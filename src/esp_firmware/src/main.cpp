/*
 * ESP Controller - Merge aus MainCode.ino + motor_encoder_test_v2.ino
 * Board : ESP32-S3 (MainPCB, custom) / ESP32 Arduino Core 3.x
 *
 * Kernaufteilung:
 *   Core 1 (loop)             : Command-Handling - Jetson-Protokoll, USB-Debug,
 *                               Button, Servo/Lenkung, Batteriemessung.
 *                               Darf blockieren.
 *   Core 0 (motorControlTask) : Antrieb - VNH5019, Encoder, PID-Positionsregler.
 *                               Einziger Ort, der die Motorpins anfasst.
 *
 * Uebergabe zwischen den Kernen: ausschliesslich ueber die setXxx/getXxx-
 * Funktionen, abgesichert per Spinlock, plus eine Queue fuer die Ergebnisse
 * abgeschlossener Positionsfahrten. Nie direkt
 * auf die Sollwerte schreiben - sonst sieht der Motortask eine neue Richtung
 * mit altem Duty.
 */

#include <Arduino.h>
#include <Preferences.h>
#include <ESP32Encoder.h>
#include <esp_timer.h>
#include "SCServo.h"

// ==========================================
// 0. KONSOLE - USB-CDC und optional UART0
// ==========================================
//
// Der ESP32-S3 kann die Konsole auf zwei Wegen ausgeben:
//   - natives USB-Serial-JTAG (GPIO19/20). Mit ARDUINO_USB_CDC_ON_BOOT=1 und
//     ARDUINO_USB_MODE=1 zeigt "Serial" genau dorthin (siehe platformio.ini).
//   - UART0 (GPIO43/44), wo bei manchen Boards ein USB-UART-Wandler haengt.
//
// Fallstrick beim Bring-up: HWCDC::write() verwirft die Daten vollstaendig,
// solange isCDC_Connected() false liefert - also solange der Host die
// CDC-Verbindung nicht aufgebaut hat. Ein Terminal, das DTR nicht setzt, sieht
// deshalb nie etwas, obwohl der COM-Port existiert und der ESP laeuft.
//
// -DCONSOLE_MIRROR_UART0=1 (Env "esp32-s3-uart0") legt die Konsole zusaetzlich
// auf UART0. Dann genuegt ein USB-TTL-Adapter an GPIO43 (ESP-TX) / GPIO44
// (ESP-RX) als Notausgang, unabhaengig vom USB-Zustand. Nur einschalten, wenn
// diese beiden Pins auf dem Board frei sind.
#ifndef CONSOLE_MIRROR_UART0
#define CONSOLE_MIRROR_UART0 0
#endif

#if CONSOLE_MIRROR_UART0
class ConsoleIO : public Print {
public:
    void begin(unsigned long baud) {
        HWCDCSerial.begin();
        Serial0.begin(baud);          // UART0 auf seinen Standardpins
    }
    // Auf UART0 immer, auf CDC nur wenn verbunden: sonst kostet jeder Aufruf
    // bis zu tx_timeout_ms (100 ms) am TX-Semaphor, und das im loop().
    size_t write(uint8_t c) override {
        Serial0.write(c);
        if (HWCDCSerial) HWCDCSerial.write(c);
        return 1;
    }
    size_t write(const uint8_t* b, size_t n) override {
        Serial0.write(b, n);
        if (HWCDCSerial) HWCDCSerial.write(b, n);
        return n;
    }
    void flush() override {
        Serial0.flush();
        if (HWCDCSerial) HWCDCSerial.flush();
    }
    int available() { return HWCDCSerial.available() + Serial0.available(); }
    int read() {
        if (HWCDCSerial.available()) return HWCDCSerial.read();
        if (Serial0.available())     return Serial0.read();
        return -1;
    }
};
static ConsoleIO Console;

// Der Core definiert "Serial" selbst als Makro auf HWCDCSerial. Umbiegen statt
// alle Aufrufstellen anzufassen - "Serial1"/"Serial2" sind eigene Tokens und
// bleiben unberuehrt.
#undef Serial
#define Serial Console
#endif  // CONSOLE_MIRROR_UART0

// ==========================================
// 1. PIN- UND PROTOKOLL-DEFINITIONEN
// ==========================================

// --- Antrieb: VNH5019 + Quadratur-Encoder ---
constexpr int PIN_MOTOR_PWM = 41;   // VNH5019 PWM
constexpr int PIN_MOTOR_INA = 42;   // VNH5019 INA
constexpr int PIN_MOTOR_INB = 38;   // VNH5019 INB
// Stromsense: auf dem neuen Board an GPIO8 = ADC1_CH7, also analog lesbar
// (das alte GPIO39 war es nicht). Skalierung siehe CS_MV_PER_A_NOMINAL.
constexpr int PIN_MOTOR_CS  = 8;    // VNH5019 CS, ADC1_CH7
constexpr int PIN_ENC_A     = 15;   // Encoder A
constexpr int PIN_ENC_B     = 16;   // Encoder B

// Fahrtrichtung. Welche Drehrichtung "vorwaerts" ist, entscheidet die
// Verdrahtung, nicht das Protokoll: an welchen Klemmen des VNH5019 die
// Motorleitungen haengen und wie herum der Encoder eingebaut ist.
// Auf true stellen, wenn ein POSITIVER Motorwert das Fahrzeug rueckwaerts
// fahren laesst.
//
// Der Schalter dreht Motorausgang UND Encoder gemeinsam - das ist Absicht.
// Wuerden die beiden unterschiedliche Vorzeichen haben, faende die
// Positionsregelung ihr Ziel nie: sie zieht dann von der Sollposition weg
// statt darauf zu, bis der Timeout kommt oder etwas kaputtgeht.
constexpr bool DRIVE_INVERT = true;

// --- Peripherie ---
// Servo-Bus ueber die Halbduplex-Buffer des neuen Boards, Pins aus ESP-Sicht:
//   ESP-TX (GPIO17) -> A-Eingang des SN74LVC1G126 (treibt die Busleitung)
//   ESP-RX (GPIO18) <- Y-Ausgang des SN74LVC1G125 (haengt an der Busleitung)
// Die Richtungsumschaltung macht die Hardware ueber die beiden OE-Eingaenge;
// die Firmware kennt keinen Direction-Pin. Zur Laufzeit per "svpin<rx>,<tx>"
// tauschbar, falls die Zuordnung doch andersherum bestueckt ist.
#define PIN_SERVO_RX    18
#define PIN_SERVO_TX    17
#define PIN_JETSON_RX   10
#define PIN_JETSON_TX   11
#define PIN_LED         13
#define PIN_BUTTON      9
#define PIN_BATTERY     1           // ADC1_CH0, Teiler 100k / 22k gegen GND

// --- PWM ---
// 10 Bit beibehalten, damit CMD_MOTOR (16-Bit-Speed 0..1023) unveraendert bleibt.
constexpr int PWM_FREQ = 20000;
constexpr int PWM_RES  = 10;
constexpr uint16_t DUTY_MAX = (1 << PWM_RES) - 1;   // 1023

// --- Protokoll: Jetson -> ESP ---
#define START_BYTE      0xA5

// Zweites Startbyte fuer Pakete mit Sendezeitstempel:
//
//   A6 <CMD> <uint32 t_tx_us> <PAYLOAD wie bei A5>
//
// Der Stempel sind die unteren 32 Bit der ESP-Uhr (esp_timer, Mikrosekunden
// seit Boot) und meint den Zeitpunkt, zu dem das *letzte Byte* des Pakets die
// Leitung verlaesst - nicht den Zeitpunkt des Aufrufs. Der Anteil fuers
// Rausschieben (10 Bit pro Byte bei 115200 Baud) wird eingerechnet, damit
// beide Seiten denselben Bezugspunkt haben und ein langes Paket nicht
// scheinbar frueher losgeht als ein kurzes.
//
// Die ESP-Uhr in die Jetson-Uhr rechnet der Offset aus dem Ping-Pong von
// CMD_TIME_SYNC / CMD_TIME_RSP (siehe docs/JETSON_BRIDGE.md).
//
// Stempeln ist per Default AUS und wird mit CMD_STAMP_MODE (oder "ts1" auf der
// USB-Konsole) eingeschaltet - eine Bridge, die 0xA6 nicht kennt, sieht sonst
// nur noch Bruch. Der Empfangspfad des ESP versteht 0xA6 immer, die Bridge
// darf ihre Befehle also jederzeit stempeln.
#define START_BYTE_TS   0xA6
#define CMD_MOTOR       0x10   // 3B: dir, speedHi, speedLo
#define CMD_SERVO       0x20   // 3B: id, pctHi, pctLo
#define CMD_LED         0x30   // 1B: on/off
#define CMD_CALIBRATE   0x40   // 0B: startet die manuelle Kalibrierung
#define CMD_CAL         0x41   // 2B: aktion, argument (manuelle Kalibrierung)
#define CMD_TORQUE      0x50   // 0B
#define CMD_TRIM        0x60   // 1B: 0=links 1=rechts 2=speichern
#define CMD_PID_SET     0x80   // 5B: paramId, int32 wert (x1000)
#define CMD_PID_GET     0x81   // 0B
#define CMD_PID_SAVE    0x83   // 0B: aktuelle Parameter ins NVS schreiben
#define CMD_MOVE        0x90   // 5B: moveId, int32 ziel in 1/10 Grad (absolut)
#define CMD_MOVE_ABORT  0x91   // 0B
#define CMD_PROGRESS    0x92   // 0B
#define CMD_BATTERY     0xA0   // 0B
#define CMD_TIME_SYNC   0xB0   // 1B: seq  -> Antwort CMD_TIME_RSP
#define CMD_STAMP_MODE  0xB2   // 1B: 0=aus 1=an -> Antwort CMD_STAMP_RSP
#define CMD_TELEM_RATE  0xC0   // 2B: uint16 Intervall in ms, 0 = aus
#define CMD_EMERGENCY   0xFF   // 0B

// Aktionen in CMD_CAL
#define CAL_ACT_START   0x00   // Kalibriermodus starten
#define CAL_ACT_MINUS   0x01   // einen Schritt Richtung Position 0
#define CAL_ACT_PLUS    0x02   // einen Schritt Richtung Position 1023
#define CAL_ACT_CENTER  0x03   // aktuelle Stellung = Mitte
#define CAL_ACT_LEFT    0x04   // aktuelle Stellung = linker Anschlag
#define CAL_ACT_RIGHT   0x05   // aktuelle Stellung = rechter Anschlag
#define CAL_ACT_SAVE    0x06   // pruefen, ins NVS schreiben, Modus beenden
#define CAL_ACT_ABORT   0x07   // abbrechen, gespeicherte Werte bleiben
#define CAL_ACT_FREE    0x08   // Torque aus - Lenkung von Hand bewegen
#define CAL_ACT_HOLD    0x09   // Torque an - aktuelle Stellung halten
#define CAL_ACT_GOTO_C  0x0A   // gemerkte Mitte anfahren
#define CAL_ACT_STEP    0x0B   // arg = neue Schrittweite in Ticks
#define CAL_ACT_STATUS  0x0C   // nur Zustand abfragen

// Status-Codes in CMD_CAL_RSP
#define CAL_ST_OK       0x00   // Aktion ausgefuehrt
#define CAL_ST_SAVED    0x01   // Kalibrierung gespeichert, Modus beendet
#define CAL_ST_REJECTED 0x02   // unvollstaendig oder unplausibel - nicht gespeichert
#define CAL_ST_NOSERVO  0x03   // Servo antwortet nicht
#define CAL_ST_INACTIVE 0x04   // Aktion braucht einen laufenden Kalibriermodus
#define CAL_ST_LIMIT    0x05   // Bereichsende 0 / 1023 erreicht

// --- Protokoll: ESP -> Jetson ---
#define CMD_CAL_RSP     0x42   // 11B: aktiv, flags, status, int16 pos/mitte/links/rechts
#define CMD_BUTTON      0x70   // 1B: 0x01 = pressed
#define CMD_PID_RSP     0x82   // 12B: int32 Kp, Ki, Kd (jeweils x1000)
#define CMD_PID_SAVED   0x84   // 1B: 0x00 = gespeichert, 0x01 = Fehler
#define CMD_MOVE_DONE   0x93   // 6B: moveId, status, int32 ist-Pos (1/10 Grad)
#define CMD_PROGRESS_RSP 0x94  // 11B: moveId, aktiv, prozent, int32 ist, int32 ziel
#define CMD_BATTERY_RSP 0xA1   // 6B: int32 pack mV, int16 zelle mV
#define CMD_BATTERY_WARN 0xA2  // 6B: wie CMD_BATTERY_RSP, ungefragt bei Unterspannung
#define CMD_TIME_RSP    0xB1   // 17B: seq, int64 t_rx_us, int64 t_tx_us
#define CMD_STAMP_RSP   0xB3   // 1B: aktueller Stempelmodus
#define CMD_TELEMETRY   0xC1   // 12B: int32 pos, int32 tempo, int16 duty, int16 mA

// Status-Codes in CMD_MOVE_DONE
#define MOVE_OK         0x00
#define MOVE_TIMEOUT    0x01
#define MOVE_ABORTED    0x02

#define JETSON_TIMEOUT  5000

// ==========================================
// 2. GETEILTER ZUSTAND ZWISCHEN DEN KERNEN
// ==========================================

enum MotorMode : uint8_t {
    MOTOR_COAST    = 0,   // H-Bruecke hochohmig, Motor rollt aus
    MOTOR_DRIVE    = 1,   // faehrt offen mit duty in Richtung reverse
    MOTOR_BRAKE    = 2,   // Kurzschlussbremse mit duty als Bremskraft
    MOTOR_POSITION = 3    // PID regelt auf Zielposition
};

struct MotorCommand {
    MotorMode mode    = MOTOR_COAST;
    bool      reverse = false;
    uint16_t  duty    = 0;          // 0..DUTY_MAX
};

// PID-Parameter. Startwerte sind Schaetzungen fuer den Pololu 25D mit 408
// Counts/U - muessen am realen Aufbau nachgezogen werden (Befehl "pid").
struct PidParams {
    float    kp        = 4.0f;    // Duty pro Count Regelabweichung
    float    ki        = 0.5f;    // Duty pro (Count * Sekunde)
    float    kd        = 0.10f;   // Duty pro (Count / Sekunde)
    float    iTermLim  = 200.0f;  // Begrenzung des I-Anteils in Duty (Anti-Windup)
    uint16_t maxDuty   = 700;     // Stellgroessenbegrenzung
    // Anlauf-Duty gegen Haftreibung. Kurz vorm Ziel ist die Regelabweichung so
    // klein, dass kp*err unter die Losbrechschwelle faellt - der Motor bleibt
    // stehen und die Fahrt laeuft in den Timeout. Solange das Ziel nicht
    // erreicht ist, wird mindestens dieser Wert angelegt. 0 = aus.
    uint16_t minDuty   = 0;
    int32_t  tolDeg10  = 50;      // Zielfenster, 1/10 Grad (50 = 5,0 Grad)
    uint32_t settleMs  = 200;     // so lange im Fenster -> fertig
    uint32_t timeoutMs = 10000;   // Abbruch, wenn Ziel nicht erreicht
};

struct MoveState {
    uint8_t  id         = 0;
    bool     active     = false;
    uint8_t  progress   = 0;      // 0..100 %
    long     startCnt   = 0;
    long     targetCnt  = 0;
};

struct MoveResult {
    uint8_t id;
    uint8_t status;
    int32_t finalDeg10;
};

static MotorCommand   g_motorCmd;
static PidParams      g_pid;
static MoveState      g_move;
static unsigned long  g_lastCmdTime = 0;   // Watchdog-Zeitstempel
static portMUX_TYPE   g_motorMux = portMUX_INITIALIZER_UNLOCKED;

static QueueHandle_t  g_moveResultQueue = nullptr;

// Telemetrie: Motortask schreibt, Core 1 liest nur zur Ausgabe.
static volatile float   g_rpm        = 0.0f;
static volatile long    g_encCount   = 0;
static volatile int16_t g_dutySigned = 0;   // + vorwaerts, - rueckwaerts

ESP32Encoder encoder;
// COUNTS_PER_REV des AKTIVEN Motors eintragen:
// ServoCity DE3 (Open-Collector): 3 PPR x 4 x 42,875 Getriebe = ~514,5
// Pololu 25D #4841 (Push-Pull)  : 48 CPR x 4,4 Getriebe = 211,2 (Ausgangswelle)
constexpr float COUNTS_PER_REV = 408.0f;

// --- Geschwindigkeitsmessung ---
// Abgetastet wird mit dem Takt des Motortasks (TASK_PERIOD, 10 ms). Gemessen
// wird ueber ein GLEITENDES Fenster der letzten g_speedWindow Abtastungen:
//
//   tempo = (count[jetzt] - count[jetzt - n]) / (t[jetzt] - t[jetzt - n])
//
// Damit kommt bei jeder Abtastung ein neuer Wert - also mit 100 Hz -, das
// Fenster darf aber trotzdem laenger sein als 10 ms. Die beiden Groessen sind
// entkoppelt, und genau das ist der Punkt der Uebung: frueher hing die
// Ausgaberate an der Fensterlaenge, ein Wert alle 100 ms.
//
// Die Differenz der beiden Randwerte IST der Mittelwert ueber das Fenster -
// alle Zwischenwerte kuerzen sich weg. Deshalb kein Mittelwert der
// Einzelmessungen und kein EMA: die Verzoegerung betraegt exakt ein halbes
// Fenster und laesst sich hinschreiben, statt als Zeitkonstante im Nebel zu
// bleiben.
//
// Was die Fensterlaenge kostet, ist Aufloesung. Ein einzelner Impuls in einem
// Fenster von n x 10 ms entspricht:
//
//   n = 1  (10 ms)   14,7 U/min   88 grad/s   <- so grob wie der Vollausschlag
//   n = 5  (50 ms)    2,9 U/min   17,6 grad/s
//   n = 10 (100 ms)   1,5 U/min    8,8 grad/s
//
// bei 408 Impulsen je Umdrehung und rund 30 U/min Hoechstdrehzahl. Ueber das
// Fenster laesst sich Rauschen gegen Verzoegerung tauschen, zur Laufzeit per
// "sw<n>" auf der Konsole.
constexpr uint8_t SPEED_WINDOW_MAX = 32;
static volatile uint8_t g_speedWindow = 1;

// --- Umrechnung Ausgangswelle: 1/10 Grad <-> Encoder-Counts ---
static inline long deg10ToCounts(int32_t deg10) {
    return lroundf(deg10 * COUNTS_PER_REV / 3600.0f);
}
static inline int32_t countsToDeg10(long counts) {
    return (int32_t)lroundf(counts * 3600.0f / COUNTS_PER_REV);
}

// ==========================================
// 3. ZUGRIFF AUF DEN GETEILTEN ZUSTAND
// ==========================================

// Offener Steuerbefehl. Fuettert den Watchdog - jeder gueltige Befehl haelt
// den Antrieb am Leben, egal ob von Jetson oder USB. Bricht eine laufende
// Positionsfahrt ab, damit sich nicht zwei Regler um den Motor streiten.
void setMotorCommand(MotorMode mode, bool reverse, uint16_t duty) {
    if (duty > DUTY_MAX) duty = DUTY_MAX;
    portENTER_CRITICAL(&g_motorMux);
    bool wasMoving = g_move.active;
    uint8_t movingId = g_move.id;
    long curCnt = g_encCount;
    g_motorCmd.mode    = mode;
    g_motorCmd.reverse = reverse;
    g_motorCmd.duty    = duty;
    g_move.active      = false;
    g_lastCmdTime      = millis();
    portEXIT_CRITICAL(&g_motorMux);

    if (wasMoving && g_moveResultQueue) {
        MoveResult r = { movingId, MOVE_ABORTED, countsToDeg10(curCnt) };
        xQueueSend(g_moveResultQueue, &r, 0);
    }
}

// Positionsfahrt starten. deltaDeg10 ist RELATIV zur aktuellen Stellung, in
// 1/10 Grad der Ausgangswelle. 900 = "drehe 90 Grad weiter", -450 = "45 Grad
// zurueck". Damit ist eine Fahrt reset-fest: sie braucht keinen Nullpunkt.
// Es laeuft immer nur eine Fahrt. Wird eine neue gestartet, waehrend noch eine
// aktiv ist, bekommt die alte ID ein MOVE_ABORTED - sonst wartet der Jetson
// ewig auf eine Quittung, die nie kommt.
void startMove(uint8_t id, int32_t deltaDeg10) {
    portENTER_CRITICAL(&g_motorMux);
    bool    superseded = g_move.active;
    uint8_t oldId      = g_move.id;
    long    curCnt     = g_encCount;
    g_motorCmd.mode   = MOTOR_POSITION;
    g_motorCmd.duty   = 0;
    g_move.id         = id;
    g_move.active     = true;
    g_move.progress   = 0;
    g_move.startCnt   = curCnt;
    g_move.targetCnt  = curCnt + deg10ToCounts(deltaDeg10);
    g_lastCmdTime     = millis();
    portEXIT_CRITICAL(&g_motorMux);

    if (superseded && g_moveResultQueue) {
        MoveResult r = { oldId, MOVE_ABORTED, countsToDeg10(curCnt) };
        xQueueSend(g_moveResultQueue, &r, 0);
    }
}

void abortMove() {
    setMotorCommand(MOTOR_COAST, false, 0);
}

MotorCommand getMotorCommand(unsigned long* lastCmdTimeOut) {
    MotorCommand c;
    portENTER_CRITICAL(&g_motorMux);
    c = g_motorCmd;
    if (lastCmdTimeOut) *lastCmdTimeOut = g_lastCmdTime;
    portEXIT_CRITICAL(&g_motorMux);
    return c;
}

MoveState getMoveState() {
    MoveState m;
    portENTER_CRITICAL(&g_motorMux);
    m = g_move;
    portEXIT_CRITICAL(&g_motorMux);
    return m;
}

PidParams getPid() {
    PidParams p;
    portENTER_CRITICAL(&g_motorMux);
    p = g_pid;
    portEXIT_CRITICAL(&g_motorMux);
    return p;
}

void setPid(const PidParams& p) {
    portENTER_CRITICAL(&g_motorMux);
    g_pid = p;
    portEXIT_CRITICAL(&g_motorMux);
}

// ==========================================
// 4. SERVO / LENKUNG - Zustand
// ==========================================

Preferences prefs;
int trimOffset = 0;

const int SERVO_ID = 1;

// SC-Serie (SCSCL, big-endian): Positionsbereich 0..1023 (10 Bit), Mitte 512.
// Der SC09 ist ein SCS-Servo, KEIN STS - er antwortet big-endian und mit halber
// Aufloesung. Deshalb SCSCL statt SMS_STS.
constexpr int SERVO_POS_MAX    = 1023;
constexpr int SERVO_CENTER_DEF = 512;
// SCSCL-WritePosEx nimmt Speed (Acc/Time intern 0). Speed 0 ist beim SC09 KEIN
// "maximal schnell", sondern "keine Geschwindigkeit" - das Ziel landet im
// Register, der Servo bleibt aber stehen. Deshalb auch beim Lenken > 0.
constexpr uint16_t SERVO_SPEED_FAST  = 1500;   // Lenken: zuegig, aber mit Speed
constexpr uint16_t SERVO_SPEED_CALIB = 500;    // Kalibrierung: gebremst
constexpr uint8_t  SERVO_ACC         = 50;      // von SCSCL ignoriert, aus Kompatibilitaet
// SC09: 0..1023 Ticks ueber ~300 Grad Vollbereich. Nur fuer die Anzeige.
constexpr float    SERVO_DEG_PER_TICK = 300.0f / (SERVO_POS_MAX + 1);

// Konvention aus der Kalibrierung: leftLimit ist der Anschlag mit der GROESSEREN
// Rohposition, rightLimit der mit der kleineren, centerLimit die von Hand
// gesetzte Geradeausstellung. softwareCenterPos = centerLimit + trimOffset.
// Die Zuordnung in CMD_SERVO rechnet symmetrisch, eine gespiegelt montierte
// Lenkung funktioniert also auch mit vertauschten Werten - dann ist nur die
// Vorzeichenrichtung des Lenkbefehls gedreht.
int softwareCenterPos = SERVO_CENTER_DEF;
int centerLimit = SERVO_CENTER_DEF;
int leftLimit  = SERVO_POS_MAX;
int rightLimit = 0;

int servoManualPos = SERVO_CENTER_DEF;   // Position der manuellen Konsolensteuerung (j/l/m)

// Serial2-Pins fuer den Servo-Bus, zur Laufzeit umstellbar ("svpin<rx>,<tx>").
// Zum Eingrenzen einer vertauschten Verkabelung, ohne loeten zu muessen.
int g_servoRx = PIN_SERVO_RX;
int g_servoTx = PIN_SERVO_TX;
uint32_t g_servoBaud   = 1000000;   // per "svbaud<n>" fuer Oszi-Messung senkbar
bool     g_servoTxTest = false;      // sendet dauerhaft 0x55 zum Oszilloskopieren

TaskHandle_t MotorControlTaskHandle;

volatile bool buttonTriggered = false;

void IRAM_ATTR buttonISR() {
    buttonTriggered = true;
}

// ==========================================
// 5. MOTORTREIBER (VNH5019) - nur vom Motortask (Core 0) aufrufen!
// ==========================================

class MotorDriver {
private:
    uint8_t pwmPin, inaPin, inbPin;

public:
    MotorDriver(uint8_t pwm, uint8_t ina, uint8_t inb)
        : pwmPin(pwm), inaPin(ina), inbPin(inb) {}

    void begin() {
        pinMode(inaPin, OUTPUT);
        pinMode(inbPin, OUTPUT);
        ledcAttach(pwmPin, PWM_FREQ, PWM_RES);
        coast();
    }

    void coast() {
        ledcWrite(pwmPin, 0);
        digitalWrite(inaPin, LOW);
        digitalWrite(inbPin, LOW);
    }

    void drive(bool reverse, uint16_t duty) {
        // != wirkt auf bool wie XOR: DRIVE_INVERT kippt die Richtung.
        const bool rev = (reverse != DRIVE_INVERT);
        digitalWrite(inaPin, rev ? LOW  : HIGH);
        digitalWrite(inbPin, rev ? HIGH : LOW);
        ledcWrite(pwmPin, duty);
    }

    // INA=INB=LOW mit Duty > 0: Kurzschlussbremse gegen GND.
    void brake(uint16_t duty) {
        digitalWrite(inaPin, LOW);
        digitalWrite(inbPin, LOW);
        ledcWrite(pwmPin, duty);
    }
};

// ==========================================
// 6. MOTOR-CONTROL TASK (Core 0)
// ==========================================

// Rampe: max. Duty-Aenderung pro Task-Zyklus.
// 25 -> volle Skala in ~410 ms. Hoeher = spritziger, aber mehr Stromspitze.
constexpr uint16_t RAMP_STEP    = 25;
constexpr uint32_t TASK_PERIOD  = 10;    // ms - zugleich das Abtastintervall
                                         // der Geschwindigkeitsmessung

void motorControlTask(void* pvParameters) {
    MotorDriver* motor = (MotorDriver*)pvParameters;

    uint16_t  appliedDuty    = 0;       // was gerade wirklich anliegt
    bool      appliedReverse = false;
    MotorMode appliedMode    = MOTOR_COAST;

    // Ringpuffer der Geschwindigkeitsmessung: je Abtastung Encoderstand und
    // Zeitpunkt. Zeitpunkt in Mikrosekunden, weil 10 ms in Millisekunden nur
    // 10 Schritte sind - der Quantisierungsfehler der Zeit waere sonst so
    // gross wie das Fenster selbst.
    long     speedCnt[SPEED_WINDOW_MAX] = {0};
    uint32_t speedUs[SPEED_WINDOW_MAX]  = {0};
    uint8_t  speedHead = 0;    // naechster Schreibplatz
    uint8_t  speedFill = 0;    // wie viele Plaetze schon gueltig sind

    // PID-Zustand
    float integral   = 0.0f;
    long  lastPosCnt = 0;
    bool  pidPrimed  = false;
    unsigned long moveStartMs = 0;
    unsigned long inWindowSince = 0;
    bool    wasMoving  = false;
    uint8_t lastMoveId = 0;

    const float dt = TASK_PERIOD / 1000.0f;

    for (;;) {
        unsigned long now = millis();
        long posCnt = (long)encoder.getCount();
        g_encCount = posCnt;

        unsigned long lastCmdTime;
        MotorCommand cmd = getMotorCommand(&lastCmdTime);
        MoveState    mv  = getMoveState();
        PidParams    pid = getPid();

        // --- Watchdog ---
        // Greift nur im offenen Steuerbetrieb. Eine Positionsfahrt darf laenger
        // als JETSON_TIMEOUT dauern, ohne dass ein Befehl nachkommt - dort ist
        // pid.timeoutMs das Sicherheitsnetz.
        if (cmd.mode == MOTOR_DRIVE && (now - lastCmdTime > JETSON_TIMEOUT)) {
            cmd.mode = MOTOR_COAST;
            cmd.duty = 0;
        }

        uint16_t wantDuty    = 0;
        bool     wantReverse = cmd.reverse;
        MotorMode wantMode   = cmd.mode;

        // ------------------------------------------------
        // PID-Positionsregler
        // ------------------------------------------------
        if (cmd.mode == MOTOR_POSITION && mv.active) {
            // Neue Fahrt = vorher keine aktiv ODER eine andere ID. Ohne den
            // ID-Vergleich wuerde eine abloesende Fahrt Integral und Startzeit
            // der alten erben und verfrueht in den Timeout laufen.
            if (!wasMoving || mv.id != lastMoveId) {
                integral      = 0.0f;
                lastPosCnt    = posCnt;
                pidPrimed     = false;
                moveStartMs   = now;
                inWindowSince = 0;
                wasMoving     = true;
                lastMoveId    = mv.id;
            }

            long  errCnt = mv.targetCnt - posCnt;
            float err    = (float)errCnt;

            long tolCnt = deg10ToCounts(pid.tolDeg10);
            if (tolCnt < 1) tolCnt = 1;

            // D-Anteil auf die Messgroesse statt auf den Fehler: kein
            // Ableitungssprung, wenn ein neues Ziel gesetzt wird.
            float dMeas = pidPrimed ? ((float)(posCnt - lastPosCnt) / dt) : 0.0f;
            lastPosCnt  = posCnt;
            pidPrimed   = true;

            integral += err * dt;
            float iTerm = pid.ki * integral;
            // Anti-Windup: I-Anteil in Duty begrenzen und Integral zurueckrechnen
            if (iTerm >  pid.iTermLim) { iTerm =  pid.iTermLim; integral = (pid.ki != 0.0f) ?  pid.iTermLim / pid.ki : 0.0f; }
            if (iTerm < -pid.iTermLim) { iTerm = -pid.iTermLim; integral = (pid.ki != 0.0f) ? -pid.iTermLim / pid.ki : 0.0f; }

            float out = pid.kp * err + iTerm - pid.kd * dMeas;

            if (labs(errCnt) <= tolCnt) {
                // Totzone: im Zielfenster nicht weiter nachregeln. Sonst steht
                // waehrend der Verweilzeit ein Rest-Duty aus dem I-Anteil an und
                // der Antrieb drueckt gegen die Reibung, ohne etwas zu bewirken.
                out = 0.0f;
                integral = 0.0f;
            } else if (pid.minDuty > 0 && fabsf(out) < (float)pid.minDuty) {
                // Haftreibung ueberwinden: ausserhalb des Zielfensters nie
                // weniger als minDuty anlegen. Richtung kommt aus dem Vorzeichen
                // der Regelabweichung, nicht aus out - out kann durch den
                // D-Anteil kurzzeitig das falsche Vorzeichen haben.
                out = (errCnt < 0) ? -(float)pid.minDuty : (float)pid.minDuty;
            }

            uint16_t lim = min<uint16_t>(pid.maxDuty, DUTY_MAX);
            if (out >  (float)lim) out =  (float)lim;
            if (out < -(float)lim) out = -(float)lim;

            wantReverse = (out < 0.0f);
            wantDuty    = (uint16_t)fabsf(out);
            wantMode    = MOTOR_POSITION;

            // --- Fortschritt ---
            long span = labs(mv.targetCnt - mv.startCnt);
            uint8_t pct = 100;
            if (span > 0) {
                long doneCnt = labs(posCnt - mv.startCnt);
                if (doneCnt > span) doneCnt = span;
                pct = (uint8_t)((doneCnt * 100) / span);
            }
            portENTER_CRITICAL(&g_motorMux);
            g_move.progress = pct;
            portEXIT_CRITICAL(&g_motorMux);

            // --- Ziel erreicht? ---
            bool finished = false;
            uint8_t status = MOVE_OK;

            if (labs(errCnt) <= tolCnt) {
                if (inWindowSince == 0) inWindowSince = now;
                if (now - inWindowSince >= pid.settleMs) { finished = true; status = MOVE_OK; }
            } else {
                inWindowSince = 0;
            }

            if (!finished && (now - moveStartMs > pid.timeoutMs)) {
                finished = true;
                status = MOVE_TIMEOUT;
            }

            if (finished) {
                portENTER_CRITICAL(&g_motorMux);
                g_move.active      = false;
                g_move.progress    = (status == MOVE_OK) ? 100 : g_move.progress;
                g_motorCmd.mode    = MOTOR_COAST;
                g_motorCmd.duty    = 0;
                portEXIT_CRITICAL(&g_motorMux);

                MoveResult r = { mv.id, status, countsToDeg10(posCnt) };
                xQueueSend(g_moveResultQueue, &r, 0);

                wantMode  = MOTOR_COAST;
                wantDuty  = 0;
                wasMoving = false;
            }
        } else {
            wasMoving = false;
            if (cmd.mode == MOTOR_POSITION) {   // Fahrt beendet, noch nicht umgeschaltet
                wantMode = MOTOR_COAST;
                wantDuty = 0;
            } else {
                wantDuty = (cmd.mode == MOTOR_COAST) ? 0 : cmd.duty;
            }
        }

        // --- Jeder Modus- oder Richtungswechsel geht ueber Duty 0 ---
        // Sonst schaltet INA/INB unter Last um (Stromspitze), oder - schlimmer -
        // ein Nothalt laesst appliedMode auf DRIVE stehen und die Bremskraft
        // wirkt als Fahrbefehl.
        bool changing = (wantMode != appliedMode) ||
                        ((wantMode == MOTOR_DRIVE || wantMode == MOTOR_POSITION) &&
                         wantReverse != appliedReverse);
        if (changing && appliedDuty > 0) {
            wantDuty = 0;
        }

        // Richtung/Modus uebernehmen, sobald die Bruecke stromlos ist.
        // MUSS vor der Rampe stehen: danach ist appliedDuty nach dem ersten
        // Rampenschritt nie wieder 0 und appliedMode haengt fuer immer auf COAST
        // - der Motor laeuft dann gar nicht erst an.
        if (appliedDuty == 0) {
            appliedReverse = wantReverse;
            appliedMode    = wantMode;
        }

        // --- Slew-Rate-Limiter ---
        // Nur beim Beschleunigen begrenzen. Reduzieren ist elektrisch
        // unkritisch (entspricht dem normalen PWM-Tastverhaeltnis) und muss
        // sofort wirken, sonst braucht der Nothalt eine halbe Sekunde.
        // Bremsen wird ebenfalls sofort voll angelegt.
        if (appliedDuty < wantDuty) {
            uint16_t step = (appliedMode == MOTOR_BRAKE) ? DUTY_MAX : RAMP_STEP;
            appliedDuty = min<uint16_t>(wantDuty, appliedDuty + step);
        } else {
            appliedDuty = wantDuty;
        }

        switch (appliedMode) {
            case MOTOR_DRIVE:
            case MOTOR_POSITION: motor->drive(appliedReverse, appliedDuty); break;
            case MOTOR_BRAKE:    motor->brake(appliedDuty);                 break;
            case MOTOR_COAST:
            default:             motor->coast();                            break;
        }

        // Was wirklich an der Bruecke anliegt, fuer Plotter und Telemetrie.
        g_dutySigned = (appliedMode == MOTOR_DRIVE || appliedMode == MOTOR_POSITION)
                       ? (int16_t)(appliedReverse ? -(int)appliedDuty : (int)appliedDuty)
                       : 0;

        // --- Geschwindigkeit: gleitendes Fenster ---
        speedCnt[speedHead] = posCnt;
        speedUs[speedHead]  = micros();
        speedHead = (uint8_t)((speedHead + 1) % SPEED_WINDOW_MAX);
        if (speedFill < SPEED_WINDOW_MAX) speedFill++;

        {
            uint8_t want = g_speedWindow;
            if (want < 1) want = 1;
            if (want > SPEED_WINDOW_MAX - 1) want = SPEED_WINDOW_MAX - 1;

            // Solange der Ring noch nicht voll ist, so weit zurueckschauen wie
            // moeglich - sonst laege der "aelteste" Wert auf einer Null aus der
            // Initialisierung und das Tempo spraenge beim Start ins Absurde.
            uint8_t back = (want < speedFill) ? want : (uint8_t)(speedFill - 1);

            if (back >= 1) {
                uint8_t newest = (uint8_t)((speedHead + SPEED_WINDOW_MAX - 1)
                                           % SPEED_WINDOW_MAX);
                uint8_t oldest = (uint8_t)((newest + SPEED_WINDOW_MAX - back)
                                           % SPEED_WINDOW_MAX);
                // Echte verstrichene Zeit statt Nenn-Takt: der Task kann
                // verspaetet drankommen, und ein zu kurz angenommenes dt
                // blaeht das Tempo auf. Die Subtraktion ist auch ueber den
                // Ueberlauf von micros() hinweg richtig (71,6 min).
                uint32_t dtUs = speedUs[newest] - speedUs[oldest];
                if (dtUs > 0) {
                    long delta = speedCnt[newest] - speedCnt[oldest];
                    g_rpm = ((float)delta / COUNTS_PER_REV)
                            * (60000000.0f / (float)dtUs);
                }
            } else {
                g_rpm = 0.0f;
            }
        }

        vTaskDelay(TASK_PERIOD / portTICK_PERIOD_MS);
    }
}

// ==========================================
// 7. BATTERIE (Core 1)
// ==========================================

// Spannungsteiler 100k oben / 22k gegen GND -> Faktor (100+22)/22.
// Per Serial ("vc<faktor>") nachziehbar und im NVS gespeichert, weil
// Widerstandstoleranzen den rechnerischen Wert um mehrere Prozent verfehlen.
constexpr float BATT_DIVIDER_NOMINAL = (100.0f + 22.0f) / 22.0f;   // 5.545
float battDivider = BATT_DIVIDER_NOMINAL;

// Ab hier ist der ADC am Anschlag: 12 dB reichen bis ~3,1 V am Pin, darueber
// liefert er stur seinen Maximalcode. Jede hoehere Spannung sieht dann gleich
// aus - der "Messwert" ist keine Messung mehr, sondern die Obergrenze selbst.
// 4S voll (16,8 V) landet ueber den Teiler bei ~3,03 V und bleibt darunter.
constexpr float BATT_ADC_CLIP_MV = 3100.0f;

constexpr int   BATT_CELLS        = 4;        // 4S
constexpr float BATT_WARN_CELL    = 3.80f;    // Warnschwelle pro Zelle
constexpr float BATT_RECOVER_CELL = 3.85f;    // Hysterese: erst darueber wieder entwarnen
constexpr uint32_t BATT_INTERVAL  = 15000;    // alle 15 s messen
constexpr uint32_t BATT_WARN_REPEAT = 60000;  // Warnung hoechstens 1x pro Minute

// --- Motorstrom (VNH5019 CS an GPIO8) ---
// Der VNH5019 spiegelt einen Bruchteil des Motorstroms auf CS; ueber den
// Messwiderstand auf dem Board wird daraus eine Spannung. Der Pololu-Traeger
// liefert ~0,14 V/A - ist der Widerstand auf dem eigenen Board anders bemessen,
// stimmt der Wert nicht. Deshalb per "ic<mV pro A>" nachziehbar und im NVS
// gespeichert. Gegenprobe mit einer Strommesszange oder dem Labornetzteil.
constexpr float CS_MV_PER_A_NOMINAL = 140.0f;
float csMvPerA  = CS_MV_PER_A_NOMINAL;
int   csZeroMv  = 0;      // Nullpunkt bei stehendem Motor ("iz")

// Rohspannung am CS-Pin in mV. Ohne Nullpunktabzug - den macht der Aufrufer.
float readMotorCsMv() {
    uint32_t sum = 0;
    for (int i = 0; i < 16; i++) sum += analogReadMilliVolts(PIN_MOTOR_CS);
    return sum / 16.0f;
}

float readMotorCurrentA() {
    if (csMvPerA <= 0.0f) return 0.0f;
    float a = (readMotorCsMv() - csZeroMv) / csMvPerA;
    return (a < 0.0f) ? 0.0f : a;   // CS kann nur Strom in eine Richtung melden
}

float    battPackV   = 0.0f;
float    battCellV   = 0.0f;
bool     battLow     = false;
uint32_t battLastRead = 0;
uint32_t battLastWarn = 0;

// Live-Ausgabe des Abgriffs ("vm"). Das 15-s-Raster taugt nicht zur Fehlersuche:
// eine wackelige Loetstelle findet man nur, wenn man daran ruettelt und dabei
// zusieht. 0 = aus. Laeuft bewusst als Flag statt als blockierende Schleife,
// damit Antrieb und Jetson-Link waehrenddessen weiterlaufen.
uint32_t g_battMonMs   = 0;
uint32_t g_battMonLast = 0;

// Spannung am Teilerabgriff in mV. analogReadMilliVolts nutzt die
// Werkskalibrierung des ADC - deutlich genauer als analogRead/4095*3.3.
// Als eigene Funktion, weil sich ohne den Rohwert ein geklippter Messwert
// nicht von einem falsch kalibrierten Teilerfaktor unterscheiden laesst.
float readBatteryMv() {
    uint32_t sum = 0;
    for (int i = 0; i < 16; i++) sum += analogReadMilliVolts(PIN_BATTERY);
    return sum / 16.0f;
}

// Liefert die Packspannung in Volt.
float readBatteryVolts() {
    return (readBatteryMv() / 1000.0f) * battDivider;
}

// Misst den Abgriff dreimal: freilaufend, dann gegen die internen Pull-
// Widerstaende des ESP (~45 kOhm). Wie stark die Spannung dabei nachgibt,
// verraet die Quellimpedanz des Knotens - und damit, ob am Pin ueberhaupt
// noch der gedachte Teiler haengt. Ohne Akku sind das die Erwartungswerte:
//   Teiler heil:      frei ~0 mV, Pullup ~1080 mV (3,3 V ueber 45k/22k)
//   22k offen, 100k an einer lebenden Schiene: Pulldown zieht auf ~1,5 V
//   Kurzschluss nach 3V3: bleibt auch mit Pulldown oben
void batteryPinDiagnose() {
    Serial.printf("Akku-Pin-Diagnose an GPIO%d (Akku sollte dafuer ab sein):\n",
                  PIN_BATTERY);

    float mvFree = readBatteryMv();

    pinMode(PIN_BATTERY, INPUT_PULLDOWN);
    vTaskDelay(50 / portTICK_PERIOD_MS);
    float mvDown = readBatteryMv();

    pinMode(PIN_BATTERY, INPUT_PULLUP);
    vTaskDelay(50 / portTICK_PERIOD_MS);
    float mvUp = readBatteryMv();

    // Zurueck auf reinen Analogeingang. Die Reihenfolge ist dieselbe wie im
    // setup(): erst lesen (haengt den Pin an den ADC-Kanal), dann daempfen.
    pinMode(PIN_BATTERY, INPUT);
    (void)analogRead(PIN_BATTERY);
    analogSetPinAttenuation(PIN_BATTERY, ADC_11db);
    vTaskDelay(50 / portTICK_PERIOD_MS);

    Serial.printf("  frei     %7.1f mV\n", mvFree);
    Serial.printf("  Pulldown %7.1f mV\n", mvDown);
    Serial.printf("  Pullup   %7.1f mV\n", mvUp);
    Serial.print("  -> ");

    if (mvFree < 300.0f) {
        Serial.println("Abgriff liegt auf Masse - das ist das erwartete Bild ohne Akku.");
        if (mvUp > 700.0f && mvUp < 1500.0f)
            Serial.println("     Pullup-Wert passt zum 22k gegen GND: Teiler ist heil.");
        else
            Serial.println("     Pullup-Wert passt aber nicht zu 22k gegen GND - Abgriff pruefen.");
    } else if (mvDown < 300.0f) {
        Serial.println("Knoten ist hochohmig: er floatet, nichts zieht ihn nach Masse.");
        Serial.println("     Der 22k-Zweig gegen GND ist offen (kalte Loetstelle/Bruch).");
    } else if (mvDown < 2500.0f) {
        Serial.println("Etwas speist ueber ~100k ein, waehrend der 22k gegen GND fehlt.");
        Serial.println("     Sieht nach dem Teiler-Oberzweig an einer lebenden Schiene aus.");
    } else {
        Serial.println("Niederohmig hart getrieben - der Pulldown kommt nicht dagegen an.");
        Serial.println("     Loetbruecke oder Fehlbestueckung nach 3V3 am wahrscheinlichsten.");
    }
}

// ==========================================
// 8. SERVO HILFSFUNKTIONEN (Kalibrierung & Torque) - Core 1
// ==========================================

void printTorque(SCSCL* servo) {
    int rawLoad = servo->ReadLoad(SERVO_ID);
    if (rawLoad != -1) {
        int actualLoad = rawLoad & 0x3FF;   // Bits 0-9
        Serial.print("ESP: Aktuelles Servo-Torque: ");
        Serial.println(actualLoad);
    } else {
        Serial.println("ESP: Fehler beim Lesen des Torques!");
    }
}

// Halbduplex-Eigenecho: jede zweite Leseanfrage verschluckt sich am zurueck-
// gespiegelten eigenen Sendepaket und liefert -1. Retry sitzt das aus - bis zu
// 6 Versuche, ein gueltiges Ergebnis ist meist beim 2. da. Betrifft nur Reads;
// WritePosEx (Lenken) braucht das nicht.
template <typename F>
int servoReadRetry(F fn, int tries = 6) {
    for (int i = 0; i < tries; i++) {
        int v = fn();
        if (v != -1) return v;
    }
    return -1;
}

// Schreibt eine rohe Servo-Position 0..1023 unter Umgehung der Lenk-Limits und
// meldet den Rueckgabewert.
void servoWriteRaw(SCSCL* servo, int pos) {
    pos = constrain(pos, 0, SERVO_POS_MAX);
    servoManualPos = pos;
    int ret = servo->WritePosEx(SERVO_ID, pos, SERVO_SPEED_CALIB, SERVO_ACC);
    Serial.printf("Servo -> Pos %d (WritePosEx ret=%d%s)\n", pos, ret,
                  ret == -1 ? " KEINE ANTWORT" : "");
}

// Testet beide Protokolle der SCServo-Lib auf demselben Bus (Serial2) und
// meldet, welches antwortet. SC-Serie (SCSCL) und STS/SMS-Serie (SMS_STS) sind
// zueinander inkompatibel - ein STS-Servo antwortet nicht auf SCSCL-Pakete.
// Serial2 mit neuen Pins fuer den Servo-Bus neu starten. RX/TX sind aus ESP-
// Sicht: rx = wo der ESP empfaengt (an U1RXD), tx = wo er sendet (an U1TXD).
void servoBusRestart(SCSCL* servo) {
    Serial2.end();
    Serial2.begin(g_servoBaud, SERIAL_8N1, g_servoRx, g_servoTx);
    servo->pSerial = &Serial2;
}

void servoSetPins(SCSCL* servo, int rx, int tx) {
    g_servoRx = rx;
    g_servoTx = tx;
    servoBusRestart(servo);
    Serial.printf("-> Servo-Bus neu: RX=IO%d  TX=IO%d @ %lu Baud. Jetzt 'svs' testen.\n",
                  g_servoRx, g_servoTx, (unsigned long)g_servoBaud);
}

void servoScanProtocols(SCSCL* scs) {
    Serial.println("--- Protokoll-Scan auf Serial2 (ID 1) ---");

    // Gegenprobe mit dem STS-Treiber (little-endian, 0..4095) auf demselben Bus.
    SMS_STS sts;
    sts.pSerial = &Serial2;
    int stsPos = servoReadRetry([&]{ return sts.ReadPos(SERVO_ID); });
    Serial.printf("  SMS_STS (STS-Serie, 0..4095): %s\n",
                  stsPos == -1 ? "keine Antwort" : String("Pos " + String(stsPos)).c_str());

    int scsPos = servoReadRetry([&]{ return scs->ReadPos(SERVO_ID); });
    Serial.printf("  SCSCL  (SC-Serie, 0..1023): %s\n",
                  scsPos == -1 ? "keine Antwort" : String("Pos " + String(scsPos)).c_str());

    if (scsPos != -1 || stsPos != -1) {
        Serial.println("  => Servo antwortet - Firmware nutzt SCSCL (SC-Serie), korrekt.");
    } else {
        Serial.println("  => Beide stumm: Verkabelung/Strom/Baudrate pruefen.");
    }
}

// Vollstaendige Servo-Diagnose. Der wichtigste Befehl, wenn sich nichts bewegt:
// er trennt Bus-, Strom-, Torque- und Mapping-Probleme voneinander.
void servoDiagnose(SCSCL* servo) {
    int pos  = servoReadRetry([&]{ return servo->ReadPos(SERVO_ID); });
    int volt = servoReadRetry([&]{ return servo->ReadVoltage(SERVO_ID); });
    int load = servoReadRetry([&]{ return servo->ReadLoad(SERVO_ID); });
    int mode = servoReadRetry([&]{ return servo->ReadMode(SERVO_ID); });

    Serial.println("--- Servo-Diagnose (ID 1) ---");
    if (pos == -1 && volt == -1) {
        Serial.println("  KEINE Antwort vom Servo-Bus.");
        Serial.printf("  -> Verkabelung Serial2 (RX IO%d / TX IO%d), 1 Mbit,\n",
                      g_servoRx, g_servoTx);
        Serial.println("     Servo-Stromversorgung und gemeinsame Masse pruefen.");
        return;
    }
    Serial.printf("  Position : %d\n", pos);
    Serial.printf("  Spannung : %.1f V%s\n", volt / 10.0f,
                  (volt != -1 && volt < 40) ? "  (< 4 V - Servo unterversorgt!)" : "");
    Serial.printf("  Last     : %d\n", load);
    Serial.printf("  Modus    : %d %s\n", mode,
                  mode == 0 ? "(Position)" : mode == 1 ? "(PWM/Rad - dreht nicht auf Position!)" : "");

    // Torque erzwingen - haeufigste Ursache fuer "haelt/bewegt sich nicht".
    int te = servo->EnableTorque(SERVO_ID, 1);
    Serial.printf("  Torque eingeschaltet (EnableTorque ret=%d)\n", te);
    if (pos != -1) {
        servo->WritePosEx(SERVO_ID, pos, SERVO_SPEED_FAST, SERVO_ACC);   // Position halten
        servoManualPos = pos;
    }
    Serial.println("  Test: 'j'/'l' bewegen, 'm' Mitte, 'sv512' Mitte-Pos.");
}

// Zwingt den SC-Servo in den Positions-Modus. Anders als STS hat die SC-Serie
// KEIN Mode-Register (33). Position vs. Dauerdreh (Wheel) haengt allein an den
// Winkel-Limits: MIN==MAX==0 => Wheel, sonst Positions-Servo. Also MIN=0,
// MAX=1023 setzen. Liegt im EPROM: unlock -> schreiben -> lock. Alles
// Schreibbefehle, daher vom Halbduplex-Echo nicht betroffen.
void servoForcePositionMode(SCSCL* servo) {
    servo->unLockEprom(SERVO_ID);
    servo->writeWord(SERVO_ID, SCSCL_MIN_ANGLE_LIMIT_L, 0);
    servo->writeWord(SERVO_ID, SCSCL_MAX_ANGLE_LIMIT_L, SERVO_POS_MAX);
    servo->LockEprom(SERVO_ID);
    vTaskDelay(20 / portTICK_PERIOD_MS);
    servo->EnableTorque(SERVO_ID, 1);
    Serial.println("-> Positions-Modus gesetzt (Limits 0..1023, Torque an).");
    Serial.println("   Jetzt 'm' / 'j' / 'l' testen.");
}

// Register-Dump + Schreib-Ruecklese-Test. Klaert die Kernfrage: kommen Writes
// ueberhaupt am Servo an? Schreibt ein Ziel und liest das Ziel-Register zurueck.
void servoRegisterDump(SCSCL* servo) {
    // SC-Serie: Mode wird aus den Winkel-Limits abgeleitet (ReadMode: 0=Position,
    // 3=Wheel). Kein eigenes Mode-Register wie bei STS.
    int mode   = servoReadRetry([&]{ return servo->ReadMode(SERVO_ID); });
    int torque = servoReadRetry([&]{ return servo->readByte(SERVO_ID, SCSCL_TORQUE_ENABLE); });
    int minA   = servoReadRetry([&]{ return servo->readWord(SERVO_ID, SCSCL_MIN_ANGLE_LIMIT_L); });
    int maxA   = servoReadRetry([&]{ return servo->readWord(SERVO_ID, SCSCL_MAX_ANGLE_LIMIT_L); });
    int pos    = servoReadRetry([&]{ return servo->ReadPos(SERVO_ID); });
    Serial.println("--- Servo-Register ---");
    Serial.printf("  Mode=%d  Torque(40)=%d  MinAng(9)=%d  MaxAng(11)=%d  Pos(56)=%d\n",
                  mode, torque, minA, maxA, pos);

    // Schreib-Test: Torque-Register direkt setzen und zuruecklesen.
    servo->writeByte(SERVO_ID, SCSCL_TORQUE_ENABLE, 1);
    vTaskDelay(20 / portTICK_PERIOD_MS);
    int tqBack = servoReadRetry([&]{ return servo->readByte(SERVO_ID, SCSCL_TORQUE_ENABLE); });
    Serial.printf("  Schreibtest Torque=1 -> zurueckgelesen %d  %s\n", tqBack,
                  tqBack == 1 ? "WRITE KOMMT AN" : "WRITE KOMMT NICHT AN!");

    // Ziel schreiben und Ziel-Register zuruecklesen.
    servo->WritePosEx(SERVO_ID, SERVO_CENTER_DEF, SERVO_SPEED_FAST, SERVO_ACC);
    vTaskDelay(20 / portTICK_PERIOD_MS);
    int goal = servoReadRetry([&]{ return servo->readWord(SERVO_ID, SCSCL_GOAL_POSITION_L); });
    Serial.printf("  WritePosEx(%d) -> GoalPos(42)=%d  %s\n", SERVO_CENTER_DEF, goal,
                  goal == SERVO_CENTER_DEF ? "ANGEKOMMEN (dreht nicht = Torque/Mechanik)"
                               : "NICHT angekommen (Write-Problem)");
}

// ==========================================
// 8b. MANUELLE LENKUNGS-KALIBRIERUNG (Core 1)
// ==========================================
//
// Der SC09 kennt keine Drehmomentbegrenzung. Faehrt er selbsttaetig gegen einen
// Anschlag, drueckt er mit vollem Moment weiter, bis Anlenkung oder Getriebe
// nachgeben - automatisches Antasten ist damit nicht sicher zu betreiben.
// Deshalb wird von Hand kalibriert: der Bediener faehrt in kleinen Schritten
// (Default 10 Ticks ~ 3 Grad) und setzt Mitte und beide Anschlaege selbst.
//
// Die gespeicherten Limits bleiben bis 'calsave' unveraendert. Ein Abbruch
// (oder ein Reset mittendrin) laesst also die alte Kalibrierung intakt.

constexpr int CAL_STEP_DEF = 10;    // Ticks pro Schritt (~2,9 Grad)
constexpr int CAL_STEP_MIN = 1;
constexpr int CAL_STEP_MAX = 200;
// Anschlaege muessen mindestens so weit auseinanderliegen (~15 Grad), sonst ist
// beim Setzen offensichtlich etwas schiefgegangen.
constexpr int CAL_MIN_SPAN = 50;

struct CalState {
    bool active     = false;
    bool freeMode   = false;                 // Torque aus, Lenkung von Hand
    int  pos        = SERVO_CENTER_DEF;      // zuletzt kommandierte Rohposition
    int  step       = CAL_STEP_DEF;
    bool haveCenter = false;
    bool haveLeft   = false;
    bool haveRight  = false;
    int  center     = SERVO_CENTER_DEF;
    int  left       = SERVO_POS_MAX;
    int  right      = 0;
};
static CalState g_cal;

static int calReadPos(SCSCL* servo) {
    return servoReadRetry([&]{ return servo->ReadPos(SERVO_ID); });
}

static int calReadLoad(SCSCL* servo) {
    int raw = servoReadRetry([&]{ return servo->ReadLoad(SERVO_ID); });
    return (raw == -1) ? -1 : (raw & 0x3FF);   // Bits 0-9, Bit 10 ist die Richtung
}

static void calPrintMark(const char* name, bool have, int value) {
    if (have) Serial.printf("  %-8s: %4d\n", name, value);
    else      Serial.printf("  %-8s:    - (noch nicht gesetzt)\n", name);
}

void calPrintStatus(SCSCL* servo) {
    int ist  = calReadPos(servo);
    int load = calReadLoad(servo);

    Serial.printf("--- Kalibrierung: %s ---\n",
                  !g_cal.active ? "inaktiv ('cal' startet sie)"
                                : (g_cal.freeMode ? "AKTIV, Torque frei" : "AKTIV"));
    Serial.printf("  Position: soll=%d  ist=%s  Last=%s\n", g_cal.pos,
                  ist  == -1 ? "?" : String(ist).c_str(),
                  load == -1 ? "?" : String(load).c_str());
    Serial.printf("  Schritt : %d Ticks (~%.1f grad)\n",
                  g_cal.step, g_cal.step * SERVO_DEG_PER_TICK);
    calPrintMark("Mitte",  g_cal.haveCenter, g_cal.center);
    calPrintMark("Links",  g_cal.haveLeft,   g_cal.left);
    calPrintMark("Rechts", g_cal.haveRight,  g_cal.right);
    Serial.printf("  gespeichert: Mitte=%d  Links=%d  Rechts=%d  Trim=%d\n",
                  centerLimit, leftLimit, rightLimit, trimOffset);
    if (g_cal.active) {
        Serial.println("  + / -      ein Schritt (auch '+50' fuer einmalig 50 Ticks)");
        Serial.println("  caln<t>    Schrittweite   calm/call/calr  Mitte/Links/Rechts merken");
        Serial.println("  calfree    Torque aus (von Hand drehen)   calhold  wieder halten");
        Serial.println("  calgo      Mitte anfahren                 calsave  speichern");
        Serial.println("  calq       abbrechen (gespeicherte Werte bleiben)");
    }
}

void calStart(SCSCL* servo) {
    // Antrieb stillsetzen - waehrend der Kalibrierung soll nichts wegrollen.
    setMotorCommand(MOTOR_COAST, false, 0);

    g_cal.active     = true;
    g_cal.freeMode   = false;
    g_cal.haveCenter = g_cal.haveLeft = g_cal.haveRight = false;
    g_cal.step       = CAL_STEP_DEF;

    servo->EnableTorque(SERVO_ID, 1);
    int p = calReadPos(servo);
    if (p != -1) {
        // Erst die Ist-Stellung uebernehmen und genau dorthin schreiben, sonst
        // springt der Servo beim Einschalten des Torques auf ein altes Ziel.
        g_cal.pos = p;
        servoManualPos = p;
        servo->WritePosEx(SERVO_ID, p, SERVO_SPEED_CALIB, SERVO_ACC);
    } else {
        // Ohne gelesene Ist-Position bewusst NICHTS schreiben: ein geratenes
        // Ziel liesse die Lenkung quer durch den Bereich schlagen.
        g_cal.pos = servoManualPos;
        Serial.println("!! Servo nicht lesbar - Position wird nicht gesetzt.");
        Serial.println("   Erst 'sv' (Diagnose), dann Kalibrierung neu starten.");
    }

    Serial.println("=== Manuelle Lenkungs-Kalibrierung ===");
    Serial.println("  1) mit + / - auf Geradeaus stellen      -> 'calm'");
    Serial.println("  2) langsam an den LINKEN Anschlag       -> 'call'");
    Serial.println("  3) langsam an den RECHTEN Anschlag      -> 'calr'");
    Serial.println("  4) 'calsave' speichert, 'calq' bricht ab");
    Serial.println("  Kurz VOR dem harten Anschlag stoppen: der SC09 hat keine");
    Serial.println("  Drehmomentbegrenzung und drueckt sonst dauerhaft dagegen.");
    Serial.println("  Alternative: 'calfree', Lenkung von Hand an den Anschlag,");
    Serial.println("  dann 'call'/'calr' - dabei wirkt gar keine Servokraft.");
    calPrintStatus(servo);
}

// Ein Schritt um delta Ticks. Meldet Ist-Position und Last zurueck, damit am
// Anschlag sichtbar wird, dass der Servo dem Ziel nicht mehr folgt.
uint8_t calMove(SCSCL* servo, int delta) {
    if (!g_cal.active) {
        Serial.println("-> Kalibrierung laeuft nicht ('cal' startet sie)");
        return CAL_ST_INACTIVE;
    }
    if (g_cal.freeMode) {
        Serial.println("-> Torque ist frei ('calhold' schaltet ihn wieder ein)");
        return CAL_ST_INACTIVE;
    }

    int target = constrain(g_cal.pos + delta, 0, SERVO_POS_MAX);
    if (target == g_cal.pos) {
        Serial.printf("-> Bereichsende %d erreicht - weiter geht es nicht\n", target);
        return CAL_ST_LIMIT;
    }

    g_cal.pos = target;
    servoManualPos = target;
    servo->WritePosEx(SERVO_ID, target, SERVO_SPEED_CALIB, SERVO_ACC);
    // Schritt zu Ende fahren lassen, bevor gemessen wird. SERVO_SPEED_CALIB ist
    // die Fahrgeschwindigkeit in Ticks/s, dazu etwas Anlauf.
    vTaskDelay((60 + abs(delta) * 1000 / (int)SERVO_SPEED_CALIB) / portTICK_PERIOD_MS);

    int ist  = calReadPos(servo);
    int load = calReadLoad(servo);
    Serial.printf("Servo -> %d (%+d)  ist=%s  Last=%s\n", target, delta,
                  ist  == -1 ? "?" : String(ist).c_str(),
                  load == -1 ? "?" : String(load).c_str());

    if (ist == -1) return CAL_ST_NOSERVO;

    // Bleibt die Ist-Position mehr als einen halben Schritt hinter dem Ziel
    // zurueck, klemmt es. Genau hier soll der Bediener aufhoeren.
    if (abs(ist - target) > max(4, abs(delta) / 2)) {
        Serial.printf("   !! folgt dem Ziel nicht (%d Ticks Abweichung) - Anschlag?\n",
                      abs(ist - target));
        Serial.println("      Anschlag hier mit 'call'/'calr' setzen und zurueckfahren.");
    }
    return CAL_ST_OK;
}

// Aktuelle Stellung als Mitte / Links / Rechts merken. Es zaehlt die ausgelesene
// Ist-Position, nicht das Kommando - im Free-Mode gibt es gar kein Kommando, und
// am Anschlag weichen beide bewusst voneinander ab.
uint8_t calSetMark(SCSCL* servo, uint8_t what) {
    if (!g_cal.active) {
        Serial.println("-> Kalibrierung laeuft nicht ('cal' startet sie)");
        return CAL_ST_INACTIVE;
    }

    uint8_t st = CAL_ST_OK;
    int p = calReadPos(servo);
    if (p == -1) {
        p  = g_cal.pos;
        st = CAL_ST_NOSERVO;
        Serial.println("   (Servo nicht lesbar - Sollposition uebernommen)");
    }
    g_cal.pos = p;   // im Free-Mode nachziehen, damit Schritte hier weitergehen

    switch (what) {
        case CAL_ACT_CENTER: g_cal.center = p; g_cal.haveCenter = true;
                             Serial.printf("-> Mitte  = %d\n", p);  break;
        case CAL_ACT_LEFT:   g_cal.left   = p; g_cal.haveLeft   = true;
                             Serial.printf("-> Links  = %d\n", p);  break;
        case CAL_ACT_RIGHT:  g_cal.right  = p; g_cal.haveRight  = true;
                             Serial.printf("-> Rechts = %d\n", p);  break;
        default: return CAL_ST_REJECTED;
    }

    if (g_cal.haveCenter && g_cal.haveLeft && g_cal.haveRight) {
        Serial.println("   Alle drei Marken gesetzt - 'calsave' speichert.");
    }
    return st;
}

void calSetStep(int ticks) {
    g_cal.step = constrain(ticks, CAL_STEP_MIN, CAL_STEP_MAX);
    Serial.printf("-> Schrittweite = %d Ticks (~%.1f grad)\n",
                  g_cal.step, g_cal.step * SERVO_DEG_PER_TICK);
}

// Torque aus: die Lenkung laesst sich von Hand bis an den Anschlag bewegen,
// ganz ohne Servokraft. Beim Wiedereinschalten wird die Ist-Stellung als Ziel
// gesetzt, sonst zieht der Servo auf sein altes Ziel zurueck.
uint8_t calSetFree(SCSCL* servo, bool freeIt) {
    if (!g_cal.active) {
        Serial.println("-> Kalibrierung laeuft nicht ('cal' startet sie)");
        return CAL_ST_INACTIVE;
    }
    g_cal.freeMode = freeIt;
    servo->EnableTorque(SERVO_ID, freeIt ? 0 : 1);

    if (freeIt) {
        Serial.println("-> Torque AUS: Lenkung von Hand stellen, dann calm/call/calr.");
        return CAL_ST_OK;
    }

    int p = calReadPos(servo);
    if (p == -1) {
        Serial.println("-> Torque AN, aber Position nicht lesbar.");
        return CAL_ST_NOSERVO;
    }
    g_cal.pos = p;
    servoManualPos = p;
    servo->WritePosEx(SERVO_ID, p, SERVO_SPEED_CALIB, SERVO_ACC);
    Serial.printf("-> Torque AN, haelt bei %d.\n", p);
    return CAL_ST_OK;
}

// Zur gemerkten Mitte fahren (oder zur gespeicherten, solange keine gesetzt ist).
uint8_t calGoCenter(SCSCL* servo) {
    if (!g_cal.active) {
        Serial.println("-> Kalibrierung laeuft nicht ('cal' startet sie)");
        return CAL_ST_INACTIVE;
    }
    if (g_cal.freeMode) calSetFree(servo, false);

    int target = g_cal.haveCenter ? g_cal.center : centerLimit;
    g_cal.pos = target;
    servoManualPos = target;
    servo->WritePosEx(SERVO_ID, target, SERVO_SPEED_CALIB, SERVO_ACC);
    Serial.printf("-> faehrt auf Mitte %d (%s)\n", target,
                  g_cal.haveCenter ? "neu gesetzt" : "gespeichert");
    return CAL_ST_OK;
}

// Prueft die drei Marken auf Plausibilitaet und schreibt sie ins NVS. Erst hier
// aendern sich die aktiven Limits.
bool calSave(SCSCL* servo) {
    if (!g_cal.active) {
        Serial.println("-> Kalibrierung laeuft nicht ('cal' startet sie)");
        return false;
    }
    if (!g_cal.haveCenter || !g_cal.haveLeft || !g_cal.haveRight) {
        Serial.printf("-> NICHT gespeichert, es fehlt:%s%s%s\n",
                      g_cal.haveCenter ? "" : " Mitte (calm)",
                      g_cal.haveLeft   ? "" : " Links (call)",
                      g_cal.haveRight  ? "" : " Rechts (calr)");
        return false;
    }

    int lo = min(g_cal.left, g_cal.right);
    int hi = max(g_cal.left, g_cal.right);
    if (hi - lo < CAL_MIN_SPAN) {
        Serial.printf("-> NICHT gespeichert: Anschlaege nur %d Ticks auseinander "
                      "(min. %d).\n", hi - lo, CAL_MIN_SPAN);
        return false;
    }
    if (g_cal.center <= lo || g_cal.center >= hi) {
        Serial.printf("-> NICHT gespeichert: Mitte %d liegt nicht zwischen den "
                      "Anschlaegen %d..%d.\n", g_cal.center, lo, hi);
        return false;
    }

    leftLimit   = g_cal.left;
    rightLimit  = g_cal.right;
    centerLimit = g_cal.center;
    // Der alte Trim bezog sich auf die alte Mitte und ist damit hinfaellig.
    trimOffset  = 0;
    softwareCenterPos = centerLimit;

    bool ok = true;
    ok &= prefs.putInt("lLim10", leftLimit)   > 0;
    ok &= prefs.putInt("rLim10", rightLimit)  > 0;
    ok &= prefs.putInt("cLim10", centerLimit) > 0;
    prefs.putInt("offset10", trimOffset);   // 0 schreibt NVS ggf. gar nicht neu

    g_cal.active   = false;
    g_cal.freeMode = false;

    servo->EnableTorque(SERVO_ID, 1);
    servo->WritePosEx(SERVO_ID, softwareCenterPos, SERVO_SPEED_CALIB, SERVO_ACC);
    servoManualPos = softwareCenterPos;

    Serial.printf("-> %s: Mitte=%d  Links=%d  Rechts=%d  (Trim auf 0 zurueckgesetzt)\n",
                  ok ? "gespeichert" : "FEHLER beim Speichern (NVS)",
                  centerLimit, leftLimit, rightLimit);
    Serial.printf("   Hub: links %+d Ticks (~%.0f grad), rechts %+d Ticks (~%.0f grad),\n",
                  leftLimit - centerLimit, (leftLimit - centerLimit) * SERVO_DEG_PER_TICK,
                  rightLimit - centerLimit, (rightLimit - centerLimit) * SERVO_DEG_PER_TICK);
    Serial.println("   davon nutzt die Lenkung 80 % je Seite (+-100 %).");
    return ok;
}

void calAbort(SCSCL* servo) {
    g_cal.active   = false;
    g_cal.freeMode = false;
    servo->EnableTorque(SERVO_ID, 1);
    servo->WritePosEx(SERVO_ID, softwareCenterPos, SERVO_SPEED_CALIB, SERVO_ACC);
    servoManualPos = softwareCenterPos;
    Serial.printf("-> Kalibrierung abgebrochen. Gespeicherte Werte bleiben "
                  "(Mitte=%d L=%d R=%d), faehrt auf Mitte %d.\n",
                  centerLimit, leftLimit, rightLimit, softwareCenterPos);
}

// Eine Aktion ausfuehren. Gemeinsamer Einstieg fuer USB-Konsole und Jetson-Link,
// damit beide Wege exakt dasselbe tun. arg ist die Schrittweite (0 = aktuelle)
// bzw. bei CAL_ACT_STEP die neue Schrittweite.
uint8_t calHandleAction(SCSCL* servo, uint8_t action, uint8_t arg) {
    switch (action) {
        case CAL_ACT_START:  calStart(servo);                       return CAL_ST_OK;
        case CAL_ACT_MINUS:  return calMove(servo, -(arg ? (int)arg : g_cal.step));
        case CAL_ACT_PLUS:   return calMove(servo,  (arg ? (int)arg : g_cal.step));
        case CAL_ACT_CENTER:
        case CAL_ACT_LEFT:
        case CAL_ACT_RIGHT:  return calSetMark(servo, action);
        case CAL_ACT_SAVE:   return calSave(servo) ? CAL_ST_SAVED : CAL_ST_REJECTED;
        case CAL_ACT_ABORT:  calAbort(servo);                       return CAL_ST_OK;
        case CAL_ACT_FREE:   return calSetFree(servo, true);
        case CAL_ACT_HOLD:   return calSetFree(servo, false);
        case CAL_ACT_GOTO_C: return calGoCenter(servo);
        case CAL_ACT_STEP:   calSetStep(arg);                       return CAL_ST_OK;
        case CAL_ACT_STATUS: calPrintStatus(servo);                 return CAL_ST_OK;
        default:
            Serial.printf("-> unbekannte Kalibrier-Aktion 0x%02X\n", action);
            return CAL_ST_REJECTED;
    }
}

// ==========================================
// 9. KOMMUNIKATION (Jetson Bridge) - Core 1
// ==========================================

// Alle Mehrbyte-Werte im Protokoll sind Big-Endian, wie im bestehenden
// CMD_MOTOR/CMD_SERVO. Floats werden als int32 x1000 uebertragen.
static inline int32_t readI32BE(const uint8_t* b) {
    return ((int32_t)b[0] << 24) | ((int32_t)b[1] << 16) |
           ((int32_t)b[2] << 8)  |  (int32_t)b[3];
}
static inline void writeI32BE(uint8_t* b, int32_t v) {
    b[0] = (uint8_t)(v >> 24); b[1] = (uint8_t)(v >> 16);
    b[2] = (uint8_t)(v >> 8);  b[3] = (uint8_t)v;
}
static inline uint32_t readU32BE(const uint8_t* b) {
    return ((uint32_t)b[0] << 24) | ((uint32_t)b[1] << 16) |
           ((uint32_t)b[2] << 8)  |  (uint32_t)b[3];
}
static inline void writeU32BE(uint8_t* b, uint32_t v) {
    b[0] = (uint8_t)(v >> 24); b[1] = (uint8_t)(v >> 16);
    b[2] = (uint8_t)(v >> 8);  b[3] = (uint8_t)v;
}
static inline void writeI64BE(uint8_t* b, int64_t v) {
    for (int i = 0; i < 8; i++) b[i] = (uint8_t)(v >> (56 - 8 * i));
}

// --- Zeitbasis des Links ---
// esp_timer_get_time() zaehlt Mikrosekunden seit Boot als int64 und laeuft -
// anders als micros() - nicht alle 71 Minuten ueber. Auf der Leitung stehen im
// Rahmenstempel nur die unteren 32 Bit (spart 4 Byte je Paket); CMD_TIME_RSP
// liefert regelmaessig den vollen Wert, an dem die Bridge den Ueberlauf
// wieder auspacken kann.
static inline int64_t nowUs() { return esp_timer_get_time(); }

// Sendezeitstempel an alle ausgehenden Pakete haengen (Rahmen 0xA6 statt
// 0xA5). Default aus, siehe START_BYTE_TS.
bool g_stampTx = false;

// --- Fahrtelemetrie ---
// Abstand zwischen zwei CMD_TELEMETRY in Millisekunden, 0 = aus. Der Jetson
// stellt das mit CMD_TELEM_RATE ein, die USB-Konsole mit "tel<ms>".
//
// Untergrenze TELEMETRY_MS_MIN (10 ms) = der Takt des Motortasks. Schneller
// zu senden waere sinnlos: Position und Geschwindigkeit entstehen beide in
// diesem Takt, und beide sind bei jedem Paket frisch. Wie fein die
// Geschwindigkeit aufgeloest ist, haengt nicht am Sendetakt, sondern am
// Messfenster - siehe g_speedWindow.
//
// Bewusst nicht im NVS: eine Bridge, die den Takt braucht, stellt ihn beim
// Verbinden selbst ein, und ein ESP ohne Gegenstelle soll nicht ins Leere
// senden.
uint16_t g_telemetryMs = 0;
constexpr uint16_t TELEMETRY_MS_MIN = 10;

bool savePidParams();   // Definition weiter unten, wird von CMD_PID_SAVE genutzt

// --- Mitschnitt des Jetson-Links auf der USB-Konsole ---
// 0 = aus, 1 = dekodierte Pakete, 2 = zusaetzlich jedes Rohbyte.
// Level 1 ist Default. Achtung: ein Motor-Heartbeat im 100-ms-Takt erzeugt
// 10 Zeilen/s - beim Dauerfahren mit "dbg0" abschalten.
uint8_t g_linkDebug = 1;

// Plotter-Modus: gibt zyklisch reine "name:wert"-Zeilen aus, wie sie der
// Serial Plotter der Arduino IDE erwartet. Solange er laeuft, schweigt der
// Link-Mitschnitt - jede Fremdzeile zerlegt sonst die Kurve.
bool g_plotMode = false;
// Vom Befehl "gp<grad>" gesetzt: Plotter laeuft nur fuer die Dauer einer Fahrt
// und schaltet sich danach selbst ab. Der Serial Plotter der Arduino IDE hat
// kein Eingabefeld - so laesst sich eine Fahrt im Monitor ausloesen und die
// Kurve danach im Plotter ansehen, ohne dazwischen etwas tippen zu muessen.
bool g_plotAuto = false;

static const char* cmdName(uint8_t cmd) {
    switch (cmd) {
        case CMD_MOTOR:         return "MOTOR";
        case CMD_SERVO:         return "SERVO";
        case CMD_LED:           return "LED";
        case CMD_CALIBRATE:     return "CALIBRATE";
        case CMD_CAL:           return "CAL";
        case CMD_CAL_RSP:       return "CAL_RSP";
        case CMD_TORQUE:        return "TORQUE";
        case CMD_TRIM:          return "TRIM";
        case CMD_PID_SET:       return "PID_SET";
        case CMD_PID_GET:       return "PID_GET";
        case CMD_PID_SAVE:      return "PID_SAVE";
        case CMD_MOVE:          return "MOVE";
        case CMD_MOVE_ABORT:    return "MOVE_ABORT";
        case CMD_PROGRESS:      return "PROGRESS";
        case CMD_BATTERY:       return "BATTERY";
        case CMD_EMERGENCY:     return "EMERGENCY";
        case CMD_BUTTON:        return "BUTTON";
        case CMD_PID_RSP:       return "PID_RSP";
        case CMD_PID_SAVED:     return "PID_SAVED";
        case CMD_MOVE_DONE:     return "MOVE_DONE";
        case CMD_PROGRESS_RSP:  return "PROGRESS_RSP";
        case CMD_BATTERY_RSP:   return "BATTERY_RSP";
        case CMD_BATTERY_WARN:  return "BATTERY_WARN";
        case CMD_TIME_SYNC:     return "TIME_SYNC";
        case CMD_TIME_RSP:      return "TIME_RSP";
        case CMD_STAMP_MODE:    return "STAMP_MODE";
        case CMD_STAMP_RSP:     return "STAMP_RSP";
        case CMD_TELEM_RATE:    return "TELEM_RATE";
        case CMD_TELEMETRY:     return "TELEMETRY";
        default:                return "???";
    }
}

static const char* calActionName(uint8_t act) {
    switch (act) {
        case CAL_ACT_START:  return "start";
        case CAL_ACT_MINUS:  return "schritt-";
        case CAL_ACT_PLUS:   return "schritt+";
        case CAL_ACT_CENTER: return "mitte setzen";
        case CAL_ACT_LEFT:   return "links setzen";
        case CAL_ACT_RIGHT:  return "rechts setzen";
        case CAL_ACT_SAVE:   return "speichern";
        case CAL_ACT_ABORT:  return "abbrechen";
        case CAL_ACT_FREE:   return "torque frei";
        case CAL_ACT_HOLD:   return "torque halten";
        case CAL_ACT_GOTO_C: return "mitte anfahren";
        case CAL_ACT_STEP:   return "schrittweite";
        case CAL_ACT_STATUS: return "status";
        default:             return "???";
    }
}

class JetsonComms {
private:
    HardwareSerial* serialPort;
    SCSCL* servo;

    unsigned long stateTime;
    uint8_t buffer[24];
    int bufIndex = 0;
    unsigned long linkBaud = 115200;

    enum State { WAITING_START, WAITING_CMD, READING_STAMP, READING_DATA };
    State currentState = WAITING_START;
    uint8_t currentCmd = 0;
    uint8_t dataLength = 0;

    // --- Zeitstempel des gerade laufenden Empfangs ---
    bool     frameStamped = false;   // Rahmen kam als 0xA6 herein
    uint8_t  stampBuf[4];
    int      stampIndex = 0;
    uint32_t frameStampUs = 0;       // Sendestempel der Gegenstelle (nur 0xA6)
    int64_t  frameRxUs    = 0;       // ESP-Uhr beim letzten Byte des Rahmens

    // Link-Statistik fuer die Fehlersuche
    uint32_t rxPackets  = 0;   // vollstaendig empfangen und ausgefuehrt
    uint32_t rxUnknown  = 0;   // unbekanntes CMD-Byte verworfen
    uint32_t rxTimeouts = 0;   // Paket blieb unvollstaendig
    uint32_t rxStray    = 0;   // Bytes ausserhalb eines Pakets (z.B. ASCII)
    uint32_t txPackets  = 0;
    unsigned long lastRxMs = 0;
    uint32_t syncRequests = 0;    // beantwortete CMD_TIME_SYNC
    int64_t  lastSyncUs   = 0;    // ESP-Uhr der letzten Antwort

    // Unterdrueckung identischer Wiederholungen im Mitschnitt. Ein Heartbeat
    // im 100-ms-Takt erzeugt sonst 10 nutzlose Zeilen/s und verdeckt genau die
    // Pakete, auf die man wartet.
    uint8_t  lastLogCmd = 0xFF;
    uint8_t  lastLogBuf[24];
    uint8_t  lastLogLen = 0;
    uint32_t repeatCount = 0;
    unsigned long repeatSince = 0;

public:
    JetsonComms(HardwareSerial* s, SCSCL* sv) : serialPort(s), servo(sv) {}

    void begin(unsigned long baud = 115200) {
        serialPort->begin(baud, SERIAL_8N1, PIN_JETSON_RX, PIN_JETSON_TX);
        linkBaud  = baud;
        stateTime = millis();
    }

    void sendButtonEvent() {
        uint8_t p[3] = {START_BYTE, CMD_BUTTON, 0x01};   // 0x01 = Pressed
        sendPacket(p, sizeof(p));
    }

    void sendMoveDone(const MoveResult& r) {
        uint8_t p[8] = {START_BYTE, CMD_MOVE_DONE, r.id, r.status};
        writeI32BE(&p[4], r.finalDeg10);
        sendPacket(p, sizeof(p));
    }

    void sendProgress() {
        MoveState m = getMoveState();
        uint8_t p[13] = {START_BYTE, CMD_PROGRESS_RSP, m.id,
                         (uint8_t)(m.active ? 1 : 0), m.progress};
        writeI32BE(&p[5], countsToDeg10(g_encCount));
        writeI32BE(&p[9], countsToDeg10(m.targetCnt));
        sendPacket(p, sizeof(p));
    }

    void sendBattery(uint8_t cmd) {
        uint8_t p[8] = {START_BYTE, cmd};
        writeI32BE(&p[2], (int32_t)lroundf(battPackV * 1000.0f));
        int16_t cellmV = (int16_t)lroundf(battCellV * 1000.0f);
        p[6] = (uint8_t)(cellmV >> 8);
        p[7] = (uint8_t)cellmV;
        sendPacket(p, sizeof(p));
    }

    void sendPidParams() {
        PidParams pid = getPid();
        uint8_t p[14] = {START_BYTE, CMD_PID_RSP};
        writeI32BE(&p[2],  (int32_t)lroundf(pid.kp * 1000.0f));
        writeI32BE(&p[6],  (int32_t)lroundf(pid.ki * 1000.0f));
        writeI32BE(&p[10], (int32_t)lroundf(pid.kd * 1000.0f));
        sendPacket(p, sizeof(p));
    }

    // Vollstaendiger Zustand der manuellen Kalibrierung. Geht nach jeder
    // CMD_CAL-Aktion raus, damit die Bridge eine Anzeige bauen kann, ohne
    // selbst mitzuzaehlen.
    void sendCalState(uint8_t status) {
        uint8_t flags = (g_cal.haveCenter ? 0x01 : 0) |
                        (g_cal.haveLeft   ? 0x02 : 0) |
                        (g_cal.haveRight  ? 0x04 : 0) |
                        (g_cal.freeMode   ? 0x08 : 0);
        uint8_t p[13] = {START_BYTE, CMD_CAL_RSP,
                         (uint8_t)(g_cal.active ? 1 : 0), flags, status};
        auto put16 = [&](int idx, int v) {
            p[idx]     = (uint8_t)((uint16_t)(int16_t)v >> 8);
            p[idx + 1] = (uint8_t)((uint16_t)(int16_t)v);
        };
        put16(5,  g_cal.pos);
        put16(7,  g_cal.haveCenter ? g_cal.center : centerLimit);
        put16(9,  g_cal.haveLeft   ? g_cal.left   : leftLimit);
        put16(11, g_cal.haveRight  ? g_cal.right  : rightLimit);
        sendPacket(p, sizeof(p));
    }

    void sendPidSaved(bool ok) {
        uint8_t p[3] = {START_BYTE, CMD_PID_SAVED, (uint8_t)(ok ? 0x00 : 0x01)};
        sendPacket(p, sizeof(p));
    }

    // Antwort auf CMD_TIME_SYNC. Traegt beide Zeitpunkte der ESP-Seite als
    // volle int64-Mikrosekunden:
    //   t_rx = letztes Byte der Anfrage empfangen
    //   t_tx = letztes Byte dieser Antwort auf der Leitung
    // Mit t1/t4 der Jetson-Seite (jeweils ebenfalls letztes Byte) ergibt sich
    // der Uhrenversatz zu ((t2-t1) + (t3-t4)) / 2 und die Umlaufzeit zu
    // (t4-t1) - (t3-t2). Siehe docs/JETSON_BRIDGE.md.
    //
    // Dieses Paket geht bewusst NIE als 0xA6 raus: sein t_tx steht schon mit
    // voller Breite in der Nutzlast, ein zusaetzlicher 32-Bit-Stempel waere
    // nur Redundanz - und die Bridge braucht die Zeitsynchronisation, bevor
    // sie ueberhaupt entscheiden kann, ob sie Stempel haben will.
    void sendTimeSync(uint8_t seq, int64_t rxUs) {
        uint8_t p[19] = {START_BYTE, CMD_TIME_RSP, seq};
        writeI64BE(&p[3], rxUs);
        // p[11..18] (t_tx) fuellt sendPacket so spaet wie moeglich selbst.
        sendPacket(p, sizeof(p), 11);
        syncRequests++;
        lastSyncUs = nowUs();
    }

    // Fahrzustand: wo die Ausgangswelle steht, wie schnell sie dreht, was an
    // der Bruecke anliegt und wie viel Strom fliesst.
    //
    // Tempo in 1/10 Grad pro Sekunde, gleiche Einheit wie die Wege in
    // CMD_MOVE. Der Motortask fuehrt intern U/min - Faktor 60 (eine Umdrehung
    // sind 3600 Zehntelgrad, eine Minute 60 Sekunden).
    void sendTelemetry() {
        uint8_t p[14] = {START_BYTE, CMD_TELEMETRY};

        // Position und Tempo sind vorzeichenbehaftet: der Encoder-Delta im
        // Motortask ist ein signed long, rueckwaerts liefert also negative
        // Werte. writeI32BE schiebt arithmetisch, das Zweierkomplement kommt
        // damit unveraendert auf die Leitung.
        writeI32BE(&p[2], countsToDeg10(g_encCount));
        writeI32BE(&p[6], (int32_t)lroundf(g_rpm * 60.0f));

        // Duty ebenso: minus heisst rueckwaerts.
        int16_t duty = g_dutySigned;
        p[10] = (uint8_t)((uint16_t)duty >> 8);
        p[11] = (uint8_t)duty;

        // Der Strom dagegen ist immer positiv - der VNH5019 meldet auf CS nur
        // den Betrag, nicht die Richtung. Wer sie braucht, liest sie am Duty ab.
        int32_t mA = (int32_t)lroundf(readMotorCurrentA() * 1000.0f);
        mA = constrain(mA, 0L, 32767L);
        p[12] = (uint8_t)((uint16_t)(int16_t)mA >> 8);
        p[13] = (uint8_t)(int16_t)mA;

        sendPacket(p, sizeof(p));
    }

    void sendStampMode() {
        uint8_t p[3] = {START_BYTE, CMD_STAMP_RSP, (uint8_t)(g_stampTx ? 1 : 0)};
        sendPacket(p, sizeof(p));
    }

    void printLinkStats() {
        Serial.printf("Link IO%d(RX)/IO%d(TX) @115200 | debug=%u\n",
                      PIN_JETSON_RX, PIN_JETSON_TX, g_linkDebug);
        Serial.printf("  RX: %lu Pakete, %lu unbekannt, %lu unvollstaendig, %lu Streubytes\n",
                      (unsigned long)rxPackets, (unsigned long)rxUnknown,
                      (unsigned long)rxTimeouts, (unsigned long)rxStray);
        Serial.printf("  TX: %lu Pakete%s\n", (unsigned long)txPackets,
                      g_stampTx ? " (mit Zeitstempel, Rahmen 0xA6)" : "");
        Serial.printf("  Zeit: %lld us seit Boot | %lu Syncs",
                      (long long)nowUs(), (unsigned long)syncRequests);
        if (lastSyncUs) Serial.printf(" | letzter Sync vor %lld ms",
                                      (long long)((nowUs() - lastSyncUs) / 1000));
        Serial.println();
        if (rxPackets == 0 && rxStray == 0) {
            Serial.println("  !! noch NICHTS empfangen - Verkabelung/Baudrate/GND pruefen");
        } else if (lastRxMs) {
            Serial.printf("  letztes Paket vor %lu ms\n", millis() - lastRxMs);
        }
    }

    void process() {
        while (serialPort->available()) {
            uint8_t byte = serialPort->read();

            if (currentState != WAITING_START && (millis() - stateTime > 100)) {
                rxTimeouts++;
                if (g_linkDebug && !g_plotMode) {
                    Serial.printf("[RX] ABBRUCH cmd=0x%02X %s nach %d/%u Byte (>100 ms Pause) - resync\n",
                                  currentCmd, cmdName(currentCmd), bufIndex, dataLength);
                }
                currentState = WAITING_START;
            }
            stateTime = millis();

            switch (currentState) {
                case WAITING_START:
                    if (byte == START_BYTE || byte == START_BYTE_TS) {
                        frameStamped = (byte == START_BYTE_TS);
                        currentState = WAITING_CMD;
                        bufIndex = 0;
                        stampIndex = 0;
                        frameStampUs = 0;
                    } else {
                        // Kein Startbyte: entweder ASCII-Klartext oder Muell nach
                        // einem Sync-Verlust. Nur im Vollmodus einzeln melden.
                        rxStray++;
                        if (g_linkDebug >= 2 && !g_plotMode) {
                            Serial.printf("[RX] sync-suche: 0x%02X%s\n", byte,
                                          (byte >= 32 && byte < 127) ? " (ASCII)" : "");
                        }
                    }
                    break;

                case WAITING_CMD: {
                    currentCmd = byte;
                    int len = payloadLength(currentCmd);
                    if (len < 0) {                  // unbekannter Befehl
                        rxUnknown++;
                        if (g_linkDebug && !g_plotMode) {
                            Serial.printf("[RX] UNBEKANNT cmd=0x%02X - verworfen\n", currentCmd);
                        }
                        currentState = WAITING_START;
                        break;
                    }
                    dataLength = (uint8_t)len;
                    // Bei 0xA6 stehen erst vier Byte Sendestempel, dann die
                    // Nutzlast.
                    currentState = frameStamped ? READING_STAMP : READING_DATA;
                    if (!frameStamped && dataLength == 0) finishFrame();
                    break;
                }

                case READING_STAMP:
                    stampBuf[stampIndex++] = byte;
                    if (stampIndex >= 4) {
                        frameStampUs = readU32BE(stampBuf);
                        currentState = READING_DATA;
                        if (dataLength == 0) finishFrame();
                    }
                    break;

                case READING_DATA:
                    buffer[bufIndex++] = byte;
                    if (bufIndex >= dataLength) finishFrame();
                    break;
            }
        }
    }

private:
    // Rahmen ist vollstaendig: Empfangszeit festhalten, protokollieren,
    // ausfuehren, zurueck auf Sync-Suche.
    //
    // frameRxUs meint das letzte Byte des Rahmens - denselben Bezugspunkt, den
    // die Bridge fuer ihr t1 nimmt. Ungenau ist daran nur, dass process() aus
    // dem loop() heraus liest: das Byte lag da schon eine Weile im
    // Treiberpuffer. Diese Verzoegerung steckt in der gemessenen Umlaufzeit,
    // die Bridge sieht sie also und kann Ausreisser verwerfen.
    void finishFrame() {
        frameRxUs = nowUs();
        logRxPacket();
        executeCommand();
        currentState = WAITING_START;
    }

    // Zeitpunkt, zu dem das letzte Byte eines n Byte langen Pakets die Leitung
    // verlassen hat. 8N1 = 10 Bit pro Byte. Vor dem Aufruf muss der Sendepuffer
    // leer sein, sonst schiebt sich der Rest des Vorgaengers davor.
    int64_t txDoneUs(size_t n) const {
        return nowUs() + (int64_t)n * 10 * 1000000 / (int64_t)linkBaud;
    }

    // Ein Paket rausschicken, mitzaehlen und optional mitschneiden.
    //
    // p enthaelt immer den ungestempelten Rahmen (START_BYTE, CMD, Nutzlast).
    // Ist g_stampTx an, baut sendPacket daraus den 0xA6-Rahmen und schiebt vier
    // Byte Zeitstempel zwischen CMD und Nutzlast.
    //
    // txStampOffset >= 0: an dieser Stelle im *ungestempelten* Rahmen steht ein
    // int64-Feld, das den eigenen Sendezeitpunkt aufnimmt (nur CMD_TIME_RSP).
    // Solche Pakete bekommen nie zusaetzlich einen Rahmenstempel.
    void sendPacket(const uint8_t* p, size_t n, int txStampOffset = -1) {
        const bool stamped = g_stampTx && txStampOffset < 0;
        const bool needsClock = stamped || txStampOffset >= 0;

        uint8_t out[48];
        size_t  k = 0;

        if (stamped) {
            out[k++] = START_BYTE_TS;
            out[k++] = p[1];
            k += 4;                              // Platzhalter fuer den Stempel
            memcpy(&out[k], p + 2, n - 2);
            k += n - 2;
        } else {
            memcpy(out, p, n);
            k = n;
        }

        // Erst den Puffer leerlaufen lassen, dann stempeln: nur so gilt die
        // Rechnung "jetzt + Uebertragungsdauer". Haengt noch eine ASCII-Zeile
        // im Puffer (CMD_EMERGENCY, CMD_TRIM), waere der Stempel sonst zu frueh.
        if (needsClock) {
            serialPort->flush();
            int64_t done = txDoneUs(k);
            if (stamped)              writeU32BE(&out[2], (uint32_t)done);
            if (txStampOffset >= 0)   writeI64BE(&out[txStampOffset], done);
        }

        serialPort->write(out, k);
        txPackets++;
        if (g_linkDebug && !g_plotMode) {
            const uint8_t* payload = &out[k - (n - 2)];   // Nutzlast am Ende
            Serial.printf("[TX] %s", cmdName(p[1]));
            if (stamped) Serial.printf(" t=%lu", (unsigned long)readU32BE(&out[2]));
            for (size_t i = 0; i < n - 2; i++) Serial.printf(" %02X", payload[i]);
            Serial.println();
        }
    }

    // Empfangenes Paket auf der USB-Konsole protokollieren.
    void logRxPacket() {
        rxPackets++;
        lastRxMs = millis();
        if (!g_linkDebug || g_plotMode) return;

        // Byte-identische Wiederholung? Dann nur zaehlen und alle 5 s eine
        // Sammelzeile ausgeben. In dbg2 (Rohbytes) bleibt alles ungefiltert.
        bool same = (currentCmd == lastLogCmd && dataLength == lastLogLen &&
                     memcmp(buffer, lastLogBuf, dataLength) == 0);
        // Zeitsync-Pakete nie zusammenfassen - beim Nachmessen des Links will
        // man jede einzelne Runde sehen, auch wenn die seq zufaellig gleich ist.
        if (currentCmd == CMD_TIME_SYNC) same = false;

        if (same && g_linkDebug < 2) {
            repeatCount++;
            if (millis() - repeatSince >= 5000) {
                Serial.printf("[RX] %-12s  %lux unveraendert in %lu s\n",
                              cmdName(currentCmd), (unsigned long)repeatCount,
                              (millis() - repeatSince) / 1000);
                repeatCount = 0;
                repeatSince = millis();
            }
            return;
        }

        if (repeatCount) {
            Serial.printf("[RX] %-12s  %lux unveraendert\n",
                          cmdName(lastLogCmd), (unsigned long)repeatCount);
            repeatCount = 0;
        }
        lastLogCmd = currentCmd;
        lastLogLen = dataLength;
        memcpy(lastLogBuf, buffer, dataLength);
        repeatSince = millis();

        Serial.printf("[RX] %-12s", cmdName(currentCmd));
        if (frameStamped) Serial.printf(" t=%lu", (unsigned long)frameStampUs);
        if (g_linkDebug >= 2) {
            Serial.print(" raw:");
            for (int i = 0; i < dataLength; i++) Serial.printf(" %02X", buffer[i]);
        }

        switch (currentCmd) {
            case CMD_MOTOR:
                Serial.printf("  dir=%u speed=%u", buffer[0], (buffer[1] << 8) | buffer[2]);
                break;
            case CMD_SERVO:
                Serial.printf("  id=%u lenkung=%d%%", buffer[0],
                              (int16_t)((buffer[1] << 8) | buffer[2]));
                break;
            case CMD_LED:
                Serial.printf("  %s", buffer[0] ? "an" : "aus");
                break;
            case CMD_TRIM:
                Serial.printf("  %s", buffer[0] == 0 ? "links" :
                                      buffer[0] == 1 ? "rechts" : "speichern");
                break;
            case CMD_PID_SET: {
                int32_t raw = readI32BE(&buffer[1]);
                Serial.printf("  param=%u wert=%.3f", buffer[0], raw / 1000.0f);
                break;
            }
            case CMD_MOVE: {
                int32_t d = readI32BE(&buffer[1]);
                Serial.printf("  id=%u um %+.1f grad", buffer[0], d / 10.0f);
                break;
            }
            case CMD_CAL:
                Serial.printf("  %s arg=%u", calActionName(buffer[0]), buffer[1]);
                break;
            case CMD_TIME_SYNC:
                Serial.printf("  seq=%u", buffer[0]);
                break;
            case CMD_STAMP_MODE:
                Serial.printf("  %s", buffer[0] ? "an" : "aus");
                break;
            case CMD_TELEM_RATE: {
                uint16_t ms = (buffer[0] << 8) | buffer[1];
                if (ms) Serial.printf("  alle %u ms (%.1f Hz)", ms, 1000.0f / ms);
                else    Serial.print("  aus");
                break;
            }
            default:
                break;   // Befehle ohne Nutzlast
        }
        Serial.println();
    }

    // Nutzlastlaenge je Befehl. -1 = unbekannt.
    static int payloadLength(uint8_t cmd) {
        switch (cmd) {
            case CMD_MOTOR:      return 3;
            case CMD_SERVO:      return 3;
            case CMD_LED:        return 1;
            case CMD_TRIM:       return 1;
            case CMD_PID_SET:    return 5;
            case CMD_MOVE:       return 5;
            case CMD_CAL:        return 2;
            case CMD_TIME_SYNC:  return 1;
            case CMD_STAMP_MODE: return 1;
            case CMD_TELEM_RATE: return 2;
            case CMD_CALIBRATE:
            case CMD_TORQUE:
            case CMD_PID_GET:
            case CMD_PID_SAVE:
            case CMD_MOVE_ABORT:
            case CMD_PROGRESS:
            case CMD_BATTERY:
            case CMD_EMERGENCY:  return 0;
            default:             return -1;
        }
    }

    void executeCommand() {
        switch (currentCmd) {

        case CMD_EMERGENCY:
            // Aktiv bremsen statt nur ausrollen, bricht auch eine Fahrt ab.
            setMotorCommand(MOTOR_BRAKE, false, DUTY_MAX);
            serialPort->println("ESP: EMERGENCY STOP EXECUTED!");
            break;

        case CMD_MOTOR: {
            bool     reverse = buffer[0];
            uint16_t speed   = (buffer[1] << 8) | buffer[2];   // 0..1023
            // Speed 0 = ausrollen. Bremsen laeuft ueber CMD_EMERGENCY.
            setMotorCommand(speed ? MOTOR_DRIVE : MOTOR_COAST, reverse, speed);
            break;
        }

        case CMD_MOVE: {
            uint8_t id = buffer[0];
            int32_t targetDeg10 = readI32BE(&buffer[1]);
            startMove(id, targetDeg10);
            break;
        }

        case CMD_MOVE_ABORT:
            abortMove();
            break;

        case CMD_PROGRESS:
            sendProgress();
            break;

        case CMD_BATTERY:
            sendBattery(CMD_BATTERY_RSP);
            break;

        // Zeitabgleich. Muss ohne Umweg beantwortet werden: jede Millisekunde,
        // die zwischen Empfang und Antwort vergeht, geht als Unsicherheit in
        // den Uhrenversatz ein. Deshalb hier direkt im Parser-Kontext und
        // nicht ueber die Queue oder den naechsten loop().
        case CMD_TIME_SYNC:
            sendTimeSync(buffer[0], frameRxUs);
            break;

        case CMD_STAMP_MODE:
            g_stampTx = (buffer[0] != 0);
            sendStampMode();
            break;

        // Takt der Fahrtelemetrie setzen. Die sofortige Antwort dient
        // gleichzeitig als Quittung - die Bridge weiss damit, dass der Takt
        // angekommen ist, ohne auf das erste regulaere Paket zu warten.
        case CMD_TELEM_RATE: {
            uint16_t ms = (buffer[0] << 8) | buffer[1];
            g_telemetryMs = ms ? (ms < TELEMETRY_MS_MIN ? TELEMETRY_MS_MIN : ms) : 0;
            sendTelemetry();
            break;
        }

        case CMD_PID_GET:
            sendPidParams();
            break;

        case CMD_PID_SAVE:
            sendPidSaved(savePidParams());
            break;

        case CMD_PID_SET: {
            uint8_t param = buffer[0];
            int32_t raw   = readI32BE(&buffer[1]);
            float   val   = raw / 1000.0f;
            PidParams p = getPid();
            switch (param) {
                case 0: p.kp        = val;                        break;
                case 1: p.ki        = val;                        break;
                case 2: p.kd        = val;                        break;
                case 3: p.iTermLim  = val;                        break;
                case 4: p.maxDuty   = (uint16_t)constrain(raw / 1000, 0L, (long)DUTY_MAX); break;
                case 5: p.tolDeg10  = raw / 1000;                 break;
                case 6: p.settleMs  = (uint32_t)(raw / 1000);     break;
                case 7: p.timeoutMs = (uint32_t)(raw / 1000);     break;
                case 8: p.minDuty   = (uint16_t)constrain(raw / 1000, 0L, (long)DUTY_MAX); break;
                default: return;   // unbekannter Parameter, still verwerfen
            }
            setPid(p);
            break;
        }

        case CMD_SERVO: {
            uint8_t id = buffer[0];
            int16_t steerPct = (buffer[1] << 8) | buffer[2];

            // Waehrend der Kalibrierung darf kein Lenkbefehl dazwischenfunken -
            // er wuerde die von Hand angefahrene Stellung sofort verwerfen.
            if (g_cal.active) {
                static uint32_t lastHint = 0;
                if (millis() - lastHint > 2000) {
                    lastHint = millis();
                    Serial.println("ESP: Lenkbefehl ignoriert - Kalibrierung laeuft.");
                }
                break;
            }

            // --- Dynamische Hub-Begrenzung auf 80% ---
            const float MAX_THROW_FACTOR = 1.0f;

            int maxDistRight = softwareCenterPos - rightLimit;
            int maxDistLeft  = leftLimit - softwareCenterPos;

            int safeRight = softwareCenterPos - (maxDistRight * MAX_THROW_FACTOR);
            int safeLeft  = softwareCenterPos + (maxDistLeft  * MAX_THROW_FACTOR);

            int physicalPos = softwareCenterPos;

            if (steerPct < -100) steerPct = -100;
            if (steerPct >  100) steerPct =  100;

            if (steerPct > 0) {
                physicalPos = map(steerPct, 0, 100, softwareCenterPos, safeLeft);
            } else if (steerPct < 0) {
                physicalPos = map(steerPct, -100, 0, safeRight, softwareCenterPos);
            }

            servo->WritePosEx(id, physicalPos, SERVO_SPEED_FAST, SERVO_ACC);
            break;
        }

        case CMD_LED:
            digitalWrite(PIN_LED, buffer[0] ? HIGH : LOW);
            break;

        // Startet die manuelle Kalibrierung. Das automatische Antasten gibt es
        // nicht mehr - der SC09 kann sein Drehmoment nicht begrenzen.
        // Die eigentlichen Schritte laufen ueber CMD_CAL.
        case CMD_CALIBRATE:
            calStart(servo);
            sendCalState(CAL_ST_OK);
            break;

        case CMD_CAL:
            sendCalState(calHandleAction(servo, buffer[0], buffer[1]));
            break;

        case CMD_TORQUE:
            printTorque(servo);
            break;

        case CMD_TRIM: {
            uint8_t action = buffer[0];
            if (action == 0x00) {          // Trim -8 Ticks
                trimOffset        -= 8;
                softwareCenterPos  = centerLimit + trimOffset;
                servo->WritePosEx(SERVO_ID, softwareCenterPos, SERVO_SPEED_FAST, SERVO_ACC);
                serialPort->printf("ESP: Trim L | Offset: %d | Pos: %d\n", trimOffset, softwareCenterPos);
            } else if (action == 0x01) {   // Trim +8 Ticks
                trimOffset        += 8;
                softwareCenterPos  = centerLimit + trimOffset;
                servo->WritePosEx(SERVO_ID, softwareCenterPos, SERVO_SPEED_FAST, SERVO_ACC);
                serialPort->printf("ESP: Trim R | Offset: %d | Pos: %d\n", trimOffset, softwareCenterPos);
            } else if (action == 0x02) {   // Offset dauerhaft speichern
                prefs.putInt("offset10", trimOffset);
                serialPort->printf("ESP: Trim-Offset (%d) im Flash gespeichert!\n", trimOffset);
            }
            break;
        }
        }
    }
};

// ==========================================
// 10. USB-DEBUG-KONSOLE (Core 1)
// ==========================================

MotorDriver planetaryMotor(PIN_MOTOR_PWM, PIN_MOTOR_INA, PIN_MOTOR_INB);
SCSCL sc09Servo;
JetsonComms jetson(&Serial1, &sc09Servo);

String cmdBuf;
uint8_t nextLocalMoveId = 1;   // IDs fuer Fahrten aus der USB-Konsole

void printPidParams() {
    PidParams p = getPid();
    Serial.printf("PID  Kp=%.3f  Ki=%.3f  Kd=%.3f\n", p.kp, p.ki, p.kd);
    Serial.printf("     iLim=%.0f  maxDuty=%u  minDuty=%u  tol=%.1f grad (%ld counts)\n",
                  p.iTermLim, p.maxDuty, p.minDuty, p.tolDeg10 / 10.0f,
                  (long)deg10ToCounts(p.tolDeg10));
    Serial.printf("     settle=%lu ms  timeout=%lu ms\n",
                  (unsigned long)p.settleMs, (unsigned long)p.timeoutMs);
}

// Eine Zeile im Format des Arduino-Serial-Plotters: "name:wert" durch
// Leerzeichen getrennt.
//
// Beide Winkel sind RELATIV zum Fahrtbeginn, nicht absolut. Absolut waeren es
// Werte um 1900 Grad, auf die der Plotter die Y-Achse skaliert - die eigentliche
// 90-Grad-Bewegung waere dann 5 % der Achshoehe und saehe aus wie eine flache
// Linie. Relativ laeuft ist von 0 auf ziel, und duty in Prozent liegt in
// derselben Groessenordnung.
void plotTick() {
    static uint32_t lastPlot = 0;
    if (!g_plotMode) return;
    uint32_t now = millis();
    if (now - lastPlot < 20) return;   // 50 Hz
    lastPlot = now;

    MoveState m = getMoveState();
    Serial.printf("ziel:%.1f ist:%.1f duty:%.1f\n",
                  countsToDeg10(m.targetCnt - m.startCnt) / 10.0f,
                  countsToDeg10(g_encCount   - m.startCnt) / 10.0f,
                  (g_dutySigned * 100.0f) / DUTY_MAX);
}

// Schreibt den kompletten Parametersatz ins NVS. Gibt false zurueck, sobald
// ein Schluessel nicht geschrieben werden konnte (volle oder defekte Partition).
// NVS verwirft Schreibvorgaenge mit unveraendertem Wert selbst, mehrfaches
// Speichern gleicher Werte kostet also keine Flash-Zyklen.
bool savePidParams() {
    PidParams p = getPid();
    bool ok = true;
    ok &= prefs.putFloat("kp",     p.kp)        > 0;
    ok &= prefs.putFloat("ki",     p.ki)        > 0;
    ok &= prefs.putFloat("kd",     p.kd)        > 0;
    ok &= prefs.putFloat("ilim",   p.iTermLim)  > 0;
    ok &= prefs.putUShort("mduty", p.maxDuty)   > 0;
    ok &= prefs.putUShort("mind",  p.minDuty)   > 0;
    ok &= prefs.putInt("tol",      p.tolDeg10)  > 0;
    ok &= prefs.putUInt("settle",  p.settleMs)  > 0;
    ok &= prefs.putUInt("tmo",     p.timeoutMs) > 0;
    return ok;
}

// Fehlende Schluessel werden per isKey() abgefangen. Ohne das protokolliert
// ESP-IDF bei jedem noch nie gespeicherten Wert eine [E]-Zeile ("nvs_get_blob"),
// die beim Debuggen wie ein echter Fehler aussieht - der Default greift aber.
void loadPidParams() {
    PidParams p;   // Defaults aus der Struktur
    if (prefs.isKey("kp"))     p.kp        = prefs.getFloat("kp",     p.kp);
    if (prefs.isKey("ki"))     p.ki        = prefs.getFloat("ki",     p.ki);
    if (prefs.isKey("kd"))     p.kd        = prefs.getFloat("kd",     p.kd);
    if (prefs.isKey("ilim"))   p.iTermLim  = prefs.getFloat("ilim",   p.iTermLim);
    if (prefs.isKey("mduty"))  p.maxDuty   = prefs.getUShort("mduty", p.maxDuty);
    if (prefs.isKey("mind"))   p.minDuty   = prefs.getUShort("mind",  p.minDuty);
    if (prefs.isKey("tol"))    p.tolDeg10  = prefs.getInt("tol",      p.tolDeg10);
    if (prefs.isKey("settle")) p.settleMs  = prefs.getUInt("settle",  p.settleMs);
    if (prefs.isKey("tmo"))    p.timeoutMs = prefs.getUInt("tmo",     p.timeoutMs);
    setPid(p);
}

void printHelp() {
    Serial.println("--- Motor (offen) ---");
    Serial.println("  f<0-255> r<0-255> b<0-255>  vor/rueck/bremsen");
    Serial.println("  c        coast            z  Encoder auf 0");
    Serial.println("  e        Telemetrie (Counts, Grad, RPM, Duty, Strom)");
    Serial.println("  i        Motorstrom    ic<mV/A> skalieren  iz  Nullpunkt");
    Serial.println("--- Position ---");
    Serial.println("  g<grad>  um X Grad weiterdrehen (relativ, z.B. g90.0 / g-45)");
    Serial.println("  gp<grad> dito, plottet automatisch bis zum Fahrtende");
    Serial.println("  q        laufende Fahrt abbrechen");
    Serial.println("  w        Fortschritt der Fahrt");
    Serial.println("--- PID ---");
    Serial.println("  kp<f> ki<f> kd<f>          Regelparameter");
    Serial.println("  kl<f> Anti-Windup-Limit    km<0-1023> max. Duty");
    Serial.println("  ka<0-1023> Anlauf-Duty     kt<grad> Zielfenster");
    Serial.println("  kn<ms> Verweilzeit         kx<ms> Timeout");
    Serial.println("  pid   anzeigen             pids  im Flash speichern");
    Serial.println("  p     Plotter an/aus (Arduino Serial Plotter, 50 Hz)");
    Serial.println("--- Jetson-Link (IO10 RX / IO11 TX) ---");
    Serial.println("  dbg      Statistik    dbg0 aus  dbg1 Pakete  dbg2 +Rohbytes");
    Serial.println("  ts       Uhr + Sync-Status   ts0/ts1  Sendezeitstempel aus/an");
    Serial.println("  tel      Fahrtelemetrie   tel<ms> Takt setzen (tel0 = aus)");
    Serial.println("  sw       Messfenster Tempo sw<n> in Vielfachen von 10 ms");
    Serial.println("  o        LED umschalten (Taster meldet sich als [TX] BUTTON)");
    Serial.println("--- Batterie ---");
    Serial.println("  v        jetzt messen      vc<faktor>  Teiler kalibrieren");
    Serial.println("  vd       Pin-Diagnose (Akku ab): prueft den Teiler am ADC-Eingang");
    Serial.println("  vm       Live-Ausgabe zum Wackeltest  vm<ms> Takt  vm0 aus");
    Serial.println("--- Lenkung: manuelle Kalibrierung ---");
    Serial.println("  cal      starten / Status (x = dasselbe)");
    Serial.println("  + / -    ein Schritt (auch '+50' = einmalig 50 Ticks)");
    Serial.println("  caln<t>  Schrittweite in Ticks (Default 10, ~2,9 grad)");
    Serial.println("  calm     Mitte  call  linker Anschlag  calr  rechter Anschlag");
    Serial.println("  calfree  Torque aus (von Hand stellen)   calhold  wieder halten");
    Serial.println("  calgo    Mitte anfahren  calsave  speichern  calq  abbrechen");
    Serial.println("--- Lenkung: Betrieb ---");
    Serial.println("  t Torque  a/d Trim Pos-/Pos+  s Trim speichern");
    Serial.println("  j/l Pos-/Pos+  m Mitte  sv Diagnose  sv<0-1023> Pos  sve0/1 Torque");
    Serial.println("  svs Protokoll-Scan  svpos Positions-Modus erzwingen (gegen Wheel-Mode)");
    Serial.println("  svpin<rx>,<tx> Serial2-Pins tauschen (z.B. svpin17,18)");
    Serial.println("  svbaud<n> Baud (Oszi)  svtx 0x55-Dauersignal an TX (Oszi-Test)");
    Serial.println("  h        diese Hilfe");
}

void handleDebugCommand(String cmd) {
    cmd.trim();
    if (cmd.length() == 0) return;

    // --- Mehrbuchstabige Befehle zuerst ---
    if (cmd.startsWith("pids")) {
        Serial.println(savePidParams() ? "-> PID im Flash gespeichert"
                                       : "-> FEHLER beim Speichern (NVS)");
        return;
    }
    if (cmd.startsWith("pid"))  { printPidParams(); return; }

    // Fahrt starten und dabei plotten. Muss vor dem einbuchstabigen 'g' stehen.
    if (cmd.startsWith("gp")) {
        float grad = cmd.substring(2).toFloat();
        uint8_t id = nextLocalMoveId++;
        if (nextLocalMoveId == 0) nextLocalMoveId = 1;
        Serial.printf("-> move id=%u um %.1f grad | Plotter bis Fahrtende\n", id, grad);
        g_plotMode = true;
        g_plotAuto = true;
        startMove(id, (int32_t)lroundf(grad * 10.0f));
        return;
    }

    // --- Manuelle Lenkungs-Kalibrierung ---
    // Muss vor den einbuchstabigen Befehlen stehen ('c' ist coast).
    if (cmd.startsWith("cal")) {
        String arg = cmd.substring(3);
        arg.trim();
        if (arg.length() == 0) {
            if (g_cal.active) calPrintStatus(&sc09Servo);
            else              calStart(&sc09Servo);
        }
        else if (arg == "?")     calPrintStatus(&sc09Servo);
        else if (arg == "m")     calSetMark(&sc09Servo, CAL_ACT_CENTER);
        else if (arg == "l")     calSetMark(&sc09Servo, CAL_ACT_LEFT);
        else if (arg == "r")     calSetMark(&sc09Servo, CAL_ACT_RIGHT);
        else if (arg == "save")  calSave(&sc09Servo);
        else if (arg == "q")     calAbort(&sc09Servo);
        else if (arg == "free")  calSetFree(&sc09Servo, true);
        else if (arg == "hold")  calSetFree(&sc09Servo, false);
        else if (arg == "go")    calGoCenter(&sc09Servo);
        else if (arg.charAt(0) == 'n') calSetStep(arg.substring(1).toInt());
        else if (arg.charAt(0) == '+' || arg.charAt(0) == '-') {
            int n = arg.substring(1).toInt();
            if (n <= 0) n = g_cal.step;
            calMove(&sc09Servo, arg.charAt(0) == '+' ? n : -n);
        }
        else {
            Serial.println("?? cal, cal+/cal-, caln<ticks>, calm/call/calr,");
            Serial.println("   calfree/calhold/calgo, calsave, calq");
        }
        return;
    }

    // --- Messfenster der Geschwindigkeit ---
    // Muss vor die einbuchstabigen Befehle ('s' = Stop).
    if (cmd.startsWith("sw")) {
        String arg = cmd.substring(2);
        arg.trim();
        if (arg.length() > 0) {
            long n = arg.toInt();
            if (n < 1)                       n = 1;
            if (n > SPEED_WINDOW_MAX - 1)    n = SPEED_WINDOW_MAX - 1;
            g_speedWindow = (uint8_t)n;
        }
        uint32_t fensterMs = (uint32_t)g_speedWindow * TASK_PERIOD;
        float schrittRpm = 60000.0f / (COUNTS_PER_REV * (float)fensterMs);
        Serial.printf("-> Messfenster %u x %lu ms = %lu ms\n",
                      g_speedWindow, (unsigned long)TASK_PERIOD,
                      (unsigned long)fensterMs);
        Serial.printf("   ein Impuls = %.1f U/min = %.1f grad/s, "
                      "Verzoegerung %.0f ms\n",
                      schrittRpm, schrittRpm * 6.0f, fensterMs / 2.0f);
        Serial.println("   Ausgaberate bleibt davon unberuehrt (siehe 'tel').");
        return;
    }

    // --- Fahrtelemetrie ---
    // Muss vor die einbuchstabigen Befehle ('t' = Torque).
    if (cmd.startsWith("tel")) {
        String arg = cmd.substring(3);
        arg.trim();
        if (arg.length() > 0) {
            long ms = arg.toInt();
            if (ms <= 0)                      g_telemetryMs = 0;
            else if (ms < TELEMETRY_MS_MIN)   g_telemetryMs = TELEMETRY_MS_MIN;
            else if (ms > 60000)              g_telemetryMs = 60000;
            else                              g_telemetryMs = (uint16_t)ms;
        }
        if (g_telemetryMs) {
            Serial.printf("-> Telemetrie alle %u ms (%.1f Hz)\n",
                          g_telemetryMs, 1000.0f / g_telemetryMs);
            Serial.printf("   Tempo aus einem Fenster von %u ms ('sw'), "
                          "aber bei jedem Paket neu.\n",
                          (unsigned)g_speedWindow * (unsigned)TASK_PERIOD);
        } else {
            Serial.println("-> Telemetrie aus ('tel100' = 10 Hz)");
        }
        Serial.printf("   jetzt: %.1f grad  %.1f grad/s  duty=%d  I=%.2f A\n",
                      countsToDeg10(g_encCount) / 10.0f, g_rpm * 6.0f,
                      g_dutySigned, readMotorCurrentA());
        return;
    }

    // --- Zeitsynchronisation / Sendezeitstempel ---
    // Muss vor die einbuchstabigen Befehle ('t' = Torque).
    if (cmd.startsWith("ts")) {
        String arg = cmd.substring(2);
        arg.trim();
        if (arg.length() > 0) {
            g_stampTx = (arg.toInt() != 0);
            jetson.sendStampMode();      // Bridge ueber die Umstellung informieren
        }
        Serial.printf("-> Sendezeitstempel %s (Rahmen 0x%02X)\n",
                      g_stampTx ? "AN" : "AUS", g_stampTx ? START_BYTE_TS : START_BYTE);
        Serial.printf("   ESP-Uhr: %lld us seit Boot (%.3f s)\n",
                      (long long)nowUs(), nowUs() / 1000000.0);
        Serial.println("   Abgleich macht der Jetson mit CMD_TIME_SYNC (0xB0).");
        return;
    }

    if (cmd.startsWith("dbg")) {
        String arg = cmd.substring(3);
        if (arg.length() > 0) {
            g_linkDebug = (uint8_t)constrain(arg.toInt(), 0L, 2L);
            Serial.printf("-> Link-Debug = %u (%s)\n", g_linkDebug,
                          g_linkDebug == 0 ? "aus" :
                          g_linkDebug == 1 ? "Pakete" : "Pakete + Rohbytes");
        }
        jetson.printLinkStats();
        return;
    }

    // --- Motorstrom kalibrieren ---
    if (cmd.startsWith("ic")) {
        float f = cmd.substring(2).toFloat();
        if (f > 1.0f && f < 5000.0f) {
            csMvPerA = f;
            prefs.putFloat("csmv", csMvPerA);
            Serial.printf("-> Stromsense = %.1f mV/A (gespeichert)\n", csMvPerA);
        } else {
            Serial.printf("-> aktuell %.1f mV/A (nominal %.1f). Format: ic140\n",
                          csMvPerA, CS_MV_PER_A_NOMINAL);
        }
        return;
    }
    // Nullpunkt bei stehendem Motor. Muss im Coast gemessen werden, sonst
    // wandert der Ruhestrom in den Offset und jede spaetere Messung ist zu klein.
    if (cmd == "iz") {
        setMotorCommand(MOTOR_COAST, false, 0);
        vTaskDelay(300 / portTICK_PERIOD_MS);
        csZeroMv = (int)lroundf(readMotorCsMv());
        prefs.putInt("csoff", csZeroMv);
        Serial.printf("-> Nullpunkt = %d mV (gespeichert)\n", csZeroMv);
        return;
    }

    if (cmd == "vd") {
        batteryPinDiagnose();
        return;
    }

    if (cmd.startsWith("vm")) {
        long ms = cmd.substring(2).toInt();
        if (cmd.length() == 2) ms = g_battMonMs ? 0 : 250;   // "vm" schaltet um
        g_battMonMs = (ms > 0) ? (uint32_t)max(ms, 50L) : 0;
        g_battMonLast = 0;
        if (g_battMonMs) Serial.printf("-> Akku-Live-Ausgabe alle %lu ms (vm0 = aus)\n",
                                       (unsigned long)g_battMonMs);
        else             Serial.println("-> Akku-Live-Ausgabe aus");
        return;
    }

    if (cmd.startsWith("vc")) {
        float f = cmd.substring(2).toFloat();
        if (f > 1.0f && f < 50.0f) {
            battDivider = f;
            prefs.putFloat("bdiv", battDivider);
            Serial.printf("-> Teilerfaktor = %.4f (gespeichert)\n", battDivider);
        } else {
            Serial.printf("-> aktueller Teilerfaktor = %.4f (nominal %.4f)\n",
                          battDivider, BATT_DIVIDER_NOMINAL);
        }
        return;
    }

    // --- Servo/Lenkung direkt (muss vor den einbuchstabigen Befehlen stehen) ---
    if (cmd.startsWith("sv")) {
        String arg = cmd.substring(2);
        if (arg.length() == 0) {
            servoDiagnose(&sc09Servo);                 // "sv" = Diagnose
        } else if (arg == "s") {
            servoScanProtocols(&sc09Servo);            // "svs" = Protokoll-Scan
        } else if (arg.startsWith("baud")) {           // "svbaud<n>" fuer Oszi
            long b = arg.substring(4).toInt();
            if (b >= 1200 && b <= 1000000) {
                g_servoBaud = (uint32_t)b;
                servoBusRestart(&sc09Servo);
                Serial.printf("-> Servo-Baud = %lu (fuer echten Betrieb wieder svbaud1000000!)\n",
                              (unsigned long)g_servoBaud);
            } else {
                Serial.printf("-> aktuell %lu Baud. Bereich 1200..1000000.\n",
                              (unsigned long)g_servoBaud);
            }
        } else if (arg == "tx") {                       // "svtx" Dauer-0x55 fuers Oszi
            g_servoTxTest = !g_servoTxTest;
            Serial.printf("-> TX-Testmuster 0x55 %s (an IO%d messen)%s\n",
                          g_servoTxTest ? "AN" : "AUS", g_servoTx,
                          g_servoTxTest ? "" : "");
            if (g_servoTxTest && g_servoBaud > 115200)
                Serial.println("   Hinweis: fuer langsame Oszis erst 'svbaud9600'.");
        } else if (arg.startsWith("pin")) {            // "svpin<rx>,<tx>"
            String rest = arg.substring(3);
            rest.replace(",", " ");
            int sp = rest.indexOf(' ');
            if (sp > 0) {
                int rx = rest.substring(0, sp).toInt();
                int tx = rest.substring(sp + 1).toInt();
                servoSetPins(&sc09Servo, rx, tx);
            } else {
                Serial.printf("-> aktuell RX=IO%d TX=IO%d. Format: svpin18,17\n",
                              g_servoRx, g_servoTx);
            }
        } else if (arg == "pos") {
            servoForcePositionMode(&sc09Servo);            // "svpos" = Positions-Modus erzwingen
        } else if (arg == "r") {
            servoRegisterDump(&sc09Servo);                 // "svr" = Register + Schreibtest
        } else if (arg == "c") {
            servoWriteRaw(&sc09Servo, SERVO_CENTER_DEF);   // "svc" = Mitte (roh)
        } else if (arg.charAt(0) == 'e') {
            int on = arg.substring(1).toInt();
            int ret = sc09Servo.EnableTorque(SERVO_ID, on ? 1 : 0);
            Serial.printf("-> Torque %s (ret=%d)\n", on ? "AN" : "AUS", ret);
        } else {
            servoWriteRaw(&sc09Servo, arg.toInt());    // "sv700" = feste Position
        }
        return;
    }

    if (cmd.length() >= 2 && cmd.charAt(0) == 'k') {
        PidParams p = getPid();
        float f = cmd.substring(2).toFloat();
        switch (cmd.charAt(1)) {
            case 'p': p.kp = f;                                           break;
            case 'i': p.ki = f;                                           break;
            case 'd': p.kd = f;                                           break;
            case 'l': p.iTermLim = f;                                     break;
            case 'm': p.maxDuty = (uint16_t)constrain((long)f, 0L, (long)DUTY_MAX); break;
            case 'a': p.minDuty = (uint16_t)constrain((long)f, 0L, (long)DUTY_MAX); break;
            case 't': p.tolDeg10 = (int32_t)lroundf(f * 10.0f);           break;
            case 'n': p.settleMs = (uint32_t)f;                           break;
            case 'x': p.timeoutMs = (uint32_t)f;                          break;
            default: Serial.println("?? unbekannter PID-Parameter");      return;
        }
        setPid(p);
        printPidParams();
        return;
    }

    // --- Einbuchstabige Befehle ---
    char c = cmd.charAt(0);

    // f/r/b nehmen weiterhin 0-255 (wie im Testskript) und werden auf 10 Bit skaliert.
    int val255 = constrain(cmd.substring(1).toInt(), 0, 255);
    uint16_t duty = (uint16_t)((val255 * DUTY_MAX) / 255);

    switch (c) {
        case 'f': setMotorCommand(MOTOR_DRIVE, false, duty); Serial.printf("-> forward %d (duty %u)\n", val255, duty); break;
        case 'r': setMotorCommand(MOTOR_DRIVE, true,  duty); Serial.printf("-> reverse %d (duty %u)\n", val255, duty); break;
        case 'b': setMotorCommand(MOTOR_BRAKE, false, duty); Serial.printf("-> brake %d\n", val255);                   break;
        case 'c': setMotorCommand(MOTOR_COAST, false, 0);    Serial.println("-> coast");                               break;

        case 'z':
            encoder.clearCount();
            Serial.println("-> encoder = 0");
            break;

        case 'e':
            Serial.printf("enc=%ld (%.1f grad)  rpm=%.1f  duty=%d  I=%.2f A (%.0f mV)\n",
                          g_encCount, countsToDeg10(g_encCount) / 10.0f,
                          g_rpm, g_dutySigned, readMotorCurrentA(), readMotorCsMv());
            break;

        // --- Motorstrom ---
        case 'i':
            Serial.printf("Motorstrom: %.2f A  (CS %.0f mV, Null %d mV, %.1f mV/A)\n",
                          readMotorCurrentA(), readMotorCsMv(), csZeroMv, csMvPerA);
            break;

        // --- Positionsfahrt ---
        case 'g': {
            float grad = cmd.substring(1).toFloat();
            uint8_t id = nextLocalMoveId++;
            if (nextLocalMoveId == 0) nextLocalMoveId = 1;
            startMove(id, (int32_t)lroundf(grad * 10.0f));
            Serial.printf("-> move id=%u um %.1f grad (ab %.1f)\n",
                          id, grad, countsToDeg10(g_encCount) / 10.0f);
            break;
        }

        case 'q':
            abortMove();
            Serial.println("-> Fahrt abgebrochen");
            break;

        case 'w': {
            MoveState m = getMoveState();
            Serial.printf("move id=%u %s  %u%%  ist=%.1f  ziel=%.1f grad\n",
                          m.id, m.active ? "AKTIV" : "idle", m.progress,
                          countsToDeg10(g_encCount) / 10.0f,
                          countsToDeg10(m.targetCnt) / 10.0f);
            break;
        }

        // --- Batterie ---
        case 'v': {
            float pinMv = readBatteryMv();
            battPackV = (pinMv / 1000.0f) * battDivider;
            battCellV = battPackV / BATT_CELLS;
            Serial.printf("Akku: %.2f V Pack | %.3f V/Zelle (%dS)%s\n",
                          battPackV, battCellV, BATT_CELLS,
                          battCellV < BATT_WARN_CELL ? "  *** NIEDRIG ***" : "");
            Serial.printf("      GPIO%d: %.1f mV am Teilerabgriff, roh %d/4095, Faktor %.4f\n",
                          PIN_BATTERY, pinMv, analogRead(PIN_BATTERY), battDivider);
            if (pinMv >= BATT_ADC_CLIP_MV) {
                Serial.println("      *** ADC AM ANSCHLAG - Packspannung nicht messbar ***");
                Serial.println("      Multimeter an GPIO1 gegen GND halten:");
                Serial.println("        ~3,0 V (Pack/5,5) -> Teiler ok, Pack zu hoch fuer den Messbereich");
                Serial.println("        deutlich hoeher   -> 22k-Zweig gegen GND offen (kalte Loetstelle?)");
            }
            break;
        }

        // --- Lenkung ---
        case 'x':   // 'c' ist coast -> Kalibrierung liegt auf 'x'
            calStart(&sc09Servo);
            break;

        case 't':
            printTorque(&sc09Servo);
            break;

        // --- Servo schrittweise stellen ---
        // Ohne Zahl gilt die Kalibrier-Schrittweite (bzw. 100 Ticks ausserhalb
        // der Kalibrierung), mit Zahl der angegebene Wert: "-40", "j25".
        case '+':
        case '-':
        case 'j':   // Position kleiner
        case 'l': { // Position groesser
            bool plus = (c == '+' || c == 'l');
            int n = cmd.substring(1).toInt();
            if (n <= 0) n = g_cal.active ? g_cal.step : 100;
            int delta = plus ? n : -n;
            if (g_cal.active) calMove(&sc09Servo, delta);
            else              servoWriteRaw(&sc09Servo, servoManualPos + delta);
            break;
        }

        case 'm':   // Mitte anfahren
            if (g_cal.active) calGoCenter(&sc09Servo);
            else              servoWriteRaw(&sc09Servo, softwareCenterPos);
            break;

        // Trim verschiebt die Mitte in Rohposition-Ticks. Welche Fahrtrichtung
        // das ist, haengt am Einbau - deshalb hier bewusst Pos-/Pos+ statt
        // links/rechts. Vorzeichen wie bisher, damit Muskelgedaechtnis und
        // CMD_TRIM gleich bleiben.
        case 'a':   // Trim Position kleiner
            trimOffset        -= 20;
            softwareCenterPos  = centerLimit + trimOffset;
            sc09Servo.WritePosEx(SERVO_ID, softwareCenterPos, SERVO_SPEED_FAST, SERVO_ACC);
            servoManualPos = softwareCenterPos;
            Serial.printf("Trim: %d | Aktuelle Pos: %d\n", trimOffset, softwareCenterPos);
            break;

        case 'd':   // Trim Position groesser
            trimOffset        += 20;
            softwareCenterPos  = centerLimit + trimOffset;
            sc09Servo.WritePosEx(SERVO_ID, softwareCenterPos, SERVO_SPEED_FAST, SERVO_ACC);
            servoManualPos = softwareCenterPos;
            Serial.printf("Trim: %d | Aktuelle Pos: %d\n", trimOffset, softwareCenterPos);
            break;

        case 's':   // Trim speichern
            prefs.putInt("offset10", trimOffset);
            Serial.println("ESP: Trim-Offset permanent gespeichert!");
            break;

        // LED umschalten. Sonst nur ueber CMD_LED vom Jetson erreichbar - beim
        // Bring-up eines neuen Boards will man sie ohne Jetson pruefen koennen.
        case 'o': {
            static bool ledOn = false;
            ledOn = !ledOn;
            digitalWrite(PIN_LED, ledOn ? HIGH : LOW);
            Serial.printf("-> LED (IO%d) %s\n", PIN_LED, ledOn ? "AN" : "AUS");
            break;
        }

        case 'p':
            g_plotMode = !g_plotMode;
            Serial.printf("-> Plotter %s%s\n", g_plotMode ? "AN" : "AUS",
                          g_plotMode ? " (soll/ist in Grad, duty in %, 50 Hz)" : "");
            break;

        case 'h':
            printHelp();
            break;

        default:
            Serial.println("?? unbekannter Befehl ('h' fuer Hilfe)");
    }
}

// ==========================================
// 11. BATTERIE-UEBERWACHUNG (Core 1)
// ==========================================

void batteryTick() {
    uint32_t now = millis();

    if (g_battMonMs && now - g_battMonLast >= g_battMonMs) {
        g_battMonLast = now;
        float mv = readBatteryMv();
        Serial.printf("vm: GPIO%d %7.1f mV  roh %4d/4095  -> %6.2f V%s\n",
                      PIN_BATTERY, mv, analogRead(PIN_BATTERY),
                      (mv / 1000.0f) * battDivider,
                      mv >= BATT_ADC_CLIP_MV ? "  ANSCHLAG" : "");
    }

    if (now - battLastRead < BATT_INTERVAL) return;
    battLastRead = now;

    float pinMv = readBatteryMv();
    battPackV = (pinMv / 1000.0f) * battDivider;
    battCellV = battPackV / BATT_CELLS;

    // Ein geklippter Wert liegt immer oben - die Unterspannungswarnung koennte
    // also nie ausloesen, egal wie leer der Akku wirklich ist. Deshalb laut
    // melden statt eine Messung vorzutaeuschen. Takt wie die Akkuwarnung.
    if (pinMv >= BATT_ADC_CLIP_MV &&
        (battLastWarn == 0 || now - battLastWarn >= BATT_WARN_REPEAT)) {
        battLastWarn = now;
        Serial.printf("ESP: WARNUNG Akku-ADC am Anschlag (%.0f mV an GPIO%d)! %.2f V ist "
                      "nur die Obergrenze, keine Messung - Unterspannungsschutz ist "
                      "blind. 'v' zeigt Details.\n",
                      pinMv, PIN_BATTERY, battPackV);
    }

    if (!battLow && battCellV < BATT_WARN_CELL) {
        battLow = true;
        battLastWarn = 0;              // sofort warnen
    } else if (battLow && battCellV > BATT_RECOVER_CELL) {
        battLow = false;               // Hysterese: erst ueber 3,85 V entwarnen
        Serial.printf("ESP: Akku wieder ok: %.2f V (%.3f V/Zelle)\n", battPackV, battCellV);
    }

    if (battLow && (battLastWarn == 0 || now - battLastWarn >= BATT_WARN_REPEAT)) {
        battLastWarn = now;
        Serial.printf("ESP: WARNUNG Akku niedrig! %.2f V Pack | %.3f V/Zelle\n",
                      battPackV, battCellV);
        jetson.sendBattery(CMD_BATTERY_WARN);
    }
}

// Fahrtelemetrie im eingestellten Takt. Laeuft auf Core 1 aus dem loop() -
// der Takt ist deshalb nicht hart, sondern haengt an der Schleifendauer.
// Genau dafuer traegt das Paket seinen Sendezeitstempel: der Jetson muss den
// Zeitpunkt nicht aus dem Takt herleiten, sondern liest ihn ab.
void telemetryTick() {
    if (g_telemetryMs == 0) return;

    static uint32_t lastSend = 0;
    uint32_t now = millis();
    if ((uint32_t)(now - lastSend) < g_telemetryMs) return;

    // Takt fortschreiben statt auf "jetzt" zu setzen: sonst schiebt sich der
    // Rest jeder loop()-Runde in den Takt und aus 100 Hz werden 90.
    lastSend += g_telemetryMs;
    // Nach einer laengeren Pause (Positionsfahrt, langer Konsolenausdruck)
    // nicht die verpassten Pakete nachfeuern, sondern neu aufsetzen.
    if ((uint32_t)(now - lastSend) > g_telemetryMs) lastSend = now;

    jetson.sendTelemetry();
}

// ==========================================
// 12. SETUP & LOOP (Core 1)
// ==========================================

void setup() {
    Serial.begin(115200);
    delay(300);

    // --- Antrieb + Encoder ---
    planetaryMotor.begin();

    ESP32Encoder::useInternalWeakPullResistors = puType::up;  // Hall open-drain
    // Spuren getauscht anhaengen, wenn die Fahrtrichtung gedreht ist - sonst
    // zaehlt der Encoder beim Vorwaertsfahren rueckwaerts.
    encoder.attachFullQuad(DRIVE_INVERT ? PIN_ENC_B : PIN_ENC_A,
                           DRIVE_INVERT ? PIN_ENC_A : PIN_ENC_B);   // 4x in HW
    // Glitch-Filter in APB-Takten (80 MHz). 250 = ~3,1 us: killt Buersten-Rauschen,
    // laesst echte Flanken durch (bei Vollspeed ~333 us Abstand).
    encoder.setFilter(250);
    encoder.clearCount();

    // --- Batterie-ADC ---
    // 12 dB Daempfung: Messbereich bis ~3,1 V am Pin. 4S voll (16,8 V) liegt
    // ueber den Teiler bei ~3,03 V - also knapp unterhalb der Klippgrenze.
    // Der erste analogRead haengt den Pin an einen ADC-Kanal. Ohne ihn schlaegt
    // analogSetPinAttenuation in Core 3.x fehl ("__analogChannelConfig(): Pin
    // is not ADC pin") und die Daempfung bleibt auf dem Default stehen.
    (void)analogRead(PIN_BATTERY);
    analogSetPinAttenuation(PIN_BATTERY, ADC_11db);

    // --- Motorstrom-ADC (VNH5019 CS) ---
    // Gleiche Reihenfolge wie beim Akku: erst lesen, dann Daempfung setzen.
    (void)analogRead(PIN_MOTOR_CS);
    analogSetPinAttenuation(PIN_MOTOR_CS, ADC_11db);

    // --- Peripherie ---
    pinMode(PIN_BUTTON, INPUT_PULLUP);
    pinMode(PIN_LED, OUTPUT);
    attachInterrupt(digitalPinToInterrupt(PIN_BUTTON), buttonISR, FALLING);

    jetson.begin(115200);

    Serial2.begin(1000000, SERIAL_8N1, PIN_SERVO_RX, PIN_SERVO_TX);
    sc09Servo.pSerial = &Serial2;
    delay(500);

    // --- NVS: Lenkung, PID, Batterie-Kalibrierung ---
    prefs.begin("steering", false);
    // 10-Bit-Keys (SC-Servo 0..1023). Alte *12-Keys aus der STS-Fehlannahme
    // werden bewusst ignoriert - sie liegen im 0..4095-Bereich.
    leftLimit   = prefs.getInt("lLim10", SERVO_POS_MAX);
    rightLimit  = prefs.getInt("rLim10", 0);
    trimOffset  = prefs.getInt("offset10", 0);
    if (prefs.isKey("bdiv"))  battDivider = prefs.getFloat("bdiv", BATT_DIVIDER_NOMINAL);
    if (prefs.isKey("csmv"))  csMvPerA    = prefs.getFloat("csmv", CS_MV_PER_A_NOMINAL);
    if (prefs.isKey("csoff")) csZeroMv    = prefs.getInt("csoff", 0);
    // Die Mitte wird seit der manuellen Kalibrierung eigenstaendig gesetzt.
    // Fehlt der Schluessel (Kalibrierung von frueher), gilt wie bisher die
    // rechnerische Mitte zwischen den Anschlaegen.
    centerLimit = prefs.getInt("cLim10", (leftLimit + rightLimit) / 2);
    softwareCenterPos = centerLimit + trimOffset;
    loadPidParams();

    // --- Motortask auf Core 0 ---
    // Queue und Pins muessen vor dem Task stehen.
    g_moveResultQueue = xQueueCreate(8, sizeof(MoveResult));
    setMotorCommand(MOTOR_COAST, false, 0);
    xTaskCreatePinnedToCore(
        motorControlTask,
        "Motor_Task",
        4096,
        (void*)&planetaryMotor,
        1,
        &MotorControlTaskHandle,
        0                        // Core 0
    );

    // Torque explizit einschalten. Ohne das nimmt der SC09 Positionsbefehle
    // zwar entgegen, haelt sie aber nicht - die Lenkung reagiert erst, nachdem
    // 'cal' oder 'tq1' den Torque gesetzt hat.
    sc09Servo.EnableTorque(SERVO_ID, 1);

    // Aktuelle Servoposition halten, um Startup-Spannung zu vermeiden
    int startPos = sc09Servo.ReadPos(SERVO_ID);
    if (startPos != -1) {
        sc09Servo.WritePosEx(SERVO_ID, startPos, SERVO_SPEED_FAST, SERVO_ACC);
    }

    // Erste Batteriemessung sofort, danach im 15-s-Raster
    battPackV = readBatteryVolts();
    battCellV = battPackV / BATT_CELLS;
    battLastRead = millis();

    // Pinbelegung beim Start ausgeben - beim Bring-up eines neuen Boards die
    // erste Kontrolle, ob die Firmware ueberhaupt die richtigen Pins bedient.
    Serial.printf("Pins: Motor PWM%d INA%d INB%d CS%d | Enc %d/%d | Servo TX%d RX%d @%lu"
                  " | Jetson RX%d TX%d | Akku %d | LED %d | Taster %d\n",
                  PIN_MOTOR_PWM, PIN_MOTOR_INA, PIN_MOTOR_INB, PIN_MOTOR_CS,
                  DRIVE_INVERT ? PIN_ENC_B : PIN_ENC_A,
                  DRIVE_INVERT ? PIN_ENC_A : PIN_ENC_B, g_servoTx, g_servoRx,
                  (unsigned long)g_servoBaud, PIN_JETSON_RX, PIN_JETSON_TX,
                  PIN_BATTERY, PIN_LED, PIN_BUTTON);
    if (DRIVE_INVERT) {
        Serial.println("Fahrtrichtung gedreht (DRIVE_INVERT) - Motor und Encoder.");
    }
    Serial.printf("System Ready. Lenkung: Mitte:%d L:%d R:%d | Trim: %d | Akku: %.2f V (%.3f V/Zelle)\n",
                  centerLimit, leftLimit, rightLimit, trimOffset, battPackV, battCellV);
    if (!prefs.isKey("cLim10")) {
        Serial.println("Hinweis: Lenkung noch nie manuell kalibriert - 'cal' starten.");
    }
    printHelp();
}

void loop() {
    // --- Button ---
    if (buttonTriggered) {
        buttonTriggered = false;
        static unsigned long lastPressTime = 0;
        if (millis() - lastPressTime > 200) {   // 200 ms Entprellzeit
            lastPressTime = millis();
            jetson.sendButtonEvent();
        }
    }

    // --- Ergebnisse abgeschlossener Positionsfahrten an den Jetson melden ---
    MoveResult r;
    while (xQueueReceive(g_moveResultQueue, &r, 0) == pdTRUE) {
        // Auto-Plotter endet mit der Fahrt - erst abschalten, dann melden,
        // sonst landet die Meldung noch im Kurvenstrom.
        if (g_plotAuto) { g_plotMode = false; g_plotAuto = false; }
        jetson.sendMoveDone(r);
        const char* txt = (r.status == MOVE_OK)      ? "DONE"
                        : (r.status == MOVE_TIMEOUT) ? "TIMEOUT"
                                                     : "ABORTED";
        Serial.printf("-> move id=%u %s bei %.1f grad\n", r.id, txt, r.finalDeg10 / 10.0f);
    }

    // --- USB-Debug-Konsole ---
    // Eingaben werden zurueckgeschickt. Ohne Echo laesst sich nicht
    // unterscheiden, ob der Monitor nichts sendet oder ob nur das Zeilenende
    // fehlt und der Befehl deshalb nie ausgefuehrt wird.
    static uint32_t lastCharMs = 0;
    static bool sawLineEnding = false;   // Monitor haengt ein \n oder \r an?

    while (Serial.available()) {
        char ch = Serial.read();
        lastCharMs = millis();
        if (ch == '\n' || ch == '\r') {
            sawLineEnding = true;
            if (!g_plotMode) Serial.println();
            handleDebugCommand(cmdBuf);
            cmdBuf = "";
        } else {
            cmdBuf += ch;
            if (!g_plotMode) Serial.print(ch);
            if (cmdBuf.length() > 40) cmdBuf = "";   // Ueberlauf verwerfen
        }
    }

    // Notausgang fuer Monitore, die kein Zeilenende senden (z.B. die
    // VS-Code-Serial-Monitor-Erweiterung mit "Line ending: None"): nach einer
    // Sendepause den Puffer trotzdem ausfuehren.
    // Nur solange noch NIE ein Zeilenende kam - sonst wuerde bei jemandem, der
    // zeichenweise in ein Terminal tippt, jede Denkpause ein halbes Kommando
    // ausloesen. Sobald der Monitor sich einmal als zeilenbasiert zu erkennen
    // gibt, ist dieser Pfad fuer immer aus.
    if (!sawLineEnding && cmdBuf.length() > 0 && millis() - lastCharMs > 400) {
        if (!g_plotMode) Serial.println("   (ohne Zeilenende empfangen)");
        handleDebugCommand(cmdBuf);
        cmdBuf = "";
    }

    // --- Jetson-Protokoll ---
    jetson.process();

    // --- Akku alle 15 s ---
    batteryTick();

    // --- Fahrtelemetrie (Position, Tempo, Duty, Strom) ---
    telemetryTick();

    // --- Plotter-Ausgabe (nur wenn mit "p" eingeschaltet) ---
    plotTick();

    // --- Servo-TX-Testmuster fuers Oszilloskop (svtx) ---
    // Fuellt den Sendepuffer mit 0x55, damit an IO_TX ein Dauer-Rechteck liegt.
    // begrenzt pro loop, sonst blockiert write() bei niedriger Baudrate.
    if (g_servoTxTest) {
        for (int i = 0; i < 8 && Serial2.availableForWrite() > 0; i++) {
            Serial2.write(0x55);
        }
    }
}
