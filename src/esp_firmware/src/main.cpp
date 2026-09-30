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
// 1. PIN- UND PROTOKOLL-DEFINITIONEN
// ==========================================

// --- Antrieb: VNH5019 + Quadratur-Encoder ---
constexpr int PIN_MOTOR_PWM = 41;   // VNH5019 PWM
constexpr int PIN_MOTOR_INA = 42;   // VNH5019 INA
constexpr int PIN_MOTOR_INB = 38;   // VNH5019 INB
constexpr int PIN_MOTOR_CS  = 39;   // VNH5019 CS (nur digital lesbar, kein ADC)
constexpr int PIN_ENC_A     = 15;   // Encoder A
constexpr int PIN_ENC_B     = 16;   // Encoder B

// --- Peripherie ---
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
#define CMD_MOTOR       0x10   // 3B: dir, speedHi, speedLo
#define CMD_SERVO       0x20   // 3B: id, pctHi, pctLo
#define CMD_LED         0x30   // 1B: on/off
#define CMD_CALIBRATE   0x40   // 0B
#define CMD_TORQUE      0x50   // 0B
#define CMD_TRIM        0x60   // 1B: 0=links 1=rechts 2=speichern
#define CMD_PID_SET     0x80   // 5B: paramId, int32 wert (x1000)
#define CMD_PID_GET     0x81   // 0B
#define CMD_PID_SAVE    0x83   // 0B: aktuelle Parameter ins NVS schreiben
#define CMD_MOVE        0x90   // 5B: moveId, int32 weite in 1/10 Grad (relativ)
#define CMD_MOVE_ABORT  0x91   // 0B
#define CMD_PROGRESS    0x92   // 0B
#define CMD_BATTERY     0xA0   // 0B
#define CMD_SYNC_REQ    0xB0   // 4B: uint32 req_id
#define CMD_TELEM_RATE  0xB3   // 1B: hz (0 = aus, max 100 = Motortask-Takt)
#define CMD_EMERGENCY   0xFF   // 0B

// --- Protokoll: ESP -> Jetson ---
#define CMD_BUTTON      0x70   // 1B: 0x01 = pressed
#define CMD_PID_RSP     0x82   // 12B: int32 Kp, Ki, Kd (jeweils x1000)
#define CMD_PID_SAVED   0x84   // 1B: 0x00 = gespeichert, 0x01 = Fehler
#define CMD_MOVE_DONE   0x93   // 6B: moveId, status, int32 ist-Pos (1/10 Grad)
#define CMD_PROGRESS_RSP 0x94  // 11B: moveId, aktiv, prozent, int32 ist, int32 ziel
#define CMD_BATTERY_RSP 0xA1   // 6B: int32 pack mV, int16 zelle mV
#define CMD_BATTERY_WARN 0xA2  // 6B: wie CMD_BATTERY_RSP, ungefragt bei Unterspannung
#define CMD_SYNC_RSP    0xB1   // 22B: uint32 req_id, int64 t2_us, int64 t3_us, uint16 crc
#define CMD_ENC_TELEM   0xB2   // 16B: uint16 seq, int64 t_us, int32 count, uint16 crc

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

// --- Encoder-Telemetrie fuer den EKF auf dem Jetson ---
// Der Zeitstempel entsteht dort, wo der Zaehler gelesen wird (im Motortask),
// nicht beim Versenden. g_encCount wird nur alle TASK_PERIOD aktualisiert - wer
// ihn asynchron liest und dann stempelt, datiert einen bis zu 10 ms alten Wert
// auf jetzt. esp_timer_get_time() ist int64 us und laeuft nicht ueber; micros()
// waere nach 71,6 min hinueber.
struct EncoderSample {
    int64_t  t_us;      // esp_timer_get_time() beim Zaehlerlesen
    int32_t  count;     // roher kumulativer Encoder-Count, ungefiltert
    uint16_t seq;       // fortlaufend, fuer Verlusterkennung auf Jetson-Seite
};

static QueueHandle_t g_encQueue = nullptr;
static uint16_t      g_encSeq   = 0;   // nur vom Motortask beschrieben

// Sendefrequenz der Telemetrie. Die Samples entstehen im Motortask-Takt
// (100 Hz), mehr ist ohne Aenderung von TASK_PERIOD nicht zu holen - hoehere
// Werte werden auf 100 begrenzt, kleinere dezimieren den Strom.
static volatile uint8_t g_telemHz = 100;

// Asynchrone Meldungen aus loop() an den Link-Task. Serial1 gehoert exklusiv
// dem Link-Task: zwei Tasks, die in dieselbe UART schreiben, verschraenken
// ihre Bytes und zerlegen das Protokoll.
struct TxPacket {
    uint8_t len;
    uint8_t buf[16];
};
static QueueHandle_t g_txQueue = nullptr;

// Befehle, die der Link-Task nicht selbst ausfuehren darf, weil sie den Servo
// anfassen oder ins NVS schreiben (beides dauert Millisekunden bis Sekunden).
struct DeferredCmd {
    uint8_t cmd;
    uint8_t len;
    uint8_t data[8];
};
static QueueHandle_t g_deferQueue = nullptr;

// Telemetrie: Motortask schreibt, Core 1 liest nur zur Ausgabe.
static volatile float   g_rpm        = 0.0f;
static volatile long    g_encCount   = 0;
static volatile int16_t g_dutySigned = 0;   // + vorwaerts, - rueckwaerts

ESP32Encoder encoder;
// COUNTS_PER_REV des AKTIVEN Motors eintragen:
// ServoCity DE3 (Open-Collector): 3 PPR x 4 x 42,875 Getriebe = ~514,5
// Pololu 25D #4841 (Push-Pull)  : 48 CPR x 4,4 Getriebe = 211,2 (Ausgangswelle)
constexpr float COUNTS_PER_REV = 408.0f;
constexpr float RPM_EMA = 0.30f;

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

int softwareCenterPos = 511;
int leftLimit  = 0;
int rightLimit = 1023;

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
        digitalWrite(inaPin, reverse ? LOW  : HIGH);
        digitalWrite(inbPin, reverse ? HIGH : LOW);
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
constexpr uint32_t TASK_PERIOD  = 10;    // ms
constexpr uint32_t RPM_INTERVAL = 100;   // ms

void motorControlTask(void* pvParameters) {
    MotorDriver* motor = (MotorDriver*)pvParameters;

    uint16_t  appliedDuty    = 0;       // was gerade wirklich anliegt
    bool      appliedReverse = false;
    MotorMode appliedMode    = MOTOR_COAST;

    unsigned long lastRpmCalc = millis();
    long  lastCount = 0;
    float rpmFilt   = 0.0f;
    bool  firstTick = true;

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

        // Stempel und Zaehlerstand gehoeren zusammen - erst die Zeit, dann
        // sofort den Zaehler, ohne etwas dazwischen.
        int64_t tSample = esp_timer_get_time();
        long posCnt = (long)encoder.getCount();
        g_encCount = posCnt;

        // Roh weitergeben: keine Glaettung, keine Einheitenumrechnung. Der EKF
        // differenziert selbst und waehlt sein Fenster. g_rpm unten ist EMA-
        // gefiltert und wegen der Filterlaufzeit fuer den EKF unbrauchbar.
        if (g_encQueue) {
            EncoderSample s = { tSample, (int32_t)posCnt, ++g_encSeq };
            xQueueSend(g_encQueue, &s, 0);   // nicht blockierend, Overflow egal
        }

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

        // --- Encoder / RPM ---
        if (now - lastRpmCalc >= RPM_INTERVAL) {
            float rdt = (now - lastRpmCalc) / 1000.0f;
            lastRpmCalc = now;

            long delta = posCnt - lastCount;
            lastCount  = posCnt;

            float rpm = (delta / COUNTS_PER_REV) * (60.0f / rdt);
            if (firstTick) { rpmFilt = rpm; firstTick = false; }
            else           { rpmFilt += RPM_EMA * (rpm - rpmFilt); }
            g_rpm = rpmFilt;
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

constexpr int   BATT_CELLS        = 4;        // 4S
constexpr float BATT_WARN_CELL    = 3.80f;    // Warnschwelle pro Zelle
constexpr float BATT_RECOVER_CELL = 3.85f;    // Hysterese: erst darueber wieder entwarnen
constexpr uint32_t BATT_INTERVAL  = 15000;    // alle 15 s messen
constexpr uint32_t BATT_WARN_REPEAT = 60000;  // Warnung hoechstens 1x pro Minute

float    battPackV   = 0.0f;
float    battCellV   = 0.0f;
bool     battLow     = false;
uint32_t battLastRead = 0;
uint32_t battLastWarn = 0;

// Liefert die Packspannung in Volt. analogReadMilliVolts nutzt die
// Werkskalibrierung des ADC - deutlich genauer als analogRead/4095*3.3.
float readBatteryVolts() {
    uint32_t sum = 0;
    for (int i = 0; i < 16; i++) sum += analogReadMilliVolts(PIN_BATTERY);
    float pinV = (sum / 16.0f) / 1000.0f;
    return pinV * battDivider;
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

// Obergrenzen fuer das Antasten. Ohne sie kann die Routine loop() dauerhaft
// blockieren: faellt der Servo nach dem ersten erfolgreichen ReadPos aus,
// liefert er nur noch -1, currentPos aendert sich nie und keine der
// Abbruchbedingungen greift. Dann steht die gesamte Befehlsverarbeitung
// (USB-Konsole UND Jetson-Link) still, waehrend der Motortask weiterlaeuft.
constexpr unsigned long PROBE_TIMEOUT_MS = 8000;
constexpr int PROBE_MAX_READ_FAILS = 10;

int probeLimit(SCSCL* servo, int directionTarget) {
    int currentPos = servo->ReadPos(SERVO_ID);
    if (currentPos == -1) {
        Serial.println("ESP: Servo antwortet nicht - Kalibrierung abgebrochen.");
        return directionTarget;
    }

    // Schrittweite 40 Ticks, um normale Gleitreibung zu ueberwinden
    int stepSize = (directionTarget == 0) ? -40 : 40;
    int stuckCounter = 0;
    int readFails = 0;
    unsigned long tStart = millis();

    Serial.printf("ESP: Starte sanftes Tasten in Richtung %d...\n", directionTarget);

    while (true) {
        int targetPos = currentPos + stepSize;
        if (targetPos < 0)    targetPos = 0;
        if (targetPos > 1023) targetPos = 1023;

        servo->WritePos(SERVO_ID, targetPos, 0, 600);
        vTaskDelay(100 / portTICK_PERIOD_MS);

        int actualPos = servo->ReadPos(SERVO_ID);
        if (actualPos != -1) {
            readFails = 0;
            if (abs(actualPos - currentPos) <= 5) {
                stuckCounter++;
                if (stuckCounter >= 3) {   // 300 ms echte Blockade
                    Serial.printf("ESP: Anschlag sanft ertastet bei Pos: %d\n", actualPos);
                    servo->WritePos(SERVO_ID, actualPos, 0, 0);   // Regeldifferenz auf 0
                    return actualPos;
                }
            } else {
                stuckCounter = 0;
            }
            currentPos = actualPos;
        } else if (++readFails >= PROBE_MAX_READ_FAILS) {
            Serial.printf("ESP: Servo %dx nicht lesbar - Abbruch bei Pos %d\n",
                          readFails, currentPos);
            return currentPos;
        }

        if (millis() - tStart > PROBE_TIMEOUT_MS) {
            Serial.printf("ESP: Antast-Timeout nach %lu ms - Abbruch bei Pos %d\n",
                          PROBE_TIMEOUT_MS, currentPos);
            return currentPos;
        }

        if (currentPos == 0    && directionTarget == 0)    return 0;
        if (currentPos == 1023 && directionTarget == 1023) return 1023;
    }
}

void runCalibrationRoutine(SCSCL* servo) {
    Serial.println("ESP: Starte sichere Mikroschritt-Kalibrierung...");

    // Antrieb waehrend der Kalibrierung stillsetzen.
    setMotorCommand(MOTOR_COAST, false, 0);

    rightLimit = probeLimit(servo, 0);

    // ZWINGEND: Lenkung um 80 Ticks entspannen, bevor die Richtung wechselt
    int relaxPos = rightLimit + 80;
    if (relaxPos > 1023) relaxPos = 1023;
    servo->WritePos(SERVO_ID, relaxPos, 0, 400);
    vTaskDelay(600 / portTICK_PERIOD_MS);

    leftLimit = probeLimit(servo, 1023);

    prefs.putInt("lLimit", leftLimit);
    prefs.putInt("rLimit", rightLimit);

    softwareCenterPos = (leftLimit + rightLimit) / 2;
    Serial.printf("ESP: Kalibrierung beendet & gespeichert. Mitte: %d\n", softwareCenterPos);

    servo->WritePos(SERVO_ID, softwareCenterPos, 0, 600);
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
static inline void writeI64BE(uint8_t* b, int64_t v) {
    for (int i = 0; i < 8; i++) b[i] = (uint8_t)(v >> (56 - 8 * i));
}

// CRC-16/CCITT-FALSE. Polynom 0x1021, Init 0xFFFF, keine Reflexion, kein
// Final-XOR. Testvektor: "123456789" -> 0x29B1.
//
// Die Variante ist hier bewusst festgeschrieben: "crc16" allein ist mehrdeutig
// (MODBUS, XMODEM, KERMIT ... unterscheiden sich in Init, Reflexion und XOR und
// liefern fuer dieselben Daten verschiedene Werte). Die Gegenstelle muss exakt
// diese nachbauen.
//
// Gerechnet wird ueber CMD-Byte + Nutzlast OHNE die beiden CRC-Bytes selbst.
// Das START_BYTE gehoert NICHT dazu - es dient nur der Resynchronisation.
static uint16_t crc16_ccitt(const uint8_t* data, size_t len) {
    uint16_t crc = 0xFFFF;
    for (size_t i = 0; i < len; i++) {
        crc ^= (uint16_t)data[i] << 8;
        for (int b = 0; b < 8; b++) {
            crc = (crc & 0x8000) ? (uint16_t)((crc << 1) ^ 0x1021)
                                 : (uint16_t)(crc << 1);
        }
    }
    return crc;
}

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
        case CMD_TORQUE:        return "TORQUE";
        case CMD_TRIM:          return "TRIM";
        case CMD_PID_SET:       return "PID_SET";
        case CMD_PID_GET:       return "PID_GET";
        case CMD_PID_SAVE:      return "PID_SAVE";
        case CMD_MOVE:          return "MOVE";
        case CMD_MOVE_ABORT:    return "MOVE_ABORT";
        case CMD_PROGRESS:      return "PROGRESS";
        case CMD_BATTERY:       return "BATTERY";
        case CMD_SYNC_REQ:      return "SYNC_REQ";
        case CMD_SYNC_RSP:      return "SYNC_RSP";
        case CMD_ENC_TELEM:     return "ENC_TELEM";
        case CMD_TELEM_RATE:    return "TELEM_RATE";
        case CMD_EMERGENCY:     return "EMERGENCY";
        case CMD_BUTTON:        return "BUTTON";
        case CMD_PID_RSP:       return "PID_RSP";
        case CMD_PID_SAVED:     return "PID_SAVED";
        case CMD_MOVE_DONE:     return "MOVE_DONE";
        case CMD_PROGRESS_RSP:  return "PROGRESS_RSP";
        case CMD_BATTERY_RSP:   return "BATTERY_RSP";
        case CMD_BATTERY_WARN:  return "BATTERY_WARN";
        default:                return "???";
    }
}

class JetsonComms {
private:
    HardwareSerial* serialPort;
    SCSCL* servo;

    unsigned long stateTime;
    uint8_t buffer[16];
    int bufIndex = 0;

    enum State { WAITING_START, WAITING_CMD, READING_DATA };
    State currentState = WAITING_START;
    uint8_t currentCmd = 0;
    uint8_t dataLength = 0;

    // Link-Statistik fuer die Fehlersuche
    uint32_t rxPackets  = 0;   // vollstaendig empfangen und ausgefuehrt
    uint32_t rxUnknown  = 0;   // unbekanntes CMD-Byte verworfen
    uint32_t rxTimeouts = 0;   // Paket blieb unvollstaendig
    uint32_t rxStray    = 0;   // Bytes ausserhalb eines Pakets (z.B. ASCII)
    uint32_t txPackets  = 0;
    unsigned long lastRxMs = 0;

    // Unterdrueckung identischer Wiederholungen im Mitschnitt. Ein Heartbeat
    // im 100-ms-Takt erzeugt sonst 10 nutzlose Zeilen/s und verdeckt genau die
    // Pakete, auf die man wartet.
    uint8_t  lastLogCmd = 0xFF;
    uint8_t  lastLogBuf[16];
    uint8_t  lastLogLen = 0;
    uint32_t repeatCount = 0;
    unsigned long repeatSince = 0;

    // Empfangszeit des Startbytes. Beim Lesen des Startbytes ist noch nicht
    // bekannt, ob ein SYNC_REQ folgt - also wird immer gestempelt und der Wert
    // nur dann verwendet, wenn es einer war.
    int64_t startByteUs = 0;

public:
    JetsonComms(HardwareSerial* s, SCSCL* sv) : serialPort(s), servo(sv) {}

    void begin(unsigned long baud = 115200) {
        // Muss vor begin() stehen, sonst bleiben die Default-Groessen stehen.
        // TX klein halten: ein grosser Sendepuffer erzeugt nur Latenz, weil
        // Pakete dann hinter aelteren warten statt sofort rauszugehen.
        serialPort->setRxBufferSize(512);
        serialPort->setTxBufferSize(256);
        serialPort->begin(baud, SERIAL_8N1, PIN_JETSON_RX, PIN_JETSON_TX);
        stateTime = millis();
    }

    // --- Encoder-Telemetrie (100 Hz) ---
    void sendEncTelem(const EncoderSample& s) {
        uint8_t p[18];
        p[0] = START_BYTE;
        p[1] = CMD_ENC_TELEM;
        p[2] = (uint8_t)(s.seq >> 8);
        p[3] = (uint8_t)s.seq;
        writeI64BE(&p[4], s.t_us);
        writeI32BE(&p[12], s.count);
        uint16_t crc = crc16_ccitt(&p[1], 15);   // CMD + 14 B Nutzlast
        p[16] = (uint8_t)(crc >> 8);
        p[17] = (uint8_t)crc;
        sendPacket(p, sizeof(p), true);          // quiet: kein Log bei 100 Hz
    }

    // --- Zeitsynchronisation ---
    // t2 = Empfang des Startbytes, t3 = unmittelbar vor dem Absenden. Der
    // Jetson rechnet offset = ((t2-t1) + (t3-t4)) / 2; die ESP-Verarbeitungszeit
    // (t3-t2) faellt dabei heraus, eine verzoegerte Antwort verfaelscht das
    // Ergebnis also nicht - solange t2 und t3 ehrlich sind.
    void sendSyncRsp(uint32_t reqId, int64_t t2) {
        uint8_t p[24];
        p[0] = START_BYTE;
        p[1] = CMD_SYNC_RSP;
        writeI32BE(&p[2], (int32_t)reqId);
        writeI64BE(&p[6], t2);

        // Ohne flush() koennte das Paket noch hinter Telemetrie im Sendepuffer
        // stehen und t3 waere um Millisekunden zu frueh. Bei 100 Hz ist der
        // Puffer praktisch immer leer, flush() kehrt sofort zurueck.
        serialPort->flush();
        int64_t t3 = esp_timer_get_time();
        writeI64BE(&p[14], t3);

        uint16_t crc = crc16_ccitt(&p[1], 21);   // CMD + 20 B Nutzlast
        p[22] = (uint8_t)(crc >> 8);
        p[23] = (uint8_t)crc;
        sendPacket(p, sizeof(p), true);
    }

    // --- Paketbau, ohne Serial-Zugriff ---
    // Getrennt vom Senden, weil dieselben Pakete aus zwei Kontexten entstehen:
    // der Link-Task sendet sie direkt, loop() legt sie in die TX-Queue.
    static uint8_t buildButton(uint8_t* p) {
        p[0] = START_BYTE; p[1] = CMD_BUTTON; p[2] = 0x01;   // 0x01 = Pressed
        return 3;
    }

    static uint8_t buildMoveDone(uint8_t* p, const MoveResult& r) {
        p[0] = START_BYTE; p[1] = CMD_MOVE_DONE; p[2] = r.id; p[3] = r.status;
        writeI32BE(&p[4], r.finalDeg10);
        return 8;
    }

    static uint8_t buildBattery(uint8_t* p, uint8_t cmd) {
        p[0] = START_BYTE; p[1] = cmd;
        writeI32BE(&p[2], (int32_t)lroundf(battPackV * 1000.0f));
        int16_t cellmV = (int16_t)lroundf(battCellV * 1000.0f);
        p[6] = (uint8_t)(cellmV >> 8);
        p[7] = (uint8_t)cellmV;
        return 8;
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
        uint8_t p[8];
        sendPacket(p, buildBattery(p, cmd));
    }

    void sendPidParams() {
        PidParams pid = getPid();
        uint8_t p[14] = {START_BYTE, CMD_PID_RSP};
        writeI32BE(&p[2],  (int32_t)lroundf(pid.kp * 1000.0f));
        writeI32BE(&p[6],  (int32_t)lroundf(pid.ki * 1000.0f));
        writeI32BE(&p[10], (int32_t)lroundf(pid.kd * 1000.0f));
        sendPacket(p, sizeof(p));
    }

    void printLinkStats() {
        Serial.printf("Link IO%d(RX)/IO%d(TX) @115200 | debug=%u\n",
                      PIN_JETSON_RX, PIN_JETSON_TX, g_linkDebug);
        Serial.printf("  RX: %lu Pakete, %lu unbekannt, %lu unvollstaendig, %lu Streubytes\n",
                      (unsigned long)rxPackets, (unsigned long)rxUnknown,
                      (unsigned long)rxTimeouts, (unsigned long)rxStray);
        Serial.printf("  TX: %lu Pakete\n", (unsigned long)txPackets);
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
                    if (byte == START_BYTE) {
                        startByteUs = esp_timer_get_time();   // t2 fuer SYNC_REQ
                        currentState = WAITING_CMD;
                        bufIndex = 0;
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
                    } else if (len == 0) {
                        dataLength = 0;
                        logRxPacket();
                        executeCommand();
                        currentState = WAITING_START;
                    } else {
                        dataLength = (uint8_t)len;
                        currentState = READING_DATA;
                    }
                    break;
                }

                case READING_DATA:
                    buffer[bufIndex++] = byte;
                    if (bufIndex >= dataLength) {
                        logRxPacket();
                        executeCommand();
                        currentState = WAITING_START;
                    }
                    break;
            }
        }
    }

private:
    // Ein Paket rausschicken, mitzaehlen und optional mitschneiden.
    // quiet=true fuer die 100-Hz-Telemetrie: 100 printf/s aus einem Task mit
    // Prioritaet 3 wuerden das restliche System lahmlegen.
    void sendPacket(const uint8_t* p, size_t n, bool quiet = false) {
        serialPort->write(p, n);
        txPackets++;
        if (g_linkDebug && !g_plotMode && !quiet) {
            Serial.printf("[TX] %s", cmdName(p[1]));
            for (size_t i = 2; i < n; i++) Serial.printf(" %02X", p[i]);
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
            case CMD_SYNC_REQ:   return 4;
            case CMD_TELEM_RATE: return 1;
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

    // An loop() weiterreichen. Die Nutzlast wird kopiert - buffer ist beim
    // naechsten Paket schon wieder ueberschrieben.
    void deferCommand() {
        if (!g_deferQueue || dataLength > sizeof(DeferredCmd::data)) return;
        DeferredCmd d;
        d.cmd = currentCmd;
        d.len = dataLength;
        memcpy(d.data, buffer, dataLength);
        xQueueSend(g_deferQueue, &d, 0);
    }

    void executeCommand() {
        switch (currentCmd) {

        // ---- Befehle fuer loop(): fassen den Servo an oder schreiben ins
        //      NVS. Beides dauert Millisekunden bis Sekunden und darf den
        //      Link-Task nicht blockieren. ----
        case CMD_SERVO:
        case CMD_CALIBRATE:
        case CMD_TORQUE:
        case CMD_TRIM:
        case CMD_PID_SAVE:
            deferCommand();
            break;

        case CMD_EMERGENCY:
            // Aktiv bremsen statt nur ausrollen, bricht auch eine Fahrt ab.
            setMotorCommand(MOTOR_BRAKE, false, DUTY_MAX);
            Serial.println("ESP: EMERGENCY STOP EXECUTED!");
            break;

        case CMD_SYNC_REQ:
            sendSyncRsp((uint32_t)readI32BE(&buffer[0]), startByteUs);
            break;

        case CMD_TELEM_RATE:
            g_telemHz = (buffer[0] > 100) ? 100 : buffer[0];
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

        case CMD_PID_GET:
            sendPidParams();
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

        case CMD_LED:
            digitalWrite(PIN_LED, buffer[0] ? HIGH : LOW);
            break;
        }
    }

public:
    // --- Verzoegerte Befehle, ausgefuehrt von loop() auf Core 1 ---
    // Laufen dort, weil sie den SCServo-Bus bedienen oder ins NVS schreiben.
    // Statusmeldungen gehen auf die USB-Konsole, nicht mehr auf Serial1: der
    // Link traegt jetzt 100 Hz Binaertelemetrie, in die kein ASCII gehoert.
    void runDeferred(const DeferredCmd& d) {
        switch (d.cmd) {

        case CMD_SERVO: {
            uint8_t id = d.data[0];
            int16_t steerPct = (d.data[1] << 8) | d.data[2];

            // --- Dynamische Hub-Begrenzung auf 80% ---
            const float MAX_THROW_FACTOR = 0.80f;

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

            servo->WritePos(id, physicalPos, 0, 0);
            break;
        }

        case CMD_CALIBRATE:
            runCalibrationRoutine(servo);
            break;

        case CMD_TORQUE:
            printTorque(servo);
            break;

        case CMD_PID_SAVE: {
            bool ok = savePidParams();
            uint8_t p[3] = {START_BYTE, CMD_PID_SAVED, (uint8_t)(ok ? 0x00 : 0x01)};
            queueTx(p, sizeof(p));
            break;
        }

        case CMD_TRIM: {
            uint8_t action = d.data[0];
            if (action == 0x00) {          // Links trimmen (-2)
                trimOffset        -= 2;
                softwareCenterPos -= 2;
                servo->WritePos(SERVO_ID, softwareCenterPos, 0, 0);
                Serial.printf("ESP: Trim L | Offset: %d | Pos: %d\n", trimOffset, softwareCenterPos);
            } else if (action == 0x01) {   // Rechts trimmen (+2)
                trimOffset        += 2;
                softwareCenterPos += 2;
                servo->WritePos(SERVO_ID, softwareCenterPos, 0, 0);
                Serial.printf("ESP: Trim R | Offset: %d | Pos: %d\n", trimOffset, softwareCenterPos);
            } else if (action == 0x02) {   // Offset dauerhaft speichern
                prefs.putInt("offset", trimOffset);
                Serial.printf("ESP: Trim-Offset (%d) im Flash gespeichert!\n", trimOffset);
            }
            break;
        }
        }
    }

    // Aus loop() aufrufen statt direkt zu senden - Serial1 gehoert dem Link-Task.
    static void queueTx(const uint8_t* p, uint8_t n) {
        if (!g_txQueue || n > sizeof(TxPacket::buf)) return;
        TxPacket t;
        t.len = n;
        memcpy(t.buf, p, n);
        xQueueSend(g_txQueue, &t, 0);
    }

    // Ein fertig gebautes Paket aus der TX-Queue rausschicken (Link-Task).
    void sendRaw(const TxPacket& t) { sendPacket(t.buf, t.len); }
};

// ==========================================
// 10. USB-DEBUG-KONSOLE (Core 1)
// ==========================================

MotorDriver planetaryMotor(PIN_MOTOR_PWM, PIN_MOTOR_INA, PIN_MOTOR_INB);
SCSCL sc09Servo;
JetsonComms jetson(&Serial1, &sc09Servo);

// ==========================================
// 9b. LINK-TASK (Core 0)
// ==========================================

// Bedient Serial1 exklusiv. Muss einen eigenen Task haben, weil loop()
// blockieren darf und das auch tut: runCalibrationRoutine() -> probeLimit()
// laeuft bis zu 8 s pro Richtung. Waehrenddessen ginge weder Telemetrie raus
// noch eine Sync-Anfrage rein.
//
// Prioritaet 3 > Motor_Task (1): der Link-Task laeuft nur wenige hundert
// Mikrosekunden pro Zyklus und muss den Motortask verdraengen duerfen, damit
// die Sync-Zeitstempel nicht um dessen Laufzeit verrutschen.
TaskHandle_t LinkTaskHandle;

void linkTask(void* pvParameters) {
    (void)pvParameters;
    uint8_t decimCounter = 0;

    for (;;) {
        // --- 1. Encoder-Telemetrie ---
        // Immer leeren, auch wenn das Senden aus ist - sonst laeuft die Queue
        // voll und der Motortask verwirft stumm.
        EncoderSample s;
        while (xQueueReceive(g_encQueue, &s, 0) == pdTRUE) {
            uint8_t hz = g_telemHz;
            if (hz == 0) continue;

            // Samples kommen im Motortask-Takt (100 Hz). Kleinere Raten durch
            // Dezimierung, hoehere gibt es ohne Aenderung von TASK_PERIOD nicht.
            uint8_t divider = (uint8_t)((100 + hz / 2) / hz);
            if (divider < 1) divider = 1;
            if (++decimCounter >= divider) {
                decimCounter = 0;
                jetson.sendEncTelem(s);
            }
        }

        // --- 2. Empfang ---
        jetson.process();

        // --- 3. Asynchrone Meldungen aus loop() ---
        TxPacket t;
        while (xQueueReceive(g_txQueue, &t, 0) == pdTRUE) {
            jetson.sendRaw(t);
        }

        vTaskDelay(1);   // 1 kHz
    }
}

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
    Serial.println("  e        Telemetrie (Counts, Grad, RPM)");
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
    Serial.println("--- Batterie ---");
    Serial.println("  v        jetzt messen      vc<faktor>  Teiler kalibrieren");
    Serial.println("--- Lenkung ---");
    Serial.println("  x Kalibrierung  t Torque  a/d Trim L/R  s Trim speichern");
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
            Serial.printf("enc=%ld (%.1f grad)  rpm=%.1f  CS(dig)=%d\n",
                          g_encCount, countsToDeg10(g_encCount) / 10.0f,
                          g_rpm, digitalRead(PIN_MOTOR_CS));
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
        case 'v':
            battPackV = readBatteryVolts();
            battCellV = battPackV / BATT_CELLS;
            Serial.printf("Akku: %.2f V Pack | %.3f V/Zelle (%dS)%s\n",
                          battPackV, battCellV, BATT_CELLS,
                          battCellV < BATT_WARN_CELL ? "  *** NIEDRIG ***" : "");
            break;

        // --- Lenkung ---
        case 'x':   // 'c' ist coast -> Kalibrierung liegt auf 'x'
            runCalibrationRoutine(&sc09Servo);
            softwareCenterPos += trimOffset;
            sc09Servo.WritePos(SERVO_ID, softwareCenterPos, 0, 500);
            break;

        case 't':
            printTorque(&sc09Servo);
            break;

        case 'a':   // Trim links
            trimOffset        -= 5;
            softwareCenterPos -= 5;
            sc09Servo.WritePos(SERVO_ID, softwareCenterPos, 0, 0);
            Serial.printf("Trim: %d | Aktuelle Pos: %d\n", trimOffset, softwareCenterPos);
            break;

        case 'd':   // Trim rechts
            trimOffset        += 5;
            softwareCenterPos += 5;
            sc09Servo.WritePos(SERVO_ID, softwareCenterPos, 0, 0);
            Serial.printf("Trim: %d | Aktuelle Pos: %d\n", trimOffset, softwareCenterPos);
            break;

        case 's':   // Trim speichern
            prefs.putInt("offset", trimOffset);
            Serial.println("ESP: Trim-Offset permanent gespeichert!");
            break;

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
    if (now - battLastRead < BATT_INTERVAL) return;
    battLastRead = now;

    battPackV = readBatteryVolts();
    battCellV = battPackV / BATT_CELLS;

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
        uint8_t p[8];
        JetsonComms::queueTx(p, JetsonComms::buildBattery(p, CMD_BATTERY_WARN));
    }
}

// ==========================================
// 12. SETUP & LOOP (Core 1)
// ==========================================

void setup() {
    Serial.begin(115200);
    delay(300);

    // --- Antrieb + Encoder ---
    planetaryMotor.begin();
    pinMode(PIN_MOTOR_CS, INPUT);

    ESP32Encoder::useInternalWeakPullResistors = puType::up;  // Hall open-drain
    encoder.attachFullQuad(PIN_ENC_A, PIN_ENC_B);             // 4x Dekodierung in HW
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
    leftLimit   = prefs.getInt("lLimit", 0);
    rightLimit  = prefs.getInt("rLimit", 1023);
    trimOffset  = prefs.getInt("offset", 0);
    if (prefs.isKey("bdiv")) battDivider = prefs.getFloat("bdiv", BATT_DIVIDER_NOMINAL);
    softwareCenterPos = ((leftLimit + rightLimit) / 2) + trimOffset;
    loadPidParams();

    // --- Tasks auf Core 0 ---
    // Alle Queues muessen stehen, bevor ein Task sie benutzt.
    g_moveResultQueue = xQueueCreate(8, sizeof(MoveResult));
    g_encQueue        = xQueueCreate(8, sizeof(EncoderSample));
    g_txQueue         = xQueueCreate(8, sizeof(TxPacket));
    g_deferQueue      = xQueueCreate(8, sizeof(DeferredCmd));
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
    xTaskCreatePinnedToCore(
        linkTask,
        "Link_Task",
        4096,
        nullptr,
        3,                       // > Motor_Task, damit Sync-Stempel praezise sind
        &LinkTaskHandle,
        0                        // Core 0
    );

    // Aktuelle Servoposition halten, um Startup-Spannung zu vermeiden
    int startPos = sc09Servo.ReadPos(SERVO_ID);
    if (startPos != -1) {
        sc09Servo.WritePos(SERVO_ID, startPos, 0, 0);
    }

    // Erste Batteriemessung sofort, danach im 15-s-Raster
    battPackV = readBatteryVolts();
    battCellV = battPackV / BATT_CELLS;
    battLastRead = millis();

    Serial.printf("System Ready. Limits: L:%d, R:%d | Trim: %d | Akku: %.2f V (%.3f V/Zelle)\n",
                  leftLimit, rightLimit, trimOffset, battPackV, battCellV);
    printHelp();
}

void loop() {
    // --- Button ---
    if (buttonTriggered) {
        buttonTriggered = false;
        static unsigned long lastPressTime = 0;
        if (millis() - lastPressTime > 200) {   // 200 ms Entprellzeit
            lastPressTime = millis();
            uint8_t p[3];
            JetsonComms::queueTx(p, JetsonComms::buildButton(p));
        }
    }

    // --- Befehle, die der Link-Task an uns weitergereicht hat ---
    DeferredCmd dc;
    while (xQueueReceive(g_deferQueue, &dc, 0) == pdTRUE) {
        jetson.runDeferred(dc);
    }

    // --- Ergebnisse abgeschlossener Positionsfahrten an den Jetson melden ---
    MoveResult r;
    while (xQueueReceive(g_moveResultQueue, &r, 0) == pdTRUE) {
        // Auto-Plotter endet mit der Fahrt - erst abschalten, dann melden,
        // sonst landet die Meldung noch im Kurvenstrom.
        if (g_plotAuto) { g_plotMode = false; g_plotAuto = false; }
        uint8_t p[8];
        JetsonComms::queueTx(p, JetsonComms::buildMoveDone(p, r));
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

    // Serial1 wird nicht mehr hier bedient - das macht linkTask auf Core 0.

    // --- Akku alle 15 s ---
    batteryTick();

    // --- Plotter-Ausgabe (nur wenn mit "p" eingeschaltet) ---
    plotTick();
}
