/*
 * VNH5019 Motortreiber + Quadratur-Encoder Testskript  (v2)
 * Board : ESP32-S3 (MainPCB, custom)
 * Core  : ESP32 Arduino 3.x
 *
 * WICHTIG: Encoder laeuft jetzt ueber die HARDWARE-PCNT-Einheit (ESP32Encoder-Lib),
 * nicht mehr ueber Software-ISRs. Das war die Ursache fuer das Unter-Zaehlen.
 *
 * >>> Bibliothek installieren: Arduino IDE -> Bibliotheksverwalter ->
 *     "ESP32Encoder" (von Kevin Harrington) installieren. <<<
 *
 * Serielle Befehle (115200 Baud, Newline senden):
 *   f<0-255>  vorwaerts | r<0-255> rueckwaerts | b<0-255> bremsen
 *   c         coast     | z        Encoder-Zaehler auf 0
 */

#include <ESP32Encoder.h>

// ---------------- Pin-Konfiguration (VERIFY!) ----------------
constexpr int PIN_PWM   = 41;   // VNH5019 PWM
constexpr int PIN_INA   = 42;   // VNH5019 INA
constexpr int PIN_INB   = 38;   // VNH5019 INB
constexpr int PIN_CS    = 39;   // VNH5019 CS (nur digital lesbar, kein ADC)
constexpr int PIN_ENC_A = 15;   // Encoder A
constexpr int PIN_ENC_B = 16;   // Encoder B

// ---------------- PWM ----------------
constexpr int PWM_FREQ = 20000;
constexpr int PWM_RES  = 8;

// ---------------- Encoder ----------------
ESP32Encoder encoder;
// --- COUNTS_PER_REV: gemessenen/berechneten Wert des AKTIVEN Motors eintragen ---
// ServoCity DE3 (Open-Collector!): 3 PPR x 4 x 42,875 Getriebe = ~514,5 (Ausgangswelle)
// Pololu 25D #4841 (Push-Pull):    48 CPR (Motorwelle) x 4,4 Getriebe = 211,2 (Ausgangswelle)
//                                  -> Motorwellen-RPM stattdessen: 48,0
constexpr float COUNTS_PER_REV = 408.0f;   // Pololu 4841 (4,4:1), Ausgangswelle. Motorwelle = 48.0f

// ---------------- Motor-Steuerung ----------------
void motorCoast()               { ledcWrite(PIN_PWM, 0);   digitalWrite(PIN_INA, LOW);  digitalWrite(PIN_INB, LOW);  }
void motorForward(uint8_t duty) { digitalWrite(PIN_INA, HIGH); digitalWrite(PIN_INB, LOW);  ledcWrite(PIN_PWM, duty); }
void motorReverse(uint8_t duty) { digitalWrite(PIN_INA, LOW);  digitalWrite(PIN_INB, HIGH); ledcWrite(PIN_PWM, duty); }
void motorBrake(uint8_t duty)   { digitalWrite(PIN_INA, LOW);  digitalWrite(PIN_INB, LOW);  ledcWrite(PIN_PWM, duty); }

// ---------------- Setup ----------------
void setup() {
  Serial.begin(115200);
  delay(300);
  Serial.println("\n=== VNH5019 + Encoder Test v2 (PCNT) [BUILD mit d-Diagnose] ===");
  Serial.println("Befehle: f<0-255> r<0-255> b<0-255> c(coast) z(zero) d(Diagnose) p(Pos-Stream)");

  pinMode(PIN_INA, OUTPUT);
  pinMode(PIN_INB, OUTPUT);
  pinMode(PIN_CS,  INPUT);
  ledcAttach(PIN_PWM, PWM_FREQ, PWM_RES);

  // --- Encoder: Hardware-Quadratur ueber PCNT ---
  ESP32Encoder::useInternalWeakPullResistors = puType::up;  // H370 latched Hall = Open-Drain -> Pull-ups noetig
  encoder.attachFullQuad(PIN_ENC_A, PIN_ENC_B);             // 4x Dekodierung in HW
  // Glitch-Filter in APB-Takten (80 MHz). 250 = ~3,1 us: killt Buersten-Rauschen
  // (Spikes < 1 us), laesst aber echte Flanken durch (bei Vollspeed ~333 us Abstand).
  // 1023 (=12,8 us) war zu nah am Limit und verdeckt nur ein HW-Rauschproblem.
  encoder.setFilter(250);
  encoder.clearCount();

  motorCoast();
}

// ---------------- Loop ----------------
String        cmdBuf;
unsigned long lastReport = 0;
long          lastCount  = 0;
float         rpmFilt    = 0.0f;   // geglaettete RPM (EMA)
bool          firstTick  = true;   // erstes Fenster verwerfen (dt-Startsprung)
bool          diagMode   = false;  // Quadratur-Diagnose (Befehl 'd')
int           lastAB     = -1;     // letzter A/B-Zustand
bool          streamMode = false;  // Positions-Stream fuer Processing (Befehl 'p')
unsigned long lastStream = 0;      // Zeitstempel letzter Stream-Ausgabe

// Glaettungsfaktor 0..1: kleiner = ruhiger aber traeger. 0.3 ist ein guter Start.
constexpr float RPM_EMA = 0.30f;

void handleCommand(String cmd) {
  cmd.trim();
  if (cmd.length() == 0) return;
  char c   = cmd.charAt(0);
  int  val = constrain(cmd.substring(1).toInt(), 0, 255);
  switch (c) {
    case 'f': motorForward(val); Serial.printf("-> forward %d\n", val); break;
    case 'r': motorReverse(val); Serial.printf("-> reverse %d\n", val); break;
    case 'b': motorBrake(val);   Serial.printf("-> brake %d\n",   val); break;
    case 'c': motorCoast();      Serial.println("-> coast");            break;
    case 'z': encoder.clearCount(); lastCount = 0; rpmFilt = 0; firstTick = true; Serial.println("-> encoder = 0"); break;
    case 'd': diagMode = !diagMode; lastAB = -1;
              Serial.printf("-> Quadratur-Diagnose %s (Motor aus, Welle langsam von Hand drehen)\n", diagMode ? "AN" : "AUS"); break;
    case 'p': streamMode = !streamMode;
              Serial.printf("-> Positions-Stream %s\n", streamMode ? "AN (POS,millis,count @50Hz)" : "AUS"); break;
    default:  Serial.println("?? unbekannter Befehl");
  }
}

void loop() {
  while (Serial.available()) {
    char ch = Serial.read();
    if (ch == '\n' || ch == '\r') { handleCommand(cmdBuf); cmdBuf = ""; }
    else                          { cmdBuf += ch; }
  }

  // --- Quadratur-Diagnose: bei jeder A/B-Aenderung ausgeben ---
  if (diagMode) {
    int ab = (digitalRead(PIN_ENC_A) << 1) | digitalRead(PIN_ENC_B);
    if (ab != lastAB) {
      Serial.printf("A=%d B=%d   cnt=%ld\n", (ab >> 1) & 1, ab & 1, (long)encoder.getCount());
      lastAB = ab;
    }
    return;   // im Diagnose-Modus keine RPM-Ausgabe
  }

  // --- Positions-Stream fuer Processing-Visualisierung (Befehl 'p') ---
  // Kompakte, maschinenlesbare Zeile: "POS,<millis>,<count>" mit 50 Hz.
  // millis kommt vom ESP32 -> praezise Geschwindigkeit trotz Serial-Jitter.
  if (streamMode) {
    unsigned long tnow = millis();
    if (tnow - lastStream >= 20) {          // 20 ms = 50 Hz
      lastStream = tnow;
      Serial.printf("POS,%lu,%ld\n", tnow, (long)encoder.getCount());
    }
    return;   // im Stream-Modus keine verbose RPM-Zeile (haelt Serial sauber)
  }

  unsigned long now = millis();
  if (now - lastReport >= 500) {
    float dt   = (now - lastReport) / 1000.0f;
    lastReport = now;

    long  cnt   = (long)encoder.getCount();
    long  delta = cnt - lastCount;
    lastCount   = cnt;

    float rpm   = (delta / COUNTS_PER_REV) * (60.0f / dt);   // Ausgangswelle (roh)
    float tps   = delta / dt;                                // Counts pro Sekunde

    // Erstes Fenster nach Start/Reset verwerfen (dt-Sprung -> Muellwert)
    if (firstTick) { rpmFilt = rpm; firstTick = false; }
    else           { rpmFilt += RPM_EMA * (rpm - rpmFilt); }  // exponentieller Mittelwert

    Serial.printf("enc=%ld   rpm=%.1f  (raw=%.1f  %.0f cps)   CS(dig)=%d\n",
                  cnt, rpmFilt, rpm, tps, digitalRead(PIN_CS));
  }
}