#include <Arduino.h>
#include "SCServo.h"

#include <Preferences.h> // Bibliothek für dauerhaftes Speichern

Preferences prefs;
int trimOffset = 0; // Der gespeicherte Korrekturwert

// ==========================================
// 1. PIN- UND PROTOKOLL-DEFINITIONEN
// ==========================================

#define PIN_MOTOR_DIR   4
#define PIN_MOTOR_PWM   5

#define PIN_SERVO_RX    18 
#define PIN_SERVO_TX    17 
#define PIN_JETSON_RX   10
#define PIN_JETSON_TX   11
#define PIN_LED         13

#define START_BYTE      0xA5
#define CMD_MOTOR       0x10
#define CMD_SERVO       0x20
#define CMD_LED         0x30
#define CMD_EMERGENCY   0xFF


// NEUE PROTOKOLL-BEFEHLE
#define CMD_CALIBRATE   0x40 
#define CMD_TORQUE      0x50 
#define CMD_TRIM 0x60

#define PIN_BUTTON      9
#define CMD_BUTTON      0x70

#define JETSON_TIMEOUT  5000 

// ==========================================
// 2. GLOBALE VARIABLEN FÜR MULTITHREADING & SERVO
// ==========================================

volatile uint16_t targetSpeed = 0;
volatile bool targetReverse = false;

// Servo Kalibrierungs-Parameter
const int SERVO_ID = 1;
const int STALL_THRESHOLD = 350; // 40 % Last für Anschlag (anpassen!)
const int CALIB_SPEED = 200;
const int DELAY_POLL = 20;

int softwareCenterPos = 511; // Standard Mitte
int leftLimit = 0;
int rightLimit = 1023;

TaskHandle_t MotorControlTaskHandle;

// ==========================================
// 3. KLASSEN FÜR AKTUATOREN
// ==========================================


volatile bool buttonTriggered = false;

// Interrupt Service Routine (ISR)
void IRAM_ATTR buttonISR() {
    buttonTriggered = true;
}





class MotorDriver {
private:
    uint8_t pwmPin, dirPin;

public:
    MotorDriver(uint8_t pwm, uint8_t dir) 
        : pwmPin(pwm), dirPin(dir) {}
        
    void begin() {
        pinMode(dirPin, OUTPUT);
        // Neue ESP32 Core 3.x API: ledcAttach(pin, frequenz, auflösung)
        ledcAttach(pwmPin, 20000, 10); 
        
        stop();
    }
    
    void drive(bool reverse, uint16_t speed) {
        if (speed > 1023) speed = 1023;
        digitalWrite(dirPin, reverse ? HIGH : LOW);
        // Neue API: ledcWrite direkt auf den Pin, nicht auf den Kanal
        ledcWrite(pwmPin, speed); 
    }
    
    void stop() {
        ledcWrite(pwmPin, 0);
    }
};




// ==========================================
// 4. MULTITHREADING: DER MOTOR-CONTROL TASK (Core 0)
// ==========================================

void motorControlTask(void * pvParameters) {
    MotorDriver* motor = (MotorDriver*)pvParameters;
    for(;;) {
        motor->drive(targetReverse, targetSpeed);
        vTaskDelay(10 / portTICK_PERIOD_MS); 
    }
}

// ==========================================
// 5. SERVO HILFSFUNKTIONEN (Kalibrierung & Torque)
// ==========================================

void printTorque(SCSCL* servo) {
    int rawLoad = servo->ReadLoad(SERVO_ID);
    if (rawLoad != -1) {
        int actualLoad = rawLoad & 0x3FF; // Bits 0-9 extrahieren
        Serial.print("ESP: Aktuelles Servo-Torque: ");
        Serial.println(actualLoad);
    } else {
        Serial.println("ESP: Fehler beim Lesen des Torques!");
    }
}



int probeLimit(SCSCL* servo, int directionTarget) {
    int currentPos = servo->ReadPos(SERVO_ID);
    if (currentPos == -1) return directionTarget;

    // Schrittweite auf 40 Ticks erhöht, um normale Gleitreibung zu überwinden
    int stepSize = (directionTarget == 0) ? -40 : 40;
    int stuckCounter = 0;

    Serial.printf("ESP: Starte sanftes Tasten in Richtung %d...\n", directionTarget);

    while (true) {
        int targetPos = currentPos + stepSize;
        
        if (targetPos < 0) targetPos = 0;
        if (targetPos > 1023) targetPos = 1023;

        // Geschwindigkeit leicht erhöht für stabileres Anlaufmoment
        servo->WritePos(SERVO_ID, targetPos, 0, 600); 
        
        // Messintervall auf 100 ms verlängert, passend zum längeren Weg
        vTaskDelay(100 / portTICK_PERIOD_MS); 

        int actualPos = servo->ReadPos(SERVO_ID);
        
        if (actualPos != -1) {
            // Toleranz für Stillstand auf <= 5 Ticks angepasst
            if (abs(actualPos - currentPos) <= 5) {
                stuckCounter++;
                // Nach 3 Zyklen (300 ms) echter Blockade abbrechen
                if (stuckCounter >= 3) {
                    Serial.printf("ESP: Anschlag sanft ertastet bei Pos: %d\n", actualPos);
                    
                    // Regeldifferenz sofort auf 0 setzen
                    servo->WritePos(SERVO_ID, actualPos, 0, 0);
                    return actualPos;
                }
            } else {
                stuckCounter = 0; 
            }
            currentPos = actualPos;
        }

        if (currentPos == 0 && directionTarget == 0) return 0;
        if (currentPos == 1023 && directionTarget == 1023) return 1023;
    }
}

void runCalibrationRoutine(SCSCL* servo) {
    Serial.println("ESP: Starte sichere Mikroschritt-Kalibrierung...");

    rightLimit = probeLimit(servo, 0);

    // ZWINGEND: Lenkung um 80 Ticks mechanisch entspannen, bevor die Richtung wechselt
    int relaxPos = rightLimit + 80; 
    if (relaxPos > 1023) relaxPos = 1023;
    servo->WritePos(SERVO_ID, relaxPos, 0, 400);
    vTaskDelay(600 / portTICK_PERIOD_MS); 

    leftLimit = probeLimit(servo, 1023);

    // --- NEU: Werte im NVS speichern ---
    prefs.putInt("lLimit", leftLimit);
    prefs.putInt("rLimit", rightLimit);
    
    softwareCenterPos = (leftLimit + rightLimit) / 2;
    Serial.printf("ESP: Kalibrierung beendet & gespeichert. Mitte: %d\n", softwareCenterPos);
    
    servo->WritePos(SERVO_ID, softwareCenterPos, 0, 600);
}

// ==========================================
// 6. KOMMUNIKATION KLASSE (Jetson Bridge)
// ==========================================



class JetsonComms {
private:
    HardwareSerial* serialPort;
    MotorDriver* motor;
    SCSCL* servo; 
    
    unsigned long lastPacketTime;
    unsigned long stateTime; 
    uint8_t buffer[10];
    int bufIndex = 0;
    
    enum State { WAITING_START, WAITING_CMD, READING_DATA, WAITING_CHECKSUM };
    State currentState = WAITING_START;
    uint8_t currentCmd = 0;
    uint8_t dataLength = 0;

public:
    JetsonComms(HardwareSerial* s, MotorDriver* m, SCSCL* sv) 
        : serialPort(s), motor(m), servo(sv) {}

    void begin(unsigned long baud = 115200) {
        serialPort->begin(baud, SERIAL_8N1, PIN_JETSON_RX, PIN_JETSON_TX); 
        feedWatchdog();
        stateTime = millis();
    }

    void feedWatchdog() {
        lastPacketTime = millis();
    }

    void checkSafety() {
        if (millis() - lastPacketTime > JETSON_TIMEOUT) {
            targetSpeed = 0; 
        }
    }

    void sendButtonEvent() {
        uint8_t packet[] = {START_BYTE, CMD_BUTTON, 0x01}; // 0x01 = Pressed
        serialPort->write(packet, sizeof(packet));
        //Serial.println("ESP: Button pressed, Event an Jetson gesendet.");
    }

    void process() {
        while (serialPort->available()) {
            uint8_t byte = serialPort->read();

            if (currentState != WAITING_START && (millis() - stateTime > 100)) {
                currentState = WAITING_START;
            }
            stateTime = millis(); 

            switch (currentState) {
                case WAITING_START:
                    if (byte == START_BYTE) {
                        currentState = WAITING_CMD;
                        bufIndex = 0;
                    }
                    break;
                case WAITING_CMD:
                    currentCmd = byte;
                    if (currentCmd == CMD_EMERGENCY) {
                        targetSpeed = 0;
                        serialPort->println("ESP: EMERGENCY STOP EXECUTED!");
                        currentState = WAITING_START;
                        feedWatchdog();
                    } else if (currentCmd == CMD_MOTOR || currentCmd == CMD_SERVO) {
                        dataLength = 3; 
                        currentState = READING_DATA;
                    } else if (currentCmd == CMD_LED || currentCmd == CMD_TRIM) {
                        dataLength = 1; 
                        currentState = READING_DATA;
                    } else if (currentCmd == CMD_CALIBRATE || currentCmd == CMD_TORQUE) {
                        executeCommand();
                        currentState = WAITING_START;
                        feedWatchdog();
                    } else {
                        currentState = WAITING_START; 
                    }
                    break;
                case READING_DATA:
                    buffer[bufIndex++] = byte;
                    if (bufIndex >= dataLength) {
                        executeCommand();
                        currentState = WAITING_START;
                        feedWatchdog();
                    }
                    break;
                case WAITING_CHECKSUM:
                    currentState = WAITING_START;
                    break;
            }
        }
    }

private:
    void executeCommand() {
        if (currentCmd == CMD_MOTOR) {
            targetReverse = buffer[0];
            targetSpeed = (buffer[1] << 8) | buffer[2];
            
            //serialPort->printf("ESP: Motor OK | Speed: %d\n", targetSpeed);
        } 
        else if (currentCmd == CMD_SERVO) {
            // SCHUTZSPERRE

            uint8_t id = buffer[0];
            int16_t steerPct = (buffer[1] << 8) | buffer[2]; 
            
            // --- NEU: DYNAMISCHE HUB-BEGRENZUNG AUF 80% ---
            const float MAX_THROW_FACTOR = 0.80; 
            
            // 1. Berechne den maximal möglichen Weg von der Mitte zum jeweiligen Anschlag
            int maxDistRight = softwareCenterPos - rightLimit; 
            int maxDistLeft = leftLimit - softwareCenterPos;
            
            // 2. Multipliziere den Weg mit dem Faktor (z.B. 80%) und addiere/subtrahiere ihn zur Mitte
            int safeRight = softwareCenterPos - (maxDistRight * MAX_THROW_FACTOR);
            int safeLeft = softwareCenterPos + (maxDistLeft * MAX_THROW_FACTOR);
            
            int physicalPos = softwareCenterPos;

            // Grenzen für den empfangenen Prozentwert absichern
            if (steerPct < -100) steerPct = -100;
            if (steerPct > 100) steerPct = 100;

            // Asymmetrisches Mapping mit den neuen, sanfteren Limits
            if (steerPct > 0) {
                physicalPos = map(steerPct, 0, 100, softwareCenterPos, safeLeft);
            } 
            else if (steerPct < 0) {
                physicalPos = map(steerPct, -100, 0, safeRight, softwareCenterPos);
            }

            servo->WritePos(id, physicalPos, 0, 0);
            
            // Debug-Ausgabe zur Kontrolle (kannst du später auskommentieren)
            //serialPort->printf("ESP: Servo | CMD: %d%% | Limit(L/R): %d/%d | Pos: %d\n", steerPct, safeLeft, safeRight, physicalPos);
        }
        else if (currentCmd == CMD_LED) {
            bool turnOn = buffer[0];
            digitalWrite(PIN_LED, turnOn ? HIGH : LOW);
            //serialPort->printf("ESP: LED OK | State: %s\n", turnOn ? "ON" : "OFF");
        }
        else if (currentCmd == CMD_CALIBRATE) {
            runCalibrationRoutine(servo);
            // Jetson mitteilen, wo die neue Mitte liegt
            //serialPort->printf("ESP: Calib OK | Center: %d\n", softwareCenterPos);
        }
        else if (currentCmd == CMD_TORQUE) {
            printTorque(servo);
        }
        else if (currentCmd == CMD_TRIM) {
            uint8_t action = buffer[0];
            
            if (action == 0x00) { // Links trimmen (-2)
                trimOffset -= 2;
                softwareCenterPos -= 2;
                servo->WritePos(SERVO_ID, softwareCenterPos, 0, 0);
                serialPort->printf("ESP: Trim L | Offset: %d | Pos: %d\n", trimOffset, softwareCenterPos);
            } 
            else if (action == 0x01) { // Rechts trimmen (+2)
                trimOffset += 2;
                softwareCenterPos += 2;
                servo->WritePos(SERVO_ID, softwareCenterPos, 0, 0);
                serialPort->printf("ESP: Trim R | Offset: %d | Pos: %d\n", trimOffset, softwareCenterPos);
            } 
            else if (action == 0x02) { // Offset dauerhaft speichern
                prefs.putInt("offset", trimOffset);
                serialPort->printf("ESP: Trim-Offset (%d) im Flash gespeichert!\n", trimOffset);
            }
        }
    }
};

// ==========================================
// 7. HAUPTPROGRAMM (SETUP & LOOP auf Core 1)
// ==========================================

MotorDriver planetaryMotor(PIN_MOTOR_PWM, PIN_MOTOR_DIR);
SCSCL sc09Servo; 
JetsonComms jetson(&Serial1, &planetaryMotor, &sc09Servo);

void setup() {
    Serial.begin(115200);

    planetaryMotor.begin();
    jetson.begin(115200);     
    // Button Setup
    pinMode(PIN_BUTTON, INPUT_PULLUP);
    pinMode(PIN_LED, OUTPUT);
    attachInterrupt(digitalPinToInterrupt(PIN_BUTTON), buttonISR, FALLING); // FALLING, da NO-Taster auf GND zieht

    Serial2.begin(1000000, SERIAL_8N1, PIN_SERVO_RX, PIN_SERVO_TX);
    sc09Servo.pSerial = &Serial2; 
    delay(500);

    // --- MULTITHREADING SETUP ---
    xTaskCreatePinnedToCore(
        motorControlTask,        
        "Motor_Task",            
        4000,                    
        (void*)&planetaryMotor,  
        1,                       
        &MotorControlTaskHandle, 
        0                        
    );

    // Lese aktuelle Position und halte sie, um Startup-Spannung (Torque 964) zu vermeiden
    int startPos = sc09Servo.ReadPos(SERVO_ID);
    if (startPos != -1) {
        sc09Servo.WritePos(SERVO_ID, startPos, 0, 0); 
    }
    
    // NVS initialisieren
    prefs.begin("steering", false);
    
    // Lade Limits (Default-Werte falls nichts gespeichert ist: 0 und 1023)
    leftLimit = prefs.getInt("lLimit", 0);
    rightLimit = prefs.getInt("rLimit", 1023);
    trimOffset = prefs.getInt("offset", 0);
    
    // Berechne die Mitte basierend auf den geladenen Werten
    softwareCenterPos = ((leftLimit + rightLimit) / 2) + trimOffset;
    
    Serial.printf("System Ready. Geladene Limits: L:%d, R:%d | Trim: %d\n", leftLimit, rightLimit, trimOffset);
    Serial.println("Trim-Modus: 'A' (Links), 'D' (Rechts), 'S' (Speichern)");
} // Ende von setup()


void loop() {
    if (buttonTriggered) {
        buttonTriggered = false; // Flag sofort zurücksetzen
        static unsigned long lastPressTime = 0;
        
        // 200 ms Entprellzeit
        if (millis() - lastPressTime > 200) { 
            lastPressTime = millis();
            jetson.sendButtonEvent();
            //Serial.println("Button Pressed!");
        }
    }
    if (Serial.available() > 0) {
        char cmd = Serial.read();
        
        // 1. Kalibrierung & Torque (wie gehabt)
        if (cmd == 'C' || cmd == 'c') {
            runCalibrationRoutine(&sc09Servo);
            // Nach Kalibrierung den aktuellen Offset wieder draufrechnen
            softwareCenterPos += trimOffset; 
            sc09Servo.WritePos(SERVO_ID, softwareCenterPos, 0, 500);
        } 
        else if (cmd == 'T' || cmd == 't') {
            printTorque(&sc09Servo);
        }
        
        // 2. LIVE-TRIMMING
        else if (cmd == 'A' || cmd == 'a') { // Trim Links
            trimOffset -= 5;
            softwareCenterPos -= 5;
            sc09Servo.WritePos(SERVO_ID, softwareCenterPos, 0, 0);
            Serial.printf("Trim: %d | Aktuelle Pos: %d\n", trimOffset, softwareCenterPos);
        }
        else if (cmd == 'D' || cmd == 'd') { // Trim Rechts
            trimOffset += 5;
            softwareCenterPos += 5;
            sc09Servo.WritePos(SERVO_ID, softwareCenterPos, 0, 0);
            Serial.printf("Trim: %d | Aktuelle Pos: %d\n", trimOffset, softwareCenterPos);
        }
        
        // 3. SPEICHERN
        else if (cmd == 'S' || cmd == 's') {
            prefs.putInt("offset", trimOffset);
            Serial.println("ESP: Trim-Offset permanent gespeichert!");
        }
    }

    jetson.process();
    jetson.checkSafety();
}