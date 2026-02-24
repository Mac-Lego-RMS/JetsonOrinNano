#include <Arduino.h>
#include "SCServo.h"
#include <ESP32Encoder.h> // Die neue Hardware-Encoder Bibliothek

// ==========================================
// 1. PIN- UND PROTOKOLL-DEFINITIONEN
// ==========================================

#define PIN_MOTOR_DIR   4
#define PIN_MOTOR_PWM   5
#define PIN_ENC_A       6   
#define PIN_ENC_B       7   

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
#define JETSON_TIMEOUT  5000 

// ==========================================
// 2. GLOBALE VARIABLEN FÜR MULTITHREADING
// ==========================================

volatile uint16_t targetSpeed = 0;
volatile bool targetReverse = false;

ESP32Encoder encoder; // Unser Hardware-Encoder Objekt
TaskHandle_t MotorControlTaskHandle;

// ==========================================
// 3. KLASSEN FÜR AKTUATOREN
// ==========================================

class MotorDriver {
private:
    uint8_t pwmPin, dirPin;
public:
    MotorDriver(uint8_t pwm, uint8_t dir) : pwmPin(pwm), dirPin(dir) {}
    void begin() {
        pinMode(dirPin, OUTPUT);
        ledcAttach(pwmPin, 20000, 10);
        stop();
    }
    void drive(bool reverse, uint16_t speed) {
        if (speed > 1023) speed = 1023;
        digitalWrite(dirPin, reverse ? HIGH : LOW);
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
    
    // Dieser Loop läuft unabhängig auf Core 0!
    for(;;) {
        // Motor mit den globalen Zielwerten ansteuern
        motor->drive(targetReverse, targetSpeed);
        
        // Exakt 10 Millisekunden warten (100 Hz)
        vTaskDelay(10 / portTICK_PERIOD_MS); 
    }
}

// ==========================================
// 5. KOMMUNIKATION KLASSE (Jetson Bridge)
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
        lastPacketTime = millis();
        stateTime = millis();
    }

    void checkSafety() {
        if (millis() - lastPacketTime > JETSON_TIMEOUT) {
            targetSpeed = 0; // Notstopp bei Verbindungsabbruch
        }
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
                        lastPacketTime = millis();
                    } else if (currentCmd == CMD_MOTOR || currentCmd == CMD_SERVO) {
                        dataLength = 3; 
                        currentState = READING_DATA;
                    } else if (currentCmd == CMD_LED) {
                        dataLength = 1; 
                        currentState = READING_DATA;
                    } else {
                        currentState = WAITING_START; 
                    }
                    break;
                case READING_DATA:
                    buffer[bufIndex++] = byte;
                    if (bufIndex >= dataLength) {
                        executeCommand();
                        currentState = WAITING_START;
                        lastPacketTime = millis();
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
            
            // Lese die Encoder Ticks direkt aus der Hardware!
            long currentTicks = encoder.getCount(); 
            
            serialPort->print("ESP: Motor OK | Speed: ");
            serialPort->print(targetSpeed);
            serialPort->print(" | Ticks: ");
            serialPort->println(currentTicks); // Sende Ticks an Jetson/ROS
        } 
        else if (currentCmd == CMD_SERVO) {
            uint8_t id = buffer[0];
            uint16_t pos = (buffer[1] << 8) | buffer[2];
            servo->WritePos(id, pos, 0, 0);
            
            serialPort->print("ESP: Servo OK | ID: ");
            serialPort->print(id);
            serialPort->print(" | Pos: ");
            serialPort->println(pos);
        }
        else if (currentCmd == CMD_LED) {
            bool turnOn = buffer[0];
            digitalWrite(PIN_LED, turnOn ? HIGH : LOW);
            serialPort->print("ESP: LED OK | State: ");
            serialPort->println(turnOn ? "ON" : "OFF");
        }
    }
};

// ==========================================
// 6. HAUPTPROGRAMM (SETUP & LOOP auf Core 1)
// ==========================================

MotorDriver planetaryMotor(PIN_MOTOR_PWM, PIN_MOTOR_DIR);
SCSCL sc09Servo; 
JetsonComms jetson(&Serial1, &planetaryMotor, &sc09Servo);

void setup() {
    Serial.begin(115200);
    
    // --- ENCODER SETUP ---
    ESP32Encoder::useInternalWeakPullResistors = puType::up;
    encoder.attachHalfQuad(PIN_ENC_A, PIN_ENC_B);
    encoder.clearCount();

    planetaryMotor.begin();
    jetson.begin(115200);     

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
        0                        /* Läuft auf Core 0 */
    );

    sc09Servo.WritePos(1, 500, 0, 0); // Servo in die Mitte
    Serial.println("System Ready. FreeRTOS Dual-Core & PCNT Encoder aktiv.");
}

void loop() {
    // Core 1 kümmert sich nur noch um ROS 2 Nachrichten
    jetson.process();
    jetson.checkSafety();
}