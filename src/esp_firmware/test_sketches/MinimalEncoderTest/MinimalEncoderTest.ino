#include <Arduino.h>
#include <ESP32Encoder.h> // Binde die Hardware-Encoder Bibliothek ein

// --- PIN DEFINITIONEN ---
#define PIN_MOTOR_DIR   4
#define PIN_MOTOR_PWM   5
#define PIN_ENC_A       6
#define PIN_ENC_B       7

// --- MOTOR KLASSE ---
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

MotorDriver motor(PIN_MOTOR_PWM, PIN_MOTOR_DIR);

// Erstelle das Encoder-Objekt
ESP32Encoder encoder;

void setup() {
    Serial.begin(115200);
    delay(2000); 
    
    Serial.println("\n==================================");
    Serial.println("   HARDWARE PCNT ENCODER TEST");
    Serial.println("==================================");

    motor.begin();

    // --- ENCODER HARDWARE SETUP ---
    // Interne Pullup-Widerstände des ESP32 aktivieren
    ESP32Encoder::useInternalWeakPullResistors = puType::up;
    
    // Encoder an Pin 6 und 7 binden. 
    // attachHalfQuad zählt 2 Flanken pro Tick (sehr stabil). 
    // Alternativ: attachFullQuad() zählt alle 4 Flanken für maximale Auflösung.
    encoder.attachHalfQuad(PIN_ENC_A, PIN_ENC_B);
    
    // Zähler auf 0 setzen
    encoder.clearCount();
}

void loop() {
    // --- Phase 1: Vorwärts fahren ---
    Serial.println("\n>>> Motor VORWÄRTS (Speed 500)");
    motor.drive(false, 500); 
    
    for (int i = 0; i < 30; i++) {
        // Zählerstand direkt aus der Hardware abfragen!
        Serial.print("Ticks: ");
        Serial.println(encoder.getCount());
        delay(100); 
    }

    // --- Phase 2: Stopp ---
    Serial.println("\n>>> Motor STOPP");
    motor.stop();
    delay(1000);

    // --- Phase 3: Rückwärts fahren ---
    Serial.println("\n>>> Motor RÜCKWÄRTS (Speed 500)");
    motor.drive(true, 500); 
    
    for (int i = 0; i < 30; i++) {
        Serial.print("Ticks: ");
        Serial.println(encoder.getCount());
        delay(100);
    }

    // --- Phase 4: Stopp ---
    Serial.println("\n>>> Motor STOPP");
    motor.stop();
    delay(2000);
}