#include <Arduino.h>
#include <ESP32Encoder.h> // hardware encoder library

// --- PIN DEFINITIONS ---
#define PIN_MOTOR_DIR   4
#define PIN_MOTOR_PWM   5
#define PIN_ENC_A       6
#define PIN_ENC_B       7

// --- MOTOR CLASS ---
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

// encoder object
ESP32Encoder encoder;

void setup() {
    Serial.begin(115200);
    delay(2000); 
    
    Serial.println("\n==================================");
    Serial.println("   HARDWARE PCNT ENCODER TEST");
    Serial.println("==================================");

    motor.begin();

    // --- ENCODER HARDWARE SETUP ---
    // enable the internal pull-up resistors of the ESP32
    ESP32Encoder::useInternalWeakPullResistors = puType::up;
    
    // attach the encoder to pins 6 and 7. 
    // attachHalfQuad counts 2 edges per tick (very stable). 
    // Alternative: attachFullQuad() counts all 4 edges for maximum resolution.
    encoder.attachHalfQuad(PIN_ENC_A, PIN_ENC_B);
    
    // reset the counter to 0
    encoder.clearCount();
}

void loop() {
    // --- Phase 1: drive forwards ---
    Serial.println("\n>>> Motor FORWARD (speed 500)");
    motor.drive(false, 500); 
    
    for (int i = 0; i < 30; i++) {
        // read the count directly from the hardware
        Serial.print("Ticks: ");
        Serial.println(encoder.getCount());
        delay(100); 
    }

    // --- Phase 2: stop ---
    Serial.println("\n>>> Motor STOP");
    motor.stop();
    delay(1000);

    // --- Phase 3: drive backwards ---
    Serial.println("\n>>> Motor BACKWARD (speed 500)");
    motor.drive(true, 500); 
    
    for (int i = 0; i < 30; i++) {
        Serial.print("Ticks: ");
        Serial.println(encoder.getCount());
        delay(100);
    }

    // --- Phase 4: stop ---
    Serial.println("\n>>> Motor STOP");
    motor.stop();
    delay(2000);
}