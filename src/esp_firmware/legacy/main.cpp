#include <Arduino.h>
#include "ActuatorController.h"
#include "CommsHandler.h"

// --- Instanziierung ---
// MD10C an GPIO 4 (PWM) und 5 (DIR)
MotorDriver planetaryMotor(PIN_MOTOR_PWM, PIN_MOTOR_DIR);

// SC09 Servo an UART2
ServoDriver sc09Servo(&Serial2);

// Jetson Kommunikation an UART1
JetsonComms jetson(&Serial1, &planetaryMotor, &sc09Servo);

void setup() {
    // Debug Serial (USB C)
    Serial.begin(115200);
    Serial.println("ESP32-S3 Actuator Controller Initialized");

    // Subsysteme starten
    planetaryMotor.begin();
    sc09Servo.begin(1000000); // 1 Mbps ist typisch für SC09
    jetson.begin(115200);     // Baudrate muss mit Jetson-Skript übereinstimmen

    // Kurzer Test beim Boot (Optional)
    Serial.println("System Ready.");
}

void loop() {
    // 1. Nachrichten vom Jetson lesen und verarbeiten
    jetson.process();

    // 2. Sicherheitsüberprüfung (Watchdog)
    jetson.checkSafety();

    // 3. Andere Hintergrundaufgaben...
    // delay() vermeiden für reaktives System!
}

//Pin 1