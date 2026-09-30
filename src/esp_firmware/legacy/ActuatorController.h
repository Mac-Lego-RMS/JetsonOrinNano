#ifndef ACTUATOR_CONTROLLER_H
#define ACTUATOR_CONTROLLER_H

#include <Arduino.h>

// --- Pin Definitionen (ESP32-S3) ---
#define PIN_MOTOR_DIR   4
#define PIN_MOTOR_PWM   5
#define PIN_SERVO_RX    17
#define PIN_SERVO_TX    18


// --- MD10C Motor Klasse ---
class MotorDriver {
private:
    uint8_t pwmPin, dirPin;
    uint8_t pwmChannel = 0; // LEDC Kanal

public:
    MotorDriver(uint8_t pwm, uint8_t dir) : pwmPin(pwm), dirPin(dir) {}

    void begin() {
        pinMode(dirPin, OUTPUT);
        // ESP32-S3 LEDC Setup (10-bit Auflösung, 20kHz Frequenz)
        ledcSetup(pwmChannel, 20000, 10);
        ledcAttachPin(pwmPin, pwmChannel);
        stop();
    }

    void drive(bool reverse, uint16_t speed) {
        // Speed limitieren auf 10-bit (0-1023)
        if (speed > 1023) speed = 1023;
        
        digitalWrite(dirPin, reverse ? HIGH : LOW);
        ledcWrite(pwmChannel, speed);
    }

    void stop() {
        ledcWrite(pwmChannel, 0);
    }
};

// --- SC09 Servo Klasse (Serial Bus Protokoll) ---
class ServoDriver {
private:
    HardwareSerial* serialPort;
    
public:
    ServoDriver(HardwareSerial* serial) : serialPort(serial) {}

    void begin(unsigned long baud = 1000000) {
        // SC09 nutzt oft 1Mbps, Pins anpassen je nach Board Layout
        serialPort->begin(baud, SERIAL_8N1, PIN_SERVO_RX, PIN_SERVO_TX);
    }

    // Sendet ein Paket gemäß SCS/Feetech Protokoll
    void setPosition(uint8_t id, uint16_t position, uint16_t time = 0) {
        if (position > 1000) position = 1000;

        uint8_t buf[9];
        buf[0] = 0xFF; // Header
        buf[1] = 0xFF; // Header
        buf[2] = id;
        buf[3] = 0x05; // Länge (Instr + P1 + P2 + P3 + Checksum) -> hier verkürzt für Demo
        buf[4] = 0x03; // WRITE Befehl
        buf[5] = 0x2A; // Register Adresse für Zielposition
        buf[6] = (position >> 8) & 0xFF; // High Byte
        buf[7] = position & 0xFF;        // Low Byte
        
        // Checksumme berechnen: ~(ID + Len + Instr + Params)
        uint8_t sum = buf[2] + buf[3] + buf[4] + buf[5] + buf[6] + buf[7];
        buf[8] = ~sum;

        serialPort->write(buf, 9);
    }
};

#endif