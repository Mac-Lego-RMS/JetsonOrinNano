#ifndef COMMS_HANDLER_H
#define COMMS_HANDLER_H

#include <Arduino.h>
#include "ActuatorController.h"

// --- Protokoll Definitionen ---
#define START_BYTE      0xA5
#define CMD_MOTOR       0x10
#define CMD_SERVO       0x20
#define CMD_EMERGENCY   0xFF
#define JETSON_TIMEOUT  1000 // Millisekunden bis Not-Halt

class JetsonComms {
private:
    HardwareSerial* serialPort;
    MotorDriver* motor;
    ServoDriver* servo;
    
    unsigned long lastPacketTime;
    uint8_t buffer[10];
    int bufIndex = 0;
    
    enum State { WAITING_START, WAITING_CMD, READING_DATA, WAITING_CHECKSUM };
    State currentState = WAITING_START;
    uint8_t currentCmd = 0;
    uint8_t dataLength = 0;

public:
    JetsonComms(HardwareSerial* s, MotorDriver* m, ServoDriver* sv) 
        : serialPort(s), motor(m), servo(sv) {}

    void begin(unsigned long baud = 115200) {
        serialPort->begin(baud, SERIAL_8N1, 10, 11); // RX=18, TX=17
        lastPacketTime = millis();
    }

    void checkSafety() {
        // Failsafe: Wenn Jetson lange schweigt, Motor aus!
        if (millis() - lastPacketTime > JETSON_TIMEOUT) {
            motor->stop();
            // Optional: LED blinken lassen zur Warnung
        }
    }

    void process() {
        while (serialPort->available()) {
            uint8_t byte = serialPort->read();

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
                        motor->stop();
                        currentState = WAITING_START;
                        lastPacketTime = millis();
                    } else if (currentCmd == CMD_MOTOR) {
                        dataLength = 3; // Dir, SpeedH, SpeedL
                        currentState = READING_DATA;
                    } else if (currentCmd == CMD_SERVO) {
                        dataLength = 3; // ID, PosH, PosL
                        currentState = READING_DATA;
                    } else {
                        currentState = WAITING_START; // Unbekannter Befehl
                    }
                    break;

                case READING_DATA:
                    buffer[bufIndex++] = byte;
                    if (bufIndex >= dataLength) {
                        currentState = WAITING_CHECKSUM; // Optional
                        // Vereinfachung: Wir führen den Befehl direkt aus 
                        // (In Prod-Code hier Checksumme prüfen!)
                        executeCommand();
                        currentState = WAITING_START;
                        lastPacketTime = millis();
                    }
                    break;
                
                case WAITING_CHECKSUM:
                    // Platzhalter für Checksummen-Logik
                    currentState = WAITING_START;
                    break;
            }
        }
    }

private:
    void executeCommand() {
        if (currentCmd == CMD_MOTOR) {
            bool reverse = buffer[0];
            uint16_t speed = (buffer[1] << 8) | buffer[2];
            motor->drive(reverse, speed);
        } 
        else if (currentCmd == CMD_SERVO) {
            uint8_t id = buffer[0];
            uint16_t pos = (buffer[1] << 8) | buffer[2];
            servo->setPosition(id, pos);
        }
    }
};

#endif