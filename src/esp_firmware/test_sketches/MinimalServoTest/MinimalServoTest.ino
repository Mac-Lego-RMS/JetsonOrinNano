#include <Arduino.h>
#include "SCServo.h"

// --- IMPORTANT: pin assignment ---
#define PIN_SERVO_RX 18
#define PIN_SERVO_TX 17

// initialise the library (SCSCL for SC models such as the SC09)
SCSCL sc; 

int activeID = 1; // start with the default ID 1

void setup() {
  // 1. serial connection to the PC (Serial Monitor)
  Serial.begin(115200);
  delay(2000); // wait briefly so the Serial Monitor can be opened
  Serial.println("\n==================================");
  Serial.println("   STARTING SERVO MINIMAL TEST");
  Serial.println("==================================");

  // 2. serial connection to the servo adapter (Serial2)
  // 1,000,000 baud (1 Mbps) is the default for the SC09
  Serial2.begin(1000000, SERIAL_8N1, PIN_SERVO_RX, PIN_SERVO_TX);
  
  // 3. tell the library to use Serial2
  sc.pSerial = &Serial2;
  delay(500);

  // 4. send a ping
  Serial.print("Pinging servo with ID ");
  Serial.print(activeID);
  Serial.println("...");

  int pingResult = sc.Ping(activeID);

  if (pingResult != -1) {
    Serial.println("[OK] SUCCESS! The servo answered.");
  } else {
    Serial.println("[ERROR] No answer from ID 1.");
    Serial.println("Using the broadcast ID (254) from now on to address all servos...");
    activeID = 254; // 254 addresses all servos on the bus
  }
}

void loop() {
  // test sequence: move back and forth
  Serial.println("-> Moving to position 300");
  sc.WritePos(activeID, 300, 0, 0); // ID, position, time (0 = fast), speed (0 = max)
  delay(2000);

  Serial.println("-> Moving to position 700");
  sc.WritePos(activeID, 700, 0, 0);
  delay(2000);
}