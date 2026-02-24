#include <Arduino.h>
#include "SCServo.h"

// --- WICHTIG: Deine Pin-Belegung ---
#define PIN_SERVO_RX 18
#define PIN_SERVO_TX 17

// Bibliothek initialisieren (SCSCL für SC-Modelle wie den SC09)
SCSCL sc; 

int activeID = 1; // Wir starten mit der Standard-ID 1

void setup() {
  // 1. Serielle Verbindung zum PC (Serial Monitor)
  Serial.begin(115200);
  delay(2000); // Kurz warten, damit du den Serial Monitor öffnen kannst
  Serial.println("\n==================================");
  Serial.println("   STARTE SERVO MINIMAL-TEST");
  Serial.println("==================================");

  // 2. Serielle Verbindung zum Servo-Adapter (Serial2)
  // 1.000.000 Baud (1 Mbps) ist der Standard für SC09
  Serial2.begin(1000000, SERIAL_8N1, PIN_SERVO_RX, PIN_SERVO_TX);
  
  // 3. Der Bibliothek sagen, dass sie Serial2 nutzen soll
  sc.pSerial = &Serial2;
  delay(500);

  // 4. Ping-Test senden
  Serial.print("Pinge Servo mit ID ");
  Serial.print(activeID);
  Serial.println(" an...");

  int pingResult = sc.Ping(activeID);

  if (pingResult != -1) {
    Serial.println("[OK] SUCCESS! Servo hat geantwortet!");
  } else {
    Serial.println("[FEHLER] Keine Antwort von ID 1.");
    Serial.println("Nutze ab jetzt Broadcast-ID (254), um alle Servos anzusprechen...");
    activeID = 254; // 254 spricht alle Servos auf dem Bus an
  }
}

void loop() {
  // Test-Sequenz: Hin und her fahren
  Serial.println("-> Fahre auf Position 300");
  sc.WritePos(activeID, 300, 0, 0); // ID, Position, Zeit(0=schnell), Speed(0=max)
  delay(2000);

  Serial.println("-> Fahre auf Position 700");
  sc.WritePos(activeID, 700, 0, 0);
  delay(2000);
}