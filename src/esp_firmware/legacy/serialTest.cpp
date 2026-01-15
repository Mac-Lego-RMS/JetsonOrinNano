#include <Arduino.h>

// Definiere die Pins
#define LED_PIN     1   // Deine Test-LED
#define JETSON_RX   18  // Verbunden mit Jetson TX
#define JETSON_TX   17  // Verbunden mit Jetson RX (für Echo/Feedback)

void setup() {
  // 1. Debugging über USB-C Kabel zum PC
  Serial.begin(115200);
  Serial.println("--- Starte Serial Test ---");

  // 2. Kommunikation zum Jetson
  // WICHTIG: Baudrate muss auf beiden Seiten gleich sein!
  Serial1.begin(115200, SERIAL_8N1, JETSON_RX, JETSON_TX);

  // 3. LED Setup
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW); // Start: Aus
}

void loop() {
  // Prüfen, ob Daten vom Jetson da sind
  if (Serial1.available() > 0) {
    char receivedChar = Serial1.read();

    // Aktion basierend auf dem Zeichen
    if (receivedChar == '1') {
      digitalWrite(LED_PIN, HIGH);
      Serial.println("Befehl empfangen: LED AN");
      Serial1.println("ESP: OK, LED is ON"); // Rückmeldung an Jetson
    } 
    else if (receivedChar == '0') {
      digitalWrite(LED_PIN, LOW);
      Serial.println("Befehl empfangen: LED AUS");
      Serial1.println("ESP: OK, LED is OFF"); // Rückmeldung an Jetson
    }
    else {
      // Unbekannte Zeichen (z.B. Newlines) ignorieren oder anzeigen
      Serial.print("Ignoriert: ");
      Serial.println(receivedChar);
    }
  }
}