/*
 * SCSerial.h
 * hardware interface layer for waveshare serial bus servo
 * date: 2023.6.28
 */


#include "SCSerial.h"

SCSerial::SCSerial()
{
	IOTimeOut = 100;
	pSerial = NULL;
}

SCSerial::SCSerial(u8 End):SCS(End)
{
	IOTimeOut = 100;
	pSerial = NULL;
}

SCSerial::SCSerial(u8 End, u8 Level):SCS(End, Level)
{
	IOTimeOut = 100;
	pSerial = NULL;
}

int SCSerial::readSCS(unsigned char *nDat, int nLen)
{
	int Size = 0;
	int ComData;
	unsigned long t_begin = millis();
	unsigned long t_user;
	while(1){
		ComData = pSerial->read();
		if(ComData!=-1){
			if(nDat){
				nDat[Size] = ComData;
			}
			Size++;
			t_begin = millis();
		}
		if(Size>=nLen){
			break;
		}
		t_user = millis() - t_begin;
		if(t_user>IOTimeOut){
			break;
		}
	}
	return Size;
}

int SCSerial::writeSCS(unsigned char *nDat, int nLen)
{
	if(nDat==NULL){
		return 0;
	}
	return pSerial->write(nDat, nLen);
}

int SCSerial::writeSCS(unsigned char bDat)
{
	return pSerial->write(&bDat, 1);
}

void SCSerial::rFlushSCS()
{
	while(pSerial->read()!=-1);
}

void SCSerial::wFlushSCS()
{
	// Halbduplex-Echo-Fix: Auf diesem Adapter trennt die Richtungsumschaltung
	// (TXEN) das eigene Sendesignal nicht sauber von RX - das gesendete Paket
	// landet als Echo im Empfangspuffer und wird sonst als vermeintliche Antwort
	// gelesen (Wert = Registeradresse + 512). Deshalb hier: warten bis TX
	// physisch raus ist, dann das Echo verwerfen. Die echte Servo-Antwort kommt
	// erst danach und bleibt erhalten.
	if(pSerial){
		pSerial->flush();            // blockiert bis TX-FIFO leer
		delayMicroseconds(50);       // Rest des Echos einlaufen lassen
		// Echo verwerfen, aber BEGRENZT: bei Busrauschen wuerde ein
		// unbegrenztes while() endlos Bytes lesen und loop() blockieren
		// (Befehle stauen sich, Servo faehrt Sekunden spaeter). Das Echo ist
		// nur wenige Bytes lang - 32 als Obergrenze ist reichlich.
		int guard = 32;
		while(guard-- > 0 && pSerial->read()!=-1);
	}
}