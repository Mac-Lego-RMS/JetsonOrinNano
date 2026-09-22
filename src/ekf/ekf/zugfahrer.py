#!/usr/bin/env python3
"""
Eine Rangierfolge Zug fuer Zug ueber die Bruecke abfahren.

Gemeinsam genutzt von ausparken_test_node und round1_controller_node: beide
fahren Zuege auf DEMSELBEN Weg. Die Messreihen aus dem Ausparktest gelten fuer
genau diese Ausfuehrung -- eine zweite Implementierung im Regler wuerde
frueher oder spaeter davon abweichen, und dann gelten die Messungen nicht mehr.

Ablauf je Zug:
  1. Lenkwert lenk_wartezeit lang wiederholen (im Stand lenken; der erste
     Befehl auf einer frischen Verbindung geht in der DDS-Erkennung verloren).
  2. Strecke als Wellendrehung in Grad auf /esp_serial_bridge/move schicken.
     Die Strecke regelt der ESP ueber die Encoder, nicht der EKF.
  3. Auf die Quittung auf /esp_serial_bridge/move_done warten.

Waehrend eines Zuges darf niemand /cmd_vel senden: die Bruecke wuerde die
Lenkung neu stellen, und ein Motorbefehl loest die laufende Fahrt ab.

Der Besitzer (``node``) muss bereitstellen:
    pub_steer, pub_move            Publisher (Float32)
    lenk_wartezeit, zug_timeout    Sekunden
    weg_toleranz_cm                Toleranz fuer die Plausibilitaetspruefung
    get_logger()
und ``quittung`` von aussen setzen, wenn move_done eintrifft:
    fahrer.quittung = (zeitpunkt, status, wert)
"""
import math

from std_msgs.msg import Float32

from ekf.ausparken import bahn, cm_zu_grad


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def pose_text(pose):
    return ('x=%+.3f m  y=%+.3f m  Kurs=%+.1f grad'
            % (pose[0], pose[1], math.degrees(pose[2])))


def umkehren(schritte):
    """Die Folge, die den Hinweg aufhebt: rueckwaerts durch die Liste, jede
    Strecke negiert, jede Lenkung unveraendert. Aus dem Ausparken wird so das
    Einparken -- am Roboter nachgemessen auf rund 2 cm genau."""
    return [(lenk, -cm) for lenk, cm in reversed(schritte)]


class Fahrer:
    """Faehrt eine Schrittfolge ueber die Bruecke ab.

    Erst lenken, dann die Positionsfahrt ausloesen, dann auf die Quittung
    warten. Kein /cmd_vel dazwischen.
    """

    def __init__(self, node, schritte, name):
        self.node = node
        self.schritte = schritte
        self.name = name
        self.i = 0
        self.phase = 'lenken'
        self.gesendet = False
        self.t0 = 0.0
        self.quittung = None
        self.los_t = None
        self.pose0 = None
        self.fehler = None

    @property
    def fertig(self):
        return self.i >= len(self.schritte)

    def takt(self, jetzt, pose):
        """Einmal pro Regeltakt aufrufen. True, wenn fertig oder abgebrochen
        (dann steht der Grund in ``fehler``)."""
        if self.fertig:
            return True
        lenk, cm = self.schritte[self.i]

        if self.phase == 'lenken':
            if not self.gesendet:
                self.gesendet = True
                self.t0 = jetzt
            self.node.pub_steer.publish(Float32(data=float(lenk)))
            if jetzt - self.t0 < self.node.lenk_wartezeit:
                return False
            self.quittung = None
            self.los_t = jetzt
            self.pose0 = pose
            self.node.pub_move.publish(Float32(data=float(cm_zu_grad(cm))))
            self.phase = 'fahren'
            self.node.get_logger().info(
                "%s Zug %d/%d: Lenkung %+.0f %%, %+.1f cm"
                % (self.name, self.i + 1, len(self.schritte), lenk, cm))
            return False

        q = self.quittung
        if q is not None and q[0] >= self.los_t:
            status = q[1]
            plan = bahn((0.0, 0.0, 0.0), [(lenk, cm)])[-1][0]
            soll_dreh = math.degrees(plan[2])
            soll_weg = math.hypot(plan[0], plan[1])
            ist_dreh = math.degrees(wrap(pose[2] - self.pose0[2]))
            ist_weg = math.hypot(pose[0] - self.pose0[0], pose[1] - self.pose0[1])
            self.node.get_logger().info(
                "%s Zug %d fertig: Drehung %+.1f grad (Modell %+.1f), "
                "Weg %.1f cm (Modell %.1f)%s"
                % (self.name, self.i + 1, ist_dreh, soll_dreh,
                   ist_weg * 100, soll_weg * 100,
                   '' if status == 0 else '  [Status %d]' % status))
            self.node.get_logger().info("           steht bei  %s"
                                        % pose_text(pose))
            if status == 2:
                self.fehler = ("Zug %d wurde von einem Motorbefehl abgeloest"
                               % (self.i + 1))
                return True
            # ACHTUNG: ist_weg kommt aus der EKF-Pose. Springt die Lokalisierung,
            # schlaegt diese Pruefung fehl, obwohl der Zug korrekt gefahren ist
            # (so geschehen im CCW-Test, Lauf 5). Sobald geklaert ist, was
            # move_done in data[2] meldet, sollte hier q[2] stehen.
            if status == 1 and abs(ist_weg - soll_weg) > \
                    self.node.weg_toleranz_cm / 100.0:
                self.fehler = ("Zug %d: Zeitueberschreitung UND %.1f cm zu "
                               "kurz" % (self.i + 1, (soll_weg - ist_weg) * 100))
                return True
            self.i += 1
            self.phase = 'lenken'
            self.gesendet = False
            return self.fertig

        if jetzt - self.los_t > self.node.zug_timeout:
            self.fehler = ("Zug %d ohne Quittung nach %.0f s -- laeuft der "
                           "esp_serial_bridge?" % (self.i + 1, self.node.zug_timeout))
            return True
        return False