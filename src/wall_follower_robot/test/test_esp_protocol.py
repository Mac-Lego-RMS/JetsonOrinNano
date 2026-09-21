"""Hardware-freie Tests fuer Serialisierung und Parsing der ESP-Bridge."""

from wall_follower_robot.esp_serial_bridge import (
    CMD_BATTERY_WARN,
    CMD_BUTTON,
    CMD_MOVE_DONE,
    EspProtocol,
    EV_PACKET,
    EV_TEXT,
    PacketParser,
    decode_battery,
    decode_move_done,
    decode_pid_rsp,
    decode_progress,
)


# --- Serialisierung: die *1000-Kodierung ist die Hauptfehlerquelle -----------

def test_pid_set_float_x1000():
    # Kp = 4.5 -> 4500 als int32 big-endian
    assert EspProtocol.pid_set(0, 4.5) == bytes([0xA5, 0x80, 0x00, 0x00, 0x00, 0x11, 0x94])


def test_pid_set_integer_param_still_x1000():
    # maxDuty = 700 MUSS als 700000 kodiert werden, nicht als 700.
    pkt = EspProtocol.pid_set(4, 700)
    assert pkt[:3] == bytes([0xA5, 0x80, 0x04])
    assert int.from_bytes(pkt[3:], 'big') == 700000


def test_pid_set_timeout_large_value():
    # Timeout 15 s -> 15000000
    pkt = EspProtocol.pid_set(7, 15000.0)  # ms als realer Wert
    assert int.from_bytes(pkt[3:], 'big', signed=True) == 15000000


def test_motor_big_endian_speed():
    assert EspProtocol.motor(0, 1023) == bytes([0xA5, 0x10, 0x00, 0x03, 0xFF])
    assert EspProtocol.motor(1, 0) == bytes([0xA5, 0x10, 0x01, 0x00, 0x00])


def test_motor_speed_clamped():
    assert EspProtocol.motor(0, 5000) == bytes([0xA5, 0x10, 0x00, 0x03, 0xFF])


def test_servo_negative_two_complement():
    # -100 als int16 big-endian = 0xFF9C
    assert EspProtocol.servo(1, -100) == bytes([0xA5, 0x20, 0x01, 0xFF, 0x9C])
    assert EspProtocol.servo(1, 100) == bytes([0xA5, 0x20, 0x01, 0x00, 0x64])


def test_move_negative_target():
    # -45.0 Grad -> -450 in 1/10 Grad
    pkt = EspProtocol.move(7, -450)
    assert pkt[:3] == bytes([0xA5, 0x90, 0x07])
    assert int.from_bytes(pkt[3:], 'big', signed=True) == -450


def test_zero_payload_commands():
    assert EspProtocol.emergency() == bytes([0xA5, 0xFF])
    assert EspProtocol.pid_save() == bytes([0xA5, 0x83])
    assert EspProtocol.calibrate() == bytes([0xA5, 0x40])


# --- Parser ------------------------------------------------------------------

def test_parse_button():
    events = PacketParser().feed(bytes([0xA5, CMD_BUTTON, 0x01]))
    assert events == [(EV_PACKET, CMD_BUTTON, bytes([0x01]))]


def test_parse_move_done_roundtrip():
    payload = bytes([0x07, 0x00]) + (905).to_bytes(4, 'big', signed=True)
    events = PacketParser().feed(bytes([0xA5, CMD_MOVE_DONE]) + payload)
    assert len(events) == 1
    _, cmd, pl = events[0]
    assert cmd == CMD_MOVE_DONE
    assert decode_move_done(pl) == (7, 0, 905)


def test_parse_battery():
    payload = (15200).to_bytes(4, 'big', signed=True) + (3800).to_bytes(2, 'big', signed=True)
    events = PacketParser().feed(bytes([0xA5, CMD_BATTERY_WARN]) + payload)
    assert decode_battery(events[0][2]) == (15200, 3800)


def test_ascii_text_between_packets():
    stream = b'System Ready. v1.0\n' + bytes([0xA5, CMD_BUTTON, 0x01])
    events = PacketParser().feed(stream)
    assert (EV_TEXT, 'System Ready. v1.0') in events
    assert (EV_PACKET, CMD_BUTTON, bytes([0x01])) in events


def test_unknown_cmd_resyncs():
    # 0x99 ist unbekannt -> verwerfen, danach gueltiges Button-Paket erkennen.
    stream = bytes([0xA5, 0x99, 0x12, 0xA5, CMD_BUTTON, 0x01])
    events = PacketParser().feed(stream)
    packets = [e for e in events if e[0] == EV_PACKET]
    assert packets == [(EV_PACKET, CMD_BUTTON, bytes([0x01]))]


def test_split_packet_across_feeds():
    parser = PacketParser()
    assert parser.feed(bytes([0xA5, CMD_MOVE_DONE, 0x07])) == []
    rest = bytes([0x00]) + (100).to_bytes(4, 'big', signed=True)
    events = parser.feed(rest)
    assert len(events) == 1
    assert decode_move_done(events[0][2]) == (7, 0, 100)


def test_pid_rsp_decode():
    payload = (4250).to_bytes(4, 'big', signed=True) \
        + (300).to_bytes(4, 'big', signed=True) \
        + (80).to_bytes(4, 'big', signed=True)
    kp, ki, kd = decode_pid_rsp(payload)
    assert (round(kp, 3), round(ki, 3), round(kd, 3)) == (4.25, 0.3, 0.08)


def test_progress_decode():
    payload = bytes([0x07, 0x01, 0x32]) \
        + (450).to_bytes(4, 'big', signed=True) \
        + (900).to_bytes(4, 'big', signed=True)
    events = PacketParser().feed(bytes([0xA5, 0x94]) + payload)
    assert decode_progress(events[0][2]) == (7, 1, 50, 450, 900)


def test_stale_partial_packet_is_dropped():
    parser = PacketParser(timeout=0.1)
    # Angefangenes Paket bei t=0
    parser.feed(bytes([0xA5, CMD_MOVE_DONE, 0x07]), now=0.0)
    # Neuer, kompletter Button-Frame deutlich spaeter -> altes Teilpaket verworfen
    events = parser.feed(bytes([0xA5, CMD_BUTTON, 0x01]), now=1.0)
    assert events == [(EV_PACKET, CMD_BUTTON, bytes([0x01]))]
