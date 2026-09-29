"""The ONE table of /rosout log-text patterns used by the toolkit.

The controller (src/ekf/ekf/round1_controller_node.py) and a few other nodes
report results only as log text, e.g. the parking result. The code base is
about to be translated from German to English, so every event has one regular
expression for the CURRENT German text (copied from the source) and one
TOLERANT regular expression for a plausible English translation. When the
translation lands, check the English strings in the source against the
'en' rows below and adjust them here -- nowhere else.

Columns
-------
key        event name used by the tools (several rows may share a key)
lang       'de' = current source text, 'en' = expected translation
min_level  minimum rcl log level for the row to count (10 DEBUG, 20 INFO,
           30 WARN, 40 ERROR); keeps e.g. a harmless "emergency stop service
           ready" info line from counting as an emergency stop
regex      Python regex; named groups become the parsed values. Numbers are
           matched by NUM (both '.' and ',' decimal separators)
source     where the German string is produced

Numbers are returned as float by parse_number().
"""
import re
from dataclasses import dataclass

NUM = r'[-+]?\d+(?:[.,]\d+)?'

LEVEL_NAMES = {10: 'DEBUG', 20: 'INFO', 30: 'WARN', 40: 'ERROR', 50: 'FATAL'}


@dataclass(frozen=True)
class LogPattern:
    key: str
    lang: str
    min_level: int
    regex: str
    source: str
    flags: int = 0

    def compiled(self):
        return re.compile(self.regex.replace('NUM', NUM), self.flags)


I = re.IGNORECASE

# fmt: off
PATTERNS = [
    # --- parking result -------------------------------------------------------
    # "EINGEPARKT. base_link %.1f cm von der Aussenbande (erwartet %s), Kurs
    #  %+.1f grad zur Bande = %.1f cm Achsdifferenz (Regel: hoechstens 2 cm)."
    LogPattern('parked', 'de', 20,
               r'EINGEPARKT\.\s*base_link\s+(?P<dist_cm>NUM)\s*cm\s+von\s+der\s+Aussenbande\s*'
               r'\(erwartet\s*(?P<expected>[^)]*)\),\s*Kurs\s*(?P<heading_deg>NUM)\s*grad\s+zur\s+Bande\s*'
               r'=\s*(?P<axle_cm>NUM)\s*cm\s+Achsdifferenz',
               'round1_controller_node.py _einparken_fertig()'),
    # e.g. "PARKED. base_link 4.3 cm from the outer wall (expected 4.0 cm),
    #       heading +0.8 deg to the wall = 0.1 cm axle difference (rule: ...)"
    LogPattern('parked', 'en', 20,
               r'PARKED\b.*?(?P<dist_cm>NUM)\s*cm\s+(?:from|to|off)\s+(?:the\s+)?outer\s+'
               r'(?:wall|barrier|band|boundary|border)\w*.*?\(expected\s*(?P<expected>[^)]*)\).*?'
               r'(?:heading|course|yaw)\s*(?P<heading_deg>NUM)\s*(?:deg|degrees?|°).*?'
               r'(?P<axle_cm>NUM)\s*cm\s+(?:of\s+)?axle',
               'expected translation', I),
    # Fallback report without numbers: "EINGEPARKT bei (%.2f, %.2f)."
    LogPattern('parked_at', 'de', 20,
               r'EINGEPARKT\s+bei\s*\((?P<x>NUM),\s*(?P<y>NUM)\)',
               'round1_controller_node.py _einparken_fertig()'),
    LogPattern('parked_at', 'en', 20,
               r'PARKED\s+at\s*\((?P<x>NUM),\s*(?P<y>NUM)\)',
               'expected translation', I),
    # "Endlage laut Buchtmessung: Heck %.1f cm, Front %.1f cm Luft%s."
    LogPattern('bay_clearance', 'de', 20,
               r'Endlage\s+laut\s+Buchtmessung:\s*Heck\s*(?P<rear_cm>NUM)\s*cm,\s*Front\s*(?P<front_cm>NUM)\s*cm',
               'round1_controller_node.py _einparken_fertig()'),
    LogPattern('bay_clearance', 'en', 20,
               r'(?:final|end)\s+(?:pose|position).*?bay.*?rear\s*(?P<rear_cm>NUM)\s*cm,\s*front\s*(?P<front_cm>NUM)\s*cm',
               'expected translation', I),
    # "Einparken rueckwaerts fertig: %.1f cm neben der Parklinie, Kurs %+.1f grad, ..."
    LogPattern('park_reverse_done', 'de', 20,
               r'Einparken\s+rueckwaerts\s+fertig:\s*(?P<line_offset_cm>NUM)\s*cm\s+neben\s+der\s+Parklinie,\s*'
               r'Kurs\s*(?P<heading_deg>NUM)',
               'round1_controller_node.py (PARK_RUECK)'),
    LogPattern('park_reverse_done', 'en', 20,
               r'(?:parking|park-in|parking in)\s+reverse\w*\s+(?:done|finished|complete)\w*[:\s]*'
               r'(?P<line_offset_cm>NUM)\s*cm\s+(?:beside|from|off|next to)\s+the\s+parking\s+line,\s*'
               r'(?:heading|course)\s*(?P<heading_deg>NUM)',
               'expected translation', I),

    # --- emergency stops and aborts --------------------------------------------
    # "NOTSTOP: Lokalisierung seit %.1f s 'lost' -- ...", "NOTSTOP: Einlenkpunkt ..."
    # (the task description spells it "Notstopp"; both spellings are accepted)
    LogPattern('estop', 'de', 30, r'\bNOTSTOPP?\b[:\s-]*(?P<reason>.*)',
               'round1_controller_node.py control_loop()/_drive()', I),
    # "NOTHALT" = the bridge's ~/emergency topic was triggered (warn level)
    LogPattern('estop', 'de', 30, r'^\s*(?P<reason>NOTHALT)\b',
               'esp_serial_bridge.py _on_emergency()'),
    LogPattern('estop', 'en', 30, r'\bEMERGENCY[ _-]?(?:STOP|HALT)\b[:\s-]*(?P<reason>.*)',
               'expected translation', I),
    LogPattern('estop', 'en', 30, r'\bE-?STOP\b[:\s-]*(?P<reason>.*)',
               'expected translation', I),
    # "NOTFALL-RANGIEREN %d/%d: %s -- setzt %.0f cm zurueck (...), dann neu planen."
    # (since commit e740c8a: backs up and re-plans instead of an emergency stop;
    #  after rangier_max attempts per corner the NOTSTOP follows as before)
    LogPattern('manoeuvre', 'de', 30,
               r'NOTFALL-RANGIEREN\s*(?P<attempt>\d+)\s*/\s*(?P<max>\d+):\s*(?P<reason>.*?)\s*--\s*setzt\s*'
               r'(?P<back_cm>NUM)\s*cm\s+zurueck',
               'round1_controller_node.py _rangieren()'),
    LogPattern('manoeuvre', 'en', 30,
               r'EMERGENCY[ _-]?(?:MANOEUVRE|MANEUVER|MANOEUVRING|MANEUVERING|RECOVERY|REVERSING)\s*'
               r'(?P<attempt>\d+)\s*/\s*(?P<max>\d+):\s*(?P<reason>.*?)\s*--\s*'
               r'(?:backs?|backing|reverses|reversing)(?:\s+up)?\s*(?P<back_cm>NUM)\s*cm',
               'expected translation', I),
    # "Rangieren: schon %d Versuche an dieser Ecke -- gibt auf (%s)."
    LogPattern('manoeuvre_giveup', 'de', 30, r'Rangieren:\s*schon\s*(?P<attempts>\d+)\s*Versuche.*gibt\s+auf',
               'round1_controller_node.py _rangieren()'),
    LogPattern('manoeuvre_giveup', 'en', 30,
               r'(?:manoeuvr|maneuver)\w*:?\s*already\s*(?P<attempts>\d+)\s*attempts.*(?:gives?|giving)\s+up',
               'expected translation', I),
    # "%s abgebrochen: %s" with Einparken / Ausparken
    LogPattern('abort', 'de', 30, r'(?P<phase>Einparken|Ausparken)\s+abgebrochen:\s*(?P<reason>.*)',
               'round1_controller_node.py _ausparken_abbruch()'),
    LogPattern('abort', 'en', 30,
               r'(?P<phase>parking(?:\s+in|\s+out)?|unparking|park-in|park-out|pull-out|leaving\s+the\s+(?:bay|spot))'
               r'\s+(?:aborted|cancell?ed)[:\s-]*(?P<reason>.*)',
               'expected translation', I),

    # --- race progress ---------------------------------------------------------
    # f"ZIEL ({corner_count} Ecken, {front_dist:.2f} m vor Frontwand, v=...). STOP."
    LogPattern('finish', 'de', 20, r'\bZIEL\s*\((?P<corners>\d+)\s*Ecken',
               'round1_controller_node.py _drive()'),
    LogPattern('finish', 'en', 20, r'\b(?:FINISH|GOAL|TARGET)\w*\s*\((?P<corners>\d+)\s*corners',
               'expected translation', I),
    # "Drei Runden fertig (%d Ecken) -- ..."
    LogPattern('three_laps', 'de', 20, r'Drei\s+Runden\s+fertig\s*\((?P<corners>\d+)\s*Ecken',
               'round1_controller_node.py _park_uebergang()'),
    LogPattern('three_laps', 'en', 20,
               r'Three\s+laps\s+(?:done|complete|completed|finished)\s*\((?P<corners>\d+)\s*corners',
               'expected translation', I),
    # f"TURN fertig Ecke {corner_count} (theta=..., ziel=...)."
    LogPattern('corner_done', 'de', 20, r'TURN\s+fertig\s+Ecke\s*(?P<corner>\d+)',
               'round1_controller_node.py _turn()'),
    LogPattern('corner_done', 'en', 20,
               r'TURN\s+(?:done|finished|complete|completed)[,:]?\s*(?:at\s+)?corner\s*(?P<corner>\d+)',
               'expected translation', I),
    # "Start."
    LogPattern('start', 'de', 20, r'^\s*Start\.\s*$', 'round1_controller_node.py control_loop()'),

    # --- configuration echoed at start-up --------------------------------------
    # "Regler: Totzeit-Vorausberechnung %.3f s (Verstaerkung %.2f), ..."
    LogPattern('dead_time', 'de', 20,
               r'Totzeit-Vorausberechnung\s*(?P<dead_time_s>NUM)\s*s(?:\s*\(Verstaerkung\s*(?P<gain>NUM))?',
               'round1_controller_node.py __init__()'),
    LogPattern('dead_time', 'en', 20,
               r'dead[- ]time(?:\s+(?:prediction|compensation|look-?ahead|pre-?diction))?\s*(?P<dead_time_s>NUM)\s*s'
               r'(?:\s*\((?:gain)\s*(?P<gain>NUM))?',
               'expected translation', I),

    # --- sensor / localisation problems ----------------------------------------
    # f"GYRO AUSGEFALLEN: {grund}. ..."
    LogPattern('gyro_fail', 'de', 30, r'GYRO\s+AUSGEFALLEN[:\s]*(?P<reason>.*)', 'ekf_node.py _gyro_pruefen()'),
    LogPattern('gyro_fail', 'en', 30, r'GYRO\s+(?:FAILED|FAILURE|LOST|DOWN|DEAD|OUT)\b[:\s]*(?P<reason>.*)',
               'expected translation', I),
    # "Lokalisierung: %s -> %s%s."  (controller, on every state change)
    LogPattern('loc_change', 'de', 20, r'Lokalisierung:\s*(?P<old>[\w-]+)\s*->\s*(?P<new>\w+)',
               'round1_controller_node.py lok_state_cb()'),
    LogPattern('loc_change', 'en', 20, r'Locali[sz]ation(?:\s+state)?:\s*(?P<old>[\w-]+)\s*->\s*(?P<new>\w+)',
               'expected translation', I),
]
# fmt: on

_COMPILED = [(p, p.compiled()) for p in PATTERNS]


def parse_number(text):
    """'4,3' / '+0.8' / '4.0 cm' -> float, or NaN."""
    if text is None:
        return float('nan')
    m = re.search(NUM, str(text))
    if not m:
        return float('nan')
    return float(m.group(0).replace(',', '.'))


def match_line(msg, level=20):
    """All pattern matches for one log line.

    Returns a list of (key, lang, groups_dict); at most one match per key
    (the first row of the table that matches wins).
    """
    out, seen = [], set()
    for pat, rx in _COMPILED:
        if pat.key in seen or level < pat.min_level:
            continue
        m = rx.search(msg)
        if m:
            seen.add(pat.key)
            out.append((pat.key, pat.lang, {k: (v.strip() if isinstance(v, str) else v)
                                            for k, v in m.groupdict().items()}))
    return out


def scan_log(log_df):
    """Run match_line over a /rosout table (columns t_bag, level, name, msg).

    Returns a list of dicts {t_bag, key, lang, node, msg, **groups}, in time
    order.
    """
    events = []
    if log_df is None or len(log_df) == 0:
        return events
    for row in log_df.itertuples(index=False):
        text = str(getattr(row, 'msg', '') or '')
        level = int(getattr(row, 'level', 20) or 20)
        for key, lang, groups in match_line(text, level):
            ev = {'t_bag': float(row.t_bag), 'key': key, 'lang': lang,
                  'node': str(getattr(row, 'name', '')), 'level': level, 'msg': text}
            ev.update(groups)
            events.append(ev)
    return events


def estop_kind(reason):
    """Coarse category of an emergency-stop reason text."""
    r = (reason or '').lower()
    if re.search(r'lokalisierung|locali[sz]ation|\blost\b', r):
        return 'localisation_lost'
    if re.search(r'einlenkpunkt|turn-?in|t_a\b|bogen|arc', r):
        return 'turn_in_point'
    if re.search(r'vor der nase|in front of|ahead of the nose|nose', r):
        return 'obstacle_ahead'
    if 'nothalt' in r or 'halt' in r:
        return 'bridge_emergency'
    return 'other'


# --------------------------------------------------------------------------
# Source check: after the translation, run
#     python3 logpatterns.py --check-source
# It greps the node sources for a short anchor of every event (German anchor
# or English anchor regex) and reports which events can no longer be found,
# i.e. whose 'en' row above must be checked against the new text.
# --------------------------------------------------------------------------
ANCHORS = {
    'parked': ('EINGEPARKT. base_link', r'PARKED\b'),
    'parked_at': ('EINGEPARKT bei', r'PARKED at'),
    'bay_clearance': ('Endlage laut Buchtmessung', r'bay'),
    'park_reverse_done': ('Einparken rueckwaerts fertig', r'reverse'),
    'estop': ('NOTSTOP', r'EMERGENCY[ _-]?STOP|E-?STOP'),
    'manoeuvre': ('NOTFALL-RANGIEREN', r'EMERGENCY[ _-]?(MANOEUV|MANEUV|RECOVERY|REVERS)'),
    'manoeuvre_giveup': ('gibt auf', r'giv(es|ing) up'),
    'abort': ('abgebrochen: %s', r'(aborted|cancell?ed)'),
    'finish': ('ZIEL (', r'(FINISH|GOAL|TARGET)\w*\s*\('),
    'three_laps': ('Drei Runden fertig', r'Three laps'),
    'corner_done': ('TURN fertig Ecke', r'TURN (done|finished|complete)'),
    'start': ('"Start."', r'"Start\."'),
    'dead_time': ('Totzeit-Vorausberechnung', r'dead[- ]time'),
    'gyro_fail': ('GYRO AUSGEFALLEN', r'GYRO (FAILED|FAILURE|LOST|DOWN|DEAD|OUT)'),
    'loc_change': ('Lokalisierung: %s -> %s', r'Locali[sz]ation(\s+state)?: %s -> %s'),
}


def check_source(src_dir):
    from pathlib import Path
    files = [p for d in ('ekf', 'esp_bridge') for p in (Path(src_dir) / d).rglob('*.py')]
    text = '\n'.join(p.read_text(errors='replace') for p in files)
    missing = 0
    for key, (de, en) in ANCHORS.items():
        has_de, has_en = de in text, re.search(en, text, re.IGNORECASE) is not None
        state = 'German' if has_de else ('English?' if has_en else 'NOT FOUND')
        if not has_de:
            missing += 1
        print(f'  {key:18s} {state}')
    print(f'{missing} event(s) without the German anchor -- check their "en" rows in PATTERNS'
          if missing else 'all German anchors still present in the source')
    return missing


if __name__ == '__main__':
    import sys
    from pathlib import Path
    if '--check-source' in sys.argv:
        check_source(Path(__file__).resolve().parents[2] / 'src')
    else:
        for p in PATTERNS:
            print(f'{p.key:18s} {p.lang}  level>={p.min_level}  {p.source}')
