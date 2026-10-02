"""Rating matrices for the mobility chapter.

Three comparisons in the journal are judgements rather than measurements:
the steering generations, the motor candidates and the mechanical
trade-offs. They are drawn as matrices instead of tables. Every cell keeps
its value; the colour only grades it from -- (red) over o (grey) to ++
(blue), and the symbol is printed as well, so the grade never depends on
colour alone. Each grade restates what the chapter text says about that
value; the sources are given next to the data below.

    python docs/analysis/draw_mobility_ratings.py
"""
import argparse
import sys
from pathlib import Path

from matplotlib.patches import FancyBboxPatch

import style

FIGURES = Path(__file__).resolve().parents[1] / 'figures'

# diverging scale: two hues, neutral grey midpoint; light tints so the ink
# on top stays readable
GRADE = {'--': '#f2b3a3', '-': '#f9dbd1', 'o': '#efeeea', '+': '#d4e5f9',
         '++': '#a9caf1', None: '#f7f7f5'}
SYMBOL = {'--': '−−', '-': '−', 'o': 'o', '+': '+', '++': '++', None: ''}

# 02-mobility.md, Development of the steering
STEERING = {
    'columns': ['Regional final\nLEGO rack', 'National final\ndirect link',
                'Napoleon\nAckermann linkage'],
    'highlight': 2,
    'rows': [
        ('Reversal play', [('2–4°', '--'), ('< 0.5° (measured)', '++'),
                           ('≈ 0.8° expected ‡', '+')]),
        ('Ackermann share', [('0 %', '--'), ('0 %', '--'), ('≈ 100 %', '++')]),
        ('Mechanical lock', [('not documented', None), ('± 35°', 'o'),
                             ('58° inner / 36.5° outer', '++')]),
        ('Robustness', [('gears skipped\nunder load', '--'),
                        ('broke or came loose\nat LEGO H-profiles', '-'),
                        ('steel tie-rod ends,\nno wear', '++')]),
    ],
}

# 02-mobility.md, Motor selection; data sheets [5]-[7]
MOTORS = {
    'columns': ['Pololu 20D 31:1\n(national final)', '25GA370\n(chosen)',
                'Pololu 25D\n4.4:1 HP', 'Pololu 25D\n9.7:1 HP', 'Pololu 37D\n10:1'],
    'highlight': 1,
    'rows': [
        ('Speed $v_0$\n(Ø 32 mm, 1:1)', [('1.31 m/s\n(Ø 67 mm, diff.)', 'o'),
                                       ('1.68 m/s', '++'),
                                       ('3.69 m/s; we use\n10–20 % of it', '--'),
                                       ('1.68 m/s', '++'), ('1.68 m/s', '++')]),
        ('Stall torque', [('2.4 kg·cm', 'o'), ('not specified;\ntraction-limited', None),
                          ('1.7 kg·cm', '-'), ('3.9 kg·cm', '+'), ('4.9 kg·cm', '++')]),
        ('Encoder', [('none', '--'), ('408 counts/rev,\n3.3 V', '++'),
                     ('211 counts/rev', '-'), ('465 counts/rev,\nnot 3.3 V', '--'),
                     ('640 counts/rev', '++')]),
        ('Mass', [('43 g', '++'), ('94 g', 'o'), ('95 g', 'o'), ('95 g', 'o'),
                  ('190 g', '--')]),
        ('Size, fit', [('Ø 20 × 43 mm', '+'), ('Ø 24.4 × 70 mm,\nbeside the Jetson', '+'),
                       ('Ø 25 × 63 mm', '+'), ('Ø 25 × 63 mm', '+'),
                       ('Ø 37 × 65 mm,\nraises the Jetson', '--')]),
    ],
}

# 02-mobility.md, Mechanical trade-offs
TRADEOFFS = {
    'columns': ['Gained', 'Given up'],
    'highlight': None,
    'rows': [
        ('Compact steering\nlinkage', [('short front,\nAckermann geometry', '+'),
                                       ('slightly less lock than a\nlarger linkage', '-')]),
        ('Small steering\nparts in SLA', [('precision,\nwear resistance', '+'),
                                          ('a second printer\nand process', '-')]),
        ('Rigid axle instead\nof a differential', [('lengthways motor,\nshorter chassis', '+'),
                                                   ('tire scrub, driven\nlock only ≈ 22–25°', '-')]),
        ('Low LiDAR\nposition', [('small camera–LiDAR\noffset', '+'),
                                 ('rear 120° blocked\n(240° used)', '-')]),
        ('Purchased steel\ntie-rod ends', [('no wear,\nreproducible geometry', '+'),
                                           ('20° articulation,\nservo rotated 12°', '-')]),
        ('Cast silicone\ntires, 32 mm', [('grip,\nlow vehicle', '+'),
                                         ('casting effort,\nregular cleaning', '-')]),
        ('25GA370 instead of\na larger motor', [('mass, height,\nno PCB change', '+'),
                                                ('nothing measurable:\ntraction-limited', 'o')]),
    ],
}


def cell(ax, x, y, w, h, text, grade):
    ax.add_patch(FancyBboxPatch((x + 0.04, y + 0.04), w - 0.08, h - 0.08,
                                boxstyle='round,pad=0,rounding_size=0.06',
                                fc=GRADE[grade], ec='none'))
    ax.text(x + 0.1, y + h / 2, SYMBOL[grade], ha='left', va='center',
            fontsize=7, color=style.INK_2, fontweight='bold')
    ax.text(x + 0.36, y + h / 2, text, ha='left', va='center', fontsize=6.6,
            color=style.INK, linespacing=1.15)


def matrix(spec, out_dir, stem, title, caption, label_w=1.35, col_w=1.45, row_h=0.62,
           legend=True):
    cols, rows = spec['columns'], spec['rows']
    width = label_w + col_w * len(cols)
    height = 0.62 + row_h * len(rows) + 0.45
    fig, ax = style.figure(1, 1, width=min(7.0, width), height=height * min(7.0, width) / width)
    ax.set_xlim(0, width)
    ax.set_ylim(-0.45, height - 0.45)
    ax.axis('off')
    top = row_h * len(rows)
    for j, name in enumerate(cols):
        x = label_w + j * col_w
        hi = spec['highlight'] == j
        ax.text(x + 0.1, top + 0.3, name, ha='left', va='center', fontsize=7.5,
                color=style.INK if hi else style.INK_2,
                fontweight='bold' if hi else 'normal', linespacing=1.15)
    for i, (label, values) in enumerate(rows):
        y = top - (i + 1) * row_h
        ax.text(0.02, y + row_h / 2, label, ha='left', va='center', fontsize=7.5,
                color=style.INK, linespacing=1.15)
        for j, (text, grade) in enumerate(values):
            cell(ax, label_w + j * col_w, y, col_w, row_h, text, grade)
    if spec['highlight'] is not None:
        x = label_w + spec['highlight'] * col_w
        ax.add_patch(FancyBboxPatch((x, 0), col_w, top + 0.62,
                                    boxstyle='round,pad=0,rounding_size=0.08',
                                    fc='none', ec=style.INK_2, lw=1.0))
    # legend: the five grades
    x = label_w
    for g in ('--', '-', 'o', '+', '++') if legend else ():
        ax.add_patch(FancyBboxPatch((x, -0.38), 0.32, 0.24,
                                    boxstyle='round,pad=0,rounding_size=0.05',
                                    fc=GRADE[g], ec='none'))
        ax.text(x + 0.16, -0.26, SYMBOL[g], ha='center', va='center', fontsize=7,
                color=style.INK_2, fontweight='bold')
        x += 0.4
    if legend:
        ax.text(x + 0.05, -0.26, 'worse  →  better', va='center', fontsize=7,
                color=style.MUTED)
    ax.set_title(title, loc='left')
    style.save(fig, out_dir, stem, caption)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out-dir', default=str(FIGURES))
    a = ap.parse_args(argv)
    style.apply_style()
    src = 'Grades restate the text of chapter 1  |  draw_mobility_ratings.py'
    matrix(STEERING, a.out_dir, 'steering_generations',
           'Three steering generations', src, label_w=1.3, col_w=1.9)
    matrix(MOTORS, a.out_dir, 'motor_selection',
           'Drive motor candidates', 'Data sheets [5]–[7]; ' + src,
           label_w=1.15, col_w=1.37, row_h=0.6)
    matrix(TRADEOFFS, a.out_dir, 'mobility_tradeoffs',
           'What each decision gained and gave up', src,
           label_w=1.55, col_w=2.2, row_h=0.56, legend=False)
    return 0


if __name__ == '__main__':
    sys.exit(main())
