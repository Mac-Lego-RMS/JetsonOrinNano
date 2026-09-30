"""Vehicle-level wiring diagram: which connector of the main PCB goes where.

The schematic shows the board; this shows the vehicle around it. Connector
references and pin functions follow schemes/MainPCB/MainPCB.kicad_sch.

    python docs/analysis/draw_wiring.py
"""
import argparse
import sys
from pathlib import Path

from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

import style

FIGURES = Path(__file__).resolve().parents[1] / 'figures'
POWER, SIGNAL, USB = style.CAT[7], style.CAT[0], style.CAT[6]

BOARD = (37, 17, 63, 65)          # x0, y0, x1, y1
JETSON = (27, 1.5, 73, 11.5)

LEFT = [   # name, pins, connector, kind, y
    ('4S LiPo, 450 / 1150 mAh', 'XT30, 12.0-16.8 V', 'J12', POWER, 58.5),
    ('Second source', 'bench supply or fresh pack, XT30', 'J9', POWER, 49),
    ('Main switch, 10 A', 'switches 15Vin to 15Vsw', 'J1', POWER, 39.5),
    ('Laptop (maintenance)', 'USB-C: flash, calibrate, bench power', 'J8', USB, 30),
]
RIGHT = [
    ('RPLIDAR S3', '5 V · TX · RX · GND', 'J14', SIGNAL, 61),
    ('BNO055 IMU', '3.3 V · SDA · SCL · GND', 'J15', SIGNAL, 53.5),
    ('SC09 steering servo', '5 V · Data (half duplex) · GND', 'J3', SIGNAL, 46),
    ('Drive motor + Hall encoder', 'M+ · M- · 3.3 V · GND · A · B', 'J5', POWER, 38.5),
    ('Start button', 'Detect · GND', 'J7', SIGNAL, 31),
    ('Status LED, addressable', '5 V · DIN · GND', 'J11', SIGNAL, 23.5),
]
DOWN = [   # connector, kind, x, label
    ('J16', POWER, 43, 'XT30, battery voltage'),
    ('J10', SIGNAL, 50, '40-pin header'),
    ('J13', USB, 57, 'FFC to USB'),
]


def box(ax, x0, y0, x1, y1, fill='#f6f6f3', edge=style.AXIS, lw=0.8):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                boxstyle='round,pad=0,rounding_size=0.8',
                                fc=fill, ec=edge, lw=lw))


def component(ax, x0, x1, y, name, pins):
    box(ax, x0, y - 3.2, x1, y + 3.2, fill=style.SURFACE)
    xm = (x0 + x1) / 2
    ax.text(xm, y + 0.9, name, ha='center', va='center', fontsize=7.2,
            fontweight='bold', color=style.INK)
    ax.text(xm, y - 1.4, pins, ha='center', va='center', fontsize=6, color=style.INK_2)


def wire(ax, xs, ys, col):
    ax.plot(xs, ys, color=col, lw=2.2, solid_capstyle='butt', zorder=1)


def draw(out_dir):
    style.apply_style()
    fig, ax = style.figure(1, 1, width=7.0, height=4.9)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 70)
    ax.axis('off')

    bx0, by0, bx1, by1 = BOARD
    box(ax, bx0, by0, bx1, by1, fill='#eef3fb', edge=style.CAT[0], lw=1.2)
    xm = (bx0 + bx1) / 2
    ax.text(xm, by1 - 3, 'Main PCB, V5', ha='center', fontsize=8.5, fontweight='bold')
    for k, line in enumerate(['stacked on the Jetson',
                              '',
                              'LTC4412 x2  ideal-diode inputs',
                              'MAX17504  15 V to 5 V',
                              'TPS259230  eFuse',
                              'AMS1117  5 V to 3.3 V',
                              '',
                              'ESP32-S3  motor, servo,',
                              'encoder, button, LED',
                              'CP2102N  LiDAR UART to USB',
                              'IMU I2C passed to J10']):
        ax.text(xm, by1 - 7 - 2.6 * k, line, ha='center', fontsize=6.2, color=style.INK_2)

    for name, pins, ref, col, y in LEFT:
        component(ax, 1, 25, y, name, pins)
        wire(ax, [25, bx0], [y, y], col)
        ax.text(bx0 - 0.8, y + 0.9, ref, ha='right', fontsize=6.5, fontweight='bold', color=col)
    for name, pins, ref, col, y in RIGHT:
        component(ax, 75, 99, y, name, pins)
        wire(ax, [bx1, 75], [y, y], col)
        ax.text(bx1 + 0.8, y + 0.9, ref, fontsize=6.5, fontweight='bold', color=col)

    jx0, jy0, jx1, jy1 = JETSON
    box(ax, jx0, jy0, jx1, jy1, fill=style.SURFACE)
    ax.text((jx0 + jx1) / 2, jy0 + 7.1, 'Jetson Orin Nano on A603 carrier', ha='center',
            fontsize=7.2, fontweight='bold')
    ax.text((jx0 + jx1) / 2, jy0 + 4.0,
            'J16: power, XT30 at battery voltage  ·  J13: USB 3.0 port (LiDAR)',
            ha='center', fontsize=6, color=style.INK_2)
    ax.text((jx0 + jx1) / 2, jy0 + 1.5,
            'J10: 40-pin header, UART 115200 (pins 8/10), I2C (pins 3/5)',
            ha='center', fontsize=6, color=style.INK_2)
    for ref, col, x, lbl in DOWN:
        wire(ax, [x, x], [by0, jy1], col)
        ax.text(x + 0.8, (by0 + jy1) / 2 - 0.8, ref, fontsize=6.5, fontweight='bold', color=col)

    component(ax, 1, 23, 6.5, 'USB camera', 'direct to a Jetson USB port')
    wire(ax, [23, jx0], [6.5, 6.5], USB)

    handles = [Line2D([], [], color=c, lw=2.2, label=l) for c, l in
               ((POWER, 'power'), (SIGNAL, 'signal (with supply and GND)'), (USB, 'USB'))]
    ax.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, -0.07), ncol=3,
              frameon=False, fontsize=7)
    style.save(fig, out_dir, 'wiring',
               'Source: schemes/MainPCB/MainPCB.kicad_sch  |  draw_wiring.py')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out-dir', default=str(FIGURES))
    a = ap.parse_args(argv)
    draw(a.out_dir)
    return 0


if __name__ == '__main__':
    sys.exit(main())
