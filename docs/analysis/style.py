"""Shared figure style for all plot scripts (printed engineering journal).

Design rules (from the team's data-viz guideline):
  * light, white surface (printed PDF); recessive hairline grid and axes,
    no top/right spines, no dual y-axes;
  * categorical colours in a FIXED order (validated colour-blind safe on the
    white surface; slots 1-3 are safe for scatter plots with all pairs), never
    cycled beyond 8 -- fold the rest into small multiples;
  * magnitude = one hue light->dark (SEQ_CMAP), ordered groups = ORDINAL ramp;
  * status colours (good / warning / critical) only for states, always with a
    text label; red/green pillars always get a second encoding (marker shape);
  * text never takes a series colour; 1.5 pt lines, >= 8 px markers;
  * short title, axis labels with units, source bag in a small footer.

Figures are saved as SVG (for the PDF) and PNG (GitHub preview).
"""
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt                     # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, ListedColormap  # noqa: E402

# --- surfaces and ink --------------------------------------------------------
SURFACE = '#ffffff'
INK = '#0b0b0b'           # primary text
INK_2 = '#52514e'         # secondary text, axis labels
MUTED = '#898781'         # ticks, footer, reference lines
GRID = '#e1e0d9'          # hairline grid
AXIS = '#c3c2b7'          # baseline / spines

# --- categorical (fixed order) -------------------------------------------------
CAT = ['#2a78d6',   # 1 blue
       '#eb6834',   # 2 orange
       '#1baf7a',   # 3 aqua
       '#eda100',   # 4 yellow
       '#e87ba4',   # 5 magenta
       '#008300',   # 6 green
       '#4a3aa7',   # 7 violet
       '#e34948']   # 8 red

# --- sequential blue ramp (100 -> 700) -------------------------------------------
BLUE_RAMP = ['#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7',
             '#3987e5', '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#104281', '#0d366b']
SEQ_CMAP = LinearSegmentedColormap.from_list('seq_blue', BLUE_RAMP[2:])
# ordinal ramp for ordered groups (e.g. run ranges): start at step 250
ORDINAL = BLUE_RAMP[3:]

# --- status (reserved for states, always labelled) -------------------------------
STATUS = {'good': '#0ca30c', 'warning': '#fab219', 'serious': '#ec835a',
          'critical': '#d03b3b'}
LOC_STATE_COLORS = {'ok': STATUS['good'], 'recovering': STATUS['warning'],
                    'lost': STATUS['critical']}

# --- semantic: pillar colours (with marker shape as second channel) -------------
PILLAR = {'red': {'color': '#e34948', 'marker': 's', 'label': 'red pillar'},
          'green': {'color': '#008300', 'marker': '^', 'label': 'green pillar'},
          'unknown': {'color': MUTED, 'marker': 'o', 'label': 'colour unknown'}}

LINE_W = 1.5              # pt  (~2 px)
MARKER_S = 6              # pt  (~8 px)


def apply_style():
    plt.rcParams.update({
        'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
        'savefig.facecolor': SURFACE,
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Liberation Sans', 'Arial', 'Helvetica'],
        'font.size': 9, 'axes.titlesize': 10, 'axes.titleweight': 'bold',
        'axes.titlelocation': 'left', 'axes.titlecolor': INK, 'axes.titlepad': 14,
        'axes.labelsize': 9, 'axes.labelcolor': INK_2, 'text.color': INK,
        'axes.edgecolor': AXIS, 'axes.linewidth': 0.8,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
        'grid.linestyle': '-', 'axes.axisbelow': True,
        'xtick.color': AXIS, 'ytick.color': AXIS,
        'xtick.labelcolor': INK_2, 'ytick.labelcolor': INK_2,
        'xtick.labelsize': 8, 'ytick.labelsize': 8,
        'axes.prop_cycle': matplotlib.cycler(color=CAT),
        'lines.linewidth': LINE_W, 'lines.solid_capstyle': 'round',
        'lines.solid_joinstyle': 'round', 'lines.markersize': MARKER_S,
        'legend.frameon': False, 'legend.fontsize': 8, 'legend.labelcolor': INK_2,
        'figure.dpi': 100, 'savefig.dpi': 200,
        'svg.fonttype': 'path',     # glyphs as paths: the PDF looks the same everywhere
        'axes.formatter.use_mathtext': True,
    })


apply_style()


def figure(nrows=1, ncols=1, width=7.0, height=None, **kw):
    """Figure with the house size (7 in = one column of an A4 journal page)."""
    height = height or 2.4 * nrows + 0.4
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, height), **kw)
    return fig, axes


def footer(fig, text):
    """Small caption in the lower left corner (source bag, tool)."""
    fig.text(0.01, 0.003, text, fontsize=7, color=MUTED, ha='left', va='bottom')


def source_caption(names, tool):
    names = list(names)
    if len(names) > 4:
        src = f'{len(names)} bags ({names[0]} ... {names[-1]})'
    else:
        src = ', '.join(names)
    return f'Source: {src}  |  {tool}'


def save(fig, out_dir, stem, caption=None):
    """Write <stem>.svg and <stem>.png into out_dir; returns the paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if caption:
        footer(fig, caption)
    fig.tight_layout(rect=(0, 0.035 if caption else 0, 1, 1))
    paths = []
    for ext in ('svg', 'png'):
        p = out_dir / f'{stem}.{ext}'
        fig.savefig(p, bbox_inches='tight', pad_inches=0.08)
        paths.append(p)
    plt.close(fig)
    print(f'  wrote {paths[0]} (+ .png)')
    return paths


def ref_line(ax, x=None, y=None, label=None, color=MUTED, text_pos='top'):
    """Thin solid reference line with a small text label in muted ink."""
    if x is not None:
        ax.axvline(x, color=color, lw=0.9, zorder=3)
        if label:
            ymax = ax.get_ylim()[1]
            ax.text(x, ymax, f' {label}', color=INK_2, fontsize=7, va='top', ha='left',
                    rotation=0, zorder=4)
    if y is not None:
        ax.axhline(y, color=color, lw=0.9, zorder=3)
        if label:
            xmin = ax.get_xlim()[0]
            ax.text(xmin, y, f' {label}', color=INK_2, fontsize=7, va='bottom', ha='left',
                    zorder=4)


def stat_lines(ax, values, unit='', fmt='{:.2f}'):
    """Median / p95 / max as labelled vertical lines on a histogram."""
    import numpy as np
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {}
    stats = {'median': float(np.median(v)), 'p95': float(np.percentile(v, 95)),
             'max': float(np.max(v))}
    x_lo, x_hi = ax.get_xlim()
    shades = {'median': INK, 'p95': INK_2, 'max': MUTED}
    lines = []
    for k, x in stats.items():
        off = x > x_hi
        if not off:
            ax.axvline(x, color=shades[k], lw=0.9, zorder=3)
            ax.text(x, 1.0, k, transform=ax.get_xaxis_transform(), color=INK_2, fontsize=6.5,
                    ha='center', va='bottom', zorder=4)
        lines.append(f'{k} {fmt.format(x)}{unit}' + (' (off scale)' if off else ''))
    ax.text(0.98, 0.95, '\n'.join(lines), transform=ax.transAxes, ha='right', va='top',
            fontsize=7.5, color=INK_2, linespacing=1.5,
            bbox=dict(boxstyle='round,pad=0.35', fc=SURFACE, ec=GRID, lw=0.6))
    return stats


def hist(ax, values, bins=40, color=CAT[0], **kw):
    """Histogram with thin gaps between bars (the 'surface gap'). The bin
    count shrinks for small samples."""
    import numpy as np
    n = int(np.isfinite(np.asarray(values, float)).sum())
    if isinstance(bins, int):
        bins = max(6, min(bins, n // 4 if n else 6))
    return ax.hist(values, bins=bins, color=color, edgecolor=SURFACE, linewidth=0.6, **kw)


def legend_below(ax, ncol=3, **kw):
    """Legend under the x-axis label, so it never covers data."""
    return ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=ncol,
                     borderaxespad=0.0, **kw)


def legend_above(ax, ncol=3, **kw):
    """Legend in the title row, right-aligned (titles are left-aligned)."""
    return ax.legend(loc='lower right', bbox_to_anchor=(1.0, 1.0), ncol=ncol,
                     borderaxespad=0.2, **kw)


def seq_colors(n):
    """n colours from the ordinal blue ramp (light -> dark)."""
    if n <= 1:
        return [ORDINAL[len(ORDINAL) // 2]]
    idx = [round(i * (len(ORDINAL) - 1) / (n - 1)) for i in range(n)]
    return [ORDINAL[i] for i in idx]


def no_data(ax, text='no data in this bag'):
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha='center', va='center',
            color=MUTED, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)


__all__ = ['apply_style', 'figure', 'save', 'footer', 'source_caption', 'ref_line',
           'stat_lines', 'hist', 'legend_below', 'legend_above', 'seq_colors', 'no_data', 'CAT', 'SEQ_CMAP', 'ORDINAL',
           'STATUS', 'LOC_STATE_COLORS', 'PILLAR', 'INK', 'INK_2', 'MUTED', 'GRID', 'AXIS',
           'SURFACE', 'ListedColormap', 'plt']
