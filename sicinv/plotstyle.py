"""Shared figure style so every plot in the repo reads as one set."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
DATA = ROOT / "data"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
REF = "#52514e"          # datasheet / reference data (neutral, not a series hue)

# categorical slots, fixed order (validated colorblind-safe for adjacent use)
C1, C2, C3, C4, C5, C6, C7, C8 = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100",
    "#e87ba4", "#008300", "#4a3aa7", "#e34948",
)
SERIES = [C1, C2, C3, C4, C5, C6, C7, C8]

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK_2,
    "axes.titlecolor": INK,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.labelsize": 9.5,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.frameon": False,
    "legend.fontsize": 8.5,
    "lines.linewidth": 2.0,
    "lines.markersize": 6,
    "font.family": "DejaVu Sans",
    "axes.prop_cycle": matplotlib.cycler(color=SERIES),
})


def ref_points(ax, x, y, label="datasheet", **kw):
    """Datasheet points: open neutral circles so they never compete with model lines."""
    kw.setdefault("color", REF)
    ax.plot(x, y, "o", mfc="none", mew=1.4, ms=7, label=label, **kw)


def save(fig, name: str):
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return FIG / name
