"""
Figure for the hyperparameter tuning results (report section 4.5, Figure 5).

Reads hyperparameter_results.csv and writes hyperparameter_results.png.

Why a dot plot rather than bars: the three configurations differ by about 0.05 in
validation loss and 0.01 in BLEU-4. On a bar chart with a zero baseline those
differences are invisible, and truncating a bar axis misrepresents the magnitudes,
because a bar encodes value through its length from zero. A dot plot encodes through
position alone, so a zoomed axis is honest. Every value is also printed beside its
marker, so the figure cannot be misread off the axis.

Validation loss and BLEU-4 are on different scales and get a panel each. They are never
placed on one pair of axes.

Run from the project root:
    python make_tuning_figure.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Slots 1 and 2 of the reference categorical palette, used unmodified.
SERIES_1 = "#2a78d6"   # blue, validation loss
SERIES_2 = "#eb6834"   # orange, BLEU-4
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#d8d7d2"
SURFACE = "#fcfcfb"


def _panel(ax, labels, values, colour, title, fmt, pad_frac=0.28):
    """One Cleveland dot plot: a faint guide per row, a marker at the value, a label."""
    y = range(len(labels))

    lo, hi = min(values), max(values)
    span = (hi - lo) or abs(hi) or 1.0
    left, right = lo - span * pad_frac * 2.2, hi + span * pad_frac * 2.2

    for i, value in zip(y, values):
        ax.plot([left, right], [i, i], color=GRID, linewidth=1, zorder=1)
        ax.plot([value], [i], marker="o", markersize=11, color=colour,
                markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
        ax.annotate(fmt.format(value), (value, i), textcoords="offset points",
                    xytext=(0, 13), ha="center", fontsize=10, color=TEXT_PRIMARY)

    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=10, color=TEXT_PRIMARY)
    ax.set_ylim(-0.6, len(labels) - 0.4)
    ax.set_xlim(left, right)
    ax.invert_yaxis()

    ax.set_title(title, fontsize=11, color=TEXT_PRIMARY, pad=14, loc="left")
    ax.tick_params(axis="x", labelsize=9, colors=TEXT_SECONDARY, length=0)
    ax.tick_params(axis="y", length=0)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.set_facecolor(SURFACE)


def write_figure(df, out_path):
    labels = [f"{r['LSTM units']} units, dropout {r['Dropout']}"
              for _, r in df.iterrows()]

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    fig.patch.set_facecolor(SURFACE)

    _panel(axes[0], labels, list(df["Best val loss"]), SERIES_1,
           "Best validation loss (lower is better)", "{:.4f}")
    _panel(axes[1], labels, list(df["BLEU-4"]), SERIES_2,
           "BLEU-4, greedy decoding on 250 test images (higher is better)", "{:.4f}")

    axes[1].set_yticklabels([])

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def main():
    df = pd.read_csv(os.path.join(BASE_DIR, "hyperparameter_results.csv"))
    out_path = os.path.join(BASE_DIR, "hyperparameter_results.png")
    write_figure(df, out_path)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
