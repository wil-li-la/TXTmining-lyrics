"""For each numeric feature, plot mean by year with a LOESS smoother.

Styled to match the app's brutalist "Swiss Industrial Print" look (taste-skill:
industrial-brutalist-ui): documentation-paper ground, carbon-ink line + box frame,
acid-yellow data markers, monospace type. Uses DejaVu Sans Mono, which ships with
matplotlib, so no font install is needed in the Docker image.
"""
import os

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

NON_PLOT = {"song_id", "year", "decade", "genre", "embedding"}

PAPER = "#F4F4F0"
INK = "#111111"
ACCENT = "#E9FF3A"


def _apply_brutalist_style() -> None:
    """Set matplotlib rcParams for the Swiss-Industrial-Print plot look."""
    plt.rcParams.update({
        "figure.facecolor": PAPER,
        "axes.facecolor": PAPER,
        "savefig.facecolor": PAPER,
        "font.family": "monospace",
        "font.monospace": ["DejaVu Sans Mono"],
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.linewidth": 1.5,
        "axes.grid": False,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "figure.dpi": 120,
    })


def plot_trend(df: pd.DataFrame, feature: str, out_dir: str) -> None:
    yearly = df.groupby("year")[feature].mean().reset_index()
    label = feature.replace("_", " ").upper()
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.regplot(
        data=yearly, x="year", y=feature, lowess=True, ax=ax,
        scatter_kws={"s": 34, "facecolor": ACCENT, "edgecolor": INK,
                     "linewidths": 1.1, "alpha": 1.0, "zorder": 3},
        line_kws={"color": INK, "linewidth": 2.4, "zorder": 2},
    )
    # All four spines visible — a hard blueprint frame.
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
        spine.set_color(INK)
    ax.set_title(f"{label}  //  BY YEAR", loc="left", fontweight="bold")
    ax.set_xlabel("YEAR")
    ax.set_ylabel(label)
    ax.margins(x=0.02)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{feature}.png"))
    plt.close(fig)


def main(feat_path: str = "data/song_features.parquet",
         out_dir: str = "output/trends") -> None:
    _apply_brutalist_style()
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_parquet(feat_path)
    numeric = [c for c in df.columns if c not in NON_PLOT and df[c].dtype != object]
    for col in numeric:
        print(f"  plotting {col}")
        plot_trend(df, col, out_dir)
    print(f"Saved {len(numeric)} trend plots -> {out_dir}")


if __name__ == "__main__":
    main()
