"""For each numeric feature, plot mean by year with a LOESS smoother."""
import os

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

NON_PLOT = {"song_id", "year", "decade", "genre", "embedding"}


def plot_trend(df: pd.DataFrame, feature: str, out_dir: str) -> None:
    yearly = df.groupby("year")[feature].mean().reset_index()
    plt.figure(figsize=(8, 4))
    sns.regplot(data=yearly, x="year", y=feature, lowess=True,
                scatter_kws={"alpha": 0.6}, line_kws={"color": "red"})
    plt.title(f"{feature} over time")
    plt.xlabel("Year")
    plt.ylabel(feature)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{feature}.png"), dpi=120)
    plt.close()


def main(feat_path: str = "data/song_features.parquet",
         out_dir: str = "output/trends") -> None:
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_parquet(feat_path)
    numeric = [c for c in df.columns if c not in NON_PLOT and df[c].dtype != object]
    for col in numeric:
        print(f"  plotting {col}")
        plot_trend(df, col, out_dir)
    print(f"Saved {len(numeric)} trend plots -> {out_dir}")


if __name__ == "__main__":
    main()
