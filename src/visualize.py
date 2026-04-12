"""Generate word clouds per decade from TF-IDF weights."""

import joblib
import numpy as np
import pandas as pd
from wordcloud import WordCloud
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def generate_decade_wordcloud(
    tfidf_matrix,
    vocab: list[str],
    decade_labels: pd.Series,
    decade: str,
    output_path: str,
):
    """Generate and save a word cloud for a specific decade.

    Aggregates TF-IDF scores across all songs in the decade
    to find the most representative words.
    """
    mask = (decade_labels == decade).values
    subset = tfidf_matrix[mask]
    mean_scores = np.asarray(subset.mean(axis=0)).flatten()
    word_scores = {vocab[i]: mean_scores[i] for i in range(len(vocab)) if mean_scores[i] > 0}

    wc = WordCloud(
        width=800,
        height=400,
        background_color="white",
        max_words=100,
        colormap="viridis",
        max_font_size=80,
        min_font_size=12,
        relative_scaling=0.3,
    )
    wc.generate_from_frequencies(word_scores)

    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation="bilinear")
    plt.title(f"Top Words — {decade}", fontsize=16)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved {output_path}")


def main():
    """Generate word clouds for all four decades."""
    df = pd.read_csv("data/lyrics_processed.csv")
    matrix = joblib.load("data/tfidf_matrix.pkl")
    vocab = joblib.load("data/tfidf_vocab.pkl")

    decades = ["1990s", "2000s", "2010s", "2020s"]
    for decade in decades:
        output_path = f"output/wordclouds/{decade}.png"
        generate_decade_wordcloud(matrix, vocab, df["decade"], decade, output_path)

    print(f"\nDone. Word clouds saved to output/wordclouds/")


if __name__ == "__main__":
    main()
