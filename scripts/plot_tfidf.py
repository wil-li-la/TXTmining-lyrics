"""Compute the TF-IDF matrix from processed lyrics and plot it three ways.

A full ~6.8k docs x 5k terms sparse matrix is unreadable as a raw heatmap,
so we render it at three altitudes:

  1. output/tfidf/tfidf_decade_heatmap.png
     Mean TF-IDF per decade x top terms — the interpretable summary.
  2. output/tfidf/tfidf_sample_heatmap.png
     A random sample of songs x top terms — the real matrix structure/sparsity.
  3. output/tfidf/tfidf_sparsity.png
     A spy plot of a matrix block — how sparse TF-IDF actually is.

Run:  ./venv/bin/python scripts/plot_tfidf.py
"""

import os

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer

# Inlined to mirror src/features.py:build_tfidf — a `src/features/` package
# shadows that module, so we reproduce its config here rather than import it.
def build_tfidf(lyrics, max_features=5000):
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(lyrics)
    vocab = vectorizer.get_feature_names_out().tolist()
    return matrix, vocab, vectorizer

OUT_DIR = "output/tfidf"
# Decades present in the corpus, in chronological order (filled in main()).
DECADES = []
TOP_TERMS = 30        # term columns shown in the heatmaps
SAMPLE_SONGS = 60     # rows shown in the per-song heatmap
RNG = np.random.default_rng(42)


def top_unigram_indices(matrix, vocab, k):
    """Indices of the k unigram terms with the highest global mean TF-IDF."""
    mean_scores = np.asarray(matrix.mean(axis=0)).flatten()
    # unigrams only (no space) keep the term axis readable
    order = np.argsort(mean_scores)[::-1]
    picked = [i for i in order if " " not in vocab[i]][:k]
    return picked


def plot_decade_heatmap(matrix, vocab, decades, term_idx, path):
    """Mean TF-IDF per decade for the top terms."""
    terms = [vocab[i] for i in term_idx]
    rows = []
    for dec in DECADES:
        mask = (decades == dec).values
        rows.append(np.asarray(matrix[mask][:, term_idx].mean(axis=0)).flatten())
    data = pd.DataFrame(rows, index=DECADES, columns=terms)

    plt.figure(figsize=(14, 5))
    sns.heatmap(data, cmap="magma", linewidths=0.4, linecolor="white",
                cbar_kws={"label": "mean TF-IDF"})
    plt.title(f"TF-IDF by decade — top {len(terms)} terms", fontsize=15, pad=12)
    plt.xlabel("term")
    plt.ylabel("decade")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  saved {path}")


def plot_sample_heatmap(matrix, vocab, titles, term_idx, path):
    """A random sample of songs x top terms — the real matrix block."""
    n = matrix.shape[0]
    sample = np.sort(RNG.choice(n, size=min(SAMPLE_SONGS, n), replace=False))
    block = matrix[sample][:, term_idx].toarray()
    terms = [vocab[i] for i in term_idx]

    plt.figure(figsize=(14, 10))
    sns.heatmap(block, cmap="viridis", xticklabels=terms, yticklabels=False,
                cbar_kws={"label": "TF-IDF weight"})
    plt.title(f"TF-IDF matrix — {len(sample)} sampled songs x top {len(terms)} terms",
              fontsize=15, pad=12)
    plt.xlabel("term")
    plt.ylabel(f"{len(sample)} sampled songs")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  saved {path}")


def plot_sparsity(matrix, path):
    """Spy plot of a matrix block showing TF-IDF sparsity."""
    rows = min(300, matrix.shape[0])
    cols = min(300, matrix.shape[1])
    block = matrix[:rows, :cols].toarray()
    density = (matrix.nnz / (matrix.shape[0] * matrix.shape[1])) * 100

    plt.figure(figsize=(8, 8))
    plt.spy(block, markersize=0.6, aspect="auto", color="#222222")
    plt.title(f"TF-IDF sparsity — {rows}x{cols} block\n"
              f"full matrix {matrix.shape[0]}x{matrix.shape[1]}, "
              f"{density:.2f}% non-zero", fontsize=13, pad=12)
    plt.xlabel("term index")
    plt.ylabel("song index")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  saved {path}")


def main():
    global DECADES
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv("data/lyrics_processed.csv").dropna(subset=["lyrics_clean"])
    DECADES = sorted(df["decade"].dropna().unique())
    print(f"Loaded {len(df)} songs with cleaned lyrics across {len(DECADES)} decades.")

    matrix, vocab, vectorizer = build_tfidf(df["lyrics_clean"])
    print(f"TF-IDF matrix: {matrix.shape[0]} docs x {matrix.shape[1]} terms "
          f"({matrix.nnz:,} non-zeros, "
          f"{matrix.nnz / (matrix.shape[0] * matrix.shape[1]) * 100:.2f}% dense)")

    # persist the freshly computed artifacts alongside the existing ones
    joblib.dump(matrix, "data/tfidf_matrix.pkl")
    joblib.dump(vocab, "data/tfidf_vocab.pkl")
    joblib.dump(vectorizer, "data/tfidf_vectorizer.pkl")

    term_idx = top_unigram_indices(matrix, vocab, TOP_TERMS)
    plot_decade_heatmap(matrix, vocab, df["decade"].reset_index(drop=True), term_idx,
                        f"{OUT_DIR}/tfidf_decade_heatmap.png")
    plot_sample_heatmap(matrix, vocab, df["title"].tolist(), term_idx,
                        f"{OUT_DIR}/tfidf_sample_heatmap.png")
    plot_sparsity(matrix, f"{OUT_DIR}/tfidf_sparsity.png")

    print(f"\nDone. Plots in {OUT_DIR}/")


if __name__ == "__main__":
    main()
