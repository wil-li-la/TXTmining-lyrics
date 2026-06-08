"""Generate word clouds per decade from TF-IDF weights.

The naive approach — averaging per-song TF-IDF across a decade — is misleading
on a small corpus (~24 songs/decade): a word that spikes in a single song
(e.g. "macarena", "scrub", "creep") dominates the whole decade even though it
is not representative. We make the per-decade scoring robust by:

  1. Unigrams only — bigrams fragment a word cloud and read as noise.
  2. A presence filter — a word must appear in at least MIN_SONGS distinct
     songs of the decade, which removes single-song artifacts outright.
  3. A gentle distinctiveness boost — score = mean_TFIDF * (mean/overall)^BETA,
     so each decade surfaces its own flavour instead of the same universal
     pop vocabulary (love / know / baby) appearing in all four clouds.
"""

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from wordcloud import WordCloud
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# A word must appear in at least this fraction of a decade's songs (with a hard
# floor) to be eligible — the main defence against single-song spikes. Scales
# with corpus size: ~18 songs for a 900-song decade, the floor for small ones.
MIN_SONG_FRACTION = 0.02
MIN_SONGS_FLOOR = 5
# Distinctiveness exponent: 0 = pure frequency (clouds look alike),
# 1 = pure distinctiveness (reintroduces rare spikes). 0.5 is the sweet spot.
BETA = 0.5
MAX_WORDS = 60

# Profanity + slurs are filtered from the *rendered* clouds only (the TF-IDF
# matrix and decade classifier keep the full vocabulary). Without this, the
# 2020s cloud is dominated by explicit terms that crowd out genuine themes.
PROFANITY = frozenset({
    # f-word family
    "fuck", "fucks", "fucked", "fucking", "fuckin", "fucker", "fuckers",
    "fuckboy", "motherfucker", "motherfuckers", "motherfucking", "motherfuckin",
    # shit family
    "shit", "shits", "shitty", "bullshit", "shittin",
    # bitch family
    "bitch", "bitches", "bitchin", "bitchy", "bitchass",
    # the n-word and variants (slurs)
    "nigga", "niggas", "niggaz", "nigger", "niggers", "niggah", "niggahs",
    # sexual / anatomical
    "pussy", "pussies", "dick", "dicks", "cock", "cum", "cumming", "ass",
    "asses", "asshole", "assholes", "hoe", "hoes", "ho", "hos", "slut",
    "sluts", "whore", "whores", "tits", "titties", "titty", "thot", "thots",
    # general profanity / intensifiers
    "damn", "goddamn", "hell", "piss", "pissed", "crap", "fck", "shyt",
})


def decade_word_scores(matrix, vocab, decade_mask, overall_mean):
    """Robust per-decade word scores: frequency grounded, mildly distinctive.

    Returns {word: score} for unigrams present in enough distinct songs of the
    decade, scored by mean decade TF-IDF weighted by how much the decade
    over-uses the word relative to the whole corpus.
    """
    subset = matrix[decade_mask]
    n_songs = subset.shape[0]
    min_songs = max(MIN_SONGS_FLOOR, int(MIN_SONG_FRACTION * n_songs))
    mean_decade = np.asarray(subset.mean(axis=0)).flatten()
    songs_with_word = np.asarray((subset > 0).sum(axis=0)).flatten()

    scores = {}
    for i, word in enumerate(vocab):
        if " " in word:                      # unigrams only
            continue
        if word in PROFANITY:                # display-layer profanity filter
            continue
        if songs_with_word[i] < min_songs:   # drop single-song spikes
            continue
        distinctiveness = (mean_decade[i] + 1e-6) / (overall_mean[i] + 1e-6)
        scores[word] = float(mean_decade[i] * (distinctiveness ** BETA))
    return scores


def generate_decade_wordcloud(scores: dict, decade: str, output_path: str):
    """Render and save a word cloud from a {word: score} mapping."""
    wc = WordCloud(
        width=800,
        height=400,
        background_color="white",
        max_words=MAX_WORDS,
        colormap="viridis",
        max_font_size=80,
        min_font_size=12,
        relative_scaling=0.4,
        prefer_horizontal=0.95,
    )
    wc.generate_from_frequencies(scores)

    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation="bilinear")
    plt.title(f"Top Words — {decade}", fontsize=16)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved {output_path}  ({len(scores)} eligible words)")


def main():
    """Generate robust word clouds for all four decades."""
    df = pd.read_csv("data/lyrics_processed.csv").dropna(subset=["lyrics_clean"])
    df = df.reset_index(drop=True)
    matrix = sp.csr_matrix(joblib.load("data/tfidf_matrix.pkl"))
    vocab = joblib.load("data/tfidf_vocab.pkl")
    overall_mean = np.asarray(matrix.mean(axis=0)).flatten()

    decades = sorted(df["decade"].dropna().unique())
    for decade in decades:
        mask = (df["decade"] == decade).values
        scores = decade_word_scores(matrix, vocab, mask, overall_mean)
        top = sorted(scores, key=scores.get, reverse=True)[:12]
        print(f"{decade}: {', '.join(top)}")
        generate_decade_wordcloud(scores, decade, f"output/wordclouds/{decade}.png")

    print("\nDone. Word clouds saved to output/wordclouds/")


if __name__ == "__main__":
    main()
