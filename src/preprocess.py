"""Preprocess lyrics: clean text, remove stopwords, deduplicate chorus lines."""

import re
import pandas as pd
from nltk.corpus import stopwords

STOP_WORDS = set(stopwords.words("english"))


def deduplicate_lines(text: str) -> str:
    """Remove duplicate lines (sentence-level dedup for chorus reduction)."""
    seen = set()
    result = []
    for line in text.split("\n"):
        line_stripped = line.strip()
        if line_stripped and line_stripped not in seen:
            seen.add(line_stripped)
            result.append(line_stripped)
    return "\n".join(result)


def clean_text(text: str) -> str:
    """Lowercase, remove punctuation, remove stopwords."""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOP_WORDS]
    return " ".join(tokens)


def main():
    """Read raw lyrics, preprocess, and save."""
    df = pd.read_csv("data/lyrics.csv")
    df["lyrics_clean"] = (
        df["lyrics"]
        .apply(deduplicate_lines)
        .apply(clean_text)
    )
    df.to_csv("data/lyrics_processed.csv", index=False)
    print(f"Preprocessed {len(df)} songs → data/lyrics_processed.csv")
    # Show a sample
    print(f"\nSample (first 200 chars):\n{df['lyrics_clean'].iloc[0][:200]}")


if __name__ == "__main__":
    main()
