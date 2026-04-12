"""Preprocess lyrics: clean text, remove stopwords, deduplicate chorus lines."""

import re
import pandas as pd
from nltk.corpus import stopwords

# NLTK stopwords + song filler words and contraction fragments
STOP_WORDS = set(stopwords.words("english")) | {
    # Interjections / vocal fillers common in lyrics
    "oh", "ooh", "yeah", "ya", "yea", "hey", "ayy", "uh", "uhh",
    "ah", "ahh", "la", "na", "da", "du", "mm", "mmm", "hmm",
    "whoa", "wo", "woo", "aye", "yo", "huh", "shh",
    "uhhuh", "uhoh", "nana", "shaha", "bop", "wop",
    "lalalalalala", "lalalala", "lalala", "dodododo",
    # Contraction leftovers after apostrophe removal
    "im", "ive", "youre", "youve", "youll", "youd",
    "dont", "cant", "wont", "didnt", "doesnt", "isnt", "wasnt",
    "werent", "havent", "hasnt", "hadnt", "wouldnt", "couldnt",
    "shouldnt", "aint", "gonna", "wanna", "gotta", "til",
    "thats", "hes", "shes", "its", "lets", "theyre", "whos",
    "ill", "wed", "hed", "shed", "theyd", "theyve", "theyll",
    # Genius artifacts
    "contributors", "translations", "lyrics", "read", "more",
}


def strip_genius_header(text: str) -> str:
    """Remove Genius metadata header (Contributors, Translations, description).

    The header ends with a pattern like 'Read More' or 'Lyrics' followed by
    the actual lyrics content.
    """
    # Match up to "Lyrics" followed by optional description ending with "Read More"
    text = re.sub(
        r"^\d+\s*Contributor.*?Lyrics(?:.*?Read More\s*)?",
        "",
        text,
        count=1,
        flags=re.DOTALL,
    )
    return text.strip()


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
    """Lowercase, remove punctuation, remove stopwords and short tokens."""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    # Remove repeated-syllable nonsense (e.g. "lalalalala", "nananana")
    text = re.sub(r"\b([a-z]{2,3})\1{2,}\b", "", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]
    return " ".join(tokens)


def main():
    """Read raw lyrics, preprocess, and save."""
    df = pd.read_csv("data/lyrics.csv")
    df = df.dropna(subset=["lyrics"])
    df = df[df["lyrics"].str.strip() != ""].reset_index(drop=True)
    df["lyrics_clean"] = (
        df["lyrics"]
        .apply(strip_genius_header)
        .apply(deduplicate_lines)
        .apply(clean_text)
    )
    df = df[df["lyrics_clean"].str.strip() != ""].reset_index(drop=True)
    df.to_csv("data/lyrics_processed.csv", index=False)
    print(f"Preprocessed {len(df)} songs → data/lyrics_processed.csv")
    # Show a sample
    print(f"\nSample (first 200 chars):\n{df['lyrics_clean'].iloc[0][:200]}")


if __name__ == "__main__":
    main()
