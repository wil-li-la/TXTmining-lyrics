"""Preprocessing producing three lyric views for the feature pipeline.

- raw:       line breaks preserved, original casing; used by rhyme, repetition.
- tokenized: lowercased, punctuation stripped, stopwords KEPT; used by pronouns.
- clean:     tokenized minus stopwords + 3-char minimum; used by TF-IDF, EmoLex, concreteness.
"""
import re
from nltk.corpus import stopwords

_STOP = set(stopwords.words("english")) | {
    # song fillers
    "oh", "ooh", "yeah", "ya", "yea", "hey", "ayy", "uh", "uhh",
    "ah", "ahh", "la", "na", "da", "du", "mm", "mmm", "hmm",
    "whoa", "wo", "woo", "aye", "yo", "huh", "shh",
    # contraction fragments
    "im", "ive", "youre", "youve", "youll", "youd",
    "dont", "cant", "wont", "didnt", "doesnt", "isnt", "wasnt",
    "werent", "havent", "hasnt", "hadnt", "wouldnt", "couldnt",
    "shouldnt", "aint", "gonna", "wanna", "gotta", "til",
    "thats", "hes", "shes", "its", "lets", "theyre", "whos",
    "ill", "wed", "hed", "shed", "theyd", "theyve", "theyll",
    # Genius artifacts
    "contributors", "translations", "lyrics", "read", "more", "embed",
}

def _strip_genius_header(text: str) -> str:
    text = re.sub(r"^\d+\s*Contributor.*?Lyrics(?:.*?Read More\s*)?", "",
                  text, count=1, flags=re.DOTALL)
    text = re.sub(r"\d+Embed\s*$", "", text)
    return text.strip()

def make_views(text: str) -> dict[str, str]:
    """Return {raw, tokenized, clean} views of a single song's lyrics."""
    raw = _strip_genius_header(text)

    lower = raw.lower()
    tokens_with_stopwords = re.findall(r"[a-z']+", lower)
    tokenized = " ".join(t.strip("'") for t in tokens_with_stopwords if t.strip("'"))

    clean_tokens = [
        t for t in tokenized.split()
        if t not in _STOP and len(t) > 2
    ]
    clean = " ".join(clean_tokens)

    return {"raw": raw, "tokenized": tokenized, "clean": clean}
