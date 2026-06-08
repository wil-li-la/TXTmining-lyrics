"""Preprocess lyrics: clean text, remove stopwords, deduplicate chorus lines."""

import re
import pandas as pd
from nltk.corpus import stopwords

try:
    from nltk.corpus import wordnet as _wn
    _wn.synsets("test")  # force-load; raises LookupError if data missing
except LookupError:
    _wn = None

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
    "theres", "wheres", "heres", "whats", "hows", "youll",
    # Slang / dropped-letter fragments
    "yall", "tryna", "gon", "finna", "bout", "cmon", "ima", "imma",
    "gimme", "lemme", "kinda", "sorta", "outta", "lotta", "nah", "naw",
    # Genius / dataset artifacts ("miscellaneous" is a walkerkq category tag
    # embedded at the front of ~89 songs; "instrumental" marks lyric-less tracks)
    "contributors", "translations", "lyrics", "read", "more",
    "miscellaneous", "instrumental",
}

# Foreign-language function words (Spanish/French/German) that leak in from
# non-English songs in the corpus (Despacito, Macarena, Bailando, ...).
# These are not English content words and pollute the word clouds.
FOREIGN_WORDS = {
    # Spanish
    "que", "quiero", "vida", "loca", "tus", "con", "para", "baila", "bailando",
    "sin", "los", "las", "una", "uno", "voy", "eres", "ser", "ven", "como",
    "por", "mas", "muy", "corazon", "amor", "todo", "nada", "noche", "dia",
    "soy", "dale", "mundo", "aqui", "asi", "bien", "esta", "este", "esa",
    "poquito", "despacito", "shakira", "fuego", "calor", "quiere", "tiene",
    "pero", "porque", "tambien", "donde", "cuando", "siempre", "nunca",
    "ahora", "tengo", "contigo", "conmigo", "gusta", "sabe", "puedo",
    "vamos", "dame", "mami", "bebe", "ella", "hace", "cosas", "ojos",
    "manos", "eso", "esto", "otra", "otro", "hola", "gracias", "quien",
    "nadie", "algo", "mejor", "ahi", "ese", "esos", "esas", "estas",
    "hoy", "hasta", "nos", "estoy", "mira", "oye", "dios", "vez", "fui",
    "buena", "bueno", "toda", "vas", "fue", "dije", "dijo",
    # French
    "pour", "avec", "dans", "vie", "oui", "non", "est", "les", "des", "une",
    "moi", "toi", "mon", "deja",
    # German
    "ich", "nicht", "ein", "und", "aber", "der", "die", "das",
    # ambiguous short foreign tokens also common as English noise
    "el", "en", "mi", "tu", "yo", "si",
}
STOP_WORDS |= FOREIGN_WORDS

# Colloquial "-in'" spellings → restore the dropped final g so they merge with
# their canonical form ("livin" -> "living", "killin" -> "killing"). A token is
# only restored when "<token>g" is a real English word, which protects real
# words that already end in "in" (skin, begin, again, within, brain).
# Curated fallback covers the common cases when the wordnet corpus is absent.
_DROPPED_G_FALLBACK = {
    "livin", "comin", "goin", "doin", "makin", "takin", "tryin", "lovin",
    "feelin", "wishin", "layin", "playin", "sayin", "runnin", "lookin",
    "talkin", "walkin", "gettin", "givin", "holdin", "fallin", "callin",
    "singin", "dancin", "missin", "kissin", "waitin", "workin", "nothin",
    "somethin", "mornin", "evenin", "rollin", "burnin", "turnin", "movin",
    "hopin", "cryin", "smilin", "shinin", "flyin", "dyin", "lyin", "bein",
    "leavin", "killin", "chillin", "droppin", "hangin", "trippin", "dreamin",
    "somethin", "nothin", "everythin", "anythin",
}
_restore_cache: dict[str, str] = {}


def restore_dropped_g(token: str) -> str:
    """Map a colloquial '-in' token to its '-ing' form when that is a real word."""
    if len(token) < 4 or not token.endswith("in"):
        return token
    if token in _restore_cache:
        return _restore_cache[token]
    candidate = token + "g"
    # Restore if "<token>g" is a real word (wordnet) OR a known colloquial form.
    # The fallback also covers function words wordnet lacks (something, nothing).
    in_dict = _wn is not None and bool(_wn.synsets(candidate))
    restored = candidate if (in_dict or token in _DROPPED_G_FALLBACK) else token
    _restore_cache[token] = restored
    return restored


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
    """Lowercase, remove punctuation, normalize slang, drop stopwords/short tokens."""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    # Remove repeated-syllable nonsense, incl. 2x repeats (e.g. "lalalala",
    # "nanana", "ohoh", "dohdoh"). Anchored to whole tokens of 1-3 char syllables.
    text = re.sub(r"\b([a-z]{1,3})\1+\b", "", text)
    tokens = text.split()
    # Restore the dropped g on colloquial "-in'" spellings before filtering.
    tokens = [restore_dropped_g(t) for t in tokens]
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]
    return " ".join(tokens)


def main():
    """Read the full corpus (1960s-2020s), preprocess, and save.

    Uses data/lyrics_full.csv (~4.9k songs, lyrics in `lyrics_raw`) rather than
    the legacy 96-song data/lyrics.csv — the larger corpus is what the rest of
    the v2 analysis uses, covers the 1960s-1980s, and makes per-decade word
    clouds robust (no single song can dominate ~900 songs).
    """
    df = pd.read_csv("data/lyrics_full.csv")
    df = df.rename(columns={"lyrics_raw": "lyrics"})
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
    print("Songs per decade:")
    print(df["decade"].value_counts().sort_index().to_string())
    print(f"\nSample (first 200 chars):\n{df['lyrics_clean'].iloc[0][:200]}")


if __name__ == "__main__":
    main()
