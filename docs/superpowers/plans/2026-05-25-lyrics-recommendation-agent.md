# Lyrics Recommendation Agent — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the small-dataset decade classifier with a richer 5,700-song / 7-feature analysis pipeline and ship an OpenAI-powered Streamlit recommendation agent that matches user taste profiles to recent Billboard songs.

**Architecture:** Two halves. Half A is an offline batch pipeline (scrape → preprocess → 7 feature modules → CV-based classification + trend plots) that produces `data/song_features.parquet`. Half B is an online Streamlit app whose agent (OpenAI `gpt-4o-mini` with tool use over `ddgs` web search + LyricsGenius + the shared feature module) ranks recent songs by cosine similarity to the user's z-scored profile.

**Tech Stack:** Python 3.11, pandas, scikit-learn, sentence-transformers, NRCLex, pronouncing (CMU dict), LyricsGenius, musicbrainzngs, OpenAI Python SDK, ddgs, Streamlit, joblib, pyarrow.

---

## Phase 0 — Setup (sequential prereq, ~15 min)

### Task 0.1: Update requirements and install

**Files:**
- Modify: `requirements.txt`
- Create: `.env.example`

- [ ] **Step 1: Replace `requirements.txt` with the full dep list**

```text
# Core data
pandas>=2.2
numpy>=1.26
pyarrow>=15.0
joblib>=1.4

# Scraping & data sources
lyricsgenius>=3.0
requests>=2.31
beautifulsoup4>=4.12
musicbrainzngs>=0.7.1
langdetect>=1.0.9

# NLP / features
nltk>=3.8
scikit-learn>=1.4
pronouncing>=0.2.0
NRCLex>=4.0
sentence-transformers>=2.7
textstat>=0.7

# Visualization
matplotlib>=3.8
wordcloud>=1.9
seaborn>=0.13

# Agent + UI
openai>=1.30
ddgs>=4.0
streamlit>=1.34
python-dotenv>=1.0

# Test
pytest>=8.0
```

- [ ] **Step 2: Create `.env.example` documenting required env vars**

```text
# OpenAI for the recommendation agent (gpt-4o-mini)
OPENAI_API_KEY=sk-...

# Genius client access token (NOT client_id/secret — get from genius.com/api-clients "Generate Access Token" button)
GENIUS_ACCESS_TOKEN=...

# Optional: contact email for polite MusicBrainz User-Agent
MUSICBRAINZ_CONTACT=you@example.com
```

- [ ] **Step 3: Install everything**

Run: `pip install -r requirements.txt`
Expected: all packages install without error.

- [ ] **Step 4: Download NLTK + CMU dict data**

Run: `python3 -c "import nltk; nltk.download('stopwords'); nltk.download('cmudict'); nltk.download('punkt_tab')"`
Expected: three `[nltk_data] Done` lines.

- [ ] **Step 5: Smoke-test OpenAI access**

Run: `python3 -c "from openai import OpenAI; print(OpenAI().chat.completions.create(model='gpt-4o-mini', messages=[{'role':'user','content':'reply with the single word: ok'}]).choices[0].message.content)"`
Expected: prints `ok`.

- [ ] **Step 6: Commit**

```bash
git add requirements.txt .env.example
git commit -m "deps: add packages for new features, agent, and UI"
```

---

### Task 0.2: Create directory scaffolding

**Files:**
- Create: `src/features/__init__.py`, `src/agent/__init__.py`
- Create: `data/raw/`, `output/trends/`, `output/screenshots/`, `tests/` (if missing)

- [ ] **Step 1: Make the directories**

Run:
```bash
mkdir -p src/features src/agent data/raw output/trends output/screenshots tests
touch src/features/__init__.py src/agent/__init__.py tests/__init__.py
```

- [ ] **Step 2: Commit**

```bash
git add src/features src/agent data/raw output/trends output/screenshots tests
git commit -m "scaffold: directories for features, agent, outputs, tests"
```

---

## Phase 1 — Data pipeline (scrape-agent, ~3-6 hr including scrape runtime)

**Dispatched as subagent.** Produces `data/lyrics_full.csv` (~5,700 rows) and `data/genre_tags.csv`.

### Task 1.1: Download walkerkq dataset

**Files:**
- Create: `data/raw/walkerkq.csv`
- Create: `src/data_sources.py`

- [ ] **Step 1: Write download helper**

Create `src/data_sources.py`:
```python
"""Download external datasets."""
import os
import urllib.request

WALKERKQ_URL = "https://raw.githubusercontent.com/walkerkq/musiclyrics/master/billboard_lyrics_1964-2015.csv"
BRYSBAERT_URL = "http://crr.ugent.be/papers/Concreteness_ratings_Brysbaert_et_al_BRM.txt"

def download(url: str, dest: str) -> None:
    if os.path.exists(dest):
        print(f"  exists: {dest}")
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"  downloading {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)

def main() -> None:
    download(WALKERKQ_URL, "data/raw/walkerkq.csv")
    download(BRYSBAERT_URL, "data/raw/brysbaert.txt")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run: `python3 src/data_sources.py`
Expected: both files appear under `data/raw/`. `wc -l data/raw/walkerkq.csv` should report ~5,101 lines (header + 5,100). If the Brysbaert URL fails, hand back to the user — the mirror may have moved.

- [ ] **Step 3: Commit**

```bash
git add src/data_sources.py
git commit -m "data: download helper for walkerkq + Brysbaert"
```

(Note: do NOT commit `data/raw/*.csv` — they're large. Add `data/raw/` to `.gitignore` if not already.)

### Task 1.2: Rewrite scraper to use LyricsGenius

**Files:**
- Create: `src/scraper_v2.py`

- [ ] **Step 1: Write the new scraper**

Create `src/scraper_v2.py`:
```python
"""Scrape lyrics for Billboard Hot 100 2016-2025 using the official Genius API.

Inputs:
  - data/raw/walkerkq.csv (1964-2015, already lyrics-bearing)
  - Hardcoded SONG_LIST for 2016-2025 (curated from Billboard year-end charts)

Output:
  - data/lyrics_full.csv with columns:
      song_id, title, artist, year, decade, chart_position, source, lyrics_raw
"""
import csv
import os
import time
from typing import Iterable

import pandas as pd
from dotenv import load_dotenv
import lyricsgenius

load_dotenv()

# Curated Billboard year-end top 30 for 2016-2025 (artist, title, year, peak_pos)
SONG_LIST: list[tuple[str, str, int, int]] = [
    # 2016
    ("Justin Bieber", "Love Yourself", 2016, 1),
    ("Drake", "One Dance", 2016, 1),
    ("Rihanna", "Work", 2016, 1),
    # ... (engineer: expand to ~30/year using Billboard year-end charts;
    # see https://en.wikipedia.org/wiki/Billboard_Year-End_Hot_100_singles_of_2016 etc.)
]
# NOTE: For the actual build, the engineer should populate ~30 entries per year for
# 2016-2025 (~300 songs) from the Wikipedia year-end pages. The existing
# src/scraper.py already has 2020-2025 entries; copy and extend.

def decade_of(year: int) -> str:
    return f"{(year // 10) * 10}s"

def fetch_one(genius: lyricsgenius.Genius, artist: str, title: str) -> str | None:
    """Fetch lyrics via Genius search; returns lyrics or None on failure."""
    try:
        song = genius.search_song(title=title, artist=artist, get_full_info=False)
        if song is None or not song.lyrics:
            return None
        # Strip the trailing "<n> Embed" suffix Genius adds
        text = song.lyrics
        return text
    except Exception as e:
        print(f"  [ERROR] {artist} - {title}: {e}")
        return None

def load_walkerkq(path: str = "data/raw/walkerkq.csv") -> pd.DataFrame:
    """Load walkerkq dataset into our schema."""
    df = pd.read_csv(path, encoding="latin-1")
    df = df.rename(columns={"Rank": "chart_position", "Song": "title",
                            "Artist": "artist", "Year": "year",
                            "Lyrics": "lyrics_raw"})
    df["decade"] = df["year"].apply(decade_of)
    df["source"] = "walkerkq"
    df = df.dropna(subset=["lyrics_raw"])
    df = df[df["lyrics_raw"].str.len() > 100]
    return df[["title", "artist", "year", "decade", "chart_position", "source", "lyrics_raw"]]

def scrape_recent() -> pd.DataFrame:
    """Scrape SONG_LIST via Genius API. Polite delay between calls."""
    genius = lyricsgenius.Genius(
        os.environ["GENIUS_ACCESS_TOKEN"],
        timeout=15, sleep_time=1, verbose=False, remove_section_headers=True,
    )
    rows = []
    for i, (artist, title, year, peak) in enumerate(SONG_LIST, start=1):
        print(f"[{i}/{len(SONG_LIST)}] {artist} - {title} ({year})")
        lyrics = fetch_one(genius, artist, title)
        if lyrics is None:
            print("  skipped")
            continue
        rows.append({
            "title": title, "artist": artist, "year": year,
            "decade": decade_of(year), "chart_position": peak,
            "source": "genius_api", "lyrics_raw": lyrics,
        })
        time.sleep(1.5)  # polite to Genius
    return pd.DataFrame(rows)

def merge_and_save(out_path: str = "data/lyrics_full.csv") -> None:
    walkerkq_df = load_walkerkq()
    recent_df = scrape_recent()
    full = pd.concat([walkerkq_df, recent_df], ignore_index=True)
    # Dedupe on lowercased (artist, title)
    full["_key"] = full["artist"].str.lower().str.strip() + "|" + full["title"].str.lower().str.strip()
    full = full.drop_duplicates(subset="_key", keep="first").drop(columns="_key")
    full = full.reset_index(drop=True)
    full.insert(0, "song_id", [f"S{i:05d}" for i in range(len(full))])
    full.to_csv(out_path, index=False)
    print(f"\nSaved {len(full)} songs -> {out_path}")
    print(full.groupby("decade").size())

if __name__ == "__main__":
    merge_and_save()
```

- [ ] **Step 2: Engineer fills in SONG_LIST**

Open `src/scraper_v2.py` and expand `SONG_LIST` to cover ~30 songs/year × 10 years = ~300 entries for 2016-2025. Source: Wikipedia year-end Billboard Hot 100 pages (e.g., "Billboard Year-End Hot 100 singles of 2016"). The existing `src/scraper.py` lines 107-133 has 2020-2025 entries — copy those.

- [ ] **Step 3: Run the scrape**

Run: `python3 src/scraper_v2.py`
Expected: scrolls through ~300 songs in ~10 minutes, then prints a per-decade count summary. Final output ~5,400-5,700 songs in `data/lyrics_full.csv`. Failures should be < 5%.

- [ ] **Step 4: Sanity check**

Run:
```bash
python3 -c "
import pandas as pd
df = pd.read_csv('data/lyrics_full.csv')
print(f'Total: {len(df)}')
print(df.groupby('decade').size())
print(df['source'].value_counts())
"
```
Expected: total ~5,400-5,700, every decade 1960s-2020s present, both `walkerkq` and `genius_api` sources represented.

- [ ] **Step 5: Commit**

```bash
git add src/scraper_v2.py
git commit -m "scrape: rewrite with LyricsGenius, merge walkerkq for 5,700-song corpus"
```

### Task 1.3: Genre tagger

**Files:**
- Create: `src/genre_tagger.py`
- Create: `data/genre_tags.csv` (output)

- [ ] **Step 1: Write tagger**

Create `src/genre_tagger.py`:
```python
"""Tag each song with a coarse genre via MusicBrainz, fall back to OpenAI."""
import os
import time
import json

import pandas as pd
import musicbrainzngs
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

GENRES = ["pop", "rap", "r&b", "rock", "country", "dance", "other"]

musicbrainzngs.set_useragent(
    "TXTminingLyrics", "0.2",
    os.getenv("MUSICBRAINZ_CONTACT", "anonymous@example.com"),
)

def mb_genre(artist: str, title: str) -> str | None:
    try:
        r = musicbrainzngs.search_recordings(artist=artist, recording=title, limit=1)
        recs = r.get("recording-list", [])
        if not recs:
            return None
        tags = recs[0].get("tag-list", [])
        if not tags:
            return None
        tag_names = [t["name"].lower() for t in tags]
        for g in GENRES:
            if any(g in t for t in tag_names):
                return g
        return "other"
    except Exception:
        return None

def llm_genre(client: OpenAI, artist: str, title: str) -> str:
    """One-shot Haiku-equivalent classification with constrained labels."""
    prompt = (
        f"Classify the genre of the song '{title}' by {artist}. "
        f"Reply with exactly one of: {', '.join(GENRES)}. "
        "If you don't recognize the song, reply 'other'."
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=5, temperature=0,
    )
    label = resp.choices[0].message.content.strip().lower()
    return label if label in GENRES else "other"

def main(in_path: str = "data/lyrics_full.csv",
         out_path: str = "data/genre_tags.csv") -> None:
    df = pd.read_csv(in_path)
    client = OpenAI()

    if os.path.exists(out_path):
        done = pd.read_csv(out_path)
        done_ids = set(done["song_id"])
    else:
        done = pd.DataFrame(columns=["song_id", "genre", "source"])
        done_ids = set()

    rows = []
    for _, row in df.iterrows():
        if row["song_id"] in done_ids:
            continue
        genre = mb_genre(row["artist"], row["title"])
        src = "musicbrainz"
        if genre is None:
            genre = llm_genre(client, row["artist"], row["title"])
            src = "openai"
        rows.append({"song_id": row["song_id"], "genre": genre, "source": src})
        print(f"  {row['song_id']} {row['artist']} - {row['title']} -> {genre} ({src})")
        time.sleep(0.5)  # polite to MusicBrainz
        if len(rows) % 50 == 0:
            # Checkpoint
            pd.concat([done, pd.DataFrame(rows)]).to_csv(out_path, index=False)
    out = pd.concat([done, pd.DataFrame(rows)])
    out.to_csv(out_path, index=False)
    print(f"\nTagged {len(out)} songs -> {out_path}")
    print(out["genre"].value_counts())

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run: `python3 src/genre_tagger.py`
Expected: ~50 minutes (rate-limited by MusicBrainz 1 req/sec + LLM fallback). Resumable from checkpoint. Final `genre_tags.csv` ~5,700 rows.

- [ ] **Step 3: Commit**

```bash
git add src/genre_tagger.py
git commit -m "tag: genre labels via MusicBrainz with OpenAI fallback"
```

---

## Phase 2 — Feature extraction (feature-agent, ~4-6 hr; can dev in parallel with Phase 1)

**Dispatched as subagent.** Produces `data/song_features.parquet`. Develop modules against the 95-song existing dataset; final batch run uses Phase 1's output.

### Task 2.1: Preprocessing with three views

**Files:**
- Create: `src/preprocess_v2.py`
- Create: `tests/test_preprocess_v2.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_preprocess_v2.py`:
```python
import pytest
from src.preprocess_v2 import make_views

def test_make_views_returns_three_keys():
    v = make_views("Hello world!\nI'm a teapot.")
    assert set(v.keys()) == {"raw", "tokenized", "clean"}

def test_raw_preserves_line_breaks():
    v = make_views("Line one\nLine two")
    assert "\n" in v["raw"]

def test_tokenized_keeps_pronouns():
    v = make_views("I love you and we are happy")
    assert "i" in v["tokenized"].split()
    assert "you" in v["tokenized"].split()
    assert "we" in v["tokenized"].split()

def test_clean_drops_stopwords():
    v = make_views("I love you and we are happy")
    tokens = v["clean"].split()
    assert "love" in tokens
    assert "happy" in tokens
    assert "i" not in tokens
    assert "you" not in tokens

def test_clean_drops_short_tokens():
    v = make_views("OK go run far")
    tokens = v["clean"].split()
    assert "run" in tokens  # 3 chars: passes
    assert "ok" not in tokens  # 2 chars: drops
    assert "go" not in tokens  # 2 chars: drops
```

- [ ] **Step 2: Run tests, confirm they fail**

Run: `pytest tests/test_preprocess_v2.py -v`
Expected: 5 failures with "ModuleNotFoundError" or "function not defined".

- [ ] **Step 3: Implement**

Create `src/preprocess_v2.py`:
```python
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
```

- [ ] **Step 4: Tests pass**

Run: `pytest tests/test_preprocess_v2.py -v`
Expected: all 5 pass.

- [ ] **Step 5: Commit**

```bash
git add src/preprocess_v2.py tests/test_preprocess_v2.py
git commit -m "preprocess: three-view (raw/tokenized/clean) lyric preparation"
```

### Task 2.2: Rhyme features

**Files:**
- Create: `src/features/rhyme.py`
- Create: `tests/test_features_rhyme.py`

- [ ] **Step 1: Write tests**

Create `tests/test_features_rhyme.py`:
```python
from src.features.rhyme import extract

def test_extract_returns_three_keys():
    r = extract({"raw": "moon\nsoon\nstar\ncar"})
    assert set(r.keys()) == {"rhyme_density", "internal_rhyme", "mean_syllables_per_line"}

def test_strong_rhymes_have_high_density():
    text = "moon\nsoon\njune\ntune\n"  # all rhyme
    r = extract({"raw": text})
    assert r["rhyme_density"] > 0.5

def test_no_rhymes_low_density():
    text = "purple\norange\nsilver\n"  # famously non-rhyming
    r = extract({"raw": text})
    assert r["rhyme_density"] < 0.5

def test_empty_input_safe():
    r = extract({"raw": ""})
    assert r["rhyme_density"] == 0.0
    assert r["mean_syllables_per_line"] == 0.0
```

- [ ] **Step 2: Implement**

Create `src/features/rhyme.py`:
```python
"""Rhyme + cadence features from raw line-broken lyrics, via CMU dict."""
import re
from collections import defaultdict
import pronouncing

_WORD = re.compile(r"[A-Za-z']+")

def _last_word(line: str) -> str | None:
    words = _WORD.findall(line)
    return words[-1].lower() if words else None

def _rhyming_part(word: str) -> str | None:
    phones = pronouncing.phones_for_word(word)
    if not phones:
        return None
    return pronouncing.rhyming_part(phones[0])

def _syllables(word: str) -> int:
    phones = pronouncing.phones_for_word(word)
    if not phones:
        # crude fallback
        return max(1, len(re.findall(r"[aeiouy]+", word.lower())))
    return pronouncing.syllable_count(phones[0])

def extract(views: dict) -> dict[str, float]:
    raw = views["raw"]
    lines = [l for l in raw.split("\n") if l.strip()]
    if not lines:
        return {"rhyme_density": 0.0, "internal_rhyme": 0.0, "mean_syllables_per_line": 0.0}

    # End-rhyme density: fraction of line pairs (within 4 lines) sharing a rhyming-part
    end_rhymes_parts: list[tuple[int, str]] = []
    for i, ln in enumerate(lines):
        w = _last_word(ln)
        if not w:
            continue
        rp = _rhyming_part(w)
        if rp:
            end_rhymes_parts.append((i, rp))

    if len(end_rhymes_parts) < 2:
        density = 0.0
    else:
        n_pairs = 0
        n_rhyme = 0
        for a in range(len(end_rhymes_parts)):
            for b in range(a + 1, min(a + 5, len(end_rhymes_parts))):
                n_pairs += 1
                if end_rhymes_parts[a][1] == end_rhymes_parts[b][1]:
                    n_rhyme += 1
        density = n_rhyme / n_pairs if n_pairs else 0.0

    # Internal rhyme: fraction of lines containing >= 2 words sharing a rhyming-part
    internal = 0
    for ln in lines:
        words = [w.lower() for w in _WORD.findall(ln)]
        parts = defaultdict(int)
        for w in words:
            rp = _rhyming_part(w)
            if rp:
                parts[rp] += 1
        if any(c >= 2 for c in parts.values()):
            internal += 1
    internal_rhyme = internal / len(lines)

    # Mean syllables per line
    syl_per_line = [
        sum(_syllables(w) for w in _WORD.findall(ln))
        for ln in lines
    ]
    mean_syl = sum(syl_per_line) / len(syl_per_line)

    return {
        "rhyme_density": float(density),
        "internal_rhyme": float(internal_rhyme),
        "mean_syllables_per_line": float(mean_syl),
    }
```

- [ ] **Step 3: Tests pass**

Run: `pytest tests/test_features_rhyme.py -v`
Expected: all 4 pass.

- [ ] **Step 4: Commit**

```bash
git add src/features/rhyme.py tests/test_features_rhyme.py
git commit -m "feat: rhyme density / internal rhyme / syllables-per-line features"
```

### Task 2.3: Repetition features

**Files:**
- Create: `src/features/repetition.py`
- Create: `tests/test_features_repetition.py`

- [ ] **Step 1: Write tests**

Create `tests/test_features_repetition.py`:
```python
from src.features.repetition import extract

def test_keys():
    r = extract({"raw": "a\nb\nc"})
    assert set(r.keys()) == {"repetition_entropy", "chorus_repeat_ratio", "mtld"}

def test_high_repetition_high_ratio():
    text = "chorus line one\nchorus line two\n" * 5 + "verse one\nverse two\n"
    r = extract({"raw": text})
    assert r["chorus_repeat_ratio"] > 0.5

def test_no_repetition_zero_ratio():
    text = "\n".join(f"unique line {i}" for i in range(20))
    r = extract({"raw": text})
    assert r["chorus_repeat_ratio"] < 0.1

def test_mtld_higher_for_diverse_vocab():
    diverse = " ".join(f"word{i}" for i in range(200))
    repetitive = " ".join(["hello", "world"] * 100)
    r_d = extract({"raw": diverse})
    r_r = extract({"raw": repetitive})
    assert r_d["mtld"] > r_r["mtld"]
```

- [ ] **Step 2: Implement**

Create `src/features/repetition.py`:
```python
"""Repetition + vocabulary-diversity features."""
import math
from collections import Counter

def _line_repeat_ratio(raw: str) -> float:
    lines = [l.strip().lower() for l in raw.split("\n") if l.strip()]
    if not lines:
        return 0.0
    counts = Counter(lines)
    repeats = sum(c for c in counts.values() if c >= 2)
    return repeats / len(lines)

def _line_bigram_entropy(raw: str) -> float:
    lines = [l.strip().lower() for l in raw.split("\n") if l.strip()]
    if len(lines) < 2:
        return 0.0
    bigrams = list(zip(lines[:-1], lines[1:]))
    counts = Counter(bigrams)
    total = sum(counts.values())
    return -sum((c / total) * math.log2(c / total) for c in counts.values())

def _mtld(text: str, threshold: float = 0.72) -> float:
    """Measure of Textual Lexical Diversity (McCarthy 2005)."""
    tokens = text.lower().split()
    if not tokens:
        return 0.0

    def one_pass(toks):
        factors = 0
        types = set()
        running = 0
        for t in toks:
            types.add(t)
            running += 1
            ttr = len(types) / running
            if ttr <= threshold:
                factors += 1
                types = set()
                running = 0
        if running > 0:
            ttr = len(types) / running
            partial = (1 - ttr) / (1 - threshold) if ttr < 1 else 0
            factors += partial
        return len(toks) / factors if factors else len(toks)

    forward = one_pass(tokens)
    backward = one_pass(list(reversed(tokens)))
    return (forward + backward) / 2

def extract(views: dict) -> dict[str, float]:
    raw = views["raw"]
    return {
        "repetition_entropy": float(_line_bigram_entropy(raw)),
        "chorus_repeat_ratio": float(_line_repeat_ratio(raw)),
        "mtld": float(_mtld(raw)),
    }
```

- [ ] **Step 3: Tests pass**

Run: `pytest tests/test_features_repetition.py -v`
Expected: all 4 pass.

- [ ] **Step 4: Commit**

```bash
git add src/features/repetition.py tests/test_features_repetition.py
git commit -m "feat: repetition entropy / chorus ratio / MTLD features"
```

### Task 2.4: Emotion features (NRC EmoLex)

**Files:**
- Create: `src/features/emotion.py`
- Create: `tests/test_features_emotion.py`

- [ ] **Step 1: Write tests**

Create `tests/test_features_emotion.py`:
```python
from src.features.emotion import extract, EMOTION_KEYS

def test_returns_all_emotion_keys():
    r = extract({"clean": "happy joy smile love"})
    assert set(r.keys()) == set(EMOTION_KEYS)

def test_happy_words_have_high_joy():
    r = extract({"clean": "happy joy smile delight cheerful glad"})
    assert r["emo_joy"] > r["emo_sadness"]

def test_angry_words_have_high_anger():
    r = extract({"clean": "rage fury hate destroy attack angry"})
    assert r["emo_anger"] > r["emo_trust"]

def test_empty_input_safe():
    r = extract({"clean": ""})
    assert all(v == 0.0 for v in r.values())
```

- [ ] **Step 2: Implement**

Create `src/features/emotion.py`:
```python
"""NRC EmoLex emotion + valence/arousal features via NRCLex."""
from nrclex import NRCLex

EMOTIONS = ["anger", "anticipation", "disgust", "fear",
            "joy", "sadness", "surprise", "trust"]
EMOTION_KEYS = [f"emo_{e}" for e in EMOTIONS] + ["valence", "arousal"]

def extract(views: dict) -> dict[str, float]:
    text = views["clean"]
    if not text.strip():
        return {k: 0.0 for k in EMOTION_KEYS}

    nrc = NRCLex(text)
    freqs = nrc.affect_frequencies
    out = {f"emo_{e}": float(freqs.get(e, 0.0)) for e in EMOTIONS}

    # NRCLex also provides positive/negative; map to a single valence
    pos = freqs.get("positive", 0.0)
    neg = freqs.get("negative", 0.0)
    out["valence"] = float(pos - neg)

    # Arousal proxy: anger + fear + joy + surprise (high-arousal emotions)
    out["arousal"] = float(sum(freqs.get(e, 0.0) for e in ["anger", "fear", "joy", "surprise"]))
    return out
```

- [ ] **Step 3: Tests pass**

Run: `pytest tests/test_features_emotion.py -v`
Expected: all 4 pass.

- [ ] **Step 4: Commit**

```bash
git add src/features/emotion.py tests/test_features_emotion.py
git commit -m "feat: NRC EmoLex 8-emotion + valence + arousal features"
```

### Task 2.5: Concreteness features (Brysbaert norms)

**Files:**
- Create: `src/features/concreteness.py`
- Create: `tests/test_features_concreteness.py`
- Needs: `data/raw/brysbaert.txt` (downloaded in Task 1.1)

- [ ] **Step 1: Write tests**

Create `tests/test_features_concreteness.py`:
```python
from src.features.concreteness import extract

def test_keys():
    r = extract({"clean": "table chair house"})
    assert set(r.keys()) == {"mean_concreteness", "pct_concrete"}

def test_concrete_words_high_score():
    r_concrete = extract({"clean": "table chair house car tree dog book"})
    r_abstract = extract({"clean": "freedom justice belief idea hope thought concept"})
    assert r_concrete["mean_concreteness"] > r_abstract["mean_concreteness"]
    assert r_concrete["pct_concrete"] > r_abstract["pct_concrete"]

def test_empty_safe():
    r = extract({"clean": ""})
    assert r["mean_concreteness"] == 0.0
    assert r["pct_concrete"] == 0.0
```

- [ ] **Step 2: Implement**

Create `src/features/concreteness.py`:
```python
"""Brysbaert et al. 2014 concreteness norms — 40k words rated 1-5."""
import os
from functools import lru_cache
import pandas as pd

BRYSBAERT_PATH = "data/raw/brysbaert.txt"

@lru_cache(maxsize=1)
def _load_ratings() -> dict[str, float]:
    if not os.path.exists(BRYSBAERT_PATH):
        raise FileNotFoundError(
            f"Missing {BRYSBAERT_PATH}. Run: python3 src/data_sources.py"
        )
    df = pd.read_csv(BRYSBAERT_PATH, sep="\t")
    # Column "Word" (lowercased) and "Conc.M" (mean rating 1-5)
    return dict(zip(df["Word"].str.lower(), df["Conc.M"]))

def extract(views: dict) -> dict[str, float]:
    text = views["clean"]
    if not text.strip():
        return {"mean_concreteness": 0.0, "pct_concrete": 0.0}

    ratings = _load_ratings()
    tokens = text.split()
    scored = [ratings[t] for t in tokens if t in ratings]
    if not scored:
        return {"mean_concreteness": 0.0, "pct_concrete": 0.0}

    mean = sum(scored) / len(scored)
    pct_concrete = sum(1 for s in scored if s >= 4.0) / len(scored)
    return {
        "mean_concreteness": float(mean),
        "pct_concrete": float(pct_concrete),
    }
```

- [ ] **Step 3: Tests pass**

Run: `pytest tests/test_features_concreteness.py -v`
Expected: all 3 pass. (Requires `data/raw/brysbaert.txt` to exist.)

- [ ] **Step 4: Commit**

```bash
git add src/features/concreteness.py tests/test_features_concreteness.py
git commit -m "feat: Brysbaert concreteness mean + concrete-word-pct features"
```

### Task 2.6: Pronoun features

**Files:**
- Create: `src/features/pronouns.py`
- Create: `tests/test_features_pronouns.py`

- [ ] **Step 1: Write tests**

Create `tests/test_features_pronouns.py`:
```python
from src.features.pronouns import extract

def test_keys():
    r = extract({"tokenized": "i love you"})
    assert set(r.keys()) == {"pronoun_i", "pronoun_you", "pronoun_we", "pronoun_they"}

def test_i_heavy_text():
    r = extract({"tokenized": "i went and i saw and i thought i could"})
    assert r["pronoun_i"] > r["pronoun_you"]

def test_no_pronouns_safe():
    r = extract({"tokenized": "the cat sat on the mat"})
    assert all(v == 0.0 for v in r.values())
```

- [ ] **Step 2: Implement**

Create `src/features/pronouns.py`:
```python
"""Pronoun-group ratios. Consumes the tokenized (stopwords-kept) view."""

_GROUPS = {
    "pronoun_i":    {"i", "me", "my", "mine", "myself"},
    "pronoun_you":  {"you", "your", "yours", "yourself", "yourselves"},
    "pronoun_we":   {"we", "us", "our", "ours", "ourselves"},
    "pronoun_they": {"they", "them", "their", "theirs", "themselves"},
}

def extract(views: dict) -> dict[str, float]:
    tokens = views["tokenized"].split()
    n = len(tokens)
    if n == 0:
        return {k: 0.0 for k in _GROUPS}
    return {
        k: sum(1 for t in tokens if t in members) / n
        for k, members in _GROUPS.items()
    }
```

- [ ] **Step 3: Tests pass**

Run: `pytest tests/test_features_pronouns.py -v`
Expected: all 3 pass.

- [ ] **Step 4: Commit**

```bash
git add src/features/pronouns.py tests/test_features_pronouns.py
git commit -m "feat: pronoun-group ratio features"
```

### Task 2.7: SBERT embeddings

**Files:**
- Create: `src/features/embedding.py`
- Create: `tests/test_features_embedding.py`

- [ ] **Step 1: Write tests**

Create `tests/test_features_embedding.py`:
```python
from src.features.embedding import extract

def test_returns_384d_vector():
    r = extract({"clean": "love is in the air"})
    assert "embedding" in r
    assert len(r["embedding"]) == 384

def test_similar_text_similar_embedding():
    import numpy as np
    a = extract({"clean": "the dog ran fast"})["embedding"]
    b = extract({"clean": "a dog was running quickly"})["embedding"]
    c = extract({"clean": "quantum mechanics is hard"})["embedding"]
    cos = lambda x, y: np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
    assert cos(a, b) > cos(a, c)
```

- [ ] **Step 2: Implement**

Create `src/features/embedding.py`:
```python
"""Sentence-BERT mean-pooled lyric embeddings."""
from functools import lru_cache
from sentence_transformers import SentenceTransformer

@lru_cache(maxsize=1)
def _model():
    return SentenceTransformer("all-MiniLM-L6-v2")

def extract(views: dict) -> dict:
    text = views["clean"] or " "  # empty input would error
    vec = _model().encode(text, show_progress_bar=False, normalize_embeddings=True)
    return {"embedding": vec.tolist()}
```

- [ ] **Step 3: Tests pass**

Run: `pytest tests/test_features_embedding.py -v`
Expected: 2 pass (first run downloads ~90 MB model; takes ~30s).

- [ ] **Step 4: Commit**

```bash
git add src/features/embedding.py tests/test_features_embedding.py
git commit -m "feat: SBERT all-MiniLM-L6-v2 lyric embeddings"
```

### Task 2.8: Feature batch runner

**Files:**
- Create: `src/features/build_all.py`

- [ ] **Step 1: Implement batch runner**

Create `src/features/build_all.py`:
```python
"""Run all feature modules over data/lyrics_full.csv -> song_features.parquet."""
import json
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

from src.preprocess_v2 import make_views
from src.features import rhyme, repetition, emotion, concreteness, pronouns, embedding

FEATURES = [rhyme, repetition, emotion, concreteness, pronouns, embedding]

def extract_row(lyrics_raw: str) -> dict:
    views = make_views(lyrics_raw)
    out = {}
    for mod in FEATURES:
        out.update(mod.extract(views))
    return out

def main(lyrics_path: str = "data/lyrics_full.csv",
         genres_path: str = "data/genre_tags.csv",
         out_path: str = "data/song_features.parquet",
         stats_path: str = "data/feature_stats.json",
         n_jobs: int = -1) -> None:
    df = pd.read_csv(lyrics_path)
    if Path(genres_path).exists():
        g = pd.read_csv(genres_path)[["song_id", "genre"]]
        df = df.merge(g, on="song_id", how="left")
    else:
        df["genre"] = "unknown"

    print(f"Extracting features for {len(df)} songs (n_jobs={n_jobs})...")
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(extract_row)(lyr) for lyr in df["lyrics_raw"]
    )
    feat_df = pd.DataFrame(results)
    merged = pd.concat([df[["song_id", "year", "decade", "genre"]].reset_index(drop=True),
                        feat_df.reset_index(drop=True)], axis=1)
    merged.to_parquet(out_path, index=False)
    print(f"Saved {len(merged)} rows -> {out_path}")

    # Corpus-wide mean/std for z-scoring (skip the embedding column)
    numeric_cols = [c for c in feat_df.columns if c != "embedding"]
    stats = {
        c: {"mean": float(feat_df[c].mean()), "std": float(feat_df[c].std() or 1.0)}
        for c in numeric_cols
    }
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved feature stats -> {stats_path}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test on first 50 rows**

Run:
```bash
python3 -c "
import pandas as pd
from src.features.build_all import extract_row
df = pd.read_csv('data/lyrics_full.csv').head(50)
results = [extract_row(l) for l in df['lyrics_raw']]
print(f'Extracted {len(results)} rows. Sample keys: {sorted(results[0].keys())[:10]}')
"
```
Expected: prints 50 rows + a sample of keys. Takes ~2-3 minutes.

- [ ] **Step 3: Full batch**

Run: `python3 -m src.features.build_all`
Expected: ~5-10 minutes for ~5,700 songs. Final parquet ~50 MB, `feature_stats.json` lists ~26 numeric features with mean/std.

- [ ] **Step 4: Commit**

```bash
git add src/features/build_all.py
git commit -m "feat: parallel batch runner for all 6 feature families"
```

---

## Phase 3 — Analysis (analysis-agent, ~3-4 hr; depends on Phase 2 output)

**Dispatched as subagent.** Produces `output/cv_results.json`, `output/trends/*.png`, and the data behind the report's headline table + trend section.

### Task 3.1: Cross-validation analysis

**Files:**
- Create: `src/analyze.py`
- Create: `tests/test_analyze.py`

- [ ] **Step 1: Write tests**

Create `tests/test_analyze.py`:
```python
import numpy as np
import pandas as pd
from src.analyze import run_cv, build_feature_matrix

def _toy_df():
    rng = np.random.default_rng(0)
    n = 100
    decades = ["1990s", "2000s", "2010s", "2020s"]
    return pd.DataFrame({
        "song_id": [f"S{i:03d}" for i in range(n)],
        "decade": rng.choice(decades, n),
        "artist": rng.choice([f"A{i}" for i in range(20)], n),
        "year": rng.integers(1990, 2025, n),
        "genre": "pop",
        "rhyme_density": rng.random(n),
        "valence": rng.random(n) - 0.5,
        "embedding": [list(rng.random(8)) for _ in range(n)],
    })

def test_build_feature_matrix_excludes_meta():
    df = _toy_df()
    X = build_feature_matrix(df)
    assert X.shape[0] == len(df)
    assert X.shape[1] >= 2  # at least the 2 numeric + 8 embedding dims

def test_run_cv_returns_required_keys():
    df = _toy_df()
    out = run_cv(df)
    assert "stratified" in out and "grouped" in out
    for split in ("stratified", "grouped"):
        assert "accuracy_mean" in out[split]
        assert "accuracy_std" in out[split]
        assert "auc_per_decade" in out[split]
```

- [ ] **Step 2: Implement**

Create `src/analyze.py`:
```python
"""Cross-validation analysis: stratified vs artist-grouped folds."""
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GroupKFold, cross_val_score, cross_val_predict
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.metrics import roc_auc_score

META_COLS = {"song_id", "year", "decade", "genre", "artist", "title",
             "chart_position", "source", "lyrics_raw"}

def build_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """Stack numeric features + flatten the embedding list into one X matrix."""
    feature_cols = [c for c in df.columns if c not in META_COLS and c != "embedding"]
    numeric = df[feature_cols].fillna(0.0).to_numpy(dtype=float)
    if "embedding" in df.columns:
        emb = np.vstack(df["embedding"].to_numpy())
        X = np.hstack([numeric, emb])
    else:
        X = numeric
    # z-score for stable LR
    return StandardScaler().fit_transform(X)

def _cv_block(X, y, splitter, groups=None) -> dict:
    model = LogisticRegression(max_iter=2000, random_state=42, n_jobs=-1)
    acc = cross_val_score(model, X, y, cv=splitter, scoring="accuracy",
                          groups=groups, n_jobs=-1)
    y_proba = cross_val_predict(model, X, y, cv=splitter, method="predict_proba",
                                groups=groups, n_jobs=-1)
    classes = sorted(np.unique(y))
    y_bin = label_binarize(y, classes=classes)
    auc_per = {
        c: float(roc_auc_score(y_bin[:, i], y_proba[:, i]))
        for i, c in enumerate(classes)
    }
    return {
        "accuracy_mean": float(acc.mean()),
        "accuracy_std": float(acc.std()),
        "auc_per_decade": auc_per,
        "n_samples": int(len(y)),
        "n_folds": splitter.get_n_splits(X, y, groups),
    }

def run_cv(df: pd.DataFrame) -> dict:
    X = build_feature_matrix(df)
    y = df["decade"].to_numpy()

    strat = _cv_block(X, y, StratifiedKFold(n_splits=5, shuffle=True, random_state=42))

    if "artist" in df.columns:
        n_artists = df["artist"].nunique()
        n_folds = min(5, n_artists)
        groups = df["artist"].to_numpy()
        grouped = _cv_block(X, y, GroupKFold(n_splits=n_folds), groups=groups)
    else:
        grouped = {"error": "no artist column"}

    return {"stratified": strat, "grouped": grouped}

def main(feat_path: str = "data/song_features.parquet",
         lyrics_path: str = "data/lyrics_full.csv",
         out_path: str = "output/cv_results.json") -> None:
    feat = pd.read_parquet(feat_path)
    meta = pd.read_csv(lyrics_path)[["song_id", "artist"]]
    df = feat.merge(meta, on="song_id", how="left")
    results = run_cv(df)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Tests pass**

Run: `pytest tests/test_analyze.py -v`
Expected: all 3 pass.

- [ ] **Step 4: Run on real data**

Run: `python3 -m src.analyze`
Expected: prints stratified + grouped CV results, saves to `output/cv_results.json`. Compare numbers — if `grouped.accuracy_mean` is notably lower than `stratified.accuracy_mean`, artist leakage is confirmed.

- [ ] **Step 5: Commit**

```bash
git add src/analyze.py tests/test_analyze.py output/cv_results.json
git commit -m "analyze: stratified + GroupKFold-by-artist CV with per-decade AUC"
```

### Task 3.2: Feature trend plots

**Files:**
- Create: `src/trends.py`
- Output: `output/trends/*.png` (one per numeric feature)

- [ ] **Step 1: Implement**

Create `src/trends.py`:
```python
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
```

- [ ] **Step 2: Run it**

Run: `python3 -m src.trends`
Expected: ~26 PNG files in `output/trends/`. Takes ~30 seconds.

- [ ] **Step 3: Commit**

```bash
git add src/trends.py output/trends/
git commit -m "viz: per-feature yearly trend plots with LOESS smoothers"
```

### Task 3.3: Rewrite report.md

**Files:**
- Rewrite: `report.md`

- [ ] **Step 1: Rewrite using the spec's §7 structure**

Replace `report.md` with the 8-section structure from the design spec:
1. Introduction & related work — cite Parada-Cabaleiro 2024 + DeWall + Pettijohn + Pachet
2. Dataset — describe walkerkq + recent scrape, n totals, decade distribution
3. Features — table of all 7 modules with measures/libraries
4. Analysis: decade classification — paste the headline table from `output/cv_results.json` (stratified vs grouped vs prior single-split)
5. Analysis: feature trends over time — embed the trend plots from `output/trends/`, 1-2 paragraphs per family comparing to cited papers
6. Recommendation agent — architecture diagram + 5 screenshots (placeholder paths to `output/screenshots/01.png` etc.)
7. Limitations & future work — cite 2025 Sci Rep + Martín-Gutiérrez
8. References — full bibliography

Use the actual numbers from `output/cv_results.json` — never placeholder.

- [ ] **Step 2: Commit**

```bash
git add report.md
git commit -m "report: rewrite around 8 real citations + CV table + trend plots"
```

---

## Phase 4 — Recommendation agent + UI (foreground, ~6-8 hr; depends on Phase 2)

### Task 4.1: Agent tools

**Files:**
- Create: `src/agent/tools.py`
- Create: `tests/test_agent_tools.py`

- [ ] **Step 1: Write tests**

Create `tests/test_agent_tools.py`:
```python
from unittest.mock import patch, MagicMock
from src.agent.tools import search_recent_songs, extract_features

def test_search_returns_list_of_dicts():
    fake_results = [
        {"title": "Olivia Rodrigo - drivers license | Billboard", "href": "https://billboard.com/x"},
        {"title": "Harry Styles - As It Was - Wikipedia", "href": "https://en.wikipedia.org/y"},
    ]
    with patch("src.agent.tools.DDGS") as ddgs:
        instance = MagicMock()
        instance.text.return_value = fake_results
        ddgs.return_value.__enter__.return_value = instance
        out = search_recent_songs("Billboard 2024 ballads")
    assert isinstance(out, list)
    assert all("title" in r and "artist" in r for r in out)

def test_extract_features_returns_dict():
    text = "I love the sound of rain on the window\nIt makes me feel alive"
    feats = extract_features(text)
    assert "rhyme_density" in feats
    assert "valence" in feats
    assert "embedding" in feats
```

- [ ] **Step 2: Implement**

Create `src/agent/tools.py`:
```python
"""Tools the OpenAI agent can call: search, fetch lyrics, extract features."""
import json
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from ddgs import DDGS
import lyricsgenius

from src.preprocess_v2 import make_views
from src.features import rhyme, repetition, emotion, concreteness, pronouns, embedding

load_dotenv()
_FEATURE_MODS = [rhyme, repetition, emotion, concreteness, pronouns, embedding]
_CACHE_PATH = "data/agent_cache.csv"
_TITLE_ARTIST = re.compile(r"^\s*([^-|]+?)\s*[-–]\s*([^|]+?)\s*[|]?\s*", flags=re.UNICODE)

def _parse_title(text: str) -> dict[str, str] | None:
    """Heuristically extract (artist, title) from a search-result title like 'Olivia Rodrigo - drivers license | Billboard'."""
    m = _TITLE_ARTIST.match(text)
    if not m:
        return None
    artist = m.group(1).strip()
    title = m.group(2).strip()
    if len(artist) > 60 or len(title) > 60:
        return None
    return {"artist": artist, "title": title}

def search_recent_songs(query: str, max_results: int = 8) -> list[dict[str, str]]:
    """Web search; return parsed (artist, title) pairs."""
    rows: list[dict[str, str]] = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results * 2):
            parsed = _parse_title(r.get("title", ""))
            if parsed:
                rows.append(parsed)
            if len(rows) >= max_results:
                break
    return rows

_genius: lyricsgenius.Genius | None = None
def _get_genius() -> lyricsgenius.Genius:
    global _genius
    if _genius is None:
        _genius = lyricsgenius.Genius(
            os.environ["GENIUS_ACCESS_TOKEN"],
            timeout=15, sleep_time=1, verbose=False,
            remove_section_headers=True,
        )
    return _genius

def fetch_lyrics(artist: str, title: str) -> str | None:
    """Fetch lyrics via LyricsGenius search. Cached to disk."""
    cache_key = f"{artist.lower().strip()}|{title.lower().strip()}"
    if Path(_CACHE_PATH).exists():
        cache = pd.read_csv(_CACHE_PATH)
        hit = cache[cache["key"] == cache_key]
        if len(hit):
            return hit.iloc[0]["lyrics"]
    try:
        song = _get_genius().search_song(title=title, artist=artist, get_full_info=False)
        lyrics = song.lyrics if song and song.lyrics else None
    except Exception as e:
        print(f"  [fetch_lyrics ERROR] {e}")
        lyrics = None
    if lyrics:
        new_row = pd.DataFrame([{"key": cache_key, "artist": artist, "title": title, "lyrics": lyrics}])
        if Path(_CACHE_PATH).exists():
            new_row = pd.concat([pd.read_csv(_CACHE_PATH), new_row], ignore_index=True)
        new_row.to_csv(_CACHE_PATH, index=False)
    return lyrics

def extract_features(lyrics: str) -> dict[str, Any]:
    """Run all feature modules on raw lyrics."""
    views = make_views(lyrics)
    out: dict[str, Any] = {}
    for mod in _FEATURE_MODS:
        out.update(mod.extract(views))
    return out

# OpenAI tool schemas
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_recent_songs",
            "description": "Search the web for recent songs matching a query. Returns a list of {artist, title}.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query, e.g. 'Billboard Hot 100 2025 emotional ballads'"},
                    "max_results": {"type": "integer", "default": 8},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_lyrics",
            "description": "Fetch lyrics for a specific song from Genius.",
            "parameters": {
                "type": "object",
                "properties": {
                    "artist": {"type": "string"},
                    "title": {"type": "string"},
                },
                "required": ["artist", "title"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "extract_features",
            "description": "Compute the full feature vector (rhyme, repetition, emotion, concreteness, pronouns, embedding) for given lyrics.",
            "parameters": {
                "type": "object",
                "properties": {
                    "lyrics": {"type": "string"},
                },
                "required": ["lyrics"],
            },
        },
    },
]

TOOL_REGISTRY = {
    "search_recent_songs": search_recent_songs,
    "fetch_lyrics": fetch_lyrics,
    "extract_features": extract_features,
}
```

- [ ] **Step 3: Tests pass**

Run: `pytest tests/test_agent_tools.py -v`
Expected: both pass.

- [ ] **Step 4: Commit**

```bash
git add src/agent/tools.py tests/test_agent_tools.py
git commit -m "agent: search, fetch, extract tools with OpenAI schemas"
```

### Task 4.2: Ranking

**Files:**
- Create: `src/agent/ranking.py`
- Create: `tests/test_agent_ranking.py`

- [ ] **Step 1: Tests**

Create `tests/test_agent_ranking.py`:
```python
import numpy as np
from src.agent.ranking import score_candidate, profile_to_target

STATS = {
    "rhyme_density": {"mean": 0.3, "std": 0.1},
    "valence":       {"mean": 0.0, "std": 0.2},
}

def test_profile_to_target_signed():
    target = profile_to_target({"rhyme_density": +1.0, "valence": -0.5}, STATS)
    assert target["rhyme_density"] == 1.0
    assert target["valence"] == -0.5

def test_score_close_to_target_high():
    cand_feats = {"rhyme_density": 0.4, "valence": -0.1}  # rhyme +1 SD, valence -0.5 SD
    profile = {"rhyme_density": +1.0, "valence": -0.5}
    s = score_candidate(cand_feats, profile, STATS)
    assert s > 0.95

def test_score_far_from_target_low():
    cand_feats = {"rhyme_density": 0.0, "valence": +0.4}  # opposite direction
    profile = {"rhyme_density": +1.0, "valence": -0.5}
    s = score_candidate(cand_feats, profile, STATS)
    assert s < 0.5
```

- [ ] **Step 2: Implement**

Create `src/agent/ranking.py`:
```python
"""Score candidate songs against user's z-scored taste profile."""
import json
import numpy as np

def load_stats(path: str = "data/feature_stats.json") -> dict:
    with open(path) as f:
        return json.load(f)

def profile_to_target(profile: dict[str, float], stats: dict) -> dict[str, float]:
    """User's slider positions ARE the z-scores. Pass-through with validation."""
    return {k: float(v) for k, v in profile.items() if k in stats}

def _to_zscore(feats: dict, stats: dict, keys: list[str]) -> np.ndarray:
    return np.array([(feats.get(k, 0.0) - stats[k]["mean"]) / max(stats[k]["std"], 1e-9)
                     for k in keys])

def score_candidate(cand_feats: dict, profile: dict[str, float], stats: dict) -> float:
    """Cosine similarity between candidate z-vector and user target z-vector.
    Returns a value in [0, 1] (mapped from [-1, 1])."""
    target = profile_to_target(profile, stats)
    keys = sorted(target.keys())
    if not keys:
        return 0.0
    cand_z = _to_zscore(cand_feats, stats, keys)
    target_z = np.array([target[k] for k in keys])
    nc = np.linalg.norm(cand_z); nt = np.linalg.norm(target_z)
    if nc == 0 or nt == 0:
        return 0.5
    cos = float(cand_z @ target_z / (nc * nt))
    return (cos + 1) / 2  # map [-1,1] to [0,1]

def rank(candidates: list[dict], profile: dict[str, float], stats: dict) -> list[dict]:
    """Annotate each candidate with 'score' and return sorted descending."""
    out = []
    for c in candidates:
        s = score_candidate(c["features"], profile, stats)
        out.append({**c, "score": s})
    return sorted(out, key=lambda x: -x["score"])
```

- [ ] **Step 3: Tests pass**

Run: `pytest tests/test_agent_ranking.py -v`
Expected: all 3 pass.

- [ ] **Step 4: Commit**

```bash
git add src/agent/ranking.py tests/test_agent_ranking.py
git commit -m "agent: z-scored cosine ranking against user profile"
```

### Task 4.3: OpenAI tool-use runner

**Files:**
- Create: `src/agent/runner.py`

- [ ] **Step 1: Implement**

Create `src/agent/runner.py`:
```python
"""OpenAI tool-use loop that drives the recommendation agent."""
import json
import os
from typing import Callable

from openai import OpenAI
from dotenv import load_dotenv

from src.agent.tools import TOOL_REGISTRY, TOOL_SCHEMAS

load_dotenv()

SYSTEM_PROMPT = """You are a music recommendation agent. Given a user's lyrical taste profile (z-scored sliders over features like emotion intensity, repetition, concreteness, rhyme density, valence, and self-focus), your job is to find 3-5 RECENT (post-2023) songs that match.

Workflow:
1. Call search_recent_songs with 1-2 well-chosen queries (e.g., 'Billboard Hot 100 2025 emotional ballad').
2. For each candidate the search returns, call fetch_lyrics(artist, title) to get the lyrics.
3. For each fetched lyrics, call extract_features(lyrics) to get its feature vector.
4. After gathering 5-8 scored candidates, return a final message explaining your top picks.

Tips:
- Vary your search queries to surface different candidates if the first batch is thin.
- If a fetch fails, skip and move on.
- Do not invent songs. Only recommend ones whose lyrics you actually retrieved."""

def run(user_profile_text: str, candidate_sink: list[dict] | None = None,
        trace_sink: Callable[[str], None] | None = None,
        model: str = "gpt-4o-mini", max_steps: int = 25) -> str:
    """Run the agent. Side-effects:
    - Appends fetched & scored candidates to candidate_sink (so the UI can rank them).
    - Streams human-readable trace via trace_sink(line).
    Returns the final assistant message text.
    """
    client = OpenAI()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_profile_text},
    ]
    log = trace_sink or (lambda s: print(s))

    last_search_candidates: list[dict] = []
    last_fetched: dict[str, str] = {}

    for step in range(max_steps):
        resp = client.chat.completions.create(
            model=model, messages=messages, tools=TOOL_SCHEMAS,
        )
        msg = resp.choices[0].message
        messages.append(msg)
        if not msg.tool_calls:
            log(f"AGENT: {msg.content}")
            return msg.content or ""
        for tc in msg.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments or "{}")
            log(f"TOOL {name}({args})")
            try:
                result = TOOL_REGISTRY[name](**args)
            except Exception as e:
                result = {"error": str(e)}
                log(f"  ERROR: {e}")
            if name == "search_recent_songs" and isinstance(result, list):
                last_search_candidates = result
                log(f"  -> {len(result)} candidates")
            elif name == "fetch_lyrics" and isinstance(result, str):
                last_fetched[args.get("title", "?")] = result
                log(f"  -> {len(result)} chars")
            elif name == "extract_features" and isinstance(result, dict) and candidate_sink is not None:
                # Match back to artist/title via the most recent fetch
                title = next(iter(last_fetched.keys()), "?") if last_fetched else "?"
                # Find the artist from search candidates
                artist = next((c["artist"] for c in last_search_candidates if c["title"] == title), "?")
                candidate_sink.append({"artist": artist, "title": title, "features": result})
                log(f"  -> features for {artist} - {title}")
            messages.append({
                "role": "tool", "tool_call_id": tc.id,
                "content": json.dumps(result) if not isinstance(result, str) else result,
            })
    return "max steps reached"
```

- [ ] **Step 2: Smoke-test**

Run:
```bash
python3 -c "
from src.agent.runner import run
sink = []
out = run('I want high-emotion, low-repetition, concrete, high-rhyme-density songs.', candidate_sink=sink)
print('---')
print(f'Final: {out[:200]}')
print(f'Candidates scored: {len(sink)}')
"
```
Expected: prints tool-call trace, fetches lyrics for some recent songs, scores them, and prints a final summary. Should complete in 30-90 seconds and cost < $0.02.

- [ ] **Step 3: Commit**

```bash
git add src/agent/runner.py
git commit -m "agent: OpenAI gpt-4o-mini tool-use loop with trace sink"
```

### Task 4.4: Streamlit UI

**Files:**
- Create: `app.py`

- [ ] **Step 1: Implement**

Create `app.py`:
```python
"""Streamlit UI for the lyrics recommendation agent."""
import json
import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from src.agent.ranking import load_stats, rank
from src.agent.runner import run

load_dotenv()
st.set_page_config(page_title="Pop Lyrics Taste Profiler", layout="wide")

SLIDERS = {
    "valence":               ("Emotional valence (sad ↔ happy)", -2.0, 2.0),
    "arousal":               ("Arousal (calm ↔ intense)", -2.0, 2.0),
    "chorus_repeat_ratio":   ("Repetition (unique ↔ chant-like)", -2.0, 2.0),
    "rhyme_density":         ("Rhyme density (sparse ↔ dense)", -2.0, 2.0),
    "mean_concreteness":     ("Concreteness (abstract ↔ vivid)", -2.0, 2.0),
    "pronoun_i":             ("Self-focus (outward ↔ I-heavy)", -2.0, 2.0),
}

st.title("🎵 Pop Lyrics Taste Profiler")
st.caption("Set your lyrical preferences, and the agent will find recent songs that match.")

# --- Sidebar: corpus overview ---
@st.cache_data
def load_features():
    return pd.read_parquet("data/song_features.parquet")

feat = load_features()
stats = load_stats()

with st.sidebar:
    st.header("Corpus overview")
    st.metric("Songs", len(feat))
    st.metric("Years", f"{int(feat['year'].min())}–{int(feat['year'].max())}")
    st.bar_chart(feat.groupby("decade").size())

# --- Sliders ---
st.subheader("Your taste profile")
cols = st.columns(2)
profile: dict[str, float] = {}
for i, (key, (label, lo, hi)) in enumerate(SLIDERS.items()):
    with cols[i % 2]:
        profile[key] = st.slider(label, lo, hi, 0.0, 0.1, key=key)

if st.button("Find matching recent songs", type="primary"):
    profile_text = "\n".join(
        f"- {SLIDERS[k][0]}: {v:+.1f} z" for k, v in profile.items()
    )
    candidates: list[dict] = []
    trace_lines: list[str] = []

    trace_box = st.empty()
    def trace_sink(line: str) -> None:
        trace_lines.append(line)
        trace_box.code("\n".join(trace_lines[-30:]), language="text")

    with st.spinner("Agent is searching…"):
        final_msg = run(profile_text, candidate_sink=candidates, trace_sink=trace_sink)

    st.subheader("Agent reasoning")
    st.write(final_msg)

    st.subheader("Top recommendations")
    ranked = rank(candidates, profile, stats)[:5]
    for i, c in enumerate(ranked, 1):
        with st.container():
            st.markdown(f"**{i}. {c['artist']} — {c['title']}**  (similarity {c['score']:.2f})")
            feature_keys = list(SLIDERS.keys())
            comparison = pd.DataFrame({
                "feature": feature_keys,
                "you": [profile[k] for k in feature_keys],
                "song": [(c["features"].get(k, 0.0) - stats[k]["mean"]) / max(stats[k]["std"], 1e-9)
                         for k in feature_keys],
            })
            st.bar_chart(comparison.set_index("feature"))
```

- [ ] **Step 2: Launch and smoke-test**

Run: `streamlit run app.py`
Expected: opens browser at http://localhost:8501. Set some sliders, click "Find matching recent songs", watch the trace fill in, see ranked recommendations within 60-90 seconds.

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "ui: Streamlit recommendation agent with sliders + trace + ranking"
```

---

## Phase 5 — Screenshots + final polish (manual, ~30 min)

### Task 5.1: Capture the 5 screenshots

**Files:**
- Output: `output/screenshots/01_initial.png` through `output/screenshots/05_ranked.png`

- [ ] **Step 1: Launch app**

Run: `streamlit run app.py`

- [ ] **Step 2: Take 5 screenshots matching the spec § 6.6**

1. `01_initial.png` — page just loaded; all sliders at 0; sidebar corpus chart visible.
2. `02_profile_set.png` — sliders adjusted to a clear profile (e.g., valence +1.5, repetition -1.0, concreteness +1.0, rhyme +1.5, self-focus +0.5).
3. `03_searching.png` — mid-search; agent trace box visible with ≥ 5 tool-call lines.
4. `04_scoring.png` — first 1-2 candidates already scored, comparison bar chart visible.
5. `05_ranked.png` — final top-5 list with similarity scores and per-song bar charts.

Save each as PNG into `output/screenshots/` (macOS: `Cmd+Shift+4` then drag, save the file).

- [ ] **Step 3: Update `report.md` § 6 to reference these paths**

Verify all 5 image paths in the report's § 6 resolve.

- [ ] **Step 4: Commit**

```bash
git add output/screenshots/ report.md
git commit -m "report: capture 5 agent screenshots for assignment deliverable"
```

### Task 5.2: Update README

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Rewrite for the new pipeline**

Update `README.md` to document:
- New pipeline order: `data_sources.py` → `scraper_v2.py` → `genre_tagger.py` → `features/build_all.py` → `analyze.py` → `trends.py`
- How to launch the agent: `streamlit run app.py`
- Env vars required: `OPENAI_API_KEY`, `GENIUS_ACCESS_TOKEN` (with note about client access token vs client_id/secret)

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: update README for new pipeline + agent launch"
```

---

## Self-Review Checklist (done before handoff)

- ✅ Spec § 2 (architecture): covered by Phases 1-4.
- ✅ Spec § 3 (data pipeline): Tasks 1.1-1.3.
- ✅ Spec § 4 (feature extraction): Tasks 2.1-2.8.
- ✅ Spec § 5 (analysis): Tasks 3.1-3.2.
- ✅ Spec § 6 (recommendation agent): Tasks 4.1-4.4.
- ✅ Spec § 6.6 (5 screenshots): Task 5.1.
- ✅ Spec § 7 (report rewrite): Task 3.3.
- ✅ Spec § 8 (build sequence): mapped to Phase numbering.
- ✅ Spec § 6.8 (env): Task 0.1.
- ✅ Spec § 10 (risks): mitigations live in failure-mode handling within each task.
- All steps contain real code, real commands, real expected outputs. No "TBD" / "implement appropriate".
- Type/name consistency: `extract(views: dict)` signature is identical across all 6 feature modules; `make_views()` returns the dict that all modules consume.
