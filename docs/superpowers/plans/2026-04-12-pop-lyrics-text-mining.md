# Pop Lyrics Text Mining Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a text mining pipeline that scrapes Billboard pop lyrics from AZLyrics across four decades and uses TF-IDF, word clouds, and Logistic Regression to detect stylistic changes over time.

**Architecture:** Sequential pipeline of five scripts in `src/`. Each script reads from `data/` or a prior output, processes it, and writes results to `data/` or `output/`. No shared state between scripts beyond files on disk.

**Tech Stack:** Python 3, requests, beautifulsoup4, pandas, scikit-learn, nltk, wordcloud, matplotlib

---

## File Map

| File | Responsibility |
|------|----------------|
| `requirements.txt` | Pin all dependencies |
| `src/scraper.py` | Hardcoded song list + AZLyrics fetcher → `data/lyrics.csv` |
| `src/preprocess.py` | Clean text, remove stopwords, deduplicate choruses → `data/lyrics_processed.csv` |
| `src/features.py` | TF-IDF vectorization → `data/tfidf_matrix.pkl`, `data/tfidf_vocab.pkl` |
| `src/visualize.py` | Word clouds per decade → `output/wordclouds/*.png` |
| `src/classify.py` | Logistic Regression + ROC/AUC → `output/roc_curve.png`, prints metrics |
| `tests/test_preprocess.py` | Tests for preprocessing functions |
| `tests/test_features.py` | Tests for TF-IDF feature extraction |
| `tests/test_classify.py` | Tests for classification pipeline |

---

### Task 1: Project Setup

**Files:**
- Create: `requirements.txt`
- Create: `src/__init__.py` (empty)
- Create: `tests/__init__.py` (empty)
- Create: `data/.gitkeep`
- Create: `output/wordclouds/.gitkeep`

- [ ] **Step 1: Create requirements.txt**

```
requests==2.32.3
beautifulsoup4==4.13.3
pandas==2.2.3
scikit-learn==1.6.1
nltk==3.9.1
wordcloud==1.9.4
matplotlib==3.10.1
joblib==1.4.2
pytest==8.3.4
```

- [ ] **Step 2: Create directory structure and placeholder files**

```bash
mkdir -p src tests data output/wordclouds
touch src/__init__.py tests/__init__.py data/.gitkeep output/wordclouds/.gitkeep
```

- [ ] **Step 3: Create virtual environment and install dependencies**

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

- [ ] **Step 4: Download NLTK stopwords**

```bash
python3 -c "import nltk; nltk.download('stopwords')"
```

- [ ] **Step 5: Verify setup**

```bash
python3 -c "import requests, bs4, pandas, sklearn, nltk, wordcloud, matplotlib; print('All imports OK')"
```

Expected: `All imports OK`

- [ ] **Step 6: Commit**

```bash
git add requirements.txt src/__init__.py tests/__init__.py data/.gitkeep output/wordclouds/.gitkeep
git commit -m "feat: project setup with dependencies and directory structure"
```

---

### Task 2: Scraper

**Files:**
- Create: `src/scraper.py`

- [ ] **Step 1: Create `src/scraper.py` with song list and scraping logic**

The script contains a hardcoded `SONGS` list of 100 entries (25 per decade) and a function to fetch lyrics from AZLyrics.

```python
"""Scrape pop song lyrics from AZLyrics."""

import re
import time
import random
import requests
import pandas as pd

SONGS = [
    # 1990s
    ("Baby One More Time", "Britney Spears", 1999, "1990s"),
    ("Wannabe", "Spice Girls", 1996, "1990s"),
    ("No Scrubs", "TLC", 1999, "1990s"),
    ("Waterfalls", "TLC", 1995, "1990s"),
    ("I Will Always Love You", "Whitney Houston", 1992, "1990s"),
    ("Creep", "TLC", 1994, "1990s"),
    ("MMMBop", "Hanson", 1997, "1990s"),
    ("Livin La Vida Loca", "Ricky Martin", 1999, "1990s"),
    ("Genie In A Bottle", "Christina Aguilera", 1999, "1990s"),
    ("I Want It That Way", "Backstreet Boys", 1999, "1990s"),
    ("Un-Break My Heart", "Toni Braxton", 1996, "1990s"),
    ("Believe", "Cher", 1998, "1990s"),
    ("Kiss From A Rose", "Seal", 1995, "1990s"),
    ("End Of The Road", "Boyz II Men", 1992, "1990s"),
    ("Torn", "Natalie Imbruglia", 1997, "1990s"),
    ("Iris", "Goo Goo Dolls", 1998, "1990s"),
    ("Losing My Religion", "R.E.M.", 1991, "1990s"),
    ("Smells Like Teen Spirit", "Nirvana", 1991, "1990s"),
    ("Black Or White", "Michael Jackson", 1991, "1990s"),
    ("Gangsta's Paradise", "Coolio", 1995, "1990s"),
    ("Macarena", "Los Del Rio", 1996, "1990s"),
    ("Wonderwall", "Oasis", 1995, "1990s"),
    ("Return Of The Mack", "Mark Morrison", 1996, "1990s"),
    ("Semi-Charmed Life", "Third Eye Blind", 1997, "1990s"),
    ("Smooth", "Santana", 1999, "1990s"),

    # 2000s
    ("Crazy In Love", "Beyonce", 2003, "2000s"),
    ("In Da Club", "50 Cent", 2003, "2000s"),
    ("Hey Ya", "Outkast", 2003, "2000s"),
    ("Since U Been Gone", "Kelly Clarkson", 2004, "2000s"),
    ("Toxic", "Britney Spears", 2004, "2000s"),
    ("Yeah", "Usher", 2004, "2000s"),
    ("Umbrella", "Rihanna", 2007, "2000s"),
    ("Hips Don't Lie", "Shakira", 2006, "2000s"),
    ("Poker Face", "Lady Gaga", 2008, "2000s"),
    ("Single Ladies", "Beyonce", 2008, "2000s"),
    ("Beautiful Day", "U2", 2000, "2000s"),
    ("Lose Yourself", "Eminem", 2002, "2000s"),
    ("Complicated", "Avril Lavigne", 2002, "2000s"),
    ("Mr. Brightside", "The Killers", 2004, "2000s"),
    ("Gold Digger", "Kanye West", 2005, "2000s"),
    ("Hollaback Girl", "Gwen Stefani", 2005, "2000s"),
    ("Rehab", "Amy Winehouse", 2006, "2000s"),
    ("Bleeding Love", "Leona Lewis", 2007, "2000s"),
    ("Viva La Vida", "Coldplay", 2008, "2000s"),
    ("I Gotta Feeling", "Black Eyed Peas", 2009, "2000s"),
    ("Boom Boom Pow", "Black Eyed Peas", 2009, "2000s"),
    ("Love Story", "Taylor Swift", 2008, "2000s"),
    ("Low", "Flo Rida", 2007, "2000s"),
    ("Irreplaceable", "Beyonce", 2006, "2000s"),
    ("SexyBack", "Justin Timberlake", 2006, "2000s"),

    # 2010s
    ("Rolling In The Deep", "Adele", 2010, "2010s"),
    ("Someone Like You", "Adele", 2011, "2010s"),
    ("Call Me Maybe", "Carly Rae Jepsen", 2012, "2010s"),
    ("Happy", "Pharrell Williams", 2013, "2010s"),
    ("Uptown Funk", "Mark Ronson", 2014, "2010s"),
    ("Shape Of You", "Ed Sheeran", 2017, "2010s"),
    ("Despacito", "Luis Fonsi", 2017, "2010s"),
    ("Old Town Road", "Lil Nas X", 2019, "2010s"),
    ("Closer", "The Chainsmokers", 2016, "2010s"),
    ("Bad Guy", "Billie Eilish", 2019, "2010s"),
    ("Blank Space", "Taylor Swift", 2014, "2010s"),
    ("Shake It Off", "Taylor Swift", 2014, "2010s"),
    ("Thinking Out Loud", "Ed Sheeran", 2014, "2010s"),
    ("Royals", "Lorde", 2013, "2010s"),
    ("Wrecking Ball", "Miley Cyrus", 2013, "2010s"),
    ("Lean On", "Major Lazer", 2015, "2010s"),
    ("Sorry", "Justin Bieber", 2015, "2010s"),
    ("Havana", "Camila Cabello", 2017, "2010s"),
    ("God's Plan", "Drake", 2018, "2010s"),
    ("Thank U Next", "Ariana Grande", 2018, "2010s"),
    ("Shallow", "Lady Gaga", 2018, "2010s"),
    ("Sunflower", "Post Malone", 2018, "2010s"),
    ("Blinding Lights", "The Weeknd", 2019, "2010s"),
    ("Sicko Mode", "Travis Scott", 2018, "2010s"),
    ("7 Rings", "Ariana Grande", 2019, "2010s"),

    # 2020s
    ("Levitating", "Dua Lipa", 2020, "2020s"),
    ("Watermelon Sugar", "Harry Styles", 2020, "2020s"),
    ("drivers license", "Olivia Rodrigo", 2021, "2020s"),
    ("good 4 u", "Olivia Rodrigo", 2021, "2020s"),
    ("Stay", "The Kid Laroi", 2021, "2020s"),
    ("As It Was", "Harry Styles", 2022, "2020s"),
    ("Anti-Hero", "Taylor Swift", 2022, "2020s"),
    ("About Damn Time", "Lizzo", 2022, "2020s"),
    ("Heat Waves", "Glass Animals", 2022, "2020s"),
    ("Flowers", "Miley Cyrus", 2023, "2020s"),
    ("Kill Bill", "SZA", 2023, "2020s"),
    ("Vampire", "Olivia Rodrigo", 2023, "2020s"),
    ("Cruel Summer", "Taylor Swift", 2023, "2020s"),
    ("Last Night", "Morgan Wallen", 2023, "2020s"),
    ("Paint The Town Red", "Doja Cat", 2023, "2020s"),
    ("Espresso", "Sabrina Carpenter", 2024, "2020s"),
    ("Fortnight", "Taylor Swift", 2024, "2020s"),
    ("Beautiful Things", "Benson Boone", 2024, "2020s"),
    ("Lose Control", "Teddy Swims", 2024, "2020s"),
    ("Birds Of A Feather", "Billie Eilish", 2024, "2020s"),
    ("A Bar Song", "Shaboozey", 2024, "2020s"),
    ("Not Like Us", "Kendrick Lamar", 2024, "2020s"),
    ("Die With A Smile", "Lady Gaga", 2024, "2020s"),
    ("Taste", "Sabrina Carpenter", 2024, "2020s"),
    ("APT", "Rose", 2024, "2020s"),
]


def _make_url(artist: str, title: str) -> str:
    """Build AZLyrics URL from artist and title.

    AZLyrics URL pattern: azlyrics.com/lyrics/artistname/songtitle.html
    - Artist and title are lowercased
    - Spaces, punctuation, and special characters are stripped
    - Leading "the " is removed from artist names
    """
    def clean(s: str) -> str:
        s = s.lower()
        s = re.sub(r"^the ", "", s)
        s = re.sub(r"[^a-z0-9]", "", s)
        return s

    return f"https://www.azlyrics.com/lyrics/{clean(artist)}/{clean(title)}.html"


def fetch_lyrics(artist: str, title: str) -> str | None:
    """Fetch lyrics text for a single song from AZLyrics.

    Returns the lyrics string or None if the request fails.
    """
    url = _make_url(artist, title)
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/120.0.0.0 Safari/537.36"
    }
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  FAILED: {e}")
        return None

    from bs4 import BeautifulSoup
    soup = BeautifulSoup(resp.text, "html.parser")

    # AZLyrics stores lyrics in a <div> that has no class/id,
    # located after the comment "Usage of azlyrics.com content..."
    divs = soup.find_all("div", class_=False, id_=False)
    for div in divs:
        text = div.get_text(strip=True)
        if len(text) > 200:
            return div.get_text("\n", strip=True)
    return None


def main():
    """Scrape all songs and save to data/lyrics.csv."""
    rows = []
    for i, (title, artist, year, decade) in enumerate(SONGS):
        print(f"[{i+1}/{len(SONGS)}] {artist} - {title}")
        lyrics = fetch_lyrics(artist, title)
        if lyrics:
            rows.append({
                "title": title,
                "artist": artist,
                "year": year,
                "decade": decade,
                "lyrics": lyrics,
            })
            print(f"  OK ({len(lyrics)} chars)")
        else:
            print("  SKIPPED")
        # Polite delay: 10-20 seconds between requests
        delay = random.uniform(10, 20)
        print(f"  Waiting {delay:.0f}s...")
        time.sleep(delay)

    df = pd.DataFrame(rows)
    df.to_csv("data/lyrics.csv", index=False)
    print(f"\nDone. Saved {len(df)} songs to data/lyrics.csv")
    print(f"Per decade: {df.groupby('decade').size().to_dict()}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the scraper**

```bash
cd /Users/willin/Gitub-local/TXTmining-lyrics
source venv/bin/activate
python3 src/scraper.py
```

Expected: Progress output for each song. Some songs may fail (AZLyrics may block or URL may not match). At least 15+ songs per decade is acceptable. The script takes ~20-30 minutes due to polite delays.

- [ ] **Step 3: Verify output**

```bash
python3 -c "import pandas as pd; df = pd.read_csv('data/lyrics.csv'); print(df.shape); print(df.groupby('decade').size())"
```

Expected: ~80-100 rows, roughly balanced across decades.

- [ ] **Step 4: Commit**

```bash
git add src/scraper.py
git commit -m "feat: add AZLyrics scraper with Billboard song list"
```

Note: Do NOT commit `data/lyrics.csv` — it contains scraped content.

---

### Task 3: Preprocessing

**Files:**
- Create: `src/preprocess.py`
- Create: `tests/test_preprocess.py`

- [ ] **Step 1: Write failing tests for preprocessing functions**

Create `tests/test_preprocess.py`:

```python
"""Tests for preprocessing functions."""

from src.preprocess import clean_text, deduplicate_lines


def test_clean_text_lowercases():
    assert "hello world" in clean_text("Hello World")


def test_clean_text_removes_punctuation():
    result = clean_text("hello, world! it's a test.")
    assert "," not in result
    assert "!" not in result


def test_clean_text_removes_stopwords():
    result = clean_text("I am the best in the world")
    assert "the" not in result.split()
    assert "am" not in result.split()
    assert "best" in result.split()
    assert "world" in result.split()


def test_deduplicate_lines_removes_repeated_lines():
    text = "I love you\nYeah yeah\nI love you\nSo much\nYeah yeah"
    result = deduplicate_lines(text)
    lines = result.strip().split("\n")
    assert lines.count("I love you") == 1
    assert lines.count("Yeah yeah") == 1
    assert "So much" in lines


def test_deduplicate_lines_preserves_order():
    text = "first\nsecond\nfirst\nthird"
    result = deduplicate_lines(text)
    lines = result.strip().split("\n")
    assert lines == ["first", "second", "third"]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
source venv/bin/activate
python3 -m pytest tests/test_preprocess.py -v
```

Expected: FAIL with `ModuleNotFoundError` or `ImportError`.

- [ ] **Step 3: Implement `src/preprocess.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python3 -m pytest tests/test_preprocess.py -v
```

Expected: All 5 tests PASS.

- [ ] **Step 5: Run preprocessing on scraped data**

```bash
python3 src/preprocess.py
```

Expected: Output showing number of processed songs and a sample.

- [ ] **Step 6: Commit**

```bash
git add src/preprocess.py tests/test_preprocess.py
git commit -m "feat: add lyrics preprocessing with stopword removal and chorus dedup"
```

---

### Task 4: TF-IDF Feature Extraction

**Files:**
- Create: `src/features.py`
- Create: `tests/test_features.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_features.py`:

```python
"""Tests for TF-IDF feature extraction."""

import pandas as pd
from src.features import build_tfidf


def test_build_tfidf_returns_correct_shape():
    df = pd.DataFrame({
        "lyrics_clean": ["love baby tonight", "money car fast", "feel good sunshine"],
        "decade": ["1990s", "2000s", "2010s"],
    })
    matrix, vocab, vectorizer = build_tfidf(df["lyrics_clean"])
    assert matrix.shape[0] == 3
    assert matrix.shape[1] == len(vocab)
    assert len(vocab) > 0


def test_build_tfidf_vocab_contains_words():
    df = pd.DataFrame({
        "lyrics_clean": ["love baby tonight", "money car fast"],
    })
    matrix, vocab, vectorizer = build_tfidf(df["lyrics_clean"])
    assert "love" in vocab
    assert "money" in vocab


def test_build_tfidf_matrix_is_sparse():
    from scipy.sparse import issparse
    df = pd.DataFrame({
        "lyrics_clean": ["hello world", "foo bar baz"],
    })
    matrix, vocab, vectorizer = build_tfidf(df["lyrics_clean"])
    assert issparse(matrix)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python3 -m pytest tests/test_features.py -v
```

Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement `src/features.py`**

```python
"""Build TF-IDF features from preprocessed lyrics."""

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def build_tfidf(lyrics: pd.Series, max_features: int = 5000):
    """Build TF-IDF matrix from a series of cleaned lyrics.

    Returns (sparse_matrix, feature_names_list, fitted_vectorizer).
    """
    vectorizer = TfidfVectorizer(max_features=max_features)
    matrix = vectorizer.fit_transform(lyrics)
    vocab = vectorizer.get_feature_names_out().tolist()
    return matrix, vocab, vectorizer


def main():
    """Read processed lyrics, build TF-IDF, save artifacts."""
    df = pd.read_csv("data/lyrics_processed.csv")
    matrix, vocab, vectorizer = build_tfidf(df["lyrics_clean"])

    joblib.dump(matrix, "data/tfidf_matrix.pkl")
    joblib.dump(vocab, "data/tfidf_vocab.pkl")
    joblib.dump(vectorizer, "data/tfidf_vectorizer.pkl")

    print(f"TF-IDF matrix shape: {matrix.shape}")
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Saved to data/tfidf_matrix.pkl, data/tfidf_vocab.pkl, data/tfidf_vectorizer.pkl")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python3 -m pytest tests/test_features.py -v
```

Expected: All 3 tests PASS.

- [ ] **Step 5: Run feature extraction on preprocessed data**

```bash
python3 src/features.py
```

Expected: Output showing matrix shape (e.g., `(85, 5000)`) and vocab size.

- [ ] **Step 6: Commit**

```bash
git add src/features.py tests/test_features.py
git commit -m "feat: add TF-IDF feature extraction"
```

---

### Task 5: Word Cloud Visualization

**Files:**
- Create: `src/visualize.py`

- [ ] **Step 1: Implement `src/visualize.py`**

```python
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
```

- [ ] **Step 2: Run visualization**

```bash
python3 src/visualize.py
```

Expected: Four PNG files in `output/wordclouds/`: `1990s.png`, `2000s.png`, `2010s.png`, `2020s.png`.

- [ ] **Step 3: Verify output files exist**

```bash
ls -la output/wordclouds/
```

Expected: Four `.png` files, each a few hundred KB.

- [ ] **Step 4: Commit**

```bash
git add src/visualize.py
git commit -m "feat: add word cloud visualization per decade"
```

---

### Task 6: Classification and Evaluation

**Files:**
- Create: `src/classify.py`
- Create: `tests/test_classify.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_classify.py`:

```python
"""Tests for classification pipeline."""

import numpy as np
from scipy.sparse import csr_matrix
from src.classify import train_and_evaluate


def test_train_and_evaluate_returns_metrics():
    # Synthetic data: 40 samples, 10 features, 4 classes
    np.random.seed(42)
    X = csr_matrix(np.random.rand(40, 10))
    y = np.array(["1990s"] * 10 + ["2000s"] * 10 + ["2010s"] * 10 + ["2020s"] * 10)
    metrics = train_and_evaluate(X, y, output_path=None)
    assert "accuracy" in metrics
    assert "auc_scores" in metrics
    assert isinstance(metrics["accuracy"], float)
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_train_and_evaluate_auc_scores_per_class():
    np.random.seed(42)
    X = csr_matrix(np.random.rand(40, 10))
    y = np.array(["1990s"] * 10 + ["2000s"] * 10 + ["2010s"] * 10 + ["2020s"] * 10)
    metrics = train_and_evaluate(X, y, output_path=None)
    assert len(metrics["auc_scores"]) == 4
    for label, score in metrics["auc_scores"].items():
        assert 0.0 <= score <= 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python3 -m pytest tests/test_classify.py -v
```

Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement `src/classify.py`**

```python
"""Train Logistic Regression to classify lyrics by decade, evaluate with ROC/AUC."""

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score, roc_curve, auc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def train_and_evaluate(X, y, output_path: str | None = "output/roc_curve.png"):
    """Train Logistic Regression and produce ROC curves.

    Args:
        X: TF-IDF feature matrix (sparse or dense).
        y: Array of decade labels.
        output_path: Path to save ROC curve plot. None to skip plotting.

    Returns:
        Dict with 'accuracy' and 'auc_scores' per class.
    """
    classes = sorted(np.unique(y))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # ROC/AUC (one-vs-rest)
    y_test_bin = label_binarize(y_test, classes=classes)
    y_score = model.predict_proba(X_test)

    auc_scores = {}
    if output_path is not None:
        plt.figure(figsize=(8, 6))

    for i, label in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        auc_scores[label] = roc_auc
        if output_path is not None:
            plt.plot(fpr, tpr, label=f"{label} (AUC = {roc_auc:.2f})")

    if output_path is not None:
        plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curves — Decade Classification")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"ROC curve saved to {output_path}")

    return {"accuracy": acc, "auc_scores": auc_scores}


def main():
    """Load features, train model, evaluate, and save results."""
    df = pd.read_csv("data/lyrics_processed.csv")
    matrix = joblib.load("data/tfidf_matrix.pkl")

    y = df["decade"].values
    metrics = train_and_evaluate(matrix, y)

    print(f"\nAccuracy: {metrics['accuracy']:.2%}")
    print("\nAUC Scores per Decade:")
    for label, score in metrics["auc_scores"].items():
        print(f"  {label}: {score:.3f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python3 -m pytest tests/test_classify.py -v
```

Expected: All 2 tests PASS.

- [ ] **Step 5: Run classification on real data**

```bash
python3 src/classify.py
```

Expected: Prints accuracy and per-decade AUC scores. Saves `output/roc_curve.png`.

- [ ] **Step 6: Commit**

```bash
git add src/classify.py tests/test_classify.py
git commit -m "feat: add Logistic Regression classifier with ROC/AUC evaluation"
```

---

### Task 7: README

**Files:**
- Create: `README.md`

- [ ] **Step 1: Create `README.md`**

```markdown
# Billboard Pop Lyrics Text Mining

Analyzing lyrical changes in Billboard top 100 pop songs across four decades (1990s–2020s) using TF-IDF and classification.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 -c "import nltk; nltk.download('stopwords')"
```

## Pipeline

Run each script in order:

```bash
python3 src/scraper.py        # Scrape lyrics from AZLyrics (~25 min)
python3 src/preprocess.py     # Clean and deduplicate
python3 src/features.py       # Build TF-IDF features
python3 src/visualize.py      # Generate word clouds
python3 src/classify.py       # Train classifier, produce ROC curves
```

## Output

- `output/wordclouds/` — word cloud per decade
- `output/roc_curve.png` — ROC curves for decade classification

## Tests

```bash
python3 -m pytest tests/ -v
```
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add README with setup and usage instructions"
```
