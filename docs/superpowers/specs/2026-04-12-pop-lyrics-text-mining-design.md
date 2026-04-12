# Pop Lyrics Text Mining — Design Spec

## Objective

Explore whether lyrical styles in Billboard top 100 pop music have changed across decades (1990s, 2000s, 2010s, 2020s) using TF-IDF features, word clouds, and classification.

## Dataset

- **Source:** AZLyrics (web scraping)
- **Language:** English
- **Scope:** 25 songs per decade, 100 songs total across 4 decades
- **Artist selection:** Whatever is easiest to collect from Billboard top 100 lists per decade

## Project Structure

```
TXTmining-lyrics/
├── docs/                        # project guide and specs
├── data/
│   └── lyrics.csv               # scraped lyrics with metadata
├── src/
│   ├── scraper.py               # AZLyrics scraping logic
│   ├── preprocess.py            # tokenization, stopword removal, dedup
│   ├── features.py              # TF-IDF vectorization
│   ├── visualize.py             # word clouds per decade
│   └── classify.py              # Logistic Regression, ROC/AUC
├── output/
│   ├── wordclouds/              # generated word cloud images
│   └── roc_curve.png            # ROC curve plot
├── requirements.txt
└── README.md
```

## Data Schema

### `data/lyrics.csv` (after scraping)

| Column | Type   | Example                      |
|--------|--------|------------------------------|
| title  | str    | "Shape of You"               |
| artist | str    | "Ed Sheeran"                 |
| decade | str    | "2010s"                      |
| year   | int    | 2017                         |
| lyrics | str    | "The club isn't the best..." |

### `data/lyrics_processed.csv` (after preprocessing)

Adds one column:

| Column       | Type | Description                                         |
|--------------|------|-----------------------------------------------------|
| lyrics_clean | str  | Lowercased, stopwords removed, chorus deduplicated  |

## Pipeline

Scripts run sequentially. Each reads from and writes to `data/` or `output/`.

### 1. `src/scraper.py`

- Contains a predefined list of (artist, song, year, decade) tuples curated from Billboard top 100 lists
- Fetches lyrics from AZLyrics with polite delays between requests to avoid rate limiting
- Saves raw results to `data/lyrics.csv`

### 2. `src/preprocess.py`

- Reads `data/lyrics.csv`
- Lowercases text, removes punctuation and special characters
- Removes English stopwords (via NLTK or scikit-learn's built-in list)
- Sentence-level deduplication to reduce chorus repetition
- Writes `data/lyrics_processed.csv`

### 3. `src/features.py`

- Reads `data/lyrics_processed.csv`
- Builds TF-IDF vectors using scikit-learn's `TfidfVectorizer`
- Saves TF-IDF matrix and feature names for downstream use (pickle or sparse matrix)

### 4. `src/visualize.py`

- Generates one word cloud per decade from TF-IDF weights
- Saves images to `output/wordclouds/`

### 5. `src/classify.py`

- Logistic Regression (one-vs-rest for 4 decades)
- Train/test split
- Outputs ROC curves per class and AUC scores
- Saves plot to `output/roc_curve.png`

## Dependencies

- `requests` + `beautifulsoup4` — scraping
- `pandas` — data handling
- `scikit-learn` — TF-IDF, Logistic Regression, metrics
- `nltk` — stopwords
- `wordcloud` + `matplotlib` — visualization
- `joblib` or `pickle` — model/matrix persistence

## Expected Outputs

- Word clouds highlighting representative words per decade
- TF-IDF-based analysis of distinctive lyrical features
- Classification results (accuracy, ROC curves, AUC scores)
- Insights into how pop music language reflects cultural and stylistic shifts
