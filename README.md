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
python3 src/scraper.py        # Scrape lyrics from Genius (~15 min)
python3 src/preprocess.py     # Clean and deduplicate
python3 src/features.py       # Build TF-IDF features
python3 src/visualize.py      # Generate word clouds
python3 src/classify.py       # Train classifier, produce ROC curves
```

## Output

- `output/wordclouds/` — word cloud per decade
- `output/roc_curve.png` — ROC curves for decade classification

