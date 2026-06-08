# Expand the 2020s corpus & refresh analysis

**Date:** 2026-06-08
**Branch:** fix/agent-lyrics-lrclib

## Problem

The decade classifier in `src/analyze.py` reports a very strong one-vs-rest
AUC for the 2020s, but `data/lyrics_full.csv` holds only ~140 2020s songs
versus 456–952 for every other decade. The small sample makes the result easy
to overstate (see the `analysis-caveats` memory). We want to raise the 2020s
count to a comparable size and re-test whether the signal survives.

## Goal

Expand the 2020s slice of `data/lyrics_full.csv` from ~140 to ~450–540 songs
(targeting ~100/year for 2020–2025), re-run the full pipeline, and report the
before/after 2020s AUC so the caveat can be softened or kept based on evidence.

## Design

### 1. Data — `data/raw/billboard_2016_2025.csv`

New reviewable data file with columns `artist,title,year,peak`, holding the
Billboard Year-End Hot 100 for 2020–2025 (~100/year) plus the existing
2016–2019 entries lifted out of `scraper_v2.py`. Compiled by hand from the
Year-End charts. Deeper-cut titles that are mis-named simply miss the lyric
lookup and are logged — a missing song, never a wrong one.

### 2. Scraper — `src/scraper_v2.py`

- Replace the inline `SONG_LIST` constant with `load_song_list(csv_path)` that
  reads the new CSV into the same `(artist, title, year, peak)` tuples.
- Rewrite `fetch_one` to call **lrclib first** (key-free `lrclib.net/api/get`
  by artist + clean title), falling back to a Genius search only on a miss.
  This keeps re-runs off the Genius rate limit and matches how lyrics are
  fetched elsewhere in the repo (`src/agent/tools.py`).
- `merge_and_save` is otherwise unchanged: walkerkq (1964–2015) is concatenated
  first so older `song_id`s stay stable, dedup on lowercased `artist|title`,
  positional `song_id` reassigned.

### 3. Regenerate keyed artifacts (in order)

Because `song_id` is positional, anything keyed by it must be rebuilt:

1. `data/genre_tags.csv` — **delete + full regen** via `src/genre_tagger.py`
   (resume-by-id would misalign shifted ids). ~6 min, trivial gpt-4o-mini cost.
2. `data/song_features.parquet` — `src/features/build_all.py` (local
   Sentence-BERT embeddings, no API cost).
3. `output/cv_results.json` — `src/analyze.py` → new per-decade AUC.

### 4. Refresh presentation layer

`src/preprocess.py` → `data/lyrics_processed.csv`, then tf-idf
(`scripts/plot_tfidf.py`), `src/visualize.py` (wordclouds), `src/trends.py`.

### 5. Report

Capture before/after 2020s AUC (stratified + artist-grouped) and n. Update the
small-sample caveat in `report.md` and the `analysis-caveats` memory to reflect
the new n and whether the signal held.

## Error handling / risk

- lrclib/Genius misses are expected and logged; if the 2020s lands below ~300,
  surface it rather than silently proceeding.
- Caveat wording is driven by the actual new n and AUC, not assumed.
- Corpus is built locally (Genius text scraping works off datacenter IPs), then
  committed; the HF Space only reads the resulting CSV.

## Verification

- Decade counts before/after.
- 2020s AUC before/after printed side by side (stratified + grouped).
- Spot-check 3–5 newly fetched lyrics for correct title/artist.
