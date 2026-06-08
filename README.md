---
title: Pop Lyrics Taste Profiler
emoji: 🎵
colorFrom: gray
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
short_description: Set your taste profile; agent finds recent matching songs.
---

# Pop Lyrics Taste Profiler

A Streamlit app that lets you set a lyrical taste profile (6 sliders over emotion, repetition, concreteness, rhyme density, self-focus) and have an OpenAI-powered agent find recent Billboard songs that match — by live-fetching lyrics from Genius, scoring them on the same feature vector, and ranking by cosine similarity to your profile.

Built on a 4,869-song corpus spanning 1965–2025 (Billboard Year-End Hot 100 via `walkerkq/musiclyrics` + a 2016–2025 supplemental scrape).

## Two tabs

- **🎧 Recommend** — adjust sliders, click _Find matching recent songs_, watch the agent search Genius, fetch lyrics, extract features, and rank.
- **📊 Analyze** — per-decade style profiles, cross-decade heatmap, year-by-year trend plots, and the classifier result (38% accuracy, no artist leakage).

## Required secrets (set in Space settings)

| Name | Get it from |
|---|---|
| `OPENAI_API_KEY` | https://platform.openai.com/api-keys (used for the agent's tool-use loop with `gpt-4o-mini`) |
| `GENIUS_ACCESS_TOKEN` | https://genius.com/api-clients — click **Generate Access Token** on your client (this is NOT the same as `client_id` / `client_secret`) |

## Local development

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python3 -c "import nltk; nltk.download('stopwords'); nltk.download('cmudict')"
cp .env.example .env  # then fill in OPENAI_API_KEY and GENIUS_ACCESS_TOKEN
streamlit run app.py
```

## Re-running the offline pipeline (optional)

```bash
python3 src/data_sources.py             # downloads walkerkq + Brysbaert
python3 src/scraper_v2.py               # 2016–2025 Genius scrape (~10 min)
python3 src/genre_tagger.py             # genre labels via OpenAI batch (~5 min)
python3 -m src.features.build_all       # 7 feature modules over all songs (~5 min)
python3 -m src.analyze                  # CV + ROC
python3 -m src.trends                   # 22 per-feature trend plots
```

## Architecture & methodology

See `docs/superpowers/specs/2026-05-25-lyrics-recommendation-agent-design.md` for the full design spec, and `report.md` for the analysis writeup.
