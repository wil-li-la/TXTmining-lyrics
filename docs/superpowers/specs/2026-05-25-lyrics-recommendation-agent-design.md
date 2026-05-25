# Lyrics Recommendation Agent — Design

**Date:** 2026-05-25
**Status:** Approved (pending user review of this written spec)
**Supersedes:** 2026-04-12-pop-lyrics-text-mining-design.md (extends rather than replaces)

## 1. Motivation

The current project (`report.md`) classifies Billboard pop songs by decade using TF-IDF + Logistic Regression on 95 songs. Three weaknesses make the result hard to defend:

1. **Tiny dataset (n=95)** with no cross-validation — the headline 34.5% accuracy is within sampling noise.
2. **Likely artist leakage** — the model may be learning artist identity (Beyoncé, Drake) rather than era characteristics.
3. **No clear utility** — "predict the decade" has no real-world use case.

This redesign addresses all three and adds the week's assignment requirement: a working agent with five screenshots demonstrating its use.

The unifying framing: characterize lyric style with rich features, validate the analysis with proper CV, then **close the loop with a recommendation agent** that takes a user's taste profile and finds *recent* songs matching it.

## 2. Architecture overview

Two halves of one system:

**Half A — Offline analysis pipeline** (produces artifacts used by the report and the agent):

```
raw lyrics (walkerkq 5,100 + 2016–2025 scrape ≈ 5,700 songs)
   ↓ preprocess.py    (raw + tokenized + clean views)
   ↓ src/features/    (TF-IDF, rhyme, repetition, EmoLex, concreteness, pronouns, SBERT)
   ↓ analyze.py       (Stratified 5-fold CV, GroupKFold by artist)
   → report.md (rewritten, 8 real citations anchored in Parada-Cabaleiro 2024)
   → data/song_features.parquet ← consumed by Half B
```

**Half B — Online recommendation agent** (the assignment deliverable):

```
Streamlit UI
   ↓ user adjusts taste-profile sliders over feature dimensions
Recommendation Agent (OpenAI gpt-4o-mini with tool use)
   ↓ tool: search_recent_songs   (ddgs web search wrapper)
   ↓ tool: fetch_lyrics          (LyricsGenius)
   ↓ tool: extract_features      (shared src/features/ module)
   ↓ deterministic Python ranking: cosine distance to user profile
   → top-5 recommendations with per-feature explanations
```

Key design points:
- The same `src/features/` module is shared between offline batch and the online agent's `extract_features` tool. One source of truth.
- `data/song_features.parquet` is the boundary artifact. Half B can be developed independently once Half A produces this file.
- During development we spin out three Claude Code subagents in parallel (scrape, feature, recommendation); in production it's one user-facing agent.

## 3. Data pipeline

### Sources

| Source | Coverage | Songs | Method |
|---|---|---|---|
| walkerkq/musiclyrics | Billboard Year-End Hot 100, 1965–2015 | ~5,100 | Download CSV directly from GitHub |
| Existing scrape | Billboard Hot 100, 2020–2025 | ~95 | Keep |
| New scrape | Billboard Hot 100, 2016–2019 + 2025 fill | ~500 | Reuse `src/scraper.py` with LyricsGenius |
| **Total** | 1965–2025 | **~5,700** | |

### Schema (`data/lyrics_full.csv`)

```
song_id | title | artist | year | decade | chart_position | source | lyrics_raw
```

### Genre tagging (new — enables optional genre filtering in the agent UI)

- **Primary:** MusicBrainz API query by `(artist, title)` — free, no auth, returns canonical genre tags.
- **Fallback:** LLM-assisted tagging (single Haiku call per uncovered song with a constrained label set: `pop / rap / r&b / rock / country / dance`) for the ~10–20% MusicBrainz misses.
- Cache to `data/genre_tags.csv` so we never re-query.

### Deduplication & sanity

- Deduplicate on `(artist_lower, title_lower)` — walkerkq + new scrape overlap at the 2015 boundary.
- Drop rows where `lyrics_raw < 100 chars` (instrumentals, scraping failures).
- Drop rows where `langdetect` flags > 50% non-English (Parada-Cabaleiro restricted to English; we match for comparability).

### Preprocessing change from current pipeline

**Stop sentence-level chorus dedup before feature extraction.** The current `preprocess.py` collapses repeated chorus lines, which destroys the repetition signal we now want to measure. Instead, preprocess produces three views:

- `lyrics_raw` — untouched, line-break-preserving. Used by rhyme, repetition.
- `lyrics_tokenized` — lowercased, punctuation stripped, **stopwords kept**. Used by pronouns.
- `lyrics_clean` — `lyrics_tokenized` minus stopwords + 3-char minimum filter. Used by TF-IDF, EmoLex, concreteness.

Each feature module declares which view it consumes.

### Failure modes

- Genius rate limit → exponential backoff in scraper, resume from last `song_id`.
- MusicBrainz miss → fall through to LLM tagger; if both fail, label `genre = "unknown"` (still usable downstream).
- Lyrics fetch returns instrumental marker → drop with logged reason.

## 4. Feature extraction

Seven feature modules under `src/features/` — TF-IDF (retained for the classifier in §5) plus six new families that populate `song_features.parquet`. Every module exposes `extract(lyrics_views: dict) → dict[str, float]` so offline batch and the agent's online tool consume identical code.

| Module | Measures | Library | Input view | Output dims |
|---|---|---|---|---|
| `tfidf.py` (refactor existing) | Vocabulary distinctiveness | sklearn `TfidfVectorizer(max_features=1000, min_df=5, max_df=0.9, ngram_range=(1,2))` | clean | sparse vector (separate from parquet; used by classifier only) |
| `rhyme.py` | End-rhyme density, internal-rhyme rate, mean syllables/line | `pronouncing` (CMU dict) | raw | 3 floats |
| `repetition.py` | Line-bigram entropy, chorus-repeat ratio, type-token ratio (MTLD) | stdlib + custom | raw | 3 floats |
| `emotion.py` | 8 NRC emotions + valence + arousal | `NRCLex` | clean | 10 floats (proportions, sum-normalized) |
| `concreteness.py` | Mean concreteness rating, % concrete-words (≥ 4.0) | Brysbaert 40k-word CSV (OSF download, joined manually) | clean | 2 floats |
| `pronouns.py` | I/me, you, we/us, they/them ratios | stdlib regex on tokens | tokenized | 4 floats |
| `embedding.py` | Semantic style | `sentence-transformers` `all-MiniLM-L6-v2`, mean-pooled | clean | 384-d dense vector |

### Output artifact — `data/song_features.parquet`

```
song_id | year | decade | genre |
  rhyme_density | internal_rhyme | mean_syllables_per_line |
  repetition_entropy | chorus_repeat_ratio | mtld |
  emo_anger | emo_anticipation | ... | valence | arousal |
  mean_concreteness | pct_concrete |
  pronoun_i | pronoun_you | pronoun_we | pronoun_they |
  embedding (list[float, 384])
```

~5,700 rows × ~410 columns. Parquet handles the embedding column cleanly.

### Parallelization

All 6 modules are pure functions of `lyrics_views`. Build with `joblib.Parallel(n_jobs=-1)`: one row at a time, each row computes all 6 in sequence. ~5,700 songs × ~50 ms/song ≈ 5 minutes on a laptop.

### Mapping to the agent's UI

The taste-profile picker exposes ~6–8 sliders mapped directly to a subset of these columns: emotion intensity, repetition, concreteness, rhyme density, valence, arousal, pronoun-self vs other. The agent computes cosine distance over the *z-scored* feature vector for candidate songs vs the user's slider settings.

## 5. Analysis upgrades

Replaces the current `src/classify.py` with `src/analyze.py`. Two CV runs, one comparison table.

### 5.1 Stratified 5-fold CV

Replaces the single 70/30 split. Report mean ± std accuracy and per-decade AUC. `sklearn.model_selection.cross_val_score` + `cross_val_predict`.

### 5.2 GroupKFold by artist

`sklearn.model_selection.GroupKFold(n_splits=5)` with `groups=artist`. No artist appears in both train and test. **Directly tests the artist-leakage critique** — if AUC drops materially under this split, the original model was learning artists, not eras.

### 5.3 Headline comparison table (the report's main methodological contribution)

| Eval | Accuracy | 2020s AUC | 1990s AUC | Reading |
|---|---|---|---|---|
| Old: 70/30 split, n=95 | 34.5% | 0.90 | 0.73 | (prior report) |
| New: Stratified 5-fold, n≈5,700 | ? | ? | ? | (real estimate) |
| New: GroupKFold by artist | ? | ? | ? | (leakage-corrected) |

Whatever the numbers, this table is the project's defensible analytical contribution.

### 5.4 Top-terms-per-decade

Keep the existing per-decade mean-TF-IDF method, but tighten `min_df=5` and `max_features=1000`. The "creep / macarena" noise was mostly a small-dataset artifact and should go away at n=5,700. If it doesn't, revisit Monroe et al. 2008 log-odds (omitted for now to control scope).

### 5.5 Explicitly cut

- Permutation test (at n=5,700, significance is a foregone conclusion; adds noise to the writeup).
- Monroe log-odds (only needed if §5.4 fix is insufficient).

## 6. Recommendation agent (assignment deliverable)

### 6.1 Stack

| Layer | Choice | Why |
|---|---|---|
| LLM | OpenAI Chat Completions API, `gpt-4o-mini` | User has no Anthropic API access; gpt-4o-mini supports tool use cheaply (~$0.01/session) |
| Web search | `ddgs` (DuckDuckGo Python wrapper) | Keyless, zero-setup; swap to Tavily if quality issues |
| Lyrics fetch | LyricsGenius | Reuse existing dependency |
| UI | Streamlit | Fastest path to sliders + charts + chat trace |
| Feature extraction | Same `src/features/` modules from §4 | One source of truth |

### 6.2 Tools

```
search_recent_songs(query: str) → list[{artist, title, year}]
    # ddgs query; LLM crafts like "Billboard Hot 100 2025 emotional ballads"

fetch_lyrics(artist: str, title: str) → str
    # LyricsGenius; caches to data/agent_cache.csv

extract_features(lyrics: str) → dict
    # shared src/features/ module; returns same dict shape as song_features.parquet
```

### 6.3 Ranking (deterministic, not LLM judgment)

1. Load corpus mean/std for each feature from `song_features.parquet`.
2. Convert user's slider position (−2 to +2) directly into a target z-score vector.
3. For each candidate: extract features → z-score using corpus stats → cosine similarity to target.
4. Return top 5 with per-feature breakdown.

LLM does the **search strategy** + **natural-language explanations**, not the scoring. This keeps recommendations reproducible.

### 6.4 OpenAI tool-use loop (code shape)

```python
client = OpenAI()  # OPENAI_API_KEY from env

tools = [
    {"type": "function", "function": {"name": "search_recent_songs", ...}},
    {"type": "function", "function": {"name": "fetch_lyrics", ...}},
    {"type": "function", "function": {"name": "extract_features", ...}},
]

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": user_profile_as_text},
]

while True:
    resp = client.chat.completions.create(
        model="gpt-4o-mini", messages=messages, tools=tools
    )
    msg = resp.choices[0].message
    messages.append(msg)
    if not msg.tool_calls:
        break
    for tc in msg.tool_calls:
        result = TOOL_REGISTRY[tc.function.name](**json.loads(tc.function.arguments))
        messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)})
```

### 6.5 Streamlit UI layout

```
┌──────────────────────────────────────────────────┐
│  Pop Lyrics Taste Profiler                       │
├──────────────────────────────────────────────────┤
│  [Corpus overview: feature distributions]        │
├──────────────────────────────────────────────────┤
│  Your taste:                                     │
│    Emotion intensity   [─────●──]  +1.2          │
│    Repetition          [──●──────] −0.8          │
│    Concreteness        [────●────]  0.0          │
│    Rhyme density       [──────●──] +1.5          │
│    Valence (positive)  [───●─────] −0.5          │
│    Self-focused        [─────●───] +0.7          │
│  [ Find matching recent songs ]                  │
├──────────────────────────────────────────────────┤
│  Agent trace:                                    │
│    🔍 Searched: "Billboard Hot 100 2025 high..." │
│    🎵 Fetched: Olivia Rodrigo — "drivers license"│
│    📊 Scored: similarity 0.87                    │
│    ...                                           │
├──────────────────────────────────────────────────┤
│  Top recommendations:                            │
│    1. Song A — 0.91   [feature radar chart]      │
│    2. Song B — 0.86   [feature radar chart]      │
│    3. Song C — 0.83   [feature radar chart]      │
└──────────────────────────────────────────────────┘
```

### 6.6 Five screenshots

1. **Initial state** — sliders at 0, corpus-overview chart showing decade feature distributions.
2. **Profile set** — sliders adjusted, user about to click "Find matches".
3. **Agent searching** — live tool-use trace: search query → candidates found → fetching lyrics.
4. **Per-song scoring** — one candidate with its feature radar overlaid on the user's target.
5. **Final ranked output** — top 5 recommendations with similarity scores and short LLM-generated explanations.

### 6.7 Failure modes

- LyricsGenius returns nothing → agent logs "couldn't fetch, skipping" in trace, moves to next candidate.
- Web search returns < 5 candidates → agent reformulates query and retries once.
- All candidates score below 0.5 similarity → return what we have + UI note ("low confidence; try widening your profile").

### 6.8 Environment

`OPENAI_API_KEY` and `GENIUS_ACCESS_TOKEN` (already used by current scraper).

## 7. Citations & report rewrite

`report.md` is rewritten end-to-end.

### 7.1 New structure

```
1. Introduction & related work        ← citations land here
2. Dataset                            ← walkerkq + scrape extension
3. Features                           ← 6 families, methods table
4. Analysis: decade classification    ← CV + GroupKFold table (§5)
5. Analysis: feature trends over time ← line plots per feature, 1965→2025
6. Recommendation agent               ← architecture + 5 screenshots
7. Limitations & future work
8. References
```

### 7.2 Citation map

| Paper | Lands in | Used for |
|---|---|---|
| **Parada-Cabaleiro et al. 2024** (*Sci Rep*) | §1, §5 | **Anchor citation** — partial replication of their lexical-complexity + repetition findings on Billboard-only |
| DeWall et al. 2011 (*Psych Aesthetics*) | §1, §5 | Template for pronoun / I-vs-we trend analysis |
| Pettijohn & Sacco 2009 (*J Lang Soc Psych*) | §1 | Prior work motivating decade-scale lyric trends |
| Brand et al. 2019 | §5 | Comparator for emotion-over-time trends |
| 2025 *Sci Rep* (societal crises) | §5, §7 | Recent extension; future-work hook for adding economic correlates |
| Pachet & Roy 2008 (*ISMIR*) | §1 | Frames the project's skeptical posture toward hit-song-science claims |
| Martín-Gutiérrez et al. 2023 (arXiv) | §1, §7 | Hit-prediction lyrics+audio benchmark — future-work pointer |
| Akhtar et al. 2025 (arXiv) | §3 | Methodological cite for SBERT-on-lyrics |

**8 real citations** — anchor + 7 supporting. More becomes padding.

### 7.3 New section: §5 feature trends over time

For each of the 6 feature families, plot mean by year (1965→2025) with a LOESS smoother. Two paragraphs of prose per family comparing what we observe to what the cited papers report. Even if §4's classifier is weak, these trend plots may show robust shifts (e.g., declining lexical diversity per Parada-Cabaleiro).

### 7.4 What gets cut from current report

- The single-split metrics in current §4 (replaced by §5's CV table).
- The cherry-picked "external → internal" narrative in current Finding #4 (replaced by §5 evidence-driven trends).
- The 2020s AUC=0.90 overclaim (the new CV numbers will tell a different, more honest story).

### 7.5 Target length

~3,500–4,500 words. Current report is ~1,800 — roughly 2× longer with real depth, not padding.

## 8. Build sequence

Three independent subagents during implementation (per the "spin out an agent per sub-task" requirement):

1. **scrape-agent** — fetches walkerkq, runs 2016–2019 + 2025 Genius scrape, merges to `data/lyrics_full.csv`, runs MusicBrainz + LLM genre tagger.
2. **feature-agent** — implements `src/features/{rhyme,repetition,emotion,concreteness,pronouns,embedding}.py`, builds `data/song_features.parquet`.
3. **analysis-agent** — implements `src/analyze.py` (CV + GroupKFold), produces the headline table and per-feature trend plots; updates `report.md` with the 8 citations.

Then a final foreground build for the **recommendation agent** (`src/agent/`) + Streamlit UI (`app.py`) — needs `song_features.parquet` from steps 1–2, so it's sequenced last.

## 9. Out of scope (deferred to future work)

- Audio-feature integration (Spotify API is closed to new apps as of Nov 2024; AcousticBrainz dump is an option but adds a multi-day data-engineering task).
- Hit-prediction modeling (chart position is in the data; predicting it is a separate project).
- Multi-language support (Parada-Cabaleiro restricted to English; we match).
- Monroe et al. 2008 log-odds for distinctive terms (only if §5.4 fix is insufficient).
- Permutation testing (significance is foregone at n=5,700).

## 10. Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| `ddgs` rate-limited or empty results | Medium | Swap to Tavily (1k free searches/month) |
| LyricsGenius scrape fails for many recent songs | Medium | Cache aggressively; degrade gracefully (skip + log) |
| MusicBrainz coverage poor for older songs | Medium | LLM fallback labelled; "unknown" genre still usable |
| GroupKFold reveals near-chance accuracy | High | This is itself the finding — own it in the report |
| Feature extraction takes >> 5 min at n=5,700 | Low | Profile; SBERT embedding is the likely bottleneck — batch it |
| Streamlit + OpenAI tool-use UX feels clunky | Medium | Hand-tune the system prompt; pre-record fallback screenshots if live demo fails |

## 11. Success criteria

- `data/song_features.parquet` exists with ≥ 5,500 rows × all feature columns populated.
- `report.md` rewritten with the headline CV table populated and 8 real citations.
- `streamlit run app.py` launches the agent UI; a user session end-to-end produces 5 recommendations in < 60 seconds.
- 5 screenshots captured matching §6.6.
- Total ≤ 7 days of build effort.
