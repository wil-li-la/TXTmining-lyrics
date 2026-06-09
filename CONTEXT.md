# Pop Lyrics Taste Profiler

A text-mining project over ~5,200 pop songs (1965–2025) with two faces: an **Analyze** report on how lyrical style shifts across decades/genres, and a **Recommend** flow that matches a listener's stated lyrical taste to songs in the catalog.

## Language

**Taste Profile**:
A listener's set of slider positions, expressed directly as z-scores, describing the lyrical style they prefer (e.g. happy vs. sad, abstract vs. vivid).
_Avoid_: preferences, query, settings

**Recommendation**:
A song surfaced because it matches a Taste Profile. The result list is the top few Recommendations.
_Avoid_: suggestion, result, match (as a noun)

**Candidate Pool**:
The set of songs eligible to become Recommendations. Here it is the whole catalog (1965–2025), not just recent songs.
_Avoid_: corpus (that's the analysis dataset), search results

**Provenance-safe feature**:
A lyrical feature whose value is comparable across all eras because it does not depend on line structure — valence, arousal, concreteness, self-focus, and the sentence embedding. Only these drive cross-era Relevance.
_Avoid_: bag-of-words feature, reliable feature

**Line-based feature**:
A feature derived from line breaks — **rhyme density** and **repetition**. Untrustworthy for pre-2016 songs because the source data stripped line breaks, fixing these features near zero. Not used for cross-era Relevance.
_Avoid_: structural feature, prosody

**Recent song**:
A song from **2016 or later** — the cutoff where Line-based features become trustworthy (lyrics scraped fresh, line breaks intact).
_Avoid_: new song, current song

**Relevance**:
How well a candidate's Provenance-safe features match the Taste Profile (cosine similarity). The primary ranking signal.
_Avoid_: score (overloaded), similarity

**Diversity** (a.k.a. dedup):
Removing near-duplicate Recommendations — alternate mixes, clean edits, re-releases of the same song — using sentence-embedding closeness. Affects which candidates survive, never their Relevance.
_Avoid_: variety, novelty

## Flagged ambiguities

**"Agent"** — historically meant a live OpenAI tool-use loop that searched Genius and fetched lyrics on the fly. As of the recommender redesign (see `docs/adr/0001`), Recommendations come from deterministic retrieval over a precomputed feature index; the language model only writes the natural-language explanation. Use **Recommender** for the subsystem and reserve **Agent** for the retained live-search *fallback* path only.

## Example dialogue

> **Dev:** A user slides "rhyme density" to +2. Why isn't that changing the top results?
> **Expert:** Because rhyme is a Line-based feature, and the Candidate Pool is the whole catalog. For pre-2016 songs that feature is fake-zero, so we can't rank on it across eras — it only acts as a tie-breaker among Recent songs that already tie on the Provenance-safe features.
> **Dev:** And if two of the top songs are the same track, different mix?
> **Expert:** Diversity drops the near-duplicate. Relevance picked them both; dedup keeps one.
