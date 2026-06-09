# Recommend by retrieval over the feature index, ranked on provenance-safe features

## Status

accepted

## Context

The original recommender was a live OpenAI tool-use loop: the model searched Genius, fetched lyrics, and extracted features on the fly, and a deterministic step ranked whatever it returned. The model's system prompt and tool schema hardcoded six example artist+song pairs (Sabrina Carpenter, Olivia Rodrigo, Chappell Roan, Tate McRae, Gracie Abrams, Billie Eilish). A small model copies such examples near-verbatim, so the **Candidate Pool was effectively those few artists regardless of the Taste Profile**. Worse, live free-text search *cannot* target the profile — a song's features are unknown until after it is fetched — so the sliders only re-ordered a biased, near-constant pool. Result: recommendations were dominated by Sabrina Carpenter and a fixed set of 2023–25 pop songwriters.

## Decision

Recommendations are produced by **deterministic retrieval over the precomputed feature index** (`data/song_features.parquet`, the whole catalog 1965–2025), not by live search. Relevance is cosine similarity between the Taste Profile and each song's **provenance-safe features only** — valence, arousal, concreteness, self-focus. The sentence **embedding is used for Diversity (dedup of near-duplicate songs), not Relevance**. The two **line-based sliders (rhyme density, repetition) act as a tie-breaker only among Recent (2016+) songs** whose primary scores are within epsilon. The language model is retained solely to write the "why these match" explanation over the already-ranked top results. The old live-search loop is kept as a fallback for the (practically impossible) case where retrieval returns nothing — with its hardcoded examples replaced so the fallback cannot reintroduce the bias.

## Considered options

- **De-bias the live agent (keep tool-use search).** Rejected: live search fundamentally cannot make Candidate selection profile-driven; at best it diversifies a random pool and re-ranks it — strictly worse than deterministic retrieval that the project already has the data for.
- **Whole catalog ranked on all features.** Rejected: pre-2016 line-based features are fixed near zero (line breaks were stripped from the source data), so ranking on them swaps the visible artist bias for an invisible *era* artifact — "low rhyme/repetition" profiles flood old songs; "high rhyme/repetition" profiles never see them.
- **Restrict the pool to 2016+ (where all features are valid).** Rejected: shrinks discovery to ~570 songs; we preferred whole-catalog reach with a feature subset that is fair across eras.

## Consequences

- The recommender is now **deterministic and OpenAI-free for correctness** — same profile yields the same shortlist (a claim `report.md` previously made but did not honor). Only the explanation prose needs the model.
- **Rhyme and repetition are demoted** from primary ranking signals to a Recent-only tie-breaker; they remain visible in each result's feature-comparison chart.
- `report.md` §6 (live tool-use narrative) and §6.3 (search-trace screenshots) become inaccurate and are rewritten/regenerated as part of this change.
- A prolific artist can still occupy multiple slots if they genuinely match (no per-artist cap was added) — accepted, because that reflects real Relevance rather than a prompt artifact.
