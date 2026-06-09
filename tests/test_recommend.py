"""Tests for whole-catalog retrieval recommendation (ADR-0001).

Relevance is ranked ONLY on provenance-safe features (valence, arousal,
concreteness, self-focus). Line-based features (rhyme, repetition) are a
tie-breaker among Recent (2016+) songs only. The embedding does dedup, not
relevance.
"""
import numpy as np
import pandas as pd

from src.agent.ranking import (
    safe_relevance,
    line_match,
    dedupe_by_embedding,
    recommend,
    SAFE_FEATURES,
    LINE_FEATURES,
)

# mean 0 / std 1 so raw feature value == z-score (easy to reason about)
STATS = {
    k: {"mean": 0.0, "std": 1.0}
    for k in ["valence", "arousal", "mean_concreteness", "pronoun_i",
              "rhyme_density", "chorus_repeat_ratio"]
}


def _song(song_id, year, emb, **feats):
    base = {k: 0.0 for k in STATS}
    base.update(feats)
    return {"song_id": song_id, "artist": f"artist-{song_id}",
            "title": f"title-{song_id}", "year": year,
            "embedding": np.array(emb, dtype=float), **base}


def test_safe_relevance_ignores_line_features():
    # profile matches on all 4 safe features but candidate is OPPOSITE on rhyme.
    profile = {"valence": 1.0, "arousal": 1.0, "mean_concreteness": 1.0,
               "pronoun_i": 1.0, "rhyme_density": 2.0}
    feats = {"valence": 1.0, "arousal": 1.0, "mean_concreteness": 1.0,
             "pronoun_i": 1.0, "rhyme_density": -2.0}
    # rhyme mismatch must not pull the score down — it's excluded from relevance.
    assert safe_relevance(feats, profile, STATS) > 0.95


def test_line_match_only_uses_line_features():
    profile = {"valence": 1.0, "rhyme_density": 1.0, "chorus_repeat_ratio": 1.0}
    feats = {"valence": -1.0, "rhyme_density": 1.0, "chorus_repeat_ratio": 1.0}
    # valence is ignored by line_match; the two line features match perfectly.
    assert line_match(feats, profile, STATS) > 0.95


def test_recommend_ranks_by_safe_features():
    profile = {"valence": 2.0, "arousal": 0.0, "mean_concreteness": 0.0,
               "pronoun_i": 0.0}
    df = pd.DataFrame([
        _song("A", 2024, [1, 0, 0], valence=2.0),   # closest
        _song("B", 2024, [0, 1, 0], valence=0.0),
        _song("C", 2024, [0, 0, 1], valence=-2.0),  # farthest
    ])
    out = recommend(df, profile, STATS, k=3)
    assert [r["song_id"] for r in out] == ["A", "B", "C"]


def test_recommend_tiebreak_prefers_recent_line_match():
    # X and Y tie on safe features; both Recent; X matches the line profile.
    profile = {"valence": 1.0, "rhyme_density": 1.0, "chorus_repeat_ratio": 1.0}
    df = pd.DataFrame([
        _song("Y", 2024, [0, 1, 0], valence=1.0,
              rhyme_density=-1.0, chorus_repeat_ratio=-1.0),
        _song("X", 2023, [1, 0, 0], valence=1.0,
              rhyme_density=1.0, chorus_repeat_ratio=1.0),
    ])
    out = recommend(df, profile, STATS, k=2)
    assert [r["song_id"] for r in out] == ["X", "Y"]


def test_recommend_tiebreak_does_not_apply_to_pre2016():
    # Same setup, but both songs are pre-2016: line features must NOT reorder
    # them, so input order is preserved (P listed before Q stays first).
    profile = {"valence": 1.0, "rhyme_density": 1.0, "chorus_repeat_ratio": 1.0}
    df = pd.DataFrame([
        _song("P", 1999, [1, 0, 0], valence=1.0,
              rhyme_density=-1.0, chorus_repeat_ratio=-1.0),  # worse line match
        _song("Q", 1999, [0, 1, 0], valence=1.0,
              rhyme_density=1.0, chorus_repeat_ratio=1.0),    # better line match
    ])
    out = recommend(df, profile, STATS, k=2)
    assert [r["song_id"] for r in out] == ["P", "Q"]


def test_no_line_preference_does_not_favor_recent_in_tie():
    # The listener expresses NO rhyme/repetition preference. A Recent song must
    # not jump ahead of a tied pre-2016 song just for being recent.
    profile = {"valence": 1.0, "rhyme_density": 0.0, "chorus_repeat_ratio": 0.0}
    df = pd.DataFrame([
        _song("old", 1990, [1, 0, 0], valence=1.0),  # tied on safe, listed first
        _song("new", 2024, [0, 1, 0], valence=1.0),  # tied on safe, recent
    ])
    out = recommend(df, profile, STATS, k=2)
    assert [r["song_id"] for r in out] == ["old", "new"]


def test_dedupe_removes_near_duplicate_embeddings():
    rows = [
        {"song_id": "1", "embedding": np.array([1.0, 0.0, 0.0])},
        {"song_id": "2", "embedding": np.array([0.999, 0.001, 0.0])},  # ~dup of 1
        {"song_id": "3", "embedding": np.array([0.0, 1.0, 0.0])},      # distinct
    ]
    kept = dedupe_by_embedding(rows, threshold=0.95)
    assert [r["song_id"] for r in kept] == ["1", "3"]


def test_recommend_dedupes_near_duplicate_songs():
    profile = {"valence": 2.0}
    df = pd.DataFrame([
        _song("orig", 2024, [1, 0, 0], valence=2.0),
        _song("mix", 2024, [0.999, 0.001, 0.0], valence=2.0),  # alt mix
        _song("other", 2024, [0, 1, 0], valence=1.5),
    ])
    out = recommend(df, profile, STATS, k=5)
    ids = [r["song_id"] for r in out]
    assert "orig" in ids and "other" in ids
    assert "mix" not in ids  # near-duplicate dropped


def test_safe_and_line_feature_constants():
    assert SAFE_FEATURES == ["valence", "arousal", "mean_concreteness", "pronoun_i"]
    assert LINE_FEATURES == ["rhyme_density", "chorus_repeat_ratio"]
