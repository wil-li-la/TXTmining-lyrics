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
    return np.array([
        (feats.get(k, 0.0) - stats[k]["mean"]) / max(stats[k]["std"], 1e-9)
        for k in keys
    ])


def score_candidate(cand_feats: dict, profile: dict[str, float], stats: dict) -> float:
    """Cosine similarity between candidate z-vector and user target z-vector.
    Returns a value in [0, 1] (mapped from [-1, 1])."""
    target = profile_to_target(profile, stats)
    keys = sorted(target.keys())
    if not keys:
        return 0.0
    cand_z = _to_zscore(cand_feats, stats, keys)
    target_z = np.array([target[k] for k in keys])
    nc = np.linalg.norm(cand_z)
    nt = np.linalg.norm(target_z)
    if nc == 0 or nt == 0:
        return 0.5
    cos = float(cand_z @ target_z / (nc * nt))
    return (cos + 1) / 2


def rank(candidates: list[dict], profile: dict[str, float], stats: dict) -> list[dict]:
    """Annotate each candidate with 'score' and return sorted descending."""
    out = []
    for c in candidates:
        s = score_candidate(c["features"], profile, stats)
        out.append({**c, "score": s})
    return sorted(out, key=lambda x: -x["score"])


# ---------------------------------------------------------------------------
# Whole-catalog retrieval recommender (ADR-0001).
#
# Relevance is ranked ONLY on provenance-safe features — values comparable
# across all eras. Line-based features (rhyme, repetition) are fixed near zero
# for pre-2016 songs (source data stripped line breaks), so they cannot rank
# the catalog; they only break ties among Recent (2016+) songs. The sentence
# embedding does Diversity (dedup), never Relevance.
# ---------------------------------------------------------------------------

SAFE_FEATURES = ["valence", "arousal", "mean_concreteness", "pronoun_i"]
LINE_FEATURES = ["rhyme_density", "chorus_repeat_ratio"]
RECENT_YEAR = 2016


def _cosine_over(feats: dict, profile: dict[str, float], stats: dict,
                 keys: list[str]) -> float:
    """Cosine similarity over the given feature keys, mapped to [0, 1].

    Only keys present in BOTH the profile and stats are used; if none remain,
    returns the neutral 0.5 (no signal)."""
    keys = [k for k in keys if k in profile and k in stats]
    if not keys:
        return 0.5
    cand_z = _to_zscore(feats, stats, keys)
    target_z = np.array([float(profile[k]) for k in keys])
    nc = np.linalg.norm(cand_z)
    nt = np.linalg.norm(target_z)
    if nc == 0 or nt == 0:
        return 0.5
    return (float(cand_z @ target_z / (nc * nt)) + 1) / 2


def safe_relevance(feats: dict, profile: dict[str, float], stats: dict) -> float:
    """Primary relevance: cosine over provenance-safe features only."""
    return _cosine_over(feats, profile, stats, SAFE_FEATURES)


def line_match(feats: dict, profile: dict[str, float], stats: dict) -> float:
    """Secondary signal: cosine over line-based features (rhyme, repetition)."""
    return _cosine_over(feats, profile, stats, LINE_FEATURES)


def _emb_cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(a @ b / (na * nb))


def dedupe_by_embedding(rows: list[dict], threshold: float = 0.95) -> list[dict]:
    """Drop rows whose embedding is near-identical to an already-kept row.

    Assumes `rows` is already sorted best-first, so the higher-ranked member of
    a near-duplicate pair (e.g. alternate mixes of one song) is the one kept."""
    kept: list[dict] = []
    kept_embs: list[np.ndarray] = []
    for r in rows:
        emb = np.asarray(r["embedding"], dtype=float)
        if any(_emb_cosine(emb, e) >= threshold for e in kept_embs):
            continue
        kept.append(r)
        kept_embs.append(emb)
    return kept


def recommend(index_df, profile: dict[str, float], stats: dict, k: int = 5,
              recent_year: int = RECENT_YEAR, tie_epsilon: float = 0.01,
              dedup_threshold: float = 0.95) -> list[dict]:
    """Rank the whole catalog against the taste profile and return the top k.

    1. Primary score = safe-feature cosine, computed for every song.
    2. Songs are bucketed by primary score (width `tie_epsilon`); within a
       bucket, Recent (>= recent_year) songs are ordered by line-feature match.
       Pre-recent songs get no line bonus, so line features never reorder them.
    3. Near-duplicate songs are removed by embedding similarity.
    """
    rows: list[dict] = []
    for rec in index_df.to_dict("records"):
        primary = safe_relevance(rec, profile, stats)
        recent = int(rec.get("year", 0) or 0) >= recent_year
        # Recent songs are tie-broken by line-feature match; pre-recent songs
        # sit at the neutral midpoint (0.5) — equal to a recent song with no
        # line preference — so recency alone never reorders a tie.
        bonus = line_match(rec, profile, stats) if recent else 0.5
        rec = dict(rec)
        rec["score"] = primary
        rec["_bucket"] = round(primary / tie_epsilon)
        rec["_line"] = bonus
        rows.append(rec)
    rows.sort(key=lambda r: (r["_bucket"], r["_line"]), reverse=True)
    rows = dedupe_by_embedding(rows, dedup_threshold)
    for r in rows:
        r.pop("_bucket", None)
        r.pop("_line", None)
    return rows[:k]
