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
