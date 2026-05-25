"""Brysbaert et al. 2014 concreteness norms — 40k words rated 1-5."""
import os
from functools import lru_cache
import pandas as pd

BRYSBAERT_PATH = "data/raw/brysbaert.txt"

@lru_cache(maxsize=1)
def _load_ratings() -> dict[str, float]:
    if not os.path.exists(BRYSBAERT_PATH):
        raise FileNotFoundError(
            f"Missing {BRYSBAERT_PATH}. Run: python3 src/data_sources.py"
        )
    df = pd.read_csv(BRYSBAERT_PATH, sep="\t")
    # Column "Word" (lowercased) and "Conc.M" (mean rating 1-5)
    return dict(zip(df["Word"].str.lower(), df["Conc.M"]))

def extract(views: dict) -> dict[str, float]:
    text = views["clean"]
    if not text.strip():
        return {"mean_concreteness": 0.0, "pct_concrete": 0.0}

    ratings = _load_ratings()
    tokens = text.split()
    scored = [ratings[t] for t in tokens if t in ratings]
    if not scored:
        return {"mean_concreteness": 0.0, "pct_concrete": 0.0}

    mean = sum(scored) / len(scored)
    pct_concrete = sum(1 for s in scored if s >= 4.0) / len(scored)
    return {
        "mean_concreteness": float(mean),
        "pct_concrete": float(pct_concrete),
    }
