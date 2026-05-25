"""Run all feature modules over data/lyrics_full.csv -> song_features.parquet."""
import json
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

from src.preprocess_v2 import make_views
from src.features import rhyme, repetition, emotion, concreteness, pronouns, embedding

FEATURES = [rhyme, repetition, emotion, concreteness, pronouns, embedding]

def extract_row(lyrics_raw: str) -> dict:
    views = make_views(lyrics_raw)
    out = {}
    for mod in FEATURES:
        out.update(mod.extract(views))
    return out

def main(lyrics_path: str = "data/lyrics_full.csv",
         genres_path: str = "data/genre_tags.csv",
         out_path: str = "data/song_features.parquet",
         stats_path: str = "data/feature_stats.json",
         n_jobs: int = -1) -> None:
    df = pd.read_csv(lyrics_path)
    if Path(genres_path).exists():
        g = pd.read_csv(genres_path)[["song_id", "genre"]]
        df = df.merge(g, on="song_id", how="left")
    else:
        df["genre"] = "unknown"

    print(f"Extracting features for {len(df)} songs (n_jobs={n_jobs})...")
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(extract_row)(lyr) for lyr in df["lyrics_raw"]
    )
    feat_df = pd.DataFrame(results)
    merged = pd.concat([df[["song_id", "year", "decade", "genre"]].reset_index(drop=True),
                        feat_df.reset_index(drop=True)], axis=1)
    merged.to_parquet(out_path, index=False)
    print(f"Saved {len(merged)} rows -> {out_path}")

    # Corpus-wide mean/std for z-scoring (skip the embedding column)
    numeric_cols = [c for c in feat_df.columns if c != "embedding"]
    stats = {
        c: {"mean": float(feat_df[c].mean()), "std": float(feat_df[c].std() or 1.0)}
        for c in numeric_cols
    }
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved feature stats -> {stats_path}")

if __name__ == "__main__":
    main()
