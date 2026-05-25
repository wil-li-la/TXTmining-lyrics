"""Cross-validation analysis: stratified vs artist-grouped folds."""
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GroupKFold, cross_val_score, cross_val_predict
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.metrics import roc_auc_score

META_COLS = {"song_id", "year", "decade", "genre", "artist", "title",
             "chart_position", "source", "lyrics_raw"}


def build_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """Stack numeric features + flatten the embedding list into one X matrix."""
    feature_cols = [c for c in df.columns if c not in META_COLS and c != "embedding"]
    numeric = df[feature_cols].fillna(0.0).to_numpy(dtype=float)
    if "embedding" in df.columns:
        emb = np.vstack([np.asarray(e, dtype=float) for e in df["embedding"]])
        X = np.hstack([numeric, emb])
    else:
        X = numeric
    return StandardScaler().fit_transform(X)


def _cv_block(X, y, splitter, groups=None) -> dict:
    model = LogisticRegression(max_iter=2000, random_state=42, n_jobs=-1)
    acc = cross_val_score(model, X, y, cv=splitter, scoring="accuracy",
                          groups=groups, n_jobs=-1)
    y_proba = cross_val_predict(model, X, y, cv=splitter, method="predict_proba",
                                groups=groups, n_jobs=-1)
    classes = sorted(np.unique(y))
    y_bin = label_binarize(y, classes=classes)
    auc_per = {
        c: float(roc_auc_score(y_bin[:, i], y_proba[:, i]))
        for i, c in enumerate(classes)
    }
    return {
        "accuracy_mean": float(acc.mean()),
        "accuracy_std": float(acc.std()),
        "auc_per_decade": auc_per,
        "n_samples": int(len(y)),
        "n_folds": splitter.get_n_splits(X, y, groups),
    }


def run_cv(df: pd.DataFrame) -> dict:
    X = build_feature_matrix(df)
    y = df["decade"].to_numpy()

    strat = _cv_block(X, y, StratifiedKFold(n_splits=5, shuffle=True, random_state=42))

    if "artist" in df.columns:
        n_artists = df["artist"].nunique()
        n_folds = min(5, n_artists)
        groups = df["artist"].to_numpy()
        grouped = _cv_block(X, y, GroupKFold(n_splits=n_folds), groups=groups)
    else:
        grouped = {"error": "no artist column"}

    return {"stratified": strat, "grouped": grouped}


def main(feat_path: str = "data/song_features.parquet",
         lyrics_path: str = "data/lyrics_full.csv",
         out_path: str = "output/cv_results.json") -> None:
    feat = pd.read_parquet(feat_path)
    meta = pd.read_csv(lyrics_path)[["song_id", "artist"]]
    df = feat.merge(meta, on="song_id", how="left")
    results = run_cv(df)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
