"""Redeploy runtime artifacts to the HF Space (no git remote; uses huggingface_hub).

Pushes only changed runtime files needed by the deployed app, then triggers a
Docker rebuild. See memory hf-space-deploy for the full recipe.
"""
import os

from dotenv import load_dotenv
from huggingface_hub import HfApi

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

REPO = "wil-li-la/pop-lyrics-taste-profiler"
ALLOW = [
    "app.py",
    "data/song_features.parquet",
    "data/feature_stats.json",
    "data/lyrics_processed.csv",
    "output/cv_results.json",
    "output/trends/*.png",
]


def main() -> None:
    api = HfApi(token=os.environ["HF_TOKEN"])
    res = api.upload_folder(
        folder_path=".",
        repo_id=REPO,
        repo_type="space",
        allow_patterns=ALLOW,
        commit_message="Expand 2020s corpus to 476 songs; refresh features, CV, trends",
    )
    print("upload done:", res)


if __name__ == "__main__":
    main()
