"""Sentence-BERT mean-pooled lyric embeddings."""
from functools import lru_cache
from sentence_transformers import SentenceTransformer

@lru_cache(maxsize=1)
def _model():
    return SentenceTransformer("all-MiniLM-L6-v2")

def extract(views: dict) -> dict:
    text = views["clean"] or " "  # empty input would error
    vec = _model().encode(text, show_progress_bar=False, normalize_embeddings=True)
    return {"embedding": vec.tolist()}
