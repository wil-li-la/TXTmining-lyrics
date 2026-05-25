from src.features.embedding import extract

def test_returns_384d_vector():
    r = extract({"clean": "love is in the air"})
    assert "embedding" in r
    assert len(r["embedding"]) == 384

def test_similar_text_similar_embedding():
    import numpy as np
    a = extract({"clean": "the dog ran fast"})["embedding"]
    b = extract({"clean": "a dog was running quickly"})["embedding"]
    c = extract({"clean": "quantum mechanics is hard"})["embedding"]
    cos = lambda x, y: np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
    assert cos(a, b) > cos(a, c)
