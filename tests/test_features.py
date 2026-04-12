"""Tests for TF-IDF feature extraction."""

import pandas as pd
from src.features import build_tfidf


def test_build_tfidf_returns_correct_shape():
    df = pd.DataFrame({
        "lyrics_clean": ["love baby tonight", "money car fast", "feel good sunshine"],
        "decade": ["1990s", "2000s", "2010s"],
    })
    matrix, vocab, vectorizer = build_tfidf(df["lyrics_clean"])
    assert matrix.shape[0] == 3
    assert matrix.shape[1] == len(vocab)
    assert len(vocab) > 0


def test_build_tfidf_vocab_contains_words():
    df = pd.DataFrame({
        "lyrics_clean": ["love baby tonight", "money car fast"],
    })
    matrix, vocab, vectorizer = build_tfidf(df["lyrics_clean"])
    assert "love" in vocab
    assert "money" in vocab


def test_build_tfidf_matrix_is_sparse():
    from scipy.sparse import issparse
    df = pd.DataFrame({
        "lyrics_clean": ["hello world", "foo bar baz"],
    })
    matrix, vocab, vectorizer = build_tfidf(df["lyrics_clean"])
    assert issparse(matrix)
