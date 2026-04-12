"""Build TF-IDF features from preprocessed lyrics."""

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def build_tfidf(lyrics: pd.Series, max_features: int = 5000):
    """Build TF-IDF matrix from a series of cleaned lyrics.

    Returns (sparse_matrix, feature_names_list, fitted_vectorizer).
    """
    vectorizer = TfidfVectorizer(max_features=max_features)
    matrix = vectorizer.fit_transform(lyrics)
    vocab = vectorizer.get_feature_names_out().tolist()
    return matrix, vocab, vectorizer


def main():
    """Read processed lyrics, build TF-IDF, save artifacts."""
    df = pd.read_csv("data/lyrics_processed.csv")
    matrix, vocab, vectorizer = build_tfidf(df["lyrics_clean"])

    joblib.dump(matrix, "data/tfidf_matrix.pkl")
    joblib.dump(vocab, "data/tfidf_vocab.pkl")
    joblib.dump(vectorizer, "data/tfidf_vectorizer.pkl")

    print(f"TF-IDF matrix shape: {matrix.shape}")
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Saved to data/tfidf_matrix.pkl, data/tfidf_vocab.pkl, data/tfidf_vectorizer.pkl")


if __name__ == "__main__":
    main()
