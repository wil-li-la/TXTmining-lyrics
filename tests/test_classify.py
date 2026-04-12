"""Tests for classification pipeline."""

import numpy as np
from scipy.sparse import csr_matrix
from src.classify import train_and_evaluate


def test_train_and_evaluate_returns_metrics():
    # Synthetic data: 40 samples, 10 features, 4 classes
    np.random.seed(42)
    X = csr_matrix(np.random.rand(40, 10))
    y = np.array(["1990s"] * 10 + ["2000s"] * 10 + ["2010s"] * 10 + ["2020s"] * 10)
    metrics = train_and_evaluate(X, y, output_path=None)
    assert "accuracy" in metrics
    assert "auc_scores" in metrics
    assert isinstance(metrics["accuracy"], float)
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_train_and_evaluate_auc_scores_per_class():
    np.random.seed(42)
    X = csr_matrix(np.random.rand(40, 10))
    y = np.array(["1990s"] * 10 + ["2000s"] * 10 + ["2010s"] * 10 + ["2020s"] * 10)
    metrics = train_and_evaluate(X, y, output_path=None)
    assert len(metrics["auc_scores"]) == 4
    for label, score in metrics["auc_scores"].items():
        assert 0.0 <= score <= 1.0
