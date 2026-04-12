"""Train Logistic Regression to classify lyrics by decade, evaluate with ROC/AUC."""

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score, roc_curve, auc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def train_and_evaluate(X, y, output_path: str | None = "output/roc_curve.png"):
    """Train Logistic Regression and produce ROC curves.

    Args:
        X: TF-IDF feature matrix (sparse or dense).
        y: Array of decade labels.
        output_path: Path to save ROC curve plot. None to skip plotting.

    Returns:
        Dict with 'accuracy' and 'auc_scores' per class.
    """
    classes = sorted(np.unique(y))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # ROC/AUC (one-vs-rest)
    y_test_bin = label_binarize(y_test, classes=classes)
    y_score = model.predict_proba(X_test)

    auc_scores = {}
    if output_path is not None:
        plt.figure(figsize=(8, 6))

    for i, label in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        auc_scores[label] = roc_auc
        if output_path is not None:
            plt.plot(fpr, tpr, label=f"{label} (AUC = {roc_auc:.2f})")

    if output_path is not None:
        plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curves — Decade Classification")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"ROC curve saved to {output_path}")

    return {"accuracy": acc, "auc_scores": auc_scores}


def main():
    """Load features, train model, evaluate, and save results."""
    df = pd.read_csv("data/lyrics_processed.csv")
    matrix = joblib.load("data/tfidf_matrix.pkl")

    y = df["decade"].values
    metrics = train_and_evaluate(matrix, y)

    print(f"\nAccuracy: {metrics['accuracy']:.2%}")
    print("\nAUC Scores per Decade:")
    for label, score in metrics["auc_scores"].items():
        print(f"  {label}: {score:.3f}")


if __name__ == "__main__":
    main()
