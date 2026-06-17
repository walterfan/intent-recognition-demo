"""Test-set evaluation: accuracy report and confusion matrix."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from .config import Config, DEFAULT_CONFIG, CONFUSION_PATH, ARTIFACT_DIR
from .data import Dataset
from .predict import IntentPredictor


def evaluate(predictor: IntentPredictor, dataset: Dataset, config: Config = DEFAULT_CONFIG):
    """Evaluate on the test split; return (confusion matrix, label order)."""
    texts = [t for t, _ in dataset.test]
    y_true = [label for _, label in dataset.test]
    y_pred = [preds[0][0] for preds in predictor.predict_batch(texts, k=1)]

    labels = dataset.labels
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0))
    return cm, labels


def plot_confusion_matrix(cm: np.ndarray, labels: list[str], path=CONFUSION_PATH) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("Confusion Matrix")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved confusion matrix to {path}")


def main() -> None:
    import argparse

    from .data import build_dataset_by_name

    parser = argparse.ArgumentParser(description="Evaluate the saved model on a test split.")
    parser.add_argument(
        "--dataset",
        choices=["synthetic", "snips"],
        default="synthetic",
        help="Must match the dataset used for training to reproduce the same test split.",
    )
    args = parser.parse_args()

    config = DEFAULT_CONFIG
    dataset = build_dataset_by_name(args.dataset, config)
    predictor = IntentPredictor.from_saved()
    cm, labels = evaluate(predictor, dataset, config)
    plot_confusion_matrix(cm, labels)


if __name__ == "__main__":
    main()
