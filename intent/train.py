"""Training loop with validation, early stopping, persistence, and curves."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .config import Config, DEFAULT_CONFIG, MODEL_PATH, CURVE_PATH, ARTIFACT_DIR
from .data import Dataset, build_dataset, build_dataset_by_name
from .model import IntentClassifier
from .preprocess import Vocabulary, numericalize


@dataclass
class History:
    train_loss: list[float]
    val_loss: list[float]
    train_acc: list[float]
    val_acc: list[float]


def _encode(
    samples: list[tuple[str, str]],
    vocab: Vocabulary,
    label_to_idx: dict[str, int],
    max_len: int,
) -> TensorDataset:
    xs = [numericalize(text, vocab, max_len) for text, _ in samples]
    ys = [label_to_idx[label] for _, label in samples]
    return TensorDataset(torch.tensor(xs, dtype=torch.long), torch.tensor(ys, dtype=torch.long))


def _set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _run_epoch(model, loader, criterion, optimizer=None) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss, correct, total = 0.0, 0, 0
    for xb, yb in loader:
        with torch.set_grad_enabled(is_train):
            logits = model(xb)
            loss = criterion(logits, yb)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * xb.size(0)
        correct += (logits.argmax(1) == yb).sum().item()
        total += xb.size(0)
    return total_loss / total, correct / total


def train(config: Config = DEFAULT_CONFIG, dataset: Dataset | None = None):
    """Train a classifier and return (model, vocab, labels, history)."""
    _set_seed(config.seed)
    dataset = dataset or build_dataset(config)
    labels = dataset.labels
    label_to_idx = {label: i for i, label in enumerate(labels)}

    vocab = Vocabulary.build([t for t, _ in dataset.train], min_freq=config.min_freq)

    train_ds = _encode(dataset.train, vocab, label_to_idx, config.max_len)
    val_ds = _encode(dataset.val, vocab, label_to_idx, config.max_len)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size)

    model = IntentClassifier(
        vocab_size=len(vocab),
        num_classes=len(labels),
        embed_dim=config.embed_dim,
        hidden_dim=config.hidden_dim,
        attention_dim=config.attention_dim,
        dropout=config.dropout,
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    history = History([], [], [], [])
    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improve = 0

    for epoch in range(1, config.max_epochs + 1):
        tr_loss, tr_acc = _run_epoch(model, train_loader, criterion, optimizer)
        va_loss, va_acc = _run_epoch(model, val_loader, criterion)
        history.train_loss.append(tr_loss)
        history.val_loss.append(va_loss)
        history.train_acc.append(tr_acc)
        history.val_acc.append(va_acc)
        print(
            f"Epoch {epoch:02d} | train_loss={tr_loss:.4f} acc={tr_acc:.3f} "
            f"| val_loss={va_loss:.4f} acc={va_acc:.3f}"
        )

        if va_loss < best_val - 1e-4:
            best_val = va_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= config.patience:
                print(f"Early stopping at epoch {epoch} (best val_loss={best_val:.4f})")
                break

    model.load_state_dict(best_state)
    return model, vocab, labels, history


def save_model(model: IntentClassifier, vocab: Vocabulary, labels: list[str], config: Config) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "vocab": vocab.to_dict(),
            "labels": labels,
            "config": {
                "embed_dim": config.embed_dim,
                "hidden_dim": config.hidden_dim,
                "attention_dim": config.attention_dim,
                "dropout": config.dropout,
                "max_len": config.max_len,
            },
        },
        MODEL_PATH,
    )
    print(f"Saved model to {MODEL_PATH}")


def load_model(path=MODEL_PATH):
    """Reload a saved bundle into (model, vocab, labels, saved_config)."""
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    vocab = Vocabulary.from_dict(bundle["vocab"])
    labels = bundle["labels"]
    cfg = bundle["config"]
    model = IntentClassifier(
        vocab_size=len(vocab),
        num_classes=len(labels),
        embed_dim=cfg["embed_dim"],
        hidden_dim=cfg["hidden_dim"],
        attention_dim=cfg["attention_dim"],
        dropout=cfg["dropout"],
    )
    model.load_state_dict(bundle["model_state"])
    model.eval()
    return model, vocab, labels, cfg


def plot_history(history: History, path=CURVE_PATH) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history.train_loss) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(epochs, history.train_loss, label="train")
    ax1.plot(epochs, history.val_loss, label="val")
    ax1.set_title("Loss")
    ax1.set_xlabel("epoch")
    ax1.legend()
    ax2.plot(epochs, history.train_acc, label="train")
    ax2.plot(epochs, history.val_acc, label="val")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("epoch")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved training curves to {path}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train the intent classifier.")
    parser.add_argument(
        "--dataset",
        choices=["synthetic", "snips"],
        default="synthetic",
        help="Data source: synthetic templates (default) or the SNIPS benchmark.",
    )
    args = parser.parse_args()

    config = DEFAULT_CONFIG
    dataset = build_dataset_by_name(args.dataset, config)
    model, vocab, labels, history = train(config, dataset)
    save_model(model, vocab, labels, config)
    plot_history(history)


if __name__ == "__main__":
    main()
