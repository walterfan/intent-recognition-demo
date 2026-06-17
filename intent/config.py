"""Shared hyperparameters and paths for the intent recognition demo."""

from dataclasses import dataclass, field
from pathlib import Path

ARTIFACT_DIR = Path(__file__).resolve().parent.parent / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "model.pt"
CURVE_PATH = ARTIFACT_DIR / "training_curves.png"
CONFUSION_PATH = ARTIFACT_DIR / "confusion_matrix.png"


@dataclass
class Config:
    # Reproducibility
    seed: int = 42

    # Data
    samples_per_intent: int = 120
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    # Probability of injecting colloquial noise (filler words / small typos) into
    # each generated sample. 0.0 keeps the clean, easy demo; raise it (e.g. 0.2)
    # to simulate messier real-world input and watch accuracy drop.
    noise_prob: float = 0.0

    # Preprocessing
    max_len: int = 16
    min_freq: int = 1

    # Model
    embed_dim: int = 64
    hidden_dim: int = 64
    attention_dim: int = 64
    dropout: float = 0.3

    # Training
    batch_size: int = 32
    max_epochs: int = 40
    patience: int = 5
    learning_rate: float = 1e-3

    # Prediction
    top_k: int = 3

    def __post_init__(self) -> None:
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")


DEFAULT_CONFIG = Config()
