"""Single/batch Top-K intent prediction and an interactive CLI."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .config import DEFAULT_CONFIG
from .model import IntentClassifier
from .preprocess import Vocabulary, numericalize, tokenize
from .train import load_model


@dataclass
class Explanation:
    """Why the model made a prediction: tokens + attention + ranked intents."""

    text: str
    tokens: list[str]
    attention: list[float]  # one weight per token, aligned with `tokens`
    top_k: list[tuple[str, float]]  # (label, confidence), descending


def render_bar(value: float, width: int = 20) -> str:
    """Render a 0..1 value as a fixed-width unicode bar for terminal output."""
    filled = int(round(max(0.0, min(1.0, value)) * width))
    return "\u2588" * filled + "\u00b7" * (width - filled)


class IntentPredictor:
    def __init__(self, model: IntentClassifier, vocab: Vocabulary, labels: list[str], max_len: int):
        self.model = model
        self.model.eval()
        self.vocab = vocab
        self.labels = labels
        self.max_len = max_len

    @classmethod
    def from_saved(cls, path=None) -> "IntentPredictor":
        model, vocab, labels, cfg = load_model(path) if path else load_model()
        return cls(model, vocab, labels, cfg["max_len"])

    def _probs(self, texts: list[str]) -> torch.Tensor:
        ids = [numericalize(t, self.vocab, self.max_len) for t in texts]
        x = torch.tensor(ids, dtype=torch.long)
        with torch.no_grad():
            return F.softmax(self.model(x), dim=1)

    def predict(self, text: str, k: int = DEFAULT_CONFIG.top_k) -> list[tuple[str, float]]:
        """Return Top-K (label, confidence) for a single text, descending."""
        return self.predict_batch([text], k)[0]

    def predict_batch(self, texts: list[str], k: int = DEFAULT_CONFIG.top_k) -> list[list[tuple[str, float]]]:
        """Return one ranked Top-K result per input text, preserving order."""
        if not texts:
            return []
        probs = self._probs(texts)
        k = min(k, len(self.labels))
        results: list[list[tuple[str, float]]] = []
        top_probs, top_idx = probs.topk(k, dim=1)
        for row_probs, row_idx in zip(top_probs.tolist(), top_idx.tolist()):
            results.append([(self.labels[i], float(p)) for p, i in zip(row_probs, row_idx)])
        return results

    def explain(self, text: str, k: int = DEFAULT_CONFIG.top_k) -> Explanation:
        """Return tokens, per-token attention weights, and Top-K for one text.

        Makes the prediction interpretable: the attention weights show which
        tokens the model focused on when choosing the intent.
        """
        tokens = tokenize(text)[: self.max_len]
        ids = numericalize(text, self.vocab, self.max_len)
        x = torch.tensor([ids], dtype=torch.long)
        with torch.no_grad():
            logits, weights = self.model(x, return_attention=True)
            probs = F.softmax(logits, dim=1)[0]
        attention = weights[0][: len(tokens)].tolist()
        k = min(k, len(self.labels))
        top_probs, top_idx = probs.topk(k)
        top_k = [(self.labels[i], float(p)) for p, i in zip(top_probs.tolist(), top_idx.tolist())]
        return Explanation(text=text, tokens=tokens, attention=attention, top_k=top_k)


def print_explanation(exp: Explanation) -> None:
    """Pretty-print Top-K probabilities and the attention each token received."""
    print("  Top-K intents:")
    for label, conf in exp.top_k:
        print(f"    {label:<14} {render_bar(conf)} {conf:6.1%}")
    if exp.tokens:
        peak = max(exp.attention) or 1.0
        print("  Attention (which words drove the decision):")
        for tok, w in zip(exp.tokens, exp.attention):
            # Normalize by the peak so the most-attended token fills the bar.
            print(f"    {tok:<14} {render_bar(w / peak)} {w:6.1%}")


def interactive(path=None) -> None:
    predictor = IntentPredictor.from_saved(path)
    print("Interactive intent recognition. Type a sentence, or 'quit'/'exit' to stop.")
    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if text.lower() in {"quit", "exit"}:
            break
        if not text:
            continue
        print_explanation(predictor.explain(text))


def main() -> None:
    interactive()


if __name__ == "__main__":
    main()
