"""End-to-end pipeline walkthrough: show every stage from text to intent.

Run after training a model:

    python -m intent.demo                 # use built-in example sentences
    python -m intent.demo "play some jazz" "is the flu serious"

For each sentence it prints the intermediate result of every pipeline stage
(tokenize -> vocab ids -> padding -> model -> Top-K + attention), so the
otherwise hidden "black box" steps become easy to follow.
"""

from __future__ import annotations

import sys

from .predict import IntentPredictor, print_explanation
from .preprocess import PAD_IDX, PAD_TOKEN, UNK_IDX, UNK_TOKEN, numericalize, tokenize

EXAMPLE_SENTENCES = [
    "play some jazz for me",
    "what are the symptoms of the flu",
    "set an alarm for seven am",
    "will it rain tomorrow",
]


def walk_through(predictor: IntentPredictor, text: str) -> None:
    vocab = predictor.vocab
    max_len = predictor.max_len

    print("=" * 60)
    print(f"[0] raw text        : {text!r}")

    tokens = tokenize(text)
    print(f"[1] tokenize        : {tokens}")

    # Map each token to its vocab id, flagging out-of-vocabulary tokens.
    pairs = []
    for tok in tokens:
        idx = vocab.index(tok)
        flag = f"  <- OOV, mapped to {UNK_TOKEN}({UNK_IDX})" if idx == UNK_IDX else ""
        pairs.append(f"{tok!r} -> {idx}{flag}")
    print("[2] token -> id     :")
    for line in pairs:
        print(f"      {line}")

    padded = numericalize(text, vocab, max_len)
    pad_count = padded.count(PAD_IDX)
    print(f"[3] pad/truncate    : (max_len={max_len}, {PAD_TOKEN}={PAD_IDX})")
    print(f"      {padded}")
    if pad_count:
        print(f"      ({pad_count} trailing {PAD_TOKEN} added)")

    exp = predictor.explain(text)
    print("[4] model -> intent :")
    print_explanation(exp)


def main(argv: list[str] | None = None) -> None:
    argv = argv if argv is not None else sys.argv[1:]
    sentences = argv if argv else EXAMPLE_SENTENCES
    predictor = IntentPredictor.from_saved()
    for text in sentences:
        walk_through(predictor, text)
    print("=" * 60)


if __name__ == "__main__":
    main()
