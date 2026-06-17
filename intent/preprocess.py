"""Tokenization, vocabulary building, and text numericalization."""

from __future__ import annotations

import re

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
PAD_IDX = 0
UNK_IDX = 1

# Matches a run of CJK characters; used to decide character-level fallback.
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def tokenize(text: str) -> list[str]:
    """Whitespace tokenizer with character-level fallback for CJK text.

    Latin text is lowercased and split on whitespace. CJK characters are split
    individually since they are not whitespace-delimited.
    """
    text = text.strip().lower()
    if not text:
        return []
    tokens: list[str] = []
    for chunk in text.split():
        if _CJK_RE.search(chunk):
            # Split CJK characters individually, keep non-CJK runs together.
            buf = ""
            for ch in chunk:
                if _CJK_RE.match(ch):
                    if buf:
                        tokens.append(buf)
                        buf = ""
                    tokens.append(ch)
                else:
                    buf += ch
            if buf:
                tokens.append(buf)
        else:
            tokens.append(chunk)
    return tokens


class Vocabulary:
    """Maps tokens to integer indices with reserved pad/unk entries."""

    def __init__(self, token_to_idx: dict[str, int]):
        self.token_to_idx = token_to_idx
        self.idx_to_token = {idx: tok for tok, idx in token_to_idx.items()}

    def __len__(self) -> int:
        return len(self.token_to_idx)

    def index(self, token: str) -> int:
        return self.token_to_idx.get(token, UNK_IDX)

    @classmethod
    def build(cls, texts: list[str], min_freq: int = 1) -> "Vocabulary":
        counts: dict[str, int] = {}
        for text in texts:
            for tok in tokenize(text):
                counts[tok] = counts.get(tok, 0) + 1
        token_to_idx = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX}
        for tok in sorted(counts):
            if counts[tok] >= min_freq:
                token_to_idx[tok] = len(token_to_idx)
        return cls(token_to_idx)

    def to_dict(self) -> dict[str, int]:
        return dict(self.token_to_idx)

    @classmethod
    def from_dict(cls, token_to_idx: dict[str, int]) -> "Vocabulary":
        return cls(dict(token_to_idx))


def numericalize(text: str, vocab: Vocabulary, max_len: int) -> list[int]:
    """Convert text to a fixed-length list of indices (padded/truncated)."""
    ids = [vocab.index(tok) for tok in tokenize(text)]
    ids = ids[:max_len]
    if len(ids) < max_len:
        ids = ids + [PAD_IDX] * (max_len - len(ids))
    return ids
