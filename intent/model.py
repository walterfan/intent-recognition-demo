"""Bidirectional LSTM with additive attention for intent classification."""

from __future__ import annotations

import torch
import torch.nn as nn

from .preprocess import PAD_IDX


class AdditiveAttention(nn.Module):
    """Bahdanau-style attention pooling over LSTM outputs."""

    def __init__(self, hidden_dim: int, attention_dim: int):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, attention_dim)
        self.score = nn.Linear(attention_dim, 1, bias=False)

    def forward(
        self, outputs: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # outputs: (batch, seq, hidden); mask: (batch, seq) with True for real tokens.
        energy = self.score(torch.tanh(self.proj(outputs))).squeeze(-1)  # (batch, seq)
        energy = energy.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(energy, dim=1)  # (batch, seq), sums to 1 over real tokens
        context = (outputs * weights.unsqueeze(-1)).sum(dim=1)  # (batch, hidden)
        return context, weights


class IntentClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        embed_dim: int = 64,
        hidden_dim: int = 64,
        attention_dim: int = 64,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, batch_first=True, bidirectional=True
        )
        self.attention = AdditiveAttention(hidden_dim * 2, attention_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        # x: (batch, seq) of token indices.
        mask = x != PAD_IDX
        embedded = self.embedding(x)
        outputs, _ = self.lstm(embedded)  # (batch, seq, hidden*2)
        context, weights = self.attention(outputs, mask)
        logits = self.classifier(self.dropout(context))  # (batch, num_classes)
        if return_attention:
            return logits, weights  # weights: (batch, seq) attention over tokens
        return logits
