"""
encoder.py
----------
Neural network encoders that convert raw game-state tensors into
a latent representation suitable for policy and value heads.

Two variants:
  - MLPEncoder      : simple feed-forward, fast, lower capacity
  - TransformerEncoder : attention over card-play history, higher capacity
                        and naturally suited to sequence-based reasoning
                        (card counting, void detection)
"""

import torch
import torch.nn as nn
from typing import Optional


class MLPEncoder(nn.Module):
    """
    Simple multi-layer perceptron encoder.
    Input:  flat state vector of shape (batch, obs_dim)
    Output: latent vector of shape (batch, embed_dim)
    """

    def __init__(self, obs_dim: int, embed_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.obs_dim = obs_dim
        self.embed_dim = embed_dim
        # TODO: define self.net as a Sequential MLP
        # Suggested architecture:
        #   Linear(obs_dim, hidden_dim) → LayerNorm → ReLU → Dropout
        #   Linear(hidden_dim, hidden_dim) → LayerNorm → ReLU → Dropout
        #   Linear(hidden_dim, embed_dim)
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, obs_dim) → (batch, embed_dim)"""
        raise NotImplementedError


class CardHistoryEncoder(nn.Module):
    """
    Encodes the sequence of (player, card) pairs played so far using
    a small Transformer. Naturally captures:
      - Which cards are gone from each suit (void detection)
      - Relative order of high-card elimination
      - Player tendencies from their play sequence
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        max_seq_len: int = 52,  # max cards played in a round
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        # Card embedding: 52 possible cards → embed_dim
        # Player embedding: 4 players → embed_dim
        # Positional encoding: trick number (0-12) + position in trick (0-3)
        raise NotImplementedError

    def forward(
        self,
        card_ids: torch.Tensor,       # (batch, seq_len) int64, 0-51
        player_ids: torch.Tensor,     # (batch, seq_len) int64, 0-3
        padding_mask: torch.Tensor,   # (batch, seq_len) bool, True = padding
    ) -> torch.Tensor:
        """Returns (batch, embed_dim) pooled representation of the history."""
        raise NotImplementedError

    def _build_positional_encoding(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Return sinusoidal positional encodings of shape (seq_len, embed_dim)."""
        raise NotImplementedError


class TransformerEncoder(nn.Module):
    """
    Full state encoder combining:
      - Static features (hand, bids, tricks won, position) via MLP
      - History sequence (cards played in order) via CardHistoryEncoder
      - Fusion of both into a single latent vector
    """

    def __init__(
        self,
        static_dim: int,     # dimension of non-sequential state features
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        hidden_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.static_encoder = MLPEncoder(static_dim, embed_dim, hidden_dim, dropout)
        self.history_encoder = CardHistoryEncoder(embed_dim, num_heads, num_layers, dropout=dropout)

        # Fusion layer: concatenate static + history → embed_dim
        raise NotImplementedError

    def forward(
        self,
        static_features: torch.Tensor,    # (batch, static_dim)
        card_history_ids: torch.Tensor,   # (batch, seq_len) int64
        player_history_ids: torch.Tensor, # (batch, seq_len) int64
        history_mask: torch.Tensor,       # (batch, seq_len) bool
    ) -> torch.Tensor:
        """Returns fused latent (batch, embed_dim)."""
        raise NotImplementedError
