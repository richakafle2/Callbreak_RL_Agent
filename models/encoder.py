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

import math

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
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, obs_dim) → (batch, embed_dim)"""
        return self.net(x)


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
        self.card_embedding = nn.Embedding(52, embed_dim)
        # Player embedding: 4 players → embed_dim
        self.player_embedding = nn.Embedding(4, embed_dim)

        self.input_norm = nn.LayerNorm(embed_dim)
        self.input_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(
        self,
        card_ids: torch.Tensor,       # (batch, seq_len) int64, 0-51
        player_ids: torch.Tensor,     # (batch, seq_len) int64, 0-3
        padding_mask: torch.Tensor,   # (batch, seq_len) bool, True = padding
    ) -> torch.Tensor:
        """Returns (batch, embed_dim) pooled representation of the history."""
        batch_size, seq_len = card_ids.shape

        # No cards played yet (start of round) — nothing to attend over.
        if seq_len == 0:
            return torch.zeros(batch_size, self.embed_dim, device=card_ids.device)

        # nn.TransformerEncoder produces NaN (or, with the nested-tensor
        # fast path used in eval mode, raises a RuntimeError) for any batch
        # row whose src_key_padding_mask is entirely True — i.e. zero valid
        # keys for that row to attend to. This isn't a rare edge case here:
        # every round starts with zero cards played, so a fully-padded row
        # happens on literally every bid at the start of every round. If
        # left unhandled, NaN would flow into ppo_update()'s backward pass
        # (which runs this same forward with grad enabled, unlike
        # PPOAgent's @torch.no_grad() inference calls) and silently corrupt
        # every weight in the model via torch.where-style NaN-gradient
        # leakage if "fixed" only after the fact.
        #
        # Fix: for any fully-padded row, unmask exactly one dummy position
        # so attention always has >=1 valid key (self-attention on a single
        # token is well-defined and NaN-free). The pooling step below still
        # uses the ORIGINAL padding_mask, so that dummy position is excluded
        # from the output — it influences nothing, it just keeps softmax
        # from dividing by zero valid keys.
        fully_padded = padding_mask.all(dim=1)  # (batch,)
        attn_mask = padding_mask.clone()
        if fully_padded.any():
            attn_mask[fully_padded, 0] = False

        card_emb = self.card_embedding(card_ids)      # (batch, seq_len, embed_dim)
        player_emb = self.player_embedding(player_ids)  # (batch, seq_len, embed_dim)
        pos_emb = self._build_positional_encoding(seq_len, card_ids.device)  # (seq_len, embed_dim)
        pos_emb = pos_emb.unsqueeze(0)  # (1, seq_len, embed_dim) — broadcasts over batch

        x = card_emb + player_emb + pos_emb
        x = self.input_norm(x)
        x = self.input_dropout(x)

        # nn.TransformerEncoder treats True in src_key_padding_mask as "ignore this position".
        encoded = self.transformer(x, src_key_padding_mask=attn_mask)  # (batch, seq_len, embed_dim)

        # Masked mean-pool over the real (non-padding) positions. Uses the
        # ORIGINAL padding_mask (not attn_mask), so the dummy unmasked
        # position for fully-padded rows contributes nothing here — those
        # rows correctly pool to zero, matching "no history yet".
        valid = (~padding_mask).unsqueeze(-1).to(encoded.dtype)  # (batch, seq_len, 1)
        summed = (encoded * valid).sum(dim=1)                     # (batch, embed_dim)
        counts = valid.sum(dim=1).clamp(min=1.0)                  # (batch, 1) avoid /0
        pooled = summed / counts

        return pooled

    def _build_positional_encoding(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Return sinusoidal positional encodings of shape (seq_len, embed_dim)."""
        pe = torch.zeros(seq_len, self.embed_dim, device=device)
        position = torch.arange(0, seq_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.embed_dim, 2, dtype=torch.float, device=device)
            * (-math.log(10000.0) / self.embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe


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
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(
        self,
        static_features: torch.Tensor,    # (batch, static_dim)
        card_history_ids: torch.Tensor,   # (batch, seq_len) int64
        player_history_ids: torch.Tensor, # (batch, seq_len) int64
        history_mask: torch.Tensor,       # (batch, seq_len) bool
    ) -> torch.Tensor:
        """Returns fused latent (batch, embed_dim)."""
        static_latent = self.static_encoder(static_features)
        history_latent = self.history_encoder(card_history_ids, player_history_ids, history_mask)

        fused = torch.cat([static_latent, history_latent], dim=-1)
        return self.fusion(fused)
