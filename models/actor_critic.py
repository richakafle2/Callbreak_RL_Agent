"""
actor_critic.py
---------------
Combined Actor-Critic network for Call Break.

Architecture:
  Encoder (MLP or Transformer)
       │
  ┌────┴────┐
  ▼         ▼
Policy    Value
Head      Head
  │         │
π(a|s)    V(s)

The policy head is split into two sub-heads:
  - BidHead  : outputs logits over 13 bid actions
  - PlayHead : outputs logits over 52 card actions (masked by legal plays)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from models.encoder import MLPEncoder, TransformerEncoder


class BidHead(nn.Module):
    """
    Maps latent encoding → distribution over bids 1-13.
    During inference, only bids in [min_bid, max_bid] are legal.
    """

    def __init__(self, embed_dim: int, hidden_dim: int, num_bids: int = 13):
        super().__init__()
        # TODO: define layers
        raise NotImplementedError

    def forward(
        self, latent: torch.Tensor, bid_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        latent: (batch, embed_dim)
        bid_mask: (batch, 13) bool — True where bid is legal
        Returns log-probabilities (batch, 13) with illegal actions masked to -inf.
        """
        raise NotImplementedError


class PlayHead(nn.Module):
    """
    Maps latent encoding → distribution over 52 card plays.
    Illegal plays are masked out before softmax.
    """

    def __init__(self, embed_dim: int, hidden_dim: int, num_cards: int = 52):
        super().__init__()
        # TODO: define layers
        raise NotImplementedError

    def forward(
        self, latent: torch.Tensor, legal_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        latent: (batch, embed_dim)
        legal_mask: (batch, 52) bool — True where card play is legal
        Returns log-probabilities (batch, 52) with illegal cards masked to -inf.
        """
        raise NotImplementedError

    def _apply_legal_mask(
        self, logits: torch.Tensor, legal_mask: torch.Tensor
    ) -> torch.Tensor:
        """Set logits of illegal actions to -1e9 before softmax."""
        raise NotImplementedError


class ValueHead(nn.Module):
    """Maps latent encoding → scalar state value estimate V(s)."""

    def __init__(self, embed_dim: int, hidden_dim: int):
        super().__init__()
        # TODO: define layers (MLP → scalar)
        raise NotImplementedError

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """latent: (batch, embed_dim) → (batch, 1)"""
        raise NotImplementedError


class ActorCritic(nn.Module):
    """
    Full Actor-Critic model combining encoder + policy heads + value head.
    Handles both bidding and playing phases via `phase` argument.
    """

    def __init__(
        self,
        obs_dim: int,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
        encoder_type: str = "transformer",  # "mlp" | "transformer"
    ):
        super().__init__()
        self.encoder_type = encoder_type

        if encoder_type == "transformer":
            self.encoder = TransformerEncoder(
                static_dim=obs_dim,
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                hidden_dim=hidden_dim,
                dropout=dropout,
            )
        else:
            self.encoder = MLPEncoder(obs_dim, embed_dim, hidden_dim, dropout)

        self.bid_head  = BidHead(embed_dim, hidden_dim)
        self.play_head = PlayHead(embed_dim, hidden_dim)
        self.value_head = ValueHead(embed_dim, hidden_dim)

    def forward(
        self,
        obs: torch.Tensor,
        legal_mask: torch.Tensor,
        phase: str = "play",                  # "bid" | "play"
        # Transformer-specific inputs (ignored for MLP encoder):
        card_history: Optional[torch.Tensor] = None,
        player_history: Optional[torch.Tensor] = None,
        history_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          log_probs : (batch, action_dim)  — log π(a|s), masked
          value     : (batch, 1)           — V(s)
        """
        raise NotImplementedError

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        legal_mask: torch.Tensor,
        phase: str = "play",
        deterministic: bool = False,
        **encoder_kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample (or argmax) an action; return action, log_prob, entropy, value.

        Returns:
          action    : (batch,)  sampled action indices
          log_prob  : (batch,)  log π(action | s)
          entropy   : (batch,)  policy entropy — used in PPO loss
          value     : (batch,)  V(s)
        """
        raise NotImplementedError

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        legal_mask: torch.Tensor,
        phase: str = "play",
        **encoder_kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Re-evaluate stored actions during PPO update.

        Returns:
          log_prob  : (batch,)
          entropy   : (batch,)
          value     : (batch,)
        """
        raise NotImplementedError
