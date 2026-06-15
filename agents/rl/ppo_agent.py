"""
ppo_agent.py
------------
RL-trained agent that wraps the ActorCritic model for use as a
Call Break player (implements BaseAgent interface).
Used during evaluation and as the opponent in self-play.
"""

import torch
import numpy as np
from typing import Dict, Optional

from agents.base_agent import BaseAgent
from models.actor_critic import ActorCritic
from utils.state_encoder import StateEncoder


class PPOAgent(BaseAgent):
    """
    Inference wrapper around a trained ActorCritic model.
    Does not update weights — use PPOTrainer for training.
    """

    def __init__(
        self,
        player_id: int,
        model: ActorCritic,
        encoder: StateEncoder,
        device: torch.device = torch.device("cpu"),
        deterministic: bool = True,
        name: str = "PPO",
    ):
        super().__init__(player_id, name)
        self.model = model
        self.model.eval()
        self.encoder = encoder
        self.device = device
        self.deterministic = deterministic

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def bid(self, observation: Dict) -> int:
        """
        Encode observation, run BidHead, return highest-probability legal bid.
        Bid values are 1-13; action indices are 0-12.
        """
        raise NotImplementedError

    def play(self, observation: Dict) -> int:
        """
        Encode observation, run PlayHead with legal mask, return card index.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _obs_to_tensor(self, observation: Dict) -> torch.Tensor:
        """Encode observation dict to a (1, obs_dim) tensor."""
        raise NotImplementedError

    def _build_bid_mask(self, observation: Dict) -> torch.Tensor:
        """Return (1, 13) bool mask for legal bids."""
        raise NotImplementedError

    def _build_play_mask(self, observation: Dict) -> torch.Tensor:
        """Return (1, 52) bool mask for legal card plays."""
        raise NotImplementedError

    @torch.no_grad()
    def _forward(
        self,
        obs_tensor: torch.Tensor,
        legal_mask: torch.Tensor,
        phase: str,
    ) -> int:
        """
        Run a single forward pass and sample/argmax an action.
        Returns action as an int.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Checkpoint loading
    # ------------------------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        model_config: Dict,
        player_id: int = 0,
        device: Optional[torch.device] = None,
    ) -> "PPOAgent":
        """
        Load a PPOAgent from a saved checkpoint file.
        model_config should match the architecture used during training.
        """
        raise NotImplementedError
