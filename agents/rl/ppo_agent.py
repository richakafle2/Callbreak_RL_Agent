"""
ppo_agent.py
------------
RL-trained agent that wraps the ActorCritic model for use as a
Call Break player (implements BaseAgent interface).
Used during evaluation and as the opponent in self-play.

ASSUMPTIONS (flag if any of these don't match the real classes):
  - BaseAgent.__init__(self, player_id, name) exists, matching the call in
    __init__ below.
  - StateEncoder.encode(observation_dict, player_id) -> np.ndarray of shape
    (OBS_DIM,), matching how CallBreakEnv calls it.
  - observation dicts (from Round.get_observation) contain:
      "legal_plays"  -> list of Card objects, each with a .index attribute
      "legal_bids"   -> list of ints in [1, 13] (falls back to range(1,14)
                        if absent, mirroring CallBreakEnv's own fallback)
  - ActorCritic exposes `bid_forward(obs_tensor) -> (logits[1,13], value[1])`
    and `play_forward(obs_tensor) -> (logits[1,52], value[1])`, matching the
    two-head design (BidHead/PlayHead) described for the model skeleton.
  - Checkpoints are saved either as a raw state_dict or as a dict containing
    a "model_state_dict" key (same convention used elsewhere in this project).
"""

import torch
import numpy as np
from typing import Dict, Optional

from agents.base_agent import BaseAgent
from models.actor_critic import ActorCritic
from utils.state_encoder import StateEncoder


NUM_BID_ACTIONS = 13
NUM_CARDS = 52
MASK_FILL_VALUE = -1e9


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
        self.model.to(device)
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
        obs_tensor = self._obs_to_tensor(observation)
        legal_mask = self._build_bid_mask(observation)
        action_index = self._forward(obs_tensor, legal_mask, phase="bid")
        return action_index + 1

    def play(self, observation: Dict) -> int:
        """
        Encode observation, run PlayHead with legal mask, return card index.
        """
        obs_tensor = self._obs_to_tensor(observation)
        legal_mask = self._build_play_mask(observation)
        return self._forward(obs_tensor, legal_mask, phase="play")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _obs_to_tensor(self, observation: Dict) -> torch.Tensor:
        """Encode observation dict to a (1, obs_dim) tensor."""
        encoded = self.encoder.encode(observation, player_id=self.player_id)
        tensor = torch.as_tensor(encoded, dtype=torch.float32, device=self.device)
        return tensor.unsqueeze(0)

    def _build_bid_mask(self, observation: Dict) -> torch.Tensor:
        """Return (1, 13) bool mask for legal bids."""
        mask = torch.zeros((1, NUM_BID_ACTIONS), dtype=torch.bool, device=self.device)
        legal_bids = observation.get("legal_bids", list(range(1, NUM_BID_ACTIONS + 1)))
        for bid_value in legal_bids:
            mask[0, bid_value - 1] = True
        return mask

    def _build_play_mask(self, observation: Dict) -> torch.Tensor:
        """Return (1, 52) bool mask for legal card plays."""
        mask = torch.zeros((1, NUM_CARDS), dtype=torch.bool, device=self.device)
        for card in observation.get("legal_plays", []):
            mask[0, card.index] = True
        return mask

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
        if phase == "bid":
            logits, _ = self.model.bid_forward(obs_tensor)
        elif phase == "play":
            logits, _ = self.model.play_forward(obs_tensor)
        else:
            raise ValueError(f"Unknown phase '{phase}', expected 'bid' or 'play'.")

        masked_logits = logits.masked_fill(~legal_mask, MASK_FILL_VALUE)

        if self.deterministic:
            action = torch.argmax(masked_logits, dim=-1)
        else:
            dist = torch.distributions.Categorical(logits=masked_logits)
            action = dist.sample()

        return int(action.item())

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
        resolved_device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        model = ActorCritic(**model_config)
        checkpoint = torch.load(checkpoint_path, map_location=resolved_device)
        state_dict = (
            checkpoint["model_state_dict"]
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
            else checkpoint
        )
        model.load_state_dict(state_dict)

        encoder = StateEncoder()

        return cls(
            player_id=player_id,
            model=model,
            encoder=encoder,
            device=resolved_device,
            deterministic=True,
            name="PPO",
        )