"""
self_play.py
------------
Self-play pool manager.

Maintains a pool of past model checkpoints. During the self-play curriculum
stage, opponents are sampled from this pool (uniform or prioritised by Elo).
The current model is added to the pool periodically.
"""

import os
import random
import torch
from typing import List, Optional, Tuple
from models.actor_critic import ActorCritic


class SelfPlayPool:
    """
    Fixed-size ring buffer of past agent checkpoints.
    Supports uniform and prioritised (Elo-weighted) sampling.
    """

    def __init__(
        self,
        pool_size: int = 20,
        sample_strategy: str = "uniform",  # "uniform" | "prioritized"
        checkpoint_dir: str = "checkpoints/pool",
        seed: Optional[int] = None,
    ):
        self.pool_size = pool_size
        self.sample_strategy = sample_strategy
        self.checkpoint_dir = checkpoint_dir
        self._rng = random.Random(seed)

        os.makedirs(checkpoint_dir, exist_ok=True)

        # Metadata for each pool entry
        self._pool: List[dict] = []
        # Each entry: {
        #   'path': str,        checkpoint file path
        #   'step': int,        training step when added
        #   'elo': float,       current Elo rating
        #   'win_rate': float,  running win rate vs current agent
        # }

    # ------------------------------------------------------------------
    # Pool management
    # ------------------------------------------------------------------

    def add(self, model: ActorCritic, step: int, elo: float = 1000.0) -> str:
        """
        Save `model` to disk and add it to the pool.
        If pool is full, evict the oldest entry.
        Returns the path where the checkpoint was saved.
        """
        raise NotImplementedError

    def _evict_oldest(self) -> None:
        """Remove the oldest entry from the pool (and delete the file)."""
        raise NotImplementedError

    def update_elo(self, path: str, new_elo: float) -> None:
        """Update the Elo rating for the checkpoint at `path`."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(self, n: int = 1) -> List[str]:
        """
        Sample `n` checkpoint paths from the pool.
        'uniform' strategy: uniform at random.
        'prioritized' strategy: weights proportional to Elo rating.
        Returns list of checkpoint file paths.
        """
        raise NotImplementedError

    def load_agent(self, path: str, model_config: dict, device: torch.device) -> ActorCritic:
        """Load and return an ActorCritic model from a checkpoint path."""
        raise NotImplementedError

    def sample_opponents(
        self, n: int, model_config: dict, device: torch.device
    ) -> List[ActorCritic]:
        """
        Sample `n` opponents from the pool and return them as loaded models.
        Convenience wrapper around sample() + load_agent().
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._pool)

    def is_empty(self) -> bool:
        return len(self._pool) == 0

    def summary(self) -> List[dict]:
        """Return a copy of pool metadata (no model objects)."""
        return list(self._pool)
