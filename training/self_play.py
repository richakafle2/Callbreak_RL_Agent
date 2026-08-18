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

        self._load_existing_checkpoints()

    def _load_existing_checkpoints(self) -> None:
        """
        Populate the in-memory pool from any checkpoint files already
        sitting in checkpoint_dir (e.g. left over from a previous process).
        Without this, restarting training silently drops all past-self
        opponents even though their files are still on disk. Entries are
        sorted by step so eviction order (oldest-first) stays correct.
        """
        if not os.path.isdir(self.checkpoint_dir):
            return

        found = []
        for fname in os.listdir(self.checkpoint_dir):
            if not fname.endswith(".pt"):
                continue
            path = os.path.join(self.checkpoint_dir, fname)
            try:
                step = int(fname[len("step_"):-len(".pt")])
            except ValueError:
                step = 0
            found.append({"path": path, "step": step, "elo": 1000.0, "win_rate": 0.0})

        found.sort(key=lambda e: e["step"])
        # Respect pool_size in case more files exist on disk than the
        # configured pool size (e.g. pool_size was lowered between runs).
        self._pool = found[-self.pool_size:] if self.pool_size else found

    # ------------------------------------------------------------------
    # Pool management
    # ------------------------------------------------------------------

    def add(self, model: ActorCritic, step: int, elo: float = 1000.0) -> str:
        """
        Save `model` to disk and add it to the pool.
        If pool is full, evict the oldest entry.
        Returns the path where the checkpoint was saved.
        """
        path = os.path.join(self.checkpoint_dir, f"step_{step}.pt")
        torch.save({"model_state_dict": model.state_dict(), "step": step}, path)

        if len(self._pool) >= self.pool_size:
            self._evict_oldest()

        self._pool.append({
            "path": path,
            "step": step,
            "elo": elo,
            "win_rate": 0.0,
        })
        return path

    def _evict_oldest(self) -> None:
        """Remove the oldest entry from the pool (and delete the file)."""
        if not self._pool:
            return
        oldest = self._pool.pop(0)  # entries are appended in chronological order
        try:
            os.remove(oldest["path"])
        except FileNotFoundError:
            pass

    def update_elo(self, path: str, new_elo: float) -> None:
        """Update the Elo rating for the checkpoint at `path`."""
        for entry in self._pool:
            if entry["path"] == path:
                entry["elo"] = new_elo
                return
        raise ValueError(f"No pool entry found for path: {path}")

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
        if self.is_empty():
            raise RuntimeError("Cannot sample from an empty SelfPlayPool.")

        paths = [entry["path"] for entry in self._pool]

        if self.sample_strategy == "uniform":
            return self._rng.choices(paths, k=n)
        elif self.sample_strategy == "prioritized":
            weights = [max(entry["elo"], 1e-6) for entry in self._pool]
            return self._rng.choices(paths, weights=weights, k=n)
        else:
            raise ValueError(
                f"Unknown sample_strategy '{self.sample_strategy}', "
                "expected 'uniform' or 'prioritized'."
            )

    def load_agent(self, path: str, model_config: dict, device: torch.device) -> ActorCritic:
        """Load and return an ActorCritic model from a checkpoint path."""
        model = ActorCritic(**model_config)
        checkpoint = torch.load(path, map_location=device)
        state_dict = (
            checkpoint["model_state_dict"]
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
            else checkpoint
        )
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model

    def sample_opponents(
        self, n: int, model_config: dict, device: torch.device
    ) -> List[ActorCritic]:
        """
        Sample `n` opponents from the pool and return them as loaded models.
        Convenience wrapper around sample() + load_agent().
        """
        paths = self.sample(n)
        return [self.load_agent(path, model_config, device) for path in paths]

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