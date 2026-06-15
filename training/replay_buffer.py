"""
replay_buffer.py
----------------
PPO rollout buffer — stores trajectories collected from parallel environments
and computes Generalised Advantage Estimation (GAE) before each policy update.
"""

import numpy as np
import torch
from typing import Dict, Generator, Optional, Tuple


class RolloutBuffer:
    """
    Stores experience from `num_envs` parallel environments for `rollout_steps`
    steps. After collection, call compute_returns_and_advantages() then
    iterate with get_batches() to feed the PPO update.
    """

    def __init__(
        self,
        rollout_steps: int,
        num_envs: int,
        obs_dim: int,
        action_dim: int,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        device: torch.device = torch.device("cpu"),
    ):
        self.rollout_steps = rollout_steps
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device

        self._ptr = 0       # current insertion pointer
        self._full = False

        # Allocated storage (pre-allocated for speed)
        self.observations  = np.zeros((rollout_steps, num_envs, obs_dim),  dtype=np.float32)
        self.actions       = np.zeros((rollout_steps, num_envs),           dtype=np.int64)
        self.log_probs     = np.zeros((rollout_steps, num_envs),           dtype=np.float32)
        self.rewards       = np.zeros((rollout_steps, num_envs),           dtype=np.float32)
        self.values        = np.zeros((rollout_steps, num_envs),           dtype=np.float32)
        self.dones         = np.zeros((rollout_steps, num_envs),           dtype=np.float32)
        self.legal_masks   = np.zeros((rollout_steps, num_envs, action_dim), dtype=np.bool_)
        self.phases        = np.empty((rollout_steps, num_envs),           dtype=object)

        # Computed after collection
        self.advantages: Optional[np.ndarray] = None
        self.returns:    Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Insertion
    # ------------------------------------------------------------------

    def add(
        self,
        obs:        np.ndarray,   # (num_envs, obs_dim)
        action:     np.ndarray,   # (num_envs,)
        log_prob:   np.ndarray,   # (num_envs,)
        reward:     np.ndarray,   # (num_envs,)
        value:      np.ndarray,   # (num_envs,)
        done:       np.ndarray,   # (num_envs,) float 0/1
        legal_mask: np.ndarray,   # (num_envs, action_dim)
        phase:      np.ndarray,   # (num_envs,) str "bid"/"play"
    ) -> None:
        """Insert one step of experience from all environments."""
        raise NotImplementedError

    def is_full(self) -> bool:
        return self._full

    def reset(self) -> None:
        """Clear the buffer for the next rollout."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # GAE computation
    # ------------------------------------------------------------------

    def compute_returns_and_advantages(self, last_values: np.ndarray, last_dones: np.ndarray) -> None:
        """
        Compute GAE advantages and discounted returns in-place.

        Args:
            last_values : (num_envs,) — V(s_T) from the final state
            last_dones  : (num_envs,) — whether the final state is terminal

        Sets:
            self.advantages : (rollout_steps, num_envs)
            self.returns    : (rollout_steps, num_envs)

        GAE formula (per env):
            δ_t = r_t + γ * V(s_{t+1}) * (1 - done_t) - V(s_t)
            A_t = δ_t + (γ * λ) * (1 - done_t) * A_{t+1}
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Batch iteration
    # ------------------------------------------------------------------

    def get_batches(self, batch_size: int) -> Generator[Dict[str, torch.Tensor], None, None]:
        """
        Yield shuffled mini-batches of experience as torch tensors.
        Call compute_returns_and_advantages() before iterating.

        Each batch dict contains:
          'obs', 'actions', 'log_probs_old', 'advantages',
          'returns', 'legal_masks', 'phases'
        """
        raise NotImplementedError

    def _flatten(self, arr: np.ndarray) -> np.ndarray:
        """Reshape (rollout_steps, num_envs, ...) → (rollout_steps*num_envs, ...)."""
        return arr.reshape(-1, *arr.shape[2:])

    def __len__(self) -> int:
        return self.rollout_steps * self.num_envs if self._full else self._ptr * self.num_envs
