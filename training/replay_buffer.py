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
        history_len: int = 52,
    ):
        self.rollout_steps = rollout_steps
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device
        self.history_len = history_len

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
        # History tensors for the transformer encoder path. Unused (but
        # still allocated and stored) when encoder_type is "mlp" — cheap to
        # keep unconditional here since ActorCritic's MLP branch ignores
        # them anyway (see ActorCritic._encode), and this keeps the buffer
        # encoder-agnostic rather than branching on config.
        self.card_history_ids   = np.zeros((rollout_steps, num_envs, history_len), dtype=np.int64)
        self.player_history_ids = np.zeros((rollout_steps, num_envs, history_len), dtype=np.int64)
        self.history_masks      = np.ones((rollout_steps, num_envs, history_len),  dtype=np.bool_)

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
        card_history_ids:   Optional[np.ndarray] = None,  # (num_envs, history_len)
        player_history_ids: Optional[np.ndarray] = None,  # (num_envs, history_len)
        history_mask:        Optional[np.ndarray] = None,  # (num_envs, history_len)
    ) -> None:
        """Insert one step of experience from all environments."""
        if self._full:
            raise RuntimeError(
                "RolloutBuffer is full — call reset() before collecting a new rollout."
            )

        self.observations[self._ptr] = obs
        self.actions[self._ptr]      = action
        self.log_probs[self._ptr]    = log_prob
        self.rewards[self._ptr]      = reward
        self.values[self._ptr]       = value
        self.dones[self._ptr]        = done
        self.legal_masks[self._ptr]  = legal_mask
        self.phases[self._ptr]       = phase
        if card_history_ids is not None:
            self.card_history_ids[self._ptr] = card_history_ids
        if player_history_ids is not None:
            self.player_history_ids[self._ptr] = player_history_ids
        if history_mask is not None:
            self.history_masks[self._ptr] = history_mask

        self._ptr += 1
        if self._ptr >= self.rollout_steps:
            self._full = True

    def is_full(self) -> bool:
        return self._full

    def reset(self) -> None:
        """Clear the buffer for the next rollout."""
        self._ptr = 0
        self._full = False
        self.advantages = None
        self.returns = None
        # Not strictly necessary to zero the arrays (they get overwritten on
        # the next add() pass before being read), but it avoids ever reading
        # stale data from a previous rollout if a bug causes a partial fill.
        self.observations.fill(0)
        self.actions.fill(0)
        self.log_probs.fill(0)
        self.rewards.fill(0)
        self.values.fill(0)
        self.dones.fill(0)
        self.legal_masks.fill(False)
        self.phases.fill(None)
        self.card_history_ids.fill(0)
        self.player_history_ids.fill(0)
        self.history_masks.fill(True)  # True = padding; matches the "no history yet" default

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
        advantages = np.zeros((self.rollout_steps, self.num_envs), dtype=np.float32)
        last_advantage = np.zeros(self.num_envs, dtype=np.float32)

        for t in reversed(range(self.rollout_steps)):
            if t == self.rollout_steps - 1:
                next_value = last_values
                next_done = last_dones
            else:
                next_value = self.values[t + 1]
                next_done = self.dones[t + 1]

            not_done = 1.0 - next_done
            delta = self.rewards[t] + self.gamma * next_value * not_done - self.values[t]
            last_advantage = delta + self.gamma * self.gae_lambda * not_done * last_advantage
            advantages[t] = last_advantage

        self.advantages = advantages
        self.returns = advantages + self.values

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
        if self.advantages is None or self.returns is None:
            raise RuntimeError(
                "Call compute_returns_and_advantages() before get_batches()."
            )

        obs         = self._flatten(self.observations)
        actions     = self._flatten(self.actions)
        log_probs   = self._flatten(self.log_probs)
        advantages  = self._flatten(self.advantages)
        returns     = self._flatten(self.returns)
        legal_masks = self._flatten(self.legal_masks)
        phases      = self._flatten(self.phases)
        card_history_ids   = self._flatten(self.card_history_ids)
        player_history_ids = self._flatten(self.player_history_ids)
        history_masks       = self._flatten(self.history_masks)

        # Normalize advantages for stability (standard PPO trick), guarding
        # against a degenerate std of 0 on a tiny/uniform rollout.
        adv_std = advantages.std()
        if adv_std > 1e-8:
            advantages = (advantages - advantages.mean()) / (adv_std + 1e-8)

        n_samples = obs.shape[0]
        indices = np.arange(n_samples)
        np.random.shuffle(indices)

        for start in range(0, n_samples, batch_size):
            end = start + batch_size
            batch_idx = indices[start:end]

            yield {
                "obs":            torch.as_tensor(obs[batch_idx], dtype=torch.float32, device=self.device),
                "actions":        torch.as_tensor(actions[batch_idx], dtype=torch.int64, device=self.device),
                "log_probs_old":  torch.as_tensor(log_probs[batch_idx], dtype=torch.float32, device=self.device),
                "advantages":     torch.as_tensor(advantages[batch_idx], dtype=torch.float32, device=self.device),
                "returns":        torch.as_tensor(returns[batch_idx], dtype=torch.float32, device=self.device),
                "legal_masks":    torch.as_tensor(legal_masks[batch_idx], dtype=torch.bool, device=self.device),
                "card_history_ids":   torch.as_tensor(card_history_ids[batch_idx], dtype=torch.int64, device=self.device),
                "player_history_ids": torch.as_tensor(player_history_ids[batch_idx], dtype=torch.int64, device=self.device),
                "history_masks":       torch.as_tensor(history_masks[batch_idx], dtype=torch.bool, device=self.device),
                # Phases stay as a plain numpy array of strings ("bid"/"play"),
                # not a tensor — ActorCritic.evaluate_actions() takes `phase`
                # as a single string, so ppo_update() must split each batch
                # into a bid-subset and play-subset before calling it.
                "phases":         phases[batch_idx],
            }

    def _flatten(self, arr: np.ndarray) -> np.ndarray:
        """Reshape (rollout_steps, num_envs, ...) → (rollout_steps*num_envs, ...)."""
        return arr.reshape(-1, *arr.shape[2:])

    def __len__(self) -> int:
        return self.rollout_steps * self.num_envs if self._full else self._ptr * self.num_envs