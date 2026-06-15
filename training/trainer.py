"""
trainer.py
----------
PPO training loop for the Call Break agent.

Flow:
  1. Collect rollout_steps of experience from num_envs parallel envs.
  2. Compute GAE returns and advantages.
  3. Run num_epochs of mini-batch PPO updates.
  4. Repeat until total_timesteps reached.
  5. Periodically evaluate and save checkpoints.
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Optional

from models.actor_critic import ActorCritic
from training.replay_buffer import RolloutBuffer
from training.curriculum import CurriculumScheduler
from evaluation.evaluator import Evaluator
from utils.logger import Logger


class PPOTrainer:
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Model
        self.model = ActorCritic(
            obs_dim=config["obs_dim"],
            **config["model"],
        ).to(self.device)

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config["ppo"]["learning_rate"],
        )

        self.buffer = RolloutBuffer(
            rollout_steps=config["ppo"]["rollout_steps"],
            num_envs=config["training"]["num_envs"],
            obs_dim=config["obs_dim"],
            action_dim=52,
            gamma=config["ppo"]["gamma"],
            gae_lambda=config["ppo"]["gae_lambda"],
            device=self.device,
        )

        self.curriculum = CurriculumScheduler(config["curriculum"])
        self.evaluator  = Evaluator(config["evaluation"])
        self.logger     = Logger(config["training"]["log_dir"])

        self.global_step: int = 0
        self.best_eval_score: float = -float("inf")

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self, total_timesteps: int) -> None:
        """
        Outer training loop.
        1. Build parallel environments via _make_envs().
        2. Loop: collect_rollout() → ppo_update() → maybe evaluate/save.
        3. Advance curriculum based on eval metrics.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Rollout collection
    # ------------------------------------------------------------------

    def collect_rollout(self, envs) -> Dict:
        """
        Run all parallel environments for rollout_steps steps.
        Stores experience in self.buffer.
        Returns stats dict: {'mean_reward', 'episode_lengths', ...}.

        Steps:
          1. For each step t in rollout_steps:
             a. Get obs from envs.
             b. Call model.get_action_and_value(obs, legal_masks).
             c. Step envs with chosen actions.
             d. Store transition in buffer.
          2. Compute final values for GAE bootstrap.
          3. Call buffer.compute_returns_and_advantages(last_values, last_dones).
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

    def ppo_update(self) -> Dict:
        """
        Run num_epochs passes over the rollout buffer doing mini-batch updates.
        Returns loss statistics dict.

        PPO loss = policy_loss + value_loss_coeff * value_loss - entropy_coeff * entropy

        Policy loss (clipped):
          ratio = exp(new_log_prob - old_log_prob)
          L_clip = -min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)

        Value loss:
          L_v = 0.5 * (V(s) - R)²  (optionally clipped)

        Entropy bonus:
          L_e = mean(entropy)
        """
        raise NotImplementedError

    def _compute_policy_loss(
        self,
        new_log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
    ) -> torch.Tensor:
        """Compute clipped PPO policy loss."""
        raise NotImplementedError

    def _compute_value_loss(
        self,
        values: torch.Tensor,
        returns: torch.Tensor,
        old_values: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute (optionally clipped) value function loss."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Environment management
    # ------------------------------------------------------------------

    def _make_envs(self, opponents: List) -> object:
        """
        Create a vectorised set of CallBreakEnv instances.
        Each env has the same opponent configuration from curriculum.
        Returns a VecEnv-like wrapper.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str, metadata: Optional[Dict] = None) -> None:
        """Save model weights, optimizer state, and training metadata."""
        raise NotImplementedError

    def load_checkpoint(self, path: str) -> Dict:
        """Load checkpoint; return metadata dict."""
        raise NotImplementedError

    def _save_if_best(self, eval_score: float, step: int) -> None:
        """Save as best.pt if eval_score exceeds self.best_eval_score."""
        raise NotImplementedError
