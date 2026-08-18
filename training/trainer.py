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

IMPORTANT GAPS FILLED IN THAT WEREN'T IN THE ORIGINAL STUB (flag if any of
these should be designed differently):

  1. CallBreakEnv is a single-agent Gymnasium env (no native vectorization).
     `_make_envs()` builds a small `_SyncVectorEnv` wrapper around a plain
     Python list of CallBreakEnv instances and steps them all synchronously
     each rollout step. If you want true parallelism (subprocess/thread
     workers), this wrapper is the piece to replace.

  2. The environment can have different envs sitting in different phases
     ("bid" vs "play") at the *same* rollout step, but ActorCritic's
     forward/get_action_and_value/evaluate_actions all take a single scalar
     `phase` argument, not a per-sample one. Both collect_rollout() and
     ppo_update() handle this by grouping envs/samples by phase and making
     one forward pass per phase-subset, then recombining. This is the
     central design choice this file makes.

  3. Bid actions and play actions both live in the env's unified
     Discrete(52) action space (bid action index == bid_value - 1, which
     conveniently lines up 1:1 with BidHead's own 13-wide logits). Because
     of that, legal masks are stored at width 52 in the buffer regardless of
     phase, but sliced to width 13 before being handed to BidHead.

  4. SelfPlayPool isn't constructed in __init__ (only CurriculumScheduler,
     Evaluator, and Logger are). It's built lazily inside train() the first
     time the curriculum reaches the self_play stage, using
     config.get("self_play", {}) for pool_size/sample_strategy/checkpoint_dir.

  5. Evaluator's exact method signature wasn't available to me here; I've
     assumed `evaluator.evaluate(model, opponents, num_games) -> {"win_rate":
     float, ...}`. If evaluator.py's real signature differs, `_run_evaluation`
     is the one place that needs updating.

  6. [FIX] Curriculum advancement bug: _run_evaluation() used to feed the
     *global* evaluation-suite win rate (config['evaluation']['opponents'] --
     random/safe_bet/safe_play/basic_bet blended together) into
     curriculum.record_result(), which is what should_advance() checks
     against each stage's threshold_winrate. That meant a stage could only
     ever advance once the agent was already beating the ENTIRE fixed suite
     at the threshold rate -- including opponents that stage doesn't even
     train against. In practice this meant "mixed" (threshold 0.65) never
     advanced, since overall win rate landed at 64.1% / 62.0% across two
     full 10M-step runs -- just under the bar, for a bar that was never the
     right bar to check for that stage.

     Fixed by running a SECOND, stage-specific evaluation using
     evaluator.run_games() directly against self._get_current_opponents()
     (the actual opponents this stage trains against, including live
     self-play model instances -- run_games() takes pre-instantiated agents,
     so this works for self_play too), and feeding THAT win rate into
     curriculum.record_result(). The global-suite eval is kept exactly as
     before, purely for logging and best-checkpoint selection.
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Optional, Tuple

from models.actor_critic import ActorCritic
from training.replay_buffer import RolloutBuffer
from training.curriculum import CurriculumScheduler
from training.self_play import SelfPlayPool
from evaluation.evaluator import Evaluator
from utils.logger import Logger
from utils.state_encoder import StateEncoder, MAX_HISTORY_LEN
from environment.callbreak_env import CallBreakEnv


NUM_CARDS = 52
NUM_BID_ACTIONS = 13


class _SyncVectorEnv:
    """Minimal synchronous vector-env wrapper around a list of CallBreakEnv
    instances. Auto-resets any env that terminates so the rollout can keep
    running continuously across episode boundaries."""

    def __init__(self, envs: List[CallBreakEnv]):
        self.envs = envs
        self.num_envs = len(envs)
        self._last_obs: Optional[np.ndarray] = None
        self._last_infos: Optional[List[Dict]] = None

    def reset(self):
        obs_list, info_list = [], []
        for env in self.envs:
            obs, info = env.reset()
            obs_list.append(obs)
            info_list.append(info)
        self._last_obs = np.stack(obs_list)
        self._last_infos = info_list
        return self._last_obs, self._last_infos

    def step(self, actions: np.ndarray):
        obs_list, reward_list, done_list, info_list = [], [], [], []
        for env, action in zip(self.envs, actions):
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
            if done:
                # Auto-reset so the next rollout step has a valid observation.
                obs, info = env.reset()
            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(float(done))
            info_list.append(info)

        self._last_obs = np.stack(obs_list)
        self._last_infos = info_list
        return (
            self._last_obs,
            np.asarray(reward_list, dtype=np.float32),
            np.asarray(done_list, dtype=np.float32),
            info_list,
        )

    @property
    def last_obs(self):
        return self._last_obs

    @property
    def last_infos(self):
        return self._last_infos


def _build_legal_mask(legal_actions: List[int], width: int) -> np.ndarray:
    mask = np.zeros(width, dtype=np.bool_)
    for idx in legal_actions:
        if idx < width:
            mask[idx] = True
    return mask


_HISTORY_ENCODER = StateEncoder()  # stateless w.r.t. encode_history(); one shared instance is fine


def _build_history_batch(infos: List[Dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build (num_envs, MAX_HISTORY_LEN) card/player history id arrays and a
    padding mask from a batch of env info dicts, for the transformer
    encoder's CardHistoryEncoder. No-op cost for the mlp encoder path,
    since ActorCritic._encode() ignores these entirely when encoder_type
    is "mlp" -- kept unconditional here to keep this file encoder-agnostic.
    """
    num_envs = len(infos)
    card_ids = np.zeros((num_envs, MAX_HISTORY_LEN), dtype=np.int64)
    player_ids = np.zeros((num_envs, MAX_HISTORY_LEN), dtype=np.int64)
    masks = np.ones((num_envs, MAX_HISTORY_LEN), dtype=np.bool_)
    for i, info in enumerate(infos):
        c, p, m = _HISTORY_ENCODER.encode_history(info)
        card_ids[i] = c
        player_ids[i] = p
        masks[i] = m
    return card_ids, player_ids, masks


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
            history_len=MAX_HISTORY_LEN,
        )

        self.curriculum = CurriculumScheduler(config["curriculum"])
        self.evaluator  = Evaluator(config["evaluation"])
        self.logger     = Logger(config["training"]["log_dir"])

        self.global_step: int = 0
        self.best_eval_score: float = -float("inf")

        # Built lazily once the curriculum reaches the self_play stage
        # (see _ensure_self_play_pool).
        self._self_play_pool: Optional[SelfPlayPool] = None

        self.checkpoint_dir = config["training"].get("checkpoint_dir", "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

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
        eval_interval = self.config["training"]["eval_interval"]
        checkpoint_interval = self.config["training"].get(
            "checkpoint_interval", eval_interval
        )
        steps_per_rollout = (
            self.config["ppo"]["rollout_steps"] * self.config["training"]["num_envs"]
        )

        opponents = self._get_current_opponents()
        envs = self._make_envs(opponents)
        last_eval_step = 0
        last_checkpoint_step = 0

        while self.global_step < total_timesteps:
            rollout_stats = self.collect_rollout(envs)
            loss_stats = self.ppo_update()
            self.buffer.reset()
            self.global_step += steps_per_rollout

            # [FIX] Re-enabled: this was commented out, which meant both
            # prior runs trained with zero visibility into which curriculum
            # stage was active at any point over 10M steps.
            #
            # [FIX 2] log_scalars() forwards every value straight into
            # TensorBoard's add_scalar(), which requires a float -- passing
            # the stage NAME ("mixed") crashed immediately. Logging the
            # numeric stage index instead keeps this TensorBoard-safe; the
            # human-readable name is still visible via the existing
            # "[curriculum] advanced to stage: ..." console print below,
            # which isn't going through the scalar logger at all.
            self.logger.log_scalars(
                {
                    **rollout_stats,
                    **loss_stats,
                    "curriculum_stage_idx": float(self.curriculum._current_idx),
                },
                step=self.global_step,
            )

            if self.global_step - last_eval_step >= eval_interval:
                last_eval_step = self.global_step
                eval_result = self._run_evaluation()
                # [FIX 2] eval_result only carries the numeric stage index
                # (TensorBoard-safe), so print the human-readable name +
                # the win rate that actually gates advancement here, on
                # every eval cycle -- not just when a stage change happens.
                print(
                    f"[eval] step={self.global_step:,} "
                    f"stage={self.curriculum.stage_name} "
                    f"stage_win_rate={eval_result['eval_stage_win_rate']:.1%} "
                    f"(threshold={self.curriculum.current_stage.threshold_winrate}) "
                    f"overall_win_rate={eval_result['eval_overall_win_rate']:.1%}"
                )
                self.logger.log_scalars(eval_result, step=self.global_step)
                self._save_if_best(eval_result["eval_overall_win_rate"], self.global_step)

                min_self_play_timesteps = self.config["curriculum"].get(
                    "self_play_timesteps"
                )
                advanced = self.curriculum.advance(min_self_play_timesteps)
                if advanced:
                    print(f"[curriculum] advanced to stage: {self.curriculum.stage_name}")
                    opponents = self._get_current_opponents()
                    envs = self._make_envs(opponents)

            if self.global_step - last_checkpoint_step >= checkpoint_interval:
                last_checkpoint_step = self.global_step
                ckpt_path = os.path.join(
                    self.checkpoint_dir, f"step_{self.global_step}.pt"
                )
                self.save_checkpoint(ckpt_path, metadata={"step": self.global_step})

                if self.curriculum.is_self_play:
                    pool = self._ensure_self_play_pool()
                    pool.add(self.model, self.global_step)
                    # self_play is the terminal curriculum stage, so
                    # curriculum.advance() never returns True again once
                    # we're here -- meaning opponents/envs would otherwise
                    # never be rebuilt again, and freshly-added pool
                    # checkpoints would never actually get sampled into
                    # training. Rebuild now so newer past-selves enter
                    # rotation as the pool grows.
                    opponents = self._get_current_opponents()
                    envs = self._make_envs(opponents)

    # ------------------------------------------------------------------
    # Rollout collection
    # ------------------------------------------------------------------

    def collect_rollout(self, envs: "_SyncVectorEnv") -> Dict:
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
        if envs.last_obs is None:
            obs, infos = envs.reset()
        else:
            obs, infos = envs.last_obs, envs.last_infos

        episode_rewards: List[float] = []
        running_episode_reward = np.zeros(envs.num_envs, dtype=np.float32)

        for _ in range(self.buffer.rollout_steps):
            actions = np.zeros(envs.num_envs, dtype=np.int64)
            log_probs = np.zeros(envs.num_envs, dtype=np.float32)
            values = np.zeros(envs.num_envs, dtype=np.float32)
            legal_masks_52 = np.zeros((envs.num_envs, NUM_CARDS), dtype=np.bool_)
            phases = np.array(
                [info["phase"] for info in infos], dtype=object
            )
            card_history_ids, player_history_ids, history_masks = _build_history_batch(infos)

            for phase in ("bid", "play"):
                idx = np.where(phases == phase)[0]
                if idx.size == 0:
                    continue

                width = NUM_BID_ACTIONS if phase == "bid" else NUM_CARDS
                mask_subset = np.stack(
                    [_build_legal_mask(infos[i]["legal_actions"], width) for i in idx]
                )
                obs_subset = obs[idx]

                obs_tensor = torch.as_tensor(obs_subset, dtype=torch.float32, device=self.device)
                mask_tensor = torch.as_tensor(mask_subset, dtype=torch.bool, device=self.device)
                card_hist_tensor = torch.as_tensor(
                    card_history_ids[idx], dtype=torch.int64, device=self.device
                )
                player_hist_tensor = torch.as_tensor(
                    player_history_ids[idx], dtype=torch.int64, device=self.device
                )
                hist_mask_tensor = torch.as_tensor(
                    history_masks[idx], dtype=torch.bool, device=self.device
                )

                with torch.no_grad():
                    action, log_prob, _entropy, value = self.model.get_action_and_value(
                        obs_tensor, mask_tensor, phase=phase, deterministic=False,
                        card_history=card_hist_tensor,
                        player_history=player_hist_tensor,
                        history_mask=hist_mask_tensor,
                    )

                actions[idx] = action.cpu().numpy()
                log_probs[idx] = log_prob.cpu().numpy()
                values[idx] = value.cpu().numpy()
                # Store the mask at full 52-width for the buffer, regardless
                # of phase; bid masks simply leave indices 13-51 as False.
                legal_masks_52[idx, :width] = mask_subset

            next_obs, rewards, dones, next_infos = envs.step(actions)
            running_episode_reward += rewards
            for env_idx, done in enumerate(dones):
                if done:
                    episode_rewards.append(float(running_episode_reward[env_idx]))
                    running_episode_reward[env_idx] = 0.0

            self.buffer.add(
                obs=obs,
                action=actions,
                log_prob=log_probs,
                reward=rewards,
                value=values,
                done=dones,
                legal_mask=legal_masks_52,
                phase=phases,
                card_history_ids=card_history_ids,
                player_history_ids=player_history_ids,
                history_mask=history_masks,
            )

            obs, infos = next_obs, next_infos

        # Bootstrap value for the state the rollout stopped on.
        last_values = self._compute_values(obs, infos)
        last_dones = np.array(
            [0.0 for _ in infos], dtype=np.float32
        )  # envs auto-reset on done, so the state we're bootstrapping from
           # is always a fresh (non-terminal) observation.
        self.buffer.compute_returns_and_advantages(last_values, last_dones)

        return {
            "mean_reward": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
            "episodes_completed": len(episode_rewards),
        }

    def _compute_values(self, obs: np.ndarray, infos: List[Dict]) -> np.ndarray:
        """Run a forward pass purely to get V(s) for every env's current
        state, grouped by phase (see module docstring, point 2)."""
        values = np.zeros(len(infos), dtype=np.float32)
        phases = np.array([info["phase"] for info in infos], dtype=object)
        card_history_ids, player_history_ids, history_masks = _build_history_batch(infos)

        for phase in ("bid", "play"):
            idx = np.where(phases == phase)[0]
            if idx.size == 0:
                continue
            width = NUM_BID_ACTIONS if phase == "bid" else NUM_CARDS
            mask_subset = np.stack(
                [_build_legal_mask(infos[i]["legal_actions"], width) for i in idx]
            )
            obs_tensor = torch.as_tensor(obs[idx], dtype=torch.float32, device=self.device)
            mask_tensor = torch.as_tensor(mask_subset, dtype=torch.bool, device=self.device)
            card_hist_tensor = torch.as_tensor(
                card_history_ids[idx], dtype=torch.int64, device=self.device
            )
            player_hist_tensor = torch.as_tensor(
                player_history_ids[idx], dtype=torch.int64, device=self.device
            )
            hist_mask_tensor = torch.as_tensor(
                history_masks[idx], dtype=torch.bool, device=self.device
            )
            with torch.no_grad():
                _log_probs, value = self.model.forward(
                    obs_tensor, mask_tensor, phase=phase,
                    card_history=card_hist_tensor,
                    player_history=player_hist_tensor,
                    history_mask=hist_mask_tensor,
                )
            values[idx] = value.squeeze(-1).cpu().numpy()

        return values

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
        ppo_cfg = self.config["ppo"]
        num_epochs = ppo_cfg["num_epochs"]
        batch_size = ppo_cfg.get("batch_size", 256)
        value_loss_coeff = ppo_cfg.get("value_loss_coeff", 0.5)
        entropy_coeff = ppo_cfg.get("entropy_coeff", 0.01)
        max_grad_norm = ppo_cfg.get("max_grad_norm", 0.5)

        policy_losses, value_losses, entropies = [], [], []

        for _ in range(num_epochs):
            for batch in self.buffer.get_batches(batch_size):
                phases = batch["phases"]  # numpy object array of "bid"/"play"
                total_policy_loss = torch.zeros((), device=self.device)
                total_value_loss = torch.zeros((), device=self.device)
                total_entropy = torch.zeros((), device=self.device)
                total_count = 0

                for phase in ("bid", "play"):
                    idx = np.where(phases == phase)[0]
                    if idx.size == 0:
                        continue

                    width = NUM_BID_ACTIONS if phase == "bid" else NUM_CARDS
                    idx_t = torch.as_tensor(idx, device=self.device)

                    obs_subset = batch["obs"][idx_t]
                    actions_subset = batch["actions"][idx_t]
                    old_log_probs_subset = batch["log_probs_old"][idx_t]
                    advantages_subset = batch["advantages"][idx_t]
                    returns_subset = batch["returns"][idx_t]
                    mask_subset = batch["legal_masks"][idx_t][:, :width]
                    card_hist_subset = batch["card_history_ids"][idx_t]
                    player_hist_subset = batch["player_history_ids"][idx_t]
                    hist_mask_subset = batch["history_masks"][idx_t]

                    new_log_probs, entropy, values = self.model.evaluate_actions(
                        obs_subset, actions_subset, mask_subset, phase=phase,
                        card_history=card_hist_subset,
                        player_history=player_hist_subset,
                        history_mask=hist_mask_subset,
                    )

                    policy_loss = self._compute_policy_loss(
                        new_log_probs, old_log_probs_subset, advantages_subset
                    )
                    value_loss = self._compute_value_loss(values, returns_subset)

                    n = idx.size
                    total_policy_loss = total_policy_loss + policy_loss * n
                    total_value_loss = total_value_loss + value_loss * n
                    total_entropy = total_entropy + entropy.mean() * n
                    total_count += n

                if total_count == 0:
                    continue

                policy_loss = total_policy_loss / total_count
                value_loss = total_value_loss / total_count
                entropy_mean = total_entropy / total_count

                loss = policy_loss + value_loss_coeff * value_loss - entropy_coeff * entropy_mean

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                self.optimizer.step()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy_mean.item())

        return {
            "policy_loss": float(np.mean(policy_losses)) if policy_losses else 0.0,
            "value_loss": float(np.mean(value_losses)) if value_losses else 0.0,
            "entropy": float(np.mean(entropies)) if entropies else 0.0,
        }

    def _compute_policy_loss(
        self,
        new_log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
    ) -> torch.Tensor:
        """Compute clipped PPO policy loss."""
        clip_epsilon = self.config["ppo"].get("clip_epsilon", 0.2)
        ratio = torch.exp(new_log_probs - old_log_probs)
        unclipped = ratio * advantages
        clipped = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
        return -torch.min(unclipped, clipped).mean()

    def _compute_value_loss(
        self,
        values: torch.Tensor,
        returns: torch.Tensor,
        old_values: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute (optionally clipped) value function loss."""
        if old_values is None:
            return 0.5 * (values - returns).pow(2).mean()

        clip_epsilon = self.config["ppo"].get("value_clip_epsilon", 0.2)
        clipped_values = old_values + torch.clamp(
            values - old_values, -clip_epsilon, clip_epsilon
        )
        unclipped_loss = (values - returns).pow(2)
        clipped_loss = (clipped_values - returns).pow(2)
        return 0.5 * torch.max(unclipped_loss, clipped_loss).mean()

    # ------------------------------------------------------------------
    # Environment management
    # ------------------------------------------------------------------

    def _make_envs(self, opponents: List) -> _SyncVectorEnv:
        """
        Create a vectorised set of CallBreakEnv instances.
        Each env has the same opponent configuration from curriculum.
        Returns a VecEnv-like wrapper.
        """
        num_envs = self.config["training"]["num_envs"]
        num_rounds = self.config["training"].get("num_rounds_per_episode", 5)
        reward_shaping = self.config["training"].get("reward_shaping", True)
        base_seed = self.config["training"].get("seed", 0)

        envs = [
            CallBreakEnv(
                opponent_agents=opponents,
                num_rounds=num_rounds,
                reward_shaping=reward_shaping,
                seed=base_seed + i,
            )
            for i in range(num_envs)
        ]
        return _SyncVectorEnv(envs)

    def _get_current_opponents(self) -> List:
        """
        Wrapper around curriculum.get_opponents() that guarantees the
        SelfPlayPool exists (and set_self_play_context has been called)
        before we ever ask the curriculum for self_play opponents. Needed
        because pool creation otherwise only happens inside the checkpoint
        block in train(), which can run *after* get_opponents() is first
        called for the self_play stage (e.g. immediately on curriculum
        advancement, or when --stage self_play force-starts training there).
        """
        if self.curriculum.is_self_play:
            self._ensure_self_play_pool()
        return self.curriculum.get_opponents(self._self_play_pool)

    def _ensure_self_play_pool(self) -> SelfPlayPool:
        if self._self_play_pool is None:
            sp_cfg = self.config.get("self_play", {})
            self._self_play_pool = SelfPlayPool(
                pool_size=sp_cfg.get("pool_size", 20),
                sample_strategy=sp_cfg.get("sample_strategy", "uniform"),
                checkpoint_dir=sp_cfg.get(
                    "checkpoint_dir", os.path.join(self.checkpoint_dir, "pool")
                ),
                seed=self.config["training"].get("seed"),
            )
            # ActorCritic requires obs_dim alongside everything in
            # config["model"] (see __init__ above: ActorCritic(obs_dim=...,
            # **config["model"])). config["model"] alone is missing obs_dim,
            # so SelfPlayPool.load_agent() would fail to reconstruct the
            # model without it.
            pool_model_config = {"obs_dim": self.config["obs_dim"], **self.config["model"]}
            self.curriculum.set_self_play_context(
                self._self_play_pool, pool_model_config, self.device
            )
        return self._self_play_pool

    def _run_evaluation(self) -> Dict:
        """
        Runs TWO evaluations:

          1. Global suite eval, against config['evaluation']['opponents']
             (random/safe_bet/safe_play/basic_bet blended). Used purely for
             logging and best-checkpoint selection (_save_if_best) -- this
             is the number you want to track as "how good is the agent
             overall," and it should NOT gate curriculum advancement, since
             it includes opponents the current stage may not even train
             against.

          2. [FIX] Stage-specific eval, against self._get_current_opponents()
             -- the actual opponents this stage trains against (works for
             self_play too, since run_games() accepts pre-instantiated
             agents, including live self-play model wrappers). THIS is what
             gets fed into curriculum.record_result(), because
             should_advance() is checking "has the agent mastered what this
             stage is teaching it," not "does it already beat everything."

        Previously, step 2 didn't exist -- the global win rate was recorded
        into the curriculum directly, which is why both prior runs stalled
        in the "mixed" stage (threshold 0.65) despite performing reasonably:
        overall win rate (64.1% / 62.0%) never crossed 0.65, because it
        includes safe_bet (~49%) and basic_bet (~22%) dragging the average
        down -- neither of which "mixed"'s own threshold was ever meant to
        require beating at that rate in isolation.
        """
        from agents.rl.ppo_agent import PPOAgent
        from utils.state_encoder import StateEncoder

        eval_agent = PPOAgent(
            player_id=0,
            model=self.model,
            encoder=StateEncoder(),
            device=self.device,
            deterministic=True,
            name="rl_agent",
        )

        num_games = self.config["evaluation"].get("num_games", 100)

        # --- 1. Global suite: logging / best-checkpoint selection only ---
        result = self.evaluator.evaluate(eval_agent, num_games=num_games)

        # --- 2. [FIX] Stage-specific: this is what gates curriculum advancement ---
        stage_opponents = self._get_current_opponents()
        stage_games = self.evaluator.run_games(eval_agent, stage_opponents, num_games)
        stage_wins = stage_games["wins"]

        for i in range(num_games):
            self.curriculum.record_result(i < stage_wins)

        flat = {
            "eval_overall_win_rate": result["overall_win_rate"],
            "eval_bid_accuracy": result["bid_accuracy"],
            "eval_avg_overtrick": result["avg_overtrick"],
            "eval_elo": result["elo"],
            # New: makes it possible to see, per eval point, exactly what
            # value curriculum advancement is being gated on.
            "eval_stage_win_rate": stage_wins / num_games,
            # [FIX 2] Same TensorBoard float-only constraint as above --
            # log the numeric index here too, not the stage name string.
            "curriculum_stage_idx": float(self.curriculum._current_idx),
        }
        for opp_type, wr in result["win_rates"].items():
            flat[f"eval_win_rate_{opp_type}"] = wr
        for opp_type, score in result["avg_scores"].items():
            flat[f"eval_avg_score_{opp_type}"] = score

        return flat

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str, metadata: Optional[Dict] = None) -> None:
        """Save model weights, optimizer state, and training metadata."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "best_eval_score": self.best_eval_score,
            "curriculum_stage_idx": self.curriculum._current_idx,
        }
        if metadata:
            checkpoint.update(metadata)
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str) -> Dict:
        """Load checkpoint; return metadata dict."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.global_step = checkpoint.get("global_step", 0)
        self.best_eval_score = checkpoint.get("best_eval_score", -float("inf"))
        if "curriculum_stage_idx" in checkpoint:
            self.curriculum._current_idx = checkpoint["curriculum_stage_idx"]

        return {
            k: v
            for k, v in checkpoint.items()
            if k not in ("model_state_dict", "optimizer_state_dict")
        }

    def _save_if_best(self, eval_score: float, step: int) -> None:
        """Save as best.pt if eval_score exceeds self.best_eval_score."""
        if eval_score > self.best_eval_score:
            self.best_eval_score = eval_score
            best_path = os.path.join(self.checkpoint_dir, "best.pt")
            self.save_checkpoint(best_path, metadata={"step": step, "eval_score": eval_score})