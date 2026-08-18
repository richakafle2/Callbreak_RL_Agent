"""
curriculum.py
-------------
Curriculum scheduler that manages training stage progression.

Stages (from config):
  1. random    — all random opponents        → advance at 70% win rate
  2. mixed     — random + safe_play mix      → advance at 65%
  3. safe      — all safe_play opponents     → advance at 60%
  4. self_play — pool of past selves         → run for fixed timesteps

Each stage has a `threshold_winrate` that triggers advancement.

NOTE: `record_result()` and `advance()` below aren't part of the original
skeleton's method stubs (only `_build_stages`, `current_stage`, `stage_name`,
`is_self_play`, and `get_opponents` were present as stubs). Something has to
feed game outcomes into each CurriculumStage's win/loss counters and decide
when to move to the next stage, and `CurriculumStage.should_advance()`
implies that shape of API, so I've added the minimal pair of methods needed
to make the scheduler usable end-to-end. Flag if trainer.py should own this
bookkeeping instead.

[FIX] advance() previously checked `if self.is_self_play: return False`
before ever reaching the self_play-specific "ran for enough timesteps"
check below it -- since self_play IS the last stage, that guard fired
every single time and the timestep check was unreachable dead code. This
meant advance() could never report "self_play curriculum complete," even
once min_self_play_timesteps was satisfied, silently defeating the
logging/detection purpose the docstring describes. Fixed by checking
self_play's own completion criteria first, and only using is_last_stage to
gate whether there's actually a next stage index to move into.
"""

from typing import Dict, List, Optional, Any
from agents.random_agent import RandomAgent
from agents.heuristic.safe_bet_agent import SafeBetAgent
from agents.heuristic.safe_play_agent import SafePlayAgent
from agents.heuristic.basic_bet_agent import BasicBetAgent
from agents.rl.ppo_agent import PPOAgent
from utils.state_encoder import StateEncoder


AGENT_REGISTRY = {
    "random":    RandomAgent,
    "safe_bet":  SafeBetAgent,
    "safe_play": SafePlayAgent,
    "basic_bet": BasicBetAgent,
}


class CurriculumStage:
    def __init__(self, name: str, opponents: List[str], threshold_winrate: Optional[float]):
        self.name = name
        self.opponents = opponents                  # list of 3 agent type strings
        self.threshold_winrate = threshold_winrate  # None = run until timestep limit
        self.timesteps_in_stage: int = 0
        self.games_played: int = 0
        self.wins: int = 0

    @property
    def win_rate(self) -> float:
        return self.wins / self.games_played if self.games_played > 0 else 0.0

    def should_advance(self) -> bool:
        """Return True when win rate has exceeded the threshold."""
        if self.threshold_winrate is None:
            return False
        if self.games_played < 100:    # need minimum sample before advancing
            return False
        return self.win_rate >= self.threshold_winrate


class CurriculumScheduler:
    """Manages progression through training curriculum stages."""

    def __init__(self, curriculum_config: Dict):
        self.stages: List[CurriculumStage] = []
        self._current_idx: int = 0
        self._self_play_pool: Optional[Any] = None  # SelfPlayPool injected later
        self._model_config: Optional[Dict] = None   # injected via set_self_play_context
        self._device: Optional[Any] = None           # injected via set_self_play_context

        self._build_stages(curriculum_config)

    def _build_stages(self, config: Dict) -> None:
        """Parse config['stages'] into CurriculumStage objects."""
        stage_specs = config["stages"]
        for spec in stage_specs:
            stage = CurriculumStage(
                name=spec["name"],
                opponents=spec.get("opponents", []),
                threshold_winrate=spec.get("threshold_winrate"),
            )
            self.stages.append(stage)

        if not self.stages:
            raise ValueError("Curriculum config produced zero stages.")

    # ------------------------------------------------------------------
    # Stage access
    # ------------------------------------------------------------------

    @property
    def current_stage(self) -> CurriculumStage:
        return self.stages[self._current_idx]

    @property
    def stage_name(self) -> str:
        return self.current_stage.name

    @property
    def is_self_play(self) -> bool:
        return self.current_stage.name == "self_play"

    @property
    def is_last_stage(self) -> bool:
        return self._current_idx == len(self.stages) - 1

    # ------------------------------------------------------------------
    # Self-play context injection
    # ------------------------------------------------------------------

    def set_self_play_context(self, pool: Any, model_config: Dict, device: Any) -> None:
        """
        Wire up the SelfPlayPool + the model architecture/device needed to
        rehydrate ActorCritic checkpoints into playable opponents. Must be
        called before get_opponents() is used during the self_play stage.
        """
        self._self_play_pool = pool
        self._model_config = model_config
        self._device = device

    # ------------------------------------------------------------------
    # Opponent building
    # ------------------------------------------------------------------

    def get_opponents(self, self_play_pool: Optional[Any] = None) -> List:
        """
        Return a list of 3 instantiated opponent agents for the current stage.
        """
        pool = self_play_pool if self_play_pool is not None else self._self_play_pool

        if self.is_self_play:
            if pool is None:
                raise RuntimeError(
                    "Reached the self_play stage but no SelfPlayPool was "
                    "provided — call set_self_play_context() first."
                )
            if pool.is_empty():
                # Bootstrap case: pool hasn't been seeded with any checkpoint
                # yet (e.g. self_play is reached before the first snapshot is
                # saved). Fall back to the strongest heuristic so training
                # doesn't stall on an empty sample().
                return [BasicBetAgent(player_id=seat + 1) for seat in range(3)]

            if self._model_config is None or self._device is None:
                raise RuntimeError(
                    "SelfPlayPool is set but model_config/device are missing "
                    "— call set_self_play_context() with all three arguments."
                )

            models = pool.sample_opponents(3, self._model_config, self._device)
            return [
                PPOAgent(
                    player_id=seat + 1,
                    model=model,
                    encoder=StateEncoder(),
                    device=self._device,
                    deterministic=False,  # sample, don't argmax, for opponent variety
                    name=f"pool_opponent_{seat + 1}",
                )
                for seat, model in enumerate(models)
            ]

        return [
            AGENT_REGISTRY[name](player_id=seat + 1)
            for seat, name in enumerate(self.current_stage.opponents)
        ]

    # ------------------------------------------------------------------
    # Progress tracking / advancement
    # ------------------------------------------------------------------

    def record_result(self, won: bool, timesteps: int = 0) -> None:
        """
        Record the outcome of one game (round) played under the current
        stage, and accumulate timesteps spent in this stage. Call this once
        per completed episode/round during training and evaluation.
        """
        stage = self.current_stage
        stage.games_played += 1
        if won:
            stage.wins += 1
        stage.timesteps_in_stage += timesteps

    def advance(self, min_self_play_timesteps: Optional[int] = None) -> bool:
        """
        Move to the next curriculum stage if the current stage's promotion
        criteria are met. Returns True if the stage index actually advanced.

        - Non-self_play stages advance once `should_advance()` is True
          (win rate over threshold with enough games played).
        - The self_play stage has no win-rate threshold (`threshold_winrate`
          is None) and instead runs for a fixed number of timesteps. Since
          self_play is the final stage, there's no next index to move into
          -- but we still want `stage.should_advance()`-equivalent detection
          to work here so callers (e.g. trainer.py's logging) can tell when
          the fixed-timestep run is "done," even though the index itself
          won't move.

        [FIX] Previously `if self.is_last_stage: return False` ran BEFORE
        the self_play timestep check, so that check was unreachable dead
        code -- self_play is always the last stage. Now the self_play
        completion check runs first (as a pure detection signal, no index
        change), and is_last_stage is only used to decide whether advancing
        the index is possible at all.
        """
        stage = self.current_stage

        if stage.name == "self_play":
            if min_self_play_timesteps is None:
                return False
            return stage.timesteps_in_stage >= min_self_play_timesteps

        if self.is_last_stage:
            return False

        ready = stage.should_advance()
        if ready:
            self._current_idx += 1
        return ready
