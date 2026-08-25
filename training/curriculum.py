"""
curriculum.py
-------------
Curriculum scheduler that manages training stage progression.

Stages (from config):
  1. random      — all random opponents          → advance at 70% win rate
  2. mixed       — random + safe_play + basic_bet → advance at 55%
  3. safe        — all safe_play opponents        → advance at 60%
  4. mixed_self  — basic_bet + self + safe_play    → advance at 55%
  5. self_play   — basic_bet + self + self         → run for fixed timesteps

Each stage has a `threshold_winrate` that triggers advancement. A stage with
`threshold_winrate: null` instead runs for a fixed number of timesteps
(see `advance()`), which is currently only true of the terminal `self_play`
stage.

Any stage whose `opponents` list includes `"self"` draws that opponent from
a SelfPlayPool of past checkpoints rather than from AGENT_REGISTRY --
`get_opponents()` resolves this per-seat, so a stage can mix heuristic and
self-play opponents in the same game (e.g. `mixed_self`), not just switch
wholesale from one to the other.
"""

from typing import Dict, List, Optional, Any, Tuple
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
    def __init__(
        self,
        name: str,
        opponents: List[str],
        threshold_winrate: Optional[float],
        recent_window: int = 5,
    ):
        self.name = name
        self.opponents = opponents                  # list of 3 agent type strings
        self.threshold_winrate = threshold_winrate  # None = run until timestep limit
        self.timesteps_in_stage: int = 0
        self.games_played: int = 0
        self.wins: int = 0

        # Rolling window of (wins, games) per eval batch, most recent last.
        # should_advance() gates on this instead of the all-time cumulative
        # rate, so stale performance from right after entering the stage
        # (e.g. while a self-play pool was still empty and falling back to
        # a heuristic) doesn't permanently drag down the average that
        # decides advancement.
        self._recent_window = recent_window
        self._recent_batches: List[Tuple[int, int]] = []  # (wins, games) per batch

    @property
    def win_rate(self) -> float:
        """All-time cumulative win rate since entering this stage."""
        return self.wins / self.games_played if self.games_played > 0 else 0.0

    @property
    def recent_win_rate(self) -> float:
        """Win rate over just the last `recent_window` eval batches."""
        if not self._recent_batches:
            return 0.0
        wins = sum(w for w, _ in self._recent_batches)
        games = sum(g for _, g in self._recent_batches)
        return wins / games if games > 0 else 0.0

    def record_batch(self, wins: int, games: int) -> None:
        """
        Record one eval batch's (wins, games) as a single windowed sample.
        Call this once per evaluation cycle (e.g. once per 200-game eval),
        in addition to record_result() per individual game, so
        recent_win_rate reflects only the last few cycles rather than the
        whole stage's history.
        """
        self._recent_batches.append((wins, games))
        if len(self._recent_batches) > self._recent_window:
            self._recent_batches.pop(0)

    def should_advance(self) -> bool:
        """Return True when the recent-window win rate has exceeded the threshold."""
        if self.threshold_winrate is None:
            return False
        if self.games_played < 100:    # need minimum sample before advancing
            return False
        if not self._recent_batches:
            return False
        return self.recent_win_rate >= self.threshold_winrate


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
        """True if the current stage's opponent list includes any "self" slots."""
        return "self" in self.current_stage.opponents

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
        called before get_opponents() is used during any stage whose
        opponent list contains "self".
        """
        self._self_play_pool = pool
        self._model_config = model_config
        self._device = device

    # ------------------------------------------------------------------
    # Opponent building
    # ------------------------------------------------------------------

    def get_opponents(self, self_play_pool: Optional[Any] = None) -> List:
        """
        Return a list of 3 instantiated opponent agents for the current
        stage, resolved seat-by-seat against config['stages'][...]['opponents'].

        Any seat whose name is "self" is filled from the self-play pool (or
        the BasicBetAgent bootstrap fallback if the pool is still empty).
        Every other seat is built directly from AGENT_REGISTRY, exactly as
        configured.
        """
        pool = self_play_pool if self_play_pool is not None else self._self_play_pool
        opponents_spec = self.current_stage.opponents
        self_seats = [seat for seat, name in enumerate(opponents_spec) if name == "self"]

        self_agents_by_seat: Dict[int, Any] = {}
        if self_seats:
            if pool is None:
                raise RuntimeError(
                    f"Stage '{self.stage_name}' includes 'self' opponents but "
                    "no SelfPlayPool was provided — call "
                    "set_self_play_context() first."
                )

            if pool.is_empty():
                # Bootstrap case: pool hasn't been seeded with any checkpoint
                # yet (e.g. this stage is reached before the first snapshot
                # is saved). Fall back to the strongest heuristic for just
                # the "self" seats, leaving any other configured seats alone.
                for seat in self_seats:
                    self_agents_by_seat[seat] = BasicBetAgent(player_id=seat + 1)
            else:
                if self._model_config is None or self._device is None:
                    raise RuntimeError(
                        "SelfPlayPool is set but model_config/device are "
                        "missing — call set_self_play_context() with all "
                        "three arguments."
                    )
                models = pool.sample_opponents(
                    len(self_seats), self._model_config, self._device
                )
                for seat, model in zip(self_seats, models):
                    self_agents_by_seat[seat] = PPOAgent(
                        player_id=seat + 1,
                        model=model,
                        encoder=StateEncoder(),
                        device=self._device,
                        deterministic=False,  # sample, don't argmax, for variety
                        name=f"pool_opponent_{seat + 1}",
                    )

        opponents = []
        for seat, name in enumerate(opponents_spec):
            if seat in self_agents_by_seat:
                opponents.append(self_agents_by_seat[seat])
            else:
                opponents.append(AGENT_REGISTRY[name](player_id=seat + 1))
        return opponents

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

        - Stages with a numeric threshold_winrate advance once
          should_advance() is True (win rate over threshold with enough
          games played).
        - A stage with threshold_winrate == None instead runs for a fixed
          number of timesteps (min_self_play_timesteps), and never advances
          the stage index -- this is used for the terminal self_play stage,
          which has nowhere further to advance to.
        """
        stage = self.current_stage

        if stage.threshold_winrate is None:
            if min_self_play_timesteps is None:
                return False
            return stage.timesteps_in_stage >= min_self_play_timesteps

        if self.is_last_stage:
            return False

        ready = stage.should_advance()
        if ready:
            self._current_idx += 1
        return ready