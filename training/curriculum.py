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
"""

from typing import Dict, List, Optional, Any
from agents.random_agent import RandomAgent
from agents.heuristic.safe_bet_agent import SafeBetAgent
from agents.heuristic.safe_play_agent import SafePlayAgent
from agents.heuristic.basic_bet_agent import BasicBetAgent


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

        self._build_stages(curriculum_config)

    def _build_stages(self, config: Dict) -> None:
        """Parse config['stages'] into CurriculumStage objects."""
        raise NotImplementedError

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

    # ------------------------------------------------------------------
    # Opponent building
    # ------------------------------------------------------------------

    def get_opponents(self, self_play_pool: Optional[Any] = None) -> List:
        """
        Return a list of 3 instantiated opponent agents for the current stage.
        For self_play stages, samples from self_play_pool.
        """
        raise NotImplementedError

    def _build_opponent(self, agent_type: str, player_id: int) -> Any:
        """Instantiate an opponent agent by type string."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Progress tracking
    # ------------------------------------------------------------------

    def record_game_result(self, agent_won: bool) -> None:
        """Update win/game counters for the current stage."""
        raise NotImplementedError

    def step_timesteps(self, n: int) -> None:
        """Advance the timestep counter for the current stage."""
        raise NotImplementedError

    def try_advance(self) -> bool:
        """
        Check if the current stage's threshold is met and advance if so.
        Returns True if the stage changed.
        """
        raise NotImplementedError

    def _advance_stage(self) -> None:
        """Move to the next stage; reset its counters."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # State / serialization
    # ------------------------------------------------------------------

    def state_dict(self) -> Dict:
        """Serialise scheduler state for checkpointing."""
        raise NotImplementedError

    def load_state_dict(self, state: Dict) -> None:
        """Restore scheduler state from checkpoint."""
        raise NotImplementedError
