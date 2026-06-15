"""
elo.py
------
Elo rating system for tracking relative agent strength.
Used to measure progress against baselines and within the self-play pool.
"""

import math
from typing import Dict, Optional


class EloRating:
    def __init__(self, initial_rating: float = 1000.0, k_factor: float = 32.0):
        self.initial_rating = initial_rating
        self.k_factor = k_factor
        self._ratings: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Rating management
    # ------------------------------------------------------------------

    def get_rating(self, agent_name: str) -> float:
        """Return the current Elo rating for `agent_name` (default initial)."""
        return self._ratings.get(agent_name, self.initial_rating)

    def set_rating(self, agent_name: str, rating: float) -> None:
        self._ratings[agent_name] = rating

    def register(self, agent_name: str, rating: Optional[float] = None) -> None:
        """Register an agent with an optional initial rating."""
        self._ratings[agent_name] = rating or self.initial_rating

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, winner: str, loser: str) -> Tuple[float, float]:
        """
        Update Elo ratings for a completed match.
        Returns (new_winner_rating, new_loser_rating).

        Formula:
          E_a = 1 / (1 + 10^((R_b - R_a) / 400))
          R_a' = R_a + K * (S_a - E_a)
          where S_a = 1 for win, 0 for loss.
        """
        raise NotImplementedError

    def update_draw(self, agent_a: str, agent_b: str) -> Tuple[float, float]:
        """
        Update for a draw (S_a = S_b = 0.5).
        Returns (new_a_rating, new_b_rating).
        """
        raise NotImplementedError

    def expected_score(self, agent_a: str, agent_b: str) -> float:
        """
        Return the expected score for agent_a against agent_b.
        E_a = 1 / (1 + 10^((R_b - R_a) / 400))
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Multi-player (4-player game adaptation)
    # ------------------------------------------------------------------

    def update_multiplayer(
        self,
        agent_names: list,
        scores: list,
    ) -> Dict[str, float]:
        """
        Update Elo for a 4-player game using pairwise comparison.
        For each pair (i, j): treat higher scorer as winner, lower as loser.
        Returns updated ratings dict.

        This is an approximation; true multi-player Elo is an active research area.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def leaderboard(self) -> list:
        """Return list of (name, rating) tuples sorted by rating descending."""
        return sorted(self._ratings.items(), key=lambda x: x[1], reverse=True)

    def __repr__(self) -> str:
        board = ", ".join(f"{k}:{v:.0f}" for k, v in self.leaderboard())
        return f"EloRating({board})"


# Fix missing import for type hint
from typing import Tuple
