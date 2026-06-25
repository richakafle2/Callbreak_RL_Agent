"""
elo.py
------
Elo rating system for tracking relative agent strength.
Used to measure progress against baselines and within the self-play pool.
"""

import math
from typing import Dict, Optional, Tuple   # moved to top — was at bottom, caused NameError


class EloRating:
    def __init__(self, initial_rating: float = 1000.0, k_factor: float = 32.0):
        self.initial_rating = initial_rating
        self.k_factor = k_factor
        self._ratings: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Rating management
    # ------------------------------------------------------------------

    def get_rating(self, agent_name: str) -> float:
        return self._ratings.get(agent_name, self.initial_rating)

    def set_rating(self, agent_name: str, rating: float) -> None:
        self._ratings[agent_name] = rating

    def register(self, agent_name: str, rating: Optional[float] = None) -> None:
        # fixed: `rating or self.initial_rating` silently replaces 0.0 with
        # initial_rating because 0.0 is falsy. Explicit None check instead.
        self._ratings[agent_name] = rating if rating is not None else self.initial_rating

    # ------------------------------------------------------------------
    # Core helper
    # ------------------------------------------------------------------

    def _expected(self, rating_a: float, rating_b: float) -> float:
        """E_a = 1 / (1 + 10^((R_b - R_a) / 400))"""
        return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))  # fixed: ** not ^

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, winner: str, loser: str) -> Tuple[float, float]:
        """
        Update Elo ratings for a completed match.
        Returns (new_winner_rating, new_loser_rating).
        """
        R_w = self.get_rating(winner)
        R_l = self.get_rating(loser)

        E_w = self._expected(R_w, R_l)
        E_l = 1.0 - E_w

        new_winner = R_w + self.k_factor * (1.0 - E_w)   # S = 1 for win
        new_loser  = R_l + self.k_factor * (0.0 - E_l)   # S = 0 for loss

        self.set_rating(winner, new_winner)   # fixed: ratings never stored
        self.set_rating(loser,  new_loser)    # fixed: ratings never stored
        return new_winner, new_loser          # fixed: nothing was returned

    def update_draw(self, agent_a: str, agent_b: str) -> Tuple[float, float]:
        """Update for a draw (S_a = S_b = 0.5)."""
        R_a = self.get_rating(agent_a)
        R_b = self.get_rating(agent_b)
        E_a = self._expected(R_a, R_b)
        E_b = 1.0 - E_a
        new_a = R_a + self.k_factor * (0.5 - E_a)
        new_b = R_b + self.k_factor * (0.5 - E_b)
        self.set_rating(agent_a, new_a)
        self.set_rating(agent_b, new_b)
        return new_a, new_b

    def expected_score(self, agent_a: str, agent_b: str) -> float:
        """E_a = 1 / (1 + 10^((R_b - R_a) / 400))"""
        return self._expected(self.get_rating(agent_a), self.get_rating(agent_b))

    # ------------------------------------------------------------------
    # Multi-player (4-player game adaptation)
    # ------------------------------------------------------------------

    def update_multiplayer(self, agent_names: list, scores: list) -> Dict[str, float]:
        """
        Pairwise Elo update for a 4-player game.
        K is divided by (n-1) so total update per player matches a
        single head-to-head match in magnitude.
        """
        n = len(agent_names)
        deltas: Dict[str, float] = {name: 0.0 for name in agent_names}
        k_scaled = self.k_factor / (n - 1)

        for i in range(n):
            for j in range(i + 1, n):
                name_i, name_j = agent_names[i], agent_names[j]
                E_i = self._expected(self.get_rating(name_i), self.get_rating(name_j))
                E_j = 1.0 - E_i
                if scores[i] > scores[j]:
                    S_i, S_j = 1.0, 0.0
                elif scores[i] < scores[j]:
                    S_i, S_j = 0.0, 1.0
                else:
                    S_i, S_j = 0.5, 0.5
                deltas[name_i] += k_scaled * (S_i - E_i)
                deltas[name_j] += k_scaled * (S_j - E_j)

        updated: Dict[str, float] = {}
        for name in agent_names:
            new_rating = self.get_rating(name) + deltas[name]
            self.set_rating(name, new_rating)
            updated[name] = new_rating
        return updated

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def leaderboard(self) -> list:
        return sorted(self._ratings.items(), key=lambda x: x[1], reverse=True)

    def __repr__(self) -> str:
        board = ", ".join(f"{k}:{v:.0f}" for k, v in self.leaderboard())
        return f"EloRating({board})"
