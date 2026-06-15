"""
metrics.py
----------
Game-specific metrics for evaluating Call Break agent performance beyond
simple win rate.
"""

from typing import Dict, List, Tuple
import numpy as np


class CallBreakMetrics:
    """Stateful metrics tracker across evaluation episodes."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self._bids: List[int] = []
        self._tricks_won: List[int] = []
        self._scores: List[float] = []
        self._game_results: List[bool] = []   # True = agent won overall game

    # ------------------------------------------------------------------
    # Per-round recording
    # ------------------------------------------------------------------

    def record_round(self, bid: int, tricks_won: int, score: float) -> None:
        """Record the outcome of one round."""
        self._bids.append(bid)
        self._tricks_won.append(tricks_won)
        self._scores.append(score)

    def record_game(self, agent_won: bool) -> None:
        self._game_results.append(agent_won)

    # ------------------------------------------------------------------
    # Metric computation
    # ------------------------------------------------------------------

    def bid_accuracy(self) -> float:
        """Fraction of rounds where the agent met or exceeded their bid."""
        raise NotImplementedError

    def mean_bid(self) -> float:
        """Average bid across rounds (proxy for risk appetite)."""
        raise NotImplementedError

    def mean_score(self) -> float:
        """Mean per-round score."""
        raise NotImplementedError

    def mean_overtrick(self) -> float:
        """Average excess tricks when bid was met (lower = more efficient)."""
        raise NotImplementedError

    def win_rate(self) -> float:
        """Fraction of games won (agent had highest total score)."""
        raise NotImplementedError

    def bid_calibration(self) -> Dict:
        """
        Return calibration analysis: for each bid value (1-13),
        what fraction of the time did the agent actually win that many tricks?

        Returns: {'bid': [...], 'actual_mean_tricks': [...], 'accuracy': [...]}
        """
        raise NotImplementedError

    def score_distribution(self) -> Dict:
        """Return percentile stats of per-round scores."""
        raise NotImplementedError

    def summary(self) -> Dict:
        """Return all metrics as a flat dict."""
        return {
            "win_rate":       self.win_rate(),
            "bid_accuracy":   self.bid_accuracy(),
            "mean_bid":       self.mean_bid(),
            "mean_score":     self.mean_score(),
            "mean_overtrick": self.mean_overtrick(),
        }

    # ------------------------------------------------------------------
    # Comparison utilities
    # ------------------------------------------------------------------

    @staticmethod
    def compare(metrics_a: Dict, metrics_b: Dict) -> Dict:
        """
        Return a dict of deltas (a - b) for common metric keys.
        Positive delta means `a` is better.
        """
        raise NotImplementedError

    @staticmethod
    def elo_win_probability(rating_a: float, rating_b: float) -> float:
        """Expected win probability for agent with rating_a vs rating_b."""
        raise NotImplementedError
