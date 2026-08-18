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
        if not self._bids:
            return 0.0
        hits = sum(1 for b, t in zip(self._bids, self._tricks_won) if t >= b)
        return hits / len(self._bids)

    def mean_bid(self) -> float:
        """Average bid across rounds (proxy for risk appetite)."""
        if not self._bids:
            return 0.0
        return float(np.mean(self._bids))

    def mean_score(self) -> float:
        """Mean per-round score."""
        if not self._scores:
            return 0.0
        return float(np.mean(self._scores))

    def mean_overtrick(self) -> float:
        """Average excess tricks when bid was met (lower = more efficient)."""
        overtricks = [
            t - b for b, t in zip(self._bids, self._tricks_won) if t >= b
        ]
        if not overtricks:
            return 0.0
        return float(np.mean(overtricks))

    def win_rate(self) -> float:
        """Fraction of games won (agent had highest total score)."""
        if not self._game_results:
            return 0.0
        return float(np.mean(self._game_results))

    def bid_calibration(self) -> Dict:
        """
        Return calibration analysis: for each bid value (1-13),
        what fraction of the time did the agent actually win that many tricks?

        Returns: {'bid': [...], 'actual_mean_tricks': [...], 'accuracy': [...]}
        """
        bids_arr = np.asarray(self._bids)
        tricks_arr = np.asarray(self._tricks_won)

        bid_values, mean_tricks, accuracy = [], [], []
        for bid_value in range(1, 14):
            mask = bids_arr == bid_value
            if not np.any(mask):
                continue
            bid_values.append(bid_value)
            mean_tricks.append(float(tricks_arr[mask].mean()))
            accuracy.append(float((tricks_arr[mask] >= bid_value).mean()))

        return {
            "bid": bid_values,
            "actual_mean_tricks": mean_tricks,
            "accuracy": accuracy,
        }

    def score_distribution(self) -> Dict:
        """Return percentile stats of per-round scores."""
        if not self._scores:
            return {
                "min": 0.0, "p25": 0.0, "median": 0.0,
                "p75": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0,
            }
        scores_arr = np.asarray(self._scores)
        return {
            "min": float(np.min(scores_arr)),
            "p25": float(np.percentile(scores_arr, 25)),
            "median": float(np.median(scores_arr)),
            "p75": float(np.percentile(scores_arr, 75)),
            "max": float(np.max(scores_arr)),
            "mean": float(np.mean(scores_arr)),
            "std": float(np.std(scores_arr)),
        }

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
        common_keys = set(metrics_a.keys()) & set(metrics_b.keys())
        return {
            key: metrics_a[key] - metrics_b[key]
            for key in common_keys
            if isinstance(metrics_a[key], (int, float)) and isinstance(metrics_b[key], (int, float))
        }