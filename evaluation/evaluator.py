"""
evaluator.py
------------
Runs head-to-head evaluation matches between the RL agent and baselines.
Tracks win rates, average scores, bid accuracy, and Elo ratings.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np

from agents.base_agent import BaseAgent
from agents.random_agent import RandomAgent
from agents.heuristic.safe_bet_agent import SafeBetAgent
from agents.heuristic.safe_play_agent import SafePlayAgent
from agents.heuristic.basic_bet_agent import BasicBetAgent
from evaluation.elo import EloRating


BASELINE_REGISTRY = {
    "random":    RandomAgent,
    "safe_bet":  SafeBetAgent,
    "safe_play": SafePlayAgent,
    "basic_bet": BasicBetAgent,
}


class Evaluator:
    """
    Runs evaluation episodes between the learning agent and heuristic baselines.
    The learning agent always occupies seat 0; opponents fill seats 1-3.
    """

    def __init__(self, eval_config: Dict):
        self.num_games = eval_config.get("num_games", 200)
        self.opponent_types = eval_config.get("opponents", ["random", "safe_play"])
        self.elo = EloRating(
            initial_rating=eval_config.get("elo", {}).get("initial_rating", 1000),
            k_factor=eval_config.get("elo", {}).get("k_factor", 32),
        )
        self._results_history: List[Dict] = []

    # ------------------------------------------------------------------
    # Main evaluation entry point
    # ------------------------------------------------------------------

    def evaluate(self, agent: BaseAgent, num_games: Optional[int] = None) -> Dict:
        """
        Run `num_games` games against each opponent type in self.opponent_types.

        Returns a metrics dict:
          {
            'win_rates': {'random': 0.72, 'safe_play': 0.61, ...},
            'avg_scores': {'random': 3.4, 'safe_play': 2.8, ...},
            'bid_accuracy': float,          # fraction of rounds where agent met bid
            'avg_overtrick': float,         # mean tricks above bid when bid met
            'elo': float,                   # current Elo estimate
            'overall_win_rate': float,
          }
        """
        scores = {}
        for opp in self.oppo
        if num_games:
            for game in num_games:


    def run_games(
        self,
        agent: BaseAgent,
        opponents: List[BaseAgent],
        num_games: int,
    ) -> Dict:
        """
        Run `num_games` full Call Break games (each = 5 rounds by default).
        Returns raw results: {
          'wins': int,
          'scores': List[float],
          'opponent_scores': List[List[float]],
          'bid_met_rate': float,
          'avg_tricks_over_bid': float,
        }
        """
        raise NotImplementedError

    def _run_single_game(
        self, agent: BaseAgent, opponents: List[BaseAgent]
    ) -> Tuple[bool, float, List[float]]:
        """
        Run one complete Call Break game (num_rounds rounds).
        Returns:
          won          : bool — agent won (highest total score)
          agent_score  : float — agent's total score
          opp_scores   : List[float] — opponent total scores
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Metrics helpers
    # ------------------------------------------------------------------

    def _compute_bid_accuracy(self, round_results: List[Dict]) -> float:
        """
        Compute fraction of rounds where the agent met their bid.
        round_results: list of {'bid': int, 'tricks_won': int}
        """
        raise NotImplementedError

    def _compute_avg_overtrick(self, round_results: List[Dict]) -> float:
        """Average tricks above bid for rounds where bid was met."""
        raise NotImplementedError

    def _build_opponents(self, opponent_types: List[str]) -> List[BaseAgent]:
        """Instantiate a list of opponent agents from their type names."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # History / reporting
    # ------------------------------------------------------------------

    def log_result(self, result: Dict) -> None:
        """Append an evaluation result to history."""
        self._results_history.append(result)

    def get_history(self) -> List[Dict]:
        return list(self._results_history)

    def print_summary(self, result: Dict) -> None:
        """Pretty-print an evaluation result dict."""
        raise NotImplementedError
