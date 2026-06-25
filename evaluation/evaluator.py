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
from environment.round import Round
from environment.card import Card


BASELINE_REGISTRY = {
    "random":    RandomAgent,
    "safe_bet":  SafeBetAgent,
    "safe_play": SafePlayAgent,
    "basic_bet": BasicBetAgent,
}

NUM_ROUNDS_PER_GAME = 5


class Evaluator:
    def __init__(self, eval_config: Dict):
        self.num_games = eval_config.get("num_games", 200)
        self.opponent_types = eval_config.get("opponents", ["random", "safe_play"])
        self.elo = EloRating(
            initial_rating=eval_config.get("elo", {}).get("initial_rating", 1000),
            k_factor=eval_config.get("elo", {}).get("k_factor", 32),
        )
        self._results_history: List[Dict] = []

    def evaluate(self, agent: BaseAgent, num_games: Optional[int] = None) -> Dict:
        n = num_games or self.num_games       # fixed: was `for game in num_games` (int not iterable)
        agent_name = getattr(agent, "name", "rl_agent")
        self.elo.register(agent_name)

        win_rates: Dict[str, float] = {}
        avg_scores: Dict[str, float] = {}
        all_round_results: List[Dict] = []
        total_wins = 0

        for opp_type in self.opponent_types:  # fixed: was self.oppo (truncated / nonexistent)
            opponents = self._build_opponents([opp_type] * 3)
            results = self.run_games(agent, opponents, n)

            win_rates[opp_type] = results["wins"] / n
            avg_scores[opp_type] = float(np.mean(results["scores"])) if results["scores"] else 0.0
            all_round_results.extend(results.get("round_results", []))
            total_wins += results["wins"]

            self.elo.register(opp_type)
            for won in results["game_results"]:
                if won:
                    self.elo.update(agent_name, opp_type)
                else:
                    self.elo.update(opp_type, agent_name)

        total_games = n * len(self.opponent_types)
        result = {
            "win_rates":         win_rates,
            "avg_scores":        avg_scores,
            "bid_accuracy":      self._compute_bid_accuracy(all_round_results),
            "avg_overtrick":     self._compute_avg_overtrick(all_round_results),
            "elo":               self.elo.get_rating(agent_name),
            "overall_win_rate":  total_wins / total_games if total_games > 0 else 0.0,
        }
        self.log_result(result)
        return result

    def run_games(self, agent, opponents, num_games) -> Dict:
        wins = 0
        scores: List[float] = []
        opponent_scores: List[List[float]] = []
        game_results: List[bool] = []
        all_round_results: List[Dict] = []

        for _ in range(num_games):
            agent.reset()
            for opp in opponents:
                opp.reset()
            won, agent_score, opp_score_list, round_results = self._run_single_game(agent, opponents)
            wins += int(won)
            scores.append(agent_score)
            opponent_scores.append(opp_score_list)
            game_results.append(won)
            all_round_results.extend(round_results)

        return {
            "wins":                wins,
            "scores":              scores,
            "opponent_scores":     opponent_scores,
            "bid_met_rate":        self._compute_bid_accuracy(all_round_results),
            "avg_tricks_over_bid": self._compute_avg_overtrick(all_round_results),
            "game_results":        game_results,
            "round_results":       all_round_results,
        }

    def _run_single_game(self, agent, opponents) -> Tuple[bool, float, List[float], List[Dict]]:
        all_agents = [agent] + list(opponents)
        total_scores = [0.0] * 4
        round_results: List[Dict] = []

        for round_num in range(NUM_ROUNDS_PER_GAME):
            r = Round(round_number=round_num)
            r.deal()

            for pid in range(4):
                obs = r.get_observation(pid)
                bid = int(np.clip(all_agents[pid].bid(obs), 1, 13))
                r.place_bid(pid, bid)

            while not r.playing.is_complete:
                pid = r.playing.current_player
                obs = r.get_observation(pid)
                card_index = all_agents[pid].play(obs)
                legal = r.playing.legal_plays(pid)
                r.play_card(pid, self._find_card_by_index(card_index, legal))

            for i, score in enumerate(r.scores):
                total_scores[i] += score
            round_results.append({"bid": r.bidding.bids[0], "tricks_won": r.playing.tricks_won[0]})

        agent_score = total_scores[0]
        opp_scores  = total_scores[1:]
        won = agent_score > max(opp_scores)
        return won, agent_score, opp_scores, round_results

    def _compute_bid_accuracy(self, round_results: List[Dict]) -> float:
        if not round_results:
            return 0.0
        return sum(1 for r in round_results if r["tricks_won"] >= r["bid"]) / len(round_results)

    def _compute_avg_overtrick(self, round_results: List[Dict]) -> float:
        over = [r["tricks_won"] - r["bid"] for r in round_results if r["tricks_won"] >= r["bid"]]
        return float(np.mean(over)) if over else 0.0

    def _build_opponents(self, opponent_types: List[str]) -> List[BaseAgent]:
        opponents = []
        for seat, opp_type in enumerate(opponent_types, start=1):
            cls = BASELINE_REGISTRY.get(opp_type)
            if cls is None:
                raise ValueError(
                    f"Unknown opponent type '{opp_type}'. Available: {list(BASELINE_REGISTRY)}"
                )
            opponents.append(cls(player_id=seat))
        return opponents

    def log_result(self, result: Dict) -> None:
        self._results_history.append(result)

    def get_history(self) -> List[Dict]:
        return list(self._results_history)

    def print_summary(self, result: Dict) -> None:
        W = 44
        print(f"\n╔{'═'*W}╗")
        print(f"║{'  Evaluation Summary':^{W}}║")
        print(f"╠{'═'*W}╣")
        print(f"║  {'Overall win rate':<22} {result['overall_win_rate']:>7.1%}       ║")
        print(f"║  {'Elo rating':<22} {result['elo']:>7.0f}       ║")
        print(f"║  {'Bid accuracy':<22} {result['bid_accuracy']:>7.1%}       ║")
        print(f"║  {'Avg overtrick':<22} {result['avg_overtrick']:>7.2f}       ║")
        print(f"╠{'═'*W}╣")
        print(f"║  {'Opponent':<14} {'Win rate':>9} {'Avg score':>9}   ║")
        print(f"║  {'-'*14} {'-'*9} {'-'*9}   ║")
        for opp in result["win_rates"]:
            print(f"║  {opp:<14} {result['win_rates'][opp]:>9.1%} {result['avg_scores'].get(opp,0):>9.2f}   ║")
        print(f"╚{'═'*W}╝\n")

    def _find_card_by_index(self, index: int, cards: List[Card]) -> Card:
        for card in cards:
            if card.index == index:
                return card
        return cards[0]