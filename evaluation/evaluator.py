"""
evaluator.py
------------
Runs head-to-head evaluation matches between the RL agent and baselines.
Tracks win rates, average scores, bid accuracy, Elo ratings, and per-seat
bid comparisons (see _compute_bid_stats / print_bid_comparison).
"""

from typing import Dict, List, Optional, Tuple
from collections import defaultdict
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

_HIGH_CARD_POINTS = {14: 4, 13: 3, 12: 2, 11: 1}  # Ace, King, Queen, Jack


def _hand_strength(hand: List[Card]) -> float:
    """
    Fixed, opponent-agnostic hand-strength score used ONLY for this
    diagnostic -- not what any agent (PPO or heuristic) actually uses
    internally to decide its own bid. That's the point: this is a common
    yardstick applied identically to every seat, so "does this agent's bid
    track hand strength" can be compared fairly across agent types that may
    have entirely different (or no) internal hand-evaluation logic.

    Standard Call Break heuristic: high-card points (A=4,K=3,Q=2,J=1) across
    all suits, plus a flat bonus per trump (spade) card, since spades beat
    every other suit regardless of rank and are disproportionately valuable
    for winning tricks.
    """
    hcp = sum(_HIGH_CARD_POINTS.get(int(c.rank), 0) for c in hand)
    trump_count = sum(1 for c in hand if c.is_trump)
    return hcp + 1.5 * trump_count


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
        n = num_games or self.num_games
        agent_name = getattr(agent, "name", "rl_agent")
        self.elo.register(agent_name)
        play_stats_by_matchup: Dict[str, Dict] = {}
        win_rates: Dict[str, float] = {}
        avg_scores: Dict[str, float] = {}
        bid_stats_by_matchup: Dict[str, Dict] = {}
        all_round_results: List[Dict] = []
        bid_stats_by_strength_matchup: Dict[str, Dict] = {}
        total_wins = 0

        for opp_type in self.opponent_types:
            opponents = self._build_opponents([opp_type] * 3)
            results = self.run_games(agent, opponents, n)

            win_rates[opp_type] = results["wins"] / n
            avg_scores[opp_type] = float(np.mean(results["scores"])) if results["scores"] else 0.0
            all_round_results.extend(results.get("round_results", []))
            total_wins += results["wins"]

            bid_stats_by_matchup[opp_type] = results["bid_stats"]
            play_stats_by_matchup[opp_type] = results["play_stats"]

            bid_stats_by_strength_matchup[opp_type] = results["bid_stats_by_strength"]

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
            "bid_stats_by_matchup": bid_stats_by_matchup, 
            "play_stats_by_matchup": play_stats_by_matchup, 
            "bid_stats_by_strength_matchup": bid_stats_by_strength_matchup,
        }
        self.log_result(result)
        return result

    def run_games(self, agent, opponents, num_games) -> Dict:
        wins = 0
        scores: List[float] = []
        opponent_scores: List[List[float]] = []
        game_results: List[bool] = []
        all_round_results: List[Dict] = []
        all_play_events: List[Dict] = []  # NEW

        for _ in range(num_games):
            agent.reset()
            for opp in opponents:
                opp.reset()
            won, agent_score, opp_score_list, round_results, play_events = self._run_single_game(agent, opponents)
            wins += int(won)
            scores.append(agent_score)
            opponent_scores.append(opp_score_list)
            game_results.append(won)
            all_round_results.extend(round_results)
            all_play_events.extend(play_events)  # NEW

        return {
            "wins":                wins,
            "scores":              scores,
            "opponent_scores":     opponent_scores,
            "bid_met_rate":        self._compute_bid_accuracy(all_round_results),
            "avg_tricks_over_bid": self._compute_avg_overtrick(all_round_results),
            "game_results":        game_results,
            "round_results":       all_round_results,
            "bid_stats":           self._compute_bid_stats(all_round_results),
            "play_stats":          self._compute_play_stats(all_play_events),  
            "bid_stats_by_strength": self._compute_bid_stats_by_strength_tercile(all_round_results),        
        }
    
    def _compute_play_stats(self, play_events: List[Dict]) -> Dict[str, Dict]:
        """
        Play-phase diagnostics, grouped by seat name, testing two specific
        hypotheses:

        1. Denial-bonus over-firing: trick_won_reward() adds
            denial_bonus * needy_opponents on EVERY trick won, including
            overtricks -- not gated on tricks_won <= bid. If this is driving
            unwanted "grab everything" behavior, expect BOTH a high
            post_bid_win_rate AND a high avg_needy_on_post_bid_wins together
            -- the agent keeps fighting for tricks it doesn't need, especially
            when the reward for doing so is high.
        2. Trump conservation: whether played_trump rate rises once the bid
            is already met (post_bid_trump_rate vs pre_bid_trump_rate). A rise
            suggests trump is being burned on unneeded tricks, which can leave
            the agent without stopping power late against aggressive opponents.

        NOTE: "post-bid" here is per-CARD, not per-round -- it means "this
        specific play happened after the bid was already met," not "this
        round ended in an overtrick."
        """
        by_name: Dict[str, List[Dict]] = defaultdict(list)
        for ev in play_events:
            by_name[ev["name"]].append(ev)

        stats = {}
        for name, evs in by_name.items():
            pre_bid = [e for e in evs if not e["already_met_bid_before"]]
            post_bid = [e for e in evs if e["already_met_bid_before"]]

            pre_trump_opps = [e for e in pre_bid if e["had_trump_option"]]
            post_trump_opps = [e for e in post_bid if e["had_trump_option"]]

            pre_trump_rate = (
                sum(1 for e in pre_trump_opps if e["played_trump"]) / len(pre_trump_opps)
                if pre_trump_opps else float("nan")
            )
            post_trump_rate = (
                sum(1 for e in post_trump_opps if e["played_trump"]) / len(post_trump_opps)
                if post_trump_opps else float("nan")
            )

            post_bid_wins = [e for e in post_bid if e["won_trick"]]
            post_bid_win_rate = len(post_bid_wins) / len(post_bid) if post_bid else float("nan")
            avg_needy_on_post_bid_wins = (
                float(np.mean([e["needy_opponents_if_won"] for e in post_bid_wins]))
                if post_bid_wins else float("nan")
            )

            trick_win_rate = sum(1 for e in evs if e["won_trick"]) / len(evs) if evs else 0.0

            stats[name] = {
                "n_plays":                    len(evs),
                "trick_win_rate":             trick_win_rate,
                "pre_bid_trump_rate":         pre_trump_rate,
                "post_bid_trump_rate":        post_trump_rate,
                "post_bid_win_rate":          post_bid_win_rate,
                "avg_needy_on_post_bid_wins": avg_needy_on_post_bid_wins,
                "n_post_bid_plays":           len(post_bid),
            }
        return stats

    def _run_single_game(self, agent, opponents) -> Tuple[bool, float, List[float], List[Dict], List[Dict]]:
        all_agents = [agent] + list(opponents)
        seat_names = [getattr(a, "name", f"seat_{i}") for i, a in enumerate(all_agents)]

        total_scores = [0.0] * 4
        round_results: List[Dict] = []
        play_events: List[Dict] = []  # NEW: one row per card played, for play-phase diagnostics

        for round_num in range(NUM_ROUNDS_PER_GAME):
            r = Round(round_number=round_num)
            r.deal()

            seat_hand_strength: Dict[int, float] = {}

            for pid in range(4):
                obs = r.get_observation(pid)
                bid = int(np.clip(all_agents[pid].bid(obs), 1, 13))
                r.place_bid(pid, bid)
                seat_hand_strength[pid] = _hand_strength(obs["hand"])

            trick_num = 1               # NEW: 1-indexed trick position within the round
            trick_buffer: List[Dict] = []  # NEW: holds this trick's up-to-4 events until winner is known

            while not r.playing.is_complete:
                pid = r.playing.current_player
                obs = r.get_observation(pid)
                card_index = all_agents[pid].play(obs)
                legal = r.playing.legal_plays(pid)
                played_card = self._find_card_by_index(card_index, legal)

                # Snapshot BEFORE resolving -- these describe the situation the
                # agent faced when it chose this card, not the outcome after.
                bid_this_seat = r.bidding.bids[pid]
                tricks_before = r.playing.tricks_won[pid]
                had_trump_option = any(getattr(c, "is_trump", False) for c in legal)
                played_trump = getattr(played_card, "is_trump", False)

                winner = r.play_card(pid, played_card)

                trick_buffer.append({
                    "seat": pid,
                    "name": seat_names[pid],
                    "trick_num": trick_num,
                    "bid": bid_this_seat,
                    "tricks_won_before": tricks_before,
                    "already_met_bid_before": tricks_before >= bid_this_seat,
                    "had_trump_option": had_trump_option,
                    "played_trump": played_trump,
                    "played_trump_rank": getattr(played_card, "rank", None) if played_trump else None,
                    "won_trick": False,           # patched below once winner is known
                    "needy_opponents_if_won": None,
                })

                if winner is not None:
                    # NEW: winner is only known on the 4th card and isn't
                    # necessarily whoever just played -- patch the correct
                    # seat's buffered event rather than the current pid's.
                    for ev in trick_buffer:
                        if ev["seat"] == winner:
                            ev["won_trick"] = True
                            # Mirrors RewardShaper/callbreak_env's denial-bonus
                            # calc exactly, so this is directly comparable to
                            # what the agent was actually rewarded for.
                            other_seats = [s for s in range(4) if s != winner]
                            ev["needy_opponents_if_won"] = sum(
                                1 for s in other_seats
                                if r.playing.tricks_won[s] < r.bidding.bids[s]
                            )
                            break
                    play_events.extend(trick_buffer)
                    trick_buffer = []
                    trick_num += 1

            for i, score in enumerate(r.scores):
                total_scores[i] += score

            for pid in range(4):
                round_results.append({
                    "seat": pid,
                    "name": seat_names[pid],
                    "bid": r.bidding.bids[pid],
                    "tricks_won": r.playing.tricks_won[pid],
                    "hand_strength": seat_hand_strength[pid],
                })

        agent_score = total_scores[0]
        opp_scores  = total_scores[1:]
        won = agent_score > max(opp_scores)
        return won, agent_score, opp_scores, round_results, play_events

    def _compute_bid_accuracy(self, round_results: List[Dict]) -> float:
        # Unchanged behavior: only counts seat 0 (the RL agent), same as
        # before -- round_results now also contains other seats' entries,
        # so this must filter to seat 0 explicitly to avoid silently mixing
        # in opponents' bid-met rates.
        agent_rows = [r for r in round_results if r.get("seat", 0) == 0]
        if not agent_rows:
            return 0.0
        return sum(1 for r in agent_rows if r["tricks_won"] >= r["bid"]) / len(agent_rows)

    def _compute_bid_stats_by_strength_tercile(self, round_results: List[Dict]) -> Dict[str, Dict[str, Dict]]:
        """
        Same bid-vs-hand-strength diagnostic as _compute_bid_stats, but split
        into three buckets (weak / medium / strong) by hand_strength within
        this call's pooled rows, instead of one aggregate number.

        Tercile cut points come from the POOLED hand_strength values across
        ALL seats in round_results, not per-seat -- strength is just a
        function of the 13 dealt cards, and every seat draws from the same
        distribution, so pooling gives more stable cut points than computing
        them separately per seat name would.

        This tests a specific failure mode that aggregate bid_strength_corr
        can't see: two agents can have near-identical overall correlation
        (PPO ~0.85 vs BasicBet ~0.77) while one of them still underbids
        specifically on its STRONGEST hands -- where the forgone bid, and the
        forgone score, is largest. Aggregate corr/gap can't distinguish "gap
        spread evenly across all hands" from "gap concentrated in one bucket."
        """
        if len(round_results) < 6:
            return {}

        all_strengths = [r["hand_strength"] for r in round_results]
        tercile_edges = np.percentile(all_strengths, [33.33, 66.67])

        def _bucket(s: float) -> str:
            if s <= tercile_edges[0]:
                return "weak"
            elif s <= tercile_edges[1]:
                return "medium"
            return "strong"

        by_name_bucket: Dict[str, Dict[str, List[Dict]]] = defaultdict(lambda: defaultdict(list))
        for r in round_results:
            by_name_bucket[r["name"]][_bucket(r["hand_strength"])].append(r)

        stats: Dict[str, Dict[str, Dict]] = {}
        for name, buckets in by_name_bucket.items():
            stats[name] = {}
            for label in ("weak", "medium", "strong"):
                rows = buckets.get(label, [])
                if not rows:
                    stats[name][label] = {
                        "avg_bid": float("nan"), "avg_tricks_won": float("nan"),
                        "avg_gap": float("nan"), "avg_hand_strength": float("nan"),
                        "n_rounds": 0,
                    }
                    continue
                bids = [r["bid"] for r in rows]
                tricks = [r["tricks_won"] for r in rows]
                strengths = [r["hand_strength"] for r in rows]
                stats[name][label] = {
                    "avg_bid": float(np.mean(bids)),
                    "avg_tricks_won": float(np.mean(tricks)),
                    "avg_gap": float(np.mean([t - b for t, b in zip(tricks, bids)])),
                    "avg_hand_strength": float(np.mean(strengths)),
                    "n_rounds": len(rows),
                }
        return stats
    def _compute_avg_overtrick(self, round_results: List[Dict]) -> float:
        agent_rows = [r for r in round_results if r.get("seat", 0) == 0]
        over = [r["tricks_won"] - r["bid"] for r in agent_rows if r["tricks_won"] >= r["bid"]]
        return float(np.mean(over)) if over else 0.0

    def _compute_bid_stats(self, round_results: List[Dict]) -> Dict[str, Dict]:
        """
        Aggregate avg bid / avg tricks_won / avg (tricks_won - bid), AND the
        bid-to-hand-strength correlation, grouped by seat name -- across
        every seat in every round passed in, not just the RL agent.

        The correlation is the calibration check: a high correlation means
        this agent bids more when its hand is objectively stronger (by the
        fixed _hand_strength yardstick) and less when it's weaker. A low or
        near-zero correlation, even with a perfectly reasonable AVERAGE bid,
        means the agent is bidding roughly the same regardless of hand
        quality -- leaving expected value on the table on strong hands and
        risking misses on weak ones, independent of any incentive/reward
        issue. That's a distinct problem from what the denial bonus or
        underbidding checks address.
        """
        by_name: Dict[str, List[Dict]] = defaultdict(list)
        for r in round_results:
            by_name[r["name"]].append(r)

        stats = {}
        for name, rows in by_name.items():
            bids = [r["bid"] for r in rows]
            tricks = [r["tricks_won"] for r in rows]
            strengths = [r["hand_strength"] for r in rows]

            if len(rows) >= 2 and np.std(bids) > 1e-9 and np.std(strengths) > 1e-9:
                bid_strength_corr = float(np.corrcoef(bids, strengths)[0, 1])
            else:
                # Undefined (constant bids or constant strengths in this
                # sample) rather than silently reporting 0.0, which would
                # look like "no calibration" instead of "not enough
                # variance to measure."
                bid_strength_corr = float("nan")

            stats[name] = {
                "avg_bid": float(np.mean(bids)),
                "avg_tricks_won": float(np.mean(tricks)),
                "avg_bid_vs_tricks_gap": float(np.mean([t - b for t, b in zip(tricks, bids)])),
                "avg_hand_strength": float(np.mean(strengths)),
                "bid_strength_corr": bid_strength_corr,
                "n_rounds": len(rows),
            }
        return stats

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
        
    def print_bid_by_strength(self, result: Dict, matchup: Optional[str] = None) -> None:
        """
        matchup: restrict to one opponent_type (e.g. "basic_bet"); None prints all.
        """
        print("\nBid comparison by hand-strength tercile:")
        print(f"  {'Matchup':<12} {'Seat':<14} {'Bucket':<7} {'AvgBid':>7} {'AvgTricks':>10} {'AvgGap':>8} {'AvgStr':>7} {'N':>5}")
        print(f"  {'-'*12} {'-'*14} {'-'*7} {'-'*7} {'-'*10} {'-'*8} {'-'*7} {'-'*5}")
        for opp_type, by_name in result.get("bid_stats_by_strength_matchup", {}).items():
            if matchup and opp_type != matchup:
                continue
            for name, buckets in by_name.items():
                for label in ("weak", "medium", "strong"):
                    s = buckets[label]
                    if s["n_rounds"] == 0:
                        continue
                    print(
                        f"  {opp_type:<12} {name:<14} {label:<7} "
                        f"{s['avg_bid']:>7.2f} {s['avg_tricks_won']:>10.2f} "
                        f"{s['avg_gap']:>+8.2f} {s['avg_hand_strength']:>7.2f} {s['n_rounds']:>5}"
                    )
        print()
    def print_play_diagnostics(self, result: Dict) -> None:
        """
        Read pre_bid_trump_rate vs post_bid_trump_rate as a pair: a large jump
        upward once the bid is already met is the trump-conservation red flag.
        Read post_bid_win_rate alongside avg_needy_on_post_bid_wins together:
        both high is the signature of the denial bonus driving "keep grabbing
        tricks" behavior specifically when opponents are still short.
        """
        print("\nPlay-phase diagnostics (trump timing / post-bid behavior), by matchup:")
        print(
            f"  {'Matchup':<12} {'Seat':<14} {'TrickWin%':>9} "
            f"{'PreTrump%':>9} {'PostTrump%':>10} {'PostWin%':>9} {'AvgNeedy':>9}"
        )
        print(f"  {'-'*12} {'-'*14} {'-'*9} {'-'*9} {'-'*10} {'-'*9} {'-'*9}")

        def _fmt(v: float) -> str:
            return f"{v*100:>8.1f}%" if not np.isnan(v) else f"{'n/a':>9}"

        for opp_type, stats in result.get("play_stats_by_matchup", {}).items():
            for name, s in stats.items():
                needy = s["avg_needy_on_post_bid_wins"]
                needy_str = f"{needy:>9.2f}" if not np.isnan(needy) else f"{'n/a':>9}"
                print(
                    f"  {opp_type:<12} {name:<14} "
                    f"{s['trick_win_rate']*100:>8.1f}% "
                    f"{_fmt(s['pre_bid_trump_rate'])} {_fmt(s['post_bid_trump_rate'])} "
                    f"{_fmt(s['post_bid_win_rate'])} {needy_str}"
                )
        print()

    def print_bid_comparison(self, result: Dict) -> None:
        """
        Prints, per matchup and per seat: average bid, average tricks won,
        average gap, and the bid-to-hand-strength correlation. A consistent
        gap in avg bid is evidence for/against underbidding; a consistent
        gap in bid_strength_corr (this agent's correlation much lower than
        the opponent's, within the SAME matchup/games) is evidence for/
        against a separate bid-calibration problem.
        """
        print("\nBid comparison (avg bid / avg tricks won / avg gap / bid-strength corr), by matchup:")
        print(f"  {'Matchup':<12} {'Seat':<14} {'Avg bid':>8} {'Avg tricks':>11} {'Avg gap':>9} {'Bid-str r':>10}")
        print(f"  {'-'*12} {'-'*14} {'-'*8} {'-'*11} {'-'*9} {'-'*10}")
        for opp_type, stats in result.get("bid_stats_by_matchup", {}).items():
            for name, s in stats.items():
                corr_str = f"{s['bid_strength_corr']:>+10.2f}" if not np.isnan(s['bid_strength_corr']) else f"{'n/a':>10}"
                print(
                    f"  {opp_type:<12} {name:<14} "
                    f"{s['avg_bid']:>8.2f} {s['avg_tricks_won']:>11.2f} "
                    f"{s['avg_bid_vs_tricks_gap']:>+9.2f} {corr_str}"
                )
        print()

    def _find_card_by_index(self, index: int, cards: List[Card]) -> Card:
        for card in cards:
            if card.index == index:
                return card
        return cards[0]