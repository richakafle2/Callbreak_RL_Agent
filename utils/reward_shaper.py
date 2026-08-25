"""
reward_shaper.py
----------------
Reward shaping logic for the Call Break RL agent.

The base reward is only available at the end of each 13-trick round,
making credit assignment difficult. Shaping adds intermediate signals
to accelerate learning without changing the optimal policy (potential-based
shaping guarantees policy invariance when using Φ(s') - Φ(s) form, PROVIDED
Φ(terminal) == 0 for all terminal states).

Shaping components:
  1. Trick reward     : small positive when a needed trick is won; overtricks
                         are reward-neutral (round_score already rewards them
                         at the end -- see note below), except trump waste is
                         still penalized.
  2. Bid progress     : small negative for losing a trick that was still needed
  3. Trump conservation: small negative for wasting high trumps on unneeded tricks
  4. Final round score: the true game reward (bid met / exceeded / failed)

NOTE ON DOUBLE SHAPING:
  trick_won_reward / trick_lost_reward and shaped_reward (potential-based) are
  two independent per-trick shaping signals. If both are summed into the
  agent's reward in the same step (check trainer.py / round.py), the effective
  shaping is their SUM, and any bias in one compounds with the other. They've
  been made directionally consistent here, but ideally only one mechanism
  should be active at a time -- verify wiring before assuming this file alone
  fixes the low-bid overweighting behavior.
"""

import math
from typing import List, Optional


class RewardShaper:
    def __init__(
        self,
        enabled: bool = True,
        trick_reward: float = 0.1,
        bid_progress_bonus: float = 0.02,
        trump_waste_penalty: float = -0.03,
        overtrick_bonus_rate: float = 0.1,
        underbid_penalty_rate: float = 1.0,
        potential_scale: float = 1.0,
        denial_bonus: float = 0.05,
    ):
        self.enabled = enabled
        self.trick_reward = trick_reward
        self.bid_progress_bonus = bid_progress_bonus
        self.trump_waste_penalty = trump_waste_penalty
        self.overtrick_bonus_rate = overtrick_bonus_rate
        self.underbid_penalty_rate = underbid_penalty_rate
        self.potential_scale = potential_scale
        self.denial_bonus = denial_bonus

    # ------------------------------------------------------------------
    # Per-trick intermediate reward
    # ------------------------------------------------------------------

    def trick_won_reward(
        self,
        tricks_won: int,
        bid: int,
        card_played=None,
        opponent_bids: Optional[List[int]] = None,
        opponent_tricks_won: Optional[List[int]] = None,
    ) -> float:
        """
        Called when the agent wins a trick.

        Args:
            tricks_won : agent's tricks won AFTER this trick (inclusive)
            bid        : agent's declared bid
            card_played: the Card used to win (for trump waste detection)
            opponent_bids       : each opponent's declared bid, if available
            opponent_tricks_won : each opponent's tricks won BEFORE this
                trick (i.e. unaffected by it, since they didn't win it) --
                used to detect denial.

        Returns a shaped reward signal.
        """
        if not self.enabled:
            return 0.0

        if tricks_won <= bid:
            reward = self.trick_reward
        else:
            
            reward = self.trick_reward * self.overtrick_bonus_rate 

        # Trump conservation: penalize using a high trump (Ace/King) to win
        # a trick that wasn't needed to reach the bid.
        if (
            card_played is not None
            and tricks_won > bid
            and getattr(card_played, "is_trump", False)
            and getattr(card_played, "rank", 0) >= 13  # King (13) or Ace (14)
        ):
            reward += self.trump_waste_penalty

        #
        # This is a heuristic, not an exact counterfactual ("would this
        # specific opponent have won the trick without my card") -- it
        # simply rewards taking tricks away from the pool while an opponent
        # still needs some. That's directionally correct and much cheaper
        # than reconstructing per-trick counterfactuals from play history.
        if opponent_bids is not None and opponent_tricks_won is not None:
            needy_opponents = sum(
                1 for ob, ot in zip(opponent_bids, opponent_tricks_won) if ot < ob
            )
            reward += self.denial_bonus * needy_opponents

        return reward

    def trick_lost_reward(self, tricks_won: int, bid: int) -> float:
        """
        Called when the agent loses a trick.
        Returns a small negative signal only if we needed the trick.
        """
        if not self.enabled:
            return 0.0

        needed_the_trick = tricks_won < bid
        return -self.bid_progress_bonus if needed_the_trick else 0.0

    # ------------------------------------------------------------------
    # End-of-round true reward
    # ------------------------------------------------------------------

    def round_score(self, tricks_won: int, bid: int) -> float:
        """
        Compute the official Call Break score for one round.

        - Met or exceeded bid : bid + overtrick_bonus_rate * (tricks_won - bid)
        - Failed to meet bid  : -underbid_penalty_rate * bid
        """
        if tricks_won >= bid:
            return bid + self.overtrick_bonus_rate * (tricks_won - bid)
        return -self.underbid_penalty_rate * bid

    # ------------------------------------------------------------------
    # Game-level aggregation
    # ------------------------------------------------------------------

    def final_game_reward(self, total_score: float, opponent_scores: List[float]) -> float:
        """
        Optional: reward based on relative standing vs opponents.
        E.g. normalise by the range of scores, or use rank-based reward.
        """
        if not opponent_scores:
            return 0.0

        all_scores = opponent_scores + [total_score]
        score_range = max(all_scores) - min(all_scores)

        if score_range < 1e-8:
            # Everyone tied -- no informative margin to reward/penalize.
            return 0.0

        margin = total_score - (sum(opponent_scores) / len(opponent_scores))
        return margin / score_range

    # ------------------------------------------------------------------
    # Potential-based shaping (policy-invariant)
    # ------------------------------------------------------------------

    def potential(self, tricks_won: int, bid: int, tricks_remaining: int) -> float:
        """
        State potential Φ(s) for potential-based shaping.

        Design goals fixed from the previous version:
          1. Bid-invariant magnitude: winning/losing a trick should move Φ
             by a comparable amount whether bid=1 or bid=10. The old
             `tricks_won / bid` ratio made low-bid rounds dominate PPO's
             advantage estimates.
          2. Φ(terminal) == 0: required for the potential-based shaping
             policy-invariance guarantee. The old version multiplied by
             `progress` (-> 1.0 at the end), which left a nonzero residual
             at the terminal state and could bias the learned policy, not
             just accelerate learning toward the same optimum.

        Φ(s) = tanh(deviation / DEVIATION_NORM) * (tricks_remaining / 13)

        `expected_pace` is where the agent "should" be if progressing toward
        its bid linearly across the round. Being ahead of pace is good
        (positive potential), behind pace is bad, and the whole thing decays
        to exactly 0 as tricks_remaining -> 0.

        Note: a hard clip(deviation, -1, 1) was tried first but saturates
        immediately for bid=1 (the first trick alone can hit +/-1), which
        recreates a milder version of the original bid-skew bug. tanh gives
        a soft squash instead of a hard cutoff, so low-bid rounds don't
        spike to the ceiling on trick one. Some residual bid-dependence is
        expected and appropriate here -- one trick genuinely is a bigger
        fraction of a bid=1 round than a bid=10 round -- the goal is just to
        avoid the previous ~1/bid blowup, not to erase that difference
        entirely.
        """
        if bid <= 0:
            return 0.0

        tricks_elapsed = 13 - tricks_remaining
        expected_pace = bid * (tricks_elapsed / 13.0)
        deviation = tricks_won - expected_pace

        deviation_norm = 3.0  # tricks of deviation considered "very off pace"
        weight = tricks_remaining / 13.0
        return math.tanh(deviation / deviation_norm) * weight * self.potential_scale

    def shaped_reward(
        self,
        raw_reward: float,
        prev_tricks_won: int,
        curr_tricks_won: int,
        bid: int,
        tricks_remaining_before: int,
        tricks_remaining_after: int,
    ) -> float:
        """
        Full potential-based shaping:
          shaped_r = raw_r + γ * Φ(s') - Φ(s)
        where γ is the discount factor (approximated as 1 here for simplicity).
        """
        if not self.enabled:
            return raw_reward

        phi_before = self.potential(prev_tricks_won, bid, tricks_remaining_before)
        phi_after = self.potential(curr_tricks_won, bid, tricks_remaining_after)

        return raw_reward + phi_after - phi_before