"""
reward_shaper.py
----------------
Reward shaping logic for the Call Break RL agent.

The base reward is only available at the end of each 13-trick round,
making credit assignment difficult. Shaping adds intermediate signals
to accelerate learning without changing the optimal policy (potential-based
shaping guarantees policy invariance when using Φ(s') - Φ(s) form).

Shaping components:
  1. Trick reward     : +0.1 when a needed trick is won, -0.05 for excess tricks
  2. Bid progress     : small positive when closing gap to bid target
  3. Trump conservation: small negative for wasting high trumps early
  4. Final round score: the true game reward (bid met / exceeded / failed)
"""

from typing import List, Optional


class RewardShaper:
    def __init__(
        self,
        enabled: bool = True,
        trick_reward: float = 0.1,
        excess_trick_penalty: float = -0.05,
        bid_progress_bonus: float = 0.02,
        trump_waste_penalty: float = -0.03,
        overtrick_bonus_rate: float = 0.1,
        underbid_penalty_rate: float = 1.0,
    ):
        self.enabled = enabled
        self.trick_reward = trick_reward
        self.excess_trick_penalty = excess_trick_penalty
        self.bid_progress_bonus = bid_progress_bonus
        self.trump_waste_penalty = trump_waste_penalty
        self.overtrick_bonus_rate = overtrick_bonus_rate
        self.underbid_penalty_rate = underbid_penalty_rate

    # ------------------------------------------------------------------
    # Per-trick intermediate reward
    # ------------------------------------------------------------------

    def trick_won_reward(self, tricks_won: int, bid: int, card_played=None) -> float:
        """
        Called when the agent wins a trick.

        Args:
            tricks_won : tricks won AFTER this trick (inclusive)
            bid        : agent's declared bid
            card_played: the Card used to win (for trump waste detection)

        Returns a shaped reward signal.
        """
        if not self.enabled:
            return 0.0

        if tricks_won <= bid:
            reward = self.trick_reward
        else:
            # Already met the bid -- winning further tricks isn't harmful to
            # the score (overtricks add a small bonus at round end), but we
            # discourage the *play policy* from chasing unnecessary tricks
            # at the cost of good cards, since that can cost tricks later.
            reward = self.excess_trick_penalty

        # Trump conservation: penalize using a high trump (Ace/King) to win
        # a trick that wasn't needed to reach the bid.
        if (
            card_played is not None
            and tricks_won > bid
            and getattr(card_played, "is_trump", False)
            and getattr(card_played, "rank", 0) >= 13  # King (13) or Ace (14)
        ):
            reward += self.trump_waste_penalty

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
        High potential = closer to meeting bid with tricks to spare.

        Φ(s) = (tricks_won / bid) * tricks_remaining_factor
        """
        if bid <= 0:
            return 0.0

        # Cap the ratio so a large pile of overtricks doesn't blow up the
        # potential unboundedly -- being 2x over bid isn't twice as good as
        # being exactly on pace.
        pace_ratio = min(tricks_won / bid, 1.5)

        # Fraction of the round completed so far: potential should carry
        # more weight as the round progresses and the current trajectory
        # becomes a more reliable signal of the final outcome.
        progress = 1.0 - (tricks_remaining / 13.0)

        return pace_ratio * progress

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