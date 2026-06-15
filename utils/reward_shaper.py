"""
reward_shaper.py
----------------
Reward shaping for the Call Break RL agent.

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
        raise NotImplementedError

    def trick_lost_reward(self, tricks_won: int, bid: int) -> float:
        """
        Called when the agent loses a trick.
        Returns a small negative signal only if we needed the trick.
        """
        if not self.enabled:
            return 0.0
        raise NotImplementedError

    # ------------------------------------------------------------------
    # End-of-round true reward
    # ------------------------------------------------------------------

    def round_score(self, tricks_won: int, bid: int) -> float:
        """
        Compute the official Call Break score for one round.

        - Met or exceeded bid : bid + overtrick_bonus_rate * (tricks_won - bid)
        - Failed to meet bid  : -underbid_penalty_rate * bid
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Game-level aggregation
    # ------------------------------------------------------------------

    def final_game_reward(self, total_score: float, opponent_scores: list) -> float:
        """
        Optional: reward based on relative standing vs opponents.
        E.g. normalise by the range of scores, or use rank-based reward.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Potential-based shaping (policy-invariant)
    # ------------------------------------------------------------------

    def potential(self, tricks_won: int, bid: int, tricks_remaining: int) -> float:
        """
        State potential Φ(s) for potential-based shaping.
        High potential = closer to meeting bid with tricks to spare.

        Φ(s) = (tricks_won / bid) * tricks_remaining_factor
        """
        raise NotImplementedError

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
        raise NotImplementedError
