"""
safe_bet_agent.py
-----------------
Heuristic agent that bids conservatively based on hand strength.

Bid heuristic:
  - Count "winning" cards: aces, kings in suits where ace is likely gone
    by round 1, and trump cards above a threshold rank.
  - Add partial credit for queens and suited sequences.
  - Clamp to [min_bid, max_bid].

Play heuristic:
  - Plays the lowest legal card that still wins the trick.
  - If it cannot win, plays its lowest card (duck).
"""

from typing import Dict, List
from agents.base_agent import BaseAgent
from environment.card import Card, Suit, Rank


class SafeBetAgent(BaseAgent):
    def __init__(self, player_id: int, min_bid: int = 1, max_bid: int = 13):
        super().__init__(player_id, name="SafeBet")
        self.min_bid = min_bid
        self.max_bid = max_bid

    # ------------------------------------------------------------------
    # Bidding
    # ------------------------------------------------------------------

    def bid(self, observation: Dict) -> int:
        """
        Estimate the number of tricks this hand can reliably win.
        Steps:
          1. _count_sure_tricks(hand)  — aces + high trumps
          2. _count_probable_tricks(hand)  — kings, queens, mid-trumps
          3. total = sure + 0.5 * probable, clamped to [min_bid, max_bid]
        """
        hand = observation[]
        return self._count_sure_tricks(hand)
        raise NotImplementedError

    def _count_sure_tricks(self, hand: List[Card]) -> int:
        """
        Count cards that are almost certainly winners:
          - Ace of any suit
          - Trump cards rank >= QUEEN if we hold 3+ trumps
        """
        raise NotImplementedError

    def _count_probable_tricks(self, hand: List[Card]) -> float:
        """
        Count cards that are likely winners but not guaranteed:
          - Kings (unless ace not yet played — treat as 0.7 probable)
          - Mid-range trumps (9, 10, J)
        Returns a float; will be halved in bid().
        """
        raise NotImplementedError

    def _trump_count(self, hand: List[Card]) -> int:
        """Return the number of trump (spade) cards in hand."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Playing
    # ------------------------------------------------------------------

    def play(self, observation: Dict) -> int:
        """
        Safe play strategy:
          - If we can win the trick with a non-trump, play the lowest winner.
          - If we must trump to win and it's worthwhile, play lowest trump that wins.
          - Otherwise, duck with the lowest card we can afford to lose.
        """
        raise NotImplementedError

    def _lowest_winning_play(
        self, legal_plays: List[Card], current_trick_plays: List, led_suit
    ) -> Card | None:
        """
        Return the lowest-ranked card in legal_plays that would currently
        win the trick, or None if no such card exists.
        """
        raise NotImplementedError

    def _lowest_card(self, cards: List[Card]) -> Card:
        """Return the card with the lowest effective value."""
        raise NotImplementedError

    def _trick_is_worth_winning(self, observation: Dict) -> bool:
        """
        Return True if winning this trick helps us meet our bid.
        (tricks_won[player_id] < bids[player_id])
        """
        raise NotImplementedError
