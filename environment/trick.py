"""
trick.py
--------
Manages a single trick in Call Break:
  - Tracks which cards have been played and by whom
  - Determines legal plays for a given hand
  - Determines the winner of the trick
"""

from typing import List, Optional, Tuple
from environment.card import Card, Suit


class Trick:
    def __init__(self, leading_player: int, num_players: int = 4):
        self.leading_player = leading_player
        self.num_players = num_players

        # Cards played this trick: index corresponds to play order.
        # Each entry is (player_id, Card).
        self.plays: List[Tuple[int, Card]] = []
        self.led_suit: Optional[Suit] = None

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    @property
    def is_complete(self) -> bool:
        """True when all players have played a card."""
        return len(self.plays) == self.num_players

    @property
    def current_player(self) -> int:
        """
        Return the player_id whose turn it is to play.
        Cycles from leading_player around.
        """
        raise NotImplementedError

    def cards_played(self) -> List[Card]:
        """Return just the Card objects played so far (ordered)."""
        raise NotImplementedError

    def current_winner(self) -> Optional[int]:
        """
        Return the player_id currently winning the trick
        (highest card by Call Break rules: trump > led suit > others).
        Returns None if no cards played yet.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Legal moves
    # ------------------------------------------------------------------

    def legal_plays(self, hand: List[Card]) -> List[Card]:
        """
        Return the subset of `hand` that are legal plays in this trick.

        Call Break rules:
          1. If you can follow the led suit, you MUST.
          2. If you cannot follow suit but have a higher trump than any
             trump already played, you MUST play a higher trump.
          3. If you cannot follow suit and cannot beat existing trumps,
             you may play any card.
          4. On the lead (first card), any card is legal.
        """
        raise NotImplementedError

    def _must_follow_suit(self, hand: List[Card]) -> List[Card]:
        """Return hand cards that match led_suit, or [] if none."""
        raise NotImplementedError

    def _trump_cards(self, hand: List[Card]) -> List[Card]:
        """Return all trump (spade) cards in hand."""
        raise NotImplementedError

    def _higher_trumps(self, hand: List[Card]) -> List[Card]:
        """Return trump cards in hand that beat the current best trump played."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Playing a card
    # ------------------------------------------------------------------

    def play_card(self, player_id: int, card: Card) -> None:
        """
        Record that `player_id` played `card`.
        Sets led_suit on the first play.
        Raises ValueError if: the trick is complete, it is not player_id's
        turn, or the card is not a legal play from their hand (caller must
        validate legal_plays before calling this).
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------

    def resolve(self) -> int:
        """
        Return the player_id who wins this trick.
        Raises RuntimeError if the trick is not complete.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        plays_str = ", ".join(f"P{pid}:{card}" for pid, card in self.plays)
        return f"Trick(led_suit={self.led_suit}, plays=[{plays_str}])"
