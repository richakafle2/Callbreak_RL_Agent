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
        return (self.leading_player + len(self.plays)) % self.num_players

    def cards_played(self) -> List[Card]:
        """Return just the Card objects played so far (ordered)."""
        return [card for _, card in self.plays]

    def current_winner(self) -> Optional[int]:
        """
        Return the player_id currently winning the trick
        (highest card by Call Break rules: trump > led suit > others).
        Returns None if no cards played yet.
        """
        if not self.plays:
            return None

        best_player, best_card = self.plays[0]
        for player_id, card in self.plays[1:]:
            if card.beats(best_card, self.led_suit):
                best_player, best_card = player_id, card
        return best_player

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
        if not self.plays:
            # Leading the trick: any card is legal.
            return list(hand)

        follow = self._must_follow_suit(hand)
        if follow:
            return follow

        # Void in the led suit — check forced-overtrump rule.
        # Only applies when a trump is already committed to the trick;
        # if no trump has been played yet there is nothing to overtrump.
        higher = self._higher_trumps(hand)
        if higher:
            return higher

        # No led-suit cards and can't beat an existing trump (or no trump
        # played yet) — any card is legal.
        return list(hand)

    def _must_follow_suit(self, hand: List[Card]) -> List[Card]:
        """Return hand cards that match led_suit, or [] if none."""
        if self.led_suit is None:
            return []
        return [card for card in hand if card.suit == self.led_suit]

    def _trump_cards(self, hand: List[Card]) -> List[Card]:
        """Return all trump (spade) cards in hand."""
        return [card for card in hand if card.is_trump]

    def _best_trump_played(self) -> Optional[Card]:
        """
        Return the highest trump already played in this trick, or None.
        Internal helper used by _higher_trumps.
        """
        trumps = [card for _, card in self.plays if card.is_trump]
        if not trumps:
            return None
        return max(trumps, key=lambda c: c.rank)

    def _higher_trumps(self, hand: List[Card]) -> List[Card]:
        """
        Return trump cards in hand that beat the current best trump played.
        Returns [] if no trump has been played yet — the forced-overtrump
        rule only triggers once a trump is already committed to the trick.
        """
        best = self._best_trump_played()
        if best is None:
            return []
        return [card for card in self._trump_cards(hand) if card.rank > best.rank]

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
        if self.is_complete:
            raise ValueError("Cannot play a card; this trick is already complete.")
        if player_id != self.current_player:
            raise ValueError(
                f"It is player {self.current_player}'s turn, not player {player_id}'s."
            )

        if self.led_suit is None:
            self.led_suit = card.suit

        self.plays.append((player_id, card))

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------

    def resolve(self) -> int:
        """
        Return the player_id who wins this trick.
        Raises RuntimeError if the trick is not complete.
        """
        if not self.is_complete:
            raise RuntimeError(
                f"Cannot resolve: only {len(self.plays)}/{self.num_players} cards played."
            )
        return self.current_winner()

    def __repr__(self) -> str:
        plays_str = ", ".join(f"P{pid}:{card}" for pid, card in self.plays)
        return f"Trick(led_suit={self.led_suit}, plays=[{plays_str}])"