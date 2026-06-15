"""
deck.py
-------
Deck creation, shuffling, and dealing for Call Break.
"""

import random
from typing import List
from environment.card import Card, Suit, Rank


class Deck:
    def __init__(self, seed: int | None = None):
        self._rng = random.Random(seed)
        self.cards: List[Card] = []
        self._build()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build(self) -> None:
        """Populate self.cards with all 52 cards."""
        self.cards = self.all_cards()

    def shuffle(self) -> None:
        """Shuffle the deck in place using self._rng."""
        random.shuffle(self.cards, seed=self._rng)

    def reset(self, seed: int | None = None) -> None:
        """Rebuild and shuffle the deck; optionally re-seed the RNG."""
        raise 

    # ------------------------------------------------------------------
    # Dealing
    # ------------------------------------------------------------------

    def deal(self, num_players: int = 4) -> List[List[Card]]:
        """
        Deal all 52 cards evenly to `num_players` players.
        Returns a list of hands: [[cards for p0], [cards for p1], ...]
        Each hand is sorted by suit then rank for readability.
        Raises ValueError if 52 % num_players != 0.
        """
        

    def deal_hand(self, n: int = 13) -> List[Card]:
        """
        Draw and return the next `n` cards from the deck.
        Raises IndexError if not enough cards remain.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def all_cards() -> List[Card]:
        """Return a fresh unsorted list of all 52 cards."""
        cards = []
        for suit in range(4):
             for rank in range(2, 15):
                cards.append(Card(Rank(rank), Suit(suit)))
        return cards
                 

    def __len__(self) -> int:
        return len(self.cards)

    def __repr__(self) -> str:
        return f"Deck({len(self.cards)} cards remaining)"
