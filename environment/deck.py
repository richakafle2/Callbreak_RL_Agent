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

    def _build(self) -> None:
        self.cards = self.all_cards()

    def shuffle(self) -> None:
        self._rng.shuffle(self.cards)           # fixed: call on the Random instance

    def reset(self, seed: int | None = None) -> None:
        self._build()
        self._rng = random.Random(seed)         # fixed: wrap in Random(), not assign raw int
        self.shuffle()

    def deal(self, num_players: int = 4) -> List[List[Card]]:
        if 52 % num_players != 0:
            raise ValueError(f"Cannot deal 52 cards evenly to {num_players} players.")

        dealt_hands: List[List[Card]] = [[] for _ in range(num_players)]
        for i, card in enumerate(self.cards):
            dealt_hands[i % num_players].append(card)  # fixed: % num_players, not % 4

        for hand in dealt_hands:
            hand.sort(key=lambda c: (c.suit, c.rank))  # fixed: proper sort key

        return dealt_hands

    def deal_hand(self, n: int = 13) -> List[Card]:
        if len(self.cards) < n:
            raise IndexError(f"Only {len(self.cards)} cards remain; cannot deal {n}.")
        hand = self.cards[:n]
        self.cards = self.cards[n:]
        return hand

    @staticmethod
    def all_cards() -> List[Card]:
        return [Card(Rank(rank), Suit(suit))
                for suit in range(4)
                for rank in range(2, 15)]

    def __len__(self) -> int:
        return len(self.cards)

    def __repr__(self) -> str:
        return f"Deck({len(self.cards)} cards remaining)"