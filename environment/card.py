"""
card.py
-------
Card and Suit definitions for Call Break.
Spades are always trump in Call Break.
"""

from enum import IntEnum
from dataclasses import dataclass


class Suit(IntEnum):
    CLUBS    = 0
    DIAMONDS = 1
    HEARTS   = 2
    SPADES   = 3   # always trump

    def __str__(self) -> str:
        return self.name.capitalize()


class Rank(IntEnum):
    TWO   = 2
    THREE = 3
    FOUR  = 4
    FIVE  = 5
    SIX   = 6
    SEVEN = 7
    EIGHT = 8
    NINE  = 9
    TEN   = 10
    JACK  = 11
    QUEEN = 12
    KING  = 13
    ACE   = 14

    def __str__(self) -> str:
        names = {11: "J", 12: "Q", 13: "K", 14: "A"}
        return names.get(self.value, str(self.value))


@dataclass(frozen=True)
class Card:
    rank: Rank
    suit: Suit

    @property
    def is_trump(self) -> bool:
        return self.suit == Suit.SPADES   # use the enum, not bare 3

    @property
    def index(self) -> int:
        return self.suit * 13 + (self.rank - 2)

    def beats(self, other: "Card", led_suit: Suit) -> bool:
        # Priority 1: trump vs non-trump
        if self.is_trump and not other.is_trump:
            return True
        if not self.is_trump and other.is_trump:
            return False
        # Same suit (both trump, or both non-trump): rank decides
        if self.suit == other.suit:
            return self.rank > other.rank
        # Different non-trump suits: only led suit can win
        if self.suit == led_suit:
            return True
        return False

    def __str__(self) -> str:
        return f"{self.rank}{self.suit.name[0]}"

    def __repr__(self) -> str:
        return f"Card({self.rank}, {self.suit})"

    @classmethod
    def from_index(cls, index: int) -> "Card":
        suit = Suit(index // 13)       # wrap in Suit enum
        rank = Rank(index % 13 + 2)   # wrap in Rank enum
        return cls(rank, suit)         # return an instance, not the class