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

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_trump(self) -> bool:
        """Return True if this card is a spade (trump suit)."""
        return self.suit == 3

    @property
    def index(self) -> int:
        """
        Unique 0-51 index for this card.
        Convention: index = suit * 13 + (rank - 2)
        Used for tensor encoding.
        """
        return self.suit * 13 + (self.rank - 2)

    # ------------------------------------------------------------------
    # Comparison helpers
    # ------------------------------------------------------------------

    def beats(self, other: Card, led_suit: Suit) -> bool:
        """
        Return True if this card beats `other` given the led suit.
        Rules:
          - Trump beats any non-trump.
          - Same suit: higher rank wins.
          - Non-trump, different suit from led: loses to led suit cards.
        """
        # other card is a spade so this card must be a spade to beat
        if other.suit == 3:
            if self.suit == 3:
                return self.rank > other.rank
            else: 
                return False

        # other card is the lead suit 
        if other.suit == led_suit:
            if self.suit == led_suit:
                return self.rank > other.rank
            else: 
                return False
            
        # default, but should never come to this is where both are not lead or spade so just return higher rank
        
        return self.rank > other.rank
    

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __str__(self) -> str:
        return f"{self.rank}{self.suit.name[0]}"

    def __repr__(self) -> str:
        return f"Card({self.rank}, {self.suit})"

    # ------------------------------------------------------------------
    # Class methods
    # ------------------------------------------------------------------

    @classmethod
    def from_index(cls, index: int) -> Card:
        """Reconstruct a Card from its 0-51 index."""
        cls.suit = index // 13
        cls.rank = index % 13 + 2
        return cls
