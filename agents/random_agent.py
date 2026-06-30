"""
random_agent.py
---------------
Uniformly random agent — the weakest baseline.
  - Bids a random integer between min_bid and max_bid.
  - Plays a uniformly random legal card.
"""

import random
from typing import Dict, Optional
from agents.base_agent import BaseAgent


class RandomAgent(BaseAgent):
    def __init__(
        self,
        player_id: int,
        min_bid: int = 1,
        max_bid: int = 13,
        seed: Optional[int] = None,
    ):
        super().__init__(player_id, name="Random")
        self.min_bid = min_bid
        self.max_bid = max_bid
        self._rng = random.Random(seed)

    def bid(self, observation: Dict) -> int:
        """Return a uniformly random bid in [min_bid, max_bid]."""
        return random.randint(self.min_bid, self.max_bid)

    def play(self, observation: Dict) -> int:
        """Return the card index of a uniformly random legal card."""
        legal_cards = observation["legal_cards"]
        return self._rng.choice(legal_cards)
