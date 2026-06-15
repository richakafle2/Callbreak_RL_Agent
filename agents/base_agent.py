"""
base_agent.py
-------------
Abstract interface that every Call Break agent must implement.
Both heuristic and RL agents share this contract so the environment
can call opponents uniformly.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import numpy as np


class BaseAgent(ABC):
    """
    Abstract Call Break agent.

    Each agent must implement:
      - bid(observation)  → int   (bid 1-13)
      - play(observation) → int   (card index 0-51)

    Optionally override:
      - observe(observation, action, reward, next_obs, done)
            for agents that learn from experience.
      - reset()  called at the start of each new game.
    """

    def __init__(self, player_id: int, name: str = "Agent"):
        self.player_id = player_id
        self.name = name

    # ------------------------------------------------------------------
    # Decision interface
    # ------------------------------------------------------------------

    @abstractmethod
    def bid(self, observation: Dict) -> int:
        """
        Given the current observation (from Round.get_observation),
        return an integer bid in [1, 13].

        Observation keys available at bid time:
          - 'hand': List[Card]
          - 'bids': List[Optional[int]]   (None for players not yet bid)
          - 'player_id': int
        """
        raise NotImplementedError

    @abstractmethod
    def play(self, observation: Dict) -> int:
        """
        Given the current observation, return the index (0-51) of the
        card to play. Must be a legal play (in observation['legal_plays']).

        Observation keys available at play time:
          - 'hand': List[Card]
          - 'legal_plays': List[Card]
          - 'bids': List[int]
          - 'tricks_won': List[int]
          - 'cards_played_history': List[Tuple[int, Card]]
          - 'current_trick_plays': List[Tuple[int, Card]]
          - 'led_suit': Optional[Suit]
          - 'tricks_remaining': int
          - 'player_id': int
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Optional learning hooks
    # ------------------------------------------------------------------

    def observe(
        self,
        observation: Dict,
        action: int,
        reward: float,
        next_observation: Dict,
        done: bool,
    ) -> None:
        """
        Called after each action to give the agent experience.
        No-op by default; override in learning agents.
        """
        pass

    def reset(self) -> None:
        """Called at the start of each new game. Override if stateful."""
        pass

    # ------------------------------------------------------------------
    # Utility helpers (available to all subclasses)
    # ------------------------------------------------------------------

    def _card_to_index(self, card) -> int:
        """Convert a Card object to its 0-51 index."""
        return card.index

    def _index_to_card(self, index: int, hand: List) -> Optional[object]:
        """Return the Card in hand whose index matches, or None."""
        for card in hand:
            if card.index == index:
                return card
        return None

    def _encode_hand(self, hand: List) -> np.ndarray:
        """Return a 52-dim binary numpy vector for the given hand."""
        vec = np.zeros(52, dtype=np.float32)
        for card in hand:
            vec[card.index] = 1.0
        return vec

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.player_id}, name={self.name})"
