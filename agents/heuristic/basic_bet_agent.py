"""
basic_bet_agent.py
------------------
Simple greedy heuristic — a step above random.
  - Bids by counting aces and trumps naively.
  - Always tries to win the current trick regardless of bid status.
"""

from typing import Dict, List
from agents.base_agent import BaseAgent
from environment.card import Card, Suit, Rank


class BasicBetAgent(BaseAgent):
    def __init__(self, player_id: int):
        super().__init__(player_id, name="BasicBet")

    # ------------------------------------------------------------------
    # Bidding — naive count
    # ------------------------------------------------------------------

    def bid(self, observation: Dict) -> int:
        """
        Simple counting:
          - Each ace = 1 trick
          - Each trump >= JACK = 1 trick
          - Each trump < JACK = 0.5 trick
          - Round down, minimum 1
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Playing — greedy, always try to win
    # ------------------------------------------------------------------

    def play(self, observation: Dict) -> int:
        """
        Greedy play:
          - If we can win the trick, play the lowest winning card.
          - If we cannot win, play the lowest card in hand.
        Does NOT consider whether winning the trick is strategically useful.
        """
        raise NotImplementedError

    def _can_win(self, legal_plays: List[Card], observation: Dict) -> bool:
        """Return True if any legal play would win the current trick."""
        raise NotImplementedError

    def _lowest_winner(self, legal_plays: List[Card], observation: Dict) -> Card:
        """Return the lowest card that wins the current trick."""
        raise NotImplementedError

    def _lowest_card(self, cards: List[Card]) -> Card:
        """Return the card with the lowest effective rank."""
        raise NotImplementedError
