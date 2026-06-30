"""
basic_bet_agent.py
------------------
Simple greedy heuristic — a step above random.
  - Bids by counting aces and trumps naively.
  - Always tries to win the current trick regardless of bid status.
"""

from typing import Dict, List, Optional
from agents.base_agent import BaseAgent
from environment.card import Card, Suit, Rank


class BasicBetAgent(BaseAgent):
    def __init__(self, player_id: int):
        super().__init__(player_id, name="BasicBet")

    def bid(self, observation: Dict) -> int:
        """
        Simple counting:
          - Each ace = 1 trick
          - Each trump >= JACK = 1 trick
          - Each trump < JACK = 0.5 trick
          - Round down, minimum 1
        """
        hand = observation["hand"]
        bet = 0.0
        for card in hand:
            if card.rank == Rank.ACE:       # fixed: Rank.ACE not rank.ACE
                bet += 1
            elif card.is_trump:             # fixed: card.is_trump, not observation["trump_suit"]
                if card.rank >= Rank.JACK:  # fixed: Rank.JACK not rank.JACK
                    bet += 1
                else:
                    bet += 0.5
        return max(1, int(bet))

    def play(self, observation: Dict) -> int:
        """
        Greedy play:
          - If we can win the trick, play the lowest winning card.
          - If we cannot win, play the lowest card in hand.
        """
        legal_plays = observation["legal_plays"]    # fixed: "legal_cards" → "legal_plays"

        if self._can_win(legal_plays, observation):
            return self._lowest_winner(legal_plays, observation).index
        return self._lowest_card(legal_plays).index

    def _current_best(self, observation: Dict) -> Optional[Card]:
        """Return the card currently winning the trick, or None if leading."""
        plays = observation.get("current_trick_plays", [])
        if not plays:
            return None
        led_suit = observation.get("led_suit")
        _, best_card = plays[0]
        for _, card in plays[1:]:
            if card.beats(best_card, led_suit):
                best_card = card
        return best_card

    def _can_win(self, legal_plays: List[Card], observation: Dict) -> bool:
        """Return True if any legal play would win the current trick."""
        best = self._current_best(observation)
        if best is None:
            return True     # leading: any card wins
        led_suit = observation.get("led_suit")
        return any(card.beats(best, led_suit) for card in legal_plays)

    def _lowest_winner(self, legal_plays: List[Card], observation: Dict) -> Card:
        """Return the lowest card that wins the current trick."""
        best = self._current_best(observation)
        if best is None:
            return self._lowest_card(legal_plays)   # leading: play cheapest
        led_suit = observation.get("led_suit")
        winners = [c for c in legal_plays if c.beats(best, led_suit)]
        return self._lowest_card(winners)

    def _lowest_card(self, cards: List[Card]) -> Card:
        """Return the lowest-value card: non-trump first, then by rank."""
        return min(cards, key=lambda c: (c.is_trump, c.rank))
