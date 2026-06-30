"""
safe_play_agent.py
------------------
Heuristic agent with basic card-counting and safe play logic.

This agent observes what has been played and uses that to:
  - Identify "established" cards (those that are now the highest in their suit)
  - Avoid wasting high cards when the trick is already won
  - Lead established cards to win tricks safely
  - Track which suits opponents are likely void in
"""

from typing import Dict, List, Optional, Set
from agents.base_agent import BaseAgent
from environment.card import Card, Suit, Rank


class SafePlayAgent(BaseAgent):
    def __init__(self, player_id: int):
        super().__init__(player_id, name="SafePlay")
        self._played_cards: Set[int] = set()
        self._known_void: Dict[int, Set[Suit]] = {}   # player_id -> suits they're void in

    def reset(self) -> None:
        self._played_cards.clear()
        self._known_void.clear()

    # ------------------------------------------------------------------
    # Bidding
    # ------------------------------------------------------------------

    def bid(self, observation: Dict) -> int:
        hand = observation["hand"]
        estimate = self._estimate_trick_count(hand)
        return max(1, int(estimate))

    def _estimate_trick_count(self, hand: List[Card]) -> float:
        """
        For each suit, cards above the highest already-played card count:
          - Trump > NINE: 0.8 each, trump <= NINE: 0.3
          - Ace (non-trump): 0.9
          - Other established non-trump: 0.6
        """
        total = 0.0
        for suit in Suit:
            suit_cards = [c for c in hand if c.suit == suit]
            if not suit_cards:
                continue
            highest_played = self._highest_played_rank(suit)
            for card in suit_cards:
                if highest_played is not None and card.rank <= highest_played:
                    continue
                if suit == Suit.SPADES:
                    weight = 0.8 if card.rank > Rank.NINE else 0.3
                elif card.rank == Rank.ACE:
                    weight = 0.9
                else:
                    weight = 0.6
                total += weight
        return total

    # ------------------------------------------------------------------
    # Playing
    # ------------------------------------------------------------------

    def play(self, observation: Dict) -> int:
        self._update_memory(observation.get("cards_played_history", []))
        legal_plays = observation["legal_plays"]
        trick_plays = observation.get("current_trick_plays", [])

        if not trick_plays:
            card = self._choose_lead(legal_plays, observation)
        else:
            card = self._choose_follow(legal_plays, observation)
        return card.index

    def _update_memory(self, history: List) -> None:
        """
        Rebuilds played_cards and void detection from scratch each call —
        avoids incremental-diff bugs. Chunks history into 4-card tricks;
        any player who didn't follow the led suit is flagged void in it.
        """
        self._played_cards = {card.index for _, card in history}
        self._known_void.clear()
        trick_size = 4
        complete = len(history) - (len(history) % trick_size)
        for i in range(0, complete, trick_size):
            trick = history[i:i + trick_size]
            led_suit = trick[0][1].suit
            for pid, card in trick:
                if card.suit != led_suit:
                    self._known_void.setdefault(pid, set()).add(led_suit)

    def _choose_lead(self, hand: List[Card], observation: Optional[Dict] = None) -> Card:
        established = self._established_cards(hand)
        if established:
            non_trump_est = [c for c in established if not c.is_trump]
            return self._lowest_card(non_trump_est or established)

        if observation is not None and self._is_behind_on_bid(observation):
            trumps = [c for c in hand if c.is_trump]
            if trumps:
                return self._lowest_card(trumps)

        non_trumps = [c for c in hand if not c.is_trump]
        return self._lowest_card(non_trumps or hand)

    def _choose_follow(self, hand: List[Card], observation: Dict) -> Card:
        trick_plays = observation.get("current_trick_plays", [])
        led_suit = observation.get("led_suit")
        if self._is_behind_on_bid(observation):
            winner = self._lowest_winning_card(hand, trick_plays, led_suit)
            if winner is not None:
                return winner
        return self._lowest_card(hand)

    # ------------------------------------------------------------------
    # Card-counting helpers
    # ------------------------------------------------------------------

    def _established_cards(self, hand: List[Card]) -> List[Card]:
        result = []
        for card in hand:
            top = self._highest_remaining(card.suit)
            if top is not None and card.rank == top:
                result.append(card)
        return result

    def _highest_remaining(self, suit: Suit) -> Optional[Rank]:
        played_ranks = {
            Card.from_index(idx).rank
            for idx in self._played_cards
            if Card.from_index(idx).suit == suit
        }
        remaining = [r for r in Rank if r not in played_ranks]
        return max(remaining) if remaining else None

    def _highest_played_rank(self, suit: Suit) -> Optional[Rank]:
        played_ranks = [
            Card.from_index(idx).rank
            for idx in self._played_cards
            if Card.from_index(idx).suit == suit
        ]
        return max(played_ranks) if played_ranks else None

    def _suit_void_probability(self, player_id: int, suit: Suit) -> float:
        """Binary signal: 1.0 if player_id was observed not following `suit`, else 0.0."""
        return 1.0 if suit in self._known_void.get(player_id, set()) else 0.0

    # ------------------------------------------------------------------
    # Play-selection helpers
    # ------------------------------------------------------------------

    def _lowest_winning_card(self, legal_plays, current_trick_plays, led_suit) -> Optional[Card]:
        if not legal_plays:
            return None
        if not current_trick_plays:
            return self._lowest_card(legal_plays)
        _, best = current_trick_plays[0]
        for _, card in current_trick_plays[1:]:
            if card.beats(best, led_suit):
                best = card
        winners = [c for c in legal_plays if c.beats(best, led_suit)]
        return self._lowest_card(winners) if winners else None

    def _lowest_card(self, cards: List[Card]) -> Card:
        """Prefers discarding non-established cards first, then lowest non-trump rank."""
        established_idx = {c.index for c in self._established_cards(cards)}
        pool = [c for c in cards if c.index not in established_idx] or cards
        return min(pool, key=lambda c: (c.is_trump, c.rank))

    def _is_behind_on_bid(self, observation: Dict) -> bool:
        bids = observation.get("bids", [])
        tricks_won = observation.get("tricks_won", [])
        pid = self.player_id
        if pid >= len(bids) or pid >= len(tricks_won):
            return True
        bid = bids[pid]
        return True if bid is None else tricks_won[pid] < bid