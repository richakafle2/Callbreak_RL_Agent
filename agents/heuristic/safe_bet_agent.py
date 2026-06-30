"""
safe_bet_agent.py
-----------------
Heuristic agent that bids conservatively based on hand strength.

Bid heuristic:
  - Count "winning" cards: aces, kings in suits where ace is likely gone
    by round 1, and trump cards above a threshold rank.
  - Add partial credit for queens and suited sequences.
  - Clamp to [min_bid, max_bid].

Play heuristic:
  - Plays the lowest legal card that still wins the trick.
  - If it cannot win, plays its lowest card (duck).
"""

from typing import Dict, List, Optional
from agents.base_agent import BaseAgent
from environment.card import Card, Suit, Rank


class SafeBetAgent(BaseAgent):
    def __init__(self, player_id: int, min_bid: int = 1, max_bid: int = 13):
        super().__init__(player_id, name="SafeBet")
        self.min_bid = min_bid
        self.max_bid = max_bid

    # ------------------------------------------------------------------
    # Bidding
    # ------------------------------------------------------------------

    def bid(self, observation: Dict) -> int:
        """
        Estimate the number of tricks this hand can reliably win.
          total = sure + 0.5 * probable, clamped to [min_bid, max_bid]
        """
        hand = observation["hand"]                  # fixed: was `observation` (whole dict)
        sure     = self._count_sure_tricks(hand)
        probable = self._count_probable_tricks(hand)
        total    = sure + 0.5 * probable            # fixed: was `return sure` (ignored probable)
        return max(self.min_bid, min(self.max_bid, int(total)))

    def _count_sure_tricks(self, hand: List[Card]) -> int:
        """
        Cards almost certain to win a trick:
          - Ace of any suit            (highest in suit, only trumped)
          - Trump >= QUEEN             (if we hold 3+ trumps for support)
        """
        tc   = self._trump_count(hand)
        sure = 0
        for card in hand:
            if card.rank == Rank.ACE:
                sure += 1                             # aces always count
            elif card.is_trump and tc >= 3 and card.rank >= Rank.QUEEN:
                sure += 1                             # supported high trump
        return sure

    def _count_probable_tricks(self, hand: List[Card]) -> float:
        """
        Cards likely — but not certain — to win:
          - King of any suit  : 0.7 (ace may still be out)
          - Isolated high trump (QUEEN with < 3 trumps): 0.7
          - Mid-range trumps NINE–JACK : 0.5

        Returns a raw count; bid() halves this before adding to sure tricks.
        """
        tc       = self._trump_count(hand)
        probable = 0.0
        for card in hand:
            if card.rank == Rank.ACE:
                continue                              # already counted as sure
            if card.rank == Rank.KING:
                probable += 0.7                       # king is strong but not guaranteed
            elif card.is_trump and card.rank >= Rank.QUEEN and tc < 3:
                probable += 0.7                       # high trump without suit support
            elif card.is_trump and Rank.NINE <= card.rank <= Rank.JACK:
                probable += 0.5                       # mid-range trump, risky
        return probable

    def _trump_count(self, hand: List[Card]) -> int:
        """Return the total number of trump (spade) cards in hand."""
        return sum(1 for card in hand if card.is_trump)

    # ------------------------------------------------------------------
    # Playing
    # ------------------------------------------------------------------

    def play(self, observation: Dict) -> int:
        """
        Safe play strategy (priority order):
          1. Win with a non-trump card (preserves trumps for later).
          2. Win by trumping — only if this trick is worth winning.
          3. Duck: discard the lowest-value card we can afford to lose.
        """
        legal_plays          = observation["legal_plays"]
        current_trick_plays  = observation.get("current_trick_plays", [])
        led_suit             = observation.get("led_suit")
        worth_winning        = self._trick_is_worth_winning(observation)

        if worth_winning:
            # Step 1: try the lowest non-trump winner
            non_trumps = [c for c in legal_plays if not c.is_trump]
            nt_winner  = self._lowest_winning_play(non_trumps, current_trick_plays, led_suit)
            if nt_winner is not None:
                return nt_winner.index

            # Step 2: trump to win if no non-trump option exists
            trumps       = [c for c in legal_plays if c.is_trump]
            trump_winner = self._lowest_winning_play(trumps, current_trick_plays, led_suit)
            if trump_winner is not None:
                return trump_winner.index

        # Step 3: duck — shed the cheapest card
        return self._lowest_card(legal_plays).index

    def _lowest_winning_play(
        self,
        legal_plays: List[Card],
        current_trick_plays: List,
        led_suit,
    ) -> Optional[Card]:
        """
        Return the lowest card in `legal_plays` that would currently win
        the trick, or None if no such card exists.

        When leading (no cards played yet) every card wins; return the
        cheapest one so we preserve high cards for later tricks.
        """
        if not legal_plays:
            return None

        if not current_trick_plays:
            # Leading: all cards win; play cheapest to keep high cards back
            return self._lowest_card(legal_plays)

        # Find the card currently winning the trick
        _, best = current_trick_plays[0]
        for _, card in current_trick_plays[1:]:
            if card.beats(best, led_suit):
                best = card

        winners = [c for c in legal_plays if c.beats(best, led_suit)]
        if not winners:
            return None
        return self._lowest_card(winners)

    def _lowest_card(self, cards: List[Card]) -> Card:
        """
        Return the lowest-value card to play.
        Non-trumps are preferred (cheaper to spend than trumps);
        within a suit, lower rank is preferred.
        """
        return min(cards, key=lambda c: (c.is_trump, c.rank))

    def _trick_is_worth_winning(self, observation: Dict) -> bool:
        """
        Return True when winning this trick still helps us reach our bid.
        (i.e. tricks already won < bid declared)
        Defaults to True when bid data is unavailable.
        """
        bids       = observation.get("bids", [])
        tricks_won = observation.get("tricks_won", [])
        pid        = self.player_id
        if pid >= len(bids) or pid >= len(tricks_won):
            return True
        bid = bids[pid]
        won = tricks_won[pid]
        if bid is None:
            return True                 # bidding phase hasn't completed yet
        return won < bid
