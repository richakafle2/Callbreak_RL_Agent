"""
round.py
--------
Manages a full round of Call Break:
  Phase 1 — Bidding: each player declares how many tricks they will win.
  Phase 2 — Play: 13 tricks are played out.
  Phase 3 — Scoring: compute each player's score for this round.
"""

from typing import List, Dict, Optional, Tuple
from environment.card import Card
from environment.deck import Deck
from environment.trick import Trick


class BiddingPhase:
    """Encapsulates the bidding state for one round."""

    def __init__(self, num_players: int = 4, min_bid: int = 1, max_bid: int = 13):
        self.num_players = num_players
        self.min_bid = min_bid
        self.max_bid = max_bid
        self.bids: List[Optional[int]] = [None] * num_players
        self.current_bidder: int = 0

    def place_bid(self, player_id: int, bid: int) -> None:
        """
        Record `bid` for `player_id`.
        Raises ValueError if:
          - bid is out of [min_bid, max_bid]
          - it is not player_id's turn
          - player_id has already bid
        """
        raise NotImplementedError

    def is_complete(self) -> bool:
        """True when all players have placed a bid."""
        raise NotImplementedError

    def valid_bids(self, player_id: int) -> List[int]:
        """Return list of integer bids legal for player_id (typically min_bid..max_bid)."""
        raise NotImplementedError


class PlayingPhase:
    """Manages the 13-trick playing sequence for one round."""

    def __init__(self, hands: List[List[Card]], starting_player: int = 0):
        self.hands: List[List[Card]] = [list(h) for h in hands]
        self.starting_player = starting_player
        self.tricks_played: List[Trick] = []
        self.tricks_won: List[int] = [0] * len(hands)
        self.current_trick: Optional[Trick] = None
        self._start_new_trick(starting_player)

    # ------------------------------------------------------------------
    # Trick management
    # ------------------------------------------------------------------

    def _start_new_trick(self, leading_player: int) -> None:
        """Initialise a new Trick with the given leader."""
        raise NotImplementedError

    def play_card(self, player_id: int, card: Card) -> Optional[int]:
        """
        Play `card` for `player_id` in the current trick.
        If the trick is now complete, resolve it and start the next one.
        Returns the winner player_id if a trick was just resolved, else None.
        Raises ValueError if the card is not a legal play.
        """
        raise NotImplementedError

    def legal_plays(self, player_id: int) -> List[Card]:
        """Return the legal plays for player_id given the current trick state."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    @property
    def is_complete(self) -> bool:
        """True when all 13 tricks have been played."""
        raise NotImplementedError

    @property
    def current_player(self) -> int:
        """player_id whose turn it is to play."""
        raise NotImplementedError

    def tricks_remaining(self) -> int:
        """Number of tricks not yet played."""
        raise NotImplementedError

    def cards_played_history(self) -> List[Tuple[int, Card]]:
        """All (player_id, card) pairs played so far across all tricks."""
        raise NotImplementedError


class Round:
    """
    Orchestrates a complete Call Break round:
      1. Deal cards.
      2. Run bidding phase.
      3. Run playing phase.
      4. Calculate scores.
    """

    def __init__(
        self,
        round_number: int,
        num_players: int = 4,
        deck: Optional[Deck] = None,
        seed: Optional[int] = None,
    ):
        self.round_number = round_number
        self.num_players = num_players
        self.deck = deck or Deck(seed=seed)
        self.hands: List[List[Card]] = []
        self.bidding: Optional[BiddingPhase] = None
        self.playing: Optional[PlayingPhase] = None
        self.scores: List[float] = []

    def deal(self) -> List[List[Card]]:
        """
        Shuffle and deal cards to all players.
        Returns the list of hands and initialises self.hands and self.bidding.
        """
        raise NotImplementedError

    def get_observation(self, player_id: int) -> Dict:
        """
        Return a dictionary of everything player_id can observe:
          - hand: their current cards
          - bids: all bids placed so far (None if not yet placed)
          - tricks_won: tricks won per player so far
          - cards_played_history: [(player_id, card), ...]
          - current_trick_plays: plays in the current trick
          - led_suit: suit led in current trick (None if leading)
          - tricks_remaining: int
          - legal_plays or legal_bids: depending on phase
        """
        raise NotImplementedError

    def calculate_scores(self) -> List[float]:
        """
        Compute and store per-player scores after playing phase completes.

        Scoring:
          - Met or exceeded bid:  score = bid + 0.1 * (tricks_won - bid)
          - Failed to meet bid:   score = -bid
        Returns the scores list.
        Raises RuntimeError if the playing phase is not complete.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"Round({self.round_number}, bids={self.bidding and self.bidding.bids})"
