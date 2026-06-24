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
        if bid < self.min_bid or bid > self.max_bid:
            raise ValueError(
                f"Bid {bid} is out of range [{self.min_bid}, {self.max_bid}]"
            )
        if player_id != self.current_bidder:
            raise ValueError(
                f"It is player {self.current_bidder}'s turn to bid, not player {player_id}'s"
            )
        if self.bids[player_id] is not None:
            raise ValueError(f"Player {player_id} has already placed a bid")

        self.bids[player_id] = bid
        self.current_bidder = (self.current_bidder + 1) % self.num_players

    def is_complete(self) -> bool:
        """True when all players have placed a bid."""
        return None not in self.bids

    def valid_bids(self, player_id: int) -> List[int]:
        """Return list of integer bids legal for player_id (typically min_bid..max_bid)."""
        return [i for i in range(self.min_bid, self.max_bid + 1)]


class PlayingPhase:
    """Manages the 13-trick playing sequence for one round."""

    def __init__(self, hands: List[List[Card]], starting_player: int = 0):
        self.hands: List[List[Card]] = [list(h) for h in hands]
        self.starting_player = starting_player
        self.tricks_played: List[Trick] = []
        self.tricks_won: List[int] = [0] * len(hands)
        self.current_trick: Optional[Trick] = None

        self.num_players = len(hands)
        # Total tricks this phase will play = cards dealt to each player
        # (13 in standard Call Break, but kept general for testability).
        self._total_tricks = len(hands[0]) if hands and hands[0] else 0

        self._start_new_trick(starting_player)

    # ------------------------------------------------------------------
    # Trick management
    # ------------------------------------------------------------------

    def _start_new_trick(self, leading_player: int) -> None:
        """Initialise a new Trick with the given leader."""
        self.current_trick = Trick(leading_player=leading_player, num_players=self.num_players)

    def play_card(self, player_id: int, card: Card) -> Optional[int]:
        """
        Play `card` for `player_id` in the current trick.
        If the trick is now complete, resolve it and start the next one.
        Returns the winner player_id if a trick was just resolved, else None.
        Raises ValueError if the card is not a legal play.
        """
        if self.current_trick is None:
            raise ValueError("Playing phase is already complete; no trick in progress.")
        if card not in self.hands[player_id]:
            raise ValueError(f"Player {player_id} does not hold {card}")

        legal = self.legal_plays(player_id)
        if card not in legal:
            raise ValueError(
                f"{card} is not a legal play for player {player_id} "
                f"given the current trick (legal: {legal})"
            )

        # Trick.play_card is responsible for enforcing turn order
        # (raises ValueError if it isn't player_id's turn).
        self.current_trick.play_card(player_id, card)
        self.hands[player_id].remove(card)

        if not self.current_trick.is_complete:
            return None

        winner = self.current_trick.resolve()
        self.tricks_won[winner] += 1
        self.tricks_played.append(self.current_trick)

        if len(self.tricks_played) < self._total_tricks:
            self._start_new_trick(winner)
        else:
            self.current_trick = None  # all tricks played

        return winner

    def legal_plays(self, player_id: int) -> List[Card]:
        """Return the legal plays for player_id given the current trick state."""
        if self.current_trick is None:
            return []
        return self.current_trick.legal_plays(self.hands[player_id])

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    @property
    def is_complete(self) -> bool:
        """True when all tricks have been played."""
        return len(self.tricks_played) == self._total_tricks

    @property
    def current_player(self) -> int:
        """player_id whose turn it is to play."""
        if self.current_trick is None:
            raise RuntimeError("Playing phase is complete; there is no current player.")
        return self.current_trick.current_player

    def tricks_remaining(self) -> int:
        """Number of tricks not yet played."""
        return self._total_tricks - len(self.tricks_played)

    def cards_played_history(self) -> List[Tuple[int, Card]]:
        """All (player_id, card) pairs played so far across all tricks."""
        history: List[Tuple[int, Card]] = []
        for trick in self.tricks_played:
            history.extend(trick.plays)
        if self.current_trick is not None:
            history.extend(self.current_trick.plays)
        return history


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
        self.deck.reset()  # rebuild to a fresh 52 + shuffle, in case deck was reused
        self.hands = self.deck.deal(num_players=self.num_players)
        self.bidding = BiddingPhase(num_players=self.num_players)
        self.playing = None
        self.scores = []
        return self.hands

    # ------------------------------------------------------------------
    # Phase orchestration (wraps BiddingPhase / PlayingPhase so callers
    # don't need to manually wire the bid → play → score transitions)
    # ------------------------------------------------------------------

    def place_bid(self, player_id: int, bid: int) -> None:
        """
        Place a bid for player_id. Automatically starts the playing phase
        once all players have bid.
        """
        if self.bidding is None:
            raise RuntimeError("Round has not been dealt yet; call deal() first.")
        self.bidding.place_bid(player_id, bid)
        if self.bidding.is_complete():
            self._start_playing_phase()

    def play_card(self, player_id: int, card: Card) -> Optional[int]:
        """
        Play a card for player_id. Automatically calculates scores once
        the final trick is resolved.
        Returns the trick winner if a trick was just completed, else None.
        """
        if self.playing is None:
            raise RuntimeError("Playing phase has not started; bidding is not complete.")
        winner = self.playing.play_card(player_id, card)
        if self.playing.is_complete:
            self.calculate_scores()
        return winner

    def _start_playing_phase(self) -> None:
        """
        Transition from bidding to playing once all bids are in.
        The lead player rotates by round number so a different player
        leads the first trick each round (mirrors dealer rotation).
        """
        if self.bidding is None or not self.bidding.is_complete():
            raise RuntimeError("Cannot start playing phase before bidding is complete.")
        starting_player = self.round_number % self.num_players
        self.playing = PlayingPhase(hands=self.hands, starting_player=starting_player)

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def get_observation(self, player_id):
        if self.bidding is None:
            raise RuntimeError("Round has not been dealt yet; call deal() first.")

        if not self.bidding.is_complete():
            return {
                "round_number": self.round_number, "player_id": player_id,
                "hand": list(self.hands[player_id]),      # ← fine here, no cards removed yet
                "bids": list(self.bidding.bids),
                "phase": "bid",
                "legal_bids": self.bidding.valid_bids(player_id),
                "tricks_won": [0] * self.num_players,
                "cards_played_history": [],
                "current_trick_plays": [],
                "led_suit": None,
                "tricks_remaining": len(self.hands[player_id]),
            }

        if self.playing is None:
            self._start_playing()

        ct = self.playing.current_trick
        return {
            "round_number": self.round_number, "player_id": player_id,
            "hand": list(self.playing.hands[player_id]),  # ← fixed: tracks removed cards
            "bids": list(self.bidding.bids),
            "phase": "play",
            "tricks_won": list(self.playing.tricks_won),
            "cards_played_history": self.playing.cards_played_history(),
            "current_trick_plays": list(ct.plays) if ct else [],
            "led_suit": ct.led_suit if ct else None,
            "tricks_remaining": self.playing.tricks_remaining(),
            "legal_plays": [] if self.playing.is_complete else self.playing.legal_plays(player_id),
        }

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def calculate_scores(self) -> List[float]:
        """
        Compute and store per-player scores after playing phase completes.

        Scoring:
          - Met or exceeded bid:  score = bid + 0.1 * (tricks_won - bid)
          - Failed to meet bid:   score = -bid
        Returns the scores list.
        Raises RuntimeError if the playing phase is not complete.
        """
        if self.playing is None or not self.playing.is_complete:
            raise RuntimeError("Cannot calculate scores before the playing phase is complete.")

        scores: List[float] = []
        for player_id in range(self.num_players):
            bid = self.bidding.bids[player_id]
            won = self.playing.tricks_won[player_id]
            if won >= bid:
                score = bid + 0.1 * (won - bid)
            else:
                score = -float(bid)
            scores.append(score)

        self.scores = scores
        return scores

    def __repr__(self) -> str:
        return f"Round({self.round_number}, bids={self.bidding and self.bidding.bids})"
