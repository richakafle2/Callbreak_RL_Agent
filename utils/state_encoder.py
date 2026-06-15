"""
state_encoder.py
----------------
Converts a Call Break game observation dictionary into a flat numpy
float32 vector for consumption by neural network models.

Observation vector layout (174 dimensions total):

 Idx   Len  Description
 ---   ---  -----------
   0    52  Hand cards (binary: 1 = in hand)
  52    52  Cards played globally this round (binary)
 104    52  Cards in current trick, position-encoded (4 players × 13 ranks)
            Encoded as: player_offset * 13 + rank_offset_within_suit
            (captures which player played which card in this trick)
 156     4  Bids, normalised by 13
 160     4  Tricks won, normalised by 13
 164     4  One-hot: which player is leading the current trick
 168     1  Tricks remaining, normalised by 13
 169     1  Bid deficit = (bid - tricks_won) / 13  (negative = ahead)
 170     4  One-hot: learning agent's position at table (always player 0 = [1,0,0,0])
 ---   ---
 174  TOTAL
"""

import numpy as np
from typing import Dict, List, Optional
from environment.card import Card, Suit


OBS_DIM = 174


class StateEncoder:
    """
    Stateless encoder: converts observation dicts to numpy vectors.
    Can also encode separate static / history components for the
    Transformer encoder variant.
    """

    def __init__(self, num_players: int = 4, num_cards: int = 52):
        self.num_players = num_players
        self.num_cards = num_cards

    # ------------------------------------------------------------------
    # Full flat encoding (MLP / baseline)
    # ------------------------------------------------------------------

    def encode(self, observation: Dict, player_id: int = 0) -> np.ndarray:
        """
        Encode a complete observation dict into a (OBS_DIM,) float32 array.

        observation keys used:
          'hand', 'cards_played_history', 'current_trick_plays',
          'bids', 'tricks_won', 'tricks_remaining', 'leading_player'
        """
        raise NotImplementedError

    def _encode_hand(self, hand: List[Card]) -> np.ndarray:
        """Return 52-dim binary vector: 1 where card is in hand."""
        raise NotImplementedError

    def _encode_played_globally(self, history: List) -> np.ndarray:
        """
        Return 52-dim binary vector of all cards played this round
        (across all completed tricks + current trick).
        """
        raise NotImplementedError

    def _encode_current_trick(self, trick_plays: List) -> np.ndarray:
        """
        Encode the current trick as a 52-dim vector.
        Each position in the 52-card space is set to a value in [0, 1]
        indicating which player played it (player_id / num_players)
        so positional information is preserved.
        """
        raise NotImplementedError

    def _encode_bids(self, bids: List[Optional[int]]) -> np.ndarray:
        """Normalise bids to [0, 1]; encode un-placed bids as 0."""
        raise NotImplementedError

    def _encode_tricks_won(self, tricks_won: List[int]) -> np.ndarray:
        """Normalise tricks won to [0, 1]."""
        raise NotImplementedError

    def _encode_leading_player(self, leading_player: int) -> np.ndarray:
        """Return 4-dim one-hot vector for the current trick leader."""
        raise NotImplementedError

    def _encode_tricks_remaining(self, tricks_remaining: int) -> float:
        """Normalise tricks remaining to [0, 1]."""
        raise NotImplementedError

    def _encode_bid_deficit(self, bid: int, tricks_won: int) -> float:
        """
        Encode urgency: (bid - tricks_won) / 13.
        Positive = need more tricks, negative = ahead of bid.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Transformer-specific: split into static + sequence components
    # ------------------------------------------------------------------

    def encode_static(self, observation: Dict, player_id: int = 0) -> np.ndarray:
        """
        Return the non-sequential features only (hand, bids, tricks, etc.)
        for use as the static input to TransformerEncoder.
        """
        raise NotImplementedError

    def encode_history_sequence(
        self, history: List
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return (card_ids, player_ids, padding_mask) for CardHistoryEncoder.

        card_ids    : (max_seq,) int64 — 0-51 card index, 0 = padding
        player_ids  : (max_seq,) int64 — 0-3 player index, 0 = padding
        padding_mask: (max_seq,) bool  — True where position is padding

        max_seq = 52 (full round of plays).
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Action encoding / decoding
    # ------------------------------------------------------------------

    @staticmethod
    def bid_action_to_int(action: int) -> int:
        """Convert action index (0-12) to actual bid value (1-13)."""
        return action + 1

    @staticmethod
    def int_to_bid_action(bid: int) -> int:
        """Convert bid value (1-13) to action index (0-12)."""
        return bid - 1

    @staticmethod
    def card_to_action(card: Card) -> int:
        """Convert a Card to its action index (0-51)."""
        return card.index

    @staticmethod
    def action_to_card(action: int, hand: List[Card]) -> Optional[Card]:
        """Return the Card in hand corresponding to action index, or None."""
        for card in hand:
            if card.index == action:
                return card
        return None
