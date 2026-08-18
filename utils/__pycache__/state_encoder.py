"""
state_encoder.py
----------------
Converts a Round observation dict into fixed-size numeric tensors for the
neural network: a flat vector for the MLP encoder, or a (static features,
card-play history sequence) pair for the Transformer encoder.

Confirmed against round.py / callbreak_env.py: 'hand', 'cards_played_history',
'current_trick_plays', 'bids', 'tricks_won', 'legal_plays', 'led_suit' are
all real keys returned by Round.get_observation(). 'tricks_remaining' is
also real (Round.get_observation() returns it directly). 'leading_player'
is NOT currently emitted by Round.get_observation() or CallBreakEnv's info
dict — encode()'s default of 0 is used in practice everywhere today, so
_encode_leading_player never sees a real value. Flagging this rather than
silently treating it as confirmed.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from environment.card import Card

# --- Component sizes (kept as named constants so OBS_DIM can't silently
# drift out of sync with encode()'s actual concatenation order) ---
_HAND_DIM = 52
_PLAYED_GLOBALLY_DIM = 52
_CURRENT_TRICK_DIM = 52
_BIDS_DIM = 4
_TRICKS_WON_DIM = 4
_LEADING_PLAYER_DIM = 4
_TRICKS_REMAINING_DIM = 1
_BID_DEFICIT_DIM = 1

OBS_DIM = (
    _HAND_DIM
    + _PLAYED_GLOBALLY_DIM
    + _CURRENT_TRICK_DIM
    + _BIDS_DIM
    + _TRICKS_WON_DIM
    + _LEADING_PLAYER_DIM
    + _TRICKS_REMAINING_DIM
    + _BID_DEFICIT_DIM
)  # 170

# Static-features width for the Transformer encoder variant: same as OBS_DIM
# but WITHOUT the flat 'played globally' bag-of-cards vector, since that
# same information is instead carried by the ordered history sequence
# (encode_history) fed to CardHistoryEncoder.
STATIC_DIM = OBS_DIM - _PLAYED_GLOBALLY_DIM  # 118

MAX_HISTORY_LEN = 52  # at most 52 cards get played in a round (13 tricks * 4)


class StateEncoder:
    """
    Encodes Round observations into neural-network-ready tensors.
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
        hand_vec = self._encode_hand(observation.get("hand", []))
        played_vec = self._encode_played_globally(observation.get("cards_played_history", []))
        trick_vec = self._encode_current_trick(observation.get("current_trick_plays", []))
        bids_vec = self._encode_bids(observation.get("bids", [None] * self.num_players))
        tricks_won_vec = self._encode_tricks_won(observation.get("tricks_won", [0] * self.num_players))
        leader_vec = self._encode_leading_player(observation.get("leading_player", 0))

        tricks_remaining = observation.get("tricks_remaining", 13)
        tricks_remaining_scalar = self._encode_tricks_remaining(tricks_remaining)

        my_bid = observation.get("bids", [None] * self.num_players)[player_id]
        my_tricks_won = observation.get("tricks_won", [0] * self.num_players)[player_id]
        bid_deficit_scalar = self._encode_bid_deficit(
            my_bid if my_bid is not None else 0, my_tricks_won
        )

        return np.concatenate(
            [
                hand_vec,
                played_vec,
                trick_vec,
                bids_vec,
                tricks_won_vec,
                leader_vec,
                np.array([tricks_remaining_scalar], dtype=np.float32),
                np.array([bid_deficit_scalar], dtype=np.float32),
            ]
        ).astype(np.float32)

    def _encode_hand(self, hand: List[Card]) -> np.ndarray:
        """Return 52-dim binary vector: 1 where card is in hand."""
        vec = np.zeros(self.num_cards, dtype=np.float32)
        for card in hand:
            vec[card.index] = 1.0
        return vec

    def _encode_played_globally(self, history: List) -> np.ndarray:
        """
        Return 52-dim binary vector of all cards played this round
        (across all completed tricks + current trick).

        `history` is expected to be a flat list of Card objects (or
        (player, Card) pairs -- either works since we only need .index).
        """
        vec = np.zeros(self.num_cards, dtype=np.float32)
        for entry in history:
            card = entry[1] if isinstance(entry, (tuple, list)) else entry
            vec[card.index] = 1.0
        return vec

    def _encode_current_trick(self, trick_plays: List) -> np.ndarray:
        """
        Encode the current trick as a 52-dim vector.
        Each position in the 52-card space is set to a value in [0, 1]
        indicating which player played it (player_id / num_players)
        so positional information is preserved.
        """
        vec = np.zeros(self.num_cards, dtype=np.float32)
        for seat, card in trick_plays:
            # +1 so seat 0's play is distinguishable from "no play" (0.0).
            vec[card.index] = (seat + 1) / self.num_players
        return vec

    def _encode_bids(self, bids: List[Optional[int]]) -> np.ndarray:
        """Normalise bids to [0, 1]; encode un-placed bids as 0."""
        vec = np.zeros(self.num_players, dtype=np.float32)
        for i, bid in enumerate(bids):
            if bid is not None:
                vec[i] = bid / 13.0
        return vec

    def _encode_tricks_won(self, tricks_won: List[int]) -> np.ndarray:
        """Normalise tricks won to [0, 1]."""
        return np.asarray(tricks_won, dtype=np.float32) / 13.0

    def _encode_leading_player(self, leading_player: int) -> np.ndarray:
        """Return 4-dim one-hot vector for the current trick leader."""
        vec = np.zeros(self.num_players, dtype=np.float32)
        vec[leading_player] = 1.0
        return vec

    def _encode_tricks_remaining(self, tricks_remaining: int) -> float:
        """Normalise tricks remaining to [0, 1]."""
        return float(tricks_remaining) / 13.0

    def _encode_bid_deficit(self, bid: int, tricks_won: int) -> float:
        """
        Encode urgency: (bid - tricks_won) / 13.
        Positive = need more tricks, negative = ahead of bid.
        """
        return float(bid - tricks_won) / 13.0

    # ------------------------------------------------------------------
    # Transformer-specific: split into static + sequence components
    # ------------------------------------------------------------------

    def encode_static(self, observation: Dict, player_id: int = 0) -> np.ndarray:
        """
        Return the non-sequential features only (hand, bids, tricks, etc.)
        for use as the static input to TransformerEncoder. Identical to
        encode() but omits the flat 'played globally' bag-of-cards vector,
        since encode_history() below carries that information with richer
        (ordered, per-player) structure instead.
        """
        hand_vec = self._encode_hand(observation.get("hand", []))
        trick_vec = self._encode_current_trick(observation.get("current_trick_plays", []))
        bids_vec = self._encode_bids(observation.get("bids", [None] * self.num_players))
        tricks_won_vec = self._encode_tricks_won(observation.get("tricks_won", [0] * self.num_players))
        leader_vec = self._encode_leading_player(observation.get("leading_player", 0))

        tricks_remaining = observation.get("tricks_remaining", 13)
        tricks_remaining_scalar = self._encode_tricks_remaining(tricks_remaining)

        my_bid = observation.get("bids", [None] * self.num_players)[player_id]
        my_tricks_won = observation.get("tricks_won", [0] * self.num_players)[player_id]
        bid_deficit_scalar = self._encode_bid_deficit(
            my_bid if my_bid is not None else 0, my_tricks_won
        )

        return np.concatenate(
            [
                hand_vec,
                trick_vec,
                bids_vec,
                tricks_won_vec,
                leader_vec,
                np.array([tricks_remaining_scalar], dtype=np.float32),
                np.array([bid_deficit_scalar], dtype=np.float32),
            ]
        ).astype(np.float32)

    def encode_history(
        self, observation: Dict, max_len: int = MAX_HISTORY_LEN
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Encode 'cards_played_history' as a padded sequence for
        CardHistoryEncoder / TransformerEncoder.

        Returns:
          card_history_ids   : (max_len,) int64, card index (0-51); padded
                                positions are filled with 0 (masked out, so
                                the value itself is never read as real data).
          player_history_ids : (max_len,) int64, seat that played the card
                                (0-3); padded positions filled with 0.
          history_mask        : (max_len,) bool, True = padding (matches
                                CardHistoryEncoder's `padding_mask` convention,
                                where True marks entries to exclude from the
                                pooled representation).

        `history` is expected to be a list of (seat, Card) tuples in the
        order they were played this round.
        """
        history = observation.get("cards_played_history", [])
        card_ids = np.zeros(max_len, dtype=np.int64)
        player_ids = np.zeros(max_len, dtype=np.int64)
        mask = np.ones(max_len, dtype=np.bool_)  # start fully padded

        # Only the most recent `max_len` plays matter if a round could
        # somehow exceed it (shouldn't happen at 52, but guards against it).
        trimmed = history[-max_len:]
        for i, entry in enumerate(trimmed):
            seat, card = entry
            card_ids[i] = card.index
            player_ids[i] = seat
            mask[i] = False  # real data, not padding

        return card_ids, player_ids, mask