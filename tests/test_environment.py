"""
tests/test_environment.py
--------------------------
Unit tests for Call Break game engine components.
Run with: pytest tests/test_environment.py -v
"""

import pytest
from environment.card import Card, Suit, Rank
from environment.deck import Deck
from environment.trick import Trick
from environment.round import Round, BiddingPhase


# ======================================================================
# Card tests
# ======================================================================

class TestCard:
    def test_trump_detection(self):
        ace_spades = Card(Rank.ACE, Suit.SPADES)
        ace_hearts = Card(Rank.ACE, Suit.HEARTS)
        assert ace_spades.is_trump is True
        assert ace_hearts.is_trump is False

    def test_index_uniqueness(self):
        """Each card must have a unique index 0-51."""
        indices = set()
        for suit in Suit:
            for rank in Rank:
                idx = Card(rank, suit).index
                assert 0 <= idx <= 51
                assert idx not in indices
                indices.add(idx)
        assert len(indices) == 52

    def test_from_index_roundtrip(self):
        """Card.from_index(card.index) should reconstruct the original card."""
        for suit in Suit:
            for rank in Rank:
                card = Card(rank, suit)
                assert Card.from_index(card.index) == card

    def test_beats_trump_over_nontrump(self):
        two_spades = Card(Rank.TWO, Suit.SPADES)
        ace_hearts = Card(Rank.ACE, Suit.HEARTS)
        assert two_spades.beats(ace_hearts, led_suit=Suit.HEARTS)

    def test_beats_higher_rank_same_suit(self):
        ace_hearts = Card(Rank.ACE, Suit.HEARTS)
        king_hearts = Card(Rank.KING, Suit.HEARTS)
        assert ace_hearts.beats(king_hearts, led_suit=Suit.HEARTS)
        assert not king_hearts.beats(ace_hearts, led_suit=Suit.HEARTS)

    def test_beats_offsuit_loses(self):
        """A card not following suit and not trump cannot beat led-suit cards."""
        two_clubs = Card(Rank.TWO, Suit.CLUBS)
        three_hearts = Card(Rank.THREE, Suit.HEARTS)
        # clubs not led, hearts led → clubs loses
        assert not two_clubs.beats(three_hearts, led_suit=Suit.HEARTS)


# ======================================================================
# Deck tests
# ======================================================================

class TestDeck:
    def test_build_52_cards(self):
        deck = Deck()
        assert len(deck) == 52

    def test_deal_four_hands(self):
        deck = Deck(seed=42)
        deck.shuffle()
        hands = deck.deal(num_players=4)
        assert len(hands) == 4
        assert all(len(h) == 13 for h in hands)
        # No duplicate cards across hands
        all_cards = [c for h in hands for c in h]
        assert len(set(c.index for c in all_cards)) == 52

    def test_deal_uneven_raises(self):
        deck = Deck()
        with pytest.raises(ValueError):
            deck.deal(num_players=3)   # 52 % 3 != 0

    def test_shuffle_changes_order(self):
        d1 = Deck(seed=1)
        d2 = Deck(seed=2)
        d1.shuffle()
        d2.shuffle()
        # Very unlikely to be identical
        assert [c.index for c in d1.cards] != [c.index for c in d2.cards]


# ======================================================================
# Trick tests
# ======================================================================

class TestTrick:
    def _make_trick(self, leading_player: int = 0) -> Trick:
        return Trick(leading_player=leading_player, num_players=4)

    def test_initial_state(self):
        trick = self._make_trick(0)
        assert not trick.is_complete
        assert trick.led_suit is None
        assert trick.current_player == 0

    def test_led_suit_set_on_first_play(self):
        trick = self._make_trick(0)
        card = Card(Rank.FIVE, Suit.HEARTS)
        hand = [card]
        # Must be legal (first play)
        assert card in trick.legal_plays(hand)
        trick.play_card(0, card)
        assert trick.led_suit == Suit.HEARTS

    def test_must_follow_suit(self):
        trick = self._make_trick(0)
        trick.play_card(0, Card(Rank.FIVE, Suit.HEARTS))
        hand = [Card(Rank.ACE, Suit.HEARTS), Card(Rank.TEN, Suit.CLUBS)]
        legal = trick.legal_plays(hand)
        assert Card(Rank.ACE, Suit.HEARTS) in legal
        assert Card(Rank.TEN, Suit.CLUBS) not in legal

    def test_trump_when_void(self):
        trick = self._make_trick(0)
        trick.play_card(0, Card(Rank.FIVE, Suit.HEARTS))
        hand = [Card(Rank.TWO, Suit.SPADES), Card(Rank.TEN, Suit.CLUBS)]
        legal = trick.legal_plays(hand)
        # Void in hearts — both are legal (spade trump or any discard)
        assert Card(Rank.TWO, Suit.SPADES) in legal

    def test_trick_winner(self):
        trick = self._make_trick(0)
        trick.play_card(0, Card(Rank.FIVE, Suit.HEARTS))
        trick.play_card(1, Card(Rank.ACE, Suit.HEARTS))
        trick.play_card(2, Card(Rank.TWO, Suit.SPADES))   # trump
        trick.play_card(3, Card(Rank.KING, Suit.HEARTS))
        assert trick.is_complete
        assert trick.resolve() == 2   # trump wins

    def test_current_winner_updates(self):
        trick = self._make_trick(0)
        trick.play_card(0, Card(Rank.FIVE, Suit.HEARTS))
        assert trick.current_winner() == 0
        trick.play_card(1, Card(Rank.ACE, Suit.HEARTS))
        assert trick.current_winner() == 1


# ======================================================================
# Bidding phase tests
# ======================================================================

class TestBiddingPhase:
    def test_bidding_order(self):
        phase = BiddingPhase(num_players=4)
        for i in range(4):
            assert phase.current_bidder == i
            phase.place_bid(i, bid=3)
        assert phase.is_complete()

    def test_invalid_bid_raises(self):
        phase = BiddingPhase()
        with pytest.raises(ValueError):
            phase.place_bid(0, bid=0)   # below min_bid
        with pytest.raises(ValueError):
            phase.place_bid(0, bid=14)  # above max_bid

    def test_wrong_player_raises(self):
        phase = BiddingPhase()
        with pytest.raises(ValueError):
            phase.place_bid(1, bid=3)   # player 1 can't bid before player 0


# ======================================================================
# Round tests
# ======================================================================

class TestRound:
    def test_full_round_scores(self):
        """A complete round should produce 4 scores without errors."""
        # TODO: implement once game logic is in place
        pytest.skip("Implement after Round is complete")

    def test_observation_structure(self):
        """get_observation should return expected keys."""
        pytest.skip("Implement after Round is complete")
