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

# ======================================================================
# Shared helpers (at module level, used by TestRound)
# ======================================================================

def make_round(round_number: int = 0, seed: int = 42) -> Round:
    """Return a freshly dealt Round ready for bidding."""
    r = Round(round_number=round_number, seed=seed)
    r.deal()
    return r

def bid_all(r: Round, bids: tuple = (3, 3, 3, 3)) -> None:
    """Place bids for all four players from player 0 onward."""
    for player_id, bid in enumerate(bids):
        r.place_bid(player_id, bid)

def play_to_completion(r: Round) -> None:
    """Play every remaining trick by always choosing the first legal card."""
    while not r.playing.is_complete:
        player = r.playing.current_player
        legal  = r.playing.legal_plays(player)
        r.play_card(player, legal[0])


# ======================================================================
# Round tests
# ======================================================================

class TestRound:

    def _expected_score(self, bid: int, won: int) -> float:
        """Mirror of Round.calculate_scores formula — used to verify correctness."""
        return bid + 0.1 * (won - bid) if won >= bid else -float(bid)

    # ------------------------------------------------------------------
    # test_full_round_scores
    # ------------------------------------------------------------------

    def test_full_round_scores(self):
        """A complete round produces exactly 4 scores matching the formula."""
        r = make_round(seed=42)
        bid_all(r, bids=(3, 3, 3, 3))
        play_to_completion(r)

        assert len(r.scores) == 4
        assert sum(r.playing.tricks_won) == 13

        for i in range(4):
            bid = r.bidding.bids[i]
            won = r.playing.tricks_won[i]
            assert r.scores[i] == pytest.approx(self._expected_score(bid, won)), \
                f"Player {i}: bid={bid}, won={won}, score={r.scores[i]}"

    def test_scores_formula_holds_across_seeds(self):
        """Formula is applied correctly regardless of deal or bids."""
        for seed in range(10):
            r = make_round(seed=seed)
            bid_all(r, bids=(3, 3, 3, 3))
            play_to_completion(r)
            for i in range(4):
                assert r.scores[i] == pytest.approx(
                    self._expected_score(r.bidding.bids[i], r.playing.tricks_won[i])
                )

    def test_scores_exceeded_bid_earns_bonus(self):
        """
        Bidding low (1) ensures at least one player exceeds their bid.
        Their score must be bid + 0.1 * excess, which is > bid.
        """
        r = make_round(seed=0)
        bid_all(r, bids=(1, 1, 1, 1))   # only 4 tricks bid, 13 to distribute
        play_to_completion(r)

        exceeded = [i for i in range(4) if r.playing.tricks_won[i] > 1]
        assert len(exceeded) > 0, "Expected at least one player to exceed a bid of 1"
        for i in exceeded:
            excess = r.playing.tricks_won[i] - 1
            assert r.scores[i] == pytest.approx(1 + 0.1 * excess)
            assert r.scores[i] > 1

    def test_scores_failed_bid_is_negative(self):
        """
        Bidding 13 with 4 players means at least 3 will fail.
        Each failed player's score must equal exactly -bid = -13.
        """
        r = make_round(seed=7)
        bid_all(r, bids=(13, 13, 13, 13))
        play_to_completion(r)

        failed = [i for i in range(4) if r.playing.tricks_won[i] < 13]
        assert len(failed) >= 3
        for i in failed:
            assert r.scores[i] == pytest.approx(-13.0)

    def test_scores_raise_before_round_complete(self):
        """calculate_scores raises RuntimeError if called mid-round."""
        r = make_round(seed=1)
        bid_all(r)
        for _ in range(3 * 4):          # play 3 full tricks (12 cards played)
            player = r.playing.current_player
            r.play_card(player, r.playing.legal_plays(player)[0])
        with pytest.raises(RuntimeError):
            r.calculate_scores()

    def test_scores_stable_on_repeated_call(self):
        """calculate_scores called twice returns the same values."""
        r = make_round(seed=3)
        bid_all(r)
        play_to_completion(r)
        assert r.calculate_scores() == r.scores

    def test_total_tricks_always_13(self):
        """13 tricks are always fully distributed regardless of bids."""
        for seed in range(5):
            r = make_round(seed=seed)
            bid_all(r)
            play_to_completion(r)
            assert sum(r.playing.tricks_won) == 13

    # ------------------------------------------------------------------
    # test_observation_structure
    # ------------------------------------------------------------------

    def test_observation_structure(self):
        """get_observation returns all required keys with correct values."""

        # ---- Bidding phase -------------------------------------------
        r = make_round(seed=42)
        obs = r.get_observation(0)

        required_bid_keys = {
            "phase", "player_id", "round_number",
            "hand", "bids", "legal_bids",
            "tricks_won", "cards_played_history",
            "current_trick_plays", "led_suit", "tricks_remaining",
        }
        assert required_bid_keys.issubset(obs.keys()), \
            f"Missing bid-phase keys: {required_bid_keys - obs.keys()}"

        assert obs["phase"]                 == "bid"
        assert obs["player_id"]             == 0
        assert obs["round_number"]          == 0
        assert len(obs["hand"])             == 13
        assert obs["bids"]                  == [None, None, None, None]
        assert obs["tricks_won"]            == [0, 0, 0, 0]
        assert obs["cards_played_history"]  == []
        assert obs["current_trick_plays"]   == []
        assert obs["led_suit"]              is None
        assert obs["tricks_remaining"]      == 13
        assert all(1 <= b <= 13 for b in obs["legal_bids"])
        assert len(obs["legal_bids"])       == 13

        # ---- Bids are reflected mid-bidding --------------------------
        r.place_bid(0, 5)
        obs_mid = r.get_observation(0)
        assert obs_mid["bids"][0] == 5
        assert obs_mid["bids"][1] is None       # player 1 hasn't bid yet

        r.place_bid(1, 3)
        r.place_bid(2, 3)
        r.place_bid(3, 2)

        # ---- Playing phase -------------------------------------------
        obs_play = r.get_observation(0)

        required_play_keys = {
            "phase", "player_id", "round_number",
            "hand", "bids", "tricks_won",
            "cards_played_history", "current_trick_plays",
            "led_suit", "tricks_remaining", "legal_plays",
        }
        assert required_play_keys.issubset(obs_play.keys()), \
            f"Missing play-phase keys: {required_play_keys - obs_play.keys()}"

        assert obs_play["phase"]                == "play"
        assert obs_play["bids"]                 == [5, 3, 3, 2]
        assert obs_play["tricks_won"]           == [0, 0, 0, 0]
        assert obs_play["tricks_remaining"]     == 13
        assert obs_play["cards_played_history"] == []
        assert len(obs_play["legal_plays"])     > 0

    def test_observation_hand_shrinks_after_each_trick(self):
        """Player 0's hand loses exactly one card per trick played."""
        r = make_round(seed=5)
        bid_all(r)
        for trick_num in range(13):
            assert len(r.get_observation(0)["hand"]) == 13 - trick_num
            for _ in range(4):
                player = r.playing.current_player
                r.play_card(player, r.playing.legal_plays(player)[0])

    def test_observation_history_grows_with_each_play(self):
        """cards_played_history gains one entry for every card played."""
        r = make_round(seed=8)
        bid_all(r)
        cards_played = 0
        while not r.playing.is_complete:
            assert len(r.get_observation(0)["cards_played_history"]) == cards_played
            player = r.playing.current_player
            r.play_card(player, r.playing.legal_plays(player)[0])
            cards_played += 1
        assert len(r.get_observation(0)["cards_played_history"]) == 52

    def test_observation_led_suit_set_after_first_play_in_trick(self):
        """led_suit is None before the lead card is played, set afterward."""
        r = make_round(seed=2)
        bid_all(r)
        assert r.get_observation(0)["led_suit"] is None
        leader    = r.playing.current_player
        lead_card = r.playing.legal_plays(leader)[0]
        r.play_card(leader, lead_card)
        assert r.get_observation(0)["led_suit"] == lead_card.suit

    def test_observation_legal_plays_always_subset_of_hand(self):
        """Every card in legal_plays must be in the player's current hand."""
        r = make_round(seed=9)
        bid_all(r)
        for _ in range(8):
            player = r.playing.current_player
            r.play_card(player, r.playing.legal_plays(player)[0])
        for pid in range(4):
            obs      = r.get_observation(pid)
            hand_set = {c.index for c in obs["hand"]}
            legal_set = {c.index for c in obs["legal_plays"]}
            assert legal_set.issubset(hand_set), \
                f"Player {pid} has legal plays not in hand: {legal_set - hand_set}"

    def test_observation_before_deal_raises(self):
        """get_observation before deal() raises RuntimeError."""
        with pytest.raises(RuntimeError):
            Round(round_number=0).get_observation(0)