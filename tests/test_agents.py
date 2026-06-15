"""
tests/test_agents.py
---------------------
Unit tests for heuristic and RL agent behaviour.
"""

import pytest
from environment.card import Card, Suit, Rank
from agents.random_agent import RandomAgent
from agents.heuristic.safe_bet_agent import SafeBetAgent
from agents.heuristic.safe_play_agent import SafePlayAgent
from agents.heuristic.basic_bet_agent import BasicBetAgent


def make_observation(hand, legal_plays=None, bids=None, tricks_won=None):
    """Helper to build a minimal observation dict for agent testing."""
    return {
        "hand": hand,
        "legal_plays": legal_plays or hand,
        "bids": bids or [3, 3, 3, 3],
        "tricks_won": tricks_won or [0, 0, 0, 0],
        "cards_played_history": [],
        "current_trick_plays": [],
        "led_suit": None,
        "tricks_remaining": 13,
        "player_id": 0,
    }


class TestRandomAgent:
    def test_bid_in_range(self):
        agent = RandomAgent(player_id=0, min_bid=1, max_bid=13, seed=42)
        hand = [Card(Rank.ACE, Suit.SPADES), Card(Rank.KING, Suit.HEARTS)]
        obs = make_observation(hand)
        for _ in range(50):
            bid = agent.bid(obs)
            assert 1 <= bid <= 13

    def test_play_returns_legal_card(self):
        hand = [Card(Rank.ACE, Suit.SPADES), Card(Rank.KING, Suit.HEARTS)]
        legal = [Card(Rank.KING, Suit.HEARTS)]
        obs = make_observation(hand, legal_plays=legal)
        agent = RandomAgent(player_id=0, seed=0)
        action = agent.play(obs)
        assert action == Card(Rank.KING, Suit.HEARTS).index


class TestSafeBetAgent:
    def test_bid_at_least_one(self):
        agent = SafeBetAgent(player_id=0)
        hand = [Card(r, Suit.CLUBS) for r in [Rank.TWO, Rank.THREE, Rank.FOUR]]
        obs = make_observation(hand)
        assert agent.bid(obs) >= 1

    def test_ace_heavy_hand_bids_high(self):
        agent = SafeBetAgent(player_id=0)
        hand = [
            Card(Rank.ACE, Suit.SPADES),
            Card(Rank.ACE, Suit.HEARTS),
            Card(Rank.ACE, Suit.CLUBS),
            Card(Rank.ACE, Suit.DIAMONDS),
            Card(Rank.KING, Suit.SPADES),
        ]
        obs = make_observation(hand)
        assert agent.bid(obs) >= 4

    def test_play_returns_legal_card(self):
        agent = SafeBetAgent(player_id=0)
        hand = [Card(Rank.ACE, Suit.HEARTS), Card(Rank.TWO, Suit.SPADES)]
        obs = make_observation(hand, legal_plays=hand)
        action = agent.play(obs)
        assert action in [c.index for c in hand]


class TestSafePlayAgent:
    def test_memory_resets(self):
        agent = SafePlayAgent(player_id=0)
        agent._played_cards.add(0)
        agent.reset()
        assert len(agent._played_cards) == 0

    def test_play_from_legal_only(self):
        agent = SafePlayAgent(player_id=0)
        hand = [Card(Rank.ACE, Suit.HEARTS), Card(Rank.TWO, Suit.HEARTS)]
        legal = [Card(Rank.TWO, Suit.HEARTS)]
        obs = make_observation(hand, legal_plays=legal)
        action = agent.play(obs)
        assert action == Card(Rank.TWO, Suit.HEARTS).index


class TestBasicBetAgent:
    def test_bid_with_aces(self):
        agent = BasicBetAgent(player_id=0)
        hand = [
            Card(Rank.ACE, Suit.HEARTS),
            Card(Rank.ACE, Suit.CLUBS),
            Card(Rank.TWO, Suit.SPADES),
        ]
        obs = make_observation(hand)
        # Should count at least 2 (the two aces)
        assert agent.bid(obs) >= 2

    def test_play_wins_when_possible(self):
        """BasicBetAgent should prefer winning the trick."""
        agent = BasicBetAgent(player_id=0)
        # Current trick: 5 of hearts led
        current_trick = [(0, Card(Rank.FIVE, Suit.HEARTS))]
        hand = [Card(Rank.ACE, Suit.HEARTS), Card(Rank.TWO, Suit.CLUBS)]
        obs = make_observation(hand, legal_plays=[Card(Rank.ACE, Suit.HEARTS)])
        obs["current_trick_plays"] = current_trick
        obs["led_suit"] = Suit.HEARTS
        action = agent.play(obs)
        assert action == Card(Rank.ACE, Suit.HEARTS).index
