"""
callbreak_env.py
----------------
OpenAI Gymnasium-compatible environment for Call Break.

Observation space  — flat numpy array (see StateEncoder for breakdown, 174-dim)
Action space       — Discrete(52) for play, Discrete(13) for bid
Reward             — shaped per-trick + final round score
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from environment.round import Round
from environment.card import Card
from utils.state_encoder import StateEncoder, OBS_DIM
from utils.reward_shaper import RewardShaper

NUM_CARDS = 52
NUM_BID_ACTIONS = 13   # bids 1-13 → action indices 0-12


class CallBreakEnv(gym.Env):
    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(self, opponent_agents, num_rounds=5, reward_shaping=True, seed=None, render_mode=None):
        super().__init__()
        assert len(opponent_agents) == 3
        self.opponent_agents = opponent_agents
        self.num_rounds = num_rounds
        self.render_mode = render_mode
        self.encoder = StateEncoder()
        self.reward_shaper = RewardShaper(enabled=reward_shaping)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Discrete(NUM_CARDS)
        self._rng = np.random.default_rng(seed)
        self._round: Optional[Round] = None
        self._round_number: int = 0
        self._total_scores: List[float] = [0.0] * 4

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._round_number = 0
        self._total_scores = [0.0] * 4
        for agent in self.opponent_agents:
            agent.reset()
        self._start_new_round()
        return self._get_observation(), self._build_info()

    def step(self, action):
        reward = 0.0
        terminated = False
        truncated = False
        phase = self._round.get_observation(0)["phase"]

        if phase == "bid":
            self._apply_agent_bid(action)
            reward += self._run_opponents_until_agent_turn()
        else:
            reward += self._apply_agent_play(action)

            if self._round.playing.is_complete:
                # Agent's play completed the round (they were last to play)
                reward, terminated = self._finalise_round(reward)
            else:
                # Normal case: opponents finish the trick (and maybe the round)
                reward += self._run_opponents_until_agent_turn()
                if self._round.playing.is_complete:
                    reward, terminated = self._finalise_round(reward)

        return self._get_observation(), float(reward), terminated, truncated, self._build_info()

    def _get_observation(self):
        if self._round is None or self._round_number >= self.num_rounds:
            return np.zeros(OBS_DIM, dtype=np.float32)
        return self.encoder.encode(self._round.get_observation(0), player_id=0)

    def _get_legal_action_mask(self):
        mask = np.zeros(NUM_CARDS, dtype=bool)
        if self._round is None or self._round_number >= self.num_rounds:
            return mask
        obs = self._round.get_observation(0)
        if obs["phase"] == "bid":
            for bid in obs.get("legal_bids", list(range(1, 14))):
                mask[bid - 1] = True
        else:
            for card in obs.get("legal_plays", []):
                mask[card.index] = True
        return mask

    def _start_new_round(self):
        int_seed = int(self._rng.integers(0, 2**31))
        self._round = Round(round_number=self._round_number, seed=int_seed)
        self._round.deal()

    def _run_opponents_until_agent_turn(self):
        shaped_reward = 0.0
        while True:
            if not self._round.bidding.is_complete():
                bidder = self._round.bidding.current_bidder
                if bidder == 0:
                    break
                obs = self._round.get_observation(bidder)
                self._round.place_bid(bidder, self.opponent_agents[bidder - 1].bid(obs))
                continue

            if self._round.playing is None or self._round.playing.is_complete:
                break
            if self._round.playing.current_player == 0:
                break

            player = self._round.playing.current_player
            obs = self._round.get_observation(player)
            card_index = self.opponent_agents[player - 1].play(obs)
            legal = self._round.playing.legal_plays(player)
            winner = self._round.play_card(player, self._find_card_by_index(card_index, legal))

            if winner is not None:
                curr_tricks = self._round.playing.tricks_won[0]
                if winner == 0:
                    shaped_reward += self.reward_shaper.trick_won_reward(
                        curr_tricks, self._round.bidding.bids[0]
                    )
                if self._round.playing.is_complete:
                    break

        return shaped_reward

    def _apply_agent_bid(self, action):
        self._round.place_bid(0, int(np.clip(action + 1, 1, 13)))

    def _apply_agent_play(self, card_index):
        legal = self._round.playing.legal_plays(0)
        card = self._find_card_by_index(card_index, legal)
        if card not in legal:
            raise ValueError(f"Card index {card_index} not legal. Legal: {[c.index for c in legal]}")
        winner = self._round.play_card(0, card)
        if winner is None:
            return 0.0
        curr_tricks = self._round.playing.tricks_won[0]
        bid = self._round.bidding.bids[0]
        if winner == 0:
            return self.reward_shaper.trick_won_reward(curr_tricks, bid)
        return self.reward_shaper.trick_lost_reward(curr_tricks, bid)

    def _round_end_reward(self):
        return self.reward_shaper.round_score(
            self._round.playing.tricks_won[0],
            self._round.bidding.bids[0],
        )

    def _advance_to_next_round(self):
        for i, score in enumerate(self._round.scores):
            self._total_scores[i] += score
        self._round_number += 1
        if self._round_number < self.num_rounds:
            self._start_new_round()

    def _finalise_round(self, reward_so_far):
        reward_so_far += self._round_end_reward()
        self._advance_to_next_round()
        if self._round_number >= self.num_rounds:
            return reward_so_far, True
        reward_so_far += self._run_opponents_until_agent_turn()
        return reward_so_far, False

    def render(self):
        if self._round is None or self._round_number >= self.num_rounds:
            return "Game over."
        obs = self._round.get_observation(0)
        hand = sorted(obs["hand"], key=lambda c: (c.suit, c.rank))
        lines = [
            f"╔══ Round {self._round_number + 1}/{self.num_rounds}  Phase: {obs['phase'].upper()} ══╗",
            f"  Hand       : {' '.join(str(c) for c in hand)}",
            f"  Bids       : {obs['bids']}",
            f"  Tricks won : {obs.get('tricks_won', ['-'] * 4)}",
        ]
        trick_plays = obs.get("current_trick_plays", [])
        if trick_plays:
            lines.append(f"  Trick      : {'  '.join(f'P{p}:{c}' for p,c in trick_plays)}")
        lines.append(f"  Totals     : {[f'{s:.1f}' for s in self._total_scores]}")
        lines.append(f"╚{'═' * 40}╝")
        output = "\n".join(lines)
        if self.render_mode == "human":
            print(output); return None
        return output

    def close(self): pass

    def _find_card_by_index(self, index, cards):
        for card in cards:
            if card.index == index:
                return card
        return cards[0] if cards else (_ for _ in ()).throw(ValueError(f"No cards (index {index})"))

    def _build_info(self):
        info = {"round_number": self._round_number, "total_scores": list(self._total_scores),
                "phase": "done", "bids": [None]*4, "tricks_won": [0]*4, "legal_actions": []}
        if self._round is None or self._round_number >= self.num_rounds:
            return info
        obs = self._round.get_observation(0)
        info.update({"phase": obs["phase"], "bids": obs["bids"],
                     "tricks_won": obs.get("tricks_won", [0]*4),
                     "legal_actions": np.where(self._get_legal_action_mask())[0].tolist()})
        return info