"""
play.py
-------
Interactive CLI to watch or play against the trained agent.

Usage:
  python scripts/play.py                          # watch 4 heuristic agents play
  python scripts/play.py --checkpoint best.pt     # watch trained agent vs heuristics
  python scripts/play.py --human                  # play the learnable seat yourself

NOTE ON ARCHITECTURE (please correct me if wrong):
  CallBreakEnv is single-agent-vs-3-fixed-opponents: seat 0 is the seat the
  env steps for; opponent_agents[0:3] are seats 1-3 and are stepped internally
  inside env.step()/env.reset() via their .bid(obs)/.play(obs)/.reset()
  interface, where `obs` there is the *dict* form from Round.get_observation(),
  not the flat encoded array the env returns to the caller.

  Because of this, "play as one of the 4 seats" (from the original docstring)
  isn't quite supported by the env as written -- only seat 0 is addressable
  from outside. --human therefore always puts the human in seat 0. If you want
  a human to sit in seats 1-3 instead, the env itself would need to expose
  those seats, since right now they're hard-wired to opponent_agents.

  I'm assuming heuristic agents' methods return:
      agent.bid(obs_dict)  -> int, the actual bid value (1-13)
      agent.play(obs_dict) -> int, a card index (0-51)
  matching how CallBreakEnv._run_opponents_until_agent_turn() calls them.

  For a trained checkpoint, I'm assuming ActorCritic exposes separate bid/play
  forward passes over the flat encoded observation vector; see TrainedAgent
  below for the exact shape assumptions -- flag if these don't match.
"""

import argparse
import time
import sys

try:
    import yaml
except ImportError:
    yaml = None

import numpy as np
import torch

from environment.round import Round
from environment.card import Card
from environment.callbreak_env import CallBreakEnv

from agents.random_agent import RandomAgent
from agents.basic_bet_agent import BasicBetAgent
from agents.safe_bet_agent import SafeBetAgent
from agents.safe_play_agent import SafePlayAgent
from models.actor_critic import ActorCritic


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--human",      action="store_true")
    parser.add_argument("--config",     type=str, default="config/config.yaml")
    parser.add_argument("--rounds",     type=int, default=5)
    parser.add_argument("--delay",      type=float, default=0.5,
                         help="Seconds between moves in watch mode")
    return parser.parse_args()


def load_config(path):
    if yaml is None:
        return {}
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}


class TrainedAgent:
    """Wraps an ActorCritic checkpoint so it can act as seat 0."""

    def __init__(self, checkpoint_path, config):
        encoder_type = config.get("model", {}).get("encoder_type", "mlp")
        self.model = ActorCritic(encoder_type=encoder_type)
        state = torch.load(checkpoint_path, map_location="cpu")
        state_dict = state.get("model_state_dict", state) if isinstance(state, dict) else state
        self.model.load_state_dict(state_dict)
        self.model.eval()

    @torch.no_grad()
    def act(self, obs_array, info):
        """obs_array: flat encoded observation from env.reset()/step().
        info: the info dict from the same env call, used for phase + legal mask.
        Returns an action index in the env's Discrete(52) action space
        (0-12 meaningful during bidding, 0-51 during play)."""
        obs_tensor = torch.as_tensor(obs_array, dtype=torch.float32).unsqueeze(0)
        legal = info["legal_actions"]

        if info["phase"] == "bid":
            logits, _ = self.model.bid_forward(obs_tensor)   # expected shape (1, 13)
            full_logits = torch.full((1, 52), -1e9)
            full_logits[0, :13] = logits[0]
        else:
            full_logits, _ = self.model.play_forward(obs_tensor)  # expected shape (1, 52)

        mask = torch.full((1, 52), -1e9)
        mask[0, legal] = 0.0
        masked = full_logits + mask
        return int(torch.argmax(masked, dim=-1).item())


def build_opponents(config):
    """Seats 1, 2, 3 -- always heuristic, per CallBreakEnv's fixed-opponent design."""
    return [BasicBetAgent(), SafeBetAgent(), SafePlayAgent()]


def build_seat0_agent(args, config):
    """Returns an object exposing .act(obs_array, info) for seat 0,
    or None if the human should be prompted instead."""
    if args.human:
        return None
    if args.checkpoint:
        return TrainedAgent(args.checkpoint, config)
    return RandomAgent()  # default: seat 0 is a 4th heuristic, matching original "watch 4 agents" mode


class HeuristicSeat0Wrapper:
    """Adapts a heuristic agent (which expects dict obs) to act(obs_array, info),
    by pulling the dict-form observation directly off the env's internal Round
    rather than the flat encoded array."""

    def __init__(self, heuristic, env):
        self.heuristic = heuristic
        self.env = env

    def act(self, obs_array, info):
        dict_obs = self.env._round.get_observation(0)
        if info["phase"] == "bid":
            bid_value = self.heuristic.bid(dict_obs)
            return bid_value - 1  # convert raw bid (1-13) to action index (0-12)
        return self.heuristic.play(dict_obs)  # already a card index


def format_card_index(idx):
    return str(Card.from_index(idx))


def prompt_human_bid(env, info):
    dict_obs = env._round.get_observation(0)
    hand = sorted(dict_obs["hand"], key=lambda c: (c.suit, c.rank))
    print("Your hand:", " ".join(str(c) for c in hand))
    legal_bids = [a + 1 for a in info["legal_actions"]]
    while True:
        raw = input(f"Enter your bid {legal_bids}: ")
        try:
            bid = int(raw)
            if bid - 1 in info["legal_actions"]:
                return bid - 1
        except ValueError:
            pass
        print("Invalid bid, try again.")


def prompt_human_play(env, info):
    legal = info["legal_actions"]
    dict_obs = env._round.get_observation(0)
    trick = dict_obs.get("current_trick_plays", [])
    if trick:
        print("Current trick:", "  ".join(f"P{p}:{c}" for p, c in trick))
    print("Legal plays:")
    for idx in legal:
        print(f"  [{idx}] {format_card_index(idx)}")
    while True:
        raw = input("Choose a card index to play: ")
        try:
            choice = int(raw)
            if choice in legal:
                return choice
        except ValueError:
            pass
        print("Invalid choice, pick one of the listed indices.")


def main():
    args = parse_args()
    config = load_config(args.config)
    watch_mode = not args.human

    opponents = build_opponents(config)
    env = CallBreakEnv(
        opponent_agents=opponents,
        num_rounds=args.rounds,
        render_mode="human",
    )

    seat0_agent = build_seat0_agent(args, config)
    if seat0_agent is not None and not isinstance(seat0_agent, TrainedAgent):
        seat0_agent = HeuristicSeat0Wrapper(seat0_agent, env)

    obs, info = env.reset()
    terminated = truncated = False

    try:
        while not terminated:
            env.render()

            if seat0_agent is None:
                if info["phase"] == "bid":
                    action = prompt_human_bid(env, info)
                else:
                    action = prompt_human_play(env, info)
            else:
                action = seat0_agent.act(obs, info)

            obs, reward, terminated, truncated, info = env.step(action)

            if watch_mode:
                time.sleep(args.delay)
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(0)

    env.render()
    print("\nFinal totals:", info["total_scores"])


if __name__ == "__main__":
    main()
    
