"""
play.py
-------
Interactive CLI to watch or play against the trained agent.

Usage:
  python scripts/play.py                          # watch 4 agents play
  python scripts/play.py --checkpoint best.pt     # watch trained agent vs heuristics
  python scripts/play.py --human                  # play as one of the 4 seats
"""

import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--human",      action="store_true")
    parser.add_argument("--config",     type=str, default="config/config.yaml")
    parser.add_argument("--rounds",     type=int, default=5)
    parser.add_argument("--delay",      type=float, default=0.5,
                        help="Seconds between moves in watch mode")
    return parser.parse_args()


def main():
    args = parse_args()
    # TODO: Build env, agents, run interactive game loop with render()
    raise NotImplementedError("Implement interactive play loop")


if __name__ == "__main__":
    main()
