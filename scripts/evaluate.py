"""
evaluate.py
-----------
Evaluate a trained agent against all heuristic baselines.

Usage:
  python scripts/evaluate.py --checkpoint checkpoints/best.pt
  python scripts/evaluate.py --checkpoint checkpoints/best.pt --games 500 --vs random safe_play
"""

import argparse
import yaml
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Call Break RL agent")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config",     type=str, default="config/config.yaml")
    parser.add_argument("--games",      type=int, default=200)
    parser.add_argument("--vs",         nargs="+",
                        default=["random", "safe_bet", "safe_play", "basic_bet"],
                        help="Opponent types to evaluate against")
    parser.add_argument("--seed",       type=int, default=0)
    parser.add_argument("--device",     type=str, default=None)
    parser.add_argument("--output",     type=str, default=None,
                        help="Optional path to save results JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    from utils.state_encoder import OBS_DIM
    config["obs_dim"] = OBS_DIM

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # Load agent from checkpoint
    from agents.rl.ppo_agent import PPOAgent
    agent = PPOAgent.from_checkpoint(
        checkpoint_path=args.checkpoint,
        model_config={"obs_dim": config["obs_dim"], **config["model"]},
        player_id=0,
        device=device,
    )

    # Run evaluation
    from evaluation.evaluator import Evaluator
    config["evaluation"]["num_games"] = args.games
    config["evaluation"]["opponents"] = args.vs
    evaluator = Evaluator(config["evaluation"])
    results = evaluator.evaluate(agent, num_games=args.games)

    evaluator.print_summary(results)

    if args.output:
        import json
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
