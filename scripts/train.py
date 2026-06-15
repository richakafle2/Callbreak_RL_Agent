"""
train.py
--------
Main entry point for training the Call Break RL agent.

Usage:
  python scripts/train.py --config config/config.yaml
  python scripts/train.py --config config/config.yaml --stage random
  python scripts/train.py --config config/config.yaml --resume checkpoints/step_500000.pt
"""

import argparse
import os
import yaml
import torch
import numpy as np
import random


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Call Break RL agent")
    parser.add_argument("--config",  type=str, default="config/config.yaml")
    parser.add_argument("--stage",   type=str, default=None,
                        help="Override curriculum starting stage")
    parser.add_argument("--resume",  type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--seed",    type=int, default=None)
    parser.add_argument("--device",  type=str, default=None,
                        help="cuda | cpu (default: auto)")
    parser.add_argument("--debug",   action="store_true",
                        help="Use small hyperparameters for quick debugging")
    return parser.parse_args()


def load_config(path: str) -> dict:
    """Load YAML config and return as dict."""
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def apply_debug_overrides(config: dict) -> dict:
    """Shrink hyperparameters for quick smoke-test runs."""
    config["ppo"]["rollout_steps"] = 128
    config["training"]["num_envs"] = 4
    config["training"]["total_timesteps"] = 10_000
    config["training"]["eval_interval"] = 2_000
    config["ppo"]["num_epochs"] = 2
    return config


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    # Apply overrides
    if args.debug:
        config = apply_debug_overrides(config)
    if args.seed is not None:
        config["training"]["seed"] = args.seed
    set_seed(config["training"]["seed"])

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    # Add obs_dim to config (computed from StateEncoder)
    from utils.state_encoder import OBS_DIM
    config["obs_dim"] = OBS_DIM

    # Build trainer
    from training.trainer import PPOTrainer
    trainer = PPOTrainer(config)

    # Resume if requested
    if args.resume:
        print(f"Resuming from {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Override starting stage
    if args.stage:
        print(f"Starting from curriculum stage: {args.stage}")
        # TODO: advance curriculum to requested stage

    # Run training
    print(f"Starting training for {config['training']['total_timesteps']:,} steps")
    trainer.train(total_timesteps=config["training"]["total_timesteps"])
    print("Training complete.")


if __name__ == "__main__":
    main()
