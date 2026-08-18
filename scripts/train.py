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

# Curriculum order, per the 4-stage design: random -> mixed -> safe -> self-play.
# NOTE: I'm assuming these are the exact string keys PPOTrainer's curriculum
# uses internally. If trainer.py names them differently (e.g. "self_play" vs
# "selfplay"), update this list to match.
CURRICULUM_STAGES = ["random", "mixed", "safe", "self_play"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Call Break RL agent")
    parser.add_argument("--config",  type=str, default="config/config.yaml")
    parser.add_argument("--stage",   type=str, default=None,
                         choices=CURRICULUM_STAGES,
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


def advance_curriculum_to_stage(trainer, stage: str) -> None:
    """Force the trainer's curriculum to start at `stage` instead of stage 0.

    trainer.curriculum is a CurriculumScheduler. It tracks progress as a
    plain integer index (`_current_idx`) into `self.stages`, a list of
    CurriculumStage objects (each with a `.name`). There is no set_stage()
    method, so we look up the target stage's index by name and set
    `_current_idx` directly. train() re-derives opponents from the current
    stage via get_opponents() at the top of its loop, so this is safe to
    call any time before trainer.train() runs.
    """
    stage_names = [s.name for s in trainer.curriculum.stages]

    if stage not in stage_names:
        raise ValueError(
            f"Unknown curriculum stage '{stage}'. Expected one of {stage_names}."
        )

    trainer.curriculum._current_idx = stage_names.index(stage)


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
        advance_curriculum_to_stage(trainer, args.stage)

    # Run training
    print(f"Starting training for {config['training']['total_timesteps']:,} steps")
    trainer.train(total_timesteps=config["training"]["total_timesteps"])
    print("Training complete.")


if __name__ == "__main__":
    main()
