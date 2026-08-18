"""
logger.py
---------
Structured logging with console output and optional TensorBoard.
"""

import os
import time
import json
import logging
from typing import Any, Dict, Optional


class Logger:
    def __init__(
        self,
        log_dir: str,
        experiment_name: str = "callbreak_rl",
        use_tensorboard: bool = True,
        console_level: int = logging.INFO,
    ):
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        self.use_tensorboard = use_tensorboard
        self._writer = None
        self._step: int = 0

        os.makedirs(log_dir, exist_ok=True)

        # Console + file logger
        self._logger = logging.getLogger(experiment_name)
        self._logger.setLevel(console_level)
        self._setup_handlers()

        # TensorBoard
        if use_tensorboard:
            self._setup_tensorboard()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_handlers(self) -> None:
        """Attach console and file handlers to self._logger."""
        # Guard against duplicate handlers if a Logger with the same
        # experiment_name is constructed more than once in a process
        # (e.g. across curriculum stages, or in a notebook re-run).
        if self._logger.handlers:
            return

        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self._logger.addHandler(console_handler)

        log_file = os.path.join(self.log_dir, f"{self.experiment_name}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        self._logger.addHandler(file_handler)

        # Don't also propagate to the root logger's handlers (avoids
        # duplicate lines if the root logger also has a StreamHandler).
        self._logger.propagate = False

    def _setup_tensorboard(self) -> None:
        """Create a TensorBoard SummaryWriter; store in self._writer."""
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            self._logger.warning(
                "TensorBoard requested but torch.utils.tensorboard is "
                "unavailable (tensorboard package not installed?) -- "
                "continuing without it."
            )
            self._writer = None
            return

        tb_dir = os.path.join(self.log_dir, "tensorboard")
        os.makedirs(tb_dir, exist_ok=True)
        self._writer = SummaryWriter(log_dir=tb_dir)

    # ------------------------------------------------------------------
    # Logging interface
    # ------------------------------------------------------------------

    def log_scalar(self, tag: str, value: float, step: Optional[int] = None) -> None:
        """Log a single scalar value (to TensorBoard + console)."""
        effective_step = step if step is not None else self._step
        if step is not None:
            self._step = step

        if self._writer is not None:
            self._writer.add_scalar(tag, value, effective_step)

        self._logger.debug(f"{tag} = {value:.6f} (step {effective_step})")

    def log_scalars(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log multiple scalar metrics at once."""
        for tag, value in metrics.items():
            self.log_scalar(tag, value, step)

    def log_training_step(self, step: int, losses: Dict[str, float], fps: float) -> None:
        """Log training losses and throughput."""
        self.log_scalars({f"train/{k}": v for k, v in losses.items()}, step)
        self.log_scalar("train/fps", fps, step)

        loss_summary = ", ".join(f"{k}={v:.4f}" for k, v in losses.items())
        self.info(f"step {step}: {loss_summary}, fps={fps:.1f}")

    def log_evaluation(self, step: int, eval_metrics: Dict) -> None:
        """Log evaluation results (win rates, Elo, etc.)."""
        scalar_metrics = {
            f"eval/{k}": v for k, v in eval_metrics.items() if isinstance(v, (int, float))
        }
        self.log_scalars(scalar_metrics, step)

        summary = ", ".join(f"{k}={v}" for k, v in eval_metrics.items())
        self.info(f"[eval @ step {step}] {summary}")

    def log_curriculum_advance(self, old_stage: str, new_stage: str, step: int) -> None:
        """Log a curriculum stage transition."""
        self.info(f"[curriculum] step {step}: advanced from '{old_stage}' to '{new_stage}'")

    def info(self, msg: str) -> None:
        self._logger.info(msg)

    def warning(self, msg: str) -> None:
        self._logger.warning(msg)

    def debug(self, msg: str) -> None:
        self._logger.debug(msg)

    # ------------------------------------------------------------------
    # JSON result export
    # ------------------------------------------------------------------

    def save_results(self, results: Dict, filename: str) -> None:
        """Write a results dict to a JSON file in log_dir."""
        path = os.path.join(self.log_dir, filename)
        with open(path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        self.info(f"Saved results to {path}")

    def close(self) -> None:
        if self._writer:
            self._writer.close()