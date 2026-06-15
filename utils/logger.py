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
        raise NotImplementedError

    def _setup_tensorboard(self) -> None:
        """Create a TensorBoard SummaryWriter; store in self._writer."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Logging interface
    # ------------------------------------------------------------------

    def log_scalar(self, tag: str, value: float, step: Optional[int] = None) -> None:
        """Log a single scalar value (to TensorBoard + console)."""
        raise NotImplementedError

    def log_scalars(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log multiple scalar metrics at once."""
        for tag, value in metrics.items():
            self.log_scalar(tag, value, step)

    def log_training_step(self, step: int, losses: Dict[str, float], fps: float) -> None:
        """Log training losses and throughput."""
        raise NotImplementedError

    def log_evaluation(self, step: int, eval_metrics: Dict) -> None:
        """Log evaluation results (win rates, Elo, etc.)."""
        raise NotImplementedError

    def log_curriculum_advance(self, old_stage: str, new_stage: str, step: int) -> None:
        """Log a curriculum stage transition."""
        raise NotImplementedError

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
        raise NotImplementedError

    def close(self) -> None:
        if self._writer:
            self._writer.close()
