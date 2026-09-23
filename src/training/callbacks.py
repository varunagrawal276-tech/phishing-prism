"""
Training callbacks for PRISM-Phish.

Early stopping, logging, and metric tracking.
"""

import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    Early stopping based on a monitored metric.

    Stops training when the metric hasn't improved for `patience` evaluations.
    """

    def __init__(
        self,
        patience: int = 3,
        min_delta: float = 0.001,
        mode: str = "max",
        metric_name: str = "val_f1_macro",
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.metric_name = metric_name
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, metric_value: float) -> bool:
        """
        Check if training should stop.

        Args:
            metric_value: Current value of the monitored metric.

        Returns:
            True if training should stop.
        """
        if self.best_score is None:
            self.best_score = metric_value
            return False

        if self.mode == "max":
            improved = metric_value > self.best_score + self.min_delta
        else:
            improved = metric_value < self.best_score - self.min_delta

        if improved:
            self.best_score = metric_value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(
                    f"Early stopping triggered: {self.metric_name} hasn't improved "
                    f"for {self.patience} evaluations. Best: {self.best_score:.4f}"
                )
                return True

        return False


class TrainingLogger:
    """Logs training progress."""

    def __init__(self, log_every_n_steps: int = 50):
        self.log_every_n_steps = log_every_n_steps
        self.step = 0
        self.epoch_start_time = None

    def on_epoch_start(self, epoch: int):
        self.epoch_start_time = time.time()
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch + 1}")
        logger.info(f"{'='*60}")

    def on_epoch_end(self, epoch: int, metrics: dict):
        duration = time.time() - self.epoch_start_time
        logger.info(f"Epoch {epoch + 1} completed in {duration:.1f}s")
        for key, value in sorted(metrics.items()):
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.4f}")

    def on_step(self, step: int, loss: float, lr: float = None):
        self.step = step
        if step % self.log_every_n_steps == 0:
            msg = f"  Step {step}: loss={loss:.4f}"
            if lr is not None:
                msg += f", lr={lr:.2e}"
            logger.info(msg)
