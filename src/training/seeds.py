"""
Seed management for reproducible experiments.
"""

import os
import random
from typing import Optional

import numpy as np


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Set random seeds for reproducibility across all libraries.

    Args:
        seed: Random seed value.
        deterministic: If True, enforce deterministic CUDA operations
                       (may reduce performance).
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            if deterministic:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                # PyTorch 2.x deterministic algorithms
                try:
                    torch.use_deterministic_algorithms(True, warn_only=True)
                except AttributeError:
                    pass
    except ImportError:
        pass


DEFAULT_SEEDS = [42, 52, 62]
EXTENDED_SEEDS = [42, 52, 62, 72, 82]
