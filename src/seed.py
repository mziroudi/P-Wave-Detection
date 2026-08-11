"""Global seeding for reproducible experiments."""

from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int = 42, deterministic_torch: bool = True) -> None:
    """Seed Python ``random``, NumPy, and PyTorch (if installed)."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
