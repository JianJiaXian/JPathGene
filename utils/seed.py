"""Reproducible seeding across python / numpy / torch."""
import os
import random

import numpy as np


def set_seed(seed: int = 42, deterministic: bool = True):
    """Seed all RNGs. Returns the seed for convenience/logging."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:  # torch not available (e.g. pure figure scripts)
        pass
    return seed
