"""
Seed utility functions for reproducibility across random, NumPy, and PyTorch.
If `seed` is None or 0, randomness is not fixed.
"""

import random
import numpy as np

def set_seed(seed: int | None):
    """
    Set random seeds across Python, NumPy, and (optionally) PyTorch.
    Pass None or 0 to allow random behavior (non-deterministic generation).
    """
    if seed is None or seed == 0:
        # Don't fix randomness
        try:
            import torch
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass
        return

    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
