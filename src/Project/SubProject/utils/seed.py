import os
import random
from typing import Optional


def set_seed(seed: int = 42, deterministic: bool = True, env_var: Optional[str] = None) -> int:
    """Set RNG seeds for Python, NumPy, and PyTorch (if available).

    - If `env_var` is provided and set, it overrides the `seed`.
    - When `deterministic` is True, enables PyTorch deterministic algorithms when possible.
    Returns the final seed used.
    """
    if env_var and (v := os.getenv(env_var)):
        try:
            seed = int(v)
        except ValueError:
            pass

    try:
        import numpy as np  # type: ignore
    except Exception:  # pragma: no cover - optional dependency
        np = None  # type: ignore

    try:
        import torch  # type: ignore
    except Exception:  # pragma: no cover - optional dependency
        torch = None  # type: ignore

    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass
            try:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            except Exception:
                pass

    return seed

