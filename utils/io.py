"""Config loading, device selection, and small JSON/CSV helpers."""
import json
import os
from typing import Any, Dict

import yaml

# Root for all run artifacts. Override with PG_OUT_ROOT so that concurrent runs
# on different cohorts (e.g. the demo vs. TCGA-BRCA) write to separate trees and
# never clobber each other's checkpoints/tables. Defaults to "outputs".
OUT_ROOT = os.environ.get("PG_OUT_ROOT", "outputs")


def load_config(path: str) -> Dict[str, Any]:
    """Load a YAML config into a plain dict."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = os.path.abspath(path)
    return cfg


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def save_json(obj: Any, path: str):
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def load_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def _json_default(o):
    try:
        import numpy as np

        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:
        pass
    return str(o)


def get_device(preferred: str = "cuda"):
    """Return a torch.device, gracefully falling back to CPU."""
    import torch

    if preferred.startswith("cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get(cfg: Dict, dotted: str, default=None):
    """Nested lookup: get(cfg, 'model.target.ema', False)."""
    cur = cfg
    for key in dotted.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur
