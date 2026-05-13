"""
Configuration utilities for the Stage1 pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict
import yaml


def load_config(path: str) -> Dict[str, Any]:
    """
    Load a YAML config file.

    The config controls:
    - feature columns
    - split ratio
    - sequence length strategy
    - model hyperparameters
    - training hyperparameters
    """
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if cfg is None:
        cfg = {}

    return cfg


def get_nested(cfg: Dict[str, Any], path: str, default=None):
    """
    Small helper for nested config access.

    Example:
        get_nested(cfg, "features.flow.numerical", [])
    """
    cur = cfg
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur
