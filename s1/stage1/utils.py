"""
General utilities.
"""

from __future__ import annotations

import json
import os
import random
from typing import Any, Dict

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True):
    """
    统一设置所有随机种子，确保可复现性

    Args:
        seed: 随机种子值
        deterministic: 是否使用确定性算法（会降低性能但确保可复现）
    """
    # 环境变量
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.use_deterministic_algorithms(True)

    print(f"[INFO] 随机种子已设置: {seed}")

def worker_init_fn(worker_id: int):
    """
    DataLoader worker 的初始化函数
    确保每个 worker 的随机状态是可确定的
    """
    # 基于 worker_id 和基础种子生成 worker 特定的种子
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(obj: Dict[str, Any], path: str) -> None:
    """
    Save dictionary as UTF-8 JSON.
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
