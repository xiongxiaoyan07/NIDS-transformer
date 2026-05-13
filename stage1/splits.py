"""
Train/validation/test split logic.

This module supports:
1. Normal train/val/test split from one dataset.
2. External final test files:
   - Training CSVs are used only for train/val.
   - External CSVs are used only for final test.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def stratified_train_val_test_split(
    flows: pd.DataFrame,
    flow_id_col: str,
    label_col: str,
    train_size: float,
    val_size: float,
    test_size: float,
    seed: int,
    stratify: bool = True,
) -> Dict[str, List[int]]:
    """
    Split flows into train/val/test.

    The split is flow-level, not packet-level.
    This prevents packets from the same flow appearing in both train and test.

    If label=1 is rare, the function tries to ensure test contains at least one positive sample.
    """
    if abs(train_size + val_size + test_size - 1.0) > 1e-6:
        raise ValueError("train_size + val_size + test_size must be 1.")

    df = flows[[flow_id_col, label_col]].drop_duplicates(flow_id_col).copy()
    ids = df[flow_id_col].to_numpy()
    y = df[label_col].astype(int).to_numpy()

    can_stratify = stratify and _can_stratify(y)

    if can_stratify:
        train_val_ids, test_ids, train_val_y, _ = train_test_split(
            ids,
            y,
            test_size=test_size,
            random_state=seed,
            stratify=y,
        )

        relative_val_size = val_size / (train_size + val_size)

        if _can_stratify(train_val_y):
            train_ids, val_ids, _, _ = train_test_split(
                train_val_ids,
                train_val_y,
                test_size=relative_val_size,
                random_state=seed,
                stratify=train_val_y,
            )
        else:
            train_ids, val_ids = train_test_split(
                train_val_ids,
                test_size=relative_val_size,
                random_state=seed,
                shuffle=True,
            )
    else:
        train_ids, val_ids, test_ids = _random_split(
            ids,
            train_size=train_size,
            val_size=val_size,
            test_size=test_size,
            seed=seed,
        )

    splits = {
        "train": [int(x) for x in train_ids],
        "val": [int(x) for x in val_ids],
        "test": [int(x) for x in test_ids],
    }

    _ensure_test_has_positive_if_possible(splits, df, flow_id_col, label_col)
    return splits


def train_val_split_for_external_test(
    flows: pd.DataFrame,
    flow_id_col: str,
    label_col: str,
    train_size: float,
    val_size: float,
    seed: int,
    stratify: bool = True,
) -> Dict[str, List[int]]:
    """
    Split only train/val when external test files are used.
    """
    total = train_size + val_size
    train_ratio = train_size / total
    val_ratio = val_size / total

    df = flows[[flow_id_col, label_col]].drop_duplicates(flow_id_col).copy()
    ids = df[flow_id_col].to_numpy()
    y = df[label_col].astype(int).to_numpy()

    if stratify and _can_stratify(y):
        train_ids, val_ids, _, _ = train_test_split(
            ids,
            y,
            test_size=val_ratio,
            random_state=seed,
            stratify=y,
        )
    else:
        train_ids, val_ids = train_test_split(
            ids,
            test_size=val_ratio,
            random_state=seed,
            shuffle=True,
        )

    return {
        "train": [int(x) for x in train_ids],
        "val": [int(x) for x in val_ids],
    }


def _can_stratify(y: np.ndarray) -> bool:
    if len(np.unique(y)) < 2:
        return False
    counts = pd.Series(y).value_counts()
    return int(counts.min()) >= 2


def _random_split(ids, train_size, val_size, test_size, seed):
    rng = np.random.default_rng(seed)
    ids = np.array(ids)
    rng.shuffle(ids)

    n = len(ids)
    n_test = max(1, int(round(n * test_size)))
    n_val = max(1, int(round(n * val_size)))

    test_ids = ids[:n_test]
    val_ids = ids[n_test:n_test + n_val]
    train_ids = ids[n_test + n_val:]
    return train_ids, val_ids, test_ids


def _ensure_test_has_positive_if_possible(
    splits: Dict[str, List[int]],
    df: pd.DataFrame,
    flow_id_col: str,
    label_col: str,
) -> None:
    """
    If label=1 exists but test has none, move one positive from train/val into test.
    """
    id_to_label = dict(zip(df[flow_id_col].tolist(), df[label_col].astype(int).tolist()))

    all_have_positive = any(v == 1 for v in id_to_label.values())
    if not all_have_positive:
        return

    test_has_positive = any(id_to_label.get(fid, 0) == 1 for fid in splits["test"])
    if test_has_positive:
        return

    for src in ["train", "val"]:
        for fid in list(splits[src]):
            if id_to_label.get(fid, 0) == 1:
                splits[src].remove(fid)
                splits["test"].append(fid)
                return
