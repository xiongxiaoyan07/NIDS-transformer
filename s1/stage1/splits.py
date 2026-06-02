# splits.py
"""
Train/validation/test split logic.

This module supports:
1. Original random / stratified train-val-test split.
2. Original random / stratified train-val split for external test.
3. Chronological train-val-test split.
4. Chronological train-val split for external test.

Important:
- Split is always flow-level, not packet-level.
- This prevents packets from the same flow appearing in both train and test,
  which is consistent with your original implementation.
  注意一点：Chronological split 和严格 stratify 本质上不能同时完全保证。随机 stratify 会打乱时间顺序，
  而严格时间划分可能导致 label=1 集中在某个时间段。所以上面代码采用的是更合理的折中方案：
  严格保持 train → val → test 的时间连续性，只在边界附近小范围移动切分点，让 label=1 的比例尽量接近。
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# =============================================================================
# 原始方法 1：随机 stratified train/val/test split
# 保留不动，方便做 baseline / ablation
# =============================================================================
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
    Original random stratified split.
    Split flows into train/val/test.

    This is the method in your current Stage1 code:
    - flow-level split
    - optionally stratified by label
    - random split, not chronological
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

    print("[INFO] splits.py ------ stratified_train_val_test_split")

    return splits


# =============================================================================
# 原始方法 2：外部 test 时，随机 stratified train/val split
# 保留不动，方便和 Chronological external-test 版本对比
# =============================================================================
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
    Original random train/val split when external test files are used.

    In this mode:
    - local training CSVs are split into train/val
    - external CSVs are used as final test
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


# =============================================================================
# 新方法 1：Chronological train/val/test split
# =============================================================================
def chronological_train_val_test_split(
    flows: pd.DataFrame,
    flow_id_col: str,
    label_col: str,
    time_col: str,
    train_size: float,
    val_size: float,
    test_size: float,
    seed: int = 42,
    stratify: bool = True,
    boundary_tolerance: float = 0.05,
) -> Dict[str, List[int]]:
    """
    Chronological train/val/test split.

    Recommended for continuous PCAP or real internal traffic:
        train: earliest 70%
        val:   middle 10%
        test:  latest 20%

    Key design:
    1. Strictly preserves chronological order.
       The split is always contiguous in time:
           train block -> val block -> test block

    2. Flow-level split.
       The same flow_id never appears in more than one split.

    3. Best-effort label balancing.
       Pure chronological split and perfect stratification are mathematically
       conflicting objectives. Random stratification can perfectly mix labels,
       but it destroys time order. Therefore, this function keeps chronological
       order first, and only searches nearby split boundaries to make label=1
       ratios closer across train/val/test.

    Args:
        flows:
            Flow-level dataframe.

        flow_id_col:
            Flow ID column.

        label_col:
            Label column. Usually binary label, where label=1 is malicious.

        time_col:
            Flow timestamp column.
            Recommended values:
                - flow start timestamp
                - first packet timestamp
            It must represent the chronological order of flows.

        train_size, val_size, test_size:
            Split ratios. Usually 0.70 / 0.10 / 0.20.

        seed:
            Kept for API compatibility.
            Chronological split itself does not shuffle data.

        stratify:
            If True, use boundary search to improve label distribution while
            preserving chronological order.

        boundary_tolerance:
            How much the split boundary can move around the nominal ratio.
            Example:
                boundary_tolerance=0.05 means each boundary can move within
                +/- 5% of total number of flows.
            Larger value gives better label balance but less exact split ratio.

    Returns:
        {
            "train": [flow_ids...],
            "val":   [flow_ids...],
            "test":  [flow_ids...],
        }
    """
    if abs(train_size + val_size + test_size - 1.0) > 1e-6:
        raise ValueError("train_size + val_size + test_size must be 1.")

    # 1. 只保留 flow_id / label / time 三列，并按 flow_id 去重。
    #    这与你原代码的 flow-level split 逻辑一致，避免同一个 flow 的 packet
    #    出现在多个 split 中。
    required_cols = [flow_id_col, label_col, time_col]
    _ensure_columns(flows, required_cols, "flows dataframe")

    df = flows[required_cols].drop_duplicates(flow_id_col).copy()

    # 2. 处理时间列和 label 列。
    #    如果 time_col 有 NaN，会被排到最后，这通常不理想，所以这里直接报错。
    if df[time_col].isna().any():
        missing_count = int(df[time_col].isna().sum())
        raise ValueError(
            f"time_col={time_col!r} contains {missing_count} missing values. "
            "Chronological split requires valid timestamps."
        )

    df[label_col] = pd.to_numeric(
        df[label_col],
        errors="coerce",
    ).fillna(0).astype(int)

    # 3. 核心：按时间排序。
    #    mergesort 是稳定排序；如果多个 flow 时间戳相同，会保持原相对顺序。
    df = df.sort_values(time_col, kind="mergesort").reset_index(drop=True)

    n = len(df)
    if n < 3:
        raise ValueError(
            f"Need at least 3 flows for train/val/test split, got {n}."
        )

    # 4. 计算理论边界。
    #    b1 是 train 结束位置；
    #    b2 是 val 结束位置；
    #    test 从 b2 到最后。
    nominal_b1 = int(round(n * train_size))
    nominal_b2 = int(round(n * (train_size + val_size)))
    print("[INFO] splits.py ------ chronological_train_val_test_split-----------nominal_b1=", nominal_b1, "nominal_b2=",nominal_b2)

    # 5. 如果 stratify=True，则在理论边界附近搜索更好的边界。
    #    注意：这里不是随机 stratify，而是 chronological-aware boundary search。
    #    它不会打乱时间顺序，只移动 train/val/test 的边界。
    if stratify:
        # b1, b2 = _choose_chronological_boundaries_with_label_balance(
        #     labels=df[label_col].to_numpy(dtype=int),
        #     nominal_b1=nominal_b1,
        #     nominal_b2=nominal_b2,
        #     train_size=train_size,
        #     val_size=val_size,
        #     test_size=test_size,
        #     boundary_tolerance=boundary_tolerance,
        # )
        b1, b2 = _choose_chronological_boundaries_with_label_balance_fast_stride(
            labels=df[label_col].to_numpy(dtype=int),
            nominal_b1=nominal_b1,
            nominal_b2=nominal_b2,
            train_size=train_size,
            val_size=val_size,
            test_size=test_size,
            boundary_tolerance=boundary_tolerance,
            chunk_size=512,
            search_stride=5,
        )
    else:
        b1, b2 = nominal_b1, nominal_b2

    print("[INFO] splits.py ------ chronological_train_val_test_split-----------b1=", b1, "b2=",b2)
    # 6. 边界安全修正，保证每个 split 至少有一个样本。
    b1, b2 = _sanitize_three_way_boundaries(n=n, b1=b1, b2=b2)

    train_df = df.iloc[:b1]
    print("[INFO] splits.py ------ chronological_train_val_test_split-----------train_df \n", train_df.head(3))
    val_df = df.iloc[b1:b2]
    print("[INFO] splits.py ------ chronological_train_val_test_split-----------val_df \n", val_df.head(3))
    test_df = df.iloc[b2:]
    print("[INFO] splits.py ------ chronological_train_val_test_split-----------test_df \n", test_df.head(3))

    splits = {
        "train": [int(x) for x in train_df[flow_id_col].tolist()],
        "val": [int(x) for x in val_df[flow_id_col].tolist()],
        "test": [int(x) for x in test_df[flow_id_col].tolist()],
    }

    _print_split_report(
        name="chronological_train_val_test_split",
        df=df,
        splits=splits,
        flow_id_col=flow_id_col,
        label_col=label_col,
        time_col=time_col,
    )

    # 7. 与原代码一致：如果全局存在 label=1，但 test 没有 label=1，做一次提醒。
    #    这里不自动从 train/val 移到 test，因为那会破坏严格时间顺序。
    _warn_if_positive_missing_in_any_split(
        splits=splits,
        df=df,
        flow_id_col=flow_id_col,
        label_col=label_col,
    )

    print("[INFO] splits.py ------ chronological_train_val_test_split")

    return splits


# =============================================================================
# 新方法 2：Chronological train/val split for external test
# =============================================================================
def chronological_train_val_split_for_external_test(
    flows: pd.DataFrame,
    flow_id_col: str,
    label_col: str,
    time_col: str,
    train_size: float,
    val_size: float,
    seed: int = 42,
    stratify: bool = True,
    boundary_tolerance: float = 0.05,
) -> Dict[str, List[int]]:
    """
    Chronological train/val split when external test files are used.

    Use case:
        - packet_csv / flow_csv:
            used only for train and val
        - external_packet_csv / external_flow_csv:
            used only for final test

    Split logic:
        train: earliest part of the training CSVs
        val:   latest part of the training CSVs

    Example:
        If train_size=0.70 and val_size=0.10 in your original global config,
        then inside local training files:
            train_ratio = 0.70 / (0.70 + 0.10) = 87.5%
            val_ratio   = 0.10 / (0.70 + 0.10) = 12.5%

    Important:
        This function only returns train/val.
        The external test IDs should still be assigned in pipeline.py from
        external_flow_csv.

    Args:
        flows:
            Flow dataframe from the training CSVs.

        flow_id_col:
            Flow ID column.

        label_col:
            Label column.

        time_col:
            Timestamp column used for chronological ordering.

        train_size, val_size:
            Original train/val sizes from config.

        seed:
            Kept for API compatibility.

        stratify:
            If True, searches near the nominal train/val boundary to make
            label=1 ratio in train and val closer while preserving time order.

        boundary_tolerance:
            Boundary search range around nominal split point.

    Returns:
        {
            "train": [flow_ids...],
            "val":   [flow_ids...],
        }
    """
    if train_size <= 0 or val_size <= 0:
        raise ValueError("train_size and val_size must be positive.")

    total = train_size + val_size
    train_ratio = train_size / total
    val_ratio = val_size / total

    required_cols = [flow_id_col, label_col, time_col]
    _ensure_columns(flows, required_cols, "flows dataframe")

    df = flows[required_cols].drop_duplicates(flow_id_col).copy()

    if df[time_col].isna().any():
        missing_count = int(df[time_col].isna().sum())
        raise ValueError(
            f"time_col={time_col!r} contains {missing_count} missing values. "
            "Chronological split requires valid timestamps."
        )

    df[label_col] = pd.to_numeric(
        df[label_col],
        errors="coerce",
    ).fillna(0).astype(int)

    # 按时间排序，保证 train 是过去，val 是更靠后的未来。
    df = df.sort_values(time_col, kind="mergesort").reset_index(drop=True)

    n = len(df)
    if n < 2:
        raise ValueError(
            f"Need at least 2 flows for train/val split, got {n}."
        )

    nominal_b = int(round(n * train_ratio))

    if stratify:
        b = _choose_chronological_boundary_with_label_balance(
            labels=df[label_col].to_numpy(dtype=int),
            nominal_b=nominal_b,
            left_ratio=train_ratio,
            right_ratio=val_ratio,
            boundary_tolerance=boundary_tolerance,
        )
    else:
        b = nominal_b

    b = _sanitize_two_way_boundary(n=n, b=b)

    train_df = df.iloc[:b]
    val_df = df.iloc[b:]

    splits = {
        "train": [int(x) for x in train_df[flow_id_col].tolist()],
        "val": [int(x) for x in val_df[flow_id_col].tolist()],
    }

    _print_split_report(
        name="chronological_train_val_split_for_external_test",
        df=df,
        splits=splits,
        flow_id_col=flow_id_col,
        label_col=label_col,
        time_col=time_col,
    )

    _warn_if_positive_missing_in_any_split(
        splits=splits,
        df=df,
        flow_id_col=flow_id_col,
        label_col=label_col,
    )

    print("[INFO] splits.py ------ chronological_train_val_split_for_external_test")

    return splits


# =============================================================================
# Helper functions
# =============================================================================
def _choose_chronological_boundaries_with_label_balance_fast_stride(
    labels: np.ndarray,
    nominal_b1: int,
    nominal_b2: int,
    train_size: float,
    val_size: float,
    test_size: float,
    boundary_tolerance: float,
    chunk_size: int = 512,
    search_stride: int = 1,
) -> Tuple[int, int]:
    labels = np.asarray(labels, dtype=np.float64)
    if labels.ndim != 1:
        raise ValueError("labels must be a 1D array")

    n = len(labels)
    if n < 3:
        raise ValueError("Need at least 3 samples to create non-empty train/val/test splits")

    search_stride = max(1, int(search_stride))

    nominal_b1 = int(np.clip(nominal_b1, 1, n - 2))
    nominal_b2 = int(np.clip(nominal_b2, nominal_b1 + 1, n - 1))

    total_pos = float(labels.sum())
    global_pos_rate = total_pos / n

    tol = max(1, int(round(n * boundary_tolerance)))

    b1_min = max(1, nominal_b1 - tol)
    b1_max = min(n - 2, nominal_b1 + tol)

    b2_min = max(2, nominal_b2 - tol)
    b2_max = min(n - 1, nominal_b2 + tol)

    b1_candidates = np.arange(b1_min, b1_max + 1, search_stride, dtype=np.int64)
    b2_candidates = np.arange(b2_min, b2_max + 1, search_stride, dtype=np.int64)

    # Make sure nominal boundaries are included.
    b1_candidates = np.unique(np.concatenate([b1_candidates, np.array([nominal_b1], dtype=np.int64)]))
    b2_candidates = np.unique(np.concatenate([b2_candidates, np.array([nominal_b2], dtype=np.int64)]))

    prefix_pos = np.empty(n + 1, dtype=np.float64)
    prefix_pos[0] = 0.0
    prefix_pos[1:] = np.cumsum(labels, dtype=np.float64)

    best_score = np.inf
    best_pair = (nominal_b1, nominal_b2)

    b2_all = b2_candidates[None, :]

    for start in range(0, len(b1_candidates), chunk_size):
        end = min(start + chunk_size, len(b1_candidates))

        b1 = b1_candidates[start:end, None]
        b2 = b2_all

        valid = b2 > b1

        train_len = b1.astype(np.float64)
        val_len = (b2 - b1).astype(np.float64)
        test_len = (n - b2).astype(np.float64)

        train_pos = prefix_pos[b1]
        val_pos = prefix_pos[b2] - prefix_pos[b1]
        test_pos = total_pos - prefix_pos[b2]

        train_rate = train_pos / train_len
        val_rate = val_pos / val_len
        test_rate = test_pos / test_len

        label_balance_score = (
            (train_rate - global_pos_rate) ** 2
            + (val_rate - global_pos_rate) ** 2
            + (test_rate - global_pos_rate) ** 2
        )

        size_score = (
            (train_len / n - train_size) ** 2
            + (val_len / n - val_size) ** 2
            + (test_len / n - test_size) ** 2
        )

        if total_pos > 0:
            missing_positive_penalty = (
                (train_pos == 0).astype(np.float64)
                + (val_pos == 0).astype(np.float64)
                + (test_pos == 0).astype(np.float64)
            )
        else:
            missing_positive_penalty = 0.0

        score = (
            10.0 * label_balance_score
            + size_score
            + 5.0 * missing_positive_penalty
        )

        score = np.where(valid, score, np.inf)

        local_flat_idx = int(np.argmin(score))
        local_score = float(score.ravel()[local_flat_idx])

        if local_score < best_score:
            local_i, local_j = np.unravel_index(local_flat_idx, score.shape)
            best_score = local_score
            best_pair = (
                int(b1_candidates[start + local_i]),
                int(b2_candidates[local_j]),
            )

    return best_pair


def _choose_chronological_boundary_with_label_balance(
    labels: np.ndarray,
    nominal_b: int,
    left_ratio: float,
    right_ratio: float,
    boundary_tolerance: float,
) -> int:
    """
    Two-way chronological boundary search.

    Used for external-test mode:
        [0:b]  -> train
        [b:n]  -> val

    Preserves chronological order.
    """
    print("***************splits.py ******** _choose_chronological_boundary_with_label_balance")
    print("***************splits.py ******** Two-way chronological boundary search. Used for external-test mode")
    n = len(labels)
    global_pos_rate = float(labels.mean()) if n > 0 else 0.0

    tol = max(1, int(round(n * boundary_tolerance)))

    b_min = max(1, nominal_b - tol)
    b_max = min(n - 1, nominal_b + tol)

    best_score = float("inf")
    best_b = nominal_b

    target_sizes = np.array([left_ratio, right_ratio], dtype=np.float64)

    for b in range(b_min, b_max + 1):
        left_labels = labels[:b]
        right_labels = labels[b:]

        if len(left_labels) == 0 or len(right_labels) == 0:
            continue

        pos_rates = np.array(
            [
                left_labels.mean(),
                right_labels.mean(),
            ],
            dtype=np.float64,
        )

        actual_sizes = np.array(
            [
                len(left_labels) / n,
                len(right_labels) / n,
            ],
            dtype=np.float64,
        )

        label_balance_score = float(
            np.sum((pos_rates - global_pos_rate) ** 2)
        )

        size_score = float(
            np.sum((actual_sizes - target_sizes) ** 2)
        )

        missing_positive_penalty = 0.0
        if labels.sum() > 0:
            if left_labels.sum() == 0:
                missing_positive_penalty += 1.0
            if right_labels.sum() == 0:
                missing_positive_penalty += 1.0

        score = (
            10.0 * label_balance_score
            + 1.0 * size_score
            + 5.0 * missing_positive_penalty
        )

        if score < best_score:
            best_score = score
            best_b = b

    return best_b


def _sanitize_three_way_boundaries(
    n: int,
    b1: int,
    b2: int,
) -> Tuple[int, int]:
    """
    Ensure:
        0 < b1 < b2 < n
    """
    print("********** splits.py *****************_sanitize_three_way_boundaries: 边界安全修正，保证每个 split 至少有一个样本")
    b1 = int(b1)
    b2 = int(b2)

    b1 = max(1, min(b1, n - 2))
    b2 = max(b1 + 1, min(b2, n - 1))

    return b1, b2


def _sanitize_two_way_boundary(
    n: int,
    b: int,
) -> int:
    """
    Ensure:
        0 < b < n
    """
    b = int(b)
    b = max(1, min(b, n - 1))
    return b


def _print_split_report(
    name: str,
    df: pd.DataFrame,
    splits: Dict[str, List[int]],
    flow_id_col: str,
    label_col: str,
    time_col: str,
) -> None:
    """
    Print split statistics:
        - number of flows
        - label counts
        - label=1 ratio
        - time range

    This is very useful for verifying chronological split.
    """
    print("\n" + "=" * 80)
    print(f"[INFO] Split report: {name}")
    print("=" * 80)

    id_to_row = df.set_index(flow_id_col)

    for split_name, ids in splits.items():
        sub = id_to_row.loc[ids].reset_index()

        n = len(sub)
        label_counts = sub[label_col].astype(int).value_counts().sort_index().to_dict()
        pos_count = int(label_counts.get(1, 0))
        pos_rate = pos_count / n if n > 0 else 0.0

        t_min = sub[time_col].min() if n > 0 else None
        t_max = sub[time_col].max() if n > 0 else None

        print(f"[INFO] {split_name}:")
        print(f"       num_flows     = {n}")
        print(f"       label_counts  = {label_counts}")
        print(f"       label1_ratio  = {pos_rate:.6f}")
        print(f"       time_range    = [{t_min}, {t_max}]")

    print("=" * 80 + "\n")


def _warn_if_positive_missing_in_any_split(
    splits: Dict[str, List[int]],
    df: pd.DataFrame,
    flow_id_col: str,
    label_col: str,
) -> None:
    """
    Warn if label=1 exists globally but is missing in any split.

    For chronological split, we should NOT automatically move a positive sample
    from train/val to test, because that may break chronological order.

    Your original random split had _ensure_test_has_positive_if_possible(),
    which can move one positive into test. That is acceptable for random split,
    but not ideal for strict chronological split.
    """
    id_to_label = dict(
        zip(
            df[flow_id_col].tolist(),
            df[label_col].astype(int).tolist(),
        )
    )

    global_has_positive = any(v == 1 for v in id_to_label.values())
    if not global_has_positive:
        return

    for split_name, ids in splits.items():
        has_positive = any(id_to_label.get(fid, 0) == 1 for fid in ids)
        if not has_positive:
            print(
                f"[WARNING] {split_name} has no label=1 samples. "
                "This may happen with strict chronological split when positives "
                "are temporally clustered."
            )


def _ensure_columns(
    df: pd.DataFrame,
    required: List[str],
    name: str,
) -> None:
    """
    Check required columns.
    """
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def _can_stratify(y: np.ndarray) -> bool:
    """
    Original helper:
    Stratify is possible only when:
        - at least 2 classes exist
        - every class has at least 2 samples
    """
    if len(np.unique(y)) < 2:
        return False
    counts = pd.Series(y).value_counts()
    return int(counts.min()) >= 2


def _random_split(ids, train_size, val_size, test_size, seed):
    """
    Original random split fallback.
    """
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
    Original helper for random split.

    If label=1 exists but test has none, move one positive from train/val into test.

    Note:
        This is kept only for the original random/stratified split.
        Do not use this helper in strict chronological split, because moving
        samples across splits may break time order.
    """
    id_to_label = dict(zip(df[flow_id_col].tolist(), df[label_col].astype(int).tolist()))

    all_have_positive = any(v == 1 for v in id_to_label.values())
    if not all_have_positive:
        return

    test_has_positive = any(id_to_label.get(fid, 0) == 1 for fid in splits["test"])
    if test_has_positive:
        return

    print("moving one positive from train/val into test")
    for src in ["train", "val"]:
        for fid in list(splits[src]):
            if id_to_label.get(fid, 0) == 1:
                splits[src].remove(fid)
                splits["test"].append(fid)
                return
