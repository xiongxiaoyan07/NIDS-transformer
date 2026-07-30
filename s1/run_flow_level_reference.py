#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Experiment A: flow-level reference baselines.

Train and evaluate the following models using ONLY flow CSV statistics:
    - Random Forest
    - XGBoost
    - LightGBM
    - Flow MLP

The script is intentionally independent from the Stage1 packet dataloader and
from features.flow_fusion.enabled. It never reads a packet CSV.

Example:
    python run_flow_level_reference.py \
        --flow_csv /path/to/stage1_flows.csv \
        --config configs/stage1_config.yaml \
        --out_dir results/experiment_A \
        --split_method chronological \
        --time_col flow_start_timestamp_us \
        --seed 130

Dependencies:
    pip install numpy pandas scikit-learn joblib pyyaml torch xgboost lightgbm
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyTorch is required for the Flow MLP baseline.") from exc


MODEL_ORDER = ["random_forest", "xgboost", "lightgbm", "flow_mlp"]
MODEL_DISPLAY_NAMES = {
    "random_forest": "RandomForest",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "flow_mlp": "FlowMLP",
}
# (flow_id,flow_start_timestamp_us,flow_end_timestamp_us,source_ip,source_port,destination_ip,destination_port,
#  protocol,flow_duration,total_fwd_packets,total_backward_packets,total_length_of_fwd_packets,total_length_of_bwd_packets,
#  fwd_packet_length_max,fwd_packet_length_min,fwd_packet_length_mean,fwd_packet_length_std,bwd_packet_length_max,bwd_packet_length_min,
#  bwd_packet_length_mean,bwd_packet_length_std,flow_bytes_per_s,flow_packets_per_s,flow_iat_mean,flow_iat_std,flow_iat_max,flow_iat_min,
#  fwd_iat_total,fwd_iat_mean,fwd_iat_std,fwd_iat_max,fwd_iat_min,bwd_iat_total,bwd_iat_mean,bwd_iat_std,bwd_iat_max,bwd_iat_min,fwd_psh_flags,
#  bwd_psh_flags,fwd_urg_flags,bwd_urg_flags,fwd_header_length,bwd_header_length,fwd_packets_per_s,bwd_packets_per_s,min_packet_length,
#  max_packet_length,packet_length_mean,packet_length_std,packet_length_variance,fin_flag_count,syn_flag_count,rst_flag_count,psh_flag_count,
#  ack_flag_count,urg_flag_count,cwe_flag_count,ece_flag_count,down_up_ratio,average_packet_size,avg_fwd_segment_size,avg_bwd_segment_size,
#  has_init_win_bytes_forward,init_win_bytes_forward,has_init_win_bytes_backward,init_win_bytes_backward,act_data_pkt_fwd,min_seg_size_forward,
#  active_mean,active_std,active_max,active_min,idle_mean,idle_std,idle_max,idle_min,label)
DEFAULT_EXCLUDE_COLUMNS = {
    "flow_id",
    "flow_start_timestamp_us",
    "flow_end_timestamp_us",
    "source_ip",
    "source_port",
    "destination_ip",
    "destination_id",
    "destination_port"
}


@dataclass
class SplitData:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


@dataclass
class MetricResult:
    threshold: float
    accuracy: float
    precision_class1: float
    recall_class1: float
    f1_class1: float
    macro_f1: float
    macro_recall: float
    macro_precision: float
    weighted_f1: float
    weighted_recall: float
    weighted_precision: float
    roc_auc: float
    pr_auc: float
    tpr: float
    fpr: float
    tn: int
    fp: int
    fn: int
    tp: int
    inference_seconds: float
    samples_per_second: float


class FlowMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (256, 128, 64),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run flow-only reference baselines from a flow CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--flow_csv", required=True, help="Path to stage1_flows.csv")
    parser.add_argument("--config", default=None, help="Optional Stage1 YAML config")
    parser.add_argument("--out_dir", default="./flow_reference_results")

    parser.add_argument("--flow_id_col", default=None)
    parser.add_argument("--label_col", default=None)
    parser.add_argument("--time_col", default=None)

    parser.add_argument(
        "--feature_cols",
        nargs="*",
        default=None,
        help="Explicit flow feature columns. If omitted, config or safe auto-selection is used.",
    )
    parser.add_argument(
        "--exclude_cols",
        nargs="*",
        default=[],
        help="Additional columns excluded from model input.",
    )
    parser.add_argument(
        "--max_categories",
        type=int,
        default=128,
        help="Drop categorical columns with more unique values than this threshold.",
    )

    parser.add_argument(
        "--split_method",
        choices=["stratified", "chronological"],
        default="stratified",
    )
    parser.add_argument("--val_size", type=float, default=0.10)
    parser.add_argument("--test_size", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=130)

    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_ORDER,
        default=MODEL_ORDER,
    )
    parser.add_argument(
        "--imbalance",
        choices=["none", "class_weight"],
        default="class_weight",
        help="Use exactly one model-level class-imbalance correction. No sampler is used.",
    )
    parser.add_argument(
        "--threshold_objective",
        choices=["f1"],
        default="f1",
    )
    parser.add_argument(
        "--min_precision",
        type=float,
        default=None,
        help="Optional validation precision constraint during threshold search.",
    )

    # Tree models
    parser.add_argument("--rf_estimators", type=int, default=500)
    parser.add_argument("--rf_max_depth", type=int, default=None)
    parser.add_argument("--xgb_estimators", type=int, default=500)
    parser.add_argument("--lgbm_estimators", type=int, default=500)

    # Flow MLP
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--mlp_hidden_dims", type=int, nargs="+", default=[256, 128, 64])
    parser.add_argument("--mlp_dropout", type=float, default=0.3)
    parser.add_argument("--mlp_epochs", type=int, default=200)
    parser.add_argument("--mlp_batch_size", type=int, default=256)
    parser.add_argument("--mlp_lr", type=float, default=1e-3)
    parser.add_argument("--mlp_weight_decay", type=float, default=1e-4)
    parser.add_argument("--mlp_patience", type=int, default=20)

    args = parser.parse_args()
    if args.val_size <= 0 or args.test_size <= 0:
        parser.error("val_size and test_size must both be greater than 0")
    if args.val_size + args.test_size >= 1:
        parser.error("val_size + test_size must be less than 1")
    if args.min_precision is not None and not (0 <= args.min_precision <= 1):
        parser.error("min_precision must be in [0, 1]")
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def load_yaml(path: Optional[str]) -> Dict[str, Any]:
    if path is None:
        return {}
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit("PyYAML is required when --config is used.") from exc
    with open(path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return data


def resolve_columns(args: argparse.Namespace, cfg: Dict[str, Any]) -> Tuple[str, str, str]:
    data_cfg = cfg.get("data", {}) if isinstance(cfg.get("data", {}), dict) else {}
    flow_id_col = args.flow_id_col or data_cfg.get("flow_id_col", "flow_id")
    label_col = args.label_col or data_cfg.get("label_col", "label")
    time_col = args.time_col or data_cfg.get("flow_time_col", "flow_start_timestamp_us")
    return str(flow_id_col), str(label_col), str(time_col)


def _flatten_feature_block(block: Any) -> List[str]:
    if not isinstance(block, dict):
        return []
    result: List[str] = []
    for key in ("numerical", "numeric", "categorical", "binary", "features", "columns"):
        values = block.get(key)
        if isinstance(values, list):
            result.extend(str(value) for value in values)
    return result


def feature_columns_from_config(cfg: Dict[str, Any]) -> List[str]:
    """Read common flow-feature layouts without depending on fusion.enabled."""
    features_cfg = cfg.get("features", {})
    if not isinstance(features_cfg, dict):
        return []

    candidates: List[str] = []
    candidates.extend(_flatten_feature_block(features_cfg.get("flow")))
    candidates.extend(_flatten_feature_block(features_cfg.get("flow_features")))

    flow_fusion = features_cfg.get("flow_fusion")
    candidates.extend(_flatten_feature_block(flow_fusion))

    direct = features_cfg.get("flow_feature_cols")
    if isinstance(direct, list):
        candidates.extend(str(value) for value in direct)

    # Preserve config order and remove duplicates.
    return list(dict.fromkeys(candidates))


def clean_flow_dataframe(
    df: pd.DataFrame,
    flow_id_col: str,
    label_col: str,
    time_col: str,
) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(column).strip() for column in df.columns]
    if "record_type" in df.columns:
        df = df.drop(columns=["record_type"])
    df = df.replace([np.inf, -np.inf], np.nan)

    required = [flow_id_col, label_col]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Flow CSV missing required columns: {missing}")

    df[flow_id_col] = pd.to_numeric(df[flow_id_col], errors="coerce")
    df[label_col] = pd.to_numeric(df[label_col], errors="coerce")
    df = df[df[flow_id_col].notna() & df[label_col].notna()].copy()
    df = df[df[flow_id_col] != 0].copy()

    if time_col in df.columns:
        df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
        df = df[df[time_col].notna() & (df[time_col] != 0)].copy()

    df[flow_id_col] = df[flow_id_col].astype(np.int64)
    return df.reset_index(drop=True)


def encode_binary_labels(labels: pd.Series) -> Tuple[np.ndarray, Dict[str, int]]:
    values = labels.dropna().unique().tolist()
    if len(values) != 2:
        raise ValueError(
            f"This Experiment A script currently expects exactly two classes; found {values}"
        )

    if set(values) == {0, 1}:
        mapping = {"0": 0, "1": 1}
        encoded = labels.astype(int).to_numpy(dtype=np.int64)
        return encoded, mapping

    sorted_values = sorted(values, key=lambda value: str(value))
    raw_mapping = {value: index for index, value in enumerate(sorted_values)}
    encoded = labels.map(raw_mapping).to_numpy(dtype=np.int64)
    mapping = {str(key): int(value) for key, value in raw_mapping.items()}
    return encoded, mapping


def select_feature_columns(
    df: pd.DataFrame,
    explicit_cols: Optional[Sequence[str]],
    config_cols: Sequence[str],
    flow_id_col: str,
    label_col: str,
    time_col: str,
    extra_excludes: Sequence[str],
    max_categories: int,
) -> Tuple[List[str], List[str], List[str], List[str]]:
    excluded = set(DEFAULT_EXCLUDE_COLUMNS)
    excluded.update({flow_id_col, label_col, time_col})
    excluded.update(str(column) for column in extra_excludes)

    if explicit_cols:
        requested = list(dict.fromkeys(str(column) for column in explicit_cols))
        missing = [column for column in requested if column not in df.columns]
        if missing:
            raise ValueError(f"Requested feature columns missing from flow CSV: {missing}")
        selected = [column for column in requested if column not in excluded]
        source = "command line"
    else:
        usable_config_cols = [
            column for column in config_cols if column in df.columns and column not in excluded
        ]
        if usable_config_cols:
            selected = list(dict.fromkeys(usable_config_cols))
            source = "config"
        else:
            selected = [column for column in df.columns if column not in excluded]
            source = "automatic inference"

    if not selected:
        raise ValueError("No flow feature columns remain after exclusions.")

    numeric_cols: List[str] = []
    categorical_cols: List[str] = []
    dropped_cols: List[str] = []

    for column in selected:
        series = df[column]
        if pd.api.types.is_bool_dtype(series.dtype) or pd.api.types.is_numeric_dtype(series.dtype):
            numeric_cols.append(column)
            continue

        numeric_version = pd.to_numeric(series, errors="coerce")
        non_null = int(series.notna().sum())
        numeric_fraction = float(numeric_version.notna().sum()) / max(non_null, 1)
        if numeric_fraction >= 0.98:
            df[column] = numeric_version
            numeric_cols.append(column)
            continue

        unique_count = int(series.nunique(dropna=True))
        if unique_count <= max_categories:
            categorical_cols.append(column)
        else:
            dropped_cols.append(column)

    final_cols = numeric_cols + categorical_cols
    if not final_cols:
        raise ValueError("No usable numeric or low-cardinality categorical flow features found.")

    print(f"[INFO] Feature selection source: {source}")
    print(f"[INFO] Numeric flow features ({len(numeric_cols)}): {numeric_cols}")
    print(f"[INFO] Categorical flow features ({len(categorical_cols)}): {categorical_cols}")
    if dropped_cols:
        print(
            "[WARN] Dropped high-cardinality non-numeric columns: "
            f"{dropped_cols}"
        )

    return final_cols, numeric_cols, categorical_cols, dropped_cols


def build_preprocessor(
    numeric_cols: Sequence[str],
    categorical_cols: Sequence[str],
) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # scikit-learn renamed sparse -> sparse_output. Support both APIs.
    try:
        one_hot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # scikit-learn < 1.2
        one_hot = OneHotEncoder(handle_unknown="ignore", sparse=False)

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", one_hot),
        ]
    )

    transformers = []
    if numeric_cols:
        transformers.append(("numeric", numeric_pipeline, list(numeric_cols)))
    if categorical_cols:
        transformers.append(("categorical", categorical_pipeline, list(categorical_cols)))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def make_splits(
    df: pd.DataFrame,
    y: np.ndarray,
    method: str,
    val_size: float,
    test_size: float,
    seed: int,
    time_col: str,
) -> SplitData:
    indices = np.arange(len(df), dtype=np.int64)

    if method == "chronological":
        if time_col not in df.columns:
            raise ValueError(
                f"Chronological split requested, but time column '{time_col}' is absent."
            )
        ordered = np.argsort(df[time_col].to_numpy(), kind="mergesort")
        n_total = len(ordered)
        n_test = max(1, int(math.ceil(n_total * test_size)))
        n_val = max(1, int(math.ceil(n_total * val_size)))
        n_train = n_total - n_val - n_test
        if n_train <= 0:
            raise ValueError("Not enough rows for the requested chronological split sizes.")
        return SplitData(
            train_idx=ordered[:n_train],
            val_idx=ordered[n_train : n_train + n_val],
            test_idx=ordered[n_train + n_val :],
        )

    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )
    adjusted_val_size = val_size / (1.0 - test_size)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=adjusted_val_size,
        random_state=seed,
        stratify=y[train_val_idx],
    )
    return SplitData(
        train_idx=np.asarray(train_idx, dtype=np.int64),
        val_idx=np.asarray(val_idx, dtype=np.int64),
        test_idx=np.asarray(test_idx, dtype=np.int64),
    )


def validate_split_classes(y: np.ndarray, split: SplitData) -> None:
    for split_name, split_idx in (
        ("train", split.train_idx),
        ("val", split.val_idx),
        ("test", split.test_idx),
    ):
        values, counts = np.unique(y[split_idx], return_counts=True)
        distribution = {int(value): int(count) for value, count in zip(values, counts)}
        print(f"[INFO] {split_name} label distribution: {distribution}")
        if len(values) < 2:
            warnings.warn(
                f"{split_name} contains only one class; AUC metrics may be undefined.",
                RuntimeWarning,
            )


def binary_class_weight(y_train: np.ndarray) -> Tuple[float, Dict[int, float]]:
    counts = np.bincount(y_train, minlength=2).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError(f"Training split is missing a class: counts={counts.tolist()}")
    total = counts.sum()
    weights = total / (2.0 * counts)
    class_weight = {0: float(weights[0]), 1: float(weights[1])}
    scale_pos_weight = float(counts[0] / counts[1])
    return scale_pos_weight, class_weight


def find_optimal_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    min_precision: Optional[float] = None,
) -> Tuple[float, Dict[str, float]]:
    best_threshold = 0.5
    best_tuple = (-1.0, -1.0, -1.0)  # f1, recall, precision
    best_info = {"f1": 0.0, "precision": 0.0, "recall": 0.0}

    thresholds = np.unique(
        np.concatenate(
            [
                np.linspace(0.01, 0.99, 197),
                np.clip(scores, 0.0, 1.0),
            ]
        )
    )

    constrained_candidates = 0
    for threshold in thresholds:
        predictions = (scores >= threshold).astype(np.int64)
        precision = precision_score(y_true, predictions, zero_division=0)
        recall = recall_score(y_true, predictions, zero_division=0)
        f1 = f1_score(y_true, predictions, zero_division=0)

        if min_precision is not None and precision < min_precision:
            continue
        constrained_candidates += 1
        current_tuple = (f1, recall, precision)
        if current_tuple > best_tuple:
            best_tuple = current_tuple
            best_threshold = float(threshold)
            best_info = {
                "f1": float(f1),
                "precision": float(precision),
                "recall": float(recall),
            }

    if min_precision is not None and constrained_candidates == 0:
        warnings.warn(
            "No validation threshold satisfied min_precision; falling back to unconstrained F1.",
            RuntimeWarning,
        )
        return find_optimal_threshold(y_true, scores, min_precision=None)

    return best_threshold, best_info


def safe_auc(metric_fn, y_true: np.ndarray, scores: np.ndarray) -> float:
    try:
        return float(metric_fn(y_true, scores))
    except ValueError:
        return float("nan")


def evaluate_binary(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    inference_seconds: float,
) -> MetricResult:
    predictions = (scores >= threshold).astype(np.int64)
    cm = confusion_matrix(y_true, predictions, labels=[0, 1])
    tn, fp, fn, tp = [int(value) for value in cm.ravel()]
    tpr = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)
    samples_per_second = len(y_true) / max(inference_seconds, 1e-12)

    return MetricResult(
        threshold=float(threshold),
        accuracy=float(accuracy_score(y_true, predictions)),
        precision_class1=float(precision_score(y_true, predictions, zero_division=0)),
        recall_class1=float(recall_score(y_true, predictions, zero_division=0)),
        f1_class1=float(f1_score(y_true, predictions, zero_division=0)),
        macro_f1=float(f1_score(y_true, predictions, average="macro", zero_division=0)),
        weighted_f1=float(
            f1_score(y_true, predictions, average="weighted", zero_division=0)
        ),
        roc_auc=safe_auc(roc_auc_score, y_true, scores),
        pr_auc=safe_auc(average_precision_score, y_true, scores),
        tpr=float(tpr),
        fpr=float(fpr),
        tn=tn,
        fp=fp,
        fn=fn,
        tp=tp,
        inference_seconds=float(inference_seconds),
        samples_per_second=float(samples_per_second),
    )


def predict_scores_sklearn(model: Any, x: np.ndarray) -> Tuple[np.ndarray, float]:
    start = time.perf_counter()
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(x)[:, 1]
    elif hasattr(model, "decision_function"):
        raw_scores = model.decision_function(x)
        scores = 1.0 / (1.0 + np.exp(-raw_scores))
    else:
        raise TypeError(f"Model {type(model).__name__} exposes no probability-like output.")
    elapsed = time.perf_counter() - start
    return np.asarray(scores, dtype=np.float64), elapsed


def train_random_forest(
    x_train: np.ndarray,
    y_train: np.ndarray,
    args: argparse.Namespace,
) -> RandomForestClassifier:
    class_weight = "balanced_subsample" if args.imbalance == "class_weight" else None
    model = RandomForestClassifier(
        n_estimators=args.rf_estimators,
        max_depth=args.rf_max_depth,
        class_weight=class_weight,
        random_state=args.seed,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    return model


def train_xgboost(
    x_train: np.ndarray,
    y_train: np.ndarray,
    args: argparse.Namespace,
    scale_pos_weight: float,
) -> Any:
    try:
        import xgboost as xgb
    except ImportError as exc:
        raise RuntimeError(
            "XGBoost is not installed. Run: pip install xgboost"
        ) from exc

    model = xgb.XGBClassifier(
        n_estimators=args.xgb_estimators,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight if args.imbalance == "class_weight" else 1.0,
        random_state=args.seed,
        n_jobs=-1,
        tree_method="hist",
    )
    model.fit(x_train, y_train, verbose=False)
    return model


def train_lightgbm(
    x_train: np.ndarray,
    y_train: np.ndarray,
    args: argparse.Namespace,
) -> Any:
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise RuntimeError(
            "LightGBM is not installed. Run: pip install lightgbm"
        ) from exc

    model = lgb.LGBMClassifier(
        n_estimators=args.lgbm_estimators,
        max_depth=-1,
        num_leaves=31,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        class_weight="balanced" if args.imbalance == "class_weight" else None,
        random_state=args.seed,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(x_train, y_train)
    return model


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        warnings.warn("CUDA requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_arg)


def mlp_scores(
    model: FlowMLP,
    x: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> Tuple[np.ndarray, float]:
    model.eval()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x.astype(np.float32, copy=False))),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    outputs: List[np.ndarray] = []
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        for (batch_x,) in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            logits = model(batch_x)
            outputs.append(torch.sigmoid(logits).cpu().numpy())
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return np.concatenate(outputs).astype(np.float64), elapsed


def train_flow_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    args: argparse.Namespace,
    scale_pos_weight: float,
    out_dir: Path,
) -> FlowMLP:
    device = resolve_device(args.device)
    print(f"[INFO] FlowMLP device: {device}")

    train_dataset = TensorDataset(
        torch.from_numpy(x_train.astype(np.float32, copy=False)),
        torch.from_numpy(y_train.astype(np.float32, copy=False)),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.mlp_batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(args.seed),
    )

    model = FlowMLP(
        input_dim=x_train.shape[1],
        hidden_dims=args.mlp_hidden_dims,
        dropout=args.mlp_dropout,
    ).to(device)

    if args.imbalance == "class_weight":
        pos_weight = torch.tensor([scale_pos_weight], dtype=torch.float32, device=device)
    else:
        pos_weight = None
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.mlp_lr,
        weight_decay=args.mlp_weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=max(2, args.mlp_patience // 4),
    )

    x_val_tensor = torch.from_numpy(x_val.astype(np.float32, copy=False)).to(device)
    y_val_tensor = torch.from_numpy(y_val.astype(np.float32, copy=False)).to(device)

    best_val_loss = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    patience_counter = 0
    history: List[Dict[str, float]] = []

    for epoch in range(1, args.mlp_epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_examples = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_loss_sum += float(loss.item()) * len(batch_y)
            train_examples += len(batch_y)

        model.eval()
        with torch.no_grad():
            val_logits = model(x_val_tensor)
            val_loss = float(criterion(val_logits, y_val_tensor).item())

        train_loss = train_loss_sum / max(train_examples, 1)
        scheduler.step(val_loss)
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        )

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"[FlowMLP] epoch={epoch:03d} "
                f"train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
                f"best_val_loss={best_val_loss:.6f}"
            )

        if patience_counter >= args.mlp_patience:
            print(f"[FlowMLP] early stopping at epoch {epoch}")
            break

    if best_state is None:
        raise RuntimeError("FlowMLP training did not produce a valid checkpoint.")
    model.load_state_dict(best_state)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": int(x_train.shape[1]),
            "hidden_dims": list(args.mlp_hidden_dims),
            "dropout": float(args.mlp_dropout),
            "best_val_loss": float(best_val_loss),
        },
        out_dir / "flow_mlp_best.pt",
    )
    pd.DataFrame(history).to_csv(out_dir / "flow_mlp_history.csv", index=False)
    return model


def save_predictions(
    output_path: Path,
    flow_ids: np.ndarray,
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> None:
    predictions = (scores >= threshold).astype(np.int64)
    pd.DataFrame(
        {
            "flow_id": flow_ids.astype(np.int64),
            "label": y_true.astype(np.int64),
            "score_class1": scores.astype(np.float64),
            "prediction": predictions,
            "threshold": np.full(len(y_true), threshold, dtype=np.float64),
        }
    ).to_csv(output_path, index=False)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_yaml(args.config)
    flow_id_col, label_col, time_col = resolve_columns(args, cfg)

    print(f"[INFO] Reading flow CSV: {args.flow_csv}")
    flows = pd.read_csv(args.flow_csv, low_memory=False)
    flows = clean_flow_dataframe(flows, flow_id_col, label_col, time_col)
    print(f"[INFO] Clean flow rows: {len(flows):,}")

    y, label_mapping = encode_binary_labels(flows[label_col])
    config_feature_cols = feature_columns_from_config(cfg)
    feature_cols, numeric_cols, categorical_cols, dropped_cols = select_feature_columns(
        df=flows,
        explicit_cols=args.feature_cols,
        config_cols=config_feature_cols,
        flow_id_col=flow_id_col,
        label_col=label_col,
        time_col=time_col,
        extra_excludes=args.exclude_cols,
        max_categories=args.max_categories,
    )

    split = make_splits(
        df=flows,
        y=y,
        method=args.split_method,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
        time_col=time_col,
    )
    validate_split_classes(y, split)

    split_labels = np.full(len(flows), "", dtype=object)
    split_labels[split.train_idx] = "train"
    split_labels[split.val_idx] = "val"
    split_labels[split.test_idx] = "test"
    split_manifest = pd.DataFrame(
        {
            "flow_id": flows[flow_id_col].to_numpy(dtype=np.int64),
            "split": split_labels,
            "label": y,
        }
    )
    if time_col in flows.columns:
        split_manifest[time_col] = flows[time_col].to_numpy()
    split_manifest.to_csv(out_dir / "flow_reference_split.csv", index=False)

    x_frame = flows[feature_cols].copy()
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    x_train = preprocessor.fit_transform(x_frame.iloc[split.train_idx])
    x_val = preprocessor.transform(x_frame.iloc[split.val_idx])
    x_test = preprocessor.transform(x_frame.iloc[split.test_idx])

    x_train = np.asarray(x_train, dtype=np.float32)
    x_val = np.asarray(x_val, dtype=np.float32)
    x_test = np.asarray(x_test, dtype=np.float32)
    y_train = y[split.train_idx]
    y_val = y[split.val_idx]
    y_test = y[split.test_idx]

    print(f"[INFO] Encoded flow feature dimension: {x_train.shape[1]}")
    joblib.dump(preprocessor, out_dir / "flow_reference_preprocessor.joblib")

    scale_pos_weight, class_weight = binary_class_weight(y_train)
    print(f"[INFO] imbalance={args.imbalance}")
    print(f"[INFO] scale_pos_weight={scale_pos_weight:.6f}")
    print(f"[INFO] balanced class weights={class_weight}")

    feature_manifest = {
        "flow_csv": str(Path(args.flow_csv).resolve()),
        "flow_id_col": flow_id_col,
        "label_col": label_col,
        "time_col": time_col,
        "label_mapping": label_mapping,
        "feature_columns": feature_cols,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "dropped_high_cardinality_columns": dropped_cols,
        "encoded_feature_dim": int(x_train.shape[1]),
        "split_method": args.split_method,
        "seed": args.seed,
        "val_size": args.val_size,
        "test_size": args.test_size,
        "imbalance": args.imbalance,
        "note": "Flow-only input. No packet sequence and no packet loader are used.",
    }
    with open(out_dir / "flow_reference_manifest.json", "w", encoding="utf-8") as file:
        json.dump(json_safe(feature_manifest), file, ensure_ascii=False, indent=2)

    results: Dict[str, Dict[str, Any]] = {}
    failures: Dict[str, str] = {}

    for model_key in args.models:
        display_name = MODEL_DISPLAY_NAMES[model_key]
        print("\n" + "=" * 72)
        print(f"[MODEL] {display_name}")
        print("=" * 72)

        train_start = time.perf_counter()
        try:
            if model_key == "random_forest":
                model = train_random_forest(x_train, y_train, args)
                joblib.dump(model, out_dir / "random_forest.joblib")
                val_scores, val_inference = predict_scores_sklearn(model, x_val)
                test_scores, test_inference = predict_scores_sklearn(model, x_test)

            elif model_key == "xgboost":
                model = train_xgboost(x_train, y_train, args, scale_pos_weight)
                joblib.dump(model, out_dir / "xgboost.joblib")
                val_scores, val_inference = predict_scores_sklearn(model, x_val)
                test_scores, test_inference = predict_scores_sklearn(model, x_test)

            elif model_key == "lightgbm":
                model = train_lightgbm(x_train, y_train, args)
                joblib.dump(model, out_dir / "lightgbm.joblib")
                val_scores, val_inference = predict_scores_sklearn(model, x_val)
                test_scores, test_inference = predict_scores_sklearn(model, x_test)

            elif model_key == "flow_mlp":
                model = train_flow_mlp(
                    x_train=x_train,
                    y_train=y_train,
                    x_val=x_val,
                    y_val=y_val,
                    args=args,
                    scale_pos_weight=scale_pos_weight,
                    out_dir=out_dir,
                )
                device = resolve_device(args.device)
                val_scores, val_inference = mlp_scores(
                    model, x_val, device, args.mlp_batch_size
                )
                test_scores, test_inference = mlp_scores(
                    model, x_test, device, args.mlp_batch_size
                )
            else:  # pragma: no cover
                raise ValueError(f"Unsupported model: {model_key}")

            training_seconds = time.perf_counter() - train_start
            threshold, threshold_info = find_optimal_threshold(
                y_true=y_val,
                scores=val_scores,
                min_precision=args.min_precision,
            )
            val_metrics = evaluate_binary(
                y_true=y_val,
                scores=val_scores,
                threshold=threshold,
                inference_seconds=val_inference,
            )
            test_metrics = evaluate_binary(
                y_true=y_test,
                scores=test_scores,
                threshold=threshold,
                inference_seconds=test_inference,
            )

            results[display_name] = {
                "training_seconds": float(training_seconds),
                "validation_threshold_search": threshold_info,
                "validation": asdict(val_metrics),
                "test": asdict(test_metrics),
            }

            save_predictions(
                out_dir / f"predictions_val_{model_key}.csv",
                flows.iloc[split.val_idx][flow_id_col].to_numpy(),
                y_val,
                val_scores,
                threshold,
            )
            save_predictions(
                out_dir / f"predictions_test_{model_key}.csv",
                flows.iloc[split.test_idx][flow_id_col].to_numpy(),
                y_test,
                test_scores,
                threshold,
            )

            print(
                f"[{display_name}] threshold={threshold:.4f} "
                f"test_F1={test_metrics.f1_class1:.4f} "
                f"test_MacroF1={test_metrics.macro_f1:.4f} "
                f"test_PRAUC={test_metrics.pr_auc:.4f} "
                f"test_ROCAUC={test_metrics.roc_auc:.4f} "
                f"test_FPR={test_metrics.fpr:.4f}"
            )

        except Exception as exc:
            failures[display_name] = f"{type(exc).__name__}: {exc}"
            print(f"[ERROR] {display_name} failed: {failures[display_name]}", file=sys.stderr)

    if not results:
        raise SystemExit(f"All requested models failed: {failures}")

    summary_rows: List[Dict[str, Any]] = []
    for model_name, model_result in results.items():
        row: Dict[str, Any] = {
            "model": model_name,
            "training_seconds": model_result["training_seconds"],
        }
        row.update(model_result["test"])
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["f1_class1", "pr_auc"], ascending=False
    )
    summary_df.to_csv(out_dir / "flow_reference_test_results.csv", index=False)

    complete_results = {
        "manifest": feature_manifest,
        "results": results,
        "failures": failures,
    }
    with open(out_dir / "flow_reference_results.json", "w", encoding="utf-8") as file:
        json.dump(json_safe(complete_results), file, ensure_ascii=False, indent=2)

    print("\n" + "=" * 72)
    print("[FINAL TEST RESULTS]")
    print("=" * 72)
    display_columns = [
        "model",
        "threshold",
        "precision_class1",
        "recall_class1",
        "f1_class1",
        "macro_f1",
        "pr_auc",
        "roc_auc",
        "fpr",
        "training_seconds",
    ]
    print(summary_df[display_columns].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    if failures:
        print(f"\n[WARN] Some optional models failed: {failures}")
    print(f"\n[INFO] Results saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()