#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Binary flow-level baselines for custom Suricata Stage1 flow CSV files.

Default target:
    dataset/Wednesday-workingHours-stage1_flows.csv

The script maps label 0 to benign/class0 and any non-zero label to attack/class1.
It drops flow identifiers, timestamps, and endpoint IPs by default, then trains
classical ML baselines and selects the decision threshold on the validation set.

Example:
    python s1/suricata_flow_binary_baseline.py ^
      --flow_csv dataset/Wednesday-workingHours-stage1_flows.csv ^
      --out_dir s1/results/wednesday_suricata_flow_binary ^
      --models logreg extratrees xgb lgbm
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import RobustScaler


DEFAULT_FLOW_CSV = "dataset/Wednesday-workingHours-stage1_flows.csv"
DEFAULT_OUT_DIR = "s1/results/wednesday_suricata_flow_binary_baseline"
DEFAULT_MODELS = ["logreg", "extratrees", "xgb", "lgbm"]

IDENTIFIER_COLUMNS = {
    "flow_id",
    "flow_start_timestamp_us",
    "flow_end_timestamp_us",
    "source_ip",
    "destination_ip",
    "src_ip",
    "dst_ip",
}

PORT_COLUMNS = {
    "source_port",
    "destination_port",
    "src_port",
    "dst_port",
}

PROTOCOL_COLUMNS = {
    "protocol",
}

STAGE1_FLOW_LOG1P_COLUMNS = {
    "flow_duration",
    "total_fwd_packets",
    "total_backward_packets",
    "total_length_of_fwd_packets",
    "total_length_of_bwd_packets",
    "fwd_packet_length_max",
    "fwd_packet_length_min",
    "fwd_packet_length_mean",
    "fwd_packet_length_std",
    "bwd_packet_length_max",
    "bwd_packet_length_min",
    "bwd_packet_length_mean",
    "bwd_packet_length_std",
    "flow_bytes_per_s",
    "flow_packets_per_s",
    "flow_iat_mean",
    "flow_iat_std",
    "flow_iat_max",
    "flow_iat_min",
    "fwd_iat_total",
    "fwd_iat_mean",
    "fwd_iat_std",
    "fwd_iat_max",
    "fwd_iat_min",
    "bwd_iat_total",
    "bwd_iat_mean",
    "bwd_iat_std",
    "bwd_iat_max",
    "bwd_iat_min",
    "fwd_header_length",
    "bwd_header_length",
    "fwd_packets_per_s",
    "bwd_packets_per_s",
    "min_packet_length",
    "max_packet_length",
    "packet_length_mean",
    "packet_length_std",
    "packet_length_variance",
    "average_packet_size",
    "avg_fwd_segment_size",
    "avg_bwd_segment_size",
    "init_win_bytes_forward",
    "init_win_bytes_backward",
    "act_data_pkt_fwd",
    "active_mean",
    "active_std",
    "active_max",
    "active_min",
    "idle_mean",
    "idle_std",
    "idle_max",
    "idle_min",
}


@dataclass
class PreparedData:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray
    test_flow_ids: Optional[np.ndarray]
    feature_cols: List[str]
    log1p_cols: List[str]
    dropped_feature_cols: Dict[str, Any]
    imputer: SimpleImputer
    scaler: Optional[RobustScaler]


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def save_json(obj: Dict[str, Any], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(obj), f, indent=2, ensure_ascii=False)


def write_lines(lines: Iterable[str], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(f"{line}\n")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def split_csv_arg(value: Optional[str]) -> List[str]:
    if value is None or str(value).strip() == "":
        return []
    return [x.strip() for x in str(value).split(",") if x.strip()]


def find_required_column(columns: Sequence[str], requested: str) -> str:
    if requested in columns:
        return requested
    lower_map = {str(c).strip().lower(): c for c in columns}
    key = str(requested).strip().lower()
    if key in lower_map:
        return lower_map[key]
    raise ValueError(f"Required column {requested!r} not found.")


def make_binary_labels(label_series: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(label_series, errors="coerce")
    if numeric.notna().mean() >= 0.95:
        return (numeric.fillna(0).to_numpy(dtype=np.float64) != 0).astype(np.int64)

    normalized = label_series.astype(str).str.strip().str.upper()
    benign_values = {"0", "BENIGN", "NORMAL", "BACKGROUND"}
    return (~normalized.isin(benign_values)).astype(np.int64).to_numpy()


def load_flow_csv(args: argparse.Namespace) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    flow_csv = Path(args.flow_csv)
    if not flow_csv.exists():
        raise FileNotFoundError(flow_csv)

    print(f"[INFO] loading flow CSV: {flow_csv}")
    df = pd.read_csv(flow_csv, low_memory=False)
    raw_rows, raw_cols = df.shape
    df = normalize_columns(df)

    label_col = find_required_column(df.columns, args.label_col)
    before = len(df)
    df = df[df[label_col].notna()].copy()
    dropped_missing_label_rows = before - len(df)
    y = make_binary_labels(df[label_col])

    flow_id_col = None
    duplicate_flow_id_rows = None
    if args.flow_id_col in df.columns:
        flow_id_col = args.flow_id_col
    else:
        try:
            flow_id_col = find_required_column(df.columns, args.flow_id_col)
        except ValueError:
            flow_id_col = None

    if flow_id_col is not None:
        duplicate_flow_id_rows = int(df[flow_id_col].duplicated().sum())
        if args.drop_duplicate_flow_ids:
            before = len(df)
            df = df.drop_duplicates(subset=[flow_id_col], keep="first").reset_index(drop=True)
            y = make_binary_labels(df[label_col])
            print(f"[WARN] dropped duplicated flow_id rows: {before - len(df):,}")

    label_counts_raw = df[label_col].value_counts(dropna=False).sort_index().to_dict()
    binary_counts = {
        int(k): int(v) for k, v in pd.Series(y).value_counts().sort_index().items()
    }
    report = {
        "flow_csv": str(flow_csv),
        "file_size_mb": flow_csv.stat().st_size / 1024 / 1024,
        "raw_num_rows": int(raw_rows),
        "raw_num_columns": int(raw_cols),
        "num_rows_after_label_filter": int(len(df)),
        "num_columns": int(len(df.columns)),
        "label_column": label_col,
        "label_counts_raw": label_counts_raw,
        "binary_counts": binary_counts,
        "attack_ratio": float(np.mean(y)) if len(y) else 0.0,
        "flow_id_column": flow_id_col,
        "duplicate_flow_id_rows": duplicate_flow_id_rows,
        "dropped_missing_label_rows": int(dropped_missing_label_rows),
        "dropped_duplicate_flow_ids": bool(args.drop_duplicate_flow_ids),
    }
    print(
        "[DATA] rows={:,}, cols={}, binary_counts={}, attack_ratio={:.6f}".format(
            len(df), len(df.columns), binary_counts, report["attack_ratio"]
        )
    )
    if duplicate_flow_id_rows is not None:
        print(f"[DATA] duplicated flow_id rows={duplicate_flow_id_rows:,}")
    return df, report


def choose_feature_columns(
    df: pd.DataFrame,
    label_col: str,
    args: argparse.Namespace,
) -> Tuple[List[str], Dict[str, Any]]:
    excluded = {label_col}
    excluded.update(IDENTIFIER_COLUMNS)
    if not args.include_ports:
        excluded.update(PORT_COLUMNS)
    if not args.include_protocol:
        excluded.update(PROTOCOL_COLUMNS)
    excluded.update(split_csv_arg(args.drop_columns))

    candidate_cols = [c for c in df.columns if c not in excluded]
    dropped: Dict[str, Any] = {
        "identifier_time_endpoint_label_or_disabled": [c for c in df.columns if c in excluded],
        "non_numeric": [],
    }

    numeric_cols: List[str] = []
    for col in candidate_cols:
        converted = pd.to_numeric(df[col], errors="coerce")
        valid_ratio = float(converted.notna().mean())
        if valid_ratio >= args.min_numeric_ratio:
            numeric_cols.append(col)
        else:
            dropped["non_numeric"].append(col)

    if not numeric_cols:
        raise ValueError("No numeric feature columns left after filtering.")

    print(f"[DATA] numeric feature candidates={len(numeric_cols)}")
    return numeric_cols, dropped


def split_indices(df: pd.DataFrame, y: np.ndarray, args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    total = args.train_size + args.val_size + args.test_size
    if abs(total - 1.0) > 1e-6:
        raise ValueError("train_size + val_size + test_size must equal 1.0")

    indices = np.arange(len(df), dtype=np.int64)
    if args.split == "chronological":
        time_col = find_required_column(df.columns, args.time_col)
        time_values = pd.to_numeric(df[time_col], errors="coerce")
        if time_values.isna().any():
            raise ValueError(f"Time column {time_col!r} contains non-numeric values.")
        order = np.argsort(time_values.to_numpy(), kind="mergesort")
        indices = indices[order]
        n = len(indices)
        b1 = int(round(n * args.train_size))
        b2 = int(round(n * (args.train_size + args.val_size)))
        b1 = min(max(b1, 1), n - 2)
        b2 = min(max(b2, b1 + 1), n - 1)
        return indices[:b1], indices[b1:b2], indices[b2:]

    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=args.test_size,
        random_state=args.seed,
        shuffle=True,
        stratify=y if args.stratify else None,
    )
    val_fraction = args.val_size / (args.train_size + args.val_size)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_fraction,
        random_state=args.seed,
        shuffle=True,
        stratify=y[train_val_idx] if args.stratify else None,
    )
    return train_idx, val_idx, test_idx


def select_log1p_columns(
    x_numeric: pd.DataFrame,
    args: argparse.Namespace,
) -> List[str]:
    mode = str(args.log1p_mode).lower()
    if mode == "none":
        return []
    if mode == "stage1_selected":
        return [col for col in x_numeric.columns if col in STAGE1_FLOW_LOG1P_COLUMNS]
    if mode == "all_nonnegative":
        log_cols: List[str] = []
        excluded = set(PORT_COLUMNS) | set(PROTOCOL_COLUMNS)
        for col in x_numeric.columns:
            if col in excluded:
                continue
            values = x_numeric[col].replace([np.inf, -np.inf], np.nan).dropna()
            if values.empty:
                continue
            if float(values.min()) >= 0.0:
                log_cols.append(col)
        return log_cols
    raise ValueError(f"Unsupported log1p_mode: {args.log1p_mode!r}")


def apply_log1p(x_numeric: pd.DataFrame, log1p_cols: Sequence[str]) -> pd.DataFrame:
    if not log1p_cols:
        return x_numeric
    x_numeric = x_numeric.copy()
    for col in log1p_cols:
        if col not in x_numeric.columns:
            continue
        values = pd.to_numeric(x_numeric[col], errors="coerce")
        values = values.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        values = values.clip(lower=0.0)
        x_numeric[col] = np.log1p(values.astype(np.float64))
    return x_numeric


def numeric_frame(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    args: argparse.Namespace,
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    x_numeric = df.loc[:, feature_cols].apply(pd.to_numeric, errors="coerce")
    inf_count = pd.Series(
        np.isinf(x_numeric.to_numpy(dtype=float, copy=False)).sum(axis=0),
        index=x_numeric.columns,
    ).sort_values(ascending=False)
    x_numeric = x_numeric.replace([np.inf, -np.inf], np.nan)
    log1p_cols = select_log1p_columns(x_numeric, args)
    x_numeric = apply_log1p(x_numeric, log1p_cols)
    return x_numeric, inf_count, log1p_cols


def fit_transform_features(
    df: pd.DataFrame,
    feature_cols: List[str],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], List[str], Dict[str, Any], SimpleImputer, Optional[RobustScaler]]:
    x_all, inf_count, log1p_cols = numeric_frame(df, feature_cols, args)
    missing_count = x_all.isna().sum().sort_values(ascending=False)
    x_train_raw = x_all.iloc[train_idx]

    keep_cols: List[str] = []
    dropped_all_nan: List[str] = []
    dropped_constant: List[str] = []
    for col in feature_cols:
        s = x_train_raw[col].dropna()
        if s.empty:
            dropped_all_nan.append(col)
        elif s.nunique(dropna=True) <= 1:
            dropped_constant.append(col)
        else:
            keep_cols.append(col)

    if not keep_cols:
        raise ValueError("No usable feature columns after all-NaN/constant filtering.")

    x_train_raw = x_all.iloc[train_idx][keep_cols]
    x_val_raw = x_all.iloc[val_idx][keep_cols]
    x_test_raw = x_all.iloc[test_idx][keep_cols]

    imputer = SimpleImputer(strategy=args.impute_strategy)
    x_train = imputer.fit_transform(x_train_raw)
    x_val = imputer.transform(x_val_raw)
    x_test = imputer.transform(x_test_raw)

    scaler: Optional[RobustScaler] = None
    if args.scale:
        scaler = RobustScaler(
            with_centering=True,
            with_scaling=True,
            quantile_range=(args.robust_q_low, args.robust_q_high),
        )
        x_train = scaler.fit_transform(x_train)
        x_val = scaler.transform(x_val)
        x_test = scaler.transform(x_test)

    if args.clip_value is not None and args.clip_value > 0:
        clip = float(args.clip_value)
        x_train = np.clip(x_train, -clip, clip)
        x_val = np.clip(x_val, -clip, clip)
        x_test = np.clip(x_test, -clip, clip)

    dropped = {
        "all_nan_on_train": dropped_all_nan,
        "constant_on_train": dropped_constant,
        "top_missing_columns_full": {
            str(k): int(v) for k, v in missing_count[missing_count > 0].head(30).to_dict().items()
        },
        "top_inf_columns_full": {
            str(k): int(v) for k, v in inf_count[inf_count > 0].head(30).to_dict().items()
        },
        "log1p_mode": args.log1p_mode,
        "log1p_columns": log1p_cols,
    }
    print(
        "[DATA] final features={}, log1p_cols={}, dropped_all_nan={}, dropped_constant={}".format(
            len(keep_cols), len(log1p_cols), len(dropped_all_nan), len(dropped_constant)
        )
    )
    return (
        x_train.astype(np.float32),
        x_val.astype(np.float32),
        x_test.astype(np.float32),
        keep_cols,
        [col for col in log1p_cols if col in keep_cols],
        dropped,
        imputer,
        scaler,
    )


def print_split_report(name: str, y: np.ndarray) -> None:
    counts = np.bincount(y.astype(np.int64), minlength=2)
    ratio = counts[1] / max(1, counts.sum())
    print(
        f"[SPLIT] {name}: n={len(y):,}, class0={counts[0]:,}, "
        f"class1={counts[1]:,}, pos_ratio={ratio:.6f}"
    )


def prepare_data(df: pd.DataFrame, dataset_report: Dict[str, Any], args: argparse.Namespace) -> PreparedData:
    label_col = dataset_report["label_column"]
    y = make_binary_labels(df[label_col])

    feature_candidates, dropped_selection = choose_feature_columns(df, label_col, args)
    train_idx, val_idx, test_idx = split_indices(df, y, args)
    x_train, x_val, x_test, final_cols, log1p_cols, dropped_transform, imputer, scaler = fit_transform_features(
        df=df,
        feature_cols=feature_candidates,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        args=args,
    )

    print_split_report("train", y[train_idx])
    print_split_report("val", y[val_idx])
    print_split_report("test", y[test_idx])

    flow_id_col = dataset_report.get("flow_id_column")
    test_flow_ids = None
    if flow_id_col is not None and flow_id_col in df.columns:
        test_flow_ids = df.iloc[test_idx][flow_id_col].to_numpy()

    dropped_feature_cols = {**dropped_selection, **dropped_transform}
    return PreparedData(
        x_train=x_train,
        y_train=y[train_idx],
        x_val=x_val,
        y_val=y[val_idx],
        x_test=x_test,
        y_test=y[test_idx],
        train_indices=train_idx,
        val_indices=val_idx,
        test_indices=test_idx,
        test_flow_ids=test_flow_ids,
        feature_cols=final_cols,
        log1p_cols=log1p_cols,
        dropped_feature_cols=dropped_feature_cols,
        imputer=imputer,
        scaler=scaler,
    )


def safe_binary_log_loss(y_true: np.ndarray, prob1: np.ndarray) -> float:
    prob1 = np.asarray(prob1, dtype=np.float64)
    prob1 = np.clip(prob1, 1e-7, 1.0 - 1e-7)
    return float(log_loss(y_true, np.stack([1.0 - prob1, prob1], axis=1), labels=[0, 1]))


def metrics_at_threshold(y_true: np.ndarray, prob1: np.ndarray, threshold: float) -> Dict[str, Any]:
    prob1 = np.asarray(prob1, dtype=np.float64)
    y_pred = (prob1 >= threshold).astype(np.int64)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    precision_macro, recall_macro, _, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, _, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    per_class_f1 = f1_score(y_true, y_pred, labels=[0, 1], average=None, zero_division=0)

    try:
        auc = float(roc_auc_score(y_true, prob1))
    except Exception:
        auc = float("nan")
    try:
        pr_auc = float(average_precision_score(y_true, prob1))
    except Exception:
        pr_auc = float("nan")

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_macro),
        "macro_recall": float(recall_macro),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_precision": float(precision_weighted),
        "weighted_recall": float(recall_weighted),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision_label1": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_label1": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_label1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "auc": auc,
        "pr_auc": pr_auc,
        "per_class_f1": {str(i): float(v) for i, v in enumerate(per_class_f1)},
        "confusion_matrix": cm.tolist(),
        "num_classes": 2,
        "num_samples": int(len(y_true)),
        "loss": safe_binary_log_loss(y_true, prob1),
        "threshold": float(threshold),
    }


def threshold_search(
    y_true: np.ndarray,
    prob1: np.ndarray,
    steps: int,
    metric: str,
) -> Tuple[float, Dict[str, Any], pd.DataFrame]:
    rows: List[Dict[str, Any]] = []
    for threshold in np.linspace(0.001, 0.999, int(steps)):
        m = metrics_at_threshold(y_true, prob1, float(threshold))
        rows.append(
            {
                "threshold": float(threshold),
                "accuracy": m["accuracy"],
                "precision_label1": m["precision_label1"],
                "recall_label1": m["recall_label1"],
                "f1_label1": m["f1_label1"],
                "macro_f1": m["macro_f1"],
                "weighted_f1": m["weighted_f1"],
                "auc": m["auc"],
                "pr_auc": m["pr_auc"],
            }
        )
    table = pd.DataFrame(rows)
    if metric not in table.columns:
        raise ValueError(f"Unsupported threshold metric {metric!r}. Available: {list(table.columns)}")
    best_idx = int(table[metric].idxmax())
    best_threshold = float(table.loc[best_idx, "threshold"])
    return best_threshold, metrics_at_threshold(y_true, prob1, best_threshold), table


def scale_pos_weight(y_train: np.ndarray) -> float:
    counts = np.bincount(y_train.astype(np.int64), minlength=2)
    if counts[1] == 0:
        return 1.0
    return float(counts[0] / counts[1])


def build_model(name: str, args: argparse.Namespace, y_train: np.ndarray) -> Any:
    name = name.lower()
    if name == "logreg":
        return LogisticRegression(
            max_iter=args.logreg_max_iter,
            class_weight="balanced" if args.class_weight else None,
            solver="lbfgs",
            random_state=args.seed,
        )
    if name == "extratrees":
        return ExtraTreesClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            class_weight="balanced" if args.class_weight else None,
            n_jobs=args.n_jobs,
            random_state=args.seed,
        )
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            class_weight="balanced" if args.class_weight else None,
            n_jobs=args.n_jobs,
            random_state=args.seed,
        )
    if name == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=tuple(args.mlp_hidden),
            activation="relu",
            alpha=args.mlp_alpha,
            batch_size=args.mlp_batch_size,
            learning_rate_init=args.mlp_lr,
            max_iter=args.mlp_max_iter,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            random_state=args.seed,
            verbose=args.verbose_model,
        )
    if name == "xgb":
        try:
            from xgboost import XGBClassifier
        except Exception as exc:
            raise RuntimeError("xgboost is not installed.") from exc
        return XGBClassifier(
            n_estimators=args.xgb_n_estimators,
            max_depth=args.xgb_max_depth,
            learning_rate=args.xgb_learning_rate,
            subsample=args.xgb_subsample,
            colsample_bytree=args.xgb_colsample_bytree,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method=args.xgb_tree_method,
            scale_pos_weight=scale_pos_weight(y_train) if args.class_weight else 1.0,
            n_jobs=args.n_jobs,
            random_state=args.seed,
        )
    if name == "lgbm":
        try:
            from lightgbm import LGBMClassifier
        except Exception as exc:
            raise RuntimeError("lightgbm is not installed.") from exc
        return LGBMClassifier(
            n_estimators=args.lgbm_n_estimators,
            max_depth=args.lgbm_max_depth,
            learning_rate=args.lgbm_learning_rate,
            num_leaves=args.lgbm_num_leaves,
            subsample=args.lgbm_subsample,
            colsample_bytree=args.lgbm_colsample_bytree,
            objective="binary",
            class_weight="balanced" if args.class_weight else None,
            n_jobs=args.n_jobs,
            random_state=args.seed,
            verbose=-1,
        )
    raise ValueError(f"Unknown model {name!r}")


def predict_prob1(model: Any, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(x)
        if prob.ndim == 2 and prob.shape[1] >= 2:
            return prob[:, 1].astype(np.float64)
        return prob.ravel().astype(np.float64)
    if hasattr(model, "decision_function"):
        score = model.decision_function(x)
        return (1.0 / (1.0 + np.exp(-score))).astype(np.float64)
    return model.predict(x).astype(np.float64)


def save_feature_importance(model_name: str, model: Any, feature_cols: List[str], out_dir: Path) -> None:
    rows: Optional[pd.DataFrame] = None
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=np.float64)
        rows = pd.DataFrame({"feature": feature_cols, "importance": values, "abs_importance": np.abs(values)})
    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_)
        if coef.ndim == 2:
            coef = coef[0]
        rows = pd.DataFrame({"feature": feature_cols, "coefficient": coef, "abs_importance": np.abs(coef)})

    if rows is None:
        return
    rows.sort_values("abs_importance", ascending=False).to_csv(
        out_dir / f"{model_name}_feature_importance.csv",
        index=False,
    )


def train_and_evaluate_model(
    model_name: str,
    data: PreparedData,
    args: argparse.Namespace,
    out_dir: Path,
) -> Dict[str, Any]:
    model_name = model_name.lower()
    model = build_model(model_name, args, data.y_train)

    print(f"\n[MODEL] {model_name}: fitting...")
    started = time.time()
    model.fit(data.x_train, data.y_train)
    fit_seconds = time.time() - started

    val_prob = predict_prob1(model, data.x_val)
    test_prob = predict_prob1(model, data.x_test)

    threshold, val_best, threshold_table = threshold_search(
        data.y_val,
        val_prob,
        steps=args.threshold_steps,
        metric=args.threshold_metric,
    )
    test_metrics = metrics_at_threshold(data.y_test, test_prob, threshold)
    oracle_threshold, oracle_metrics, _ = threshold_search(
        data.y_test,
        test_prob,
        steps=args.threshold_steps,
        metric=args.threshold_metric,
    )

    threshold_table.to_csv(out_dir / f"{model_name}_val_threshold_search.csv", index=False)
    save_feature_importance(model_name, model, data.feature_cols, out_dir)

    if args.save_predictions:
        pred = (test_prob >= threshold).astype(np.int64)
        pred_df = pd.DataFrame(
            {
                "row_index": data.test_indices,
                "flow_id": data.test_flow_ids if data.test_flow_ids is not None else "",
                "y_true": data.y_test,
                "prob_label1": test_prob,
                "y_pred": pred,
            }
        )
        pred_df.to_csv(out_dir / f"{model_name}_test_predictions.csv", index=False)

    result = {
        "model": model_name,
        "fit_seconds": fit_seconds,
        "selected_threshold_from_val": threshold,
        "val_best_metrics": val_best,
        "test_metrics": test_metrics,
        "test_oracle_best_threshold": oracle_threshold,
        "test_oracle_best_metrics_diagnostic_only": oracle_metrics,
    }
    save_json(result, out_dir / f"{model_name}_metrics.json")

    print(
        "[RESULT] {} | P1={:.4f} | R1={:.4f} | F1_1={:.4f} | "
        "PR_AUC={:.4f} | AUC={:.4f} | threshold={:.4f} | fit={:.1f}s".format(
            model_name,
            test_metrics["precision_label1"],
            test_metrics["recall_label1"],
            test_metrics["f1_label1"],
            test_metrics["pr_auc"],
            test_metrics["auc"],
            threshold,
            fit_seconds,
        )
    )
    return result


def save_summary_outputs(
    data: PreparedData,
    dataset_report: Dict[str, Any],
    results: List[Dict[str, Any]],
    args: argparse.Namespace,
    out_dir: Path,
) -> None:
    write_lines(data.feature_cols, out_dir / "feature_columns.txt")
    save_json(data.dropped_feature_cols, out_dir / "dropped_feature_columns.json")

    report = dict(dataset_report)
    report["args"] = vars(args)
    report["num_final_features"] = len(data.feature_cols)
    report["log1p_columns_used"] = data.log1p_cols
    report["split_sizes"] = {
        "train": int(len(data.y_train)),
        "val": int(len(data.y_val)),
        "test": int(len(data.y_test)),
    }
    report["split_binary_counts"] = {
        "train": dict(pd.Series(data.y_train).value_counts().sort_index()),
        "val": dict(pd.Series(data.y_val).value_counts().sort_index()),
        "test": dict(pd.Series(data.y_test).value_counts().sort_index()),
    }
    report["dropped_feature_columns"] = data.dropped_feature_cols
    save_json(report, out_dir / "dataset_report.json")

    rows = []
    for result in results:
        test = result["test_metrics"]
        oracle = result["test_oracle_best_metrics_diagnostic_only"]
        rows.append(
            {
                "model": result["model"],
                "fit_seconds": result["fit_seconds"],
                "threshold": result["selected_threshold_from_val"],
                "accuracy": test["accuracy"],
                "precision_label1": test["precision_label1"],
                "recall_label1": test["recall_label1"],
                "f1_label1": test["f1_label1"],
                "macro_f1": test["macro_f1"],
                "weighted_f1": test["weighted_f1"],
                "auc": test["auc"],
                "pr_auc": test["pr_auc"],
                "loss": test["loss"],
                "confusion_matrix": json.dumps(test["confusion_matrix"]),
                "oracle_threshold_test_only": result["test_oracle_best_threshold"],
                "oracle_f1_label1_test_only": oracle["f1_label1"],
            }
        )
    summary = pd.DataFrame(rows).sort_values("f1_label1", ascending=False)
    summary.to_csv(out_dir / "binary_baseline_results.csv", index=False)
    save_json({"dataset_report": report, "results": results}, out_dir / "binary_baseline_summary.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Binary baselines for custom Suricata flow CSV.")
    parser.add_argument("--flow_csv", default=DEFAULT_FLOW_CSV)
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--label_col", default="label")
    parser.add_argument("--flow_id_col", default="flow_id")
    parser.add_argument("--time_col", default="flow_start_timestamp_us")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--seed", type=int, default=130)

    parser.add_argument("--split", choices=["random", "chronological"], default="random")
    parser.add_argument("--train_size", type=float, default=0.7)
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--stratify", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--drop_duplicate_flow_ids", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--include_ports", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include_protocol", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--drop_columns", default="", help="Comma-separated extra columns to drop.")
    parser.add_argument("--min_numeric_ratio", type=float, default=0.90)
    parser.add_argument("--impute_strategy", choices=["median", "mean", "most_frequent"], default="median")
    parser.add_argument(
        "--log1p_mode",
        choices=["stage1_selected", "all_nonnegative", "none"],
        default="stage1_selected",
        help="stage1_selected matches the flow log1p list in Stage1 YAML configs.",
    )
    parser.add_argument("--scale", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--robust_q_low", type=float, default=5.0)
    parser.add_argument("--robust_q_high", type=float, default=95.0)
    parser.add_argument("--clip_value", type=float, default=50.0)

    parser.add_argument("--threshold_metric", default="f1_label1")
    parser.add_argument("--threshold_steps", type=int, default=999)
    parser.add_argument("--class_weight", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--n_jobs", type=int, default=-1)

    parser.add_argument("--n_estimators", type=int, default=500)
    parser.add_argument("--max_depth", type=int, default=None)
    parser.add_argument("--min_samples_leaf", type=int, default=1)
    parser.add_argument("--logreg_max_iter", type=int, default=2000)

    parser.add_argument("--xgb_n_estimators", type=int, default=600)
    parser.add_argument("--xgb_max_depth", type=int, default=6)
    parser.add_argument("--xgb_learning_rate", type=float, default=0.05)
    parser.add_argument("--xgb_subsample", type=float, default=0.9)
    parser.add_argument("--xgb_colsample_bytree", type=float, default=0.9)
    parser.add_argument("--xgb_tree_method", default="hist")

    parser.add_argument("--lgbm_n_estimators", type=int, default=800)
    parser.add_argument("--lgbm_max_depth", type=int, default=-1)
    parser.add_argument("--lgbm_learning_rate", type=float, default=0.03)
    parser.add_argument("--lgbm_num_leaves", type=int, default=63)
    parser.add_argument("--lgbm_subsample", type=float, default=0.9)
    parser.add_argument("--lgbm_colsample_bytree", type=float, default=0.9)

    parser.add_argument("--mlp_hidden", type=int, nargs="+", default=[256, 128])
    parser.add_argument("--mlp_alpha", type=float, default=1e-4)
    parser.add_argument("--mlp_batch_size", type=int, default=2048)
    parser.add_argument("--mlp_lr", type=float, default=1e-3)
    parser.add_argument("--mlp_max_iter", type=int, default=100)
    parser.add_argument("--verbose_model", action="store_true")

    parser.add_argument("--save_predictions", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    warnings.filterwarnings(
        "ignore",
        message="X does not have valid feature names.*",
        category=UserWarning,
    )
    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    df, dataset_report = load_flow_csv(args)
    data = prepare_data(df, dataset_report, args)

    results: List[Dict[str, Any]] = []
    for model_name in args.models:
        results.append(train_and_evaluate_model(model_name, data, args, out_dir))

    save_summary_outputs(data, dataset_report, results, args, out_dir)
    print(f"\n[DONE] outputs saved to: {out_dir.resolve()}")
    if results:
        best = max(results, key=lambda r: r["test_metrics"]["f1_label1"])
        m = best["test_metrics"]
        print(
            "[BEST] {} | P1={:.4f} | R1={:.4f} | F1_1={:.4f} | PR_AUC={:.4f} | AUC={:.4f}".format(
                best["model"],
                m["precision_label1"],
                m["recall_label1"],
                m["f1_label1"],
                m["pr_auc"],
                m["auc"],
            )
        )


if __name__ == "__main__":
    main()
