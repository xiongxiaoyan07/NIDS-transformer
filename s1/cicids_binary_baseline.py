#!/usr/bin/env python3
# -*- coding: utf-8 -*-

r"""
Binary baseline for CICIDS-style MachineLearningCSV files.

This script is intentionally standalone. It reads one CICIDS CSV, converts
BENIGN to class 0 and every other label to class 1, cleans numeric features,
selects a validation threshold for class-1 F1, and evaluates on the test set.

Example:
    python s1/cicids_binary_baseline.py ^
      --csv_path C:\Users\XiaoyanXiong\Downloads\MachineLearningCSV\MachineLearningCVE\Wednesday-workingHours.pcap_ISCX.csv ^
      --out_dir s1/results/cicids_wednesday_binary ^
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


IDENTIFIER_COLUMNS = {
    "Flow ID",
    "Src IP",
    "Source IP",
    "Dst IP",
    "Destination IP",
    "Timestamp",
}

PORT_COLUMNS = {
    "Source Port",
    "Src Port",
    "Destination Port",
    "Dst Port",
}

DEFAULT_MODELS = ["logreg", "extratrees", "xgb", "lgbm"]


@dataclass
class PreparedData:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    test_original_labels: np.ndarray
    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray
    feature_cols: List[str]
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


def duplicate_base_name(col: str) -> str:
    if "." in col:
        base, suffix = col.rsplit(".", 1)
        if suffix.isdigit():
            return base
    return col


def drop_duplicate_header_columns(
    df: pd.DataFrame,
    keep: str = "first",
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Pandas renames duplicate CSV headers as name, name.1, name.2, ...
    CICIDS files commonly contain a duplicated Fwd Header Length column.
    """
    if keep != "first":
        raise ValueError("Only keep='first' is currently supported.")

    seen = set()
    keep_cols: List[str] = []
    dropped: List[str] = []
    for col in df.columns:
        base = duplicate_base_name(col)
        if base in seen:
            dropped.append(col)
        else:
            keep_cols.append(col)
            seen.add(base)
    return df.loc[:, keep_cols].copy(), dropped


def replace_inf_like_values(df: pd.DataFrame) -> pd.DataFrame:
    return df.replace(
        {
            "Infinity": np.inf,
            "infinity": np.inf,
            "Inf": np.inf,
            "inf": np.inf,
            "-Infinity": -np.inf,
            "-infinity": -np.inf,
            "-Inf": -np.inf,
            "-inf": -np.inf,
            "NaN": np.nan,
            "nan": np.nan,
            "": np.nan,
        }
    )


def find_label_column(columns: Sequence[str], requested: str) -> str:
    if requested in columns:
        return requested
    stripped_to_actual = {str(c).strip(): c for c in columns}
    if requested.strip() in stripped_to_actual:
        return stripped_to_actual[requested.strip()]
    lower_to_actual = {str(c).strip().lower(): c for c in columns}
    if requested.strip().lower() in lower_to_actual:
        return lower_to_actual[requested.strip().lower()]
    raise ValueError(f"Label column {requested!r} not found. Available tail: {list(columns)[-10:]}")


def load_cicids_csv(args: argparse.Namespace) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    print(f"[INFO] loading CSV: {csv_path}")
    read_kwargs: Dict[str, Any] = {"low_memory": False}
    if args.max_rows is not None and args.max_rows > 0:
        read_kwargs["nrows"] = int(args.max_rows)

    df = pd.read_csv(csv_path, **read_kwargs)
    raw_num_rows, raw_num_columns = df.shape
    df = normalize_columns(df)

    duplicate_dropped: List[str] = []
    if args.drop_duplicate_header_columns:
        df, duplicate_dropped = drop_duplicate_header_columns(df)

    label_col = find_label_column(df.columns, args.label_col)
    df[label_col] = df[label_col].astype(str).str.strip()

    before = len(df)
    df = df[df[label_col].ne("") & df[label_col].str.lower().ne("nan")].copy()
    dropped_empty_labels = before - len(df)

    sampled_rows = 0
    if args.sample_rows is not None and 0 < args.sample_rows < len(df):
        sampled_rows = int(args.sample_rows)
        y_for_sample = binary_labels_from_series(df[label_col], benign_label=args.benign_label)
        sample_parts = []
        for cls in [0, 1]:
            cls_idx = np.flatnonzero(y_for_sample == cls)
            if len(cls_idx) == 0:
                continue
            cls_n = max(1, int(round(sampled_rows * len(cls_idx) / len(df))))
            cls_n = min(cls_n, len(cls_idx))
            sampled_idx = np.random.default_rng(args.seed + cls).choice(
                cls_idx,
                size=cls_n,
                replace=False,
            )
            sample_parts.append(df.iloc[sampled_idx])
        df = (
            pd.concat(sample_parts, axis=0)
            .sample(frac=1.0, random_state=args.seed)
            .reset_index(drop=True)
        )

    label_counts = df[label_col].value_counts(dropna=False).to_dict()
    y = binary_labels_from_series(df[label_col], benign_label=args.benign_label)
    binary_counts = pd.Series(y).value_counts().sort_index().to_dict()

    report = {
        "csv_path": str(csv_path),
        "file_size_mb": csv_path.stat().st_size / 1024 / 1024,
        "raw_num_rows": int(raw_num_rows),
        "raw_num_columns": int(raw_num_columns),
        "num_rows": int(len(df)),
        "num_columns_after_duplicate_drop": int(len(df.columns)),
        "label_column": label_col,
        "benign_label": args.benign_label,
        "label_counts": label_counts,
        "binary_counts": binary_counts,
        "attack_ratio": float(np.mean(y)) if len(y) else 0.0,
        "dropped_duplicate_header_columns": duplicate_dropped,
        "dropped_empty_label_rows": int(dropped_empty_labels),
        "max_rows_read": int(args.max_rows) if args.max_rows is not None else None,
        "sampled_rows_after_load": sampled_rows or None,
    }
    print(
        "[DATA] rows={:,}, cols={}, binary_counts={}, attack_ratio={:.6f}".format(
            len(df), len(df.columns), binary_counts, report["attack_ratio"]
        )
    )
    return df, report


def binary_labels_from_series(labels: pd.Series, benign_label: str) -> np.ndarray:
    normalized = labels.astype(str).str.strip().str.upper()
    benign = str(benign_label).strip().upper()
    return (normalized != benign).astype(np.int64).to_numpy()


def choose_feature_columns(
    df: pd.DataFrame,
    label_col: str,
    args: argparse.Namespace,
) -> Tuple[List[str], Dict[str, Any]]:
    excluded = {label_col}
    excluded.update(IDENTIFIER_COLUMNS)
    if not args.include_ports:
        excluded.update(PORT_COLUMNS)

    user_drop = [x.strip() for x in args.drop_columns.split(",") if x.strip()]
    excluded.update(user_drop)

    missing_user_drop = [col for col in user_drop if col not in df.columns]
    candidate_cols = [c for c in df.columns if c not in excluded]

    dropped = {
        "identifier_or_label_or_port": [c for c in df.columns if c in excluded],
        "requested_drop_not_found": missing_user_drop,
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


def split_indices(
    df: pd.DataFrame,
    y: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if abs(args.train_size + args.val_size + args.test_size - 1.0) > 1e-6:
        raise ValueError("train_size + val_size + test_size must equal 1.0")

    indices = np.arange(len(df), dtype=np.int64)
    if args.split == "chronological":
        if args.timestamp_col not in df.columns:
            raise ValueError(
                f"Chronological split needs --timestamp_col {args.timestamp_col!r}, "
                "but that column is not present."
            )
        ts = pd.to_datetime(df[args.timestamp_col], errors="coerce")
        if ts.isna().all():
            ts = pd.to_numeric(df[args.timestamp_col], errors="coerce")
        order = np.argsort(ts.to_numpy(), kind="mergesort")
        indices = indices[order]

        n = len(indices)
        b1 = int(round(n * args.train_size))
        b2 = int(round(n * (args.train_size + args.val_size)))
        b1 = min(max(b1, 1), n - 2)
        b2 = min(max(b2, b1 + 1), n - 1)
        train_idx = indices[:b1]
        val_idx = indices[b1:b2]
        test_idx = indices[b2:]
        return train_idx, val_idx, test_idx

    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=args.test_size,
        random_state=args.seed,
        shuffle=True,
        stratify=y if args.stratify else None,
    )
    val_fraction_of_train_val = args.val_size / (args.train_size + args.val_size)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_fraction_of_train_val,
        random_state=args.seed,
        shuffle=True,
        stratify=y[train_val_idx] if args.stratify else None,
    )
    return train_idx, val_idx, test_idx


def numeric_frame(df: pd.DataFrame, feature_cols: Sequence[str]) -> pd.DataFrame:
    x = replace_inf_like_values(df.loc[:, feature_cols])
    x = x.apply(pd.to_numeric, errors="coerce")
    x = x.replace([np.inf, -np.inf], np.nan)
    return x


def fit_transform_features(
    df: pd.DataFrame,
    feature_cols: List[str],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], Dict[str, Any], SimpleImputer, Optional[RobustScaler]]:
    x_raw = replace_inf_like_values(df.loc[:, feature_cols])
    x_numeric = x_raw.apply(pd.to_numeric, errors="coerce")
    inf_mask = np.isinf(x_numeric.to_numpy(dtype=float, copy=False))
    inf_like_count = pd.Series(inf_mask.sum(axis=0), index=x_numeric.columns).sort_values(ascending=False)
    x_all = x_numeric.replace([np.inf, -np.inf], np.nan)
    missing_count = x_all.isna().sum().sort_values(ascending=False)

    x_train_raw = x_all.iloc[train_idx].copy()
    dropped_all_nan: List[str] = []
    dropped_constant: List[str] = []
    keep_cols: List[str] = []

    for col in feature_cols:
        train_col = x_train_raw[col]
        finite_non_na = train_col.dropna()
        if finite_non_na.empty:
            dropped_all_nan.append(col)
            continue
        if finite_non_na.nunique(dropna=True) <= 1:
            dropped_constant.append(col)
            continue
        keep_cols.append(col)

    if not keep_cols:
        raise ValueError("No usable feature columns after train-set all-NaN/constant filtering.")

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
            str(k): int(v)
            for k, v in missing_count[missing_count > 0].head(30).to_dict().items()
        },
        "top_inf_like_columns_full": {
            str(k): int(v)
            for k, v in inf_like_count[inf_like_count > 0].head(30).to_dict().items()
        },
    }

    print(
        "[DATA] final features={}, dropped_all_nan={}, dropped_constant={}".format(
            len(keep_cols), len(dropped_all_nan), len(dropped_constant)
        )
    )

    return (
        x_train.astype(np.float32),
        x_val.astype(np.float32),
        x_test.astype(np.float32),
        keep_cols,
        dropped,
        imputer,
        scaler,
    )


def prepare_data(df: pd.DataFrame, report: Dict[str, Any], args: argparse.Namespace) -> PreparedData:
    label_col = report["label_column"]
    y = binary_labels_from_series(df[label_col], benign_label=args.benign_label)
    original_labels = df[label_col].astype(str).to_numpy()

    feature_candidates, dropped_from_selection = choose_feature_columns(df, label_col, args)
    train_idx, val_idx, test_idx = split_indices(df, y, args)

    transformed = fit_transform_features(
        df=df,
        feature_cols=feature_candidates,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        args=args,
    )
    x_train, x_val, x_test, final_cols, dropped_from_transform, imputer, scaler = transformed

    dropped_feature_cols: Dict[str, Any] = {
        **dropped_from_selection,
        **{k: v for k, v in dropped_from_transform.items() if isinstance(v, list)},
    }
    dropped_feature_cols["top_missing_columns_full"] = dropped_from_transform[
        "top_missing_columns_full"
    ]
    dropped_feature_cols["top_inf_like_columns_full"] = dropped_from_transform[
        "top_inf_like_columns_full"
    ]

    print_split_report("train", y[train_idx])
    print_split_report("val", y[val_idx])
    print_split_report("test", y[test_idx])

    return PreparedData(
        x_train=x_train,
        y_train=y[train_idx],
        x_val=x_val,
        y_val=y[val_idx],
        x_test=x_test,
        y_test=y[test_idx],
        test_original_labels=original_labels[test_idx],
        train_indices=train_idx,
        val_indices=val_idx,
        test_indices=test_idx,
        feature_cols=final_cols,
        dropped_feature_cols=dropped_feature_cols,
        imputer=imputer,
        scaler=scaler,
    )


def print_split_report(name: str, y: np.ndarray) -> None:
    counts = np.bincount(y.astype(np.int64), minlength=2)
    ratio = counts[1] / max(1, counts.sum())
    print(f"[SPLIT] {name}: n={len(y):,}, class0={counts[0]:,}, class1={counts[1]:,}, pos_ratio={ratio:.6f}")


def safe_binary_log_loss(y_true: np.ndarray, prob1: np.ndarray) -> float:
    prob1 = np.asarray(prob1, dtype=np.float64)
    prob1 = np.clip(prob1, 1e-7, 1.0 - 1e-7)
    y_score = np.stack([1.0 - prob1, prob1], axis=1)
    return float(log_loss(y_true, y_score, labels=[0, 1]))


def metrics_at_threshold(
    y_true: np.ndarray,
    prob1: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    prob1 = np.asarray(prob1, dtype=np.float64)
    y_pred = (prob1 >= threshold).astype(np.int64)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    precision_macro, recall_macro, _, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, _, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    per_class_f1_arr = f1_score(y_true, y_pred, average=None, labels=[0, 1], zero_division=0)
    per_class_f1 = {str(i): float(v) for i, v in enumerate(per_class_f1_arr)}

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
        "per_class_f1": per_class_f1,
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
    thresholds = np.linspace(0.001, 0.999, int(steps))
    rows: List[Dict[str, Any]] = []
    for th in thresholds:
        m = metrics_at_threshold(y_true, prob1, float(th))
        rows.append(
            {
                "threshold": float(th),
                "precision_label1": m["precision_label1"],
                "recall_label1": m["recall_label1"],
                "f1_label1": m["f1_label1"],
                "macro_f1": m["macro_f1"],
                "weighted_f1": m["weighted_f1"],
                "accuracy": m["accuracy"],
                "auc": m["auc"],
                "pr_auc": m["pr_auc"],
            }
        )

    table = pd.DataFrame(rows)
    if metric not in table.columns:
        raise ValueError(f"Unsupported threshold metric {metric!r}. Available: {list(table.columns)}")
    best_idx = int(table[metric].idxmax())
    best_threshold = float(table.loc[best_idx, "threshold"])
    best_metrics = metrics_at_threshold(y_true, prob1, best_threshold)
    return best_threshold, best_metrics, table


def class_weight_scale_pos_weight(y_train: np.ndarray) -> float:
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
            scale_pos_weight=class_weight_scale_pos_weight(y_train) if args.class_weight else 1.0,
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
    pred = model.predict(x)
    return pred.astype(np.float64)


def save_feature_importance(
    model_name: str,
    model: Any,
    feature_cols: List[str],
    out_dir: Path,
) -> None:
    rows: Optional[pd.DataFrame] = None
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=np.float64)
        rows = pd.DataFrame(
            {
                "feature": feature_cols,
                "importance": values,
                "abs_importance": np.abs(values),
            }
        )
    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_)
        if coef.ndim == 2:
            coef = coef[0]
        rows = pd.DataFrame(
            {
                "feature": feature_cols,
                "coefficient": coef,
                "abs_importance": np.abs(coef),
            }
        )

    if rows is None:
        return
    rows = rows.sort_values("abs_importance", ascending=False)
    rows.to_csv(out_dir / f"{model_name}_feature_importance.csv", index=False)


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
    oracle_threshold, oracle_test_metrics, _ = threshold_search(
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
                "original_label": data.test_original_labels,
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
        "test_oracle_best_metrics_diagnostic_only": oracle_test_metrics,
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
    out_dir: Path,
) -> None:
    write_lines(data.feature_cols, out_dir / "feature_columns.txt")
    save_json(data.dropped_feature_cols, out_dir / "dropped_feature_columns.json")

    dataset_report = dict(dataset_report)
    dataset_report["num_final_features"] = len(data.feature_cols)
    dataset_report["split_sizes"] = {
        "train": int(len(data.y_train)),
        "val": int(len(data.y_val)),
        "test": int(len(data.y_test)),
    }
    dataset_report["split_binary_counts"] = {
        "train": dict(pd.Series(data.y_train).value_counts().sort_index()),
        "val": dict(pd.Series(data.y_val).value_counts().sort_index()),
        "test": dict(pd.Series(data.y_test).value_counts().sort_index()),
    }
    dataset_report["dropped_feature_columns"] = data.dropped_feature_cols
    save_json(dataset_report, out_dir / "dataset_report.json")

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
    summary_df = pd.DataFrame(rows).sort_values("f1_label1", ascending=False)
    summary_df.to_csv(out_dir / "binary_baseline_results.csv", index=False)

    save_json(
        {
            "dataset_report": dataset_report,
            "results": results,
        },
        out_dir / "binary_baseline_summary.json",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Binary baseline for CICIDS MachineLearningCSV files."
    )
    parser.add_argument("--csv_path", required=True, help="Path to CICIDS CSV file.")
    parser.add_argument("--out_dir", required=True, help="Directory for outputs.")
    parser.add_argument("--label_col", default="Label")
    parser.add_argument("--benign_label", default="BENIGN")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--seed", type=int, default=130)
    parser.add_argument("--max_rows", type=int, default=None, help="Optional smoke-test row limit.")
    parser.add_argument("--sample_rows", type=int, default=None, help="Optional stratified smoke-test sample after loading.")

    parser.add_argument("--split", choices=["random", "chronological"], default="random")
    parser.add_argument("--timestamp_col", default="Timestamp")
    parser.add_argument("--train_size", type=float, default=0.7)
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--stratify", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--include_ports", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--drop_columns", default="", help="Comma-separated extra columns to drop.")
    parser.add_argument("--drop_duplicate_header_columns", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min_numeric_ratio", type=float, default=0.90)
    parser.add_argument("--impute_strategy", choices=["median", "mean", "most_frequent"], default="median")
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

    df, dataset_report = load_cicids_csv(args)
    data = prepare_data(df, dataset_report, args)

    results: List[Dict[str, Any]] = []
    for model_name in args.models:
        result = train_and_evaluate_model(model_name, data, args, out_dir)
        results.append(result)

    save_summary_outputs(data, dataset_report, results, out_dir)
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
