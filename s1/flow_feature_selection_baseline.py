from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler


IDENTIFIER_COLUMNS = {
    "flow_id",
    "label",
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


@dataclass
class NumericPreprocessor:
    feature_cols: List[str]
    imputer: SimpleImputer
    scaler: Optional[Any]
    clip_bounds: Dict[str, Tuple[float, float]]
    log1p_cols: List[str]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def save_json(obj: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(obj), f, indent=2, ensure_ascii=False)


def parse_top_k(values: Sequence[str]) -> List[Any]:
    parsed: List[Any] = []
    for value in values:
        if str(value).lower() == "all":
            parsed.append("all")
        else:
            parsed.append(int(value))
    return parsed


def split_csv_arg(value: Optional[str]) -> List[str]:
    if value is None or str(value).strip() == "":
        return []
    return [x.strip() for x in str(value).split(",") if x.strip()]


def safe_binary_log_loss(y_true: np.ndarray, prob1: np.ndarray) -> float:
    prob1 = np.asarray(prob1, dtype=np.float64)
    prob1 = np.clip(prob1, 1e-7, 1.0 - 1e-7)
    return float(log_loss(y_true, np.stack([1.0 - prob1, prob1], axis=1), labels=[0, 1]))


def metrics_at_threshold(
    y_true: np.ndarray,
    prob1: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    prob1 = np.asarray(prob1, dtype=np.float64)
    pred = (prob1 >= threshold).astype(np.int64)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])

    try:
        auc = float(roc_auc_score(y_true, prob1))
    except Exception:
        auc = float("nan")

    try:
        pr_auc = float(average_precision_score(y_true, prob1))
    except Exception:
        pr_auc = float("nan")

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision_label1": float(
            precision_score(y_true, pred, pos_label=1, zero_division=0)
        ),
        "recall_label1": float(
            recall_score(y_true, pred, pos_label=1, zero_division=0)
        ),
        "f1_label1": float(f1_score(y_true, pred, pos_label=1, zero_division=0)),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(y_true, pred, average="weighted", zero_division=0)
        ),
        "auc": auc,
        "pr_auc": pr_auc,
        "loss": safe_binary_log_loss(y_true, prob1),
        "confusion_matrix": cm.tolist(),
    }


def find_best_threshold(
    y_true: np.ndarray,
    prob1: np.ndarray,
    steps: int,
) -> Tuple[float, Dict[str, Any]]:
    thresholds = np.linspace(0.001, 0.999, int(steps))
    rows = [metrics_at_threshold(y_true, prob1, float(th)) for th in thresholds]
    best = max(rows, key=lambda row: row["f1_label1"])
    return float(best["threshold"]), best


def load_flow_csv(args: argparse.Namespace) -> pd.DataFrame:
    print(f"[INFO] loading flow CSV: {args.flow_csv}")
    df = pd.read_csv(args.flow_csv, low_memory=False)

    required = [args.flow_id_col, args.label_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in flow CSV: {missing}")

    df[args.flow_id_col] = pd.to_numeric(df[args.flow_id_col], errors="coerce")
    df[args.label_col] = pd.to_numeric(df[args.label_col], errors="coerce")
    df = df.dropna(subset=[args.flow_id_col, args.label_col]).copy()
    df[args.flow_id_col] = df[args.flow_id_col].astype(np.int64)
    df[args.label_col] = df[args.label_col].astype(np.int64)

    before = len(df)
    df = df.drop_duplicates(args.flow_id_col, keep="first").reset_index(drop=True)
    dropped = before - len(df)
    if dropped > 0:
        print(f"[WARN] dropped duplicated flow_id rows: {dropped}")

    df = df[df[args.label_col].isin([0, 1])].copy()
    df = df.reset_index(drop=True)
    print(f"[INFO] loaded flows={len(df):,}, columns={len(df.columns)}")

    counts = df[args.label_col].value_counts().sort_index().to_dict()
    print(f"[INFO] label counts: {counts}")
    return df


def split_flows(
    df: pd.DataFrame,
    args: argparse.Namespace,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if abs(args.train_size + args.val_size + args.test_size - 1.0) > 1e-6:
        raise ValueError("train_size + val_size + test_size must equal 1.0")

    if args.split == "chronological":
        if args.time_col not in df.columns:
            raise ValueError(
                f"Chronological split requires time column {args.time_col!r}"
            )
        split_df = df.copy()
        split_df[args.time_col] = pd.to_numeric(split_df[args.time_col], errors="coerce")
        split_df = split_df.dropna(subset=[args.time_col]).copy()
        split_df = split_df.sort_values(args.time_col, kind="mergesort").reset_index(drop=True)

        n = len(split_df)
        b1 = int(round(n * args.train_size))
        b2 = int(round(n * (args.train_size + args.val_size)))
        b1 = min(max(b1, 1), n - 2)
        b2 = min(max(b2, b1 + 1), n - 1)

        train_df = split_df.iloc[:b1].copy()
        val_df = split_df.iloc[b1:b2].copy()
        test_df = split_df.iloc[b2:].copy()
    else:
        y = df[args.label_col].to_numpy(dtype=np.int64)
        stratify_y = y if args.stratify else None
        train_val_df, test_df = train_test_split(
            df,
            test_size=args.test_size,
            random_state=args.seed,
            shuffle=True,
            stratify=stratify_y,
        )
        relative_val = args.val_size / (args.train_size + args.val_size)
        train_val_y = train_val_df[args.label_col].to_numpy(dtype=np.int64)
        stratify_train_val = train_val_y if args.stratify else None
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=relative_val,
            random_state=args.seed,
            shuffle=True,
            stratify=stratify_train_val,
        )

    for name, part in [("train", train_df), ("val", val_df), ("test", test_df)]:
        counts = part[args.label_col].value_counts().sort_index().to_dict()
        pos_rate = float(part[args.label_col].mean()) if len(part) else 0.0
        print(
            f"[SPLIT] {name}: n={len(part):,}, label_counts={counts}, "
            f"pos_rate={pos_rate:.6f}"
        )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def infer_feature_columns(df: pd.DataFrame, args: argparse.Namespace) -> List[str]:
    user_features = split_csv_arg(args.feature_columns)
    if user_features:
        missing = [col for col in user_features if col not in df.columns]
        if missing:
            raise ValueError(f"Requested feature columns are missing: {missing}")
        return user_features

    exclude = set(IDENTIFIER_COLUMNS)
    exclude.add(args.flow_id_col)
    exclude.add(args.label_col)

    if not args.include_ports:
        exclude.update(PORT_COLUMNS)

    if not args.include_protocol:
        exclude.add("protocol")

    exclude.update(split_csv_arg(args.exclude_columns))

    candidates: List[str] = []
    for col in df.columns:
        if col in exclude:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        numeric_ratio = float(values.notna().mean())
        if numeric_ratio >= args.min_numeric_ratio:
            candidates.append(col)

    if not candidates:
        raise ValueError("No numeric feature columns found after exclusions.")

    print(f"[INFO] feature candidates={len(candidates)}")
    return candidates


def numeric_frame(df: pd.DataFrame, feature_cols: Sequence[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for col in feature_cols:
        out[col] = pd.to_numeric(df[col], errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def choose_log1p_cols(
    train_raw: pd.DataFrame,
    *,
    enabled: bool,
    max_unique_for_binary: int,
) -> List[str]:
    if not enabled:
        return []

    cols: List[str] = []
    for col in train_raw.columns:
        values = train_raw[col].dropna()
        if values.empty:
            continue
        if values.min() < 0:
            continue
        if values.nunique(dropna=True) <= max_unique_for_binary:
            continue
        q95 = values.quantile(0.95)
        if q95 > 10:
            cols.append(col)
    return cols


def fit_numeric_preprocessor(
    train_df: pd.DataFrame,
    feature_cols: Sequence[str],
    args: argparse.Namespace,
) -> NumericPreprocessor:
    raw = numeric_frame(train_df, feature_cols)
    log1p_cols = choose_log1p_cols(
        raw,
        enabled=args.auto_log1p,
        max_unique_for_binary=args.binary_unique_threshold,
    )

    work = raw.copy()
    for col in log1p_cols:
        work[col] = np.log1p(work[col].clip(lower=0))

    clip_bounds: Dict[str, Tuple[float, float]] = {}
    if args.clip:
        for col in work.columns:
            values = work[col].dropna()
            if values.empty:
                clip_bounds[col] = (0.0, 0.0)
                continue
            lo = float(values.quantile(args.clip_lower_quantile))
            hi = float(values.quantile(args.clip_upper_quantile))
            if lo > hi:
                lo, hi = hi, lo
            clip_bounds[col] = (lo, hi)
            work[col] = work[col].clip(lo, hi)

    imputer = SimpleImputer(strategy="median")
    imputer.fit(work)

    if args.scaler == "robust":
        scaler: Optional[Any] = RobustScaler()
    elif args.scaler == "standard":
        scaler = StandardScaler()
    elif args.scaler == "none":
        scaler = None
    else:
        raise ValueError(f"Unsupported scaler: {args.scaler}")

    filled = imputer.transform(work)
    if scaler is not None:
        scaler.fit(filled)

    return NumericPreprocessor(
        feature_cols=list(feature_cols),
        imputer=imputer,
        scaler=scaler,
        clip_bounds=clip_bounds,
        log1p_cols=log1p_cols,
    )


def transform_numeric(
    df: pd.DataFrame,
    preprocessor: NumericPreprocessor,
) -> np.ndarray:
    work = numeric_frame(df, preprocessor.feature_cols)

    for col in preprocessor.log1p_cols:
        work[col] = np.log1p(work[col].clip(lower=0))

    for col, (lo, hi) in preprocessor.clip_bounds.items():
        if col in work.columns:
            work[col] = work[col].clip(lo, hi)

    x = preprocessor.imputer.transform(work)
    if preprocessor.scaler is not None:
        x = preprocessor.scaler.transform(x)
    return x.astype(np.float32)


def compute_feature_ranking(
    train_df: pd.DataFrame,
    y_train: np.ndarray,
    feature_cols: Sequence[str],
    args: argparse.Namespace,
) -> pd.DataFrame:
    print("[INFO] computing feature ranking...")

    raw = numeric_frame(train_df, feature_cols)
    imputed = raw.copy()
    for col in imputed.columns:
        median = imputed[col].median()
        imputed[col] = imputed[col].fillna(0.0 if pd.isna(median) else median)

    rows: List[Dict[str, Any]] = []
    for col in feature_cols:
        values = imputed[col].to_numpy(dtype=np.float64)
        unique_count = int(pd.Series(values).nunique(dropna=True))
        non_null_rate = float(raw[col].notna().mean())
        std = float(np.nanstd(values))

        if unique_count <= 1 or std == 0:
            auc = float("nan")
            signed_ap = float("nan")
            uni_importance = 0.0
            direction = 0
        else:
            try:
                auc = float(roc_auc_score(y_train, values))
                direction = 1 if auc >= 0.5 else -1
                uni_importance = float(abs(auc - 0.5) * 2.0)
                signed_values = values if direction == 1 else -values
                signed_ap = float(average_precision_score(y_train, signed_values))
            except Exception:
                auc = float("nan")
                signed_ap = float("nan")
                uni_importance = 0.0
                direction = 0

        rows.append(
            {
                "feature": col,
                "train_non_null_rate": non_null_rate,
                "train_unique_count": unique_count,
                "train_std": std,
                "univariate_auc": auc,
                "univariate_direction": direction,
                "univariate_importance": uni_importance,
                "univariate_signed_pr_auc": signed_ap,
            }
        )

    ranking = pd.DataFrame(rows)

    pre = fit_numeric_preprocessor(train_df, feature_cols, args)
    x_rank = transform_numeric(train_df, pre)

    try:
        mi = mutual_info_classif(
            x_rank,
            y_train,
            random_state=args.seed,
            n_neighbors=3,
            discrete_features=False,
        )
        ranking["mutual_info"] = mi.astype(float)
    except Exception as exc:
        print(f"[WARN] mutual_info_classif failed: {exc}")
        ranking["mutual_info"] = 0.0

    try:
        et = ExtraTreesClassifier(
            n_estimators=args.importance_trees,
            random_state=args.seed,
            class_weight="balanced",
            n_jobs=args.n_jobs,
            min_samples_leaf=args.importance_min_samples_leaf,
        )
        et.fit(x_rank, y_train)
        ranking["extratrees_importance"] = et.feature_importances_.astype(float)
    except Exception as exc:
        print(f"[WARN] ExtraTrees importance failed: {exc}")
        ranking["extratrees_importance"] = 0.0

    score_cols = [
        "univariate_importance",
        "mutual_info",
        "extratrees_importance",
    ]
    norm_parts = []
    for col in score_cols:
        values = ranking[col].fillna(0.0).to_numpy(dtype=np.float64)
        max_value = float(np.max(values)) if len(values) else 0.0
        if max_value > 0:
            norm_parts.append(values / max_value)
        else:
            norm_parts.append(np.zeros_like(values))

    ranking["combined_score"] = np.vstack(norm_parts).mean(axis=0)
    ranking = ranking.sort_values(
        ["combined_score", "extratrees_importance", "univariate_importance"],
        ascending=False,
    ).reset_index(drop=True)
    ranking["combined_rank"] = np.arange(1, len(ranking) + 1)
    return ranking


def select_top_features(
    ranking: pd.DataFrame,
    train_df: pd.DataFrame,
    k: Any,
    args: argparse.Namespace,
) -> List[str]:
    ordered = ranking["feature"].tolist()
    if k == "all":
        return ordered

    target_k = int(k)
    if args.corr_threshold <= 0 or args.corr_threshold >= 1:
        return ordered[:target_k]

    raw = numeric_frame(train_df, ordered)
    selected: List[str] = []

    for feature in ordered:
        if len(selected) >= target_k:
            break
        if not selected:
            selected.append(feature)
            continue

        current = raw[feature]
        too_correlated = False
        for kept in selected:
            corr = current.corr(raw[kept], method="spearman")
            if pd.notna(corr) and abs(float(corr)) >= args.corr_threshold:
                too_correlated = True
                break
        if not too_correlated:
            selected.append(feature)

    if len(selected) < target_k:
        for feature in ordered:
            if len(selected) >= target_k:
                break
            if feature not in selected:
                selected.append(feature)

    return selected


def get_prob1(model: Any, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1].astype(np.float64)
        return proba.reshape(-1).astype(np.float64)

    if hasattr(model, "decision_function"):
        score = model.decision_function(x)
        return (1.0 / (1.0 + np.exp(-score))).astype(np.float64)

    pred = model.predict(x)
    return np.asarray(pred, dtype=np.float64)


def build_model(name: str, y_train: np.ndarray, args: argparse.Namespace) -> Optional[Any]:
    counts = np.bincount(y_train.astype(int), minlength=2)
    n0, n1 = int(counts[0]), int(counts[1])
    scale_pos_weight = n0 / max(n1, 1)

    if name == "logreg":
        return LogisticRegression(
            class_weight="balanced",
            max_iter=args.logreg_max_iter,
            solver="lbfgs",
            n_jobs=args.n_jobs,
            random_state=args.seed,
        )

    if name == "extratrees":
        return ExtraTreesClassifier(
            n_estimators=args.n_estimators,
            random_state=args.seed,
            class_weight="balanced",
            n_jobs=args.n_jobs,
            min_samples_leaf=args.min_samples_leaf,
        )

    if name == "rf":
        return RandomForestClassifier(
            n_estimators=args.n_estimators,
            random_state=args.seed,
            class_weight="balanced",
            n_jobs=args.n_jobs,
            min_samples_leaf=args.min_samples_leaf,
        )

    if name == "xgb":
        try:
            import xgboost as xgb
        except Exception as exc:
            print(f"[SKIP] xgboost is not available: {exc}")
            return None

        return xgb.XGBClassifier(
            n_estimators=args.xgb_n_estimators,
            max_depth=args.xgb_max_depth,
            learning_rate=args.xgb_lr,
            subsample=args.xgb_subsample,
            colsample_bytree=args.xgb_colsample_bytree,
            min_child_weight=args.xgb_min_child_weight,
            reg_lambda=args.xgb_reg_lambda,
            reg_alpha=args.xgb_reg_alpha,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method=args.xgb_tree_method,
            scale_pos_weight=scale_pos_weight,
            n_jobs=args.n_jobs,
            random_state=args.seed,
        )

    if name == "lgbm":
        try:
            import lightgbm as lgb
        except Exception as exc:
            print(f"[SKIP] lightgbm is not available: {exc}")
            return None

        return lgb.LGBMClassifier(
            objective="binary",
            n_estimators=args.lgbm_n_estimators,
            learning_rate=args.lgbm_lr,
            num_leaves=args.lgbm_num_leaves,
            max_depth=args.lgbm_max_depth,
            min_child_samples=args.lgbm_min_child_samples,
            subsample=args.lgbm_subsample,
            colsample_bytree=args.lgbm_colsample_bytree,
            reg_lambda=args.lgbm_reg_lambda,
            reg_alpha=args.lgbm_reg_alpha,
            scale_pos_weight=scale_pos_weight,
            random_state=args.seed,
            n_jobs=args.n_jobs,
            verbosity=-1,
        )

    raise ValueError(f"Unknown model: {name}")


def evaluate_feature_set(
    feature_set_name: str,
    selected_features: List[str],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    y_train = train_df[args.label_col].to_numpy(dtype=np.int64)
    y_val = val_df[args.label_col].to_numpy(dtype=np.int64)
    y_test = test_df[args.label_col].to_numpy(dtype=np.int64)

    pre = fit_numeric_preprocessor(train_df, selected_features, args)
    x_train = transform_numeric(train_df, pre)
    x_val = transform_numeric(val_df, pre)
    x_test = transform_numeric(test_df, pre)

    rows: List[Dict[str, Any]] = []

    for model_name in args.models:
        model = build_model(model_name, y_train, args)
        if model is None:
            continue

        print(
            f"[MODEL] feature_set={feature_set_name}, "
            f"num_features={len(selected_features)}, model={model_name}"
        )
        model.fit(x_train, y_train)
        val_prob = get_prob1(model, x_val)
        test_prob = get_prob1(model, x_test)

        threshold, val_metrics = find_best_threshold(
            y_val,
            val_prob,
            steps=args.threshold_steps,
        )
        test_metrics = metrics_at_threshold(y_test, test_prob, threshold)
        _, test_oracle = find_best_threshold(
            y_test,
            test_prob,
            steps=args.threshold_steps,
        )

        row = {
            "feature_set": feature_set_name,
            "num_features": len(selected_features),
            "model": model_name,
            "threshold": threshold,
            "test_precision_label1": test_metrics["precision_label1"],
            "test_recall_label1": test_metrics["recall_label1"],
            "test_f1_label1": test_metrics["f1_label1"],
            "test_macro_f1": test_metrics["macro_f1"],
            "test_pr_auc": test_metrics["pr_auc"],
            "test_auc": test_metrics["auc"],
            "test_loss": test_metrics["loss"],
            "test_confusion_matrix": test_metrics["confusion_matrix"],
            "val_precision_label1": val_metrics["precision_label1"],
            "val_recall_label1": val_metrics["recall_label1"],
            "val_f1_label1": val_metrics["f1_label1"],
            "test_oracle_threshold": test_oracle["threshold"],
            "test_oracle_f1_label1": test_oracle["f1_label1"],
            "selected_features": selected_features,
        }
        rows.append(row)

        print(
            f"  -> TEST P1={row['test_precision_label1']:.4f} "
            f"R1={row['test_recall_label1']:.4f} "
            f"F1_1={row['test_f1_label1']:.4f} "
            f"PR_AUC={row['test_pr_auc']:.4f} "
            f"AUC={row['test_auc']:.4f} "
            f"threshold={row['threshold']:.4f}"
        )

        if args.save_predictions:
            pred_df = pd.DataFrame(
                {
                    args.flow_id_col: test_df[args.flow_id_col].to_numpy(),
                    "y_true": y_test,
                    "prob1": test_prob,
                    "pred": (test_prob >= threshold).astype(np.int64),
                    "model": model_name,
                    "feature_set": feature_set_name,
                }
            )
            pred_path = os.path.join(
                args.out_dir,
                f"predictions_{feature_set_name}_{model_name}.csv",
            )
            pred_df.to_csv(pred_path, index=False)

    return rows


def save_selected_feature_files(
    selected_map: Dict[str, List[str]],
    out_dir: str,
) -> None:
    selected_dir = os.path.join(out_dir, "selected_features")
    ensure_dir(selected_dir)
    for name, features in selected_map.items():
        path = os.path.join(selected_dir, f"{name}.txt")
        with open(path, "w", encoding="utf-8") as f:
            for feature in features:
                f.write(f"{feature}\n")


def main() -> None:
    args = parse_args()
    ensure_dir(args.out_dir)

    df = load_flow_csv(args)
    train_df, val_df, test_df = split_flows(df, args)

    y_train = train_df[args.label_col].to_numpy(dtype=np.int64)

    if args.ranking_csv:
        print(f"[INFO] loading existing ranking: {args.ranking_csv}")
        ranking = pd.read_csv(args.ranking_csv)
        if "feature" not in ranking.columns:
            raise ValueError("--ranking_csv must contain a 'feature' column.")
        feature_cols = [str(x) for x in ranking["feature"].tolist() if str(x) in df.columns]
        ranking = ranking[ranking["feature"].isin(feature_cols)].copy()
        if "combined_score" not in ranking.columns:
            ranking["combined_score"] = np.arange(len(ranking), 0, -1, dtype=float)
        if "combined_rank" not in ranking.columns:
            ranking["combined_rank"] = np.arange(1, len(ranking) + 1)
    else:
        feature_cols = infer_feature_columns(df, args)
        ranking = compute_feature_ranking(train_df, y_train, feature_cols, args)

    ranking_path = os.path.join(args.out_dir, "feature_ranking.csv")
    ranking.to_csv(ranking_path, index=False)

    selected_map: Dict[str, List[str]] = {}
    for k in args.top_k:
        name = "all" if k == "all" else f"top{k}"
        selected_map[name] = select_top_features(ranking, train_df, k, args)

    save_selected_feature_files(selected_map, args.out_dir)

    all_rows: List[Dict[str, Any]] = []
    for name, selected in selected_map.items():
        selected_df = pd.DataFrame(
            {
                "rank_in_set": np.arange(1, len(selected) + 1),
                "feature": selected,
            }
        )
        selected_df.to_csv(
            os.path.join(args.out_dir, f"selected_features_{name}.csv"),
            index=False,
        )
        all_rows.extend(
            evaluate_feature_set(
                name,
                selected,
                train_df,
                val_df,
                test_df,
                args,
            )
        )

    results_df = pd.DataFrame(all_rows)
    results_csv = os.path.join(args.out_dir, "feature_selection_baseline_results.csv")
    if len(results_df) > 0:
        flattened = results_df.copy()
        flattened["selected_features"] = flattened["selected_features"].apply(
            lambda xs: ",".join(xs)
        )
        flattened.to_csv(results_csv, index=False)

    summary_path = os.path.join(args.out_dir, "feature_selection_summary.json")
    save_json(
        {
            "args": vars(args),
            "input_flow_csv": os.path.abspath(args.flow_csv),
            "feature_candidates": feature_cols,
            "top_ranked_features": ranking.head(30).to_dict(orient="records"),
            "results": all_rows,
            "artifacts": {
                "feature_ranking_csv": os.path.abspath(ranking_path),
                "results_csv": os.path.abspath(results_csv),
            },
        },
        summary_path,
    )

    print("\n[TOP 20 FEATURES]")
    cols = [
        "combined_rank",
        "feature",
        "combined_score",
        "univariate_auc",
        "univariate_signed_pr_auc",
        "mutual_info",
        "extratrees_importance",
    ]
    print(ranking[cols].head(20).to_string(index=False))

    print("\n[RESULTS] Test at validation-selected threshold")
    if len(results_df) == 0:
        print("No model results were produced.")
    else:
        view_cols = [
            "feature_set",
            "num_features",
            "model",
            "test_precision_label1",
            "test_recall_label1",
            "test_f1_label1",
            "test_pr_auc",
            "test_auc",
            "threshold",
        ]
        print(
            results_df.sort_values("test_f1_label1", ascending=False)[view_cols]
            .to_string(index=False)
        )

    print(f"\n[INFO] saved ranking: {ranking_path}")
    print(f"[INFO] saved results: {results_csv}")
    print(f"[INFO] saved summary: {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rank raw Suricata flow CSV features and rerun flow-only baselines "
            "with selected top-K feature subsets."
        )
    )

    parser.add_argument("--flow_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--flow_id_col", default="flow_id")
    parser.add_argument("--label_col", default="label")
    parser.add_argument("--time_col", default="flow_start_timestamp_us")

    parser.add_argument(
        "--split",
        choices=["chronological", "random"],
        default="chronological",
    )
    parser.add_argument("--stratify", action="store_true")
    parser.add_argument("--train_size", type=float, default=0.70)
    parser.add_argument("--val_size", type=float, default=0.10)
    parser.add_argument("--test_size", type=float, default=0.20)

    parser.add_argument("--seed", type=int, default=130)
    parser.add_argument("--n_jobs", type=int, default=-1)

    parser.add_argument("--feature_columns", default=None)
    parser.add_argument("--exclude_columns", default=None)
    parser.add_argument(
        "--ranking_csv",
        default=None,
        help="Reuse an existing feature_ranking.csv instead of recomputing ranking.",
    )
    parser.add_argument("--min_numeric_ratio", type=float, default=0.95)
    parser.add_argument("--include_ports", action="store_true")
    parser.add_argument("--include_protocol", action="store_true")

    parser.add_argument("--auto_log1p", action="store_true", default=True)
    parser.add_argument("--no_auto_log1p", action="store_false", dest="auto_log1p")
    parser.add_argument("--binary_unique_threshold", type=int, default=2)
    parser.add_argument("--clip", action="store_true", default=True)
    parser.add_argument("--no_clip", action="store_false", dest="clip")
    parser.add_argument("--clip_lower_quantile", type=float, default=0.001)
    parser.add_argument("--clip_upper_quantile", type=float, default=0.999)
    parser.add_argument(
        "--scaler",
        choices=["robust", "standard", "none"],
        default="robust",
    )

    parser.add_argument(
        "--top_k",
        nargs="+",
        default=["10", "20", "30", "40", "60", "all"],
        help="Feature set sizes to evaluate. Use 'all' for all candidate features.",
    )
    parser.add_argument(
        "--corr_threshold",
        type=float,
        default=1.0,
        help=(
            "Skip highly correlated ranked features when building top-K sets. "
            "Use 1.0 to disable correlation filtering for speed."
        ),
    )
    parser.add_argument("--importance_trees", type=int, default=400)
    parser.add_argument("--importance_min_samples_leaf", type=int, default=2)

    parser.add_argument(
        "--models",
        nargs="+",
        default=["logreg", "extratrees", "xgb", "lgbm"],
        choices=["logreg", "extratrees", "rf", "xgb", "lgbm"],
    )
    parser.add_argument("--threshold_steps", type=int, default=999)
    parser.add_argument("--save_predictions", action="store_true")

    parser.add_argument("--n_estimators", type=int, default=600)
    parser.add_argument("--min_samples_leaf", type=int, default=1)
    parser.add_argument("--logreg_max_iter", type=int, default=2000)

    parser.add_argument("--xgb_n_estimators", type=int, default=800)
    parser.add_argument("--xgb_max_depth", type=int, default=6)
    parser.add_argument("--xgb_lr", type=float, default=0.03)
    parser.add_argument("--xgb_subsample", type=float, default=0.85)
    parser.add_argument("--xgb_colsample_bytree", type=float, default=0.85)
    parser.add_argument("--xgb_min_child_weight", type=float, default=3.0)
    parser.add_argument("--xgb_reg_lambda", type=float, default=2.0)
    parser.add_argument("--xgb_reg_alpha", type=float, default=0.0)
    parser.add_argument("--xgb_tree_method", default="hist")

    parser.add_argument("--lgbm_n_estimators", type=int, default=1000)
    parser.add_argument("--lgbm_lr", type=float, default=0.03)
    parser.add_argument("--lgbm_num_leaves", type=int, default=63)
    parser.add_argument("--lgbm_max_depth", type=int, default=-1)
    parser.add_argument("--lgbm_min_child_samples", type=int, default=30)
    parser.add_argument("--lgbm_subsample", type=float, default=0.85)
    parser.add_argument("--lgbm_colsample_bytree", type=float, default=0.85)
    parser.add_argument("--lgbm_reg_lambda", type=float, default=2.0)
    parser.add_argument("--lgbm_reg_alpha", type=float, default=0.0)

    args = parser.parse_args()
    args.top_k = parse_top_k(args.top_k)
    return args


if __name__ == "__main__":
    main()
