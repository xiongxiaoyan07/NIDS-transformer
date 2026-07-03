from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


def save_json(obj: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(obj), f, indent=2, ensure_ascii=False)


def sanitize_features(x: np.ndarray, clip_value: float = 50.0) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=clip_value, neginf=-clip_value)
    x = np.clip(x, -clip_value, clip_value)
    return x.astype(np.float32)


def load_flow_npz(npz_path: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    data = np.load(npz_path, allow_pickle=True)
    if "flow_feats" not in data:
        raise KeyError(f"{npz_path} does not contain 'flow_feats'.")
    if "labels" not in data:
        raise KeyError(f"{npz_path} does not contain 'labels'.")

    x = sanitize_features(data["flow_feats"])
    y = np.asarray(data["labels"], dtype=np.int64)
    arrays = {k: data[k] for k in data.files}

    if len(x) != len(y):
        raise ValueError(f"Length mismatch in {npz_path}: x={len(x)}, y={len(y)}")

    return x, y, arrays


def maybe_load_preprocessor_feature_names(
    preprocessor_joblib: Optional[str],
) -> Optional[List[str]]:
    if not preprocessor_joblib:
        return None
    if not os.path.exists(preprocessor_joblib):
        raise FileNotFoundError(preprocessor_joblib)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    try:
        import joblib
    except Exception as exc:
        print(f"[WARN] joblib is not available, cannot read preprocessor: {exc}")
        return None

    preprocessor = joblib.load(preprocessor_joblib)

    names: List[str] = []
    names.extend(list(getattr(preprocessor, "flow_num_cols", [])))

    flow_cat_cols = list(getattr(preprocessor, "flow_cat_cols", []))
    flow_ohe = getattr(preprocessor, "flow_ohe", None)
    if flow_cat_cols and flow_ohe is not None:
        try:
            cat_names = flow_ohe.get_feature_names_out(flow_cat_cols)
        except TypeError:
            cat_names = flow_ohe.get_feature_names_out()
        names.extend([str(x) for x in cat_names])

    names.extend(list(getattr(preprocessor, "flow_bin_cols", [])))
    return names


def get_feature_names(
    train_arrays: Dict[str, np.ndarray],
    num_features: int,
    preprocessor_joblib: Optional[str],
) -> List[str]:
    for key in ["flow_feature_names", "feature_names"]:
        if key in train_arrays:
            names = [str(x) for x in train_arrays[key].tolist()]
            if len(names) == num_features:
                return names
            print(
                f"[WARN] {key} length={len(names)} does not match "
                f"flow feature dim={num_features}; ignoring it."
            )

    names = maybe_load_preprocessor_feature_names(preprocessor_joblib)
    if names is not None:
        if len(names) == num_features:
            return names
        print(
            "[WARN] preprocessor feature name count does not match "
            f"flow feature dim: names={len(names)}, dim={num_features}. "
            "Using generated names."
        )

    return [f"flow_feat_{idx:03d}" for idx in range(num_features)]


def minmax01(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    lo = float(np.min(values))
    hi = float(np.max(values))
    if hi <= lo:
        return np.zeros_like(values, dtype=np.float64)
    return (values - lo) / (hi - lo)


def class_ratio(y: np.ndarray) -> float:
    counts = np.bincount(y.astype(int), minlength=2)
    return float(counts[1]) / max(float(counts.sum()), 1.0)


def compute_feature_ranking(
    x_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: Sequence[str],
    *,
    seed: int,
    n_jobs: int,
    xgb_tree_method: str,
    use_xgb: bool,
    use_lgbm: bool,
    use_extratrees: bool,
) -> pd.DataFrame:
    n_features = x_train.shape[1]
    rows = pd.DataFrame(
        {
            "feature_index": np.arange(n_features, dtype=np.int64),
            "feature_name": list(feature_names),
            "variance": np.var(x_train, axis=0),
            "nonzero_rate": np.mean(np.abs(x_train) > 1e-12, axis=0),
            "missing_or_bad_rate_after_sanitize": np.zeros(n_features, dtype=np.float64),
        }
    )

    try:
        f_scores, f_pvalues = f_classif(x_train, y_train)
        rows["f_classif"] = np.nan_to_num(f_scores, nan=0.0, posinf=0.0, neginf=0.0)
        rows["f_classif_pvalue"] = np.nan_to_num(
            f_pvalues, nan=1.0, posinf=1.0, neginf=1.0
        )
    except Exception as exc:
        print(f"[WARN] f_classif failed: {exc}")
        rows["f_classif"] = 0.0
        rows["f_classif_pvalue"] = 1.0

    try:
        mi = mutual_info_classif(
            x_train,
            y_train,
            discrete_features=False,
            random_state=seed,
        )
        rows["mutual_info"] = np.nan_to_num(mi, nan=0.0, posinf=0.0, neginf=0.0)
    except Exception as exc:
        print(f"[WARN] mutual_info_classif failed: {exc}")
        rows["mutual_info"] = 0.0

    counts = np.bincount(y_train.astype(int), minlength=2)
    scale_pos_weight = float(counts[0]) / max(float(counts[1]), 1.0)

    importance_cols: List[str] = ["f_classif", "mutual_info"]

    if use_xgb:
        try:
            import xgboost as xgb

            model = xgb.XGBClassifier(
                n_estimators=600,
                max_depth=5,
                learning_rate=0.03,
                subsample=0.85,
                colsample_bytree=0.85,
                min_child_weight=3.0,
                reg_lambda=2.0,
                reg_alpha=0.0,
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method=xgb_tree_method,
                scale_pos_weight=scale_pos_weight,
                n_jobs=n_jobs,
                random_state=seed,
            )
            model.fit(x_train, y_train)
            rows["xgb_importance"] = np.nan_to_num(
                model.feature_importances_, nan=0.0, posinf=0.0, neginf=0.0
            )
            importance_cols.append("xgb_importance")
        except Exception as exc:
            print(f"[WARN] XGBoost importance skipped: {exc}")
            rows["xgb_importance"] = 0.0

    if use_lgbm:
        try:
            import lightgbm as lgb

            model = lgb.LGBMClassifier(
                objective="binary",
                n_estimators=700,
                learning_rate=0.03,
                num_leaves=63,
                max_depth=-1,
                min_child_samples=30,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=2.0,
                reg_alpha=0.0,
                scale_pos_weight=scale_pos_weight,
                random_state=seed,
                n_jobs=n_jobs,
                verbosity=-1,
            )
            model.fit(x_train, y_train)
            try:
                imp = model.booster_.feature_importance(importance_type="gain")
            except Exception:
                imp = model.feature_importances_
            rows["lgbm_importance"] = np.nan_to_num(
                imp, nan=0.0, posinf=0.0, neginf=0.0
            )
            importance_cols.append("lgbm_importance")
        except Exception as exc:
            print(f"[WARN] LightGBM importance skipped: {exc}")
            rows["lgbm_importance"] = 0.0

    if use_extratrees:
        try:
            model = ExtraTreesClassifier(
                n_estimators=500,
                max_depth=None,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=seed,
                n_jobs=n_jobs,
            )
            model.fit(x_train, y_train)
            rows["extratrees_importance"] = np.nan_to_num(
                model.feature_importances_, nan=0.0, posinf=0.0, neginf=0.0
            )
            importance_cols.append("extratrees_importance")
        except Exception as exc:
            print(f"[WARN] ExtraTrees importance skipped: {exc}")
            rows["extratrees_importance"] = 0.0

    score_parts = [minmax01(rows[col].to_numpy()) for col in importance_cols]
    rows["ensemble_score"] = np.mean(np.vstack(score_parts), axis=0)
    rows["is_near_constant"] = rows["variance"] <= 1e-8

    rows = rows.sort_values(
        ["is_near_constant", "ensemble_score", "mutual_info", "f_classif"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)
    rows["rank"] = np.arange(1, len(rows) + 1, dtype=np.int64)
    return rows


def binary_metrics_at_threshold(
    y_true: np.ndarray,
    prob1: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    pred = (prob1 >= threshold).astype(np.int64)
    result = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision_label1": float(
            precision_score(y_true, pred, pos_label=1, zero_division=0)
        ),
        "recall_label1": float(recall_score(y_true, pred, pos_label=1, zero_division=0)),
        "f1_label1": float(f1_score(y_true, pred, pos_label=1, zero_division=0)),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(y_true, pred, average="weighted", zero_division=0)
        ),
        "confusion_matrix": confusion_matrix(y_true, pred, labels=[0, 1]).tolist(),
    }

    try:
        result["auc"] = float(roc_auc_score(y_true, prob1))
    except Exception:
        result["auc"] = float("nan")
    try:
        result["pr_auc"] = float(average_precision_score(y_true, prob1))
    except Exception:
        result["pr_auc"] = float("nan")

    return result


def threshold_search(
    y_true: np.ndarray,
    prob1: np.ndarray,
    *,
    threshold_min: float,
    threshold_max: float,
    threshold_steps: int,
) -> Dict[str, Any]:
    thresholds = np.linspace(threshold_min, threshold_max, threshold_steps)
    rows = [
        binary_metrics_at_threshold(y_true, prob1, float(th))
        for th in thresholds
    ]
    best = max(rows, key=lambda r: r["f1_label1"])
    return {"best_f1": best}


def fit_predict_model(
    model_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    *,
    seed: int,
    n_jobs: int,
    xgb_tree_method: str,
) -> Tuple[np.ndarray, np.ndarray]:
    counts = np.bincount(y_train.astype(int), minlength=2)
    scale_pos_weight = float(counts[0]) / max(float(counts[1]), 1.0)

    if model_name == "xgb":
        import xgboost as xgb

        model = xgb.XGBClassifier(
            n_estimators=800,
            max_depth=6,
            learning_rate=0.03,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=3.0,
            reg_lambda=2.0,
            reg_alpha=0.0,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method=xgb_tree_method,
            scale_pos_weight=scale_pos_weight,
            n_jobs=n_jobs,
            random_state=seed,
        )
        model.fit(x_train, y_train)
        return model.predict_proba(x_val)[:, 1], model.predict_proba(x_test)[:, 1]

    if model_name == "lgbm":
        import lightgbm as lgb

        model = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=1000,
            learning_rate=0.03,
            num_leaves=63,
            max_depth=-1,
            min_child_samples=30,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=2.0,
            reg_alpha=0.0,
            scale_pos_weight=scale_pos_weight,
            random_state=seed,
            n_jobs=n_jobs,
            verbosity=-1,
        )
        model.fit(x_train, y_train)
        return model.predict_proba(x_val)[:, 1], model.predict_proba(x_test)[:, 1]

    if model_name == "extratrees":
        model = ExtraTreesClassifier(
            n_estimators=600,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=seed,
            n_jobs=n_jobs,
        )
        model.fit(x_train, y_train)
        return model.predict_proba(x_val)[:, 1], model.predict_proba(x_test)[:, 1]

    raise ValueError(f"Unsupported model_name: {model_name}")


def parse_eval_models(spec: str) -> List[str]:
    values = [x.strip().lower() for x in spec.split(",") if x.strip()]
    allowed = {"xgb", "lgbm", "extratrees"}
    bad = [x for x in values if x not in allowed]
    if bad:
        raise ValueError(f"Unsupported eval models: {bad}. Allowed: {sorted(allowed)}")
    return values


def parse_topk_values(spec: str, max_k: int) -> List[int]:
    values: List[int] = []
    for item in [x.strip().lower() for x in spec.split(",") if x.strip()]:
        if item in {"all", "-1"}:
            k = max_k
        elif item.endswith("%"):
            pct = float(item[:-1]) / 100.0
            k = int(round(max_k * pct))
        else:
            k = int(item)
        k = max(1, min(int(k), max_k))
        values.append(k)
    return sorted(set(values))


def evaluate_feature_subsets(
    ranking: pd.DataFrame,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    topk_values: Sequence[int],
    eval_models: Sequence[str],
    seed: int,
    n_jobs: int,
    xgb_tree_method: str,
    threshold_min: float,
    threshold_max: float,
    threshold_steps: int,
) -> pd.DataFrame:
    ranked_nonconstant = ranking[~ranking["is_near_constant"]].copy()
    if ranked_nonconstant.empty:
        ranked_nonconstant = ranking.copy()

    ranked_indices = ranked_nonconstant["feature_index"].astype(int).tolist()
    rows: List[Dict[str, Any]] = []

    for k in topk_values:
        indices = ranked_indices[:k]
        for model_name in eval_models:
            print(f"[EVAL] model={model_name}, top_k={k}")
            try:
                val_prob, test_prob = fit_predict_model(
                    model_name,
                    x_train[:, indices],
                    y_train,
                    x_val[:, indices],
                    x_test[:, indices],
                    seed=seed,
                    n_jobs=n_jobs,
                    xgb_tree_method=xgb_tree_method,
                )
            except Exception as exc:
                print(f"[WARN] evaluation skipped for {model_name} top_k={k}: {exc}")
                continue

            val_search = threshold_search(
                y_val,
                val_prob,
                threshold_min=threshold_min,
                threshold_max=threshold_max,
                threshold_steps=threshold_steps,
            )
            th = float(val_search["best_f1"]["threshold"])
            val_metrics = binary_metrics_at_threshold(y_val, val_prob, th)
            test_metrics = binary_metrics_at_threshold(y_test, test_prob, th)

            rows.append(
                {
                    "model": model_name,
                    "top_k": int(k),
                    "threshold_from_val": th,
                    "val_precision_label1": val_metrics["precision_label1"],
                    "val_recall_label1": val_metrics["recall_label1"],
                    "val_f1_label1": val_metrics["f1_label1"],
                    "val_pr_auc": val_metrics["pr_auc"],
                    "val_auc": val_metrics["auc"],
                    "test_precision_label1": test_metrics["precision_label1"],
                    "test_recall_label1": test_metrics["recall_label1"],
                    "test_f1_label1": test_metrics["f1_label1"],
                    "test_pr_auc": test_metrics["pr_auc"],
                    "test_auc": test_metrics["auc"],
                    "test_confusion_matrix": json.dumps(test_metrics["confusion_matrix"]),
                    "selected_feature_indices": json.dumps(indices),
                }
            )

    return pd.DataFrame(rows)


def save_selected_npz(
    source_npz: str,
    output_npz: str,
    selected_indices: Sequence[int],
    selected_names: Sequence[str],
) -> None:
    _, _, arrays = load_flow_npz(source_npz)
    save_dict: Dict[str, Any] = {}
    for key, value in arrays.items():
        save_dict[key] = value

    save_dict["flow_feats"] = sanitize_features(arrays["flow_feats"])[:, selected_indices]
    save_dict["flow_feature_names"] = np.array(list(selected_names), dtype=object)
    save_dict["selected_flow_feature_indices"] = np.array(selected_indices, dtype=np.int64)
    save_dict["selected_flow_feature_names"] = np.array(list(selected_names), dtype=object)

    np.savez_compressed(output_npz, **save_dict)


def write_selected_feature_artifacts(
    ranking: pd.DataFrame,
    selected_indices: Sequence[int],
    out_dir: str,
) -> List[str]:
    selected_df = ranking[
        ranking["feature_index"].isin([int(x) for x in selected_indices])
    ].copy()
    selected_df = selected_df.sort_values("rank")

    selected_csv = os.path.join(out_dir, "selected_features.csv")
    selected_txt = os.path.join(out_dir, "selected_features.txt")
    selected_npy = os.path.join(out_dir, "selected_feature_indices.npy")

    selected_df.to_csv(selected_csv, index=False)
    np.save(selected_npy, np.array(selected_indices, dtype=np.int64))
    with open(selected_txt, "w", encoding="utf-8") as f:
        for _, row in selected_df.iterrows():
            f.write(
                f"{int(row['rank']):03d}\t"
                f"{int(row['feature_index'])}\t"
                f"{row['feature_name']}\t"
                f"score={float(row['ensemble_score']):.8f}\n"
            )

    return selected_df["feature_name"].astype(str).tolist()


def choose_best_row(results: pd.DataFrame, metric: str) -> pd.Series:
    if results.empty:
        raise ValueError("No feature subset evaluation results were produced.")
    if metric not in results.columns:
        raise ValueError(f"Selection metric not found: {metric}")
    return results.sort_values(metric, ascending=False).iloc[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rank flow features, evaluate top-K subsets, and optionally write "
            "selected NPZ files for rerunning flow-only baselines."
        )
    )
    parser.add_argument("--train_npz", required=True)
    parser.add_argument("--val_npz", required=True)
    parser.add_argument("--test_npz", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--preprocessor_joblib", default=None)

    parser.add_argument("--seed", type=int, default=130)
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("--xgb_tree_method", default="hist")

    parser.add_argument(
        "--importance_models",
        default="xgb,lgbm,extratrees",
        help="Comma-separated subset of xgb,lgbm,extratrees used in ranking.",
    )
    parser.add_argument(
        "--eval_models",
        default="xgb,lgbm,extratrees",
        help="Comma-separated subset of xgb,lgbm,extratrees used for top-K validation.",
    )
    parser.add_argument("--topk", default="5,10,15,20,30,40,50,all")
    parser.add_argument(
        "--selection_metric",
        default="val_f1_label1",
        choices=[
            "val_f1_label1",
            "val_pr_auc",
            "val_auc",
            "test_f1_label1",
            "test_pr_auc",
        ],
        help=(
            "Use val_* for formal model selection. test_* is diagnostic only "
            "and should not be used for final reporting."
        ),
    )
    parser.add_argument("--threshold_min", type=float, default=0.001)
    parser.add_argument("--threshold_max", type=float, default=0.999)
    parser.add_argument("--threshold_steps", type=int, default=999)
    parser.add_argument("--save_selected_npz", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    safe_mkdir(args.out_dir)

    print("[INFO] loading NPZ files...")
    x_train, y_train, train_arrays = load_flow_npz(args.train_npz)
    x_val, y_val, _ = load_flow_npz(args.val_npz)
    x_test, y_test, _ = load_flow_npz(args.test_npz)

    if x_train.shape[1] != x_val.shape[1] or x_train.shape[1] != x_test.shape[1]:
        raise ValueError(
            "Feature dimension mismatch: "
            f"train={x_train.shape}, val={x_val.shape}, test={x_test.shape}"
        )

    feature_names = get_feature_names(
        train_arrays,
        x_train.shape[1],
        args.preprocessor_joblib,
    )

    print(
        f"[DATA] train={x_train.shape}, val={x_val.shape}, test={x_test.shape}, "
        f"train_pos_ratio={class_ratio(y_train):.6f}"
    )

    importance_models = set(parse_eval_models(args.importance_models))
    eval_models = parse_eval_models(args.eval_models)

    ranking = compute_feature_ranking(
        x_train,
        y_train,
        feature_names,
        seed=args.seed,
        n_jobs=args.n_jobs,
        xgb_tree_method=args.xgb_tree_method,
        use_xgb="xgb" in importance_models,
        use_lgbm="lgbm" in importance_models,
        use_extratrees="extratrees" in importance_models,
    )

    ranking_path = os.path.join(args.out_dir, "flow_feature_importance_ranked.csv")
    ranking.to_csv(ranking_path, index=False)
    print(f"[INFO] saved ranking: {ranking_path}")

    max_k = int((~ranking["is_near_constant"]).sum())
    if max_k <= 0:
        max_k = len(ranking)
    topk_values = parse_topk_values(args.topk, max_k)
    print(f"[INFO] evaluating top_k values: {topk_values}")

    results = evaluate_feature_subsets(
        ranking,
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test,
        topk_values=topk_values,
        eval_models=eval_models,
        seed=args.seed,
        n_jobs=args.n_jobs,
        xgb_tree_method=args.xgb_tree_method,
        threshold_min=args.threshold_min,
        threshold_max=args.threshold_max,
        threshold_steps=args.threshold_steps,
    )

    results_path = os.path.join(args.out_dir, "flow_feature_subset_eval.csv")
    results.to_csv(results_path, index=False)
    print(f"[INFO] saved subset evaluation: {results_path}")

    best_row = choose_best_row(results, args.selection_metric)
    selected_indices = json.loads(str(best_row["selected_feature_indices"]))
    selected_indices = [int(x) for x in selected_indices]
    selected_names = write_selected_feature_artifacts(
        ranking,
        selected_indices,
        args.out_dir,
    )

    selected_npz_paths: Dict[str, str] = {}
    if args.save_selected_npz:
        selected_npz_dir = os.path.join(
            args.out_dir,
            f"selected_npz_top{len(selected_indices)}_{best_row['model']}",
        )
        safe_mkdir(selected_npz_dir)

        selected_npz_paths = {
            "train_npz": os.path.join(selected_npz_dir, "train_selected_flow_feats.npz"),
            "val_npz": os.path.join(selected_npz_dir, "val_selected_flow_feats.npz"),
            "test_npz": os.path.join(selected_npz_dir, "test_selected_flow_feats.npz"),
        }
        save_selected_npz(
            args.train_npz,
            selected_npz_paths["train_npz"],
            selected_indices,
            selected_names,
        )
        save_selected_npz(
            args.val_npz,
            selected_npz_paths["val_npz"],
            selected_indices,
            selected_names,
        )
        save_selected_npz(
            args.test_npz,
            selected_npz_paths["test_npz"],
            selected_indices,
            selected_names,
        )
        print(f"[INFO] saved selected NPZ files: {selected_npz_dir}")

    summary = {
        "args": vars(args),
        "data_shapes": {
            "train": list(x_train.shape),
            "val": list(x_val.shape),
            "test": list(x_test.shape),
            "train_pos_ratio": class_ratio(y_train),
            "val_pos_ratio": class_ratio(y_val),
            "test_pos_ratio": class_ratio(y_test),
        },
        "ranking_csv": os.path.abspath(ranking_path),
        "subset_eval_csv": os.path.abspath(results_path),
        "selection_metric": args.selection_metric,
        "best_row": best_row.to_dict(),
        "selected_feature_count": len(selected_indices),
        "selected_feature_indices": selected_indices,
        "selected_feature_names": selected_names,
        "selected_npz_paths": {
            k: os.path.abspath(v) for k, v in selected_npz_paths.items()
        },
    }
    summary_path = os.path.join(args.out_dir, "flow_feature_selection_summary.json")
    save_json(summary, summary_path)

    print("\n[BEST SUBSET]")
    print(
        f"model={best_row['model']} top_k={int(best_row['top_k'])} "
        f"val_f1={float(best_row['val_f1_label1']):.4f} "
        f"test_f1={float(best_row['test_f1_label1']):.4f} "
        f"test_pr_auc={float(best_row['test_pr_auc']):.4f}"
    )
    print(f"[INFO] selected feature count: {len(selected_indices)}")
    print(f"[INFO] saved summary: {summary_path}")

    if args.save_selected_npz:
        print("\n[RE-RUN YOUR FULL FLOW-ONLY BASELINE WITH SELECTED FEATURES]")
        print(
            "python s1/flow_only_baseline.py "
            f"--train_npz {selected_npz_paths['train_npz']} "
            f"--val_npz {selected_npz_paths['val_npz']} "
            f"--test_npz {selected_npz_paths['test_npz']} "
            f"--out_dir {os.path.join(args.out_dir, 'baseline_on_selected_features')} "
            "--models mlp xgb lgbm"
        )


if __name__ == "__main__":
    main()
