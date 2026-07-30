import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score

try:
    from scipy.stats import mannwhitneyu
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


K_LIST = [16, 32, 64, 128, 256]


def safe_auc(y_true, score):
    y_true = np.asarray(y_true)
    score = np.asarray(score)

    mask = np.isfinite(score)
    y_true = y_true[mask]
    score = score[mask]

    if len(np.unique(y_true)) < 2:
        return np.nan

    try:
        auc = roc_auc_score(y_true, score)
        return max(auc, 1.0 - auc)
    except Exception:
        return np.nan


def cohen_d(x1, x0):
    x1 = np.asarray(x1, dtype=float)
    x0 = np.asarray(x0, dtype=float)

    x1 = x1[np.isfinite(x1)]
    x0 = x0[np.isfinite(x0)]

    if len(x1) < 2 or len(x0) < 2:
        return np.nan

    n1, n0 = len(x1), len(x0)
    s1, s0 = np.var(x1, ddof=1), np.var(x0, ddof=1)

    pooled = np.sqrt(((n1 - 1) * s1 + (n0 - 1) * s0) / max(n1 + n0 - 2, 1))

    if pooled == 0:
        return 0.0

    return (np.mean(x1) - np.mean(x0)) / pooled


def label_entropy_from_ratio(r):
    r = np.asarray(r, dtype=float)
    eps = 1e-12
    r = np.clip(r, eps, 1.0 - eps)
    return -(r * np.log2(r) + (1.0 - r) * np.log2(1.0 - r))


def compute_previous_k_stats(labels, groups, context_name, k_list):
    n = len(labels)
    results = {}

    for k in k_list:
        ratio = np.zeros(n, dtype=np.float32)
        attack_count = np.zeros(n, dtype=np.int32)
        benign_count = np.zeros(n, dtype=np.int32)
        neighbor_count = np.zeros(n, dtype=np.int32)

        for _, idxs in groups.items():
            idxs = np.asarray(idxs, dtype=np.int64)

            if len(idxs) <= 1:
                continue

            group_labels = labels[idxs].astype(np.int32)
            cumsum = np.concatenate([[0], np.cumsum(group_labels)])

            local_pos = np.arange(len(idxs))
            starts = np.maximum(0, local_pos - k)
            counts = local_pos - starts

            sums = cumsum[local_pos] - cumsum[starts]

            valid = counts > 0
            global_idxs = idxs[valid]

            attack_count[global_idxs] = sums[valid]
            neighbor_count[global_idxs] = counts[valid]
            benign_count[global_idxs] = counts[valid] - sums[valid]
            ratio[global_idxs] = sums[valid] / counts[valid]

        entropy = label_entropy_from_ratio(ratio)
        entropy[neighbor_count == 0] = 0.0

        results[f"{context_name}_attack_ratio_k{k}"] = ratio
        results[f"{context_name}_attack_count_k{k}"] = attack_count
        results[f"{context_name}_benign_count_k{k}"] = benign_count
        results[f"{context_name}_neighbor_count_k{k}"] = neighbor_count
        results[f"{context_name}_label_entropy_k{k}"] = entropy.astype(np.float32)

        print(
            f"[{context_name}] K={k:<3d} "
            f"mean_neighbor_count={neighbor_count.mean():.2f} "
            f"mean_attack_ratio={ratio.mean():.4f}"
        )

    return results


def compute_time_stats(labels, k_list):
    n = len(labels)
    groups = {"all": np.arange(n)}
    return compute_previous_k_stats(labels, groups, "time", k_list)


def build_endpoint_key(df):
    src = df["source_ip"].astype(str).to_numpy()
    dst = df["destination_ip"].astype(str).to_numpy()

    endpoint = []
    for s, d in zip(src, dst):
        if s <= d:
            endpoint.append(f"{s}__{d}")
        else:
            endpoint.append(f"{d}__{s}")

    return pd.Series(endpoint, index=df.index)


def statistical_validation(df, contexts, k_list, out_dir):
    rows = []
    y = df["label"].to_numpy(dtype=int)

    for context in contexts:
        for k in k_list:
            ratio_col = f"{context}_attack_ratio_k{k}"
            count_col = f"{context}_neighbor_count_k{k}"
            entropy_col = f"{context}_label_entropy_k{k}"

            valid = df[count_col].to_numpy() > 0

            for feature_col in [ratio_col, entropy_col, count_col]:
                values = df[feature_col].to_numpy(dtype=float)

                attack_values = values[(y == 1) & valid]
                benign_values = values[(y == 0) & valid]

                if SCIPY_AVAILABLE and len(attack_values) > 0 and len(benign_values) > 0:
                    try:
                        _, p_value = mannwhitneyu(
                            attack_values,
                            benign_values,
                            alternative="two-sided"
                        )
                    except Exception:
                        p_value = np.nan
                else:
                    p_value = np.nan

                rows.append({
                    "context": context,
                    "K": k,
                    "feature": feature_col,
                    "valid_flows": int(valid.sum()),
                    "attack_valid_flows": int(((y == 1) & valid).sum()),
                    "benign_valid_flows": int(((y == 0) & valid).sum()),
                    "attack_mean": float(np.mean(attack_values)) if len(attack_values) else np.nan,
                    "benign_mean": float(np.mean(benign_values)) if len(benign_values) else np.nan,
                    "attack_median": float(np.median(attack_values)) if len(attack_values) else np.nan,
                    "benign_median": float(np.median(benign_values)) if len(benign_values) else np.nan,
                    "mean_difference": float(np.mean(attack_values) - np.mean(benign_values))
                    if len(attack_values) and len(benign_values) else np.nan,
                    "cohen_d": float(cohen_d(attack_values, benign_values)),
                    "mannwhitney_p": float(p_value) if np.isfinite(p_value) else np.nan,
                    "auc_as_single_feature": float(safe_auc(y[valid], values[valid])),
                })

    stats = pd.DataFrame(rows)
    stats_path = out_dir / "context_statistical_validation.csv"
    stats.to_csv(stats_path, index=False)

    print(f"[INFO] saved statistical validation: {stats_path}")
    return stats


def plot_attack_ratio_histograms(df, contexts, k_list, out_dir):
    y = df["label"].to_numpy(dtype=int)

    for context in contexts:
        for k in k_list:
            ratio_col = f"{context}_attack_ratio_k{k}"
            count_col = f"{context}_neighbor_count_k{k}"

            valid = df[count_col].to_numpy() > 0
            attack_values = df.loc[(y == 1) & valid, ratio_col].to_numpy()
            benign_values = df.loc[(y == 0) & valid, ratio_col].to_numpy()

            plt.figure(figsize=(8, 5))
            plt.hist(benign_values, bins=40, alpha=0.6, density=True, label="Benign flow")
            plt.hist(attack_values, bins=40, alpha=0.6, density=True, label="Attack flow")
            plt.xlabel("Neighbor attack ratio")
            plt.ylabel("Density")
            plt.title(f"{context} context, K={k}")
            plt.legend()
            plt.tight_layout()

            save_path = out_dir / f"hist_{context}_k{k}.png"
            plt.savefig(save_path, dpi=200)
            plt.close()


def plot_auc_summary(stats, out_dir):
    sub = stats[stats["feature"].str.contains("attack_ratio")].copy()

    plt.figure(figsize=(10, 6))

    for context in sorted(sub["context"].unique()):
        tmp = sub[sub["context"] == context].sort_values("K")
        plt.plot(
            tmp["K"],
            tmp["auc_as_single_feature"],
            marker="o",
            label=context
        )

    plt.xlabel("K")
    plt.ylabel("AUC using context attack ratio only")
    plt.title("Context label relevance")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    save_path = out_dir / "context_auc_summary.png"
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_effect_size_summary(stats, out_dir):
    sub = stats[stats["feature"].str.contains("attack_ratio")].copy()

    plt.figure(figsize=(10, 6))

    for context in sorted(sub["context"].unique()):
        tmp = sub[sub["context"] == context].sort_values("K")
        plt.plot(
            tmp["K"],
            tmp["cohen_d"],
            marker="o",
            label=context
        )

    plt.xlabel("K")
    plt.ylabel("Cohen's d")
    plt.title("Effect size of neighbor attack ratio")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    save_path = out_dir / "context_effect_size_summary.png"
    plt.savefig(save_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--flow_csv", type=str, default="dataset/ar002_et12_20260511_002-stage1_flows.csv")
    parser.add_argument("--out_dir", type=str, default="context_analysis")
    parser.add_argument("--k_list", type=str, default="16,32,64,128,256")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    k_list = [int(x.strip()) for x in args.k_list.split(",") if x.strip()]

    print("=" * 80)
    print("Loading flow CSV")
    print("=" * 80)

    df = pd.read_csv(args.flow_csv)

    required_cols = [
        "flow_id",
        "flow_start_timestamp_us",
        "source_ip",
        "destination_ip",
        "label",
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)

    df = df.sort_values("flow_start_timestamp_us", kind="mergesort").reset_index(drop=True)
    df["global_index"] = np.arange(len(df), dtype=np.int64)

    labels = df["label"].to_numpy(dtype=np.int8)

    print(f"Total flows: {len(df):,}")
    print("Label counts:")
    print(df["label"].value_counts().sort_index())

    print("\nBuilding groups...")

    source_groups = df.groupby("source_ip", sort=False).indices
    destination_groups = df.groupby("destination_ip", sort=False).indices

    endpoint_key = build_endpoint_key(df)
    endpoint_groups = endpoint_key.groupby(endpoint_key, sort=False).indices

    print(f"Unique sources: {len(source_groups):,}")
    print(f"Unique destinations: {len(destination_groups):,}")
    print(f"Unique endpoints: {len(endpoint_groups):,}")

    print("\nComputing context features...")

    all_results = {}

    all_results.update(compute_time_stats(labels, k_list))
    all_results.update(compute_previous_k_stats(labels, source_groups, "source", k_list))
    all_results.update(compute_previous_k_stats(labels, destination_groups, "destination", k_list))
    all_results.update(compute_previous_k_stats(labels, endpoint_groups, "endpoint", k_list))

    for col, values in all_results.items():
        df[col] = values

    feature_path = out_dir / "context_features.csv"
    df.to_csv(feature_path, index=False)
    print(f"\n[INFO] saved context features: {feature_path}")

    contexts = ["time", "source", "destination", "endpoint"]

    print("\nRunning statistical validation...")
    stats = statistical_validation(df, contexts, k_list, out_dir)

    print("\nGenerating figures...")
    plot_attack_ratio_histograms(df, contexts, k_list, out_dir)
    plot_auc_summary(stats, out_dir)
    plot_effect_size_summary(stats, out_dir)

    summary = stats[
        stats["feature"].str.contains("attack_ratio")
    ].sort_values(
        ["auc_as_single_feature", "cohen_d"],
        ascending=False
    )

    summary_path = out_dir / "context_attack_ratio_ranking.csv"
    summary.to_csv(summary_path, index=False)

    print("\nTop context settings by AUC:")
    print(
        summary[
            [
                "context",
                "K",
                "attack_mean",
                "benign_mean",
                "mean_difference",
                "cohen_d",
                "auc_as_single_feature",
                "mannwhitney_p",
            ]
        ].head(20)
    )

    print("\nDone.")
    print(f"Outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()