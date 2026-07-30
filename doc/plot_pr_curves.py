"""Plot aligned test-set precision-recall curves from Stage 2 prediction CSVs.

Example:
    python plot_pr_curves.py \
      --prediction "No context=/path/non_context/stage2_predictions_test.csv" \
      --prediction "Proposed=/path/proposed/stage2_predictions_test.csv" \
      --output figures/stage2_pr_curves.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {"flow_id", "label", "prob_label_1"}


def precision_recall(y_true: np.ndarray, scores: np.ndarray):
    order = np.argsort(-scores, kind="mergesort")
    y = y_true[order].astype(np.int64)
    sorted_scores = scores[order]
    threshold_indices = np.r_[np.where(np.diff(sorted_scores))[0], y.size - 1]
    tp = np.cumsum(y)[threshold_indices]
    fp = 1 + threshold_indices - tp
    positives = int(y.sum())
    if positives == 0:
        raise ValueError("The supplied test set contains no positive examples.")
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / positives
    precision = np.r_[1.0, precision]
    recall = np.r_[0.0, recall]
    average_precision = float(np.sum(np.diff(recall) * precision[1:]))
    return precision, recall, average_precision


def load_prediction(spec: str):
    if "=" not in spec:
        raise ValueError(f"Expected NAME=CSV, received: {spec}")
    name, raw_path = spec.split("=", 1)
    path = Path(raw_path).expanduser().resolve()
    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    frame = frame[["flow_id", "label", "prob_label_1"]].sort_values("flow_id").reset_index(drop=True)
    return name.strip(), path, frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction", action="append", required=True, help="Model name and CSV as NAME=PATH")
    parser.add_argument("--output", required=True, type=Path, help="Output PDF path")
    args = parser.parse_args()

    loaded = [load_prediction(spec) for spec in args.prediction]
    reference_ids = loaded[0][2]["flow_id"].to_numpy()
    reference_labels = loaded[0][2]["label"].to_numpy(dtype=np.int64)
    for name, path, frame in loaded[1:]:
        if not np.array_equal(reference_ids, frame["flow_id"].to_numpy()):
            raise ValueError(f"Flow IDs are not aligned for {name}: {path}")
        if not np.array_equal(reference_labels, frame["label"].to_numpy(dtype=np.int64)):
            raise ValueError(f"Labels are not aligned for {name}: {path}")

    colors = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#6B7280"]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for (name, _, frame), color in zip(loaded, colors, strict=False):
        precision, recall, ap = precision_recall(
            frame["label"].to_numpy(dtype=np.int64),
            frame["prob_label_1"].to_numpy(dtype=np.float64),
        )
        ax.step(recall, precision, where="post", linewidth=1.5, color=color, label=f"{name} (AP={ap:.4f})")

    prevalence = float(reference_labels.mean())
    ax.axhline(prevalence, color="#6B7280", linestyle="--", linewidth=1.0, label=f"Prevalence={prevalence:.4f}")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.grid(color="#D1D5DB", linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower left", fontsize=8)
    fig.tight_layout()

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
