from __future__ import annotations

import argparse
import bisect
import os
from collections import defaultdict
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

from run_stage2 import print_context_diagnostics, print_data_summary
from stage2.config import load_config, save_config
from stage2.context import ContextIndexBuilder
from stage2.data_io import prepare_sorted_stage2_data
from stage2.dataset import Stage2Dataset
from stage2.model import build_stage2_model
from stage2.trainer import Stage2Trainer
from stage2.utils import get_device, safe_mkdir, set_seed


class LeakyContextIndexBuilder:
    """Intentionally leaky context builder for negative-control experiments.

    This builder breaks the clean train/val/test boundary on purpose:
    every target can use context flows from all splits and from both past and
    future chronological positions. Do not use its metrics as valid results.
    """

    def __init__(self, meta_df: pd.DataFrame, cfg: Dict[str, Any]):
        self.meta_df = meta_df.reset_index(drop=True)
        self.cfg = cfg
        self.source_col = cfg["data"].get("source_col", "source_id")
        self.destination_col = cfg["data"].get("destination_col", "destination_id")

        ctx = cfg["context"]
        raw_method = ctx.get("method", "time_only")
        self.method = "source_destination" if raw_method in DUAL_CONTEXT_METHODS else raw_method
        self.window_size = int(ctx.get("window_size", 128))
        self.include_target = True
        self.k = max(self.window_size - 1, 0)

        if self.method not in {"time_only", "source_host", "destination_host", "endpoint", "source_destination"}:
            raise ValueError(f"Unknown context.method for leaky builder: {self.method}")

        print(
            "[LEAKY] Context builder enabled: "
            f"method={self.method}, window_size={self.window_size}, "
            "all splits visible, past+future context visible"
        )

    def _nearest_from_sorted(self, values: Sequence[int], row_idx: int) -> List[int]:
        if self.k == 0 or not values:
            return []

        pos = bisect.bisect_left(values, row_idx)
        left = pos - 1
        right = pos
        chosen: List[int] = []

        while len(chosen) < self.k and (left >= 0 or right < len(values)):
            left_dist = abs(values[left] - row_idx) if left >= 0 else None
            right_dist = abs(values[right] - row_idx) if right < len(values) else None

            if right < len(values) and values[right] == row_idx:
                right += 1
                continue

            take_left = False
            if left >= 0 and right < len(values):
                take_left = left_dist <= right_dist
            elif left >= 0:
                take_left = True

            if take_left:
                candidate = int(values[left])
                left -= 1
            else:
                candidate = int(values[right])
                right += 1

            if candidate != row_idx:
                chosen.append(candidate)

        return sorted(chosen)

    def _time_context(self, row_idx: int, n: int) -> List[int]:
        if self.k == 0:
            return []

        half_left = self.k // 2
        start = max(0, row_idx - half_left)
        end = min(n, start + self.k + 1)
        start = max(0, end - self.k - 1)
        ctx = [i for i in range(start, end) if i != row_idx]

        if len(ctx) > self.k:
            ctx = sorted(ctx, key=lambda i: (abs(i - row_idx), i))[: self.k]
        return sorted(ctx)

    def build(self) -> List[Any]:
        n = len(self.meta_df)
        sources = self.meta_df[self.source_col].astype(str).tolist()
        destinations = self.meta_df[self.destination_col].astype(str).tolist()

        source_map: Dict[str, List[int]] = defaultdict(list)
        dest_map: Dict[str, List[int]] = defaultdict(list)
        endpoint_map: Dict[str, List[int]] = defaultdict(list)
        for idx, (src, dst) in enumerate(zip(sources, destinations)):
            source_map[src].append(idx)
            dest_map[dst].append(idx)
            endpoint_map[src].append(idx)
            if dst != src:
                endpoint_map[dst].append(idx)

        contexts: List[Any] = []
        for idx in range(n):
            src = sources[idx]
            dst = destinations[idx]

            if self.method == "source_destination":
                src_ctx = self._nearest_from_sorted(source_map[src], idx)
                dst_ctx = self._nearest_from_sorted(dest_map[dst], idx)
                src_ctx.append(idx)
                dst_ctx.append(idx)
                contexts.append(
                    {
                        "source": np.array(src_ctx, dtype=np.int64),
                        "destination": np.array(dst_ctx, dtype=np.int64),
                    }
                )
                continue

            if self.method == "time_only":
                ctx = self._time_context(idx, n)
            elif self.method == "source_host":
                ctx = self._nearest_from_sorted(source_map[src], idx)
            elif self.method == "destination_host":
                ctx = self._nearest_from_sorted(dest_map[dst], idx)
            else:
                merged = sorted(set(endpoint_map[src]).union(endpoint_map[dst]))
                ctx = self._nearest_from_sorted(merged, idx)

            ctx.append(idx)
            contexts.append(np.array(ctx, dtype=np.int64))

        return contexts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run intentionally leaky Stage2 experiment. Do not report as valid metrics."
    )
    parser.add_argument("--stage1_dir", required=True, type=str)
    parser.add_argument("--out_dir", required=True, type=str)
    parser.add_argument("--config", default=None, type=str)
    parser.add_argument(
        "--context_method",
        default=None,
        choices=["time_only", "source_host", "destination_host", "endpoint", "source_destination"],
    )
    parser.add_argument("--context_policy", default=None, type=str)
    parser.add_argument("--window_size", default=None, type=int)
    parser.add_argument("--seed", default=None, type=int)
    parser.add_argument("--epochs", default=None, type=int)
    parser.add_argument("--batch_size", default=None, type=int)
    parser.add_argument("--model_type", default=None, type=str)
    parser.add_argument("--cls_head", default=None, type=int)
    parser.add_argument("--device", default=None, type=str)
    parser.add_argument(
        "--leak_train_on_eval",
        action="store_true",
        help="Also train the Stage2 classifier on train+val+test target rows.",
    )
    parser.add_argument(
        "--leak_eval_fraction",
        default=1.0,
        type=float,
        help=(
            "Fraction of original val and test target rows to relabel as train "
            "when --leak_train_on_eval is set. Use 0.5 for half leakage."
        ),
    )
    parser.add_argument(
        "--leak_eval_seed",
        default=None,
        type=int,
        help="Seed for selecting which val/test rows are leaked into train.",
    )
    parser.add_argument(
        "--clean_context",
        action="store_true",
        help="Use the normal ContextIndexBuilder and only apply target-row leakage.",
    )
    return parser.parse_args()


def apply_cli_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    cfg["data"]["stage1_dir"] = args.stage1_dir
    if args.context_method is not None:
        cfg["context"]["method"] = args.context_method
    if args.context_policy is not None:
        cfg["context"]["context_policy"] = args.context_policy
    if args.window_size is not None:
        cfg["context"]["window_size"] = args.window_size
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size
    if args.model_type is not None:
        cfg["model"]["model_type"] = args.model_type
    if args.cls_head is not None:
        cfg["model"]["cls_head"] = args.cls_head
    if args.device is not None:
        cfg["training"]["device"] = args.device

    cfg["context"]["include_target"] = True
    cfg["_leakage_experiment"] = {
        "warning": "Intentional data leakage. Do not report these metrics as valid.",
        "context_leakage": "all train/val/test splits visible; past and future flows visible",
        "train_target_leakage": bool(args.leak_train_on_eval),
        "leak_eval_fraction": float(args.leak_eval_fraction),
        "leak_eval_seed": args.leak_eval_seed,
        "clean_context": bool(args.clean_context),
    }
    return cfg


def build_datasets(
    meta_df: pd.DataFrame,
    z_sorted: np.ndarray,
    context_indices: List[Any],
    cfg: Dict[str, Any],
    leak_train_on_eval: bool,
    leak_eval_fraction: float,
    leak_eval_seed: int,
) -> Dict[str, Stage2Dataset]:
    source_col = cfg["data"].get("source_col", "source_id")
    destination_col = cfg["data"].get("destination_col", "destination_id")

    train_meta = meta_df
    if leak_train_on_eval:
        train_meta = meta_df.copy()
        leak_eval_fraction = float(leak_eval_fraction)
        if leak_eval_fraction <= 0.0 or leak_eval_fraction > 1.0:
            raise ValueError("--leak_eval_fraction must be in (0, 1]")

        rng = np.random.default_rng(int(leak_eval_seed))
        leaked_counts: Dict[str, int] = {}
        for split in ["val", "test"]:
            rows = train_meta.index[train_meta["split"] == split].to_numpy(dtype=np.int64)
            n_leak = int(round(len(rows) * leak_eval_fraction))
            if n_leak <= 0:
                continue
            leaked_rows = rng.choice(rows, size=n_leak, replace=False)
            train_meta.loc[leaked_rows, "split"] = "train"
            leaked_counts[split] = int(n_leak)

        print(
            "[LEAKY] Training target rows include original train plus sampled eval rows: "
            f"fraction={leak_eval_fraction:.3f}, leaked_counts={leaked_counts}"
        )

    return {
        "train": Stage2Dataset(
            meta_df_sorted=train_meta,
            z_sorted=z_sorted,
            context_indices=context_indices,
            target_split="train",
            source_col=source_col,
            destination_col=destination_col,
        ),
        "val": Stage2Dataset(
            meta_df_sorted=meta_df,
            z_sorted=z_sorted,
            context_indices=context_indices,
            target_split="val",
            source_col=source_col,
            destination_col=destination_col,
        ),
        "test": Stage2Dataset(
            meta_df_sorted=meta_df,
            z_sorted=z_sorted,
            context_indices=context_indices,
            target_split="test",
            source_col=source_col,
            destination_col=destination_col,
        ),
    }


def main() -> None:
    args = parse_args()
    cfg = apply_cli_overrides(load_config(args.config), args)

    safe_mkdir(args.out_dir)
    save_config(cfg, os.path.join(args.out_dir, "stage2_leaky_config_used.yaml"))

    print("\n" + "=" * 80)
    print("[LEAKY WARNING] This run intentionally introduces train/val/test leakage.")
    print("[LEAKY WARNING] Use it only as a negative-control demonstration.")
    print("=" * 80 + "\n")

    set_seed(int(cfg.get("seed", 42)))
    device = get_device(cfg["training"].get("device", "auto"))
    print(f"[INFO] Stage2 leaky device: {device}")

    meta_df, z_sorted = prepare_sorted_stage2_data(
        stage1_dir=cfg["data"]["stage1_dir"],
        cfg=cfg,
    )

    if args.clean_context:
        print("[LEAKY] clean_context=True: using normal ContextIndexBuilder.")
        context_indices = ContextIndexBuilder(meta_df, cfg).build()
    else:
        context_indices = LeakyContextIndexBuilder(meta_df, cfg).build()
    print("[LEAKY] first context indices:", context_indices[:3])
    print_context_diagnostics(
        meta_df=meta_df,
        context_indices=context_indices,
        include_target=True,
    )
    print_data_summary(meta_df, z_sorted, context_indices)

    datasets = build_datasets(
        meta_df=meta_df,
        z_sorted=z_sorted,
        context_indices=context_indices,
        cfg=cfg,
        leak_train_on_eval=bool(args.leak_train_on_eval),
        leak_eval_fraction=float(args.leak_eval_fraction),
        leak_eval_seed=int(args.leak_eval_seed if args.leak_eval_seed is not None else cfg.get("seed", 42)),
    )

    input_dim = int(z_sorted.shape[1])
    model = build_stage2_model(cfg, input_dim=input_dim).to(device)
    print(model)

    trainer = Stage2Trainer(
        model=model,
        datasets=datasets,
        cfg=cfg,
        device=device,
        out_dir=args.out_dir,
        input_dim=input_dim,
    )
    trainer.fit()
    trainer.final_evaluate_and_save(meta_df=meta_df, context_indices=context_indices)
    print(f"[LEAKY] Finished. Outputs saved to: {args.out_dir}")


if __name__ == "__main__":
    main()


# !python /content/drive/MyDrive/s2/run_stage2_leaky.py \
#   --stage1_dir /content/drive/MyDrive/s1/0704/0704C_ar002_et12_20260511_001 \
#   --config /content/drive/MyDrive/s2/0709ar002_et12_20260511_001/target_query_gated_v1/stage2_config_used.yaml \
#   --out_dir /content/drive/MyDrive/s2/LEAKY/half_val_test_to_train_clean_context \
#   --context_method source_host \
#   --context_policy online \
#   --window_size 128 \
#   --seed 130 \
#   --epochs 100 \
#   --batch_size 128 \
#   --leak_train_on_eval \
#   --leak_eval_fraction 0.5 \
#   --leak_eval_seed 130 \
#   --clean_context

# !python /content/drive/MyDrive/s2/run_stage2_leaky.py \
#   --stage1_dir /content/drive/MyDrive/s1/0704/0704C_ar002_et12_20260511_001 \
#   --config /content/drive/MyDrive/s2/0709ar002_et12_20260511_001/leakage/stage2_config_used.yaml \
#   --out_dir /content/drive/MyDrive/s2/0709ar002_et12_20260511_001/leakage \
#   --context_method source_host \
#   --context_policy online \
#   --window_size 128 \
#   --seed 130 \
#   --epochs 100 \
#   --batch_size 128 \
#   --leak_train_on_eval