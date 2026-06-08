from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from stage2.config import load_config, save_config
from stage2.context import ContextIndexBuilder
from stage2.data_io import prepare_sorted_stage2_data
from stage2.dataset import Stage2Dataset
from stage2.model import Stage2Transformer
from stage2.trainer import Stage2Trainer
from stage2.utils import get_device, safe_mkdir, set_seed

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage2 inter-flow Transformer")
    parser.add_argument(
        "--stage1_dir",
        type=str,
        required=True,
        help="Directory containing Stage1 outputs",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Output directory for Stage2 run",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Stage2 YAML config path",
    )
    parser.add_argument(
        "--context_method",
        type=str,
        default=None,
        choices=["time_only", "source_host", "destination_host", "endpoint"],
        help="Override context.method",
    )
    parser.add_argument(
        "--context_policy",
        type=str,
        default=None,
        choices=["online", "split_isolated", "train_only_for_eval"],
        help="Override context.context_policy",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=None,
        help="Override context.window_size",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override training.epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override training.batch_size",
    )
    # model --- use_positional_encoding
    # model --- pooling
    return parser.parse_args()

def apply_cli_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    cfg["data"]["stage1_dir"] = args.stage1_dir

    if args.context_method is not None:
        cfg["context"]["method"] = args.context_method
    if args.context_policy is not None:
        cfg["context"]["context_policy"] = args.context_policy
    if args.window_size is not None:
        cfg["context"]["window_size"] = args.window_size
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size
    return cfg

def print_data_summary(meta_df: pd.DataFrame, z_sorted: np.ndarray, context_indices: List[np.ndarray]) -> None:
    print("[INFO] Stage2 loaded flows:")
    print(meta_df["split"].value_counts().to_string())

    print("[INFO] label counts by split:")
    print(pd.crosstab(meta_df["split"], meta_df["label"]).to_string())

    print(f"[INFO] z shape: {z_sorted.shape}")

    lengths = np.array([len(x) for x in context_indices], dtype=np.int64)
    print(
        "[INFO] context lengths: "
        f"min={lengths.min()}, "
        f"mean={lengths.mean():.2f}, "
        f"p50={np.percentile(lengths, 50):.1f}, "
        f"p95={np.percentile(lengths, 95):.1f}, "
        f"max={lengths.max()}"
    )

def build_datasets(
    meta_df: pd.DataFrame,
    z_sorted: np.ndarray,
    context_indices: List[np.ndarray],
) -> Dict[str, Stage2Dataset]:
    return {
        split: Stage2Dataset(
            meta_df_sorted=meta_df,
            z_sorted=z_sorted,
            context_indices=context_indices,
            target_split=split,
        )
        for split in ["train", "val", "test"]
    }

def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    cfg = apply_cli_overrides(cfg, args)

    safe_mkdir(args.out_dir)
    save_config(cfg, os.path.join(args.out_dir, "stage2_config_used.yaml"))

    set_seed(int(cfg.get("seed", 42)))
    device = get_device(cfg["training"].get("device", "auto"))
    print(f"[INFO] Stage2 --------- device: {device}")

    print("[INFO] Stage2 --------- loading data... prepare_sorted_stage2_data")
    meta_df, z_sorted = prepare_sorted_stage2_data(
        stage1_dir=cfg["data"]["stage1_dir"],
        cfg=cfg,
    )

    print("[INFO] Stage2 --------- ContextIndexBuilder")
    context_builder = ContextIndexBuilder(meta_df, cfg)
    context_indices = context_builder.build()

    print_data_summary(meta_df, z_sorted, context_indices)

    print("[INFO] Stage2 --------- build_datasets")
    datasets = build_datasets(meta_df, z_sorted, context_indices)

    input_dim = int(z_sorted.shape[1])
    print("[INFO] Stage2 --------- input_dim = ", input_dim)
    print("[INFO] Stage2 --------- Building model")
    model = Stage2Transformer(cfg, input_dim=input_dim).to(device)
    print(model)

    print("[INFO] Stage2 --------- Building trainer")
    trainer = Stage2Trainer(
        model=model,
        datasets=datasets,
        cfg=cfg,
        device=device,
        out_dir=args.out_dir,
        input_dim=input_dim,
    )
    print("[INFO] Stage2 --------- Training")
    trainer.fit()
    print("[INFO] Stage2 --------- Evaluating")
    trainer.final_evaluate_and_save(meta_df=meta_df, context_indices=context_indices)

    print(f"[INFO] Stage2 finished. Outputs saved to: {args.out_dir}")

if __name__ == "__main__":
    main()
















