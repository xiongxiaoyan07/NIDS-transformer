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
from stage2.model import build_stage2_model
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
        "--seed",
        type=int,
        default=None,
        help="Override seed",
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
    parser.add_argument(
        "--model_pooling",
        type=str,
        default=None,
        choices=["last", "mean", "attention"],
        help="Override model.pooling",
    )
    parser.add_argument(
        "--cls_head",
        type=int,
        default=None,
        help="Override model.cls_head",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        choices=[
            "no_context_mlp",
            "target_query_gated",
            "target_query",
            "target_query_residual",
            "target_query_residual_attention",
            "lstm",
            "transformer",
            "residual_transformer",
        ],
        help="Override model.model_type",
    )

    parser.add_argument(
        "--lstm_hidden_dim",
        type=int,
        default=None,
        help="Override model.lstm_hidden_dim",
    )

    parser.add_argument(
        "--lstm_num_layers",
        type=int,
        default=None,
        help="Override model.lstm_num_layers",
    )

    parser.add_argument(
        "--lstm_bidirectional",
        action="store_true",
        help="Override model.lstm_bidirectional=True",
    )
    parser.add_argument(
        "--context_scale",
        type=float,
        default=None,
        help="Override model.context_scale for residual context models",
    )
    parser.add_argument(
        "--gate_bias_init",
        type=float,
        default=None,
        help="Override model.gate_bias_init for gated residual models",
    )
    parser.add_argument(
        "--use_context_length_feature",
        action="store_true",
        help="Override model.use_context_length_feature=True",
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
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.model_pooling is not None:
        cfg["model"]["pooling"] = args.model_pooling
    if args.cls_head is not None:
        cfg["model"]["cls_head"] = args.cls_head
    if args.model_type is not None:
        cfg["model"]["model_type"] = args.model_type
    if args.context_scale is not None:
        cfg["model"]["context_scale"] = args.context_scale
    if args.gate_bias_init is not None:
        cfg["model"]["gate_bias_init"] = args.gate_bias_init
    if args.use_context_length_feature:
        cfg["model"]["use_context_length_feature"] = True

    if args.lstm_hidden_dim is not None:
        cfg["model"]["lstm_hidden_dim"] = args.lstm_hidden_dim

    if args.lstm_num_layers is not None:
        cfg["model"]["lstm_num_layers"] = args.lstm_num_layers

    if args.lstm_bidirectional:
        cfg["model"]["lstm_bidirectional"] = True
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
    print(
        f"[INFO] Stage2 --------- seed: "
        f"{int(cfg.get('seed', 42))}"
    )
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
    print("[INFO] Stage2 --------- model_type =", cfg["model"].get("model_type", "transformer"))
    print("[INFO] Stage2 --------- model.pooling =", cfg["model"].get("pooling", "last"))
    model = build_stage2_model(cfg, input_dim=input_dim).to(device)
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















