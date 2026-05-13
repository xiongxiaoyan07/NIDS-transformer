#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Entry point for Stage1 training.

Examples:

1) Normal train/val/test split:

python run_stage1.py \
  --packet_csv /home/xxiong/pcaps/stage1_packets.csv \
  --flow_csv /home/xxiong/pcaps/stage1_flows.csv \
  --config configs/stage1_config.yaml \
  --out_dir ./stage1_artifacts

2) External final test:

python run_stage1.py \
  --packet_csv ./train_stage1_packets.csv \
  --flow_csv ./train_stage1_flows.csv \
  --external_packet_csv ./final_test_packets.csv \
  --external_flow_csv ./final_test_flows.csv \
  --config configs/stage1_config.yaml \
  --out_dir ./stage1_artifacts_external_test
"""

from __future__ import annotations

import argparse
import os

import torch

from stage1.config import load_config
from stage1.model import Stage1TimeAwareTransformer
from stage1.pipeline import build_dataloaders
from stage1.trainer import train_model
from stage1.export_embeddings import export_embeddings
from stage1.utils import set_seed, safe_mkdir, save_json


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--packet_csv", required=True)
    parser.add_argument("--flow_csv", required=True)
    parser.add_argument("--config", default="configs/stage1_config.yaml")
    parser.add_argument("--out_dir", default="./stage1_artifacts")

    parser.add_argument("--external_packet_csv", default=None)
    parser.add_argument("--external_flow_csv", default=None)

    parser.add_argument("--export_embeddings", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    cfg = load_config(args.config)
    seed = int(cfg.get("seed", 42))

    set_seed(seed)
    safe_mkdir(args.out_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")

    loaders, preprocessor, metadata = build_dataloaders(
        packet_csv=args.packet_csv,
        flow_csv=args.flow_csv,
        cfg=cfg,
        out_dir=args.out_dir,
        external_packet_csv=args.external_packet_csv,
        external_flow_csv=args.external_flow_csv,
    )

    print("[INFO] preprocessor summary:")
    print(metadata["preprocessor"])

    input_dim = preprocessor.input_dim()
    model = Stage1TimeAwareTransformer(input_dim=input_dim, cfg=cfg).to(device)

    run_summary = train_model(
        model=model,
        loaders=loaders,
        cfg=cfg,
        out_dir=args.out_dir,
        device=device,
    )

    save_json(run_summary, os.path.join(args.out_dir, "stage1_run_summary.json"))

    if args.export_embeddings:
        # Export embeddings for all three splits.
        for split_name in ["train", "val", "test"]:
            out_path = os.path.join(args.out_dir, f"stage1_{split_name}_embeddings.npz")
            export_embeddings(
                model=model,
                loader=loaders[split_name],
                device=device,
                output_npz_path=out_path,
            )
            print(f"[INFO] saved embeddings: {out_path}")


if __name__ == "__main__":
    main()
