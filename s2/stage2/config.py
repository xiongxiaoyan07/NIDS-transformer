from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional

import yaml


DEFAULT_CFG: Dict[str, Any] = {
    "seed": 42,
    "data": {
        "stage1_dir": None,
        "embedding_files": {
            "train": "stage1_train_embeddings.npz",
            "val": "stage1_val_embeddings.npz",
            "test": "stage1_test_embeddings.npz",
        },
        "metadata_csv": "stage1_flow_metadata.csv",
        "timestamp_col": "flow_start_timestamp_us",
        "source_col": "source_id",
        "destination_col": "destination_id",
    },
    "context": {
        # time_only / source_host / destination_host / endpoint / source_destination
        "method": "time_only",
        "window_size": 16,
        "include_target": True,

        # online:
        #   context can use any earlier flow in global chronological order.
        # split_isolated:
        #   context only uses earlier flows from the same split.
        # train_only_for_eval:
        #   train target uses earlier train flows;
        #   val/test target uses earlier train flows only.
        "context_policy": "split_isolated",

        # endpoint options:
        #   same_endpoint: previous flows touching either current endpoint.
        #   same_source_or_dest: previous flows with same source OR same destination only.
        "endpoint_mode": "same_endpoint",
        "deduplicate": True,
    },
    "model": {
        # None means use Stage1 z dimension directly.
        # model_type can be transformer / lstm / gru / cnn_lstm /
        # no_context_mlp / target_query_gated /
        # target_query_residual / relation_aware_attention /
        # source_destination_attention / residual_transformer.
        "d_model": None,
        "nhead": 8,
        "num_layers": 2,
        "dim_feedforward": 512,
        "dropout": 0.3,
        "num_classes": 2,
        "pooling": "last",  # last / mean / attention
        "use_positional_encoding": True,
        "max_len": 512,
    },
    "training": {
        "epochs": 500,
        "batch_size": 64,
        "lr": 3.0e-4,
        "weight_decay": 1.0e-4,
        "num_workers": 0,
        "patience": 20,
        "metric_for_best": "f1_label1",
        "use_weighted_sampler": True,
        "class_weighted_loss": True,
        "grad_clip_norm": 1.0,
        "threshold": 0.5,
        "auto_threshold": True,
        "threshold_metric": "f1_label1",
        "threshold_min": 0.01,
        "threshold_max": 0.99,
        "threshold_steps": 199,
        "amp": True,
        "device": "auto",
    },
}
def deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    out = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    cfg = deepcopy(DEFAULT_CFG)
    if path is None:
        return cfg

    with open(path, "r", encoding="utf-8") as f:
        user_cfg = yaml.safe_load(f) or {}

    return deep_update(cfg, user_cfg)


def save_config(cfg: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
