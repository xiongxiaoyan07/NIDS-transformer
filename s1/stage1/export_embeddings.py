"""
Export Stage1 flow embeddings z_i.

These embeddings can be used as Stage2 inter-flow Transformer input.
"""

from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def export_embeddings(
    model,
    loader,
    device,
    output_npz_path: str,
    split_name: str | None = None,
) -> None:
    """
    Export Stage1 z_intra embeddings.

    Saved fields:
        flow_id
        label
        z
        split
        flow_start_timestamp_us
        source_id
        destination_id

    Notes:
        - z is the Stage1 pooled flow-level representation.
        - source_id/destination_id/timestamp are raw metadata for Stage2.
        - They are not Stage1 model features.
    """
    model.eval()

    all_flow_ids = []
    all_z = []
    all_labels = []
    all_splits = []

    all_flow_start_ts = []
    all_source_ids = []
    all_destination_ids = []

    for batch in loader:
        x = batch["x"].to(device)
        t = batch["time"].to(device)
        mask = batch["mask"].to(device)

        flow_feats = batch.get("flow_feats")
        if flow_feats is not None:
            flow_feats = flow_feats.to(device)

        logits, z, _ = model(
            x,
            t,
            mask,
            flow_feats=flow_feats,
            return_embedding=True,
        )

        batch_flow_ids = batch["flow_id"].cpu().numpy().tolist()
        batch_labels = batch["label"].cpu().numpy().tolist()

        all_flow_ids.extend(batch_flow_ids)
        all_labels.extend(batch_labels)
        all_z.append(z.detach().cpu().numpy())

        if split_name is not None:
            all_splits.extend([split_name] * len(batch_flow_ids))

        if "flow_start_timestamp_us" in batch:
            all_flow_start_ts.extend(
                batch["flow_start_timestamp_us"].cpu().numpy().tolist()
            )

        if "source_id" in batch:
            all_source_ids.extend([str(x) for x in batch["source_id"]])

        if "destination_id" in batch:
            all_destination_ids.extend([str(x) for x in batch["destination_id"]])

    z_mat = np.concatenate(all_z, axis=0)

    save_dict = {
        "flow_id": np.array(all_flow_ids, dtype=np.int64),
        "label": np.array(all_labels, dtype=np.int64),
        "z": z_mat.astype(np.float32),
    }

    if split_name is not None:
        save_dict["split"] = np.array(all_splits)

    if len(all_flow_start_ts) == len(all_flow_ids):
        save_dict["flow_start_timestamp_us"] = np.array(
            all_flow_start_ts,
            dtype=np.int64,
        )

    if len(all_source_ids) == len(all_flow_ids):
        save_dict["source_id"] = np.array(all_source_ids)

    if len(all_destination_ids) == len(all_flow_ids):
        save_dict["destination_id"] = np.array(all_destination_ids)

    np.savez_compressed(output_npz_path, **save_dict)

    print(f"[INFO] Exported Stage1 embeddings: {output_npz_path}")
    print(f"[INFO] Keys: {list(save_dict.keys())}")
    print(f"[INFO] z shape: {z_mat.shape}")
