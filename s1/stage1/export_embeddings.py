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
    # 我知道为什么train中有重复数据了，那是因为loader，train的loader
    # "train": DataLoader(
    #             datasets["train"],
    #             batch_size=batch_size,
    #             sampler=sampler,  # 使用sampler而不是shuffle
    #             num_workers=num_workers,
    #             collate_fn=custom_collate_fn,
    #             pin_memory=True,
    #         ),
    # sampler = WeightedRandomSampler(
    #     weights=sample_weights,
    #     num_samples=len(sample_weights),
    #     replacement=True
    # )
    # WeightedRandomSampler用于在数据加载阶段实现加权随机采样的工具，其最核心的用途是针对不平衡数据集，通过在训练时调整样本被选中的概率，来解决模型预测偏向多数类的问题
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

    # 检查重复
    # unique_ids, counts = np.unique(np.array(all_flow_ids, dtype=np.int64), return_counts=True)
    # dup_ids = unique_ids[counts > 1]
    # print("[INFO] Stage1 export_embeddings: ", split_name)
    # #train: 这里已经看是有重复的了-----修改之后这里就不存在重复的flow_id了
    # if len(dup_ids) > 0:
    #     total = (counts[counts > 1] - 1).sum()  # 重复的总条目数（排除第一次出现）
    #     examples = dup_ids[:10].tolist()
    #     print(f"[INFO] Stage1 export_embeddings---------------Duplicated flow_id in {split_name}. Total duplicated: {total} Examples: {examples}")

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
