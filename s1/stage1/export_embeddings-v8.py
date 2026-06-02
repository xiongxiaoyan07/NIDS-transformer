"""
Export Stage1 flow embeddings z_i.

These embeddings can be used as Stage2 inter-flow Transformer input.
"""

from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def export_embeddings(model, loader, device, output_npz_path: str) -> None:
    model.eval()

    all_flow_ids = []
    all_z = []
    all_labels = []

    for batch in loader:
        x = batch["x"].to(device)
        t = batch["time"].to(device)
        mask = batch["mask"].to(device)

        logits, z, _ = model(x, t, mask, return_embedding=True)

        all_flow_ids.extend(batch["flow_id"].cpu().numpy().tolist())
        all_labels.extend(batch["label"].cpu().numpy().tolist())
        all_z.append(z.detach().cpu().numpy())

    z_mat = np.concatenate(all_z, axis=0)

    np.savez_compressed(
        output_npz_path,
        flow_id=np.array(all_flow_ids),
        label=np.array(all_labels),
        z=z_mat,
    )
