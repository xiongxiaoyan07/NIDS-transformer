from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        position = torch.arange(max_len).float().unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)

        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.size(1)
        if length > self.pe.size(1):
            raise ValueError(
                f"Sequence length {length} > max positional length {self.pe.size(1)}. "
                "Increase model.max_len."
            )
        return x + self.pe[:, :length, :]

class AttentionPooling(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.score = nn.Linear(d_model, 1)

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores = self.score(h).squeeze(-1)
        scores = scores.masked_fill(~mask, -1e9)
        weights = torch.softmax(scores, dim=1)
        return torch.sum(h * weights.unsqueeze(-1), dim=1)

class Stage2Transformer(nn.Module):
    def __init__(self, cfg: Dict[str, Any], input_dim: int):
        super().__init__()
        model_cfg = cfg["model"]

        d_model = model_cfg.get("d_model") or input_dim
        self.input_dim = int(input_dim)
        self.d_model = int(d_model)
        self.pooling = model_cfg.get("pooling", "last")

        if self.pooling not in {"last", "mean", "attention"}:
            raise ValueError(f"Unknown model.pooling: {self.pooling}")

        self.input_proj = (
            nn.Identity()
            if self.input_dim == self.d_model
            else nn.Linear(self.input_dim, self.d_model)
        )

        self.pos = (
            SinusoidalPositionalEncoding(
                self.d_model,
                max_len=int(model_cfg.get("max_len", 512)),
            )
            if bool(model_cfg.get("use_positional_encoding", True))
            else nn.Identity()
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(model_cfg.get("nhead", 8)),
            dim_feedforward=int(model_cfg.get("dim_feedforward", 512)),
            dropout=float(model_cfg.get("dropout", 0.3)),
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=int(model_cfg.get("num_layers", 2)),
        )

        if self.pooling == "attention":
            self.att_pool = AttentionPooling(self.d_model)

        self.cls_head = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Dropout(float(model_cfg.get("dropout", 0.3))),
            nn.Linear(self.d_model, int(model_cfg.get("num_classes", 2))),
        )

    def forward(self, context_z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # context_z: [B, L, input_dim]
        # mask: True means valid token.
        # Transformer src_key_padding_mask: True means padding token.
        x = self.input_proj(context_z)
        x = self.pos(x)

        key_padding_mask = ~mask
        #src_key_padding_mask True 代表需要被忽略的位置（即填充位置），False 代表需要保留的正常Token
        h = self.encoder(x, src_key_padding_mask=key_padding_mask)

        if self.pooling == "last":
            lengths = mask.long().sum(dim=1).clamp(min=1)
            last_idx = lengths - 1
            pooled = h[torch.arange(h.size(0), device=h.device), last_idx]

        elif self.pooling == "mean":
            denom = mask.float().sum(dim=1, keepdim=True).clamp(min=1.0)
            pooled = (h * mask.unsqueeze(-1).float()).sum(dim=1) / denom

        else:
            pooled = self.att_pool(h, mask)

        return self.cls_head(pooled)










