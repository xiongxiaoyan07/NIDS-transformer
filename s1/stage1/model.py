"""
Stage1 model:
- record-level projection
- positional encoding
- time-aware encoding based on flow_iat_us
- Transformer encoder
- masked pooling
- classifier
"""

from __future__ import annotations

import math
from typing import Dict, Any

import torch
import torch.nn as nn


class RecordLevelProjection(nn.Module):
    """
    Input encoding:
        e_i,t = W x_i,t + b

    This is the record-level projection before Transformer.
    """

    def __init__(self, input_dim: int, d_model: int, dropout: float):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class TimeAwareEncoding(nn.Module):
    """
    Add position and continuous-time information.

    e_tilde_i,t = e_i,t + PE(t) + TimeMLP(log(1 + flow_iat_us_t))

    Config switches:
        use_positional_encoding
        use_time_encoding
    """

    def __init__(
        self,
        d_model: int,
        max_len: int,
        dropout: float,
        use_positional_encoding: bool,
        use_time_encoding: bool,
    ):
        super().__init__()
        self.use_positional_encoding = use_positional_encoding
        self.use_time_encoding = use_time_encoding
        self.dropout = nn.Dropout(dropout)

        # Sinusoidal position encoding.
        # Registered as buffer because it is not trainable.
        pe = self._build_sinusoidal_pe(max_len, d_model)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

        # Continuous-time encoding.
        # Input should be log(1 + flow_iat_us), shape [B, L].
        self.time_mlp = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    @staticmethod
    def _build_sinusoidal_pe(max_len: int, d_model: int) -> torch.Tensor:
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)

        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])

        return pe

    def forward(self, e: torch.Tensor, time_log: torch.Tensor) -> torch.Tensor:
        """
        Args:
            e:
                Projected packet features.
                Shape: [B, L, d_model]

            time_log:
                log(1 + flow_iat_us).
                Shape: [B, L]

        Returns:
            Encoded packet sequence.
            Shape: [B, L, d_model]
        """
        seq_len = e.size(1)
        out = e

        if self.use_positional_encoding:
            out = out + self.pe[:, :seq_len, :]

        if self.use_time_encoding:
            out = out + self.time_mlp(time_log.unsqueeze(-1))

        return self.dropout(out)


class Stage1TimeAwareTransformer(nn.Module):
    """
    Stage1 intra-flow packet-sequence Transformer.

    Input:
        x:
            [B, L, input_dim]
            x_i,t = [packet header features ; flow-level features ; temporal features]

        time_log:
            [B, L]
            log(1 + flow_iat_us)

        mask:
            [B, L]
            True  = real packet
            False = padding

    Output:
        logits:
            [B, 2]

        optionally:
            z:
                [B, d_model], flow-level embedding for Stage2.
            h:
                [B, L, d_model], packet-level hidden states.
    """

    def __init__(self, input_dim: int, cfg: Dict[str, Any]):
        super().__init__()

        model_cfg = cfg.get("model", {})
        seq_cfg = cfg.get("sequence", {})

        d_model = int(model_cfg.get("d_model", 128))
        nhead = int(model_cfg.get("nhead", 4))
        num_layers = int(model_cfg.get("num_layers", 2))
        dim_feedforward = int(model_cfg.get("dim_feedforward", 256))
        dropout = float(model_cfg.get("dropout", 0.1))
        max_seq_len = int(seq_cfg.get("max_seq_len", 64))

        use_positional_encoding = bool(model_cfg.get("use_positional_encoding", True))
        use_time_encoding = bool(model_cfg.get("use_time_encoding", True))

        self.projection = RecordLevelProjection(
            input_dim=input_dim,
            d_model=d_model,
            dropout=dropout,
        )

        self.time_encoding = TimeAwareEncoding(
            d_model=d_model,
            max_len=max_seq_len,
            dropout=dropout,
            use_positional_encoding=use_positional_encoding,
            use_time_encoding=use_time_encoding,
        )

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 2),
        )

    def forward(
        self,
        x: torch.Tensor,
        time_log: torch.Tensor,
        mask: torch.Tensor,
        return_embedding: bool = False,
    ):
        """
        Args:
            x:        [B, L, input_dim]
            time_log: [B, L]
            mask:     [B, L], True for real packet, False for padding
        """
        e = self.projection(x)
        e = self.time_encoding(e, time_log)

        # PyTorch wants True where tokens should be ignored.
        src_key_padding_mask = ~mask.bool()

        h = self.encoder(e, src_key_padding_mask=src_key_padding_mask)
        z = self.masked_mean_pool(h, mask)

        logits = self.classifier(z)

        if return_embedding:
            return logits, z, h

        print("[INFO] model.py ------ forward")

        return logits

    @staticmethod
    def masked_mean_pool(h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_float = mask.float().unsqueeze(-1)
        h = h * mask_float
        denom = mask_float.sum(dim=1).clamp(min=1.0)
        return h.sum(dim=1) / denom
