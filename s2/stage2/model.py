from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# ============================================================
# Utility functions
# ============================================================

def last_valid_token(h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    h:    [B, L, D]
    mask: [B, L], True means valid token.

    Works for both left padding and right padding.
    Returns the last valid token representation for each sample.
    """
    B, L, _ = h.shape
    device = h.device

    positions = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
    masked_positions = positions.masked_fill(~mask, -1)
    last_idx = masked_positions.max(dim=1).values.clamp(min=0)

    return h[torch.arange(B, device=device), last_idx]


def masked_mean_pooling(h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    h:    [B, L, D]
    mask: [B, L]
    """
    denom = mask.float().sum(dim=1, keepdim=True).clamp(min=1.0)
    return (h * mask.unsqueeze(-1).float()).sum(dim=1) / denom


def compact_left_padded_sequences(
    x: torch.Tensor,
    mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Your collate_fn uses left padding:
        [PAD] [PAD] z1 z2 z3

    PyTorch pack_padded_sequence expects valid tokens to start from index 0:
        z1 z2 z3 [PAD] [PAD]

    This function converts left-padded sequences into right-padded sequences.

    Args:
        x:    [B, L, D]
        mask: [B, L], True means valid token

    Returns:
        compact_x: [B, L_valid_max, D], valid tokens right-padded
        lengths:   [B]
    """
    B, L, D = x.shape
    lengths = mask.long().sum(dim=1).clamp(min=1)
    max_len = int(lengths.max().item())

    compact_x = x.new_zeros(B, max_len, D)

    for i in range(B):
        valid_tokens = x[i, mask[i]]
        cur_len = valid_tokens.size(0)

        if cur_len == 0:
            # Theoretically should not happen if include_target=True.
            # Keep a zero token to avoid empty sequence.
            compact_x[i, 0] = 0.0
        else:
            compact_x[i, :cur_len] = valid_tokens

    return compact_x, lengths


# ============================================================
# Positional Encoding
# ============================================================
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


# ============================================================
# Attention Pooling
# ============================================================

class AttentionPooling(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.score = nn.Linear(d_model, 1)

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores = self.score(h).squeeze(-1)
        scores = scores.masked_fill(~mask, -1e9)
        weights = torch.softmax(scores, dim=1)
        return torch.sum(h * weights.unsqueeze(-1), dim=1)

# ============================================================
# Shared classifier builder
# ============================================================

def build_cls_head(d_model: int,cls_head_config: int, dropout: float, num_classes: int = 2) -> nn.Module:
    """
    Keep the classifier style close to your original Stage2Transformer.
    """
    # ---- Classifier ----
    # dropout = float(model_cfg.get("dropout", 0.3))
    # cls_head_config = int(model_cfg.get("cls_head", 0))
    print("[INFO] Stage2Transformer.__init__*********** cls_head_config=",cls_head_config)
    if cls_head_config == 2:
        return nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model,d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )
    elif cls_head_config == 3:
        return nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.3),
            nn.Linear(d_model // 2, num_classes),
        )
    else:
        return nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

# ============================================================
# Baseline 1: No Context MLP
# ============================================================

class Stage2NoContextMLP(nn.Module):
    """
    No-context baseline.

    It ignores historical context and only uses the current flow's Stage1 embedding.

    Since ContextIndexBuilder appends the current flow at the end when include_target=True,
    and your collate_fn left-pads valid tokens to the right, the current flow is the
    last valid token.

    Forward signature is kept as:
        forward(context_z, mask) -> logits

    Therefore, Stage2Trainer can be reused without modification.
    """

    def __init__(self, cfg: Dict[str, Any], input_dim: int):
        super().__init__()

        model_cfg = cfg["model"]
        d_model = model_cfg.get("d_model") or input_dim
        dropout = float(model_cfg.get("dropout", 0.3))
        num_classes = int(model_cfg.get("num_classes", 2))
        cls_head_config = int(model_cfg.get("cls_head", 0))

        self.input_dim = int(input_dim)
        self.d_model = int(d_model)

        self.input_proj = (
            nn.Identity()
            if self.input_dim == self.d_model
            else nn.Linear(self.input_dim, self.d_model)
        )

        self.cls_head = build_cls_head(
            d_model=self.d_model,
            dropout=dropout,
            num_classes=num_classes,
            cls_head_config=cls_head_config
        )

    def forward(self, context_z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Only take current flow representation.
        # This is the last valid token when include_target=True.
        current_z = last_valid_token(context_z, mask)

        x = self.input_proj(current_z)
        return self.cls_head(x)


# ============================================================
# Baseline 2: LSTM Context Encoder
# ============================================================

class Stage2LSTM(nn.Module):
    """
    LSTM context baseline.

    It uses the same context_z and mask as Transformer, but replaces TransformerEncoder
    with LSTM. This makes the comparison fair:
        same Stage1 embedding
        same ContextIndexBuilder
        same Dataset
        same Trainer
        different context encoder
    """

    def __init__(self, cfg: Dict[str, Any], input_dim: int):
        super().__init__()

        model_cfg = cfg["model"]

        d_model = model_cfg.get("d_model") or input_dim
        hidden_dim = int(model_cfg.get("lstm_hidden_dim", d_model))
        num_layers = int(model_cfg.get("lstm_num_layers", model_cfg.get("num_layers", 2)))
        dropout = float(model_cfg.get("dropout", 0.3))
        bidirectional = bool(model_cfg.get("lstm_bidirectional", False))
        pooling = model_cfg.get("pooling", "last")
        num_classes = int(model_cfg.get("num_classes", 2))
        cls_head_config = int(model_cfg.get("cls_head", 0))

        if pooling not in {"last", "mean", "attention"}:
            raise ValueError(f"Unknown model.pooling for LSTM: {pooling}")

        self.input_dim = int(input_dim)
        self.d_model = int(d_model)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.pooling = pooling

        self.input_proj = (
            nn.Identity()
            if self.input_dim == self.d_model
            else nn.Linear(self.input_dim, self.d_model)
        )

        lstm_dropout = dropout if num_layers > 1 else 0.0

        self.lstm = nn.LSTM(
            input_size=self.d_model,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=bidirectional,
        )

        out_dim = hidden_dim * (2 if bidirectional else 1)

        if self.pooling == "attention":
            self.att_pool = AttentionPooling(out_dim)

        self.cls_head = build_cls_head(
            d_model=out_dim,
            dropout=dropout,
            num_classes=num_classes,
            cls_head_config=cls_head_config
        )

    def forward(self, context_z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Convert left-padded input into right-padded valid sequence.
        x, lengths = compact_left_padded_sequences(context_z, mask)

        x = self.input_proj(x)

        lengths_cpu = lengths.detach().cpu()

        packed = pack_padded_sequence(
            x,
            lengths_cpu,
            batch_first=True,
            enforce_sorted=False,
        )

        packed_out, (h_n, c_n) = self.lstm(packed)

        h, _ = pad_packed_sequence(
            packed_out,
            batch_first=True,
            total_length=x.size(1),
        )

        # h is now right-padded.
        B, L, _ = h.shape
        device = h.device
        right_mask = (
            torch.arange(L, device=device).unsqueeze(0)
            < lengths.to(device).unsqueeze(1)
        )

        if self.pooling == "last":
            if self.bidirectional:
                # Last layer forward and backward hidden states.
                pooled = torch.cat([h_n[-2], h_n[-1]], dim=-1)
            else:
                pooled = h_n[-1]

        elif self.pooling == "mean":
            pooled = masked_mean_pooling(h, right_mask)

        else:
            pooled = self.att_pool(h, right_mask)

        return self.cls_head(pooled)



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

        print(
            "[INFO] Stage2Transformer.__init__ use_positional_encoding=",
            model_cfg.get("use_positional_encoding", True),
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

        dropout = float(model_cfg.get("dropout", 0.3))
        num_classes = int(model_cfg.get("num_classes", 2))
        cls_head_config = int(model_cfg.get("cls_head", 0))

        print("[INFO] Stage2Transformer.__init__*********** pooling=",self.pooling)
        print("[INFO] Stage2Transformer.__init__*********** dropout=",dropout)

        self.cls_head = build_cls_head(
            d_model=self.d_model,
            dropout=dropout,
            num_classes=num_classes,
            cls_head_config=cls_head_config
        )

    def forward(self, context_z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # context_z: [B, L, input_dim]
        # mask: True means valid token.
        # Transformer src_key_padding_mask: True means padding token.
        x = self.input_proj(context_z)
        x = self.pos(x)

        key_padding_mask = ~mask
        #src_key_padding_mask True 代表需要被忽略的位置（即填充位置），False 代表需要保留的正常Token
        h = self.encoder(
            x,
            src_key_padding_mask=key_padding_mask,
        )

        if self.pooling == "last":
            # Correct for left padding.
            pooled = last_valid_token(h, mask)

        elif self.pooling == "mean":
            pooled = masked_mean_pooling(h, mask)

        else:
            pooled = self.att_pool(h, mask)

        return self.cls_head(pooled)


# ============================================================
# Model factory
# ============================================================

def build_stage2_model(cfg: Dict[str, Any], input_dim: int) -> nn.Module:
    model_cfg = cfg["model"]
    model_type = model_cfg.get("model_type", "transformer")

    if model_type == "no_context_mlp":
        return Stage2NoContextMLP(cfg, input_dim=input_dim)

    if model_type == "lstm":
        return Stage2LSTM(cfg, input_dim=input_dim)

    if model_type == "transformer":
        return Stage2Transformer(cfg, input_dim=input_dim)

    raise ValueError(
        f"Unknown model.model_type: {model_type}. "
        "Choose from: no_context_mlp, lstm, transformer."
    )







