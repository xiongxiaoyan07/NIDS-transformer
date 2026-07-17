from __future__ import annotations

from typing import Any, Dict, Tuple, List, Optional

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
    def __init__(self, d_model: int, max_len: int = 512, position_mode: str = "age"):
        super().__init__()
        if position_mode not in {"age", "order", "absolute"}:
            raise ValueError(
                f"Unknown model.position_mode: {position_mode}. "
                "Choose from: age, order, absolute."
            )
        self.position_mode = position_mode

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

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x:    [B, L, D]
        mask: [B, L], True means valid token.

        position_mode:
            absolute: physical tensor positions 0..L-1.
            order:    valid tokens get positions 0..len-1 from oldest to target.
            age:      target/current flow gets 0; older history gets larger ids.
        """
        length = x.size(1)

        if mask is None or self.position_mode == "absolute":
            if length > self.pe.size(1):
                raise ValueError(
                    f"Sequence length {length} > max positional length {self.pe.size(1)}. "
                    "Increase model.max_len."
                )
            return x + self.pe[:, :length, :]

        if mask.shape != x.shape[:2]:
            raise ValueError(f"mask shape {tuple(mask.shape)} does not match x shape {tuple(x.shape[:2])}")

        valid_order = mask.long().cumsum(dim=1) - 1
        lengths = mask.long().sum(dim=1, keepdim=True).clamp(min=1)

        if self.position_mode == "age":
            position_ids = lengths - 1 - valid_order
        else:
            position_ids = valid_order

        position_ids = position_ids.masked_fill(~mask, 0).clamp(min=0)
        max_position_id = int(position_ids.max().item()) if position_ids.numel() > 0 else 0
        if max_position_id >= self.pe.size(1):
            raise ValueError(
                f"Position id {max_position_id} >= max positional length {self.pe.size(1)}. "
                "Increase model.max_len."
            )

        pe = self.pe[0][position_ids]
        return x + pe * mask.unsqueeze(-1).to(dtype=x.dtype)


# ============================================================
# Attention Pooling
# ============================================================

class AttentionPooling(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.score = nn.Linear(d_model, 1)

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores = self.score(h).squeeze(-1)
        scores = scores.masked_fill(~mask, -1e4)
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


class Stage2GRU(nn.Module):
    """
    GRU context baseline.

    Uses the same Stage2 inputs and pooling options as Stage2LSTM, but replaces
    the recurrent encoder with GRU.
    """

    def __init__(self, cfg: Dict[str, Any], input_dim: int):
        super().__init__()

        model_cfg = cfg["model"]

        d_model = model_cfg.get("d_model") or input_dim
        hidden_dim = int(model_cfg.get("gru_hidden_dim", model_cfg.get("lstm_hidden_dim", d_model)))
        num_layers = int(model_cfg.get("gru_num_layers", model_cfg.get("lstm_num_layers", model_cfg.get("num_layers", 2))))
        dropout = float(model_cfg.get("dropout", 0.3))
        bidirectional = bool(model_cfg.get("gru_bidirectional", model_cfg.get("lstm_bidirectional", False)))
        pooling = model_cfg.get("pooling", "last")
        num_classes = int(model_cfg.get("num_classes", 2))
        cls_head_config = int(model_cfg.get("cls_head", 0))

        if pooling not in {"last", "mean", "attention"}:
            raise ValueError(f"Unknown model.pooling for GRU: {pooling}")

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

        self.gru = nn.GRU(
            input_size=self.d_model,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        out_dim = hidden_dim * (2 if bidirectional else 1)
        if self.pooling == "attention":
            self.att_pool = AttentionPooling(out_dim)

        self.cls_head = build_cls_head(
            d_model=out_dim,
            dropout=dropout,
            num_classes=num_classes,
            cls_head_config=cls_head_config,
        )

    def forward(self, context_z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x, lengths = compact_left_padded_sequences(context_z, mask)
        x = self.input_proj(x)

        packed = pack_padded_sequence(
            x,
            lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_out, h_n = self.gru(packed)

        h, _ = pad_packed_sequence(
            packed_out,
            batch_first=True,
            total_length=x.size(1),
        )

        B, L, _ = h.shape
        device = h.device
        right_mask = (
            torch.arange(L, device=device).unsqueeze(0)
            < lengths.to(device).unsqueeze(1)
        )

        if self.pooling == "last":
            if self.bidirectional:
                pooled = torch.cat([h_n[-2], h_n[-1]], dim=-1)
            else:
                pooled = h_n[-1]
        elif self.pooling == "mean":
            pooled = masked_mean_pooling(h, right_mask)
        else:
            pooled = self.att_pool(h, right_mask)

        return self.cls_head(pooled)


class Stage2CNNLSTMCompat(nn.Module):
    """
    CNN+LSTM context baseline compatible with Stage2Trainer.

    Multi-kernel 1D CNN extracts local context patterns before an LSTM encoder.
    The output is raw class logits with shape [B, num_classes].
    """

    def __init__(self, cfg: Dict[str, Any], input_dim: int):
        super().__init__()

        model_cfg = cfg["model"]

        d_model = model_cfg.get("d_model") or input_dim
        dropout = float(model_cfg.get("dropout", 0.3))
        pooling = model_cfg.get("pooling", "last")
        num_classes = int(model_cfg.get("num_classes", 2))
        cls_head_config = int(model_cfg.get("cls_head", 0))
        cnn_kernel_sizes = model_cfg.get("cnn_kernel_sizes", [3, 5, 7])
        cnn_out_channels = int(model_cfg.get("cnn_out_channels", d_model))
        hidden_dim = int(model_cfg.get("cnn_lstm_hidden_dim", model_cfg.get("lstm_hidden_dim", d_model)))
        num_layers = int(model_cfg.get("cnn_lstm_num_layers", model_cfg.get("lstm_num_layers", model_cfg.get("num_layers", 2))))
        bidirectional = bool(model_cfg.get("cnn_lstm_bidirectional", model_cfg.get("lstm_bidirectional", False)))

        if pooling not in {"last", "mean", "attention"}:
            raise ValueError(f"Unknown model.pooling for CNN+LSTM: {pooling}")

        self.input_dim = int(input_dim)
        self.d_model = int(d_model)
        self.pooling = pooling
        self.bidirectional = bidirectional

        self.input_proj = (
            nn.Identity()
            if self.input_dim == self.d_model
            else nn.Linear(self.input_dim, self.d_model)
        )

        self.convs = nn.ModuleList([
            nn.Conv1d(self.d_model, cnn_out_channels, int(k), padding=int(k) // 2)
            for k in cnn_kernel_sizes
        ])

        cnn_total = cnn_out_channels * len(cnn_kernel_sizes)
        self.cnn_project = nn.Sequential(
            nn.Linear(cnn_total, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.lstm = nn.LSTM(
            input_size=self.d_model,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        out_dim = hidden_dim * (2 if bidirectional else 1)
        if self.pooling == "attention":
            self.att_pool = AttentionPooling(out_dim)

        self.cls_head = build_cls_head(
            d_model=out_dim,
            dropout=dropout,
            num_classes=num_classes,
            cls_head_config=cls_head_config,
        )

    def forward(self, context_z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x, lengths = compact_left_padded_sequences(context_z, mask)
        x = self.input_proj(x)

        B, L, _ = x.shape
        device = x.device
        right_mask = (
            torch.arange(L, device=device).unsqueeze(0)
            < lengths.to(device).unsqueeze(1)
        )
        valid_mask = right_mask.unsqueeze(-1).to(dtype=x.dtype)

        z = x * valid_mask
        z_t = z.transpose(1, 2)
        conv_outputs = [conv(z_t).transpose(1, 2) for conv in self.convs]
        z = torch.cat(conv_outputs, dim=-1)
        z = self.cnn_project(z) * valid_mask

        packed = pack_padded_sequence(
            z,
            lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_out, (h_n, c_n) = self.lstm(packed)

        h, _ = pad_packed_sequence(
            packed_out,
            batch_first=True,
            total_length=L,
        )

        if self.pooling == "last":
            if self.bidirectional:
                pooled = torch.cat([h_n[-2], h_n[-1]], dim=-1)
            else:
                pooled = h_n[-1]
        elif self.pooling == "mean":
            pooled = masked_mean_pooling(h, right_mask)
        else:
            pooled = self.att_pool(h, right_mask)

        return self.cls_head(pooled)


class Stage2CNNLSTM(nn.Module):
    """
    CNN+LSTM Baseline for Stage 2

    动机：
    1. CNN 捕捉局部 n-gram 流模式
    2. LSTM 捕捉长距离时序依赖
    3. 是时间序列分类的强基线
    """

    def __init__(
            self,
            d_model: int,
            num_layers: int = 2,
            dropout: float = 0.1,
            pooling: str = "last",
            cls_head: List[int] = [128, 64],
            activation: str = "gelu",
            class_alpha: Optional[torch.Tensor] = None,

            # CNN specific
            cnn_kernel_sizes: List[int] = [3, 5, 7],
            cnn_out_channels: int = 128,
            lstm_hidden_size: Optional[int] = None,
            lstm_num_layers: int = 2,
    ):
        super().__init__()
        self.pooling = pooling

        # Multi-scale CNN
        self.convs = nn.ModuleList([
            nn.Conv1d(d_model, cnn_out_channels, k, padding=k // 2)
            for k in cnn_kernel_sizes
        ])

        # Nonlinear projection
        cnn_total = cnn_out_channels * len(cnn_kernel_sizes)
        self.cnn_project = nn.Sequential(
            nn.Linear(cnn_total, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # LSTM encoder
        lstm_hidden = lstm_hidden_size or d_model
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=lstm_hidden,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_num_layers > 1 else 0,
        )

        # 输出维度 × 2 (bidirectional)
        lstm_output_dim = lstm_hidden * 2
        self.proj = nn.Linear(lstm_output_dim, d_model) if lstm_output_dim != d_model else nn.Identity()

        # Pooling
        if pooling == "attention":
            self.pooler = AttentionPooling(d_model)
        else:
            self.pooler = None

        # Classification head
        self.classifier = self._build_classifier(d_model, cls_head, dropout, activation)

        # Class balancing
        self.class_alpha = class_alpha

    def _build_classifier(self, d_model, cls_head, dropout, activation):
        layers = []
        in_dim = d_model
        for h_dim in cls_head:
            layers.append(nn.Linear(in_dim, h_dim))
            if activation == "gelu":
                layers.append(nn.GELU())
            else:
                layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 1))
        return nn.Sequential(*layers)

    def forward(self, context_z, mask, return_logits=False):
        """
        context_z: [B, L, d_model]
        mask: [B, L]
        """
        B, L, D = context_z.shape

        # 1. Pad mask for CNN
        valid_mask = mask.unsqueeze(-1).float()  # [B, L, 1]

        # 2. CNN feature extraction
        z = context_z * valid_mask
        z_t = z.transpose(1, 2)  # [B, D, L]

        conv_outputs = []
        for conv in self.convs:
            conv_out = conv(z_t)  # [B, C, L]
            conv_out = conv_out.transpose(1, 2)  # [B, L, C]
            conv_outputs.append(conv_out)

        # Concatenate multi-scale features
        z = torch.cat(conv_outputs, dim=-1)  # [B, L, C_total]

        # Project back to d_model
        z = self.cnn_project(z)  # [B, L, d_model]
        z = z * valid_mask  # Re-apply mask

        # 3. Pack for LSTM
        lengths = mask.sum(dim=1).cpu()
        nonzero_mask = lengths > 0
        if not nonzero_mask.all():
            z = z[nonzero_mask]
            lengths = lengths[nonzero_mask]
            valid_mask = valid_mask[nonzero_mask]

        packed = pack_padded_sequence(
            z, lengths, batch_first=True, enforce_sorted=False
        )
        lstm_out, _ = self.lstm(packed)

        # Back to dense but with original order
        z_dense, _ = pad_packed_sequence(
            lstm_out, batch_first=True, total_length=L
        )
        z_dense = z_dense * valid_mask

        # Project if needed
        z_dense = self.proj(z_dense)

        # 4. Pooling
        if self.pooling == "last":
            lengths = mask.sum(dim=1).long()
            lengths = torch.clamp(lengths - 1, min=0)
            z_pooled = z_dense[torch.arange(len(z_dense)), lengths]
        elif self.pooling == "mean":
            z_pooled = (z_dense * valid_mask).sum(1) / valid_mask.sum(1).clamp(min=1)
        elif self.pooling == "attention":
            z_pooled = self.pooler(z_dense, mask)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

        # 5. Classification
        logits = self.classifier(z_pooled).squeeze(-1)

        if return_logits:
            return logits
        return torch.sigmoid(logits)

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

        self.use_positional_encoding = bool(model_cfg.get("use_positional_encoding", True))
        self.position_mode = str(model_cfg.get("position_mode", "age"))
        self.pos = (
            SinusoidalPositionalEncoding(
                self.d_model,
                max_len=int(model_cfg.get("max_len", 512)),
                position_mode=self.position_mode,
            )
            if self.use_positional_encoding
            else None
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

        print(
            f"[INFO] Stage2Transformer.__init__*********** "
            f"use_positional_encoding={self.use_positional_encoding}, "
            f"position_mode={self.position_mode}, "
            f"dim_feedforward={model_cfg.get('dim_feedforward', 512)}, "
            f"nhead={model_cfg.get('nhead', 8)}, dropout={dropout}, "
            f"num_layers={model_cfg.get('num_layers', 2)}, pooling={self.pooling}, "
            f"num_classes={num_classes}, cls_head_config={cls_head_config}"
        )

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
        if self.use_positional_encoding:
            x = self.pos(x, mask)

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


class Stage2ResidualContextTransformer(nn.Module):
    """
    Residual context Transformer.

    final_logits = base_logits(current_z) + context_scale * delta_logits(context)

    This design keeps the strong no-context current-flow classifier
    and lets historical context only learn a correction term.
    """

    def __init__(self, cfg: Dict[str, Any], input_dim: int):
        super().__init__()

        model_cfg = cfg["model"]

        d_model = model_cfg.get("d_model") or input_dim
        dropout = float(model_cfg.get("dropout", 0.25))
        num_classes = int(model_cfg.get("num_classes", 2))
        cls_head_config = int(model_cfg.get("cls_head", 1))

        self.input_dim = int(input_dim)
        self.d_model = int(d_model)
        self.context_scale = float(model_cfg.get("context_scale", 0.5))
        self.pooling = model_cfg.get("pooling", "last")

        self.input_proj = (
            nn.Identity()
            if self.input_dim == self.d_model
            else nn.Linear(self.input_dim, self.d_model)
        )

        self.use_positional_encoding = bool(model_cfg.get("use_positional_encoding", True))
        self.position_mode = str(model_cfg.get("position_mode", "age"))

        self.pos = (
            SinusoidalPositionalEncoding(
                self.d_model,
                max_len=int(model_cfg.get("max_len", 512)),
                position_mode=self.position_mode,
            )
            if self.use_positional_encoding
            else None
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(model_cfg.get("nhead", 4)),
            dim_feedforward=int(model_cfg.get("dim_feedforward", 256)),
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=int(model_cfg.get("num_layers", 1)),
        )

        # Strong no-context branch
        self.base_head = build_cls_head(
            d_model=self.d_model,
            cls_head_config=cls_head_config,
            dropout=dropout,
            num_classes=num_classes,
        )

        # Context correction branch
        self.delta_head = nn.Sequential(
            nn.LayerNorm(self.d_model * 3),
            nn.Dropout(dropout),
            nn.Linear(self.d_model * 3, self.d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model, num_classes),
        )

        # Gate controls how much context correction is used
        self.gate = nn.Sequential(
            nn.LayerNorm(self.d_model * 3),
            nn.Linear(self.d_model * 3, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, 1),
            nn.Sigmoid(),
        )

        print(
            "[INFO] Stage2ResidualContextTransformer: "
            f"d_model={self.d_model}, nhead={model_cfg.get('nhead', 4)}, "
            f"num_layers={model_cfg.get('num_layers', 1)}, "
            f"dropout={dropout}, context_scale={self.context_scale}"
        )

    def _history_mask_without_current(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Remove the current-flow token from the mask.
        Since current flow is the last valid token, set that position to False.
        """
        history_mask = mask.clone()
        B, L = mask.shape
        positions = torch.arange(L, device=mask.device).unsqueeze(0).expand(B, L)
        last_idx = positions.masked_fill(~mask, -1).max(dim=1).values.clamp(min=0)
        history_mask[torch.arange(B, device=mask.device), last_idx] = False
        return history_mask

    def forward(self, context_z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # current-flow representation before context encoding
        x = self.input_proj(context_z)
        current_z = last_valid_token(x, mask)

        base_logits = self.base_head(current_z)

        # Use only previous flows for context branch
        history_mask = self._history_mask_without_current(mask)
        no_history = ~history_mask.any(dim=1)
        has_no_history = bool(no_history.any().item())

        if self.use_positional_encoding:
            x_ctx = self.pos(x, mask)
        else:
            x_ctx = x

        safe_history_mask = history_mask
        if has_no_history:
            safe_history_mask = history_mask.clone()
            # Transformer attention cannot receive a row with all keys masked.
            # Use the current token as a temporary finite key, then zero out
            # the context residual for these rows below.
            B, L = mask.shape
            positions = torch.arange(L, device=mask.device).unsqueeze(0).expand(B, L)
            current_idx = positions.masked_fill(~mask, -1).max(dim=1).values.clamp(min=0)
            row_idx = torch.arange(B, device=mask.device)
            safe_history_mask[row_idx[no_history], current_idx[no_history]] = True

        key_padding_mask = ~safe_history_mask

        h = self.encoder(
            x_ctx,
            src_key_padding_mask=key_padding_mask,
        )

        # Summarize previous flows only
        context_summary = masked_mean_pooling(h, safe_history_mask)
        if has_no_history:
            context_summary = context_summary.masked_fill(no_history.unsqueeze(-1), 0.0)

        fused = torch.cat(
            [
                current_z,
                context_summary,
                current_z - context_summary,
            ],
            dim=-1,
        )

        delta_logits = self.delta_head(fused)
        gate = self.gate(fused)
        if has_no_history:
            gate = gate.masked_fill(no_history.unsqueeze(-1), 0.0)

        logits = base_logits + self.context_scale * gate * delta_logits
        return logits
# ============================================================
# Model factory
# ============================================================

def build_stage2_model(cfg: Dict[str, Any], input_dim: int) -> nn.Module:
    model_cfg = cfg["model"]
    model_type = model_cfg.get("model_type", "transformer")

    if model_type == "no_context_mlp":
        return Stage2NoContextMLP(cfg, input_dim=input_dim)

    if model_type in {"target_query_gated", "target_query"}:
        from .target_query import Stage2TargetQueryGatedAttention

        return Stage2TargetQueryGatedAttention(cfg, input_dim=input_dim)

    if model_type in {"target_query_residual", "target_query_residual_attention"}:
        from .target_query import Stage2TargetQueryResidualAttention

        return Stage2TargetQueryResidualAttention(cfg, input_dim=input_dim)

    if model_type == "residual_transformer":
        return Stage2ResidualContextTransformer(cfg, input_dim)

    if model_type == "lstm":
        return Stage2LSTM(cfg, input_dim=input_dim)

    if model_type == "gru":
        return Stage2GRU(cfg, input_dim=input_dim)

    if model_type in {"cnn_lstm", "cnnlstm"}:
        return Stage2CNNLSTMCompat(cfg, input_dim=input_dim)

    if model_type == "transformer":
        return Stage2Transformer(cfg, input_dim=input_dim)

    raise ValueError(
        f"Unknown model.model_type: {model_type}. "
        "Choose from: no_context_mlp, target_query_gated, target_query_residual, "
        "residual_transformer, lstm, gru, cnn_lstm, transformer."
    )
