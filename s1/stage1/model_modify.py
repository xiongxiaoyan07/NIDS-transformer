"""
Stage1 model:
- record-level projection
- time-aware encoding (paper: TE(pos+p) instead of PE(pos) + MLP(time))
- Transformer encoder
- masked pooling
- classifier
"""

from __future__ import annotations

import math
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .build_mlp import build_mlp


class RecordLevelProjection(nn.Module):
    """
    Input encoding:
        e_i,t = W x_i,t + b

    This is the record-level projection before Transformer.
    """

    def __init__(self, input_dim: int, d_model: int, dropout: float, mlp_cfg: dict | None = None):
        super().__init__()
        self.proj = build_mlp(
            input_dim=input_dim,
            output_dim=d_model,
            mlp_cfg=mlp_cfg,
            default_dropout=dropout,
            legacy_layer_factory=lambda: nn.Linear(input_dim, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class TimeAwareEncoding(nn.Module):
    """
    Configurable position/time encoding.

    Supported fusion modes:
      - joint:         original TE(pos + p), kept for paper reproduction.
      - joint_scaled:  TE(a_pos * pos + a_time * p), with learnable gains.
      - separate_add:  e + a_pos * PE(pos) + a_time * TE(p).
      - concat:        concatenate independent position/time encodings and mix.
      - gated:         token-wise, feature-wise gate between position and time.

    The time branch can optionally combine:
      1) cumulative elapsed time p_t = log(1 + sum_{k<=t} IAT_k + alpha)
      2) local packet IAT log(1 + IAT_t)

    This preserves absolute/long-range timing while also exposing burst-level timing.
    """

    VALID_FUSION_MODES = {
        "joint",
        "joint_scaled",
        "separate_add",
        "concat",
        "gated",
    }

    def __init__(
        self,
        d_model: int,
        max_len: int,
        dropout: float,
        use_positional_encoding: bool,
        use_time_encoding: bool,
        alpha: float = 1e-7,
        fusion_mode: str = "joint",
        use_local_iat: bool = False,
        normalize_local_iat: bool = True,
        post_encoding_norm: bool = False,
        learnable_gains: bool = True,
        local_time_hidden_dim: int | None = None,
        max_time_log: float = 30.0,
    ):
        super().__init__()

        fusion_mode = str(fusion_mode).lower()
        if fusion_mode not in self.VALID_FUSION_MODES:
            raise ValueError(
                f"Unknown time encoding fusion_mode={fusion_mode!r}. "
                f"Expected one of {sorted(self.VALID_FUSION_MODES)}"
            )

        self.use_positional_encoding = bool(use_positional_encoding)
        self.use_time_encoding = bool(use_time_encoding)
        self.alpha = float(alpha)
        self.fusion_mode = fusion_mode
        self.use_local_iat = bool(use_local_iat and use_time_encoding)
        self.normalize_local_iat = bool(normalize_local_iat)
        self.learnable_gains = bool(learnable_gains)
        self.max_time_log = float(max_time_log)
        self.d_model = int(d_model)

        self.dropout = nn.Dropout(dropout)
        self.post_norm = (
            nn.LayerNorm(d_model) if post_encoding_norm else nn.Identity()
        )

        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        self.register_buffer("div_term", div_term, persistent=False)

        position = torch.arange(0, max_len, dtype=torch.float32)
        self.register_buffer("position", position, persistent=False)

        # Gains are constrained to (0, 2). Initial raw value 0 gives gain 1.
        if self.learnable_gains:
            self.raw_pos_gain = nn.Parameter(torch.tensor(0.0))
            self.raw_time_gain = nn.Parameter(torch.tensor(0.0))
            self.raw_local_gain = nn.Parameter(torch.tensor(0.0))
            self.raw_residual_gain = nn.Parameter(torch.tensor(0.0))
        else:
            self.register_buffer("raw_pos_gain", torch.tensor(0.0), persistent=False)
            self.register_buffer("raw_time_gain", torch.tensor(0.0), persistent=False)
            self.register_buffer("raw_local_gain", torch.tensor(0.0), persistent=False)
            self.register_buffer("raw_residual_gain", torch.tensor(0.0), persistent=False)

        # Local IAT branch: use both absolute log-IAT and per-flow normalized log-IAT.
        if self.use_local_iat:
            hidden = local_time_hidden_dim or max(16, d_model // 2)
            self.local_time_encoder = nn.Sequential(
                nn.Linear(2, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, d_model),
                nn.LayerNorm(d_model),
            )
            self.time_component_mixer = nn.Sequential(
                nn.Linear(2 * d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.LayerNorm(d_model),
            )

        if self.use_positional_encoding and self.use_time_encoding:
            if self.fusion_mode == "concat":
                self.concat_fusion = nn.Sequential(
                    nn.Linear(2 * d_model, d_model),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.LayerNorm(d_model),
                )
            elif self.fusion_mode == "gated":
                # The original projected packet embedding helps decide which signal
                # should dominate for each packet and each latent dimension.
                self.fusion_gate = nn.Sequential(
                    nn.Linear(3 * d_model, d_model),
                    nn.GELU(),
                    nn.Linear(d_model, d_model),
                    nn.Sigmoid(),
                )
                self.gated_fusion_norm = nn.LayerNorm(d_model)

    @staticmethod
    def _gain(raw: torch.Tensor) -> torch.Tensor:
        """Map an unconstrained scalar to (0, 2), initialized at 1."""
        return 2.0 * torch.sigmoid(raw)

    def _sinusoidal_from_scalar(self, scalar: torch.Tensor) -> torch.Tensor:
        """
        Args:
            scalar: [B, L]
        Returns:
            encoding: [B, L, d_model]
        """
        div_term = self.div_term.to(device=scalar.device, dtype=scalar.dtype)
        arg = scalar.unsqueeze(-1) * div_term

        encoding = scalar.new_zeros(
            scalar.size(0), scalar.size(1), self.d_model
        )
        encoding[..., 0::2] = torch.sin(arg)

        odd_dim = encoding[..., 1::2].shape[-1]
        if odd_dim > 0:
            encoding[..., 1::2] = torch.cos(arg[..., :odd_dim])

        return encoding

    @staticmethod
    def _masked_standardize(
        values: torch.Tensor,
        mask: torch.Tensor,
        eps: float = 1e-5,
    ) -> torch.Tensor:
        mask_f = mask.to(dtype=values.dtype)
        count = mask_f.sum(dim=1, keepdim=True).clamp(min=1.0)
        mean = (values * mask_f).sum(dim=1, keepdim=True) / count
        variance = (
            ((values - mean) ** 2) * mask_f
        ).sum(dim=1, keepdim=True) / count
        normalized = (values - mean) / torch.sqrt(variance + eps)
        return normalized * mask_f

    def _compute_time_features(
        self,
        time_log: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            cumulative_coordinate: [B, L]
            local_log_iat:         [B, L]
        """
        mask_f = mask.to(dtype=time_log.dtype)

        # expm1 can overflow in reduced precision. Compute this branch in FP32.
        time_log_fp32 = time_log.float().clamp(min=0.0, max=self.max_time_log)
        time_log_fp32 = time_log_fp32 * mask.float()

        intervals = torch.expm1(time_log_fp32).clamp(min=0.0)
        cumulative = torch.cumsum(intervals, dim=1)
        cumulative_coordinate = torch.log1p(cumulative + self.alpha)
        cumulative_coordinate = cumulative_coordinate * mask.float()

        return (
            cumulative_coordinate.to(dtype=time_log.dtype),
            time_log_fp32.to(dtype=time_log.dtype) * mask_f,
        )

    def _build_position_encoding(
        self,
        batch_size: int,
        seq_len: int,
        dtype: torch.dtype,
        device: torch.device,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pos_scalar = self.position[:seq_len].to(device=device, dtype=dtype)
        pos_scalar = pos_scalar.unsqueeze(0).expand(batch_size, -1)
        pos_encoding = self._sinusoidal_from_scalar(pos_scalar)
        pos_encoding = pos_encoding * mask.unsqueeze(-1).to(dtype=dtype)
        return pos_scalar, pos_encoding

    def _build_time_encoding(
        self,
        time_log: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cumulative_coordinate, local_log_iat = self._compute_time_features(
            time_log=time_log,
            mask=mask,
        )
        cumulative_encoding = self._sinusoidal_from_scalar(
            cumulative_coordinate
        )

        if not self.use_local_iat:
            return cumulative_coordinate, cumulative_encoding

        if self.normalize_local_iat:
            local_relative = self._masked_standardize(local_log_iat, mask)
        else:
            local_relative = local_log_iat

        # tanh keeps the absolute log-IAT channel numerically bounded while
        # preserving whether an interval is small or large.
        local_absolute = torch.tanh(local_log_iat / 10.0)
        local_input = torch.stack(
            [local_absolute, local_relative], dim=-1
        )
        local_encoding = self.local_time_encoder(local_input)
        local_encoding = local_encoding * mask.unsqueeze(-1).to(
            dtype=local_encoding.dtype
        )

        local_gain = self._gain(self.raw_local_gain)
        mixed_time = self.time_component_mixer(
            torch.cat(
                [cumulative_encoding, local_gain * local_encoding],
                dim=-1,
            )
        )
        mixed_time = mixed_time * mask.unsqueeze(-1).to(
            dtype=mixed_time.dtype
        )
        return cumulative_coordinate, mixed_time

    def forward(
        self,
        e: torch.Tensor,
        time_log: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            e:        [B, L, d_model]
            time_log: [B, L], log(1 + IAT)
            mask:     [B, L], True for real packets
        """
        batch_size, seq_len, _ = e.shape

        if time_log.dim() == 3:
            time_log = time_log.squeeze(-1)

        if mask is None:
            mask = torch.ones(
                batch_size,
                seq_len,
                dtype=torch.bool,
                device=e.device,
            )
        else:
            mask = mask.bool()

        if not self.use_positional_encoding and not self.use_time_encoding:
            return self.dropout(e)

        pos_scalar = None
        pos_encoding = None
        if self.use_positional_encoding:
            pos_scalar, pos_encoding = self._build_position_encoding(
                batch_size=batch_size,
                seq_len=seq_len,
                dtype=e.dtype,
                device=e.device,
                mask=mask,
            )

        cumulative_coordinate = None
        time_encoding = None
        if self.use_time_encoding:
            cumulative_coordinate, time_encoding = self._build_time_encoding(
                time_log=time_log.to(dtype=e.dtype),
                mask=mask,
            )

        pos_gain = self._gain(self.raw_pos_gain)
        time_gain = self._gain(self.raw_time_gain)
        residual_gain = self._gain(self.raw_residual_gain)

        if self.use_positional_encoding and self.use_time_encoding:
            if self.fusion_mode == "joint":
                # Exact original formulation for reproducibility.
                joint_scalar = pos_scalar + cumulative_coordinate
                fused_encoding = self._sinusoidal_from_scalar(joint_scalar)

            elif self.fusion_mode == "joint_scaled":
                joint_scalar = (
                    pos_gain * pos_scalar
                    + time_gain * cumulative_coordinate
                )
                fused_encoding = self._sinusoidal_from_scalar(joint_scalar)

            elif self.fusion_mode == "separate_add":
                fused_encoding = (
                    pos_gain * pos_encoding
                    + time_gain * time_encoding
                )

            elif self.fusion_mode == "concat":
                fused_encoding = self.concat_fusion(
                    torch.cat(
                        [pos_gain * pos_encoding, time_gain * time_encoding],
                        dim=-1,
                    )
                )

            else:  # gated
                pos_branch = pos_gain * pos_encoding
                time_branch = time_gain * time_encoding
                gate = self.fusion_gate(
                    torch.cat([e, pos_branch, time_branch], dim=-1)
                )
                fused_encoding = (
                    gate * pos_branch
                    + (1.0 - gate) * time_branch
                )
                fused_encoding = self.gated_fusion_norm(fused_encoding)

        elif self.use_positional_encoding:
            fused_encoding = pos_gain * pos_encoding

        else:
            fused_encoding = time_gain * time_encoding

        fused_encoding = fused_encoding * mask.unsqueeze(-1).to(
            dtype=fused_encoding.dtype
        )
        output = e + residual_gain * fused_encoding
        output = self.post_norm(output)
        return self.dropout(output)


class AttentionPooling(nn.Module):
    """
    Learned attention pooling, replacing simple masked_mean_pool
    """

    def __init__(self, d_model):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, h, mask):
        # h: [B, L, D], mask: [B, L]
        attn_weights = self.attention(h).squeeze(-1)  # [B, L]

        # Set very small weights for padding positions
        attn_weights = attn_weights.masked_fill(~mask.bool(), -1e4)
        # attn_weights = attn_weights.masked_fill(~mask.bool(), -1e9)
        attn_weights = torch.softmax(attn_weights, dim=-1).unsqueeze(-1)  # [B, L, 1]

        # Weighted sum
        z = (h * attn_weights).sum(dim=1)  # [B, D]
        return z


class FlowFeatureEncoder(nn.Module):
    """
    Encodes flow-level statistical features into d_model space.
    Used only when inject_to_packets=False.
    """

    def __init__(self, flow_dim: int, d_model: int, dropout: float):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(flow_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

    def forward(self, flow_feats: torch.Tensor) -> torch.Tensor:
        """
        Args:
            flow_feats: [B, flow_dim]
        Returns:
            flow_encoded: [B, d_model]
        """
        return self.encoder(flow_feats)

class FlowFusion(nn.Module):
    """
    Fuses packet-level representation (from Transformer) with flow-level features.
    Supports three fusion methods: concat, gated, add.
    """

    def __init__(self, d_model: int, fusion_method: str, dropout: float):
        super().__init__()
        self.fusion_method = fusion_method

        if fusion_method == "concat":
            self.fusion_layer = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        elif fusion_method == "gated":
            self.gate = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.Sigmoid()
            )
            # Optional: add a small projection after gating
            self.post_gate = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Dropout(dropout),
            )
        elif fusion_method == "add":
            # Optional projection for pooled representation
            self.pooled_proj = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model),
                nn.Dropout(dropout),
            )
            # Optional projection for flow representation
            self.flow_proj = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model),
                nn.Dropout(dropout),
            )
        else:
            raise ValueError(f"Unsupported fusion method: {fusion_method}")

    def forward(self, pooled: torch.Tensor, flow_encoded: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pooled: [B, d_model] - representation from Transformer + pooling
            flow_encoded: [B, d_model] - encoded flow-level features
        Returns:
            fused: [B, d_model]
        """
        if self.fusion_method == "concat":
            # Concatenate and project
            concat_feat = torch.cat([pooled, flow_encoded], dim=-1)  # [B, 2*d_model]
            return self.fusion_layer(concat_feat)  # [B, d_model]

        elif self.fusion_method == "gated":
            # Compute dynamic gates
            concat_feat = torch.cat([pooled, flow_encoded], dim=-1)  # [B, 2*d_model]
            gate = self.gate(concat_feat)  # [B, d_model], values in [0, 1]

            # Gated fusion
            fused = gate * pooled + (1 - gate) * flow_encoded  # [B, d_model]
            return self.post_gate(fused)

        elif self.fusion_method == "add":
            # Additive fusion with optional projections
            pooled_proj = self.pooled_proj(pooled)  # [B, d_model]
            flow_proj = self.flow_proj(flow_encoded)  # [B, d_model]
            return pooled_proj + flow_proj  # [B, d_model]


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
    # print("[INFO] Stage1Transformer.__init__*********** cls_head_config=",cls_head_config)
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

class Stage1TimeAwareTransformer(nn.Module):
    """
    Stage1 intra-flow packet-sequence Transformer.

    Supports three modes:
    1. inject_to_packets=True: Flow features concatenated to each packet (æ–¹æ¡ˆA)
    2. inject_to_packets=False, use_flow_features=True: Hierarchical fusion (æ–¹æ¡ˆC)
    3. use_flow_features=False: Only packet features (æ–¹æ¡ˆB)

    Input:
        x:
            [B, L, input_dim]
            Mode 1: [packet features ; flow features]
            Mode 2/3: [packet features only]

        time_log:
            [B, L]
            log(1 + flow_iat_us)

        mask:
            [B, L]
            True  = real packet
            False = padding

        flow_feats (optional):
            [B, flow_dim]
            Only used in æ–¹æ¡ˆC

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
        flow_fusion_cfg = cfg.get("features", {}).get("flow_fusion", {})
        self.use_flow_fusion = flow_fusion_cfg.get("enabled", False)
        self.inject_to_packets = flow_fusion_cfg.get("inject_to_packets", True)
        self.fusion_method = flow_fusion_cfg.get("method", "gated")

        # Model hyperparameters
        d_model = int(model_cfg.get("d_model", 128))
        nhead = int(model_cfg.get("nhead", 4))
        num_layers = int(model_cfg.get("num_layers", 2))
        dim_feedforward = int(model_cfg.get("dim_feedforward", 256))
        dropout = float(model_cfg.get("dropout", 0.1))
        max_seq_len = int(seq_cfg.get("max_seq_len", 64))

        use_positional_encoding = bool(model_cfg.get("use_positional_encoding", True))
        use_time_encoding = bool(model_cfg.get("use_time_encoding", True))

        record_projection_cfg = model_cfg.get("record_projection", None)

        print(f"[INFO]------model --- __init__---d_model={d_model}, nhead={nhead}, num_layers={num_layers}, "
              f"dim_feedforward={dim_feedforward}, dropout={dropout}, max_seq_len={max_seq_len}, use_positional_encoding={use_positional_encoding}, "
              f"use_time_encoding={use_time_encoding}")
        # Time encoding configuration
        time_encoding_cfg = model_cfg.get("time_encoding", {})
        alpha = float(time_encoding_cfg.get("alpha", 1e-07))
        time_fusion_mode = str(
            time_encoding_cfg.get("fusion_mode", "joint")
        ).lower()
        use_local_iat = bool(
            time_encoding_cfg.get("use_local_iat", False)
        )
        normalize_local_iat = bool(
            time_encoding_cfg.get("normalize_local_iat", True)
        )
        post_encoding_norm = bool(
            time_encoding_cfg.get("post_encoding_norm", False)
        )
        learnable_gains = bool(
            time_encoding_cfg.get("learnable_gains", True)
        )
        local_time_hidden_dim = time_encoding_cfg.get(
            "local_time_hidden_dim", None
        )
        if local_time_hidden_dim is not None:
            local_time_hidden_dim = int(local_time_hidden_dim)
        max_time_log = float(
            time_encoding_cfg.get("max_time_log", 30.0)
        )
        # Record-level projection (always needed)
        self.projection = RecordLevelProjection(
            input_dim=input_dim,
            d_model=d_model,
            dropout=dropout,
            mlp_cfg=record_projection_cfg
        )

        # Time-aware encoding (always needed)
        self.time_encoding = TimeAwareEncoding(
            d_model=d_model,
            max_len=max_seq_len,
            dropout=dropout,
            use_positional_encoding=use_positional_encoding,
            use_time_encoding=use_time_encoding,
            alpha=alpha,
            fusion_mode=time_fusion_mode,
            use_local_iat=use_local_iat,
            normalize_local_iat=normalize_local_iat,
            post_encoding_norm=post_encoding_norm,
            learnable_gains=learnable_gains,
            local_time_hidden_dim=local_time_hidden_dim,
            max_time_log=max_time_log,
        )

        # Transformer encoder (always needed)
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

        # Attention pooling (always needed)
        self.attention_pool = AttentionPooling(d_model)

        # å¦‚æžœæ˜¯åˆ†å±‚æ³¨å…¥æ¨¡å¼(æ–¹æ¡ˆC)ï¼Œåˆå§‹åŒ–FlowFeatureEncoderå’ŒFlowFusion  Hierarchical fusion mode (æ–¹æ¡ˆC)
        if self.use_flow_fusion and not self.inject_to_packets:
            flow_feature_dim = cfg.get("_flow_feature_dim", 0)
            if flow_feature_dim > 0:
                self.flow_encoder = FlowFeatureEncoder(
                    flow_dim=flow_feature_dim,
                    d_model=d_model,
                    dropout=dropout
                )

                self.flow_fusion = FlowFusion(
                    d_model=d_model,
                    fusion_method=self.fusion_method,
                    dropout=dropout
                )

                print(f"[INFO---model.__init__] æ–¹æ¡ˆC - åˆ†å±‚ç‰¹å¾æ³¨å…¥å·²å¯ç”¨")
                print(f"[INFO]   Flowç‰¹å¾ç»´åº¦: {flow_feature_dim}")
                print(f"[INFO]   èžåˆæ–¹æ³•: {self.fusion_method}")
                print(f"[INFO]   æ¨¡åž‹ç»´åº¦: {d_model}")
            else:
                print(f"[WARNING] å¯ç”¨äº†flow_fusionä½†flow_feature_dim=0ï¼Œå°†ç¦ç”¨flowèžåˆ")
                self.use_flow_fusion = False
        elif self.inject_to_packets:
            print(f"[INFO] æ–¹æ¡ˆA - Flowç‰¹å¾æ‹¼æŽ¥åˆ°æ¯ä¸ªpacket")
            print(f"[INFO]   è¾“å…¥ç»´åº¦(å«flow): {input_dim}")
        else:
            print(f"[INFO] æ–¹æ¡ˆB - ä»…ä½¿ç”¨Packetç‰¹å¾")
            print(f"[INFO]   è¾“å…¥ç»´åº¦: {input_dim}")

        # Print encoding mode
        if use_positional_encoding and use_time_encoding:
            print(f"[INFO] Encoding: Position + Time, fusion_mode={time_fusion_mode}, local_iat={use_local_iat}")
        elif use_positional_encoding:
            print(f"[INFO] Encoding: Positional Encoding only")
        elif use_time_encoding:
            print(f"[INFO] Encoding: Time Encoding only (no position index)")
        else:
            print(f"[INFO] Encoding: None")

        # ---- Classifier ----
        cls_head_config = int(model_cfg.get("cls_head", 0))

        print("[INFO] Stage1Transformer.__init__*********** cls_head_config=", cls_head_config)

        self.classifier = build_cls_head(
            d_model=d_model,
            dropout=dropout,
            num_classes=2,
            cls_head_config=cls_head_config
        )

    def forward(
            self,
            x: torch.Tensor,
            time_log: torch.Tensor,
            mask: torch.Tensor,
            flow_feats: Optional[torch.Tensor] = None,
            return_embedding: bool = False,
    ):
        """
        Args:
            x:        [B, L, packet_dim] or [B, L, packet_dim + flow_dim]
            time_log: [B, L]
            mask:     [B, L], True for real packet, False for padding
            flow_feats: [B, flow_dim], optional, only used when inject_to_packets=False
            return_embedding: if True, returns (logits, z, h)
        """
        # 1. Record-level projection
        e = self.projection(x)  # [B, L, d_model]

        # 2. Time-aware encoding
        e = self.time_encoding(e, time_log, mask=mask)  # [B, L, d_model]

        # 3. Transformer encoding
        src_key_padding_mask = ~mask.bool()
        h = self.encoder(e, src_key_padding_mask=src_key_padding_mask)  # [B, L, d_model]

        # 4. Pooling to get flow-level representation
        z = self.attention_pool(h, mask)  # [B, d_model]

        # 5. Flow feature fusion (æ–¹æ¡ˆC: åˆ†å±‚æ³¨å…¥)
        if self.use_flow_fusion and not self.inject_to_packets and flow_feats is not None:
            # Encode flow features
            flow_encoded = self.flow_encoder(flow_feats)  # [B, d_model]

            # Fuse with pooled representation
            z = self.flow_fusion(z, flow_encoded)  # [B, d_model]

        # 6. Classification
        logits = self.classifier(z)  # [B, 2]

        if return_embedding:
            return logits, z, h

        return logits

    @staticmethod
    def masked_mean_pool(h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Simple mean pooling (kept for reference)."""
        mask_float = mask.float().unsqueeze(-1)
        h = h * mask_float
        denom = mask_float.sum(dim=1).clamp(min=1.0)
        return h.sum(dim=1) / denom

    def get_model_info(self) -> Dict[str, Any]:
        """Return model configuration info for debugging."""
        return {
            "mode": "inject_to_packets" if self.inject_to_packets
            else ("hierarchical_fusion" if self.use_flow_fusion else "packet_only"),
            "use_flow_features": self.use_flow_fusion,
            "fusion_method": self.fusion_method if self.use_flow_fusion else None,
        }