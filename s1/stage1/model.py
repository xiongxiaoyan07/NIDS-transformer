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
from typing import Dict, Any, Optional

import torch
import torch.nn as nn

from .build_mlp import build_mlp


class RecordLevelProjection(nn.Module):
    """
    Input encoding:
        e_i,t = W x_i,t + b

    This is the record-level projection before Transformer.
    """

    def __init__(self, input_dim: int, d_model: int, dropout: float, mlp_cfg: dict | None = None):
        super().__init__()
        # 如果你旧版 projection 不是纯 Linear，而是 Linear + Dropout / LayerNorm，
        # 可以把 legacy_layer_factory 改成旧版原始结构。
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


class AttentionPooling(nn.Module):
    """
    学习性的注意力池化，替代简单的 masked_mean_pool
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

        # 对padding位置设置极小权重
        attn_weights = attn_weights.masked_fill(~mask.bool(), -1e9)
        attn_weights = torch.softmax(attn_weights, dim=-1).unsqueeze(-1)  # [B, L, 1]

        # 加权求和
        z = (h * attn_weights).sum(dim=1)  # [B, D]
        return z

# 在 model.py 的 forward 方法中：
# 替换 z = self.masked_mean_pool(h, mask)
# 为：
# z = self.attention_pool(h, mask)
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

class Stage1TimeAwareTransformer(nn.Module):
    """
    Stage1 intra-flow packet-sequence Transformer.

    Supports three modes:
    1. inject_to_packets=True: Flow features concatenated to each packet (方案A)
    2. inject_to_packets=False, use_flow_features=True: Hierarchical fusion (方案C)
    3. use_flow_features=False: Only packet features (方案B)

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
            Only used in Mode 2 （方案C）

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
        # 获取flow_fusion配置
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

        # 如果是分层注入模式(方案C)，初始化FlowFeatureEncoder和FlowFusion
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

                print(f"[INFO] 方案C - 分层特征注入已启用")
                print(f"[INFO]   Flow特征维度: {flow_feature_dim}")
                print(f"[INFO]   融合方法: {self.fusion_method}")
                print(f"[INFO]   模型维度: {d_model}")
            else:
                print(f"[WARNING] 启用了flow_fusion但flow_feature_dim=0，将禁用flow融合")
                self.use_flow_fusion = False
        elif self.inject_to_packets:
            print(f"[INFO] 方案A - Flow特征拼接到每个packet")
            print(f"[INFO]   输入维度(含flow): {input_dim}")
        else:
            print(f"[INFO] 方案B - 仅使用Packet特征")
            print(f"[INFO]   输入维度: {input_dim}")

        # ---- Classifier ----
        # For extremely imbalanced data, use simpler classifier with high dropout
        if dropout > 0.25:
            self.classifier = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, 2),
            )
        else:
            self.classifier = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout * 0.3),
                nn.Linear(d_model // 2, 2),
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
        e = self.time_encoding(e, time_log)  # [B, L, d_model]

        # 3. Transformer encoding
        src_key_padding_mask = ~mask.bool()
        h = self.encoder(e, src_key_padding_mask=src_key_padding_mask)  # [B, L, d_model]

        # 4. Pooling to get flow-level representation
        z = self.attention_pool(h, mask)  # [B, d_model]

        # 5. Flow feature fusion (方案C: 分层注入)
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
