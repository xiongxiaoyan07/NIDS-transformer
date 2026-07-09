from __future__ import annotations

import math
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import SinusoidalPositionalEncoding, build_cls_head


def _last_valid_indices(mask: torch.Tensor) -> torch.Tensor:
    """Return the index of the last valid token for each row."""
    batch_size, seq_len = mask.shape
    positions = torch.arange(seq_len, device=mask.device).unsqueeze(0).expand(batch_size, seq_len)
    return positions.masked_fill(~mask, -1).max(dim=1).values.clamp(min=0)


def _inverse_softplus(value: float) -> float:
    value = float(value)
    if value <= 0.0:
        return -20.0
    if value > 20.0:
        return value
    return math.log(math.expm1(value))


class Stage2TargetQueryGatedAttention(nn.Module):
    """Classify the current flow with target-query attention over history.

    The current flow is the query. Earlier flows are keys/values. A learned gate
    decides how much of the context update should be injected into the current
    flow representation, so the model can fall back to the strong current-flow
    Stage1 embedding when history is noisy.
    """

    def __init__(self, cfg: Dict[str, Any], input_dim: int):
        super().__init__()

        model_cfg = cfg["model"]
        self.input_dim = int(input_dim)
        self.d_model = int(model_cfg.get("d_model") or input_dim)
        self.use_positional_encoding = bool(model_cfg.get("use_positional_encoding", True))
        self.position_mode = str(model_cfg.get("position_mode", "age"))

        dropout = float(model_cfg.get("dropout", 0.3))
        nhead = int(model_cfg.get("nhead", 8))
        dim_feedforward = int(model_cfg.get("dim_feedforward", 512))
        num_classes = int(model_cfg.get("num_classes", 2))
        cls_head_config = int(model_cfg.get("cls_head", 0))

        self.input_proj = (
            nn.Identity()
            if self.input_dim == self.d_model
            else nn.Linear(self.input_dim, self.d_model)
        )

        self.pos = (
            SinusoidalPositionalEncoding(
                self.d_model,
                max_len=int(model_cfg.get("max_len", 512)),
                position_mode=self.position_mode,
            )
            if self.use_positional_encoding
            else None
        )

        self.query_norm = nn.LayerNorm(self.d_model)
        self.history_norm = nn.LayerNorm(self.d_model)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_dropout = nn.Dropout(dropout)

        # Features: current, summary, current-summary, current*summary.
        fusion_dim = self.d_model * 4
        update_dim = self.d_model * 3

        self.context_update = nn.Sequential(
            nn.LayerNorm(update_dim),
            nn.Linear(update_dim, self.d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model, self.d_model),
        )

        self.gate = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, self.d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model, self.d_model),
            nn.Sigmoid(),
        )

        self.fused_norm = nn.LayerNorm(self.d_model)
        self.ffn = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, self.d_model),
            nn.Dropout(dropout),
        )
        self.final_norm = nn.LayerNorm(self.d_model)

        self.cls_head = build_cls_head(
            d_model=self.d_model,
            cls_head_config=cls_head_config,
            dropout=dropout,
            num_classes=num_classes,
        )

        print(
            "[INFO] Stage2TargetQueryGatedAttention.__init__ "
            f"use_positional_encoding={self.use_positional_encoding}, "
            f"position_mode={self.position_mode}, nhead={nhead}, "
            f"dim_feedforward={dim_feedforward}, dropout={dropout}, "
            f"cls_head_config={cls_head_config}"
        )

    def forward(self, context_z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # context_z: [B, L, input_dim], left padded. The current flow is the
        # last valid token because ContextIndexBuilder appends it last.
        tokens = self.input_proj(context_z)
        batch_size = tokens.size(0)
        row_idx = torch.arange(batch_size, device=tokens.device)
        current_idx = _last_valid_indices(mask)

        current = tokens[row_idx, current_idx]

        attn_tokens = tokens
        if self.pos is not None:
            attn_tokens = self.pos(tokens, mask)

        query = attn_tokens[row_idx, current_idx].unsqueeze(1)

        history_mask = mask.clone()
        history_mask[row_idx, current_idx] = False
        no_history = ~history_mask.any(dim=1)
        has_no_history = bool(no_history.any().item())

        safe_history_mask = history_mask.clone()
        history_tokens = attn_tokens.clone()
        if has_no_history:
            history_tokens[no_history] = 0.0
            safe_history_mask[no_history, 0] = True

        attn_out, _ = self.cross_attn(
            query=self.query_norm(query),
            key=self.history_norm(history_tokens),
            value=self.history_norm(history_tokens),
            key_padding_mask=~safe_history_mask,
            need_weights=False,
        )

        summary = self.attn_dropout(attn_out.squeeze(1))
        if has_no_history:
            summary = summary.masked_fill(no_history.unsqueeze(-1), 0.0)

        fusion_features = torch.cat(
            [current, summary, current - summary, current * summary],
            dim=-1,
        )
        update_features = torch.cat(
            [summary, current - summary, current * summary],
            dim=-1,
        )

        gate = self.gate(fusion_features)
        context_update = self.context_update(update_features)
        if has_no_history:
            gate = gate.masked_fill(no_history.unsqueeze(-1), 0.0)
            context_update = context_update.masked_fill(no_history.unsqueeze(-1), 0.0)

        fused = self.fused_norm(current + gate * context_update)
        fused = self.final_norm(fused + self.ffn(fused))
        return self.cls_head(fused)


class Stage2TargetQueryResidualAttention(nn.Module):
    """Target-query attention with conservative residual logit correction.

    This variant keeps a strong current-flow branch and lets context produce
    only a gated delta on top of the base logits:

        logits = base_logits(current) + context_scale * gate * delta_logits

    It is meant for settings where context improves recall/ranking but can
    increase false positives if injected too aggressively.
    """

    def __init__(self, cfg: Dict[str, Any], input_dim: int):
        super().__init__()

        model_cfg = cfg["model"]
        context_cfg = cfg.get("context", {})

        self.input_dim = int(input_dim)
        self.d_model = int(model_cfg.get("d_model") or input_dim)
        self.use_positional_encoding = bool(model_cfg.get("use_positional_encoding", True))
        self.position_mode = str(model_cfg.get("position_mode", "age"))
        self.context_scale = float(model_cfg.get("context_scale", 0.2))
        self.use_context_length_feature = bool(model_cfg.get("use_context_length_feature", False))
        self.window_size = max(1, int(context_cfg.get("window_size", model_cfg.get("max_len", 512))))

        dropout = float(model_cfg.get("dropout", 0.25))
        nhead = int(model_cfg.get("nhead", 4))
        dim_feedforward = int(model_cfg.get("dim_feedforward", 256))
        num_classes = int(model_cfg.get("num_classes", 2))
        cls_head_config = int(model_cfg.get("cls_head", 1))

        self.input_proj = (
            nn.Identity()
            if self.input_dim == self.d_model
            else nn.Linear(self.input_dim, self.d_model)
        )

        self.pos = (
            SinusoidalPositionalEncoding(
                self.d_model,
                max_len=int(model_cfg.get("max_len", 512)),
                position_mode=self.position_mode,
            )
            if self.use_positional_encoding
            else None
        )

        self.query_norm = nn.LayerNorm(self.d_model)
        self.history_norm = nn.LayerNorm(self.d_model)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_dropout = nn.Dropout(dropout)

        self.context_ffn = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, self.d_model),
            nn.Dropout(dropout),
        )
        self.context_norm = nn.LayerNorm(self.d_model)

        extra_dim = 2 if self.use_context_length_feature else 0
        fusion_dim = self.d_model * 4 + extra_dim

        self.base_head = build_cls_head(
            d_model=self.d_model,
            cls_head_config=cls_head_config,
            dropout=dropout,
            num_classes=num_classes,
        )

        self.delta_head = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, self.d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model, num_classes),
        )

        self.gate = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, self.d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model, 1),
            nn.Sigmoid(),
        )

        gate_bias_init = float(model_cfg.get("gate_bias_init", -2.0))
        nn.init.constant_(self.gate[-2].bias, gate_bias_init)

        if bool(model_cfg.get("zero_init_delta", True)):
            nn.init.zeros_(self.delta_head[-1].weight)
            nn.init.zeros_(self.delta_head[-1].bias)

        print(
            "[INFO] Stage2TargetQueryResidualAttention.__init__ "
            f"use_positional_encoding={self.use_positional_encoding}, "
            f"position_mode={self.position_mode}, nhead={nhead}, "
            f"dim_feedforward={dim_feedforward}, dropout={dropout}, "
            f"context_scale={self.context_scale}, gate_bias_init={gate_bias_init}, "
            f"use_context_length_feature={self.use_context_length_feature}, "
            f"cls_head_config={cls_head_config}"
        )

    def forward(self, context_z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        tokens = self.input_proj(context_z)
        batch_size = tokens.size(0)
        row_idx = torch.arange(batch_size, device=tokens.device)
        current_idx = _last_valid_indices(mask)

        current = tokens[row_idx, current_idx]
        base_logits = self.base_head(current)

        attn_tokens = tokens
        if self.pos is not None:
            attn_tokens = self.pos(tokens, mask)

        query = attn_tokens[row_idx, current_idx].unsqueeze(1)

        history_mask = mask.clone()
        history_mask[row_idx, current_idx] = False
        history_len = history_mask.float().sum(dim=1, keepdim=True)
        no_history = ~history_mask.any(dim=1)
        has_no_history = bool(no_history.any().item())

        safe_history_mask = history_mask.clone()
        history_tokens = attn_tokens
        if has_no_history:
            history_tokens = attn_tokens.clone()
            history_tokens[no_history] = 0.0
            safe_history_mask[no_history, 0] = True

        attn_out, _ = self.cross_attn(
            query=self.query_norm(query),
            key=self.history_norm(history_tokens),
            value=self.history_norm(history_tokens),
            key_padding_mask=~safe_history_mask,
            need_weights=False,
        )

        summary = self.attn_dropout(attn_out.squeeze(1))
        summary = self.context_norm(summary + self.context_ffn(summary))
        if has_no_history:
            summary = summary.masked_fill(no_history.unsqueeze(-1), 0.0)

        fusion_parts = [current, summary, current - summary, current * summary]
        if self.use_context_length_feature:
            denom = torch.log1p(
                torch.tensor(float(max(self.window_size - 1, 1)), device=tokens.device, dtype=tokens.dtype)
            )
            length_feature = torch.log1p(history_len.to(dtype=tokens.dtype)) / denom
            no_history_feature = no_history.unsqueeze(-1).to(dtype=tokens.dtype)
            fusion_parts.extend([length_feature, no_history_feature])

        fused = torch.cat(fusion_parts, dim=-1)
        delta_logits = self.delta_head(fused)
        gate = self.gate(fused)

        if has_no_history:
            gate = gate.masked_fill(no_history.unsqueeze(-1), 0.0)

        return base_logits + self.context_scale * gate * delta_logits


class Stage2RelationAwareAttention(nn.Module):
    """Target-query attention with relation-aware score bias for mixed context.

    The model keeps the conservative residual-logit structure:

        logits = base_logits(current) + context_scale * gate * delta_logits

    Relation features only change attention scores, not the current-flow base
    branch. The expected relation feature order is:
    [same_source, same_destination, same_endpoint, normalized_age].
    """

    uses_relation_features = True

    def __init__(self, cfg: Dict[str, Any], input_dim: int):
        super().__init__()

        model_cfg = cfg["model"]
        context_cfg = cfg.get("context", {})

        self.input_dim = int(input_dim)
        self.d_model = int(model_cfg.get("d_model") or input_dim)
        self.use_positional_encoding = bool(model_cfg.get("use_positional_encoding", True))
        self.position_mode = str(model_cfg.get("position_mode", "age"))
        self.context_scale = float(model_cfg.get("context_scale", 0.2))
        self.use_context_length_feature = bool(model_cfg.get("use_context_length_feature", True))
        self.window_size = max(1, int(context_cfg.get("window_size", model_cfg.get("max_len", 512))))

        dropout = float(model_cfg.get("dropout", 0.25))
        self.nhead = int(model_cfg.get("nhead", 4))
        dim_feedforward = int(model_cfg.get("dim_feedforward", 256))
        num_classes = int(model_cfg.get("num_classes", 2))
        cls_head_config = int(model_cfg.get("cls_head", 1))

        if self.d_model % self.nhead != 0:
            raise ValueError(
                f"model.d_model ({self.d_model}) must be divisible by model.nhead ({self.nhead})"
            )
        self.head_dim = self.d_model // self.nhead

        self.input_proj = (
            nn.Identity()
            if self.input_dim == self.d_model
            else nn.Linear(self.input_dim, self.d_model)
        )

        self.pos = (
            SinusoidalPositionalEncoding(
                self.d_model,
                max_len=int(model_cfg.get("max_len", 512)),
                position_mode=self.position_mode,
            )
            if self.use_positional_encoding
            else None
        )

        self.query_norm = nn.LayerNorm(self.d_model)
        self.history_norm = nn.LayerNorm(self.d_model)
        self.q_proj = nn.Linear(self.d_model, self.d_model)
        self.k_proj = nn.Linear(self.d_model, self.d_model)
        self.v_proj = nn.Linear(self.d_model, self.d_model)
        self.out_proj = nn.Linear(self.d_model, self.d_model)
        self.attn_dropout = nn.Dropout(dropout)

        self.same_source_bias = nn.Parameter(
            torch.full((self.nhead,), float(model_cfg.get("relation_source_bias_init", 0.10)))
        )
        self.same_destination_bias = nn.Parameter(
            torch.full((self.nhead,), float(model_cfg.get("relation_destination_bias_init", 0.05)))
        )
        self.same_endpoint_bias = nn.Parameter(
            torch.full((self.nhead,), float(model_cfg.get("relation_endpoint_bias_init", 0.00)))
        )
        age_lambda_init = float(model_cfg.get("relation_age_lambda_init", 0.10))
        self.age_lambda_raw = nn.Parameter(
            torch.full((self.nhead,), _inverse_softplus(age_lambda_init))
        )
        self.relation_bias_scale = float(model_cfg.get("relation_bias_scale", 1.0))

        self.context_ffn = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, self.d_model),
            nn.Dropout(dropout),
        )
        self.context_norm = nn.LayerNorm(self.d_model)

        extra_dim = 2 if self.use_context_length_feature else 0
        fusion_dim = self.d_model * 4 + extra_dim

        self.base_head = build_cls_head(
            d_model=self.d_model,
            cls_head_config=cls_head_config,
            dropout=dropout,
            num_classes=num_classes,
        )

        self.delta_head = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, self.d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model, num_classes),
        )

        self.gate = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, self.d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model, 1),
            nn.Sigmoid(),
        )

        gate_bias_init = float(model_cfg.get("gate_bias_init", -2.0))
        nn.init.constant_(self.gate[-2].bias, gate_bias_init)

        if bool(model_cfg.get("zero_init_delta", True)):
            nn.init.zeros_(self.delta_head[-1].weight)
            nn.init.zeros_(self.delta_head[-1].bias)

        print(
            "[INFO] Stage2RelationAwareAttention.__init__ "
            f"use_positional_encoding={self.use_positional_encoding}, "
            f"position_mode={self.position_mode}, nhead={self.nhead}, "
            f"dim_feedforward={dim_feedforward}, dropout={dropout}, "
            f"context_scale={self.context_scale}, gate_bias_init={gate_bias_init}, "
            f"use_context_length_feature={self.use_context_length_feature}, "
            f"relation_source_bias_init={model_cfg.get('relation_source_bias_init', 0.10)}, "
            f"relation_destination_bias_init={model_cfg.get('relation_destination_bias_init', 0.05)}, "
            f"relation_endpoint_bias_init={model_cfg.get('relation_endpoint_bias_init', 0.00)}, "
            f"relation_age_lambda_init={age_lambda_init}, "
            f"relation_bias_scale={self.relation_bias_scale}, "
            f"cls_head_config={cls_head_config}"
        )

    def _relation_bias(self, relation_features: torch.Tensor) -> torch.Tensor:
        same_source = relation_features[..., 0].unsqueeze(1)
        same_destination = relation_features[..., 1].unsqueeze(1)
        same_endpoint = relation_features[..., 2].unsqueeze(1)
        age = relation_features[..., 3].unsqueeze(1)

        source_bias = self.same_source_bias.view(1, self.nhead, 1)
        destination_bias = self.same_destination_bias.view(1, self.nhead, 1)
        endpoint_bias = self.same_endpoint_bias.view(1, self.nhead, 1)
        age_lambda = F.softplus(self.age_lambda_raw).view(1, self.nhead, 1)

        return self.relation_bias_scale * (
            same_source * source_bias
            + same_destination * destination_bias
            + same_endpoint * endpoint_bias
            - age * age_lambda
        )

    def forward(
        self,
        context_z: torch.Tensor,
        mask: torch.Tensor,
        relation_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        tokens = self.input_proj(context_z)
        batch_size, seq_len, _ = tokens.shape
        row_idx = torch.arange(batch_size, device=tokens.device)
        current_idx = _last_valid_indices(mask)

        current = tokens[row_idx, current_idx]
        base_logits = self.base_head(current)

        if relation_features is None:
            relation_features = torch.zeros(
                batch_size,
                seq_len,
                4,
                device=tokens.device,
                dtype=tokens.dtype,
            )
        else:
            relation_features = relation_features.to(device=tokens.device, dtype=tokens.dtype)

        attn_tokens = tokens
        if self.pos is not None:
            attn_tokens = self.pos(tokens, mask)

        query = attn_tokens[row_idx, current_idx].unsqueeze(1)
        history_mask = mask.clone()
        history_mask[row_idx, current_idx] = False
        history_len = history_mask.float().sum(dim=1, keepdim=True)
        no_history = ~history_mask.any(dim=1)
        has_no_history = bool(no_history.any().item())

        safe_history_mask = history_mask.clone()
        history_tokens = attn_tokens
        if has_no_history:
            history_tokens = attn_tokens.clone()
            history_tokens[no_history] = 0.0
            safe_history_mask[no_history, current_idx[no_history]] = True

        q = self.q_proj(self.query_norm(query)).view(
            batch_size, 1, self.nhead, self.head_dim
        ).transpose(1, 2)
        k = self.k_proj(self.history_norm(history_tokens)).view(
            batch_size, seq_len, self.nhead, self.head_dim
        ).transpose(1, 2)
        v = self.v_proj(self.history_norm(history_tokens)).view(
            batch_size, seq_len, self.nhead, self.head_dim
        ).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)).squeeze(2)
        scores = scores / math.sqrt(float(self.head_dim))
        scores = scores + self._relation_bias(relation_features)
        scores = scores.masked_fill(~safe_history_mask.unsqueeze(1), torch.finfo(scores.dtype).min)

        attn_weights = torch.softmax(scores.float(), dim=-1).to(dtype=scores.dtype)
        attn_weights = self.attn_dropout(attn_weights)
        summary = torch.matmul(attn_weights.unsqueeze(2), v).squeeze(2)
        summary = summary.contiguous().view(batch_size, self.d_model)
        summary = self.out_proj(summary)
        summary = self.context_norm(summary + self.context_ffn(summary))

        if has_no_history:
            summary = summary.masked_fill(no_history.unsqueeze(-1), 0.0)

        fusion_parts = [current, summary, current - summary, current * summary]
        if self.use_context_length_feature:
            denom = torch.log1p(
                torch.tensor(float(max(self.window_size - 1, 1)), device=tokens.device, dtype=tokens.dtype)
            )
            length_feature = torch.log1p(history_len.to(dtype=tokens.dtype)) / denom
            no_history_feature = no_history.unsqueeze(-1).to(dtype=tokens.dtype)
            fusion_parts.extend([length_feature, no_history_feature])

        fused = torch.cat(fusion_parts, dim=-1)
        delta_logits = self.delta_head(fused)
        gate = self.gate(fused)
        if has_no_history:
            gate = gate.masked_fill(no_history.unsqueeze(-1), 0.0)

        return base_logits + self.context_scale * gate * delta_logits


class Stage2SourceDestinationAttention(nn.Module):
    """Dual-branch source/destination target-query attention.

    Source and destination histories are encoded separately:

        summary_src = attention(current, source_history)
        summary_dst = attention(current, destination_history)

    The final prediction keeps a current-flow base branch and uses the two
    summaries only as a gated residual logit correction.
    """

    def __init__(self, cfg: Dict[str, Any], input_dim: int):
        super().__init__()

        model_cfg = cfg["model"]
        context_cfg = cfg.get("context", {})

        self.input_dim = int(input_dim)
        self.d_model = int(model_cfg.get("d_model") or input_dim)
        self.use_positional_encoding = bool(model_cfg.get("use_positional_encoding", True))
        self.position_mode = str(model_cfg.get("position_mode", "age"))
        self.context_scale = float(model_cfg.get("context_scale", 0.2))
        self.source_context_scale = float(model_cfg.get("source_context_scale", self.context_scale))
        self.destination_context_scale = float(model_cfg.get("destination_context_scale", self.context_scale))
        self.use_context_length_feature = bool(model_cfg.get("use_context_length_feature", True))
        self.use_multiplicative_fusion = bool(model_cfg.get("use_multiplicative_fusion", False))
        self.window_size = max(1, int(context_cfg.get("window_size", model_cfg.get("max_len", 512))))

        dropout = float(model_cfg.get("dropout", 0.25))
        nhead = int(model_cfg.get("nhead", 4))
        dim_feedforward = int(model_cfg.get("dim_feedforward", 256))
        num_classes = int(model_cfg.get("num_classes", 2))
        cls_head_config = int(model_cfg.get("cls_head", 1))

        self.input_proj = (
            nn.Identity()
            if self.input_dim == self.d_model
            else nn.Linear(self.input_dim, self.d_model)
        )

        self.pos = (
            SinusoidalPositionalEncoding(
                self.d_model,
                max_len=int(model_cfg.get("max_len", 512)),
                position_mode=self.position_mode,
            )
            if self.use_positional_encoding
            else None
        )

        self.src_query_norm = nn.LayerNorm(self.d_model)
        self.src_history_norm = nn.LayerNorm(self.d_model)
        self.src_cross_attn = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.src_context_ffn = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, self.d_model),
            nn.Dropout(dropout),
        )
        self.src_context_norm = nn.LayerNorm(self.d_model)

        self.dst_query_norm = nn.LayerNorm(self.d_model)
        self.dst_history_norm = nn.LayerNorm(self.d_model)
        self.dst_cross_attn = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.dst_context_ffn = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, self.d_model),
            nn.Dropout(dropout),
        )
        self.dst_context_norm = nn.LayerNorm(self.d_model)
        self.attn_dropout = nn.Dropout(dropout)

        branch_dim = self.d_model * 3
        if self.use_multiplicative_fusion:
            branch_dim += self.d_model
        if self.use_context_length_feature:
            branch_dim += 2

        self.base_head = build_cls_head(
            d_model=self.d_model,
            cls_head_config=cls_head_config,
            dropout=dropout,
            num_classes=num_classes,
        )

        self.source_delta_head = nn.Sequential(
            nn.LayerNorm(branch_dim),
            nn.Dropout(dropout),
            nn.Linear(branch_dim, self.d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model, num_classes),
        )

        self.source_gate = nn.Sequential(
            nn.LayerNorm(branch_dim),
            nn.Linear(branch_dim, self.d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model, 1),
            nn.Sigmoid(),
        )

        self.destination_delta_head = nn.Sequential(
            nn.LayerNorm(branch_dim),
            nn.Dropout(dropout),
            nn.Linear(branch_dim, self.d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model, num_classes),
        )

        self.destination_gate = nn.Sequential(
            nn.LayerNorm(branch_dim),
            nn.Linear(branch_dim, self.d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model, 1),
            nn.Sigmoid(),
        )

        gate_bias_init = float(model_cfg.get("gate_bias_init", -2.0))
        source_gate_bias_init = float(model_cfg.get("source_gate_bias_init", gate_bias_init))
        destination_gate_bias_init = float(model_cfg.get("destination_gate_bias_init", gate_bias_init))
        nn.init.constant_(self.source_gate[-2].bias, source_gate_bias_init)
        nn.init.constant_(self.destination_gate[-2].bias, destination_gate_bias_init)

        if bool(model_cfg.get("zero_init_delta", True)):
            nn.init.zeros_(self.source_delta_head[-1].weight)
            nn.init.zeros_(self.source_delta_head[-1].bias)
            nn.init.zeros_(self.destination_delta_head[-1].weight)
            nn.init.zeros_(self.destination_delta_head[-1].bias)

        print(
            "[INFO] Stage2SourceDestinationAttention.__init__ "
            f"use_positional_encoding={self.use_positional_encoding}, "
            f"position_mode={self.position_mode}, nhead={nhead}, "
            f"dim_feedforward={dim_feedforward}, dropout={dropout}, "
            f"context_scale={self.context_scale}, gate_bias_init={gate_bias_init}, "
            f"source_gate_bias_init={source_gate_bias_init}, "
            f"destination_gate_bias_init={destination_gate_bias_init}, "
            f"source_context_scale={self.source_context_scale}, "
            f"destination_context_scale={self.destination_context_scale}, "
            f"use_context_length_feature={self.use_context_length_feature}, "
            f"use_multiplicative_fusion={self.use_multiplicative_fusion}, "
            f"cls_head_config={cls_head_config}"
        )

    def _branch_summary(
        self,
        context_z: torch.Tensor,
        mask: torch.Tensor,
        query_norm: nn.LayerNorm,
        history_norm: nn.LayerNorm,
        cross_attn: nn.MultiheadAttention,
        context_ffn: nn.Sequential,
        context_norm: nn.LayerNorm,
    ):
        tokens = self.input_proj(context_z)
        batch_size = tokens.size(0)
        row_idx = torch.arange(batch_size, device=tokens.device)
        current_idx = _last_valid_indices(mask)
        current = tokens[row_idx, current_idx]

        attn_tokens = tokens
        if self.pos is not None:
            attn_tokens = self.pos(tokens, mask)

        query = attn_tokens[row_idx, current_idx].unsqueeze(1)
        history_mask = mask.clone()
        history_mask[row_idx, current_idx] = False
        history_len = history_mask.float().sum(dim=1, keepdim=True)
        no_history = ~history_mask.any(dim=1)
        has_no_history = bool(no_history.any().item())

        safe_history_mask = history_mask.clone()
        history_tokens = attn_tokens
        if has_no_history:
            history_tokens = attn_tokens.clone()
            history_tokens[no_history] = 0.0
            safe_history_mask[no_history, 0] = True

        attn_out, _ = cross_attn(
            query=query_norm(query),
            key=history_norm(history_tokens),
            value=history_norm(history_tokens),
            key_padding_mask=~safe_history_mask,
            need_weights=False,
        )

        summary = self.attn_dropout(attn_out.squeeze(1))
        summary = context_norm(summary + context_ffn(summary))
        if has_no_history:
            summary = summary.masked_fill(no_history.unsqueeze(-1), 0.0)

        return current, summary, history_len, no_history

    def forward(
        self,
        source_context_z: torch.Tensor,
        source_mask: torch.Tensor,
        destination_context_z: torch.Tensor,
        destination_mask: torch.Tensor,
    ) -> torch.Tensor:
        current_src, summary_src, source_len, no_source = self._branch_summary(
            source_context_z,
            source_mask,
            self.src_query_norm,
            self.src_history_norm,
            self.src_cross_attn,
            self.src_context_ffn,
            self.src_context_norm,
        )
        current_dst, summary_dst, destination_len, no_destination = self._branch_summary(
            destination_context_z,
            destination_mask,
            self.dst_query_norm,
            self.dst_history_norm,
            self.dst_cross_attn,
            self.dst_context_ffn,
            self.dst_context_norm,
        )

        current = 0.5 * (current_src + current_dst)
        base_logits = self.base_head(current)

        source_parts = [
            current,
            summary_src,
            current - summary_src,
        ]
        destination_parts = [
            current,
            summary_dst,
            current - summary_dst,
        ]
        if self.use_multiplicative_fusion:
            source_parts.append(current * summary_src)
            destination_parts.append(current * summary_dst)

        if self.use_context_length_feature:
            denom = torch.log1p(
                torch.tensor(float(max(self.window_size - 1, 1)), device=current.device, dtype=current.dtype)
            )
            source_length_feature = torch.log1p(source_len.to(dtype=current.dtype)) / denom
            destination_length_feature = torch.log1p(destination_len.to(dtype=current.dtype)) / denom
            source_parts.extend(
                [
                    source_length_feature,
                    no_source.unsqueeze(-1).to(dtype=current.dtype),
                ]
            )
            destination_parts.extend(
                [
                    destination_length_feature,
                    no_destination.unsqueeze(-1).to(dtype=current.dtype),
                ]
            )

        source_fused = torch.cat(source_parts, dim=-1)
        destination_fused = torch.cat(destination_parts, dim=-1)

        source_delta_logits = self.source_delta_head(source_fused)
        source_gate = self.source_gate(source_fused)
        destination_delta_logits = self.destination_delta_head(destination_fused)
        destination_gate = self.destination_gate(destination_fused)

        if bool(no_source.any().item()):
            source_gate = source_gate.masked_fill(no_source.unsqueeze(-1), 0.0)
        if bool(no_destination.any().item()):
            destination_gate = destination_gate.masked_fill(no_destination.unsqueeze(-1), 0.0)

        return (
            base_logits
            + self.source_context_scale * source_gate * source_delta_logits
            + self.destination_context_scale * destination_gate * destination_delta_logits
        )
