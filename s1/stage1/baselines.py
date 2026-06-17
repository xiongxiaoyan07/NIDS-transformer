# models/baselines.py
"""
传统模型和经典深度学习基线的统一接口
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
import math


# ============================================
# Level 2: 经典深度学习模型
# ============================================

class FlowLevelMLP(nn.Module):
    """
    基于流级聚合特征的 MLP 分类器

    与 Stage1 的 FlowFeatureEncoder 对应，不使用包序列
    """

    def __init__(
            self,
            input_dim: int,
            hidden_dims: list = [256, 128, 64],
            dropout: float = 0.3,
            num_classes: int = 2,
            use_batch_norm: bool = True
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

        layers = []
        in_dim = input_dim

        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(in_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(in_dim, num_classes)

    def forward(self, x, mask=None, time_log=None, flow_feats=None, return_embedding=False):
        """
        Args:
            x: [B, L, D] 包序列（我们聚合成流特征）
            flow_feats: [B, D_flow] 流级聚合特征

        Returns:
            logits: [B, num_classes]
        """
        # 使用 flow_feats 或聚合 packet 特征
        if flow_feats is not None:
            features = flow_feats
        else:
            # 简单聚合：对有效包取均值
            if mask is not None:
                mask_expanded = mask.unsqueeze(-1).float()  # [B, L, 1]
                x_masked = x * mask_expanded
                features = x_masked.sum(dim=1) / (mask_expanded.sum(dim=1) + 1e-8)
            else:
                features = x.mean(dim=1)

        embedding = self.feature_extractor(features)
        logits = self.classifier(embedding)

        if return_embedding:
            return logits, embedding, None
        return logits


# ============================================
# Level 3: 序列模型基线
# ============================================

class LSTMClassifier(nn.Module):
    """
    LSTM 序列分类器（无时间感知）

    支持单向/双向、多层
    """

    def __init__(
            self,
            input_dim: int,
            hidden_dim: int = 128,
            num_layers: int = 2,
            dropout: float = 0.2,
            bidirectional: bool = False,
            pooling: str = 'attention',  # 'mean', 'last', 'attention'
            num_classes: int = 2,
            use_time: bool = False  # 是否使用时间特征
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.pooling_type = pooling
        self.use_time = use_time

        # 如果使用时间特征，扩展输入维度
        lstm_input_dim = input_dim + 1 if use_time else input_dim

        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim

        if pooling == 'attention':
            self.pooling = AttentionPooling(lstm_output_dim)
        else:
            self.pooling = None

        self.classifier = nn.Linear(lstm_output_dim, num_classes)

    def forward(self, x, mask=None, time_log=None, flow_feats=None, return_embedding=False):
        """
        Args:
            x: [B, L, D] 包特征序列
            mask: [B, L] 掩码
            time_log: [B, L] 对数时间间隔
            flow_feats: [B, D_flow] 流特征
            return_embedding: 是否返回嵌入表示
        """
        B, L, D = x.shape
        device = x.device

        # 可选：拼接时间特征
        if self.use_time and time_log is not None:
            time_feat = time_log.unsqueeze(-1)  # [B, L, 1]
            x = torch.cat([x, time_feat], dim=-1)

        # 计算序列长度
        if mask is not None:
            lengths = mask.sum(dim=1).long()  # [B]
            # 确保最小长度为1
            lengths = torch.clamp(lengths, min=1)

            # 打包变长序列（提升效率）
            x_packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )

            lstm_out, (hidden, _) = self.lstm(x_packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                lstm_out, batch_first=True, total_length=L
            )
        else:
            lstm_out, (hidden, _) = self.lstm(x)

        # 池化策略
        if self.pooling_type == 'attention':
            if self.pooling is None:
                self.pooling = AttentionPooling(lstm_out.size(-1)).to(device)
            z = self.pooling(lstm_out, mask)
        elif self.pooling_type == 'mean':
            if mask is not None:
                mask_expanded = mask.unsqueeze(-1).float()
                z = (lstm_out * mask_expanded).sum(dim=1) / (mask_expanded.sum(dim=1) + 1e-8)
            else:
                z = lstm_out.mean(dim=1)
        elif self.pooling_type == 'last':
            if mask is not None:
                # 获取每个序列最后一个有效位置
                last_indices = (lengths - 1).clamp(min=0).unsqueeze(-1).unsqueeze(-1)
                last_indices = last_indices.expand(-1, lstm_out.size(-1)).unsqueeze(1)
                z = lstm_out.gather(1, last_indices).squeeze(1)
            else:
                z = lstm_out[:, -1, :]
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling_type}")

        logits = self.classifier(z)

        if return_embedding:
            return logits, z, lstm_out
        return logits


class BiLSTMClassifier(LSTMClassifier):
    """双向 LSTM"""

    def __init__(self, *args, **kwargs):
        kwargs['bidirectional'] = True
        super().__init__(*args, **kwargs)


class GRUClassifier(nn.Module):
    """
    GRU 序列分类器
    """

    def __init__(
            self,
            input_dim: int,
            hidden_dim: int = 128,
            num_layers: int = 2,
            dropout: float = 0.2,
            bidirectional: bool = False,
            pooling: str = 'attention',
            num_classes: int = 2,
            use_time: bool = False
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.pooling_type = pooling
        self.use_time = use_time

        gru_input_dim = input_dim + 1 if use_time else input_dim

        self.gru = nn.GRU(
            input_size=gru_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        gru_output_dim = hidden_dim * 2 if bidirectional else hidden_dim

        if pooling == 'attention':
            self.pooling = AttentionPooling(gru_output_dim)
        else:
            self.pooling = None

        self.classifier = nn.Linear(gru_output_dim, num_classes)

    def forward(self, x, mask=None, time_log=None, flow_feats=None, return_embedding=False):
        B, L, D = x.shape
        device = x.device

        if self.use_time and time_log is not None:
            time_feat = time_log.unsqueeze(-1)
            x = torch.cat([x, time_feat], dim=-1)

        if mask is not None:
            lengths = mask.sum(dim=1).long()
            lengths = torch.clamp(lengths, min=1)
            x_packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            gru_out, hidden = self.gru(x_packed)
            gru_out, _ = nn.utils.rnn.pad_packed_sequence(
                gru_out, batch_first=True, total_length=L
            )
        else:
            gru_out, hidden = self.gru(x)

        if self.pooling_type == 'attention':
            if self.pooling is None:
                self.pooling = AttentionPooling(gru_out.size(-1)).to(device)
            z = self.pooling(gru_out, mask)
        elif self.pooling_type == 'mean':
            if mask is not None:
                mask_expanded = mask.unsqueeze(-1).float()
                z = (gru_out * mask_expanded).sum(dim=1) / (mask_expanded.sum(dim=1) + 1e-8)
            else:
                z = gru_out.mean(dim=1)
        elif self.pooling_type == 'last':
            if mask is not None:
                last_indices = (lengths - 1).clamp(min=0).unsqueeze(-1).unsqueeze(-1)
                last_indices = last_indices.expand(-1, gru_out.size(-1)).unsqueeze(1)
                z = gru_out.gather(1, last_indices).squeeze(1)
            else:
                z = gru_out[:, -1, :]

        logits = self.classifier(z)

        if return_embedding:
            return logits, z, gru_out
        return logits


class CNN1DClassifier(nn.Module):
    """
    1D CNN 序列分类器

    使用多个卷积核提取局部模式，类似文本分类的 TextCNN
    """

    def __init__(
            self,
            input_dim: int,
            num_filters: int = 128,
            kernel_sizes: list = [3, 5, 7],
            dropout: float = 0.3,
            pooling: str = 'adaptive',  # 'adaptive', 'max'
            num_classes: int = 2,
            use_time: bool = False
    ):
        super().__init__()
        self.input_dim = input_dim
        self.use_time = use_time
        cnn_input_dim = input_dim + 1 if use_time else input_dim

        # 多个并行的卷积层（不同核大小）
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=cnn_input_dim,
                out_channels=num_filters,
                kernel_size=k,
                padding='same'  # 保持序列长度
            )
            for k in kernel_sizes
        ])

        total_dim = num_filters * len(kernel_sizes)

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(total_dim, num_classes)

        self.pooling_type = pooling

    def forward(self, x, mask=None, time_log=None, flow_feats=None, return_embedding=False):
        """
        Args:
            x: [B, L, D]
        """
        B, L, D = x.shape

        # 可选：拼接时间特征
        if self.use_time and time_log is not None:
            time_feat = time_log.unsqueeze(-1)
            x = torch.cat([x, time_feat], dim=-1)

        # 对掩码位置置零
        if mask is not None:
            x = x * mask.unsqueeze(-1).float()

        # [B, L, D] -> [B, D, L] (Conv1d 要求)
        x = x.transpose(1, 2)

        conv_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(x))  # [B, num_filters, L]

            if self.pooling_type == 'adaptive':
                # 自适应平均池化 -> [B, num_filters, 1]
                pooled = F.adaptive_avg_pool1d(conv_out, 1).squeeze(-1)
            elif self.pooling_type == 'max':
                # 全局最大池化（考虑掩码）
                if mask is not None:
                    # 将填充位置设为很小的值
                    mask_expanded = mask.unsqueeze(1).float()  # [B, 1, L]
                    conv_out = conv_out * mask_expanded + (1 - mask_expanded) * (-1e9)
                pooled = F.max_pool1d(conv_out, conv_out.size(-1)).squeeze(-1)

            conv_outputs.append(pooled)

        # 拼接所有卷积核的输出
        z = torch.cat(conv_outputs, dim=1)  # [B, num_filters * len(kernel_sizes)]
        z = self.dropout(z)

        logits = self.classifier(z)

        if return_embedding:
            return logits, z, None
        return logits


class StandardTransformer(nn.Module):
    """
    标准 Transformer 分类器（仅位置编码，无时间感知）

    与你的 Time-Aware Transformer 结构相同，但使用标准位置编码
    """

    def __init__(
            self,
            input_dim: int,
            d_model: int = 128,
            num_heads: int = 8,
            num_layers: int = 3,
            dim_feedforward: int = 256,
            dropout: float = 0.1,
            max_seq_len: int = 32,
            num_classes: int = 2,
            use_flow_features: bool = False,
            flow_feature_dim: Optional[int] = None
    ):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.use_flow_features = use_flow_features

        # 线性投影到 d_model
        self.input_projection = nn.Linear(input_dim, d_model)

        # 标准位置编码（仅位置）
        self.position_encoding = PositionalEncoding(d_model, max_len=max_seq_len)

        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-LN
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # 注意力池化
        self.attention_pooling = AttentionPooling(d_model)

        # 可选的流特征编码器
        if use_flow_features and flow_feature_dim is not None:
            self.flow_encoder = FlowFeatureEncoder(flow_feature_dim, d_model)
            self.fusion = FlowFusion(d_model, method='gated')
        else:
            self.flow_encoder = None

        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x, mask=None, time_log=None, flow_feats=None, return_embedding=False):
        """
        Args:
            x: [B, L, D]
            mask: [B, L]
            time_log: [B, L] - 忽略（不使用时间信息）
            flow_feats: [B, D_flow] or None
        """
        B, L, D = x.shape

        # 投影
        h = self.input_projection(x)  # [B, L, d_model]

        # 仅添加位置编码（不使用时间）
        h = self.position_encoding(h)

        # Transformer 编码
        if mask is not None:
            # TransformerEncoder 需要 mask 为 [L, L] 或 [B*num_heads, L, L]
            # 或 src_key_padding_mask 为 [B, L]
            h = self.transformer(h, src_key_padding_mask=~mask)
        else:
            h = self.transformer(h)

        # 池化
        z = self.attention_pooling(h, mask)

        # 融合流特征
        if self.flow_encoder is not None and flow_feats is not None:
            z_flow = self.flow_encoder(flow_feats)
            z = self.fusion(z, z_flow)

        logits = self.classifier(z)

        if return_embedding:
            return logits, z, h
        return logits


# ============================================
# 辅助模块（从你的 model.py 复用）
# ============================================

class AttentionPooling(nn.Module):
    """基于注意力的池化层"""

    def __init__(self, input_dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = input_dim // 2

        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: [B, L, D]
            mask: [B, L]
        Returns:
            z: [B, D]
        """
        attn_weights = self.attention(x).squeeze(-1)  # [B, L]

        if mask is not None:
            attn_weights = attn_weights.masked_fill(~mask, -1e9)

        attn_weights = F.softmax(attn_weights, dim=-1).unsqueeze(-1)  # [B, L, 1]

        z = (x * attn_weights).sum(dim=1)  # [B, D]
        return z


class PositionalEncoding(nn.Module):
    """标准正弦位置编码"""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class FlowFeatureEncoder(nn.Module):
    """流特征编码器"""

    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, x):
        return self.encoder(x)


class FlowFusion(nn.Module):
    """流特征门控融合"""

    def __init__(self, d_model: int, method: str = 'gated'):
        super().__init__()
        self.method = method

        if method == 'gated':
            self.gate = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.Sigmoid()
            )
            self.post_gate = nn.Linear(d_model, d_model)

    def forward(self, pooled, flow_encoded):
        if self.method == 'gated':
            gate_input = torch.cat([pooled, flow_encoded], dim=-1)
            gate = self.gate(gate_input)
            fused = gate * pooled + (1 - gate) * flow_encoded
            return self.post_gate(fused)
        elif self.method == 'concat':
            return torch.cat([pooled, flow_encoded], dim=-1)
        elif self.method == 'add':
            return pooled + flow_encoded
        return pooled