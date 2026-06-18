# models/comparison.py
"""
模型对比实验管理器
统一训练、评估、记录所有基线模型
"""

import os
import json
import time
import copy
import pickle
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report
)
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier

from stage1.utils import set_seed
from .baselines import (
    FlowLevelMLP, LSTMClassifier, BiLSTMClassifier,
    GRUClassifier, CNN1DClassifier, StandardTransformer
)
from .model import Stage1TimeAwareTransformer
# 损失函数和优化器
from .losses import FocalLossWithLabelSmoothing


class ModelBenchmark:
    """
    统一的模型基准测试框架

    用法:
        benchmark = ModelBenchmark(config)
        benchmark.add_baseline_models()
        benchmark.add_your_model()
        results = benchmark.run_all()
        benchmark.generate_report(results)
    """

    def __init__(
            self,
            input_dim,
            flow_feature_dim,
            config: Dict[str, Any],
            train_loader: DataLoader,
            val_loader: DataLoader,
            test_loader: DataLoader,
            device: str = 'cuda'
    ):
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device

        self.models = {}  # name -> model_instance
        self.results = {}  # name -> metrics_dict

        # 从 config 提取关键参数
        self.input_dim = input_dim
        self.max_seq_len = config['sequence']['max_seq_len']
        self.num_classes = config['model'].get('num_classes',2)

        # 检查是否有流特征（方案C）
        self.has_flow_feats = (flow_feature_dim is not None and flow_feature_dim > 0)
        self.flow_feature_dim = flow_feature_dim

    def add_baseline_models(self):
        """添加所有基线模型"""
        print("=" * 60)
        print("添加基线模型...self.input_dim", self.input_dim)
        print("=" * 60)

        # ============================================
        # Level 2: 经典深度学习
        # ============================================
        # self.models['MLP'] = FlowLevelMLP(
        #     input_dim=self.flow_feature_dim if self.has_flow_feats else self.input_dim,
        #     hidden_dims=[256, 128, 64],
        #     dropout=0.3,
        #     num_classes=self.num_classes
        # )
        # print("[✓] MLP")

        if self.has_flow_feats:
            self.models['MLP_with_flow'] = FlowLevelMLP(
                input_dim=self.flow_feature_dim,
                hidden_dims=[256, 128, 64],
                dropout=0.3,
                num_classes=self.num_classes
            )
            print("[✓] MLP (flow features only)")

        # ============================================
        # Level 3: 序列模型（无时间感知）
        # ============================================
        # self.models['LSTM-attention'] = LSTMClassifier(
        #     input_dim=self.input_dim,
        #     hidden_dim=128,
        #     num_layers=2,
        #     dropout=0.2,
        #     bidirectional=False,
        #     pooling='attention',
        #     num_classes=self.num_classes
        # )
        # print("[✓] LSTM-attention")

        self.models['LSTM'] = LSTMClassifier(
            input_dim=self.input_dim,
            hidden_dim=128,
            num_layers=2,
            dropout=0.2,
            bidirectional=False,
            pooling='mean',
            num_classes=self.num_classes
        )
        print("[✓] LSTM-mean")

        self.models['BiLSTM'] = BiLSTMClassifier(
            input_dim=self.input_dim,
            hidden_dim=128,
            num_layers=2,
            dropout=0.2,
            pooling='mean',
            num_classes=self.num_classes
        )
        print("[✓] BiLSTM-mean")

        # self.models['GRU-attention'] = GRUClassifier(
        #     input_dim=self.input_dim,
        #     hidden_dim=128,
        #     num_layers=2,
        #     dropout=0.2,
        #     bidirectional=False,
        #     pooling='attention',
        #     num_classes=self.num_classes
        # )
        # print("[✓] GRU-attention")

        self.models['GRU'] = GRUClassifier(
            input_dim=self.input_dim,
            hidden_dim=128,
            num_layers=2,
            dropout=0.2,
            bidirectional=False,
            pooling='mean',
            num_classes=self.num_classes
        )
        print("[✓] GRU-mean")

        # self.models['CNN1D-adaptive'] = CNN1DClassifier(
        #     input_dim=self.input_dim,
        #     num_filters=128,
        #     kernel_sizes=[3, 5, 7],
        #     dropout=0.3,
        #     pooling='adaptive',
        #     num_classes=self.num_classes
        # )
        # print("[✓] CNN-1D-adaptive")

        self.models['CNN1D'] = CNN1DClassifier(
            input_dim=self.input_dim,
            num_filters=128,
            kernel_sizes=[3, 5, 7],
            dropout=0.3,
            pooling='max',
            num_classes=self.num_classes
        )
        print("[✓] CNN-1D-max")

        self.models['StandardTransformer'] = StandardTransformer(
            input_dim=self.input_dim,
            d_model=128,
            num_heads=4,
            num_layers=2,
            dim_feedforward=128,
            dropout=0.1,
            max_seq_len=self.max_seq_len,
            num_classes=self.num_classes,
            use_flow_features=self.has_flow_feats,
            flow_feature_dim=self.flow_feature_dim
        )
        print("[✓] Standard Transformer")

        # ============================================
        # Level 3 变体：序列模型 + 时间输入
        # ============================================
        self.models['LSTM_with_time'] = LSTMClassifier(
            input_dim=self.input_dim,
            hidden_dim=128,
            num_layers=2,
            dropout=0.2,
            bidirectional=False,
            use_time=True,  # 拼接时间特征
            pooling='mean',
            num_classes=self.num_classes
        )
        print("[✓] LSTM-mean + Time Feature")

        self.models['CNN1D_with_time'] = CNN1DClassifier(
            input_dim=self.input_dim,
            num_filters=128,
            kernel_sizes=[3, 5, 7],
            dropout=0.3,
            use_time=True,
            pooling='max',
            num_classes=self.num_classes
        )
        print("[✓] CNN-1D-max + Time Feature")

        print(f"共添加 {len(self.models)} 个基线模型\n")

    def add_ablations(self, base_config: Dict[str, Any]):
        """添加消融模型变体"""
        print("=" * 60)
        print("添加消融模型...")
        print("=" * 60)

        # 1. 无时间感知（use_time_encoding=False）
        config_no_time = copy.deepcopy(base_config)
        config_no_time['model']['use_time_encoding'] = False
        self.models['Ours_PositionOnly'] = Stage1TimeAwareTransformer(
            input_dim=self.input_dim,
            cfg=config_no_time
        )
        print("[✓] Ours - No Time Awareness - PositionOnly")

        # 2. 无位置编码（use_positional_encoding=False）
        config_no_pos = copy.deepcopy(base_config)
        config_no_pos['model']['use_positional_encoding'] = False
        self.models['Ours_TimeOnly'] = Stage1TimeAwareTransformer(
            input_dim=self.input_dim,
            cfg=config_no_pos
        )
        print("[✓] Ours - No Position Encoding - TimeOnly")

        # 3. 无时间、无位置（纯内容）
        config_none = copy.deepcopy(base_config)
        config_none['model']['use_time_encoding'] = False
        config_none['model']['use_positional_encoding'] = False
        self.models['Ours_NoEncoding'] = Stage1TimeAwareTransformer(
            input_dim=self.input_dim,
            cfg=config_none
        )
        print("[✓] Ours - No Encoding")

        # # 4. 既有时间、又有位置
        # config_both = copy.deepcopy(base_config)
        # self.models['Ours_BothEncoding'] = Stage1TimeAwareTransformer(
        #     input_dim=self.input_dim,
        #     cfg=config_both
        # )
        # print("[✓] Ours - Both Time and Position Encoding")
        #
        # # 5. 无流特征（如果使用了方案C）
        # if self.has_flow_feats:
        #     config_no_flow = copy.deepcopy(base_config)
        #     config_no_flow['fusion'] = {'method': 'none'}
        #     self.models['Ours_NoFlowFeats'] = Stage1TimeAwareTransformer(
        #         input_dim=self.input_dim,
        #         cfg=config_no_flow
        #     )
        #     print("[✓] Ours - No Flow Features")

        print(f"共添加 {len(self.models)} 个模型（含消融）\n")

    def train_deep_model(
            self,
            model: nn.Module,
            model_name: str,
            num_epochs: int = 50,
            lr: float = 1e-3,
            weight_decay: float = 1e-4,
            patience: int = 10,
            verbose: bool = True
    ) -> Dict[str, Any]:
        """训练深度学习模型"""
        print(f"\n{'@' * 50}")
        print(f"train_deep_model: {model_name},num_epochs={num_epochs},lr={lr},weight_decay={weight_decay},patience={patience},verbose={verbose}")
        print(f"{'@' * 50}")
        model = model.to(self.device)

        # 计算类别权重（处理不平衡）
        alpha = self._compute_class_weights()

        criterion = FocalLossWithLabelSmoothing(
            alpha=alpha, gamma=2.0, label_smoothing=0.1
        )
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

        # 学习率调度
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs
        )
        # ⚠️ 如果需要时使用 scheduler，种子不影响
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer, mode='max', patience=5
        # )

        # 训练循环
        best_val_f1 = 0.0
        best_model_state = None
        patience_counter = 0
        history = {'train_loss': [], 'val_loss': [], 'val_f1_class1': []}

        for epoch in range(num_epochs):
            # 训练阶段
            model.train()
            train_loss = 0.0
            for batch in self.train_loader:
                x = batch['x'].to(self.device)
                mask = batch.get('mask', None)
                if mask is not None:
                    mask = mask.to(self.device)
                time_log = batch.get('time', None)
                if time_log is not None:
                    time_log = time_log.to(self.device)
                flow_feats = batch.get('flow_feats', None)
                if flow_feats is not None:
                    flow_feats = flow_feats.to(self.device)
                labels = batch['label'].to(self.device)

                optimizer.zero_grad()
                logits = model(x, mask=mask, time_log=time_log, flow_feats=flow_feats)
                loss = criterion(logits, labels)
                loss.backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(self.train_loader)
            history['train_loss'].append(train_loss)

            # 验证阶段
            val_metrics = self.evaluate_model(model)
            history['val_loss'].append(val_metrics['loss_mean'])
            history['val_f1_class1'].append(val_metrics['f1_class1'])

            if val_metrics['f1_class1'] > best_val_f1:
                best_val_f1 = val_metrics['f1_class1']
                best_model_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if verbose and (epoch + 1) % 5 == 0:
                print(f"[{model_name}] Epoch {epoch + 1}/{num_epochs} | "
                      f"Train Loss: {train_loss:.4f} | "
                      f"Val Loss: {val_metrics['loss_mean']:.4f} | "
                      f"Val F1_class1: {val_metrics['f1_class1']:.4f} | "
                      f"Best F1_class1: {best_val_f1:.4f}")

            scheduler.step()

            if patience_counter >= patience:
                if verbose:
                    print(f"[{model_name}] Early stopping at epoch {epoch + 1}")
                break

        # 恢复最佳模型
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        return history

    def train_ml_model(
            self,
            model,
            model_name: str
    ) -> Dict[str, Any]:
        """训练传统机器学习模型（XGBoost/LightGBM/Random Forest）"""

        # 准备数据：聚合包特征 + 流特征
        X_train, y_train = self._prepare_flow_level_data(self.train_loader)
        X_val, y_val = self._prepare_flow_level_data(self.val_loader)

        if model_name == 'XGBoost':
            # 计算样本权重（处理不平衡）
            scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
            model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                scale_pos_weight=scale_pos_weight,
                use_label_encoder=False,
                eval_metric='logloss',
                random_state=42
            )
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                # early_stopping_rounds=20,
                verbose=False
            )
        elif model_name == 'LightGBM':
            class_weight = {
                0: 1.0,
                1: (y_train == 0).sum() / (y_train == 1).sum()
            }
            model = lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                class_weight=class_weight,
                random_state=42
            )
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)]
            )
        elif model_name == 'RandomForest':
            class_weight = {
                0: 1.0,
                1: (y_train == 0).sum() / (y_train == 1).sum()
            }
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train, y_train)

        return model

    def evaluate_model(
            self,
            model: nn.Module,
            loader: Optional[DataLoader] = None,
            threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        评估深度学习模型

        Args:
            model: 模型
            loader: 数据加载器（None 则使用验证集）
            threshold: 决策阈值（None 则在验证集上搜索最优阈值）
        """
        if loader is None:
            loader = self.val_loader

        model.eval()
        all_probs = []
        all_labels = []
        all_losses = []

        criterion = nn.CrossEntropyLoss(reduction='none')

        with torch.no_grad():
            for batch in loader:
                x = batch['x'].to(self.device)
                mask = batch.get('mask', None)
                if mask is not None:
                    mask = mask.to(self.device)
                time_log = batch.get('time', None)
                if time_log is not None:
                    time_log = time_log.to(self.device)
                flow_feats = batch.get('flow_feats', None)
                if flow_feats is not None:
                    flow_feats = flow_feats.to(self.device)
                labels = batch['label'].to(self.device)

                logits = model(x, mask=mask, time_log=time_log, flow_feats=flow_feats)
                probs = torch.softmax(logits, dim=1)

                loss = criterion(logits, labels)

                all_probs.append(probs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                all_losses.append(loss.cpu().numpy())

        all_probs = np.concatenate(all_probs, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        all_losses = np.concatenate(all_losses, axis=0)

        # 二分类：取正类概率
        if all_probs.shape[1] == 2:
            scores = all_probs[:, 1]
        else:
            scores = all_probs.max(axis=1)

        # 搜索最优阈值
        if threshold is None:
            threshold = self._find_optimal_threshold(scores, all_labels)

        predictions = (scores >= threshold).astype(int)

        # 计算指标
        metrics = {
            'threshold': threshold,
            'loss_mean': all_losses.mean(),
            'accuracy': accuracy_score(all_labels, predictions),
            'precision_macro': precision_score(all_labels, predictions, average='macro', zero_division=0),
            'recall_macro': recall_score(all_labels, predictions, average='macro', zero_division=0),
            'f1_macro': f1_score(all_labels, predictions, average='macro', zero_division=0),
            'precision_weighted': precision_score(all_labels, predictions, average='weighted', zero_division=0),
            'recall_weighted': recall_score(all_labels, predictions, average='weighted', zero_division=0),
            'f1_weighted': f1_score(all_labels, predictions, average='weighted', zero_division=0),
            # 针对少数类（label=1）的指标
            'precision_class1': precision_score(all_labels, predictions, pos_label=1, zero_division=0),
            'recall_class1': recall_score(all_labels, predictions, pos_label=1, zero_division=0),
            'f1_class1': f1_score(all_labels, predictions, pos_label=1, zero_division=0),
            # AUC 指标（不需要阈值）
            'roc_auc': roc_auc_score(all_labels, scores) if len(np.unique(all_labels)) > 1 else 0.5,
            'pr_auc': average_precision_score(all_labels, scores) if len(np.unique(all_labels)) > 1 else 0.5,
            # 混淆矩阵
            'confusion_matrix': confusion_matrix(all_labels, predictions).tolist(),
            # 预测分数分布
            'prob_mean': float(scores.mean()),
            'prob_std': float(scores.std()),
        }

        return metrics

    def evaluate_ml_model(
            self,
            model,
            loader: DataLoader,
            model_name: str
    ) -> Dict[str, Any]:
        """评估传统机器学习模型"""
        X, y = self._prepare_flow_level_data(loader)

        # 预测
        probs = model.predict_proba(X)
        if probs.shape[1] == 2:
            scores = probs[:, 1]
        else:
            scores = probs.max(axis=1)

        # 搜索最优阈值
        threshold = self._find_optimal_threshold(scores, y)
        predictions = (scores >= threshold).astype(int)

        metrics = {
            'threshold': threshold,
            'accuracy': accuracy_score(y, predictions),
            'precision_macro': precision_score(y, predictions, average='macro', zero_division=0),
            'recall_macro': recall_score(y, predictions, average='macro', zero_division=0),
            'f1_macro': f1_score(y, predictions, average='macro', zero_division=0),
            'precision_weighted': precision_score(y, predictions, average='weighted', zero_division=0),
            'recall_weighted': recall_score(y, predictions, average='weighted', zero_division=0),
            'f1_weighted': f1_score(y, predictions, average='weighted', zero_division=0),
            'precision_class1': precision_score(y, predictions, pos_label=1, zero_division=0),
            'recall_class1': recall_score(y, predictions, pos_label=1, zero_division=0),
            'f1_class1': f1_score(y, predictions, pos_label=1, zero_division=0),
            'roc_auc': roc_auc_score(y, scores) if len(np.unique(y)) > 1 else 0.5,
            'pr_auc': average_precision_score(y, scores) if len(np.unique(y)) > 1 else 0.5,
            'confusion_matrix': confusion_matrix(y, predictions).tolist(),
        }

        return metrics

    def run_all(self, num_epochs: int = 50, seed: int = 42) -> Dict[str, Any]:
        """
        运行所有模型的训练和评估

        Returns:
            包含所有模型结果的字典
        """
        print("\n" + "=" * 70)
        print("开始模型对比实验")
        print("=" * 70)
        all_results = {}

        for model_name, model in self.models.items():
            print(f"\n{'─' * 50}")
            print(f"训练模型: {model_name}")
            print(f"{'─' * 50}")
            set_seed(seed)

            start_time = time.time()

            # 判断模型类型
            if model_name in ['XGBoost', 'LightGBM', 'RandomForest']:
                # 传统机器学习模型
                model = self.train_ml_model(model, model_name)

                # 评估
                val_metrics = self.evaluate_ml_model(model, self.val_loader, model_name)
                test_metrics = self.evaluate_ml_model(model, self.test_loader, model_name)
            else:
                # 深度学习模型
                history = self.train_deep_model(
                    model, model_name,
                    num_epochs=num_epochs,
                    lr=1e-3,
                    weight_decay=1e-4,
                    patience=10
                )

                # 评估
                val_metrics = self.evaluate_model(model, self.val_loader)
                test_metrics = self.evaluate_model(model, self.test_loader, threshold=val_metrics['threshold'])

                # 保存训练历史
                val_metrics['training_history'] = history

            training_time = time.time() - start_time

            # 计算模型参数数量
            if hasattr(model, 'parameters'):
                # PyTorch模型
                num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            elif isinstance(model, (xgb.XGBClassifier, lgb.LGBMClassifier, RandomForestClassifier)):
                # sklearn模型：估计树的数量
                num_params = model.n_estimators
            else:
                num_params = 0

            # 汇总结果
            result = {
                'model_name': model_name,
                'num_params': num_params,
                'training_time_seconds': training_time,
                'val_metrics': val_metrics,
                'test_metrics': test_metrics,
            }
            all_results[model_name] = result

            # 实时输出核心指标
            print(f"\n[{model_name}] 验证集:")
            print(f"  F1 (macro): {val_metrics['f1_macro']:.4f}")
            print(f"  F1 (class1): {val_metrics['f1_class1']:.4f}")
            print(f"  ROC-AUC: {val_metrics['roc_auc']:.4f}")
            print(f"  PR-AUC: {val_metrics['pr_auc']:.4f}")
            print(f"[{model_name}] 测试集:")
            print(f"  F1 (macro): {test_metrics['f1_macro']:.4f}")
            print(f"  F1 (class1): {test_metrics['f1_class1']:.4f}")
            print(f"  ROC-AUC: {test_metrics['roc_auc']:.4f}")
            print(f"  PR-AUC: {test_metrics['pr_auc']:.4f}")
            print(f"  训练时间: {training_time:.1f}s")

        self.results = all_results
        return all_results

    def generate_report(
            self,
            results: Dict[str, Any],
            output_dir: str = 'results/comparison'
    ):
        """
        生成详细的对比报告

        包含：表格、图表、统计分析
        """
        os.makedirs(output_dir, exist_ok=True)

        # ============================================
        # 1. 生成对比表格
        # ============================================
        self._generate_comparison_tables(results, output_dir)

        # ============================================
        # 2. 生成对比图表
        # ============================================
        self._generate_comparison_plots(results, output_dir)

        # ============================================
        # 3. 统计分析：显著性检验
        # ============================================
        self._generate_statistical_analysis(results, output_dir)

        # ============================================
        # 4. 保存原始结果
        # ============================================
        results_file = os.path.join(output_dir, 'all_results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n原始结果已保存至: {results_file}")

        print(f"\n对比实验完成！报告保存在: {output_dir}")

    # ============================================
    # 辅助方法
    # ============================================

    def _prepare_flow_level_data(self, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """
        将包级数据聚合为流级特征

        Returns:
            X: [N, D] 聚合后的流特征
            y: [N] 标签
        """
        all_features = []
        all_labels = []

        for batch in loader:
            x = batch['x']  # [B, L, D]
            mask = batch.get('mask', None)
            flow_feats = batch.get('flow_feats', None)

            # 策略1：优先使用已有的流级特征
            if flow_feats is not None:
                features = flow_feats.numpy()
            elif mask is not None:
                # 策略2：对有效包取均值
                x_np = x.numpy()
                mask_np = mask.numpy()
                mask_expanded = mask_np[:, :, np.newaxis]
                features = (x_np * mask_expanded).sum(axis=1) / (mask_expanded.sum(axis=1) + 1e-8)
            else:
                features = x.mean(dim=1).numpy()

            all_features.append(features)
            all_labels.append(batch['label'].numpy())

        X = np.concatenate(all_features, axis=0)
        y = np.concatenate(all_labels, axis=0)

        return X, y

    def _compute_class_weights(self) -> torch.Tensor:
        """计算类别权重（处理不平衡）"""
        all_labels = []
        for batch in self.train_loader:
            all_labels.append(batch['label'].numpy())
        all_labels = np.concatenate(all_labels, axis=0)

        # 计算每个类别的样本数
        unique, counts = np.unique(all_labels, return_counts=True)
        n_samples = len(all_labels)
        n_classes = len(unique)

        # 逆频率权重
        alpha = torch.zeros(n_classes, dtype=torch.float32)
        for i, count in zip(unique, counts):
            alpha[i] = n_samples / (n_classes * count)

        return alpha

    def _find_optimal_threshold(
            self,
            scores: np.ndarray,
            labels: np.ndarray
    ) -> float:
        """在验证集上搜索最优决策阈值（最大化 F1）"""
        best_threshold = 0.5
        best_f1 = 0.0

        for threshold in np.arange(0.1, 0.95, 0.02):
            predictions = (scores >= threshold).astype(int)
            f1 = f1_score(labels, predictions, average='macro', zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        return best_threshold

    def _generate_comparison_tables(
            self,
            results: Dict[str, Any],
            output_dir: str
    ):
        """生成对比表格（CSV 和 LaTeX）"""

        # 提取主要指标
        rows = []
        for model_name, result in results.items():
            test_metrics = result['test_metrics']
            row = {
                'Model': model_name,
                'Params': result['num_params'],
                'Time (s)': f"{result['training_time_seconds']:.1f}",
                'Accuracy': f"{test_metrics['accuracy']:.4f}",
                'F1 Macro': f"{test_metrics['f1_macro']:.4f}",
                'F1 Class1': f"{test_metrics['f1_class1']:.4f}",
                'Precision Class1': f"{test_metrics['precision_class1']:.4f}",
                'Recall Class1': f"{test_metrics['recall_class1']:.4f}",
                'ROC-AUC': f"{test_metrics['roc_auc']:.4f}",
                'PR-AUC': f"{test_metrics['pr_auc']:.4f}",
                'threshold': f"{test_metrics['threshold']:.4f}",
                'precision_macro': f"{test_metrics['precision_macro']:.4f}",
                'recall_macro': f"{test_metrics['recall_macro']:.4f}",
                'precision_weighted': f"{test_metrics['precision_weighted']:.4f}",
                'recall_weighted': f"{test_metrics['recall_weighted']:.4f}",
                'f1_weighted': f"{test_metrics['f1_weighted']:.4f}",
                'confusion_matrix': f"{test_metrics['confusion_matrix']}"
            }
            rows.append(row)

        df = pd.DataFrame(rows)

        # 按 F1 Macro 排序
        df = df.sort_values('F1 Macro', ascending=False)

        # 保存 CSV
        csv_path = os.path.join(output_dir, 'comparison_table.csv')
        df.to_csv(csv_path, index=False)
        print(f"\n对比表格已保存至: {csv_path}")

        # 生成 LaTeX 表格
        latex_path = os.path.join(output_dir, 'comparison_table.tex')
        with open(latex_path, 'w') as f:
            f.write(df.to_latex(index=False, escape=False, column_format='l' + 'c' * (len(df.columns) - 1)))
        print(f"LaTeX 表格已保存至: {latex_path}")

        # 分模型组对比
        model_groups = {
            'ML Models': ['XGBoost', 'LightGBM', 'RandomForest'],
            'DL Without Sequence': ['MLP', 'MLP_with_flow'],
            'Sequence Models': ['LSTM', 'BiLSTM', 'GRU', 'CNN1D', 'StandardTransformer'],
            'Sequence + Raw Time': ['LSTM_with_time', 'CNN1D_with_time'],
            'Ablation Models': [k for k in results.keys() if k.startswith('Ours_')],
            'Time-Aware Transformer': ['Time-Aware Transformer']
        }

        # 为每组生成子表格
        for group_name, model_names in model_groups.items():
            group_df = df[df['Model'].isin(model_names)]
            if len(group_df) > 0:
                group_csv = os.path.join(output_dir, f'comparison_{group_name.replace(" ", "_")}.csv')
                group_df.to_csv(group_csv, index=False)
                print(f"  {group_name}: {len(group_df)} 个模型")

    def _generate_comparison_plots(
            self,
            results: Dict[str, Any],
            output_dir: str
    ):
        """生成对比图表"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')  # 非交互式后端

            # 提取数据
            model_names = []
            f1_macro_values = []
            f1_class1_values = []
            roc_auc_values = []
            pr_auc_values = []
            training_times = []

            for model_name, result in results.items():
                model_names.append(model_name)
                test_metrics = result['test_metrics']
                f1_macro_values.append(test_metrics['f1_macro'] * 100)
                f1_class1_values.append(test_metrics['f1_class1'] * 100)
                roc_auc_values.append(test_metrics['roc_auc'] * 100)
                pr_auc_values.append(test_metrics['pr_auc'] * 100)
                training_times.append(result['training_time_seconds'])

            # ============================================
            # 图1：性能对比条形图
            # ============================================
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))

            # 按F1排序的索引
            sorted_idx = np.argsort(f1_macro_values)

            # F1 Macro
            ax = axes[0, 0]
            colors = ['#FF6B6B' if 'Time-Aware' in name else
                      '#4ECDC4' if name.startswith('Ours_') else
                      '#45B7D1' for name in model_names]
            bars = ax.barh(
                [model_names[i] for i in sorted_idx],
                [f1_macro_values[i] for i in sorted_idx],
                color=[colors[i] for i in sorted_idx],
                edgecolor='black',
                linewidth=0.5
            )
            ax.set_xlabel('F1 Macro (%)', fontsize=12)
            ax.set_title('Overall Performance (F1 Macro)', fontsize=14, fontweight='bold')
            # ax.axvline(x=f1_macro_values[sorted_idx][-1], color='red', linestyle='--', alpha=0.7)
            sorted_values = np.sort(f1_macro_values)
            best_f1 = float(sorted_values[-1])
            ax.axvline(x=best_f1, color='red', linestyle='--', alpha=0.7)
            ax.grid(axis='x', alpha=0.3)

            # 在条形上显示数值
            for i, (bar, val) in enumerate(zip(bars,
                                               [f1_macro_values[i] for i in sorted_idx])):
                ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                        f'{val:.1f}%', va='center', fontsize=9)

            # F1 Class1
            ax = axes[0, 1]
            sorted_idx_class1 = np.argsort(f1_class1_values)
            bars = ax.barh(
                [model_names[i] for i in sorted_idx_class1],
                [f1_class1_values[i] for i in sorted_idx_class1],
                color=[colors[i] for i in sorted_idx_class1],
                edgecolor='black',
                linewidth=0.5
            )
            ax.set_xlabel('F1 (Attack Class) %', fontsize=12)
            ax.set_title('Attack Detection Performance', fontsize=14, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)

            for i, (bar, val) in enumerate(zip(bars,
                                               [f1_class1_values[i] for i in sorted_idx_class1])):
                ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                        f'{val:.1f}%', va='center', fontsize=9)

            # ROC-AUC vs PR-AUC
            ax = axes[1, 0]
            scatter = ax.scatter(
                roc_auc_values, pr_auc_values,
                c=np.arange(len(model_names)),
                cmap='viridis',
                s=100,
                edgecolor='black',
                linewidth=0.5,
                alpha=0.8
            )
            ax.set_xlabel('ROC-AUC (%)', fontsize=12)
            ax.set_ylabel('PR-AUC (%)', fontsize=12)
            ax.set_title('ROC-AUC vs PR-AUC', fontsize=14, fontweight='bold')
            ax.plot([50, 100], [0, 100], 'r--', alpha=0.3, label='Random')
            ax.legend()
            ax.grid(alpha=0.3)

            # 标注模型名
            for i, name in enumerate(model_names):
                ax.annotate(name, (roc_auc_values[i], pr_auc_values[i]),
                            xytext=(5, 5), textcoords='offset points',
                            fontsize=7, alpha=0.8)

            # 训练时间
            ax = axes[1, 1]
            sorted_idx_time = np.argsort(training_times)
            bars = ax.barh(
                [model_names[i] for i in sorted_idx_time],
                [training_times[i] for i in sorted_idx_time],
                color=[colors[i] for i in sorted_idx_time],
                edgecolor='black',
                linewidth=0.5
            )
            ax.set_xlabel('Training Time (seconds)', fontsize=12)
            ax.set_title('Computational Efficiency', fontsize=14, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)

            # 时间标注
            for bar, val in zip(bars, [training_times[i] for i in sorted_idx_time]):
                ax.text(val + 1, bar.get_y() + bar.get_height() / 2,
                        f'{val:.0f}s', va='center', fontsize=9)

            plt.tight_layout()
            plot_path = os.path.join(output_dir, 'comparison_plots.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"\n对比图表已保存至: {plot_path}")

            # ============================================
            # 图2：消融实验专项图
            # ============================================
            ablation_models = {k: v for k, v in results.items() if k.startswith('Ours_')}
            if len(ablation_models) > 0:
                fig, ax = plt.subplots(figsize=(12, 6))

                ablation_names = []
                ablation_f1 = []
                for name, result in ablation_models.items():
                    ablation_names.append(name.replace('Ours_', ''))
                    ablation_f1.append(result['test_metrics']['f1_macro'] * 100)

                sorted_idx = np.argsort(ablation_f1)[::-1]

                colors_ablation = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(ablation_names)))
                bars = ax.bar(
                    [ablation_names[i] for i in sorted_idx],
                    [ablation_f1[i] for i in sorted_idx],
                    color=[colors_ablation[i] for i in sorted_idx],
                    edgecolor='black',
                    linewidth=1
                )

                ax.set_ylabel('F1 Macro (%)', fontsize=12)
                ax.set_title('Ablation Study: Impact of Each Component', fontsize=14, fontweight='bold')
                ax.grid(axis='y', alpha=0.3)
                plt.xticks(rotation=45, ha='right')

                # 数值标注
                for bar, val in zip(bars, [ablation_f1[i] for i in sorted_idx]):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                            f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

                # 差异标注
                if len(ablation_f1) > 0:
                    max_f1 = max(ablation_f1)
                    for bar, val in zip(bars, [ablation_f1[i] for i in sorted_idx]):
                        if val < max_f1:
                            diff = max_f1 - val
                            ax.text(bar.get_x() + bar.get_width() / 2., height + 3.5,
                                    f'Δ{max_f1:.1f}%', ha='center', va='bottom',
                                    fontsize=8, color='red', fontstyle='italic')

                plt.tight_layout()
                ablation_plot_path = os.path.join(output_dir, 'ablation_study.png')
                plt.savefig(ablation_plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"消融实验图已保存至: {ablation_plot_path}")

        except ImportError:
            print("警告: matplotlib 未安装，跳过图表生成")

    def _generate_statistical_analysis(
            self,
            results: Dict[str, Any],
            output_dir: str
    ):
        """统计显著性检验"""
        try:
            from scipy import stats

            # 找出我们的模型和其他模型的性能差异
            our_model_name = [k for k in results.keys() if 'Time-Aware' in k
                              and not k.startswith('Ours_')]
            if not our_model_name:
                our_model_name = [k for k in results.keys() if k.startswith('Time-Aware')]

            stats_results = []

            if our_model_name:
                our_result = results[our_model_name[0]]
                our_f1 = our_result['test_metrics']['f1_class1']

                # 与每个基线模型比较
                for model_name, result in results.items():
                    if model_name != our_model_name[0]:
                        baseline_f1 = result['test_metrics']['f1_class1']

                        # 计算相对提升
                        relative_improvement = (
                                    (our_f1 - baseline_f1) / baseline_f1 * 100) if baseline_f1 > 0 else float('inf')

                        stats_results.append({
                            'Model': model_name,
                            'F1 Class1': baseline_f1,
                            'ΔF1': our_f1 - baseline_f1,
                            '相对提升(%)': f"{relative_improvement:.2f}%",
                        })

                # 保存统计结果
                stats_df = pd.DataFrame(stats_results)
                stats_df = stats_df.sort_values('F1 Class1', ascending=False)

                stats_csv = os.path.join(output_dir, 'statistical_improvement.csv')
                stats_df.to_csv(stats_csv, index=False)
                print(f"\n统计提升分析已保存至: {stats_csv}")

                # 打印关键发现
                print("\n" + "=" * 60)
                print("关键发现:")
                print("=" * 60)

                # 相对于最强基线的提升
                best_baseline = stats_df.iloc[0]
                print(f"相对于最强基线 ({best_baseline['Model']}):")
                print(f"  我们的 F1: {our_f1:.4f}")
                print(f"  基线 F1: {best_baseline['F1 Class1']:.4f}")
                print(f"  绝对提升: {best_baseline['ΔF1']:.4f}")
                print(f"  相对提升: {best_baseline['相对提升(%)']}")

                # 平均提升
                avg_improvement = stats_df['ΔF1'].mean()
                print(f"\n相对于所有基线的平均提升: {avg_improvement:.4f}")

        except ImportError:
            print("警告: scipy 未安装，跳过统计分析")