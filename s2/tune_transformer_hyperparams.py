# experiments/tune_transformer_hyperparams.py
"""
Stage2 Transformer 超参数网格搜索脚本

功能：
1. 基于 run_stage2.py 架构，对 Transformer 的关键超参数进行网格搜索
2. 自动记录每个组合的验证集性能
3. 生成总结报告（CSV + JSON）
4. 支持断点续跑（已完成组合自动跳过）

用法：
python experiments/tune_transformer_hyperparams.py \
    --stage1_dir /path/to/stage1/outputs \
    --output_dir results/tune_transformer \
    --search_mode grid \
    --num_workers 4
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import yaml

# 添加项目根目录
sys.path.insert(0, str(Path(__file__).parent.parent))

from stage2.config import DEFAULT_CFG, deep_update, load_config, save_config
from stage2.context import ContextIndexBuilder
from stage2.data_io import prepare_sorted_stage2_data
from stage2.dataset import Stage2Dataset
from stage2.model import build_stage2_model
from stage2.trainer import Stage2Trainer
from stage2.utils import get_device, safe_mkdir, set_seed

# ============================================================================
# 超参数搜索空间定义
# ============================================================================

HPARAM_SEARCH_SPACE = {
    # 架构参数
    "model.num_layers": {
        "values": [2, 4, 6, 8],
        "description": "Transformer encoder 层数",
        "type": "int",
    },
    "model.nhead": {
        "values": [4, 8, 16],
        "description": "多头注意力头数",
        "type": "int",
    },
    "model.dim_feedforward": {
        "values": [128,256, 512, 1024],
        "description": "前馈网络隐藏层维度",
        "type": "int",
    },
    "model.dropout": {
        "values": [0.1, 0.2, 0.3, 0.4, 0.5],
        "description": "Dropout 比率",
        "type": "float",
    },
    "model.pooling": {
        "values": ["last", "mean", "attention"],
        "description": "序列池化策略",
        "type": "categorical",
    },
    "model.use_positional_encoding": {
        "values": [True, False],
        "description": "是否使用位置编码",
        "type": "bool",
    },
    "model.cls_head": {
        "values": [1, 2, 3],
        "description": "分类头层数",
        "type": "int",
    },

    # 训练参数
    "training.lr": {
        "values": [1e-4, 3e-4, 5e-4, 1e-3],
        "description": "学习率",
        "type": "float",
    },
    "training.batch_size": {
        "values": [32, 64, 128, 256],
        "description": "批次大小",
        "type": "int",
    },
    "training.weight_decay": {
        "values": [1e-5, 1e-4, 1e-3],
        "description": "权重衰减",
        "type": "float",
    },
    "training.focal_gamma": {
        "values": [1.0, 2.0, 3.0, 5.0],
        "description": "Focal Loss gamma",
        "type": "float",
    },
    "training.label_smoothing": {
        "values": [0.0, 0.02, 0.05, 0.1],
        "description": "标签平滑",
        "type": "float",
    },
    "training.use_weighted_sampler": {
        "values": [True, False],
        "description": "是否使用加权采样器",
        "type": "bool",
    },

    # 上下文参数
    "context.window_size": {
        "values": [8, 16, 32, 64, 128, 256],
        "description": "上下文窗口大小",
        "type": "int",
    },
    "context.method": {
        "values": ["time_only", "source_host", "destination_host", "endpoint"],
        "description": "上下文构建方法",
        "type": "categorical",
    },
}

# ============================================================================
# 预设搜索模式
# ============================================================================

PRESET_SEARCH_MODES = {
    "test": [
        "model.cls_head",
    ],
    # 快速模式：架构关键参数 + 核心训练参数
    "quick": [
        "model.num_layers",
        "model.nhead",
        "model.pooling",
        "training.lr",
        "training.batch_size",
        "context.window_size",
    ],

    # 完整模式：所有参数
    "full": list(HPARAM_SEARCH_SPACE.keys()),

    # 架构搜索：仅架构参数
    "architecture": [
        "model.num_layers",
        "model.nhead",
        "model.dim_feedforward",
        "model.dropout",
        "model.pooling",
        "model.use_positional_encoding",
        "model.cls_head",
    ],

    # 训练搜索：仅训练超参数
    "training": [
        "training.lr",
        "training.batch_size",
        "training.weight_decay",
        "training.focal_gamma",
        "training.label_smoothing",
        "training.use_weighted_sampler",
    ],

    # 上下文搜索
    "context": [
        "context.window_size",
        "context.method",
    ],
}


# ============================================================================
# 工具函数
# ============================================================================

def set_nested_value(cfg: Dict[str, Any], key_path: str, value: Any) -> Dict[str, Any]:
    """设置嵌套配置值，例如 'model.num_layers' -> cfg['model']['num_layers']"""
    keys = key_path.split(".")
    current = cfg
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value
    return cfg


def get_nested_value(cfg: Dict[str, Any], key_path: str) -> Any:
    """获取嵌套配置值"""
    keys = key_path.split(".")
    current = cfg
    for key in keys:
        current = current[key]
    return current


def generate_param_combinations(
        search_params: List[str],
        max_combinations: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """生成超参数组合"""
    param_values = {}
    for param in search_params:
        if param not in HPARAM_SEARCH_SPACE:
            raise ValueError(f"Unknown parameter: {param}")
        param_values[param] = HPARAM_SEARCH_SPACE[param]["values"]

    # 笛卡尔积
    keys = list(param_values.keys())
    values_list = list(param_values.values())
    combinations = list(itertools.product(*values_list))

    # 限制组合数量
    if max_combinations is not None and len(combinations) > max_combinations:
        np.random.seed(42)
        indices = np.random.choice(len(combinations), max_combinations, replace=False)
        combinations = [combinations[i] for i in indices]

    result = []
    for combo in combinations:
        result.append(dict(zip(keys, combo)))

    return result


def build_hparam_config(
        base_cfg: Dict[str, Any],
        hparams: Dict[str, Any],
) -> Dict[str, Any]:
    """从基配置和超参数构建新配置"""
    cfg = deepcopy(base_cfg)
    for key_path, value in hparams.items():
        cfg = set_nested_value(cfg, key_path, value)
    return cfg


def generate_run_id(hparams: Dict[str, Any]) -> str:
    """为超参数组合生成唯一的运行ID"""
    parts = []
    for key in sorted(hparams.keys()):
        short_key = key.split(".")[-1]  # 只用最后一部分
        value = hparams[key]
        if isinstance(value, float):
            parts.append(f"{short_key}={value:.4f}")
        elif isinstance(value, bool):
            parts.append(f"{short_key}={int(value)}")
        else:
            parts.append(f"{short_key}={value}")
    return "_".join(parts)


def check_completed(output_dir: str, run_id: str) -> bool:
    """检查某个运行是否已完成"""
    metrics_path = os.path.join(output_dir, run_id, "stage2_metrics.json")
    return os.path.exists(metrics_path)


# ============================================================================
# 主搜索类
# ============================================================================

class TransformerHyperparamTuner:
    """
    Stage2 Transformer 超参数调优器

    功能：
    1. 网格搜索 / 随机搜索
    2. 自动记录每个试验的完整结果
    3. 生成汇总报告和排名
    """

    def __init__(
            self,
            stage1_dir: str,
            base_config_path: Optional[str],
            output_dir: str,
            search_params: List[str],
            max_combinations: Optional[int] = 100,
            num_workers: int = 0,
            device: str = "auto",
            epochs_per_trial: Optional[int] = None,
            patience_per_trial: Optional[int] = None,
    ):
        self.stage1_dir = stage1_dir
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.search_params = search_params
        self.max_combinations = max_combinations
        self.num_workers = num_workers
        self.device_str = device

        # 可选：减少每个 trial 的 epoch 数以加速搜索
        self.epochs_per_trial = epochs_per_trial
        self.patience_per_trial = patience_per_trial

        # 加载基配置
        self.base_cfg = load_config(base_config_path)
        self.base_cfg["data"]["stage1_dir"] = stage1_dir

        # 预加载数据（所有 trial 共享）
        self.device = get_device(device)
        print(f"[INFO] Device: {self.device}")
        print("[INFO] Pre-loading Stage1 data...")

        self.meta_df, self.z_sorted = prepare_sorted_stage2_data(
            stage1_dir=stage1_dir,
            cfg=self.base_cfg,
        )

        print(f"[INFO] Loaded {len(self.meta_df)} flows")
        print(f"[INFO] z shape: {self.z_sorted.shape}")

        # 结果记录
        self.results: List[Dict[str, Any]] = []
        self.best_result: Optional[Dict[str, Any]] = None
        self.best_score = -float("inf")

    def _build_datasets(
            self,
            cfg: Dict[str, Any],
    ) -> Dict[str, Stage2Dataset]:
        """为给定配置构建数据集"""
        context_builder = ContextIndexBuilder(self.meta_df, cfg)
        context_indices = context_builder.build()

        datasets = {}
        for split in ["train", "val", "test"]:
            datasets[split] = Stage2Dataset(
                meta_df_sorted=self.meta_df,
                z_sorted=self.z_sorted,
                context_indices=context_indices,
                target_split=split,
            )

        return datasets, context_indices

    def _run_single_trial(
            self,
            hparams: Dict[str, Any],
            run_id: str,
            trial_idx: int,
            total_trials: int,
    ) -> Dict[str, Any]:
        """运行单个超参数试验"""

        print(f"\n{'=' * 70}")
        print(f"Trial {trial_idx + 1}/{total_trials}: {run_id}")
        print(f"hparams: {hparams}")
        print(f"{'=' * 70}")

        # 构建配置
        cfg = build_hparam_config(self.base_cfg, hparams)

        # 可选：覆盖训练配置以加速搜索
        if self.epochs_per_trial is not None:
            cfg["training"]["epochs"] = self.epochs_per_trial
        if self.patience_per_trial is not None:
            cfg["training"]["patience"] = self.patience_per_trial

        # 设置 seed
        seed = cfg.get("seed", 42)
        set_seed(seed)

        # 创建输出目录
        trial_dir = self.output_dir / run_id
        safe_mkdir(str(trial_dir))

        # 保存配置
        save_config(cfg, str(trial_dir / "config.yaml"))

        try:
            # 构建数据集
            datasets, context_indices = self._build_datasets(cfg)

            # 构建模型
            model = build_stage2_model(cfg, input_dim=self.z_sorted.shape[1])
            model = model.to(self.device)

            n_params = sum(p.numel() for p in model.parameters())
            print(f"[INFO] Model parameters: {n_params:,}")

            # 构建训练器并训练
            trainer = Stage2Trainer(
                model=model,
                datasets=datasets,
                cfg=cfg,
                device=self.device,
                out_dir=str(trial_dir),
                input_dim=self.z_sorted.shape[1],
            )

            # 训练
            start_time = time.time()
            trainer.fit()
            train_time = time.time() - start_time

            # 最终评估
            # context_builder = ContextIndexBuilder(self.meta_df, cfg)
            # context_indices = context_builder.build()

            final_metrics = trainer.final_evaluate_and_save(
                meta_df=self.meta_df,
                context_indices=context_indices,
            )

            # 提取关键指标
            trial_result = {
                "run_id": run_id,
                "trial_idx": trial_idx,
                "hparams": hparams,
                "config_summary": {
                    k: v for k, v in hparams.items()
                },
                "num_params": n_params,
                "train_time_seconds": train_time,
                "best_epoch": final_metrics.get("best_epoch", -1),
                "threshold": final_metrics.get("threshold", 0.5),
            }

            # 添加各 split 的指标
            for split in ["train", "val", "test"]:
                if split in final_metrics.get("splits", {}):
                    metrics = final_metrics["splits"][split]
                    for metric_name in [
                        "loss", "accuracy", "confusion_matrix",
                        "macro_f1","macro_precision", "macro_recall",
                        "weighted_f1", "weighted_precision", "weighted_recall"
                        "f1_label1", "precision_label1", "recall_label1",
                        "auc", "pr_auc",
                    ]:
                        if metric_name in metrics:
                            trial_result[f"{split}_{metric_name}"] = metrics[metric_name]

            # 判断是否是最佳
            val_f1 = trial_result.get("val_f1_label1", 0)
            is_best = val_f1 > self.best_score

            if is_best:
                self.best_score = val_f1
                self.best_result = trial_result
                print(f"[🏆 NEW BEST] val_f1_label1 = {val_f1:.4f}")

            print(f"[Trial {trial_idx + 1} complete]")
            print(f"  val_f1_label1: {val_f1:.4f}")
            print(f"  val_macro_f1: {trial_result.get('val_macro_f1', 0):.4f}")
            print(f"  val_auc: {trial_result.get('val_auc', 0):.4f}")
            print(f"  test_f1_label1: {trial_result.get('test_f1_label1', 0):.4f}")
            print(f"  test_macro_f1: {trial_result.get('test_macro_f1', 0):.4f}")
            print(f"  train_time: {train_time:.1f}s")

        except Exception as e:
            print(f"[ERROR] Trial {run_id} failed: {e}")
            import traceback
            traceback.print_exc()

            trial_result = {
                "run_id": run_id,
                "trial_idx": trial_idx,
                "hparams": hparams,
                "error": str(e),
                "val_f1_label1": 0,
            }

        return trial_result

    def run(self) -> pd.DataFrame:
        """执行完整的超参数搜索"""

        # 生成所有超参数组合
        param_combos = generate_param_combinations(
            self.search_params,
            max_combinations=self.max_combinations,
        )

        total_trials = len(param_combos)
        print(f"\n{'=' * 70}")
        print(f"Hyperparameter Search Configuration")
        print(f"{'=' * 70}")
        print(f"Search parameters: {self.search_params}")
        print(f"Total combinations: {total_trials}")
        print(f"Output directory: {self.output_dir}")
        print(f"{'=' * 70}\n")

        # 运行所有试验
        for trial_idx, hparams in enumerate(param_combos):
            run_id = generate_run_id(hparams)

            # 检查是否已完成
            if check_completed(str(self.output_dir), run_id):
                print(f"[SKIP] Trial {trial_idx + 1}/{total_trials}: {run_id} (already completed)")
                continue

            # 运行试验
            trial_result = self._run_single_trial(
                hparams=hparams,
                run_id=run_id,
                trial_idx=trial_idx,
                total_trials=total_trials,
            )

            self.results.append(trial_result)

            # 实时保存结果
            self._save_progress()

        # 生成最终报告
        return self._generate_final_report()

    def _save_progress(self) -> None:
        """保存搜索进度"""
        # 保存所有结果
        results_df = pd.DataFrame(self.results)
        results_df.to_csv(
            self.output_dir / "all_trials.csv",
            index=False,
        )

        # 保存最佳结果
        if self.best_result:
            with open(self.output_dir / "best_params.json", "w") as f:
                json.dump(self.best_result, f, indent=2, ensure_ascii=False)

    def _generate_final_report(self) -> pd.DataFrame:
        """生成最终搜索报告"""

        if not self.results:
            print("[WARNING] No results to report")
            return pd.DataFrame()

        df = pd.DataFrame(self.results)

        # 按验证集 F1 排序
        if "val_f1_label1" in df.columns:
            df = df.sort_values("val_f1_label1", ascending=False)

        # 保存排名
        df.to_csv(
            self.output_dir / "hparam_search_results_sorted.csv",
            index=False,
        )

        # 生成分析报告
        self._generate_analysis(df)

        # 保存最佳配置
        self._save_best_config(df)

        # 打印总结
        self._print_summary(df)

        return df

    def _generate_analysis(self, df: pd.DataFrame) -> None:
        """生成超参数影响分析"""

        analysis_lines = []
        analysis_lines.append("=" * 80)
        analysis_lines.append("HYPERPARAMETER IMPACT ANALYSIS")
        analysis_lines.append("=" * 80)

        # 对每个搜索参数分析其影响
        for param in self.search_params:
            param_short = param.split(".")[-1]

            if param_short not in df["hparams"].iloc[0]:
                continue

            analysis_lines.append(f"\n--- {param} ---")

            # 按参数值分组计算平均性能
            param_values = df["hparams"].apply(lambda x: x.get(param))
            grouped = df.groupby(param_values)

            if "val_f1_label1" in df.columns:
                agg = grouped["val_f1_label1"].agg(["mean", "std", "count"])
                agg = agg.sort_values("mean", ascending=False)

                for value, row in agg.iterrows():
                    analysis_lines.append(
                        f"  {param_short}={value}: "
                        f"mean={row['mean']:.4f}, "
                        f"std={row['std']:.4f}, "
                        f"n={int(row['count'])}"
                    )

            # 找出该参数的最佳值
            if "val_f1_label1" in df.columns:
                best_idx = df["val_f1_label1"].idxmax()
                best_value = df.loc[best_idx, "hparams"].get(param)
                analysis_lines.append(f"  → Best: {param_short}={best_value}")

        # 保存分析报告
        with open(self.output_dir / "hparam_analysis.txt", "w") as f:
            f.write("\n".join(analysis_lines))

        print("\n".join(analysis_lines))

    def _save_best_config(self, df: pd.DataFrame) -> None:
        """保存最佳配置为可用文件"""

        if df.empty:
            return

        best_row = df.iloc[0]
        best_hparams = best_row["hparams"]

        best_cfg = build_hparam_config(self.base_cfg, best_hparams)

        save_config(best_cfg, str(self.output_dir / "best_config.yaml"))

        print(f"\n[INFO] Best config saved to: {self.output_dir / 'best_config.yaml'}")

        # 同时生成可直接使用的命令行
        cli_args = []
        for key, value in best_hparams.items():
            param_name = key.replace(".", "_")
            if isinstance(value, bool):
                if value:
                    cli_args.append(f"--{param_name}")
            else:
                cli_args.append(f"--{param_name} {value}")

        cli_cmd = f"python run_stage2.py --stage1_dir {self.stage1_dir} --config {self.output_dir / 'best_config.yaml'}"

        with open(self.output_dir / "best_config_cli.txt", "w") as f:
            f.write(f"# Best hyperparameters CLI command\n")
            f.write(f"{cli_cmd}\n")

        print(f"[INFO] CLI command: {cli_cmd}")

    def _print_summary(self, df: pd.DataFrame) -> None:
        """打印搜索总结"""

        print(f"\n{'=' * 70}")
        print(f"HYPERPARAMETER SEARCH SUMMARY")
        print(f"{'=' * 70}")
        print(f"Total trials: {len(df)}")
        print(f"Output: {self.output_dir}")

        if "val_f1_label1" in df.columns:
            print(f"\nTop 5 Configurations:")
            print("-" * 70)

            for rank, (_, row) in enumerate(df.head(5).iterrows(), 1):
                print(f"\n#{rank} {row['run_id']}")
                print(f"  val_f1_label1: {row['val_f1_label1']:.4f}")
                print(f"  val_auc: {row.get('val_auc', 'N/A')}")
                print(f"  test_f1_label1: {row.get('test_f1_label1', 'N/A'):.4f}")
                print(f"  test_auc: {row.get('test_auc', 'N/A')}")

            best_val_f1 = df["val_f1_label1"].max()
            worst_val_f1 = df["val_f1_label1"].min()
            mean_val_f1 = df["val_f1_label1"].mean()
            std_val_f1 = df["val_f1_label1"].std()

            print(f"\nPerformance Range:")
            print(f"  Best:  {best_val_f1:.4f}")
            print(f"  Worst: {worst_val_f1:.4f}")
            print(f"  Mean:  {mean_val_f1:.4f} ± {std_val_f1:.4f}")

        print(f"\n{'=' * 70}")


# ============================================================================
# 命令行接口
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage2 Transformer Hyperparameter Tuner"
    )

    # 必需参数
    parser.add_argument(
        "--stage1_dir", type=str, required=True,
        help="Stage1 输出目录",
    )
    parser.add_argument(
        "--output_dir", type=str, default="results/hparam_search",
        help="搜索输出目录",
    )

    # 搜索模式
    parser.add_argument(
        "--search_mode", type=str, default="quick",
        choices=["test", "quick", "full", "architecture", "training", "context", "custom"],
        help="预设搜索模式",
    )
    parser.add_argument(
        "--search_params", type=str, nargs="*", default=None,
        help="自定义搜索参数（当 search_mode=custom 时使用）",
    )
    parser.add_argument(
        "--max_combinations", type=int, default=100,
        help="最大试验组合数",
    )

    # 配置和环境
    parser.add_argument(
        "--config", type=str, default=None,
        help="基础 YAML 配置文件",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="设备 (auto/cuda/cpu)",
    )
    parser.add_argument(
        "--num_workers", type=int, default=0,
        help="DataLoader workers 数量",
    )

    # 搜索加速
    parser.add_argument(
        "--epochs_per_trial", type=int, default=None,
        help="每个 trial 的最大 epoch 数（None 使用配置文件中的值）",
    )
    parser.add_argument(
        "--patience_per_trial", type=int, default=None,
        help="每个 trial 的 early stopping patience",
    )
    parser.add_argument(
        "--skip_completed", action="store_true",
        help="跳过已完成的 trial",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # 确定搜索参数
    if args.search_mode == "custom":
        if not args.search_params:
            raise ValueError("custom mode requires --search_params")
        search_params = args.search_params
    else:
        search_params = PRESET_SEARCH_MODES[args.search_mode]

    print(f"[INFO] Search mode: {args.search_mode}")
    print(f"[INFO] Search parameters: {search_params}")
    print(f"[INFO] Max combinations: {args.max_combinations}")

    # 创建调优器
    tuner = TransformerHyperparamTuner(
        stage1_dir=args.stage1_dir,
        base_config_path=args.config,
        output_dir=args.output_dir,
        search_params=search_params,
        max_combinations=args.max_combinations,
        num_workers=args.num_workers,
        device=args.device,
        epochs_per_trial=args.epochs_per_trial,
        patience_per_trial=args.patience_per_trial,
    )

    # 运行搜索
    results_df = tuner.run()

    print(f"\n[INFO] Hyperparameter search complete!")
    print(f"[INFO] Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()