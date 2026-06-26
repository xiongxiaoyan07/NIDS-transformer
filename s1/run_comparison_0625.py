# run_comparison.py
"""
运行模型对比实验的主脚本（优化版）
优化点：
1. 设置 PyTorch 优化标志
2. 启用 cuDNN benchmark（在确定性模式下谨慎使用）
3. 使用 torch.compile（PyTorch 2.0+）
4. 优化数据加载（更多 workers, pin_memory, prefetch_factor）
"""

import os
import sys
import argparse
import yaml
import json
from pathlib import Path
import warnings

from stage1.utils import set_seed

warnings.filterwarnings('ignore')

import torch
import numpy as np
import random

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from stage1.comparison import ModelBenchmark
from stage1.model import Stage1TimeAwareTransformer
from stage1.pipeline import build_dataloaders

import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description='模型对比实验')
    # 配置文件
    parser.add_argument('--config', type=str, required=True,
                        help='配置文件路径 (YAML)')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='数据目录路径')
    parser.add_argument('--tensor_dir', type=str, required=True,
                        help='预计算张量目录')

    # 实验设置
    parser.add_argument('--experiment_name', type=str, default='comparison',
                        help='实验名称')
    parser.add_argument('--output_dir', type=str, default='results/comparison',
                        help='输出目录')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='训练轮数')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备 (cuda/cpu)')

    # 对比范围
    parser.add_argument('--include_ml', action='store_true', default=True,
                        help='包含传统机器学习基线')
    parser.add_argument('--include_dl', action='store_true', default=True,
                        help='包含深度学习基线')
    parser.add_argument('--include_ablations', action='store_true', default=True,
                        help='包含消融实验')
    parser.add_argument('--skip_xgboost', action='store_true',
                        help='跳过 XGBoost (节省时间)')

    # 批量运行
    parser.add_argument('--batch_mode', action='store_true',
                        help='批量模式：运行多个数据集/配置')
    parser.add_argument('--configs_dir', type=str,
                        help='批量模式的配置文件目录')

    return parser.parse_args()


def run_single_experiment(args, config: dict):
    """运行单个实验（优化版）"""

    # 创建输出目录
    experiment_dir = os.path.join(args.output_dir, args.experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)

    # 加载数据
    print("\n" + "=" * 60)
    print("加载数据...seed", args.seed)
    print("=" * 60)

    print("\n[INFO] Loading data...")
    loaders, preprocessor, metadata = build_dataloaders(
        packet_csv=f"{args.data_dir}ar002_et12_20260511_002-stage1_packets.csv",
        flow_csv=f"{args.data_dir}ar002_et12_20260511_002-stage1_flows.csv",
        cfg=config,
        out_dir=args.tensor_dir,
        seed=args.seed
    )

    # ================================
    # DEBUG：检查 DataLoader 是否确定性
    # ================================
    print("\n[DEBUG] --TRAIN--- Checking train_loader consistency...")
    for i, batch in enumerate(loaders["train"]):
        print("[DEBUG --TRAIN--- batch 0 flow_ids]:", batch["flow_id"][:5])
        if i == 2:
            break

    print("\n[DEBUG] --TRAIN-trainNoSampler-- Checking train_loader  consistency...")
    for i, batch in enumerate(loaders["trainNoSampler"]):
        print("[DEBUG --TRAIN-trainNoSampler-- batch 0 flow_ids]:", batch["flow_id"][:5])
        if i == 2:
            break

    print("\n[DEBUG] --VAL--- Checking VAL_loader consistency...")
    for i, batch in enumerate(loaders["val"]):
        print("[DEBUG --VAL--- batch 0 flow_ids]:", batch["flow_id"][:5])
        if i == 2:
            break

    print("\n[DEBUG] --TEST--- Checking TEST_loader consistency...")
    for i, batch in enumerate(loaders["test"]):
        print("[DEBUG --TEST--- batch 0 flow_ids]:", batch["flow_id"][:5])
        if i == 2:
            break

    train_cfg = config.get("training", {})
    use_weighted_sampler = train_cfg.get("use_weighted_sampler", False)
    print("[run_single_experiment]--- use_weighted_sampler = ", use_weighted_sampler)
    if use_weighted_sampler:
        train_loader = loaders['train']
    else:
        train_loader = loaders['trainNoSampler']

    val_loader = loaders['val']
    test_loader = loaders['test']

    print(f"训练集: {len(train_loader.dataset)} 个流")
    print(f"验证集: {len(val_loader.dataset)} 个流")
    print(f"测试集: {len(test_loader.dataset)} 个流")

    # 计算标签分布（🚀 简化：只读一次）
    for loader, name in [(train_loader, '训练'), (val_loader, '验证'), (test_loader, '测试')]:
        all_labels = []
        for batch in loader:
            all_labels.append(batch['label'].numpy())
        all_labels = np.concatenate(all_labels, axis=0)
        unique, counts = np.unique(all_labels, return_counts=True)
        print(f"  {name}集分布: {dict(zip(unique, counts))}")

    # 创建基准测试实例
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"\n使用设备: {device}")

    # 🚀 GPU 预热（减少首次 CUDA 调用开销）
    if device == 'cuda':
        torch.cuda.empty_cache()
        # 用小张量预热 GPU
        _ = torch.zeros(1, device=device)
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")

    flow_fusion_cfg = config.get("features", {}).get("flow_fusion", {})
    inject_to_packets = flow_fusion_cfg.get("inject_to_packets", True)
    use_flow_features = flow_fusion_cfg.get("enabled", False)

    if inject_to_packets and use_flow_features:
        # 方案A: flow特征拼接到packet
        input_dim = preprocessor.input_dim()
        flow_dim = 0
        print(f"[INFO] 方案A - Flow特征拼接到每个Packet")
        print(f"[INFO]   Input dim (with flow): {input_dim}")
    elif not inject_to_packets and use_flow_features and preprocessor.has_flow_features():
        # 方案C: 分层特征注入
        input_dim = preprocessor.packet_feature_dim()
        flow_dim = preprocessor.flow_feature_dim()
        config["_flow_feature_dim"] = flow_dim
        print(f"[INFO] 方案C - 分层特征注入")
        print(f"[INFO]   Packet input dim: {input_dim}")
        print(f"[INFO]   Flow feature dim: {flow_dim}")
    else:
        # 方案B: 仅packet特征
        input_dim = preprocessor.packet_feature_dim()
        flow_dim = 0
        print(f"[INFO] 方案B - 仅使用Packet特征")
        print(f"[INFO]   Input dim: {input_dim}")

    benchmark = ModelBenchmark(
        input_dim=input_dim,
        flow_feature_dim = flow_dim,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device
    )

    # # 添加我们的模型（Time-Aware Transformer）
    # benchmark.models['Ours_BothEncoding'] = Stage1TimeAwareTransformer(
    #     input_dim=benchmark.input_dim,
    #     cfg=config
    # )
    # print("[✓] Time-Aware Transformer (Our Model - Both Time and Position Encoding)")

    # 添加消融模型
    if args.include_ablations:
        benchmark.add_ablations(config)

    # 添加基线模型
    if args.include_dl:
        benchmark.add_baseline_models()

    # # 添加传统ML模型  方案B的配置下， 无法使用这些基础ML
    # if args.include_ml:
    #     if not args.skip_xgboost:
    #         benchmark.models['XGBoost'] = 'XGBoost'
    #         print("[✓] XGBoost")
    #     benchmark.models['LightGBM'] = 'LightGBM'
    #     print("[✓] LightGBM")
    #     benchmark.models['RandomForest'] = 'RandomForest'
    #     print("[✓] Random Forest")


    # 运行所有模型
    print(f"\n开始训练和评估 {len(benchmark.models)} 个模型...")
    results = benchmark.run_all(num_epochs=args.num_epochs, seed=args.seed)

    # 生成报告
    benchmark.generate_report(results, experiment_dir)

    # 保存配置文件
    config_save_path = os.path.join(experiment_dir, 'config.yaml')
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    return results


def run_batch_experiments(args):
    """批量运行多个实验"""
    config_files = sorted(Path(args.configs_dir).glob('*.yaml'))

    print(f"批处理模式: 找到 {len(config_files)} 个配置文件")

    all_results = {}
    for config_file in config_files:
        config_name = config_file.stem
        print(f"\n{'=' * 70}")
        print(f"运行实验: {config_name}")
        print(f"{'=' * 70}")

        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

        args.experiment_name = f"comparison_{config_name}"
        results = run_single_experiment(args, config)
        all_results[config_name] = results

    # 生成跨实验汇总报告
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    summary_rows = []
    for exp_name, results in all_results.items():
        for model_name, result in results.items():
            test_metrics = result['test_metrics']
            summary_rows.append({
                'Experiment': exp_name,
                'Model': model_name,
                'F1 Macro': test_metrics['f1_macro'],
                'F1 Class1': test_metrics['f1_class1'],
                'ROC-AUC': test_metrics['roc_auc'],
                'PR-AUC': test_metrics['pr_auc'],
            })

    summary_df = pd.DataFrame(summary_rows)

    # 透视表：实验 x 模型的 F1 Macro
    pivot_table = summary_df.pivot_table(
        index='Model',
        columns='Experiment',
        values='F1 Macro',
        aggfunc='mean'
    )
    # 保存汇总报告
    pivot_csv = os.path.join(output_dir, 'cross_experiment_summary.csv')
    pivot_table.to_csv(pivot_csv)
    print(f"\n跨实验汇总已保存至: {pivot_csv}")

    # 计算每个模型在所有实验中的平均排名
    model_ranks = {}
    for model in pivot_table.index:
        ranks = []
        for exp in pivot_table.columns:
            exp_values = pivot_table[exp].dropna()
            if model in exp_values.index:
                rank = exp_values.rank(ascending=False)[model]
                ranks.append(rank)
        if ranks:
            model_ranks[model] = np.mean(ranks)

    rank_df = pd.DataFrame(
        list(model_ranks.items()),
        columns=['Model', 'Avg Rank']
    ).sort_values('Avg Rank')

    rank_csv = os.path.join(output_dir, 'model_rankings.csv')
    rank_df.to_csv(rank_csv, index=False)

    print("\n模型平均排名（越小越好）:")
    for _, row in rank_df.iterrows():
        print(f"  {row['Model']}: {row['Avg Rank']:.2f}")

    return all_results


def main():
    args = parse_args()

    # ============================================
    # ⚠️ 第一步：在所有操作之前设置随机种子
    # ============================================
    set_seed(args.seed, deterministic=True)

    # 🚀 CUDA 优化配置（确定性模式下的最佳实践）
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision('high')  # 使用 TF32（Ampere+ GPU 加速）

    # 🚀 设置环境变量优化（不影响确定性）
    if torch.cuda.is_available():
        # 优化 CUDA 内存分配
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        # 减少 CUDA 同步开销
        torch.cuda.set_sync_debug_mode('warn')  # 生产模式，仅在错误时警告

    # 加载配置
    with open(args.config, 'r', encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # ⚠️ 将种子信息记录到 config 中
    config['_seed'] = args.seed
    config['_deterministic'] = True

    # 运行实验
    if args.batch_mode:
        if not args.configs_dir:
            raise ValueError("批处理模式需要指定 --configs_dir")
        run_batch_experiments(args)
    else:
        run_single_experiment(args, config)


if __name__ == '__main__':
    main()