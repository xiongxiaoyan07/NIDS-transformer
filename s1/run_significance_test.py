# scripts/run_significance_test.py
"""
运行统计显著性检验的独立脚本

用法:
    python scripts/run_significance_test.py \
        --config config/stage1_config.yaml \
        --model_a_path checkpoints/time_aware_transformer.pt \
        --model_b_path checkpoints/standard_transformer.pt \
        --data_dir data/processed \
        --output results/significance_test.json
"""

import os
import sys
import argparse
import yaml
import json
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import torch
from stage1.model import Stage1TimeAwareTransformer
from stage1.baselines import StandardTransformer
from stage1.statistical_tests import ModelSignificanceTester
from stage1.pipeline import build_dataloaders


def load_model(model_class, config, checkpoint_path, device):
    """加载模型和权重"""
    # 这个不在配置里，要小心
    input_dim = config['model']['input_dim']

    if model_class == 'TimeAwareTransformer':
        model = Stage1TimeAwareTransformer(input_dim, config)
    elif model_class == 'StandardTransformer':
        model = StandardTransformer(
            input_dim=input_dim,
            d_model=config['model']['d_model'],
            num_heads=config['model']['num_heads'],
            num_layers=config['model']['num_layers'],
            dim_feedforward=config['model']['dim_feedforward'],
            dropout=config['model']['dropout'],
            max_seq_len=config['data']['max_seq_len'],
            num_classes=config['model']['num_classes'],
            use_flow_features=config.get('fusion', {}).get('method') == 'gated',
            flow_feature_dim=config['model'].get('flow_feature_dim', None)
        )
    else:
        raise ValueError(f"Unknown model class: {model_class}")

    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"加载模型权重: {checkpoint_path}")

    model = model.to(device)
    model.eval()

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--model_a_path', type=str, required=True,
                        help='我们的模型权重路径')
    parser.add_argument('--model_b_path', type=str, required=True,
                        help='基线模型权重路径')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--tensor_dir', type=str, required=True)
    parser.add_argument('--output', type=str, default='results/significance_test.json')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--model_a_name', type=str, default='Time-Aware Transformer')
    parser.add_argument('--model_b_name', type=str, default='Standard Transformer')
    args = parser.parse_args()

    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # 加载数据
    print("加载数据...")
    loaders, preprocessor, metadata = build_dataloaders(
        packet_csv=f"{args.data_dir}firewall_test_20250731_160700-stage1_packets.csv",
        flow_csv=f"{args.data_dir}firewall_test_20250731_160700-stage1_flows.csv",
        cfg=config,
        out_dir=args.tensor_dir,
    )
    test_loader = loaders['test']
    print(f"测试集: {len(test_loader.dataset)} 个样本")

    # 加载模型
    device = args.device if torch.cuda.is_available() else 'cpu'

    model_a = load_model('TimeAwareTransformer', config, args.model_a_path, device)
    model_b = load_model('StandardTransformer', config, args.model_b_path, device)

    # 运行显著性检验
    tester = ModelSignificanceTester(
        model_a=model_a,
        model_b=model_b,
        model_a_name=args.model_a_name,
        model_b_name=args.model_b_name,
        device=device,
        n_bootstrap=1000,
        alpha=0.05
    )

    results = tester.run_all_tests(test_loader, output_path=args.output)

    # 输出简要结论
    print("\n" + "=" * 70)
    summary = results['summary']
    print(f"最终结论: {summary['overall_conclusion']}")
    print("=" * 70)


if __name__ == '__main__':
    main()