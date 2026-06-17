# scripts/check_reproducibility.py
"""
验证结果可复现性的脚本
运行两次，比较两次的结果是否完全一致
"""

import sys

sys.path.append('..')

import torch
import numpy as np
from stage1.utils import set_seed


def test_reproducibility(seed=42):
    print(f"测试种子 {seed} 的可复现性...")

    # 第一次运行
    set_seed(seed)

    # 记录初始随机值
    torch_rand_1 = torch.randn(3, 3)
    np_rand_1 = np.random.randn(3, 3)

    # 创建模型
    model_1 = torch.nn.Linear(10, 5)
    weights_1 = model_1.weight.data.clone()

    # 重置种子
    set_seed(seed)

    # 第二次运行
    torch_rand_2 = torch.randn(3, 3)
    np_rand_2 = np.random.randn(3, 3)

    # 创建模型
    model_2 = torch.nn.Linear(10, 5)
    weights_2 = model_2.weight.data.clone()

    # 检查是否一致
    torch_equal = torch.allclose(torch_rand_1, torch_rand_2)
    np_equal = np.allclose(np_rand_1, np_rand_2)
    weights_equal = torch.allclose(weights_1, weights_2)

    print(f"  PyTorch 随机数一致: {torch_equal}")
    print(f"  NumPy 随机数一致: {np_equal}")
    print(f"  模型权重一致: {weights_equal}")

    if torch_equal and np_equal and weights_equal:
        print(f"  ✅ 种子 {seed} 完全可复现")
    else:
        print(f"  ❌ 种子 {seed} 不可复现，存在问题！")

    return torch_equal and np_equal and weights_equal


if __name__ == '__main__':
    # 测试多个种子
    for seed in [42, 123, 2024]:
        test_reproducibility(seed)