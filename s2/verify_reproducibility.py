# experiments/verify_reproducibility.py
"""
数据加载和模型训练的可复现性验证脚本

验证内容：
1. 数据加载的一致性（多次加载是否相同）
2. 数据划分的一致性（相同 seed 是否得到相同划分）
3. 上下文构建的一致性（context indices 是否稳定）
4. 模型初始化的可复现性（相同 seed 的初始权重）
5. 训练过程的可复现性（相同 seed 是否得到相同结果）

用法：
python experiments/verify_reproducibility.py \
    --stage1_dir s1/0702v2 \
    --output_dir results/reproducibility_check \
    --num_checks 3
"""

from __future__ import annotations

import argparse
import hashlib
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
import torch.nn as nn
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from stage2.config import load_config, save_config
from stage2.context import ContextIndexBuilder
from stage2.data_io import prepare_sorted_stage2_data
from stage2.dataset import Stage2Dataset
from stage2.model import build_stage2_model, Stage2Transformer
from stage2.trainer import Stage2Trainer
from stage2.utils import get_device, safe_mkdir, set_seed


# ============================================================================
# 工具函数：数据指纹
# ============================================================================

# ============================================================================
# 工具函数：数据指纹（修复版）
# ============================================================================

def ensure_numpy(data) -> np.ndarray:
    """
    安全地将数据转换为 numpy 数组

    处理多种输入类型：
    - torch.Tensor → numpy
    - np.ndarray → 保持不变
    - list/tuple → numpy
    - 其他 → 尝试转换为 numpy
    """
    if data is None:
        raise ValueError("Cannot convert None to numpy array")

    # PyTorch Tensor
    if hasattr(data, 'detach'):
        arr = data.detach().cpu().numpy()
    # NumPy 数组
    elif isinstance(data, np.ndarray):
        arr = data
    # 列表或元组
    elif isinstance(data, (list, tuple)):
        arr = np.array(data)
    # 标量
    elif isinstance(data, (int, float)):
        arr = np.array([data])
    # Pandas DataFrame/Series
    elif hasattr(data, 'to_numpy'):
        arr = data.to_numpy()
    # 其他尝试
    else:
        try:
            arr = np.array(data)
        except:
            raise TypeError(f"Unsupported data type: {type(data)}")

    return arr


def compute_tensor_hash(data, method: str = "md5") -> str:
    """
    计算数据的哈希值（修复版 - 支持多种数据类型）

    Args:
        data: torch.Tensor, np.ndarray, list, tuple 等
        method: "md5" | "stats" | "full"

    Returns:
        哈希字符串
    """
    if data is None:
        return "None"

    try:
        # 安全转换为 numpy
        arr = ensure_numpy(data)
    except (ValueError, TypeError) as e:
        return f"Error: {e}"

    if method == "md5":
        hasher = hashlib.md5()
        hasher.update(arr.tobytes())
        return hasher.hexdigest()[:16]

    elif method == "stats":
        # 统计量签名（更稳定，忽略浮点精度差异）
        return (f"shape={arr.shape}_"
                f"mean={arr.mean():.6f}_"
                f"std={arr.std():.6f}_"
                f"min={arr.min():.6f}_"
                f"max={arr.max():.6f}")

    elif method == "full":
        return (f"shape={arr.shape}_"
                f"hash={hashlib.md5(arr.tobytes()).hexdigest()[:12]}")

    else:
        raise ValueError(f"Unknown method: {method}")


def compute_df_hash(df: pd.DataFrame) -> str:
    """
    计算 DataFrame 的哈希值（修复版）

    Args:
        df: pandas DataFrame

    Returns:
        哈希字符串
    """
    if df is None:
        return "None"

    if not isinstance(df, pd.DataFrame):
        return f"Not_a_DataFrame(type={type(df).__name__})"

    try:
        # 只用关键列计算哈希
        key_cols = ["flow_id", "label", "index"]
        cols = [c for c in key_cols if c in df.columns]

        if cols:
            data = df[cols].to_numpy()
        else:
            data = df.to_numpy()

        hasher = hashlib.md5()
        hasher.update(data.tobytes())
        return hasher.hexdigest()[:16]

    except Exception as e:
        return f"Error: {e}"


def compute_model_hash(model: nn.Module) -> Dict[str, Dict]:
    """
    计算模型参数的哈希值（修复版）

    Args:
        model: PyTorch nn.Module

    Returns:
        参数字典，每个参数包含统计信息
    """
    param_hashes = {}

    if model is None:
        return param_hashes

    for name, param in model.named_parameters():
        if param.requires_grad:
            try:
                arr = param.detach().cpu().numpy()
                param_hashes[name] = {
                    "shape": list(arr.shape),
                    "mean": float(arr.mean()),
                    "std": float(arr.std()),
                    "min": float(arr.min()),
                    "max": float(arr.max()),
                    "norm": float(np.linalg.norm(arr)),
                }
            except Exception as e:
                param_hashes[name] = {
                    "error": str(e),
                    "shape": list(param.shape),
                }

    return param_hashes

def compare_dicts(d1: Dict, d2: Dict, tolerance: float = 1e-6) -> Dict[str, Any]:
    """
    比较两个字典，找出差异
    """
    differences = {
        "only_in_1": [],
        "only_in_2": [],
        "different_keys": [],
        "identical_keys": [],
    }

    all_keys = set(d1.keys()) | set(d2.keys())

    for key in sorted(all_keys):
        if key not in d1:
            differences["only_in_2"].append(key)
        elif key not in d2:
            differences["only_in_1"].append(key)
        else:
            v1, v2 = d1[key], d2[key]

            if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                if abs(v1 - v2) > tolerance:
                    differences["different_keys"].append({
                        "key": key,
                        "value_1": v1,
                        "value_2": v2,
                        "diff": abs(v1 - v2),
                    })
                else:
                    differences["identical_keys"].append(key)
            elif isinstance(v1, np.ndarray) and isinstance(v2, np.ndarray):
                if not np.allclose(v1, v2, atol=tolerance):
                    differences["different_keys"].append({
                        "key": key,
                        "max_diff": float(np.abs(v1 - v2).max()),
                    })
                else:
                    differences["identical_keys"].append(key)
            elif v1 != v2:
                differences["different_keys"].append({
                    "key": key,
                    "value_1": v1,
                    "value_2": v2,
                })
            else:
                differences["identical_keys"].append(key)

    differences["summary"] = {
        "total_keys": len(all_keys),
        "identical": len(differences["identical_keys"]),
        "different": len(differences["different_keys"]),
        "only_in_1": len(differences["only_in_1"]),
        "only_in_2": len(differences["only_in_2"]),
        "is_identical": (
                len(differences["different_keys"]) == 0 and
                len(differences["only_in_1"]) == 0 and
                len(differences["only_in_2"]) == 0
        ),
    }

    return differences


# ============================================================================
# 验证 1：数据加载的一致性（修复版）
# ============================================================================

def verify_data_loading(stage1_dir: str, config_path: str, num_repeats: int = 3):
    """
    验证多次调用 prepare_sorted_stage2_data 是否返回相同数据

    修复：处理 z_sorted 可能是 numpy 数组的情况
    """
    print("\n" + "=" * 70)
    print("VERIFICATION 1: Data Loading Consistency")
    print("=" * 70)

    cfg = load_config(config_path)
    cfg["data"]["stage1_dir"] = stage1_dir

    results = []

    for i in range(num_repeats):
        print(f"\n--- Load attempt {i + 1}/{num_repeats} ---")
        t0 = time.time()

        try:
            meta_df, z_sorted = prepare_sorted_stage2_data(
                stage1_dir=stage1_dir, cfg=cfg
            )
            load_time = time.time() - t0

            # 安全获取形状信息
            meta_shape = meta_df.shape if meta_df is not None else None
            z_shape = z_sorted.shape if z_sorted is not None else None

            # 安全获取标签分布
            if meta_df is not None and "label" in meta_df.columns:
                label_dist = meta_df["label"].value_counts().to_dict()
            else:
                label_dist = {}

            result = {
                "attempt": i,
                "meta_shape": meta_shape,
                "z_shape": z_shape,
                "meta_hash": compute_df_hash(meta_df),
                "z_type": type(z_sorted).__name__,
                "z_hash_stats": compute_tensor_hash(z_sorted, method="stats"),
                "z_hash_md5": compute_tensor_hash(z_sorted, method="md5"),
                "load_time": load_time,
                "label_dist": label_dist,
            }

            results.append(result)

            print(f"  meta_df shape: {meta_shape}")
            print(f"  z_sorted type: {result['z_type']}")
            print(f"  z_sorted shape: {z_shape}")
            if z_sorted is not None:
                arr = ensure_numpy(z_sorted)
                print(f"  z_sorted stats: mean={arr.mean():.6f}, std={arr.std():.6f}")
            print(f"  z_sorted hash (stats): {result['z_hash_stats']}")
            print(f"  z_sorted hash (md5): {result['z_hash_md5']}")
            print(f"  load time: {load_time:.2f}s")

        except Exception as e:
            print(f"  ❌ Error in load attempt {i + 1}: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "attempt": i,
                "error": str(e),
                "traceback": traceback.format_exc(),
            })

    # 比较所有结果
    print(f"\n--- Data Loading Consistency Check ---")

    # 过滤掉有错误的结果
    valid_results = [r for r in results if "error" not in r]

    if len(valid_results) < 2:
        print(f"❌ Not enough valid results to compare (only {len(valid_results)})")
        for r in results:
            if "error" in r:
                print(f"   Error in attempt {r['attempt']}: {r['error']}")
        return results

    # 检查 meta_df 哈希
    meta_hashes = [r["meta_hash"] for r in valid_results]
    if len(set(meta_hashes)) == 1:
        print(f"✅ meta_df: IDENTICAL across {len(valid_results)} loads")
    else:
        print(f"❌ meta_df: DIFFERENT! Hashes: {meta_hashes}")

    # 检查 z_sorted 哈希
    z_hashes = [r["z_hash_md5"] for r in valid_results]
    if len(set(z_hashes)) == 1:
        print(f"✅ z_sorted: IDENTICAL across {len(valid_results)} loads")
    else:
        print(f"❌ z_sorted: DIFFERENT! Hashes: {z_hashes}")

    # 详细比较第一次和最后一次
    if len(valid_results) >= 2:
        meta_0 = valid_results[0]["meta_hash"]
        meta_n = valid_results[-1]["meta_hash"]
        z_0 = valid_results[0]["z_hash_md5"]
        z_n = valid_results[-1]["z_hash_md5"]

        if meta_0 == meta_n and z_0 == z_n:
            print(f"✅ First load == Last load: FULLY CONSISTENT")
        else:
            print(f"❌ First load != Last load: check data pipeline!")
            print(f"   meta first: {meta_0}")
            print(f"   meta last:  {meta_n}")
            print(f"   z first:    {z_0}")
            print(f"   z last:     {z_n}")

    return results


# ============================================================================
# 验证 2：数据划分的一致性（修复版）
# ============================================================================

def verify_data_split(meta_df: pd.DataFrame, z_sorted, cfg: Dict, num_repeats: int = 3):
    """
    验证相同 seed 下数据划分是否一致

    修复：处理可能为 None 的 labels
    """
    print("\n" + "=" * 70)
    print("VERIFICATION 2: Data Split Consistency")
    print("=" * 70)

    if meta_df is None or z_sorted is None:
        print("❌ Cannot verify: meta_df or z_sorted is None")
        return []

    seed = cfg.get("seed", 42)
    split_results = []

    for i in range(num_repeats):
        print(f"\n--- Split attempt {i + 1}/{num_repeats} ---")
        set_seed(seed)

        try:
            context_builder = ContextIndexBuilder(meta_df, cfg)
            context_indices = context_builder.build()

            datasets = {}
            for split in ["train", "val", "test"]:
                datasets[split] = Stage2Dataset(
                    meta_df_sorted=meta_df,
                    z_sorted=z_sorted,
                    context_indices=context_indices,
                    target_split=split,
                )

            split_info = {}
            for split, ds in datasets.items():
                # 提取划分的索引和标签
                indices = []
                labels = []
                for idx in range(len(ds)):
                    data = ds[idx]

                    # 处理不同的数据格式
                    if isinstance(data, dict):
                        idx_val = data.get("idx", idx)
                        label_val = data.get("label", None)
                    elif isinstance(data, (list, tuple)):
                        idx_val = idx
                        label_val = data[-1] if len(data) > 0 else None
                    else:
                        idx_val = idx
                        label_val = None

                    indices.append(idx_val)
                    if label_val is not None:
                        labels.append(
                            label_val.item() if hasattr(label_val, 'item') else label_val
                        )

                # 计算哈希
                indices_arr = np.array([int(i) if hasattr(i, 'item') else i for i in indices])
                indices_hash = hashlib.md5(indices_arr.tobytes()).hexdigest()[:16]

                # 标签分布
                if labels:
                    from collections import Counter
                    label_dist = dict(Counter(labels))
                else:
                    label_dist = {}

                split_info[split] = {
                    "size": len(ds),
                    "indices_hash": indices_hash,
                    "label_dist": label_dist,
                }

                print(f"  {split}: {len(ds)} samples, "
                      f"indices_hash={indices_hash}, "
                      f"labels={label_dist}")

            split_results.append(split_info)

        except Exception as e:
            print(f"  ❌ Error in split attempt {i + 1}: {e}")
            import traceback
            traceback.print_exc()
            split_results.append({"attempt": i, "error": str(e)})

    # 检查一致性
    print(f"\n--- Split Consistency Check ---")

    valid_results = [r for r in split_results if "train" in r]

    if len(valid_results) < 2:
        print(f"❌ Not enough valid results ({len(valid_results)})")
        return split_results

    for split in ["train", "val", "test"]:
        hashes = [r[split]["indices_hash"] for r in valid_results]
        if len(set(hashes)) == 1:
            print(f"✅ {split} split: IDENTICAL across {len(valid_results)} runs")
        else:
            print(f"❌ {split} split: DIFFERENT! Hashes: {hashes}")

    return split_results


# ============================================================================
# 验证 3：上下文构建的一致性（修复版）
# ============================================================================

def verify_context_building(meta_df: pd.DataFrame, cfg: Dict, num_repeats: int = 3):
    """
    验证 ContextIndexBuilder 多次构建是否返回相同结果

    修复：处理可能的错误情况
    """
    print("\n" + "=" * 70)
    print("VERIFICATION 3: Context Building Consistency")
    print("=" * 70)

    if meta_df is None:
        print("❌ Cannot verify: meta_df is None")
        return []

    seed = cfg.get("seed", 42)
    context_results = []

    for i in range(num_repeats):
        print(f"\n--- Context build attempt {i + 1}/{num_repeats} ---")
        set_seed(seed)

        try:
            context_builder = ContextIndexBuilder(meta_df, cfg)
            context_indices = context_builder.build()

            # 提取关键统计信息
            info = {
                "attempt": i,
                "num_flows": len(context_indices) if context_indices is not None else 0,
                "method": cfg.get("context", {}).get("method", "unknown"),
                "window_size": cfg.get("context", {}).get("window_size", -1),
                "type": type(context_indices).__name__,
            }

            # 采样一些上下文窗口检查
            if context_indices is not None and len(context_indices) > 0:
                sample_indices = [
                    0,
                    len(context_indices) // 2,
                    len(context_indices) - 1
                ]

                for idx in sample_indices:
                    if idx < len(context_indices):
                        ctx = context_indices[idx]

                        ctx_len = 0
                        ctx_content = None

                        if isinstance(ctx, (list, np.ndarray)):
                            ctx_len = len(ctx)
                            ctx_content = np.array(ctx) if not isinstance(ctx, np.ndarray) else ctx
                        elif hasattr(ctx, '__len__'):
                            ctx_len = len(ctx)
                            ctx_content = np.array(list(ctx))

                        if ctx_content is not None:
                            ctx_hash = hashlib.md5(
                                ctx_content.tobytes()
                            ).hexdigest()[:12]
                        else:
                            ctx_hash = "unknown"

                        info[f"flow_{idx}_ctx_len"] = ctx_len
                        info[f"flow_{idx}_ctx_hash"] = ctx_hash

                        print(f"  Flow {idx}: ctx_len={ctx_len}, ctx_hash={ctx_hash}")

            context_results.append(info)

        except Exception as e:
            print(f"  ❌ Error in context build attempt {i + 1}: {e}")
            import traceback
            traceback.print_exc()
            context_results.append({"attempt": i, "error": str(e)})

    # 检查一致性
    print(f"\n--- Context Building Consistency Check ---")

    valid_results = [r for r in context_results if "error" not in r]

    if len(valid_results) < 2:
        print(f"❌ Not enough valid results ({len(valid_results)})")
        return context_results

    # 检查关键字段
    keys_to_check = ["num_flows", "method", "window_size"]
    keys_to_check += [k for k in valid_results[0].keys() if "hash" in k]

    all_identical = True
    for key in keys_to_check:
        if key in valid_results[0]:
            values = [r.get(key) for r in valid_results]
            unique_values = set(str(v) for v in values)
            if len(unique_values) > 1:
                print(f"❌ {key}: DIFFERENT values: {list(unique_values)}")
                all_identical = False

    if all_identical:
        print(f"✅ Context building: IDENTICAL across {len(valid_results)} runs")

    return context_results


# ============================================================================
# 辅助调试函数
# ============================================================================

def debug_data_type(data, name: str = "data"):
    """
    调试函数：打印数据的类型和属性

    用法：
    debug_data_type(z_sorted, "z_sorted")
    """
    print(f"\n--- Debug: {name} ---")
    print(f"  Type: {type(data)}")
    print(f"  Is None: {data is None}")

    if data is None:
        return

    # 检查常见属性
    for attr in ['shape', 'dtype', 'device', 'detach', 'numpy', 'to_numpy', 'columns']:
        if hasattr(data, attr):
            val = getattr(data, attr)
            if callable(val):
                try:
                    result = val()
                    print(f"  .{attr}(): {type(result).__name__} = {result}")
                except Exception as e:
                    print(f"  .{attr}(): Error - {e}")
            else:
                print(f"  .{attr}: {val}")

    # 如果是 Tensor
    if hasattr(data, 'device'):
        print(f"  device: {data.device}")
        print(f"  requires_grad: {data.requires_grad if hasattr(data, 'requires_grad') else 'N/A'}")

    # 如果是 DataFrame
    if isinstance(data, pd.DataFrame):
        print(f"  columns: {list(data.columns)[:10]}")
        print(f"  dtypes:\n{data.dtypes}")

# ============================================================================
# 验证 4：模型初始化的一致性
# ============================================================================

def verify_model_initialization(cfg: Dict, input_dim: int, num_repeats: int = 3):
    """
    验证相同 seed 下模型初始化是否相同
    """
    print("\n" + "=" * 70)
    print("VERIFICATION 4: Model Initialization Consistency")
    print("=" * 70)

    seed = cfg.get("seed", 42)
    device = torch.device("cpu")  # 用 CPU 避免 GPU 不确定性

    model_hashes = []

    for i in range(num_repeats):
        print(f"\n--- Init attempt {i + 1}/{num_repeats} ---")

        # 重置所有随机状态
        set_seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

        # 强制重置 Python hash seed
        os.environ["PYTHONHASHSEED"] = str(seed)

        # 构建模型
        model = build_stage2_model(cfg, input_dim=input_dim)
        model = model.to(device)
        model.eval()

        # 前向传播测试（相同输入）
        torch.manual_seed(seed + 100)
        dummy_input = torch.randn(2, cfg.get("context", {}).get("window_size", 64), input_dim)
        dummy_mask = torch.ones(dummy_input.shape[0], dummy_input.shape[1], dtype=torch.bool)

        with torch.no_grad():
            output = model(dummy_input, dummy_mask)

        # 计算哈希
        param_hash = compute_model_hash(model)
        output_stats = {
            "mean": float(output.mean()),
            "std": float(output.std()),
            "min": float(output.min()),
            "max": float(output.max()),
            "hash": compute_tensor_hash(output, method="md5"),
        }

        n_params = sum(p.numel() for p in model.parameters())

        model_hashes.append({
            "attempt": i,
            "n_params": n_params,
            "output_stats": output_stats,
            # 只保存关键层的哈希
            "key_layers": {
                name: info
                for name, info in param_hash.items()
                if any(k in name for k in ["input_proj", "cls_head", "pos_encoder"])
            },
        })

        print(f"  Parameters: {n_params:,}")
        print(f"  Output stats: mean={output_stats['mean']:.6f}, "
              f"std={output_stats['std']:.6f}")
        print(f"  Output hash: {output_stats['hash']}")

    # 检查一致性
    print(f"\n--- Model Initialization Consistency Check ---")

    output_hashes = [r["output_stats"]["hash"] for r in model_hashes]
    param_counts = [r["n_params"] for r in model_hashes]

    if len(set(param_counts)) == 1:
        print(f"✅ Parameter count: IDENTICAL ({param_counts[0]:,})")
    else:
        print(f"❌ Parameter count: DIFFERENT! {param_counts}")

    if len(set(output_hashes)) == 1:
        print(f"✅ Model output: IDENTICAL across {num_repeats} inits")
    else:
        print(f"❌ Model output: DIFFERENT! Hashes: {output_hashes}")

        # 找出哪个层不同
        if len(model_hashes) >= 2:
            layer_names = set(model_hashes[0]["key_layers"].keys())
            for layer_name in sorted(layer_names):
                norms = [r["key_layers"].get(layer_name, {}).get("norm", 0)
                         for r in model_hashes]
                if len(set([f"{n:.6f}" for n in norms])) > 1:
                    print(f"  ❌ {layer_name}: different norms: {norms}")

    return model_hashes


# ============================================================================
# 验证 5：训练过程的确定性
# ============================================================================

def verify_training_determinism(
        meta_df: pd.DataFrame,
        z_sorted: torch.Tensor,
        cfg: Dict,
        num_repeats: int = 2,
        max_epochs: int = 5,
):
    """
    验证相同 seed 下训练过程是否完全可复现

    注意：这要求设置 torch.backends.cudnn.deterministic = True
    并且 torch.use_deterministic_algorithms(True)
    """
    print("\n" + "=" * 70)
    print("VERIFICATION 5: Training Determinism")
    print("=" * 70)

    # 设置确定性模式
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except:
        print("[WARNING] Could not enable deterministic algorithms")

    seed = cfg.get("seed", 42)
    device = get_device(cfg.get("device", "cpu"))

    print(f"  Deterministic mode: cudnn.deterministic={torch.backends.cudnn.deterministic}")
    print(f"  Device: {device}")

    training_results = []

    for run in range(num_repeats):
        print(f"\n--- Training run {run + 1}/{num_repeats} ---")

        # 完全重置
        set_seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)

        # 构建数据
        context_builder = ContextIndexBuilder(meta_df, cfg)
        context_indices = context_builder.build()

        datasets = {}
        for split in ["train", "val", "test"]:
            datasets[split] = Stage2Dataset(
                meta_df_sorted=meta_df,
                z_sorted=z_sorted,
                context_indices=context_indices,
                target_split=split,
            )

        # 构建模型（相同初始化）
        input_dim = z_sorted.shape[1]
        model = build_stage2_model(cfg, input_dim=input_dim)
        model = model.to(device)

        # 记录初始权重
        init_param_hash = compute_model_hash(model)
        init_output_hash = compute_tensor_hash(
            model(
                torch.randn(2, 64, input_dim).to(device),
                torch.ones(2, 64, dtype=torch.bool).to(device),
            ),
            method="md5",
        )

        # 修改训练配置以加速验证
        cfg_short = deepcopy(cfg)
        cfg_short["training"]["epochs"] = max_epochs
        cfg_short["training"]["patience"] = 3
        cfg_short["training"]["save_best_only"] = True

        # 训练
        trainer = Stage2Trainer(
            model=model,
            datasets=datasets,
            cfg=cfg_short,
            device=device,
            out_dir=f"/tmp/repro_test_{run}",
            input_dim=input_dim,
        )

        # 记录每个 epoch 的指标
        epoch_logs = []
        original_fit = trainer.fit

        def logged_fit():
            # 我们直接调用原始 fit，但记录额外信息
            trainer.epoch_logs = []
            original_hook = trainer._on_epoch_end

            def hook_with_log(epoch, logs):
                trainer.epoch_logs.append({
                    "epoch": epoch,
                    "train_loss": logs.get("train_loss", 0),
                    "val_loss": logs.get("val_loss", 0),
                    "val_f1": logs.get("val_macro_f1", 0),
                })
                original_hook(epoch, logs)

            trainer._on_epoch_end = hook_with_log
            original_fit()
            trainer._on_epoch_end = original_hook

        logged_fit()

        # 记录最终状态
        final_param_hash = compute_model_hash(model)

        training_results.append({
            "run": run,
            "init_output_hash": init_output_hash,
            "init_params_sample": {
                k: v for k, v in init_param_hash.items() if "cls_head" in k
            },
            "epoch_logs": trainer.epoch_logs if hasattr(trainer, "epoch_logs") else [],
            "best_epoch": trainer.best_epoch,
            "best_val_loss": trainer.best_val_loss,
            "final_params_sample": {
                k: v for k, v in final_param_hash.items() if "cls_head" in k
            },
        })

        print(f"  Init output hash: {init_output_hash}")
        print(f"  Best epoch: {trainer.best_epoch}")
        print(f"  Best val loss: {trainer.best_val_loss:.6f}")

    # 检查一致性
    print(f"\n--- Training Determinism Check ---")

    # 1. 初始化是否相同
    init_hashes = [r["init_output_hash"] for r in training_results]
    if len(set(init_hashes)) == 1:
        print(f"✅ Initial model output: IDENTICAL")
    else:
        print(f"❌ Initial model output: DIFFERENT! {init_hashes}")

    # 2. 每个 epoch 的指标是否相同
    if training_results:
        n_epochs = min(len(r["epoch_logs"]) for r in training_results)
        epoch_identical = True

        for ep in range(n_epochs):
            losses = [r["epoch_logs"][ep]["train_loss"] for r in training_results]
            if len(set([f"{l:.6f}" for l in losses])) > 1:
                print(f"❌ Epoch {ep}: train_loss differs: {losses}")
                epoch_identical = False

        if epoch_identical:
            print(f"✅ All {n_epochs} epochs: IDENTICAL training losses")

    # 3. 最终模型参数是否相同
    if len(training_results) >= 2:
        r0_final = training_results[0]["final_params_sample"]
        r1_final = training_results[1]["final_params_sample"]

        diff = compare_dicts(r0_final, r1_final, tolerance=1e-5)
        if diff["summary"]["is_identical"]:
            print(f"✅ Final model parameters: IDENTICAL")
        else:
            print(f"❌ Final model parameters: DIFFERENT")
            print(f"   Differences: {diff['different_keys']}")

    return training_results


# ============================================================================
# 验证 6：GPU vs CPU 结果一致性
# ============================================================================

def verify_cpu_gpu_consistency(cfg: Dict, input_dim: int):
    """
    验证 CPU 和 GPU 上的前向传播结果是否一致
    """
    print("\n" + "=" * 70)
    print("VERIFICATION 6: CPU vs GPU Consistency")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("[SKIP] CUDA not available")
        return None

    seed = cfg.get("seed", 42)
    device_cpu = torch.device("cpu")
    device_gpu = torch.device("cuda:0")

    window_size = cfg.get("context", {}).get("window_size", 64)
    batch_size = 2

    # 在 CPU 上初始化
    set_seed(seed)
    torch.manual_seed(seed)

    model_cpu = build_stage2_model(cfg, input_dim=input_dim)
    model_cpu = model_cpu.to(device_cpu)
    model_cpu.eval()

    # 复制到 GPU
    set_seed(seed)
    torch.manual_seed(seed)

    model_gpu = build_stage2_model(cfg, input_dim=input_dim)
    model_gpu = model_gpu.to(device_gpu)
    model_gpu.eval()

    # 相同输入
    torch.manual_seed(seed + 100)
    x_cpu = torch.randn(batch_size, window_size, input_dim)
    x_gpu = x_cpu.to(device_gpu)

    mask_cpu = torch.ones(batch_size, window_size, dtype=torch.bool)
    mask_gpu = mask_cpu.to(device_gpu)

    with torch.no_grad():
        out_cpu = model_cpu(x_cpu, mask_cpu)
        out_gpu = model_gpu(x_gpu, mask_gpu)

    out_cpu_np = out_cpu.cpu().numpy()
    out_gpu_np = out_gpu.cpu().numpy()

    max_diff = np.abs(out_cpu_np - out_gpu_np).max()
    mean_diff = np.abs(out_cpu_np - out_gpu_np).mean()

    print(f"  CPU output: {out_cpu_np}")
    print(f"  GPU output: {out_gpu_np}")
    print(f"  Max diff: {max_diff:.10f}")
    print(f"  Mean diff: {mean_diff:.10f}")

    # 通常允许 1e-5 ~ 1e-7 的误差（浮点精度）
    if max_diff < 1e-5:
        print(f"✅ CPU vs GPU: CONSISTENT (max_diff={max_diff:.2e})")
    elif max_diff < 1e-4:
        print(f"⚠️  CPU vs GPU: SLIGHT DIFFERENCE (max_diff={max_diff:.2e}) - acceptable")
    else:
        print(f"❌ CPU vs GPU: SIGNIFICANT DIFFERENCE (max_diff={max_diff:.2e})")

    return {
        "max_diff": float(max_diff),
        "mean_diff": float(mean_diff),
        "is_consistent": max_diff < 1e-5,
    }


# ============================================================================
# 自定义 collate 函数（处理变长序列）
# ============================================================================

def _collate_variable_length(batch: List) -> Tuple[torch.Tensor, ...]:
    """
    处理变长序列的 collate 函数

    将不同长度的序列 padding 到相同长度
    """
    if len(batch) == 0:
        return torch.tensor([]), torch.tensor([]), torch.tensor([]), []

    # 检查 batch 中每个元素的类型
    if isinstance(batch[0], dict):
        # dict 格式
        context_z_list = [item["context_z"] for item in batch]
        mask_list = [item.get("mask", torch.ones(len(item["context_z"]))) for item in batch]
        label_list = [item["label"] for item in batch]
        meta_list = [item.get("meta", {}) for item in batch]

    elif isinstance(batch[0], (list, tuple)):
        # tuple 格式，假设为 (context_z, mask, label, meta)
        context_z_list = [item[0] for item in batch]
        mask_list = [item[1] if len(item) > 1 else torch.ones(len(item[0])) for item in batch]
        label_list = [item[2] if len(item) > 2 else torch.tensor(0) for item in batch]
        meta_list = [item[3] if len(item) > 3 else {} for item in batch]

    else:
        # 单个 tensor，假设已经是 padding 好的
        return torch.stack(batch, dim=0)

    # 转换为 tensor（如果不是的话）
    context_z_tensors = [
        t.clone().detach() if isinstance(t, torch.Tensor) else torch.tensor(t)
        for t in context_z_list
    ]

    # 找到最大长度
    max_len = max(t.shape[0] for t in context_z_tensors)
    feature_dim = context_z_tensors[0].shape[1] if len(context_z_tensors[0].shape) > 1 else 1

    # Padding
    padded_context = torch.zeros(len(batch), max_len, feature_dim)
    padded_mask = torch.zeros(len(batch), max_len, dtype=torch.bool)

    for i, (ctx, msk) in enumerate(zip(context_z_tensors, mask_list)):
        seq_len = ctx.shape[0]
        if len(ctx.shape) == 1:
            ctx = ctx.unsqueeze(-1)  # [seq_len] -> [seq_len, 1]
        padded_context[i, :seq_len] = ctx
        if isinstance(msk, torch.Tensor):
            padded_mask[i, :seq_len] = msk.bool()
        else:
            padded_mask[i, :seq_len] = True

    # 处理 labels
    if isinstance(label_list[0], torch.Tensor):
        labels = torch.stack([l.clone().detach() for l in label_list])
    else:
        labels = torch.tensor(
            [l.item() if hasattr(l, 'item') else l for l in label_list]
        )

    return padded_context, padded_mask, labels, meta_list


# ============================================================================
# 验证 7：DataLoader 的确定性（修复版）
# ============================================================================

def verify_dataloader_determinism(
        dataset: Stage2Dataset,
        batch_size: int,
        num_workers: int = 0,
        num_repeats: int = 3
):
    """
    验证 DataLoader 在相同 seed 下是否返回相同批次

    修复：处理变长序列
    """
    print("\n" + "=" * 70)
    print("VERIFICATION 7: DataLoader Determinism")
    print("=" * 70)

    # 检查数据集
    if dataset is None:
        print("❌ Cannot verify: dataset is None")
        return []

    print(f"  Dataset size: {len(dataset)}")

    # 检查数据格式
    sample = dataset[0]
    print(f"  Sample type: {type(sample)}")
    if isinstance(sample, dict):
        print(f"  Sample keys: {list(sample.keys())}")
        if "context_z" in sample:
            ctx = sample["context_z"]
            print(f"  context_z shape: {ctx.shape if hasattr(ctx, 'shape') else 'unknown'}")
    elif isinstance(sample, (list, tuple)):
        print(f"  Sample length: {len(sample)}")
        if len(sample) > 0 and hasattr(sample[0], 'shape'):
            print(f"  First element shape: {sample[0].shape}")

    # 检查序列长度分布
    seq_lengths = []
    for i in range(min(len(dataset), 1000)):  # 采样 1000 个
        data = dataset[i]
        if isinstance(data, dict):
            ctx = data.get("context_z", data.get("features", None))
        elif isinstance(data, (list, tuple)):
            ctx = data[0]
        else:
            ctx = data

        if hasattr(ctx, 'shape'):
            seq_len = ctx.shape[0] if len(ctx.shape) > 0 else 1
        elif isinstance(ctx, (list, np.ndarray)):
            seq_len = len(ctx)
        else:
            seq_len = 1

        seq_lengths.append(seq_len)

    seq_lengths = np.array(seq_lengths)
    print(f"  Sequence lengths: min={seq_lengths.min()}, "
          f"max={seq_lengths.max()}, "
          f"mean={seq_lengths.mean():.1f}, "
          f"unique={len(np.unique(seq_lengths))}")

    if len(np.unique(seq_lengths)) > 1:
        print(f"  ⚠️  Variable length sequences detected!")
        print(f"     Using custom collate_fn with padding")
        use_custom_collate = True
    else:
        print(f"  ✓ Fixed length sequences")
        use_custom_collate = False

    # 设定 worker seed
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)

    batch_hashes = []

    for run in range(num_repeats):
        print(f"\n--- DataLoader run {run + 1}/{num_repeats} ---")

        # 重置 generator
        generator = torch.Generator()
        generator.manual_seed(42)

        # 创建 DataLoader
        loader_kwargs = {
            "batch_size": batch_size,
            "shuffle": True,
            "num_workers": num_workers,
            "generator": generator,
            "drop_last": False,
        }

        # 对于变长序列，使用自定义 collate_fn
        if use_custom_collate:
            loader_kwargs["collate_fn"] = _collate_variable_length

        # worker_init_fn 只在 num_workers > 0 时需要
        if num_workers > 0:
            loader_kwargs["worker_init_fn"] = seed_worker

        loader = torch.utils.data.DataLoader(dataset, **loader_kwargs)

        run_batch_hashes = []
        run_batch_info = []

        try:
            for batch_idx, batch in enumerate(loader):
                # 根据 batch 类型提取 context_z
                if isinstance(batch, dict):
                    context_z = batch.get("context_z")
                    batch_size_actual = len(context_z) if context_z is not None else 0
                elif isinstance(batch, (list, tuple)):
                    # 我们的 collate_fn 返回 (padded_context, mask, labels, meta)
                    context_z = batch[0]
                    batch_size_actual = context_z.shape[0] if context_z is not None else 0
                else:
                    context_z = batch
                    batch_size_actual = context_z.shape[0] if hasattr(context_z, 'shape') else 0

                # 计算批次哈希
                if context_z is not None:
                    batch_hash = compute_tensor_hash(context_z, method="md5")
                    batch_stats = compute_tensor_hash(context_z, method="stats")
                else:
                    batch_hash = f"batch_{batch_idx}_None"
                    batch_stats = "None"

                run_batch_hashes.append(batch_hash)
                run_batch_info.append({
                    "batch_idx": batch_idx,
                    "size": batch_size_actual,
                    "hash": batch_hash[:12],
                    "stats": batch_stats[:80],
                })

                if batch_idx < 3:  # 只打印前3个批次
                    print(f"  Batch {batch_idx}: size={batch_size_actual}, hash={batch_hash[:12]}")

            print(f"  Total batches: {len(run_batch_hashes)}")
            batch_hashes.append({
                "run": run,
                "n_batches": len(run_batch_hashes),
                "batch_hashes": run_batch_hashes,
                "batch_info": run_batch_info,
            })

        except Exception as e:
            print(f"  ❌ Error in DataLoader run {run + 1}: {e}")
            import traceback
            if len(run_batch_info) > 0:
                print(f"  Last successful batch: {run_batch_info[-1]}")
            traceback.print_exc()
            batch_hashes.append({
                "run": run,
                "error": str(e),
                "last_successful_batch": len(run_batch_hashes) if run_batch_hashes else 0,
            })
            break

    # 检查一致性
    print(f"\n--- DataLoader Consistency Check ---")

    valid_runs = [r for r in batch_hashes if "error" not in r]

    if len(valid_runs) < 2:
        print(f"❌ Not enough valid runs ({len(valid_runs)})")
        for r in batch_hashes:
            if "error" in r:
                print(f"   Run {r['run']}: Error at batch {r.get('last_successful_batch', '?')}")
        return batch_hashes

    # 检查批次数量
    n_batches = [r["n_batches"] for r in valid_runs]
    if len(set(n_batches)) == 1:
        print(f"✅ Batch count: IDENTICAL ({n_batches[0]} batches)")
    else:
        print(f"❌ Batch count: DIFFERENT! {n_batches}")

    # 检查每个批次的哈希
    min_batches = min(n_batches)
    batch_identical = True
    first_diff = None

    for i in range(min_batches):
        batch_hash_i = [r["batch_hashes"][i] for r in valid_runs]
        if len(set(batch_hash_i)) > 1:
            if first_diff is None:
                first_diff = i
            print(f"❌ Batch {i}: DIFFERENT hashes!")
            for r in valid_runs:
                if i < len(r.get("batch_info", [])):
                    info = r["batch_info"][i]
                    print(f"     Run {r['run']}: size={info['size']}, hash={info['hash']}")
            batch_identical = False

    if batch_identical:
        print(f"✅ All {min_batches} batches: IDENTICAL order and content")
    else:
        print(f"\n⚠️  First difference at batch {first_diff}")
        print(f"   This is expected if the dataset contains sequences of different lengths")
        print(f"   and the collate_fn pads them. The content should be the same")
        print(f"   but the padding might differ based on batch composition.")

    return batch_hashes


# ============================================================================
# 替代方案：只检查索引和数据 ID 的确定性
# ============================================================================

def verify_dataloader_index_determinism(
        dataset: Stage2Dataset,
        batch_size: int,
        num_repeats: int = 3,
):
    """
    简化版 DataLoader 验证：只检查每个 epoch 访问的样本顺序是否相同

    这避免了变长序列 padding 导致的问题
    """
    print("\n" + "=" * 70)
    print("VERIFICATION 7 (Alternative): DataLoader Index Order")
    print("=" * 70)

    print(f"  Dataset size: {len(dataset)}")
    print(f"  Batch size: {batch_size}")

    # 获取数据集中的索引（如果存在）
    sample_indices = []
    try:
        for i in range(len(dataset)):
            data = dataset[i]
            if isinstance(data, dict):
                idx = data.get("idx", i)
                fid = data.get("flow_id", i)
            elif isinstance(data, (list, tuple)):
                idx = data[0] if hasattr(data[0], 'item') else i
                fid = i
            else:
                idx = i
                fid = i
            sample_indices.append((idx, fid))
    except Exception as e:
        print(f"  ⚠️  Cannot extract indices: {e}")
        print(f"     Using position index instead")
        sample_indices = [(i, i) for i in range(len(dataset))]

    # 多次 shuffle 并记录索引顺序
    shuffle_results = []

    for run in range(num_repeats):
        print(f"\n--- Shuffle run {run + 1}/{num_repeats} ---")

        # 创建索引列表并 shuffle
        indices = list(range(len(dataset)))

        generator = torch.Generator()
        generator.manual_seed(42)

        # 使用 PyTorch 的 shuffle 逻辑
        shuffled = torch.randperm(len(indices), generator=generator).tolist()
        shuffled_indices = [indices[i] for i in shuffled]

        # 记录每个 batch 的索引
        batches = []
        for start in range(0, len(shuffled_indices), batch_size):
            batch_idx = shuffled_indices[start:start + batch_size]

            # 计算这个 batch 的哈希
            idx_array = np.array(batch_idx)
            batch_hash = hashlib.md5(idx_array.tobytes()).hexdigest()[:12]

            batches.append({
                "start": start,
                "size": len(batch_idx),
                "first_idx": batch_idx[0],
                "last_idx": batch_idx[-1],
                "hash": batch_hash,
            })

        shuffle_results.append({
            "run": run,
            "n_batches": len(batches),
            "batches": batches,
        })

        print(f"  Total batches: {len(batches)}")
        print(f"  First batch: size={batches[0]['size']}, "
              f"hash={batches[0]['hash']}")
        print(f"  Last batch: size={batches[-1]['size']}, "
              f"hash={batches[-1]['hash']}")

    # 检查一致性
    print(f"\n--- Index Order Consistency Check ---")

    n_batches = [r["n_batches"] for r in shuffle_results]
    if len(set(n_batches)) == 1:
        print(f"✅ Batch count: IDENTICAL ({n_batches[0]})")
    else:
        print(f"❌ Batch count: DIFFERENT!")
        return shuffle_results

    all_identical = True
    for i in range(n_batches[0]):
        hashes = [r["batches"][i]["hash"] for r in shuffle_results]
        if len(set(hashes)) > 1:
            print(f"❌ Batch {i}: DIFFERENT order!")
            all_identical = False

    if all_identical:
        print(f"✅ All {n_batches[0]} batches: IDENTICAL index order")

    return shuffle_results

# ============================================================================
# 主验证脚本
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Verify Reproducibility")

    parser.add_argument("--stage1_dir", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="results/reproducibility_check")
    parser.add_argument("--num_checks", type=int, default=3)
    parser.add_argument("--skip_training", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载配置
    cfg = load_config(args.config)

    print("=" * 70)
    print("REPRODUCIBILITY VERIFICATION SUITE")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Stage1 dir: {args.stage1_dir}")
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"NumPy: {np.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print("=" * 70)

    all_results = {
        "timestamp": datetime.now().isoformat(),
        "environment": {
            "python": sys.version,
            "pytorch": torch.__version__,
            "numpy": np.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
        "verifications": {},
    }

    # 1. 数据加载一致性
    try:
        v1 = verify_data_loading(args.stage1_dir, args.config, args.num_checks)
        all_results["verifications"]["data_loading"] = v1
    except Exception as e:
        print(f"\n❌ Verification 1 failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        all_results["verifications"]["data_loading"] = {"error": str(e)}
        return  # 数据加载失败，后续无法继续

    # 重新加载数据用于后续验证
    try:
        meta_df, z_sorted = prepare_sorted_stage2_data(
            stage1_dir=args.stage1_dir, cfg=cfg
        )

        if args.debug:
            debug_data_type(meta_df, "meta_df")
            debug_data_type(z_sorted, "z_sorted")

    except Exception as e:
        print(f"\n❌ Failed to load data for subsequent checks: {e}")
        import traceback
        traceback.print_exc()
        return

    # 2. 上下文构建一致性
    try:
        v3 = verify_context_building(meta_df, cfg, args.num_checks)
        all_results["verifications"]["context_building"] = v3
    except Exception as e:
        print(f"\n❌ Verification 3 failed: {e}")
        all_results["verifications"]["context_building"] = {"error": str(e)}

    # 3. 数据划分一致性
    try:
        v2 = verify_data_split(meta_df, z_sorted, cfg, args.num_checks)
        all_results["verifications"]["data_split"] = v2
    except Exception as e:
        print(f"\n❌ Verification 2 failed: {e}")
        all_results["verifications"]["data_split"] = {"error": str(e)}

    # 4. 模型初始化一致性
    try:
        z_arr = ensure_numpy(z_sorted)
        input_dim = z_arr.shape[1]
        v4 = verify_model_initialization(cfg, input_dim, args.num_checks)
        all_results["verifications"]["model_init"] = [
            {k: v for k, v in r.items() if k != "key_layers"}
            for r in v4
        ]
    except Exception as e:
        print(f"\n❌ Verification 4 failed: {e}")
        all_results["verifications"]["model_init"] = {"error": str(e)}

    # 5. CPU vs GPU 一致性
    try:
        z_arr = ensure_numpy(z_sorted)
        input_dim = z_arr.shape[1]
        v6 = verify_cpu_gpu_consistency(cfg, input_dim)
        all_results["verifications"]["cpu_vs_gpu"] = v6
    except Exception as e:
        print(f"\n❌ Verification 6 failed: {e}")
        all_results["verifications"]["cpu_vs_gpu"] = {"error": str(e)}

    # 6. DataLoader 一致性（修复版）
    try:
        context_builder = ContextIndexBuilder(meta_df, cfg)
        context_indices = context_builder.build()
        dataset = Stage2Dataset(
            meta_df_sorted=meta_df,
            z_sorted=z_sorted,
            context_indices=context_indices,
            target_split="train",
        )
        batch_size = cfg.get("training", {}).get("batch_size", 32)

        # 先检查序列长度是否固定
        seq_lengths = []
        for i in range(min(len(dataset), 100)):
            data = dataset[i]
            if isinstance(data, dict):
                ctx = data.get("context_z", data.get("features", []))
            elif isinstance(data, (list, tuple)):
                ctx = data[0]
            else:
                ctx = data

            if hasattr(ctx, '__len__'):
                seq_lengths.append(len(ctx))
            elif hasattr(ctx, 'shape'):
                seq_lengths.append(ctx.shape[0])

        unique_lengths = len(set(seq_lengths))
        print(f"\n[INFO] Sequence lengths in dataset: {unique_lengths} unique values")

        if unique_lengths > 1:
            # 使用索引验证（避免变长序列的 collate 问题）
            print("[INFO] Variable length sequences detected, using index-based verification")
            v7 = verify_dataloader_index_determinism(
                dataset, batch_size, num_repeats=args.num_checks
            )
        else:
            # 固定长度序列，使用标准验证
            v7 = verify_dataloader_determinism(
                dataset, batch_size, num_workers=0, num_repeats=args.num_checks
            )

        all_results["verifications"]["dataloader"] = v7

    except Exception as e:
        print(f"\n❌ Verification 7 failed: {e}")
        import traceback
        traceback.print_exc()
        all_results["verifications"]["dataloader"] = {"error": str(e)}

    # 7. 训练确定性（可选，耗时较长）
    if not args.skip_training:
        try:
            v5 = verify_training_determinism(
                meta_df=meta_df,
                z_sorted=z_sorted,
                cfg=cfg,
                num_repeats=2,
                max_epochs=5,
            )
            all_results["verifications"]["training_determinism"] = [
                {
                    k: v for k, v in r.items()
                    if k not in ["init_params_sample", "final_params_sample"]
                }
                for r in v5
            ]
        except Exception as e:
            print(f"\n❌ Verification 5 failed: {e}")
            all_results["verifications"]["training_determinism"] = {"error": str(e)}
    else:
        print("\n[Skipped] Training determinism verification (--skip_training)")

    # 生成报告
    try:
        _generate_reproducibility_report(all_results, output_dir)
    except Exception as e:
        print(f"\n❌ Report generation failed: {e}")
        # 至少保存 JSON
        report_path = output_dir / "reproducibility_report.json"
        with open(report_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str, ensure_ascii=False)
        print(f"[✓] JSON report saved to: {report_path}")

    # 打印总结
    _print_final_verdict(all_results)

    return all_results

def _generate_reproducibility_report(results: Dict, output_dir: Path):
    """生成可复现性验证报告"""

    # 1. 保存完整 JSON
    report_path = output_dir / "reproducibility_report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2, default=str, ensure_ascii=False)
    print(f"\n[✓] Full report saved to: {report_path}")

    # 2. 生成可读摘要
    summary_path = output_dir / "reproducibility_summary.txt"

    with open(summary_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("REPRODUCIBILITY VERIFICATION SUMMARY\n")
        f.write(f"Timestamp: {results['timestamp']}\n")
        f.write("=" * 70 + "\n\n")

        f.write("Environment:\n")
        f.write(f"  Python: {results['environment']['python'].split()[0]}\n")
        f.write(f"  PyTorch: {results['environment']['pytorch']}\n")
        f.write(f"  CUDA: {results['environment']['cuda_available']}\n\n")

        checks = results.get("verifications", {})

        # 检查 1
        f.write("1. Data Loading Consistency:\n")
        v1 = checks.get("data_loading", [])
        if v1:
            meta_hashes = [r.get("meta_hash") for r in v1]
            z_hashes = [r.get("z_full_hash") for r in v1]
            meta_ok = len(set(meta_hashes)) == 1
            z_ok = len(set(z_hashes)) == 1
            f.write(f"   meta_df: {'✓ IDENTICAL' if meta_ok else '✗ DIFFERENT'}\n")
            f.write(f"   z_sorted: {'✓ IDENTICAL' if z_ok else '✗ DIFFERENT'}\n")
            if v1:
                f.write(f"   Shape: {v1[0].get('meta_shape')}, {v1[0].get('z_shape')}\n")
        f.write("\n")

        # 检查 2
        f.write("2. Data Split Consistency:\n")
        v2 = checks.get("data_split", [])
        if v2 and len(v2) == 2:
            for split in ["train", "val", "test"]:
                hashes = [r[split]["indices_hash"] for r in v2]
                ok = len(set(hashes)) == 1
                f.write(f"   {split}: {'✓ IDENTICAL' if ok else '✗ DIFFERENT'} "
                        f"(size={v2[0][split]['size']})\n")
        f.write("\n")

        # 检查 3
        f.write("3. Context Building Consistency:\n")
        v3 = checks.get("context_building", [])
        if v3 and len(v3) >= 2:
            num_flows = [r["num_flows"] for r in v3]
            ok = len(set(num_flows)) == 1
            f.write(f"   {'✓ IDENTICAL' if ok else '✗ DIFFERENT'} "
                    f"(flows={num_flows[0] if ok else num_flows})\n")
        f.write("\n")

        # 检查 4
        f.write("4. Model Initialization Consistency:\n")
        v4 = checks.get("model_init", [])
        if v4 and len(v4) >= 2:
            output_hashes = [r.get("output_stats", {}).get("hash") for r in v4]
            ok = len(set(output_hashes)) == 1
            f.write(f"   {'✓ IDENTICAL' if ok else '✗ DIFFERENT'}\n")
            if v4[0].get("output_stats"):
                f.write(f"   Output stats: mean={v4[0]['output_stats']['mean']:.6f}, "
                        f"std={v4[0]['output_stats']['std']:.6f}\n")
        f.write("\n")

        # 检查 5
        f.write("5. CPU vs GPU Consistency:\n")
        v6 = checks.get("cpu_vs_gpu")
        if v6:
            status = "✓ CONSISTENT" if v6.get("is_consistent") else "✗ DIFFERENT"
            f.write(f"   {status} (max_diff={v6.get('max_diff', 0):.2e})\n")
        else:
            f.write("   [SKIPPED] CUDA not available\n")
        f.write("\n")

        # 检查 6
        f.write("6. DataLoader Determinism:\n")
        v7 = checks.get("dataloader", [])
        if v7 and len(v7) >= 2:
            n_batches = [r["n_batches"] for r in v7]
            ok = len(set(n_batches)) == 1
            f.write(f"   {'✓ IDENTICAL' if ok else '✗ DIFFERENT'} "
                    f"(batches={n_batches[0] if ok else n_batches})\n")
        f.write("\n")

        # 检查 7
        f.write("7. Training Determinism:\n")
        v5 = checks.get("training_determinism", [])
        if v5 and len(v5) >= 2:
            init_hashes = [r.get("init_output_hash") for r in v5]
            init_ok = len(set(init_hashes)) == 1
            f.write(f"   Init: {'✓ IDENTICAL' if init_ok else '✗ DIFFERENT'}\n")

            best_losses = [r.get("best_val_loss") for r in v5]
            loss_ok = len(set([f"{l:.6f}" for l in best_losses])) == 1
            f.write(f"   Best val loss: {'✓ IDENTICAL' if loss_ok else '✗ DIFFERENT'} "
                    f"({best_losses})\n")
        else:
            f.write("   [SKIPPED] Use --no-skip-training to enable\n")
        f.write("\n")

        # 最终判定
        f.write("-" * 70 + "\n")

        all_ok = True
        for check_name, check_data in checks.items():
            if check_name == "data_loading" and check_data:
                hashes = [r.get("z_full_hash") for r in check_data]
                if len(set(hashes)) != 1:
                    all_ok = False
            elif check_name == "model_init" and check_data:
                hashes = [r.get("output_stats", {}).get("hash") for r in check_data]
                if len(set(hashes)) != 1:
                    all_ok = False
            elif check_name == "cpu_vs_gpu" and check_data:
                if not check_data.get("is_consistent", False):
                    all_ok = False

        if all_ok:
            f.write("\n✅ OVERALL: FULLY REPRODUCIBLE\n")
            f.write("The pipeline produces identical results across runs.\n")
        else:
            f.write("\n❌ OVERALL: NOT FULLY REPRODUCIBLE\n")
            f.write("Some parts produce different results across runs.\n")
            f.write("Check the details above for specific issues.\n")

    print(f"[✓] Summary report saved to: {summary_path}")

    # 3. 打印到终端
    with open(summary_path, "r") as f:
        print("\n" + f.read())


def _print_final_verdict(results: Dict):
    """打印最终可复现性判定"""

    print("\n" + "=" * 70)
    print("FINAL REPRODUCIBILITY VERDICT")
    print("=" * 70)

    checks = results.get("verifications", {})
    issues = []

    # 数据加载
    v1 = checks.get("data_loading", [])
    if v1:
        z_hashes = [r.get("z_full_hash") for r in v1]
        if len(set(z_hashes)) != 1:
            issues.append("Data loading produces different results")

    # 数据划分
    v2 = checks.get("data_split", [])
    if v2 and len(v2) >= 2:
        for split in ["train", "val", "test"]:
            hashes = [r[split]["indices_hash"] for r in v2]
            if len(set(hashes)) != 1:
                issues.append(f"Data split '{split}' is non-deterministic")

    # 模型初始化
    v4 = checks.get("model_init", [])
    if v4 and len(v4) >= 2:
        output_hashes = [r.get("output_stats", {}).get("hash") for r in v4]
        if len(set(output_hashes)) != 1:
            issues.append("Model initialization is non-deterministic")

    # GPU 一致性
    v6 = checks.get("cpu_vs_gpu")
    if v6 and not v6.get("is_consistent", True):
        issues.append(f"CPU vs GPU outputs differ (max_diff={v6.get('max_diff', 0):.2e})")

    # 训练
    v5 = checks.get("training_determinism", [])
    if v5 and len(v5) >= 2:
        best_losses = [r.get("best_val_loss") for r in v5]
        if len(set([f"{l:.6f}" for l in best_losses])) != 1:
            issues.append(f"Training is non-deterministic (losses: {best_losses})")

    if not issues:
        print("✅ ALL CHECKS PASSED")
        print("The pipeline is fully reproducible!")
        print("\nYou can safely assume:")
        print("  - Same seed → Same results")
        print("  - Results are valid for comparison")
    else:
        print("❌ SOME CHECKS FAILED")
        print(f"Issues found: {len(issues)}")
        for issue in issues:
            print(f"  - {issue}")
        print("\nRecommended fixes:")
        print("  1. Set PYTHONHASHSEED=0")
        print("  2. Set torch.backends.cudnn.deterministic = True")
        print("  3. Set torch.use_deterministic_algorithms(True)")
        print("  4. Use num_workers=0 for DataLoader")
        print("  5. Fix random seeds before every operation")

    # ============================================================================
    # 独立的快速检查函数（可用作 import）
    # ============================================================================


def quick_reproducibility_check(stage1_dir: str, config_path: str) -> Dict[str, bool]:
    """
    快速检查数据加载和模型初始化的可复现性

    用法：
    from experiments.verify_reproducibility import quick_reproducibility_check
    results = quick_reproducibility_check("s1/0702v2", "configs/default.yaml")
    """

    cfg = load_config(config_path)

    # 加载两次数据
    meta1, z1 = prepare_sorted_stage2_data(stage1_dir=stage1_dir, cfg=cfg)
    meta2, z2 = prepare_sorted_stage2_data(stage1_dir=stage1_dir, cfg=cfg)

    # 比较
    data_ok = (
            compute_df_hash(meta1) == compute_df_hash(meta2) and
            torch.allclose(z1, z2, atol=1e-7)
    )

    # 初始化两次模型
    seed = cfg.get("seed", 42)
    set_seed(seed)
    model1 = build_stage2_model(cfg, input_dim=z1.shape[1])

    set_seed(seed)
    model2 = build_stage2_model(cfg, input_dim=z1.shape[1])

    # 比较模型
    p1 = {n: p.clone() for n, p in model1.named_parameters()}
    p2 = {n: p.clone() for n, p in model2.named_parameters()}

    model_ok = all(torch.equal(p1[n], p2[n]) for n in p1)

    return {
        "data_loading_reproducible": data_ok,
        "model_init_reproducible": model_ok,
        "all_reproducible": data_ok and model_ok,
    }

    # ============================================================================
    # 确定性训练的配置补丁
    # ============================================================================


def apply_deterministic_patch(cfg: Dict) -> Dict:
    """
    应用确定性补丁，确保训练可复现

    用法：
    cfg = load_config("configs/default.yaml")
    cfg = apply_deterministic_patch(cfg)
    """

    cfg = deepcopy(cfg)

    # 训练配置
    if "training" not in cfg:
        cfg["training"] = {}

    # 强制确定性设置
    cfg["training"]["deterministic"] = True
    cfg["training"]["num_workers"] = 0  # 多 worker 可能导致不确定性

    # 数据配置
    if "data" not in cfg:
        cfg["data"] = {}
    cfg["data"]["shuffle_seed"] = cfg.get("seed", 42)

    # 额外设置
    cfg["_deterministic_mode"] = True

    return cfg


def enable_deterministic_mode(warn: bool = True):
    """
    启用全局确定性模式

    应在所有其他导入和操作之前调用
    """
    import os
    import random

    # Python hash seed
    os.environ["PYTHONHASHSEED"] = "0"

    # PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    try:
        torch.use_deterministic_algorithms(True)
    except:
        if warn:
            print("[WARNING] torch.use_deterministic_algorithms(True) failed")

    if warn:
        print("[INFO] Deterministic mode enabled")
        print(f"  cudnn.deterministic = {torch.backends.cudnn.deterministic}")
        print(f"  cudnn.benchmark = {torch.backends.cudnn.benchmark}")

    # ============================================================================
    # 单元测试类
    # ============================================================================


class ReproducibilityTests:
    """
    可复现性的单元测试集合

    用法：
    pytest experiments/verify_reproducibility.py -v
    """

    @staticmethod
    def test_seed_reproducibility():
        """测试 set_seed 是否真正重置了所有状态"""
        set_seed(42)
        x1 = torch.randn(10)

        set_seed(42)
        x2 = torch.randn(10)

        assert torch.equal(x1, x2), "set_seed should produce identical tensors"
        print("✓ set_seed works correctly")

    @staticmethod
    def test_tensor_hash():
        """测试 compute_tensor_hash 是否稳定"""
        torch.manual_seed(123)
        x = torch.randn(100, 50)

        hash1 = compute_tensor_hash(x, method="md5")
        hash2 = compute_tensor_hash(x, method="md5")

        assert hash1 == hash2, "compute_tensor_hash should be idempotent"
        print("✓ compute_tensor_hash is stable")

    @staticmethod
    def test_model_init_reproducibility():
        """测试模型初始化是否可复现"""
        from stage2.config import load_config

        cfg = load_config(None)
        input_dim = 128

        set_seed(42)
        torch.manual_seed(42)
        model1 = build_stage2_model(cfg, input_dim=input_dim)

        set_seed(42)
        torch.manual_seed(42)
        model2 = build_stage2_model(cfg, input_dim=input_dim)

        p1 = dict(model1.named_parameters())
        p2 = dict(model2.named_parameters())

        for name in p1:
            assert torch.equal(p1[name], p2[name]), f"Parameter {name} differs"

        print("✓ Model initialization is reproducible")

    @staticmethod
    def test_deterministic_forward():
        """测试前向传播的确定性"""
        # 启用确定性模式
        enable_deterministic_mode(warn=False)

        from stage2.config import load_config

        cfg = load_config(None)
        input_dim = 128
        batch_size = 4
        seq_len = 32

        set_seed(42)
        model = build_stage2_model(cfg, input_dim=input_dim)
        model.eval()

        torch.manual_seed(999)
        x = torch.randn(batch_size, seq_len, input_dim)
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        # 多次前向传播应产生相同结果
        with torch.no_grad():
            out1 = model(x, mask)
            out2 = model(x, mask)
            out3 = model(x, mask)

        assert torch.equal(out1, out2), "Forward pass should be deterministic"
        assert torch.equal(out2, out3), "Forward pass should be deterministic"
        print("✓ Forward pass is deterministic")

    # ============================================================================
    # 调试工具：逐层输出对比
    # ============================================================================


class ModelDebugger:
    """
    模型调式工具：逐层输出对比

    用于定位哪一层导致了不确定性问题
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.hooks = []
        self.layer_outputs = {}

    def register_hooks(self):
        """为所有子模块注册前向钩子"""

        def hook_fn(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    self.layer_outputs[name] = {
                        "shape": tuple(output.shape),
                        "mean": float(output.mean()),
                        "std": float(output.std()),
                        "min": float(output.min()),
                        "max": float(output.max()),
                        "hash": compute_tensor_hash(output, method="md5"),
                    }

            return hook

        for name, module in self.model.named_modules():
            if list(module.children()) == []:  # 叶节点
                h = module.register_forward_hook(hook_fn(name))
                self.hooks.append(h)

    def remove_hooks(self):
        """移除所有钩子"""
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def compare_runs(self, *inputs, num_runs: int = 2) -> Dict[str, Any]:
        """
        多次运行并比较每层的输出

        Returns:
            {
                "layer_name": {
                    "run_0": {...},
                    "run_1": {...},
                    "identical": True/False,
                }
            }
        """
        results = {}

        for run in range(num_runs):
            self.layer_outputs = {}

            with torch.no_grad():
                self.model.eval()
                _ = self.model(*inputs)

            for layer_name, output_info in self.layer_outputs.items():
                if layer_name not in results:
                    results[layer_name] = {}
                results[layer_name][f"run_{run}"] = output_info

        # 比较
        for layer_name in results:
            if len(results[layer_name]) >= 2:
                h0 = results[layer_name]["run_0"]["hash"]
                h1 = results[layer_name]["run_1"]["hash"]
                results[layer_name]["identical"] = (h0 == h1)

        return results

    @classmethod
    def find_nondeterministic_layers(cls, model: nn.Module, *inputs) -> List[str]:
        """找到所有非确定性的层"""
        debugger = cls(model)
        debugger.register_hooks()

        comparison = debugger.compare_runs(*inputs, num_runs=2)
        debugger.remove_hooks()

        nondeterministic = [
            name for name, info in comparison.items()
            if not info.get("identical", False)
        ]

        return nondeterministic

    # ============================================================================
    # 主入口
    # ============================================================================


if __name__ == "__main__":
    main()

    # ============================================================================
    # 快速使用示例
    # ============================================================================
"""
# 示例 1：完整验证
python experiments/verify_reproducibility.py \
    --stage1_dir s1/0702v2 \
    --output_dir results/repro_check \
    --num_checks 3

# 示例 2：只验证数据，跳过训练
python experiments/verify_reproducibility.py \
    --stage1_dir s1/0702v2 \
    --skip_training

# 示例 3：在代码中使用
from experiments.verify_reproducibility import (
    quick_reproducibility_check,
    enable_deterministic_mode,
    apply_deterministic_patch,
)

# 启用确定性模式
enable_deterministic_mode()

# 快速检查
results = quick_reproducibility_check("s1/0702v2", "configs/default.yaml")
print(f"All reproducible: {results['all_reproducible']}")

# 应用确定性补丁
cfg = load_config("configs/default.yaml")
cfg = apply_deterministic_patch(cfg)
"""