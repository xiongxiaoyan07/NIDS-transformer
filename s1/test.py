# numerical_proof.py
"""
证明 expm1→cumsum→log ≈ cumsum（在 log1p 变换后的数据上）
"""
import numpy as np


def prove_equivalence():
    # 模拟你的数据
    # 假设原始间隔（微秒）
    raw_intervals = np.array([0, 200000, 500000, 1500000, 30, 5000, 3, 800000])

    # 预处理后的存储值
    time_log = np.log1p(raw_intervals)
    print("原始间隔 (微秒):", raw_intervals)
    print("存储的 time_log:", time_log)
    print("  expm1(time_log):", np.expm1(time_log))  # 验证可逆性

    print("\n" + "=" * 60)
    print("方法对比:")
    print("=" * 60)

    # 方法1：直接累积 time_log
    method1 = np.cumsum(time_log)
    print(f"\n方法1 - cumsum(time_log):")
    print(f"  值: {method1}")
    print(f"  范围: [{method1.min():.2f}, {method1.max():.2f}]")

    # 方法2：expm1 → cumsum → log（你的代码）
    method2_intervals = np.expm1(time_log)
    method2_cumsum = np.cumsum(method2_intervals)
    method2 = np.log1p(method2_cumsum)
    print(f"\n方法2 - log1p(cumsum(expm1(time_log))):")
    print(f"  expm1: {method2_intervals}")
    print(f"  cumsum: {method2_cumsum}")
    print(f"  最终: {method2}")
    print(f"  范围: [{method2.min():.2f}, {method2.max():.2f}]")

    # 比较
    diff = np.abs(method1 - method2)
    print(f"\n差异: {diff}")
    print(f"最大差异: {diff.max():.10f}")
    print(f"平均差异: {diff.mean():.10f}")

    # 为什么相似？
    # log(1 + cumsum(expm1(time_log))) ≈ log(cumsum(exp(time_log))) ≈ cumsum(time_log)
    # 因为 exp(log1p(x)) = 1+x，log(1+cumsum(x)) ≈ log(cumsum(1+x)) ≈ cumsum(log(1+x))

    print("\n" + "=" * 60)
    print("结论:")
    print("=" * 60)
    print(f"两种方法高度相似！差异 < {diff.max():.10f}")
    print("你的 _compute_time_feature 做的是无用功")


if __name__ == '__main__':
    prove_equivalence()
# import numpy as np
# import os
# from pathlib import Path
#
#
# def detailed_inspect_npz(tensor_dir='./tensors/precomputed'):
#     """
#     详细检查 .npz 文件中的每个字段
#     """
#     files = os.listdir(tensor_dir)
#     npz_files = [f for f in files if f.endswith('.npz')]
#
#     if not npz_files:
#         print(f"❌ 在 {tensor_dir} 中没有找到 .npz 文件")
#         return
#
#     print("=" * 80)
#     print("🔍 详细数据分析")
#     print("=" * 80)
#
#     npz_file = [f for f in npz_files if 'train' in f.lower()][0]
#     # 逐个文件分析
#     # for npz_file in train_file:
#     print(f"\n{'=' * 80}")
#     print(f"📄 文件: {npz_file}")
#     print(f"{'=' * 80}")
#
#     data = np.load(os.path.join(tensor_dir, npz_file), allow_pickle=True)
#     # data = np.load(npz_file, allow_pickle=True)
#
#     # ============================================
#     # 1. X - 特征数据
#     # ============================================
#     print(f"\n{'─' * 80}")
#     print(f"📊 1. X (特征数据)")
#     print(f"{'─' * 80}")
#     x = data['x']
#     print(f"   Shape: {x.shape}  # (样本数, 序列长度, 特征维度)")
#     print(f"   Dtype: {x.dtype}")
#     print(f"   样本数: {x.shape[0]}")
#     print(f"   序列长度: {x.shape[1]}")
#     print(f"   特征维度: {x.shape[2]}")
#
#     # 总体统计
#     print(f"\n   总体统计:")
#     print(f"     Min:  {x.min():.6f}")
#     print(f"     Max:  {x.max():.6f}")
#     print(f"     Mean: {x.mean():.6f}")
#     print(f"     Std:  {x.std():.6f}")
#
#     # # 每个特征维度的统计
#     # print(f"\n   各特征维度统计 (前10维):")
#     # for i in range(min(10, x.shape[2])):
#     #     # 只考虑非 padding 部分
#     #     feature_slice = x[:, :, i]
#     #     valid = feature_slice[data['mask']]  # 使用 mask 过滤
#     #     print(f"     Dim {i:2d}: min={valid.min():8.4f}, max={valid.max():8.4f}, "
#     #           f"mean={valid.mean():8.4f}, std={valid.std():8.4f}")
#
#     # 展示样本
#     print(f"\n   示例样本 (前3个包的前5个特征):")
#     sample_idx = 0
#     valid_len = data['mask'][sample_idx].sum()
#     print(f"     样本 {sample_idx} (有效长度: {valid_len}):")
#     print(f"     特征维度 0-4:")
#     for pos in range(min(5, valid_len)):
#         print(f"       位置 {pos}: {x[sample_idx, pos, :5]}")
#
#     # ============================================
#     # 2. TIME - 时间数据 ⭐ 重点关注
#     # ============================================
#     print(f"\n{'─' * 80}")
#     print(f"⏱️  2. TIME (时间数据) ⭐")
#     print(f"{'─' * 80}")
#     time = data['time']
#     print(f"   Shape: {time.shape}  # (样本数, 序列长度)")
#     print(f"   Dtype: {time.dtype}")
#
#     # 使用 mask 过滤出有效时间值
#     mask = data['mask']
#     valid_times = time[mask]  # 只取有效的时间值
#
#     print(f"\n   有效时间值统计:")
#     print(f"     总数: {len(valid_times):,}")
#     print(f"     Min:  {valid_times.min():.10f}")
#     print(f"     Max:  {valid_times.max():.10f}")
#     print(f"     Mean: {valid_times.mean():.10f}")
#     print(f"     Median: {np.median(valid_times):.10f}")
#     print(f"     Std:  {valid_times.std():.10f}")
#
#     # 分位数
#     print(f"\n   分位数分布:")
#     for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
#         val = np.percentile(valid_times, p)
#         print(f"     {p:2d}%: {val:.10f}")
#
#     # 特殊值
#     zeros = (valid_times == 0).sum()
#     negatives = (valid_times < 0).sum()
#     nans = np.isnan(valid_times).sum()
#     infs = np.isinf(valid_times).sum()
#     print(f"\n   特殊值:")
#     print(f"     零值: {zeros:,} ({zeros / len(valid_times) * 100:.2f}%)")
#     print(f"     负值: {negatives:,} ({negatives / len(valid_times) * 100:.2f}%)")
#     print(f"     NaN: {nans:,}")
#     print(f"     Inf: {infs:,}")
#
#     # 🔍 判断时间数据的格式
#     print(f"\n   🔍 时间数据格式判断:")
#     if valid_times.max() < 1.0:
#         print(f"     ✅ 范围 [0, 1): 可能是归一化的原始间隔或 log1p 变换")
#         # 检查是否可能是 log1p 变换
#         print(f"     💡 如果原始间隔是 0.1-100 秒，log1p 后 ≈ 0.095-4.62")
#         print(f"     💡 如果原始间隔是 0.001-1 秒，log1p 后 ≈ 0.001-0.69")
#     elif valid_times.max() < 10:
#         print(f"     ⚡ 范围 [0, 10): 可能是原始秒数或 log1p(大间隔)")
#     elif valid_times.max() < 1000:
#         print(f"     ⚡ 范围 [0, 1000): 可能是毫秒级时间戳")
#     else:
#         print(f"     ⚠️  大范围值: 可能是原始微秒级时间戳")
#
#     # 示例：几个样本的时间序列
#     print(f"\n   示例时间序列:")
#     for i in range(min(3, time.shape[0])):
#         sample_time = time[i]
#         sample_mask = mask[i]
#         valid_time = sample_time[sample_mask]
#         valid_len = sample_mask.sum()
#
#         print(f"\n     样本 {i}: 有效长度={valid_len}")
#         print(f"       前10个值: {valid_time[:10]}")
#         print(f"       后10个值: {valid_time[-10:] if len(valid_time) > 10 else valid_time}")
#         print(f"       统计: min={valid_time.min():.6f}, max={valid_time.max():.6f}, "
#               f"mean={valid_time.mean():.6f}, std={valid_time.std():.6f}")
#
#         # 检查是否可能是 log1p 变换
#         # 对值做 expm1 反变换
#         reversed_intervals = np.expm1(valid_time)
#         print(f"       expm1 反变换: {reversed_intervals[:5]}")
#         print(f"       如果是log1p: 原始间隔范围 [{reversed_intervals.min():.6f}, "
#               f"{reversed_intervals.max():.6f}]")
#
#         # 累积时间
#         cumulative = np.cumsum(valid_time)
#         print(f"       累积值: {cumulative[:5]}")
#         print(f"       最终累积值: {cumulative[-1]:.6f}")
#
#         # 如果是 log1p 变换，累积实际时间
#         cumulative_real = np.cumsum(reversed_intervals)
#         print(f"       实际累积时间 (秒): {cumulative_real[:5]}")
#         print(f"       最终实际累积时间: {cumulative_real[-1]:.6f} 秒 = "
#               f"{cumulative_real[-1] / 60:.4f} 分钟")
#
#     # ============================================
#     # 3. MASK
#     # ============================================
#     print(f"\n{'─' * 80}")
#     print(f"🎭 3. MASK")
#     print(f"{'─' * 80}")
#     mask = data['mask']
#     print(f"   Shape: {mask.shape}")
#     print(f"   Dtype: {mask.dtype}")
#
#     # 序列长度分布
#     seq_lengths = mask.sum(axis=1)
#     print(f"\n   序列长度统计:")
#     print(f"     Min:    {seq_lengths.min()}")
#     print(f"     Max:    {seq_lengths.max()}")
#     print(f"     Mean:   {seq_lengths.mean():.2f}")
#     print(f"     Median: {np.median(seq_lengths):.0f}")
#     print(f"     Std:    {seq_lengths.std():.2f}")
#
#     # 长度分布
#     unique_lengths, length_counts = np.unique(seq_lengths, return_counts=True)
#     print(f"\n   序列长度分布 (前10):")
#     for length, count in zip(unique_lengths[:10], length_counts[:10]):
#         print(f"     长度 {length:2d}: {count:6d} 样本 ({count / len(seq_lengths) * 100:5.1f}%)")
#
#     # ============================================
#     # 4. LABELS
#     # ============================================
#     print(f"\n{'─' * 80}")
#     print(f"🏷️  4. LABELS")
#     print(f"{'─' * 80}")
#     labels = data['labels']
#     print(f"   Shape: {labels.shape}")
#     print(f"   Dtype: {labels.dtype}")
#
#     unique_labels, label_counts = np.unique(labels, return_counts=True)
#     print(f"\n   类别分布:")
#     total = len(labels)
#     for label, count in zip(unique_labels, label_counts):
#         print(f"     类别 {label}: {count:6d} 样本 ({count / total * 100:5.1f}%)")
#
#     imbalance_ratio = label_counts.max() / (label_counts.min() + 1e-8)
#     print(f"\n   不平衡比: {imbalance_ratio:.2f}:1")
#     print(f"   ⚠️  {'严重不平衡' if imbalance_ratio > 10 else '中度不平衡' if imbalance_ratio > 3 else '相对平衡'}")
#
#     # ============================================
#     # 5. FLOW_IDS
#     # ============================================
#     print(f"\n{'─' * 80}")
#     print(f"🆔 5. FLOW_IDS")
#     print(f"{'─' * 80}")
#     flow_ids = data['flow_ids']
#     print(f"   Shape: {flow_ids.shape}")
#     print(f"   Dtype: {flow_ids.dtype}")
#     print(f"   唯一流数量: {len(np.unique(flow_ids)):,}")
#     print(f"   示例 ID: {flow_ids[:10]}")
#
#     # ============================================
#     # 6. FLOW_START_TIMESTAMP
#     # ============================================
#     print(f"\n{'─' * 80}")
#     print(f"🕐 6. FLOW_START_TIMESTAMP_US")
#     print(f"{'─' * 80}")
#     timestamps = data['flow_start_timestamp_us']
#     print(f"   Shape: {timestamps.shape}")
#     print(f"   Dtype: {timestamps.dtype}")
#     print(f"   范围: {timestamps.min():,} - {timestamps.max():,}")
#
#     # 检查是否可以转换为日期
#     try:
#         from datetime import datetime
#         sample_ts = timestamps[0]
#         if sample_ts > 1e12:  # 微秒
#             dt = datetime.fromtimestamp(sample_ts / 1e6)
#             print(f"   第一个时间戳: {sample_ts:,} μs = {dt} (微秒)")
#         elif sample_ts > 1e9:  # 秒
#             dt = datetime.fromtimestamp(sample_ts)
#             print(f"   第一个时间戳: {sample_ts:,} s = {dt} (秒)")
#         else:
#             print(f"   第一个时间戳: {sample_ts:,} (相对时间)")
#     except:
#         print(f"   第一个时间戳: {timestamps[0]:,}")
#
#     # ============================================
#     # 7. SOURCE_ID
#     # ============================================
#     print(f"\n{'─' * 80}")
#     print(f"📤 7. SOURCE_ID")
#     print(f"{'─' * 80}")
#     source_ids = data['source_id']
#     print(f"   Shape: {source_ids.shape}")
#     print(f"   Dtype: {source_ids.dtype}")
#     print(f"   前10个: {source_ids[:10]}")
#
#     unique_sources, source_counts = np.unique(source_ids, return_counts=True)
#     print(f"   唯一源数量: {len(unique_sources)}")
#     print(f"   每个源的样本数: min={source_counts.min()}, max={source_counts.max()}, "
#           f"mean={source_counts.mean():.1f}")
#
#     # ============================================
#     # 8. DESTINATION_ID
#     # ============================================
#     print(f"\n{'─' * 80}")
#     print(f"📥 8. DESTINATION_ID")
#     print(f"{'─' * 80}")
#     dest_ids = data['destination_id']
#     print(f"   Shape: {dest_ids.shape}")
#     print(f"   Dtype: {dest_ids.dtype}")
#     print(f"   前10个: {dest_ids[:10]}")
#
#     unique_dests, dest_counts = np.unique(dest_ids, return_counts=True)
#     print(f"   唯一目的数量: {len(unique_dests)}")
#
#     # ============================================
#     # 9. FLOW_FEATS
#     # ============================================
#     print(f"\n{'─' * 80}")
#     print(f"🌊 9. FLOW_FEATS (流级别特征)")
#     print(f"{'─' * 80}")
#     flow_feats = data['flow_feats']
#     print(f"   Shape: {flow_feats.shape}")
#     print(f"   Dtype: {flow_feats.dtype}")
#     print(f"   特征维度: {flow_feats.shape[1]}")
#
#     print(f"\n   流特征统计:")
#     print(f"     Min:  {flow_feats.min():.6f}")
#     print(f"     Max:  {flow_feats.max():.6f}")
#     print(f"     Mean: {flow_feats.mean():.6f}")
#     print(f"     Std:  {flow_feats.std():.6f}")
#
#     print(f"\n   示例 (前3个样本的前5个流特征):")
#     for i in range(min(3, flow_feats.shape[0])):
#         print(f"     样本 {i}: {flow_feats[i, :5]}")
#
#     # ============================================
#     # 10. TIME 数据特殊分析
#     # ============================================
#     print(f"\n{'─' * 80}")
#     print(f"🔬 10. TIME 数据深入分析 (关键!)")
#     print(f"{'─' * 80}")
#
#     # 取几个样本做详细分析
#     for i in [0, 10, 100]:  # 分析3个样本
#         if i >= time.shape[0]:
#             continue
#
#         sample_time = time[i]
#         sample_mask = mask[i]
#         valid_time = sample_time[sample_mask]
#
#         print(f"\n   样本 {i} (label={labels[i]}):")
#         print(f"     原始 time 值: {valid_time[:10]}")
#
#         # 假设1: 是 log1p(间隔)
#         intervals_v1 = np.expm1(valid_time)
#         print(f"     如果是 log1p(间隔): expm1 = {intervals_v1[:10]}")
#         print(f"       -> 累积时间: {np.cumsum(intervals_v1)[:5]}")
#
#         # 假设2: 是原始间隔（秒）
#         print(f"     如果是原始间隔（秒）:")
#         print(f"       -> 累积时间: {np.cumsum(valid_time)[:5]}")
#         print(f"       -> 总时长: {np.sum(valid_time):.4f} 秒")
#
#         # 假设3: 是标准化/归一化后的值
#         norm_mean = valid_time.mean()
#         norm_std = valid_time.std()
#         print(f"     统计: mean={norm_mean:.6f}, std={norm_std:.6f}")
#         print(f"     变异系数: {norm_std / (norm_mean + 1e-8):.4f}")
#
#     data.close()
#
# print(f"\n{'=' * 80}")
# print(f"✅ 分析完成")
# print(f"{'=' * 80}")
#
#
# def summarize_time_data(tensor_dir='./tensors/precomputed'):
#     """
#     专门总结时间数据的格式
#     """
#     files = os.listdir(tensor_dir)
#     npz_files = [f for f in files if f.endswith('.npz')]
#
#     npz_file = [f for f in npz_files if 'train' in f.lower()][0]
#
#     # for npz_file in train_file:
#     data = np.load(os.path.join(tensor_dir, npz_file), allow_pickle=True)
#     # data = np.load(npz_file, allow_pickle=True)
#     time = data['time']
#     mask = data['mask']
#     valid_times = time[mask]
#
#     print(f"\n文件: {npz_file}")
#     print(f"时间值范围: [{valid_times.min():.8f}, {valid_times.max():.8f}]")
#     print(f"均值: {valid_times.mean():.8f}")
#
#     # 判断格式
#     if valid_times.max() < 0.1:
#         print("⚠️  值很小 (< 0.1), 可能是 log1p 变换或高度归一化")
#     elif valid_times.max() < 2:
#         print("💡 值适中 (< 2), 很可能是 log1p 变换后的间隔")
#     elif valid_times.max() < 100:
#         print("💡 值中等 (< 100), 可能是原始秒数或毫秒")
#     else:
#         print("💡 值较大, 可能是原始时间戳")
#
#     data.close()
#
#
# if __name__ == '__main__':
#     # 运行详细分析
#     detailed_inspect_npz()
#
#     # 专门总结时间数据
#     print("\n" + "=" * 80)
#     print("时间数据格式总结")
#     print("=" * 80)
#     summarize_time_data()
#
# # # 在 Jupyter Notebook 或 Python 脚本中运行
# #
# # import numpy as np
# # import matplotlib.pyplot as plt
# #
# # # 1. 直接加载你的数据文件
# # tensor_dir = './tensors/precomputed/'
# #
# # # 找到具体的文件
# # import os
# #
# # files = os.listdir(tensor_dir)
# # npz_files = [f for f in files if f.endswith('.npz')]
# # print("找到的 .npz 文件:", npz_files)
# #
# # # 2. 加载训练数据
# # train_file = [f for f in npz_files if 'train' in f.lower()][0]
# # data = np.load(os.path.join(tensor_dir, train_file), allow_pickle=True)
# #
# # print("\n文件中的键:")
# # for key in data.keys():
# #     arr = data[key]
# #     print(f"  {key}: shape={arr.shape}, dtype={arr.dtype}")\
# #
# #
# # # 3. 查看时间数据
# # for key in data.keys():
# #     # if 'time' in key.lower():
# #     time_data = data[key]
# #     print(f"\n{'=' * 60}")
# #     print(f"字段: {key}")
# #     print(f"{'=' * 60}")
# #     print(f"Shape: {time_data.shape}")
# #     print(f"Dtype: {time_data.dtype}")
# #
# #     # 查看前3个样本
# #     for i in range(min(3, time_data.shape[0])):
# #         sample = time_data[i]
# #         # 显示非零值
# #         # non_zero = sample[sample != 0] if np.any(sample != 0) else sample
# #         print(f"\n样本 {i}:")
# #         print(f"  前20个值: {sample[:20]}")
# #         print(f"  长度: {len(sample)}")
# #         print(f"  统计: min={sample.min():.6f}, max={sample.max():.6f}, "
# #               f"mean={sample.mean():.6f}, std={sample.std():.6f}")
# #
# #     # 总体统计
# #     all_valid = time_data #time_data[time_data != 0] if np.any(time_data != 0) else time_data.flatten()
# #     print(f"\n总体统计 (所有样本):")
# #     print(f"  Min:    {all_valid.min():.10f}")
# #     print(f"  Max:    {all_valid.max():.10f}")
# #     print(f"  Mean:   {all_valid.mean():.10f}")
# #     print(f"  Median: {np.median(all_valid):.10f}")
# #     print(f"  Std:    {all_valid.std():.10f}")
# #
# #     # 值分布
# #     unique_values = np.unique(all_valid)
# #     print(f"  唯一值数量: {len(unique_values)}")
# #
# # data.close()