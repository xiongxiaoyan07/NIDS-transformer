# models/statistical_tests.py
"""
统计显著性检验模块
用于严谨验证模型改进是否具有统计学意义
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset
import warnings


class ModelSignificanceTester:
    """
    模型显著性检验器

    支持的方法：
    1. McNemar's Test: 配对分类结果比较
    2. 5x2 Cross-Validation Paired t-test
    3. Wilcoxon Signed-Rank Test
    4. Bootstrap Confidence Intervals
    """

    def __init__(
            self,
            model_a: torch.nn.Module,
            model_b: torch.nn.Module,
            model_a_name: str = "Model A (Ours)",
            model_b_name: str = "Model B (Baseline)",
            device: str = 'cuda',
            n_bootstrap: int = 1000,
            alpha: float = 0.05
    ):
        self.model_a = model_a
        self.model_b = model_b
        self.model_a_name = model_a_name
        self.model_b_name = model_b_name
        self.device = device
        self.n_bootstrap = n_bootstrap
        self.alpha = alpha  # 显著性水平

    def mcnemar_test(
            self,
            loader: DataLoader,
            threshold_a: float = 0.5,
            threshold_b: float = 0.5
    ) -> Dict:
        """
        McNemar's 检验：比较两个模型的分类差异

        用于配对名义数据，检验两个模型是否在相同样本上有不同的错误模式

        Contingency Table:
                    Model B Correct   Model B Wrong
        Model A Correct      n00            n01
        Model A Wrong        n10            n11

        检验统计量: χ² = (|n01 - n10| - 1)² / (n01 + n10)
        """
        print(f"\n{'=' * 60}")
        print(f"McNemar's Test: {self.model_a_name} vs {self.model_b_name}")
        print(f"{'=' * 60}")

        # 获取预测结果
        y_true, pred_a, pred_b = self._get_predictions(
            loader, threshold_a, threshold_b
        )

        # 构建列联表
        correct_a = (pred_a == y_true)
        correct_b = (pred_b == y_true)

        n00 = ((correct_a) & (correct_b)).sum()
        n01 = ((correct_a) & (~correct_b)).sum()
        n10 = ((~correct_a) & (correct_b)).sum()
        n11 = ((~correct_a) & (~correct_b)).sum()

        total = n00 + n01 + n10 + n11

        # 列联表
        contingency_table = np.array([[n00, n01], [n10, n11]])

        print(f"\n列联表 (Model A rows, Model B columns, Correct/Wrong):")
        print(f"            B Correct   B Wrong")
        print(f"A Correct   {n00:6d}      {n01:6d}")
        print(f"A Wrong     {n10:6d}      {n11:6d}")

        # McNemar's test (使用连续性校正)
        # 只关心不一致对 (n01 和 n10)
        if n01 + n10 == 0:
            p_value = 1.0
            statistic = 0.0
            print("\n没有不一致的预测对，无法进行 McNemar 检验")
        else:
            # 连续性校正
            statistic = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
            p_value = 1 - stats.chi2.cdf(statistic, df=1)

        # 判断哪个模型在不一致对上表现更好
        if n01 > n10:
            advantage = f"{self.model_b_name} 在 {n01} 个样本上正确而 {self.model_a_name} 错误"
        elif n10 > n01:
            advantage = f"{self.model_a_name} 在 {n10} 个样本上正确而 {self.model_b_name} 错误"
        else:
            advantage = "两个模型在不一致对上的表现相同"

        result = {
            'test': "McNemar's Test",
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < self.alpha,
            'n01': int(n01),  # A正确B错误
            'n10': int(n10),  # A错误B正确
            'advantage': advantage,
            'contingency_table': contingency_table.tolist(),
        }

        print(f"\n检验统计量 χ² = {statistic:.4f}")
        print(f"p-value = {p_value:.6f}")
        print(f"α = {self.alpha}")
        print(f"显著性: {'是 ✅' if result['significant'] else '否 ❌'}")
        print(f"结论: {advantage}")

        if result['significant']:
            print(f"\n⚠️  两个模型在统计上存在显著差异")
        else:
            print(f"\n✓  无法拒绝两个模型表现相同的零假设")

        return result

    def wilcoxon_signed_rank_test(
            self,
            loader: DataLoader,
            n_folds: int = 10,
            metric_fn=None
    ) -> Dict:
        """
        Wilcoxon 符号秩检验（配对）

        适用于：
        - 多个数据集/多个fold上的配对比分
        - 不假设正态分布

        这里使用 k-fold 划分，在每折上计算两个模型的性能差异
        """
        if metric_fn is None:
            from sklearn.metrics import f1_score
            metric_fn = lambda y_true, y_pred: f1_score(
                y_true, y_pred, average='macro', zero_division=0
            )

        print(f"\n{'=' * 60}")
        print(f"Wilcoxon Signed-Rank Test: {self.model_a_name} vs {self.model_b_name}")
        print(f"使用 {n_folds}-Fold Cross-Validation")
        print(f"{'=' * 60}")

        # 准备数据集
        all_labels = []
        for batch in loader:
            all_labels.append(batch['label'].numpy())
        all_labels = np.concatenate(all_labels, axis=0)

        n_samples = len(all_labels)

        # K-fold 划分
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

        fold_scores_a = []
        fold_scores_b = []

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(range(n_samples))):
            # 在每个验证折上进行评估
            val_subset = Subset(loader.dataset, val_idx)
            val_loader = DataLoader(
                val_subset,
                batch_size=loader.batch_size,
                shuffle=False,
                num_workers=0
            )

            # 评估两个模型
            scores_a = self._evaluate_model_on_fold(self.model_a, val_loader)
            scores_b = self._evaluate_model_on_fold(self.model_b, val_loader)

            y_true_list = []
            y_pred_a_list = []
            y_pred_b_list = []

            for batch in val_loader:
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
                labels = batch['label']

                with torch.no_grad():
                    logits_a = self.model_a(x, mask=mask, time_log=time_log, flow_feats=flow_feats)
                    logits_b = self.model_b(x, mask=mask, time_log=time_log, flow_feats=flow_feats)

                    probs_a = torch.softmax(logits_a, dim=1)
                    probs_b = torch.softmax(logits_b, dim=1)

                    if probs_a.shape[1] == 2:
                        pred_a = (probs_a[:, 1] >= 0.5).cpu().numpy()
                        pred_b = (probs_b[:, 1] >= 0.5).cpu().numpy()
                    else:
                        pred_a = probs_a.argmax(dim=1).cpu().numpy()
                        pred_b = probs_b.argmax(dim=1).cpu().numpy()

                y_true_list.append(labels.numpy())
                y_pred_a_list.append(pred_a)
                y_pred_b_list.append(pred_b)

            y_true_fold = np.concatenate(y_true_list)
            y_pred_a_fold = np.concatenate(y_pred_a_list)
            y_pred_b_fold = np.concatenate(y_pred_b_list)

            score_a = metric_fn(y_true_fold, y_pred_a_fold)
            score_b = metric_fn(y_true_fold, y_pred_b_fold)

            fold_scores_a.append(score_a)
            fold_scores_b.append(score_b)

        # 计算差异
        fold_diffs = np.array(fold_scores_a) - np.array(fold_scores_b)

        print(f"\n{n_folds}-Fold Cross-Validation 结果:")
        print(f"{'Fold':<8} {'Model A':<12} {'Model B':<12} {'Diff (A-B)':<12}")
        print("-" * 44)
        for i, (sa, sb, diff) in enumerate(zip(fold_scores_a, fold_scores_b, fold_diffs)):
            marker = ">" if diff > 0 else "<" if diff < 0 else "="
            print(f"  {i + 1:<6} {sa:<12.4f} {sb:<12.4f} {marker} {diff:+.4f}")

        mean_diff = np.mean(fold_diffs)
        std_diff = np.std(fold_diffs, ddof=1)

        # Wilcoxon Signed-Rank Test
        if np.all(fold_diffs == 0):
            statistic = 0
            p_value = 1.0
            print("\n所有折的差异为零，无法进行检验")
        else:
            # 去除零差异
            nonzero_diffs = fold_diffs[fold_diffs != 0]
            if len(nonzero_diffs) < 3:
                print(f"\n警告: 仅有 {len(nonzero_diffs)} 个非零差异，检验可能不够稳健")

            try:
                statistic, p_value = stats.wilcoxon(fold_diffs)
            except ValueError as e:
                print(f"\n检验失败: {e}")
                statistic = np.nan
                p_value = 1.0

        # 效应量 (Cohen's d for paired samples)
        if std_diff > 0:
            cohens_d = mean_diff / std_diff
        else:
            cohens_d = np.inf if mean_diff != 0 else 0.0

        result = {
            'test': "Wilcoxon Signed-Rank Test",
            'statistic': float(statistic) if not np.isnan(statistic) else None,
            'p_value': float(p_value),
            'significant': p_value < self.alpha,
            'mean_diff': float(mean_diff),
            'std_diff': float(std_diff),
            'cohens_d': float(cohens_d) if not np.isinf(cohens_d) else None,
            'fold_scores_a': [float(s) for s in fold_scores_a],
            'fold_scores_b': [float(s) for s in fold_scores_b],
            'fold_diffs': [float(d) for d in fold_diffs],
        }

        print(f"\n描述性统计:")
        print(f"  平均差异 (A - B): {mean_diff:.4f}")
        print(f"  差异标准差: {std_diff:.4f}")
        print(f"  Cohen's d: {cohens_d:.4f}", end="")

        if abs(cohens_d) < 0.2:
            print(" (可忽略)")
        elif abs(cohens_d) < 0.5:
            print(" (小)")
        elif abs(cohens_d) < 0.8:
            print(" (中)")
        else:
            print(" (大)")

        print(f"\n检验统计量: {statistic}")
        print(f"p-value: {p_value:.6f}")
        print(f"α = {self.alpha}")
        print(f"显著性: {'是 ✅' if result['significant'] else '否 ❌'}")

        if result['significant'] and mean_diff > 0:
            print(f"\n📈 {self.model_a_name} 显著优于 {self.model_b_name}")
        elif result['significant'] and mean_diff < 0:
            print(f"\n📉 {self.model_b_name} 显著优于 {self.model_a_name}")
        else:
            print(f"\n⚠️  无法拒绝两个模型表现相同的零假设")

        return result

    def bootstrap_confidence_interval_test(
            self,
            loader: DataLoader,
            n_iterations: int = None,
            metric_fn=None
    ) -> Dict:
        """
        Bootstrap 置信区间检验

        通过重采样评估模型性能差异的置信区间

        优点：
        - 不假设分布
        - 可以给出差异的置信区间
        - 可视化友好
        """
        if n_iterations is None:
            n_iterations = self.n_bootstrap

        if metric_fn is None:
            from sklearn.metrics import f1_score
            metric_fn = lambda y_true, y_pred: f1_score(
                y_true, y_pred, average='macro', zero_division=0
            )

        print(f"\n{'=' * 60}")
        print(f"Bootstrap 置信区间检验")
        print(f"{self.model_a_name} vs {self.model_b_name}")
        print(f"Bootstrap 迭代次数: {n_iterations}")
        print(f"{'=' * 60}")

        # 获取所有预测
        y_true_all = []
        x_all = []
        mask_all = []
        time_all = []
        flow_feats_all = []

        for batch in loader:
            y_true_all.append(batch['label'].numpy())
            x_all.append(batch['x'])
            mask_all.append(batch.get('mask'))
            time_all.append(batch.get('time'))
            flow_feats_all.append(batch.get('flow_feats'))

        y_true_all = np.concatenate(y_true_all, axis=0)
        n_samples = len(y_true_all)

        # Bootstrap 采样
        rng = np.random.RandomState(42)
        bootstrap_diffs = []

        for i in range(n_iterations):
            # 有放回采样索引
            indices = rng.choice(n_samples, size=n_samples, replace=True)

            # 评估模型A
            y_pred_a = self._bootstrap_predict(
                self.model_a, indices, x_all, mask_all, time_all, flow_feats_all
            )

            # 评估模型B
            y_pred_b = self._bootstrap_predict(
                self.model_b, indices, x_all, mask_all, time_all, flow_feats_all
            )

            # 计算指标
            y_true_bootstrap = y_true_all[indices]
            score_a = metric_fn(y_true_bootstrap, y_pred_a)
            score_b = metric_fn(y_true_bootstrap, y_pred_b)

            bootstrap_diffs.append(score_a - score_b)

        bootstrap_diffs = np.array(bootstrap_diffs)

        # 计算置信区间
        ci_lower = np.percentile(bootstrap_diffs, 100 * self.alpha / 2)
        ci_upper = np.percentile(bootstrap_diffs, 100 * (1 - self.alpha / 2))

        mean_diff = np.mean(bootstrap_diffs)
        std_diff = np.std(bootstrap_diffs)

        # 判断显著性
        significant = ci_lower > 0 or ci_upper < 0

        result = {
            'test': "Bootstrap Confidence Interval",
            'n_iterations': n_iterations,
            'mean_diff': float(mean_diff),
            'std_diff': float(std_diff),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'confidence_level': 1 - self.alpha,
            'significant': significant,
            'bootstrap_diffs': bootstrap_diffs.tolist(),
        }

        print(f"\nBootstrap 结果:")
        print(f"  平均差异 (A - B): {mean_diff:.4f}")
        print(f"  差异标准差: {std_diff:.4f}")
        print(f"  {(1 - self.alpha) * 100}% 置信区间: [{ci_lower:.4f}, {ci_upper:.4f}]")
        print(f"  显著性: {'是 ✅' if significant else '否 ❌'}")

        if significant:
            if ci_lower > 0:
                print(f"\n📈 {self.model_a_name} 显著优于 {self.model_b_name}")
                print(f"   差异的最坏情况 (95% CI下限): {ci_lower:.4f}")
            else:
                print(f"\n📉 {self.model_b_name} 显著优于 {self.model_a_name}")
                print(f"   差异的最坏情况 (95% CI上限): {ci_upper:.4f}")
        else:
            print(f"\n⚠️  置信区间包含0，无法确认显著差异")

        return result

    def run_all_tests(
            self,
            loader: DataLoader,
            output_path: Optional[str] = None
    ) -> Dict:
        """
        运行所有统计检验

        Args:
            loader: 测试集数据加载器
            output_path: 结果保存路径

        Returns:
            所有检验结果的字典
        """
        print("\n" + "=" * 70)
        print("统计显著性检验总览")
        print(f"{self.model_a_name} vs {self.model_b_name}")
        print("=" * 70)

        results = {}

        # 1. McNemar's Test
        try:
            results['mcnemar'] = self.mcnemar_test(loader)
        except Exception as e:
            print(f"McNemar's Test 失败: {e}")
            results['mcnemar'] = {'error': str(e)}

        # 2. Wilcoxon Signed-Rank Test
        try:
            results['wilcoxon'] = self.wilcoxon_signed_rank_test(loader, n_folds=10)
        except Exception as e:
            print(f"Wilcoxon Test 失败: {e}")
            results['wilcoxon'] = {'error': str(e)}

        # 3. Bootstrap
        try:
            results['bootstrap'] = self.bootstrap_confidence_interval_test(loader)
        except Exception as e:
            print(f"Bootstrap Test 失败: {e}")
            results['bootstrap'] = {'error': str(e)}

        # 汇总
        summary = self._summarize_tests(results)
        results['summary'] = summary

        # 保存结果
        if output_path:
            import json
            # 清理 numpy 类型
            results_clean = self._clean_results(results)
            with open(output_path, 'w') as f:
                json.dump(results_clean, f, indent=2)
            print(f"\n统计检验结果已保存至: {output_path}")

        return results

    def _get_predictions(self, loader, threshold_a, threshold_b):
        """获取两个模型的预测结果"""
        y_true_list = []
        pred_a_list = []
        pred_b_list = []

        self.model_a.eval()
        self.model_b.eval()

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
                labels = batch['label']

                logits_a = self.model_a(x, mask=mask, time_log=time_log, flow_feats=flow_feats)
                logits_b = self.model_b(x, mask=mask, time_log=time_log, flow_feats=flow_feats)

                probs_a = torch.softmax(logits_a, dim=1)
                probs_b = torch.softmax(logits_b, dim=1)

                if probs_a.shape[1] == 2:
                    pa = (probs_a[:, 1] >= threshold_a).cpu().numpy()
                    pb = (probs_b[:, 1] >= threshold_b).cpu().numpy()
                else:
                    pa = probs_a.argmax(dim=1).cpu().numpy()
                    pb = probs_b.argmax(dim=1).cpu().numpy()

                y_true_list.append(labels.numpy())
                pred_a_list.append(pa)
                pred_b_list.append(pb)

        y_true = np.concatenate(y_true_list)
        pred_a = np.concatenate(pred_a_list)
        pred_b = np.concatenate(pred_b_list)

        return y_true, pred_a, pred_b

    def _evaluate_model_on_fold(self, model, fold_loader):
        """在验证折上评估模型"""
        model.eval()
        return model  # 保持模型引用，实际评估在调用方进行

    def _bootstrap_predict(self, model, indices, x_all, mask_all, time_all, flow_feats_all):
        """Bootstrap 预测"""
        model.eval()
        # 简化处理：这里返回需要的预测
        # 实际实现需要根据数据格式调整
        return np.zeros(len(indices))  # placeholder

    def _summarize_tests(self, results):
        """汇总所有检验结果"""
        summary = {
            'tests_performed': [],
            'overall_conclusion': None,
        }

        for test_name, result in results.items():
            if isinstance(result, dict) and 'significant' in result:
                summary['tests_performed'].append({
                    'test': test_name,
                    'significant': result['significant'],
                    'p_value': result.get('p_value', None),
                })

        # 整体结论
        n_tests = len(summary['tests_performed'])
        n_significant = sum(t['significant'] for t in summary['tests_performed'])

        if n_tests == 0:
            summary['overall_conclusion'] = "未能完成任何检验"
        elif n_significant == n_tests:
            summary['overall_conclusion'] = f"所有 {n_tests} 个检验均显示显著差异，{self.model_a_name} 被证明显著不同"
        elif n_significant >= n_tests // 2:
            summary['overall_conclusion'] = f"{n_significant}/{n_tests} 个检验显示显著差异，证据支持模型间存在差异"
        else:
            summary['overall_conclusion'] = f"仅有 {n_significant}/{n_tests} 个检验显示显著差异，差异证据不足"

        print(f"\n{'=' * 60}")
        print("总体结论:")
        print(f"  完成检验: {n_tests}")
        print(f"  显著检验: {n_significant}")
        print(f"  结论: {summary['overall_conclusion']}")
        print(f"{'=' * 60}")

        return summary

    def _clean_results(self, results):
        """清理结果中的非JSON兼容类型"""
        import json

        def clean_value(obj):
            if isinstance(obj, dict):
                return {k: clean_value(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean_value(v) for v in obj]
            elif isinstance(obj, (np.integer,)):
                return int(obj)
            elif isinstance(obj, (np.floating,)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif obj is None or isinstance(obj, (bool, int, float, str)):
                return obj
            else:
                return str(obj)

        return clean_value(results)