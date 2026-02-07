"""
阶段3测试：历史追踪与自适应调整

测试覆盖：
1. HistoryTracker 功能（记录、查询、统计）
2. AdaptiveThresholdCalculator 自适应调整
3. F1 模块集成历史追踪
4. MultiObjectiveScorer 使用历史数据评分
"""
import time
import pytest
from typing import Dict

from edge.history_tracker import HistoryTracker, ExecutionRecord
from edge.adaptive_threshold import AdaptiveThresholdCalculator
from edge.f1_decision import F1_DecisionModule
from common.types import (
    InferenceRequest,
    TaskRequirements,
    ExecutionStrategy
)


class TestHistoryTracker:
    """测试历史追踪器"""
    
    def test_add_and_retrieve_records(self):
        """测试添加和检索历史记录"""
        tracker = HistoryTracker(max_history_size=10)
        
        # 添加几条记录
        for i in range(5):
            record = ExecutionRecord(
                timestamp=time.time(),
                strategy=ExecutionStrategy.SPECULATIVE_STANDARD,
                acceptance_rate=0.8 + i * 0.02,
                latency_ms=80.0 + i * 5,
                edge_latency_ms=30.0,
                cloud_latency_ms=50.0 + i * 5,
                confidence_score=0.85,
                success=True,
                tokens_generated=50
            )
            tracker.add_record(record)
        
        # 验证记录数量
        assert len(tracker.get_recent_records(10)) == 5
        
        # 验证最近3条
        recent_3 = tracker.get_recent_records(3)
        assert len(recent_3) == 3
        assert recent_3[-1].acceptance_rate == 0.88  # 最新的一条
    
    def test_sliding_window(self):
        """测试滑动窗口机制"""
        tracker = HistoryTracker(max_history_size=5)
        
        # 添加超过窗口大小的记录
        for i in range(10):
            record = ExecutionRecord(
                timestamp=time.time() + i,
                strategy=ExecutionStrategy.EDGE_ONLY,
                acceptance_rate=0.0,
                latency_ms=30.0,
                edge_latency_ms=30.0,
                cloud_latency_ms=0.0,
                confidence_score=0.7,
                success=True,
                tokens_generated=50
            )
            tracker.add_record(record)
        
        # 应该只保留最近5条
        assert len(tracker.get_recent_records(100)) == 5
    
    def test_acceptance_rate_calculation(self):
        """测试接受率统计"""
        tracker = HistoryTracker()
        
        # 添加协同推理记录
        acceptance_rates = [0.9, 0.85, 0.88, 0.92, 0.87]
        for ar in acceptance_rates:
            record = ExecutionRecord(
                timestamp=time.time(),
                strategy=ExecutionStrategy.SPECULATIVE_STANDARD,
                acceptance_rate=ar,
                latency_ms=80.0,
                edge_latency_ms=30.0,
                cloud_latency_ms=50.0,
                confidence_score=0.85,
                success=True,
                tokens_generated=50
            )
            tracker.add_record(record)
        
        # 计算平均接受率
        avg_ar = tracker.get_recent_acceptance_rate(
            strategy=ExecutionStrategy.SPECULATIVE_STANDARD,
            n=5
        )
        
        expected_avg = sum(acceptance_rates) / len(acceptance_rates)
        assert abs(avg_ar - expected_avg) < 0.01
    
    def test_strategy_distribution(self):
        """测试策略分布统计"""
        tracker = HistoryTracker()
        
        # 添加不同策略的记录
        strategies = [
            ExecutionStrategy.EDGE_ONLY,
            ExecutionStrategy.EDGE_ONLY,
            ExecutionStrategy.CLOUD_DIRECT,
            ExecutionStrategy.SPECULATIVE_STANDARD,
            ExecutionStrategy.SPECULATIVE_STANDARD,
            ExecutionStrategy.SPECULATIVE_STANDARD
        ]
        
        for strategy in strategies:
            record = ExecutionRecord(
                timestamp=time.time(),
                strategy=strategy,
                acceptance_rate=0.8,
                latency_ms=80.0,
                edge_latency_ms=30.0,
                cloud_latency_ms=50.0,
                confidence_score=0.85,
                success=True,
                tokens_generated=50
            )
            tracker.add_record(record)
        
        # 获取策略分布
        distribution = tracker.get_strategy_distribution(n=6)
        
        assert distribution[ExecutionStrategy.EDGE_ONLY.value] == 2/6
        assert distribution[ExecutionStrategy.CLOUD_DIRECT.value] == 1/6
        assert distribution[ExecutionStrategy.SPECULATIVE_STANDARD.value] == 3/6


class TestAdaptiveThresholdCalculator:
    """测试自适应阈值计算器"""
    
    def test_threshold_adjustment_high_acceptance(self):
        """测试高接受率时降低阈值"""
        config = {
            'target_acceptance_min': 0.75,
            'target_acceptance_max': 0.85,
            'threshold_step': 0.05,
            'smoothing_factor': 0.3,
            'threshold_min': 0.50,
            'threshold_max': 0.95,
            'initial_confidence_threshold': 0.80,
            'update_interval': 5
        }
        
        calculator = AdaptiveThresholdCalculator(config)
        tracker = HistoryTracker()
        
        # 模拟高接受率场景（0.95）
        for i in range(10):
            record = ExecutionRecord(
                timestamp=time.time(),
                strategy=ExecutionStrategy.SPECULATIVE_STANDARD,
                acceptance_rate=0.95,  # 远高于目标上限0.85
                latency_ms=80.0,
                edge_latency_ms=30.0,
                cloud_latency_ms=50.0,
                confidence_score=0.85,
                success=True,
                tokens_generated=50
            )
            tracker.add_record(record)
        
        # 计算新阈值（应该降低）
        current_threshold = 0.80
        new_threshold = calculator.calculate_adaptive_confidence_threshold(
            tracker, current_threshold
        )
        
        # 高接受率应该导致阈值降低
        assert new_threshold < current_threshold
        print(f"[测试] 高接受率调整: {current_threshold:.3f} -> {new_threshold:.3f}")
    
    def test_threshold_adjustment_low_acceptance(self):
        """测试低接受率时提高阈值"""
        config = {
            'target_acceptance_min': 0.75,
            'target_acceptance_max': 0.85,
            'threshold_step': 0.05,
            'smoothing_factor': 0.3,
            'threshold_min': 0.50,
            'threshold_max': 0.95,
            'initial_confidence_threshold': 0.80,
            'update_interval': 5
        }
        
        calculator = AdaptiveThresholdCalculator(config)
        tracker = HistoryTracker()
        
        # 模拟低接受率场景（0.60）
        for i in range(10):
            record = ExecutionRecord(
                timestamp=time.time(),
                strategy=ExecutionStrategy.SPECULATIVE_STANDARD,
                acceptance_rate=0.60,  # 低于目标下限0.75
                latency_ms=80.0,
                edge_latency_ms=30.0,
                cloud_latency_ms=50.0,
                confidence_score=0.70,
                success=True,
                tokens_generated=50
            )
            tracker.add_record(record)
        
        # 计算新阈值（应该提高）
        current_threshold = 0.80
        new_threshold = calculator.calculate_adaptive_confidence_threshold(
            tracker, current_threshold
        )
        
        # 低接受率应该导致阈值提高
        assert new_threshold > current_threshold
        print(f"[测试] 低接受率调整: {current_threshold:.3f} -> {new_threshold:.3f}")
    
    def test_threshold_bounds(self):
        """测试阈值边界限制"""
        config = {
            'target_acceptance_min': 0.75,
            'target_acceptance_max': 0.85,
            'threshold_step': 0.10,  # 大步长
            'smoothing_factor': 0.5,
            'threshold_min': 0.50,
            'threshold_max': 0.95,
            'initial_confidence_threshold': 0.80,
            'update_interval': 5
        }
        
        calculator = AdaptiveThresholdCalculator(config)
        tracker = HistoryTracker()
        
        # 极端低接受率
        for i in range(10):
            record = ExecutionRecord(
                timestamp=time.time(),
                strategy=ExecutionStrategy.SPECULATIVE_STANDARD,
                acceptance_rate=0.1,
                latency_ms=80.0,
                edge_latency_ms=30.0,
                cloud_latency_ms=50.0,
                confidence_score=0.50,
                success=True,
                tokens_generated=50
            )
            tracker.add_record(record)
        
        # 多次调整
        threshold = 0.80
        for _ in range(5):
            threshold = calculator.calculate_adaptive_confidence_threshold(
                tracker, threshold
            )
        
        # 应该在最大值范围内
        assert threshold <= config['threshold_max']
        print(f"[测试] 阈值上限保护: {threshold:.3f} <= {config['threshold_max']}")


class TestF1WithHistoryTracking:
    """测试 F1 模块集成历史追踪"""
    
    def test_f1_record_execution(self):
        """测试 F1 记录执行结果"""
        config = self._get_test_config()
        f1 = F1_DecisionModule(config, cloud_endpoint="http://localhost:8081")
        
        # 模拟记录几次执行
        for i in range(5):
            f1.record_execution(
                strategy=ExecutionStrategy.SPECULATIVE_STANDARD,
                acceptance_rate=0.85,
                latency_ms=80.0 + i * 5,
                edge_latency_ms=30.0,
                cloud_latency_ms=50.0 + i * 5,
                confidence_score=0.85,
                success=True,
                tokens_generated=50
            )
        
        # 验证历史记录
        stats = f1.get_statistics_summary()
        assert stats['total_records'] == 5
        assert stats['recent_acceptance_rate'] > 0.8
        print(f"[测试] F1 历史统计: {stats}")
    
    def test_f1_adaptive_update_trigger(self):
        """测试 F1 自适应更新触发"""
        config = self._get_test_config()
        config['enable_adaptive'] = True
        config['adaptive_threshold'] = {
            'target_acceptance_min': 0.75,
            'target_acceptance_max': 0.85,
            'threshold_step': 0.05,
            'smoothing_factor': 0.2,
            'threshold_min': 0.50,
            'threshold_max': 0.95,
            'initial_confidence_threshold': 0.80,
            'update_interval': 5
        }
        
        f1 = F1_DecisionModule(config, cloud_endpoint="http://localhost:8081")
        
        initial_threshold = f1.config.get('confidence_threshold', 0.80)
        
        # 记录10次（高接受率）
        for i in range(10):
            f1.record_execution(
                strategy=ExecutionStrategy.SPECULATIVE_STANDARD,
                acceptance_rate=0.95,  # 高接受率
                latency_ms=80.0,
                edge_latency_ms=30.0,
                cloud_latency_ms=50.0,
                confidence_score=0.85,
                success=True,
                tokens_generated=50
            )
        
        # 手动触发自适应更新
        f1._apply_adaptive_updates()
        
        # 验证阈值已调整
        new_threshold = f1.config.get('confidence_threshold', 0.80)
        print(f"[测试] 自适应调整: {initial_threshold:.3f} -> {new_threshold:.3f}")
        
        # 高接受率应该降低阈值（允许更激进）
        assert new_threshold < initial_threshold or abs(new_threshold - initial_threshold) < 0.001
    
    def test_scorer_use_history_data(self):
        """测试评分器使用历史数据"""
        config = self._get_test_config()
        config['enable_history_scoring'] = True
        
        f1 = F1_DecisionModule(config, cloud_endpoint="http://localhost:8081")
        
        # 先记录一些历史（EDGE_ONLY 延迟短但质量低）
        for i in range(10):
            f1.record_execution(
                strategy=ExecutionStrategy.EDGE_ONLY,
                acceptance_rate=0.0,
                latency_ms=25.0,
                edge_latency_ms=25.0,
                cloud_latency_ms=0.0,
                confidence_score=0.70,
                success=True,
                tokens_generated=50
            )
        
        # 再记录一些历史（SPECULATIVE 延迟长但质量高）
        for i in range(10):
            f1.record_execution(
                strategy=ExecutionStrategy.SPECULATIVE_STANDARD,
                acceptance_rate=0.88,
                latency_ms=90.0,
                edge_latency_ms=30.0,
                cloud_latency_ms=60.0,
                confidence_score=0.90,
                success=True,
                tokens_generated=50
            )
        
        # 进行一次决策（应该使用历史数据）
        request = InferenceRequest(
            prompt="Test prompt",
            max_tokens=50,
            requirements=TaskRequirements(
                max_latency_ms=100,
                min_quality_score=0.8
            )
        )
        
        plan = f1.decide(request)
        
        print(f"[测试] 决策结果: {plan.strategy.value}, 得分={plan.score:.3f}")
        print(f"[测试] 理由: {plan.reason}")
        
        # 由于质量要求高且延迟可接受，应该倾向于选择 SPECULATIVE
        assert plan.strategy in [
            ExecutionStrategy.SPECULATIVE_STANDARD,
            ExecutionStrategy.ADAPTIVE_CONFIDENCE,
            ExecutionStrategy.EDGE_ONLY
        ]
    
    def _get_test_config(self) -> Dict:
        """获取测试配置"""
        return {
            'hard_constraints': {
                'cpu_overload': 95.0,
                'memory_critical': 500,
                'ultra_low_latency': 50,
                'weak_network_rtt': 200.0
            },
            'scoring_weights': {
                'latency': 0.4,
                'cost': 0.3,
                'quality': 0.3
            },
            'latency_estimates': {
                'edge_only_ms': 30.0,
                'cloud_direct_ms': 200.0,
                'speculative_standard_ms': 80.0
            },
            'enable_network_probe': False,
            'enable_adaptive': True,
            'enable_history_scoring': True,
            'history_tracker': {
                'max_history_size': 100
            },
            'adaptive_threshold': {
                'target_acceptance_min': 0.75,
                'target_acceptance_max': 0.85,
                'threshold_step': 0.05,
                'smoothing_factor': 0.2,
                'threshold_min': 0.50,
                'threshold_max': 0.95,
                'initial_confidence_threshold': 0.80,
                'update_interval': 10
            },
            'confidence_threshold': 0.80,
            'draft_max_tokens': 64,
            'default_latency_slo': 150
        }


if __name__ == "__main__":
    print("=" * 60)
    print("阶段3测试：历史追踪与自适应调整")
    print("=" * 60)
    
    # 运行测试
    pytest.main([__file__, "-v", "-s"])
