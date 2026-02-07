"""
阶段3简单测试：历史追踪与自适应调整（最小依赖版本）
仅测试核心模块，不依赖网络监控等外部组件
"""
import time
import sys
from typing import Dict

# 添加项目路径
sys.path.insert(0, '/Users/hefen/Desktop/husband/20260203/CECOFramework-main_new/vllm_llama_inference_framework')

from edge.history_tracker import HistoryTracker, ExecutionRecord
from edge.adaptive_threshold import AdaptiveThresholdCalculator
from common.types import ExecutionStrategy


def test_history_tracker():
    """测试历史追踪器"""
    print("\n" + "=" * 60)
    print("测试1：HistoryTracker 基础功能")
    print("=" * 60)
    
    tracker = HistoryTracker(max_history_size=10)
    
    # 添加记录
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
    
    # 验证
    recent = tracker.get_recent_records(10)
    print(f"✓ 添加记录数量: {len(recent)}")
    assert len(recent) == 5, f"Expected 5 records, got {len(recent)}"
    
    # 测试接受率统计
    avg_ar = tracker.get_recent_acceptance_rate(
        strategy=ExecutionStrategy.SPECULATIVE_STANDARD, n=5
    )
    print(f"✓ 平均接受率: {avg_ar:.2%}")
    assert 0.82 < avg_ar < 0.90, f"Acceptance rate {avg_ar} out of range"
    
    # 测试滑动窗口
    print("\n测试滑动窗口机制...")
    tracker2 = HistoryTracker(max_history_size=5)
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
        tracker2.add_record(record)
    
    assert len(tracker2.get_recent_records(100)) == 5, "Sliding window failed"
    print("✓ 滑动窗口正常工作（保留最近5条）")
    
    # 测试策略分布统计
    print("\n测试策略分布...")
    tracker3 = HistoryTracker()
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
        tracker3.add_record(record)
    
    distribution = tracker3.get_strategy_distribution(n=6)
    print(f"✓ 策略分布: {distribution}")
    assert abs(distribution[ExecutionStrategy.EDGE_ONLY.value] - 2/6) < 0.01
    assert abs(distribution[ExecutionStrategy.SPECULATIVE_STANDARD.value] - 3/6) < 0.01
    
    print("✅ HistoryTracker 所有测试通过\n")


def test_adaptive_calculator():
    """测试自适应阈值计算器"""
    print("=" * 60)
    print("测试2：AdaptiveThresholdCalculator 自适应调整")
    print("=" * 60)
    
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
    
    # 场景1：高接受率（应该降低阈值）
    print("\n场景1：高接受率（0.95）")
    for i in range(10):
        record = ExecutionRecord(
            timestamp=time.time(),
            strategy=ExecutionStrategy.SPECULATIVE_STANDARD,
            acceptance_rate=0.95,
            latency_ms=80.0,
            edge_latency_ms=30.0,
            cloud_latency_ms=50.0,
            confidence_score=0.85,
            success=True,
            tokens_generated=50
        )
        tracker.add_record(record)
    
    current_threshold = 0.80
    new_threshold = calculator.calculate_adaptive_confidence_threshold(
        tracker, current_threshold
    )
    
    print(f"✓ 高接受率调整: {current_threshold:.3f} -> {new_threshold:.3f}")
    assert new_threshold < current_threshold, f"Expected threshold to decrease, got {new_threshold}"
    
    # 场景2：低接受率（应该提高阈值）
    print("\n场景2：低接受率（0.60）")
    tracker2 = HistoryTracker()
    for i in range(10):
        record = ExecutionRecord(
            timestamp=time.time(),
            strategy=ExecutionStrategy.SPECULATIVE_STANDARD,
            acceptance_rate=0.60,
            latency_ms=80.0,
            edge_latency_ms=30.0,
            cloud_latency_ms=50.0,
            confidence_score=0.70,
            success=True,
            tokens_generated=50
        )
        tracker2.add_record(record)
    
    new_threshold2 = calculator.calculate_adaptive_confidence_threshold(
        tracker2, current_threshold
    )
    
    print(f"✓ 低接受率调整: {current_threshold:.3f} -> {new_threshold2:.3f}")
    assert new_threshold2 > current_threshold, f"Expected threshold to increase, got {new_threshold2}"
    
    # 场景3：测试边界保护
    print("\n场景3：边界保护测试")
    tracker3 = HistoryTracker()
    for i in range(10):
        record = ExecutionRecord(
            timestamp=time.time(),
            strategy=ExecutionStrategy.SPECULATIVE_STANDARD,
            acceptance_rate=0.1,  # 极低接受率
            latency_ms=80.0,
            edge_latency_ms=30.0,
            cloud_latency_ms=50.0,
            confidence_score=0.50,
            success=True,
            tokens_generated=50
        )
        tracker3.add_record(record)
    
    threshold = 0.80
    for _ in range(5):
        threshold = calculator.calculate_adaptive_confidence_threshold(
            tracker3, threshold
        )
    
    assert threshold <= config['threshold_max'], f"Threshold {threshold} exceeds max {config['threshold_max']}"
    print(f"✓ 阈值上限保护: {threshold:.3f} <= {config['threshold_max']}")
    
    print("✅ AdaptiveThresholdCalculator 所有测试通过\n")


def test_statistics_summary():
    """测试统计摘要功能"""
    print("=" * 60)
    print("测试3：统计摘要功能")
    print("=" * 60)
    
    tracker = HistoryTracker(max_history_size=100)
    
    # 添加多种策略的记录
    for i in range(20):
        if i < 10:
            strategy = ExecutionStrategy.SPECULATIVE_STANDARD
            ar = 0.85
        else:
            strategy = ExecutionStrategy.EDGE_ONLY
            ar = 0.0
        
        record = ExecutionRecord(
            timestamp=time.time(),
            strategy=strategy,
            acceptance_rate=ar,
            latency_ms=80.0 if strategy == ExecutionStrategy.SPECULATIVE_STANDARD else 30.0,
            edge_latency_ms=30.0,
            cloud_latency_ms=50.0 if strategy == ExecutionStrategy.SPECULATIVE_STANDARD else 0.0,
            confidence_score=0.85 if strategy == ExecutionStrategy.SPECULATIVE_STANDARD else 0.70,
            success=True,
            tokens_generated=50
        )
        tracker.add_record(record)
    
    summary = tracker.get_statistics_summary()
    
    print(f"✓ 总记录数: {summary['total_records']}")
    print(f"✓ 平均接受率: {summary['recent_acceptance_rate']:.2%}")
    print(f"✓ 平均延迟: {summary['avg_latency_ms']:.1f}ms")
    print(f"✓ 成功率: {summary['success_rate']:.2%}")
    print(f"✓ 策略分布: {summary['strategy_distribution']}")
    
    assert summary['total_records'] == 20
    assert summary['success_rate'] == 1.0
    assert len(summary['by_strategy']) == 4  # 4种策略
    
    print("✅ 统计摘要功能测试通过\n")


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("阶段3核心模块测试套件")
    print("测试范围：HistoryTracker + AdaptiveThresholdCalculator")
    print("=" * 60)
    
    try:
        test_history_tracker()
        test_adaptive_calculator()
        test_statistics_summary()
        
        print("\n" + "=" * 60)
        print("✅ 所有核心模块测试通过！")
        print("=" * 60)
        print("\n提示：完整集成测试需要在有 psutil、aiohttp 等依赖的环境中运行")
        return 0
    
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
