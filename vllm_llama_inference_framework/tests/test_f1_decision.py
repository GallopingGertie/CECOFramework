"""
F1 决策模块单元测试
"""
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.types import (
    InferenceRequest,
    TaskRequirements,
    SystemStats,
    DecisionContext,
    ExecutionStrategy
)
from edge.f1_decision import F1_DecisionModule
from edge.decision_engine import HardConstraintChecker, MultiObjectiveScorer


def test_hard_constraint_cpu_overload():
    """测试 CPU 过载硬约束"""
    print("\n=== 测试: CPU 过载硬约束 ===")
    
    context = DecisionContext(
        request=InferenceRequest(prompt="test"),
        system_state=SystemStats(cpu_usage=98.0, memory_available_mb=2000),
        task_requirements=TaskRequirements()
    )
    
    checker = HardConstraintChecker(config={'cpu_overload': 95.0})
    decision = checker.check(context)
    
    assert decision is not None, "应该触发硬约束"
    assert decision.strategy == ExecutionStrategy.CLOUD_DIRECT, "应该选择 CLOUD_DIRECT"
    assert "CPU过载" in decision.reason, "理由应包含 CPU过载"
    
    print(f"✅ 决策: {decision.strategy.value}")
    print(f"✅ 理由: {decision.reason}")


def test_hard_constraint_ultra_low_latency():
    """测试超低延迟硬约束"""
    print("\n=== 测试: 超低延迟硬约束 ===")
    
    context = DecisionContext(
        request=InferenceRequest(prompt="test"),
        system_state=SystemStats(cpu_usage=50.0, memory_available_mb=2000),
        task_requirements=TaskRequirements(max_latency_ms=30)
    )
    
    checker = HardConstraintChecker(config={'ultra_low_latency': 50})
    decision = checker.check(context)
    
    assert decision is not None, "应该触发硬约束"
    assert decision.strategy == ExecutionStrategy.EDGE_ONLY, "应该选择 EDGE_ONLY"
    assert "超低延迟" in decision.reason, "理由应包含超低延迟"
    
    print(f"✅ 决策: {decision.strategy.value}")
    print(f"✅ 理由: {decision.reason}")


def test_hard_constraint_privacy():
    """测试隐私约束"""
    print("\n=== 测试: 隐私约束 ===")
    
    context = DecisionContext(
        request=InferenceRequest(prompt="test"),
        system_state=SystemStats(cpu_usage=50.0, memory_available_mb=2000),
        task_requirements=TaskRequirements(privacy_level=2)  # 绝密级别
    )
    
    checker = HardConstraintChecker(config={})
    decision = checker.check(context)
    
    assert decision is not None, "应该触发硬约束"
    assert decision.strategy == ExecutionStrategy.EDGE_ONLY, "应该选择 EDGE_ONLY"
    assert "隐私" in decision.reason, "理由应包含隐私"
    
    print(f"✅ 决策: {decision.strategy.value}")
    print(f"✅ 理由: {decision.reason}")


def test_scoring_latency_exceeds_slo():
    """测试延迟超过 SLO 时得分为 0"""
    print("\n=== 测试: 延迟超过 SLO ===")
    
    context = DecisionContext(
        request=InferenceRequest(prompt="test"),system_state=SystemStats(cpu_usage=50.0, memory_available_mb=2000),
        task_requirements=TaskRequirements(max_latency_ms=50)  # 要求 50ms
    )
    
    scorer = MultiObjectiveScorer(config={
        'latency_estimates': {
            'cloud_direct_ms': 200
        }
    })
    
    # CLOUD_DIRECT 预期延迟 200ms，超过 SLO
    score = scorer._score_latency(ExecutionStrategy.CLOUD_DIRECT, context)
    assert score == 0.0, "超过 SLO 应得分为 0"
    
    print(f"✅ CLOUD_DIRECT 延迟得分: {score} (预期延迟200ms > SLO 50ms)")


def test_e2e_decision_time_sensitive():
    """测试时敏任务的端到端决策"""
    print("\n=== 测试: 时敏任务端到端决策 ===")
    
    request = InferenceRequest(
        prompt="Quick question: What is 2+2?",
        requirements=TaskRequirements(
            max_latency_ms=100,  # 时敏任务
            priority=3
        )
    )
    
    config = {
        'state_monitor': {},
        'hard_constraints': {'ultra_low_latency': 50},
        'scoring_weights': {}
    }
    
    f1 = F1_DecisionModule(config=config)
    plan = f1.decide(request)
    
    # 应该选择 EDGE_ONLY（触发硬约束或得分最高）
    print(f"✅ 决策策略: {plan.strategy.value}")
    print(f"✅ 决策得分: {plan.score:.3f}")
    print(f"✅ 决策理由: {plan.reason}")
    print(f"✅ Draft 长度: {plan.draft_max_tokens}")
    
    assert plan.strategy in [ExecutionStrategy.EDGE_ONLY], \
        f"时敏任务应选择 EDGE_ONLY，实际: {plan.strategy.value}"


def test_e2e_decision_high_quality():
    """测试高质量要求任务"""
    print("\n=== 测试: 高质量要求任务 ===")
    
    request = InferenceRequest(
        prompt="Write a detailed essay about artificial intelligence.",
        requirements=TaskRequirements(
            min_quality_score=0.95,
            max_latency_ms=5000
        )
    )
    
    config = {
        'state_monitor': {},
        'hard_constraints': {},
        'scoring_weights': {'quality': 0.5}  # 提高质量权重
    }
    
    f1 = F1_DecisionModule(config=config)
    plan = f1.decide(request)
    
    print(f"✅ 决策策略: {plan.strategy.value}")
    print(f"✅ 决策得分: {plan.score:.3f}")
    print(f"✅ 决策理由: {plan.reason}")
    
    # 应该选择云端相关策略
    assert plan.strategy in [
        ExecutionStrategy.CLOUD_DIRECT,
        ExecutionStrategy.SPECULATIVE_STANDARD,
        ExecutionStrategy.ADAPTIVE_CONFIDENCE
    ], f"高质量任务应选择云端策略，实际: {plan.strategy.value}"


def test_fallback_plan():
    """测试降级策略"""
    print("\n=== 测试: 降级策略 ===")
    
    config = {
        'state_monitor': {},
        'hard_constraints': {},
        'scoring_weights': {}
    }
    
    f1 = F1_DecisionModule(config=config)
    
    # 模拟正常系统状态
    context = DecisionContext(
        request=InferenceRequest(prompt="test"),
        system_state=SystemStats(cpu_usage=50.0, memory_available_mb=2000),
        task_requirements=TaskRequirements()
    )
    
    plan = f1._fallback_plan(context, reason="测试降级")
    
    print(f"✅ 降级策略: {plan.strategy.value}")
    print(f"✅ 降级理由: {plan.reason}")
    
    assert plan.strategy in [
        ExecutionStrategy.SPECULATIVE_STANDARD,
        ExecutionStrategy.CLOUD_DIRECT,
        ExecutionStrategy.EDGE_ONLY
    ], "降级策略应该是可执行的"


def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("F1 决策模块单元测试")
    print("=" * 60)
    
    tests = [
        test_hard_constraint_cpu_overload,
        test_hard_constraint_ultra_low_latency,
        test_hard_constraint_privacy,
        test_scoring_latency_exceeds_slo,
        test_e2e_decision_time_sensitive,
        test_e2e_decision_high_quality,
        test_fallback_plan
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"❌ 测试失败: {test.__name__}")
            print(f"   错误: {e}")
            failed += 1
        except Exception as e:
            print(f"❌ 测试异常: {test.__name__}")
            print(f"   异常: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"测试结果: {passed} 通过, {failed} 失败")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
