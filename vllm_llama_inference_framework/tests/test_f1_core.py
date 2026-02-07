"""
F1 决策模块简化测试（不需要 psutil）
测试核心决策逻辑
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
from edge.decision_engine import HardConstraintChecker, MultiObjectiveScorer
from edge.execution_planner import ExecutionPlanner


def test_hard_constraint_cpu_overload():
    """测试 CPU 过载硬约束"""
    print("\n=== 测试 1: CPU 过载硬约束 ===")
    
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
    print("\n=== 测试 2: 超低延迟硬约束 ===")
    
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
    print("\n=== 测试 3: 隐私约束 ===")
    
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


def test_scoring_all_strategies():
    """测试所有策略的评分"""
    print("\n=== 测试 4: 策略评分 ===")
    
    context = DecisionContext(
        request=InferenceRequest(prompt="test"),
        system_state=SystemStats(cpu_usage=50.0, memory_available_mb=2000),
        task_requirements=TaskRequirements(max_latency_ms=1000)
    )
    
    scorer = MultiObjectiveScorer(config={
        'scoring_weights': {'latency': 0.4, 'cost': 0.3, 'quality': 0.3},
        'latency_estimates': {
            'edge_only_ms': 30,
            'cloud_direct_ms': 200,
            'speculative_standard_ms': 80
        }
    })
    
    scored = scorer.score_strategies(context)
    
    print("所有策略得分:")
    for s in scored:
        print(f"  {s.strategy.value}: {s.score:.3f}")
    
    assert len(scored) == 4, "应该有4个策略"
    assert all(s.score >= 0 for s in scored), "得分应该非负"
    
    # 找到最高分
    best = max(scored, key=lambda x: x.score)
    print(f"✅ 最优策略: {best.strategy.value} (得分={best.score:.3f})")


def test_execution_planner():
    """测试执行计划生成"""
    print("\n=== 测试 5: 执行计划生成 ===")
    
    context = DecisionContext(
        request=InferenceRequest(prompt="test"),
        system_state=SystemStats(cpu_usage=50.0, memory_available_mb=2000),
        task_requirements=TaskRequirements(max_latency_ms=500, min_quality_score=0.9)
    )
    
    planner = ExecutionPlanner(config={})
    plan = planner.generate_plan(
        ExecutionStrategy.SPECULATIVE_STANDARD,
        context,
        score=0.85
    )
    
    print(f"✅ 策略: {plan.strategy.value}")
    print(f"✅ 置信度阈值: {plan.confidence_threshold}")
    print(f"✅ Draft 长度: {plan.draft_max_tokens}")
    print(f"✅ 参数: {plan.params}")
    
    assert plan.strategy == ExecutionStrategy.SPECULATIVE_STANDARD
    assert plan.draft_max_tokens > 0
    assert 0.5 <= plan.confidence_threshold <= 0.95


def test_dynamic_threshold():
    """测试动态阈值计算"""
    print("\n=== 测试 6: 动态阈值计算 ===")
    
    # 高质量要求
    context_high = DecisionContext(
        request=InferenceRequest(prompt="test"),
        system_state=SystemStats(cpu_usage=50.0, memory_available_mb=2000),
        task_requirements=TaskRequirements(min_quality_score=0.95)
    )
    
    # 低质量要求
    context_low = DecisionContext(
        request=InferenceRequest(prompt="test"),
        system_state=SystemStats(cpu_usage=50.0, memory_available_mb=2000),
        task_requirements=TaskRequirements(min_quality_score=0.6)
    )
    
    planner = ExecutionPlanner(config={})
    
    plan_high = planner.generate_plan(ExecutionStrategy.ADAPTIVE_CONFIDENCE, context_high)
    plan_low = planner.generate_plan(ExecutionStrategy.ADAPTIVE_CONFIDENCE, context_low)
    
    print(f"高质量阈值: {plan_high.confidence_threshold:.2f}")
    print(f"低质量阈值: {plan_low.confidence_threshold:.2f}")
    
    assert plan_high.confidence_threshold > plan_low.confidence_threshold, \
        "高质量要求应该有更高的置信度阈值"
    
    print("✅ 动态阈值计算正确")


def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("F1 决策模块核心逻辑测试")
    print("=" * 60)
    
    tests = [
        test_hard_constraint_cpu_overload,
        test_hard_constraint_ultra_low_latency,
        test_hard_constraint_privacy,
        test_scoring_all_strategies,
        test_execution_planner,
        test_dynamic_threshold
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
    print(f"测试结果: ✅ {passed} 通过, ❌ {failed} 失败")
    print("=" * 60)
    
    if passed == len(tests):
        print("\n🎉 所有测试通过！F1 模块核心逻辑正常工作")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
