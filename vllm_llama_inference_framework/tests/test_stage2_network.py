"""
阶段2测试：网络感知决策
测试弱网检测和网络延迟对决策的影响
"""
import sys
import os
import asyncio

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.types import (
    InferenceRequest,
    TaskRequirements,
    SystemStats,
    NetworkStats,
    DecisionContext,
    ExecutionStrategy
)
from edge.f1_decision import F1_DecisionModule
from edge.decision_engine import HardConstraintChecker
from edge.monitor import StateMonitor


async def test_weak_network_detection():
    """测试弱网检测硬约束"""
    print("\n=== 阶段2测试1: 弱网检测 ===")
    
    # 构造弱网环境
    context = DecisionContext(
        request=InferenceRequest(prompt="test"),
        system_state=SystemStats(cpu_usage=50.0, memory_available_mb=2000),
        task_requirements=TaskRequirements(),
        network_state=NetworkStats(
            rtt_ms=250.0,  # 高延迟
            bandwidth_up=1.0,
            bandwidth_down=1.0,
            stability=0.6,
            is_weak_network=True
        )
    )
    
    checker = HardConstraintChecker(config={'weak_network_rtt': 200.0})
    decision = checker.check(context)
    
    assert decision is not None, "弱网应触发硬约束"
    assert decision.strategy == ExecutionStrategy.EDGE_ONLY, "弱网应选择 EDGE_ONLY"
    assert "弱网" in decision.reason or "网络延迟" in decision.reason, "理由应包含弱网相关信息"
    
    print(f"✅ 决策: {decision.strategy.value}")
    print(f"✅ 理由: {decision.reason}")


async def test_network_aware_scoring():
    """测试网络延迟对评分的影响"""
    print("\n=== 阶段2测试2: 网络感知评分 ===")
    
    from edge.decision_engine import MultiObjectiveScorer
    
    # 场景1: 低延迟网络
    context_good = DecisionContext(
        request=InferenceRequest(prompt="test"),
        system_state=SystemStats(cpu_usage=50.0, memory_available_mb=2000),
        task_requirements=TaskRequirements(max_latency_ms=2000),
        network_state=NetworkStats(
            rtt_ms=20.0,  # 低延迟
            bandwidth_up=100.0,
            bandwidth_down=100.0,
            stability=0.95,
            is_weak_network=False
        )
    )
    
    # 场景2: 高延迟网络
    context_bad = DecisionContext(
        request=InferenceRequest(prompt="test"),
        system_state=SystemStats(cpu_usage=50.0, memory_available_mb=2000),
        task_requirements=TaskRequirements(max_latency_ms=2000),
        network_state=NetworkStats(
            rtt_ms=150.0,  # 高延迟但未达到弱网阈值
            bandwidth_up=10.0,
            bandwidth_down=10.0,
            stability=0.7,
            is_weak_network=False
        )
    )
    
    scorer = MultiObjectiveScorer(config={
        'scoring_weights': {'latency': 0.4, 'cost': 0.3, 'quality': 0.3},
        'latency_estimates': {
            'edge_only_ms': 30,
            'cloud_direct_ms': 200,
            'speculative_standard_ms': 80
        }
    })
    
    # 计算 CLOUD_DIRECT 的延迟得分
    score_good_cloud = scorer._score_latency(ExecutionStrategy.CLOUD_DIRECT, context_good)
    score_bad_cloud = scorer._score_latency(ExecutionStrategy.CLOUD_DIRECT, context_bad)
    
    print(f"低延迟网络(RTT=20ms) CLOUD_DIRECT 得分: {score_good_cloud:.3f}")
    print(f"高延迟网络(RTT=150ms) CLOUD_DIRECT 得分: {score_bad_cloud:.3f}")
    
    # 高延迟网络应该导致 CLOUD_DIRECT 得分更低
    assert score_good_cloud > score_bad_cloud, "高延迟网络应降低云端策略得分"
    
    print("✅ 网络延迟正确影响评分")


async def test_state_monitor_network_probe():
    """测试网络探测功能"""
    print("\n=== 阶段2测试3: 网络探测 ===")
    
    # 使用模拟模式
    monitor = StateMonitor(
        cloud_endpoint="http://localhost:8081",
        config={'network_probe_interval': 5.0}
    )
    
    # 设置模拟网络
    monitor.set_simulation_network(rtt=100.0, bandwidth=50.0)
    
    # 探测网络
    net_stats = await monitor.probe_network()
    
    assert net_stats is not None, "应返回网络状态"
    assert net_stats.rtt_ms == 100.0, "RTT应为模拟值"
    assert not net_stats.is_weak_network, "100ms不应判定为弱网"
    
    print(f"✅ 网络状态: RTT={net_stats.rtt_ms}ms, 弱网={net_stats.is_weak_network}")
    
    # 测试弱网判定
    monitor.set_simulation_network(rtt=250.0, bandwidth=1.0)
    net_stats2 = await monitor.probe_network(force=True)
    
    assert net_stats2.is_weak_network, "250ms应判定为弱网"
    print(f"✅ 弱网检测: RTT={net_stats2.rtt_ms}ms, 弱网={net_stats2.is_weak_network}")


async def test_e2e_weak_network_scenario():
    """端到端测试：弱网场景"""
    print("\n=== 阶段2测试4: 端到端弱网场景 ===")
    
    config = {
        'state_monitor': {},
        'hard_constraints': {'weak_network_rtt': 200.0},
        'scoring_weights': {'latency': 0.4, 'cost': 0.3, 'quality': 0.3},
        'enable_network_probe': True
    }
    
    f1 = F1_DecisionModule(config, cloud_endpoint="http://localhost:8081")
    
    # 设置模拟弱网
    f1.state_monitor.set_simulation_network(rtt=250.0, bandwidth=1.0)
    
    request = InferenceRequest(
        prompt="Test in weak network",
        requirements=TaskRequirements(max_latency_ms=1000)
    )
    
    # 异步决策
    plan = await f1.decide_async(request)
    
    print(f"✅ 决策策略: {plan.strategy.value}")
    print(f"✅ 决策理由: {plan.reason}")
    
    # 弱网时应选择 EDGE_ONLY
    assert plan.strategy == ExecutionStrategy.EDGE_ONLY, \
        f"弱网应选择 EDGE_ONLY，实际: {plan.strategy.value}"


async def test_e2e_good_network_scenario():
    """端到端测试：良好网络场景"""
    print("\n=== 阶段2测试5: 端到端良好网络场景 ===")
    
    config = {
        'state_monitor': {},
        'hard_constraints': {},
        'scoring_weights': {'latency': 0.4, 'cost': 0.3, 'quality': 0.3},
        'enable_network_probe': True,
        'latency_estimates': {
            'edge_only_ms': 30,
            'cloud_direct_ms': 200,
            'speculative_standard_ms': 80
        }
    )
    
    f1 = F1_DecisionModule(config, cloud_endpoint="http://localhost:8081")
    
    # 设置模拟良好网络
    f1.state_monitor.set_simulation_network(rtt=15.0, bandwidth=100.0)
    
    request = InferenceRequest(
        prompt="Test in good network",
        requirements=TaskRequirements(
            max_latency_ms=3000,
            min_quality_score=0.9  # 高质量要求
        )
    )
    
    # 异步决策
    plan = await f1.decide_async(request)
    
    print(f"✅ 决策策略: {plan.strategy.value}")
    print(f"✅ 决策得分: {plan.score:.3f}")
    print(f"✅ 决策理由: {plan.reason}")
    
    # 良好网络 + 高质量要求，应选择云端相关策略
    assert plan.strategy in [
        ExecutionStrategy.CLOUD_DIRECT,
        ExecutionStrategy.SPECULATIVE_STANDARD,
        ExecutionStrategy.ADAPTIVE_CONFIDENCE
    ], f"良好网络+高质量应选择云端策略，实际: {plan.strategy.value}"


async def run_all_tests():
    """运行所有阶段2测试"""
    print("=" * 60)
    print("阶段2测试：网络感知决策")
    print("=" * 60)
    
    tests = [
        test_weak_network_detection,
        test_network_aware_scoring,
        test_state_monitor_network_probe,
        test_e2e_weak_network_scenario,
        test_e2e_good_network_scenario
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            await test()
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
        print("\n🎉 阶段2所有测试通过！网络感知功能正常工作")
    
    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
