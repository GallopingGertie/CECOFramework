"""
阶段4测试：硬件感知功能（简化版，无外部依赖）

测试覆盖：
1. GPU 模式下的参数调整
2. CPU 模式下的参数调整
3. 硬约束检查（GPU过载 vs CPU过载）
4. ExecutionPlanner 硬件感知参数生成
"""
import sys
sys.path.insert(0, '/Users/hefen/Desktop/husband/20260203/CECOFramework-main_new/vllm_llama_inference_framework')

from edge.execution_planner import ExecutionPlanner
from edge.decision_engine import HardConstraintChecker
from common.types import (
    InferenceRequest,
    TaskRequirements,
    SystemStats,
    DecisionContext,
    ExecutionStrategy
)


class TestHardwareAwareDecision:
    """测试硬件感知决策"""
    
    def test_gpu_mode_parameters(self):
        """测试 GPU 模式下的参数调整"""
        print("\n" + "=" * 60)
        print("测试1：GPU 模式参数调整")
        print("=" * 60)
        
        config = self._get_gpu_config()
        planner = ExecutionPlanner(config)
        
        # 创建 GPU 模式的决策上下文
        context = DecisionContext(
            request=InferenceRequest(prompt="测试", max_tokens=100),
            system_state=SystemStats(
                cpu_usage=50.0,
                memory_available_mb=2000.0,
                gpu_usage=30.0,
                gpu_memory_free_mb=4000.0,
                device_type="gpu"
            ),
            task_requirements=TaskRequirements(
                max_latency_ms=1500,  # 修改：使用更高的延迟要求，避免触发延迟调整
                min_quality_score=0.8
            ),
            network_state=None
        )
        
        # 测试 EDGE_ONLY 策略
        plan_edge = planner.generate_plan(
            ExecutionStrategy.EDGE_ONLY,
            context,
            score=0.8
        )
        
        print(f"✓ GPU 模式 EDGE_ONLY:")
        print(f"  - Draft tokens: {plan_edge.draft_max_tokens}")
        print(f"  - 预期: 256 (GPU 模式更强)")
        assert plan_edge.draft_max_tokens == 256, f"Expected 256, got {plan_edge.draft_max_tokens}"
        
        # 测试 SPECULATIVE 策略
        plan_spec = planner.generate_plan(
            ExecutionStrategy.SPECULATIVE_STANDARD,
            context,
            score=0.85
        )
        
        print(f"\n✓ GPU 模式 SPECULATIVE:")
        print(f"  - Draft tokens: {plan_spec.draft_max_tokens}")
        print(f"  - SLO延迟: {context.task_requirements.max_latency_ms}ms")
        print(f"  - 预期: 96 (GPU 模式协同更多draft)")
        print(f"  - 调试: 基础参数={planner.strategy_defaults[ExecutionStrategy.SPECULATIVE_STANDARD]}")
        assert plan_spec.draft_max_tokens == 96, f"Expected 96, got {plan_spec.draft_max_tokens}"
        
        print("\n✅ GPU 模式参数测试通过")
    
    def test_cpu_mode_parameters(self):
        """测试 CPU 模式下的参数调整"""
        print("\n" + "=" * 60)
        print("测试2：CPU 模式参数调整")
        print("=" * 60)
        
        config = self._get_cpu_config()
        planner = ExecutionPlanner(config)
        
        # 创建 CPU 模式的决策上下文
        context = DecisionContext(
            request=InferenceRequest(prompt="测试", max_tokens=100),
            system_state=SystemStats(
                cpu_usage=50.0,
                memory_available_mb=2000.0,
                device_type="cpu"
            ),
            task_requirements=TaskRequirements(
                max_latency_ms=1500,  # 修改：使用更高的延迟要求，避免触发延迟调整
                min_quality_score=0.8
            ),
            network_state=None
        )
        
        # 测试 EDGE_ONLY 策略
        plan_edge = planner.generate_plan(
            ExecutionStrategy.EDGE_ONLY,
            context,
            score=0.8
        )
        
        print(f"✓ CPU 模式 EDGE_ONLY:")
        print(f"  - Draft tokens: {plan_edge.draft_max_tokens}")
        print(f"  - 预期: 128 (CPU 模式能力受限)")
        assert plan_edge.draft_max_tokens == 128, f"Expected 128, got {plan_edge.draft_max_tokens}"
        
        # 测试 SPECULATIVE 策略
        plan_spec = planner.generate_plan(
            ExecutionStrategy.SPECULATIVE_STANDARD,
            context,
            score=0.85
        )
        
        print(f"\n✓ CPU 模式 SPECULATIVE:")
        print(f"  - Draft tokens: {plan_spec.draft_max_tokens}")
        print(f"  - 预期: 48 (CPU 模式协同draft较少)")
        assert plan_spec.draft_max_tokens == 48, f"Expected 48, got {plan_spec.draft_max_tokens}"
        
        print("\n✅ CPU 模式参数测试通过")
    
    def test_gpu_overload_detection(self):
        """测试 GPU 过载检测"""
        print("\n" + "=" * 60)
        print("测试3：GPU 过载检测")
        print("=" * 60)
        
        config = self._get_gpu_config()
        checker = HardConstraintChecker(config['hard_constraints'])
        
        # GPU 过载场景
        context = DecisionContext(
            request=InferenceRequest(prompt="测试", max_tokens=100),
            system_state=SystemStats(
                cpu_usage=50.0,
                memory_available_mb=2000.0,
                gpu_usage=90.0,  # GPU 过载（>85%）
                gpu_memory_free_mb=4000.0,
                device_type="gpu"
            ),
            task_requirements=TaskRequirements(
                max_latency_ms=200,
                min_quality_score=0.8
            ),
            network_state=None
        )
        
        decision = checker.check(context)
        
        print(f"✓ GPU 使用率: 90% (阈值: 85%)")
        print(f"✓ 触发硬约束: {decision is not None}")
        print(f"✓ 强制策略: {decision.strategy.value if decision else 'None'}")
        print(f"✓ 原因: {decision.reason if decision else 'None'}")
        
        assert decision is not None, "GPU过载应该触发硬约束"
        assert decision.strategy == ExecutionStrategy.CLOUD_DIRECT, "GPU过载应该转移到云端"
        assert "GPU 过载" in decision.reason
        
        print("\n✅ GPU 过载检测测试通过")
    
    def test_cpu_overload_detection(self):
        """测试 CPU 过载检测"""
        print("\n" + "=" * 60)
        print("测试4：CPU 过载检测")
        print("=" * 60)
        
        config = self._get_cpu_config()
        checker = HardConstraintChecker(config['hard_constraints'])
        
        # CPU 过载场景
        context = DecisionContext(
            request=InferenceRequest(prompt="测试", max_tokens=100),
            system_state=SystemStats(
                cpu_usage=95.0,  # CPU 过载（>90%）
                memory_available_mb=2000.0,
                device_type="cpu"
            ),
            task_requirements=TaskRequirements(
                max_latency_ms=200,
                min_quality_score=0.8
            ),
            network_state=None
        )
        
        decision = checker.check(context)
        
        print(f"✓ CPU 使用率: 95% (阈值: 90%)")
        print(f"✓ 触发硬约束: {decision is not None}")
        print(f"✓ 强制策略: {decision.strategy.value if decision else 'None'}")
        print(f"✓ 原因: {decision.reason if decision else 'None'}")
        
        assert decision is not None, "CPU过载应该触发硬约束"
        assert decision.strategy == ExecutionStrategy.CLOUD_DIRECT, "CPU过载应该转移到云端"
        assert "CPU 过载" in decision.reason
        
        print("\n✅ CPU 过载检测测试通过")
    
    def test_latency_adjustment(self):
        """测试延迟要求下的参数调整"""
        print("\n" + "=" * 60)
        print("测试5：延迟约束下的参数调整")
        print("=" * 60)
        
        config = self._get_gpu_config()
        planner = ExecutionPlanner(config)
        
        # 极低延迟要求
        context_low_latency = DecisionContext(
            request=InferenceRequest(prompt="测试", max_tokens=100),
            system_state=SystemStats(
                cpu_usage=50.0,
                memory_available_mb=2000.0,
                gpu_usage=30.0,
                device_type="gpu"
            ),
            task_requirements=TaskRequirements(
                max_latency_ms=400,  # 极低延迟（<500ms）
                min_quality_score=0.8
            ),
            network_state=None
        )
        
        plan = planner.generate_plan(
            ExecutionStrategy.SPECULATIVE_STANDARD,
            context_low_latency,
            score=0.8
        )
        
        print(f"✓ GPU 模式 + 极低延迟要求 (400ms):")
        print(f"  - Draft tokens: {plan.draft_max_tokens}")
        print(f"  - 预期: ≤32 (延迟约束进一步缩短)")
        assert plan.draft_max_tokens <= 32, f"Expected ≤32, got {plan.draft_max_tokens}"
        
        print("\n✅ 延迟调整测试通过")
    
    def _get_gpu_config(self):
        """获取 GPU 模式测试配置"""
        return {
            'hardware': {
                'device_type': 'gpu',
                'gpu_overload_threshold': 85.0,
                'gpu_memory_critical_mb': 1000
            },
            'hard_constraints': {
                'cpu_overload': 90.0,
                'gpu_overload': 85.0,
                'memory_critical': 500,
                'ultra_low_latency': 50,
                'weak_network_rtt': 200.0
            },
            'hardware_adaptive': {
                'gpu_mode': {
                    'edge_only_max_tokens': 256,
                    'collaborative_draft_tokens': 96,
                    'task_complexity_threshold': 0.7
                },
                'cpu_mode': {
                    'edge_only_max_tokens': 128,
                    'collaborative_draft_tokens': 48,
                    'task_complexity_threshold': 0.8
                }
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
            }
        }
    
    def _get_cpu_config(self):
        """获取 CPU 模式测试配置"""
        config = self._get_gpu_config()
        config['hardware']['device_type'] = 'cpu'
        return config


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("阶段4测试套件：硬件感知功能")
    print("=" * 60)
    
    try:
        test = TestHardwareAwareDecision()
        
        test.test_gpu_mode_parameters()
        test.test_cpu_mode_parameters()
        test.test_gpu_overload_detection()
        test.test_cpu_overload_detection()
        test.test_latency_adjustment()
        
        print("\n" + "=" * 60)
        print("✅ 所有硬件感知测试通过！")
        print("=" * 60)
        print("\n总结：")
        print("- ✅ GPU 模式参数正确（256 tokens EDGE, 96 tokens SPEC）")
        print("- ✅ CPU 模式参数正确（128 tokens EDGE, 48 tokens SPEC）")
        print("- ✅ GPU 过载检测有效")
        print("- ✅ CPU 过载检测有效")
        print("- ✅ 延迟约束下的参数调整正确")
        print("\n配置说明：")
        print("- 在 config.yaml 中设置 hardware.device_type 为 'gpu' 或 'cpu'")
        print("- GPU 模式会生成更多 token，适合处理更重的任务")
        print("- CPU 模式生成较少 token，只能处理简单任务")
        return 0
    
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
