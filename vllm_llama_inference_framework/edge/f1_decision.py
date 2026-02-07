"""
F1 决策模块 - 主入口
阶段2扩展：集成网络监控
阶段3扩展：集成历史追踪与自适应调整
阶段4扩展：硬件感知（GPU/CPU）
"""
from typing import Dict, Any, Optional
import traceback
import asyncio

from common.types import (
    InferenceRequest,
    SystemStats,
    NetworkStats,
    DecisionContext,
    ExecutionPlan,
    ExecutionStrategy
)
from edge.monitor import StateMonitor  # 阶段2：使用新的 monitor
from edge.decision_engine import DecisionEngine
from edge.execution_planner import ExecutionPlanner

# 阶段3：导入历史追踪和自适应模块
from edge.history_tracker import HistoryTracker, ExecutionRecord
from edge.adaptive_threshold import AdaptiveThresholdCalculator


class F1_DecisionModule:
    """
    F1 决策模块
    
    阶段1：基础决策
    阶段2：支持网络感知
    阶段3：支持历史追踪与自适应调整
    阶段4：硬件感知（GPU/CPU差异化决策）
    
    负责基于全局状态进行智能决策，选择最优的执行策略
    """
    
    def __init__(self, config: Dict[str, Any], cloud_endpoint: str = "http://localhost:8081"):
        self.config = config
        self.cloud_endpoint = cloud_endpoint
        
        # 阶段4：获取硬件配置
        hw_config = config.get('hardware', {})
        self.device_type = hw_config.get('device_type', 'cpu')
        
        # 阶段3：初始化历史追踪器
        history_config = config.get('history_tracker', {})
        max_history = history_config.get('max_history_size', 100)
        self.history_tracker = HistoryTracker(max_history_size=max_history)
        
        # 阶段3：初始化自适应阈值计算器
        adaptive_config = config.get('adaptive_threshold', {})
        self.adaptive_calculator = AdaptiveThresholdCalculator(adaptive_config)
        
        # 阶段2：初始化新的 StateMonitor（阶段4：传入完整配置以支持硬件监控）
        self.state_monitor = StateMonitor(cloud_endpoint, config)
        
        # 阶段3：初始化 DecisionEngine 时传入 history_tracker
        self.decision_engine = DecisionEngine(config, history_tracker=self.history_tracker)
        self.execution_planner = ExecutionPlanner(config)
        
        # 阶段2：网络探测开关
        self.enable_network_probe = config.get('enable_network_probe', True)
        
        # 阶段3：自适应开关
        self.enable_adaptive = config.get('enable_adaptive', True)
        
        print("[F1] 决策模块初始化完成（阶段4：硬件感知）")
        print(f"[F1] 硬件类型: {self.device_type.upper()}")
        print(f"[F1] 配置: 硬约束={config.get('hard_constraints', {})}")
        print(f"[F1] 评分权重={config.get('scoring_weights', {})}")
        print(f"[F1] 网络探测: {'启用' if self.enable_network_probe else '禁用'}")
        print(f"[F1] 自适应调整: {'启用' if self.enable_adaptive else '禁用'}")
        print(f"[F1] 历史窗口大小: {max_history}")
    
    async def decide_async(self, 
                          request: InferenceRequest, 
                          current_system_state: Optional[SystemStats] = None
                         ) -> ExecutionPlan:
        """
        核心决策接口（异步版本，支持网络探测和自适应调整）
        
        Args:
            request: 推理请求
            current_system_state: 可选的系统状态（如果外部已采集）
        
        Returns:
            ExecutionPlan: 执行计划
        """
        try:
            # 阶段3：触发自适应参数更新（每 N 次执行更新一次）
            if self.enable_adaptive and self.adaptive_calculator.should_update():
                self._apply_adaptive_updates()
            
            # 1. 构造决策上下文（包含网络状态）
            context = await self._build_context_async(request, current_system_state)
            
            # 打印决策上下文（调试用）
            self._log_context(context)
            
            # 2. 检查硬约束
            hard_decision = self.decision_engine.check_hard_constraints(context)
            if hard_decision:
                plan = self.execution_planner.generate_plan(
                    hard_decision.strategy, context, reason=hard_decision.reason
                )
                print(f"[F1] 硬约束触发: {plan.strategy.value} - {plan.reason}")
                return plan
            
            # 3. 多目标评分
            scored_strategies = self.decision_engine.score_strategies(context)
            
            # 4. 过滤不可行策略（得分为 0）
            valid_strategies = [s for s in scored_strategies if s.score > 0]
            
            if not valid_strategies:
                # 降级：所有策略都不满足，使用安全默认策略
                print("[F1] 警告: 所有策略得分为0，使用降级策略")
                return self._fallback_plan(context, reason="所有策略得分为0，降级处理")
            
            best_strategy = max(valid_strategies, key=lambda x: x.score)
            
            # 5. 生成执行计划
            plan = self.execution_planner.generate_plan(
                best_strategy.strategy, context, score=best_strategy.score
            )
            
            print(f"[F1] 决策完成: {plan.strategy.value} (得分={plan.score:.3f})")
            return plan
        
        except Exception as e:
            # 任何异常都不应该影响推理流程
            print(f"[F1] 决策异常: {e}, 使用降级策略")
            traceback.print_exc()
            return self._fallback_plan(
                DecisionContext(
                    request=request,
                    system_state=SystemStats(cpu_usage=50.0, memory_available_mb=2000.0),
                    task_requirements=request.requirements,
                    network_state=None
                ),
                reason=f"异常降级: {str(e)}"
            )
    
    def decide(self, 
               request: InferenceRequest, 
               current_system_state: Optional[SystemStats] = None
              ) -> ExecutionPlan:
        """
        同步决策接口（为了兼容性，内部调用异步版本）
        """
        loop = None
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(
            self.decide_async(request, current_system_state)
        )
    
    async def _build_context_async(self, 
                                   request: InferenceRequest,
                                   current_system_state: Optional[SystemStats]) -> DecisionContext:
        """构造决策上下文（阶段2：包含网络状态）"""
        try:
            # 获取系统状态
            if current_system_state:
                system_state = current_system_state
            else:
                system_state = self.state_monitor.get_system_stats()
        except Exception as e:
            # 状态采集失败，使用保守估计
            print(f"[F1] 状态采集失败: {e}, 使用默认值")
            system_state = SystemStats(
                cpu_usage=50.0,  # 假设中等负载
                memory_available_mb=2000.0
            )
        
        # 阶段2：获取网络状态
        network_state = None
        if self.enable_network_probe:
            try:
                network_state = await self.state_monitor.probe_network()
            except Exception as e:
                print(f"[F1] 网络探测失败: {e}")
                # 网络探测失败不影响决策，network_state 保持 None
        
        # 提取任务需求
        try:
            task_requirements = request.requirements
        except Exception as e:
            print(f"[F1] SLO 提取失败: {e}, 使用默认值")
            from common.types import TaskRequirements
            task_requirements = TaskRequirements()
        
        return DecisionContext(
            request=request,
            system_state=system_state,
            task_requirements=task_requirements,
            network_state=network_state  # 阶段2：加入网络状态
        )
    
    def _fallback_plan(self, 
                       context: DecisionContext,
                       reason: str = "降级策略") -> ExecutionPlan:
        """
        降级执行计划（安全默认策略）
        
        优先级：
        1. 如果系统状态正常，使用 SPECULATIVE_STANDARD（平衡）
        2. 如果系统过载，使用 CLOUD_DIRECT（卸载负载）
        3. 极端情况，使用 EDGE_ONLY（保证可用性）
        """
        try:
            if context.system_state.cpu_usage < 90:
                strategy = ExecutionStrategy.SPECULATIVE_STANDARD
            else:
                strategy = ExecutionStrategy.CLOUD_DIRECT
        except Exception:
            strategy = ExecutionStrategy.EDGE_ONLY
        
        return self.execution_planner.generate_plan(
            strategy, context, score=0.0, reason=reason
        )
    
    def _log_context(self, context: DecisionContext):
        """打印决策上下文（阶段2：包含网络状态）"""
        try:
            base_info = (f"[F1] 上下文: CPU={context.system_state.cpu_usage:.1f}%, "
                        f"内存={context.system_state.memory_available_mb:.0f}MB, "
                        f"SLO延迟<{context.task_requirements.max_latency_ms}ms, "
                        f"质量>{context.task_requirements.min_quality_score:.2f}, "
                        f"优先级={context.task_requirements.priority}")
            
            # 阶段2：加入网络状态
            if context.network_state:
                net_info = (f", 网络RTT={context.network_state.rtt_ms:.1f}ms, "
                           f"弱网={'是' if context.network_state.is_weak_network else '否'}")
                print(base_info + net_info)
            else:
                print(base_info)
        except Exception:
            pass
    
    # ========== 阶段3新增：历史记录和自适应方法 ==========
    
    def record_execution(self, 
                         strategy: ExecutionStrategy,
                         acceptance_rate: float,
                         latency_ms: float,
                         edge_latency_ms: float,
                         cloud_latency_ms: float,
                         confidence_score: float,
                         success: bool,
                         tokens_generated: int = 0):
        """
        记录一次执行结果（阶段3）
        
        Args:
            strategy: 使用的执行策略
            acceptance_rate: Draft 接受率
            latency_ms: 总延迟
            edge_latency_ms: 边端延迟
            cloud_latency_ms: 云端延迟
            confidence_score: 置信度分数
            success: 是否成功
            tokens_generated: 生成的 token 数
        """
        import time
        record = ExecutionRecord(
            timestamp=time.time(),
            strategy=strategy,
            acceptance_rate=acceptance_rate,
            latency_ms=latency_ms,
            edge_latency_ms=edge_latency_ms,
            cloud_latency_ms=cloud_latency_ms,
            confidence_score=confidence_score,
            success=success,
            tokens_generated=tokens_generated
        )
        self.history_tracker.add_record(record)
    
    def _apply_adaptive_updates(self):
        """
        应用自适应参数更新（阶段3）
        
        根据历史数据调整：
        - 置信度阈值
        - Draft 长度
        - 评分权重
        """
        try:
            # 构造当前参数字典
            current_params = {
                'confidence_threshold': self.config.get('confidence_threshold', 0.80),
                'draft_max_tokens': self.config.get('draft_max_tokens', 64),
                'task_latency_slo': self.config.get('default_latency_slo', 150),
                'scoring_weights': self.config.get('scoring_weights', {
                    'latency': 0.4, 'cost': 0.3, 'quality': 0.3
                })
            }
            
            # 调用自适应计算器更新参数
            new_params = self.adaptive_calculator.update_parameters(
                self.history_tracker,
                current_params
            )
            
            # 应用新参数到配置（下次决策生效）
            if 'confidence_threshold' in new_params:
                self.config['confidence_threshold'] = new_params['confidence_threshold']
            
            if 'draft_max_tokens' in new_params:
                self.config['draft_max_tokens'] = new_params['draft_max_tokens']
            
            if 'scoring_weights' in new_params:
                self.config['scoring_weights'] = new_params['scoring_weights']
        
        except Exception as e:
            print(f"[F1] 自适应更新失败: {e}")
    
    def get_statistics_summary(self) -> Dict:
        """
        获取历史统计摘要（阶段3）
        
        Returns:
            统计摘要字典
        """
        return self.history_tracker.get_statistics_summary()
    
    def get_current_config(self) -> Dict:
        """
        获取当前配置（包含自适应调整后的值）
        
        Returns:
            当前配置字典
        """
        return {
            'confidence_threshold': self.config.get('confidence_threshold', 0.80),
            'draft_max_tokens': self.config.get('draft_max_tokens', 64),
            'scoring_weights': self.config.get('scoring_weights', {}),
            'enable_network_probe': self.enable_network_probe,
            'enable_adaptive': self.enable_adaptive
        }
