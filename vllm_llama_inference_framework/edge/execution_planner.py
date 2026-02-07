"""
F1 执行计划生成器
根据策略和上下文生成详细的执行参数
（阶段4扩展：硬件感知参数调整）
"""
from typing import Dict, Any
from common.types import (
    ExecutionStrategy,
    ExecutionPlan,
    DecisionContext
)


class ExecutionPlanner:
    """执行计划生成器（硬件感知）"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # 硬件自适应配置（新增）
        hw_adaptive = config.get('hardware_adaptive', {})
        self.gpu_mode_config = hw_adaptive.get('gpu_mode', {})
        self.cpu_mode_config = hw_adaptive.get('cpu_mode', {})
        
        # 策略默认参数配置（基础值，会根据硬件调整）
        self.strategy_defaults = {
            ExecutionStrategy.EDGE_ONLY: {
                'draft_max_tokens': 128,  # 会根据硬件类型调整
                'use_kv_cache': True
            },
            ExecutionStrategy.CLOUD_DIRECT: {
                'max_tokens': 128,
                'use_kv_cache': True
            },
            ExecutionStrategy.SPECULATIVE_STANDARD: {
                'draft_max_tokens': 64,  # 会根据硬件类型调整
                'confidence_threshold': 0.8,
                'verify_timeout_ms': 5000
            },
            ExecutionStrategy.ADAPTIVE_CONFIDENCE: {
                'draft_max_tokens': 64,  # 会根据硬件类型调整
                'confidence_threshold': 'dynamic',
                'verify_timeout_ms': 5000
            }
        }
    
    def generate_plan(self, 
                      strategy: ExecutionStrategy,
                      context: DecisionContext,
                      score: float = 0.0,
                      reason: str = "") -> ExecutionPlan:
        """
        生成执行计划
        
        根据策略和上下文，生成详细的执行参数
        """
        
        # 获取策略的默认参数
        base_params = self.strategy_defaults.get(strategy, {}).copy()
        
        # 根据上下文动态调整参数
        adjusted_params = self._adjust_params(strategy, base_params, context)
        
        # 计算置信度阈值（对于需要的策略）
        if strategy in [ExecutionStrategy.SPECULATIVE_STANDARD, 
                       ExecutionStrategy.ADAPTIVE_CONFIDENCE]:
            threshold = self._calculate_confidence_threshold(strategy, context)
            adjusted_params['confidence_threshold'] = threshold
        
        return ExecutionPlan(
            strategy=strategy,
            params=adjusted_params,
            confidence_threshold=adjusted_params.get('confidence_threshold', 0.8),
            draft_max_tokens=adjusted_params.get('draft_max_tokens', 64),
            reason=reason or f"Score: {score:.3f}",
            score=score
        )
    
    def _adjust_params(self, 
                       strategy: ExecutionStrategy,
                       base_params: Dict[str, Any],
                       context: DecisionContext) -> Dict[str, Any]:
        """
        根据上下文动态调整参数（阶段4增强：硬件感知）
        
        核心逻辑：
        - GPU 模式：可以处理更重的任务，生成更多 token
        - CPU 模式：只能处理简单任务，生成较少 token
        """
        params = base_params.copy()
        device_type = context.system_state.device_type
        
        # ===== 硬件感知调整（核心功能）=====
        if device_type == "gpu":
            # GPU 模式：能力更强
            if strategy == ExecutionStrategy.EDGE_ONLY:
                # GPU 端侧可以负责更重的任务
                params['draft_max_tokens'] = self.gpu_mode_config.get(
                    'edge_only_max_tokens', 256
                )
            elif strategy in [ExecutionStrategy.SPECULATIVE_STANDARD,
                             ExecutionStrategy.ADAPTIVE_CONFIDENCE]:
                # GPU 协同推理可以生成更多 draft tokens
                params['draft_max_tokens'] = self.gpu_mode_config.get(
                    'collaborative_draft_tokens', 96
                )
        else:
            # CPU 模式：能力受限
            if strategy == ExecutionStrategy.EDGE_ONLY:
                # CPU 端侧只能负责简单任务
                params['draft_max_tokens'] = self.cpu_mode_config.get(
                    'edge_only_max_tokens', 128
                )
            elif strategy in [ExecutionStrategy.SPECULATIVE_STANDARD,
                             ExecutionStrategy.ADAPTIVE_CONFIDENCE]:
                # CPU 协同推理生成较少 draft tokens
                params['draft_max_tokens'] = self.cpu_mode_config.get(
                    'collaborative_draft_tokens', 48
                )
        
        # ===== 调整1: 根据延迟要求进一步调整 draft 长度（在硬件基础上微调）=====
        if strategy in [ExecutionStrategy.SPECULATIVE_STANDARD,
                       ExecutionStrategy.ADAPTIVE_CONFIDENCE]:
            slo_latency = context.task_requirements.max_latency_ms
            
            if slo_latency < 500:  # 极低延迟要求
                # 进一步缩短 draft 长度（但不低于硬件模式的1/3）
                min_tokens = params['draft_max_tokens'] // 3
                params['draft_max_tokens'] = max(min_tokens, min(params['draft_max_tokens'], 32))
            elif slo_latency < 1000:  # 低延迟要求
                # 适度缩短（保留75%）
                params['draft_max_tokens'] = int(params['draft_max_tokens'] * 0.75)
        
        # ===== 调整2: 根据质量要求调整阈值 =====
        if context.task_requirements.min_quality_score > 0.9:
            if 'confidence_threshold' in params and params['confidence_threshold'] != 'dynamic':
                params['confidence_threshold'] = max(params.get('confidence_threshold', 0.8), 0.85)
        
        # ===== 调整3: 根据系统负载调整超时 =====
        # GPU 模式检查 GPU 负载
        if device_type == "gpu" and context.system_state.gpu_usage > 70:
            if 'verify_timeout_ms' in params:
                params['verify_timeout_ms'] = int(params['verify_timeout_ms'] * 1.2)
        # CPU 模式检查 CPU 负载
        elif device_type == "cpu" and context.system_state.cpu_usage > 80:
            if 'verify_timeout_ms' in params:
                params['verify_timeout_ms'] = int(params['verify_timeout_ms'] * 1.2)
        
        return params
    
    def _calculate_confidence_threshold(self,
                                        strategy: ExecutionStrategy,
                                        context: DecisionContext) -> float:
        """
        动态计算置信度阈值
        
        阶段 1: 基于任务质量要求
        阶段 3: 可结合历史统计（如最近的平均接受率）
        """
        if strategy == ExecutionStrategy.SPECULATIVE_STANDARD:
            # 固定阈值策略
            return 0.8
        
        elif strategy == ExecutionStrategy.ADAPTIVE_CONFIDENCE:
            # 动态阈值策略
            base_threshold = 0.75
            
            # 根据质量要求调整
            if context.task_requirements.min_quality_score > 0.9:
                base_threshold += 0.1
            elif context.task_requirements.min_quality_score < 0.7:
                base_threshold -= 0.1
            
            # 根据优先级调整
            if context.task_requirements.priority >= 3:
                # 紧急任务降低阈值，加快响应
                base_threshold -= 0.05
            
            return max(0.5, min(0.95, base_threshold))
        
        return 0.8
