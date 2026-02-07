"""
F1 决策引擎
包含硬约束检查器和多目标评分器
（阶段3扩展：集成历史统计）
"""
from typing import Dict, Any, List, Optional
from common.types import (
    DecisionContext, 
    ExecutionStrategy, 
    HardDecision,
    ScoredStrategy
)

# 阶段3导入（可选依赖，避免循环导入）
try:
    from edge.history_tracker import HistoryTracker
except ImportError:
    HistoryTracker = None


class HardConstraintChecker:
    """硬约束规则检查（阶段4扩展：GPU感知）"""
    
    def __init__(self, config: Dict[str, Any]):
        # 可配置的阈值
        self.cpu_overload_threshold = config.get('cpu_overload', 95.0)
        self.gpu_overload_threshold = config.get('gpu_overload', 85.0)  # 新增
        self.memory_critical_mb = config.get('memory_critical', 500)
        self.ultra_low_latency_ms = config.get('ultra_low_latency', 50)
        # 阶段2新增：网络阈值
        self.weak_network_rtt_threshold = config.get('weak_network_rtt', 200.0)  # 200ms判定为弱网
    
    def check(self, context: DecisionContext) -> Optional[HardDecision]:
        """
        检查硬约束，返回强制决策或 None
        
        优先级顺序（从高到低）：
        1. 系统过载保护（CPU/GPU）
        2. 极端延迟要求
        3. 隐私要求
        4. 弱网检测（阶段2新增）
        5. 紧急任务
        """
        
        # ===== 规则1: 系统过载保护（硬件感知）=====
        device_type = context.system_state.device_type
        
        if device_type == "gpu":
            # GPU 模式：检查 GPU 使用率
            if context.system_state.gpu_usage > self.gpu_overload_threshold:
                return HardDecision(
                    strategy=ExecutionStrategy.CLOUD_DIRECT,
                    reason=f"GPU 过载 ({context.system_state.gpu_usage:.1f}%)"
                )
        else:
            # CPU 模式：检查 CPU 使用率
            if context.system_state.cpu_usage > self.cpu_overload_threshold:
                return HardDecision(
                    strategy=ExecutionStrategy.CLOUD_DIRECT,
                    reason=f"CPU 过载 ({context.system_state.cpu_usage:.1f}%)"
                )
        
        # 内存检查（通用）
        if context.system_state.memory_available_mb < self.memory_critical_mb:
            return HardDecision(
                strategy=ExecutionStrategy.CLOUD_DIRECT,
                reason=f"内存不足 ({context.system_state.memory_available_mb:.0f}MB)"
            )
        
        # ===== 规则2: 极端延迟要求（时敏性任务）=====
        if context.task_requirements.max_latency_ms < self.ultra_low_latency_ms:
            return HardDecision(
                strategy=ExecutionStrategy.EDGE_ONLY,
                reason=f"超低延迟要求 (<{self.ultra_low_latency_ms}ms), 无法等待云端"
            )
        
        # ===== 规则3: 隐私要求 =====
        if context.task_requirements.privacy_level >= 2:  # 绝密级别
            return HardDecision(
                strategy=ExecutionStrategy.EDGE_ONLY,
                reason="隐私敏感数据，不可上云"
            )
        
        # ===== 规则4（阶段2新增）: 弱网检测 =====
        if context.network_state:
            # 方式1: 使用is_weak_network标记
            if context.network_state.is_weak_network:
                return HardDecision(
                    strategy=ExecutionStrategy.EDGE_ONLY,
                    reason=f"弱网环境 (RTT={context.network_state.rtt_ms:.1f}ms), 避免云端调用"
                )
            
            # 方式2: 使用RTT阈值（双重保险）
            if context.network_state.rtt_ms > self.weak_network_rtt_threshold:
                return HardDecision(
                    strategy=ExecutionStrategy.EDGE_ONLY,
                    reason=f"网络延迟过高 (RTT={context.network_state.rtt_ms:.1f}ms > {self.weak_network_rtt_threshold}ms)"
                )
        
        # ===== 规则5: 紧急任务 + 质量要求不高 =====
        if (context.task_requirements.priority >= 3 and 
            context.task_requirements.min_quality_score < 0.7):
            return HardDecision(
                strategy=ExecutionStrategy.EDGE_ONLY,
                reason="紧急任务且质量要求低，快速响应优先"
            )
        
        # 无硬约束触发，进入评分阶段
        return None


class MultiObjectiveScorer:
    """多目标评分器（阶段3扩展：支持历史数据）"""
    
    def __init__(self, config: Dict[str, Any], history_tracker: Optional['HistoryTracker'] = None):
        # 权重配置（可在 config.yaml 中调整）
        weights = config.get('scoring_weights', {})
        self.w_latency = weights.get('latency', 0.4)     # 延迟权重
        self.w_cost = weights.get('cost', 0.3)           # 成本权重
        self.w_quality = weights.get('quality', 0.3)     # 质量权重
        
        # 参考值（用于归一化）- 从配置读取或使用默认值
        estimates = config.get('latency_estimates', {})
        self.ref_edge_latency_ms = estimates.get('edge_only_ms', 30.0)
        self.ref_cloud_latency_ms = estimates.get('cloud_direct_ms', 200.0)
        self.ref_speculative_latency_ms = estimates.get('speculative_standard_ms', 80.0)
        
        # 阶段3：历史追踪器（可选）
        self.history_tracker = history_tracker
        self.enable_history_scoring = config.get('enable_history_scoring', True)
    
    def score_strategies(self, context: DecisionContext) -> List[ScoredStrategy]:
        """为所有策略打分"""
        
        strategies = [
            ExecutionStrategy.EDGE_ONLY,
            ExecutionStrategy.CLOUD_DIRECT,
            ExecutionStrategy.SPECULATIVE_STANDARD,
            ExecutionStrategy.ADAPTIVE_CONFIDENCE
        ]
        
        scored = []
        for strategy in strategies:
            score = self._calculate_score(strategy, context)
            scored.append(ScoredStrategy(strategy=strategy, score=score))
        
        return scored
    
    def _calculate_score(self, 
                         strategy: ExecutionStrategy, 
                         context: DecisionContext) -> float:
        """
        计算单个策略的综合得分
        
        Score = w1 * latency_score + w2 * cost_score + w3 * quality_score
        每个维度归一化到 [0, 1]，越高越好
        """
        
        # 1. 延迟得分（越低越好 → 转换为越高越好）
        latency_score = self._score_latency(strategy, context)
        
        # 2. 成本得分（云端调用成本高）
        cost_score = self._score_cost(strategy, context)
        
        # 3. 质量得分（基于历史接受率、模型能力等）
        quality_score = self._score_quality(strategy, context)
        
        # 综合得分
        total_score = (
            self.w_latency * latency_score +
            self.w_cost * cost_score +
            self.w_quality * quality_score
        )
        
        # 考虑任务优先级调整
        if context.task_requirements.priority >= 2:
            # 高优先级任务更重视延迟
            total_score += 0.1 * latency_score
        
        return total_score
    
    def _score_latency(self, 
                       strategy: ExecutionStrategy, 
                       context: DecisionContext) -> float:
        """
        延迟得分计算（阶段3增强：使用历史实际延迟）
        
        根据策略估算延迟，并与 SLO 要求对比
        """
        # 阶段3：优先使用历史实际延迟
        if self.enable_history_scoring and self.history_tracker:
            historical_latency = self.history_tracker.get_avg_latency(strategy, n=20)
            # 如果有足够的历史数据，使用实际延迟
            if len(self.history_tracker.get_records_by_strategy(strategy, n=20)) >= 5:
                latency = historical_latency
            else:
                # 历史数据不足，使用基础估算
                base_latency = {
                    ExecutionStrategy.EDGE_ONLY: self.ref_edge_latency_ms,
                    ExecutionStrategy.CLOUD_DIRECT: self.ref_cloud_latency_ms,
                    ExecutionStrategy.SPECULATIVE_STANDARD: self.ref_speculative_latency_ms,
                    ExecutionStrategy.ADAPTIVE_CONFIDENCE: self.ref_speculative_latency_ms * 0.9
                }
                latency = base_latency.get(strategy, 100.0)
        else:
            # 估算各策略的预期延迟（基础值）
            base_latency = {
                ExecutionStrategy.EDGE_ONLY: self.ref_edge_latency_ms,
                ExecutionStrategy.CLOUD_DIRECT: self.ref_cloud_latency_ms,
                ExecutionStrategy.SPECULATIVE_STANDARD: self.ref_speculative_latency_ms,
                ExecutionStrategy.ADAPTIVE_CONFIDENCE: self.ref_speculative_latency_ms * 0.9
            }
            latency = base_latency.get(strategy, 100.0)
        
        # 阶段2新增：如果有网络状态，调整云端相关策略的延迟
        if context.network_state:
            network_rtt = context.network_state.rtt_ms
            
            if strategy == ExecutionStrategy.CLOUD_DIRECT:
                # 纯云端：延迟 = 基础延迟 + 2 * RTT（往返）
                latency = latency + 2 * network_rtt
            
            elif strategy in [ExecutionStrategy.SPECULATIVE_STANDARD, 
                             ExecutionStrategy.ADAPTIVE_CONFIDENCE]:
                # 协同推理：延迟 = 基础延迟 + 1 * RTT（单次验证）
                latency = latency + network_rtt
        
        slo_latency = context.task_requirements.max_latency_ms
        
        # 如果预期延迟超过 SLO，严重惩罚
        if latency > slo_latency:
            return 0.0
        
        # 归一化：延迟越低分数越高
        score = max(0.0, 1.0 - latency / slo_latency)
        return score
    
    def _score_cost(self, 
                    strategy: ExecutionStrategy, 
                    context: DecisionContext) -> float:
        """
        成本得分（云端调用成本最高）
        """
        cost_map = {
            ExecutionStrategy.EDGE_ONLY: 1.0,           # 最低成本
            ExecutionStrategy.SPECULATIVE_STANDARD: 0.6, # 中等（部分云端）
            ExecutionStrategy.ADAPTIVE_CONFIDENCE: 0.7,  # 略低于标准协同
            ExecutionStrategy.CLOUD_DIRECT: 0.0          # 最高成本
        }
        return cost_map.get(strategy, 0.5)
    
    def _score_quality(self, 
                       strategy: ExecutionStrategy, 
                       context: DecisionContext) -> float:
        """
        质量得分（阶段3增强：使用历史接受率和成功率）
        
        阶段1: 基于策略的预期质量
        阶段3: 结合历史统计（Draft 接受率、成功率）
        """
        # 预期质量映射（基础值）
        quality_map = {
            ExecutionStrategy.EDGE_ONLY: 0.7,           # 小模型质量较低
            ExecutionStrategy.CLOUD_DIRECT: 1.0,        # 大模型质量最高
            ExecutionStrategy.SPECULATIVE_STANDARD: 0.95, # 云端验证保证
            ExecutionStrategy.ADAPTIVE_CONFIDENCE: 0.92   # 动态调整略低
        }
        
        base_quality = quality_map.get(strategy, 0.8)
        
        # 阶段3：使用历史数据调整质量评分
        if self.enable_history_scoring and self.history_tracker:
            # 获取该策略的历史成功率
            success_rate = self.history_tracker.get_success_rate(strategy, n=20)
            
            # 对于协同推理策略，使用接受率调整质量得分
            if strategy in [ExecutionStrategy.SPECULATIVE_STANDARD, 
                           ExecutionStrategy.ADAPTIVE_CONFIDENCE]:
                acceptance_rate = self.history_tracker.get_recent_acceptance_rate(strategy, n=20)
                
                # 如果有足够的历史数据
                if len(self.history_tracker.get_records_by_strategy(strategy, n=20)) >= 5:
                    # 接受率越高，说明 Draft 质量越好
                    # 质量得分 = 基础质量 * 成功率 * (0.8 + 0.2 * 接受率)
                    quality_boost = 0.8 + 0.2 * acceptance_rate
                    base_quality = base_quality * success_rate * quality_boost
            else:
                # 非协同策略，仅用成功率调整
                if len(self.history_tracker.get_records_by_strategy(strategy, n=20)) >= 5:
                    base_quality = base_quality * success_rate
        
        # 如果任务要求高质量，提升云端相关策略的分数
        if context.task_requirements.min_quality_score > 0.9:
            if strategy in [ExecutionStrategy.CLOUD_DIRECT, 
                           ExecutionStrategy.SPECULATIVE_STANDARD]:
                base_quality = min(1.0, base_quality + 0.1)
        
        return base_quality


class DecisionEngine:
    """决策引擎（整合硬约束和评分）（阶段3扩展：支持历史追踪器）"""
    
    def __init__(self, config: Dict[str, Any], history_tracker: Optional['HistoryTracker'] = None):
        self.hard_constraint_checker = HardConstraintChecker(
            config.get('hard_constraints', {})
        )
        self.multi_objective_scorer = MultiObjectiveScorer(config, history_tracker)
    
    def check_hard_constraints(self, context: DecisionContext) -> Optional[HardDecision]:
        """检查硬约束"""
        return self.hard_constraint_checker.check(context)
    
    def score_strategies(self, context: DecisionContext) -> List[ScoredStrategy]:
        """评分所有策略"""
        return self.multi_objective_scorer.score_strategies(context)
