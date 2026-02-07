"""
阶段3: AdaptiveThresholdCalculator - 自适应阈值计算器
基于历史统计动态调整决策参数
"""
from typing import Dict, Optional
from edge.history_tracker import HistoryTracker
from common.types import ExecutionStrategy


class AdaptiveThresholdCalculator:
    """
    自适应阈值计算器
    
    根据历史执行数据动态调整:
    1. confidence_threshold (置信度阈值)
    2. draft_max_tokens (Draft 长度)
    3. 评分权重
    """
    
    def __init__(self, config: Dict):
        """
        初始化自适应计算器
        
        Args:
            config: 配置字典
        """
        self.config = config
        
        # 目标接受率范围
        self.target_acceptance_min = config.get('target_acceptance_min', 0.80)
        self.target_acceptance_max = config.get('target_acceptance_max', 0.90)
        
        # 调整步长
        self.threshold_step = config.get('threshold_step', 0.05)
        self.smoothing_factor = config.get('smoothing_factor', 0.1)  # 平滑系数
        
        # 阈值范围限制
        self.threshold_min = config.get('threshold_min', 0.50)
        self.threshold_max = config.get('threshold_max', 0.95)
        
        # 当前自适应值（初始值从配置读取）
        self.current_confidence_threshold = config.get('initial_confidence_threshold', 0.80)
        
        # 更新计数器（每 N 次执行更新一次参数）
        self.update_interval = config.get('update_interval', 10)
        self._execution_count = 0
    
    def should_update(self) -> bool:
        """
        判断是否应该更新参数
        
        Returns:
            是否到达更新时机
        """
        self._execution_count += 1
        return self._execution_count % self.update_interval == 0
    
    def calculate_adaptive_confidence_threshold(self,
                                                 history: HistoryTracker,
                                                 current_threshold: float) -> float:
        """
        计算自适应置信度阈值
        
        算法:
        1. 获取最近 20 次 SPECULATIVE 策略的接受率
        2. 如果接受率 > 目标上限 → 降低阈值（可以更激进）
        3. 如果接受率 < 目标下限 → 提高阈值（需要更保守）
        4. 使用移动平均平滑调整
        
        Args:
            history: 历史追踪器
            current_threshold: 当前阈值
        
        Returns:
            新的置信度阈值
        """
        # 获取最近的接受率
        recent_ar = history.get_recent_acceptance_rate(
            strategy=ExecutionStrategy.SPECULATIVE_STANDARD,
            n=20
        )
        
        # 如果样本不足，保持当前值
        if len(history.get_records_by_strategy(ExecutionStrategy.SPECULATIVE_STANDARD, n=20)) < 5:
            return current_threshold
        
        # 计算调整量
        adjustment = 0.0
        
        if recent_ar > self.target_acceptance_max:
            # 接受率过高，降低阈值
            overshoot = recent_ar - self.target_acceptance_max
            adjustment = -self.threshold_step * (overshoot / 0.1)  # 归一化
            reason = f"接受率过高 ({recent_ar:.2%}), 降低阈值"
        
        elif recent_ar < self.target_acceptance_min:
            # 接受率过低，提高阈值
            undershoot = self.target_acceptance_min - recent_ar
            adjustment = self.threshold_step * (undershoot / 0.1)
            reason = f"接受率过低 ({recent_ar:.2%}), 提高阈值"
        
        else:
            # 在目标范围内，保持稳定
            reason = f"接受率正常 ({recent_ar:.2%}), 保持阈值"
            adjustment = 0.0
        
        # 使用移动平均平滑调整
        new_threshold = current_threshold * (1 - self.smoothing_factor) + \
                        (current_threshold + adjustment) * self.smoothing_factor
        
        # 限制在合理范围内
        new_threshold = max(self.threshold_min, min(self.threshold_max, new_threshold))
        
        # 打印调整信息
        if abs(new_threshold - current_threshold) > 0.01:
            print(f"[Adaptive] {reason}")
            print(f"[Adaptive] 阈值调整: {current_threshold:.3f} → {new_threshold:.3f}")
        
        return new_threshold
    
    def calculate_adaptive_draft_length(self,
                                         history: HistoryTracker,
                                         current_length: int,
                                         task_latency_requirement: int) -> int:
        """
        计算自适应 Draft 长度
        
        算法:
        1. 如果平均延迟接近 SLO 上限 → 减少 draft 长度
        2. 如果平均延迟远低于 SLO → 可以增加 draft 长度（提高质量）
        
        Args:
            history: 历史追踪器
            current_length: 当前 draft 长度
            task_latency_requirement: 任务延迟要求 (ms)
        
        Returns:
            新的 draft 长度
        """
        # 获取最近的平均延迟
        recent_latency = history.get_avg_latency(
            strategy=ExecutionStrategy.SPECULATIVE_STANDARD,
            n=20
        )
        
        # 计算延迟余量
        latency_margin = task_latency_requirement - recent_latency
        latency_ratio = latency_margin / task_latency_requirement
        
        if latency_ratio < 0.1:  # 延迟紧张
            new_length = max(32, current_length - 8)
        elif latency_ratio > 0.5:  # 延迟充裕
            new_length = min(128, current_length + 8)
        else:
            new_length = current_length
        
        return new_length
    
    def calculate_adaptive_weights(self,
                                    history: HistoryTracker,
                                    current_weights: Dict[str, float]) -> Dict[str, float]:
        """
        计算自适应评分权重
        
        算法:
        1. 根据各策略的历史成功率调整权重
        2. 如果某策略成功率低，降低其相关维度的权重
        
        Args:
            history: 历史追踪器
            current_weights: 当前权重 {'latency': 0.4, 'cost': 0.3, 'quality': 0.3}
        
        Returns:
            新的权重字典
        """
        # 阶段3 MVP：保持权重不变，避免过于复杂
        # 未来可以根据策略成功率微调
        return current_weights
    
    def update_parameters(self,
                          history: HistoryTracker,
                          current_params: Dict) -> Dict:
        """
        统一更新所有自适应参数
        
        Args:
            history: 历史追踪器
            current_params: 当前参数字典
        
        Returns:
            更新后的参数字典
        """
        new_params = current_params.copy()
        
        # 更新置信度阈值
        if 'confidence_threshold' in current_params:
            new_params['confidence_threshold'] = self.calculate_adaptive_confidence_threshold(
                history,
                current_params['confidence_threshold']
            )
            # 缓存当前值
            self.current_confidence_threshold = new_params['confidence_threshold']
        
        # 更新 draft 长度（需要任务 SLO）
        if 'draft_max_tokens' in current_params and 'task_latency_slo' in current_params:
            new_params['draft_max_tokens'] = self.calculate_adaptive_draft_length(
                history,
                current_params['draft_max_tokens'],
                current_params['task_latency_slo']
            )
        
        # 更新权重
        if 'scoring_weights' in current_params:
            new_params['scoring_weights'] = self.calculate_adaptive_weights(
                history,
                current_params['scoring_weights']
            )
        
        return new_params
    
    def get_current_threshold(self) -> float:
        """获取当前的置信度阈值"""
        return self.current_confidence_threshold
