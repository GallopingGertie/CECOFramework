"""
F1: 置信度判断逻辑
支持多种置信度计算策略，模块化设计便于消融实验
"""
import math
# ==================== 修改点: 补全 Dict, Any, Optional ====================
from typing import List, Tuple, Dict, Any, Optional
# ========================================================================
import numpy as np
from common.types import (
    TokenProb, 
    ConfidenceMetrics, 
    ConfidenceStrategy
)


class ConfidenceCalculator:
    """置信度计算器"""
    
    def __init__(self, strategy: ConfidenceStrategy = ConfidenceStrategy.MAX_PROB):
        self.strategy = strategy
        self.temperature = 1.0
        
    def calculate_confidence(self, token_probs: List[TokenProb]) -> ConfidenceMetrics:
        """
        计算置信度指标
        
        Args:
            token_probs: Token概率列表
            
        Returns:
            ConfidenceMetrics: 置信度指标
        """
        if not token_probs:
            return ConfidenceMetrics(
                confidence_score=0.0,
                strategy=self.strategy,
                token_probs=[],
                entropy=0.0,
                max_prob=0.0,
                min_prob=0.0,
                avg_prob=0.0
            )
        
        # 提取概率值
        probs = [tp.prob for tp in token_probs]
        # logprobs = [tp.logprob for tp in token_probs] # 暂时未使用
        
        # 计算基本统计
        max_prob = max(probs)
        min_prob = min(probs)
        avg_prob = np.mean(probs)
        
        # 计算熵值
        entropy = -sum(p * math.log(p + 1e-10) for p in probs if p > 0)
        
        # 根据策略计算置信度分数
        confidence_score = self._calculate_by_strategy(
            token_probs, max_prob, entropy
        )
        
        return ConfidenceMetrics(
            confidence_score=confidence_score,
            strategy=self.strategy,
            token_probs=token_probs,
            entropy=entropy,
            max_prob=max_prob,
            min_prob=min_prob,
            avg_prob=avg_prob
        )
    
    def _calculate_by_strategy(
        self, 
        token_probs: List[TokenProb], 
        max_prob: float, 
        entropy: float
    ) -> float:
        """根据策略计算置信度分数"""
        if self.strategy == ConfidenceStrategy.MAX_PROB:
            return self._max_prob_strategy(token_probs)
        elif self.strategy == ConfidenceStrategy.ENTROPY:
            return self._entropy_strategy(token_probs, entropy)
        elif self.strategy == ConfidenceStrategy.TEMPERATURE:
            return self._temperature_strategy(token_probs)
        elif self.strategy == ConfidenceStrategy.TOP_K_AGG:
            return self._top_k_agg_strategy(token_probs)
        else:
            return max_prob
    
    def _max_prob_strategy(self, token_probs: List[TokenProb]) -> float:
        """
        最大概率策略：取所有token最大概率的平均值
        """
        max_probs = [tp.prob for tp in token_probs]
        return np.mean(max_probs) if max_probs else 0.0
    
    def _entropy_strategy(
        self, 
        token_probs: List[TokenProb], 
        entropy: float
    ) -> float:
        """
        熵值策略：熵越低，置信度越高
        归一化后取 (1 - normalized_entropy)
        """
        # 归一化熵值 (假设最大熵为 log(词汇表大小)，这里简化为10)
        max_entropy = 10.0
        normalized_entropy = min(entropy / max_entropy, 1.0)
        return 1.0 - normalized_entropy
    
    def _temperature_strategy(self, token_probs: List[TokenProb]) -> float:
        """
        温度缩放策略：使用温度参数调整概率分布
        """
        logprobs = np.array([tp.logprob for tp in token_probs])
        
        # 应用温度缩放
        scaled_logits = logprobs / self.temperature
        
        # Softmax
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
        softmax_probs = exp_logits / np.sum(exp_logits)
        
        # 取最大概率作为置信度
        return float(np.max(softmax_probs))
    
    def _top_k_agg_strategy(
        self, 
        token_probs: List[TokenProb], 
        k: int = 5
    ) -> float:
        """
        Top-K 聚合策略：取Top-K概率的加权平均
        """
        all_probs = []
        for tp in token_probs:
            # 这里简化处理，实际应从完整的概率分布中取Top-K
            all_probs.append(tp.prob)
        
        # 取Top-K并加权平均
        top_k = sorted(all_probs, reverse=True)[:k]
        weights = [1.0 / (i + 1) for i in range(len(top_k))]  # 递减权重
        
        if sum(weights) == 0:
            return 0.0
            
        weighted_avg = sum(w * p for w, p in zip(weights, top_k)) / sum(weights)
        return weighted_avg
    
    def should_accept_draft(
        self, 
        confidence_metrics: ConfidenceMetrics, 
        threshold: float = 0.8
    ) -> bool:
        """
        判断是否接受生成的Draft
        
        Args:
            confidence_metrics: 置信度指标
            threshold: 置信度阈值
            
        Returns:
            bool: 是否接受
        """
        return confidence_metrics.confidence_score >= threshold
    
    def get_confidence_report(
        self, 
        confidence_metrics: ConfidenceMetrics
    ) -> str:
        """获取置信度报告"""
        report = f"""
置信度分析报告:
- 策略: {confidence_metrics.strategy.value}
- 置信度分数: {confidence_metrics.confidence_score:.4f}
- 平均概率: {confidence_metrics.avg_prob:.4f}
- 最大概率: {confidence_metrics.max_prob:.4f}
- 最小概率: {confidence_metrics.min_prob:.4f}
- 熵值: {confidence_metrics.entropy:.4f}
- Token数量: {len(confidence_metrics.token_probs)}
        """
        return report.strip()


class ConfidenceEnsemble:
    """集成多种置信度策略"""
    
    def __init__(self, strategies: List[ConfidenceStrategy] = None):
        if strategies is None:
            strategies = [
                ConfidenceStrategy.MAX_PROB,
                ConfidenceStrategy.ENTROPY,
                ConfidenceStrategy.TOP_K_AGG
            ]
        self.calculators = {
            strategy: ConfidenceCalculator(strategy)
            for strategy in strategies
        }
    
    def ensemble_confidence(
        self, 
        token_probs: List[TokenProb],
        weights: List[float] = None
    ) -> Tuple[float, Dict[str, float]]:
        """
        集成多种策略计算置信度
        
        Returns:
            (ensemble_score, individual_scores)
        """
        if weights is None:
            weights = [1.0] * len(self.calculators)
        
        individual_scores = {}
        weighted_sum = 0.0
        weight_sum = 0.0
        
        for i, (strategy, calculator) in enumerate(self.calculators.items()):
            metrics = calculator.calculate_confidence(token_probs)
            individual_scores[strategy.value] = metrics.confidence_score
            weighted_sum += weights[i] * metrics.confidence_score
            weight_sum += weights[i]
        
        ensemble_score = weighted_sum / weight_sum if weight_sum > 0 else 0.0
        return ensemble_score, individual_scores


# 消融实验支持
class AblatedConfidenceCalculator(ConfidenceCalculator):
    """用于消融实验的置信度计算器（禁用特定功能）"""
    
    def __init__(
        self, 
        strategy: ConfidenceStrategy = ConfidenceStrategy.MAX_PROB,
        disable_entropy: bool = False,
        disable_normalization: bool = False
    ):
        super().__init__(strategy)
        self.disable_entropy = disable_entropy
        self.disable_normalization = disable_normalization
    
    def _entropy_strategy(self, token_probs: List[TokenProb], entropy: float) -> float:
        """重写熵策略以支持消融"""
        if self.disable_entropy:
            # 禁用熵计算，返回基础置信度
            return super()._max_prob_strategy(token_probs)
        
        if self.disable_normalization:
            # 禁用归一化，使用原始熵值
            return max(0.0, 1.0 - entropy / 100.0)  # 简单归一化
        
        return super()._entropy_strategy(token_probs, entropy)