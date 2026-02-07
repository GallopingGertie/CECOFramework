"""
阶段3: HistoryTracker - 历史执行统计模块
记录和统计推理执行历史，支持自适应决策
"""
import time
from typing import Dict, List, Optional, Deque
from dataclasses import dataclass, field
from collections import deque
import statistics

from common.types import ExecutionStrategy


@dataclass
class ExecutionRecord:
    """单次执行记录"""
    timestamp: float
    strategy: ExecutionStrategy
    acceptance_rate: float  # Draft 接受率 (0.0-1.0)
    latency_ms: float       # 总延迟
    edge_latency_ms: float  # 边端延迟
    cloud_latency_ms: float # 云端延迟
    confidence_score: float # 置信度分数
    success: bool           # 是否成功完成
    tokens_generated: int = 0  # 生成的 token 数


class HistoryTracker:
    """
    历史执行追踪器
    
    使用滑动窗口记录最近 N 次执行，提供统计分析功能
    """
    
    def __init__(self, max_history_size: int = 100):
        """
        初始化历史追踪器
        
        Args:
            max_history_size: 滑动窗口大小（默认保留最近100条记录）
        """
        self.max_history_size = max_history_size
        self.history: Deque[ExecutionRecord] = deque(maxlen=max_history_size)
        
        # 分策略统计（加速查询）
        self._strategy_stats: Dict[ExecutionStrategy, List[ExecutionRecord]] = {
            strategy: [] for strategy in ExecutionStrategy
        }
    
    def add_record(self, record: ExecutionRecord):
        """
        添加一条执行记录
        
        Args:
            record: 执行记录
        """
        self.history.append(record)
        
        # 更新分策略统计（保持窗口大小）
        strategy_list = self._strategy_stats[record.strategy]
        strategy_list.append(record)
        if len(strategy_list) > self.max_history_size:
            strategy_list.pop(0)
    
    def get_recent_records(self, n: int = 20) -> List[ExecutionRecord]:
        """
        获取最近 n 条记录
        
        Args:
            n: 记录数量
        
        Returns:
            最近 n 条执行记录
        """
        return list(self.history)[-n:]
    
    def get_records_by_strategy(self, 
                                 strategy: ExecutionStrategy,
                                 n: Optional[int] = None) -> List[ExecutionRecord]:
        """
        获取特定策略的历史记录
        
        Args:
            strategy: 执行策略
            n: 最多返回的记录数（None 表示全部）
        
        Returns:
            该策略的历史记录
        """
        records = self._strategy_stats[strategy]
        if n is None:
            return records
        return records[-n:]
    
    def get_recent_acceptance_rate(self,
                                    strategy: Optional[ExecutionStrategy] = None,
                                    n: int = 20) -> float:
        """
        获取最近 n 次的平均接受率
        
        Args:
            strategy: 指定策略（None 表示所有策略）
            n: 统计最近 n 次
        
        Returns:
            平均接受率 (0.0-1.0)
        """
        if strategy:
            records = self.get_records_by_strategy(strategy, n)
        else:
            records = self.get_recent_records(n)
        
        if not records:
            return 0.8  # 默认值
        
        # 只统计使用了验证的记录（SPECULATIVE/ADAPTIVE）
        valid_records = [
            r for r in records
            if r.strategy in [ExecutionStrategy.SPECULATIVE_STANDARD,
                             ExecutionStrategy.ADAPTIVE_CONFIDENCE]
        ]
        
        if not valid_records:
            return 0.8
        
        acceptance_rates = [r.acceptance_rate for r in valid_records]
        return statistics.mean(acceptance_rates)
    
    def get_avg_latency(self,
                        strategy: Optional[ExecutionStrategy] = None,
                        n: int = 20) -> float:
        """
        获取平均延迟
        
        Args:
            strategy: 指定策略（None 表示所有策略）
            n: 统计最近 n 次
        
        Returns:
            平均延迟 (ms)
        """
        if strategy:
            records = self.get_records_by_strategy(strategy, n)
        else:
            records = self.get_recent_records(n)
        
        if not records:
            return 100.0  # 默认值
        
        latencies = [r.latency_ms for r in records]
        return statistics.mean(latencies)
    
    def get_success_rate(self,
                         strategy: Optional[ExecutionStrategy] = None,
                         n: int = 20) -> float:
        """
        获取成功率
        
        Args:
            strategy: 指定策略（None 表示所有策略）
            n: 统计最近 n 次
        
        Returns:
            成功率 (0.0-1.0)
        """
        if strategy:
            records = self.get_records_by_strategy(strategy, n)
        else:
            records = self.get_recent_records(n)
        
        if not records:
            return 1.0  # 默认值
        
        success_count = sum(1 for r in records if r.success)
        return success_count / len(records)
    
    def get_confidence_distribution(self, n: int = 50) -> Dict[str, float]:
        """
        获取置信度分布统计
        
        Args:
            n: 统计最近 n 次
        
        Returns:
            置信度统计 (mean, min, max, stdev)
        """
        records = self.get_recent_records(n)
        
        if not records:
            return {'mean': 0.8, 'min': 0.0, 'max': 1.0, 'stdev': 0.0}
        
        confidence_scores = [r.confidence_score for r in records]
        
        return {
            'mean': statistics.mean(confidence_scores),
            'min': min(confidence_scores),
            'max': max(confidence_scores),
            'stdev': statistics.stdev(confidence_scores) if len(confidence_scores) > 1 else 0.0
        }
    
    def get_strategy_distribution(self, n: int = 50) -> Dict[str, float]:
        """
        获取策略使用分布
        
        Args:
            n: 统计最近 n 次
        
        Returns:
            各策略使用比例
        """
        records = self.get_recent_records(n)
        
        if not records:
            return {}
        
        strategy_counts = {}
        for record in records:
            key = record.strategy.value
            strategy_counts[key] = strategy_counts.get(key, 0) + 1
        
        total = len(records)
        return {k: v / total for k, v in strategy_counts.items()}
    
    def get_statistics_summary(self) -> Dict:
        """
        获取完整统计摘要
        
        Returns:
            包含各种统计指标的字典
        """
        return {
            'total_records': len(self.history),
            'recent_acceptance_rate': self.get_recent_acceptance_rate(n=20),
            'avg_latency_ms': self.get_avg_latency(n=20),
            'success_rate': self.get_success_rate(n=20),
            'confidence_distribution': self.get_confidence_distribution(n=50),
            'strategy_distribution': self.get_strategy_distribution(n=50),
            'by_strategy': {
                strategy.value: {
                    'count': len(self.get_records_by_strategy(strategy)),
                    'avg_acceptance_rate': self.get_recent_acceptance_rate(strategy, n=20),
                    'avg_latency_ms': self.get_avg_latency(strategy, n=20),
                    'success_rate': self.get_success_rate(strategy, n=20)
                }
                for strategy in ExecutionStrategy
            }
        }
    
    def clear(self):
        """清空历史记录"""
        self.history.clear()
        for strategy_list in self._strategy_stats.values():
            strategy_list.clear()
