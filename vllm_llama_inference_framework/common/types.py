"""
公共数据类型定义
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum


class ConfidenceStrategy(Enum):
    """置信度策略"""
    MAX_PROB = "max_prob"          # 最大概率
    ENTROPY = "entropy"            # 熵值
    TEMPERATURE = "temperature"    # 温度缩放
    TOP_K_AGG = "top_k_agg"        # Top-K 聚合


class MessageType(Enum):
    """消息类型"""
    DRAFT_REQUEST = "draft_request"
    DRAFT_RESPONSE = "draft_response"
    VERIFY_REQUEST = "verify_request"
    VERIFY_RESPONSE = "verify_response"
    HEALTH_CHECK = "health_check"


@dataclass
class TokenProb:
    """Token 概率信息"""
    token_id: int
    token: str
    prob: float
    logprob: float


@dataclass
class ConfidenceMetrics:
    """置信度指标"""
    confidence_score: float
    strategy: ConfidenceStrategy
    token_probs: List[TokenProb]
    entropy: float
    max_prob: float
    min_prob: float
    avg_prob: float


@dataclass
class DraftRequest:
    """Draft 生成请求"""
    prompt: str
    max_tokens: int = 64
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 50
    confidence_threshold: float = 0.8
    strategy: ConfidenceStrategy = ConfidenceStrategy.MAX_PROB


@dataclass
class DraftResponse:
    """Draft 生成响应"""
    draft_tokens: List[str]
    draft_token_ids: List[int]
    confidence: ConfidenceMetrics
    kv_cache_info: Optional[Dict[str, Any]] = None
    latency_ms: float = 0.0


@dataclass
class VerifyRequest:
    """Draft 验证请求"""
    prompt: str
    draft_tokens: List[str]
    draft_token_ids: List[int]
    max_verify_tokens: int = 64
    temperature: float = 0.0  # 验证时通常用贪心
    confidence_threshold: float = 0.8


@dataclass
class VerifyResponse:
    """Draft 验证响应"""
    verified_tokens: List[str]
    verified_token_ids: List[int]
    accepted_count: int
    total_count: int
    acceptance_rate: float
    corrected_positions: List[int]  # 修正的位置
    final_text: str
    latency_ms: float = 0.0


@dataclass
class InferenceRequest:
    """推理请求"""
    prompt: str
    max_tokens: int = 128
    temperature: float = 0.8
    top_p: float = 0.95
    use_draft_verify: bool = True
    use_confidence_check: bool = True
    confidence_threshold: float = 0.8


@dataclass
class InferenceResponse:
    """推理响应"""
    text: str
    tokens: List[str]
    total_latency_ms: float
    edge_latency_ms: float = 0.0
    cloud_latency_ms: float = 0.0
    acceptance_rate: float = 0.0
    confidence_score: float = 0.0


@dataclass
class KVCacheInfo:
    """KV Cache 信息"""
    cache_size: int
    hit_tokens: int
    miss_tokens: int
    hit_rate: float
    cache_keys: Optional[List[str]] = None
