"""
公共数据类型定义 - 最终兼容修复版
同时支持 Monitor/Decision 模块和原有的 Draft/Confidence 模块
"""
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum

# ==================== 1. 策略与枚举定义 ====================

class ConfidenceStrategy(Enum):
    MAX_PROB = "max_prob"
    ENTROPY = "entropy"
    TEMPERATURE = "temperature"
    TOP_K_AGG = "top_k_agg"

class MessageType(Enum):
    DRAFT_REQUEST = "draft_request"
    DRAFT_RESPONSE = "draft_response"
    VERIFY_REQUEST = "verify_request"
    VERIFY_RESPONSE = "verify_response"
    HEALTH_CHECK = "health_check"
    DIRECT_INFERENCE = "direct_inference" # 新增：直接推理消息类型

class ExecutionStrategy(Enum):
    EDGE_ONLY = "edge_only"             # F1: 纯端侧
    CLOUD_DIRECT = "cloud_direct"       # 纯云端 (Baseline)
    SPECULATIVE_STANDARD = "speculative_standard" # F1+F3: 标准协同
    ADAPTIVE_CONFIDENCE = "adaptive_confidence"   # F1+F2+F3: 自适应协同

# ==================== 2. 状态感知与任务需求 (新模块) ====================

@dataclass
class TaskRequirements:
    """任务需求 (SLA/SLO)"""
    max_latency_ms: int = 5000    # 最大容忍延迟
    min_accuracy: float = 0.9     # 精度要求
    privacy_level: int = 0        # 隐私等级: 0=公开, 1=敏感, 2=绝密
    priority: int = 1             # 优先级

@dataclass
class SystemStats:
    """硬件状态"""
    cpu_usage: float        # CPU使用率 (%)
    memory_available: float # 剩余内存 (MB)
    gpu_memory_free: float  # 剩余显存 (MB)
    gpu_utilization: float  # GPU利用率 (%)

@dataclass
class NetworkStats:
    """网络状态"""
    rtt_ms: float           # 往返时延
    bandwidth_up: float     # 上行带宽 (Mbps)
    bandwidth_down: float   # 下行带宽 (Mbps)
    stability: float        # 链路稳定性 (0.0-1.0)
    is_weak_network: bool   # 是否判定为弱网

# ==================== 3. 基础推理结构 (兼容旧代码) ====================

@dataclass
class TokenProb:
    token_id: int
    token: str
    prob: float
    logprob: float

# [修复点] 加回 ConfidenceScore 以修复 ImportError
@dataclass
class ConfidenceScore:
    """(旧版兼容) 简单置信度分数"""
    logprobs: List[float] = field(default_factory=list)
    cumulative_logprob: float = 0.0
    confidence_score: float = 0.0

@dataclass
class ConfidenceMetrics:
    """(新版) 详细置信度指标"""
    confidence_score: float
    strategy: ConfidenceStrategy
    token_probs: List[TokenProb] = field(default_factory=list)
    entropy: float = 0.0
    max_prob: float = 0.0
    min_prob: float = 0.0
    avg_prob: float = 0.0

# ==================== 4. 请求与响应结构 ====================

@dataclass
class DraftRequest:
    prompt: str
    max_tokens: int = 64
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 50
    confidence_threshold: float = 0.8
    strategy: ConfidenceStrategy = ConfidenceStrategy.MAX_PROB

@dataclass
class DraftResponse:
    draft_tokens: List[str]
    draft_token_ids: List[int]
    # [修复点] 允许 ConfidenceScore 或 ConfidenceMetrics
    confidence: Union[ConfidenceScore, ConfidenceMetrics] 
    kv_cache_info: Optional[Dict[str, Any]] = None
    latency_ms: float = 0.0

@dataclass
class VerifyRequest:
    prompt: str
    draft_tokens: List[str]
    draft_token_ids: List[int]
    max_verify_tokens: int = 64
    temperature: float = 0.0
    confidence_threshold: float = 0.8

@dataclass
class VerifyResponse:
    verified_tokens: List[str]
    verified_token_ids: List[int]
    accepted_count: int
    total_count: int
    acceptance_rate: float
    corrected_positions: List[int]
    final_text: str
    latency_ms: float = 0.0

@dataclass
class InferenceRequest:
    prompt: str
    max_tokens: int = 128
    temperature: float = 0.8
    top_p: float = 0.95
    use_draft_verify: bool = True
    use_confidence_check: bool = True
    confidence_threshold: float = 0.8
    # 新增需求字段
    requirements: TaskRequirements = field(default_factory=TaskRequirements)