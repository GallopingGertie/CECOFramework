"""
公共数据类型定义 - 最终完整版
整合了 F1 决策、F2 协同、F3 缓存、F4 通信的所有数据结构
"""
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum

# ==================== 1. 枚举定义 ====================

class ConfidenceStrategy(Enum):
    """置信度计算策略"""
    MAX_PROB = "max_prob"
    ENTROPY = "entropy"
    TEMPERATURE = "temperature"
    TOP_K_AGG = "top_k_agg"

class MessageType(Enum):
    """通信消息类型"""
    DRAFT_REQUEST = "draft_request"
    DRAFT_RESPONSE = "draft_response"
    VERIFY_REQUEST = "verify_request"
    VERIFY_RESPONSE = "verify_response"
    HEALTH_CHECK = "health_check"
    DIRECT_INFERENCE = "direct_inference"

class ExecutionStrategy(Enum):
    """执行策略"""
    EDGE_ONLY = "edge_only"             # 纯端侧
    CLOUD_DIRECT = "cloud_direct"       # 纯云端
    SPECULATIVE_STANDARD = "speculative_standard" # 标准协同 (无置信度检查)
    ADAPTIVE_CONFIDENCE = "adaptive_confidence"   # 自适应协同 (有置信度检查)

# ==================== 2. 基础数据结构 ====================

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
    entropy: float = 0.0
    max_prob: float = 0.0
    min_prob: float = 0.0
    avg_prob: float = 0.0

@dataclass
class KVCacheInfo:
    """KV Cache 统计信息"""
    cache_size: int = 0
    hit_tokens: int = 0
    miss_tokens: int = 0
    hit_rate: float = 0.0
    info: str = ""

# ==================== 3. 任务与状态定义 (F1 核心) ====================

@dataclass
class TaskRequirements:
    """任务需求 (SLO)"""
    max_latency_ms: int = 5000        # 最大容忍延迟
    min_quality_score: float = 0.8    # 最低质量要求
    priority: int = 1                 # 优先级 (1-5, 5最高)
    privacy_level: int = 0            # 隐私等级 (0:公开, 1:敏感, 2:机密)

@dataclass
class SystemStats:
    """系统硬件状态（扩展：支持GPU监控）"""
    cpu_usage: float
    memory_available_mb: float
    gpu_memory_free_mb: float = 0.0
    gpu_usage: float = 0.0  # 新增：GPU使用率
    device_type: str = "cpu"  # 新增：设备类型（"cpu" 或 "gpu"）
    timestamp: float = 0.0

@dataclass
class NetworkStats:
    """网络状态（阶段2新增）"""
    rtt_ms: float  # 往返延迟
    bandwidth_mbps: float = 0.0  # 带宽（Mbps）
    packet_loss_rate: float = 0.0  # 丢包率
    is_weak_network: bool = False  # 是否弱网

# ==================== 4. 请求与响应定义 ====================

@dataclass
class InferenceRequest:
    """推理请求 (客户端 -> Edge)"""
    prompt: str
    max_tokens: int = 128
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 40
    
    # 功能开关
    use_draft_verify: bool = True
    use_confidence_check: bool = True
    
    # [新增] 必须加上这行，否则 main.py 传过来的参数会被拒
    confidence_threshold: float = 0.8 
    
    # 任务需求 (F1)
    requirements: TaskRequirements = field(default_factory=TaskRequirements)

@dataclass
class DraftRequest:
    """Draft 生成请求 (Edge 内部 或 Edge -> Mock)"""
    prompt: str
    max_tokens: int = 64
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 40
    confidence_threshold: float = 0.8

@dataclass
class DraftResponse:
    """Draft 生成响应"""
    draft_tokens: List[str]
    draft_token_ids: List[int]
    confidence: ConfidenceMetrics
    kv_cache_info: Dict[str, Any]
    latency_ms: float

@dataclass
class VerifyRequest:
    """验证请求 (Edge -> Cloud)"""
    prompt: str
    draft_tokens: List[str]
    draft_token_ids: List[int]
    confidence_threshold: float = 0.8

@dataclass
class VerifyResponse:
    """验证响应 (Cloud -> Edge)"""
    verified_tokens: List[str]
    verified_token_ids: List[int]
    accepted_count: int
    total_count: int
    acceptance_rate: float
    corrected_positions: List[int]
    final_text: str
    latency_ms: float = 0.0

# ==================== 5. 决策模块定义 (F1) ====================

@dataclass
class DecisionContext:
    """决策上下文（阶段2扩展：加入网络状态）"""
    request: InferenceRequest
    system_state: SystemStats
    task_requirements: TaskRequirements
    network_state: Optional['NetworkStats'] = None  # 阶段2新增

@dataclass
class ExecutionPlan:
    """执行计划"""
    strategy: ExecutionStrategy
    params: Dict[str, Any]
    confidence_threshold: float = 0.8
    draft_max_tokens: int = 64
    reason: str = ""
    score: float = 0.0

@dataclass
class HardDecision:
    """硬约束结果"""
    strategy: ExecutionStrategy
    reason: str

@dataclass
class ScoredStrategy:
    """策略评分"""
    strategy: ExecutionStrategy
    score: float

# ==================== 6. 最终响应定义 (F4) ====================

@dataclass
class InferenceResponse:
    """推理响应结果 (用于 HTTP 客户端解析)"""
    text: str
    tokens: List[str]
    total_latency_ms: float
    confidence_score: float = 0.0
    acceptance_rate: float = 0.0
    edge_latency_ms: float = 0.0
    cloud_latency_ms: float = 0.0