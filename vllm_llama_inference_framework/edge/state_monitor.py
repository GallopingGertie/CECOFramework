"""
F1 状态监控器
负责采集系统硬件状态和提取任务 SLO 需求
"""
import time
import psutil
from typing import Dict, Any, Optional
from common.types import SystemStats, TaskRequirements, InferenceRequest


class SystemResourceMonitor:
    """系统资源监控器（基于 psutil）"""
    
    def __init__(self, config: Dict[str, Any]):
        self.enable_gpu = config.get('monitor_gpu', False)
        
        # GPU 监控（可选）
        self.gpu_monitor = None
        if self.enable_gpu:
            try:
                import pynvml
                pynvml.nvmlInit()
                self.gpu_monitor = pynvml
                print("[StateMonitor] GPU 监控已启用")
            except Exception as e:
                print(f"[StateMonitor] GPU 监控不可用: {e}")
    
    def sample(self) -> SystemStats:
        """
        采样当前系统状态
        
        性能：通常 < 1ms
        """
        # CPU 使用率（interval=None 使用上次缓存，非阻塞）
        cpu_usage = psutil.cpu_percent(interval=None)
        
        # 内存状态
        mem = psutil.virtual_memory()
        memory_available_mb = mem.available / (1024 * 1024)
        
        # GPU 状态（如果启用）
        gpu_memory_free_mb = 0.0
        if self.gpu_monitor:
            try:
                handle = self.gpu_monitor.nvmlDeviceGetHandleByIndex(0)
                mem_info = self.gpu_monitor.nvmlDeviceGetMemoryInfo(handle)
                gpu_memory_free_mb = mem_info.free / (1024 * 1024)
            except Exception:
                pass
        
        return SystemStats(
            cpu_usage=cpu_usage,
            memory_available_mb=memory_available_mb,
            gpu_memory_free_mb=gpu_memory_free_mb,
            timestamp=time.time()
        )
    
    def get_stats_summary(self) -> Dict[str, Any]:
        """获取统计摘要（用于日志/调试）"""
        state = self.sample()
        return {
            'cpu_usage': f"{state.cpu_usage:.1f}%",
            'memory_available': f"{state.memory_available_mb:.0f}MB",
            'gpu_memory_free': f"{state.gpu_memory_free_mb:.0f}MB" if state.gpu_memory_free_mb > 0 else "N/A"
        }


class TaskAnalyzer:
    """任务需求分析器"""
    
    def __init__(self, config: Dict[str, Any]):
        # 默认 SLO 配置
        self.default_max_latency_ms = config.get('default_max_latency_ms', 5000)
        self.default_min_quality = config.get('default_min_quality', 0.8)
    
    def analyze(self, request: InferenceRequest) -> TaskRequirements:
        """
        从 InferenceRequest 中提取或推断任务需求
        
        优先级：
        1. 如果 request 中有 requirements 字段，直接使用
        2. 否则根据请求特征推断（启发式规则）
        """
        
        # 方式1: 显式指定（推荐）
        if hasattr(request, 'requirements') and request.requirements:
            return request.requirements
        
        # 方式2: 启发式推断
        return self._infer_requirements(request)
    
    def _infer_requirements(self, request: InferenceRequest) -> TaskRequirements:
        """
        启发式推断任务需求
        
        规则示例：
        - 短文本 + 低 temperature → 时延敏感（如对话）
        - 长文本 + 高 temperature → 质量优先（如创作）
        """
        
        # 基于 prompt 长度推断
        prompt_len = len(request.prompt)
        
        if prompt_len < 50:  # 短 prompt，可能是对话
            max_latency = 1000  # 1秒
            priority = 2
        elif prompt_len < 200:  # 中等长度
            max_latency = 3000
            priority = 1
        else:  # 长文本
            max_latency = self.default_max_latency_ms
            priority = 1
        
        # 基于 temperature 推断质量要求
        if request.temperature < 0.3:  # 确定性任务
            min_quality = 0.9
        else:
            min_quality = self.default_min_quality
        
        # 隐私等级默认为 0（可以通过关键词检测等方式推断）
        privacy_level = 0
        
        return TaskRequirements(
            max_latency_ms=max_latency,
            min_quality_score=min_quality,
            priority=priority,
            privacy_level=privacy_level
        )


class StateMonitor:
    """状态监控器（阶段1：硬件 + SLO）"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # 子模块
        self.resource_monitor = SystemResourceMonitor(config)
        self.task_analyzer = TaskAnalyzer(config)
        
        # 缓存机制（避免频繁采样）
        self._cache_ttl_ms = config.get('state_cache_ttl_ms', 100)  # 100ms缓存
        self._last_system_state: Optional[SystemStats] = None
        self._last_update_time: float = 0.0
        
        # 启动时采样一次，初始化 psutil 的缓存
        psutil.cpu_percent(interval=None)
    
    def get_current_state(self) -> SystemStats:
        """
        获取当前系统状态（带缓存）
        
        如果距离上次采样不到 100ms，直接返回缓存
        """
        now = time.time()
        if (self._last_system_state and 
            (now - self._last_update_time) * 1000 < self._cache_ttl_ms):
            return self._last_system_state
        
        # 重新采样
        self._last_system_state = self.resource_monitor.sample()
        self._last_update_time = now
        return self._last_system_state
    
    def extract_task_requirements(self, 
                                   request: InferenceRequest) -> TaskRequirements:
        """
        从请求中提取任务 SLO 需求
        """
        return self.task_analyzer.analyze(request)
