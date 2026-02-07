"""
Monitor: 状态监视器
阶段2扩展: 真实网络探测
阶段4扩展: 硬件感知（GPU/CPU）
"""
import time
import psutil
import asyncio
import aiohttp
from typing import Optional, Dict, Any
from collections import deque
from common.types import SystemStats, NetworkStats


class StateMonitor:
    def __init__(self, cloud_endpoint: str, config: Dict[str, Any] = None):
        self.cloud_endpoint = cloud_endpoint
        self.simulation_mode = False
        self._sim_net_stats = None
        
        # 关键: 显存覆盖开关（测试用）
        self.override_memory_mb = None
        
        # 硬件配置（新增）
        config = config or {}
        hardware_config = config.get('hardware', {})
        self.device_type = hardware_config.get('device_type', 'cpu')  # 'cpu' 或 'gpu'
        self.monitor_gpu = (self.device_type == 'gpu')
        
        # GPU 配置
        self.gpu_overload_threshold = hardware_config.get('gpu_overload_threshold', 85.0)
        self.gpu_memory_critical_mb = hardware_config.get('gpu_memory_critical_mb', 1000)
        
        # 网络监控配置
        self.network_probe_interval = config.get('network_probe_interval_ms', 5000) / 1000.0
        self.network_cache_ttl = 2.0  # 缓存2秒
        self.rtt_history_size = 10
        self.weak_network_threshold = config.get('hard_constraints', {}).get('weak_network_rtt', 150.0)
        
        # 网络状态缓存
        self._last_network_stats: Optional[NetworkStats] = None
        self._last_network_probe_time = 0.0
        self._rtt_history: deque = deque(maxlen=self.rtt_history_size)
        self._probe_failure_count = 0  # 探测失败计数
        
        # HTTP 会话
        self._session: Optional[aiohttp.ClientSession] = None
        
        print(f"[Monitor] 初始化完成 - 设备类型: {self.device_type.upper()}, GPU监控: {'启用' if self.monitor_gpu else '禁用'}")

    def set_simulation_network(self, rtt: float, bandwidth: float, is_weak: bool = False):
        """设置模拟网络状态（用于测试）"""
        self.simulation_mode = True
        self._sim_net_stats = NetworkStats(
            rtt_ms=rtt,
            bandwidth_mbps=bandwidth,
            packet_loss_rate=0.1 if is_weak else 0.01,
            is_weak_network=is_weak
        )

    def get_system_stats(self) -> SystemStats:
        """获取系统状态（支持GPU监控）"""
        cpu_pct = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory()
        mem_avail_mb = mem.available / (1024 * 1024)
        
        # 🧪 上帝模式生效点（测试用）
        if self.override_memory_mb is not None:
            mem_avail_mb = self.override_memory_mb
        
        # GPU 监控
        gpu_usage = 0.0
        gpu_memory_free_mb = 0.0
        
        if self.monitor_gpu:
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # 使用第一块GPU
                
                # GPU 使用率
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_usage = float(utilization.gpu)
                
                # GPU 显存
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_memory_free_mb = mem_info.free / (1024 * 1024)
                
                pynvml.nvmlShutdown()
            except Exception as e:
                # GPU 监控失败，使用默认值
                print(f"[Monitor] GPU监控失败: {e}")
                gpu_usage = 0.0
                gpu_memory_free_mb = 0.0
        
        return SystemStats(
            cpu_usage=cpu_pct,
            memory_available_mb=mem_avail_mb,
            gpu_memory_free_mb=gpu_memory_free_mb,
            gpu_usage=gpu_usage,
            device_type=self.device_type,
            timestamp=time.time()
        )

    async def probe_network(self, force: bool = False) -> NetworkStats:
        """
        网络探测（阶段2增强版）
        
        Args:
            force: 强制探测，忽略缓存
        
        Returns:
            NetworkStats: 网络状态
        """
        # 1. 如果是模拟模式，直接返回模拟数据
        if self.simulation_mode and self._sim_net_stats:
            return self._sim_net_stats
        
        # 2. 检查缓存
        now = time.time()
        if not force and self._last_network_stats:
            if (now - self._last_network_probe_time) < self.network_cache_ttl:
                return self._last_network_stats
        
        # 3. 真实探测
        try:
            network_stats = await self._real_probe()
            
            # 更新缓存
            self._last_network_stats = network_stats
            self._last_network_probe_time = now
            
            # 更新历史
            self._rtt_history.append(network_stats.rtt_ms)
            
            return network_stats
        
        except Exception as e:
            print(f"[Monitor] 网络探测失败: {e}")
            # 降级：返回保守估计
            return NetworkStats(
                rtt_ms=999.9,
                bandwidth_mbps=1.0,
                packet_loss_rate=0.5,
                is_weak_network=True
            )
    
    async def _real_probe(self) -> NetworkStats:
        """真实的网络探测（HTTP HEAD 请求）"""
        if not self._session:
            self._session = aiohttp.ClientSession()
        
        # 探测 RTT：向云端发送 HEAD 请求（更轻量）
        health_url = f"{self.cloud_endpoint}/health"
        
        try:
            start = time.time()
            async with self._session.head(health_url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                rtt_ms = (time.time() - start) * 1000
            
            # 重置失败计数
            self._probe_failure_count = 0
            
            # 计算丢包率（基于失败次数）
            packet_loss_rate = min(0.5, self._probe_failure_count / 10.0)
            
            # 判断是否弱网
            is_weak = (rtt_ms > self.weak_network_threshold) or (packet_loss_rate > 0.1)
            
            return NetworkStats(
                rtt_ms=rtt_ms,
                bandwidth_mbps=50.0,  # 带宽暂时使用估算值
                packet_loss_rate=packet_loss_rate,
                is_weak_network=is_weak
            )
        
        except asyncio.TimeoutError:
            self._probe_failure_count += 1
            return NetworkStats(9999.0, 0.0, 1.0, True)
        except Exception as e:
            self._probe_failure_count += 1
            print(f"[Monitor] RTT探测异常: {e}")
            return NetworkStats(999.0, 1.0, min(0.5, self._probe_failure_count / 10.0), True)
    
    async def close(self):
        """关闭资源"""
        if self._session:
            await self._session.close()
