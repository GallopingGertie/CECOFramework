"""
F4: HTTP 客户端模块
简单的 HTTP 客户端，用于边端和云端的通信
"""
import asyncio
import json
from typing import Dict, Any, Optional, List
import aiohttp
import time
from urllib.parse import urljoin

from common.types import (
    DraftRequest, 
    DraftResponse,
    VerifyRequest,
    VerifyResponse,
    MessageType,
    InferenceRequest,
    InferenceResponse
)


class HTTPClient:
    """
    HTTP 客户端
    
    特点:
    1. 支持异步请求
    2. 自动重试机制
    3. 连接池管理
    4. 请求/响应日志
    """
    
    def __init__(
        self, 
        base_url: str,
        timeout: float = 30.0,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        self.base_url = base_url.rstrip('/')
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # 会话
        self.session: Optional[aiohttp.ClientSession] = None
        
        # 统计
        self.stats = {
            'requests_sent': 0,
            'responses_received': 0,
            'errors': 0,
            'avg_latency_ms': 0.0
        }
    
    async def __aenter__(self):
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()
    
    async def start(self):
        """启动客户端"""
        connector = aiohttp.TCPConnector(
            limit=50,  # 连接池大小
            limit_per_host=20
        )
        
        self.session = aiohttp.ClientSession(
            base_url=self.base_url,
            timeout=self.timeout,
            connector=connector
        )
    
    async def stop(self):
        """停止客户端"""
        if self.session:
            await self.session.close()
    
    async def _send_request(
        self, 
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        发送 HTTP 请求 (带重试)
        
        Args:
            method: HTTP 方法
            endpoint: 端点
            data: 请求数据
            
        Returns:
            响应数据
        """
        start_time = time.time()
        
        for attempt in range(self.max_retries + 1):
            try:
                if not self.session:
                    await self.start()
                
                url = urljoin(self.base_url, endpoint)
                
                async with self.session.request(
                    method=method,
                    url=url,
                    json=data
                ) as response:
                    response_data = await response.json()
                    
                    # 更新统计
                    self._update_stats(time.time() - start_time, success=True)
                    
                    return response_data
            
            except Exception as e:
                print(f"[HTTP] 请求失败 (尝试 {attempt + 1}/{self.max_retries + 1}): {e}")
                
                if attempt < self.max_retries:
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))
                else:
                    self._update_stats(time.time() - start_time, success=False)
                    raise
    
    def _update_stats(self, latency: float, success: bool):
        """更新统计"""
        self.stats['requests_sent'] += 1
        
        if success:
            self.stats['responses_received'] += 1
            
            # 更新平均延迟
            n = self.stats['responses_received']
            old_avg = self.stats['avg_latency_ms']
            new_latency = latency * 1000
            self.stats['avg_latency_ms'] = (old_avg * (n - 1) + new_latency) / n
        else:
            self.stats['errors'] += 1
    
    async def send_draft_request(
        self, 
        draft_request: DraftRequest,
        endpoint: str = '/draft'
    ) -> DraftResponse:
        """
        发送 Draft 生成请求
        
        Args:
            draft_request: Draft 请求
            endpoint: 端点
            
        Returns:
            Draft 响应
        """
        request_data = {
            'type': MessageType.DRAFT_REQUEST.value,
            'data': draft_request.__dict__
        }
        
        response_data = await self._send_request('POST', endpoint, request_data)
        
        if response_data.get('type') == MessageType.DRAFT_RESPONSE.value:
            data = response_data['data']
            return DraftResponse(**data)
        else:
            raise ValueError(f"Unexpected response type: {response_data.get('type')}")
    
    async def send_verify_request(
        self, 
        verify_request: VerifyRequest,
        endpoint: str = '/verify'
    ) -> VerifyResponse:
        """
        发送 Draft 验证请求
        
        Args:
            verify_request: 验证请求
            endpoint: 端点
            
        Returns:
            验证响应
        """
        request_data = {
            'type': MessageType.VERIFY_REQUEST.value,
            'data': verify_request.__dict__
        }
        
        response_data = await self._send_request('POST', endpoint, request_data)
        
        if response_data.get('type') == MessageType.VERIFY_RESPONSE.value:
            data = response_data['data']
            return VerifyResponse(**data)
        else:
            raise ValueError(f"Unexpected response type: {response_data.get('type')}")
    
    async def send_inference_request(
        self, 
        inference_request: InferenceRequest,
        endpoint: str = '/inference'
    ) -> InferenceResponse:
        """
        发送推理请求
        
        Args:
            inference_request: 推理请求
            endpoint: 端点
            
        Returns:
            推理响应
        """
        request_data = inference_request.__dict__
        
        response_data = await self._send_request('POST', endpoint, request_data)
        
        # 根据响应类型创建不同的响应对象
        if 'text' in response_data:
            return InferenceResponse(
                text=response_data['text'],
                tokens=response_data.get('tokens', []),
                total_latency_ms=response_data.get('latency_ms', 0.0),
                confidence_score=response_data.get('confidence_score', 0.0),
                acceptance_rate=response_data.get('acceptance_rate', 0.0)
            )
        else:
            raise ValueError(f"Invalid inference response: {response_data}")
    
    async def send_batch_verify_requests(
        self, 
        verify_requests: List[VerifyRequest],
        endpoint: str = '/verify/batch'
    ) -> List[VerifyResponse]:
        """
        发送批量验证请求
        
        Args:
            verify_requests: 验证请求列表
            endpoint: 端点
            
        Returns:
            验证响应列表
        """
        request_data = {
            'type': 'batch_verify_request',
            'data': {
                'requests': [req.__dict__ for req in verify_requests]
            }
        }
        
        response_data = await self._send_request('POST', endpoint, request_data)
        
        if response_data.get('type') == 'batch_verify_response':
            results = response_data['data']['results']
            return [VerifyResponse(**result) for result in results]
        else:
            raise ValueError(f"Unexpected batch response: {response_data}")
    
    async def health_check(self, endpoint: str = '/health') -> Dict[str, Any]:
        """
        健康检查
        
        Args:
            endpoint: 端点
            
        Returns:
            健康状态
        """
        request_data = {
            'type': MessageType.HEALTH_CHECK.value
        }
        
        return await self._send_request('GET', endpoint, request_data)
    
    async def get_cache_stats(self, endpoint: str = '/cache/stats') -> Dict[str, Any]:
        """
        获取缓存统计
        
        Args:
            endpoint: 端点
            
        Returns:
            缓存统计
        """
        return await self._send_request('GET', endpoint)
    
    def get_client_stats(self) -> Dict[str, Any]:
        """获取客户端统计"""
        return self.stats.copy()


class EdgeCloudHTTPClient:
    """
    边端-云端通信的专用客户端
    
    封装了边端和云端之间的所有通信逻辑
    """
    
    def __init__(
        self, 
        edge_endpoint: str = "http://localhost:8080",
        cloud_endpoint: str = "http://localhost:8081",
        timeout: float = 30.0
    ):
        self.edge_client = HTTPClient(edge_endpoint, timeout=timeout)
        self.cloud_client = HTTPClient(cloud_endpoint, timeout=timeout)
    
    async def __aenter__(self):
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()
    
    async def start(self):
        """启动客户端"""
        await self.edge_client.start()
        await self.cloud_client.start()
    
    async def stop(self):
        """停止客户端"""
        await self.edge_client.stop()
        await self.cloud_client.stop()
    
    async def edge_generate_draft(
        self, 
        draft_request: DraftRequest
    ) -> DraftResponse:
        """边端生成 Draft"""
        return await self.edge_client.send_draft_request(draft_request)
    
    async def cloud_verify_draft(
        self, 
        verify_request: VerifyRequest
    ) -> VerifyResponse:
        """云端验证 Draft"""
        return await self.cloud_client.send_verify_request(verify_request)
    
    async def full_inference_pipeline(
        self, 
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 0.8,
        confidence_threshold: float = 0.8,
        use_draft_verify: bool = True,
        use_confidence_check: bool = True
    ) -> InferenceResponse:
        """
        完整的推理流程
        
        Args:
            prompt: 输入提示
            max_tokens: 最大生成token数
            temperature: 温度参数
            confidence_threshold: 置信度阈值
            use_draft_verify: 是否使用 Draft-Verify
            use_confidence_check: 是否使用置信度检查
            
        Returns:
            推理结果
        """
        start_time = asyncio.get_event_loop().time()
        
        # 1. 边端生成 Draft
        draft_request = DraftRequest(
            prompt=prompt,
            max_tokens=max_tokens // 2,
            temperature=temperature,
            confidence_threshold=confidence_threshold
        )
        
        draft_response = await self.edge_generate_draft(draft_request)
        
        # 2. 检查置信度
        if use_confidence_check:
            if draft_response.confidence.confidence_score < confidence_threshold:
                # 置信度不足，直接返回 Draft
                return InferenceResponse(
                    text=''.join(draft_response.draft_tokens),
                    tokens=draft_response.draft_tokens,
                    total_latency_ms=draft_response.latency_ms,
                    confidence_score=draft_response.confidence.confidence_score,
                    acceptance_rate=0.0
                )
        
        # 3. 云端验证
        if use_draft_verify:
            verify_request = VerifyRequest(
                prompt=prompt,
                draft_tokens=draft_response.draft_tokens,
                draft_token_ids=draft_response.draft_token_ids,
                max_verify_tokens=max_tokens,
                confidence_threshold=confidence_threshold
            )
            
            verify_response = await self.cloud_verify_draft(verify_request)
            
            # 计算总延迟
            total_latency = (asyncio.get_event_loop().time() - start_time) * 1000
            
            return InferenceResponse(
                text=verify_response.final_text,
                tokens=verify_response.verified_tokens,
                total_latency_ms=total_latency,
                edge_latency_ms=draft_response.latency_ms,
                cloud_latency_ms=verify_response.latency_ms,
                confidence_score=draft_response.confidence.confidence_score,
                acceptance_rate=verify_response.acceptance_rate
            )
        
        # 不使用验证
        total_latency = (asyncio.get_event_loop().time() - start_time) * 1000
        return InferenceResponse(
            text=''.join(draft_response.draft_tokens),
            tokens=draft_response.draft_tokens,
            total_latency_ms=total_latency,
            confidence_score=draft_response.confidence.confidence_score,
            acceptance_rate=0.0
        )


# 简单的同步封装
class SimpleHTTPClient:
    """简单的同步 HTTP 客户端"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        
    def send_request(
        self, 
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """发送同步请求"""
        async def _async_request():
            async with HTTPClient(self.base_url) as client:
                return await client._send_request(method, endpoint, data)
        
        return asyncio.run(_async_request())
    
    def health_check(self, endpoint: str = '/health') -> Dict[str, Any]:
        """健康检查"""
        return self.send_request('GET', endpoint)
