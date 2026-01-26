"""
云端服务器
"""
import asyncio
import json
from typing import Dict, Any, Optional
import aiohttp
from aiohttp import web

from common.types import (
    VerifyRequest, 
    VerifyResponse,
    MessageType,
    InferenceRequest
)
from cloud.draft_verifier import DraftVerifier
from cloud.kv_cache import VLLMKVCache


class CloudServer:
    """云端服务器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # 初始化组件
        self.kv_cache = VLLMKVCache(
            max_blocks=config.get('kv_cache_blocks', 10000),
            enable_prefix_caching=config.get('enable_prefix_caching', True)
        )
        
        self.draft_verifier = DraftVerifier(
            model_path=config.get('model_path', 'models/vllm-llama-13b'),
            acceptance_threshold=config.get('acceptance_threshold', 0.8)
        )
        
        # 统计
        self.stats = {
            'total_requests': 0,
            'total_verifications': 0,
            'avg_acceptance_rate': 0.0,
            'avg_latency_ms': 0.0
        }
    
    async def start(self):
        """启动云端服务器"""
        print(f"[Cloud] 云端服务器启动，配置: {self.config}")
    
    async def handle_verify_request(
        self, 
        request_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        处理 Draft 验证请求
        
        Args:
            request_data: 请求数据
            
        Returns:
            VerifyResponse 的字典表示
        """
        try:
            # 解析请求
            verify_request = VerifyRequest(**request_data.get('data', {}))
            
            print(f"[Cloud] 收到验证请求: prompt='{verify_request.prompt[:50]}...', "
                  f"draft_tokens={len(verify_request.draft_tokens)}")
            
            # 验证 Draft
            verify_response = await self.draft_verifier.verify_draft(verify_request)
            
            # 更新统计
            self._update_stats(verify_response)
            
            print(f"[Cloud] 验证完成: 接受率={verify_response.acceptance_rate:.2%}, "
                  f"修正位置={verify_response.corrected_positions}")
            
            # 返回响应
            return {
                'type': MessageType.VERIFY_RESPONSE.value,
                'data': verify_response.__dict__
            }
            
        except Exception as e:
            print(f"[Cloud] 验证错误: {e}")
            return {
                'type': 'error',
                'message': str(e)
            }
    
    def _update_stats(self, verify_response: VerifyResponse):
        """更新统计信息"""
        self.stats['total_verifications'] += 1
        
        # 更新平均接受率
        n = self.stats['total_verifications']
        old_avg = self.stats['avg_acceptance_rate']
        new_rate = verify_response.acceptance_rate
        self.stats['avg_acceptance_rate'] = (old_avg * (n - 1) + new_rate) / n
        
        # 更新平均延迟
        old_latency = self.stats['avg_latency_ms']
        new_latency = verify_response.latency_ms
        self.stats['avg_latency_ms'] = (old_latency * (n - 1) + new_latency) / n
    
    async def handle_health_check(self) -> Dict[str, Any]:
        """健康检查"""
        cache_stats = self.kv_cache.get_cache_stats()
        verification_stats = self.draft_verifier.get_verification_stats()
        
        return {
            'status': 'healthy',
            'component': 'cloud',
            'cache_stats': cache_stats,
            'verification_stats': verification_stats,
            'overall_stats': self.stats
        }
    
    async def handle_batch_verify(
        self, 
        request_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        批量验证请求
        
        Args:
            request_data: 包含多个验证请求的数据
            
        Returns:
            批量验证结果
        """
        try:
            verify_requests_data = request_data.get('data', {}).get('requests', [])
            
            print(f"[Cloud] 收到批量验证请求: {len(verify_requests_data)} 个请求")
            
            # 并行验证
            tasks = []
            for req_data in verify_requests_data:
                verify_request = VerifyRequest(**req_data)
                task = self.draft_verifier.verify_draft(verify_request)
                tasks.append(task)
            
            verify_responses = await asyncio.gather(*tasks)
            
            # 返回批量结果
            return {
                'type': 'batch_verify_response',
                'data': {
                    'results': [resp.__dict__ for resp in verify_responses],
                    'count': len(verify_responses)
                }
            }
            
        except Exception as e:
            print(f"[Cloud] 批量验证错误: {e}")
            return {
                'type': 'error',
                'message': str(e)
            }
    
    async def handle_direct_inference(
        self, 
        request_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        直接推理 (不使用 Draft-Verify)
        
        用于对比实验
        """
        try:
            inference_request = InferenceRequest(**request_data.get('data', {}))
            
            print(f"[Cloud] 直接推理: prompt='{inference_request.prompt[:50]}...'")
            
            # 这里应该调用 vLLM 的直接推理接口
            # 模拟直接推理结果
            result = {
                'text': f"[Cloud Direct] {inference_request.prompt} ...",
                'tokens': ['token1', 'token2', 'token3'],
                'latency_ms': 200.0,
                'method': 'direct'
            }
            
            return {
                'type': 'direct_inference_response',
                'data': result
            }
            
        except Exception as e:
            print(f"[Cloud] 直接推理错误: {e}")
            return {
                'type': 'error',
                'message': str(e)
            }


# HTTP 路由处理
async def handle_verify(request):
    """验证请求处理器"""
    server = request.app['cloud_server']
    
    try:
        request_data = await request.json()
        response = await server.handle_verify_request(request_data)
        return web.json_response(response)
    
    except Exception as e:
        return web.json_response({'error': str(e)}, status=500)


async def handle_batch_verify(request):
    """批量验证处理器"""
    server = request.app['cloud_server']
    
    try:
        request_data = await request.json()
        response = await server.handle_batch_verify(request_data)
        return web.json_response(response)
    
    except Exception as e:
        return web.json_response({'error': str(e)}, status=500)


async def handle_direct_inference(request):
    """直接推理处理器"""
    server = request.app['cloud_server']
    
    try:
        request_data = await request.json()
        response = await server.handle_direct_inference(request_data)
        return web.json_response(response)
    
    except Exception as e:
        return web.json_response({'error': str(e)}, status=500)


async def handle_health(request):
    """健康检查处理器"""
    server = request.app['cloud_server']
    
    try:
        response = await server.handle_health_check()
        return web.json_response(response)
    
    except Exception as e:
        return web.json_response({'error': str(e)}, status=500)


async def handle_cache_stats(request):
    """缓存统计处理器"""
    server = request.app['cloud_server']
    
    try:
        cache_stats = server.kv_cache.get_cache_stats()
        return web.json_response(cache_stats)
    
    except Exception as e:
        return web.json_response({'error': str(e)}, status=500)


async def main():
    """主函数"""
    # 配置
    config = {
        'model_path': 'models/vllm-llama-13b',
        'acceptance_threshold': 0.8,
        'kv_cache_blocks': 10000,
        'enable_prefix_caching': True,
        'port': 8081
    }
    
    # 创建服务器
    cloud_server = CloudServer(config)
    await cloud_server.start()
    
    # 创建 Web 应用
    app = web.Application()
    app['cloud_server'] = cloud_server
    
    # 注册路由
    app.router.add_post('/verify', handle_verify)
    app.router.add_post('/verify/batch', handle_batch_verify)
    app.router.add_post('/inference/direct', handle_direct_inference)
    app.router.add_get('/health', handle_health)
    app.router.add_get('/cache/stats', handle_cache_stats)
    
    print(f"[Cloud] 云端服务器运行在 http://localhost:{config['port']}")
    
    try:
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, 'localhost', config['port'])
        await site.start()
        
        # 保持运行
        while True:
            await asyncio.sleep(1)
    
    except KeyboardInterrupt:
        print("\n[Cloud] 收到停止信号")
    
    finally:
        await runner.cleanup()


if __name__ == '__main__':
    asyncio.run(main())
