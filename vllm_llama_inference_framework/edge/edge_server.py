"""
边端服务器
"""
import asyncio
import json
from typing import Dict, Any, Optional
import aiohttp
from aiohttp import web

from common.types import (
    DraftRequest, 
    DraftResponse,
    MessageType,
    InferenceRequest
)
from edge.draft_generator import DraftGenerator
from edge.confidence import ConfidenceCalculator
from edge.kv_cache import LlamaCppKVCache


class EdgeServer:
    """边端服务器"""
    
    def __init__(
        self, 
        config: Dict[str, Any],
        cloud_endpoint: str = "http://localhost:8081"
    ):
        self.config = config
        self.cloud_endpoint = cloud_endpoint
        
        # 初始化组件
        self.confidence_calculator = ConfidenceCalculator(
            strategy=config.get('confidence_strategy', 'max_prob')
        )
        self.kv_cache = LlamaCppKVCache(
            max_size=config.get('kv_cache_size', 1000)
        )
        self.draft_generator = DraftGenerator(
            model_path=config.get('model_path', 'models/llama-7b-q4.gguf'),
            confidence_calculator=self.confidence_calculator
        )
        
        # HTTP 客户端会话
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def start(self):
        """启动边端服务器"""
        self.session = aiohttp.ClientSession()
        print(f"[Edge] 边端服务器启动，配置: {self.config}")
    
    async def stop(self):
        """停止边端服务器"""
        if self.session:
            await self.session.close()
    
    async def handle_draft_request(
        self, 
        request_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        处理 Draft 生成请求
        
        Args:
            request_data: 请求数据
            
        Returns:
            DraftResponse 的字典表示
        """
        try:
            # 解析请求
            draft_request = DraftRequest(**request_data)
            
            print(f"[Edge] 收到 Draft 请求: prompt='{draft_request.prompt[:50]}...'")
            
            # 生成 Draft
            draft_response = await self.draft_generator.generate_draft(draft_request)
            
            # 打印置信度报告
            confidence_report = self.confidence_calculator.get_confidence_report(
                draft_response.confidence
            )
            print(f"[Edge] {confidence_report}")
            
            # 返回响应
            return {
                'type': MessageType.DRAFT_RESPONSE.value,
                'data': draft_response.__dict__
            }
            
        except Exception as e:
            print(f"[Edge] Draft 生成错误: {e}")
            return {
                'type': 'error',
                'message': str(e)
            }
    
    async def handle_health_check(self) -> Dict[str, Any]:
        """健康检查"""
        cache_stats = self.kv_cache.get_cache_stats()
        
        return {
            'status': 'healthy',
            'component': 'edge',
            'cache_stats': cache_stats,
            'confidence_strategy': self.confidence_calculator.strategy.value
        }
    
    async def process_inference(
        self, 
        inference_request: InferenceRequest
    ) -> Dict[str, Any]:
        """
        处理完整推理流程
        
        Args:
            inference_request: 推理请求
            
        Returns:
            推理结果
        """
        start_time = asyncio.get_event_loop().time()
        
        # 1. 生成 Draft
        draft_request = DraftRequest(
            prompt=inference_request.prompt,
            max_tokens=inference_request.max_tokens // 2,  # Draft 使用一半长度
            temperature=inference_request.temperature,
            confidence_threshold=inference_request.confidence_threshold
        )
        
        draft_response = await self.draft_generator.generate_draft(draft_request)
        
        # 2. 检查置信度
        if inference_request.use_confidence_check:
            is_confident = self.confidence_calculator.should_accept_draft(
                draft_response.confidence,
                threshold=inference_request.confidence_threshold
            )
            
            if not is_confident:
                print(f"[Edge] Draft 置信度不足 ({draft_response.confidence.confidence_score:.4f})，跳过验证")
                return {
                    'text': ''.join(draft_response.draft_tokens),
                    'tokens': draft_response.draft_tokens,
                    'confidence_score': draft_response.confidence.confidence_score,
                    'used_draft_verify': False,
                    'edge_latency_ms': draft_response.latency_ms,
                    'cloud_latency_ms': 0.0
                }
        
        # 3. 发送到云端验证 (如果启用)
        if inference_request.use_draft_verify:
            verify_request = {
                'type': MessageType.VERIFY_REQUEST.value,
                'data': {
                    'prompt': inference_request.prompt,
                    'draft_tokens': draft_response.draft_tokens,
                    'draft_token_ids': draft_response.draft_token_ids,
                    'confidence_threshold': inference_request.confidence_threshold
                }
            }
            
            try:
                async with self.session.post(
                    f"{self.cloud_endpoint}/verify",
                    json=verify_request
                ) as resp:
                    verify_response = await resp.json()
                    
                    if verify_response.get('type') == MessageType.VERIFY_RESPONSE.value:
                        verify_data = verify_response['data']
                        
                        return {
                            'text': verify_data['final_text'],
                            'tokens': verify_data['verified_tokens'],
                            'confidence_score': draft_response.confidence.confidence_score,
                            'acceptance_rate': verify_data['acceptance_rate'],
                            'used_draft_verify': True,
                            'edge_latency_ms': draft_response.latency_ms,
                            'cloud_latency_ms': verify_data.get('latency_ms', 0.0)
                        }
            except Exception as e:
                print(f"[Edge] 云端验证失败: {e}，使用本地Draft")
        
        # 不使用验证或验证失败，直接返回 Draft
        return {
            'text': ''.join(draft_response.draft_tokens),
            'tokens': draft_response.draft_tokens,
            'confidence_score': draft_response.confidence.confidence_score,
            'acceptance_rate': 0.0,
            'used_draft_verify': False,
            'edge_latency_ms': draft_response.latency_ms,
            'cloud_latency_ms': 0.0
        }


# HTTP 路由处理
async def handle_request(request):
    """通用请求处理器"""
    server = request.app['edge_server']
    
    try:
        request_data = await request.json()
        message_type = request_data.get('type')
        
        if message_type == MessageType.DRAFT_REQUEST.value:
            response = await server.handle_draft_request(request_data.get('data', {}))
        elif message_type == MessageType.HEALTH_CHECK.value:
            response = await server.handle_health_check()
        else:
            response = {'error': 'Unknown message type'}
        
        return web.json_response(response)
    
    except Exception as e:
        return web.json_response({'error': str(e)}, status=500)


async def handle_inference(request):
    """推理请求处理器"""
    server = request.app['edge_server']
    
    try:
        request_data = await request.json()
        inference_request = InferenceRequest(**request_data)
        
        result = await server.process_inference(inference_request)
        return web.json_response(result)
    
    except Exception as e:
        return web.json_response({'error': str(e)}, status=500)


async def handle_cache_stats(request):
    """缓存统计处理器"""
    server = request.app['edge_server']
    
    try:
        cache_stats = server.kv_cache.get_cache_stats()
        return web.json_response(cache_stats)
    
    except Exception as e:
        return web.json_response({'error': str(e)}, status=500)


async def main():
    """主函数"""
    # 配置
    config = {
        'model_path': 'models/llama-7b-q4.gguf',
        'confidence_strategy': 'max_prob',
        'kv_cache_size': 1000,
        'port': 8080
    }
    
    # 创建服务器
    edge_server = EdgeServer(config)
    await edge_server.start()
    
    # 创建 Web 应用
    app = web.Application()
    app['edge_server'] = edge_server
    
    # 注册路由
    app.router.add_post('/draft', handle_request)
    app.router.add_post('/inference', handle_inference)
    app.router.add_get('/health', handle_request)
    app.router.add_get('/cache/stats', handle_cache_stats)
    
    print(f"[Edge] 边端服务器运行在 http://localhost:{config['port']}")
    
    try:
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, 'localhost', config['port'])
        await site.start()
        
        # 保持运行
        while True:
            await asyncio.sleep(1)
    
    except KeyboardInterrupt:
        print("\n[Edge] 收到停止信号")
    
    finally:
        await edge_server.stop()
        await runner.cleanup()


if __name__ == '__main__':
    asyncio.run(main())
