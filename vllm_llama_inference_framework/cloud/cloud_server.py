"""
云端服务器 - 最终完整版 (支持真实 vLLM 直接推理)
"""
import asyncio
import json
import time
from typing import Dict, Any, Optional, List
import aiohttp
from aiohttp import web

# ==================== 1. 引入 vLLM 组件 ====================
try:
    from vllm import SamplingParams
except ImportError:
    print("⚠️ [Cloud] 未检测到 vLLM，直接推理功能将不可用")
    SamplingParams = Any
# ==========================================================

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
        
        # 初始化 KV Cache 管理器 (如果需要)
        self.kv_cache = VLLMKVCache(
            max_blocks=config.get('kv_cache_blocks', 10000),
            enable_prefix_caching=config.get('enable_prefix_caching', True)
        )
        
        # ==================== 模型路径解析逻辑 ====================
        # 优先从 config['model']['path'] 读取，兼容嵌套和扁平配置
        model_config = config.get('model', {})
        target_path = None

        if isinstance(model_config, dict):
            target_path = model_config.get('path')
        
        if not target_path:
            target_path = config.get('model_path')

        if not target_path:
            target_path = 'models/vllm-llama-13b' # 默认值

        print(f"[CloudServer] 最终读取的模型路径: {target_path}")

        # 初始化验证器 (它持有 vLLM 引擎实例)
        self.draft_verifier = DraftVerifier(
            model_path=target_path,
            acceptance_threshold=config.get('acceptance_threshold', 0.8)
        )
        
        # 统计信息
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
        处理 Draft 验证请求 (协同推理模式)
        """
        try:
            # 解析请求
            verify_request = VerifyRequest(**request_data.get('data', {}))
            
            # print(f"[Cloud] 收到验证请求: len={len(verify_request.draft_tokens)}")
            
            # 调用验证器
            verify_response = await self.draft_verifier.verify_draft(verify_request)
            
            # 更新统计
            self._update_stats(verify_response)
            
            # print(f"[Cloud] 验证完成: AR={verify_response.acceptance_rate:.2%}")
            
            return {
                'type': MessageType.VERIFY_RESPONSE.value,
                'data': verify_response.__dict__
            }
            
        except Exception as e:
            print(f"[Cloud] 验证错误: {e}")
            import traceback
            traceback.print_exc()
            return {'type': 'error', 'message': str(e)}
    
    # ==================== 2. 核心修改：真实直接推理 ====================
    async def handle_direct_inference(
        self, 
        request_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        直接推理 (CLOUD_DIRECT 模式)
        当 Edge 决定完全卸载任务时调用此接口，使用 vLLM 直接生成。
        """
        start_time = time.time()
        try:
            # 解析请求
            req_data = request_data.get('data', {})
            inference_request = InferenceRequest(**req_data)
            
            print(f"☁️ [Cloud] 直接推理请求: Prompt='{inference_request.prompt[:30]}...'")
            
            # 构造 vLLM 采样参数
            # 注意: 如果你的 InferenceRequest 里没有 top_k 等字段，这里可以写死或去 types.py 加
            sampling_params = SamplingParams(
                temperature=inference_request.temperature,
                max_tokens=inference_request.max_tokens,
                top_p=inference_request.top_p,
                # stop=["User:", "Question:"] # 可以根据需要添加停止词
            )
            
            # !!! 关键点 !!!
            # vLLM 的 generate 是同步阻塞的，必须放入 executor 运行，
            # 否则会阻塞整个 HTTP 服务器，导致无法处理其他并发请求。
            # 我们复用 draft_verifier 里的 model (它是 LLM 实例)
            loop = asyncio.get_event_loop()
            
            outputs = await loop.run_in_executor(
                None,
                lambda: self.draft_verifier.model.generate(
                    [inference_request.prompt], 
                    sampling_params
                )
            )
            
            # 获取生成结果 (Batch size = 1)
            output = outputs[0]
            generated_text = output.outputs[0].text
            
            # 简单分词用于前端展示 (vLLM 返回的是纯文本)
            # 这里简单按空格切分，或者直接返回空列表让前端自己处理
            tokens = [t for t in generated_text.split(' ') if t]
            
            latency = (time.time() - start_time) * 1000
            print(f"☁️ [Cloud] 直接推理完成. 耗时: {latency:.1f}ms, 长度: {len(generated_text)}")
            
            result = {
                'text': generated_text,
                'tokens': tokens,
                'latency_ms': latency,
                'method': 'cloud_direct_vllm'
            }
            
            return {
                'type': 'direct_inference_response',
                'data': result
            }
            
        except Exception as e:
            print(f"❌ [Cloud] 直接推理失败: {e}")
            import traceback
            traceback.print_exc()
            return {
                'type': 'error', 
                'message': f"Cloud Inference Failed: {str(e)}"
            }

    def _update_stats(self, verify_response: VerifyResponse):
        """更新统计信息"""
        self.stats['total_verifications'] += 1
        n = self.stats['total_verifications']
        
        # 增量更新平均值
        self.stats['avg_acceptance_rate'] += (verify_response.acceptance_rate - self.stats['avg_acceptance_rate']) / n
        self.stats['avg_latency_ms'] += (verify_response.latency_ms - self.stats['avg_latency_ms']) / n
    
    async def handle_health_check(self) -> Dict[str, Any]:
        """健康检查"""
        cache_stats = {}
        if hasattr(self.kv_cache, 'get_cache_stats'):
            cache_stats = self.kv_cache.get_cache_stats()
            
        return {
            'status': 'healthy',
            'component': 'cloud',
            'overall_stats': self.stats,
            'cache_stats': cache_stats
        }
    
    async def handle_batch_verify(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """批量验证请求 (可选功能)"""
        try:
            reqs_data = request_data.get('data', {}).get('requests', [])
            tasks = []
            for d in reqs_data:
                req = VerifyRequest(**d)
                tasks.append(self.draft_verifier.verify_draft(req))
            
            responses = await asyncio.gather(*tasks)
            return {
                'type': 'batch_verify_response',
                'data': {'results': [r.__dict__ for r in responses]}
            }
        except Exception as e:
            return {'type': 'error', 'message': str(e)}


# ==================== HTTP 路由映射 ====================

async def handle_verify(request):
    server = request.app['cloud_server']
    data = await request.json()
    return web.json_response(await server.handle_verify_request(data))

async def handle_batch_verify(request):
    server = request.app['cloud_server']
    data = await request.json()
    return web.json_response(await server.handle_batch_verify(data))

async def handle_direct_inference(request):
    server = request.app['cloud_server']
    data = await request.json()
    return web.json_response(await server.handle_direct_inference(data))

async def handle_health(request):
    server = request.app['cloud_server']
    return web.json_response(await server.handle_health_check())

async def handle_cache_stats(request):
    server = request.app['cloud_server']
    if hasattr(server.kv_cache, 'get_cache_stats'):
        return web.json_response(server.kv_cache.get_cache_stats())
    return web.json_response({'status': 'no stats'})


async def main():
    """主函数 (调试用)"""
    config = {
        'model_path': 'models/vllm-llama-13b', # 默认值，实际会被 yaml 覆盖
        'port': 8081
    }
    
    cloud_server = CloudServer(config)
    await cloud_server.start()
    
    app = web.Application()
    app['cloud_server'] = cloud_server
    
    # 注册路由
    app.router.add_post('/verify', handle_verify)
    app.router.add_post('/verify/batch', handle_batch_verify)
    app.router.add_post('/inference/direct', handle_direct_inference) # <--- 确保这一行存在
    app.router.add_get('/health', handle_health)
    app.router.add_get('/cache/stats', handle_cache_stats)
    
    print(f"[Cloud] Running on port {config['port']}...")
    
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', config['port'])
    await site.start()
    
    try:
        while True: await asyncio.sleep(3600)
    except KeyboardInterrupt:
        await runner.cleanup()

if __name__ == '__main__':
    asyncio.run(main())