"""
边端服务器 (修复版: 修复 TaskRequirements 字典反序列化问题)
"""
import asyncio
import json
from typing import Dict, Any, Optional, List, Tuple
import aiohttp
from aiohttp import web

from common.types import (
    DraftRequest, 
    DraftResponse,
    MessageType,
    InferenceRequest,
    ExecutionStrategy,
    TaskRequirements  # <--- [新增] 必须导入这个类
)
from edge.draft_generator import DraftGenerator
from edge.confidence import ConfidenceCalculator
from edge.kv_cache import LlamaCppKVCache
from edge.f1_decision import F1_DecisionModule


class EdgeServer:
    """边端服务器"""
    
    def __init__(
        self, 
        config: Dict[str, Any],
        cloud_endpoint: str = "http://localhost:8081"
    ):
        self.config = config 
        
        if 'edge' in config:
            self.edge_config = config['edge']
            self.comm_config = config.get('communication', {})
        else:
            self.edge_config = config
            self.comm_config = {}

        if 'cloud_endpoint' in self.comm_config:
            self.cloud_endpoint = self.comm_config['cloud_endpoint']
        else:
            self.cloud_endpoint = cloud_endpoint
        
        self.confidence_calculator = ConfidenceCalculator(
            strategy=self.edge_config.get('confidence_strategy', 'max_prob')
        )
        self.kv_cache = LlamaCppKVCache(
            max_size=self.edge_config.get('kv_cache_size', 1000)
        )
        
        model_config = self.edge_config.get('model', {})
        target_path = None
        
        if isinstance(model_config, dict):
            target_path = model_config.get('path')
        
        if not target_path:
            target_path = self.edge_config.get('model_path')
            
        if not target_path:
            target_path = 'models/llama-7b-q4.gguf'
            
        print(f"[EdgeServer] 读取到的模型路径: {target_path}")
        
        self.draft_generator = DraftGenerator(
            model_path=target_path,
            confidence_calculator=self.confidence_calculator
        )
        
        f1_config = self.edge_config.get('f1', {})
        if not f1_config:
            print("[EdgeServer] ⚠️ 警告: F1 配置为空")
        else:
            print(f"[EdgeServer] F1 模块配置已加载 (硬约束: {list(f1_config.get('hard_constraints', {}).keys())})")

        self.f1_decision = F1_DecisionModule(f1_config)
        
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def start(self):
        self.session = aiohttp.ClientSession()
        print(f"[Edge] 边端服务器组件已初始化")
    
    async def stop(self):
        if self.session:
            await self.session.close()
    
    async def handle_draft_request(
        self, 
        request_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            draft_request = DraftRequest(**request_data)
            draft_response = await self.draft_generator.generate_draft(draft_request)
            return {
                'type': MessageType.DRAFT_RESPONSE.value,
                'data': draft_response.__dict__
            }
        except Exception as e:
            print(f"[Edge] Draft 生成错误: {e}")
            return {'type': 'error', 'message': str(e)}
    
    async def handle_health_check(self) -> Dict[str, Any]:
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
        start_time = asyncio.get_event_loop().time()
        
        execution_plan = self.f1_decision.decide(inference_request)
        print(f"[Edge] F1决策: {execution_plan.strategy.value} (得分={execution_plan.score:.3f}, 理由={execution_plan.reason})")
        
        if execution_plan.strategy == ExecutionStrategy.EDGE_ONLY:
            result = await self._execute_edge_only(inference_request, execution_plan)
        elif execution_plan.strategy == ExecutionStrategy.CLOUD_DIRECT:
            result = await self._execute_cloud_direct(inference_request, execution_plan)
        elif execution_plan.strategy == ExecutionStrategy.SPECULATIVE_STANDARD:
            result = await self._execute_speculative(inference_request, execution_plan)
        elif execution_plan.strategy == ExecutionStrategy.ADAPTIVE_CONFIDENCE:
            result = await self._execute_adaptive(inference_request, execution_plan)
        else:
            print(f"[Edge] 警告: 未知策略 {execution_plan.strategy}, 降级到 EDGE_ONLY")
            result = await self._execute_edge_only(inference_request, execution_plan)
        
        total_latency = (asyncio.get_event_loop().time() - start_time) * 1000
        result['total_latency_ms'] = total_latency
        result['strategy'] = execution_plan.strategy.value
        
        # 阶段3新增：记录执行结果到历史
        try:
            self.f1_decision.record_execution(
                strategy=execution_plan.strategy,
                acceptance_rate=result.get('acceptance_rate', 0.0),
                latency_ms=total_latency,
                edge_latency_ms=result.get('edge_latency_ms', 0.0),
                cloud_latency_ms=result.get('cloud_latency_ms', 0.0),
                confidence_score=result.get('confidence_score', 0.0),
                success=True,  # 成功完成
                tokens_generated=len(result.get('tokens', []))
            )
        except Exception as e:
            # 记录失败不应影响主流程
            print(f"[Edge] 历史记录失败: {e}")
        
        return result
    
    async def _execute_edge_only(self, request, plan) -> Dict[str, Any]:
        draft_request = DraftRequest(
            prompt=request.prompt,
            max_tokens=plan.draft_max_tokens,
            temperature=request.temperature,
            top_p=request.top_p
        )
        draft_response = await self.draft_generator.generate_draft(draft_request)
        return {
            'text': ''.join(draft_response.draft_tokens),
            'tokens': draft_response.draft_tokens,
            'confidence_score': draft_response.confidence.confidence_score,
            'used_draft_verify': False,
            'edge_latency_ms': draft_response.latency_ms,
            'cloud_latency_ms': 0.0,
            'acceptance_rate': 0.0
        }
    
    async def _execute_cloud_direct(self, request, plan) -> Dict[str, Any]:
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            cloud_request = {
                'type': MessageType.DIRECT_INFERENCE.value,
                'data': {
                    'prompt': request.prompt,
                    'max_tokens': request.max_tokens,
                    'temperature': request.temperature,
                    'top_p': request.top_p
                }
            }
            
            async with self.session.post(
                f"{self.cloud_endpoint}/inference/direct",
                json=cloud_request,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as resp:
                if resp.status == 200:
                    cloud_result = await resp.json()
                    data = cloud_result.get('data', {})
                    return {
                        'text': data.get('text', ''),
                        'tokens': data.get('tokens', []),
                        'confidence_score': 1.0,
                        'used_draft_verify': False,
                        'edge_latency_ms': 0.0,
                        'cloud_latency_ms': data.get('latency_ms', 0.0),
                        'acceptance_rate': 0.0
                    }
                else:
                    raise Exception(f"Cloud returned status {resp.status}")
        except Exception as e:
            print(f"[Edge] 云端直接推理失败: {e}, 降级到边端")
            return await self._execute_edge_only(request, plan)
    
    async def _execute_speculative(self, request, plan) -> Dict[str, Any]:
        draft_request = DraftRequest(
            prompt=request.prompt,
            max_tokens=plan.draft_max_tokens,
            temperature=request.temperature,
            confidence_threshold=plan.confidence_threshold
        )
        draft_response = await self.draft_generator.generate_draft(draft_request)
        
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            verify_request = {
                'type': MessageType.VERIFY_REQUEST.value,
                'data': {
                    'prompt': request.prompt,
                    'draft_tokens': draft_response.draft_tokens,
                    'draft_token_ids': draft_response.draft_token_ids,
                    'confidence_threshold': plan.confidence_threshold
                }
            }
            
            async with self.session.post(
                f"{self.cloud_endpoint}/verify",
                json=verify_request,
                timeout=aiohttp.ClientTimeout(total=plan.params.get('verify_timeout_ms', 5000) / 1000)
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
            print(f"[Edge] 验证失败: {e}, 使用 Draft")
            return {
                'text': ''.join(draft_response.draft_tokens),
                'tokens': draft_response.draft_tokens,
                'confidence_score': draft_response.confidence.confidence_score,
                'acceptance_rate': 0.0,
                'used_draft_verify': False,
                'edge_latency_ms': draft_response.latency_ms,
                'cloud_latency_ms': 0.0
            }
    
    async def _execute_adaptive(self, request, plan) -> Dict[str, Any]:
        return await self._execute_speculative(request, plan)


async def handle_request(request):
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
    """推理请求处理器 (已修复: 字典转对象)"""
    server = request.app['edge_server']
    
    try:
        request_data = await request.json()
        
        # ==================== [修复点] ====================
        # 如果 requirements 是字典，手动转换成 TaskRequirements 对象
        if 'requirements' in request_data and isinstance(request_data['requirements'], dict):
            # 过滤掉不需要的字段（防止前端多传参导致报错）
            req_dict = request_data['requirements']
            valid_keys = TaskRequirements.__annotations__.keys()
            filtered_reqs = {k: v for k, v in req_dict.items() if k in valid_keys}
            
            # 替换为对象
            request_data['requirements'] = TaskRequirements(**filtered_reqs)
        # =================================================
        
        inference_request = InferenceRequest(**request_data)
        
        result = await server.process_inference(inference_request)
        return web.json_response(result)
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return web.json_response({'error': str(e)}, status=500)


async def handle_cache_stats(request):
    server = request.app['edge_server']
    try:
        cache_stats = server.kv_cache.get_cache_stats()
        return web.json_response(cache_stats)
    except Exception as e:
        return web.json_response({'error': str(e)}, status=500)

async def handle_simulation_control(request):
    server = request.app['edge_server']
    try:
        data = await request.json()
        print(f"[Simulation] 收到仿真参数更新: {data}")
        return web.json_response({'status': 'ok', 'params': data})
    except Exception as e:
        print(f"[Simulation] 参数更新失败: {e}")
        return web.json_response({'error': str(e)}, status=500)