"""
边端 Draft 生成器 (修复版: 开启 logits_all 支持置信度计算)
"""
import time
import math
import asyncio
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

# 尝试导入 llama_cpp
try:
    from llama_cpp import Llama
    HAS_LLAMA_CPP = True
except ImportError:
    HAS_LLAMA_CPP = False
    print("[Edge] ⚠️ 未检测到 llama-cpp-python，将使用模拟模式")

from common.types import (
    DraftRequest, 
    DraftResponse, 
    TokenProb,
    KVCacheInfo
)
from edge.confidence import ConfidenceCalculator


class DraftGenerator:
    """Draft 生成器（基于 llama.cpp 接口）"""
    
    def __init__(
        self, 
        model_path: str,
        confidence_calculator: Optional[ConfidenceCalculator] = None
    ):
        self.model_path = model_path
        self.confidence_calculator = confidence_calculator or ConfidenceCalculator()
        
        # 加载模型
        self.model = self._load_model(model_path)
        
        # 简单的内存 KV Cache 记录
        self.kv_cache_stats = {
            'hits': 0,
            'misses': 0,
            'total': 0
        }
        
    def _load_model(self, model_path: str):
        """加载 llama.cpp 模型"""
        print(f"[Edge] 加载模型: {model_path}")
        
        if HAS_LLAMA_CPP:
            try:
                # 初始化 Llama 模型
                return Llama(
                    model_path=model_path,
                    n_ctx=2048,
                    n_gpu_layers=-1, 
                    logits_all=True,  # <--- 关键修复：必须开启这个才能获取 logprobs
                    verbose=False
                )
            except Exception as e:
                print(f"[Edge] ❌ 模型加载失败: {e}")
                print("[Edge] ⚠️ 降级到模拟模式")
                return MockLlamaModel(model_path)
        else:
            return MockLlamaModel(model_path)
    
    async def generate_draft(
        self, 
        request: DraftRequest
    ) -> DraftResponse:
        """生成 Draft tokens"""
        start_time = time.time()
        
        loop = asyncio.get_running_loop()
        
        # 定义同步生成函数
        def _sync_generate():
            if isinstance(self.model, MockLlamaModel):
                return self.model.generate(
                    request.prompt, 
                    request.max_tokens,
                    request.temperature,
                    request.top_p,
                    request.top_k
                )
            else:
                # 真实 Llama.cpp 调用
                output = self.model(
                    request.prompt,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    top_k=request.top_k,
                    logprobs=5, # 需要开启 logits_all=True 才能使用此参数
                    echo=False
                )
                
                # 解析输出
                choice = output['choices'][0]
                text = choice['text']
                
                tokens = []
                token_ids = []
                token_probs_list = []
                
                if choice.get('logprobs') and choice['logprobs'].get('tokens'):
                    logprobs_data = choice['logprobs']
                    raw_tokens = logprobs_data.get('tokens', [])
                    token_logprobs = logprobs_data.get('token_logprobs', [])
                    
                    for i, t in enumerate(raw_tokens):
                        lp = token_logprobs[i] if i < len(token_logprobs) else -1.0
                        prob = math.exp(lp) if lp is not None else 0.0
                        
                        tokens.append(t)
                        token_ids.append(hash(t) % 10000) # 简化处理 ID
                        
                        token_probs_list.append(TokenProb(
                            token_id=hash(t) % 10000,
                            token=t,
                            prob=prob,
                            logprob=lp or -99.9
                        ))
                else:
                    tokens = [text]
                    token_ids = [0]
                    token_probs_list = []

                return tokens, token_ids, token_probs_list

        # 执行生成
        tokens, token_ids, token_probs = await loop.run_in_executor(None, _sync_generate)
        
        # 计算置信度
        confidence_metrics = self.confidence_calculator.calculate_confidence(
            token_probs
        )
        
        latency = (time.time() - start_time) * 1000
        
        kv_cache_info = {
            'hit_rate': 0.0,
            'info': 'managed_by_llama_cpp'
        }
        
        return DraftResponse(
            draft_tokens=tokens,
            draft_token_ids=token_ids,
            confidence=confidence_metrics,
            kv_cache_info=kv_cache_info,
            latency_ms=latency
        )


class MockLlamaModel:
    """模拟模型"""
    def __init__(self, model_path: str): self.model_path = model_path
    def generate(self, *args): return ["mock"], [0], []