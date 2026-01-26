"""
边端 Draft 生成器
使用轻量级模型快速生成 Draft tokens
"""
import time
import math
import asyncio
from typing import List, Tuple, Optional
import numpy as np

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
        
        # 模拟 llama.cpp 模型加载
        self.model = self._load_model(model_path)
        self.kv_cache = {}
        
    def _load_model(self, model_path: str):
        """
        加载 llama.cpp 模型
        
        实际实现中，这里应该调用 llama.cpp 的 Python 绑定
        例如: from llama_cpp import Llama
        """
        print(f"[Edge] 加载模型: {model_path}")
        # 这里模拟模型加载
        return MockLlamaModel(model_path)
    
    async def generate_draft(
        self, 
        request: DraftRequest
    ) -> DraftResponse:
        """
        生成 Draft tokens
        
        Args:
            request: Draft 生成请求
            
        Returns:
            DraftResponse: Draft 生成响应
        """
        start_time = time.time()
        
        # 检查 KV Cache
        cache_hit = self._check_cache(request.prompt)
        
        # 生成 tokens
        tokens, token_ids, token_probs = await self._generate_tokens_async(
            request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k
        )
        
        # 计算置信度
        confidence_metrics = self.confidence_calculator.calculate_confidence(
            token_probs
        )
        
        # 更新 KV Cache
        self._update_cache(request.prompt, token_ids)
        
        # 准备响应
        latency = (time.time() - start_time) * 1000
        
        kv_cache_info = KVCacheInfo(
            cache_size=len(self.kv_cache),
            hit_tokens=len(token_ids) if cache_hit else 0,
            miss_tokens=0 if cache_hit else len(token_ids),
            hit_rate=1.0 if cache_hit else 0.0
        )
        
        return DraftResponse(
            draft_tokens=tokens,
            draft_token_ids=token_ids,
            confidence=confidence_metrics,
            kv_cache_info=kv_cache_info.__dict__,
            latency_ms=latency
        )
    
    async def _generate_tokens_async(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int
    ) -> Tuple[List[str], List[int], List[TokenProb]]:
        """
        异步生成 tokens
        
        实际实现中应该调用 llama.cpp 的异步生成接口
        """
        # 模拟异步生成
        await asyncio.sleep(0.01)  # 模拟延迟
        
        # 模拟生成结果
        tokens = []
        token_ids = []
        token_probs = []
        
        for i in range(min(max_tokens, 10)):  # 限制最大生成数量
            token = f"_token_{i}_"
            token_id = 1000 + i
            prob = 0.9 - i * 0.05  # 递减概率
            logprob = math.log(prob)
            
            tokens.append(token)
            token_ids.append(token_id)
            token_probs.append(TokenProb(
                token_id=token_id,
                token=token,
                prob=prob,
                logprob=logprob
            ))
        
        return tokens, token_ids, token_probs
    
    def _check_cache(self, prompt: str) -> bool:
        """检查 KV Cache 是否命中"""
        return prompt in self.kv_cache
    
    def _update_cache(self, prompt: str, token_ids: List[int]):
        """更新 KV Cache"""
        # 简单的缓存策略：直接存储
        self.kv_cache[prompt] = {
            'token_ids': token_ids,
            'timestamp': time.time()
        }
        
        # LRU 清理策略
        if len(self.kv_cache) > 1000:  # 限制缓存大小
            oldest_key = min(self.kv_cache.keys(), 
                           key=lambda k: self.kv_cache[k]['timestamp'])
            del self.kv_cache[oldest_key]
    
    def get_cache_stats(self) -> dict:
        """获取缓存统计"""
        return {
            'cache_size': len(self.kv_cache),
            'memory_usage': sum(len(v['token_ids']) for v in self.kv_cache.values())
        }


class MockLlamaModel:
    """模拟 llama.cpp 模型接口"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.vocab_size = 32000  # 模拟词汇表大小
        
    def generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        return_probs: bool = True
    ) -> dict:
        """
        模拟生成接口
        
        实际 llama.cpp 返回格式可能不同，需要根据实际接口调整
        """
        # 模拟生成结果
        tokens = []
        probs = []
        
        for i in range(max_tokens):
            token_id = np.random.randint(1000, 11000)
            prob = np.random.uniform(0.1, 0.95)
            
            tokens.append(token_id)
            probs.append(prob)
        
        return {
            'tokens': tokens,
            'probs': probs,
            'text': ' '.join([f'token_{t}' for t in tokens])
        }


class DraftManager:
    """Draft 管理器，负责协调多个 Draft 生成策略"""
    
    def __init__(self, generators: List[DraftGenerator] = None):
        self.generators = generators or []
    
    async def generate_multi_draft(
        self,
        request: DraftRequest,
        num_drafts: int = 3
    ) -> List[DraftResponse]:
        """
        生成多个 Draft 候选
        
        Args:
            request: Draft 请求
            num_drafts: Draft 数量
            
        Returns:
            List[DraftResponse]: Draft 响应列表
        """
        if not self.generators:
            # 使用默认生成器
            generator = DraftGenerator("mock_model")
            tasks = [generator.generate_draft(request) for _ in range(num_drafts)]
        else:
            # 轮流使用不同的生成器
            tasks = []
            for i in range(num_drafts):
                generator = self.generators[i % len(self.generators)]
                tasks.append(generator.generate_draft(request))
        
        return await asyncio.gather(*tasks)
    
    def select_best_draft(
        self, 
        drafts: List[DraftResponse]
    ) -> DraftResponse:
        """
        选择最佳的 Draft
        
        策略：选择置信度最高的 Draft
        """
        if not drafts:
            raise ValueError("No drafts available")
        
        return max(drafts, key=lambda d: d.confidence.confidence_score)
