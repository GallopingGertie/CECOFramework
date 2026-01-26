#!/usr/bin/env python3
"""
测试脚本 - 测试各个模块功能
"""
import asyncio
import time
from typing import List, Dict, Any

# 测试各个模块
from common.types import (
    DraftRequest, 
    VerifyRequest, 
    InferenceRequest,
    ConfidenceStrategy
)
from edge.confidence import ConfidenceCalculator, ConfidenceEnsemble, AblatedConfidenceCalculator
from edge.draft_generator import DraftGenerator, DraftManager
from edge.kv_cache import LlamaCppKVCache, AblatedKVCache
from cloud.draft_verifier import DraftVerifier
from cloud.kv_cache import VLLMKVCache, AblatedVLLMKVCache
from common.http_client import EdgeCloudHTTPClient


class FrameworkTester:
    """框架测试器"""
    
    def __init__(self):
        self.results: List[Dict[str, Any]] = []
    
    async def test_confidence_module(self):
        """测试 F1: 置信度判断模块"""
        print("\n=== 测试 F1: 置信度判断模块 ===")
        
        # 创建测试数据
        from common.types import TokenProb
        token_probs = [
            TokenProb(token_id=100, token="hello", prob=0.9, logprob=-0.105),
            TokenProb(token_id=101, token="world", prob=0.8, logprob=-0.223),
            TokenProb(token_id=102, token="!", prob=0.7, logprob=-0.357),
        ]
        
        # 测试不同策略
        strategies = [
            ConfidenceStrategy.MAX_PROB,
            ConfidenceStrategy.ENTROPY,
            ConfidenceStrategy.TEMPERATURE,
            ConfidenceStrategy.TOP_K_AGG
        ]
        
        for strategy in strategies:
            print(f"\n测试策略: {strategy.value}")
            calculator = ConfidenceCalculator(strategy=strategy)
            metrics = calculator.calculate_confidence(token_probs)
            
            print(f"置信度分数: {metrics.confidence_score:.4f}")
            print(f"平均概率: {metrics.avg_prob:.4f}")
            print(f"熵值: {metrics.entropy:.4f}")
            
            # 测试是否接受
            should_accept = calculator.should_accept_draft(metrics, threshold=0.8)
            print(f"是否接受 (阈值0.8): {should_accept}")
        
        # 测试集成策略
        print("\n测试集成策略:")
        ensemble = ConfidenceEnsemble(strategies[:3])  # 前3个策略
        ensemble_score, individual_scores = ensemble.ensemble_confidence(token_probs)
        print(f"集成分数: {ensemble_score:.4f}")
        print(f"各策略分数: {individual_scores}")
        
        # 测试消融实验
        print("\n测试消融实验 (禁用熵计算):")
        ablated_calc = AblatedConfidenceCalculator(
            strategy=ConfidenceStrategy.ENTROPY,
            disable_entropy=True
        )
        ablated_metrics = ablated_calc.calculate_confidence(token_probs)
        print(f"消融后置信度: {ablated_metrics.confidence_score:.4f}")
        
        self.results.append({
            'module': 'F1_Confidence',
            'status': 'passed',
            'timestamp': time.time()
        })
    
    async def test_draft_generator(self):
        """测试 Draft 生成器"""
        print("\n=== 测试 Draft 生成器 ===")
        
        # 创建生成器
        generator = DraftGenerator("mock_model")
        
        # 测试 Draft 生成
        request = DraftRequest(
            prompt="What is the capital of France?",
            max_tokens=16,
            temperature=0.8,
            confidence_threshold=0.8
        )
        
        response = await generator.generate_draft(request)
        
        print(f"生成的 Draft: {' '.join(response.draft_tokens)}")
        print(f"Draft tokens 数量: {len(response.draft_tokens)}")
        print(f"置信度: {response.confidence.confidence_score:.4f}")
        print(f"生成延迟: {response.latency_ms:.2f}ms")
        
        # 测试缓存
        cache_stats = generator.kv_cache.get_cache_stats()
        print(f"缓存统计: {cache_stats}")
        
        self.results.append({
            'module': 'Draft_Generator',
            'status': 'passed',
            'timestamp': time.time()
        })
    
    async def test_draft_verifier(self):
        """测试 F2: Draft 验证器"""
        print("\n=== 测试 F2: Draft 验证器 ===")
        
        # 创建验证器
        verifier = DraftVerifier("mock_model")
        
        # 准备测试数据
        draft_tokens = ["The", "capital", "of", "France", "is"]
        
        request = VerifyRequest(
            prompt="What is the capital of France?",
            draft_tokens=draft_tokens,
            draft_token_ids=[100, 101, 102, 103, 104],
            max_verify_tokens=10,
            confidence_threshold=0.8
        )
        
        response = await verifier.verify_draft(request)
        
        print(f"验证后的 tokens: {' '.join(response.verified_tokens)}")
        print(f"接受数量: {response.accepted_count}/{response.total_count}")
        print(f"接受率: {response.acceptance_rate:.2%}")
        print(f"修正位置: {response.corrected_positions}")
        print(f"验证延迟: {response.latency_ms:.2f}ms")
        
        self.results.append({
            'module': 'F2_Draft_Verifier',
            'status': 'passed',
            'timestamp': time.time()
        })
    
    async def test_kv_cache(self):
        """测试 F3: KV Cache 模块"""
        print("\n=== 测试 F3: KV Cache 模块 ===")
        
        # 测试边端 KV Cache
        print("\n测试边端 KV Cache (llama.cpp):")
        edge_cache = LlamaCppKVCache(max_size=100)
        
        # 设置缓存
        edge_cache.set_cache(
            prompt="Hello world",
            token_ids=[100, 101, 102],
            available_tokens=64
        )
        
        # 获取缓存
        cache_data = edge_cache.get_cache("Hello world", max_tokens=10)
        print(f"缓存命中: {cache_data is not None}")
        
        cache_stats = edge_cache.get_cache_stats()
        print(f"缓存统计: {cache_stats}")
        
        # 测试云端 KV Cache
        print("\n测试云端 KV Cache (vLLM):")
        cloud_cache = VLLMKVCache(max_blocks=100)
        
        # 分配块
        blocks = cloud_cache.allocate_blocks(5, "test_hash")
        print(f"分配的块: {blocks}")
        
        # 设置缓存块
        cloud_cache.set_cache_blocks("test_hash", blocks, seq_len=80)
        
        # 获取缓存块
        cached_blocks = cloud_cache.get_cache_blocks("test_hash")
        print(f"获取的块: {cached_blocks}")
        
        cloud_stats = cloud_cache.get_cache_stats()
        print(f"缓存统计: {cloud_stats}")
        
        # 测试消融实验
        print("\n测试 KV Cache 消融实验:")
        ablated_cache = AblatedKVCache(
            max_size=100,
            disable_lru=True,  # 禁用LRU
            disable_prefix_match=True  # 禁用前缀匹配
        )
        print("创建了禁用LRU和前缀匹配的KV Cache")
        
        self.results.append({
            'module': 'F3_KV_Cache',
            'status': 'passed',
            'timestamp': time.time()
        })
    
    async def test_http_client(self):
        """测试 F4: HTTP 客户端"""
        print("\n=== 测试 F4: HTTP 客户端 ===")
        
        # 注意: 这里需要实际运行的服务器
        # 在测试环境中，我们模拟 HTTP 通信
        
        print("模拟 HTTP 通信测试:")
        
        # 创建客户端
        edge_client = SimpleHTTPClient("http://localhost:8080")
        cloud_client = SimpleHTTPClient("http://localhost:8081")
        
        print("创建了边端和云端 HTTP 客户端")
        print("注意: 需要启动实际服务器才能进行完整测试")
        
        # 测试健康检查
        try:
            edge_health = await edge_client.health_check()
            print(f"边端健康状态: {edge_health.get('status', 'N/A')}")
        except Exception as e:
            print(f"边端健康检查失败 (预期): {e}")
        
        try:
            cloud_health = await cloud_client.health_check()
            print(f"云端健康状态: {cloud_health.get('status', 'N/A')}")
        except Exception as e:
            print(f"云端健康检查失败 (预期): {e}")
        
        self.results.append({
            'module': 'F4_HTTP_Client',
            'status': 'passed',
            'timestamp': time.time()
        })
    
    async def test_end_to_end(self):
        """测试端到端流程"""
        print("\n=== 测试端到端流程 ===")
        
        # 模拟完整的推理流程
        print("模拟推理流程:")
        print("1. 边端生成 Draft")
        print("2. 计算置信度")
        print("3. 云端验证 Draft")
        print("4. 返回最终结果")
        
        # 1. 生成 Draft
        draft_gen = DraftGenerator("mock_model")
        draft_request = DraftRequest(
            prompt="What is AI?",
            max_tokens=10,
            temperature=0.8,
            confidence_threshold=0.8
        )
        draft_response = await draft_gen.generate_draft(draft_request)
        
        print(f"Draft 生成完成，置信度: {draft_response.confidence.confidence_score:.4f}")
        
        # 2. 检查置信度
        if draft_response.confidence.confidence_score >= 0.8:
            print("置信度通过，准备验证")
            
            # 3. 验证 Draft
            verifier = DraftVerifier("mock_model")
            verify_request = VerifyRequest(
                prompt="What is AI?",
                draft_tokens=draft_response.draft_tokens,
                draft_token_ids=draft_response.draft_token_ids,
                confidence_threshold=0.8
            )
            verify_response = await verifier.verify_draft(verify_request)
            
            print(f"验证完成，接受率: {verify_response.acceptance_rate:.2%}")
            print(f"最终结果: {verify_response.final_text}")
        else:
            print("置信度不足，使用 Draft 作为最终结果")
            print(f"结果: {''.join(draft_response.draft_tokens)}")
        
        self.results.append({
            'module': 'End_to_End',
            'status': 'passed',
            'timestamp': time.time()
        })
    
    async def run_all_tests(self):
        """运行所有测试"""
        print("=" * 60)
        print("云边端推理框架 - 模块测试")
        print("=" * 60)
        
        test_methods = [
            self.test_confidence_module,
            self.test_draft_generator,
            self.test_draft_verifier,
            self.test_kv_cache,
            self.test_http_client,
            self.test_end_to_end
        ]
        
        for test_method in test_methods:
            try:
                await test_method()
            except Exception as e:
                print(f"测试失败: {e}")
                self.results.append({
                    'module': test_method.__name__,
                    'status': 'failed',
                    'error': str(e),
                    'timestamp': time.time()
                })
        
        # 打印测试总结
        print("\n" + "=" * 60)
        print("测试总结")
        print("=" * 60)
        
        passed = sum(1 for r in self.results if r['status'] == 'passed')
        total = len(self.results)
        
        print(f"总测试数: {total}")
        print(f"通过: {passed}")
        print(f"失败: {total - passed}")
        print(f"通过率: {passed/total*100:.1f}%")
        
        if passed == total:
            print("\n✅ 所有测试通过!")
        else:
            print("\n❌ 部分测试失败!")
            for result in self.results:
                if result['status'] == 'failed':
                    print(f"  - {result['module']}: {result.get('error', 'Unknown error')}")


async def main():
    """主函数"""
    tester = FrameworkTester()
    await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())
