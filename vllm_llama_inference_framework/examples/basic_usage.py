#!/usr/bin/env python3
"""
基本使用示例
演示如何使用云边端推理框架
"""
import asyncio
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.http_client import EdgeCloudHTTPClient, SimpleHTTPClient
from common.types import InferenceRequest, DraftRequest, VerifyRequest
from edge.confidence import ConfidenceCalculator, ConfidenceStrategy
from edge.kv_cache import LlamaCppKVCache


async def example_1_basic_inference():
    """示例1: 基本推理"""
    print("\n=== 示例1: 基本推理 ===")
    
    # 使用 EdgeCloudHTTPClient 进行完整推理
    async with EdgeCloudHTTPClient() as client:
        result = await client.full_inference_pipeline(
            prompt="What is artificial intelligence?",
            max_tokens=64,
            temperature=0.8,
            use_draft_verify=True,
            use_confidence_check=True
        )
        
        print(f"问题: What is artificial intelligence?")
        print(f"回答: {result.text}")
        print(f"总延迟: {result.total_latency_ms:.2f}ms")
        print(f"边端延迟: {result.edge_latency_ms:.2f}ms")
        print(f"云端延迟: {result.cloud_latency_ms:.2f}ms")
        print(f"接受率: {result.acceptance_rate:.2%}")
        print(f"置信度: {result.confidence_score:.2%}")


async def example_2_draft_only():
    """示例2: 只使用边端生成 Draft"""
    print("\n=== 示例2: 只使用边端生成 Draft ===")
    
    async with EdgeCloudHTTPClient() as client:
        result = await client.full_inference_pipeline(
            prompt="Explain machine learning in simple terms.",
            max_tokens=50,
            use_draft_verify=False,  # 禁用云端验证
            use_confidence_check=True
        )
        
        print(f"问题: Explain machine learning in simple terms.")
        print(f"回答: {result.text}")
        print(f"延迟: {result.total_latency_ms:.2f}ms")
        print(f"模式: 仅边端 (Draft)")


async def example_3_cloud_only():
    """示例3: 只使用云端直接推理"""
    print("\n=== 示例3: 只使用云端直接推理 ===")
    
    # 直接调用云端的直接推理接口
    cloud_client = SimpleHTTPClient("http://localhost:8081")
    
    request_data = {
        'data': {
            'prompt': "What are the benefits of renewable energy?",
            'max_tokens': 100,
            'temperature': 0.8
        }
    }
    
    try:
        response = await cloud_client.send_request(
            'POST', 
            '/inference/direct', 
            request_data
        )
        
        print(f"问题: What are the benefits of renewable energy?")
        print(f"回答: {response.get('data', {}).get('text', 'N/A')}")
        print(f"延迟: {response.get('data', {}).get('latency_ms', 0):.2f}ms")
        print(f"模式: 仅云端 (直接推理)")
    
    except Exception as e:
        print(f"云端推理失败: {e}")
        print("提示: 确保云端服务器已启动")


async def example_4_confidence_strategies():
    """示例4: 测试不同的置信度策略"""
    print("\n=== 示例4: 测试不同的置信度策略 ===")
    
    from common.types import TokenProb
    import numpy as np
    
    # 模拟 token 概率
    token_probs = []
    for i in range(5):
        prob = 0.9 - i * 0.1  # 递减概率
        token_probs.append(TokenProb(
            token_id=1000 + i,
            token=f"token_{i}",
            prob=prob,
            logprob=np.log(prob)
        ))
    
    strategies = [
        ConfidenceStrategy.MAX_PROB,
        ConfidenceStrategy.ENTROPY,
        ConfidenceStrategy.TEMPERATURE,
        ConfidenceStrategy.TOP_K_AGG
    ]
    
    print("Token 概率分布:")
    for tp in token_probs:
        print(f"  {tp.token}: {tp.prob:.3f}")
    
    print("\n各策略计算的置信度:")
    for strategy in strategies:
        calculator = ConfidenceCalculator(strategy=strategy)
        metrics = calculator.calculate_confidence(token_probs)
        print(f"  {strategy.value}: {metrics.confidence_score:.4f}")


async def example_5_kv_cache_usage():
    """示例5: KV Cache 使用示例"""
    print("\n=== 示例5: KV Cache 使用示例 ===")
    
    # 创建 KV Cache
    cache = LlamaCppKVCache(max_size=100)
    
    # 添加缓存
    cache.set_cache(
        prompt="Hello world",
        token_ids=[100, 101, 102, 103],
        available_tokens=60
    )
    
    # 获取缓存
    cached_data = cache.get_cache("Hello world", max_tokens=10)
    
    if cached_data:
        print("✅ 缓存命中!")
        print(f"缓存的token IDs: {cached_data['token_ids']}")
        print(f"可用token数: {cached_data['available_tokens']}")
    else:
        print("❌ 缓存未命中")
    
    # 获取统计
    stats = cache.get_cache_stats()
    print(f"\n缓存统计:")
    print(f"  缓存大小: {stats['cache_size']}")
    print(f"  命中率: {stats['hit_rate']:.2%}")
    print(f"  命中次数: {stats['hits']}")
    print(f"  未命中次数: {stats['misses']}")


async def example_6_batch_inference():
    """示例6: 批量推理"""
    print("\n=== 示例6: 批量推理 ===")
    
    prompts = [
        "What is Python?",
        "How does async/await work?",
        "Explain list comprehensions."
    ]
    
    async with EdgeCloudHTTPClient() as client:
        print(f"批量推理 {len(prompts)} 个提示...")
        
        tasks = [
            client.full_inference_pipeline(
                prompt=prompt,
                max_tokens=30,
                use_draft_verify=True,
                use_confidence_check=True
            )
            for prompt in prompts
        ]
        
        start_time = asyncio.get_event_loop().time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = asyncio.get_event_loop().time()
        
        print(f"\n批量推理完成!")
        print(f"总耗时: {(end_time - start_time)*1000:.2f}ms")
        print(f"平均延迟: {sum(r.total_latency_ms for r in results if not isinstance(r, Exception))/len(results):.2f}ms")
        
        for i, (prompt, result) in enumerate(zip(prompts, results)):
            if isinstance(result, Exception):
                print(f"\n{i+1}. ❌ {prompt[:30]}... -> 失败: {result}")
            else:
                print(f"\n{i+1}. ✅ {prompt[:30]}... -> {result.text[:50]}...")


async def example_7_health_checks():
    """示例7: 健康检查"""
    print("\n=== 示例7: 健康检查 ===")
    
    # 检查边端
    edge_client = SimpleHTTPClient("http://localhost:8080")
    try:
        edge_health = await edge_client.health_check()
        print(f"边端状态: {edge_health.get('status', 'N/A')}")
        print(f"边端缓存命中率: {edge_health.get('cache_stats', {}).get('hit_rate', 0):.2%}")
    except Exception as e:
        print(f"边端健康检查失败: {e}")
    
    # 检查云端
    cloud_client = SimpleHTTPClient("http://localhost:8081")
    try:
        cloud_health = await cloud_client.health_check()
        print(f"云端状态: {cloud_health.get('status', 'N/A')}")
        print(f"云端平均接受率: {cloud_health.get('verification_stats', {}).get('avg_acceptance_rate', 0):.2%}")
    except Exception as e:
        print(f"云端健康检查失败: {e}")


async def main():
    """运行所有示例"""
    print("=" * 60)
    print("云边端推理框架 - 使用示例")
    print("=" * 60)
    print("\n注意: 运行这些示例前，请确保:")
    print("1. 边端服务器运行在 http://localhost:8080")
    print("2. 云端服务器运行在 http://localhost:8081")
    print("3. 或者修改示例中的端点地址")
    
    examples = [
        example_1_basic_inference,
        example_2_draft_only,
        example_3_cloud_only,
        example_4_confidence_strategies,
        example_5_kv_cache_usage,
        example_6_batch_inference,
        example_7_health_checks
    ]
    
    for i, example in enumerate(examples, 1):
        try:
            await example()
        except Exception as e:
            print(f"\n❌ 示例 {i} 失败: {e}")
            print("这可能是因为服务器未启动，请检查服务器状态")
    
    print("\n" + "=" * 60)
    print("示例运行完成!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
