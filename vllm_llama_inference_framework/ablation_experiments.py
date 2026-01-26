#!/usr/bin/env python3
"""
消融实验脚本
用于评估各个模块对整体性能的影响
"""
import asyncio
import json
import time
from typing import Dict, Any, List, Optional
import yaml

from common.types import InferenceRequest
from common.http_client import EdgeCloudHTTPClient, SimpleHTTPClient


class AblationExperiment:
    """消融实验类"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = self._load_config(config_path)
        self.results: List[Dict[str, Any]] = []
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"[Ablation] 配置文件不存在，使用默认配置")
            return {}
    
    async def run_single_experiment(
        self,
        name: str,
        description: str,
        config_overrides: Dict[str, Any],
        test_prompts: List[str]
    ) -> Dict[str, Any]:
        """
        运行单个消融实验
        
        Args:
            name: 实验名称
            description: 实验描述
            config_overrides: 配置覆盖
            test_prompts: 测试提示列表
            
        Returns:
            实验结果
        """
        print(f"\n=== 运行实验: {name} ===")
        print(f"描述: {description}")
        
        # 应用配置覆盖
        experiment_config = self._apply_config_overrides(
            self.config.copy(), 
            config_overrides
        )
        
        # 打印配置变更
        print("配置变更:")
        self._print_config_changes(config_overrides)
        
        # 运行测试
        results = await self._run_inference_tests(
            experiment_config, 
            test_prompts, 
            name
        )
        
        # 保存结果
        experiment_result = {
            'name': name,
            'description': description,
            'config_overrides': config_overrides,
            'results': results,
            'timestamp': time.time()
        }
        
        self.results.append(experiment_result)
        return experiment_result
    
    def _apply_config_overrides(
        self, 
        base_config: Dict[str, Any], 
        overrides: Dict[str, Any]
    ) -> Dict[str, Any]:
        """应用配置覆盖"""
        for key, value in overrides.items():
            if isinstance(value, dict) and key in base_config:
                base_config[key] = self._apply_config_overrides(
                    base_config[key], 
                    value
                )
            else:
                base_config[key] = value
        
        return base_config
    
    def _print_config_changes(self, overrides: Dict[str, Any]):
        """打印配置变更"""
        def print_changes(obj, prefix=""):
            for key, value in obj.items():
                if isinstance(value, dict):
                    print_changes(value, f"{prefix}{key}.")
                else:
                    print(f"  {prefix}{key}: {value}")
        
        print_changes(overrides)
    
    async def _run_inference_tests(
        self,
        config: Dict[str, Any],
        prompts: List[str],
        experiment_name: str
    ) -> List[Dict[str, Any]]:
        """运行推理测试"""
        results = []
        
        # 创建客户端
        edge_endpoint = config['communication']['edge_endpoint']
        
        # 注意: 这里使用模拟客户端，实际应该使用 EdgeCloudHTTPClient
        client = SimpleHTTPClient(edge_endpoint)
        
        for i, prompt in enumerate(prompts):
            print(f"\n  测试 {i+1}/{len(prompts)}: {prompt[:50]}...")
            
            start_time = time.time()
            
            try:
                # 从配置中提取参数
                inference_config = config.get('inference', {}).get('features', {})
                
                # 创建推理请求
                request_data = {
                    'prompt': prompt,
                    'max_tokens': 64,
                    'temperature': 0.8,
                    'use_draft_verify': inference_config.get('use_draft_verify', True),
                    'use_confidence_check': inference_config.get('use_confidence_check', True),
                    'confidence_threshold': config.get('edge', {}).get('confidence', {}).get('threshold', 0.8)
                }
                
                # 发送请求
                response = await client.send_request('POST', '/inference', request_data)
                
                end_time = time.time()
                
                # 记录结果
                result = {
                    'prompt': prompt,
                    'response': response,
                    'latency_ms': (end_time - start_time) * 1000,
                    'success': 'error' not in response
                }
                
                if result['success']:
                    print(f"  ✅ 成功 - 延迟: {result['latency_ms']:.2f}ms")
                    print(f"  结果: {response.get('text', 'N/A')[:100]}...")
                else:
                    print(f"  ❌ 失败 - 错误: {response.get('error', 'Unknown')}")
                
                results.append(result)
            
            except Exception as e:
                print(f"  ❌ 异常: {e}")
                results.append({
                    'prompt': prompt,
                    'error': str(e),
                    'success': False
                })
        
        return results
    
    async def run_baseline_experiment(self, test_prompts: List[str]):
        """运行基准实验 (所有功能启用)"""
        return await self.run_single_experiment(
            name="baseline",
            description="所有功能启用的基准配置",
            config_overrides={},
            test_prompts=test_prompts
        )
    
    async def run_all_ablation_experiments(self, test_prompts: List[str]):
        """运行所有消融实验"""
        experiments = [
            {
                'name': 'no_confidence',
                'description': '禁用置信度判断',
                'config_overrides': {
                    'inference': {
                        'features': {
                            'use_confidence_check': False
                        }
                    }
                }
            },
            {
                'name': 'no_draft_verify',
                'description': '禁用 Draft-Verify',
                'config_overrides': {
                    'inference': {
                        'features': {
                            'use_draft_verify': False
                        }
                    }
                }
            },
            {
                'name': 'no_kv_cache',
                'description': '禁用 KV Cache',
                'config_overrides': {
                    'edge': {
                        'kv_cache': {
                            'enabled': False
                        }
                    },
                    'cloud': {
                        'kv_cache': {
                            'enabled': False
                        }
                    }
                }
            },
            {
                'name': 'edge_only',
                'description': '只使用边端',
                'config_overrides': {
                    'inference': {
                        'features': {
                            'use_draft_verify': False,
                            'use_confidence_check': False
                        }
                    }
                }
            },
            {
                'name': 'cloud_only',
                'description': '只使用云端 (直接推理)',
                'config_overrides': {
                    'inference': {
                        'features': {
                            'use_draft_verify': False,
                            'use_confidence_check': False
                        }
                    }
                }
            }
        ]
        
        experiment_results = []
        
        for exp in experiments:
            result = await self.run_single_experiment(
                name=exp['name'],
                description=exp['description'],
                config_overrides=exp['config_overrides'],
                test_prompts=test_prompts
            )
            experiment_results.append(result)
        
        return experiment_results
    
    def generate_report(self) -> Dict[str, Any]:
        """生成实验报告"""
        if not self.results:
            return {'error': '没有实验结果'}
        
        # 计算汇总统计
        summary = {
            'total_experiments': len(self.results),
            'baseline_latency': None,
            'ablation_comparison': []
        }
        
        # 找到基准实验
        baseline_result = None
        for result in self.results:
            if result['name'] == 'baseline':
                baseline_result = result
                break
        
        if baseline_result:
            baseline_latencies = [
                r['latency_ms'] for r in baseline_result['results'] 
                if r['success']
            ]
            summary['baseline_latency'] = {
                'mean': sum(baseline_latencies) / len(baseline_latencies) if baseline_latencies else 0,
                'min': min(baseline_latencies) if baseline_latencies else 0,
                'max': max(baseline_latencies) if baseline_latencies else 0
            }
        
        # 与基准比较
        for result in self.results:
            if result['name'] != 'baseline':
                latencies = [
                    r['latency_ms'] for r in result['results'] 
                    if r['success']
                ]
                
                if latencies and baseline_latencies:
                    mean_latency = sum(latencies) / len(latencies)
                    baseline_mean = summary['baseline_latency']['mean']
                    
                    comparison = {
                        'experiment': result['name'],
                        'mean_latency': mean_latency,
                        'baseline_mean': baseline_mean,
                        'difference_ms': mean_latency - baseline_mean,
                        'difference_percent': ((mean_latency - baseline_mean) / baseline_mean * 100) if baseline_mean > 0 else 0
                    }
                    
                    summary['ablation_comparison'].append(comparison)
        
        return {
            'summary': summary,
            'experiments': self.results,
            'generated_at': time.time()
        }
    
    def save_report(self, filepath: str = "ablation_report.json"):
        """保存实验报告"""
        report = self.generate_report()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n[Ablation] 实验报告已保存到: {filepath}")


async def main():
    """主函数"""
    print("=" * 60)
    print("云边端推理框架 - 消融实验")
    print("=" * 60)
    
    # 创建实验实例
    experiment = AblationExperiment()
    
    # 测试提示
    test_prompts = [
        "What is the capital of France?",
        "Explain the theory of relativity in simple terms.",
        "Write a haiku about artificial intelligence.",
        "How does photosynthesis work?",
        "What are the benefits of renewable energy?",
        "Describe the process of making coffee.",
        "What is machine learning?",
        "How do airplanes fly?",
        "Explain quantum computing.",
        "What is the meaning of life?"
    ]
    
    print(f"测试提示数量: {len(test_prompts)}")
    
    # 运行基准实验
    print("\n1. 运行基准实验...")
    await experiment.run_baseline_experiment(test_prompts)
    
    # 运行消融实验
    print("\n2. 运行消融实验...")
    await experiment.run_all_ablation_experiments(test_prompts)
    
    # 生成并保存报告
    print("\n3. 生成实验报告...")
    experiment.save_report()
    
    # 打印简要结果
    report = experiment.generate_report()
    summary = report.get('summary', {})
    
    print("\n" + "=" * 60)
    print("实验结果摘要")
    print("=" * 60)
    
    if summary.get('baseline_latency'):
        baseline = summary['baseline_latency']
        print(f"\n基准实验:")
        print(f"  平均延迟: {baseline['mean']:.2f}ms")
        print(f"  最小延迟: {baseline['min']:.2f}ms")
        print(f"  最大延迟: {baseline['max']:.2f}ms")
    
    print(f"\n消融实验对比:")
    for comparison in summary.get('ablation_comparison', []):
        diff_percent = comparison['difference_percent']
        print(f"  {comparison['experiment']}:")
        print(f"    延迟: {comparison['mean_latency']:.2f}ms")
        print(f"    差异: {diff_percent:+.1f}% ({'+' if diff_percent > 0 else ''}{comparison['difference_ms']:.2f}ms)")
    
    print("\n" + "=" * 60)
    print("消融实验完成!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
