"""
主入口文件
演示如何使用云边端推理框架
"""
import asyncio
import argparse
import yaml
from typing import Dict, Any, Optional

from common.types import InferenceRequest
from common.http_client import EdgeCloudHTTPClient, SimpleHTTPClient
from edge.edge_server import EdgeServer
from cloud.cloud_server import CloudServer


class InferenceFramework:
    """推理框架主类"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = self._load_config(config_path)
        self.edge_server: Optional[EdgeServer] = None
        self.cloud_server: Optional[CloudServer] = None
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"[Config] 配置文件 {config_path} 不存在，使用默认配置")
            return self._default_config()
    
    def _default_config(self) -> Dict[str, Any]:
        """默认配置"""
        return {
            'edge': {
                'server': {'host': 'localhost', 'port': 8080},
                'confidence': {'threshold': 0.8},
                'kv_cache': {'enabled': True, 'max_size': 1000}
            },
            'cloud': {
                'server': {'host': 'localhost', 'port': 8081},
                'draft_verifier': {'acceptance_threshold': 0.8},
                'kv_cache': {'enabled': True, 'max_blocks': 10000}
            },
            'communication': {
                'edge_endpoint': 'http://localhost:8080',
                'cloud_endpoint': 'http://localhost:8081'
            }
        }
    
    async def start_servers(self):
        """启动边端和云端服务器"""
        print("[Framework] 启动服务器...")
        
        # 启动边端服务器
        self.edge_server = EdgeServer(self.config.get('edge', {}))
        await self.edge_server.start()
        
        # 启动云端服务器
        self.cloud_server = CloudServer(self.config.get('cloud', {}))
        await self.cloud_server.start()
        
        print("[Framework] 服务器启动完成")
    
    async def stop_servers(self):
        """停止服务器"""
        print("[Framework] 停止服务器...")
        
        if self.cloud_server:
            await self.cloud_server.stop()
        
        if self.edge_server:
            await self.edge_server.stop()
        
        print("[Framework] 服务器已停止")
    
    async def run_inference(
        self,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 0.8,
        use_draft_verify: Optional[bool] = None,
        use_confidence_check: Optional[bool] = None,
        confidence_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        运行推理
        
        Args:
            prompt: 输入提示
            max_tokens: 最大生成token数
            temperature: 温度参数
            use_draft_verify: 是否使用 Draft-Verify
            use_confidence_check: 是否使用置信度检查
            confidence_threshold: 置信度阈值
            
        Returns:
            推理结果
        """
        # 使用配置或参数
        inference_config = self.config.get('inference', {}).get('features', {})
        
        use_draft_verify = use_draft_verify if use_draft_verify is not None else inference_config.get('use_draft_verify', True)
        use_confidence_check = use_confidence_check if use_confidence_check is not None else inference_config.get('use_confidence_check', True)
        confidence_threshold = confidence_threshold or self.config.get('edge', {}).get('confidence', {}).get('threshold', 0.8)
        
        # 创建推理请求
        inference_request = InferenceRequest(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            use_draft_verify=use_draft_verify,
            use_confidence_check=use_confidence_check,
            confidence_threshold=confidence_threshold
        )
        
        # 使用 HTTP 客户端发送请求
        edge_endpoint = self.config['communication']['edge_endpoint']
        client = SimpleHTTPClient(edge_endpoint)
        
        try:
            response = await client.send_request(
                'POST', 
                '/inference', 
                inference_request.__dict__
            )
            return response
        except Exception as e:
            print(f"[Framework] 推理失败: {e}")
            return {'error': str(e)}
    
    async def run_interactive_mode(self):
        """交互模式"""
        print("\n=== 云边端推理框架 交互模式 ===")
        print("输入 'quit' 或 'exit' 退出")
        print("输入 'stats' 查看统计")
        print("输入 'config' 查看配置")
        print("-" * 40)
        
        while True:
            try:
                prompt = input("\n> ").strip()
                
                if prompt.lower() in ['quit', 'exit', 'q']:
                    break
                
                if prompt.lower() == 'stats':
                    await self.show_stats()
                    continue
                
                if prompt.lower() == 'config':
                    self.show_config()
                    continue
                
                if not prompt:
                    continue
                
                # 运行推理
                print("[Framework] 正在推理...")
                result = await self.run_inference(prompt)
                
                if 'error' in result:
                    print(f"[Framework] 错误: {result['error']}")
                else:
                    print(f"\n[结果] {result.get('text', 'N/A')}")
                    print(f"[统计] 延迟: {result.get('total_latency_ms', 0):.2f}ms")
                    print(f"[统计] 置信度: {result.get('confidence_score', 0):.2%}")
                    print(f"[统计] 接受率: {result.get('acceptance_rate', 0):.2%}")
                    print(f"[统计] 边端延迟: {result.get('edge_latency_ms', 0):.2f}ms")
                    print(f"[统计] 云端延迟: {result.get('cloud_latency_ms', 0):.2f}ms")
            
            except KeyboardInterrupt:
                print("\n[Framework] 收到中断信号")
                break
            except Exception as e:
                print(f"[Framework] 错误: {e}")
    
    async def show_stats(self):
        """显示统计信息"""
        print("\n=== 统计信息 ===")
        
        # 边端统计
        edge_endpoint = self.config['communication']['edge_endpoint']
        try:
            client = SimpleHTTPClient(edge_endpoint)
            edge_stats = await client.health_check()
            print(f"\n[边端] 状态: {edge_stats.get('status', 'N/A')}")
            print(f"[边端] 缓存命中率: {edge_stats.get('cache_stats', {}).get('hit_rate', 0):.2%}")
        except Exception as e:
            print(f"[边端] 无法获取统计: {e}")
        
        # 云端统计
        cloud_endpoint = self.config['communication']['cloud_endpoint']
        try:
            client = SimpleHTTPClient(cloud_endpoint)
            cloud_stats = await client.health_check()
            print(f"\n[云端] 状态: {cloud_stats.get('status', 'N/A')}")
            print(f"[云端] 平均接受率: {cloud_stats.get('verification_stats', {}).get('avg_acceptance_rate', 0):.2%}")
        except Exception as e:
            print(f"[云端] 无法获取统计: {e}")
    
    def show_config(self):
        """显示配置"""
        print("\n=== 当前配置 ===")
        print(f"边端: {self.config.get('edge', {}).get('server', {}).get('host', 'N/A')}:{self.config.get('edge', {}).get('server', {}).get('port', 'N/A')}")
        print(f"云端: {self.config.get('cloud', {}).get('server', {}).get('host', 'N/A')}:{self.config.get('cloud', {}).get('server', {}).get('port', 'N/A')}")
        
        features = self.config.get('inference', {}).get('features', {})
        print(f"Draft-Verify: {features.get('use_draft_verify', True)}")
        print(f"置信度检查: {features.get('use_confidence_check', True)}")
        print(f"KV Cache: {features.get('use_kv_cache', True)}")


async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="云边端推理框架")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/config.yaml",
        help="配置文件路径"
    )
    parser.add_argument(
        "--mode", 
        type=str, 
        default="interactive",
        choices=["interactive", "server", "client"],
        help="运行模式"
    )
    parser.add_argument(
        "--prompt", 
        type=str,
        help="直接运行推理 (client 模式)"
    )
    parser.add_argument(
        "--max-tokens", 
        type=int, 
        default=128,
        help="最大生成token数"
    )
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=0.8,
        help="温度参数"
    )
    parser.add_argument(
        "--no-draft-verify", 
        action="store_true",
        help="禁用 Draft-Verify"
    )
    parser.add_argument(
        "--no-confidence-check", 
        action="store_true",
        help="禁用置信度检查"
    )
    
    args = parser.parse_args()
    
    # 创建框架实例
    framework = InferenceFramework(args.config)
    
    if args.mode == "server":
        # 服务器模式: 启动边端和云端服务器
        print("[Framework] 服务器模式")
        await framework.start_servers()
        
        try:
            # 保持运行
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            print("\n[Framework] 收到停止信号")
        finally:
            await framework.stop_servers()
    
    elif args.mode == "client":
        # 客户端模式: 运行单次推理
        print("[Framework] 客户端模式")
        
        if not args.prompt:
            print("[Framework] 错误: 客户端模式需要提供 --prompt 参数")
            return
        
        result = await framework.run_inference(
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            use_draft_verify=not args.no_draft_verify,
            use_confidence_check=not args.no_confidence_check
        )
        
        if 'error' in result:
            print(f"[Framework] 推理失败: {result['error']}")
        else:
            print(f"\n[结果] {result.get('text', 'N/A')}")
            print(f"[延迟] {result.get('total_latency_ms', 0):.2f}ms")
            print(f"[置信度] {result.get('confidence_score', 0):.2%}")
            print(f"[接受率] {result.get('acceptance_rate', 0):.2%}")
    
    else:
        # 交互模式
        await framework.run_interactive_mode()


if __name__ == "__main__":
    asyncio.run(main())
                                                  