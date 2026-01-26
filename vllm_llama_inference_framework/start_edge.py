#!/usr/bin/env python3
"""
边端服务器启动脚本
"""
import asyncio
import argparse
import yaml

from edge.edge_server import EdgeServer


async def main():
    parser = argparse.ArgumentParser(description="启动边端服务器")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/config.yaml",
        help="配置文件路径"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        help="模型路径"
    )
    parser.add_argument(
        "--port",
        type=int,
        help="端口号"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="主机地址"
    )
    
    args = parser.parse_args()
    
    # 加载配置
    try:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"[Edge] 配置文件不存在，使用默认配置")
        config = {}
    
    # 命令行参数覆盖配置
    edge_config = config.get('edge', {})
    
    if args.model_path:
        edge_config['model_path'] = args.model_path
    if args.port:
        edge_config['server']['port'] = args.port
    if args.host:
        edge_config['server']['host'] = args.host
    
    # 创建并启动服务器
    server = EdgeServer(edge_config)
    await server.start()


if __name__ == "__main__":
    asyncio.run(main())
