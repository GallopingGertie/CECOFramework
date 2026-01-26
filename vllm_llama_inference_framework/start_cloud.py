#!/usr/bin/env python3
"""
云端服务器启动脚本
"""
import asyncio
import argparse
import yaml

from cloud.cloud_server import CloudServer


async def main():
    parser = argparse.ArgumentParser(description="启动云端服务器")
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
        print(f"[Cloud] 配置文件不存在，使用默认配置")
        config = {}
    
    # 命令行参数覆盖配置
    cloud_config = config.get('cloud', {})
    
    if args.model_path:
        cloud_config['model_path'] = args.model_path
    if args.port:
        cloud_config['server']['port'] = args.port
    if args.host:
        cloud_config['server']['host'] = args.host
    
    # 创建并启动服务器
    server = CloudServer(cloud_config)
    await server.start()


if __name__ == "__main__":
    asyncio.run(main())
