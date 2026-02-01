#!/usr/bin/env python3
"""
边端服务器启动脚本 (修正版：支持 Web 服务启动)
"""
import asyncio
import argparse
import yaml
import sys
import os

# 确保能找到模块
sys.path.append(os.getcwd())

from aiohttp import web
from edge.edge_server import (
    EdgeServer, 
    handle_request, 
    handle_inference, 
    handle_cache_stats
)

async def main():
    parser = argparse.ArgumentParser(description="启动边端服务器")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="配置文件路径")
    args = parser.parse_args()
    
    # 1. 加载配置
    try:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"❌ 配置文件 {args.config} 不存在")
        return

    # 获取 edge 部分配置
    edge_config = config.get('edge', {})
    # 尝试从 server 字段获取端口，如果没有则用默认的
    server_config = edge_config.get('server', {})
    port = server_config.get('port', 8080)
    
    print("="*40)
    print(f"正在启动 Edge 端 (Llama.cpp)...")
    
    # 2. 初始化核心逻辑
    server = EdgeServer(edge_config)
    await server.start()
    
    # 3. 构建 Web 应用
    app = web.Application()
    app['edge_server'] = server
    
    # 注册路由
    # 注意：根据 edge_server.py 里的定义，handle_request 处理了 draft 和 health
    app.router.add_post('/draft', handle_request)
    app.router.add_post('/inference', handle_inference)
    app.router.add_get('/health', handle_request)
    app.router.add_get('/cache/stats', handle_cache_stats)
    
    print(f"✅ Edge Server 启动成功! 监听端口: {port}")
    print("="*40)
    
    # 4. 启动 HTTP 服务
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', port)
    await site.start()
    
    # 5. 保持运行
    try:
        while True:
            await asyncio.sleep(3600)
    except KeyboardInterrupt:
        print("\n正在停止 Edge 服务器...")
        await server.stop()
        await runner.cleanup()

if __name__ == "__main__":
    asyncio.run(main())