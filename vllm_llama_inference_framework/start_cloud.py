#!/usr/bin/env python3
"""
云端服务器启动脚本 (修正版：支持 Web 服务启动)
"""
import asyncio
import argparse
import yaml
import sys
import os

# 确保能找到 cloud 模块
sys.path.append(os.getcwd())

from aiohttp import web
from cloud.cloud_server import (
    CloudServer, 
    handle_verify, 
    handle_batch_verify, 
    handle_direct_inference, 
    handle_health, 
    handle_cache_stats
)

async def main():
    parser = argparse.ArgumentParser(description="启动云端服务器")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="配置文件路径")
    args = parser.parse_args()
    
    # 1. 加载配置
    try:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"❌ 配置文件 {args.config} 不存在")
        return

    # 获取 cloud 部分配置
    cloud_config = config.get('cloud', {})
    port = cloud_config.get('server', {}).get('port', 8081)
    
    # 2. 初始化核心逻辑 (这里会加载 vLLM 模型，比较慢，请耐心等待)
    print("="*40)
    print("正在初始化 vLLM (4x V100)...")
    server = CloudServer(cloud_config)
    await server.start() # 这里的 start 只打印日志
    
    # 3. 构建 Web 应用 (这是之前缺失的部分)
    app = web.Application()
    app['cloud_server'] = server
    
    # 注册路由
    app.router.add_post('/verify', handle_verify)
    app.router.add_post('/verify/batch', handle_batch_verify)
    app.router.add_post('/inference/direct', handle_direct_inference)
    app.router.add_get('/health', handle_health)
    app.router.add_get('/cache/stats', handle_cache_stats)
    
    # 4. 启动 HTTP 服务
    print(f"✅ Cloud Server 启动成功! 监听端口: {port}")
    print("="*40)
    
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', port)
    await site.start()
    
    # 5. 保持运行
    try:
        while True:
            await asyncio.sleep(3600)
    except KeyboardInterrupt:
        print("\n正在停止服务器...")
        await runner.cleanup()

if __name__ == "__main__":
    asyncio.run(main())