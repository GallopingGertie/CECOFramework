"""
Edge Server 启动脚本 - 修复版 (解决 Event Loop 冲突问题)
"""
import argparse
import asyncio
import yaml
import os
from aiohttp import web

# 引入 EdgeServer 和所有的路由处理函数
from edge.edge_server import (
    EdgeServer, 
    handle_request, 
    handle_inference, 
    handle_cache_stats,
    handle_simulation_control
)

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

async def on_startup(app):
    """Web 服务启动时的钩子：此时初始化 Session"""
    print("[System] Web服务已启动，正在初始化 Edge Server 组件...")
    server = app['edge_server']
    await server.start() # 在正确的 Loop 中创建 Session

async def on_cleanup(app):
    """Web 服务关闭时的钩子：清理资源"""
    print("[System] 正在关闭 Edge Server...")
    server = app['edge_server']
    await server.stop()

async def init_app(config_path):
    config = load_config(config_path)
    edge_config = config.get('edge', {})
    
    # 1. 实例化 Server (但不调用 start)
    server = EdgeServer(config)
    
    app = web.Application()
    app['edge_server'] = server
    
    # 2. 注册生命周期钩子 (关键修复!)
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    
    # 3. 注册路由
    app.router.add_post('/v1/chat/completions', handle_inference)
    app.router.add_post('/inference', handle_inference)
    app.router.add_post('/draft', handle_request)
    app.router.add_get('/health', handle_request)
    app.router.add_get('/cache/stats', handle_cache_stats)
    app.router.add_post('/admin/simulate', handle_simulation_control)
    
    return app, edge_config

def main():
    parser = argparse.ArgumentParser(description="Start Edge Server")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config file")
    args = parser.parse_args()

    # 获取端口
    config = load_config(args.config)
    port = config.get('edge', {}).get('server', {}).get('port', 8088)

    print(f"🚀 [Startup] 正在启动 Edge Server 于端口 {port}...")
    
    # 修复：不再手动创建 Loop，直接通过 run_app 管理
    # 先构建 app 工厂
    async def app_factory():
        app, _ = await init_app(args.config)
        return app

    # 使用 web.run_app 自动处理 Loop
    web.run_app(app_factory(), host='0.0.0.0', port=port)

if __name__ == "__main__":
    main()