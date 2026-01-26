"""
F4: HTTP 服务器模块
简单的 HTTP 服务器框架，用于边端和云端
"""
import asyncio
import json
from typing import Dict, Any, Callable, Awaitable, Optional, List
from aiohttp import web, ClientSession
import aiohttp_cors
import time
from functools import wraps


def measure_latency(func):
    """测量函数执行时间的装饰器"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        latency = (time.time() - start_time) * 1000
        
        # 如果结果是字典，添加延迟信息
        if isinstance(result, dict):
            result['latency_ms'] = latency
        
        return result
    
    return wrapper


class HTTPServer:
    """
    HTTP 服务器基类
    
    特点:
    1. 支持路由注册
    2. 中间件支持
    3. CORS 支持
    4. 请求/响应日志
    5. 性能监控
    """
    
    def __init__(
        self, 
        host: str = 'localhost',
        port: int = 8080,
        enable_cors: bool = True,
        enable_metrics: bool = True
    ):
        self.host = host
        self.port = port
        self.enable_cors = enable_cors
        self.enable_metrics = enable_metrics
        
        # Web 应用
        self.app = web.Application()
        
        # 路由表
        self.routes: Dict[str, Callable] = {}
        
        # 中间件
        self.middlewares: List[Callable] = []
        
        # 统计
        self.metrics = {
            'requests_total': 0,
            'requests_success': 0,
            'requests_error': 0,
            'avg_latency_ms': 0.0
        }
        
        # 设置 CORS
        if enable_cors:
            self._setup_cors()
    
    def _setup_cors(self):
        """设置 CORS"""
        cors = aiohttp_cors.setup(self.app, defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
                allow_methods="*"
            )
        })
        
        # 为所有路由添加 CORS
        for resource in list(self.app.router.resources()):
            cors.add(resource)
    
    def register_route(
        self, 
        method: str,
        path: str,
        handler: Callable[[web.Request], Awaitable[Dict[str, Any]]]
    ):
        """
        注册路由
        
        Args:
            method: HTTP 方法 (GET, POST, etc.)
            path: 路径
            handler: 处理器函数
        """
        route_key = f"{method.upper()}:{path}"
        self.routes[route_key] = handler
        
        # 包装处理器
        async def wrapped_handler(request):
            try:
                # 记录请求
                await self._log_request(request)
                
                # 应用中间件
                for middleware in self.middlewares:
                    result = await middleware(request)
                    if result is not None:
                        return result
                
                # 调用处理器
                result = await handler(request)
                
                # 更新指标
                self._update_metrics(success=True)
                
                return web.json_response(result)
            
            except Exception as e:
                self._update_metrics(success=False)
                return web.json_response(
                    {'error': str(e)}, 
                    status=500
                )
        
        # 注册到 aiohttp
        self.app.router.add_route(method.upper(), path, wrapped_handler)
    
    def add_middleware(self, middleware: Callable[[web.Request], Awaitable[Optional[web.Response]]]):
        """添加中间件"""
        self.middlewares.append(middleware)
    
    async def _log_request(self, request: web.Request):
        """记录请求日志"""
        print(f"[HTTP] {request.method} {request.path}")
        
        if request.can_read_body:
            try:
                body = await request.json()
                print(f"[HTTP] 请求体: {json.dumps(body, indent=2)[:200]}...")
            except:
                pass
    
    def _update_metrics(self, success: bool):
        """更新指标"""
        self.metrics['requests_total'] += 1
        
        if success:
            self.metrics['requests_success'] += 1
        else:
            self.metrics['requests_error'] += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取服务器指标"""
        return self.metrics.copy()
    
    async def start(self):
        """启动服务器"""
        runner = web.AppRunner(self.app)
        await runner.setup()
        
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()
        
        print(f"[HTTP] HTTP 服务器运行在 http://{self.host}:{self.port}")
        
        return runner


class EdgeHTTPServer(HTTPServer):
    """边端 HTTP 服务器"""
    
    def __init__(self, edge_server, **kwargs):
        super().__init__(**kwargs)
        self.edge_server = edge_server
        
        # 注册边端路由
        self._register_edge_routes()
    
    def _register_edge_routes(self):
        """注册边端路由"""
        
        @measure_latency
        async def handle_draft(request):
            """处理 Draft 请求"""
            data = await request.json()
            return await self.edge_server.handle_draft_request(data)
        
        @measure_latency
        async def handle_inference(request):
            """处理推理请求"""
            data = await request.json()
            inference_request = InferenceRequest(**data)
            return await self.edge_server.process_inference(inference_request)
        
        async def handle_health(request):
            """健康检查"""
            return await self.edge_server.handle_health_check()
        
        async def handle_cache_stats(request):
            """缓存统计"""
            return self.edge_server.kv_cache.get_cache_stats()
        
        # 注册路由
        self.register_route('POST', '/draft', handle_draft)
        self.register_route('POST', '/inference', handle_inference)
        self.register_route('GET', '/health', handle_health)
        self.register_route('GET', '/cache/stats', handle_cache_stats)


class CloudHTTPServer(HTTPServer):
    """云端 HTTP 服务器"""
    
    def __init__(self, cloud_server, **kwargs):
        super().__init__(**kwargs)
        self.cloud_server = cloud_server
        
        # 注册云端路由
        self._register_cloud_routes()
    
    def _register_cloud_routes(self):
        """注册云端路由"""
        
        @measure_latency
        async def handle_verify(request):
            """处理验证请求"""
            data = await request.json()
            return await self.cloud_server.handle_verify_request(data)
        
        @measure_latency
        async def handle_batch_verify(request):
            """处理批量验证请求"""
            data = await request.json()
            return await self.cloud_server.handle_batch_verify(data)
        
        async def handle_direct_inference(request):
            """处理直接推理请求"""
            data = await request.json()
            return await self.cloud_server.handle_direct_inference(data)
        
        async def handle_health(request):
            """健康检查"""
            return await self.cloud_server.handle_health_check()
        
        async def handle_cache_stats(request):
            """缓存统计"""
            return self.cloud_server.kv_cache.get_cache_stats()
        
        # 注册路由
        self.register_route('POST', '/verify', handle_verify)
        self.register_route('POST', '/verify/batch', handle_batch_verify)
        self.register_route('POST', '/inference/direct', handle_direct_inference)
        self.register_route('GET', '/health', handle_health)
        self.register_route('GET', '/cache/stats', handle_cache_stats)


# 中间件示例
async def logging_middleware(request):
    """日志中间件"""
    start_time = time.time()
    
    # 记录请求
    print(f"[Middleware] {request.method} {request.path} - 开始")
    
    # 继续处理
    response = None
    
    # 记录响应时间
    latency = (time.time() - start_time) * 1000
    print(f"[Middleware] {request.method} {request.path} - 完成 ({latency:.2f}ms)")
    
    return response


async def auth_middleware(request):
    """认证中间件 (示例)"""
    # 检查请求头中的 API Key
    api_key = request.headers.get('X-API-Key')
    
    # 这里应该验证 API Key
    # if not is_valid_api_key(api_key):
    #     return web.json_response({'error': 'Invalid API Key'}, status=401)
    
    # 继续处理
    return None


async def rate_limit_middleware(request):
    """限流中间件 (示例)"""
    # 这里应该实现限流逻辑
    # if is_rate_limited(request.remote):
    #     return web.json_response({'error': 'Rate limit exceeded'}, status=429)
    
    # 继续处理
    return None


class HTTPClientManager:
    """HTTP 客户端管理器"""
    
    def __init__(self):
        self.clients: Dict[str, HTTPClient] = {}
    
    def register_client(
        self, 
        name: str, 
        client: HTTPClient
    ):
        """注册客户端"""
        self.clients[name] = client
    
    def get_client(self, name: str) -> Optional[HTTPClient]:
        """获取客户端"""
        return self.clients.get(name)
    
    async def start_all(self):
        """启动所有客户端"""
        for client in self.clients.values():
            await client.start()
    
    async def stop_all(self):
        """停止所有客户端"""
        for client in self.clients.values():
            await client.stop()
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """获取所有客户端的统计"""
        return {
            name: client.get_client_stats()
            for name, client in self.clients.items()
        }


# 工具函数
async def run_health_check(
    endpoint: str, 
    timeout: float = 5.0
) -> bool:
    """
    运行健康检查
    
    Args:
        endpoint: 端点 URL
        timeout: 超时时间
        
    Returns:
        是否健康
    """
    try:
        async with HTTPClient(endpoint, timeout=timeout) as client:
            response = await client.health_check()
            return response.get('status') == 'healthy'
    except Exception as e:
        print(f"[HealthCheck] 健康检查失败: {e}")
        return False


async def wait_for_service(
    endpoint: str, 
    max_wait: float = 60.0,
    check_interval: float = 2.0
) -> bool:
    """
    等待服务启动
    
    Args:
        endpoint: 端点 URL
        max_wait: 最大等待时间
        check_interval: 检查间隔
        
    Returns:
        服务是否启动
    """
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        if await run_health_check(endpoint):
            print(f"[Wait] 服务 {endpoint} 已就绪")
            return True
        
        await asyncio.sleep(check_interval)
    
    print(f"[Wait] 等待服务 {endpoint} 超时")
    return False
