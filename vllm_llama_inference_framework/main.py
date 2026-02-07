"""
主程序入口 (Client 端测试脚本) - 最终修复版
包含:
1. 强制 IPv4 直连 (解决 Windows 卡顿)
2. 智能结果显示 (正确区分 Edge/Cloud/协同模式)
"""
import argparse
import asyncio
import aiohttp
import time
import json
import sys

# 默认配置 (强制使用 IPv4 + 8088 端口)
DEFAULT_EDGE_URL = "http://127.0.0.1:8088"

async def send_inference_request(url: str, prompt: str):
    """发送推理请求到 Edge Server"""
    print(f"[Client] 正在发送请求到: {url}/inference")
    print(f"[Client] Prompt: {prompt}")
    
    # 构造符合 InferenceRequest 定义的请求体
    payload = {
        "prompt": prompt,
        "max_tokens": 128,
        "temperature": 0.7,
        "top_p": 0.9,
        "use_draft_verify": True,     # 允许协同
        "use_confidence_check": True, # 允许置信度检查
        "confidence_threshold": 0.8,  # 设置置信度阈值
        
        # 模拟高优先级的任务需求 (可选)
        "requirements": {
            "max_latency_ms": 5000,
            "min_quality_score": 0.8,
            "priority": 1
        }
    }

    try:
        timeout = aiohttp.ClientTimeout(total=600) # 设置较长超时，防止云端处理慢断开
        async with aiohttp.ClientSession(timeout=timeout) as session:
            start_time = time.time()
            
            async with session.post(f"{url}/inference", json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    
                    # 打印最终结果
                    print_result(result, prompt)
                else:
                    error_text = await response.text()
                    print(f"❌ 请求失败 (Status {response.status}): {error_text}")
                    
    except aiohttp.ClientConnectorError:
        print(f"❌ 连接失败: 无法连接到 {url}")
        print("💡 提示: 请检查 Edge Server 是否已启动 (python start_edge.py)")
        print("💡 提示: 请检查端口是否正确 (默认为 8088)")
    except Exception as e:
        print(f"❌ 发生未知错误: {e}")

def print_result(result: dict, prompt: str):
    """美化打印推理结果"""
    text = result.get('text', '')
    
    print("\n" + "="*50)
    print("🎉 推理成功！")
    print("="*50)
    print(f"📝 生成文本:\n{text.strip()}")
    print("-" * 50)
    
    # 获取各项数据
    used_verify = result.get('used_draft_verify', False)
    edge_lat = result.get('edge_latency_ms', 0.0)
    cloud_lat = result.get('cloud_latency_ms', 0.0)
    total_lat = result.get('total_latency_ms', 0.0)
    acc_rate = result.get('acceptance_rate', 0.0)
    strategy = result.get('strategy', 'unknown') # 获取 F1 决策策略名称
    
    print(f"⏱️ 总耗时: {total_lat:.2f} ms")
    
    # ==================== 智能判定模式 (核心修复) ====================
    
    # 1. 纯云端模式 (Cloud Direct)
    # 特征: 策略是 cloud_direct，或者 云端有耗时但端侧耗时为0
    if strategy == 'cloud_direct' or (cloud_lat > 0 and edge_lat == 0):
        print(f"☁️ 仅使用 Cloud 推理 (F1决策: 纯云端)")
        print(f"☁️ Cloud耗时: {cloud_lat:.2f} ms")
        print(f"⚡ Edge耗时: 0.00 ms (跳过)")

    # 2. 协同推理模式 (Speculative)
    # 特征: 使用了 verify，或者云边都有耗时
    elif used_verify or (cloud_lat > 0 and edge_lat > 0):
        print(f"🤝 协同推理模式 (Acceptance Rate: {acc_rate:.2%})")
        print(f"⚡ Edge耗时: {edge_lat:.2f} ms (Draft生成)")
        print(f"☁️ Cloud耗时: {cloud_lat:.2f} ms (验证)")

    # 3. 纯端侧模式 (Edge Only)
    # 特征: 只有端侧耗时
    else:
        print(f"💻 仅使用 Edge 推理 (F1决策: 纯端侧)")
        print(f"⚡ Edge耗时: {edge_lat:.2f} ms")
        print(f"☁️ Cloud耗时: 0.00 ms (未启用)")
        
    print("="*50 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Edge-Cloud Inference Client")
    parser.add_argument("--mode", type=str, default="client", choices=["client"], help="运行模式")
    parser.add_argument("--prompt", type=str, default="Hello, AI!", help="测试提示词")
    parser.add_argument("--url", type=str, default=DEFAULT_EDGE_URL, help="Edge Server 地址")
    
    args = parser.parse_args()
    
    # Windows 平台下的 asyncio 策略调整 (防止 Event Loop 报错)
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    if args.mode == "client":
        # 强制修正 URL，防止用户手误输入 localhost
        target_url = args.url
        if "localhost" in target_url:
            print("[Client] ⚠️ 检测到 localhost，自动转换为 127.0.0.1 以避免 Windows IPv6 问题...")
            target_url = target_url.replace("localhost", "127.0.0.1")
            
        asyncio.run(send_inference_request(target_url, args.prompt))

if __name__ == "__main__":
    main()