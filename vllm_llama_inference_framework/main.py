"""
统一入口脚本 (Client Mode)
用于向 Edge Server 发送推理请求
"""
import argparse
import asyncio
import sys
import aiohttp
import time
import json
from typing import Optional

# 默认配置
DEFAULT_EDGE_URL = "http://localhost:8080"

async def send_inference_request(
    prompt: str,
    edge_url: str = DEFAULT_EDGE_URL,
    max_tokens: int = 128,
    temperature: float = 0.7
):
    """发送推理请求到 Edge Server"""
    url = f"{edge_url}/inference"
    
    # 构造符合 InferenceRequest 的数据包
    payload = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 0.95,
        "confidence_threshold": 0.8,
        "use_draft_verify": True,     # 开启云边协同
        "use_confidence_check": True  # 开启置信度检查
    }

    print(f"[Client] 正在发送请求到: {url}")
    print(f"[Client] Prompt: {prompt}")

    start_time = time.time()
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    latency = (time.time() - start_time) * 1000
                    
                    # 打印精美结果
                    print("\n" + "="*50)
                    print("🎉 推理成功！")
                    print("="*50)
                    print(f"📝 生成文本: \n{result.get('text', '')}")
                    print("-" * 50)
                    print(f"⏱️ 总耗时: {latency:.2f} ms")
                    
                    # 显示云边协同细节
                    if result.get('used_draft_verify'):
                        print(f"☁️ 触发云端验证: 是")
                        print(f"✅ 接受率: {result.get('acceptance_rate', 0):.2%}")
                        print(f"⚡ Edge耗时: {result.get('edge_latency_ms', 0):.2f} ms")
                        print(f"🌩️ Cloud耗时: {result.get('cloud_latency_ms', 0):.2f} ms")
                    else:
                        print(f"💻 仅使用 Edge 推理 (置信度不足或未启用验证)")
                        print(f"⚡ Edge耗时: {result.get('edge_latency_ms', 0):.2f} ms")
                    print("="*50 + "\n")
                    
                else:
                    error_text = await response.text()
                    print(f"❌ 请求失败 (Status {response.status}): {error_text}")
                    
    except aiohttp.ClientConnectorError:
        print(f"❌ 连接失败: 无法连接到 Edge Server ({url})")
        print("请检查: python start_edge.py 是否已在另一个终端成功启动")
    except Exception as e:
        print(f"❌ 发生未知错误: {e}")

async def run_interactive_mode(edge_url: str):
    """交互式聊天模式"""
    print("\n" + "="*40)
    print("🤖 进入交互式聊天模式 (输入 'exit' 退出)")
    print("="*40)
    
    while True:
        try:
            prompt = input("\nUser > ").strip()
            if not prompt:
                continue
            if prompt.lower() in ['exit', 'quit', 'q']:
                print("Bye!")
                break
                
            await send_inference_request(prompt, edge_url)
            
        except KeyboardInterrupt:
            print("\nBye!")
            break

async def main():
    parser = argparse.ArgumentParser(description="vLLM+Llama.cpp 云边协同推理框架客户端")
    
    parser.add_argument("--mode", type=str, choices=["client", "interactive"], default="client", help="运行模式")
    parser.add_argument("--prompt", type=str, default="Hello, who are you?", help="推理提示词")
    parser.add_argument("--url", type=str, default=DEFAULT_EDGE_URL, help="Edge Server 地址")
    
    args = parser.parse_args()

    if args.mode == "client":
        if not args.prompt:
            print("Error: --prompt is required in client mode")
            return
        await send_inference_request(args.prompt, args.url)
        
    elif args.mode == "interactive":
        await run_interactive_mode(args.url)

if __name__ == "__main__":
    try:
        # 统一入口，只调用一次 asyncio.run
        asyncio.run(main())
    except KeyboardInterrupt:
        pass