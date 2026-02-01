import os
from vllm import LLM, SamplingParams
from llama_cpp import Llama

# ================= 路径配置 =================
# 注意：这里使用的是相对路径，因为你刚才已经把 models 文件夹移到了当前目录下
CLOUD_MODEL_PATH = "models/cloud/TinyLlama-1.1B-Chat-v1.0"
EDGE_MODEL_PATH = "models/edge/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
# ===========================================

def test_edge_cpp():
    print("\n" + "="*20 + " 正在测试 Edge (Llama.cpp) " + "="*20)
    if not os.path.exists(EDGE_MODEL_PATH):
        print(f"❌ 错误: 找不到文件 {EDGE_MODEL_PATH}")
        return

    try:
        # 测试加载 GGUF (尝试使用 GPU 加速)
        llm_edge = Llama(
            model_path=EDGE_MODEL_PATH,
            n_ctx=512,
            n_gpu_layers=-1, # 尝试让所有层都上 GPU
            verbose=False
        )
        output = llm_edge("Q: What is the capital of France? A: ", max_tokens=32)
        print(f"✅ Edge 推理成功: {output['choices'][0]['text']}")
        print("🎉 Llama.cpp Edge 端测试通过！")

    except Exception as e:
        print(f"❌ Llama.cpp 启动失败: {e}")

def test_cloud_vllm():
    print("\n" + "="*20 + " 正在测试 Cloud (vLLM 4卡并行) " + "="*20)
    if not os.path.exists(CLOUD_MODEL_PATH):
        print(f"❌ 错误: 找不到路径 {CLOUD_MODEL_PATH}")
        return

    try:
        # 核心测试：4卡 V100 并行加载
        llm = LLM(
            model=CLOUD_MODEL_PATH,
            tensor_parallel_size=4,  # <--- 强制调用 4 张显卡
            dtype="float16",         # <--- V100 必须项
            trust_remote_code=True,
            gpu_memory_utilization=0.6 # 小模型显存给少点，防止和 Edge 抢资源
        )
        
        prompts = ["Hello, I am a"]
        sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=20)
        outputs = llm.generate(prompts, sampling_params)

        for output in outputs:
            generated_text = output.outputs[0].text
            print(f"✅ vLLM 生成结果: {output.prompt!r} -> {generated_text!r}")
        print("🎉 vLLM Cloud 端测试通过！4张显卡火力全开！")
        
    except Exception as e:
        print(f"❌ vLLM 启动失败: {e}")

if __name__ == "__main__":
    # 1. 先测 Edge (通常比较快)
    test_edge_cpp()
    # 2. 再测 Cloud (vLLM 初始化比较慢)
    test_cloud_vllm()