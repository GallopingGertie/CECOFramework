# 云边端推理框架 - 使用指南

## 目录

1. [快速开始](#快速开始)
2. [模块说明](#模块说明)
3. [配置详解](#配置详解)
4. [运行模式](#运行模式)
5. [消融实验](#消融实验)
6. [性能优化](#性能优化)
7. [故障排除](#故障排除)

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 启动服务器

**启动边端服务器:**
```bash
python start_edge.py --config config/config.yaml
```

**启动云端服务器:**
```bash
python start_cloud.py --config config/config.yaml
```

**或者使用主程序启动所有服务:**
```bash
python main.py --mode server --config config/config.yaml
```

### 3. 运行推理

**交互模式:**
```bash
python main.py --mode interactive
```

**客户端模式:**
```bash
python main.py --mode client --prompt "What is artificial intelligence?"
```

## 模块说明

### F1: 置信度判断模块 (confidence.py)

**功能:**
- 基于概率分布计算置信度
- 支持多种置信度策略
- 可配置的置信度阈值

**核心类:**
- `ConfidenceCalculator`: 置信度计算器
- `ConfidenceEnsemble`: 集成多种策略
- `AblatedConfidenceCalculator`: 用于消融实验

**使用示例:**
```python
from edge.confidence import ConfidenceCalculator, ConfidenceStrategy
from common.types import TokenProb

# 创建计算器
calculator = ConfidenceCalculator(strategy=ConfidenceStrategy.MAX_PROB)

# 计算置信度
token_probs = [TokenProb(token_id=100, token="hello", prob=0.9, logprob=-0.105)]
metrics = calculator.calculate_confidence(token_probs)

# 判断是否接受
should_accept = calculator.should_accept_draft(metrics, threshold=0.8)
```

**支持的策略:**
- `MAX_PROB`: 最大概率策略
- `ENTROPY`: 熵值策略 (熵越低置信度越高)
- `TEMPERATURE`: 温度缩放策略
- `TOP_K_AGG`: Top-K 聚合策略

### F2: Draft-Verify 模块 (draft_verifier.py)

**功能:**
- 边端生成 Draft tokens
- 云端验证并修正
- 支持 speculative decoding

**核心类:**
- `DraftGenerator`: 边端 Draft 生成器
- `DraftVerifier`: 云端验证器
- `DraftManager`: Draft 管理器

**使用示例:**
```python
from edge.draft_generator import DraftGenerator
from cloud.draft_verifier import DraftVerifier
from common.types import DraftRequest, VerifyRequest

# 边端生成 Draft
draft_gen = DraftGenerator("models/llama-7b-q4.gguf")
draft_request = DraftRequest(prompt="What is AI?", max_tokens=10)
draft_response = await draft_gen.generate_draft(draft_request)

# 云端验证
verifier = DraftVerifier("models/vllm-llama-13b")
verify_request = VerifyRequest(
    prompt="What is AI?",
    draft_tokens=draft_response.draft_tokens,
    draft_token_ids=draft_response.draft_token_ids
)
verify_response = await verifier.verify_draft(verify_request)

print(f"接受率: {verify_response.acceptance_rate:.2%}")
```

### F3: KV Cache 模块 (kv_cache.py)

**边端 KV Cache (llama.cpp):**
- `LlamaCppKVCache`: 边端缓存管理器
- LRU 淘汰策略
- 前缀匹配支持

**云端 KV Cache (vLLM):**
- `VLLMKVCache`: 云端缓存管理器
- 块分配策略
- 分布式缓存同步

**使用示例:**
```python
# 边端 KV Cache
from edge.kv_cache import LlamaCppKVCache

edge_cache = LlamaCppKVCache(max_size=1000)
edge_cache.set_cache(prompt="Hello", token_ids=[100, 101], available_tokens=64)
cache_data = edge_cache.get_cache("Hello")

# 云端 KV Cache
from cloud.kv_cache import VLLMKVCache

cloud_cache = VLLMKVCache(max_blocks=10000)
blocks = cloud_cache.allocate_blocks(5, "prompt_hash")
cloud_cache.set_cache_blocks("prompt_hash", blocks, seq_len=80)
```

### F4: HTTP 通信模块 (http_client.py, http_server.py)

**功能:**
- 异步 HTTP 客户端
- 自动重试机制
- 连接池管理
- 简单的 HTTP 服务器

**核心类:**
- `HTTPClient`: HTTP 客户端
- `EdgeCloudHTTPClient`: 边端-云端通信客户端
- `HTTPServer`: HTTP 服务器基类

**使用示例:**
```python
from common.http_client import EdgeCloudHTTPClient

# 完整推理流程
async with EdgeCloudHTTPClient() as client:
    result = await client.full_inference_pipeline(
        prompt="What is AI?",
        max_tokens=128,
        use_draft_verify=True,
        use_confidence_check=True
    )
    print(f"结果: {result.text}")
    print(f"延迟: {result.total_latency_ms:.2f}ms")
```

## 配置详解

### 配置文件结构

```yaml
# 边端配置
edge:
  model:
    path: "models/llama-7b-q4.gguf"
    context_length: 2048
    gpu_layers: 0
  
  server:
    host: "localhost"
    port: 8080
  
  confidence:  # F1 配置
    strategy: "max_prob"
    threshold: 0.8
  
  kv_cache:  # F3 边端配置
    enabled: true
    max_size: 1000

# 云端配置
cloud:
  model:
    path: "models/vllm-llama-13b"
    tensor_parallel_size: 1
    gpu_memory_utilization: 0.9
  
  server:
    host: "localhost"
    port: 8081
  
  draft_verifier:  # F2 配置
    acceptance_threshold: 0.8
  
  kv_cache:  # F3 云端配置
    enabled: true
    max_blocks: 10000

# 通信配置 (F4)
communication:
  edge_endpoint: "http://localhost:8080"
  cloud_endpoint: "http://localhost:8081"
  http_client:
    timeout: 30.0
    max_retries: 3

# 推理配置
inference:
  features:
    use_draft_verify: true      # F2
    use_confidence_check: true  # F1
    use_kv_cache: true          # F3
```

### 消融实验配置

在配置文件中可以设置消融实验:

```yaml
experiments:
  ablations:
    - name: "no_confidence"
      description: "禁用置信度判断"
      config_overrides:
        inference:
          features:
            use_confidence_check: false
    
    - name: "no_draft_verify"
      description: "禁用 Draft-Verify"
      config_overrides:
        inference:
          features:
            use_draft_verify: false
```

## 运行模式

### 1. 服务器模式

启动所有服务:
```bash
python main.py --mode server
```

单独启动边端:
```bash
python start_edge.py --port 8080
```

单独启动云端:
```bash
python start_cloud.py --port 8081
```

### 2. 交互模式

```bash
python main.py --mode interactive
```

在交互模式中，可以输入:
- 任意文本进行推理
- `stats` 查看统计信息
- `config` 查看当前配置
- `quit` 或 `exit` 退出

### 3. 客户端模式

```bash
python main.py --mode client --prompt "What is AI?"
```

参数:
- `--prompt`: 输入提示
- `--max-tokens`: 最大生成token数 (默认 128)
- `--temperature`: 温度参数 (默认 0.8)
- `--no-draft-verify`: 禁用 Draft-Verify
- `--no-confidence-check`: 禁用置信度检查

### 4. 测试模式

运行模块测试:
```bash
python test_framework.py
```

## 消融实验

### 运行消融实验

```bash
python ablation_experiments.py
```

### 支持的实验

1. **baseline**: 所有功能启用的基准
2. **no_confidence**: 禁用置信度判断
3. **no_draft_verify**: 禁用 Draft-Verify
4. **no_kv_cache**: 禁用 KV Cache
5. **edge_only**: 只使用边端
6. **cloud_only**: 只使用云端

### 实验结果

实验结果会保存到 `ablation_report.json`，包含:
- 每个实验的详细结果
- 延迟对比
- 成功率统计
- 配置变更记录

### 添加自定义实验

在 `config/config.yaml` 中添加:

```yaml
experiments:
  ablations:
    - name: "my_experiment"
      description: "自定义实验"
      config_overrides:
        edge:
          confidence:
            threshold: 0.9  # 修改置信度阈值
```

## 性能优化

### 1. KV Cache 优化

**边端:**
```yaml
edge:
  kv_cache:
    max_size: 2000  # 增加缓存大小
    enable_compression: true  # 启用压缩
```

**云端:**
```yaml
cloud:
  kv_cache:
    max_blocks: 20000  # 增加块数
    enable_prefix_caching: true  # 启用前缀缓存
```

### 2. 批处理优化

```yaml
cloud:
  draft_verifier:
    batch:
      enabled: true
      max_batch_size: 32
      timeout_ms: 10
```

### 3. HTTP 优化

```yaml
communication:
  http_client:
    timeout: 60.0  # 增加超时时间
    max_retries: 5  # 增加重试次数
    connection_pool_size: 100  # 增加连接池大小
```

### 4. 模型优化

**边端:**
- 使用量化模型 (Q4, Q8)
- 减少上下文长度
- 使用 CPU 推理

**云端:**
- 使用张量并行
- 增加 GPU 内存利用率
- 启用 FlashAttention

## 故障排除

### 常见问题

**1. 连接失败**
```
Error: Cannot connect to edge/cloud server
```
- 检查服务器是否启动
- 检查端口号是否正确
- 检查防火墙设置

**2. 模型加载失败**
```
Error: Model not found
```
- 检查模型路径是否正确
- 确保模型文件存在
- 检查模型格式 (gguf for llama.cpp)

**3. 内存不足**
```
Error: Out of memory
```
- 减少批处理大小
- 减少 KV Cache 大小
- 使用更小的模型

**4. 超时错误**
```
Error: Request timeout
```
- 增加超时时间
- 检查网络连接
- 减少生成长度

### 调试模式

启用详细日志:
```python
# 在代码中设置
import logging
logging.basicConfig(level=logging.DEBUG)
```

或者在配置文件中:
```yaml
logging:
  level: "DEBUG"
```

### 性能分析

使用内置的性能监控:
```python
from common.http_client import HTTPClient

client = HTTPClient("http://localhost:8080")
# ... 使用客户端
stats = client.get_client_stats()
print(f"平均延迟: {stats['avg_latency_ms']:.2f}ms")
```

### 健康检查

```bash
# 检查边端
curl http://localhost:8080/health

# 检查云端
curl http://localhost:8081/health
```

## 示例代码

### 完整推理流程

```python
import asyncio
from common.http_client import EdgeCloudHTTPClient

async def main():
    async with EdgeCloudHTTPClient() as client:
        # 单次推理
        result = await client.full_inference_pipeline(
            prompt="Explain quantum computing",
            max_tokens=256,
            use_draft_verify=True,
            use_confidence_check=True
        )
        
        print(f"结果: {result.text}")
        print(f"总延迟: {result.total_latency_ms:.2f}ms")
        print(f"边端延迟: {result.edge_latency_ms:.2f}ms")
        print(f"云端延迟: {result.cloud_latency_ms:.2f}ms")
        print(f"接受率: {result.acceptance_rate:.2%}")
        print(f"置信度: {result.confidence_score:.2%}")

asyncio.run(main())
```

### 批量推理

```python
import asyncio
from common.http_client import EdgeCloudHTTPClient

async def batch_inference(prompts):
    async with EdgeCloudHTTPClient() as client:
        tasks = []
        for prompt in prompts:
            task = client.full_inference_pipeline(
                prompt=prompt,
                max_tokens=128
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        return results

# 使用
prompts = ["What is AI?", "How does ML work?", "Explain deep learning."]
results = asyncio.run(batch_inference(prompts))
```

### 自定义置信度策略

```python
from edge.confidence import ConfidenceCalculator, ConfidenceEnsemble
from common.types import ConfidenceStrategy

# 使用单一策略
calculator = ConfidenceCalculator(strategy=ConfidenceStrategy.ENTROPY)

# 使用集成策略
ensemble = ConfidenceEnsemble([
    ConfidenceStrategy.MAX_PROB,
    ConfidenceStrategy.ENTROPY,
    ConfidenceStrategy.TOP_K_AGG
])

# 设置权重
score, individual = ensemble.ensemble_confidence(
    token_probs,
    weights=[0.4, 0.3, 0.3]
)
```

## 贡献指南

欢迎贡献代码! 请遵循以下步骤:

1. Fork 项目
2. 创建特性分支
3. 添加测试
4. 确保测试通过
5. 提交 Pull Request

## 许可证

MIT License

## 联系方式

如有问题，请提交 GitHub Issue 或联系维护者。
