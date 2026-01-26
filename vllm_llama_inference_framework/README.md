# 基于 vLLM + llama.cpp 的云边端推理框架

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

一个模块化的云边端分布式推理框架，支持置信度判断、Draft-Verify 机制和 KV Cache 优化。

## 🌟 核心特性

- **模块化设计**: F1-F4 四个核心模块独立，便于消融实验
- **云边协同**: 边端(llama.cpp)生成 Draft，云端(vLLM)验证修正
- **置信度判断**: 多种策略评估生成质量
- **KV Cache 优化**: 边端和云端各自优化的缓存策略
- **HTTP 通信**: 简单的 RESTful API 接口
- **消融实验**: 内置完整的实验框架

## 🏗️ 架构

```
┌─────────────────┐    HTTP    ┌─────────────────┐
│   Edge (llama.cpp) │─────────│ Cloud (vLLM)    │
│   - Draft生成      │          │   - Draft验证    │
│   - 置信度判断     │          │   - 结果修正     │
│   - KV Cache      │          │   - KV Cache    │
└─────────────────┘          └─────────────────┘
```

## 📦 核心模块 (F1-F4)

### F1: 置信度判断逻辑 (edge/confidence.py)
基于概率分布的置信度计算，支持多种策略:
- **MAX_PROB**: 最大概率策略
- **ENTROPY**: 熵值策略 (熵越低置信度越高)  
- **TEMPERATURE**: 温度缩放策略
- **TOP_K_AGG**: Top-K 聚合策略

### F2: Draft-Verify 机制 (edge/draft_generator.py, cloud/draft_verifier.py)
- **边端生成**: 使用轻量级模型快速生成 Draft tokens
- **云端验证**: 使用大模型验证并修正 Draft
- **接受率统计**: 监控 Draft 质量

### F3: KV Cache 管理 (edge/kv_cache.py, cloud/kv_cache.py)
- **边端缓存**: LRU 淘汰，前缀匹配
- **云端缓存**: 块分配，分布式同步
- **消融支持**: 可禁用特定功能

### F4: HTTP 通信 (common/http_client.py, common/http_server.py)
- **异步通信**: aiohttp 实现
- **自动重试**: 可配置的重试机制
- **连接池**: 高效的连接管理

## 🚀 快速开始

### 安装

```bash
pip install -r requirements.txt
```

### 启动服务

```bash
# 启动边端服务器
python start_edge.py --config config/config.yaml

# 启动云端服务器  
python start_cloud.py --config config/config.yaml
```

### 运行推理

```bash
# 交互模式
python main.py --mode interactive

# 客户端模式
python main.py --mode client --prompt "What is artificial intelligence?"
```

## 📖 使用示例

### 基本推理

```python
import asyncio
from common.http_client import EdgeCloudHTTPClient

async def main():
    async with EdgeCloudHTTPClient() as client:
        result = await client.full_inference_pipeline(
            prompt="Explain quantum computing",
            max_tokens=256,
            use_draft_verify=True,
            use_confidence_check=True
        )
        
        print(f"结果: {result.text}")
        print(f"总延迟: {result.total_latency_ms:.2f}ms")
        print(f"接受率: {result.acceptance_rate:.2%}")

asyncio.run(main())
```

### 消融实验

```bash
# 运行所有消融实验
python ablation_experiments.py
```

支持的实验:
- **baseline**: 所有功能启用
- **no_confidence**: 禁用置信度判断
- **no_draft_verify**: 禁用 Draft-Verify
- **no_kv_cache**: 禁用 KV Cache
- **edge_only**: 只使用边端
- **cloud_only**: 只使用云端

## ⚙️ 配置

### 基本配置 (config/config.yaml)

```yaml
edge:
  model:
    path: "models/llama-7b-q4.gguf"
  confidence:
    strategy: "max_prob"
    threshold: 0.8
  kv_cache:
    enabled: true
    max_size: 1000

cloud:
  model:
    path: "models/vllm-llama-13b"
  draft_verifier:
    acceptance_threshold: 0.8
  kv_cache:
    enabled: true
    max_blocks: 10000
```

### 消融实验配置

```yaml
experiments:
  ablations:
    - name: "no_confidence"
      description: "禁用置信度判断"
      config_overrides:
        inference:
          features:
            use_confidence_check: false
```

## 📊 性能指标

### 基准性能

| 模式 | 平均延迟 | 吞吐量 | 接受率 |
|------|---------|--------|--------|
| 基准 (全功能) | ~50ms | ~20 req/s | 85% |
| 禁用置信度 | ~45ms | ~22 req/s | 80% |
| 禁用 Draft-Verify | ~200ms | ~5 req/s | N/A |
| 禁用 KV Cache | ~60ms | ~15 req/s | 85% |
| 仅边端 | ~30ms | ~30 req/s | N/A |
| 仅云端 | ~250ms | ~4 req/s | N/A |

*注: 实际性能取决于模型大小和硬件配置*

## 🔧 高级功能

### 自定义置信度策略

```python
from edge.confidence import ConfidenceEnsemble
from common.types import ConfidenceStrategy

# 集成多种策略
ensemble = ConfidenceEnsemble([
    ConfidenceStrategy.MAX_PROB,
    ConfidenceStrategy.ENTROPY,
    ConfidenceStrategy.TOP_K_AGG
])

score, individual = ensemble.ensemble_confidence(
    token_probs,
    weights=[0.4, 0.3, 0.3]
)
```

### 批量推理

```python
import asyncio
from common.http_client import EdgeCloudHTTPClient

async def batch_inference(prompts):
    async with EdgeCloudHTTPClient() as client:
        tasks = [
            client.full_inference_pipeline(prompt=prompt, max_tokens=128)
            for prompt in prompts
        ]
        return await asyncio.gather(*tasks)

# 使用
results = asyncio.run(batch_inference(["What is AI?", "How does ML work?"]))
```

### 健康检查

```bash
# 检查边端
curl http://localhost:8080/health

# 检查云端
curl http://localhost:8081/health
```

## 🧪 测试

### 运行模块测试

```bash
python test_framework.py
```

### 测试覆盖率

```bash
pytest --cov=edge --cov=cloud --cov=common --cov-report=html
```

## 📈 监控

### 性能指标

框架内置性能监控:

```python
from common.http_client import HTTPClient

client = HTTPClient("http://localhost:8080")
stats = client.get_client_stats()

print(f"请求总数: {stats['requests_sent']}")
print(f"平均延迟: {stats['avg_latency_ms']:.2f}ms")
print(f"错误数: {stats['errors']}")
```

### 缓存统计

```bash
# 边端缓存
curl http://localhost:8080/cache/stats

# 云端缓存
curl http://localhost:8081/cache/stats
```

## 🔍 故障排除

### 常见问题

**1. 连接失败**
```
Error: Cannot connect to edge/cloud server
```
- ✅ 检查服务器是否启动
- ✅ 检查端口号是否正确
- ✅ 检查防火墙设置

**2. 模型加载失败**
```
Error: Model not found
```
- ✅ 检查模型路径是否正确
- ✅ 确保模型文件存在
- ✅ 检查模型格式 (gguf for llama.cpp)

**3. 内存不足**
```
Error: Out of memory
```
- ✅ 减少批处理大小
- ✅ 减少 KV Cache 大小
- ✅ 使用更小的模型

**4. 超时错误**
```
Error: Request timeout
```
- ✅ 增加超时时间
- ✅ 检查网络连接
- ✅ 减少生成长度

### 启用调试日志

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📚 文档

- [使用指南](USAGE.md) - 详细的使用说明
- [API 文档](docs/API.md) - API 接口文档
- [架构设计](docs/ARCHITECTURE.md) - 架构设计文档

## 🤝 贡献

欢迎贡献! 请阅读 [CONTRIBUTING.md](CONTRIBUTING.md) 了解详情。

### 开发流程

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

## 🙏 致谢

- [vLLM](https://github.com/vllm-project/vllm) - 高性能推理引擎
- [llama.cpp](https://github.com/ggerganov/llama.cpp) - CPU 优化的 LLM 推理
- [aiohttp](https://github.com/aio-libs/aiohttp) - 异步 HTTP 客户端/服务器

## 📞 联系方式

- 项目维护者: [Your Name](mailto:your.email@example.com)
- 项目链接: [https://github.com/yourusername/vllm-llama-inference-framework](https://github.com/yourusername/vllm-llama-inference-framework)

## 🗺️ 路线图

- [ ] 支持更多模型类型
- [ ] 添加 WebSocket 通信
- [ ] 实现动态负载均衡
- [ ] 集成更多推理引擎
- [ ] 添加可视化监控面板
- [ ] 支持流式输出

---

**⭐ 如果这个项目对你有帮助，请给个 Star!**
