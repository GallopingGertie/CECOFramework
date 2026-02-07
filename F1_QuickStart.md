# F1 智能决策模块 - 快速开始指南

## 概览

F1 模块已从单纯的置信度计算器升级为智能决策中枢，具备：
- ✅ 硬件监控与 SLO 决策
- ✅ 网络感知（RTT 探测、弱网检测）
- ✅ 历史追踪与自适应调整

## 快速验证

### 1. 运行核心模块测试

```bash
cd /Users/hefen/Desktop/husband/20260203/CECOFramework-main_new/vllm_llama_inference_framework

# 运行阶段3核心模块测试（无外部依赖）
python3 tests/test_stage3_simple.py
```

**预期输出**：
```
============================================================
阶段3核心模块测试套件
============================================================
✅ HistoryTracker 所有测试通过
✅ AdaptiveThresholdCalculator 所有测试通过
✅ 统计摘要功能测试通过
============================================================
✅ 所有核心模块测试通过！
============================================================
```

### 2. 检查配置文件

查看 `config/config.yaml` 中的 F1 配置节：

```yaml
edge:
  f1:
    # 硬约束
    hard_constraints:
      cpu_overload: 90.0
      memory_critical: 500
      ultra_low_latency: 50
      weak_network_rtt: 200.0
    
    # 评分权重
    scoring_weights:
      latency: 0.4
      cost: 0.1
      quality: 0.3
    
    # 网络探测
    enable_network_probe: true
    
    # 自适应调整
    enable_adaptive: true
    adaptive_threshold:
      target_acceptance_min: 0.75
      target_acceptance_max: 0.85
      update_interval: 10
```

### 3. 集成到应用

```python
from edge.f1_decision import F1_DecisionModule
from common.types import InferenceRequest, TaskRequirements
import yaml

# 加载配置
with open('config/config.yaml') as f:
    config = yaml.safe_load(f)

# 初始化 F1
f1 = F1_DecisionModule(
    config['edge']['f1'], 
    cloud_endpoint="http://localhost:8081"
)

# 创建请求
request = InferenceRequest(
    prompt="你好，请问...",
    max_tokens=100,
    requirements=TaskRequirements(
        max_latency_ms=150,
        min_quality_score=0.85,
        priority=2
    )
)

# 决策
plan = await f1.decide_async(request)
print(f"策略: {plan.strategy.value}")
print(f"得分: {plan.score:.3f}")
print(f"理由: {plan.reason}")
```

## 核心功能演示

### 1. 硬约束触发

```python
# 场景1：CPU 过载
request = InferenceRequest(
    prompt="...",
    requirements=TaskRequirements(max_latency_ms=200)
)
# 如果 CPU > 90%，F1 会强制选择 CLOUD_DIRECT

# 场景2：超低延迟
request = InferenceRequest(
    prompt="...",
    requirements=TaskRequirements(max_latency_ms=30)  # < 50ms
)
# F1 会强制选择 EDGE_ONLY
```

### 2. 网络感知

```python
# 配置中启用网络探测
config['edge']['f1']['enable_network_probe'] = True

# F1 会自动探测云端 RTT
# 如果 RTT > 200ms，触发弱网保护 → EDGE_ONLY
```

### 3. 历史追踪与自适应

```python
# 记录执行结果
f1.record_execution(
    strategy=ExecutionStrategy.SPECULATIVE_STANDARD,
    acceptance_rate=0.85,
    latency_ms=80.0,
    edge_latency_ms=30.0,
    cloud_latency_ms=50.0,
    confidence_score=0.85,
    success=True,
    tokens_generated=50
)

# 每10次执行，F1 会自动调整置信度阈值
# 高接受率（>0.85）→ 降低阈值（更激进）
# 低接受率（<0.75）→ 提高阈值（更保守）

# 查看统计
stats = f1.get_statistics_summary()
print(f"平均接受率: {stats['recent_acceptance_rate']:.2%}")
print(f"平均延迟: {stats['avg_latency_ms']:.1f}ms")
```

## 测试覆盖

### 阶段1：基础决策 ✅
```bash
# 测试硬约束、多目标评分、降级策略
python3 tests/test_f1_core.py
```

### 阶段2：网络感知 ✅
```bash
# 测试网络探测、弱网检测、网络感知评分
python3 tests/test_stage2_network.py
```

### 阶段3：历史追踪 ✅
```bash
# 测试历史记录、自适应调整、统计分析
python3 tests/test_stage3_simple.py
```

## 关键配置项

| 配置项 | 默认值 | 说明 |
|-------|--------|------|
| `cpu_overload` | 90.0 | CPU 过载阈值（%） |
| `memory_critical` | 500 | 内存临界值（MB） |
| `weak_network_rtt` | 200.0 | 弱网 RTT 阈值（ms） |
| `enable_network_probe` | true | 是否启用网络探测 |
| `enable_adaptive` | true | 是否启用自适应调整 |
| `target_acceptance_min` | 0.75 | 目标接受率下限 |
| `target_acceptance_max` | 0.85 | 目标接受率上限 |
| `update_interval` | 10 | 自适应更新间隔（次） |

## 常见问题

### Q1: 如何禁用网络探测？
在 `config.yaml` 中设置：
```yaml
enable_network_probe: false
```

### Q2: 如何调整评分权重？
修改 `scoring_weights`：
```yaml
scoring_weights:
  latency: 0.5  # 更重视延迟
  cost: 0.2
  quality: 0.3
```

### Q3: 自适应调整太频繁怎么办？
增加 `update_interval`：
```yaml
adaptive_threshold:
  update_interval: 20  # 每20次执行更新一次
```

### Q4: 如何查看决策详情？
F1 会自动打印日志：
```
[F1] 上下文: CPU=45.0%, 内存=2000MB, SLO延迟<150ms, 质量>0.85, 优先级=2, 网络RTT=50.0ms
[F1] 决策完成: speculative_standard (得分=0.856)
```

## 下一步

1. **集成到生产环境**：将 F1 集成到 EdgeServer 的完整推理流程
2. **监控与调优**：观察历史统计，调整配置参数
3. **扩展功能**：根据实际需求添加更多决策维度

## 支持

- **详细文档**：[F1_Implementation_Summary.md](F1_Implementation_Summary.md)
- **测试用例**：`tests/test_f1_*.py`, `tests/test_stage*.py`
- **源代码**：`edge/f1_decision.py`, `edge/decision_engine.py`, `edge/history_tracker.py`

---

**快速开始指南** | 版本 1.0 | 2026-02-04
