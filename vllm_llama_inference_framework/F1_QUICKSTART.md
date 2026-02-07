# F1 决策模块 - 快速开始指南

## 🚀 快速测试新的 F1 模块

### 1. 运行单元测试

```bash
cd /Users/hefen/Desktop/husband/CECOFramework-main/vllm_llama_inference_framework
python3 tests/test_f1_core.py
```

**预期输出**:
```
✅ 6 通过, ❌ 0 失败
🎉 所有测试通过！F1 模块核心逻辑正常工作
```

---

### 2. 测试不同场景的决策

创建一个测试脚本 `test_scenarios.py`:

```python
from edge.f1_decision import F1_DecisionModule
from common.types import InferenceRequest, TaskRequirements

# 加载配置
import yaml
with open('config/config.yaml', 'r') as f:
    full_config = yaml.safe_load(f)

f1_config = full_config['edge']['f1']
f1 = F1_DecisionModule(f1_config)

# 场景1: 时敏任务（聊天对话）
print("\n=== 场景1: 聊天对话 ===")
request = InferenceRequest(
    prompt="Hi, how are you?",
    requirements=TaskRequirements(
        max_latency_ms=100,  # 要求快速响应
        priority=2
    )
)
plan = f1.decide(request)
print(f"决策: {plan.strategy.value}")
print(f"理由: {plan.reason}")

# 场景2: 高质量创作
print("\n=== 场景2: 文章创作 ===")
request = InferenceRequest(
    prompt="Write a detailed article about climate change...",
    requirements=TaskRequirements(
        min_quality_score=0.95,  # 要求高质量
        max_latency_ms=5000
    )
)
plan = f1.decide(request)
print(f"决策: {plan.strategy.value}")
print(f"理由: {plan.reason}")

# 场景3: 隐私敏感
print("\n=== 场景3: 隐私数据 ===")
request = InferenceRequest(
    prompt="My credit card is 1234...",
    requirements=TaskRequirements(
        privacy_level=2  # 绝密级别
    )
)
plan = f1.decide(request)
print(f"决策: {plan.strategy.value}")
print(f"理由: {plan.reason}")

# 场景4: 平衡场景
print("\n=== 场景4: 常规问答 ===")
request = InferenceRequest(
    prompt="What is machine learning?",
    requirements=TaskRequirements(
        max_latency_ms=2000,
        min_quality_score=0.8
    )
)
plan = f1.decide(request)
print(f"决策: {plan.strategy.value}")
print(f"得分: {plan.score:.3f}")
print(f"理由: {plan.reason}")
```

运行：
```bash
python3 test_scenarios.py
```

---

### 3. 查看决策日志详情

F1 模块会自动输出详细日志，显示：
- 系统状态（CPU、内存）
- 任务需求（SLO延迟、质量要求、优先级）
- 决策策略和理由
- 执行参数（draft_max_tokens、confidence_threshold等）

---

### 4. 调整配置参数

编辑 `config/config.yaml`：

```yaml
edge:
  f1:
    # 调整硬约束阈值
    hard_constraints:
      cpu_overload: 90.0      # 降低 CPU 阈值，更容易卸载到云端
      ultra_low_latency: 100  # 提高延迟阈值
    
    # 调整评分权重
    scoring_weights:
      latency: 0.5    # 更重视延迟
      cost: 0.2       # 降低成本权重
      quality: 0.3
    
    # 调整延迟估算
    latency_estimates:
      edge_only_ms: 50         # 根据实际测试调整
      cloud_direct_ms: 150
      speculative_standard_ms: 70
```

---

### 5. 集成到完整系统

#### 启动服务器

```bash
# 终端1: 启动云端服务器
python3 start_cloud.py --config config/config.yaml

# 终端2: 启动边端服务器（已集成 F1）
python3 start_edge.py --config config/config.yaml

# 终端3: 发送推理请求
python3 main.py --mode client --prompt "Hello, how are you?"
```

#### 观察 F1 决策

边端服务器日志会显示：
```
[F1] 上下文: CPU=45.0%, 内存=4000MB, SLO延迟<5000ms, 质量>0.80, 优先级=1
[F1] 决策完成: speculative_standard (得分=0.833)
[Edge] F1决策: speculative_standard (得分=0.833, 理由=Score: 0.833)
```

---

## 🔍 故障排除

### 问题1: ImportError: No module named 'psutil'

**解决方案**:
```bash
pip install psutil
```

或者使用不依赖 psutil 的测试：
```bash
python3 tests/test_f1_core.py
```

### 问题2: F1 决策总是选择同一个策略

**检查**:
1. 查看配置文件的权重设置
2. 查看延迟估算是否准确
3. 运行测试查看评分详情：
   ```python
   scorer = MultiObjectiveScorer(config)
   scored = scorer.score_strategies(context)
   for s in scored:
       print(f"{s.strategy.value}: {s.score:.3f}")
   ```

### 问题3: 想禁用 F1 使用旧逻辑

**临时方案**: 在 `edge_server.py` 的 `process_inference` 中注释掉 F1 调用：
```python
# execution_plan = self.f1_decision.decide(inference_request)
# 使用固定策略
from common.types import ExecutionStrategy, ExecutionPlan
execution_plan = ExecutionPlan(
    strategy=ExecutionStrategy.SPECULATIVE_STANDARD,
    params={'draft_max_tokens': 64, 'confidence_threshold': 0.8}
)
```

---

## 📊 查看决策统计

创建一个统计脚本 `stats.py`:

```python
from edge.f1_decision import F1_DecisionModule
from common.types import InferenceRequest, TaskRequirements, ExecutionStrategy
import random

# 初始化
config = {...}
f1 = F1_DecisionModule(config)

# 模拟100个请求
decisions = {s.value: 0 for s in ExecutionStrategy}

for _ in range(100):
    # 随机生成任务
    latency = random.choice([100, 500, 1000, 3000, 5000])
    quality = random.choice([0.6, 0.8, 0.9, 0.95])
    priority = random.choice([1, 2, 3])
    
    request = InferenceRequest(
        prompt="test",
        requirements=TaskRequirements(
            max_latency_ms=latency,
            min_quality_score=quality,
            priority=priority
        )
    )
    
    plan = f1.decide(request)
    decisions[plan.strategy.value] += 1

# 输出统计
print("决策分布:")
for strategy, count in decisions.items():
    print(f"  {strategy}: {count}%")
```

---

## 🎯 下一步

- ✅ 熟悉 F1 决策逻辑
- ✅ 根据实际测试调整配置参数
- ✅ 准备阶段2：网络感知功能

**有问题？** 查看 `F1_IMPLEMENTATION_SUMMARY.md` 获取完整文档。
