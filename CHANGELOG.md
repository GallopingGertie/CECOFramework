# F1 智能决策模块 - 更新日志

## 版本历史

### v3.0.0 - 2026-02-04 ✅ 阶段3完成

**重大更新：历史追踪与自适应调整**

#### 新增功能
- ✅ **HistoryTracker**：滑动窗口历史追踪器
  - 支持记录最近 N 次执行（默认100）
  - 提供多维度统计（接受率、延迟、成功率、策略分布）
  - 分策略查询和分析
  
- ✅ **AdaptiveThresholdCalculator**：自适应参数调整
  - 基于 EMA 平滑算法动态调整置信度阈值
  - 目标接受率范围：75%-85%
  - 边界保护：阈值限制在 [0.50, 0.95]
  - 每 N 次执行自动触发更新（默认10次）

- ✅ **增强评分器**：MultiObjectiveScorer 使用历史数据
  - 延迟评分：优先使用历史实际延迟（样本≥5时）
  - 质量评分：结合历史接受率和成功率
  - 自动学习各策略的实际性能

#### 新增文件
- `edge/history_tracker.py` - 历史追踪器（254行）
- `edge/adaptive_threshold.py` - 自适应计算器（224行）
- `tests/test_stage3_simple.py` - 核心模块测试（330行）
- `tests/test_stage3_adaptive.py` - 完整测试（400行，需pytest）

#### 修改文件
- `edge/decision_engine.py`
  - MultiObjectiveScorer 新增 `history_tracker` 参数
  - `_score_latency()` 优先使用历史实际延迟
  - `_score_quality()` 结合历史接受率和成功率

- `edge/f1_decision.py`
  - 初始化 HistoryTracker 和 AdaptiveThresholdCalculator
  - 新增 `record_execution()` 方法
  - 新增 `_apply_adaptive_updates()` 自动调整参数
  - 新增 `get_statistics_summary()` 和 `get_current_config()` 查询接口
  - `decide_async()` 支持自动触发自适应更新

- `edge/edge_server.py`
  - `process_inference()` 执行后自动记录结果

- `config/config.yaml`
  - 新增 `history_tracker` 配置节
  - 新增 `adaptive_threshold` 配置节
  - 新增 `enable_adaptive` 和 `enable_history_scoring` 开关

#### 测试结果
```
✅ 3/3 测试套件通过
- HistoryTracker：记录、查询、滑动窗口、统计
- AdaptiveThresholdCalculator：高/低接受率调整、边界保护
- 统计摘要：成功率、策略分布、置信度分布
```

#### 示例输出
```
场景1：高接受率（0.95）
[Adaptive] 接受率过高 (95.00%), 降低阈值
[Adaptive] 阈值调整: 0.800 → 0.785

场景2：低接受率（0.60）
[Adaptive] 接受率过低 (60.00%), 提高阈值
[Adaptive] 阈值调整: 0.800 → 0.823
```

---

### v2.0.0 - 2026-02-03 ✅ 阶段2完成

**重大更新：网络感知功能**

#### 新增功能
- ✅ **网络状态监控**：真实 RTT 探测
  - 使用 HTTP HEAD 请求测量云端延迟
  - 自动探测间隔（默认5秒）
  - 丢包率检测（失败次数统计）
  
- ✅ **弱网检测**：自动识别弱网环境
  - RTT > 200ms 判定为弱网
  - 丢包率 > 10% 判定为弱网
  - 触发硬约束规则：弱网 → 强制 EDGE_ONLY

- ✅ **网络感知评分**：延迟评分考虑 RTT
  - CLOUD_DIRECT：延迟 = 基础延迟 + 2×RTT
  - SPECULATIVE：延迟 = 基础延迟 + 1×RTT
  - 动态调整策略选择

#### 新增文件
- `edge/monitor.py` - 增强版状态监控（182行）
- `tests/test_stage2_network.py` - 网络感知测试（300行）

#### 修改文件
- `common/types.py`
  - 新增 `NetworkStats` 数据类
  - `DecisionContext` 新增 `network_state` 字段

- `edge/decision_engine.py`
  - `HardConstraintChecker` 新增规则4：弱网检测
  - `MultiObjectiveScorer._score_latency()` 考虑网络 RTT

- `edge/f1_decision.py`
  - 支持异步决策接口 `decide_async()`
  - `_build_context_async()` 集成网络探测
  - 新增 `enable_network_probe` 开关

- `config/config.yaml`
  - 新增 `enable_network_probe` 配置
  - 新增 `network_probe_interval_ms` 配置
  - 新增 `weak_network_rtt` 阈值

#### 测试结果
```
✅ 5/5 测试通过
- 网络探测功能正常
- 弱网检测准确
- RTT 影响延迟评分
- 硬约束弱网规则有效
```

---

### v1.0.0 - 2026-02-02 ✅ 阶段1完成

**首次发布：基础决策框架**

#### 核心功能
- ✅ **状态监控**：CPU、内存实时监控
- ✅ **硬约束决策**：5条快速剪枝规则
  1. 系统过载保护（CPU>90% 或 内存<500MB）
  2. 极端延迟要求（SLO<50ms）
  3. 隐私要求（privacy_level≥2）
  4. 紧急任务（priority≥3 且质量要求低）
  
- ✅ **多目标评分**：延迟+成本+质量综合评分
  - 延迟得分：基于 SLO 归一化
  - 成本得分：边端最低、云端最高
  - 质量得分：云端最高、边端较低

- ✅ **四种策略**：
  1. EDGE_ONLY - 纯端侧推理
  2. CLOUD_DIRECT - 纯云端推理
  3. SPECULATIVE_STANDARD - 标准协同推理
  4. ADAPTIVE_CONFIDENCE - 自适应协同推理

#### 新增文件
- `edge/state_monitor.py` - 状态监控器（120行）
- `edge/decision_engine.py` - 决策引擎（200行）
- `edge/execution_planner.py` - 执行计划生成器（80行）
- `edge/f1_decision.py` - F1 主入口（150行）
- `tests/test_f1_core.py` - 核心功能测试（250行）

#### 修改文件
- `common/types.py`
  - 新增 9 个数据类型
  - ExecutionStrategy 枚举（4种策略）
  - TaskRequirements（SLO定义）
  - SystemStats（硬件状态）
  - DecisionContext（决策上下文）
  - ExecutionPlan（执行计划）
  - HardDecision（硬约束决策）
  - ScoredStrategy（策略评分）

- `edge/edge_server.py`
  - 初始化 F1_DecisionModule
  - `process_inference()` 调用 F1 决策
  - 根据策略执行不同分支

- `config/config.yaml`
  - 新增 `f1` 配置节
  - 配置硬约束阈值
  - 配置评分权重
  - 配置延迟估算值

#### 测试结果
```
✅ 6/6 测试通过
- 硬约束触发正确
- 多目标评分合理
- 降级策略安全
```

---

## 统计数据

### 代码量
- **新增代码**：~2000+ 行
- **新增文件**：11 个
- **修改文件**：3 个

### 测试覆盖
- **总测试用例**：14 个
- **通过率**：100%
- **测试文件**：5 个

### 配置项
- **硬约束**：4 个阈值
- **评分权重**：3 个维度
- **自适应参数**：7 个配置项
- **功能开关**：3 个

---

## 升级指南

### 从 v1.0 升级到 v2.0
1. 更新 `config.yaml`，添加网络探测配置：
   ```yaml
   enable_network_probe: true
   weak_network_rtt: 200.0
   ```
2. 安装依赖：`pip install aiohttp`
3. 重启服务

### 从 v2.0 升级到 v3.0
1. 更新 `config.yaml`，添加自适应配置：
   ```yaml
   enable_adaptive: true
   history_tracker:
     max_history_size: 100
   adaptive_threshold:
     target_acceptance_min: 0.75
     target_acceptance_max: 0.85
   ```
2. EdgeServer 会自动记录执行结果，无需修改代码
3. 重启服务

---

## 已知问题

### v3.0.0
- ⚠️ 完整集成测试需要 psutil、aiohttp 等依赖
- ⚠️ 网络探测在无云端连接时会失败（已有降级处理）

### 解决方案
- 核心功能测试使用 `test_stage3_simple.py`（无外部依赖）
- 网络探测失败不影响决策流程

---

## 路线图

### v3.1.0（计划中）
- [ ] 多模型支持
- [ ] 更精细的成本模型
- [ ] 自适应 Draft 长度

### v4.0.0（未来）
- [ ] 强化学习优化权重
- [ ] 个性化决策
- [ ] 负载预测

---

**更新日志** | 最后更新：2026-02-04
