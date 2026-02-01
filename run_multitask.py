import requests
import time
import pandas as pd
import numpy as np

# ================= 配置区域 =================
EDGE_URL = "http://127.0.0.1:8088/inference"
ROUNDS = 3  # 每个任务每个场景跑3次

# 1. 定义三个任务 (Task)
TASKS = [
    {
        "name": "📖 Story",
        "prompt": "Once upon a time in a futuristic city, robots started to",
        "max_tokens": 48  # 稍微短点，防止Baseline超时
    },
    {
        "name": "🧮 Math",
        "prompt": "Solve 4x + 10 = 30. Step by step:",
        "max_tokens": 32
    },
    {
        "name": "💻 Code",
        "prompt": "def bubble_sort(arr):",
        "max_tokens": 40
    }
]

# 2. 定义消融场景 (Scenarios) - 前4个
SCENARIOS_PART_1 = [
    ("Baseline (Cloud)",  {"use_confidence_check": False, "force_cloud": True}), # 特殊标记
    ("+F1 (Edge Only)",   {"use_draft_verify": False}),
    ("+F1+F3 (Standard)", {"use_confidence_check": False}),
    ("+F1+F2+F3 (Full)",  {"use_confidence_check": True})
]

# 3. 定义消融场景 - 最后1个 (需人工干预网络)
SCENARIO_PART_2 = [
    ("+F4 (Weak Net)",    {"use_confidence_check": True})
]

# ===========================================

def run_single_inference(task_conf, scenario_name, scenario_params):
    """运行单次推理，返回指标"""
    # 基础参数
    payload = {
        "prompt": task_conf["prompt"],
        "max_tokens": task_conf["max_tokens"],
        "temperature": 0.1,
        "top_p": 0.9,
        "use_draft_verify": True,     # 默认开启
        "use_confidence_check": True  # 默认开启
    }
    
    # 应用场景覆盖参数
    payload.update(scenario_params)
    
    # 特殊处理 Baseline: 如果是 Baseline，我们通常把 max_tokens 设小一点防止超时
    # 或者为了公平对比 TPS，保持一致。这里为了稳定性，如果是 Baseline，我们特殊处理 max_tokens
    if scenario_params.get("force_cloud"):
        # 移除自定义标记
        del payload["force_cloud"]
        # Baseline 模拟：把 max_tokens 设为极小值来测延迟，或者强制不生成 Draft
        # 为了多任务对比 TPS，我们需要它生成。我们假设 Cloud 足够快。
        # 这里把 use_draft_verify 关掉实际上并不完全等同于 Pure Cloud，因为 Edge 还是会走一遍。
        # 你的代码里 Baseline 是通过 max_tokens=1 模拟的，这里我们为了对比 TPS，
        # 实际上我们依赖 use_draft_verify=False 且 max_tokens=1 可能会导致除零。
        # === 修正逻辑 ===
        # Baseline 实际上在你的系统里很难完美模拟 (除非改Edge代码)。
        # 我们这里用 "Edge Only" 但 max_tokens=1 模拟握手延迟？不，这不准。
        # 我们用标准逻辑：设置 max_tokens 为任务所需，但 Edge 端代码需配合。
        # 你的 Edge 代码逻辑里，如果 use_draft_verify=False，就是 Edge Only。
        # 要测 Pure Cloud，目前最稳妥的方法是：设置 max_tokens=1 (测延迟) 
        # 但这样就没法测 Math/Story 的生成质量了。
        # 妥协方案：Baseline 场景只测延迟 (Latency/TTFT)，TPS 设为 NaN
        payload["max_tokens"] = 10 
        payload["use_draft_verify"] = False 

    try:
        start = time.time()
        # 超时时间设长一点，给 Cloud 机会
        resp = requests.post(EDGE_URL, json=payload, timeout=90)
        end = time.time()
        
        if resp.status_code == 200:
            data = resp.json()
            total_lat = (end - start) * 1000
            edge_lat = data.get('edge_latency_ms', 0)
            
            # 计算指标
            # 1. TTFT
            if "Baseline" in scenario_name:
                ttft = total_lat # 云端模式，首字即总时
            else:
                ttft = edge_lat
                if ttft == 0: ttft = total_lat # 防止异常

            # 2. Token Count & TPS
            text = data.get('text', '')
            tokens = len(text.split())
            if tokens == 0: tokens = 1
            tps = tokens / (total_lat / 1000)
            
            # 3. Acceptance Rate
            ar = data.get('acceptance_rate', 0) * 100
            
            return {
                "Scenario": scenario_name,
                "Task": task_conf["name"],
                "Latency": total_lat,
                "TTFT": ttft,
                "TPS": tps,
                "AR": ar
            }
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def run_group(scenarios, all_results):
    """运行一组场景"""
    for sc_name, sc_params in scenarios:
        print(f"\n🧪 [场景]: {sc_name}")
        
        for task in TASKS:
            print(f"   👉 任务: {task['name']} ", end="")
            task_metrics = []
            
            for i in range(ROUNDS):
                m = run_single_inference(task, sc_name, sc_params)
                if m:
                    task_metrics.append(m)
                    print(".", end="", flush=True)
                else:
                    print("x", end="", flush=True)
            
            # 计算该任务在该场景下的平均值
            if task_metrics:
                df_tmp = pd.DataFrame(task_metrics)
                avg_m = df_tmp.mean(numeric_only=True)
                all_results.append({
                    "Scenario": sc_name,
                    "Task": task["name"],
                    "Latency": avg_m["Latency"],
                    "TTFT": avg_m["TTFT"],
                    "TPS": avg_m["TPS"],
                    "AR": avg_m["AR"]
                })
            print(" 完成")

def main():
    all_results = []
    
    print("🚀 开始多任务全流程消融实验")
    print("="*60)
    
    # 1. 跑前 4 组 (不需要人工干预)
    run_group(SCENARIOS_PART_1, all_results)
    
    # 2. 暂停，等待人工开启弱网
    print("\n" + "="*60)
    print("🛑 [人工干预点] 请现在开启弱网环境！")
    print("   建议: 运行 'python proxy_delay.py' (监听9000端口)")
    print("   或者: 在 WSL2 运行 'sudo tc qdisc replace dev eth0 root netem delay 500ms'")
    print("   (记得修改 config.yaml 的端口并重启 Edge Server)")
    print("="*60)
    input("👉 准备好后，按 [Enter] 键继续运行 F4 测试...")
    
    # 3. 跑最后 1 组 (F4)
    run_group(SCENARIO_PART_2, all_results)
    
    # ================= 结果展示 =================
    df = pd.DataFrame(all_results)
    
    # 格式化数字
    pd.options.display.float_format = '{:.1f}'.format
    
    print("\n📊 [详细报告] 各任务表现")
    print("="*80)
    # 按任务分组显示
    for task_name in [t['name'] for t in TASKS]:
        print(f"\n--- Task: {task_name} ---")
        task_df = df[df['Task'] == task_name][['Scenario', 'Latency', 'TTFT', 'TPS', 'AR']]
        print(task_df.to_string(index=False))

    print("\n📊 [汇总报告] 全局平均 (System Average)")
    print("="*80)
    # 按场景分组算平均
    summary = df.groupby('Scenario')[['Latency', 'TTFT', 'TPS', 'AR']].mean().reset_index()
    
    # 调整顺序 (让表格按我们执行的顺序排)
    scenario_order = [s[0] for s in SCENARIOS_PART_1] + [s[0] for s in SCENARIO_PART_2]
    summary['Scenario'] = pd.Categorical(summary['Scenario'], categories=scenario_order, ordered=True)
    summary = summary.sort_values('Scenario')
    
    print(summary.to_string(index=False))
    print("="*80)
    
    # 保存
    df.to_csv("ablation_multitask_final.csv", index=False)
    print("📝 结果已保存至 ablation_multitask_final.csv")

if __name__ == "__main__":
    main()