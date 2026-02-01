import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取数据
df = pd.read_csv("framework_logic_results.csv")

# 设置风格
sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 6))

# === 绘制图表：网络延迟对策略的影响 ===
# 筛选出只有网络变化的场景 (Scenario 1-4)
net_df = df[df['Scenario'].str.contains('网络|断网')]

# 为了画图，我们将策略映射为数字
strategy_map = {
    'speculative_standard': 0, # 云端重度依赖
    'adaptive_confidence': 1,  # 混合
    'edge_only': 2             # 本地独立
}
net_df['Strategy_Num'] = net_df['Strategy'].map(strategy_map)

plt.plot(net_df['RTT(ms)'], net_df['Strategy_Num'], marker='o', linestyle='--', linewidth=2, color='b')

# 美化图表
plt.yticks([0, 1, 2], ['Standard\n(Cloud Heavy)', 'Adaptive\n(Hybrid)', 'Edge Only\n(Offline)'])
plt.xscale('log') # RTT 是指数增长的，用对数坐标更好看
plt.xlabel('Network RTT (ms) [Log Scale]', fontsize=12)
plt.ylabel('Chosen Strategy', fontsize=12)
plt.title('Framework Decision Boundary Analysis', fontsize=14)
plt.grid(True, which="both", ls="-", alpha=0.2)

# 标注区域
plt.axvspan(10, 100, color='green', alpha=0.1, label='Strong Net Zone')
plt.axvspan(100, 1000, color='yellow', alpha=0.1, label='Weak Net Zone')
plt.axvspan(1000, 3000, color='red', alpha=0.1, label='Broken Net Zone')
plt.legend()

plt.tight_layout()
plt.savefig("decision_boundary.png")
print("🖼️ 图表已保存为 decision_boundary.png")