import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 加载数据
with open("routing_analysis.json", "r") as f:
    data = json.load(f)

true_labels = data["true_labels"]
expert_indices = data["expert_indices"]

num_classes = 6  # 你的 VPN 类别数
num_experts = 4  # 你的专家数

# 建立统计矩阵
routing_matrix = np.zeros((num_classes, num_experts))

# 修改点：遍历每一条样本及其激活的专家（可能是多个）
for label, experts in zip(true_labels, expert_indices):
    # 为了兼容旧的 JSON 数据（如果里面还是单个整数）
    if isinstance(experts, int):
        experts = [experts]

    # 统计该样本分配到的每一个专家
    for expert in experts:
        routing_matrix[label, expert] += 1

# 归一化（计算每个类别分给各专家的比例）
# 防止除以0
row_sums = routing_matrix.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1
routing_matrix_normalized = routing_matrix / row_sums

# 画图
plt.figure(figsize=(10, 8))
# 你的 6 个类别名字
class_names = ["vpn-audio", "vpn-chat", "vpn-file", "vpn-mail", "vpn-streaming", "vpn-voip"]

sns.heatmap(routing_matrix_normalized, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=[f"Expert {i}" for i in range(num_experts)],
            yticklabels=class_names)

plt.title("Router Dispatch Preference (Macro-MoE Top-K)")
plt.xlabel("Experts")
plt.ylabel("Traffic Classes")
plt.tight_layout()
plt.savefig("routing_heatmap_v9.png", dpi=300)
print("Heatmap saved to routing_heatmap_v9.png!")