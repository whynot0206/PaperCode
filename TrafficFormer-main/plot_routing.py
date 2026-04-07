import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


DEFAULT_CLASS_NAMES = [
    "vpn-audio",
    "vpn-chat",
    "vpn-file",
    "vpn-mail",
    "vpn-streaming",
    "vpn-voip",
]

'''
DEFAULT_CLASS_NAMES = [
    "bittorrent",
    "email",
    "facebook",
    "hangouts",
    "netflix",
    "skype",
    "spotify",
    "vimeo",
    "voipbuster",
    "youtube",
    "file"
]
'''

# =========================
# 你每次实验只改这里即可
# =========================
SAVE_TAG = "v1_4e_adapter+backbone_top2_ISCX-VPN-Service-share-adapter-test1-seed7"

# 总目录：expert_top/
BASE_SAVE_DIR = Path("expert_top")

# 本次实验子目录：expert_top/SAVE_TAG/
RUN_SAVE_DIR = BASE_SAVE_DIR / SAVE_TAG
RUN_SAVE_DIR.mkdir(parents=True, exist_ok=True)


def normalize_rows(matrix):
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return matrix / row_sums


def draw_heatmap(matrix, class_names, num_experts, title, output_path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=[f"Expert {i}" for i in range(num_experts)],
        yticklabels=class_names,
    )
    plt.title(title)
    plt.xlabel("Experts")
    plt.ylabel("Traffic Classes")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Heatmap saved to {output_path}!")


def draw_rank_grid(rank_matrices, class_names, num_experts, output_path):
    num_ranks = len(rank_matrices)
    cols = min(2, num_ranks)
    rows = math.ceil(num_ranks / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 6 * rows))
    axes = np.array(axes).reshape(-1)

    for rank_id, matrix in enumerate(rank_matrices):
        sns.heatmap(
            matrix,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=[f"Expert {i}" for i in range(num_experts)],
            yticklabels=class_names,
            ax=axes[rank_id],
            cbar=(rank_id == 0),
        )
        axes[rank_id].set_title(f"Router Dispatch Preference (Rank-{rank_id + 1})")
        axes[rank_id].set_xlabel("Experts")
        axes[rank_id].set_ylabel("Traffic Classes")

    for idx in range(num_ranks, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Heatmap saved to {output_path}!")


with open("routing_analysis.json", "r") as f:
    data = json.load(f)

true_labels = data["true_labels"]
expert_indices = data.get("rank_expert_indices", data["expert_indices"])

normalized_expert_indices = []
for expert in expert_indices:
    if isinstance(expert, list):
        normalized_expert_indices.append(expert)
    else:
        normalized_expert_indices.append([expert])

num_classes = max(true_labels) + 1 if true_labels else len(DEFAULT_CLASS_NAMES)

num_experts = 0
for experts in normalized_expert_indices:
    if experts:
        num_experts = max(num_experts, max(experts) + 1)
num_experts = max(num_experts, 1)

top_k = data.get(
    "top_k",
    max((len(experts) for experts in normalized_expert_indices), default=1)
)

if num_classes <= len(DEFAULT_CLASS_NAMES):
    class_names = DEFAULT_CLASS_NAMES[:num_classes]
else:
    class_names = [f"class-{i}" for i in range(num_classes)]

overall_routing_matrix = np.zeros((num_classes, num_experts), dtype=np.float64)
rank_routing_matrices = [
    np.zeros((num_classes, num_experts), dtype=np.float64)
    for _ in range(top_k)
]

for label, experts in zip(true_labels, normalized_expert_indices):
    if len(experts) == 0:
        continue

    overall_weight = 1.0 / len(experts)
    for rank_id, expert_id in enumerate(experts):
        overall_routing_matrix[label, expert_id] += overall_weight
        if rank_id < len(rank_routing_matrices):
            rank_routing_matrices[rank_id][label, expert_id] += 1.0

overall_routing_matrix = normalize_rows(overall_routing_matrix)
rank_routing_matrices = [normalize_rows(matrix) for matrix in rank_routing_matrices]

# =========================
# 保存路径统一管理
# =========================
overall_output = RUN_SAVE_DIR / f"{SAVE_TAG}_overall.png"
grid_output = RUN_SAVE_DIR / f"{SAVE_TAG}_ranks_grid.png"

draw_heatmap(
    overall_routing_matrix,
    class_names,
    num_experts,
    "Router Dispatch Preference (Macro-MoE Top-K)",
    overall_output,
)

for rank_id, matrix in enumerate(rank_routing_matrices):
    rank_output = RUN_SAVE_DIR / f"{SAVE_TAG}_rank{rank_id + 1}.png"
    draw_heatmap(
        matrix,
        class_names,
        num_experts,
        f"Router Dispatch Preference (Rank-{rank_id + 1})",
        rank_output,
    )

if len(rank_routing_matrices) > 1:
    draw_rank_grid(
        rank_routing_matrices,
        class_names,
        num_experts,
        grid_output,
    )