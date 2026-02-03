import torch
import torch.nn as nn
import torch.nn.functional as F


class ProtocolRouter(nn.Module):
    def __init__(self, num_experts, hidden_size):
        super(ProtocolRouter, self).__init__()
        self.num_experts = num_experts

        # 1. 门控层
        self.gate = nn.Linear(hidden_size, num_experts)

        # 【新增优化】将参数初始化为很小的值，保证初始概率接近均匀 (1/N)
        nn.init.normal_(self.gate.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.gate.bias, 0.0)

        # 2. 统计 Buffer
        self.register_buffer("usage_counter", torch.zeros(num_experts, dtype=torch.long))

    def reset_usage(self):
        self.usage_counter.zero_()

    def forward(self, proto_ids=None, inputs_embeds=None):
        # inputs_embeds: [batch_size, seq_len, hidden_size]
        router_input = inputs_embeds[:, 0, :]

        # 1. 计算 Logits
        router_logits = self.gate(router_input)

        # 【新增优化】训练阶段加入噪声 (Noisy Gating)
        # 这能防止初始化时的“赢者通吃”，强迫探索其他专家
        if self.training:
            # 产生标准正态分布噪声，幅度设为 1.0 (可调，通常 1/num_experts 左右)
            noise = torch.randn_like(router_logits) * (1.0 / self.num_experts)
            # 加上噪声再做 softmax
            # 使用 Softplus 保证噪声幅度平滑 (可选，这里简化直接加)
            router_logits = router_logits + noise

        # 2. 计算概率
        router_probs = F.softmax(router_logits, dim=-1)

        # 3. 选择专家
        _, expert_indices = torch.max(router_probs, dim=-1)

        # 4. 【修复点】更新统计 (务必确保这一段在代码里！)
        with torch.no_grad():
            # 确保 expert_indices 和 usage_counter 在同一设备
            for i in range(self.num_experts):
                self.usage_counter[i] += (expert_indices == i).sum()

        # 5. 计算 Loss (MSE 迫使平均概率趋向 1/N)
        mean_probs = router_probs.mean(dim=0)
        target_probs = torch.full_like(mean_probs, 1.0 / self.num_experts)
        load_balance_loss = F.mse_loss(mean_probs, target_probs) * self.num_experts

        return expert_indices, load_balance_loss