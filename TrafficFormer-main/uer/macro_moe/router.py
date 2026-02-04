import torch
import torch.nn as nn
import torch.nn.functional as F


class ProtocolRouter(nn.Module):
    def __init__(self, num_experts, hidden_size):
        super(ProtocolRouter, self).__init__()
        self.num_experts = num_experts

        # 1. 门控层 (Gating Network)
        # 一个简单的线性层，将输入特征映射到专家数量的维度 (hidden_size -> num_experts)
        self.gate = nn.Linear(hidden_size, num_experts)

        # 统计 Buffer (不参与梯度更新，仅用于记录每个专家被使用的次数)
        self.register_buffer("usage_counter", torch.zeros(num_experts, dtype=torch.long))

    def forward(self, proto_ids=None, inputs_embeds=None, top_k=1):
        """
        输入: inputs_embeds [batch_size, seq_len, hidden_size]
        输出: expert_indices (每个样本选中的专家ID), load_balance_loss (辅助损失)
        """
        # 选取前 32 个 token (涵盖了论文提到的关键 header 范围) 进行平均池化
        # 这样 Router 就能“看到” header 里的内容（如协议特征、长度特征等）
        router_input = torch.mean(inputs_embeds[:, :32, :], dim=1)

        # ================= Formula 5: Routing Logic =================
        # 1. 计算 Logits (未归一化的分数)
        router_logits = self.gate(router_input)

        # 2. 加上微量噪声 (Standard Trick)
        # 在训练时加入标准正态分布噪声，有助于打破由于权重初始化导致的对称性，促进负载均衡
        if self.training:
            router_logits = router_logits + torch.randn_like(router_logits) * 0.05

        # 3. 计算路由概率 (Softmax) -> [batch_size, num_experts]
        router_probs = F.softmax(router_logits, dim=-1)

        # 4. 选择 Top-k 专家
        # values: 概率值, indices: 专家的索引ID
        _, expert_indices = torch.topk(router_probs, k=top_k, dim=-1)

        # 如果 k=1，去掉最后一维，变成 [batch_size]
        if top_k == 1:
            expert_indices = expert_indices.squeeze(-1)

        # ================= Formula 9: Load Balancing Loss (辅助损失) =================
        # 这是一个惩罚项，强制要求：
        # 1. 路由器预测的概率分布 (Prob) 尽可能均匀
        # 2. 实际分配给专家的样本比例 (Fraction) 尽可能均匀

        # a. Prob_e: 每个专家被选中的平均“预测概率”
        prob_per_expert = router_probs.mean(dim=0)

        # b. Load_e: 每个专家实际接收到的样本比例 (使用 One-hot 统计)
        if top_k == 1:
            mask = F.one_hot(expert_indices, num_classes=self.num_experts).float()
        else:
            mask = F.one_hot(expert_indices, num_classes=self.num_experts).float().sum(dim=1)

        # 计算这一批次中，每个专家分到了百分之多少的数据
        fraction_per_expert = mask.mean(dim=0)

        # c. 计算 Loss
        # 公式: num_experts * sum(Prob_i * Fraction_i)
        # 当数据完全均匀分布时，该 Loss 最小
        load_balance_loss = (self.num_experts * (prob_per_expert * fraction_per_expert).sum())

        return expert_indices, load_balance_loss