import torch
import torch.nn as nn
import torch.nn.functional as F


class ProtocolRouter(nn.Module):
    def __init__(self, num_experts, hidden_size, noise_std=0.01,
                 balance_weight=0.2, entropy_weight=1.0, target_entropy=0.6):
        super(ProtocolRouter, self).__init__()
        self.num_experts = num_experts
        self.noise_std = noise_std
        self.balance_weight = balance_weight
        self.entropy_weight = entropy_weight
        self.target_entropy = target_entropy
        # 1. 门控层 (Gating Network) - 对应 Formula 5
        # 用于将输入特征映射到专家权重
        self.gate = nn.Linear(hidden_size, num_experts)

        # 统计 Buffer (用于日志打印)
        self.register_buffer("usage_counter", torch.zeros(num_experts, dtype=torch.long))

    def reset_usage(self):
        self.usage_counter.zero_()

    def forward(self, proto_ids=None, inputs_embeds=None, top_k=1):
        """
        实现 Traffic-MoE 的 Formula 5 (Routing) 和 Formula 9 (Aux Loss)
        """
        # [batch_size, hidden_size]
        # 使用 [CLS] 或 mean pooling 作为路由特征
        # router_input = torch.mean(inputs_embeds[:, :32, :], dim=1)
        router_input = torch.mean(inputs_embeds, dim=1)

        # ================= Formula 5: Routing Logic =================
        # 1. 计算 Logits
        router_logits = self.gate(router_input)

        # 2. 加上微量噪声 (Standard Trick, 可选但推荐)
        # 帮助打破训练初期的平局，防止死锁。幅度不用太大。
        # if self.training:
        #    router_logits = router_logits + torch.randn_like(router_logits) * 0.05
        if self.training and self.noise_std > 0:
            router_logits = router_logits + torch.randn_like(router_logits) * self.noise_std

        # 3. 计算路由概率 (Prob_e)
        router_probs = F.softmax(router_logits, dim=-1)  # [batch_size, num_experts]

        # 4. 选择 Top-k 专家
        # values: [batch_size, k], indices: [batch_size, k]
        # 如果是预训练，通常 k=1；论文中 Inference 用了 k=2
        router_probs_values, expert_indices = torch.topk(router_probs, k=top_k, dim=-1)

        if top_k == 1:
            expert_indices = expert_indices.squeeze(-1)
            router_probs_values = router_probs_values.squeeze(-1)  # [batch_size]

        # ================= Formula 9: Load Balancing Loss =================
        # L_aux = N * sum(Load_e * Prob_e)

        # a. Prob_e: 每个专家被选中的平均概率 (Continuous)
        # [num_experts]
        prob_per_expert = router_probs.mean(dim=0)

        # b. Load_e: 每个专家实际接收到的样本比例 (Discrete)
        # 我们使用 one_hot 技巧来计算实际的 Load
        # 结果维度: [batch_size, num_experts]
        if top_k == 1:
            mask = F.one_hot(expert_indices, num_classes=self.num_experts).float()
        else:
            # 如果是 Top-k，只要在 indices 里出现了就算选中
            mask = F.one_hot(expert_indices, num_classes=self.num_experts).float().sum(dim=1)

        # [num_experts]
        fraction_per_expert = mask.mean(dim=0)

        # c. 计算 Loss
        # 这就是 Formula 9 的核心：点积求和 * N
        # 这种 Loss 鼓励 prob 和 fraction 向量都接近均匀分布 [1/N, ..., 1/N]
        # load_balance_loss = (self.num_experts * (prob_per_expert * fraction_per_expert).sum())

        # 说明：仅用 uniform load-balance 会过强地推向“绝对均匀”。
        # 这里改为“均衡项 + 熵目标项”：
        # - 均衡项：维持最基本的专家利用率，权重较低；
        # - 熵目标项：鼓励达到一个“合适而非最大均匀”的路由熵（可配置）。
        # 说明：仅用 uniform load-balance 会过强地推向“绝对均匀”。
        uniform_balance = (self.num_experts * (prob_per_expert * fraction_per_expert).sum())

        # 【修复点】：将 fraction_per_expert 替换为 prob_per_expert，恢复梯度回传！
        norm_entropy = -(
                prob_per_expert * torch.log(prob_per_expert + 1e-9)
        ).sum() / torch.log(torch.tensor(float(self.num_experts), device=router_probs.device))

        entropy_target_loss = (norm_entropy - self.target_entropy) ** 2

        load_balance_loss = self.balance_weight * uniform_balance + self.entropy_weight * entropy_target_loss

        # ================= 统计更新 =================
        with torch.no_grad():
            # 统计这一轮每个专家实际吃了多少数据
            if top_k == 1:
                for i in range(self.num_experts):
                    self.usage_counter[i] += (expert_indices == i).sum()
            else:
                for i in range(self.num_experts):
                    self.usage_counter[i] += (expert_indices == i).any(dim=-1).sum()

        return expert_indices, load_balance_loss, router_probs_values