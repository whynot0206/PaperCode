import torch
import torch.nn as nn
import torch.nn.functional as F


class ProtocolRouter(nn.Module):
    def __init__(self, num_experts, hidden_size, noise_std=0.01,
                 balance_weight=0.2, entropy_weight=1.0, target_entropy=0.6,
                 rank1_weight=0.0, rank2_weight=0.0, rank_target_entropy=0.45):
        super(ProtocolRouter, self).__init__()
        self.num_experts = num_experts
        self.noise_std = noise_std
        self.balance_weight = balance_weight
        self.entropy_weight = entropy_weight
        self.target_entropy = target_entropy
        self.rank1_weight = rank1_weight
        self.rank2_weight = rank2_weight
        self.rank_target_entropy = rank_target_entropy
        # 1. 门控层 (Gating Network) - 对应 Formula 5
        # 用于将输入特征映射到专家权重
        self.gate = nn.Linear(hidden_size, num_experts)

        # 统计 Buffer (用于日志打印)
        self.register_buffer("usage_counter", torch.zeros(num_experts, dtype=torch.long))
        # 按 Top-k 的“第几路”分别统计：
        # rank_usage_counter[r, e] = 第 r+1 路选择 expert e 的次数
        # 形状固定为 [num_experts, num_experts]，实际只使用前 top_k 行。
        self.register_buffer("rank_usage_counter", torch.zeros(num_experts, num_experts, dtype=torch.long))

    def reset_usage(self):
        self.usage_counter.zero_()
        self.rank_usage_counter.zero_()

    def _normalized_entropy(self, prob):
        prob = prob / (prob.sum() + 1e-9)
        normalizer = torch.log(torch.tensor(float(self.num_experts), device=prob.device))
        return -(prob * torch.log(prob + 1e-9)).sum() / normalizer

    def _rank_regularizer(self, rank_indices, router_probs):
        rank_mask = F.one_hot(rank_indices, num_classes=self.num_experts).float()
        rank_fraction = rank_mask.mean(dim=0)

        # Use the selected rank mask to keep gradients on the probabilities of
        # experts that actually win this routing slot.
        rank_prob = (rank_mask * router_probs).mean(dim=0)
        rank_prob = rank_prob / (rank_prob.sum() + 1e-9)

        uniform_balance = self.num_experts * (rank_prob * rank_fraction).sum()
        entropy_target_loss = (self._normalized_entropy(rank_prob) - self.rank_target_entropy) ** 2
        return self.balance_weight * uniform_balance + self.entropy_weight * entropy_target_loss

    def forward(self, proto_ids=None, inputs_embeds=None, top_k=1):
        """
        实现 Traffic-MoE 的 Formula 5 (Routing) 和 Formula 9 (Aux Loss)
        """
        # [batch_size, hidden_size]
        # 使用 [CLS] 或 mean pooling 作为路由特征
        # router_input = torch.mean(inputs_embeds[:, :32, :], dim=1)
        # router_input = torch.mean(inputs_embeds, dim=1)
        # router_input = inputs_embeds[:, 0, :]
        # inputs_embeds 的形状通常是 [batch_size, seq_length, hidden_size]

        # 1. 动态生成 Mask（假设你的词表中 [PAD] 的 ID 对应的 embedding 通常是零向量，
        # 或者你可以根据 inputs_embeds 在 hidden_size 维度上的绝对值和来判断是否为 Padding）
        # 这里提供一种通用的特征级 Mask 估算方法（如果传入了真正的 src mask 最好）：
        # 计算每个 token 向量的 L2 范数，如果不为 0，则认为是有效 token
        token_norms = torch.norm(inputs_embeds, dim=-1)
        # mask 的形状为 [batch_size, seq_length]，有效位置为 1，Padding 位置为 0
        mask = (token_norms > 1e-5).float()

        # 2. 将 Mask 扩展到与 inputs_embeds 相同的维度 [batch_size, seq_length, hidden_size]
        mask_expanded = mask.unsqueeze(-1)

        # 3. 把 Padding 位置的特征强行清零
        masked_embeds = inputs_embeds * mask_expanded

        # 4. 对有效特征求和 [batch_size, hidden_size]
        sum_embeds = torch.sum(masked_embeds, dim=1)

        # 5. 计算每个样本真实的有效长度 [batch_size, 1]
        # 使用 clamp(min=1e-9) 防止除以 0 的崩溃
        valid_lengths = torch.sum(mask, dim=1, keepdim=True).clamp(min=1e-9)

        # 6. 计算真正的、纯净的均值特征！
        router_input = sum_embeds / valid_lengths

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
        norm_entropy = self._normalized_entropy(prob_per_expert)

        entropy_target_loss = (norm_entropy - self.target_entropy) ** 2

        load_balance_loss = self.balance_weight * uniform_balance + self.entropy_weight * entropy_target_loss

        if top_k == 1:
            if self.rank1_weight > 0:
                load_balance_loss = load_balance_loss + self.rank1_weight * self._rank_regularizer(
                    expert_indices, router_probs
                )
        else:
            if self.rank1_weight > 0:
                load_balance_loss = load_balance_loss + self.rank1_weight * self._rank_regularizer(
                    expert_indices[:, 0], router_probs
                )
            if self.rank2_weight > 0:
                load_balance_loss = load_balance_loss + self.rank2_weight * self._rank_regularizer(
                    expert_indices[:, 1], router_probs
                )

        # ================= 统计更新 =================
        with torch.no_grad():
            # 统计这一轮每个专家实际吃了多少“被选择”次数：
            # - top_k=1: 每个样本贡献 1 次
            # - top_k>1: 每个样本贡献 k 次（分别对应 rank-1...rank-k）
            if top_k == 1:
                for i in range(self.num_experts):
                    self.usage_counter[i] += (expert_indices == i).sum()
                # 第 1 路统计
                for i in range(self.num_experts):
                    self.rank_usage_counter[0, i] += (expert_indices == i).sum()
            else:
                # all-selected 统计：按“k 路总选择次数”累计
                flat_indices = expert_indices.reshape(-1)
                for i in range(self.num_experts):
                    self.usage_counter[i] += (flat_indices == i).sum()

                # 分路统计：第 r 路选择了哪个专家
                for r in range(top_k):
                    rank_indices = expert_indices[:, r]
                    for i in range(self.num_experts):
                        self.rank_usage_counter[r, i] += (rank_indices == i).sum()

        return expert_indices, load_balance_loss, router_probs_values
