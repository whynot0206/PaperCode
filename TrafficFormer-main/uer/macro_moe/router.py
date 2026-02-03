import torch
import torch.nn as nn
import torch.nn.functional as F


class ProtocolRouter(nn.Module):
    def __init__(self, num_experts, hidden_size):
        super(ProtocolRouter, self).__init__()
        self.num_experts = num_experts
        # [修改] 添加可学习的门控层
        self.gate = nn.Linear(hidden_size, num_experts)

        # 2. 【修复点】补回 usage_counter buffer，否则 trainer 会报错
        self.register_buffer(
            "usage_counter",
            torch.zeros(num_experts, dtype=torch.long)
        )

    def reset_usage(self):
        """【修复点】Trainer 需要调用此方法重置统计"""
        self.usage_counter.zero_()

    def forward(self, proto_ids=None, inputs_embeds=None):
        """
        Args:
            proto_ids: (Optional) 外部标签
            inputs_embeds: [batch_size, seq_len, hidden_size] 输入特征
        Returns:
            expert_indices: [batch_size] 路由结果
            load_balance_loss: scalar tensor 负载均衡损失
        """
        # 1. 提取路由特征 (取 [CLS] 即第一个token)
        # inputs_embeds: [batch, seq, dim] -> [batch, dim]
        router_input = inputs_embeds[:, 0, :]

        # 2. 计算门控 logits
        router_logits = self.gate(router_input)  # [batch, num_experts]

        # 3. 计算概率
        router_probs = F.softmax(router_logits, dim=-1)

        # 4. 路由决策 (Top-1)
        # 优先使用 proto_ids (如果有且需要强制路由), 否则使用学习到的路由
        if proto_ids is not None:
            if proto_ids.dim() > 1:
                routing_ids = proto_ids[:, 0]
            else:
                routing_ids = proto_ids
            expert_indices = routing_ids % self.num_experts
        else:
            _, expert_indices = torch.max(router_probs, dim=-1)

        # 5. 计算负载均衡 Loss (Load Balance Loss)
        # 目标：Batch 内每个专家被选中的概率均值应该接近 1/N
        mean_probs = router_probs.mean(dim=0)  # [num_experts]
        target_probs = torch.full_like(mean_probs, 1.0 / self.num_experts)

        # 使用 MSE Loss 迫使分布均匀
        load_balance_loss = F.mse_loss(mean_probs, target_probs) * self.num_experts

        return expert_indices, load_balance_loss