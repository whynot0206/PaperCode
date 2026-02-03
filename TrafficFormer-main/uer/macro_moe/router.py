import torch
import torch.nn as nn
import torch.nn.functional as F  # 引入 F


class ProtocolRouter(nn.Module):
    def __init__(self, num_experts):
        super(ProtocolRouter, self).__init__()
        self.num_experts = num_experts

        self.register_buffer(
            "usage_counter",
            torch.zeros(num_experts, dtype=torch.long)
        )

        # 【新增】固定一个随机向量用于哈希，避免简单的 Sum 导致分布不均
        # 假设 input_ids 长度也就是 seq_len 不会超过 1024
        self.register_buffer("hash_vector", torch.randn(1024))

    def reset_usage(self):
        self.usage_counter.zero_()

    def forward(self, proto_ids=None, input_ids=None):
        # 情况 A: 监督路由 (保持不变)
        if proto_ids is not None:
            if proto_ids.dim() > 1:
                routing_ids = proto_ids[:, 0]
            else:
                routing_ids = proto_ids
            expert_ids = routing_ids % self.num_experts

        # 情况 B: 哈希路由 (Pre-training 阶段) -> 【重点修改这里】
        elif input_ids is not None:
            batch_size, seq_len = input_ids.size()

            # 方法 1：随机负载均衡 (最推荐用于预训练 Debug，确保每个 GPU 跑满)
            # 如果你不强求“相同的句子必须去同一个专家”，直接用这个：
            # expert_ids = torch.randint(0, self.num_experts, (batch_size,), device=input_ids.device)

            # 方法 2：改进的哈希 (保持一致性：相同的句子去同一个专家)
            # 使用 float 乘法打散整数 ID 的分布
            # 取前 32 个 token 参与哈希
            hash_len = min(seq_len, 32)

            # 获取对应的随机向量片段
            v = self.hash_vector[:hash_len].to(input_ids.device)

            # 点积：(Batch, L) * (L) -> (Batch)
            # input_ids 转换为 float 参与计算
            fingerprint = torch.matmul(input_ids[:, :hash_len].float(), v)

            # 转换为整数索引
            # 使用 abs 确保正数，转为 long 后取模
            expert_ids = fingerprint.abs().long() % self.num_experts

        # 情况 C: 随机路由 (兜底)
        else:
            batch_size = 1
            if input_ids is not None:  # 这里逻辑其实有点冗余，因为上面 elif 已经处理了 input_ids
                batch_size = input_ids.size(0)
            expert_ids = torch.randint(
                0, self.num_experts,
                (batch_size,),
                device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            )

        with torch.no_grad():
            for i in range(self.num_experts):
                self.usage_counter[i] += (expert_ids == i).sum()

        return expert_ids