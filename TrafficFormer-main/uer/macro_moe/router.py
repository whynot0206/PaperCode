import torch
import torch.nn as nn


class ProtocolRouter(nn.Module):
    def __init__(self, num_experts):
        super(ProtocolRouter, self).__init__()
        self.num_experts = num_experts

    def forward(self, proto_ids=None, input_ids=None):
        """
        路由逻辑：
        1. 有 proto_ids (Fine-tuning/Adaptation): 优先使用监督信号路由
        2. 无 proto_ids (Pre-training): 使用 input_ids 进行哈希路由
        """
        # 情况 A: 监督路由 (Fine-tuning / Adaptation 阶段)
        if proto_ids is not None:
            if proto_ids.dim() > 1:
                # 假设同一条流的标签一致，取第一个
                routing_ids = proto_ids[:, 0]
            else:
                routing_ids = proto_ids
            return routing_ids % self.num_experts

        # 情况 B: 哈希路由 (Pre-training 阶段)
        # 利用 input_ids 的特征（前16个字节的求和）来决定去哪个专家
        # 这样能保证内容相似的流量总是去同一个专家，实现聚类效果
        elif input_ids is not None:
            # 简单的哈希函数：Sum(前16个token) % num_experts
            seq_len = input_ids.size(1)
            hash_len = min(seq_len, 16)
            # 这里的 input_ids 是 token index，求和可以作为一种指纹
            fingerprint = torch.sum(input_ids[:, :hash_len], dim=1)
            return fingerprint % self.num_experts

        # 情况 C: 随机路由 (兜底，极少触发)
        else:
            batch_size = 1
            if input_ids is not None: batch_size = input_ids.size(0)
            return torch.randint(0, self.num_experts, (batch_size,),
                                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))