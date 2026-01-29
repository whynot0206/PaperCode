import torch
import torch.nn as nn
from uer.macro_moe.expert import TrafficMacroExpert
from uer.macro_moe.router import ProtocolRouter


class MacroMoEEncoder(nn.Module):
    def __init__(self, args):
        super(MacroMoEEncoder, self).__init__()
        self.num_experts = args.macro_expert_num

        # 创建专家池
        self.experts = nn.ModuleList([
            TrafficMacroExpert(args) for _ in range(self.num_experts)
        ])

        # 创建路由器
        self.router = ProtocolRouter(self.num_experts)

    def set_adaptation_mode(self, mode=True):
        """全局切换模式"""
        for expert in self.experts:
            expert.set_adaptation_mode(mode)

    def forward(self, emb, seg, input_ids=None, proto=None):
        """
        Args:
            emb: Embedding 输出
            seg: Segment ID
            input_ids: 用于预训练哈希路由
            proto: 用于微调监督路由
        """
        batch_size, seq_len, _ = emb.size()

        # 1. 路由决策 (支持 Pretrain 无 proto 情况)
        expert_indices = self.router(proto_ids=proto, input_ids=input_ids)  # [batch_size]

        # 2. 根据专家索引对 Batch 进行排序
        sorted_indices = torch.argsort(expert_indices)
        emb_sorted = emb[sorted_indices]
        seg_sorted = seg[sorted_indices]

        # 3. 计算每个专家的数据量并切分
        expert_counts = torch.bincount(expert_indices, minlength=self.num_experts)
        emb_split = torch.split(emb_sorted, expert_counts.tolist(), dim=0)
        seg_split = torch.split(seg_sorted, expert_counts.tolist(), dim=0)

        # 4. 专家并行计算
        outputs_list = []
        for i in range(self.num_experts):
            count = expert_counts[i].item()
            if count > 0:
                # 对应专家处理对应数据
                sub_out = self.experts[i](emb_split[i], seg_split[i])
                outputs_list.append(sub_out)
            else:
                # 空数据占位，保持 device 一致
                outputs_list.append(torch.empty(0, seq_len, emb.size(2), device=emb.device))

        # 5. 合并结果
        final_output_sorted = torch.cat(outputs_list, dim=0)

        # 6. 恢复原始 Batch 顺序
        reverse_indices = torch.argsort(sorted_indices)
        final_output = final_output_sorted[reverse_indices]

        return final_output