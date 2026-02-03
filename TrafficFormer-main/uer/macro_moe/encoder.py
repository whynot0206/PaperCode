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

        # [修改] 创建路由器时传入 hidden_size
        self.router = ProtocolRouter(self.num_experts, args.hidden_size)

    def set_adaptation_mode(self, mode=True):
        """全局切换模式"""
        for expert in self.experts:
            expert.set_adaptation_mode(mode)

    def forward(self, emb, seg, input_ids=None, proto=None):
        """
        Args:
            emb: [batch, seq, dim]
            seg: [batch, seq]
            input_ids: (未使用，保留接口兼容)
            proto: [batch] (可选)
        Returns:
            final_output: [batch, seq, dim]
            gate_loss: scalar tensor
        """
        batch_size, seq_len, _ = emb.size()

        # 1. 路由决策 (传入 embedding)
        # [修改] 接收 expert_indices 和 gate_loss
        expert_indices, gate_loss = self.router(proto_ids=proto, inputs_embeds=emb)

        # 2. 排序
        sorted_indices = torch.argsort(expert_indices)
        emb_sorted = emb[sorted_indices]
        seg_sorted = seg[sorted_indices]

        # 3. 切分
        expert_counts = torch.bincount(expert_indices, minlength=self.num_experts)
        emb_split = torch.split(emb_sorted, expert_counts.tolist(), dim=0)
        seg_split = torch.split(seg_sorted, expert_counts.tolist(), dim=0)

        # 4. 专家计算
        outputs_list = []
        for i in range(self.num_experts):
            count = expert_counts[i].item()
            if count > 0:
                sub_out = self.experts[i](emb_split[i], seg_split[i])
                outputs_list.append(sub_out)
            else:
                outputs_list.append(torch.empty(0, seq_len, emb.size(2), device=emb.device))

        # 5. 合并
        final_output_sorted = torch.cat(outputs_list, dim=0)

        # 6. 恢复顺序
        reverse_indices = torch.argsort(sorted_indices)
        final_output = final_output_sorted[reverse_indices]

        # [修改] 返回结果和损失
        return final_output, gate_loss