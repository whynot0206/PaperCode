import torch
import torch.nn as nn
from uer.macro_moe.expert import TrafficMacroExpert
from uer.macro_moe.router import ProtocolRouter

class MacroMoEEncoder(nn.Module):
    def __init__(self, args):
        super(MacroMoEEncoder, self).__init__()
        self.num_experts = args.macro_expert_num

        self.experts = nn.ModuleList([
            TrafficMacroExpert(args) for _ in range(self.num_experts)
        ])

        # 【修复点】这里必须传入 args.hidden_size，因为 Router 里的 Linear 层需要它
        # self.router = ProtocolRouter(self.num_experts, args.hidden_size)
        self.router = ProtocolRouter(
            self.num_experts,
            args.hidden_size,
            noise_std=getattr(args, "macro_router_noise_std", 0.01),
            balance_weight=getattr(args, "macro_router_balance_weight", 0.2),
            entropy_weight=getattr(args, "macro_router_entropy_weight", 1.0),
            target_entropy=getattr(args, "macro_router_target_entropy", 0.6),
        )

    def set_adaptation_mode(self, mode=True):
        for expert in self.experts:
            expert.set_adaptation_mode(mode)

    def forward(self, emb, seg, input_ids=None, proto=None):
        batch_size, seq_len, _ = emb.size()

        # 1. 调用 Router (接收 tuple 返回值)
        # 注意：这里传入 emb 作为 inputs_embeds
        expert_indices, gate_loss = self.router(inputs_embeds=emb)

        # 2. 后续逻辑保持不变...
        sorted_indices = torch.argsort(expert_indices)
        emb_sorted = emb[sorted_indices]
        seg_sorted = seg[sorted_indices]

        expert_counts = torch.bincount(expert_indices, minlength=self.num_experts)
        emb_split = torch.split(emb_sorted, expert_counts.tolist(), dim=0)
        seg_split = torch.split(seg_sorted, expert_counts.tolist(), dim=0)

        outputs_list = []
        for i in range(self.num_experts):
            count = expert_counts[i].item()
            if count > 0:
                sub_out = self.experts[i](emb_split[i], seg_split[i])
                outputs_list.append(sub_out)
            else:
                outputs_list.append(torch.empty(0, seq_len, emb.size(2), device=emb.device))

        final_output_sorted = torch.cat(outputs_list, dim=0)
        reverse_indices = torch.argsort(sorted_indices)
        final_output = final_output_sorted[reverse_indices]

        # 3. 返回 output 和 gate_loss
        return final_output, gate_loss