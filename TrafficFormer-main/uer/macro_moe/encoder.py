import torch
import torch.nn as nn
from uer.macro_moe.expert import TrafficMacroExpert
from uer.macro_moe.router import ProtocolRouter

class MacroMoEEncoder(nn.Module):
    def __init__(self, args):
        super(MacroMoEEncoder, self).__init__()
        self.num_experts = args.macro_expert_num
        self.top_k = max(1, min(getattr(args, "macro_top_k", 1), self.num_experts))

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
        batch_size, seq_len, hidden_size = emb.size()

        # 支持 Top-k 路由（k=1 保持兼容，k>1 缓解专家塌缩）
        expert_indices, gate_loss, router_probs = self.router(inputs_embeds=emb, top_k=self.top_k)

        if expert_indices.dim() == 1:
            # 兼容旧逻辑：top_k=1
            sorted_indices = torch.argsort(expert_indices)
            emb_sorted = emb[sorted_indices]
            seg_sorted = seg[sorted_indices]
            probs_sorted = router_probs[sorted_indices]

            expert_counts = torch.bincount(expert_indices, minlength=self.num_experts)
            emb_split = torch.split(emb_sorted, expert_counts.tolist(), dim=0)
            seg_split = torch.split(seg_sorted, expert_counts.tolist(), dim=0)
            probs_split = torch.split(probs_sorted, expert_counts.tolist(), dim=0)

            outputs_list = []
            for i in range(self.num_experts):
                count = expert_counts[i].item()
                if count > 0:
                    sub_out = self.experts[i](emb_split[i], seg_split[i])
                    scale = probs_split[i].view(-1, 1, 1)
                    outputs_list.append(sub_out * scale)
                else:
                    outputs_list.append(torch.empty(0, seq_len, hidden_size, device=emb.device))

            final_output_sorted = torch.cat(outputs_list, dim=0)
            reverse_indices = torch.argsort(sorted_indices)
            final_output = final_output_sorted[reverse_indices]
            return final_output, gate_loss, expert_indices

        # top_k > 1：对每个 expert 按样本掩码聚合，再按归一化权重加权求和
        topk_probs = router_probs / (router_probs.sum(dim=-1, keepdim=True) + 1e-9)
        final_output = torch.zeros(batch_size, seq_len, hidden_size, device=emb.device, dtype=emb.dtype)

        for expert_id in range(self.num_experts):
            expert_mask = (expert_indices == expert_id)  # [batch, top_k]
            selected = expert_mask.any(dim=-1)  # [batch]
            if not selected.any():
                continue

            local_emb = emb[selected]
            local_seg = seg[selected]
            local_out = self.experts[expert_id](local_emb, local_seg)

            # 每个样本在该 expert 上可能出现一次（top-k唯一索引），取对应概率并广播。
            local_weight = (topk_probs[selected] * expert_mask[selected].float()).sum(dim=-1)
            final_output[selected] += local_out * local_weight.view(-1, 1, 1)

        return final_output, gate_loss, expert_indices
