import copy
import warnings

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from uer.encoders.transformer_encoder import TransformerEncoder
from uer.macro_moe.expert import TrafficMacroExpert, TrafficSharedAdapterExpert
from uer.macro_moe.router import ProtocolRouter


warnings.filterwarnings(
    "ignore",
    message=r"`torch\.cpu\.amp\.autocast\(args\.\.\.\)` is deprecated.*",
    category=FutureWarning,
)


class MacroMoEEncoder(nn.Module):
    def __init__(self, args):
        super(MacroMoEEncoder, self).__init__()
        self.num_experts = args.macro_expert_num
        self.top_k = max(1, min(getattr(args, "macro_top_k", 1), self.num_experts))
        self.use_shared_backbone = getattr(args, "macro_shared_backbone", False)
        self.checkpoint_experts = getattr(args, "macro_checkpoint_experts", False)
        self.disable_adapters = getattr(args, "macro_disable_adapters", False)
        self.few_shot_unfreeze_last_n = getattr(args, "few_shot_unfreeze_last_n", 0)

        # Original full-backbone design kept here for rollback reference:
        #
        # self.experts = nn.ModuleList([
        #     TrafficMacroExpert(args) for _ in range(self.num_experts)
        # ])
        if self.use_shared_backbone:
            args_copy = copy.deepcopy(args)
            if hasattr(args_copy, "is_moe"):
                args_copy.is_moe = False
            self.shared_backbone = TransformerEncoder(args_copy)
            self.experts = nn.ModuleList(
                [TrafficSharedAdapterExpert(args) for _ in range(self.num_experts)]
            )
        else:
            self.shared_backbone = None
            self.experts = nn.ModuleList(
                [TrafficMacroExpert(args) for _ in range(self.num_experts)]
            )

        self.router = ProtocolRouter(
            self.num_experts,
            args.hidden_size,
            noise_std=getattr(args, "macro_router_noise_std", 0.01),
            balance_weight=getattr(args, "macro_router_balance_weight", 0.2),
            entropy_weight=getattr(args, "macro_router_entropy_weight", 1.0),
            target_entropy=getattr(args, "macro_router_target_entropy", 0.6),
            router_feature=getattr(args, "macro_router_feature", "mean"),
            rank1_weight=getattr(args, "macro_router_rank1_weight", 0.0),
            rank2_weight=getattr(args, "macro_router_rank2_weight", 0.0),
            rank_target_entropy=getattr(args, "macro_router_rank_target_entropy", 0.45),
            specialization_weight=getattr(args, "macro_router_specialization_weight", 1.0),
            margin_weight=getattr(args, "macro_router_margin_weight", 1.0),
            target_margin=getattr(args, "macro_router_target_margin", 0.20),
            decorrelation_weight=getattr(args, "macro_router_decorrelation_weight", 0.5),
        )

    def set_adaptation_mode(self, mode=True):
        if self.shared_backbone is not None:
            if mode:
                for param in self.shared_backbone.parameters():
                    param.requires_grad = False

                unfreeze_last_n = getattr(self, "few_shot_unfreeze_last_n", None)
                if unfreeze_last_n is None:
                    unfreeze_last_n = 0

                if hasattr(self.shared_backbone, "transformer") and isinstance(self.shared_backbone.transformer, nn.ModuleList):
                    trainable_layers = min(max(int(unfreeze_last_n), 0), len(self.shared_backbone.transformer))
                    if trainable_layers > 0:
                        for layer in self.shared_backbone.transformer[-trainable_layers:]:
                            for param in layer.parameters():
                                param.requires_grad = True
                        self.shared_backbone.train()
                    else:
                        self.shared_backbone.eval()
                else:
                    self.shared_backbone.eval()
            else:
                for param in self.shared_backbone.parameters():
                    param.requires_grad = True
                self.shared_backbone.train()

            # Few-shot adaptation keeps the shared backbone frozen while allowing
            # the router to adapt to the downstream label space and reassign
            # samples to more suitable experts.
            for param in self.router.parameters():
                param.requires_grad = True
            self.router.train()

        for expert in self.experts:
            expert.set_adaptation_mode(mode)

    def _run_expert(self, expert_id, inputs):
        expert = self.experts[expert_id]
        if self.checkpoint_experts and self.training:
            return checkpoint(expert, inputs, use_reentrant=False)
        return expert(inputs)

    def _forward_shared_backbone(self, emb, seg):
        batch_size, seq_len, hidden_size = emb.size()
        shared_hidden = self.shared_backbone(emb, seg)

        if self.disable_adapters:
            zero_gate_loss = shared_hidden.new_zeros(())
            return shared_hidden, zero_gate_loss, None, None, None

        # Original routing input kept for rollback reference:
        # expert_indices, gate_loss, router_probs = self.router(inputs_embeds=emb, top_k=self.top_k)
        expert_indices, gate_loss, router_probs, router_logits, router_probs_full = self.router(
            inputs_embeds=shared_hidden, top_k=self.top_k
        )

        if expert_indices.dim() == 1:
            sorted_indices = torch.argsort(expert_indices)
            hidden_sorted = shared_hidden[sorted_indices]
            probs_sorted = router_probs[sorted_indices]

            expert_counts = torch.bincount(expert_indices, minlength=self.num_experts)
            hidden_split = torch.split(hidden_sorted, expert_counts.tolist(), dim=0)
            probs_split = torch.split(probs_sorted, expert_counts.tolist(), dim=0)

            outputs_list = []
            for i in range(self.num_experts):
                count = expert_counts[i].item()
                if count > 0:
                    delta = self._run_expert(i, hidden_split[i])
                    scale = probs_split[i].view(-1, 1, 1)
                    outputs_list.append(hidden_split[i] + delta * scale)
                else:
                    outputs_list.append(
                        torch.empty(0, seq_len, hidden_size, device=emb.device, dtype=emb.dtype)
                    )

            final_output_sorted = torch.cat(outputs_list, dim=0)
            reverse_indices = torch.argsort(sorted_indices)
            final_output = final_output_sorted[reverse_indices]
            return final_output, gate_loss, expert_indices, router_logits, router_probs_full

        topk_probs = router_probs / (router_probs.sum(dim=-1, keepdim=True) + 1e-9)
        delta_output = torch.zeros(batch_size, seq_len, hidden_size, device=emb.device, dtype=emb.dtype)

        for expert_id in range(self.num_experts):
            expert_mask = expert_indices == expert_id
            selected = expert_mask.any(dim=-1)
            if not selected.any():
                continue

            local_hidden = shared_hidden[selected]
            local_delta = self._run_expert(expert_id, local_hidden)
            local_weight = (topk_probs[selected] * expert_mask[selected].float()).sum(dim=-1)
            delta_output[selected] += local_delta * local_weight.view(-1, 1, 1)

        final_output = shared_hidden + delta_output
        return final_output, gate_loss, expert_indices, router_logits, router_probs_full

    def _forward_full_backbone_experts(self, emb, seg):
        batch_size, seq_len, hidden_size = emb.size()
        expert_indices, gate_loss, router_probs, router_logits, router_probs_full = self.router(inputs_embeds=emb, top_k=self.top_k)

        if expert_indices.dim() == 1:
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
                    outputs_list.append(
                        torch.empty(0, seq_len, hidden_size, device=emb.device, dtype=emb.dtype)
                    )

            final_output_sorted = torch.cat(outputs_list, dim=0)
            reverse_indices = torch.argsort(sorted_indices)
            final_output = final_output_sorted[reverse_indices]
            return final_output, gate_loss, expert_indices, router_logits, router_probs_full

        topk_probs = router_probs / (router_probs.sum(dim=-1, keepdim=True) + 1e-9)
        final_output = torch.zeros(batch_size, seq_len, hidden_size, device=emb.device, dtype=emb.dtype)

        for expert_id in range(self.num_experts):
            expert_mask = expert_indices == expert_id
            selected = expert_mask.any(dim=-1)
            if not selected.any():
                continue

            local_emb = emb[selected]
            local_seg = seg[selected]
            local_out = self.experts[expert_id](local_emb, local_seg)
            local_weight = (topk_probs[selected] * expert_mask[selected].float()).sum(dim=-1)
            final_output[selected] += local_out * local_weight.view(-1, 1, 1)

        return final_output, gate_loss, expert_indices, router_logits, router_probs_full

    def forward(self, emb, seg, input_ids=None, proto=None):
        if self.shared_backbone is not None:
            return self._forward_shared_backbone(emb, seg)
        return self._forward_full_backbone_experts(emb, seg)
