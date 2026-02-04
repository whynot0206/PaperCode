import torch
import torch.nn as nn
from uer.macro_moe.expert import TrafficMacroExpert
from uer.macro_moe.router import ProtocolRouter


class MacroMoEEncoder(nn.Module):
    def __init__(self, args):
        super(MacroMoEEncoder, self).__init__()
        self.num_experts = args.macro_expert_num

        # 1. 初始化专家列表
        # ModuleList 包含 N 个 TrafficMacroExpert
        self.experts = nn.ModuleList([
            TrafficMacroExpert(args) for _ in range(self.num_experts)
        ])

        # 2. 初始化路由器
        self.router = ProtocolRouter(self.num_experts, args.hidden_size)

    def set_adaptation_mode(self, mode=True):
        # 这是一个辅助函数，用于一键切换所有专家的模式 (预训练 vs 小样本适配)
        for expert in self.experts:
            expert.set_adaptation_mode(mode)

    def forward(self, emb, seg, input_ids=None, proto=None):
        batch_size, seq_len, _ = emb.size()

        # 1. 调用 Router 进行路由
        # 根据输入 embedding 决定每个样本去哪个专家
        # expert_indices: [batch_size], gate_loss: 标量
        expert_indices, gate_loss = self.router(inputs_embeds=emb)

        # 2. 数据分发逻辑 (Dispatching)
        # 为了高效计算，我们不能写 for loop 逐个处理样本。
        # 方法是：先对样本按专家ID排序，然后切片，最后拼回来。

        # a. 获取排序索引
        sorted_indices = torch.argsort(expert_indices)

        # b. 按照专家ID重排数据
        emb_sorted = emb[sorted_indices]
        seg_sorted = seg[sorted_indices]

        # c. 计算每个专家分到了多少个样本
        expert_counts = torch.bincount(expert_indices, minlength=self.num_experts)

        # d. 切分数据 (Split)
        # emb_split 是一个 tuple，包含 N 个 tensor，每个 tensor 对应一个专家的输入
        emb_split = torch.split(emb_sorted, expert_counts.tolist(), dim=0)
        seg_split = torch.split(seg_sorted, expert_counts.tolist(), dim=0)

        # 3. 专家并行计算
        outputs_list = []
        for i in range(self.num_experts):
            count = expert_counts[i].item()
            if count > 0:
                # 如果这个专家分到了数据，就进行计算
                sub_out = self.experts[i](emb_split[i], seg_split[i])
                outputs_list.append(sub_out)
            else:
                # 如果没分到数据，创建一个空的占位符 (防止后面 cat 报错或逻辑混乱)
                outputs_list.append(torch.empty(0, seq_len, emb.size(2), device=emb.device))

        # 4. 数据重组 (Combining)
        # 将所有专家的输出拼接在一起 (此时顺序还是按专家ID排序的)
        final_output_sorted = torch.cat(outputs_list, dim=0)

        # 5. 恢复原始顺序
        # 计算反向索引，将数据还原为 batch 输入时的顺序
        reverse_indices = torch.argsort(sorted_indices)
        final_output = final_output_sorted[reverse_indices]

        # 返回最终特征和路由损失 (损失将在 Trainer 中被加到总 Loss 里)
        return final_output, gate_loss