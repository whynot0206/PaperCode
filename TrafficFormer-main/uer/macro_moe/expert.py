import torch
import torch.nn as nn
import copy
from uer.encoders.transformer_encoder import TransformerEncoder
from uer.macro_moe.adapter import FewShotAdapter


class TrafficMacroExpert(nn.Module):
    """
    宏观专家 (Macro Expert)
    由 Backbone (TrafficFormer Encoder) + Adapter 组成
    """

    def __init__(self, args):
        super(TrafficMacroExpert, self).__init__()

        # 1. 准备骨干参数
        # 强制设置 is_moe=False，确保专家内部是标准的 Transformer 结构
        # 避免递归调用原有的 Micro-MoE 代码
        args_copy = copy.deepcopy(args)
        if hasattr(args_copy, 'is_moe'):
            args_copy.is_moe = False

        # 2. 初始化骨干网络 (复用原有 Encoder)
        self.backbone = TransformerEncoder(args_copy)

        # 3. 初始化适配器
        adapter_size = getattr(args, "adapter_size", 64)
        dropout = getattr(args, "dropout", 0.1)
        self.adapter = FewShotAdapter(args.hidden_size, adapter_size, dropout)

        # 模式标志
        self.adaptation_mode = False

    def set_adaptation_mode(self, mode=True):
        """
        mode=False: 预训练/全量微调阶段 (训练 Backbone, 跳过 Adapter)
        mode=True:  小样本适配阶段 (冻结 Backbone, 训练 Adapter)
        """
        self.adaptation_mode = mode

        if mode:  # 小样本适配模式
            # 冻结骨干
            for param in self.backbone.parameters():
                param.requires_grad = False
            # 激活适配器
            for param in self.adapter.parameters():
                param.requires_grad = True
            self.backbone.eval()
            self.adapter.train()
        else:  # 预训练/全量微调模式
            # 激活骨干
            for param in self.backbone.parameters():
                param.requires_grad = True
            # 冻结适配器 (预训练时不训练它)
            for param in self.adapter.parameters():
                param.requires_grad = False
            self.backbone.train()
            self.adapter.eval()

    def forward(self, emb, seg):
        # 1. 骨干提取特征
        if self.adaptation_mode:
            with torch.no_grad():
                features = self.backbone(emb, seg)
        else:
            features = self.backbone(emb, seg)

        # 2. 适配器逻辑
        # 仅在小样本适配模式下，或者为了保证输出维度一致性时通过 Adapter
        # 但在预训练阶段，我们通常希望直接优化 backbone，不经过 adapter
        if self.adaptation_mode:
            features = self.adapter(features)

        return features