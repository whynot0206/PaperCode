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
        adapter_size = getattr(args, "adapter_size", 32)
        dropout = getattr(args, "dropout", 0.1)
        self.adapter = FewShotAdapter(args.hidden_size, adapter_size, dropout)

        # 模式标志 (默认为 False，即预训练/全量微调模式)
        self.adaptation_mode = False

    def set_adaptation_mode(self, mode=True):
        """
        mode=False: 预训练/全量微调阶段 (Backbone 和 Adapter 双启用，联合训练)
        mode=True:  小样本适配阶段 (冻结 Backbone，仅训练 Adapter)
        """
        self.adaptation_mode = mode

        if mode:  # 小样本适配模式 (Few-Shot)
            # 冻结骨干
            for param in self.backbone.parameters():
                param.requires_grad = False
            # 激活适配器
            for param in self.adapter.parameters():
                param.requires_grad = True
            self.backbone.eval()
            self.adapter.train()

        else:  # 预训练/全量微调模式 (Pretrain / Full Fine-tuning)
            # 【修改点】激活骨干
            for param in self.backbone.parameters():
                param.requires_grad = True
            # 【修改点】激活适配器 (预训练时必须训练它，防止冷启动)
            for param in self.adapter.parameters():
                param.requires_grad = True
            self.backbone.train()
            self.adapter.train()

    def get_backbone_grad_norm(self):
        """
        用于观测 Backbone 是否真的在被训练
        返回 backbone 所有参数梯度的 L2 norm
        """
        total_sq_norm = 0.0
        for p in self.backbone.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2).item()
                total_sq_norm += param_norm ** 2

        # L2 Norm = sqrt(sum ||g_i||^2)
        return total_sq_norm ** 0.5

    def forward(self, emb, seg):
        # 1. 骨干提取特征
        if self.adaptation_mode:
            # 小样本模式：骨干网络不计算梯度，节省显存并防止破坏已学到的通用特征
            with torch.no_grad():
                features = self.backbone(emb, seg)
        else:
            # 预训练/全量微调模式：骨干网络正常计算梯度
            features = self.backbone(emb, seg)

        # 2. 适配器逻辑
        # 【修改点】无论是预训练还是小样本适配，特征都必须经过 Adapter！
        # 这样在 mode=False 时，Adapter 才能和 Backbone 一起被训练，获得处理流量特征的能力。
        features = self.adapter(features)

        return features