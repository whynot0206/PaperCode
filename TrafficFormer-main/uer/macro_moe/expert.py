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
        # 必须强制设置 is_moe=False。因为专家本身已经是 MoE 的一部分了，
        # 我们希望专家的内部是一个标准的 Transformer，不要再递归地变成 MoE
        args_copy = copy.deepcopy(args)
        if hasattr(args_copy, 'is_moe'):
            args_copy.is_moe = False

        # 2. 初始化骨干网络 (复用 UER 的 TransformerEncoder)
        self.backbone = TransformerEncoder(args_copy)

        # 3. 初始化适配器 (参数量很小)
        adapter_size = getattr(args, "adapter_size", 64)
        dropout = getattr(args, "dropout", 0.1)
        self.adapter = FewShotAdapter(args.hidden_size, adapter_size, dropout)

        # 模式标志：默认 False (预训练/全量微调模式)
        self.adaptation_mode = False

    def set_adaptation_mode(self, mode=True):
        """
        核心控制函数：
        mode=False: 预训练/全量微调阶段 -> 训练 Backbone, 冻结/跳过 Adapter
        mode=True:  小样本适配阶段 -> 冻结 Backbone, 训练 Adapter
        """
        self.adaptation_mode = mode

        if mode:  # 小样本适配模式
            # 冻结骨干网络的所有参数，不计算梯度
            for param in self.backbone.parameters():
                param.requires_grad = False
            # 激活适配器参数
            for param in self.adapter.parameters():
                param.requires_grad = True

            # 设置运行模式 (影响 Dropout 和 BatchNorm)
            self.backbone.eval()  # 骨干设为评估模式
            self.adapter.train()  # 适配器设为训练模式
        else:  # 预训练模式
            # 激活骨干参数
            for param in self.backbone.parameters():
                param.requires_grad = True
            # 冻结适配器 (预训练时通常不希望训练 adapter，或者让它恒等映射)
            for param in self.adapter.parameters():
                param.requires_grad = False

            self.backbone.train()
            self.adapter.eval()

    def forward(self, emb, seg):
        # 1. 骨干提取特征
        if self.adaptation_mode:
            # 如果是适配模式，显式告诉 PyTorch 不需要计算 Backbone 的梯度，节省显存
            with torch.no_grad():
                features = self.backbone(emb, seg)
        else:
            features = self.backbone(emb, seg)

        # 2. 适配器逻辑
        # 仅在小样本适配模式下才让数据流经 Adapter
        if self.adaptation_mode:
            features = self.adapter(features)

        return features