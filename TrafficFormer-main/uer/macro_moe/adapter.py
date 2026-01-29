import torch
import torch.nn as nn
from uer.layers.layer_norm import LayerNorm


class FewShotAdapter(nn.Module):
    """
    小样本适配模块 (Few-shot Adaptation Module)
    结构：Bottleneck Adapter (Linear -> Activation -> Linear) + Residual
    """

    def __init__(self, hidden_size, adapter_size=64, dropout=0.1):
        super(FewShotAdapter, self).__init__()
        self.down_project = nn.Linear(hidden_size, adapter_size)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.up_project = nn.Linear(adapter_size, hidden_size)
        self.layer_norm = LayerNorm(hidden_size)

    def forward(self, x):
        # x shape: [batch_size, seq_length, hidden_size]
        residual = x

        # 瓶颈结构计算
        out = self.down_project(x)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.up_project(out)

        # 残差连接 + 归一化
        return self.layer_norm(residual + out)