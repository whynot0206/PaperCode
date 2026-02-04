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
        # 1. 降维投影：将高维特征压缩到低维 (例如 768 -> 64)
        # 这样可以大幅减少参数量
        self.down_project = nn.Linear(hidden_size, adapter_size)

        # 2. 非线性激活函数
        self.activation = nn.ReLU()

        # 3. Dropout 防止过拟合
        self.dropout = nn.Dropout(dropout)

        # 4. 升维投影：将低维特征恢复回原始维度 (例如 64 -> 768)
        self.up_project = nn.Linear(adapter_size, hidden_size)

        # 5. 层归一化：用于残差连接后的归一化
        self.layer_norm = LayerNorm(hidden_size)

    def forward(self, x):
        # x shape: [batch_size, seq_length, hidden_size]
        residual = x  # 保存原始输入用于残差连接

        # 瓶颈结构前向计算
        out = self.down_project(x)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.up_project(out)

        # 残差连接 (Original + Adapter_Output) 并进行归一化
        # 这保证了如果 Adapter 输出为0，模型退化为原模型，利于训练稳定性
        return self.layer_norm(residual + out)