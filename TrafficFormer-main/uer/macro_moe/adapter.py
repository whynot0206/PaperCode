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


class SharedThinAdapter(nn.Module):
    """
    Lightweight delta adapter for the shared-backbone Macro-MoE path.
    Returns only the residual delta so the encoder can combine it with
    the shared hidden states.
    """

    def __init__(self, hidden_size, adapter_size=64, dropout=0.1):
        super(SharedThinAdapter, self).__init__()
        self.down_project = nn.Linear(hidden_size, adapter_size)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.up_project = nn.Linear(adapter_size, hidden_size)

    def forward(self, x):
        out = self.down_project(x)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.up_project(out)
        return out


class SharedMlpExpert(nn.Module):
    """
    Stronger shared-backbone expert implemented as a two-hidden-layer MLP.
    It keeps the same residual-delta interface as SharedThinAdapter, but
    provides substantially more expert capacity.
    """

    def __init__(self, hidden_size, adapter_size=64, dropout=0.1):
        super(SharedMlpExpert, self).__init__()
        hidden_mid = max(adapter_size, hidden_size // 4)
        self.fc1 = nn.Linear(hidden_size, hidden_mid)
        self.fc2 = nn.Linear(hidden_mid, hidden_mid)
        self.fc3 = nn.Linear(hidden_mid, hidden_size)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.fc1(x)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.fc3(out)
        return out


class SharedGatedExpert(nn.Module):
    """
    Gated expert for the shared-backbone Macro-MoE path:
        delta = W_o( SiLU(W_g x) * W_v x )
    This is stronger than a thin bottleneck adapter while remaining much
    lighter than an entire expert backbone.
    """

    def __init__(self, hidden_size, adapter_size=64, dropout=0.1):
        super(SharedGatedExpert, self).__init__()
        hidden_mid = max(adapter_size, hidden_size // 4)
        self.gate_proj = nn.Linear(hidden_size, hidden_mid)
        self.value_proj = nn.Linear(hidden_size, hidden_mid)
        self.out_proj = nn.Linear(hidden_mid, hidden_size)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        gate = self.activation(self.gate_proj(x))
        value = self.value_proj(x)
        out = gate * value
        out = self.dropout(out)
        out = self.out_proj(out)
        return out


def build_shared_expert_module(hidden_size, adapter_size=64, dropout=0.1, expert_type="thin"):
    expert_type = (expert_type or "thin").lower()
    if expert_type == "thin":
        return SharedThinAdapter(hidden_size, adapter_size, dropout)
    if expert_type == "mlp":
        return SharedMlpExpert(hidden_size, adapter_size, dropout)
    if expert_type == "gated":
        return SharedGatedExpert(hidden_size, adapter_size, dropout)
    raise ValueError("Unsupported shared expert type: {}".format(expert_type))
