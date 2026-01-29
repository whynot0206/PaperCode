import copy
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F


class MoELayer(nn.Module):
    def __init__(self, hidden_size, num_experts, expert, route_method,
                 vocab_size, hash_list):
        super(MoELayer, self).__init__()

        self.num_experts = num_experts
        self.route_method = route_method

        # ===== experts =====
        self.experts = nn.ModuleList(
            [copy.deepcopy(expert) for _ in range(num_experts)]
        )

        # ===== routing init =====
        if route_method in ["gate-token", "gate-sentence", "feature-gate", "feature-gate-top2"]:
            # feature-gate 与 gate-sentence 共用 gate 结构
            self.gate = nn.Linear(hidden_size, num_experts, bias=False).float()

        elif route_method == "hash-random":
            self.hash_list = self._random_hash_list(vocab_size)

        elif route_method == "hash-balance":
            self.hash_list = self._balance_hash_list(hash_list)

        elif route_method == "proto":
            self.hash_list = torch.tensor(range(self.num_experts))

        else:
            raise KeyError(f"Routing method {route_method} not supported.")

    # ------------------------------------------------------------------
    # hash utils
    # ------------------------------------------------------------------
    def _random_hash_list(self, vocab_size):
        return torch.randint(
            low=0, high=self.num_experts, size=(vocab_size,)
        )

    def _balance_hash_list(self, hash_list):
        with open(hash_list, "rb") as f:
            result = pickle.load(f)
        return torch.tensor(result, dtype=torch.int64)

    # ------------------------------------------------------------------
    # gate-token (原有)
    # ------------------------------------------------------------------
    def _forward_gate_token(self, x):
        bsz, seq_len, dim = x.size()
        x = x.view(-1, dim)

        logits_gate = self.gate(x)
        prob_gate = F.softmax(logits_gate, dim=-1)
        gate = torch.argmax(prob_gate, dim=-1)

        order = gate.argsort(0)
        num_tokens = F.one_hot(gate, self.num_experts).gt(0).sum(0)
        gate_load = num_tokens.clone()

        x = x[order]
        x = x.split(num_tokens.tolist(), dim=0)

        # load balance loss
        P = prob_gate.mean(0)
        f = num_tokens.float() / num_tokens.sum()
        balance_loss = self.num_experts * torch.sum(P * f)

        prob_gate = prob_gate.gather(1, gate.unsqueeze(1))
        prob_gate = prob_gate[order]
        prob_gate = prob_gate.split(num_tokens.tolist(), dim=0)

        outputs = []
        for i in range(self.num_experts):
            if x[i].size(0) > 0:
                out = self.experts[i](x[i]) * prob_gate[i]
                outputs.append(out)

        x = torch.vstack(outputs)
        x = x[order.argsort(0)]
        x = x.view(bsz, seq_len, dim)

        return x, balance_loss, gate_load

    # ------------------------------------------------------------------
    # gate-sentence (原有)
    # ------------------------------------------------------------------
    def _forward_gate_sentence(self, x, attention_mask):
        x_masked = x * attention_mask.unsqueeze(-1)
        x_avg = x_masked.sum(1) / attention_mask.unsqueeze(-1).sum(1)

        logits_gate = self.gate(x_avg)
        prob_gate = F.softmax(logits_gate, dim=-1)
        gate = torch.argmax(prob_gate, dim=-1)

        order = gate.argsort(0)
        num_sent = F.one_hot(gate, self.num_experts).gt(0).sum(0)
        gate_load = num_sent.clone()

        x = x[order]
        x = x.split(num_sent.tolist(), dim=0)

        # load balance loss
        P = prob_gate.mean(0)
        f = num_sent.float() / num_sent.sum()
        balance_loss = self.num_experts * torch.sum(P * f)

        prob_gate = prob_gate.gather(1, gate.unsqueeze(1))
        prob_gate = prob_gate[order]
        prob_gate = prob_gate.split(num_sent.tolist(), dim=0)

        outputs = []
        for i in range(self.num_experts):
            if x[i].size(0) > 0:
                out = self.experts[i](x[i])
                out = out * prob_gate[i].unsqueeze(-1)
                outputs.append(out)

        x = torch.vstack(outputs)
        x = x[order.argsort(0)]

        return x, balance_loss, gate_load

    # ------------------------------------------------------------------
    # feature-gate（新增）
    # ------------------------------------------------------------------
    def _forward_feature_gate(self, x, attention_mask):
        """
        基于流量特征的专家路由 (Sentence/Flow Level)
        """
        # 句级特征提取 (Mean Pooling)
        x_masked = x * attention_mask.unsqueeze(-1)
        # 【优化】防止除以0
        denominator = attention_mask.unsqueeze(-1).sum(1) + 1e-9
        feat = x_masked.sum(1) / denominator
        # feat: [Batch, Hidden]

        # Gate 计算
        logits_gate = self.gate(feat)
        prob_gate = F.softmax(logits_gate, dim=-1)
        gate = torch.argmax(prob_gate, dim=-1)  # Hard Routing (Top-1)

        # 按专家分发 (Sorting & Splitting)
        order = gate.argsort(0)
        num_sent = F.one_hot(gate, self.num_experts).gt(0).sum(0)
        gate_load = num_sent.clone()

        # 对 Batch 维度进行重排
        x = x[order]
        x = x.split(num_sent.tolist(), dim=0)

        # Load Balance Loss
        P = prob_gate.mean(0)
        # 【优化】防止 num_sent.sum() 为 0 (空 batch)
        total_samples = num_sent.sum() + 1e-9
        f = num_sent.float() / total_samples
        balance_loss = self.num_experts * torch.sum(P * f)

        # 准备 Soft Gating 权重
        prob_gate = prob_gate.gather(1, gate.unsqueeze(1))
        prob_gate = prob_gate[order]
        prob_gate = prob_gate.split(num_sent.tolist(), dim=0)

        # 专家计算
        outputs = []
        for i in range(self.num_experts):
            if x[i].size(0) > 0:
                # 专家处理整个序列 [Subset_Batch, Seq, Dim]
                out = self.experts[i](x[i])
                # Soft Gating 加权: out * prob
                out = out * prob_gate[i].unsqueeze(-1)
                outputs.append(out)

        # 还原顺序
        x = torch.vstack(outputs)
        x = x[order.argsort(0)]

        return x, balance_loss, gate_load

    def _forward_feature_gate_top2(self, x, attention_mask, k=2):
        """
        基于流量特征的 Top-2 MoE 路由（流/句子级别）

        参数:
            x: [B, T, D] 输入张量
            attention_mask: [B, T] 注意力掩码
            k: 激活的专家数量 (默认=2)
        返回:
            output: [B, T, D] 加权后的输出
            balance_loss: scalar 负载均衡损失
            gate_load: [num_experts] 各个专家的负载统计
        """

        B, T, D = x.size()

        # --------------------------------------------------
        # 流级特征提取 (使用平均池化 Mean Pooling)
        # --------------------------------------------------
        x_masked = x * attention_mask.unsqueeze(-1)
        # 分母加 1e-9 防止除零错误
        denom = attention_mask.sum(1, keepdim=True) + 1e-9
        feat = x_masked.sum(1) / denom  # [B, D]

        # --------------------------------------------------
        # 计算门控 Logits 和概率
        # --------------------------------------------------
        logits = self.gate(feat)  # [B, E]
        prob = F.softmax(logits, dim=-1)  # [B, E]

        # Top-K 选择 (选出概率最高的 k 个专家)
        topk_prob, topk_idx = torch.topk(prob, k=k, dim=-1)  # [B, k]

        # 归一化 Top-K 概率 (非常重要！保证权重和为1)
        topk_prob = topk_prob / (topk_prob.sum(dim=-1, keepdim=True) + 1e-9)

        # --------------------------------------------------
        # 专家负载统计 (用于后续分析或辅助 Loss)
        # --------------------------------------------------
        gate_load = torch.zeros(self.num_experts, device=x.device)
        for i in range(k):
            # 将 Top-k 中所有被选中的专家计数累加
            gate_load.scatter_add_(
                0,
                topk_idx[:, i],
                torch.ones_like(topk_idx[:, i], dtype=torch.float)
            )

        # --------------------------------------------------
        # 负载均衡损失 (Soft 版本)
        # --------------------------------------------------
        P = prob.mean(0)  # [E] 平均概率
        f = gate_load / (gate_load.sum() + 1e-9)  # [E] 实际负载频率
        balance_loss = self.num_experts * torch.sum(P * f)

        # --------------------------------------------------
        # 分发给专家 (Top-2 核心逻辑)
        # --------------------------------------------------
        expert_outputs = torch.zeros_like(x)

        # 循环 k 次，分别处理第 1 顺位、第 2 顺位...的专家计算
        for i in range(k):
            expert_id = topk_idx[:, i]  # [B] 当前顺位的专家 ID
            expert_weight = topk_prob[:, i]  # [B] 对应的归一化权重

            # 按照专家 ID 进行排序，方便批处理
            order = expert_id.argsort(0)
            sorted_x = x[order]
            sorted_weight = expert_weight[order]

            # 统计每个专家分配到的样本数量
            # (建议优化：此处可以用 torch.bincount 替代 one_hot 以节省显存)
            num_sent = F.one_hot(
                expert_id, self.num_experts
            ).sum(0).tolist()

            # 根据数量切分 Batch
            split_x = sorted_x.split(num_sent, dim=0)
            split_w = sorted_weight.split(num_sent, dim=0)

            offset = 0
            for e in range(self.num_experts):
                # 如果该专家没有分配到样本，直接跳过
                if split_x[e].size(0) == 0:
                    continue

                # 专家前向传播
                out = self.experts[e](split_x[e])  # [n, T, D]

                # 乘上路由权重
                out = out * split_w[e].view(-1, 1, 1)

                # 将结果累加到最终输出中 (关键：+= 支持多专家协作)
                # 使用 order 索引还原原来的顺序
                expert_outputs[order[offset:offset + split_x[e].size(0)]] += out
                offset += split_x[e].size(0)

        return expert_outputs, balance_loss, gate_load

    # ------------------------------------------------------------------
    # hash / proto（原有）
    # ------------------------------------------------------------------
    def _forward_hash(self, x, input_ids):
        bsz, seq_len, dim = x.size()
        x = x.view(-1, dim)

        self.hash_list = self.hash_list.to(x.device)
        gate = self.hash_list[input_ids.view(-1)]

        order = gate.argsort(0)
        num_tokens = F.one_hot(gate, self.num_experts).gt(0).sum(0)
        gate_load = num_tokens.clone()

        x = x[order]
        x = x.split(num_tokens.tolist(), dim=0)

        outputs = []
        for i in range(self.num_experts):
            if x[i].size(0) > 0:
                outputs.append(self.experts[i](x[i]))

        x = torch.vstack(outputs)
        x = x[order.argsort(0)]
        x = x.view(bsz, seq_len, dim)

        return x, 0.0, gate_load

    def _forward_proto(self, x, proto):
        bsz, seq_len, dim = x.size()
        x = x.view(-1, dim)

        gate = self.hash_list[proto.view(-1)]
        order = gate.argsort(0)
        num_tokens = F.one_hot(gate, self.num_experts).gt(0).sum(0)
        gate_load = num_tokens.clone()

        x = x[order]
        x = x.split(num_tokens.tolist(), dim=0)

        outputs = []
        for i in range(self.num_experts):
            if x[i].size(0) > 0:
                outputs.append(self.experts[i](x[i]))

        x = torch.vstack(outputs)
        x = x[order.argsort(0)]
        x = x.view(bsz, seq_len, dim)

        return x, 0.0, gate_load

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(self, x, input_ids, attention_mask, proto=None):
        if self.route_method == "gate-token":
            return self._forward_gate_token(x)

        elif self.route_method == "gate-sentence":
            return self._forward_gate_sentence(x, attention_mask)

        elif self.route_method == "feature-gate":
            return self._forward_feature_gate(x, attention_mask)

        elif self.route_method in ["hash-random", "hash-balance"]:
            return self._forward_hash(x, input_ids)

        elif self.route_method == "proto":
            bsz, seq_len, _ = x.size()
            proto = proto.repeat(seq_len, 1).t().contiguous()
            return self._forward_proto(x, proto)

        elif self.route_method == "feature-gate-top2":
            return self._forward_feature_gate_top2(
                x, attention_mask, k=2
            )

        else:
            raise KeyError("Routing method not supported.")
