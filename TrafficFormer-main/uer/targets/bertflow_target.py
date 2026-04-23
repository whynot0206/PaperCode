import torch
import torch.nn as nn

from uer.targets import *


class BertFlowTarget(MlmTarget):
    """
    BertFlow pretraining with:
      - masked byte/datagram modeling
      - hierarchical flow relation classification
      - CLS-based proto weak supervision
      - router-based proto weak supervision
    """

    def __init__(self, args, vocab_size):
        super(BertFlowTarget, self).__init__(args, vocab_size)
        self.rel_labels_num = 6
        self.proto_labels_num = getattr(args, "flow_proto_num", 2)
        self.mbm_loss_weight = getattr(args, "bertflow_mbm_loss_weight", 0.4)
        self.rel_loss_weight = getattr(args, "bertflow_rel_loss_weight", 1.0)
        self.proto_cls_loss_weight = getattr(args, "bertflow_proto_cls_loss_weight", 0.3)
        self.proto_route_loss_weight = getattr(args, "bertflow_proto_route_loss_weight", 0.3)

        self.rel_linear_1 = nn.Linear(args.hidden_size, args.hidden_size)
        self.rel_linear_2 = nn.Linear(args.hidden_size, self.rel_labels_num)

        self.proto_cls_linear_1 = nn.Linear(args.hidden_size, args.hidden_size)
        self.proto_cls_linear_2 = nn.Linear(args.hidden_size, self.proto_labels_num)

        route_input_size = max(1, getattr(args, "macro_expert_num", 1))
        self.proto_route_linear_1 = nn.Linear(route_input_size, args.hidden_size)
        self.proto_route_linear_2 = nn.Linear(args.hidden_size, self.proto_labels_num)

    def _classification_loss(self, logits, tgt):
        log_probs = self.softmax(logits)
        loss = self.criterion(log_probs, tgt)
        correct = log_probs.argmax(dim=-1).eq(tgt).sum()
        return loss, correct

    def forward(self, memory_bank, tgt, proto=None, router_probs=None, router_logits=None):
        assert type(tgt) == tuple
        tgt_mlm, tgt_rel = tgt[0], tgt[1]

        loss_mlm, correct_mlm, denominator = self.mlm(memory_bank, tgt_mlm)

        cls_repr = memory_bank[:, 0, :]

        rel_hidden = torch.tanh(self.rel_linear_1(cls_repr))
        rel_logits = self.rel_linear_2(rel_hidden)
        loss_rel, correct_rel = self._classification_loss(rel_logits, tgt_rel)

        zero = cls_repr.new_zeros(())
        zero_correct = cls_repr.new_zeros((), dtype=torch.long)

        if proto is not None and self.proto_cls_loss_weight > 0.0:
            proto_hidden = torch.tanh(self.proto_cls_linear_1(cls_repr))
            proto_logits = self.proto_cls_linear_2(proto_hidden)
            loss_proto_cls, correct_proto_cls = self._classification_loss(proto_logits, proto)
        else:
            loss_proto_cls, correct_proto_cls = zero, zero_correct

        route_signal = router_logits if router_logits is not None else router_probs
        if proto is not None and route_signal is not None and self.proto_route_loss_weight > 0.0:
            route_hidden = torch.tanh(self.proto_route_linear_1(route_signal))
            route_logits = self.proto_route_linear_2(route_hidden)
            loss_proto_route, correct_proto_route = self._classification_loss(route_logits, proto)
        else:
            loss_proto_route, correct_proto_route = zero, zero_correct

        total_loss = (
            self.mbm_loss_weight * loss_mlm
            + self.rel_loss_weight * loss_rel
            + self.proto_cls_loss_weight * loss_proto_cls
            + self.proto_route_loss_weight * loss_proto_route
        )

        batch_size = memory_bank.size(0)

        return (
            total_loss,
            loss_mlm,
            loss_rel,
            loss_proto_cls,
            loss_proto_route,
            correct_mlm,
            correct_rel,
            correct_proto_cls,
            correct_proto_route,
            denominator,
            torch.tensor(float(batch_size), device=memory_bank.device),
        )
