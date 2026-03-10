import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Pretraining models consist of three parts:
        - embedding
        - encoder
        - target
    """

    def __init__(self, args, embedding, encoder, target):
        super(Model, self).__init__()
        self.embedding = embedding
        self.encoder = encoder
        self.target = target

        if args.target in ['bert', 'bertflow', 'mlm'] and args.tie_weights:
            self.target.mlm_linear_2.weight = self.embedding.word_embedding.weight
        elif args.target in ['lm', 't5'] and args.tie_weights:
            self.target.output_layer.weight = self.embedding.word_embedding.weight

        if args.target == 't5' and args.share_embedding:
            self.target.embedding.word_embedding.weight = self.embedding.word_embedding.weight

        self.is_moe = getattr(args, "is_moe", False)

    def forward(self, src, tgt, seg, proto=None):
        emb = self.embedding(src, seg)

        # [修改] 只有 MacroMoEEncoder 会返回 tuple (output, loss)
        if "MacroMoEEncoder" in self.encoder.__class__.__name__:
            # 加上 _, 接收多出来的 expert_indices
            output, gate_loss, _ = self.encoder(emb, seg, input_ids=src, proto=proto)
        elif self.is_moe:
            output, gate_loss = self.encoder(emb, seg, src, proto)
        else:
            output = self.encoder(emb, seg)
            gate_loss = 0.0

        # 计算任务损失
        loss_info = self.target(output, tgt)

        # [修改] 将 gate_loss 附加到返回的元组末尾
        if isinstance(loss_info, tuple):
            return loss_info + (gate_loss,)
        else:
            return loss_info, gate_loss