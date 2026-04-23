import copy
import torch
from uer.layers import *
from uer.encoders import *
from uer.targets import *
from uer.models.model import Model
# [新增] 引入 MacroMoEEncoder
from uer.macro_moe.encoder import MacroMoEEncoder


def build_model(args):
    """
    Build universial encoder representations models.
    The combinations of different embedding, encoder,
    and target layers yield pretrained models of different
    properties.
    We could select suitable one for downstream tasks.
    """

    embedding = str2embedding[args.embedding](args, len(args.vocab))

    # [新增] 处理 macro_moe 参数
    if args.encoder == "macro_moe":
        encoder = MacroMoEEncoder(args)
        # 检查是否处于小样本适配阶段
        if hasattr(args, "few_shot_stage") and args.few_shot_stage:
            print(f"Loading MacroMoE: Few-shot Adaptation Mode (Backbones Frozen, Adapters Active)")
            encoder.set_adaptation_mode(True)
        else:
            print(f"Loading MacroMoE: Pre-training/Full Fine-tuning Mode (Backbones Active)")
            encoder.set_adaptation_mode(False)
    elif args.encoder == "backbone_only":
        args_copy = copy.deepcopy(args)
        args_copy.encoder = "transformer"
        args_copy.is_moe = False
        print("Loading BackboneOnly: Transformer backbone without router/adapters/experts")
        encoder = str2encoder["transformer"](args_copy)
    else:
        encoder = str2encoder[args.encoder](args)

    target = str2target[args.target](args, len(args.vocab))
    model = Model(args, embedding, encoder, target)

    return model
