import torch
from uer.layers import *
from uer.encoders import *
from uer.targets import *
from uer.models.model import Model
# [New Code] Import Macro MoE Encoder
from uer.macro_moe.encoder import MacroMoEEncoder


def build_model(args):
    """
    Build universial encoder representations models.
    """
    embedding = str2embedding[args.embedding](args, len(args.vocab))

    # [New Code] Check for Macro MoE Encoder
    if args.encoder == "macro_moe":
        encoder = MacroMoEEncoder(args)

        # Check if we are in few-shot adaptation stage
        if hasattr(args, "few_shot_stage") and args.few_shot_stage:
            print("Mode: Few-shot Adaptation (Freezing Backbones...)")
            encoder.set_adaptation_mode(True)
        else:
            print("Mode: Pre-training or Full Fine-tuning (Backbones Active)")
            encoder.set_adaptation_mode(False)

    else:
        encoder = str2encoder[args.encoder](args)

    target = str2target[args.target](args, len(args.vocab))
    model = Model(args, embedding, encoder, target)

    return model