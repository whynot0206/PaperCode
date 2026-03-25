def model_opts(parser):
    parser.add_argument("--embedding", choices=["word", "word_pos", "word_pos_seg", "word_sinusoidalpos"],
                        default="word_pos_seg",
                        help="Emebdding type.")
    parser.add_argument("--max_seq_length", type=int, default=512,
                        help="Max sequence length for word embedding.")
    parser.add_argument("--relative_position_embedding", action="store_true",
                        help="Use relative position embedding.")
    parser.add_argument("--relative_attention_buckets_num", type=int, default=32,
                        help="Buckets num of relative position embedding.")
    parser.add_argument("--remove_embedding_layernorm", action="store_true",
                        help="Remove layernorm on embedding.")
    parser.add_argument("--remove_attention_scale", action="store_true",
                        help="Remove attention scale.")
    # [修改] 添加 macro_moe 选项
    parser.add_argument("--encoder", choices=["transformer", "rnn", "lstm", "gru",
                                              "birnn", "bilstm", "bigru",
                                              "gatedcnn", "macro_moe"],
                        default="transformer", help="Encoder type.")
    parser.add_argument("--mask", choices=["fully_visible", "causal", "causal_with_prefix"], default="fully_visible",
                        help="Mask type.")
    parser.add_argument("--layernorm_positioning", choices=["pre", "post"], default="post",
                        help="Layernorm positioning.")
    parser.add_argument("--feed_forward", choices=["dense", "gated"], default="dense",
                        help="Feed forward type, specific to transformer model.")
    parser.add_argument("--remove_transformer_bias", action="store_true",
                        help="Remove bias on transformer layers.")
    parser.add_argument("--layernorm", choices=["normal", "t5"], default="normal",
                        help="Layernorm type.")
    parser.add_argument("--bidirectional", action="store_true", help="Specific to recurrent model.")
    parser.add_argument("--factorized_embedding_parameterization", action="store_true",
                        help="Factorized embedding parameterization.")
    parser.add_argument("--parameter_sharing", action="store_true", help="Parameter sharing.")

    # [新增] Macro MoE 核心参数
    parser.add_argument("--macro_expert_num", type=int, default=4,
                        help="Number of macro experts in Macro MoE.")
    parser.add_argument("--adapter_size", type=int, default=32,
                        help="Hidden size of the few-shot adapter.")
    parser.add_argument("--few_shot_stage", action='store_true',
                        help="If true, freeze backbone and only train adapters.")
    parser.add_argument("--macro_router_noise_std", type=float, default=0.01,
                        help="Std of Gaussian noise added to Macro MoE router logits during training.")
    parser.add_argument("--macro_router_balance_weight", type=float, default=0.2,
                        help="Weight of uniform load-balancing term inside Macro MoE router aux loss.")
    parser.add_argument("--macro_router_entropy_weight", type=float, default=1.0,
                        help="Weight of entropy-target term inside Macro MoE router aux loss.")
    parser.add_argument("--macro_router_target_entropy", type=float, default=0.6,
                        help="Target normalized routing entropy in [0,1]. Lower means more specialization.")
    parser.add_argument("--macro_router_rank1_weight", type=float, default=0.0,
                        help="Extra anti-collapse weight for rank-1 routing. Default 0 disables it.")
    parser.add_argument("--macro_router_rank2_weight", type=float, default=0.0,
                        help="Extra anti-collapse weight for rank-2 routing when top-k > 1. Default 0 disables it.")
    parser.add_argument("--macro_router_rank_target_entropy", type=float, default=0.45,
                        help="Target normalized entropy for rank-specific anti-collapse losses.")
    parser.add_argument("--macro_load_balance", type=float, default=0.1,
                        help="Global weight of the gate loss in total loss (replacing moebert_load_balance).")
    parser.add_argument("--macro_top_k", type=int, default=1,
                        help="Top-k experts selected by Macro MoE router. Use 2 to reduce expert collapse.")
    parser.add_argument("--macro_checkpoint_experts", action='store_true',
                        help="Enable gradient checkpointing on Macro-MoE experts to reduce GPU memory usage.")
    parser.add_argument("--macro_shared_backbone", action='store_true',
                        help="Use one shared TrafficFormer backbone and route lightweight adapter experts on top of it.")

def optimization_opts(parser):
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate.")
    parser.add_argument("--warmup", type=float, default=0.1,
                        help="Warm up value.")
    parser.add_argument("--fp16", action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit.")
    parser.add_argument("--fp16_opt_level", choices=["O0", "O1", "O2", "O3"], default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--optimizer", choices=["adamw", "adafactor"],
                        default="adamw",
                        help="Optimizer type.")
    parser.add_argument("--scheduler", choices=["linear", "cosine", "cosine_with_restarts", "polynomial",
                                                "constant", "constant_with_warmup"],
                        default="linear", help="Scheduler type.")


def training_opts(parser):
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size.")
    parser.add_argument("--seq_length", type=int, default=128,
                        help="Sequence length.")
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="Dropout.")
    parser.add_argument("--epochs_num", type=int, default=3,
                        help="Number of epochs.")
    parser.add_argument("--report_steps", type=int, default=100,
                        help="Specific steps to print prompt.")
    parser.add_argument("--seed", type=int, default=7,
                        help="Random seed.")


def finetune_opts(parser):
    # Path options.
    parser.add_argument("--pretrained_model_path", default=None, type=str,
                        help="Path of the pretrained model.")
    parser.add_argument("--output_model_path", default="models/finetuned_model.bin", type=str,
                        help="Path of the output model.")
    parser.add_argument("--vocab_path", default=None, type=str,
                        help="Path of the vocabulary file.")
    parser.add_argument("--spm_model_path", default=None, type=str,
                        help="Path of the sentence piece model.")
    parser.add_argument("--train_path", type=str,
                        help="Path of the trainset.")
    parser.add_argument("--dev_path", type=str,
                        help="Path of the devset.")
    parser.add_argument("--test_path", default=None, type=str,
                        help="Path of the testset.")
    parser.add_argument("--config_path", default="models/bert/base_config.json", type=str,
                        help="Path of the config file.")

    # Model options.
    model_opts(parser)

    # Optimization options.
    optimization_opts(parser)

    # Training options.
    training_opts(parser)


def infer_opts(parser):
    # Path options.
    parser.add_argument("--load_model_path", default=None, type=str,
                        help="Path of the input model.")
    parser.add_argument("--vocab_path", default=None, type=str,
                        help="Path of the vocabulary file.")
    parser.add_argument("--spm_model_path", default=None, type=str,
                        help="Path of the sentence piece model.")
    parser.add_argument("--test_path", type=str, required=True,
                        help="Path of the testset.")
    parser.add_argument("--prediction_path", type=str, required=True,
                        help="Path of the prediction file.")
    parser.add_argument("--config_path", default="models/bert/base_config.json", type=str,
                        help="Path of the config file.")

    # Model options.
    model_opts(parser)

    # Inference options.
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size.")
    parser.add_argument("--seq_length", type=int, default=128,
                        help="Sequence length.")
