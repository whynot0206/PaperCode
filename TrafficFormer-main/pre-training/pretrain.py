import os  # 导入操作系统接口模块
import sys  # 导入系统相关参数和函数模块

sys.path.append(os.getcwd())  # 将当前工作目录添加到系统路径
import argparse  # 导入命令行参数解析模块
import torch  # 导入PyTorch深度学习框架
import uer.trainer as trainer  # 导入UER训练器模块
from uer.utils.config import load_hyperparam  # 从UER导入超参数加载函数
from uer.opts import *  # 从UER导入所有选项


def main():  # 主函数
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)  # 创建参数解析器

    # Path options.
    parser.add_argument("--dataset_path", type=str, default="dataset.pt",  # 添加数据集路径参数
                        help="Path of the preprocessed dataset.")  # 帮助信息：预处理数据集的路径
    parser.add_argument("--vocab_path", default=None, type=str,  # 添加词汇表路径参数
                        help="Path of the vocabulary file.")  # 帮助信息：词汇表文件的路径
    parser.add_argument("--spm_model_path", default=None, type=str,  # 添加句子片段模型路径参数
                        help="Path of the sentence piece model.")  # 帮助信息：句子片段模型的路径
    parser.add_argument("--tgt_vocab_path", default=None, type=str,  # 添加目标词汇表路径参数
                        help="Path of the target vocabulary file.")  # 帮助信息：目标词汇表文件的路径
    parser.add_argument("--tgt_spm_model_path", default=None, type=str,  # 添加目标句子片段模型路径参数
                        help="Path of the target sentence piece model.")  # 帮助信息：目标句子片段模型的路径
    parser.add_argument("--pretrained_model_path", type=str, default=None,  # 添加预训练模型路径参数
                        help="Path of the pretrained model.")  # 帮助信息：预训练模型的路径
    parser.add_argument("--output_model_path", type=str, required=True,  # 添加输出模型路径参数（必需）
                        help="Path of the output model.")  # 帮助信息：输出模型的路径
    parser.add_argument("--config_path", type=str, default="models/bert/base_config.json",
                        # define the model   # 添加配置文件路径参数
                        help="Config file of model hyper-parameters.")  # 帮助信息：模型超参数的配置文件

    # Training and saving options.
    parser.add_argument("--total_steps", type=int, default=100000,  # 添加总训练步数参数
                        help="Total training steps.")  # 帮助信息：总训练步数
    parser.add_argument("--save_checkpoint_steps", type=int, default=10000,  # 添加保存检查点步数参数
                        help="Specific steps to save model checkpoint.")  # 帮助信息：保存模型检查点的特定步数
    parser.add_argument("--report_steps", type=int, default=100,  # 添加报告步数参数
                        help="Specific steps to print prompt.")  # 帮助信息：打印提示的特定步数
    parser.add_argument("--accumulation_steps", type=int, default=1,  # 添加梯度累积步数参数
                        help="Specific steps to accumulate gradient.")  # 帮助信息：累积梯度的特定步数
    parser.add_argument("--batch_size", type=int, default=32,  # 添加批次大小参数
                        help="Training batch size. The actual batch_size is [batch_size x world_size x accumulation_steps].")  # 帮助信息：训练批次大小，实际批次大小为[batch_size x world_size x accumulation_steps]
    parser.add_argument("--instances_buffer_size", type=int, default=25600,  # 添加实例缓冲区大小参数
                        help="The buffer size of instances in memory.")  # 帮助信息：内存中实例的缓冲区大小
    parser.add_argument("--labels_num", type=int, required=False,  # 添加标签数量参数
                        help="Number of prediction labels.")  # 帮助信息：预测标签的数量
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout value.")  # 添加dropout参数
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")  # 添加随机种子参数

    # Preprocess options.
    parser.add_argument("--tokenizer", choices=["bert", "char", "space"], default="bert",  # 添加分词器参数
                        help="Specify the tokenizer."  # 帮助信息：指定分词器
                             "Original Google BERT uses bert tokenizer on Chinese corpus."  # 原始Google BERT在中文语料上使用bert分词器
                             "Char tokenizer segments sentences into characters."  # 字符分词器将句子分割成字符
                             "Space tokenizer segments sentences into words according to space."  # 空格分词器根据空格将句子分割成单词
                        )

    # Model options.
    model_opts(parser)  # 添加模型选项
    parser.add_argument("--tgt_embedding", choices=["word", "word_pos", "word_pos_seg", "word_sinusoidalpos"],
                        default="word_pos_seg",  # 添加目标嵌入参数
                        help="Target embedding type.")  # 帮助信息：目标嵌入类型
    parser.add_argument("--decoder", choices=["transformer"], default="transformer", help="Decoder type.")  # 添加解码器类型参数
    parser.add_argument("--pooling", choices=["mean", "max", "first", "last"], default="first",  # 添加池化方式参数
                        help="Pooling type.")  # 帮助信息：池化类型
    parser.add_argument("--target",
                        choices=["bert", "bertflow", "lm", "mlm", "bilm", "albert", "seq2seq", "t5", "cls", "prefixlm"],
                        default="bert",  # 添加预训练目标参数
                        help="The training target of the pretraining model.")  # 帮助信息：预训练模型的训练目标
    parser.add_argument("--tie_weights", action="store_true",  # 添加权重绑定参数
                        help="Tie the word embedding and softmax weights.")  # 帮助信息：绑定词嵌入和softmax权重
    parser.add_argument("--has_lmtarget_bias", action="store_true",  # 添加语言模型目标偏置参数
                        help="Add bias on output_layer for lm target.")  # 帮助信息：为语言模型目标在输出层添加偏置

    # MOE Model Options
    parser.add_argument("--is_moe", action="store_true", help="adopt moe layer.")  # 添加是否使用MOE层参数
    parser.add_argument("--vocab_size", type=int, required=False, help="Number of vocab.")  # 添加词汇表大小参数
    parser.add_argument("--moebert_expert_dim", type=int, required=False, default=3072,
                        help="Dim of expert,default is ffn.")  # 添加MOE专家维度参数
    parser.add_argument("--moebert_expert_num", type=int, required=False, help="Number of expert.")  # 添加MOE专家数量参数
    parser.add_argument("--moebert_route_method",
                        choices=["gate-token", "feature-gate", "feature-gate-top2", "gate-sentence", "hash-random",
                                 "hash-balance", "proto"],
                        default="hash-random",  # 添加MOE路由方法参数
                        help="moebert route method.")  # 帮助信息：MOE路由方法
    parser.add_argument("--moebert_route_hash_list", default=None, type=str,
                        help="Path of moebert hash list file.")  # 添加MOE哈希列表路径参数
    parser.add_argument("--moebert_load_balance", type=float, default=0.01, help="gate loss weight.")  # 添加MOE负载平衡参数

    # Masking options.
    parser.add_argument("--whole_word_masking", action="store_true", help="Whole word masking.")  # 添加全词掩码参数
    parser.add_argument("--span_masking", action="store_true", help="Span masking.")  # 添加跨度掩码参数
    parser.add_argument("--span_geo_prob", type=float, default=0.2,  # 添加跨度几何概率参数
                        help="Hyperparameter of geometric distribution for span masking.")  # 帮助信息：跨度掩码的几何分布超参数
    parser.add_argument("--span_max_length", type=int, default=10,  # 添加跨度最大长度参数
                        help="Max length for span masking.")  # 帮助信息：跨度掩码的最大长度

    # Optimizer options.
    optimization_opts(parser)  # 添加优化器选项

    # GPU options.
    parser.add_argument("--world_size", type=int, default=1,
                        help="Total number of processes (GPUs) for training.")  # 添加世界大小参数（进程/GPU数量）
    parser.add_argument("--gpu_ranks", default=[], nargs='+', type=int,
                        help="List of ranks of each process."  # 添加GPU排名参数
                             " Each process has a unique integer rank whose value is in the interval [0, world_size), and runs in a single GPU.")  # 帮助信息：每个进程的排名列表，每个进程有一个唯一的整数排名，值在[0, world_size)区间内，并在单个GPU中运行
    parser.add_argument("--master_ip", default="tcp://localhost:12345", type=str,
                        help="IP-Port of master for training.")  # 添加主节点IP端口参数
    parser.add_argument("--backend", choices=["nccl", "gloo"], default="nccl", type=str,
                        help="Distributed backend.")  # 添加分布式后端参数

    args = parser.parse_args()  # 解析参数

    if args.target == "cls":  # 如果目标是分类
        assert args.labels_num is not None, "Cls target needs the denotation of the number of labels."  # 断言标签数量不为空，分类目标需要标签数量的说明

    # Load hyper-parameters from config file.
    if args.config_path:  # 如果有配置文件路径
        load_hyperparam(args)  # 从配置文件加载超参数

    ranks_num = len(args.gpu_ranks)  # 获取GPU排名数量

    if args.world_size > 1:  # 如果世界大小大于1（分布式训练）
        # Multiprocessing distributed mode.
        assert torch.cuda.is_available(), "No available GPUs."  # 断言有可用的GPU
        assert ranks_num <= args.world_size, "Started processes exceed `world_size` upper limit."  # 断言启动的进程不超过世界大小上限
        assert ranks_num <= torch.cuda.device_count(), "Started processes exceeds the available GPUs."  # 断言启动的进程不超过可用GPU数量
        args.dist_train = True  # 设置分布式训练标志
        args.ranks_num = ranks_num  # 设置排名数量
        print("Using distributed mode for training.")  # 打印使用分布式训练模式信息
    elif args.world_size == 1 and ranks_num == 1:  # 如果世界大小为1且排名数量为1（单GPU训练）
        # Single GPU mode.
        assert torch.cuda.is_available(), "No available GPUs."  # 断言有可用的GPU
        args.gpu_id = args.gpu_ranks[0]  # 设置GPU ID
        print("args.gpu_id:", args.gpu_id)  # 打印GPU ID
        print("torch.cuda.device_count,", torch.cuda.device_count())  # 打印GPU设备数量
        assert args.gpu_id < torch.cuda.device_count(), "Invalid specified GPU device."  # 断言指定的GPU设备有效
        args.dist_train = False  # 设置分布式训练标志为False
        args.single_gpu = True  # 设置单GPU标志
        print("Using GPU %d for training." % args.gpu_id)  # 打印使用哪个GPU进行训练
    else:  # 其他情况（CPU训练）
        # CPU mode.
        assert ranks_num == 0, "GPUs are specified, please check the arguments."  # 断言排名数量为0，如果指定了GPU请检查参数
        args.dist_train = False  # 设置分布式训练标志为False
        args.single_gpu = False  # 设置单GPU标志为False
        print("Using CPU mode for training.")  # 打印使用CPU训练模式信息

    trainer.train_and_validate(args)  # 调用训练和验证函数


if __name__ == "__main__":  # 如果是主程序
    main()  # 调用主函数

'''
python3 pre-training/pretrain.py \
        --dataset_path data_generation/data/pretrain_dataset.pt \
        --vocab_path models/encryptd_vocab.txt \
        --output_model_path models/pretrain_model_bert.bin \
        --config_path models/bert/base_config.json \
        --world_size 1 \
        --gpu_ranks 0 \
        --total_steps 90000 \
        --save_checkpoint_steps 10000 \
        --batch_size 64 \
        --embedding word_pos_seg \
        --encoder transformer \
        --mask fully_visible \
        --target bert \
        --learning_rate 1e-4
        
python3 pre-training/pretrain.py \
        --dataset_path data_generation/data/pretrain_dataset.pt \
        --vocab_path models/encryptd_vocab.txt \
        --output_model_path models/pretrain_model_bertflow.bin \
        --config_path models/bert/base_config.json \
        --world_size 1 \
        --gpu_ranks 0 \
        --total_steps 90000 \
        --save_checkpoint_steps 10000 \
        --batch_size 64 \
        --embedding word_pos_seg \
        --encoder transformer \
        --mask fully_visible \
        --target bertflow \
        --learning_rate 1e-4  
'''
'''
python3 pre-training/pretrain.py \
    --dataset_path data_generation/data/pretrain_dataset.pt \
    --vocab_path models/encryptd_vocab.txt \
    --output_model_path models/pretrain_model_macro_moe.bin \
    --config_path models/bert/base_config.json \
    --world_size 1 \
    --gpu_ranks 0 \
    --total_steps 90000 \
    --save_checkpoint_steps 10000 \
    --batch_size 64 \
    --embedding word_pos_seg \
    --encoder macro_moe \
    --macro_expert_num 4 \
    --adapter_size 64 \
    --mask fully_visible \
    --target bertflow \
    --learning_rate 1e-4

第一次参数调整
python3 pre-training/pretrain.py \
    --dataset_path data_generation/data/pretrain_dataset.pt \
    --vocab_path models/encryptd_vocab.txt \
    --output_model_path models/pretrain_model_macro_moe.bin \
    --config_path models/bert/base_config.json \
    --world_size 1 \
    --gpu_ranks 0 \
    --total_steps 90000 \
    --save_checkpoint_steps 10000 \
    --batch_size 32 \
    --embedding word_pos_seg \
    --encoder macro_moe \
    --macro_expert_num 4 \
    --adapter_size 32 \
    --mask fully_visible \
    --target bertflow \
    --learning_rate 6e-5
'''
'''
第二次参数调整
python3 pre-training/pretrain.py \
    --dataset_path data_generation/data/pretrain_dataset.pt \
    --vocab_path models/encryptd_vocab.txt \
    --output_model_path models/pretrain_model_macro_moe.bin \
    --config_path models/bert/base_config.json \
    --world_size 1 \
    --gpu_ranks 0 \
    --total_steps 90000 \
    --report_steps 100 \
    --save_checkpoint_steps 10000 \
    --batch_size 32 \
    --embedding word_pos_seg \
    --encoder macro_moe \
    --macro_expert_num 8 \
    --adapter_size 32 \
    --mask fully_visible \
    --target bertflow \
    --learning_rate 6e-5
'''
'''
第三次参数调整
python3 pre-training/pretrain.py \
    --dataset_path data_generation/data/pretrain_dataset.pt \
    --vocab_path models/encryptd_vocab.txt \
    --output_model_path models/pretrain_model_macro_moe.bin \
    --config_path models/bert/base_config.json \
    --world_size 1 \
    --gpu_ranks 0 \
    --total_steps 90000 \
    --report_steps 100 \
    --save_checkpoint_steps 10000 \
    --batch_size 32 \
    --embedding word_pos_seg \
    --encoder macro_moe \
    --macro_expert_num 4 \
    --adapter_size 32 \
    --mask fully_visible \
    --target bertflow \
    --learning_rate 6e-5 
    
'''
'''
python3 pre-training/pretrain.py \
    --dataset_path data_generation/data/pretrain_dataset.pt \
    --vocab_path models/encryptd_vocab.txt \
    --output_model_path models/pretrain_model_macro_moe.bin \
    --config_path models/bert/base_config.json \
    --world_size 1 \
    --gpu_ranks 0 \
    --total_steps 90000 \
    --report_steps 100 \
    --save_checkpoint_steps 10000 \
    --batch_size 32 \
    --embedding word_pos_seg \
    --encoder macro_moe \
    --macro_expert_num 4 \
    --adapter_size 32 \
    --mask fully_visible \
    --target bertflow \
    --learning_rate 6e-5 \
    --moebert_load_balance 0.1 
'''
'''
python3 -u pre-training/pretrain.py \
    --dataset_path data_generation/data/pretrain_dataset.pt \
    --vocab_path models/encryptd_vocab.txt \
    --output_model_path models/pretrain_model_macro_moe.bin \
    --config_path models/bert/base_config.json \
    --world_size 1 \
    --gpu_ranks 0 \
    --total_steps 90000 \
    --report_steps 100 \
    --save_checkpoint_steps 10000 \
    --batch_size 8 \
    --accumulation_steps 4 \
    --embedding word_pos_seg \
    --encoder macro_moe \
    --macro_expert_num 4 \
    --adapter_size 32 \
    --mask fully_visible \
    --target bertflow \
    --learning_rate 6e-5 \
    --moebert_load_balance 0.1
    expert-1的使用非常少
'''
'''
python3 -u pre-training/pretrain.py \
    --dataset_path data_generation/data/pretrain_dataset.pt \
    --vocab_path models/encryptd_vocab.txt \
    --output_model_path models/pretrain_model_macro_moe.bin \
    --config_path models/bert/base_config.json \
    --world_size 1 \
    --gpu_ranks 0 \
    --total_steps 90000 \
    --report_steps 100 \
    --save_checkpoint_steps 10000 \
    --batch_size 8 \
    --accumulation_steps 4 \
    --embedding word_pos_seg \
    --encoder macro_moe \
    --macro_expert_num 4 \
    --adapter_size 32 \
    --mask fully_visible \
    --target bertflow \
    --learning_rate 6e-5 \
    --moebert_load_balance 1
    这次还是expert-1全是0
'''
'''
python3 -u pre-training/pretrain.py \
    --dataset_path data_generation/data/pretrain_dataset.pt \
    --vocab_path models/encryptd_vocab.txt \
    --output_model_path models/pretrain_model_macro_moe_8e.bin \
    --config_path models/bert/base_config.json \
    --world_size 1 \
    --gpu_ranks 0 \
    --total_steps 90000 \
    --report_steps 100 \
    --save_checkpoint_steps 10000 \
    --batch_size 8 \
    --accumulation_steps 4 \
    --embedding word_pos_seg \
    --encoder macro_moe \
    --macro_expert_num 8 \
    --adapter_size 32 \
    --mask fully_visible \
    --target bertflow \
    --learning_rate 6e-5 \
    --macro_router_noise_std 0.2 \
    --moebert_load_balance 0.2 
用8个专家试一下batch_size = 8 ，这个参数的效果还不错，8个专家都能被分配到嵌入，但是acc很差，所以要再调一下
'''
'''
python3 -u pre-training/pretrain.py \
    --dataset_path data_generation/data/pretrain_dataset.pt \
    --vocab_path models/encryptd_vocab.txt \
    --output_model_path models/pretrain_model_macro_moe_4e_optimized.bin \
    --config_path models/bert/base_config.json \
    --world_size 1 \
    --gpu_ranks 0 \
    --total_steps 90000 \
    --report_steps 100 \
    --save_checkpoint_steps 10000 \
    --batch_size 8 \
    --accumulation_steps 4 \
    --embedding word_pos_seg \
    --encoder macro_moe \
    --macro_expert_num 4 \
    --adapter_size 32 \
    --mask fully_visible \
    --target bertflow \
    --learning_rate 6e-5 \
    --macro_router_noise_std 0.05 \
    --moebert_load_balance 0.1 \
    --macro_router_target_entropy 0.85
这是3.4晚上跑的4专家，为了提高acc
'''
'''
python3 -u pre-training/pretrain.py \
    --dataset_path data_generation/data/pretrain_dataset.pt \
    --vocab_path models/encryptd_vocab.txt \
    --output_model_path models/pretrain_model_macro_moe_8e_optimized.bin \
    --config_path models/bert/base_config.json \
    --world_size 1 \
    --gpu_ranks 0 \
    --total_steps 90000 \
    --report_steps 100 \
    --save_checkpoint_steps 10000 \
    --batch_size 8 \
    --accumulation_steps 4 \
    --embedding word_pos_seg \
    --encoder macro_moe \
    --macro_expert_num 8 \
    --adapter_size 32 \
    --mask fully_visible \
    --target bertflow \
    --learning_rate 6e-5 \
    --macro_router_noise_std 0.05 \
    --moebert_load_balance 0.1 \
    --macro_router_target_entropy 0.88
再次跑8专家测试

这次修改router逻辑，再来跑4个专家先来看看
python3 -u pre-training/pretrain.py \
    --dataset_path data_generation/data/pretrain_dataset.pt \
    --vocab_path models/encryptd_vocab.txt \
    --output_model_path models/pretrain_model_macro_moe_4e_ad64.bin \
    --config_path models/bert/base_config.json \
    --world_size 1 \
    --gpu_ranks 0 \
    --total_steps 90000 \
    --report_steps 100 \
    --save_checkpoint_steps 10000 \
    --batch_size 16 \
    --accumulation_steps 4 \
    --embedding word_pos_seg \
    --encoder macro_moe \
    --macro_expert_num 4 \
    --adapter_size 64 \
    --mask fully_visible \
    --target bertflow \
    --learning_rate 6e-5 \
    --macro_router_noise_std 0.05 \
    --moebert_load_balance 0.1 \
    --macro_router_target_entropy 0.88
'''
'''
python3 -u pre-training/pretrain.py \
    --dataset_path data_generation/data/pretrain_dataset.pt \
    --vocab_path models/encryptd_vocab.txt \
    --output_model_path models/pretrain_model_macro_moe_8e_optimized.bin \
    --config_path models/bert/base_config.json \
    --world_size 1 \
    --gpu_ranks 0 \
    --total_steps 90000 \
    --report_steps 100 \
    --save_checkpoint_steps 10000 \
    --batch_size 16 \
    --accumulation_steps 4 \
    --embedding word_pos_seg \
    --encoder macro_moe \
    --macro_expert_num 8 \
    --adapter_size 32 \
    --mask fully_visible \
    --target bertflow \
    --learning_rate 6e-5 \
    --macro_router_noise_std 0.05 \
    --moebert_load_balance 0.1 \
    --macro_router_target_entropy 0.88
这次直接batch_size 16*4试试
'''
'''
#### **阶段 2：全量微调 (Full Fine-tuning)**
在已知类别的数据集上进行有监督训练，更新所有参数（专家 + 分类头）。
* **改动点**：
    * 输入模型 (`--pretrained_model_path`) 使用阶段 1 训练好的 `pretrain_model_macro_moe.bin`。
    * `--encoder` 设为 `macro_moe`。
    * **不添加** `--few_shot_stage`（骨干参数继续参与微调）。

```bash
CUDA_VISIBLE_DEVICES=2 python3 fine-tuning/run_classifier.py \
    --vocab_path models/encryptd_vocab.txt \
    --train_path train_dataset.tsv \
    --dev_path valid_dataset.tsv \
    --test_path test_dataset.tsv \
    --pretrained_model_path models/pretrain_model_macro_moe.bin \
    --output_model_path models/finetuned_model_stage2.bin \
    --epochs_num 4 \
    --earlystop 4 \
    --batch_size 128 \
    --embedding word_pos_seg \
    --encoder macro_moe \
    --macro_expert_num 4 \
    --adapter_size 32 \
    --mask fully_visible \
    --seq_length 512 \
    --learning_rate 6e-5

修改第一次：
CUDA_VISIBLE_DEVICES=2 python3 fine-tuning/run_classifier.py \
    --vocab_path models/encryptd_vocab.txt \
    --train_path train_dataset.tsv \
    --dev_path valid_dataset.tsv \
    --test_path test_dataset.tsv \
    --pretrained_model_path models/pretrain_model_macro_moe.bin \
    --output_model_path models/finetuned_model_stage2.bin \
    --epochs_num 4 \
    --earlystop 4 \
    --batch_size 8 \
    --embedding word_pos_seg \
    --encoder macro_moe \
    --macro_expert_num 4 \
    --adapter_size 32 \
    --mask fully_visible \
    --seq_length 512 \
    --learning_rate 6e-5 \
    --moebert_load_balance 0.01

python3 fine-tuning/run_classifier.py \
    --vocab_path models/encryptd_vocab.txt \
    --train_path ISCX-VPN_dataset/dataset/train_dataset.tsv \
    --dev_path ISCX-VPN_dataset/dataset/valid_dataset.tsv \
    --test_path ISCX-VPN_dataset/dataset/test_dataset.tsv \
    --pretrained_model_path models/pretrain_model_macro_moe_8e_optimized.bin-90000 \
    --output_model_path models/finetuned_model_stage2_8e.bin \
    --config_path models/bert/base_config.json \
    --epochs_num 20 \
    --earlystop 4 \
    --batch_size 64 \
    --embedding word_pos_seg \
    --encoder macro_moe \
    --macro_expert_num 8 \
    --adapter_size 32 \
    --mask fully_visible \
    --seq_length 512 \
    --learning_rate 1e-4 \
    --few_shot_stage \
    --macro_router_noise_std 0.0

第二次修改
python3 fine-tuning/run_classifier.py \
    --vocab_path models/encryptd_vocab.txt \
    --train_path ISCX-VPN_dataset/dataset/train_dataset.tsv \
    --dev_path ISCX-VPN_dataset/dataset/valid_dataset.tsv \
    --test_path ISCX-VPN_dataset/dataset/test_dataset.tsv \
    --pretrained_model_path models/pretrain_model_macro_moe_8e_optimized.bin-90000 \
    --output_model_path models/finetuned_model_stage2_8e_full.bin \
    --config_path models/bert/base_config.json \
    --epochs_num 15 \
    --earlystop 5 \
    --batch_size 32 \
    --embedding word_pos_seg \
    --encoder macro_moe \
    --macro_expert_num 8 \
    --adapter_size 32 \
    --mask fully_visible \
    --seq_length 512 \
    --learning_rate 3e-5 \
    --pooling mean \
    --macro_router_noise_std 0.0

CUDA_VISIBLE_DEVICES=2 python3 fine-tuning/run_classifier.py \
    --vocab_path models/encryptd_vocab.txt \
    --train_path ISCX-VPN_dataset/dataset/train_dataset.tsv \
    --dev_path ISCX-VPN_dataset/dataset/valid_dataset.tsv \
    --test_path ISCX-VPN_dataset/dataset/test_dataset.tsv \
    --pretrained_model_path models/pretrain_model_macro_moe_8e_optimized.bin-90000 \
    --output_model_path models/finetuned_model_stage2_8e_full.bin \
    --config_path models/bert/base_config.json \
    --epochs_num 15 \
    --earlystop 5 \
    --batch_size 32 \
    --embedding word_pos_seg \
    --encoder macro_moe \
    --macro_expert_num 8 \
    --adapter_size 32 \
    --mask fully_visible \
    --seq_length 512 \
    --learning_rate 3e-5 \
    --pooling first \
    --macro_router_noise_std 0.0

第三次微调CUDA_VISIBLE_DEVICES=2 python3 fine-tuning/run_classifier.py \
    --vocab_path models/encryptd_vocab.txt \
    --train_path ISCX-VPN_dataset/dataset/train_dataset.tsv \
    --dev_path ISCX-VPN_dataset/dataset/valid_dataset.tsv \
    --test_path ISCX-VPN_dataset/dataset/test_dataset.tsv \
    --pretrained_model_path models/pretrain_model_macro_moe_8e_optimized.bin-90000 \
    --output_model_path models/finetuned_model_stage2_8e_full.bin \
    --config_path models/bert/base_config.json \
    --epochs_num 5 \
    --earlystop 4 \
    --batch_size 128 \
    --embedding word_pos_seg \
    --encoder macro_moe \
    --macro_expert_num 8 \
    --adapter_size 32 \
    --mask fully_visible \
    --seq_length 320 \
    --learning_rate 6e-5 \
    --macro_router_noise_std 0.0
    
    第五次
    python3 fine-tuning/run_classifier.py \
    --vocab_path models/encryptd_vocab.txt \
    --train_path ISCX-VPN_dataset/dataset/train_dataset.tsv \
    --dev_path ISCX-VPN_dataset/dataset/valid_dataset.tsv \
    --test_path ISCX-VPN_dataset/dataset/test_dataset.tsv \
    --pretrained_model_path models/pretrain_model_macro_moe_8e_optimized.bin-90000 \
    --output_model_path models/finetuned_model_stage2_full.bin \
    --config_path models/bert/base_config.json \
    --epochs_num 15 \
    --earlystop 5 \
    --batch_size 32 \
    --embedding word_pos_seg \
    --encoder macro_moe \
    --macro_expert_num 8 \
    --adapter_size 32 \
    --mask fully_visible \
    --seq_length 320 \
    --learning_rate 3e-5 \
    --macro_router_noise_std 0.0
    
    这一次能到40%
    python3 fine-tuning/run_classifier.py \
    --vocab_path models/encryptd_vocab.txt \
    --train_path ISCX-VPN_dataset/dataset/train_enhance5_dataset.tsv \
    --dev_path ISCX-VPN_dataset/dataset/valid_dataset.tsv \
    --test_path ISCX-VPN_dataset/dataset/test_dataset.tsv \
    --pretrained_model_path models/pretrain_model_macro_moe_8e_optimized.bin-90000 \
    --output_model_path models/finetuned_model_stage2_full_enhanced.bin \
    --config_path models/bert/base_config.json \
    --epochs_num 10 \
    --earlystop 4 \
    --batch_size 32 \
    --embedding word_pos_seg \
    --encoder macro_moe \
    --macro_expert_num 8 \
    --adapter_size 32 \
    --mask fully_visible \
    --seq_length 320 \
    --learning_rate 5e-5 \
    --macro_router_noise_std 0.0
    
    python3 fine-tuning/run_classifier.py \
    --vocab_path models/encryptd_vocab.txt \
    --train_path ISCX-VPN_dataset/dataset/train_enhance5_dataset.tsv \
    --dev_path ISCX-VPN_dataset/dataset/valid_dataset.tsv \
    --test_path ISCX-VPN_dataset/dataset/test_dataset.tsv \
    --pretrained_model_path models/pretrain_model_macro_moe_8e_optimized.bin-90000 \
    --output_model_path models/finetuned_model_stage2_final.bin \
    --config_path models/bert/base_config.json \
    --epochs_num 8 \
    --earlystop 3 \
    --batch_size 32 \
    --embedding word_pos_seg \
    --encoder macro_moe \
    --macro_expert_num 8 \
    --adapter_size 32 \
    --mask fully_visible \
    --seq_length 320 \
    --learning_rate 2e-5 \
    --macro_router_noise_std 0.05
    
    
    python3 fine-tuning/run_classifier.py \
    --pretrained_model_path models/pretrain_model_macro_moe_4e_ad64.bin-90000 \
    --vocab_path models/encryptd_vocab.txt \
    --train_path ISCX-VPN_dataset/dataset/train_enhance5_dataset.tsv \
    --dev_path ISCX-VPN_dataset/dataset/valid_dataset.tsv \
    --test_path ISCX-VPN_dataset/dataset/test_dataset.tsv \
    --config_path models/bert/base_config.json \
    --output_model_path models/finetuned_full_macro_moe.bin \
    --epochs 10 \
    --batch_size 64 \
    --learning_rate 5e-5 \
    --encoder macro_moe \
    --embedding word_pos_seg \
    --macro_expert_num 4 \
    --adapter_size 64 \
    --macro_router_noise_std 0.02 \
    --macro_router_target_entropy 0.88
    Test set evaluation.
Confusion matrix:
tensor([[62, 20, 26, 13,  3,  2,  0,  4,  0,  0,  0,  0],
        [ 1, 20,  3,  5,  0,  1,  0,  0,  0,  0,  0,  0],
        [11,  1, 72,  0,  2,  1,  1,  1,  1,  0,  0,  0],
        [ 1,  0,  0, 29,  0,  1,  0,  0,  0,  0,  0,  0],
        [ 8,  0,  3,  0, 30,  0,  0,  0,  0,  0,  0,  0],
        [ 0,  1,  0,  0,  0, 12,  0,  0,  0,  0,  0,  0],
        [ 1,  0,  2,  0,  0,  0, 35,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  0,  6,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  1,  0, 30,  0,  0,  0],
        [ 2,  0,  0,  0,  1,  0,  0,  3,  0, 14,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  0,  1,  1,  0, 17,  0],
        [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  6]])
        Report precision, recall, and f1:
        Label 0: 0.477, 0.721, 0.574
        Label 1: 0.667, 0.476, 0.556
        Label 2: 0.800, 0.679, 0.735
        Label 3: 0.935, 0.617, 0.744
        Label 4: 0.732, 0.833, 0.779
        Label 5: 0.923, 0.706, 0.800
        Label 6: 0.921, 0.946, 0.933
        Label 7: 1.000, 0.400, 0.571
        Label 8: 0.968, 0.937, 0.952
        Label 9: 0.700, 1.000, 0.824
        Label 10: 0.895, 1.000, 0.944
        Label 11: 1.000, 1.000, 1.000
        Acc. (Correct/Total): 0.7319 (333/455) 
        Macro precision: 0.8348, Micro precision: 0.7319, Weighted precision: 0.7712
        Macro recall: 0.7763, Micro recall: 0.7319, Weighted recall: 0.7319
        Macro f1: 0.7844, Micro f1: 0.7319, Weighted f1: 0.7348


在此次调一下参数：
python3 fine-tuning/run_classifier.py \
    --pretrained_model_path models/pretrain_model_macro_moe_4e_ad64.bin-90000 \
    --vocab_path models/encryptd_vocab.txt \
    --train_path ISCX-VPN_dataset/dataset/train_enhance5_dataset.tsv \
    --dev_path ISCX-VPN_dataset/dataset/valid_dataset.tsv \
    --test_path ISCX-VPN_dataset/dataset/test_dataset.tsv \
    --config_path models/bert/base_config.json \
    --output_model_path models/finetuned_full_macro_moe_2.bin \
    --epochs 20 \
    --earlystop 15 \
    --batch_size 32 \
    --learning_rate 2e-5 \
    --encoder macro_moe \
    --embedding word_pos_seg \
    --macro_expert_num 4 \
    --adapter_size 64 \
    --macro_router_noise_std 0.02 \
    --macro_router_target_entropy 0.88

Test set evaluation.
Confusion matrix:
tensor([[60, 14, 25,  4,  4,  1,  0,  3,  0,  0,  0,  0],
        [ 2, 20,  4,  4,  0,  3,  0,  0,  0,  0,  0,  0],
        [ 7,  0, 70,  1,  1,  1,  0,  1,  1,  0,  0,  0],
        [ 3,  8,  3, 38,  1,  1,  0,  1,  1,  0,  0,  0],
        [12,  0,  2,  0, 29,  0,  0,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0, 11,  0,  0,  0,  0,  0,  0],
        [ 2,  0,  1,  0,  1,  0, 36,  0,  1,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  0,  6,  1,  2,  0,  0],
        [ 0,  0,  1,  0,  0,  0,  1,  0, 27,  0,  2,  0],
        [ 0,  0,  0,  0,  0,  0,  0,  3,  0, 12,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  0,  1,  1,  0, 15,  0],
        [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  6]])
Report precision, recall, and f1:
    Label 0: 0.541, 0.698, 0.609
    Label 1: 0.606, 0.476, 0.533
    Label 2: 0.854, 0.660, 0.745
    Label 3: 0.679, 0.809, 0.738
    Label 4: 0.674, 0.806, 0.734
    Label 5: 1.000, 0.647, 0.786
    Label 6: 0.878, 0.973, 0.923
    Label 7: 0.667, 0.400, 0.500
    Label 8: 0.871, 0.844, 0.857
    Label 9: 0.800, 0.857, 0.828
    Label 10: 0.882, 0.882, 0.882
    Label 11: 1.000, 1.000, 1.000
    Acc. (Correct/Total): 0.7253 (330/455) 
    Macro precision: 0.7876, Micro precision: 0.7253, Weighted precision: 0.7432
    Macro recall: 0.7543, Micro recall: 0.7253, Weighted recall: 0.7253
    Macro f1: 0.7613, Micro f1: 0.7253, Weighted f1: 0.7250

python3 fine-tuning/run_classifier.py \
    --pretrained_model_path models/pretrain_model_macro_moe_4e_ad64.bin-90000 \
    --vocab_path models/encryptd_vocab.txt \
    --train_path ISCX-VPN_dataset/dataset/train_enhance5_dataset.tsv \
    --dev_path ISCX-VPN_dataset/dataset/valid_dataset.tsv \
    --test_path ISCX-VPN_dataset/dataset/test_dataset.tsv \
    --config_path models/bert/base_config.json \
    --output_model_path models/finetuned_full_macro_moe_3.bin \
    --epochs 10 \
    --batch_size 64 \
    --learning_rate 2e-5 \
    --encoder macro_moe \
    --embedding word_pos_seg \
    --macro_expert_num 4 \
    --adapter_size 64 \
    --macro_router_noise_std 0.02 \
    --macro_router_target_entropy 0.88

Test set evaluation.
Confusion matrix:
tensor([[64, 14, 26,  8,  3,  1,  1,  2,  0,  0,  0,  0],
        [ 2, 25,  2,  3,  0,  2,  0,  1,  0,  0,  0,  0],
        [10,  1, 72,  0,  1,  1,  2,  2,  1,  0,  0,  0],
        [ 1,  2,  2, 34,  0,  1,  0,  0,  0,  0,  0,  0],
        [ 8,  0,  2,  0, 31,  0,  0,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  2,  0, 12,  0,  0,  0,  0,  0,  0],
        [ 1,  0,  1,  0,  0,  0, 34,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  0,  5,  1,  2,  0,  0],
        [ 0,  0,  1,  0,  0,  0,  0,  0, 28,  0,  0,  0],
        [ 0,  0,  0,  0,  1,  0,  0,  4,  1, 12,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  0,  1,  1,  0, 17,  0],
        [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  6]])
    Report precision, recall, and f1:
    Label 0: 0.538, 0.744, 0.624
    Label 1: 0.714, 0.595, 0.649
    Label 2: 0.800, 0.679, 0.735
    Label 3: 0.850, 0.723, 0.782
    Label 4: 0.756, 0.861, 0.805
    Label 5: 0.857, 0.706, 0.774
    Label 6: 0.944, 0.919, 0.932
    Label 7: 0.625, 0.333, 0.435
    Label 8: 0.966, 0.875, 0.918
    Label 9: 0.667, 0.857, 0.750
    Label 10: 0.895, 1.000, 0.944
    Label 11: 1.000, 1.000, 1.000
    Acc. (Correct/Total): 0.7473 (340/455) 
    Macro precision: 0.8010, Micro precision: 0.7473, Weighted precision: 0.7661
    Macro recall: 0.7745, Micro recall: 0.7473, Weighted recall: 0.7473
    Macro f1: 0.7790, Micro f1: 0.7473, Weighted f1: 0.7487

python3 fine-tuning/run_classifier.py \
    --pretrained_model_path models/pretrain_model_macro_moe_4e_ad64.bin-90000 \
    --vocab_path models/encryptd_vocab.txt \
    --train_path ISCX-VPN_dataset/dataset/train_enhance5_dataset.tsv \
    --dev_path ISCX-VPN_dataset/dataset/valid_dataset.tsv \
    --test_path ISCX-VPN_dataset/dataset/test_dataset.tsv \
    --config_path models/bert/base_config.json \
    --output_model_path models/finetuned_full_macro_moe_4.bin \
    --epochs 12 \
    --batch_size 64 \
    --learning_rate 2e-5 \
    --encoder macro_moe \
    --embedding word_pos_seg \
    --macro_expert_num 4 \
    --adapter_size 64 \
    --macro_router_noise_std 0.0 \
    --macro_router_target_entropy 0.3

Test set evaluation.
Confusion matrix:
tensor([[60, 11, 23,  3,  5,  2,  1,  2,  0,  0,  0,  0],
        [ 3, 25,  3,  7,  0,  2,  0,  1,  0,  0,  0,  0],
        [11,  0, 75,  1,  4,  3,  2,  1,  1,  0,  0,  0],
        [ 1,  6,  2, 36,  0,  0,  0,  0,  0,  0,  0,  0],
        [ 6,  0,  0,  0, 25,  0,  0,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0, 10,  0,  0,  0,  0,  0,  0],
        [ 2,  0,  3,  0,  1,  0, 33,  0,  1,  0,  0,  0],
        [ 2,  0,  0,  0,  0,  0,  0,  5,  1,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  1,  1, 28,  0,  2,  0],
        [ 1,  0,  0,  0,  1,  0,  0,  4,  0, 14,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  0,  1,  1,  0, 15,  0],
        [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  6]])
Report precision, recall, and f1:
    Label 0: 0.561, 0.698, 0.622
    Label 1: 0.610, 0.595, 0.602
    Label 2: 0.765, 0.708, 0.735
    Label 3: 0.800, 0.766, 0.783
    Label 4: 0.806, 0.694, 0.746
    Label 5: 1.000, 0.588, 0.741
    Label 6: 0.825, 0.892, 0.857
    Label 7: 0.625, 0.333, 0.435
    Label 8: 0.875, 0.875, 0.875
    Label 9: 0.700, 1.000, 0.824
    Label 10: 0.882, 0.882, 0.882
    Label 11: 1.000, 1.000, 1.000
Acc. (Correct/Total): 0.7297 (332/455) 
Macro precision: 0.7875, Micro precision: 0.7297, Weighted precision: 0.7413
Macro recall: 0.7526, Micro recall: 0.7297, Weighted recall: 0.7297
Macro f1: 0.7585, Micro f1: 0.7297, Weighted f1: 0.7291

调整了一下数据集,原来把包含no-vpn的数据集加进去了，实际上实验的数据集并没有使用no-vpn，而是使用vpn数据集
python3 fine-tuning/run_classifier.py \
    --pretrained_model_path models/pretrain_model_macro_moe_4e_ad64.bin-90000 \
    --vocab_path models/encryptd_vocab.txt \
    --train_path ISCX-VPN_dataset/dataset/train_enhance5_dataset.tsv \
    --dev_path ISCX-VPN_dataset/dataset/valid_dataset.tsv \
    --test_path ISCX-VPN_dataset/dataset/test_dataset.tsv \
    --config_path models/bert/base_config.json \
    --output_model_path models/finetuned_full_macro_moe_5.bin \
    --epochs 10 \
    --batch_size 64 \
    --learning_rate 2e-5 \
    --encoder macro_moe \
    --embedding word_pos_seg \
    --macro_expert_num 4 \
    --adapter_size 64 \
    --macro_router_noise_std 0.02 \
    --macro_router_target_entropy 0.88
Test set evaluation.
Confusion matrix:
tensor([[33,  0,  1,  0,  1,  0],
        [ 1, 11,  0,  0,  0,  0],
        [ 3,  2, 29,  0,  1,  2],
        [ 0,  1,  1, 14,  0,  0],
        [ 0,  1,  1,  0, 15,  0],
        [ 0,  0,  0,  0,  0,  4]])
Report precision, recall, and f1:
Label 0: 0.943, 0.892, 0.917
Label 1: 0.917, 0.733, 0.815
Label 2: 0.784, 0.906, 0.841
Label 3: 0.875, 1.000, 0.933
Label 4: 0.882, 0.882, 0.882
Label 5: 1.000, 0.667, 0.800
Acc. (Correct/Total): 0.8760 (106/121) 
Macro precision: 0.9001, Micro precision: 0.8760, Weighted precision: 0.8840
Macro recall: 0.8467, Micro recall: 0.8760, Weighted recall: 0.8760
Macro f1: 0.8646, Micro f1: 0.8760, Weighted f1: 0.8752

再次调整，先用不增强的数据集试试
python3 fine-tuning/run_classifier.py \
    --pretrained_model_path models/pretrain_model_macro_moe_4e_ad64.bin-90000 \
    --vocab_path models/encryptd_vocab.txt \
    --train_path ISCX-VPN_dataset/dataset/train_dataset.tsv \
    --dev_path ISCX-VPN_dataset/dataset/valid_dataset.tsv \
    --test_path ISCX-VPN_dataset/dataset/test_dataset.tsv \
    --config_path models/bert/base_config.json \
    --output_model_path models/finetuned_full_macro_moe_6.bin \
    --epochs 5 \
    --batch_size 16 \
    --learning_rate 1e-5 \
    --encoder macro_moe \
    --embedding word_pos_seg \
    --macro_expert_num 4 \
    --adapter_size 64 \
    --macro_router_noise_std 0.0 \
    --macro_router_target_entropy 0.3 

Test set evaluation.
Confusion matrix:
tensor([[35,  0,  1,  0,  1,  1],
        [ 2, 14,  2,  0,  1,  0],
        [ 0,  0, 27,  0,  0,  0],
        [ 0,  0,  1, 14,  0,  0],
        [ 0,  1,  1,  0, 15,  0],
        [ 0,  0,  0,  0,  0,  5]])
Report precision, recall, and f1:
Label 0: 0.921, 0.946, 0.933
Label 1: 0.737, 0.933, 0.824
Label 2: 1.000, 0.844, 0.915
Label 3: 0.933, 1.000, 0.966
Label 4: 0.882, 0.882, 0.882
Label 5: 1.000, 0.833, 0.909
Acc. (Correct/Total): 0.9091 (110/121) 
Macro precision: 0.9123, Micro precision: 0.9091, Weighted precision: 0.9190
Macro recall: 0.9065, Micro recall: 0.9091, Weighted recall: 0.9091
Macro f1: 0.9048, Micro f1: 0.9091, Weighted f1: 0.9103

用作者的参数试一下
python3 fine-tuning/run_classifier.py \
    --pretrained_model_path models/pretrain_model_macro_moe_4e_ad64.bin-90000 \
    --vocab_path models/encryptd_vocab.txt \
    --train_path ISCX-VPN_dataset/dataset/train_dataset.tsv \
    --dev_path ISCX-VPN_dataset/dataset/valid_dataset.tsv \
    --test_path ISCX-VPN_dataset/dataset/test_dataset.tsv \
    --config_path models/bert/base_config.json \
    --output_model_path models/finetuned_full_macro_moe_7.bin \
    --epochs 20 \
    --batch_size 64 \
    --seq_length 320 \
    --learning_rate 2e-5 \
    --encoder macro_moe \
    --embedding word_pos_seg \
    --macro_expert_num 4 \
    --adapter_size 64 \
    --macro_router_noise_std 0.0 \
    --macro_router_target_entropy 0.3
Test set evaluation.
Confusion matrix:
tensor([[35,  1,  0,  0,  1,  1],
        [ 2, 11,  0,  0,  0,  0],
        [ 0,  0, 31,  0,  0,  0],
        [ 0,  1,  0, 14,  0,  0],
        [ 0,  2,  1,  0, 16,  0],
        [ 0,  0,  0,  0,  0,  5]])
Report precision, recall, and f1:
Label 0: 0.921, 0.946, 0.933
Label 1: 0.846, 0.733, 0.786
Label 2: 1.000, 0.969, 0.984
Label 3: 0.933, 1.000, 0.966
Label 4: 0.842, 0.941, 0.889
Label 5: 1.000, 0.833, 0.909
Acc. (Correct/Total): 0.9256 (112/121) 
Macro precision: 0.9238, Micro precision: 0.9256, Weighted precision: 0.9269
Macro recall: 0.9038, Micro recall: 0.9256, Weighted recall: 0.9256
Macro f1: 0.9111, Micro f1: 0.9256, Weighted f1: 0.9247

python3 fine-tuning/run_classifier.py \
    --pretrained_model_path models/pretrain_model_macro_moe_4e_ad64.bin-90000 \
    --vocab_path models/encryptd_vocab.txt \
    --train_path ISCX-VPN_dataset/dataset/train_dataset.tsv \
    --dev_path ISCX-VPN_dataset/dataset/valid_dataset.tsv \
    --test_path ISCX-VPN_dataset/dataset/test_dataset.tsv \
    --config_path models/bert/base_config.json \
    --output_model_path models/finetuned_full_macro_moe_8.bin \
    --epochs 20 \
    --batch_size 64 \
    --seq_length 320 \
    --learning_rate 3e-5 \
    --encoder macro_moe \
    --embedding word_pos_seg \
    --macro_expert_num 4 \
    --adapter_size 64 \
    --macro_router_noise_std 0.0 \
    --macro_router_target_entropy 0.3
    
Test set evaluation.
Confusion matrix:
tensor([[34,  1,  1,  0,  1,  0],
        [ 1, 12,  3,  0,  0,  0],
        [ 1,  0, 27,  0,  0,  1],
        [ 0,  1,  0, 14,  0,  0],
        [ 1,  1,  1,  0, 16,  0],
        [ 0,  0,  0,  0,  0,  5]])
Report precision, recall, and f1:
Label 0: 0.919, 0.919, 0.919
Label 1: 0.750, 0.800, 0.774
Label 2: 0.931, 0.844, 0.885
Label 3: 0.933, 1.000, 0.966
Label 4: 0.842, 0.941, 0.889
Label 5: 1.000, 0.833, 0.909
Acc. (Correct/Total): 0.8926 (108/121) 
Macro precision: 0.8959, Micro precision: 0.8926, Weighted precision: 0.8961
Macro recall: 0.8895, Micro recall: 0.8926, Weighted recall: 0.8926
Macro f1: 0.8903, Micro f1: 0.8926, Weighted f1: 0.8928

python3 fine-tuning/run_classifier.py \
    --pretrained_model_path models/pretrain_model_macro_moe_4e_ad64.bin-90000 \
    --vocab_path models/encryptd_vocab.txt \
    --train_path ISCX-VPN_dataset/dataset/train_dataset.tsv \
    --dev_path ISCX-VPN_dataset/dataset/valid_dataset.tsv \
    --test_path ISCX-VPN_dataset/dataset/test_dataset.tsv \
    --config_path models/bert/base_config.json \
    --output_model_path models/finetuned_full_macro_moe_7.bin \
    --epochs 25 \
    --batch_size 64 \
    --seq_length 320 \
    --learning_rate 3e-5 \
    --encoder macro_moe \
    --embedding word_pos_seg \
    --macro_expert_num 4 \
    --adapter_size 64 \
    --macro_router_noise_std 0.0 \
    --macro_router_target_entropy 0.3
    
Test set evaluation.
Confusion matrix:
tensor([[34,  1,  1,  0,  1,  0],
        [ 2, 12,  0,  0,  1,  0],
        [ 1,  0, 31,  0,  2,  1],
        [ 0,  1,  0, 14,  0,  0],
        [ 0,  1,  0,  0, 13,  0],
        [ 0,  0,  0,  0,  0,  5]])
Report precision, recall, and f1:
Label 0: 0.919, 0.919, 0.919
Label 1: 0.800, 0.800, 0.800
Label 2: 0.886, 0.969, 0.925
Label 3: 0.933, 1.000, 0.966
Label 4: 0.929, 0.765, 0.839
Label 5: 1.000, 0.833, 0.909
Acc. (Correct/Total): 0.9008 (109/121) 
Macro precision: 0.9111, Micro precision: 0.9008, Weighted precision: 0.9024
Macro recall: 0.8810, Micro recall: 0.9008, Weighted recall: 0.9008
Macro f1: 0.8929, Micro f1: 0.9008, Weighted f1: 0.8995

出现问题，全都给expert1进行处理了，现在调一下moe的参数
python3 fine-tuning/run_classifier.py \
    --pretrained_model_path models/pretrain_model_macro_moe_4e_ad64.bin-90000 \
    --vocab_path models/encryptd_vocab.txt \
    --train_path ISCX-VPN_dataset/dataset/train_dataset.tsv \
    --dev_path ISCX-VPN_dataset/dataset/valid_dataset.tsv \
    --test_path ISCX-VPN_dataset/dataset/test_dataset.tsv \
    --config_path models/bert/base_config.json \
    --output_model_path models/finetuned_full_macro_moe_9.bin \
    --epochs 20 \
    --batch_size 64 \
    --seq_length 320 \
    --learning_rate 2e-5 \
    --encoder macro_moe \
    --embedding word_pos_seg \
    --macro_expert_num 4 \
    --adapter_size 64 \
    --macro_router_noise_std 0.05 \
    --moebert_load_balance 0.1 \
    --macro_router_target_entropy 0.88 

Test set evaluation.
Confusion matrix:
tensor([[34,  0,  0,  0,  1,  0],
        [ 1, 13,  0,  0,  0,  0],
        [ 1,  0, 30,  0,  1,  1],
        [ 0,  1,  0, 14,  0,  0],
        [ 1,  1,  2,  0, 15,  0],
        [ 0,  0,  0,  0,  0,  5]])
Report precision, recall, and f1:
Label 0: 0.971, 0.919, 0.944
Label 1: 0.929, 0.867, 0.897
Label 2: 0.909, 0.937, 0.923
Label 3: 0.933, 1.000, 0.966
Label 4: 0.789, 0.882, 0.833
Label 5: 1.000, 0.833, 0.909
路由数据已保存：routing_analysis.json!
Acc. (Correct/Total): 0.9174 (111/121)
Macro precision: 0.9220, Micro precision: 0.9174, Weighted precision: 0.9211
Macro recall: 0.9065, Micro recall: 0.9174, Weighted recall: 0.9174
Macro f1: 0.9120, Micro f1: 0.9174, Weighted f1: 0.9179

专家还是不行，全都给expert1了，所以再次调整参数：
python3 fine-tuning/run_classifier.py \
    --pretrained_model_path models/pretrain_model_macro_moe_4e_ad64.bin-90000 \
    --vocab_path models/encryptd_vocab.txt \
    --train_path ISCX-VPN_dataset/dataset/train_dataset.tsv \
    --dev_path ISCX-VPN_dataset/dataset/valid_dataset.tsv \
    --test_path ISCX-VPN_dataset/dataset/test_dataset.tsv \
    --config_path models/bert/base_config.json \
    --output_model_path models/finetuned_full_macro_moe_10.bin \
    --epochs 20 \
    --batch_size 32 \
    --seq_length 320 \
    --learning_rate 3e-5 \
    --encoder macro_moe \
    --macro_expert_num 4 \
    --adapter_size 64 \
    --macro_router_noise_std 0.2 \
    --moebert_load_balance 2.0 \
    --macro_router_target_entropy 1.0
    
    
#### **阶段 3：小样本适配 (Few-shot Adaptation)**
在新场景的小样本数据集上训练，**冻结**专家骨干，仅更新适配器和分类头。
* **改动点**：
    * 输入模型 (`--pretrained_model_path`) 使用阶段 2 训练好的 `finetuned_model_stage2.bin`。
    * 使用新场景的数据集（例如 `few_shot_train.tsv`）。
    * **添加** `--few_shot_stage`：这是一个开关，加上它就会触发“冻结骨干、训练适配器”的模式。

```bash
CUDA_VISIBLE_DEVICES=2 python3 fine-tuning/run_classifier.py \
    --vocab_path models/encryptd_vocab.txt \
    --train_path few_shot_train.tsv \
    --dev_path few_shot_valid.tsv \
    --test_path few_shot_test.tsv \
    --pretrained_model_path models/finetuned_model_stage2.bin \
    --output_model_path models/few_shot_adapted_model.bin \
    --epochs_num 10 \
    --batch_size 32 \
    --embedding word_pos_seg \
    --encoder macro_moe \
    --macro_expert_num 4 \
    --adapter_size 64 \
    --mask fully_visible \
    --seq_length 512 \
    --learning_rate 1e-4 \
    --few_shot_stage
'''
'''

python3 fine-tuning/run_classifier.py \
    --pretrained_model_path models/pretrain_model_macro_moe_4e_ad64.bin-90000 \
    --vocab_path models/encryptd_vocab.txt \
    --train_path ISCX-VPN_dataset/dataset/train_dataset.tsv \
    --dev_path ISCX-VPN_dataset/dataset/valid_dataset.tsv \
    --test_path ISCX-VPN_dataset/dataset/test_dataset.tsv \
    --config_path models/bert/base_config.json \
    --output_model_path models/finetuned_full_macro_moe_test.bin \
    --epochs 20 \
    --batch_size 64 \
    --seq_length 320 \
    --learning_rate 2e-5 \
    --encoder macro_moe \
    --embedding word_pos_seg \
    --macro_expert_num 4 \
    --adapter_size 64 \
    --macro_router_noise_std 0.0 \
    --macro_router_target_entropy 0.3 \
    --moebert_load_balance 0.1
'''