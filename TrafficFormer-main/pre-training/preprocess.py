#!/usr/bin/python3  # 指定Python解释器路径
# -*- coding:utf-8 -*-  # 指定文件编码格式

import argparse  # 导入命令行参数解析模块
import os  # 导入操作系统接口模块
import sys  # 导入系统相关参数和函数模块

sys.path.append(os.getcwd())  # 将当前工作目录添加到系统路径
print("Current working directory:", os.getcwd())
import six  # 导入Python 2和3兼容库
from packaging import version  # 导入版本处理模块
from uer.utils.data import *  # 从UER导入所有数据工具
from uer.utils import *  # 从UER导入所有工具函数

assert version.parse(six.__version__) >= version.parse("1.12.0")  # 断言six版本至少为1.12.0


def main():  # 主函数
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)  # 创建参数解析器

    # Path options.
    parser.add_argument("--corpus_path", type=str, required=True,  # 添加语料库路径参数（必需）
                        help="Path of the corpus for pretraining.")  # 帮助信息：预训练语料库的路径
    parser.add_argument("--vocab_path", default=None, type=str,  # 添加词汇表路径参数
                        help="Path of the vocabulary file.")  # 帮助信息：词汇表文件的路径
    parser.add_argument("--spm_model_path", default=None, type=str,  # 添加句子片段模型路径参数
                        help="Path of the sentence piece model.")  # 帮助信息：句子片段模型的路径
    parser.add_argument("--tgt_vocab_path", default=None, type=str,  # 添加目标词汇表路径参数
                        help="Path of the target vocabulary file.")  # 帮助信息：目标词汇表文件的路径
    parser.add_argument("--tgt_spm_model_path", default=None, type=str,  # 添加目标句子片段模型路径参数
                        help="Path of the target sentence piece model.")  # 帮助信息：目标句子片段模型的路径
    parser.add_argument("--dataset_path", type=str, default="dataset.pt",  # 添加数据集路径参数
                        help="Path of the preprocessed dataset.")  # 帮助信息：预处理数据集的路径

    # Preprocess options.
    parser.add_argument("--tokenizer", choices=["bert", "char", "space"], default="bert",  # 添加分词器参数
                        help="Specify the tokenizer."  # 帮助信息：指定分词器
                             "Original Google BERT uses bert tokenizer on Chinese corpus."  # 原始Google BERT在中文语料上使用bert分词器
                             "Char tokenizer segments sentences into characters."  # 字符分词器将句子分割成字符
                             "Space tokenizer segments sentences into words according to space."  # 空格分词器根据空格将句子分割成单词
                        )
    parser.add_argument("--tgt_tokenizer", choices=["bert", "char", "space"], default="bert",  # 添加目标分词器参数
                        help="Specify the tokenizer.")  # 帮助信息：指定分词器
    parser.add_argument("--processes_num", type=int, default=1,  # 添加进程数量参数
                        help="Split the whole dataset into `processes_num` parts, "  # 帮助信息：将整个数据集分割成`processes_num`部分
                             "and each part is fed to a single process in training step.")  # 在训练步骤中，每个部分被送入单个进程
    parser.add_argument("--target",
                        choices=["bert", "bertflow", "lm", "mlm", "bilm", "albert", "seq2seq", "t5", "cls", "prefixlm"],
                        default="bert",  # 添加预训练目标参数
                        help="The training dataset target.")  # 帮助信息：训练数据集目标
    parser.add_argument("--docs_buffer_size", type=int, default=100000,  # 添加文档缓冲区大小参数
                        help="The buffer size of documents in memory, specific to targets that require negative sampling.")  # 帮助信息：内存中文档的缓冲区大小，特别适用于需要负采样的目标
    parser.add_argument("--seq_length", type=int, default=128, help="Sequence length of instances.")  # 添加序列长度参数
    parser.add_argument("--tgt_seq_length", type=int, default=128,
                        help="Target sequence length of instances.")  # 添加目标序列长度参数
    parser.add_argument("--dup_factor", type=int, default=5,  # 添加重复因子参数
                        help="Duplicate instances multiple times.")  # 帮助信息：多次重复实例
    parser.add_argument("--short_seq_prob", type=float, default=0.1,  # 添加短序列概率参数
                        help="Probability of truncating sequence."  # 帮助信息：截断序列的概率
                             "The larger value, the higher probability of using short (truncated) sequence.")  # 值越大，使用短（截断）序列的概率越高
    parser.add_argument("--full_sentences", action="store_true", help="Full sentences.")  # 添加完整句子参数
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")  # 添加随机种子参数

    # Masking options.
    parser.add_argument("--dynamic_masking", action="store_true", help="Dynamic masking.")  # 添加动态掩码参数
    parser.add_argument("--whole_word_masking", action="store_true", help="Whole word masking.")  # 添加全词掩码参数
    parser.add_argument("--span_masking", action="store_true", help="Span masking.")  # 添加跨度掩码参数
    parser.add_argument("--span_geo_prob", type=float, default=0.2,  # 添加跨度几何概率参数
                        help="Hyperparameter of geometric distribution for span masking.")  # 帮助信息：跨度掩码的几何分布超参数
    parser.add_argument("--span_max_length", type=int, default=10,  # 添加跨度最大长度参数
                        help="Max length for span masking.")  # 帮助信息：跨度掩码的最大长度

    args = parser.parse_args()  # 解析参数

    # Dynamic masking.
    if args.dynamic_masking:  # 如果使用动态掩码
        args.dup_factor = 1  # 设置重复因子为1

    # Build tokenizer.
    tokenizer = str2tokenizer[args.tokenizer](args)  # 构建分词器
    if args.target == "seq2seq":  # 如果目标是序列到序列
        args.tgt_tokenizer = str2tokenizer[args.tgt_tokenizer](args, False)  # 构建目标分词器

    # Build and save dataset.
    dataset = str2dataset[args.target](args, tokenizer.vocab, tokenizer)  # 构建数据集
    dataset.build_and_save(args.processes_num, split_by_flow=True)  # 构建并保存数据集（按流分割）


if __name__ == "__main__":  # 如果是主程序
    main()  # 调用主函数

'''python pre-training/preprocess.py \
    --corpus_path data_generation/data/pretrain/corpus_bigram.txt \
    --vocab_path models/encryptd_vocab.txt \
    --dataset_path data_generation/data/pretrain_dataset.pt \
    --target bert \
    --seq_length 320
'''
'''python pre-training/preprocess.py \
    --corpus_path data_generation/data/pretrain/corpus_bigram.txt \
    --vocab_path models/encryptd_vocab.txt \
    --dataset_path data_generation/data/pretrain_dataset.pt \
    --target bertflow \
    --processes_num 80 \
    --seq_length 512 \
'''

'''
python pre-training/preprocess.py \
    --corpus_path data_generation/data/pretrain/corpus_bigram.txt \
    --vocab_path models/encryptd_vocab.txt \
    --dataset_path data_generation/data/pretrain_dataset.pt \
    --target bertflow \
    --processes_num 32 \
    --seq_length 512 \
'''
'''
nohup python -u pre-training/preprocess.py \
     --corpus_path data_generation/data/pretrain/corpus_bigram.txt \
     --vocab_path models/encryptd_vocab.txt \
     --dataset_path data_generation/data/pretrain_dataset.pt \
     --target bertflow \
     --processes_num 80 \
     --seq_length 512 \
     > why_preprocess_32.log 2>&1 &
'''