#!/usr/bin/env python3
import os
import sys

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pretrain_data_gen import pretrain_dataset_generation, corpora_to_bigram
from vocab_gen import build_BPE, build_vocab


def main():
    print("开始生成ET-BERT预训练数据...")

    # ==========================================
    # 修正点 1: 定义清晰的路径变量
    # ==========================================
    # 原始PCAP文件夹
    raw_pcap_dir = "data/pretrain/raw_pcapng/"
    # 转换后的PCAP文件夹
    converted_pcap_dir = "data/pretrain/converted_pcap/"
    # 分割后的Flow文件夹
    split_flow_dir = "data/pretrain/split_flows/"

    # [关键] 这是一个目录，用于存放多进程生成的临时txt碎片
    # 必须以 "/" 结尾，否则 merge 函数的 path[:-1] 逻辑会出错
    corpus_temp_dir = "data/pretrain/corpus_temp/"

    # 最终合并后的语料文件路径 (由脚本自动生成，逻辑是 corpus_temp_dir 去掉末尾斜杠 + _biburst.txt)
    # 也就是: data/pretrain/corpus_temp_biburst.txt
    merged_corpus_file = "data/pretrain/corpus_temp_biburst.txt"

    # 最终的 bigram 格式文件
    bigram_file = "data/pretrain/corpus_bigram.txt"
    # 词表文件
    vocab_file = "../models/encryptd_vocab.txt"

    # 1. 生成预训练语料
    print(f"步骤1/3: 生成预训练语料 (输出到 {corpus_temp_dir})...")
    # 注意：这一步非常耗时，因为涉及到 PCAP 的读取、转换、特征提取和多进程写入
    pretrain_dataset_generation(
        pcapng_path=raw_pcap_dir,
        pcap_output_path=converted_pcap_dir,
        output_split_path=split_flow_dir,
        select_packet_len=64,
        corpora_path=corpus_temp_dir,  # [修正] 这里传入目录路径
        start_index=28,
        enhance_factor=1,
        is_multi=True
    )

    # 检查合并文件是否生成
    if not os.path.exists(merged_corpus_file):
        print(f"[错误] 未找到合并后的语料文件: {merged_corpus_file}")
        print("请检查步骤1是否完全执行成功，或检查 corpus_temp_dir 目录下是否有碎片文件。")
        return

    # 2. 转换为bigram格式
    print(f"步骤2/3: 转换为bigram格式 (读取 {merged_corpus_file})...")
    corpora_to_bigram(
        merged_corpus_file,  # [修正] 读取自动生成的正确文件名
        bigram_file
    )

    # 3. 生成词汇表
    print(f"步骤3/3: 生成词汇表 (基于 {merged_corpus_file})...")
    # 通常使用原始的 biburst 文件生成词表，或者也可以用 bigram 文件，这里保持原逻辑使用 biburst
    build_BPE(merged_corpus_file)
    build_vocab(vocab_file)

    print("预训练数据准备完成！")
    print(f"最终语料文件: {bigram_file}")
    print(f"词表文件: {vocab_file}")


if __name__ == "__main__":
    # 创建必要的目录
    os.makedirs("data/pretrain/raw_pcapng", exist_ok=True)
    os.makedirs("data/pretrain/converted_pcap", exist_ok=True)
    os.makedirs("data/pretrain/split_flows", exist_ok=True)
    # 此时不需要创建 corpus.txt 文件，而是需要清理旧环境

    print("目录结构已创建")
    print("请确保将pcapng文件放入: data/pretrain/raw_pcapng/")

    response = input("是否开始生成预训练数据? (y/n): ")
    if response.lower() == 'y':
        main()
    else:
        print("退出程序")