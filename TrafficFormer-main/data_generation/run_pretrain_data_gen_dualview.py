#!/usr/bin/env python3
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pretrain_data_gen_dualview import pretrain_dataset_generation_dualview, corpora_to_bigram
from vocab_gen import build_BPE, build_vocab


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DATA_ROOT = os.path.join(PROJECT_ROOT, "data_generation", "data")
MODE_ROOT = os.path.join(DATA_ROOT, "pretrain_moe_dual")

# 继续复用你现有的原始输入目录，不重复拷贝
RAW_PCAP_DIR = os.path.join(DATA_ROOT, "pretrain", "raw_pcapng")

# 中间产物
CONVERTED_PCAP_DIR = os.path.join(MODE_ROOT, "converted_pcap") + os.sep
SPLIT_FLOW_DIR = os.path.join(MODE_ROOT, "split_flows") + os.sep

# 多进程临时语料
CORPUS_TEMP_ROOT = os.path.join(MODE_ROOT, "corpus_temp")
RAW_TEMP_DIR = os.path.join(CORPUS_TEMP_ROOT, "raw") + os.sep
SAN_TEMP_DIR = os.path.join(CORPUS_TEMP_ROOT, "sanitized") + os.sep

# 最终语料
CORPUS_ROOT = os.path.join(MODE_ROOT, "corpus")
RAW_BIBURST = os.path.join(CORPUS_ROOT, "corpus_raw_biburst.txt")
SAN_BIBURST = os.path.join(CORPUS_ROOT, "corpus_sanitized_biburst.txt")
RAW_BIGRAM = os.path.join(CORPUS_ROOT, "corpus_raw_bigram.txt")
SAN_BIGRAM = os.path.join(CORPUS_ROOT, "corpus_sanitized_bigram.txt")

# pair 对齐信息
PAIR_ROOT = os.path.join(MODE_ROOT, "pair")
PAIR_INDEX = os.path.join(PAIR_ROOT, "pair_index.jsonl")

# vocab 中间文件
VOCAB_ROOT = os.path.join(MODE_ROOT, "vocab")
VOCAB_SOURCE = os.path.join(VOCAB_ROOT, "corpus_vocab_source.txt")

# 最终词表继续放 models/
VOCAB_FILE = os.path.join(PROJECT_ROOT, "models", "encryptd_vocab_moe_dual.txt")

def _concat_for_vocab(raw_corpus, san_corpus, out_path):
    with open(out_path, "w") as fw:
        for path in [raw_corpus, san_corpus]:
            with open(path, "r") as fr:
                while True:
                    line = fr.readline()
                    if not line:
                        break
                    fw.write(line)


def main():
    print(f"Step 1/4: generate dual-view corpus into {MODE_ROOT}")
    pretrain_dataset_generation_dualview(
        pcapng_path=RAW_PCAP_DIR,
        pcap_output_path=CONVERTED_PCAP_DIR,
        output_split_path=SPLIT_FLOW_DIR,
        select_packet_len=64,
        raw_corpora_temp_path=RAW_TEMP_DIR,
        sanitized_corpora_temp_path=SAN_TEMP_DIR,
        raw_merged_corpus_path=RAW_BIBURST,
        sanitized_merged_corpus_path=SAN_BIBURST,
        pair_index_path=PAIR_INDEX,
        start_index=28,
        enhance_factor=1,
        is_multi=True,
    )

    if not os.path.exists(RAW_BIBURST):
        print(f"[ERROR] raw merged corpus not found: {RAW_BIBURST}")
        return
    if not os.path.exists(SAN_BIBURST):
        print(f"[ERROR] sanitized merged corpus not found: {SAN_BIBURST}")
        return
    if not os.path.exists(PAIR_INDEX):
        print(f"[ERROR] pair index not found: {PAIR_INDEX}")
        return

    print(f"Step 2/4: convert raw corpus to bigram -> {RAW_BIGRAM}")
    corpora_to_bigram(RAW_BIBURST, RAW_BIGRAM)

    print(f"Step 3/4: convert sanitized corpus to bigram -> {SAN_BIGRAM}")
    corpora_to_bigram(SAN_BIBURST, SAN_BIGRAM)

    print(f"Step 4/4: build shared vocab -> {VOCAB_FILE}")
    _concat_for_vocab(RAW_BIBURST, SAN_BIBURST, VOCAB_SOURCE)
    build_BPE(VOCAB_SOURCE)
    build_vocab(VOCAB_FILE)

    print("Dual-view corpus generation finished.")
    print(f"Raw biburst: {RAW_BIBURST}")
    print(f"Sanitized biburst: {SAN_BIBURST}")
    print(f"Raw bigram: {RAW_BIGRAM}")
    print(f"Sanitized bigram: {SAN_BIGRAM}")
    print(f"Pair index: {PAIR_INDEX}")
    print(f"Shared vocab: {VOCAB_FILE}")

    print("Suggested preprocess commands:")
    print(
        "python pre-training/preprocess.py "
        f"--corpus_path {RAW_BIGRAM} "
        f"--vocab_path {VOCAB_FILE} "
        f"--dataset_path {os.path.join(DATA_ROOT, 'pretrain_dataset_moe_dual_raw.pt')} "
        "--target bertflow --processes_num 32 --seq_length 512"
    )
    print(
        "python pre-training/preprocess.py "
        f"--corpus_path {SAN_BIGRAM} "
        f"--vocab_path {VOCAB_FILE} "
        f"--dataset_path {os.path.join(DATA_ROOT, 'pretrain_dataset_moe_dual_san.pt')} "
        "--target bertflow --processes_num 32 --seq_length 512"
    )


if __name__ == "__main__":
    os.makedirs(RAW_PCAP_DIR, exist_ok=True)
    os.makedirs(CONVERTED_PCAP_DIR, exist_ok=True)
    os.makedirs(SPLIT_FLOW_DIR, exist_ok=True)

    os.makedirs(CORPUS_TEMP_ROOT, exist_ok=True)
    os.makedirs(RAW_TEMP_DIR, exist_ok=True)
    os.makedirs(SAN_TEMP_DIR, exist_ok=True)

    os.makedirs(CORPUS_ROOT, exist_ok=True)
    os.makedirs(PAIR_ROOT, exist_ok=True)
    os.makedirs(VOCAB_ROOT, exist_ok=True)

    print("MoE dual-view pretrain data pipeline")
    print(f"Put your pcap/pcapng files into: {RAW_PCAP_DIR}")
    response = input("Run dual-view corpus generation now? (y/n): ")
    if response.lower() == "y":
        main()
    else:
        print("Cancelled.")
