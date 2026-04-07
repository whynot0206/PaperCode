#!/usr/bin/env python3
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pretrain_data_gen_moe import pretrain_dataset_generation_moe, corpora_to_bigram
from vocab_gen import build_BPE, build_vocab


def main():
    raw_pcap_dir = "data/pretrain/raw_pcapng/"
    converted_pcap_dir = "data/pretrain_moe/converted_pcap/"
    split_flow_dir = "data/pretrain_moe/split_flows/"
    corpus_temp_dir = "data/pretrain_moe/corpus_temp/"
    merged_corpus_file = "data/pretrain_moe/corpus_temp_biburst.txt"
    bigram_file = "data/pretrain_moe/corpus_moe_bigram.txt"
    vocab_file = "../models/encryptd_vocab_moe.txt"

    print(f"Step 1/3: generate MoE corpus into {corpus_temp_dir}")
    pretrain_dataset_generation_moe(
        pcapng_path=raw_pcap_dir,
        pcap_output_path=converted_pcap_dir,
        output_split_path=split_flow_dir,
        select_packet_len=64,
        corpora_path=corpus_temp_dir,
        start_index=28,
        enhance_factor=1,
        is_multi=True,
    )

    if not os.path.exists(merged_corpus_file):
        print(f"[ERROR] merged corpus file not found: {merged_corpus_file}")
        return

    print(f"Step 2/3: convert merged corpus to bigram file {bigram_file}")
    corpora_to_bigram(merged_corpus_file, bigram_file)

    print(f"Step 3/3: build vocab file {vocab_file}")
    build_BPE(merged_corpus_file)
    build_vocab(vocab_file)

    print("MoE pretrain corpus generation finished.")
    print(f"Corpus bigram: {bigram_file}")
    print(f"Vocab: {vocab_file}")
    print("Suggested preprocess command:")
    print(
        "python pre-training/preprocess.py "
        "--corpus_path data_generation/data/pretrain_moe/corpus_moe_bigram.txt "
        "--vocab_path models/encryptd_vocab_moe.txt "
        "--dataset_path data_generation/data/pretrain_dataset_moe.pt "
        "--target bertflow --processes_num 32 --seq_length 512"
    )


if __name__ == "__main__":
    os.makedirs("data/pretrain_moe/raw_pcapng", exist_ok=True)
    os.makedirs("data/pretrain_moe/converted_pcap", exist_ok=True)
    os.makedirs("data/pretrain_moe/split_flows", exist_ok=True)

    print("MoE-specific pretrain data pipeline")
    print("Put your pcap/pcapng files into: data/pretrain_moe/raw_pcapng/")

    response = input("Run MoE corpus generation now? (y/n): ")
    if response.lower() == "y":
        main()
    else:
        print("Cancelled.")
