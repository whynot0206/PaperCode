import json
import os
import shutil
import multiprocessing as mp

from tqdm import tqdm

from pretrain_data_gen import (
    convert_pcapng_2_pcap,
    split_cap,
    corpora_to_bigram,
    get_bursts_dualview,
)


def _sorted_split_files(split_root):
    all_files = []
    for _p, _d, files in os.walk(split_root):
        for file in files:
            all_files.append(os.path.join(_p, file))
    return sorted(all_files)


def _merge_temp_dir_sorted(temp_dir, merged_output_file):
    temp_files = [
        f for f in os.listdir(temp_dir)
        if f.endswith("_biburst.txt")
    ]
    temp_files = sorted(
        temp_files,
        key=lambda x: int(x.split("_")[0]) if x.split("_")[0].isdigit() else x
    )

    with open(merged_output_file, "w") as fw:
        for fname in temp_files:
            path = os.path.join(temp_dir, fname)
            with open(path, "r") as fr:
                while True:
                    line = fr.readline()
                    if not line:
                        break
                    fw.write(line)


def _scan_flow_start_lines(corpus_path):
    flow_start_lines = []
    with open(corpus_path, "r") as f:
        for line_id, line in enumerate(f):
            if line.startswith("||"):
                flow_start_lines.append(line_id)
    return flow_start_lines


def _build_pair_index(raw_corpus_path, sanitized_corpus_path, pair_index_path):
    raw_flow_starts = _scan_flow_start_lines(raw_corpus_path)
    san_flow_starts = _scan_flow_start_lines(sanitized_corpus_path)

    if len(raw_flow_starts) != len(san_flow_starts):
        raise ValueError(
            f"Flow count mismatch between raw and sanitized corpora: "
            f"{len(raw_flow_starts)} vs {len(san_flow_starts)}"
        )

    with open(pair_index_path, "w") as fw:
        for pair_id, (raw_line_id, san_line_id) in enumerate(zip(raw_flow_starts, san_flow_starts)):
            record = {
                "pair_id": pair_id,
                "flow_id": pair_id,
                "raw_sample_path": raw_corpus_path,
                "raw_line_id": raw_line_id,
                "sanitized_sample_path": sanitized_corpus_path,
                "sanitized_line_id": san_line_id,
            }
            fw.write(json.dumps(record, ensure_ascii=False) + "\n")


def pretrain_dataset_generation_dualview(
    pcapng_path,
    pcap_output_path,
    output_split_path,
    select_packet_len,
    raw_corpora_temp_path,
    sanitized_corpora_temp_path,
    raw_merged_corpus_path,
    sanitized_merged_corpus_path,
    pair_index_path,
    start_index=0,
    enhance_factor=1,
    is_multi=True,
):
    if not os.listdir(pcap_output_path):
        print("Begin to convert pcapng to pcap.")
        for _parent, _dirs, files in os.walk(pcapng_path):
            for file in files:
                if "pcapng" in file:
                    convert_pcapng_2_pcap(_parent, file, pcap_output_path)
                else:
                    shutil.copy(_parent + "/" + file, pcap_output_path + file)

    if not os.path.exists(output_split_path + "splitcap"):
        print("Begin to split pcap as session flows.")
        for _p, _d, files in os.walk(pcap_output_path):
            for file in files:
                split_cap(output_split_path, pcap_output_path, file)

    split_root = output_split_path + "splitcap"
    all_files = _sorted_split_files(split_root)

    print("Begin to generate dual-view burst dataset.")

    if is_multi:
        os.makedirs(raw_corpora_temp_path, exist_ok=True)
        os.makedirs(sanitized_corpora_temp_path, exist_ok=True)

        pbar = tqdm(total=len(all_files))
        pbar.set_description("get dualview bursts")
        update = lambda *args: pbar.update(1)

        pool = mp.Pool(processes=10)
        results = []

        for file in all_files:
            result = pool.apply_async(
                get_bursts_dualview,
                (
                    file,
                    select_packet_len,
                    raw_corpora_temp_path,
                    sanitized_corpora_temp_path,
                    start_index,
                    enhance_factor,
                    True,
                ),
                callback=update,
            )
            results.append(result)

        pool.close()
        pool.join()

        for result in results:
            result.get()

        pbar.close()

        print("start merge raw files...")
        _merge_temp_dir_sorted(raw_corpora_temp_path, raw_merged_corpus_path)

        print("start merge sanitized files...")
        _merge_temp_dir_sorted(sanitized_corpora_temp_path, sanitized_merged_corpus_path)

        if os.path.exists(raw_corpora_temp_path):
            shutil.rmtree(raw_corpora_temp_path)
        if os.path.exists(sanitized_corpora_temp_path):
            shutil.rmtree(sanitized_corpora_temp_path)
    else:
        for file in tqdm(all_files):
            get_bursts_dualview(
                file,
                select_packet_len=select_packet_len,
                raw_corpora_path=raw_merged_corpus_path,
                sanitized_corpora_path=sanitized_merged_corpus_path,
                start_index=start_index,
                enhance_factor=enhance_factor,
                is_multi=False,
            )

    print("build pair_index.jsonl ...")
    _build_pair_index(raw_merged_corpus_path, sanitized_merged_corpus_path, pair_index_path)

    return 0


__all__ = [
    "pretrain_dataset_generation_dualview",
    "corpora_to_bigram",
]