import os
import shutil
import multiprocessing as mp
from tqdm import tqdm

from pretrain_data_gen import (
    convert_pcapng_2_pcap,
    split_cap,
    merge,
    get_bursts_moe,
    get_bursts_moe_pair,
    corpora_to_bigram,
)


def _sorted_split_files(split_root):
    all_files = []
    for _p, _d, files in os.walk(split_root):
        for file in files:
            all_files.append(os.path.join(_p, file))
    return sorted(all_files)


def pretrain_dataset_generation_moe(
    pcapng_path,
    pcap_output_path,
    output_split_path,
    select_packet_len,
    corpora_path,
    start_index=0,
    enhance_factor=1,
    is_multi=True,
    mode="single_sanitized",   # "single_sanitized" or "dual_view"
):
    if mode not in {"single_sanitized", "dual_view"}:
        raise ValueError(f"Unsupported mode: {mode}")

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

    print(f"Begin to generate MoE-friendly burst dataset. mode={mode}")
    split_root = output_split_path + "splitcap"
    all_files = _sorted_split_files(split_root)

    if mode == "single_sanitized":
        if is_multi:
            if not os.path.exists(corpora_path):
                os.makedirs(corpora_path)

            pbar = tqdm(total=len(all_files))
            pbar.set_description("get moe bursts")
            update = lambda *args: pbar.update(1)

            pool = mp.Pool(processes=10)
            results = []
            for file in all_files:
                result = pool.apply_async(
                    get_bursts_moe,
                    (file, select_packet_len, corpora_path, start_index, enhance_factor, True),
                    callback=update,
                )
                results.append(result)

            pool.close()
            pool.join()

            for result in results:
                result.get()

            pbar.close()

            print("start merge files...")
            merge(corpora_path)

            if os.path.exists(corpora_path):
                shutil.rmtree(corpora_path)
        else:
            for file in tqdm(all_files):
                get_bursts_moe(
                    file,
                    select_packet_len=select_packet_len,
                    corpora_path=corpora_path,
                    start_index=start_index,
                    enhance_factor=enhance_factor,
                    is_multi=False,
                )
        return 0

    # dual_view mode
    raw_temp_dir = corpora_path[:-1] + "_raw/" if corpora_path.endswith("/") else corpora_path + "_raw/"
    san_temp_dir = corpora_path[:-1] + "_san/" if corpora_path.endswith("/") else corpora_path + "_san/"

    if is_multi:
        os.makedirs(raw_temp_dir, exist_ok=True)
        os.makedirs(san_temp_dir, exist_ok=True)

        pbar = tqdm(total=len(all_files))
        pbar.set_description("get moe dual-view bursts")
        update = lambda *args: pbar.update(1)

        pool = mp.Pool(processes=10)
        results = []
        for file in all_files:
            result = pool.apply_async(
                get_bursts_moe_pair,
                (file, select_packet_len, raw_temp_dir, san_temp_dir, start_index, enhance_factor, True),
                callback=update,
            )
            results.append(result)

        pool.close()
        pool.join()

        for result in results:
            result.get()

        pbar.close()

        print("start merge raw files...")
        merge(raw_temp_dir)
        print("start merge sanitized files...")
        merge(san_temp_dir)

        if os.path.exists(raw_temp_dir):
            shutil.rmtree(raw_temp_dir)
        if os.path.exists(san_temp_dir):
            shutil.rmtree(san_temp_dir)
    else:
        raw_output_file = raw_temp_dir[:-1] + "_biburst.txt"
        san_output_file = san_temp_dir[:-1] + "_biburst.txt"

        for file in tqdm(all_files):
            get_bursts_moe_pair(
                file,
                select_packet_len=select_packet_len,
                raw_corpora_path=raw_output_file,
                san_corpora_path=san_output_file,
                start_index=start_index,
                enhance_factor=enhance_factor,
                is_multi=False,
            )

    return 0


__all__ = ["pretrain_dataset_generation_moe", "corpora_to_bigram"]