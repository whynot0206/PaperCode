import os
import shutil
import multiprocessing as mp

from tqdm import tqdm

from pretrain_data_gen import (
    convert_pcapng_2_pcap,
    split_cap,
    merge,
    get_bursts_moe,
    corpora_to_bigram,
)


def pretrain_dataset_generation_moe(pcapng_path, pcap_output_path, output_split_path, select_packet_len, corpora_path,
                                    start_index=0, enhance_factor=1, is_multi=True):
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

    print("Begin to generate MoE-friendly burst dataset.")
    if is_multi:
        all_files = []
        for _p, _d, files in os.walk(output_split_path + "splitcap"):
            for file in files:
                all_files.append(_p + "/" + file)
        pbar = tqdm(total=len(all_files))
        pbar.set_description("get moe bursts")
        update = lambda *args: pbar.update()
        if not os.path.exists(corpora_path):
            os.makedirs(corpora_path)
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
        for _p, _d, files in os.walk(output_split_path + "splitcap"):
            for file in tqdm(files):
                get_bursts_moe(
                    _p + "/" + file,
                    select_packet_len=select_packet_len,
                    corpora_path=corpora_path,
                    start_index=start_index,
                    enhance_factor=enhance_factor,
                )

    return 0


__all__ = ["pretrain_dataset_generation_moe", "corpora_to_bigram"]
