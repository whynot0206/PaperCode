#!/usr/bin/env python3
import argparse
import os
import sys


CURRENT_SCRIPT_PATH = os.path.abspath(__file__)
ROOT_PATH = os.path.dirname(os.path.dirname(CURRENT_SCRIPT_PATH))
sys.path.append(ROOT_PATH)


def manual_merge_logic(dataset_path, workers_num, keep_temp=False):
    """Merge preprocess worker shards into one dataset .pt file."""
    abs_dataset_path = os.path.abspath(dataset_path)
    temp_dir = os.path.join(os.path.dirname(abs_dataset_path), "temp_datasets")

    print(f"Dataset output: {abs_dataset_path}")
    print(f"Temporary shard directory: {temp_dir}")
    print(f"Expected workers: {workers_num}")

    if not os.path.isdir(temp_dir):
        raise FileNotFoundError(f"Temporary directory not found: {temp_dir}")

    existing_shards = [
        os.path.join(temp_dir, f"dataset-tmp-{i}.pt")
        for i in range(workers_num)
        if os.path.exists(os.path.join(temp_dir, f"dataset-tmp-{i}.pt"))
    ]
    print(f"Found shards: {len(existing_shards)}/{workers_num}")

    if not existing_shards:
        raise FileNotFoundError(
            "No preprocess shards were found. Re-run preprocess.py first, "
            "or check --dataset_path/--workers_num."
        )

    if os.path.exists(abs_dataset_path):
        os.remove(abs_dataset_path)

    with open(abs_dataset_path, "ab") as dataset_writer:
        for i in range(workers_num):
            temp_file_path = os.path.join(temp_dir, f"dataset-tmp-{i}.pt")
            if not os.path.exists(temp_file_path):
                print(f"Warning: missing shard {i}: {temp_file_path}")
                continue

            print(f"Merging shard {i + 1}/{workers_num}: {temp_file_path}")
            with open(temp_file_path, "rb") as reader:
                while True:
                    chunk = reader.read(1024 * 1024)
                    if not chunk:
                        break
                    dataset_writer.write(chunk)

            if not keep_temp:
                os.remove(temp_file_path)

    print("Merge finished.")
    print(f"Final dataset: {abs_dataset_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge preprocess temp shards into a final pretrain dataset."
    )
    parser.add_argument(
        "--dataset_path",
        default=os.path.join(ROOT_PATH, "data_generation/data/pretrain_moe/pretrain_dataset_moe.pt"),
        help="Final dataset .pt path. Its sibling temp_datasets directory is used.",
    )
    parser.add_argument(
        "--workers_num",
        type=int,
        default=80,
        help="Number of preprocess worker shards to merge.",
    )
    parser.add_argument(
        "--keep_temp",
        action="store_true",
        help="Keep temporary shard files after merging.",
    )
    args = parser.parse_args()

    manual_merge_logic(args.dataset_path, args.workers_num, args.keep_temp)


if __name__ == "__main__":
    main()
