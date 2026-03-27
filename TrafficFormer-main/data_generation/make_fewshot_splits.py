import argparse
import csv
import os
import random
import shutil
from collections import defaultdict


def read_tsv(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)
    if "label" not in reader.fieldnames or "text_a" not in reader.fieldnames:
        raise ValueError(f"Unexpected TSV format in {path}. Expected columns: label, text_a")
    return rows


def write_tsv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "text_a"], delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def sample_k_shot(rows, k, seed):
    by_label = defaultdict(list)
    for row in rows:
        by_label[row["label"]].append(row)

    sampled = []
    rng = random.Random(seed)
    stats = {}
    for label, label_rows in sorted(by_label.items(), key=lambda x: int(x[0])):
        if len(label_rows) < k:
            raise ValueError(
                f"Label {label} only has {len(label_rows)} samples, cannot sample {k}-shot."
            )
        chosen = rng.sample(label_rows, k)
        sampled.extend(chosen)
        stats[label] = len(chosen)

    rng.shuffle(sampled)
    return sampled, stats


def main():
    parser = argparse.ArgumentParser(description="Create few-shot train splits from an existing dataset TSV.")
    parser.add_argument("--dataset_dir", required=True, help="Directory containing train/valid/test_dataset.tsv")
    parser.add_argument("--shots", required=True, nargs="+", type=int, help="Shot counts per class, e.g. 5 10")
    parser.add_argument("--seeds", required=True, nargs="+", type=int, help="Sampling seeds, e.g. 1 7 13 21 42")
    parser.add_argument(
        "--output_root",
        default=None,
        help="Output root directory. Default: <dataset_dir>/../fewshot",
    )
    args = parser.parse_args()

    train_path = os.path.join(args.dataset_dir, "train_dataset.tsv")
    valid_path = os.path.join(args.dataset_dir, "valid_dataset.tsv")
    test_path = os.path.join(args.dataset_dir, "test_dataset.tsv")

    train_rows = read_tsv(train_path)
    if args.output_root is None:
        output_root = os.path.join(os.path.dirname(args.dataset_dir), "fewshot")
    else:
        output_root = args.output_root

    print(f"Reading train set from: {train_path}")
    print(f"Output root: {output_root}")

    for shot in args.shots:
        for seed in args.seeds:
            sampled_rows, stats = sample_k_shot(train_rows, shot, seed)
            split_name = f"{shot}shot_seed{seed}"
            split_dir = os.path.join(output_root, split_name, "dataset")
            write_tsv(os.path.join(split_dir, "train_dataset.tsv"), sampled_rows)
            shutil.copy2(valid_path, os.path.join(split_dir, "valid_dataset.tsv"))
            shutil.copy2(test_path, os.path.join(split_dir, "test_dataset.tsv"))
            print(f"[DONE] {split_name}: total_train={len(sampled_rows)}, per_label={stats}")


if __name__ == "__main__":
    main()
