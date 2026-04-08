import argparse
import csv
import math
import random
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build weak route labels from existing flow text for semantic router supervision."
    )
    parser.add_argument("--train_path", required=True, help="Input train TSV path.")
    parser.add_argument("--dev_path", required=True, help="Input dev TSV path.")
    parser.add_argument("--test_path", required=True, help="Input test TSV path.")
    parser.add_argument("--output_dir", required=True, help="Directory to store labeled TSV files.")
    parser.add_argument("--route_clusters", type=int, default=4, help="Number of route clusters.")
    parser.add_argument("--kmeans_iters", type=int, default=50, help="Maximum KMeans iterations.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    return parser.parse_args()


def load_tsv(path):
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    return fieldnames, rows


def split_packets(text):
    packets = []
    current = []
    for tok in text.strip().split():
        if tok == "[SEP]":
            if current:
                packets.append(current)
                current = []
            continue
        current.append(tok)
    if current:
        packets.append(current)
    return packets


def safe_std(values):
    if len(values) <= 1:
        return 0.0
    return float(np.std(np.asarray(values, dtype=np.float32)))


def extract_behavior_features(text):
    packets = split_packets(text)
    packet_count = len(packets)
    packet_lengths = [len(pkt) for pkt in packets]

    if packet_count == 0:
        return np.zeros(8, dtype=np.float32)

    token_stream = [tok for pkt in packets for tok in pkt]
    unique_ratio = len(set(token_stream)) / max(len(token_stream), 1)

    first_half = packets[: max(1, packet_count // 2)]
    second_half = packets[max(1, packet_count // 2):]
    first_half_tokens = sum(len(pkt) for pkt in first_half)
    second_half_tokens = sum(len(pkt) for pkt in second_half) if second_half else 0
    front_back_ratio = first_half_tokens / max(second_half_tokens, 1)

    packet_delta = [abs(packet_lengths[i] - packet_lengths[i - 1]) for i in range(1, packet_count)]
    mean_len = float(np.mean(packet_lengths))
    std_len = safe_std(packet_lengths)
    max_len = float(np.max(packet_lengths))
    mean_delta = float(np.mean(packet_delta)) if packet_delta else 0.0

    features = np.asarray(
        [
            float(packet_count),
            mean_len,
            std_len,
            max_len,
            unique_ratio,
            front_back_ratio,
            mean_delta,
            float(sum(packet_lengths)),
        ],
        dtype=np.float32,
    )
    return features


def standardize(train_x, other_xs):
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    train_z = (train_x - mean) / std
    others_z = [(x - mean) / std for x in other_xs]
    return train_z, others_z


def fit_kmeans(x, k, max_iters, seed):
    rng = random.Random(seed)
    indices = list(range(len(x)))
    rng.shuffle(indices)
    centers = x[indices[:k]].copy()

    for _ in range(max_iters):
        distances = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = distances.argmin(axis=1)

        new_centers = centers.copy()
        for cluster_id in range(k):
            cluster_points = x[labels == cluster_id]
            if len(cluster_points) > 0:
                new_centers[cluster_id] = cluster_points.mean(axis=0)

        if np.allclose(new_centers, centers):
            break
        centers = new_centers

    return centers


def assign_clusters(x, centers):
    distances = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    return distances.argmin(axis=1)


def add_route_labels(rows, labels):
    for row, label in zip(rows, labels):
        row["route_label"] = str(int(label))
    return rows


def save_tsv(path, fieldnames, rows):
    output_fields = list(fieldnames)
    if "route_label" not in output_fields:
        output_fields.append("route_label")
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=output_fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_fields, train_rows = load_tsv(args.train_path)
    dev_fields, dev_rows = load_tsv(args.dev_path)
    test_fields, test_rows = load_tsv(args.test_path)

    train_x = np.stack([extract_behavior_features(row["text_a"]) for row in train_rows])
    dev_x = np.stack([extract_behavior_features(row["text_a"]) for row in dev_rows])
    test_x = np.stack([extract_behavior_features(row["text_a"]) for row in test_rows])

    train_z, (dev_z, test_z) = standardize(train_x, [dev_x, test_x])
    centers = fit_kmeans(train_z, args.route_clusters, args.kmeans_iters, args.seed)

    train_labels = assign_clusters(train_z, centers)
    dev_labels = assign_clusters(dev_z, centers)
    test_labels = assign_clusters(test_z, centers)

    save_tsv(output_dir / "train_dataset_route.tsv", train_fields, add_route_labels(train_rows, train_labels))
    save_tsv(output_dir / "valid_dataset_route.tsv", dev_fields, add_route_labels(dev_rows, dev_labels))
    save_tsv(output_dir / "test_dataset_route.tsv", test_fields, add_route_labels(test_rows, test_labels))

    print("Saved route-labeled TSV files to:", output_dir)
    print("Route cluster counts (train):", np.bincount(train_labels, minlength=args.route_clusters).tolist())


if __name__ == "__main__":
    main()
