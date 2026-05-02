#!/usr/bin/env python3
import argparse
import csv
import random
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a balanced warmup CSV from a large training CSV using deterministic streaming sampling."
    )
    parser.add_argument("--input", required=True, help="Path to the source training CSV.")
    parser.add_argument("--output", required=True, help="Path to the balanced output CSV.")
    parser.add_argument(
        "--label-column",
        default="target_actionTaken",
        help="Label column used to balance the dataset.",
    )
    parser.add_argument(
        "--majority-factor",
        type=int,
        default=2,
        help="How many majority-class rows to keep relative to the minority base count.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic selection.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    # Auto-detect label range and remap to 0/1/2 if needed (-1/0/1 → 0/1/2)
    raw_counts: Counter = Counter()
    with input_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_counts[int(float(row[args.label_column]))] += 1

    raw_labels = sorted(raw_counts.keys())
    label_remap = {raw: idx for idx, raw in enumerate(raw_labels)}
    if label_remap != {0: 0, 1: 1, 2: 2}:
        print(f"Remapping labels: {label_remap}")

    counts = Counter({label_remap[k]: v for k, v in raw_counts.items()})

    required_labels = [0, 1, 2]
    missing = [label for label in required_labels if counts[label] == 0]
    if missing:
        raise ValueError(f"Missing required labels in source CSV: {missing}")

    minority_base = min(counts[0], counts[2])
    target_counts = {
        0: minority_base,
        1: min(counts[1], args.majority_factor * minority_base),
        2: minority_base,
    }

    print("Source label counts:")
    for label in sorted(counts):
        print(f"  class {label}: {counts[label]}")

    print("Target balanced counts:")
    for label in sorted(target_counts):
        print(f"  class {label}: {target_counts[label]}")

    remaining = dict(counts)
    needed = dict(target_counts)
    written = Counter()
    rnd = random.Random(args.seed)

    # Second pass: streaming exact-proportion selection.
    with input_path.open("r", newline="") as src, output_path.open("w", newline="") as dst:
        reader = csv.DictReader(src)
        writer = csv.DictWriter(dst, fieldnames=reader.fieldnames)
        writer.writeheader()

        for row in reader:
            raw_label = int(float(row[args.label_column]))
            label = label_remap.get(raw_label, raw_label)

            if label not in needed:
                remaining[label] -= 1
                continue

            if needed[label] <= 0:
                remaining[label] -= 1
                continue

            probability = needed[label] / remaining[label]
            if rnd.random() <= probability:
                row[args.label_column] = str(label)
                for col in ("target_tradeSide",):
                    if col in row and col != args.label_column:
                        raw_extra = int(float(row[col]))
                        row[col] = str(label_remap.get(raw_extra, raw_extra))
                writer.writerow(row)
                needed[label] -= 1
                written[label] += 1

            remaining[label] -= 1

    print("Written label counts:")
    for label in sorted(written):
        print(f"  class {label}: {written[label]}")

    total_written = sum(written.values())
    print(f"Done. Total written rows: {total_written}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
