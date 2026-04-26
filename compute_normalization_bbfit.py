#!/usr/bin/env python3
"""
Berekent per-feature mean en std uit de trainset (streaming, geen RAM-probleem).
Slaat op als JSON zodat train/eval scripts het kunnen laden.

Usage:
    python compute_normalization_bbfit.py \
        --train-csv /workspace/data/lstm_train_balanced_warmup.csv \
        --output-json /workspace/data/normalization_stats.json
"""
import argparse
import csv
import json
import math
from pathlib import Path

META_COLUMNS = {
    "runId", "split", "windowStartTimestamp", "windowEndTimestamp",
    "targetTimestamp", "sequenceLength",
}
TARGET_COLUMNS = {
    "target_actionTaken", "target_tradeSide", "target_tradeActionRaw",
    "target_lastTrade", "target_netEquity", "target_netEquityDelta",
    "target_isNetEquityUp",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-csv", required=True)
    p.add_argument("--output-json", required=True)
    p.add_argument("--limit-rows", type=int, default=None)
    args = p.parse_args()

    csv_path = Path(args.train_csv)

    with csv_path.open("r", newline="") as f:
        header = next(csv.reader(f))

    feature_columns = [c for c in header if c not in META_COLUMNS and c not in TARGET_COLUMNS]
    n_features = len(feature_columns)
    print(f"Feature-kolommen: {n_features}", flush=True)

    # Welford online variance voor numerieke stabiliteit
    count = 0
    mean = [0.0] * n_features
    M2 = [0.0] * n_features

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader):
            if args.limit_rows is not None and row_idx >= args.limit_rows:
                break

            count += 1
            for i, col in enumerate(feature_columns):
                x = float(row[col])
                delta = x - mean[i]
                mean[i] += delta / count
                delta2 = x - mean[i]
                M2[i] += delta * delta2

            if count % 50000 == 0:
                print(f"  {count:,} rijen verwerkt...", flush=True)

    print(f"Totaal: {count:,} rijen", flush=True)

    std = [math.sqrt(M2[i] / count) if count > 1 else 1.0 for i in range(n_features)]
    # Vervang std=0 door 1 om deling door nul te voorkomen
    std = [s if s > 1e-8 else 1.0 for s in std]

    # Korte feature-namen (zonder timestep-prefix) voor leesbaarheid
    raw_names = []
    seen = set()
    for col in feature_columns:
        if len(col) > 5 and col[0] == "t" and col[1:4].isdigit() and col[4] == "_":
            name = col[5:]
            if name not in seen:
                seen.add(name)
                raw_names.append(name)

    print("\nTop-10 features met grootste std (variabele features):")
    indexed = sorted(enumerate(std[:len(raw_names)]), key=lambda x: x[1], reverse=True)
    for rank, (i, s) in enumerate(indexed[:10], 1):
        print(f"  {rank:2d}. {raw_names[i]:<40s} mean={mean[i]:.4g}  std={s:.4g}")

    print("\nTop-10 features met kleinste std (bijna constante features):")
    for rank, (i, s) in enumerate(indexed[-10:][::-1], 1):
        print(f"  {rank:2d}. {raw_names[i]:<40s} mean={mean[i]:.4g}  std={s:.4g}")

    result = {
        "source_csv": str(csv_path),
        "n_rows": count,
        "n_features": n_features,
        "feature_columns": feature_columns,
        "mean": mean,
        "std": std,
    }

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"\nOpgeslagen: {out}", flush=True)
    print("Klaar.", flush=True)


if __name__ == "__main__":
    main()
