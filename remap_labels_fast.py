#!/usr/bin/env python3
"""
Remap target labels van -1/0/1 naar 0/1/2 in-place.
Gebruikt bekende mapping, geen scan-pass nodig.

Usage:
    python remap_labels_fast.py <csv_file> [<csv_file2> ...]
"""
import csv
import os
import sys
from pathlib import Path

LABEL_COLS = ["target_actionTaken", "target_tradeSide"]
REMAP = {"-1": "0", "0": "1", "1": "2"}


def remap_file(path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    print(f"Remapping {path} ...", flush=True)
    with path.open("r", newline="") as src, tmp.open("w", newline="") as dst:
        reader = csv.DictReader(src)
        writer = csv.DictWriter(dst, fieldnames=reader.fieldnames)
        writer.writeheader()
        for i, row in enumerate(reader):
            for col in LABEL_COLS:
                if col in row:
                    row[col] = REMAP.get(row[col], row[col])
            writer.writerow(row)
            if (i + 1) % 500_000 == 0:
                print(f"  {i+1:,} rijen ...", flush=True)
    bak = path.with_suffix(".bak")
    os.replace(path, bak)
    os.replace(tmp, path)
    print(f"  Klaar. Backup: {bak}", flush=True)


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        remap_file(Path(arg))
    print("Done.", flush=True)
