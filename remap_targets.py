import csv
import os
from pathlib import Path

files = [
    Path("lstm_train_sequences.csv"),
    Path("lstm_validation_sequences.csv"),
    Path("lstm_test_sequences.csv"),
]

label_cols = ["target_actionTaken", "target_tradeSide"]

mappings = {col: set() for col in label_cols}

with files[0].open("r", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        for col in label_cols:
            mappings[col].add(int(float(row[col])))

mappings = {
    col: {raw: idx for idx, raw in enumerate(sorted(values))}
    for col, values in mappings.items()
}

print("Learned mappings:")
for col, mp in mappings.items():
    print(col, mp)

for path in files:
    tmp_path = path.with_suffix(path.suffix + ".tmp")

    with path.open("r", newline="") as src, tmp_path.open("w", newline="") as dst:
        reader = csv.DictReader(src)
        writer = csv.DictWriter(dst, fieldnames=reader.fieldnames)
        writer.writeheader()

        for row in reader:
            for col in label_cols:
                raw = int(float(row[col]))
                if raw not in mappings[col]:
                    raise ValueError(f"{path}: unexpected label {raw} in {col}")
                row[col] = str(mappings[col][raw])
            writer.writerow(row)

    backup_path = path.with_suffix(path.suffix + ".bak")
    os.replace(path, backup_path)
    os.replace(tmp_path, path)
    print(f"Rewrote {path} (backup at {backup_path})")

print("Done.")
