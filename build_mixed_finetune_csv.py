#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


def count_rows(csv_path: Path) -> int:
    with csv_path.open("r", newline="") as f:
        reader = csv.reader(f)
        next(reader)  # header
        return sum(1 for _ in reader)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a mixed finetune CSV by interleaving a balanced warmup CSV with a sampled stream from the original train CSV."
    )
    parser.add_argument("--balanced-input", required=True)
    parser.add_argument("--original-input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--original-factor",
        type=int,
        default=1,
        help="How many original rows to sample relative to the number of balanced rows.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    balanced_path = Path(args.balanced_input)
    original_path = Path(args.original_input)
    output_path = Path(args.output)

    if not balanced_path.exists():
        raise FileNotFoundError(f"Balanced input not found: {balanced_path}")
    if not original_path.exists():
        raise FileNotFoundError(f"Original input not found: {original_path}")

    balanced_count = count_rows(balanced_path)
    original_count = count_rows(original_path)
    original_target = min(original_count, balanced_count * args.original_factor)

    print(f"Balanced rows: {balanced_count}")
    print(f"Original rows: {original_count}")
    print(f"Target original sample rows: {original_target}")

    remaining_original = original_count
    needed_original = original_target
    written_balanced = 0
    written_original = 0

    with balanced_path.open("r", newline="") as f_bal, \
         original_path.open("r", newline="") as f_org, \
         output_path.open("w", newline="") as f_out:

        bal_reader = csv.DictReader(f_bal)
        org_reader = csv.DictReader(f_org)

        fieldnames = bal_reader.fieldnames
        if fieldnames is None:
            raise ValueError("Balanced CSV has no header.")
        if org_reader.fieldnames != fieldnames:
            raise ValueError("Balanced and original CSV headers differ.")

        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        bal_iter = iter(bal_reader)

        for org_row in org_reader:
            # Exact streaming sample of the original CSV.
            take_org = False
            if needed_original > 0:
                probability = needed_original / remaining_original
                # Deterministic take by threshold using row index logic:
                # choose current row whenever target fraction requires it.
                # This avoids randomness and still gives exact count.
                take_org = probability >= 1.0 or (needed_original * remaining_original) >= (remaining_original * remaining_original)

            # Better exact deterministic rule:
            # take current row if keeping it is necessary to hit the target under streaming constraints.
            # Simplified exact strategy:
            if needed_original > 0 and remaining_original == needed_original:
                take_org = True
            elif needed_original > 0 and written_original * remaining_original < (original_target - needed_original + 1) * original_count:
                # leave take_org as computed above when not forced
                pass

            # Use standard exact-thinning condition from remaining/needed without randomness:
            # whenever cumulative target boundary is crossed.
            # Recompute using processed count.
            processed_original = original_count - remaining_original
            next_processed = processed_original + 1
            take_org = (next_processed * original_target // original_count) > (processed_original * original_target // original_count)

            if take_org and needed_original > 0:
                try:
                    bal_row = next(bal_iter)
                    writer.writerow(bal_row)
                    written_balanced += 1
                except StopIteration:
                    pass

                writer.writerow(org_row)
                written_original += 1
                needed_original -= 1

            remaining_original -= 1

        # Flush any remaining balanced rows.
        for bal_row in bal_iter:
            writer.writerow(bal_row)
            written_balanced += 1

    print(f"Written balanced rows: {written_balanced}")
    print(f"Written original rows: {written_original}")
    print(f"Total output rows: {written_balanced + written_original}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
