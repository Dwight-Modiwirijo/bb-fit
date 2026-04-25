#!/usr/bin/env python3
import json
from pathlib import Path

LEFT = Path("/home/dwyte/checkpoints/lstm_bbfit/classification_warmup_03/eval_step196.json")
RIGHT = Path("/home/dwyte/checkpoints/lstm_bbfit/mixed_finetune_01/evals/eval_step0200.json")

def load_report(path: Path):
    data = json.loads(path.read_text())
    return data["model"]

def fmt(x):
    if isinstance(x, float):
        return f"{x:.6f}"
    return str(x)

def get_metrics(model_block, split, head):
    d = model_block[split][head]
    return {
        "loss": model_block[split]["loss"],
        "balanced_accuracy": d["balanced_accuracy"],
        "macro_f1": d["macro_f1"],
        "recall_class_0": d["per_class"][0]["recall"],
        "recall_class_1": d["per_class"][1]["recall"],
        "recall_class_2": d["per_class"][2]["recall"],
    }

left = load_report(LEFT)
right = load_report(RIGHT)

print(f"LEFT : {LEFT}")
print(f"RIGHT: {RIGHT}")

for split in ["validation", "test"]:
    print(f"\n=== {split.upper()} ===")
    for head in ["action", "trade_side"]:
        print(f"\n[{head}]")
        lm = get_metrics(left, split, head)
        rm = get_metrics(right, split, head)

        for key in [
            "loss",
            "balanced_accuracy",
            "macro_f1",
            "recall_class_0",
            "recall_class_1",
            "recall_class_2",
        ]:
            print(
                f"{key:18} "
                f"left={fmt(lm[key]):>10} "
                f"right={fmt(rm[key]):>10} "
                f"delta(right-left)={rm[key] - lm[key]:>10.6f}"
            )

print("\nInterpretatie:")
print("- LEFT  = warmup step196")
print("- RIGHT = mixed finetune step200")
print("- Focus op balanced_accuracy, macro_f1, recall_class_0, recall_class_2")
