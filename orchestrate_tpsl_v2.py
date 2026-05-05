#!/usr/bin/env python3
"""
Orchestrator tpsl_v2: autonomous finetune -> evaluate -> DoD check -> repeat.

Logic per round:
  1. Finetune (resume from previous best checkpoint, higher hold weight)
  2. Evaluate -> per-class precision via threshold sweep
  3. DoD check: long_prec >= 0.25 AND short_prec >= 0.25 at any threshold
  4. If met  -> backtest -> report -> done
  5. If not  -> hold weight += 5, next round

Updates orchestrator_report.md after every round.

Usage:
    python orchestrate_tpsl_v2.py \
        --start-checkpoint /workspace/checkpoints/lstm_bbfit/tpsl_v2_finetune_02/checkpoint_best.pt \
        --start-hold-weight 15.0
"""
import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ─── Paths ────────────────────────────────────────────────────────────────────
DATA_HOST       = Path("/home/dwyte/bb-fit")
CKPT_BASE_HOST  = Path("/home/dwyte/checkpoints/lstm_bbfit")
REPORT_PATH     = DATA_HOST / "orchestrator_report.md"

# ─── Definition of Done ───────────────────────────────────────────────────────
DOD_LONG_PREC  = 0.25
DOD_SHORT_PREC = 0.25
MAX_ROUNDS     = 5

# ─── Training defaults ────────────────────────────────────────────────────────
EPOCHS_PER_ROUND  = 10
LR                = 1e-4
HOLD_WEIGHT_STEP  = 5.0

DOCKER_FLAGS = [
    "docker", "run", "--rm", "--gpus", "all", "--ipc=host",
    "--ulimit", "memlock=-1", "--ulimit", "stack=67108864",
    "-v", "/home/dwyte/bb-fit:/workspace/data",
    "-v", "/home/dwyte/Github/bb-fit:/workspace/scripts",
    "-v", "/home/dwyte/checkpoints:/workspace/checkpoints",
    "nvcr.io/nvidia/pytorch:25.06-py3",
]


def run_docker(cmd: List[str], log_path: Path) -> int:
    full_cmd = DOCKER_FLAGS + cmd
    print(f"\n[orch] {' '.join(cmd[:4])} ...")
    with log_path.open("wb") as lf:
        proc = subprocess.Popen(full_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in proc.stdout:
            sys.stdout.buffer.write(line)
            sys.stdout.buffer.flush()
            lf.write(line)
        proc.wait()
    return proc.returncode


def run_finetune(round_num: int, start_ckpt_ctr: str, hold_weight: float) -> Path:
    name        = f"tpsl_v2_orch_r{round_num:02d}"
    ckpt_ctr    = f"/workspace/checkpoints/lstm_bbfit/{name}"
    ckpt_host   = CKPT_BASE_HOST / name
    log_path    = DATA_HOST / f"{name}_train.log"

    cmd = [
        "python", "/workspace/scripts/train_lstm_bbfit.py",
        "--train-csv",      "/workspace/data/sequences_tpsl_v2/lstm_train_balanced_finetune.csv",
        "--validation-csv", "/workspace/data/sequences_tpsl_v2/lstm_validation_sequences.csv",
        "--test-csv",       "/workspace/data/sequences_tpsl_v2/lstm_test_sequences.csv",
        "--hidden-size", "512", "--num-layers", "3", "--dropout", "0.1",
        "--lr", str(LR), "--epochs", str(EPOCHS_PER_ROUND), "--batch-size", "256",
        "--class-weights", "1.0", str(hold_weight), "1.0",
        "--resume-checkpoint", start_ckpt_ctr,
        "--reset-optimizer",
        "--checkpoint-dir", ckpt_ctr,
        "--checkpoint-every-steps", "200",
        "--lr-scheduler", "plateau",
        "--scheduler-patience", "2",
    ]
    rc = run_docker(cmd, log_path)
    if rc != 0:
        raise RuntimeError(f"Finetune round {round_num} failed (exit {rc})")
    return ckpt_host


def best_checkpoint_ctr(ckpt_dir_host: Path) -> str:
    # Prefer checkpoint_best.pt saved by train script; fall back to last epoch checkpoint
    best = ckpt_dir_host / "checkpoint_best.pt"
    if best.exists():
        rel = best.relative_to(Path("/home/dwyte/checkpoints"))
        return f"/workspace/checkpoints/{rel}"
    checkpoints = sorted(ckpt_dir_host.glob("checkpoint_epoch*.pt"))
    if not checkpoints:
        raise RuntimeError(f"No checkpoints found in {ckpt_dir_host}")
    rel = checkpoints[-1].relative_to(Path("/home/dwyte/checkpoints"))
    return f"/workspace/checkpoints/{rel}"


def run_evaluate(round_num: int, ckpt_ctr: str) -> Tuple[dict, Path]:
    name      = f"tpsl_v2_orch_r{round_num:02d}"
    json_ctr  = f"/workspace/data/eval_{name}.json"
    json_host = DATA_HOST / f"eval_{name}.json"
    log_path  = DATA_HOST / f"{name}_eval.log"

    cmd = [
        "python", "/workspace/scripts/evaluate_lstm_bbfit.py",
        "--train-csv",      "/workspace/data/sequences_tpsl_v2/lstm_train_sequences.csv",
        "--validation-csv", "/workspace/data/sequences_tpsl_v2/lstm_validation_sequences.csv",
        "--test-csv",       "/workspace/data/sequences_tpsl_v2/lstm_test_sequences.csv",
        "--checkpoint", ckpt_ctr,
        "--hidden-size", "512", "--num-layers", "3", "--dropout", "0.1",
        "--output-json", json_ctr,
    ]
    rc = run_docker(cmd, log_path)
    if rc != 0:
        raise RuntimeError(f"Evaluate round {round_num} failed (exit {rc})")
    with json_host.open() as f:
        return json.load(f), json_host


def check_dod(eval_data: dict) -> Tuple[bool, Optional[dict]]:
    sweep = eval_data.get("threshold_sweep", {}).get("test", [])
    best = None
    for entry in sweep:
        lp = entry.get("long_precision", 0)
        sp = entry.get("short_precision", 0)
        if lp >= DOD_LONG_PREC and sp >= DOD_SHORT_PREC:
            if best is None or (lp + sp) > (best["long_precision"] + best["short_precision"]):
                best = entry
    return (best is not None), best


def run_backtest(ckpt_ctr: str, round_num: int) -> dict:
    name      = f"tpsl_v2_orch_r{round_num:02d}"
    json_ctr  = f"/workspace/data/backtest_{name}.json"
    csv_ctr   = f"/workspace/data/backtest_{name}_equity.csv"
    json_host = DATA_HOST / f"backtest_{name}.json"
    log_path  = DATA_HOST / f"backtest_{name}.log"

    cmd = [
        "python", "/workspace/scripts/backtest_lstm_bbfit.py",
        "--test-csv",   "/workspace/data/sequences_tpsl_v2/lstm_test_sequences.csv",
        "--checkpoint", ckpt_ctr,
        "--hidden-size", "512", "--num-layers", "3", "--dropout", "0.1",
        "--initial-capital", "10000",
        "--tp-pct", "0.036", "--sl-pct", "0.012", "--fee", "0.0018",
        "--output-json", json_ctr,
        "--output-csv",  csv_ctr,
    ]
    rc = run_docker(cmd, log_path)
    if rc != 0:
        raise RuntimeError(f"Backtest failed (exit {rc})")
    with json_host.open() as f:
        return json.load(f)


def write_report(rounds: List[dict], dod_met: bool,
                 backtest: Optional[dict], start_time: datetime) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    status = "✅ DoD reached" if dod_met else f"⏳ In progress — {len(rounds)} round(s) done"
    lines = [
        "# Orchestrator Report — tpsl_v2",
        "",
        f"**Started:** {start_time.strftime('%Y-%m-%d %H:%M')}  ",
        f"**Updated:** {now}  ",
        f"**DoD:** long_precision ≥ {DOD_LONG_PREC:.0%} AND short_precision ≥ {DOD_SHORT_PREC:.0%}  ",
        f"**Status:** {status}",
        "",
        "---",
        "",
        "## Rounds",
        "",
        "| Round | Hold Weight | Val Balanced Acc | Long Prec | Short Prec | Best Threshold | DoD |",
        "|------:|------------:|-----------------:|----------:|-----------:|---------------:|:---:|",
    ]

    for r in rounds:
        bp     = r.get("best_precision_entry")
        lp     = f"{bp['long_precision']:.1%}"  if bp else "—"
        sp     = f"{bp['short_precision']:.1%}" if bp else "—"
        thresh = f"{bp['hold_threshold']:.2f}"  if bp else "—"
        icon   = "✅" if r["dod_met"] else "❌"
        lines.append(
            f"| {r['round']} | {r['hold_weight']:.0f} | "
            f"{r['val_acc']:.1%} | {lp} | {sp} | {thresh} | {icon} |"
        )

    if dod_met and backtest:
        bt = backtest["best"]
        lines += [
            "",
            "---",
            "",
            "## Backtest Result",
            "",
            "| Metric | Value |",
            "|:-------|------:|",
            f"| Hold threshold | {bt['hold_threshold']} |",
            f"| Starting capital | €{bt['initial_capital']:,.0f} |",
            f"| Final capital | **€{bt['final_capital']:,.2f}** |",
            f"| Growth | **{bt['growth_pct']:+.1f}%** |",
            f"| Total trades | {bt['total_trades']} |",
            f"| Wins | {bt['wins']} |",
            f"| Losses | {bt['losses']} |",
            f"| Win rate | {bt['win_rate']:.1f}% |",
        ]

    if not dod_met and rounds:
        last = rounds[-1]
        bp   = last.get("best_precision_entry")
        lines += ["", "---", "", "## Next Steps", ""]
        if bp:
            lines += [
                f"- Best threshold so far: {bp['hold_threshold']:.2f}",
                f"  - Long precision:  {bp['long_precision']:.1%} (needed: {DOD_LONG_PREC:.0%})",
                f"  - Short precision: {bp['short_precision']:.1%} (needed: {DOD_SHORT_PREC:.0%})",
            ]
        next_weight = last["hold_weight"] + HOLD_WEIGHT_STEP
        if len(rounds) < MAX_ROUNDS:
            lines.append(
                f"- Next round: hold_weight={next_weight:.0f}, "
                f"{MAX_ROUNDS - len(rounds)} round(s) remaining."
            )
        else:
            lines += [
                f"- Maximum rounds ({MAX_ROUNDS}) reached.",
                "- Consider: increasing max_rounds, different architecture, or lowering DoD threshold.",
            ]

    REPORT_PATH.write_text("\n".join(lines) + "\n")
    print(f"[orch] Report updated: {REPORT_PATH}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--start-checkpoint", required=True,
                   help="Container path to starting checkpoint")
    p.add_argument("--start-hold-weight", type=float, default=15.0,
                   help="Hold class weight for round 1 (increases by 5 each round)")
    p.add_argument("--max-rounds", type=int, default=MAX_ROUNDS)
    return p.parse_args()


def main() -> None:
    args        = parse_args()
    start_time  = datetime.now()
    ckpt_ctr    = args.start_checkpoint
    hold_weight = args.start_hold_weight
    rounds: List[dict] = []
    dod_met     = False
    backtest_result: Optional[dict] = None

    print(f"[orch] ============================================================")
    print(f"[orch] Orchestrator tpsl_v2 started: {start_time.strftime('%Y-%m-%d %H:%M')}")
    print(f"[orch] Start checkpoint : {ckpt_ctr}")
    print(f"[orch] Start hold weight: {hold_weight}")
    print(f"[orch] DoD              : long_prec >= {DOD_LONG_PREC} AND short_prec >= {DOD_SHORT_PREC}")
    print(f"[orch] Max rounds       : {args.max_rounds}")
    print(f"[orch] ============================================================")

    for round_num in range(1, args.max_rounds + 1):
        print(f"\n[orch] ── Round {round_num}/{args.max_rounds} | hold_weight={hold_weight} ──")

        ckpt_dir_host = run_finetune(round_num, ckpt_ctr, hold_weight)
        ckpt_ctr      = best_checkpoint_ctr(ckpt_dir_host)
        print(f"[orch] Best checkpoint: {ckpt_ctr}")

        eval_data, _ = run_evaluate(round_num, ckpt_ctr)
        val_acc      = eval_data["model"]["validation"]["action"]["balanced_accuracy"]
        dod_met, best_entry = check_dod(eval_data)

        rounds.append({
            "round": round_num,
            "hold_weight": hold_weight,
            "val_acc": val_acc,
            "dod_met": dod_met,
            "best_precision_entry": best_entry,
            "checkpoint": ckpt_ctr,
        })

        print(f"[orch] Round {round_num}: balanced_acc={val_acc:.3f} | DoD={'YES ✅' if dod_met else 'NO ❌'}")
        if best_entry:
            print(f"[orch] Best threshold: {best_entry['hold_threshold']:.2f} | "
                  f"long_prec={best_entry['long_precision']:.3f} "
                  f"short_prec={best_entry['short_precision']:.3f}")

        write_report(rounds, dod_met, None, start_time)

        if dod_met:
            print(f"\n[orch] DoD reached! Running backtest ...")
            backtest_result = run_backtest(ckpt_ctr, round_num)
            bt = backtest_result["best"]
            print(f"[orch] Backtest: {bt['initial_capital']} -> {bt['final_capital']:,.2f} "
                  f"({bt['growth_pct']:+.1f}%) | win_rate={bt['win_rate']:.1f}%")
            write_report(rounds, True, backtest_result, start_time)
            break

        hold_weight += HOLD_WEIGHT_STEP

    if not dod_met:
        print(f"\n[orch] DoD not reached after {len(rounds)} round(s).")
        write_report(rounds, False, None, start_time)

    print(f"\n[orch] Done. Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
