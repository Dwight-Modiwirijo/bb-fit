#!/usr/bin/env python3
"""
Orchestrator scalper — auto finetune loop for scalper pipeline.

Labels: label_direction (0=Down, 1=Flat, 2=Up) from future_return_10 ±2%
Input:  sequences_scalper/ (64 timesteps × 30 v6 features, 240-min candles)
DoD:    up_prec >= 45%  (baseline random: 28.4% — model must beat it decisively)

Phases:
  0. Use last epoch warmup checkpoint
  1. Build balanced finetune CSV (1:1:1)
  2. Finetune -> evaluate -> DoD check -> repeat (max MAX_ROUNDS)

Usage:
  python orchestrate_scalper.py
  python orchestrate_scalper.py --start-round 3 --start-checkpoint /workspace/checkpoints/.../checkpoint_epoch05.pt
"""
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_HOST      = Path("/home/dwyte/bb-fit")
CKPT_BASE_HOST = Path("/home/dwyte/checkpoints/lstm_bbfit")
SCRIPTS_HOST   = Path("/home/dwyte/Github/bb-fit")
REPORT_PATH    = DATA_HOST / "orchestrator_scalper_report.md"

SEQ_DIR      = "/workspace/data/sequences_scalper"
FINETUNE_CSV = DATA_HOST / "sequences_scalper" / "lstm_train_balanced_finetune.csv"

# ── Definition of Done ─────────────────────────────────────────────────────────
DOD_UP_PREC   = 0.45   # Up precision >= 45% (baseline: 28.4%)
DOD_DOWN_PREC = 0.35   # Down precision >= 35% (baseline: 22.2%)
DOD_UP_REC    = 0.20   # Up recall >= 20%

# ── Hyperparameters ────────────────────────────────────────────────────────────
MAX_ROUNDS        = 10
EPOCHS_PER_ROUND  = 5
LR_START          = 1e-4
EDGE_WEIGHT_START = 3.0   # weight for Up and Down classes (minority)
EDGE_WEIGHT_STEP  = 1.0   # increase per round to push harder on Up/Down

DOCKER_FLAGS = [
    "docker", "run", "--rm", "--gpus", "all", "--ipc=host",
    "--ulimit", "memlock=-1", "--ulimit", "stack=67108864",
    "-v", "/home/dwyte/bb-fit:/workspace/data",
    "-v", "/home/dwyte/Github/bb-fit:/workspace/scripts",
    "-v", "/home/dwyte/checkpoints:/workspace/checkpoints",
    "nvcr.io/nvidia/pytorch:25.06-py3",
]

DOCKER_FLAGS_CPU = [
    "docker", "run", "--rm", "--ipc=host",
    "--ulimit", "memlock=-1", "--ulimit", "stack=67108864",
    "-e", "CUDA_VISIBLE_DEVICES=",
    "-v", "/home/dwyte/bb-fit:/workspace/data",
    "-v", "/home/dwyte/Github/bb-fit:/workspace/scripts",
    "-v", "/home/dwyte/checkpoints:/workspace/checkpoints",
    "nvcr.io/nvidia/pytorch:25.06-py3",
]


def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[orch {ts}] {msg}", flush=True)


def run_docker(cmd: List[str], log_path: Path, cpu_only: bool = False) -> int:
    full_cmd = (DOCKER_FLAGS_CPU if cpu_only else DOCKER_FLAGS) + cmd
    log(f"docker {' '.join(cmd[:3])} ...")
    with log_path.open("wb") as lf:
        proc = subprocess.Popen(full_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in proc.stdout:
            sys.stdout.buffer.write(line)
            sys.stdout.buffer.flush()
            lf.write(line)
        proc.wait()
    return proc.returncode


def last_epoch_checkpoint_ctr(ckpt_dir_host: Path) -> str:
    checkpoints = sorted(ckpt_dir_host.glob("checkpoint_epoch*.pt"), key=lambda p: p.stat().st_mtime)
    if not checkpoints:
        raise RuntimeError(f"No epoch checkpoints found in {ckpt_dir_host}")
    rel = checkpoints[-1].relative_to(Path("/home/dwyte/checkpoints"))
    return f"/workspace/checkpoints/{rel}"


# ── Phase 1: build finetune dataset ───────────────────────────────────────────

def build_finetune_csv() -> None:
    if FINETUNE_CSV.exists():
        log(f"Finetune CSV exists ({FINETUNE_CSV.stat().st_size // 1024 // 1024} MB), skipping.")
        return
    log("Building balanced finetune CSV (1:1:1) ...")
    cmd = [
        "python", "/workspace/scripts/build_balanced_warmup_csv.py",
        "--input",          f"{SEQ_DIR}/lstm_train_sequences.csv",
        "--output",         f"{SEQ_DIR}/lstm_train_balanced_finetune.csv",
        "--label-column",   "label_direction",
        "--majority-factor", "1",
        "--seed", "42",
    ]
    rc = run_docker(cmd, DATA_HOST / "scalper_build_finetune.log")
    if rc != 0:
        raise RuntimeError("Building finetune CSV failed")
    log(f"Finetune CSV ready: {FINETUNE_CSV}")


# ── Phase 2: finetune round ────────────────────────────────────────────────────

def run_finetune(round_num: int, start_ckpt_ctr: str, edge_weight: float, lr: float) -> Path:
    name     = f"scalper_orch_r{round_num:02d}"
    ckpt_ctr = f"/workspace/checkpoints/lstm_bbfit/{name}"
    log_path = DATA_HOST / f"{name}_train.log"

    cmd = [
        "python", "/workspace/scripts/train_lstm_bbfit.py",
        "--train-csv",      f"{SEQ_DIR}/lstm_train_balanced_finetune.csv",
        "--validation-csv", f"{SEQ_DIR}/lstm_validation_sequences.csv",
        "--test-csv",       f"{SEQ_DIR}/lstm_test_sequences.csv",
        "--label-column",   "label_direction",
        "--hidden-size", "512", "--num-layers", "3", "--dropout", "0.1",
        "--lr", str(lr),
        "--epochs", str(EPOCHS_PER_ROUND),
        "--batch-size", "256",
        "--class-weights", str(edge_weight), "1.0", str(edge_weight),
        "--focal-gamma", "2.0",
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
    return CKPT_BASE_HOST / name


def run_evaluate(round_num: int, ckpt_ctr: str) -> Tuple[dict, Path]:
    name      = f"scalper_orch_r{round_num:02d}"
    json_ctr  = f"/workspace/data/eval_{name}.json"
    json_host = DATA_HOST / f"eval_{name}.json"
    log_path  = DATA_HOST / f"{name}_eval.log"

    time.sleep(15)

    cmd = [
        "python", "/workspace/scripts/evaluate_lstm_bbfit.py",
        "--train-csv",      f"{SEQ_DIR}/lstm_train_sequences.csv",
        "--validation-csv", f"{SEQ_DIR}/lstm_validation_sequences.csv",
        "--test-csv",       f"{SEQ_DIR}/lstm_test_sequences.csv",
        "--label-column",   "label_direction",
        "--checkpoint", ckpt_ctr,
        "--hidden-size", "512", "--num-layers", "3", "--dropout", "0.1",
        "--batch-size", "32",
        "--output-json", json_ctr,
    ]
    rc = run_docker(cmd, log_path, cpu_only=True)
    if rc != 0:
        raise RuntimeError(f"Evaluate round {round_num} failed (exit {rc})")
    with json_host.open() as f:
        return json.load(f), json_host


def extract_metrics(eval_data: dict) -> dict:
    act  = eval_data["model"]["test"]["action"]
    prec = {c["class"]: c["precision"] for c in act["per_class"]}
    rec  = {c["class"]: c["recall"]    for c in act["per_class"]}
    return {
        "bal_acc":   act["balanced_accuracy"],
        "macro_f1":  act["macro_f1"],
        "down_prec": prec.get(0, 0.0),
        "flat_prec": prec.get(1, 0.0),
        "up_prec":   prec.get(2, 0.0),
        "down_rec":  rec.get(0, 0.0),
        "up_rec":    rec.get(2, 0.0),
    }


def check_dod(m: dict) -> bool:
    return (m["up_prec"]   >= DOD_UP_PREC and
            m["down_prec"] >= DOD_DOWN_PREC and
            m["up_rec"]    >= DOD_UP_REC)


def combined_score(m: dict) -> float:
    return m["up_prec"] + m["down_prec"]


# ── Report ─────────────────────────────────────────────────────────────────────

def write_report(rounds: List[dict], dod_met: bool, best_idx: int,
                 start_time: datetime) -> None:
    now    = datetime.now().strftime("%Y-%m-%d %H:%M")
    status = "✅ DoD reached!" if dod_met else f"⏳ Running — {len(rounds)} round(s) done"
    lines  = [
        "# Orchestrator Report — scalper",
        "",
        f"**Started:** {start_time.strftime('%Y-%m-%d %H:%M')}  ",
        f"**Updated:** {now}  ",
        f"**DoD:** up_prec >= {DOD_UP_PREC:.0%}, down_prec >= {DOD_DOWN_PREC:.0%}, up_rec >= {DOD_UP_REC:.0%}  ",
        f"**Status:** {status}",
        "",
        "---", "",
        "## Rounds",
        "",
        "| Round | FlatW | LR | Bal Acc | Down P | Down R | Up P | Up R | Score | DoD |",
        "|------:|------:|---:|--------:|-------:|-------:|-----:|-----:|------:|:---:|",
    ]
    for i, r in enumerate(rounds):
        star = " ★" if i == best_idx else ""
        icon = "✅" if r["dod_met"] else "❌"
        m = r["metrics"]
        lines.append(
            f"| {r['round']} | {r['edge_weight']:.1f} | {r['lr']:.0e} | "
            f"{m['bal_acc']:.1%} | {m['down_prec']:.1%} | {m['down_rec']:.1%} | "
            f"{m['up_prec']:.1%} | {m['up_rec']:.1%} | "
            f"{combined_score(m):.3f}{star} | {icon} |"
        )

    lines += ["", "---", "", "## Notes", "",
              "- **Down (0)** = price down >2% in 10 bars (40h)",
              "- **Flat (1)** = price flat ±2%",
              "- **Up   (2)** = price up >2% in 10 bars (40h)",
              f"- **Baseline** Up prec: 28.4%, Down prec: 22.2% (class frequency)",
              f"- **★** = best round (highest up_prec + down_prec)",
              "- **input_size** = 30 (29 V6_NUMERIC_FEATURES + 1 interval_code)",
              "- **label** = label_direction from future_return_10 ±2%",
              ""]

    if not dod_met and rounds:
        last = rounds[-1]
        lines += ["---", "", "## Next step", "",
                  f"- up_prec:   {last['metrics']['up_prec']:.1%}  (target: {DOD_UP_PREC:.0%})",
                  f"- up_rec:    {last['metrics']['up_rec']:.1%}  (target: {DOD_UP_REC:.0%})",
                  f"- down_prec: {last['metrics']['down_prec']:.1%}  (target: {DOD_DOWN_PREC:.0%})",
                  ""]
        if len(rounds) >= MAX_ROUNDS:
            lines.append(f"- Maximum rounds ({MAX_ROUNDS}) reached.")
        else:
            lines.append(f"- {MAX_ROUNDS - len(rounds)} round(s) remaining.")

    REPORT_PATH.write_text("\n".join(lines) + "\n")
    log(f"Report updated: {REPORT_PATH}")


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--start-round", type=int, default=1)
    p.add_argument("--start-checkpoint", default=None,
                   help="Override start checkpoint (container path)")
    return p.parse_args()


def main() -> None:
    args       = parse_args()
    start_time = datetime.now()
    rounds: List[dict] = []
    dod_met    = False
    best_idx   = 0

    log("=" * 60)
    log("Orchestrator scalper started")
    log(f"DoD: up_prec >= {DOD_UP_PREC:.0%}  AND  down_prec >= {DOD_DOWN_PREC:.0%}  AND  up_rec >= {DOD_UP_REC:.0%}")
    log(f"Max rounds: {MAX_ROUNDS}  |  Epochs per round: {EPOCHS_PER_ROUND}")
    log("label_direction: 0=Down, 1=Flat, 2=Up  (future_return_10 ±2%)")
    log("=" * 60)

    build_finetune_csv()

    if args.start_checkpoint:
        ckpt_ctr = args.start_checkpoint
    else:
        warmup_dir = CKPT_BASE_HOST / "scalper_warmup"
        ckpt_ctr   = last_epoch_checkpoint_ctr(warmup_dir)
        log(f"Using last epoch warmup checkpoint: {ckpt_ctr}")

    edge_weight   = EDGE_WEIGHT_START + (args.start_round - 1) * EDGE_WEIGHT_STEP
    lr            = LR_START
    best_score    = -1.0
    plateau_count = 0

    for round_num in range(args.start_round, MAX_ROUNDS + 1):
        log(f"\n{'='*60}")
        log(f"Round {round_num}/{MAX_ROUNDS}  |  edge_weight={edge_weight:.1f}  |  lr={lr:.0e}")

        ckpt_dir_host = run_finetune(round_num, ckpt_ctr, edge_weight, lr)
        ckpt_ctr      = last_epoch_checkpoint_ctr(ckpt_dir_host)

        eval_data, json_host = run_evaluate(round_num, ckpt_ctr)
        m = extract_metrics(eval_data)

        score   = combined_score(m)
        dod_met = check_dod(m)

        rounds.append({
            "round":       round_num,
            "edge_weight": edge_weight,
            "lr":          lr,
            "metrics":     m,
            "dod_met":     dod_met,
            "ckpt":        ckpt_ctr,
        })

        if score > best_score:
            best_score    = score
            best_idx      = len(rounds) - 1
            plateau_count = 0
            log(f"New best score: {score:.3f}  (up_prec={m['up_prec']:.1%}, down_prec={m['down_prec']:.1%})")
        else:
            plateau_count += 1
            log(f"No improvement (plateau {plateau_count}). Score={score:.3f}")
            if plateau_count >= 2:
                lr = max(lr * 0.5, 1e-6)
                log(f"Reducing LR to {lr:.0e}")
                plateau_count = 0

        write_report(rounds, dod_met, best_idx, start_time)

        if dod_met:
            log("DoD reached! Stopping.")
            break

        edge_weight += EDGE_WEIGHT_STEP

    log("Orchestrator finished.")
    log(f"Best score: {best_score:.3f} at round {rounds[best_idx]['round']}")


if __name__ == "__main__":
    main()
