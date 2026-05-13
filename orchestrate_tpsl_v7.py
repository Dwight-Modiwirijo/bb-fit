#!/usr/bin/env python3
"""
Orchestrator tpsl_v7 — auto finetune loop for the v7 feature set.

v7 vs v6:
  - Input: lstm_merged_v7.csv (14-year dataset, FiveMinutes, real TAengine indicators)
  - Feature set: 36 features (35 V7_NUMERIC_FEATURES + 1 interval_code)
  - New vs v6: is_valid_signal_int, bb_tweak_buy/sell, probe_buy/sell_count_norm, probe_growth_norm
  - DoD unchanged: long_precision >= 75%

Phases:
  0. Wait for v7_warmup/checkpoint_best.pt
  1. Build balanced finetune CSV (1:1:1)
  2. Finetune -> evaluate -> DoD check -> repeat (max MAX_ROUNDS)

Usage:
  python orchestrate_tpsl_v7.py                    # normal start
  python orchestrate_tpsl_v7.py --skip-warmup-wait # if warmup already done
  python orchestrate_tpsl_v7.py --start-round 3 --start-checkpoint /workspace/checkpoints/.../checkpoint_epoch05.pt
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
REPORT_PATH    = DATA_HOST / "orchestrator_v7_report.md"

WARMUP_CKPT_HOST = CKPT_BASE_HOST / "v7_warmup" / "checkpoint_best.pt"

SEQ_DIR      = "/workspace/data/sequences_tpsl_v7"
FINETUNE_CSV = DATA_HOST / "sequences_tpsl_v7" / "lstm_train_balanced_finetune.csv"

# ── Definition of Done ─────────────────────────────────────────────────────────
DOD_LONG_PREC  = 0.75   # break-even is 69.2% at TP=0.8%/SL=1.8%
DOD_SHORT_PREC = 0.10   # sanity check only — bot is Long-only
DOD_LONG_REC   = 0.15

# ── Hyperparameters ────────────────────────────────────────────────────────────
MAX_ROUNDS        = 10
EPOCHS_PER_ROUND  = 5
LR_START          = 1e-4
HOLD_WEIGHT_START = 3.0
HOLD_WEIGHT_STEP  = 2.0
PLATEAU_MIN_DELTA = 0.005
WARMUP_POLL_SEC   = 60

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


def best_checkpoint_ctr(ckpt_dir_host: Path) -> str:
    best = ckpt_dir_host / "checkpoint_best.pt"
    if best.exists():
        rel = best.relative_to(Path("/home/dwyte/checkpoints"))
        return f"/workspace/checkpoints/{rel}"
    checkpoints = sorted(ckpt_dir_host.glob("checkpoint_epoch*.pt"), key=lambda p: p.stat().st_mtime)
    if not checkpoints:
        raise RuntimeError(f"No checkpoints found in {ckpt_dir_host}")
    rel = checkpoints[-1].relative_to(Path("/home/dwyte/checkpoints"))
    return f"/workspace/checkpoints/{rel}"


def last_epoch_checkpoint_ctr(ckpt_dir_host: Path) -> str:
    """Return the most recent epoch checkpoint (never checkpoint_best.pt).

    Warmup checkpoint_best.pt is nearly random because the raw validation set
    is 97% Hold — the model is rewarded for predicting Hold on step 1.
    Always start finetune from the last epoch checkpoint.
    """
    checkpoints = sorted(ckpt_dir_host.glob("checkpoint_epoch*.pt"), key=lambda p: p.stat().st_mtime)
    if not checkpoints:
        raise RuntimeError(f"No epoch checkpoints found in {ckpt_dir_host}")
    rel = checkpoints[-1].relative_to(Path("/home/dwyte/checkpoints"))
    return f"/workspace/checkpoints/{rel}"


# ── Phase 0: wait for warmup ───────────────────────────────────────────────────

def wait_for_warmup() -> None:
    if WARMUP_CKPT_HOST.exists():
        log(f"Warmup checkpoint found: {WARMUP_CKPT_HOST}")
        return
    log(f"Waiting for warmup checkpoint ({WARMUP_CKPT_HOST}) ...")
    polls = 0
    while not WARMUP_CKPT_HOST.exists():
        time.sleep(WARMUP_POLL_SEC)
        polls += 1
        if polls % 10 == 0:
            log(f"Still waiting ... ({polls * WARMUP_POLL_SEC // 60} minutes)")
    log(f"Warmup done: {WARMUP_CKPT_HOST}")


# ── Phase 1: build finetune dataset ───────────────────────────────────────────

def build_finetune_csv() -> None:
    if FINETUNE_CSV.exists():
        log(f"Finetune CSV exists ({FINETUNE_CSV.stat().st_size // 1024 // 1024} MB), skipping.")
        return
    log("Building balanced finetune CSV (1:1:1) ...")
    cmd = [
        "python", "/workspace/scripts/build_balanced_warmup_csv.py",
        "--input",  f"{SEQ_DIR}/lstm_train_sequences.csv",
        "--output", f"{SEQ_DIR}/lstm_train_balanced_finetune.csv",
        "--majority-factor", "1",
        "--seed", "42",
    ]
    rc = run_docker(cmd, DATA_HOST / "v7_build_finetune.log")
    if rc != 0:
        raise RuntimeError("Building finetune CSV failed")
    log(f"Finetune CSV ready: {FINETUNE_CSV}")


# ── Phase 2: finetune round ────────────────────────────────────────────────────

def run_finetune(round_num: int, start_ckpt_ctr: str, hold_weight: float, lr: float) -> Path:
    name     = f"v7_orch_r{round_num:02d}"
    ckpt_ctr = f"/workspace/checkpoints/lstm_bbfit/{name}"
    log_path = DATA_HOST / f"{name}_train.log"

    cmd = [
        "python", "/workspace/scripts/train_lstm_bbfit.py",
        "--train-csv",      f"{SEQ_DIR}/lstm_train_balanced_finetune.csv",
        "--validation-csv", f"{SEQ_DIR}/lstm_validation_sequences.csv",
        "--test-csv",       f"{SEQ_DIR}/lstm_test_sequences.csv",
        "--hidden-size", "512", "--num-layers", "3", "--dropout", "0.1",
        "--lr", str(lr),
        "--epochs", str(EPOCHS_PER_ROUND),
        "--batch-size", "256",
        "--class-weights", "1.0", str(hold_weight), "1.0",
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
    name      = f"v7_orch_r{round_num:02d}"
    json_ctr  = f"/workspace/data/eval_{name}.json"
    json_host = DATA_HOST / f"eval_{name}.json"
    log_path  = DATA_HOST / f"{name}_eval.log"

    time.sleep(15)  # let GPU memory release after finetune container exits

    cmd = [
        "python", "/workspace/scripts/evaluate_lstm_bbfit.py",
        "--train-csv",      f"{SEQ_DIR}/lstm_train_sequences.csv",
        "--validation-csv", f"{SEQ_DIR}/lstm_validation_sequences.csv",
        "--test-csv",       f"{SEQ_DIR}/lstm_test_sequences.csv",
        "--checkpoint", ckpt_ctr,
        "--hidden-size", "512", "--num-layers", "3", "--dropout", "0.1",
        "--batch-size", "128",
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
        "bal_acc":    act["balanced_accuracy"],
        "macro_f1":   act["macro_f1"],
        "short_prec": prec.get(0, 0.0),
        "hold_prec":  prec.get(1, 0.0),
        "long_prec":  prec.get(2, 0.0),
        "short_rec":  rec.get(0, 0.0),
        "long_rec":   rec.get(2, 0.0),
    }


def check_dod(m: dict) -> bool:
    return (m["long_prec"] >= DOD_LONG_PREC and
            m["short_prec"] >= DOD_SHORT_PREC and
            m["long_rec"] >= DOD_LONG_REC)


def combined_score(m: dict) -> float:
    return m["long_prec"] + m["short_prec"]


# ── Report ─────────────────────────────────────────────────────────────────────

def write_report(rounds: List[dict], dod_met: bool, best_idx: int,
                 start_time: datetime) -> None:
    now    = datetime.now().strftime("%Y-%m-%d %H:%M")
    status = "✅ DoD reached!" if dod_met else f"⏳ Running — {len(rounds)} round(s) done"
    lines  = [
        "# Orchestrator Report — tpsl_v7",
        "",
        f"**Started:** {start_time.strftime('%Y-%m-%d %H:%M')}  ",
        f"**Updated:** {now}  ",
        f"**DoD:** long_prec >= {DOD_LONG_PREC:.0%}, short_prec >= {DOD_SHORT_PREC:.0%}, long_rec >= {DOD_LONG_REC:.0%}  ",
        f"**Status:** {status}",
        "",
        "---", "",
        "## Rounds",
        "",
        "| Round | HoldW | LR | Bal Acc | Short P | Short R | Long P | Long R | Score | DoD |",
        "|------:|------:|---:|--------:|--------:|--------:|-------:|-------:|------:|:---:|",
    ]
    for i, r in enumerate(rounds):
        star = " ★" if i == best_idx else ""
        icon = "✅" if r["dod_met"] else "❌"
        m = r["metrics"]
        lines.append(
            f"| {r['round']} | {r['hold_weight']:.1f} | {r['lr']:.0e} | "
            f"{m['bal_acc']:.1%} | {m['short_prec']:.1%} | {m['short_rec']:.1%} | "
            f"{m['long_prec']:.1%} | {m['long_rec']:.1%} | "
            f"{combined_score(m):.3f}{star} | {icon} |"
        )

    lines += ["", "---", "", "## Notes", "",
              "- **Short** = class 0 (close Long position — bot is Long-only)",
              "- **Long**  = class 2 (open Long position)",
              f"- **Break-even** at TP=0.8%/SL=1.8% = 69.2% long precision",
              f"- **★** = best round (highest long_prec + short_prec)",
              "- **input_size** = 30 (29 V6_NUMERIC_FEATURES + 1 interval_code)",
              ""]

    if not dod_met and rounds:
        last = rounds[-1]
        lines += ["---", "", "## Next step", "",
                  f"- long_prec:  {last['metrics']['long_prec']:.1%}  (target: {DOD_LONG_PREC:.0%})",
                  f"- long_rec:   {last['metrics']['long_rec']:.1%}  (target: {DOD_LONG_REC:.0%})",
                  f"- short_prec: {last['metrics']['short_prec']:.1%}  (target: {DOD_SHORT_PREC:.0%})",
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
    p.add_argument("--skip-warmup-wait", action="store_true")
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
    log("Orchestrator tpsl_v7 started")
    log(f"DoD: long_prec >= {DOD_LONG_PREC:.0%}  AND  short_prec >= {DOD_SHORT_PREC:.0%}")
    log(f"Max rounds: {MAX_ROUNDS}  |  Epochs per round: {EPOCHS_PER_ROUND}")
    log("=" * 60)

    if not args.skip_warmup_wait:
        wait_for_warmup()
    else:
        log("--skip-warmup-wait: skipped warmup wait")

    build_finetune_csv()

    if args.start_checkpoint:
        ckpt_ctr = args.start_checkpoint
    else:
        warmup_dir = CKPT_BASE_HOST / "v7_warmup"
        ckpt_ctr   = last_epoch_checkpoint_ctr(warmup_dir)
        log(f"Using last epoch warmup checkpoint: {ckpt_ctr}")

    hold_weight   = HOLD_WEIGHT_START + (args.start_round - 1) * HOLD_WEIGHT_STEP
    lr            = LR_START
    best_score    = -1.0
    plateau_count = 0

    for round_num in range(args.start_round, MAX_ROUNDS + 1):
        log(f"\n── Round {round_num}/{MAX_ROUNDS} | hold_weight={hold_weight:.1f} | lr={lr:.0e} ──")

        ckpt_dir_host = run_finetune(round_num, ckpt_ctr, hold_weight, lr)
        ckpt_ctr      = best_checkpoint_ctr(ckpt_dir_host)
        log(f"Best checkpoint: {ckpt_ctr}")

        eval_data, _ = run_evaluate(round_num, ckpt_ctr)
        metrics      = extract_metrics(eval_data)
        dod_met      = check_dod(metrics)
        score        = combined_score(metrics)

        rounds.append({
            "round":       round_num,
            "hold_weight": hold_weight,
            "lr":          lr,
            "metrics":     metrics,
            "dod_met":     dod_met,
            "checkpoint":  ckpt_ctr,
        })

        if score > best_score:
            best_score    = score
            best_idx      = len(rounds) - 1
            plateau_count = 0
        else:
            plateau_count += 1

        log(f"Round {round_num}: bal={metrics['bal_acc']:.3f} | "
            f"short P={metrics['short_prec']:.3f} R={metrics['short_rec']:.3f} | "
            f"long  P={metrics['long_prec']:.3f} R={metrics['long_rec']:.3f} | "
            f"DoD={'YES ✅' if dod_met else 'NO ❌'}")
        log(f"Score={score:.3f}  Best={best_score:.3f}  Plateau={plateau_count}")

        write_report(rounds, dod_met, best_idx, start_time)

        if dod_met:
            log("DoD reached! Done.")
            break

        if plateau_count >= 2:
            log("Plateau detected (2 rounds without improvement).")
            best_ckpt_ctr = rounds[best_idx]["checkpoint"]
            log(f"Resetting to best checkpoint: {best_ckpt_ctr}")
            ckpt_ctr      = best_ckpt_ctr
            lr            = lr / 2.0
            plateau_count = 0
            log(f"LR reduced to {lr:.0e}")
        else:
            hold_weight += HOLD_WEIGHT_STEP

    if not dod_met:
        log(f"\nDoD not reached after {len(rounds)} round(s).")
        log(f"Best round: {rounds[best_idx]['round']} "
            f"(long={rounds[best_idx]['metrics']['long_prec']:.1%}, "
            f"short={rounds[best_idx]['metrics']['short_prec']:.1%})")
        write_report(rounds, False, best_idx, start_time)

    log(f"\nDone. Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
