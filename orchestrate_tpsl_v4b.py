#!/usr/bin/env python3
"""
Orchestrator tpsl_v4b — volledig automatisch, nul handmatige stappen nodig.

Fases:
  0. Wacht op v4b_warmup/checkpoint_best.pt  (warmup loopt al)
  1. Bouw balanced finetune CSV (1:1:1) als die nog niet bestaat
  2. Finetune -> evaluate -> DoD check -> herhaal (max MAX_ROUNDS rondes)
     - hold_weight (Hold) stijgt elke ronde met HOLD_WEIGHT_STEP, Long/Short blijft 1.0
     - Als precision plateaut (< PLATEAU_MIN_DELTA over 2 rondes): reset naar
       beste checkpoint en halveer de LR
  3. Rapport in orchestrator_v4b_report.md na elke ronde

DoD (Definition of Done):
  long_precision  >= DOD_LONG_PREC   (break-even bij 2:1 = 33%; doel iets erboven)
  short_precision >= DOD_SHORT_PREC

Usage:
  python orchestrate_tpsl_v4b.py          # start normaal
  python orchestrate_tpsl_v4b.py --skip-warmup-wait   # als warmup al klaar is
"""
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_HOST      = Path("/home/dwyte/bb-fit")
CKPT_BASE_HOST = Path("/home/dwyte/checkpoints/lstm_bbfit")
SCRIPTS_HOST   = Path("/home/dwyte/Github/bb-fit")
REPORT_PATH    = DATA_HOST / "orchestrator_v4b_report.md"

WARMUP_CKPT_HOST = CKPT_BASE_HOST / "v4b_warmup" / "checkpoint_best.pt"
WARMUP_CKPT_CTR  = "/workspace/checkpoints/lstm_bbfit/v4b_warmup/checkpoint_best.pt"

SEQ_DIR         = "/workspace/data/sequences_tpsl_v4b"
FINETUNE_CSV    = DATA_HOST / "sequences_tpsl_v4b" / "lstm_train_balanced_finetune.csv"

# ── Definition of Done ────────────────────────────────────────────────────────
DOD_LONG_PREC  = 0.35   # long  (sell) precision  >= 35%  (break-even is 33% bij 2:1)
DOD_SHORT_PREC = 0.25   # short (buy)  precision  >= 25%

# ── Training hyperparameters ──────────────────────────────────────────────────
MAX_ROUNDS        = 10
EPOCHS_PER_ROUND  = 5
LR_START          = 1e-4
HOLD_WEIGHT_START = 3.0   # Hold class weight (Long/Short stay 1.0)
HOLD_WEIGHT_STEP  = 2.0
PLATEAU_MIN_DELTA = 0.005   # als verbetering < 0.5% over 2 rondes → plateau
WARMUP_POLL_SEC   = 60      # hoe vaak op warmup checkpoint pollen

DOCKER_FLAGS = [
    "docker", "run", "--rm", "--gpus", "all", "--ipc=host",
    "--ulimit", "memlock=-1", "--ulimit", "stack=67108864",
    "-v", "/home/dwyte/bb-fit:/workspace/data",
    "-v", "/home/dwyte/Github/bb-fit:/workspace/scripts",
    "-v", "/home/dwyte/checkpoints:/workspace/checkpoints",
    "nvcr.io/nvidia/pytorch:25.06-py3",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[orch {ts}] {msg}", flush=True)


def run_docker(cmd: List[str], log_path: Path) -> int:
    full_cmd = DOCKER_FLAGS + cmd
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
        raise RuntimeError(f"Geen checkpoints gevonden in {ckpt_dir_host}")
    rel = checkpoints[-1].relative_to(Path("/home/dwyte/checkpoints"))
    return f"/workspace/checkpoints/{rel}"


# ── Fase 0: wacht op warmup ───────────────────────────────────────────────────

def wait_for_warmup() -> None:
    if WARMUP_CKPT_HOST.exists():
        log(f"Warmup checkpoint gevonden: {WARMUP_CKPT_HOST}")
        return
    log(f"Wachten op warmup checkpoint ({WARMUP_CKPT_HOST}) ...")
    log(f"Polling elke {WARMUP_POLL_SEC}s. Ctrl+C om te stoppen.")
    
    polls = 0
    while not WARMUP_CKPT_HOST.exists():
        time.sleep(WARMUP_POLL_SEC)
        polls += 1
        if WARMUP_CKPT_HOST.exists():
            break
        if polls % 10 == 0:
            log(f"Warmup nog bezig ... (al {polls * WARMUP_POLL_SEC // 60} minuten aan het wachten)")
    log(f"Warmup klaar: {WARMUP_CKPT_HOST}")


# ── Fase 1: finetune dataset bouwen ──────────────────────────────────────────

def build_finetune_csv() -> None:
    if FINETUNE_CSV.exists():
        log(f"Finetune CSV bestaat al ({FINETUNE_CSV.stat().st_size // 1024 // 1024} MB), skip.")
        return
    log("Bouwen balanced finetune CSV (1:1:1) ...")
    cmd = [
        "python", "/workspace/scripts/build_balanced_warmup_csv.py",
        "--input",  f"{SEQ_DIR}/lstm_train_sequences.csv",
        "--output", f"{SEQ_DIR}/lstm_train_balanced_finetune.csv",
        "--majority-factor", "1",
        "--seed", "42",
    ]
    rc = run_docker(cmd, DATA_HOST / "v4b_build_finetune.log")
    if rc != 0:
        raise RuntimeError("Bouwen finetune CSV mislukt")
    log(f"Finetune CSV klaar: {FINETUNE_CSV}")


# ── Fase 2: finetune ronde ────────────────────────────────────────────────────

def run_finetune(round_num: int, start_ckpt_ctr: str, hold_weight: float, lr: float) -> Path:
    name     = f"v4b_orch_r{round_num:02d}"
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
        raise RuntimeError(f"Finetune ronde {round_num} mislukt (exit {rc})")
    return CKPT_BASE_HOST / name


def run_evaluate(round_num: int, ckpt_ctr: str) -> Tuple[dict, Path]:
    name      = f"v4b_orch_r{round_num:02d}"
    json_ctr  = f"/workspace/data/eval_{name}.json"
    json_host = DATA_HOST / f"eval_{name}.json"
    log_path  = DATA_HOST / f"{name}_eval.log"

    cmd = [
        "python", "/workspace/scripts/evaluate_lstm_bbfit.py",
        "--train-csv",      f"{SEQ_DIR}/lstm_train_sequences.csv",
        "--validation-csv", f"{SEQ_DIR}/lstm_validation_sequences.csv",
        "--test-csv",       f"{SEQ_DIR}/lstm_test_sequences.csv",
        "--checkpoint", ckpt_ctr,
        "--hidden-size", "512", "--num-layers", "3", "--dropout", "0.1",
        "--output-json", json_ctr,
    ]
    rc = run_docker(cmd, log_path)
    if rc != 0:
        raise RuntimeError(f"Evaluate ronde {round_num} mislukt (exit {rc})")
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
    return m["long_prec"] >= DOD_LONG_PREC and m["short_prec"] >= DOD_SHORT_PREC


def combined_score(m: dict) -> float:
    """Enkelvoudige score om de beste ronde te kiezen."""
    return m["long_prec"] + m["short_prec"]


# ── Rapport ───────────────────────────────────────────────────────────────────

def write_report(rounds: List[dict], dod_met: bool, best_idx: int,
                 start_time: datetime) -> None:
    now    = datetime.now().strftime("%Y-%m-%d %H:%M")
    status = "✅ DoD bereikt!" if dod_met else f"⏳ Bezig — {len(rounds)} ronde(s) gedaan"
    lines  = [
        "# Orchestrator Report — tpsl_v4b",
        "",
        f"**Gestart:** {start_time.strftime('%Y-%m-%d %H:%M')}  ",
        f"**Bijgewerkt:** {now}  ",
        f"**DoD:** long_precision >= {DOD_LONG_PREC:.0%} EN short_precision >= {DOD_SHORT_PREC:.0%}  ",
        f"**Status:** {status}",
        "",
        "---", "",
        "## Rondes",
        "",
        "| Ronde | HoldW | LR | Bal Acc | Short P | Short R | Long P | Long R | Score | DoD |",
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

    lines += ["", "---", "", "## Uitleg", "",
              "- **Short** = class 0 (was -1 in brondata)",
              "- **Long**  = class 2 (was +1 in brondata)",
              f"- **Break-even** bij 2:1 ratio (TP=2.4%/SL=1.2%) = 33% precision",
              f"- **★** = beste ronde (hoogste long_prec + short_prec)",
              ""]

    if not dod_met and rounds:
        last = rounds[-1]
        lines += ["---", "", "## Volgende stap", "",
                  f"- long_prec:  {last['metrics']['long_prec']:.1%} (doel: {DOD_LONG_PREC:.0%})",
                  f"- short_prec: {last['metrics']['short_prec']:.1%} (doel: {DOD_SHORT_PREC:.0%})",
                  ""]
        if len(rounds) >= MAX_ROUNDS:
            lines.append(f"- Maximum rondes ({MAX_ROUNDS}) bereikt.")
        else:
            lines.append(f"- {MAX_ROUNDS - len(rounds)} ronde(s) resterend.")

    REPORT_PATH.write_text("\n".join(lines) + "\n")
    log(f"Rapport bijgewerkt: {REPORT_PATH}")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--skip-warmup-wait", action="store_true",
                   help="Sla het wachten op warmup over (warmup al klaar)")
    p.add_argument("--start-round", type=int, default=1,
                   help="Begin bij deze ronde (voor hervatten)")
    p.add_argument("--start-checkpoint", default=None,
                   help="Override startcheckpoint (container-pad)")
    return p.parse_args()


def main() -> None:
    args       = parse_args()
    start_time = datetime.now()
    rounds: List[dict] = []
    dod_met    = False
    best_idx   = 0

    log("=" * 60)
    log("Orchestrator tpsl_v4b gestart")
    log(f"DoD: long_prec >= {DOD_LONG_PREC:.0%}  AND  short_prec >= {DOD_SHORT_PREC:.0%}")
    log(f"Max rondes: {MAX_ROUNDS}  |  Epochs per ronde: {EPOCHS_PER_ROUND}")
    log("=" * 60)

    # ── Fase 0: wacht op warmup ──
    if not args.skip_warmup_wait:
        wait_for_warmup()
    else:
        log("--skip-warmup-wait: warmup-wacht overgeslagen")

    # ── Fase 1: finetune dataset ──
    build_finetune_csv()

    # Start checkpoint
    if args.start_checkpoint:
        ckpt_ctr = args.start_checkpoint
    else:
        ckpt_ctr = WARMUP_CKPT_CTR

    hold_weight = HOLD_WEIGHT_START + (args.start_round - 1) * HOLD_WEIGHT_STEP
    lr          = LR_START
    best_score  = -1.0
    plateau_count = 0

    # ── Fase 2: finetune loop ──
    for round_num in range(args.start_round, MAX_ROUNDS + 1):
        log(f"\n── Ronde {round_num}/{MAX_ROUNDS} | hold_weight={hold_weight:.1f} | lr={lr:.0e} ──")

        ckpt_dir_host = run_finetune(round_num, ckpt_ctr, hold_weight, lr)
        ckpt_ctr      = best_checkpoint_ctr(ckpt_dir_host)
        log(f"Beste checkpoint: {ckpt_ctr}")

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

        # Track beste ronde
        if score > best_score:
            best_score = score
            best_idx   = len(rounds) - 1
            plateau_count = 0
        else:
            plateau_count += 1

        log(f"Ronde {round_num}: bal={metrics['bal_acc']:.3f} | "
            f"short P={metrics['short_prec']:.3f} R={metrics['short_rec']:.3f} | "
            f"long  P={metrics['long_prec']:.3f} R={metrics['long_rec']:.3f} | "
            f"DoD={'JA ✅' if dod_met else 'NEE ❌'}")
        log(f"Score={score:.3f}  Beste={best_score:.3f}  Plateau={plateau_count}")

        write_report(rounds, dod_met, best_idx, start_time)

        if dod_met:
            log("DoD bereikt! Klaar.")
            break

        # Plateau-detectie: reset naar beste checkpoint en halveer LR
        if plateau_count >= 2:
            log("Plateau gedetecteerd (2 rondes geen verbetering).")
            best_ckpt_ctr = rounds[best_idx]["checkpoint"]
            log(f"Reset naar beste checkpoint: {best_ckpt_ctr}")
            ckpt_ctr    = best_ckpt_ctr
            lr          = lr / 2.0
            plateau_count = 0
            log(f"LR verlaagd naar {lr:.0e}")
        else:
            hold_weight += HOLD_WEIGHT_STEP

    if not dod_met:
        log(f"\nDoD niet bereikt na {len(rounds)} ronde(s).")
        log(f"Beste ronde: {rounds[best_idx]['round']} "
            f"(long={rounds[best_idx]['metrics']['long_prec']:.1%}, "
            f"short={rounds[best_idx]['metrics']['short_prec']:.1%})")
        write_report(rounds, False, best_idx, start_time)

    log(f"\nKlaar. Rapport: {REPORT_PATH}")


if __name__ == "__main__":
    main()
