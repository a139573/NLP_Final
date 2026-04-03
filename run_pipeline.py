#!/usr/bin/env python3
"""
Master Pipeline Orchestrator.

Runs the full NLP pipeline end-to-end for a given phase, from data
extraction through final inference. Pass --toy for a fast smoke-test
that completes in under 5 minutes.

Usage:
    python run_pipeline.py --phase 3
    python run_pipeline.py --phase 3 --toy
"""

import argparse
import subprocess
import sys
import time

# ==========================================
# Terminal Formatting
# ==========================================
BOLD = "\033[1m"
GREEN = "\033[92m"
RED = "\033[91m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
RESET = "\033[0m"
SEPARATOR = "═" * 60


def banner(step: int, total: int, title: str) -> None:
    """Print a visually distinct step banner."""
    print(f"\n{CYAN}{SEPARATOR}{RESET}")
    print(f"{BOLD}{CYAN}  STEP {step}/{total}: {title}{RESET}")
    print(f"{CYAN}{SEPARATOR}{RESET}\n")


def success(msg: str) -> None:
    print(f"{GREEN}✔ {msg}{RESET}")


def fail(msg: str) -> None:
    print(f"{RED}✘ {msg}{RESET}")


def info(msg: str) -> None:
    print(f"{YELLOW}ℹ {msg}{RESET}")


# ==========================================
# Pipeline Step Runner
# ==========================================
def run_step(step: int, total: int, title: str, cmd: list[str]) -> None:
    """Execute a single pipeline step via subprocess."""
    banner(step, total, title)
    info(f"Command: {' '.join(cmd)}")

    t0 = time.time()
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        elapsed = time.time() - t0
        fail(f"{title} FAILED after {elapsed:.1f}s (exit code {exc.returncode})")
        sys.exit(exc.returncode)

    elapsed = time.time() - t0
    success(f"{title} completed in {elapsed:.1f}s")


# ==========================================
# Main
# ==========================================
def main():
    parser = argparse.ArgumentParser(
        description="Master pipeline orchestrator for Political Rhetoric NLP."
    )
    parser.add_argument(
        "--phase", type=int, choices=[1, 2, 3], required=True,
        help="Pipeline phase to execute end-to-end.",
    )
    parser.add_argument(
        "--toy", action="store_true",
        help="Enable toy mode: drastically reduce data and iterations "
             "so the full pipeline finishes in under 5 minutes.",
    )
    args = parser.parse_args()

    py = sys.executable  # Use the same Python interpreter
    total_steps = 6

    print(f"\n{BOLD}{GREEN}{'═' * 60}{RESET}")
    print(f"{BOLD}{GREEN}  NLP MASTER PIPELINE — Phase {args.phase}"
          f"{'  [TOY MODE]' if args.toy else ''}{RESET}")
    print(f"{BOLD}{GREEN}{'═' * 60}{RESET}")

    pipeline_t0 = time.time()

    # ----------------------------------------------------------
    # STEP 1: Data Extraction
    # ----------------------------------------------------------
    cmd_extract = [
        py, "src/data/extract_parlamint.py",
        "--phase", str(args.phase),
    ]
    if args.toy:
        cmd_extract += ["--n_interventions", "20"]

    run_step(1, total_steps, "DATA EXTRACTION", cmd_extract)

    # ----------------------------------------------------------
    # STEP 2: Data Cleaning
    # ----------------------------------------------------------
    cmd_clean = [py, "src/data/clean_parlamint.py"]
    run_step(2, total_steps, "DATA CLEANING", cmd_clean)

    # ----------------------------------------------------------
    # STEP 3: Feature Engineering
    # ----------------------------------------------------------
    cmd_features = [py, "src/features/build_features_roberta.py"]
    run_step(3, total_steps, "FEATURE ENGINEERING", cmd_features)

    # ----------------------------------------------------------
    # STEP 4: Training / Hyperparameter Tuning
    # ----------------------------------------------------------
    cmd_train = [
        py, "src/models/train_engine.py",
        "--phase", str(args.phase),
    ]
    if args.toy and args.phase == 3:
        cmd_train += ["--optuna-trials", "2"]

    run_step(4, total_steps, "TRAINING / TUNING", cmd_train)

    # ----------------------------------------------------------
    # STEP 5: Evaluation & Production Retraining
    # ----------------------------------------------------------
    cmd_eval = [
        py, "src/models/evaluate_model.py",
        "--phase", str(args.phase),
    ]
    run_step(5, total_steps, "EVALUATION & PRODUCTION RETRAINING", cmd_eval)

    # ----------------------------------------------------------
    # STEP 6: Inference & Reporting
    # ----------------------------------------------------------
    cmd_infer = [py, "src/visualization/inference_engine.py"]
    run_step(6, total_steps, "INFERENCE & REPORTING", cmd_infer)

    # ----------------------------------------------------------
    # Summary
    # ----------------------------------------------------------
    total_elapsed = time.time() - pipeline_t0
    minutes, seconds = divmod(int(total_elapsed), 60)

    print(f"\n{BOLD}{GREEN}{'═' * 60}{RESET}")
    print(f"{BOLD}{GREEN}  PIPELINE COMPLETE — Phase {args.phase} "
          f"finished in {minutes}m {seconds}s{RESET}")
    print(f"{BOLD}{GREEN}{'═' * 60}{RESET}\n")


if __name__ == "__main__":
    main()
