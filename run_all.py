"""Run one or all Day 22 lab steps."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

STEPS = {
    "1": "01_langsmith_rag_pipeline.py",
    "2": "02_prompt_hub_ab_routing.py",
    "3": "03_ragas_evaluation.py",
    "4": "04_guardrails_validator.py",
}


def run_step(step: str) -> None:
    script = PROJECT_ROOT / STEPS[step]
    print("\n" + "=" * 70)
    print(f"Running step {step}: {script.name}")
    print("=" * 70)
    subprocess.run([sys.executable, str(script)], cwd=PROJECT_ROOT, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Day 22 LangSmith/RAG lab.")
    parser.add_argument(
        "--step",
        choices=["1", "2", "3", "4", "all"],
        default="all",
        help="Run a single step or the full lab.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_steps = list(STEPS) if args.step == "all" else [args.step]
    for step in selected_steps:
        run_step(step)


if __name__ == "__main__":
    main()
