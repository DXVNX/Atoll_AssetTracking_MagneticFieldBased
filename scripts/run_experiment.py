"""CLI for the unified experiment harness."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.experiments.config import load_config
from src.experiments.runner import run_experiment

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--final-test", action="store_true")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    run_dir = run_experiment(config, root=ROOT, smoke=args.smoke, overwrite=args.overwrite, final_test=args.final_test, seed=args.seed)
    print(run_dir)


if __name__ == "__main__":
    main()

