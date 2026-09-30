"""Rebuild results/experiment_registry.csv from per-run artifacts."""

from pathlib import Path

from src.experiments.registry import rebuild_registry


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    registry = rebuild_registry(root / "results")
    print(f"registry rows: {len(registry)}")


if __name__ == "__main__":
    main()

