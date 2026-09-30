"""Canonical dataset loaders with an explicit official-test lock."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .schemas import adapt_uji_samples, validate_canonical_samples


class FinalTestLockedError(PermissionError):
    pass


def load_uji_trajectories(dataset_dir: str | Path, *, source_split: str = "train_public", allow_final_test: bool = False) -> pd.DataFrame:
    if source_split == "test_public" and not allow_final_test:
        raise FinalTestLockedError("public test data is locked; pass allow_final_test=True explicitly")
    if source_split not in {"train_public", "test_public"}:
        raise ValueError(f"unsupported source_split: {source_split}")
    folder = Path(dataset_dir) / "interim" / source_split
    paths = sorted(folder.glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"no canonical trajectories found in {folder}")
    frame = pd.concat((adapt_uji_samples(pd.read_csv(path)) for path in paths), ignore_index=True)
    validate_canonical_samples(frame)
    return frame

