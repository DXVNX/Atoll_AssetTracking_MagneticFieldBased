"""Reproducibility and provenance helpers."""

from __future__ import annotations

import hashlib
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np


def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(root: str | Path) -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True, check=False)
    return result.stdout.strip() or "unknown"


def environment_snapshot() -> str:
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], text=True, capture_output=True, check=False)
    return f"python={sys.version}\nexecutable={sys.executable}\n\n{freeze.stdout}"

