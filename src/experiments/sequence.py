"""Leakage-safe fixed-shape sequence construction for UJI development data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

CHANNEL_SETS = {
    "magnetometer": ("mx", "my", "mz"),
    "magnitude": ("magnitude",),
    "magnetometer_accelerometer": ("mx", "my", "mz", "ax", "ay", "az"),
}


@dataclass(frozen=True)
class SequenceConfig:
    rate_hz: float = 10.0
    length_s: float = 2.0
    channels: str = "magnetometer"

    def __post_init__(self) -> None:
        if self.rate_hz <= 0 or self.length_s <= 0:
            raise ValueError("rate_hz and length_s must be positive")
        if self.channels not in CHANNEL_SETS:
            raise ValueError(f"unsupported channel set: {self.channels}")

    @property
    def timesteps(self) -> int:
        steps = self.rate_hz * self.length_s
        if not np.isclose(steps, round(steps)):
            raise ValueError("rate_hz * length_s must be an integer")
        return int(round(steps))


@dataclass(frozen=True)
class SequenceExample:
    values: np.ndarray
    label: str
    label_index: int
    window_id: str
    trajectory_id: str
    group_id: str


@dataclass(frozen=True)
class ChannelNormalizer:
    mean: np.ndarray
    std: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.mean[:, None]) / self.std[:, None]).astype(np.float32)


def _window_samples(samples: pd.DataFrame, window: pd.Series) -> pd.DataFrame:
    trajectory = samples[samples["trajectory_id"].astype(str) == str(window.trajectory_id)]
    selected = trajectory[(trajectory["elapsed_s"] >= float(window.window_start_s) - 1e-9) & (trajectory["elapsed_s"] < float(window.window_end_s) - 1e-9)]
    if selected.empty or selected["label"].nunique() != 1:
        raise ValueError(f"window {window.window_id} is empty or crosses a segment")
    return selected.sort_values("elapsed_s")


def _resample_window(window_samples: pd.DataFrame, window: pd.Series, config: SequenceConfig) -> np.ndarray:
    target_times = float(window.window_start_s) + np.arange(config.timesteps, dtype=float) / config.rate_hz
    source_times = window_samples["elapsed_s"].to_numpy(dtype=float)
    if target_times[-1] > source_times[-1] + 1e-8:
        raise ValueError(f"window {window.window_id} does not contain enough samples for {config.rate_hz} Hz")
    channel_values = []
    for channel in ("mx", "my", "mz", "ax", "ay", "az"):
        channel_values.append(np.interp(target_times, source_times, window_samples[channel].to_numpy(dtype=float)))
    values = np.asarray(channel_values, dtype=np.float32)
    if config.channels == "magnitude":
        values = np.linalg.norm(values[:3], axis=0, keepdims=True)
    elif config.channels == "magnetometer_accelerometer":
        pass
    else:
        values = values[:3]
    return values.astype(np.float32)


def build_sequence_examples(samples: pd.DataFrame, windows: pd.DataFrame, config: SequenceConfig, labels: Iterable[str] | None = None) -> list[SequenceExample]:
    label_values = sorted(set(labels) if labels is not None else set(windows["label"]))
    label_to_index = {label: index for index, label in enumerate(label_values)}
    examples = []
    for window in windows.itertuples(index=False):
        row = pd.Series(window._asdict())
        if row.label not in label_to_index:
            continue
        selected = _window_samples(samples, row)
        values = _resample_window(selected, row, config)
        expected = (len(CHANNEL_SETS[config.channels]), config.timesteps)
        if values.shape != expected:
            raise ValueError(f"window {row.window_id} produced {values.shape}, expected {expected}")
        examples.append(SequenceExample(values, str(row.label), label_to_index[row.label], str(row.window_id), str(row.trajectory_id), str(row.group_id)))
    return examples


def fit_channel_normalizer(examples: Iterable[SequenceExample]) -> ChannelNormalizer:
    examples = list(examples)
    if not examples:
        raise ValueError("cannot fit a normalizer on no examples")
    values = np.concatenate([example.values for example in examples], axis=1)
    mean = values.mean(axis=1)
    std = values.std(axis=1)
    std = np.where(std < 1e-8, 1.0, std)
    return ChannelNormalizer(mean.astype(np.float32), std.astype(np.float32))


def normalize_examples(examples: Iterable[SequenceExample], normalizer: ChannelNormalizer) -> list[SequenceExample]:
    return [SequenceExample(normalizer.transform(example.values), example.label, example.label_index, example.window_id, example.trajectory_id, example.group_id) for example in examples]


def to_torch_dataset(examples: Iterable[SequenceExample]):
    try:
        import torch
        from torch.utils.data import Dataset
    except ImportError as exc:
        raise ImportError("PyTorch is required for torch dataset conversion") from exc

    examples = list(examples)

    class _Dataset(Dataset):
        def __len__(self):
            return len(examples)

        def __getitem__(self, index):
            example = examples[index]
            return torch.from_numpy(example.values), torch.tensor(example.label_index, dtype=torch.long), example.window_id, example.trajectory_id, example.group_id

    return _Dataset()