from pathlib import Path

from src.experiments.benchmark import benchmark_callable, serialized_size_kb


def test_benchmark_reports_positive_timings_and_model_size(tmp_path: Path):
    median, p95 = benchmark_callable(lambda: sum(range(5)), warmup=2, repetitions=5)
    assert median > 0
    assert p95 >= median
    model_path = tmp_path / "model.joblib"
    model_path.write_bytes(b"model")
    assert serialized_size_kb(model_path) > 0