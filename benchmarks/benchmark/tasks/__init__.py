"""
Tasks definition for Benchmark
"""

from .tasks import (
    BenchmarkTable,
    check_benchmark_version,
    check_task_types,
    check_splits,
    LATEST_BENCHMARK_VERSION,
)

__all__ = [
    "BenchmarkTable",
    "check_benchmark_version",
    "check_task_types",
    "check_splits",
    "LATEST_BENCHMARK_VERSION",
]
