#!/usr/bin/env python3
"""
Storage layout regression benchmark for NumPack.

This benchmark captures the current single-file mmap storage baseline before
introducing segmented/multi-file array storage. It focuses on operations that
are likely to be affected by a layout refactor:

- Initial save
- Lazy load creation
- Full eager load
- Sequential and random row reads through NumPack.getitem
- LazyArray row reads
- Row replacement
- Append
- Streamed full scan

The script writes a JSON result file that can be used as a regression baseline:

    python benchmarks/storage_layout_baseline_bench.py
    python benchmarks/storage_layout_baseline_bench.py --baseline-json path/to/baseline.json
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_PYTHON = REPO_ROOT / "python"
sys.path.insert(0, str(LOCAL_PYTHON))
sys.path.insert(1, str(REPO_ROOT))

from numpack import NumPack  # noqa: E402
import numpack  # noqa: E402


ARRAY_NAME = "data"


@dataclass(frozen=True)
class BenchConfig:
    rows: int
    cols: int
    dtype: str
    append_rows: int
    replace_rows: int
    sequential_rows: int
    random_counts: tuple[int, ...]
    stream_rows: int
    repeat: int
    warmup: int
    seed: int
    output_dir: Path
    keep_data: bool
    baseline_json: Path | None
    tolerance: float
    absolute_tolerance_ms: float


def parse_counts(raw: str) -> tuple[int, ...]:
    values = []
    for item in raw.split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    if not values:
        raise argparse.ArgumentTypeError("count list cannot be empty")
    return tuple(values)


def dtype_from_name(name: str) -> np.dtype:
    try:
        dtype = np.dtype(name)
    except TypeError as exc:
        raise argparse.ArgumentTypeError(f"unsupported dtype: {name}") from exc
    if dtype not in (np.dtype("float32"), np.dtype("float64"), np.dtype("int32"), np.dtype("int64")):
        raise argparse.ArgumentTypeError(
            "this benchmark intentionally supports float32, float64, int32, and int64"
        )
    return dtype


def parse_args() -> BenchConfig:
    parser = argparse.ArgumentParser(
        description="Capture NumPack storage-layout performance baselines."
    )
    parser.add_argument("--rows", type=int, default=100_000)
    parser.add_argument("--cols", type=int, default=128)
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--append-rows", type=int, default=10_000)
    parser.add_argument("--replace-rows", type=int, default=1_000)
    parser.add_argument("--sequential-rows", type=int, default=10_000)
    parser.add_argument("--random-counts", type=parse_counts, default=parse_counts("100,1000,10000"))
    parser.add_argument("--stream-rows", type=int, default=8192)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "benchmarks" / "results",
        help="Directory where the JSON result is written.",
    )
    parser.add_argument(
        "--keep-data",
        action="store_true",
        help="Keep the temporary benchmark dataset directory for inspection.",
    )
    parser.add_argument(
        "--baseline-json",
        type=Path,
        default=None,
        help="Optional previous JSON result. Exits non-zero on regression.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.15,
        help="Allowed slowdown ratio when --baseline-json is provided.",
    )
    parser.add_argument(
        "--absolute-tolerance-ms",
        type=float,
        default=0.05,
        help=(
            "Allowed absolute slowdown in milliseconds when --baseline-json is provided. "
            "This avoids false positives for sub-millisecond operations."
        ),
    )
    args = parser.parse_args()

    dtype_from_name(args.dtype)
    if args.rows <= 0 or args.cols <= 0:
        parser.error("--rows and --cols must be positive")
    if args.repeat <= 0 or args.warmup < 0:
        parser.error("--repeat must be positive and --warmup cannot be negative")
    if args.append_rows <= 0 or args.replace_rows <= 0:
        parser.error("--append-rows and --replace-rows must be positive")
    if args.sequential_rows <= 0 or args.stream_rows <= 0:
        parser.error("--sequential-rows and --stream-rows must be positive")

    return BenchConfig(
        rows=args.rows,
        cols=args.cols,
        dtype=args.dtype,
        append_rows=min(args.append_rows, args.rows),
        replace_rows=min(args.replace_rows, args.rows),
        sequential_rows=min(args.sequential_rows, args.rows),
        random_counts=tuple(min(count, args.rows) for count in args.random_counts),
        stream_rows=min(args.stream_rows, args.rows),
        repeat=args.repeat,
        warmup=args.warmup,
        seed=args.seed,
        output_dir=args.output_dir,
        keep_data=args.keep_data,
        baseline_json=args.baseline_json,
        tolerance=args.tolerance,
        absolute_tolerance_ms=args.absolute_tolerance_ms,
    )


def git_value(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


def now_slug() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def make_data(config: BenchConfig) -> np.ndarray:
    rng = np.random.default_rng(config.seed)
    dtype = dtype_from_name(config.dtype)
    if np.issubdtype(dtype, np.integer):
        return rng.integers(0, 10_000, size=(config.rows, config.cols), dtype=dtype)
    return rng.random((config.rows, config.cols), dtype=dtype)


def summarize(times_ms: list[float]) -> dict[str, Any]:
    sorted_times = sorted(times_ms)
    return {
        "min_ms": min(times_ms),
        "mean_ms": statistics.fmean(times_ms),
        "median_ms": statistics.median(times_ms),
        "max_ms": max(times_ms),
        "std_ms": statistics.pstdev(times_ms) if len(times_ms) > 1 else 0.0,
        "all_ms": sorted_times,
    }


def measure(
    name: str,
    func: Callable[[], Any],
    *,
    repeat: int,
    warmup: int,
    cleanup: Callable[[], None] | None = None,
) -> dict[str, Any]:
    for _ in range(warmup):
        result = func()
        consume_result(result)
        if cleanup:
            cleanup()

    times_ms: list[float] = []
    for _ in range(repeat):
        gc.collect()
        start = time.perf_counter()
        result = func()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        consume_result(result)
        times_ms.append(elapsed_ms)
        if cleanup:
            cleanup()

    stats = summarize(times_ms)
    stats["name"] = name
    return stats


def measure_with_context(
    name: str,
    setup: Callable[[], Any],
    func: Callable[[Any], Any],
    teardown: Callable[[Any], None] | None = None,
    *,
    repeat: int,
    warmup: int,
) -> dict[str, Any]:
    for _ in range(warmup):
        ctx = setup()
        try:
            result = func(ctx)
            consume_result(result)
        finally:
            if teardown:
                teardown(ctx)

    times_ms: list[float] = []
    for _ in range(repeat):
        gc.collect()
        ctx = setup()
        try:
            start = time.perf_counter()
            result = func(ctx)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            consume_result(result)
            times_ms.append(elapsed_ms)
        finally:
            if teardown:
                teardown(ctx)

    stats = summarize(times_ms)
    stats["name"] = name
    return stats


def consume_result(result: Any) -> None:
    if result is None:
        return
    if isinstance(result, np.ndarray):
        if result.size:
            _ = result.reshape(-1)[0]
        return
    if isinstance(result, (list, tuple)) and result:
        consume_result(result[0])


def create_numpack_dataset(path: Path, data: np.ndarray) -> None:
    shutil.rmtree(path, ignore_errors=True)
    with NumPack(path, drop_if_exists=True, warn_no_context=False) as npk:
        npk.save({ARRAY_NAME: data})


def open_numpack(path: Path) -> NumPack:
    npk = NumPack(path, warn_no_context=False)
    npk.open()
    return npk


def enrich_throughput(case: dict[str, Any], *, bytes_touched: int | None = None, rows: int | None = None) -> None:
    min_seconds = case["min_ms"] / 1000.0
    if min_seconds <= 0:
        return
    if bytes_touched is not None:
        case["throughput_mib_s"] = bytes_touched / (1024 * 1024) / min_seconds
    if rows is not None:
        case["rows_per_s"] = rows / min_seconds
        case["us_per_row"] = case["min_ms"] * 1000.0 / max(rows, 1)


def run_benchmarks(config: BenchConfig) -> dict[str, Any]:
    dtype = dtype_from_name(config.dtype)
    data = make_data(config)
    append_data = data[: config.append_rows].copy()
    replace_data = data[: config.replace_rows].copy()

    temp_parent = Path(tempfile.mkdtemp(prefix="numpack_storage_baseline_"))
    dataset_path = temp_parent / "dataset.npk"
    create_numpack_dataset(dataset_path, data)

    rng = np.random.default_rng(config.seed + 1)
    sequential_indices = list(range(config.sequential_rows))
    random_indices = {
        f"random_{count}": rng.integers(0, config.rows, size=count, dtype=np.int64).tolist()
        for count in config.random_counts
    }
    sorted_random_indices = {
        f"sorted_random_{count}": sorted(indices)
        for count, indices in zip(config.random_counts, random_indices.values())
    }

    npk = open_numpack(dataset_path)
    lazy = npk.load(ARRAY_NAME, lazy=True)
    _ = lazy[0]

    data_bytes = data.nbytes
    row_bytes = config.cols * dtype.itemsize
    results: dict[str, Any] = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "repo_root": str(REPO_ROOT),
            "git_commit": git_value(["rev-parse", "HEAD"]),
            "git_dirty": bool(git_value(["status", "--short"])),
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "numpy_version": np.__version__,
            "numpack_version": getattr(numpack, "__version__", "unknown"),
            "numpack_file": getattr(numpack, "__file__", "unknown"),
            "command": " ".join(sys.argv),
            "temp_parent": str(temp_parent),
        },
        "config": {
            "rows": config.rows,
            "cols": config.cols,
            "dtype": config.dtype,
            "data_mib": data_bytes / (1024 * 1024),
            "append_rows": config.append_rows,
            "replace_rows": config.replace_rows,
            "sequential_rows": config.sequential_rows,
            "random_counts": list(config.random_counts),
            "stream_rows": config.stream_rows,
            "repeat": config.repeat,
            "warmup": config.warmup,
            "seed": config.seed,
        },
        "cases": {},
    }

    cases = results["cases"]

    def setup_empty_path() -> Path:
        path = temp_parent / f"save_new_{time.perf_counter_ns()}.npk"
        shutil.rmtree(path, ignore_errors=True)
        return path

    def save_new_once(path: Path) -> None:
        with NumPack(path, drop_if_exists=True, warn_no_context=False) as save_npk:
            save_npk.save({ARRAY_NAME: data})

    def remove_path(path: Path) -> None:
        shutil.rmtree(path, ignore_errors=True)

    cases["save_new"] = measure_with_context(
        "save_new",
        setup_empty_path,
        save_new_once,
        remove_path,
        repeat=config.repeat,
        warmup=config.warmup,
    )
    enrich_throughput(cases["save_new"], bytes_touched=data_bytes, rows=config.rows)

    cases["lazy_load_create"] = measure(
        "lazy_load_create",
        lambda: npk.load(ARRAY_NAME, lazy=True),
        repeat=config.repeat,
        warmup=config.warmup,
    )

    cases["full_load"] = measure(
        "full_load",
        lambda: npk.load(ARRAY_NAME, lazy=False),
        repeat=config.repeat,
        warmup=config.warmup,
    )
    enrich_throughput(cases["full_load"], bytes_touched=data_bytes, rows=config.rows)

    cases["getitem_sequential"] = measure(
        "getitem_sequential",
        lambda: npk.getitem(ARRAY_NAME, sequential_indices),
        repeat=config.repeat,
        warmup=config.warmup,
    )
    enrich_throughput(
        cases["getitem_sequential"],
        bytes_touched=config.sequential_rows * row_bytes,
        rows=config.sequential_rows,
    )

    for name, indices in random_indices.items():
        cases[f"getitem_{name}"] = measure(
            f"getitem_{name}",
            lambda indices=indices: npk.getitem(ARRAY_NAME, indices),
            repeat=config.repeat,
            warmup=config.warmup,
        )
        enrich_throughput(
            cases[f"getitem_{name}"],
            bytes_touched=len(indices) * row_bytes,
            rows=len(indices),
        )

    for name, indices in sorted_random_indices.items():
        cases[f"getitem_{name}"] = measure(
            f"getitem_{name}",
            lambda indices=indices: npk.getitem(ARRAY_NAME, indices),
            repeat=config.repeat,
            warmup=config.warmup,
        )
        enrich_throughput(
            cases[f"getitem_{name}"],
            bytes_touched=len(indices) * row_bytes,
            rows=len(indices),
        )

    cases["lazyarray_row_0"] = measure(
        "lazyarray_row_0",
        lambda: lazy[0],
        repeat=config.repeat,
        warmup=config.warmup,
    )
    enrich_throughput(cases["lazyarray_row_0"], bytes_touched=row_bytes, rows=1)

    smallest_random = random_indices[f"random_{config.random_counts[0]}"]
    cases["lazyarray_random_small"] = measure(
        "lazyarray_random_small",
        lambda: lazy[smallest_random],
        repeat=config.repeat,
        warmup=config.warmup,
    )
    enrich_throughput(
        cases["lazyarray_random_small"],
        bytes_touched=len(smallest_random) * row_bytes,
        rows=len(smallest_random),
    )

    replace_indices = list(range(config.replace_rows))
    cases["replace_rows"] = measure(
        "replace_rows",
        lambda: npk.replace({ARRAY_NAME: replace_data}, indexes=replace_indices),
        repeat=config.repeat,
        warmup=config.warmup,
    )
    enrich_throughput(cases["replace_rows"], bytes_touched=replace_data.nbytes, rows=config.replace_rows)

    def stream_scan() -> int:
        total = 0
        for chunk in npk.stream_load(ARRAY_NAME, buffer_size=config.stream_rows):
            total += chunk.shape[0]
        return total

    cases["stream_scan"] = measure(
        "stream_scan",
        stream_scan,
        repeat=config.repeat,
        warmup=config.warmup,
    )
    enrich_throughput(cases["stream_scan"], bytes_touched=data_bytes, rows=config.rows)

    def setup_append() -> tuple[Path, NumPack]:
        path = temp_parent / f"append_{time.perf_counter_ns()}.npk"
        create_numpack_dataset(path, data)
        append_npk = open_numpack(path)
        return path, append_npk

    def append_once(ctx: tuple[Path, NumPack]) -> None:
        _, append_npk = ctx
        append_npk.append({ARRAY_NAME: append_data})

    def teardown_append(ctx: tuple[Path, NumPack]) -> None:
        path, append_npk = ctx
        try:
            append_npk.close()
        finally:
            shutil.rmtree(path, ignore_errors=True)

    cases["append_rows"] = measure_with_context(
        "append_rows",
        setup_append,
        append_once,
        teardown_append,
        repeat=config.repeat,
        warmup=config.warmup,
    )
    enrich_throughput(cases["append_rows"], bytes_touched=append_data.nbytes, rows=config.append_rows)

    npk.close()

    config.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = config.output_dir / f"storage_layout_baseline_{now_slug()}.json"
    results["metadata"]["output_path"] = str(output_path)

    if not config.keep_data:
        shutil.rmtree(temp_parent, ignore_errors=True)
        results["metadata"]["temp_parent_removed"] = True
    else:
        results["metadata"]["temp_parent_removed"] = False

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, sort_keys=True)

    return results


def compare_against_baseline(
    current: dict[str, Any],
    baseline_path: Path,
    tolerance: float,
    absolute_tolerance_ms: float,
) -> list[str]:
    with baseline_path.open("r", encoding="utf-8") as f:
        baseline = json.load(f)

    failures = []
    comparable_config_keys = (
        "rows",
        "cols",
        "dtype",
        "append_rows",
        "replace_rows",
        "sequential_rows",
        "random_counts",
        "stream_rows",
    )
    current_config = current.get("config", {})
    baseline_config = baseline.get("config", {})
    config_mismatches = [
        f"{key}: current={current_config.get(key)!r}, baseline={baseline_config.get(key)!r}"
        for key in comparable_config_keys
        if current_config.get(key) != baseline_config.get(key)
    ]
    if config_mismatches:
        return [
            "benchmark config does not match baseline; rerun with the same benchmark arguments",
            *config_mismatches,
        ]

    current_cases = current.get("cases", {})
    baseline_cases = baseline.get("cases", {})
    for name, base_case in baseline_cases.items():
        if name not in current_cases:
            continue
        base_min = base_case.get("min_ms")
        current_min = current_cases[name].get("min_ms")
        if not isinstance(base_min, (int, float)) or not isinstance(current_min, (int, float)):
            continue
        allowed = base_min * (1.0 + tolerance)
        absolute_delta = current_min - base_min
        if (
            current_min > allowed
            and absolute_delta > absolute_tolerance_ms
            and not math.isclose(current_min, allowed)
        ):
            failures.append(
                f"{name}: {current_min:.3f} ms > allowed {allowed:.3f} ms "
                f"(baseline {base_min:.3f} ms, tolerance {tolerance:.0%} "
                f"+ {absolute_tolerance_ms:.3f} ms absolute)"
            )
    return failures


def print_table(results: dict[str, Any]) -> None:
    config = results["config"]
    print(
        "\nDataset: "
        f"{config['rows']:,} x {config['cols']} {config['dtype']} "
        f"({config['data_mib']:.1f} MiB)"
    )
    print(f"NumPack: {results['metadata']['numpack_version']} @ {results['metadata']['numpack_file']}")
    print("\nCase                         min ms    mean ms      rows/s       MiB/s")
    print("-" * 74)
    for name, case in results["cases"].items():
        rows_s = case.get("rows_per_s")
        mib_s = case.get("throughput_mib_s")
        rows_text = f"{rows_s:11,.0f}" if rows_s is not None else " " * 11
        mib_text = f"{mib_s:10,.1f}" if mib_s is not None else " " * 10
        print(
            f"{name:<28} "
            f"{case['min_ms']:8.3f} "
            f"{case['mean_ms']:10.3f} "
            f"{rows_text} "
            f"{mib_text}"
        )
    print(f"\nJSON: {results['metadata']['output_path']}")


def main() -> int:
    config = parse_args()
    results = run_benchmarks(config)
    print_table(results)

    if config.baseline_json:
        failures = compare_against_baseline(
            results,
            config.baseline_json,
            config.tolerance,
            config.absolute_tolerance_ms,
        )
        if failures:
            print("\nRegression failures:")
            for failure in failures:
                print(f"  - {failure}")
            return 1
        print(
            f"\nNo regressions beyond {config.tolerance:.0%} relative "
            f"+ {config.absolute_tolerance_ms:.3f} ms absolute tolerance."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
