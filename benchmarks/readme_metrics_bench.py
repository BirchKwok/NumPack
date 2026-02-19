#!/usr/bin/env python3
"""
README 核心性能指标基准测试
验证以下指标是否仍然成立：
  Row Replacement : 344x faster than NPY
  Data Append     : 338x faster than NPY
  Lazy Loading    : 51x  faster than NPY mmap
  Full Load       : 1.64x faster than NPY
  Batch Mode      : 21x  speedup
  Writable Batch  : 92x  speedup
"""

import os, sys, gc, time, tempfile, shutil
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from numpack import NumPack

ROWS   = 100_000
COLS   = 128
DTYPE  = np.float32
REPEAT = 5


def _min_time(fn, repeat=REPEAT):
    times = []
    for _ in range(repeat):
        gc.collect()
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times)


def bench_replace(data, npy_path, npk_dir):
    """Row Replacement: NumPack.replace vs np.save (overwrite)"""
    new_rows = np.random.rand(1000, COLS).astype(DTYPE)
    indices  = list(range(1000))

    # NPY baseline: load -> modify -> save
    def npy_replace():
        arr = np.load(npy_path)
        arr[indices] = new_rows
        np.save(npy_path, arr)

    # NumPack
    npk = NumPack(npk_dir)
    npk.open()
    def npk_replace():
        npk.replace({"data": new_rows}, indexes=indices)
    npk.close()

    t_npy = _min_time(npy_replace)
    npk.open()
    t_npk = _min_time(npk_replace)
    npk.close()
    return t_npy, t_npk


def bench_append(npy_path, npk_dir):
    """Data Append: NumPack.append vs np.save (concatenate)"""
    extra = np.random.rand(10_000, COLS).astype(DTYPE)

    def npy_append():
        arr = np.load(npy_path)
        arr = np.concatenate([arr, extra])
        np.save(npy_path, arr)

    npk = NumPack(npk_dir)
    npk.open()
    def npk_append():
        npk.append({"data": extra})
    npk.close()

    # Reset NPY file to original size for each repeat
    original = np.load(npy_path)
    npy_times = []
    for _ in range(REPEAT):
        np.save(npy_path, original)
        gc.collect()
        t0 = time.perf_counter()
        arr = np.load(npy_path)
        arr = np.concatenate([arr, extra])
        np.save(npy_path, arr)
        npy_times.append(time.perf_counter() - t0)
    t_npy = min(npy_times)
    np.save(npy_path, original)

    # Reset NPK dir and benchmark append
    shutil.rmtree(npk_dir, ignore_errors=True)
    with NumPack(npk_dir, drop_if_exists=True) as n:
        n.save({"data": original})

    npk = NumPack(npk_dir)
    npk_times = []
    for _ in range(REPEAT):
        shutil.rmtree(npk_dir, ignore_errors=True)
        with NumPack(npk_dir, drop_if_exists=True) as n:
            n.save({"data": original})
        npk2 = NumPack(npk_dir)
        npk2.open()
        gc.collect()
        t0 = time.perf_counter()
        npk2.append({"data": extra})
        npk_times.append(time.perf_counter() - t0)
        npk2.close()

    t_npk = min(npk_times)
    return t_npy, t_npk


def bench_lazy(npy_path, npk_dir):
    """Lazy Loading: NumPack lazy getitem vs np.load mmap"""
    indices = list(range(0, 10_000, 10))

    def npy_lazy():
        mmap = np.load(npy_path, mmap_mode='r')
        _ = np.array(mmap[indices])

    npk = NumPack(npk_dir)
    npk.open()
    def npk_lazy():
        _ = npk.getitem("data", indices)
    npk.close()

    t_npy = _min_time(npy_lazy)
    npk.open()
    t_npk = _min_time(npk_lazy)
    npk.close()
    return t_npy, t_npk


def bench_full_load(npy_path, npk_dir):
    """Full Load: NumPack.load vs np.load"""
    def npy_load():
        _ = np.load(npy_path)

    npk = NumPack(npk_dir)
    npk.open()
    def npk_load():
        _ = npk.load("data")
    npk.close()

    t_npy = _min_time(npy_load)
    npk.open()
    t_npk = _min_time(npk_load)
    npk.close()
    return t_npy, t_npk


def bench_batch_mode(npk_dir):
    """Batch Mode: batch_mode vs normal repeated save"""
    N_ITERS = 200
    npk = NumPack(npk_dir)

    # Normal (no batch)
    npk.open()
    arr0 = npk.load("data")
    npk.close()

    normal_times = []
    for _ in range(3):
        npk.open()
        arr = npk.load("data")
        gc.collect()
        t0 = time.perf_counter()
        for _ in range(N_ITERS):
            arr[:10] *= 1.0
            npk.save({"data": arr})
        normal_times.append(time.perf_counter() - t0)
        npk.close()
    t_normal = min(normal_times)

    # batch_mode
    batch_times = []
    for _ in range(3):
        gc.collect()
        t0 = time.perf_counter()
        with NumPack(npk_dir) as n:
            with n.batch_mode():
                arr = n.load("data")
                for _ in range(N_ITERS):
                    arr[:10] *= 1.0
                    n.save({"data": arr})
        batch_times.append(time.perf_counter() - t0)
    t_batch = min(batch_times)
    return t_normal, t_batch


def bench_writable_batch(npk_dir):
    """Writable Batch Mode: writable_batch_mode vs normal repeated save"""
    N_ITERS = 200

    # Normal
    npk = NumPack(npk_dir)
    normal_times = []
    for _ in range(3):
        npk.open()
        arr = npk.load("data")
        gc.collect()
        t0 = time.perf_counter()
        for _ in range(N_ITERS):
            arr[:10] *= 1.0
            npk.save({"data": arr})
        normal_times.append(time.perf_counter() - t0)
        npk.close()
    t_normal = min(normal_times)

    # writable_batch_mode
    wb_times = []
    for _ in range(3):
        gc.collect()
        t0 = time.perf_counter()
        with NumPack(npk_dir) as n:
            with n.writable_batch_mode() as wb:
                arr = wb.load("data")
                for _ in range(N_ITERS):
                    arr[:10] *= 1.0
        wb_times.append(time.perf_counter() - t0)
    t_wb = min(wb_times)
    return t_normal, t_wb


def fmt(speedup, baseline):
    mark = "OK" if speedup >= baseline * 0.8 else "WARN"
    return f"{speedup:6.1f}x  (README: {baseline}x)  [{mark}]"


def main():
    data = np.random.rand(ROWS, COLS).astype(DTYPE)
    print(f"\nDataset: {ROWS:,} rows x {COLS} cols  ({data.nbytes/(1<<20):.1f} MB)\n")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        npy_path = tmp / "data.npy"
        npk_dir  = tmp / "data.npk"

        np.save(npy_path, data)
        with NumPack(npk_dir, drop_if_exists=True) as n:
            n.save({"data": data})

        print(f"{'Metric':<22} {'NPY (ms)':>10} {'NumPack (ms)':>13} {'Result'}")
        print("-" * 72)

        # 1. Row Replacement
        t_npy, t_npk = bench_replace(data, npy_path, npk_dir)
        sp = t_npy / t_npk
        print(f"{'Row Replacement':<22} {t_npy*1e3:>10.2f} {t_npk*1e3:>13.2f}  {fmt(sp, 344)}")

        # 2. Data Append
        t_npy, t_npk = bench_append(npy_path, npk_dir)
        sp = t_npy / t_npk
        print(f"{'Data Append':<22} {t_npy*1e3:>10.2f} {t_npk*1e3:>13.2f}  {fmt(sp, 338)}")

        # 3. Lazy Loading
        t_npy, t_npk = bench_lazy(npy_path, npk_dir)
        sp = t_npy / t_npk
        print(f"{'Lazy Loading':<22} {t_npy*1e3:>10.2f} {t_npk*1e3:>13.2f}  {fmt(sp, 51)}")

        # 4. Full Load
        t_npy, t_npk = bench_full_load(npy_path, npk_dir)
        sp = t_npy / t_npk
        print(f"{'Full Load':<22} {t_npy*1e3:>10.2f} {t_npk*1e3:>13.2f}  {fmt(sp, 1.64)}")

        # 5. Batch Mode
        t_normal, t_batch = bench_batch_mode(npk_dir)
        sp = t_normal / t_batch
        print(f"{'Batch Mode':<22} {t_normal*1e3:>10.2f} {t_batch*1e3:>13.2f}  {fmt(sp, 21)}")

        # 6. Writable Batch
        t_normal, t_wb = bench_writable_batch(npk_dir)
        sp = t_normal / t_wb
        print(f"{'Writable Batch':<22} {t_normal*1e3:>10.2f} {t_wb*1e3:>13.2f}  {fmt(sp, 92)}")

    print()


if __name__ == "__main__":
    main()
