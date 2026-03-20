"""Benchmark for Python NumPack API.

Run:
    python benches/python_api_bench.py
"""

import json
import os
import shutil
import tempfile
import time

import numpy as np

from numpack import NumPack

WARMUP_ITERS = 2
BENCH_ITERS = 10

ROWS = 100_000
COLS = 128
APPEND_ROWS = 10_000
REPLACE_COUNT = 1_000
GETITEM_COUNT = 1_000
STREAM_BUFFER = 10_000
DROP_COUNT = 500


def bench(name, warmup, iters, setup_fn, bench_fn, teardown_fn=None):
    times = []
    for i in range(warmup + iters):
        ctx = setup_fn()
        t0 = time.perf_counter()
        bench_fn(ctx)
        t1 = time.perf_counter()
        if teardown_fn:
            teardown_fn(ctx)
        if i >= warmup:
            times.append((t1 - t0) * 1000.0)
    mean = sum(times) / len(times)
    std = (sum((t - mean) ** 2 for t in times) / len(times)) ** 0.5
    mn = min(times)
    return {"name": name, "mean_ms": mean, "std_ms": std, "min_ms": mn}


def main():
    print("=================================================================")
    print("  NumPack Python API Benchmark")
    print("=================================================================")
    print()
    print(f"Parameters:")
    print(f"  rows          = {ROWS}")
    print(f"  cols          = {COLS}")
    print(f"  append_rows   = {APPEND_ROWS}")
    print(f"  replace_count = {REPLACE_COUNT}")
    print(f"  getitem_count = {GETITEM_COUNT}")
    print(f"  stream_buffer = {STREAM_BUFFER}")
    print(f"  warmup_iters  = {WARMUP_ITERS}")
    print(f"  bench_iters   = {BENCH_ITERS}")
    print()

    data = np.random.rand(ROWS, COLS).astype(np.float32)
    append_data = np.ones((APPEND_ROWS, COLS), dtype=np.float32)
    replace_data = np.full((REPLACE_COUNT, COLS), 999.0, dtype=np.float32)
    replace_indices = list(range(REPLACE_COUNT))
    getitem_indices = [(i * 97) % ROWS for i in range(GETITEM_COUNT)]
    drop_indices = list(range(DROP_COUNT))

    results = []

    # ---- save ----
    def save_setup():
        d = tempfile.mkdtemp()
        npk = NumPack(d)
        npk.open()
        return {"npk": npk, "dir": d}

    def save_bench(ctx):
        ctx["npk"].save({"arr": data})

    def save_teardown(ctx):
        ctx["npk"].close()
        shutil.rmtree(ctx["dir"], ignore_errors=True)

    results.append(bench("save", WARMUP_ITERS, BENCH_ITERS,
                         save_setup, save_bench, save_teardown))

    # ---- load ----
    tmp_load = tempfile.mkdtemp()
    npk_load = NumPack(tmp_load)
    npk_load.open()
    npk_load.save({"arr": data})

    def load_setup():
        return {"npk": npk_load}

    def load_bench(ctx):
        _ = ctx["npk"].load("arr")

    results.append(bench("load", WARMUP_ITERS, BENCH_ITERS,
                         load_setup, load_bench))
    npk_load.close()
    shutil.rmtree(tmp_load, ignore_errors=True)

    # ---- append ----
    def append_setup():
        d = tempfile.mkdtemp()
        npk = NumPack(d)
        npk.open()
        npk.save({"arr": data})
        return {"npk": npk, "dir": d}

    def append_bench(ctx):
        ctx["npk"].append({"arr": append_data})

    def append_teardown(ctx):
        ctx["npk"].close()
        shutil.rmtree(ctx["dir"], ignore_errors=True)

    results.append(bench("append", WARMUP_ITERS, BENCH_ITERS,
                         append_setup, append_bench, append_teardown))

    # ---- replace ----
    tmp_rep = tempfile.mkdtemp()
    npk_rep = NumPack(tmp_rep)
    npk_rep.open()
    npk_rep.save({"arr": data})

    def replace_setup():
        return {"npk": npk_rep}

    def replace_bench(ctx):
        ctx["npk"].replace({"arr": replace_data}, replace_indices)

    results.append(bench("replace", WARMUP_ITERS, BENCH_ITERS,
                         replace_setup, replace_bench))
    npk_rep.close()
    shutil.rmtree(tmp_rep, ignore_errors=True)

    # ---- getitem ----
    tmp_gi = tempfile.mkdtemp()
    npk_gi = NumPack(tmp_gi)
    npk_gi.open()
    npk_gi.save({"arr": data})

    def getitem_setup():
        return {"npk": npk_gi}

    def getitem_bench(ctx):
        _ = ctx["npk"].getitem("arr", getitem_indices)

    results.append(bench("getitem", WARMUP_ITERS, BENCH_ITERS,
                         getitem_setup, getitem_bench))
    npk_gi.close()
    shutil.rmtree(tmp_gi, ignore_errors=True)

    # ---- get_shape ----
    tmp_gs = tempfile.mkdtemp()
    npk_gs = NumPack(tmp_gs)
    npk_gs.open()
    npk_gs.save({"arr": data})

    def get_shape_setup():
        return {"npk": npk_gs}

    def get_shape_bench(ctx):
        _ = ctx["npk"].get_shape("arr")

    results.append(bench("get_shape", WARMUP_ITERS, BENCH_ITERS,
                         get_shape_setup, get_shape_bench))
    npk_gs.close()
    shutil.rmtree(tmp_gs, ignore_errors=True)

    # ---- drop_rows ----
    def drop_setup():
        d = tempfile.mkdtemp()
        npk = NumPack(d)
        npk.open()
        npk.save({"arr": data})
        return {"npk": npk, "dir": d}

    def drop_bench(ctx):
        ctx["npk"].drop("arr", drop_indices)

    def drop_teardown(ctx):
        ctx["npk"].close()
        shutil.rmtree(ctx["dir"], ignore_errors=True)

    results.append(bench("drop_rows", WARMUP_ITERS, BENCH_ITERS,
                         drop_setup, drop_bench, drop_teardown))

    # ---- clone ----
    tmp_cl = tempfile.mkdtemp()
    npk_cl = NumPack(tmp_cl)
    npk_cl.open()
    npk_cl.save({"arr": data})
    clone_counter = [0]

    def clone_setup():
        return {"npk": npk_cl}

    def clone_bench(ctx):
        clone_counter[0] += 1
        ctx["npk"].clone("arr", f"clone_{clone_counter[0]}")

    results.append(bench("clone", WARMUP_ITERS, BENCH_ITERS,
                         clone_setup, clone_bench))
    npk_cl.close()
    shutil.rmtree(tmp_cl, ignore_errors=True)

    # ---- stream_load ----
    tmp_sl = tempfile.mkdtemp()
    npk_sl = NumPack(tmp_sl)
    npk_sl.open()
    npk_sl.save({"arr": data})

    def stream_setup():
        return {"npk": npk_sl}

    def stream_bench(ctx):
        total = 0
        for batch in ctx["npk"].stream_load("arr", STREAM_BUFFER):
            total += batch.shape[0]

    results.append(bench("stream_load", WARMUP_ITERS, BENCH_ITERS,
                         stream_setup, stream_bench))
    npk_sl.close()
    shutil.rmtree(tmp_sl, ignore_errors=True)

    # ---- get_member_list ----
    tmp_ml = tempfile.mkdtemp()
    npk_ml = NumPack(tmp_ml)
    npk_ml.open()
    for i in range(10):
        npk_ml.save({f"arr_{i}": np.zeros((10, 4), dtype=np.float32)})

    def ml_setup():
        return {"npk": npk_ml}

    def ml_bench(ctx):
        _ = ctx["npk"].get_member_list()

    results.append(bench("get_member_list", WARMUP_ITERS, BENCH_ITERS,
                         ml_setup, ml_bench))
    npk_ml.close()
    shutil.rmtree(tmp_ml, ignore_errors=True)

    # ---- get_modify_time ----
    tmp_mt = tempfile.mkdtemp()
    npk_mt = NumPack(tmp_mt)
    npk_mt.open()
    npk_mt.save({"arr": data})

    def mt_setup():
        return {"npk": npk_mt}

    def mt_bench(ctx):
        _ = ctx["npk"].get_modify_time("arr")

    results.append(bench("get_modify_time", WARMUP_ITERS, BENCH_ITERS,
                         mt_setup, mt_bench))
    npk_mt.close()
    shutil.rmtree(tmp_mt, ignore_errors=True)

    # ---- update (compact) ----
    def update_setup():
        d = tempfile.mkdtemp()
        npk = NumPack(d)
        npk.open()
        npk.save({"arr": data})
        npk.drop("arr", drop_indices)
        return {"npk": npk, "dir": d}

    def update_bench(ctx):
        ctx["npk"].update("arr")

    def update_teardown(ctx):
        ctx["npk"].close()
        shutil.rmtree(ctx["dir"], ignore_errors=True)

    results.append(bench("update", WARMUP_ITERS, BENCH_ITERS,
                         update_setup, update_bench, update_teardown))

    # ---- has_array ----
    tmp_ha = tempfile.mkdtemp()
    npk_ha = NumPack(tmp_ha)
    npk_ha.open()
    npk_ha.save({"arr": data})

    def ha_setup():
        return {"npk": npk_ha}

    def ha_bench(ctx):
        _ = ctx["npk"].has_array("arr")

    results.append(bench("has_array", WARMUP_ITERS, BENCH_ITERS,
                         ha_setup, ha_bench))
    npk_ha.close()
    shutil.rmtree(tmp_ha, ignore_errors=True)

    # ---- reset ----
    def reset_setup():
        d = tempfile.mkdtemp()
        npk = NumPack(d)
        npk.open()
        npk.save({"arr": data})
        return {"npk": npk, "dir": d}

    def reset_bench(ctx):
        ctx["npk"].reset()

    def reset_teardown(ctx):
        ctx["npk"].close()
        shutil.rmtree(ctx["dir"], ignore_errors=True)

    results.append(bench("reset", WARMUP_ITERS, BENCH_ITERS,
                         reset_setup, reset_bench, reset_teardown))

    # ---------- Print results ----------
    print("-----------------------------------------------------------------")
    print(f"{'Operation':<20} {'Mean(ms)':>10} {'Std(ms)':>10} {'Min(ms)':>10}")
    print("-----------------------------------------------------------------")
    for r in results:
        print(f"{r['name']:<20} {r['mean_ms']:>10.3f} {r['std_ms']:>10.3f} {r['min_ms']:>10.3f}")
    print("-----------------------------------------------------------------")

    print()
    print("JSON_RESULTS_START")
    d = {r["name"]: round(r["mean_ms"], 6) for r in results}
    print(json.dumps(d))
    print("JSON_RESULTS_END")


if __name__ == "__main__":
    main()
