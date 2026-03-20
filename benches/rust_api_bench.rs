//! Benchmark for pure Rust ParallelIO API
//!
//! Run:
//!   cargo bench --bench rust_api_bench --no-default-features --features rayon

use ndarray::{Array2, ArrayD};
use numpack::{ParallelIO, DataType, NpkResult};
use std::time::Instant;
use tempfile::TempDir;

const WARMUP_ITERS: usize = 2;
const BENCH_ITERS: usize = 10;

struct BenchResult {
    name: String,
    times_ms: Vec<f64>,
}

impl BenchResult {
    fn mean_ms(&self) -> f64 {
        self.times_ms.iter().sum::<f64>() / self.times_ms.len() as f64
    }
    fn std_ms(&self) -> f64 {
        let mean = self.mean_ms();
        let var = self.times_ms.iter().map(|t| (t - mean).powi(2)).sum::<f64>()
            / self.times_ms.len() as f64;
        var.sqrt()
    }
    fn min_ms(&self) -> f64 {
        self.times_ms.iter().cloned().fold(f64::MAX, f64::min)
    }
}

/// Benchmark a closure (no setup/teardown separation).
fn bench<F: FnMut()>(name: &str, warmup: usize, iters: usize, mut f: F) -> BenchResult {
    for _ in 0..warmup {
        f();
    }
    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let start = Instant::now();
        f();
        times.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    BenchResult { name: name.to_string(), times_ms: times }
}

/// Benchmark with separate setup and teardown.
/// Only the `run` closure is timed.
fn bench_with_setup<S, R, T, F>(
    name: &str,
    warmup: usize,
    iters: usize,
    mut setup: S,
    mut run: R,
    mut teardown: T,
) -> BenchResult
where
    S: FnMut() -> F,
    R: FnMut(&F),
    T: FnMut(F),
{
    for _ in 0..warmup {
        let ctx = setup();
        run(&ctx);
        teardown(ctx);
    }
    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let ctx = setup();
        let start = Instant::now();
        run(&ctx);
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        teardown(ctx);
        times.push(elapsed);
    }
    BenchResult { name: name.to_string(), times_ms: times }
}

struct IoCtx {
    _tmp: TempDir,
    io: ParallelIO,
}

fn main() {
    println!("=================================================================");
    println!("  NumPack Rust API Benchmark");
    println!("=================================================================");
    println!();

    let rows = 100_000usize;
    let cols = 128usize;
    let append_rows_count = 10_000usize;
    let replace_count = 1_000usize;
    let getitem_count = 1_000usize;
    let stream_buffer = 10_000usize;
    let drop_count = 500usize;

    println!("Parameters:");
    println!("  rows          = {}", rows);
    println!("  cols          = {}", cols);
    println!("  append_rows   = {}", append_rows_count);
    println!("  replace_count = {}", replace_count);
    println!("  getitem_count = {}", getitem_count);
    println!("  stream_buffer = {}", stream_buffer);
    println!("  drop_count    = {}", drop_count);
    println!("  warmup_iters  = {}", WARMUP_ITERS);
    println!("  bench_iters   = {}", BENCH_ITERS);
    println!();

    // Pre-generate data
    let data = Array2::<f32>::from_shape_fn((rows, cols), |(r, c)| {
        ((r * cols + c) % 1000) as f32 / 1000.0
    }).into_dyn();

    let append_data = Array2::<f32>::ones((append_rows_count, cols)).into_dyn();
    let replace_data = Array2::<f32>::from_elem((replace_count, cols), 999.0).into_dyn();
    let replace_indices: Vec<i64> = (0..replace_count as i64).collect();
    let getitem_indices: Vec<i64> = (0..getitem_count as i64)
        .map(|i| (i * 97) % rows as i64)
        .collect();
    let drop_indices: Vec<i64> = (0..drop_count as i64).collect();

    let mut results: Vec<BenchResult> = Vec::new();

    // Helper: create IO with data pre-saved
    let make_io = |d: &ArrayD<f32>| -> IoCtx {
        let tmp = TempDir::new().unwrap();
        let io = ParallelIO::new(tmp.path().to_path_buf()).unwrap();
        io.save_arrays(&[("arr".to_string(), d.clone(), DataType::Float32)]).unwrap();
        io.sync_metadata().unwrap();
        IoCtx { _tmp: tmp, io }
    };

    // ---- save ----
    {
        let r = bench_with_setup("save", WARMUP_ITERS, BENCH_ITERS,
            || {
                let tmp = TempDir::new().unwrap();
                let io = ParallelIO::new(tmp.path().to_path_buf()).unwrap();
                (tmp, io)
            },
            |(_, io)| {
                io.save_arrays(&[("arr".to_string(), data.clone(), DataType::Float32)]).unwrap();
                io.sync_metadata().unwrap();
            },
            |_| {},
        );
        results.push(r);
    }

    // ---- load_array ----
    {
        let ctx = make_io(&data);
        let r = bench("load_array", WARMUP_ITERS, BENCH_ITERS, || {
            let _loaded: ArrayD<f32> = ctx.io.load_array("arr").unwrap();
        });
        results.push(r);
    }

    // ---- append_rows ----
    {
        let r = bench_with_setup("append_rows", WARMUP_ITERS, BENCH_ITERS,
            || make_io(&data),
            |ctx| {
                ctx.io.append_rows("arr", &append_data).unwrap();
            },
            |_| {},
        );
        results.push(r);
    }

    // ---- replace_rows ----
    {
        let ctx = make_io(&data);
        let r = bench("replace_rows", WARMUP_ITERS, BENCH_ITERS, || {
            ctx.io.replace_rows("arr", &replace_data, &replace_indices).unwrap();
        });
        results.push(r);
    }

    // ---- getitem ----
    {
        let ctx = make_io(&data);
        let r = bench("getitem", WARMUP_ITERS, BENCH_ITERS, || {
            let _rows: ArrayD<f32> = ctx.io.getitem("arr", &getitem_indices).unwrap();
        });
        results.push(r);
    }

    // ---- get_shape ----
    {
        let ctx = make_io(&data);
        let r = bench("get_shape", WARMUP_ITERS, BENCH_ITERS, || {
            let _s = ctx.io.get_shape("arr").unwrap();
        });
        results.push(r);
    }

    // ---- drop_rows ----
    {
        let r = bench_with_setup("drop_rows", WARMUP_ITERS, BENCH_ITERS,
            || make_io(&data),
            |ctx| {
                ctx.io.drop_arrays("arr", Some(&drop_indices)).unwrap();
            },
            |_| {},
        );
        results.push(r);
    }

    // ---- clone_array ----
    {
        let ctx = make_io(&data);
        let mut counter = 0u64;
        let r = bench("clone_array", WARMUP_ITERS, BENCH_ITERS, || {
            counter += 1;
            let target = format!("clone_{}", counter);
            ctx.io.clone_array("arr", &target).unwrap();
        });
        results.push(r);
    }

    // ---- stream_load ----
    {
        let ctx = make_io(&data);
        let r = bench("stream_load", WARMUP_ITERS, BENCH_ITERS, || {
            let iter = ctx.io.stream_load::<f32>("arr", stream_buffer).unwrap();
            let _total: usize = iter
                .map(|b: NpkResult<ArrayD<f32>>| b.unwrap().shape()[0])
                .sum();
        });
        results.push(r);
    }

    // ---- get_member_list ----
    {
        let tmp = TempDir::new().unwrap();
        let io = ParallelIO::new(tmp.path().to_path_buf()).unwrap();
        for i in 0..10 {
            let small = Array2::<f32>::zeros((10, 4)).into_dyn();
            io.save_arrays(&[(format!("arr_{}", i), small, DataType::Float32)]).unwrap();
        }
        io.sync_metadata().unwrap();
        let r = bench("get_member_list", WARMUP_ITERS, BENCH_ITERS, || {
            let _list = io.get_member_list();
        });
        results.push(r);
    }

    // ---- get_modify_time ----
    {
        let ctx = make_io(&data);
        let r = bench("get_modify_time", WARMUP_ITERS, BENCH_ITERS, || {
            let _t = ctx.io.get_modify_time("arr");
        });
        results.push(r);
    }

    // ---- update (compact) ----
    {
        let r = bench_with_setup("update", WARMUP_ITERS, BENCH_ITERS,
            || {
                let ctx = make_io(&data);
                ctx.io.drop_arrays("arr", Some(&drop_indices)).unwrap();
                ctx
            },
            |ctx| {
                ctx.io.update("arr").unwrap();
            },
            |_| {},
        );
        results.push(r);
    }

    // ---- has_array ----
    {
        let ctx = make_io(&data);
        let r = bench("has_array", WARMUP_ITERS, BENCH_ITERS, || {
            let _b = ctx.io.has_array("arr");
        });
        results.push(r);
    }

    // ---- reset ----
    {
        let r = bench_with_setup("reset", WARMUP_ITERS, BENCH_ITERS,
            || make_io(&data),
            |ctx| {
                ctx.io.reset().unwrap();
            },
            |_| {},
        );
        results.push(r);
    }

    // ---------- Print results ----------
    println!("-----------------------------------------------------------------");
    println!("{:<20} {:>10} {:>10} {:>10}", "Operation", "Mean(ms)", "Std(ms)", "Min(ms)");
    println!("-----------------------------------------------------------------");
    for r in &results {
        println!(
            "{:<20} {:>10.3} {:>10.3} {:>10.3}",
            r.name,
            r.mean_ms(),
            r.std_ms(),
            r.min_ms(),
        );
    }
    println!("-----------------------------------------------------------------");

    println!();
    println!("JSON_RESULTS_START");
    print!("{{");
    for (i, r) in results.iter().enumerate() {
        if i > 0 { print!(", "); }
        print!("\"{}\": {:.6}", r.name, r.mean_ms());
    }
    println!("}}");
    println!("JSON_RESULTS_END");
}
