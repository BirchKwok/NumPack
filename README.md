# NumPack

A high-performance NumPy array storage library combining Rust's speed with Python's simplicity. Optimized for frequent read/write operations on large arrays, with built-in SIMD-accelerated vector similarity search.

## Highlights

| Feature | Performance |
|---------|-------------|
| Row Replacement | **344x faster** than NPY |
| Data Append | **338x faster** than NPY |
| Lazy Loading | **51x faster** than NPY mmap |
| Full Load | **1.64x faster** than NPY |
| Batch Mode | **21x speedup** |
| Writable Batch | **92x speedup** |

**Core Capabilities:**
- Zero-copy mmap operations with minimal memory footprint
- SIMD-accelerated Vector Engine (AVX2, AVX-512, NEON, SVE)
- Batch & Writable Batch modes for high-frequency modifications
- Supports all NumPy dtypes: bool, int8-64, uint8-64, float16/32/64, complex64/128

## Installation

### Python

```bash
pip install numpack
```

**Requirements:** Python ≥ 3.9, NumPy ≥ 1.26.0

### Rust

Add to your `Cargo.toml`:

```toml
[dependencies]
numpack = "0.5.1"
ndarray = "0.16"
```

**Features:**
- `rayon` (default) - Parallel processing support
- `avx512` - AVX-512 SIMD optimizations  
- `io-uring-support` - io_uring on Linux

**Requirements:** Rust ≥ 1.70.0

<details>
<summary><b>Build from Source</b></summary>

```bash
# Prerequisites: Rust >= 1.70.0 (rustup.rs), C/C++ compiler
git clone https://github.com/BirchKwok/NumPack.git
cd NumPack
pip install maturin>=1.0,<2.0
maturin develop  # or: maturin build --release
```
</details>

#### Basic Usage

```rust
use numpack::prelude::*;
use ndarray::{ArrayD, Array2, IxDyn};
use std::path::PathBuf;

fn main() -> NpkResult<()> {
    // Create or open a NumPack storage
    let io = ParallelIO::new(PathBuf::from("data.npk"))?;
    
    // Save arrays with explicit dtype
    let data = Array2::<f32>::from_shape_fn((1000, 128), |(r, c)| (r * 128 + c) as f32);
    io.save_arrays(&[("embeddings".to_string(), data.into_dyn(), DataType::Float32)])?;
    
    // Load array back (mmap-based, with automatic cache)
    let loaded: ArrayD<f32> = io.load_array("embeddings")?;
    assert_eq!(loaded.shape(), &[1000, 128]);
    
    // In-place append (file append mode, no rewrite)
    let extra = Array2::<f32>::ones((50, 128)).into_dyn();
    io.append_rows("embeddings", &extra)?;
    assert_eq!(io.get_shape("embeddings")?, vec![1050, 128]);
    
    // Metadata is written on drop, or call sync_metadata() explicitly
    io.sync_metadata()?;
    
    Ok(())
}
```

#### API Reference

**Storage Operations:**

| Method | Description |
|--------|-------------|
| `ParallelIO::new(path)` | Create or open a storage directory |
| `save_arrays(&[(name, array, dtype)])` | Save one or more arrays (auto-parallel for large data) |
| `sync_metadata()` | Persist metadata to disk |
| `reset()` | Delete all arrays and metadata |

**Read Operations (mmap-based):**

| Method | Description |
|--------|-------------|
| `load_array::<T>(name)` | Load full array via mmap with automatic cache |
| `getitem::<T>(name, &indexes)` | Read specific rows by index (supports negative indexing) |
| `read_rows(name, &indexes)` | Read specific rows as raw bytes |
| `stream_load::<T>(name, buffer_size)` | Streaming iterator yielding batches of rows |
| `get_array_view(name)` | Get a lazy array view (mmap-backed) |

**Write Operations (in-place):**

| Method | Description |
|--------|-------------|
| `append_rows::<T>(name, &data)` | In-place append to existing array (file append mode) |
| `replace_rows::<T>(name, &data, &indices)` | In-place row replacement with pwrite |
| `clone_array(source, target)` | Deep copy an array to a new name |

**Delete & Compact:**

| Method | Description |
|--------|-------------|
| `drop_arrays(name, Some(&indices))` | Logical delete rows (bitmap-based) |
| `drop_arrays(name, None)` | Physical delete entire array |
| `compact_array(name)` | Remove logically deleted rows, reclaim space |

**Metadata & Query:**

| Method | Description |
|--------|-------------|
| `has_array(name)` | Check if an array exists |
| `list_arrays()` / `get_member_list()` | List all array names |
| `get_array_metadata(name)` | Get array metadata (shape, dtype, size, etc.) |
| `get_shape(name)` | Get logical shape (accounts for deletions) |
| `get_modify_time(name)` | Get last modification timestamp (microseconds) |

**Aliases (Python API compatible):**

| Rust Method | Python Equivalent |
|-------------|-------------------|
| `append_rows()` | `NumPack.append()` |
| `load_array()` | `NumPack.load()` |
| `getitem()` | `NumPack.getitem()` |
| `get_shape()` | `NumPack.get_shape()` |
| `get_modify_time()` | `NumPack.get_modify_time()` |
| `clone_array()` | `NumPack.clone()` |
| `get_member_list()` | `NumPack.get_member_list()` |
| `update()` | `NumPack.update()` |
| `stream_load()` | `NumPack.stream_load()` |

**Array Operations:**

```rust
use numpack::prelude::*;
use ndarray::Array2;
use std::path::PathBuf;

fn example() -> NpkResult<()> {
    let io = ParallelIO::new(PathBuf::from("data.npk"))?;

    // Save
    let data = Array2::<f32>::zeros((1000, 128)).into_dyn();
    io.save_arrays(&[("embeddings".to_string(), data, DataType::Float32)])?;

    // In-place append (no file rewrite, O(new_data) complexity)
    let extra = Array2::<f32>::ones((100, 128)).into_dyn();
    io.append_rows("embeddings", &extra)?;

    // Load full array (mmap with LRU cache, invalidated on write)
    let arr: ndarray::ArrayD<f32> = io.load_array("embeddings")?;
    assert_eq!(arr.shape(), &[1100, 128]);

    // Random access by index (mmap, contiguous block detection)
    let rows: ndarray::ArrayD<f32> = io.getitem("embeddings", &[0, 10, -1])?;
    assert_eq!(rows.shape(), &[3, 128]);

    // Replace rows in-place (pwrite, no file rewrite)
    let new_rows = Array2::<f32>::from_elem((2, 128), 42.0).into_dyn();
    io.replace_rows("embeddings", &new_rows, &[0, 1])?;

    // Logical delete + compact
    io.drop_arrays("embeddings", Some(&[5, 6, 7]))?;
    io.compact_array("embeddings")?;

    // Clone array
    io.clone_array("embeddings", "embeddings_backup")?;

    // Query
    let shape = io.get_shape("embeddings")?;
    let names = io.list_arrays();
    let mtime = io.get_modify_time("embeddings");

    // Delete entire array
    io.drop_arrays("embeddings_backup", None)?;

    io.sync_metadata()?;
    Ok(())
}
```

**Streaming Load:**

```rust
// Process large arrays in batches without loading everything into memory
let iter: StreamIterator<f32> = io.stream_load("large_data", 10000)?;
for batch_result in iter {
    let batch: ndarray::ArrayD<f32> = batch_result?;
    // Process batch (up to 10000 rows each)
    println!("Batch shape: {:?}", batch.shape());
}
```

**Data Type Mapping:**

| NumPack Type | Rust Type | Size |
|--------------|-----------|------|
| `DataType::Bool` | `bool` | 1 byte |
| `DataType::Int8` | `i8` | 1 byte |
| `DataType::Int16` | `i16` | 2 bytes |
| `DataType::Int32` | `i32` | 4 bytes |
| `DataType::Int64` | `i64` | 8 bytes |
| `DataType::Uint8` | `u8` | 1 byte |
| `DataType::Uint16` | `u16` | 2 bytes |
| `DataType::Uint32` | `u32` | 4 bytes |
| `DataType::Uint64` | `u64` | 8 bytes |
| `DataType::Float16` | `half::f16` | 2 bytes |
| `DataType::Float32` | `f32` | 4 bytes |
| `DataType::Float64` | `f64` | 8 bytes |
| `DataType::Complex64` | `num_complex::Complex32` | 8 bytes |
| `DataType::Complex128` | `num_complex::Complex64` | 16 bytes |

#### Key Design Features

- **mmap-based Reading:** All read operations (`load_array`, `getitem`, `stream_load`) use `memmap2` with an automatic cache keyed by `last_modified` timestamp. Cache is invalidated on write/append/delete.
- **In-place Append:** `append_rows` opens the data file in append mode and writes only the new data. No existing data is rewritten. Metadata is updated incrementally.
- **In-place Replace:** `replace_rows` uses positional writes (`pwrite`) to update specific rows without touching unrelated data.
- **Logical Deletion:** `drop_arrays` with indices uses a bitmap to mark rows as deleted. Read operations automatically skip deleted rows. Call `compact_array` to physically reclaim space.
- **Adaptive Parallelism:** `save_arrays` automatically uses Rayon parallel processing when saving multiple arrays with total size > 10MB.
- **Adaptive Buffering:** Write buffer sizes are tuned by data size (256KB / 4MB / 16MB for small / medium / large arrays).

#### Concurrent Access

Multiple threads can safely write to the same storage concurrently (since v0.5.1+):

```rust
use numpack::prelude::*;
use ndarray::Array2;
use std::path::PathBuf;
use std::thread;

fn concurrent_write() -> NpkResult<()> {
    let dir = "/tmp/numpack_data";
    std::fs::create_dir_all(dir)?;
    
    let handles: Vec<_> = (0..10)
        .map(|i| {
            let dir = dir.to_string();
            thread::spawn(move || {
                let io = ParallelIO::new(PathBuf::from(dir))?;
                let data = Array2::<f32>::ones((100, 128)).into_dyn();
                io.save_arrays(&[(format!("chunk_{}", i), data, DataType::Float32)])?;
                io.sync_metadata()?;
                Ok::<_, NpkError>(())
            })
        })
        .collect();
    
    for h in handles {
        h.join().unwrap()?;
    }
    
    Ok(())
}
```

**Best Practices for Concurrent Access:**
- Each thread creates its own `ParallelIO` instance
- Call `sync_metadata()` before dropping the instance
- For read-heavy workloads, use separate read instances

#### Performance Tips

```rust
// 1. Batch saves for multiple arrays (auto-parallel for large data)
let arrays: Vec<(String, ndarray::ArrayD<f32>, DataType)> = vec![
    ("a".to_string(), data_a, DataType::Float32),
    ("b".to_string(), data_b, DataType::Float32),
];
io.save_arrays(&arrays)?;

// 2. Use append_rows for incremental data (fastest, no rewrite)
let new_data = Array2::<f32>::ones((100, 128)).into_dyn();
io.append_rows("a", &new_data)?;

// 3. Use replace_rows for updating existing rows (pwrite, no rewrite)
let updated = Array2::<f32>::zeros((3, 128)).into_dyn();
io.replace_rows("a", &updated, &[0, 1, 2])?;

// 4. Use stream_load for large arrays that don't fit in memory
let iter: StreamIterator<f32> = io.stream_load("a", 50000)?;
for batch in iter {
    let batch = batch?;
    // process batch...
}

// 5. Call sync_metadata() once after all operations
io.sync_metadata()?;

// 6. Use compact_array() periodically after many deletions
io.drop_arrays("a", Some(&[0, 1, 2]))?;
io.compact_array("a")?;
```

#### Error Handling

```rust
use numpack::core::error::{NpkError, NpkResult};

match io.get_array_metadata("nonexistent") {
    Ok(meta) => println!("Found: {:?}", meta.shape),
    Err(NpkError::ArrayNotFound(name)) => println!("Array {} not found", name),
    Err(e) => eprintln!("Error: {:?}", e),
}
```

### Batch Modes

```python
# Batch Mode - cached writes (21x speedup)
with npk.batch_mode():
    for i in range(1000):
        arr = npk.load('data')
        arr[:10] *= 2.0
        npk.save({'data': arr})

# Writable Batch Mode - direct mmap (108x speedup)
with npk.writable_batch_mode() as wb:
    arr = wb.load('data')
    arr[:10] *= 2.0  # Auto-persisted
```

### Vector Engine

SIMD-accelerated similarity search (AVX2, AVX-512, NEON, SVE).

```python
from numpack.vector_engine import VectorEngine, StreamingVectorEngine

# In-memory search
engine = VectorEngine()
indices, scores = engine.top_k_search(query, candidates, 'cosine', k=10)

# Multi-query batch (30-50% faster)
all_indices, all_scores = engine.multi_query_top_k(queries, candidates, 'cosine', k=10)

# Streaming from file (for large datasets)
streaming = StreamingVectorEngine()
indices, scores = streaming.streaming_top_k_from_file(
    query, 'vectors.npk', 'embeddings', 'cosine', k=10
)
```

**Supported Metrics:** `cosine`, `dot`, `l2`, `l2sq`, `hamming`, `jaccard`, `kl`, `js`

### Format Conversion

Convert between NumPack and other formats (PyTorch, Arrow, Parquet, SafeTensors).

```python
from numpack.io import from_tensor, to_tensor, from_table, to_table

# Memory <-> .npk (zero-copy when possible)
from_tensor(tensor, 'output.npk', array_name='embeddings')  # tensor -> .npk
tensor = to_tensor('input.npk', array_name='embeddings')     # .npk -> tensor

from_table(table, 'output.npk')  # PyArrow Table -> .npk
table = to_table('input.npk')     # .npk -> PyArrow Table

# File <-> File (streaming for large files)
from numpack.io import from_pt, to_pt
from_pt('model.pt', 'output.npk')  # .pt -> .npk
to_pt('input.npk', 'output.pt')    # .npk -> .pt
```

**Supported formats:** PyTorch (.pt), Feather, Parquet, SafeTensors, NumPy (.npy), HDF5, Zarr, CSV

### Pack & Unpack

Portable `.npkg` format for easy migration and sharing.

```python
from numpack import pack, unpack, get_package_info

# Pack NumPack directory into a single .npkg file
pack('data.npk')                          # -> data.npkg (with Zstd compression)
pack('data.npk', 'backup/data.npkg')      # Custom output path

# Unpack .npkg back to NumPack directory
unpack('data.npkg')                       # -> data.npk
unpack('data.npkg', 'restored/')          # Custom restore path

# View package info without extracting
info = get_package_info('data.npkg')
print(f"Files: {info['file_count']}, Compression: {info['compression_ratio']:.1%}")
```

## Benchmarks

*Tested on macOS Apple Silicon, 1M rows × 10 columns, Float32 (38.1MB)*

| Operation | NumPack | NPY | Advantage |
|-----------|---------|-----|----------:|
| Full Load | 4.00ms | 6.56ms | **1.64x** |
| Lazy Load | 0.002ms | 0.102ms | **51x** |
| Replace 100 rows | 0.040ms | 13.74ms | **344x** |
| Append 100 rows | 0.054ms | 18.26ms | **338x** |
| Random Access (100) | 0.004ms | 0.002ms | ~equal |

<details>
<summary><b>Multi-Format Comparison</b></summary>

**Core Operations (1M × 10, Float32, ~38.1MB):**

| Operation | NumPack | NPY | Zarr | HDF5 | Parquet | Arrow |
|-----------|--------:|----:|-----:|-----:|--------:|------:|
| Save | 11.94ms | 6.48ms | 70.91ms | 58.07ms | 142.11ms | 16.85ms |
| Full Load | 4.00ms | 6.56ms | 32.86ms | 53.99ms | 16.49ms | 12.39ms |
| Lazy Load | 0.002ms | 0.102ms | 0.374ms | 0.082ms | N/A | N/A |
| Replace 100 | 0.040ms | 13.74ms | 7.61ms | 0.29ms | 162.48ms | 26.93ms |
| Append 100 | 0.054ms | 18.26ms | 9.05ms | 0.39ms | 173.45ms | 42.46ms |

**Random Access Performance:**

| Batch Size | NumPack | NPY (mmap) | Zarr | HDF5 | Parquet | Arrow |
|------------|--------:|-----------:|-----:|-----:|--------:|------:|
| 100 rows | 0.004ms | 0.002ms | 2.66ms | 0.66ms | 16.25ms | 12.43ms |
| 1K rows | 0.025ms | 0.021ms | 2.86ms | 5.02ms | 16.48ms | 12.61ms |
| 10K rows | 0.118ms | 0.112ms | 16.63ms | 505.71ms | 17.45ms | 12.81ms |

**Batch Mode Performance (100 consecutive operations):**

| Mode | Time | Speedup |
|------|-----:|--------:|
| Normal | 414ms | - |
| Batch Mode | 20.1ms | **21x** |
| Writable Batch | 4.5ms | **92x** |

**File Size:**

| Format | Size | Compression |
|--------|-----:|:-----------:|
| NumPack | 38.15MB | - |
| NPY | 38.15MB | - |
| NPZ | 34.25MB | ✓ |
| Zarr | 34.13MB | ✓ |
| HDF5 | 38.18MB | - |
| Parquet | 44.09MB | ✓ |
| Arrow | 38.16MB | - |

</details>

### When to Use NumPack

| Use Case | Recommendation |
|----------|----------------|
| Frequent modifications | ✅ **NumPack** (344x faster) |
| ML/DL pipelines | ✅ **NumPack** (zero-copy random access, no full load) |
| Vector similarity search | ✅ **NumPack** (SIMD) |
| Write-once, read-many | ✅ **NumPack** (1.64x faster read) |
| Extreme compression | ✅ **NumPack** `.npkg` (better ratio, streaming, high I/O) |
| RAG/Embedding storage | ✅ **NumPack** (fast retrieval + SIMD search) |
| Feature store | ✅ **NumPack** (real-time updates + low latency) |
| Memory-constrained environments | ✅ **NumPack** (mmap + lazy loading) |
| Multi-process data sharing | ✅ **NumPack** (zero-copy mmap) |
| Incremental data pipelines | ✅ **NumPack** (338x faster append) |
| Real-time feature updates | ✅ **NumPack** (ms-level replace) |

## Documentation

See [`docs/`](docs/) for detailed guides and [`unified_benchmark.py`](unified_benchmark.py) for benchmark code.

## Contributing

Contributions welcome! Please submit a Pull Request.

## License

Apache License 2.0 - see [LICENSE](LICENSE) for details.
