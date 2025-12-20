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

```bash
pip install numpack
```

**Requirements:** Python ≥ 3.9, NumPy ≥ 1.26.0

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

## Quick Start

```python
import numpy as np
from numpack import NumPack

with NumPack("data.npk") as npk:
    # Save
    npk.save({'embeddings': np.random.rand(10000, 128).astype(np.float32)})
    
    # Load (normal or lazy)
    data = npk.load("embeddings")
    lazy = npk.load("embeddings", lazy=True)
    
    # Modify
    npk.replace({'embeddings': new_rows}, indices=[0, 1, 2])
    npk.append({'embeddings': more_rows})
    npk.drop('embeddings', [0, 1, 2])  # drop rows
    
    # Random access
    subset = npk.getitem('embeddings', [100, 200, 300])
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
from numpack.io import from_torch, to_torch, from_arrow, to_arrow

# Memory ↔ .npk (zero-copy when possible)
from_torch(tensor, 'output.npk', array_name='embeddings')  # tensor → .npk
tensor = to_torch('input.npk', array_name='embeddings')     # .npk → tensor

from_arrow(table, 'output.npk')  # PyArrow Table → .npk
table = to_arrow('input.npk')     # .npk → PyArrow Table

# File ↔ File (streaming for large files)
from numpack.io import from_torch_file, to_torch_file
from_torch_file('model.pt', 'output.npk')  # .pt → .npk
to_torch_file('input.npk', 'output.pt')    # .npk → .pt
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
