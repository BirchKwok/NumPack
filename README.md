# NumPack

A high-performance NumPy array storage library combining Rust's speed with Python's simplicity. Optimized for frequent read/write operations on large arrays, with built-in SIMD-accelerated vector similarity search.

## Highlights

| Feature | Performance |
|---------|-------------|
| Row Replacement | **397x faster** than NPY |
| Data Append | **346x faster** than NPY |
| Lazy Loading | **45x faster** than NPY mmap |
| Full Load | **1.57x faster** than NPY |
| Random Access | Equal to NPY mmap |
| Batch Mode | **19x speedup** |

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
# Batch Mode - cached writes (19x speedup)
with npk.batch_mode():
    for i in range(1000):
        arr = npk.load('data')
        arr[:10] *= 2.0
        npk.save({'data': arr})

# Writable Batch Mode - direct mmap (10x speedup)
with npk.writable_batch_mode() as wb:
    arr = wb.load('data')
    arr[:10] *= 2.0  # Auto-persisted
```

### Vector Engine

SIMD-accelerated similarity search (AVX2, AVX-512, NEON, SVE).

```python
from numpack.vector_engine import VectorSearch, StreamingVectorSearch

# In-memory search
engine = VectorSearch()
indices, scores = engine.top_k_search(query, candidates, 'cosine', k=10)

# Multi-query batch (30-50% faster)
all_indices, all_scores = engine.multi_query_top_k(queries, candidates, 'cosine', k=10)

# Streaming from file (for large datasets)
streaming = StreamingVectorSearch()
indices, scores = streaming.streaming_top_k_from_file(
    query, 'vectors.npk', 'embeddings', 'cosine', k=10
)
```

**Supported Metrics:** `cosine`, `dot`, `l2`, `l2sq`, `hamming`, `jaccard`, `kl`, `js`

## Benchmarks

*Tested on macOS Apple Silicon, 1M rows × 10 columns, Float32 (38.1MB)*

| Operation | NumPack | NPY | Advantage |
|-----------|---------|-----|----------:|
| Full Load | 3.72ms | 5.83ms | **1.57x** |
| Lazy Load | 0.002ms | 0.090ms | **45x** |
| Replace 100 rows | 0.034ms | 13.49ms | **397x** |
| Append 100 rows | 0.059ms | 20.40ms | **346x** |
| Random Access | ~equal | ~equal | mmap-optimized |

<details>
<summary><b>Detailed Benchmarks</b></summary>

**Batch Mode Performance (100 consecutive operations):**

| Mode | Time | Speedup |
|------|------|---------|
| Normal | 40.1ms | - |
| Batch Mode | 2.1ms | 19x |
| Writable Batch | 3.9ms | 10x |

**Random/Sequential Access (mmap-optimized):**
- Random access: Equal to NPY mmap for small/medium batches
- Sequential access: Zero-copy optimized, equal to NPY mmap
- 1280x faster than HDF5 for 10K random access

</details>

### When to Use NumPack

| Use Case | Recommendation |
|----------|----------------|
| Frequent modifications | ✅ **NumPack** (397x faster) |
| ML/DL pipelines | ✅ **NumPack** |
| Vector similarity search | ✅ **NumPack** (SIMD) |
| Write-once, read-many | NPY (faster initial write) |
| Extreme compression | NPZ (10% smaller) |

## Documentation

See [`docs/`](docs/) for detailed guides and [`unified_benchmark.py`](unified_benchmark.py) for benchmark code.

## Contributing

Contributions welcome! Please submit a Pull Request.

## License

Apache License 2.0 - see [LICENSE](LICENSE) for details.
