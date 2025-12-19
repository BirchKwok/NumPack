# NumPack

A high-performance NumPy array storage library combining Rust's speed with Python's simplicity. Optimized for frequent read/write operations on large arrays, with built-in SIMD-accelerated vector similarity search.

## Highlights

| Feature | Performance |
|---------|-------------|
| Row Replacement | **318x faster** than NPY |
| Data Append | **329x faster** than NPY |
| Lazy Loading | **46x faster** than NPY mmap |
| Full Load | **1.74x faster** than NPY |
| Random Access | **27% faster** than NPY mmap |
| Batch Mode | **21x speedup** |

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
| Full Load | 3.81ms | 6.64ms | **1.74x** |
| Lazy Load | 0.002ms | 0.107ms | **46x** |
| Replace 100 rows | 0.042ms | 13.35ms | **318x** |
| Append 100 rows | 0.056ms | 18.30ms | **329x** |
| Random Access (100) | 0.002ms | 0.003ms | **27% faster** |

<details>
<summary><b>Detailed Benchmarks</b></summary>

**Batch Mode Performance (100 consecutive operations):**

| Mode | Time | Speedup |
|------|------|--------|
| Normal | 417ms | - |
| Batch Mode | 19.7ms | 21x |
| Writable Batch | 3.8ms | 108x |

**Random/Sequential Access (mmap-optimized):**
- Random access: Equal to NPY mmap for small/medium batches
- Sequential access: Zero-copy optimized, equal to NPY mmap
- 1280x faster than HDF5 for 10K random access

</details>

### When to Use NumPack

| Use Case | Recommendation |
|----------|----------------|
| Frequent modifications | ✅ **NumPack** (318x faster) |
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
