# Advanced Features API Reference

This document covers advanced NumPack features including lazy loading, streaming, indexing operations, and specialized use cases.

## Table of Contents

- [Lazy Arrays](#lazy-arrays)
- [Streaming Operations](#streaming-operations)
- [Advanced Indexing](#advanced-indexing)
- [In-Place Operations](#in-place-operations)
- [Memory Management](#memory-management)
- [Cross-Platform Considerations](#cross-platform-considerations)
- [Pack & Unpack](#pack--unpack)

---

## Lazy Arrays

### Overview

Lazy loading provides memory-mapped array access with on-demand data loading. This is ideal for:
- Large arrays that don't fit in memory
- Accessing only portions of large datasets
- Minimizing memory footprint
- Fast initialization (54x faster than NPY mmap)

### Basic Usage

```python
import numpy as np
from numpack import NumPack

# Create large dataset
with NumPack("large_data.npk", drop_if_exists=True) as npk:
    npk.save({'features': np.random.rand(10000000, 100).astype(np.float32)})

# Lazy load - only 0.002ms initialization
with NumPack("large_data.npk") as npk:
    lazy_array = npk.load('features', lazy=True)
    print(type(lazy_array))  # <class 'LazyArray'>
    print(lazy_array.shape)  # (10000000, 100)
    
    # Data is loaded on demand
    subset = lazy_array[1000:2000]  # Only loads these rows
    print(subset.shape)  # (1000, 100)
```

### LazyArray API

LazyArray supports most NumPy array operations:

#### Shape and Type Information

```python
with NumPack("data.npk") as npk:
    lazy_arr = npk.load('features', lazy=True)
    
    # Shape information
    print(lazy_arr.shape)      # (rows, columns)
    print(lazy_arr.ndim)       # 2
    print(lazy_arr.size)       # rows * columns
    print(lazy_arr.dtype)      # Data type
    print(lazy_arr.nbytes)     # Total bytes
    
    # Check dimensions
    rows, cols = lazy_arr.shape
```

#### Indexing and Slicing

LazyArray supports advanced indexing:

```python
with NumPack("data.npk") as npk:
    lazy_arr = npk.load('features', lazy=True)
    
    # Single row
    row = lazy_arr[0]
    print(row.shape)  # (100,)
    
    # Range of rows
    rows = lazy_arr[100:200]
    print(rows.shape)  # (100, 100)
    
    # Step indexing
    every_10th = lazy_arr[::10]
    print(every_10th.shape)  # (1000, 100)
    
    # Fancy indexing with list
    specific_rows = lazy_arr[[0, 10, 20, 30]]
    print(specific_rows.shape)  # (4, 100)
    
    # NumPy array indexing
    indices = np.array([1, 5, 10, 15, 20])
    selected = lazy_arr[indices]
    print(selected.shape)  # (5, 100)
    
    # Boolean indexing (loads full array)
    mask = np.random.rand(len(lazy_arr)) > 0.5
    filtered = lazy_arr[mask]
```

#### Iteration

```python
with NumPack("data.npk") as npk:
    lazy_arr = npk.load('features', lazy=True)
    
    # Iterate over rows
    for i, row in enumerate(lazy_arr):
        if i >= 10:
            break
        print(f"Row {i} shape: {row.shape}")
    
    # Batch iteration (more efficient)
    batch_size = 1000
    for i in range(0, len(lazy_arr), batch_size):
        batch = lazy_arr[i:i+batch_size]
        process_batch(batch)
```

#### Conversion to NumPy

```python
with NumPack("data.npk") as npk:
    lazy_arr = npk.load('features', lazy=True)
    
    # Convert subset to NumPy array
    subset = lazy_arr[1000:2000]
    numpy_array = np.asarray(subset)
    
    # Convert entire array (loads all into memory)
    full_array = np.asarray(lazy_arr)
    
    # Alternative: use tolist() for Python lists
    python_list = lazy_arr[0:10].tolist()
```

### Performance Comparison

| Operation | Eager Load | Lazy Load | Advantage |
|-----------|-----------|-----------|-----------|
| Initialize | 8.27ms | **0.002ms** | 4135x faster |
| Load full array | 8.27ms | 8.30ms | Similar |
| Load 1000 rows | 8.27ms | **0.05ms** | 165x faster |
| Load 1 row | 8.27ms | **0.001ms** | 8270x faster |

**Recommendation:**
- Use lazy loading when accessing < 10% of the array
- Use eager loading when you need most of the array
- Lazy loading is ideal for exploratory data analysis

### Best Practices

#### Good Usage

```python
# Load once, use multiple times
with NumPack("data.npk") as npk:
    lazy_arr = npk.load('features', lazy=True)
    
    # Efficient: Access different portions
    train_data = lazy_arr[:80000]
    val_data = lazy_arr[80000:90000]
    test_data = lazy_arr[90000:]
```

#### Avoid

```python
# Don't create lazy array repeatedly
for i in range(100):
    with NumPack("data.npk") as npk:
        lazy_arr = npk.load('features', lazy=True)
        data = lazy_arr[i]  # Inefficient

# Better: Load once
with NumPack("data.npk") as npk:
    lazy_arr = npk.load('features', lazy=True)
    for i in range(100):
        data = lazy_arr[i]
```

---

## Streaming Operations

### Overview

Streaming provides memory-efficient batch processing of large arrays without loading the entire dataset into memory.

### API Reference

#### `stream_load(array_name, buffer_size=None)`

Stream array data in batches.

**Parameters:**
- `array_name` (str): Name of the array to stream
- `buffer_size` (Optional[int]): Rows per batch. If `None`, loads one row at a time

**Returns:**
- `Iterator[np.ndarray]`: Iterator yielding batches

### Basic Example

```python
with NumPack("data.npk") as npk:
    # Create large array
    npk.save({'large_array': np.random.rand(1000000, 100).astype(np.float32)})
    
    # Stream in batches
    for batch in npk.stream_load('large_array', buffer_size=10000):
        print(f"Processing batch: {batch.shape}")
        # Process batch...
        result = process_batch(batch)
```

### Advanced Streaming Patterns

#### Pattern 1: Incremental Aggregation

```python
with NumPack("data.npk") as npk:
    # Compute statistics without loading full array
    total_sum = 0
    total_count = 0
    
    for batch in npk.stream_load('features', buffer_size=10000):
        total_sum += batch.sum()
        total_count += batch.size
    
    mean = total_sum / total_count
    print(f"Mean: {mean}")
```

#### Pattern 2: Streaming Transformation

```python
with NumPack("input.npk") as npk_in:
    with NumPack("output.npk", drop_if_exists=True) as npk_out:
        # Process and save in batches
        first_batch = True
        
        for batch in npk_in.stream_load('features', buffer_size=10000):
            # Transform
            transformed = transform_function(batch)
            
            # Save
            if first_batch:
                npk_out.save({'transformed': transformed})
                first_batch = False
            else:
                npk_out.append({'transformed': transformed})
```

#### Pattern 3: Streaming with Multiple Arrays

```python
with NumPack("data.npk") as npk:
    # Stream multiple arrays synchronously
    features_stream = npk.stream_load('features', buffer_size=1000)
    labels_stream = npk.stream_load('labels', buffer_size=1000)
    
    for feature_batch, label_batch in zip(features_stream, labels_stream):
        process(feature_batch, label_batch)
```

### Buffer Size Selection

| Buffer Size | Memory Usage | I/O Efficiency | Use Case |
|-------------|--------------|----------------|----------|
| 1 | Minimal | Low | Extreme memory constraints |
| 100 | Very Low | Low | Very tight memory |
| 1,000 | Low | Medium | Memory constrained |
| 10,000 | Medium | High | Balanced (recommended) |
| 100,000 | High | Very High | Large memory available |

**Formula for buffer size:**
```python
# Estimate based on available memory
row_size_bytes = array_shape[1] * dtype_size
available_memory_mb = 100  # MB available for buffering
buffer_size = (available_memory_mb * 1024 * 1024) // row_size_bytes
```

---

## Advanced Indexing

### Fancy Indexing

NumPack supports NumPy-style fancy indexing:

```python
with NumPack("data.npk") as npk:
    npk.save({'features': np.random.rand(10000, 100)})
    
    # Integer array indexing
    indices = np.array([0, 10, 20, 30, 40])
    subset = npk.getitem('features', indices)
    
    # List indexing
    subset = npk.getitem('features', [1, 5, 10, 15])
    
    # With lazy arrays
    lazy_arr = npk.load('features', lazy=True)
    subset = lazy_arr[[0, 100, 200, 300]]
```

### Boolean Indexing

```python
with NumPack("data.npk") as npk:
    # Create data
    features = np.random.rand(1000, 100)
    labels = np.random.randint(0, 2, size=1000)
    npk.save({'features': features, 'labels': labels})
    
    # Load for boolean indexing
    labels = npk.load('labels')
    features_lazy = npk.load('features', lazy=True)
    
    # Boolean mask
    mask = labels == 1
    positive_samples = features_lazy[mask]
    print(positive_samples.shape)
```

### Multi-Dimensional Slicing

```python
with NumPack("data.npk") as npk:
    lazy_arr = npk.load('features', lazy=True)
    
    # Row and column slicing
    subset = lazy_arr[100:200, 10:50]
    
    # Select specific columns
    columns = lazy_arr[:, [0, 5, 10, 15]]
    
    # Every nth row
    downsampled = lazy_arr[::10, :]
```

---

## In-Place Operations

### Efficient Modification Patterns

NumPack supports efficient in-place modifications:

```python
with NumPack("data.npk") as npk:
    # Create data
    npk.save({'features': np.random.rand(10000, 100)})
    
    # In-place modification with writable batch mode (most efficient)
    with npk.writable_batch_mode() as wb:
        arr = wb.load('features')
        
        # In-place operations
        arr *= 2.0
        arr += 1.0
        np.sqrt(arr, out=arr)  # In-place sqrt
        
        # No save needed - already in file
```

### Supported In-Place Operations

```python
with npk.writable_batch_mode() as wb:
    arr = wb.load('features')
    
    # Arithmetic operations
    arr += 5
    arr -= 2
    arr *= 3
    arr /= 4
    arr **= 2
    
    # NumPy universal functions (with out parameter)
    np.sqrt(arr, out=arr)
    np.exp(arr, out=arr)
    np.log(arr, out=arr)
    np.sin(arr, out=arr)
    
    # Element assignment
    arr[0] = 999.0
    arr[10:20] = 0.0
    arr[:, 0] = 1.0
    
    # Conditional assignment
    arr[arr < 0] = 0  # ReLU activation
```

---

## Memory Management

### Memory-Efficient Patterns

#### Pattern 1: Chunked Processing

```python
def process_large_array_in_chunks(npk, array_name, chunk_size=10000):
    """Process large array without loading all into memory"""
    shape = npk.get_shape(array_name)
    total_rows = shape[0]
    
    for start in range(0, total_rows, chunk_size):
        end = min(start + chunk_size, total_rows)
        
        # Process chunk
        chunk = npk.getitem(array_name, list(range(start, end)))
        result = process(chunk)
        
        # Save or aggregate result
        yield result

# Usage
with NumPack("data.npk") as npk:
    for result in process_large_array_in_chunks(npk, 'features'):
        handle_result(result)
```

#### Pattern 2: Generator-Based Pipeline

```python
def data_pipeline(npk, array_name, batch_size=1000):
    """Memory-efficient data pipeline"""
    for batch in npk.stream_load(array_name, buffer_size=batch_size):
        # Stage 1: Normalize
        normalized = (batch - batch.mean()) / batch.std()
        
        # Stage 2: Transform
        transformed = apply_transform(normalized)
        
        # Stage 3: Augment
        augmented = augment_data(transformed)
        
        yield augmented

# Usage
with NumPack("data.npk") as npk:
    for processed_batch in data_pipeline(npk, 'features'):
        train_model(processed_batch)
```

#### Pattern 3: Lazy Evaluation Chain

```python
with NumPack("data.npk") as npk:
    # Lazy load - no memory used yet
    lazy_arr = npk.load('features', lazy=True)
    
    # Define transformations (not executed yet)
    def transform_pipeline(data):
        data = data * 2.0
        data = data + 1.0
        data = np.log(data)
        return data
    
    # Process in chunks (memory efficient)
    chunk_size = 1000
    for i in range(0, len(lazy_arr), chunk_size):
        chunk = lazy_arr[i:i+chunk_size]
        transformed = transform_pipeline(chunk)
        process(transformed)
        # Chunk is freed after each iteration
```

### Memory Monitoring

```python
import psutil
import os

def get_memory_usage():
    """Get current process memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

# Monitor memory during processing
with NumPack("data.npk") as npk:
    print(f"Initial memory: {get_memory_usage():.1f} MB")
    
    # Eager loading
    arr = npk.load('features')
    print(f"After eager load: {get_memory_usage():.1f} MB")
    del arr
    
    # Lazy loading
    lazy_arr = npk.load('features', lazy=True)
    print(f"After lazy load: {get_memory_usage():.1f} MB")
    
    # Writable batch mode
    with npk.writable_batch_mode() as wb:
        arr = wb.load('features')
        print(f"After mmap load: {get_memory_usage():.1f} MB")
```

---

## Cross-Platform Considerations

### Windows-Specific Handling

NumPack has special considerations for Windows file handle management:

```python
# Windows: Use context manager (recommended)
with NumPack("data.npk") as npk:
    npk.save({'data': array})
    result = npk.load('data')

# Windows: Suppress context warning if needed
npk = NumPack("data.npk", warn_no_context=False)

# Windows: Force cleanup (if encountering handle issues)
from numpack import force_cleanup_windows_handles
force_cleanup_windows_handles()
```

### Strict Context Mode (Cross-Platform)

For applications that require strict file management:

```python
# Enforce context manager usage
npk = NumPack("data.npk", strict_context_mode=True)

# This will raise an error
npk.open()  # RuntimeError: strict mode requires 'with' statement

# Correct usage
with npk:
    npk.save({'data': array})
```

### Platform Detection

```python
from numpack import get_backend_info
import platform

info = get_backend_info()
print(f"Platform: {info['platform']}")  # Darwin, Linux, or Windows
print(f"Is Windows: {info['is_windows']}")

# Platform-specific code
if info['is_windows']:
    # Windows-specific handling
    npk = NumPack("data.npk", strict_context_mode=True)
else:
    # Unix-like systems
    npk = NumPack("data.npk")
```

---

## Performance Optimization Tips

### Tip 1: Choose the Right Loading Mode

```python
# Decision tree for loading mode
def choose_loading_mode(array_size_mb, access_pattern):
    if access_pattern == "full":
        return "eager"
    elif access_pattern == "partial" and array_size_mb > 100:
        return "lazy"
    elif access_pattern == "streaming":
        return "stream"
    elif access_pattern == "modify_inplace":
        return "writable_batch"
    else:
        return "lazy"

# Usage
with NumPack("data.npk") as npk:
    shape = npk.get_shape('features')
    size_mb = (shape[0] * shape[1] * 4) / 1024 / 1024  # float32
    
    mode = choose_loading_mode(size_mb, "partial")
    
    if mode == "eager":
        arr = npk.load('features')
    elif mode == "lazy":
        arr = npk.load('features', lazy=True)
    elif mode == "stream":
        for batch in npk.stream_load('features', buffer_size=10000):
            process(batch)
```

### Tip 2: Batch Similar Operations

```python
# Inefficient: Multiple small operations
with NumPack("data.npk") as npk:
    for i in range(100):
        arr = npk.load('features')
        arr[i] *= 2
        npk.save({'features': arr})

# Efficient: Batch mode
with NumPack("data.npk") as npk:
    with npk.batch_mode():
        for i in range(100):
            arr = npk.load('features')
            arr[i] *= 2
            npk.save({'features': arr})
```

### Tip 3: Use Appropriate Data Types

```python
# Use smaller dtypes when precision allows
with NumPack("data.npk") as npk:
    # float32 instead of float64 (50% space savings)
    npk.save({'features': array.astype(np.float32)})
    
    # int32 instead of int64 for reasonable integers
    npk.save({'labels': labels.astype(np.int32)})
    
    # uint8 for small positive integers (0-255)
    npk.save({'categories': categories.astype(np.uint8)})
```

### Tip 4: Minimize File Opens

```python
# Inefficient: Open file repeatedly
for i in range(100):
    with NumPack("data.npk") as npk:
        data = npk.load('features')
        process(data)

# Efficient: Open once
with NumPack("data.npk") as npk:
    for i in range(100):
        data = npk.load('features')
        process(data)
```

---

## Complete Advanced Example

Here's a comprehensive example combining multiple advanced features:

```python
import numpy as np
from numpack import NumPack
import psutil
import os

def get_memory_mb():
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

# Create large dataset
print("Creating dataset...")
with NumPack("advanced_demo.npk", drop_if_exists=True) as npk:
    features = np.random.rand(1000000, 100).astype(np.float32)
    labels = np.random.randint(0, 10, size=1000000).astype(np.int32)
    npk.save({'features': features, 'labels': labels})
    del features, labels

print(f"Memory after creation: {get_memory_mb():.1f} MB")

# Lazy loading for exploration
print("\n1. Lazy loading for exploration...")
with NumPack("advanced_demo.npk") as npk:
    lazy_features = npk.load('features', lazy=True)
    print(f"Shape: {lazy_features.shape}")
    print(f"Memory: {get_memory_mb():.1f} MB")
    
    # Explore subset
    sample = lazy_features[:1000]
    print(f"Sample mean: {sample.mean():.4f}")

# Streaming for statistics
print("\n2. Streaming for statistics...")
with NumPack("advanced_demo.npk") as npk:
    total_sum = 0
    total_count = 0
    
    for batch in npk.stream_load('features', buffer_size=10000):
        total_sum += batch.sum()
        total_count += batch.size
    
    mean = total_sum / total_count
    print(f"Global mean: {mean:.4f}")

# Writable batch mode for normalization
print("\n3. Writable batch mode for normalization...")
with NumPack("advanced_demo.npk") as npk:
    with npk.writable_batch_mode() as wb:
        features = wb.load('features')
        
        # Normalize in chunks
        chunk_size = 100000
        for i in range(0, len(features), chunk_size):
            chunk = features[i:i+chunk_size]
            chunk -= mean
            chunk /= chunk.std()
        
        print(f"Memory during normalization: {get_memory_mb():.1f} MB")

# Batch mode for augmentation
print("\n4. Batch mode for augmentation...")
with NumPack("advanced_demo.npk") as npk:
    with npk.batch_mode():
        features = npk.load('features')
        
        # Add noise
        features += np.random.randn(*features.shape) * 0.01
        npk.save({'features': features})
        
        # Append augmented data
        augmented = features + np.random.randn(*features.shape) * 0.05
        npk.append({'features': augmented[:10000], 
                   'labels': npk.load('labels')[:10000]})

# Final shape
with NumPack("advanced_demo.npk") as npk:
    final_shape = npk.get_shape('features')
    print(f"\nFinal shape: {final_shape}")
    print(f"Final memory: {get_memory_mb():.1f} MB")
```

This example demonstrates:
- Lazy loading for exploration
- Streaming for statistics computation
- Writable batch mode for in-place normalization
- Batch mode for augmentation
- Memory-efficient processing of large datasets

---

## Pack & Unpack

### Overview

NumPack provides a portable `.npkg` format for packaging entire NumPack directories into a single compressed file. This is ideal for:

- **Migration**: Move NumPack data between machines or environments
- **Backup**: Create compressed backups of NumPack directories
- **Sharing**: Share NumPack datasets as a single file
- **Archival**: Long-term storage with integrity verification

The `.npkg` format preserves all NumPack files:
- Metadata (`metadata.npkm`)
- Data files (`data_*.npkd`)
- Deletion bitmaps (`deleted_*.npkb`)

### `pack(source, target=None, compression=True, overwrite=False)`

Pack a NumPack directory into a single `.npkg` file.

#### Parameters

- `source` (str | Path): Path to the NumPack directory to pack
- `target` (str | Path, optional): Output `.npkg` file path. Default: `<source>.npkg`
- `compression` (bool): Enable Zstd compression. Default: `True`
- `overwrite` (bool): Overwrite existing file. Default: `False`

#### Returns

- `Path`: Path to the created `.npkg` file

#### Example

```python
from numpack import pack

# Basic pack - creates data.npkg in same directory
pack('data.npk')

# Custom output path
pack('data.npk', '/backup/my_data.npkg')

# Without compression (faster, larger file)
pack('data.npk', 'data_uncompressed.npkg', compression=False)

# Overwrite existing package
pack('data.npk', 'data.npkg', overwrite=True)
```

---

### `unpack(source, target=None, overwrite=False, verify=True)`

Unpack a `.npkg` file into a NumPack directory.

#### Parameters

- `source` (str | Path): Path to the `.npkg` file
- `target` (str | Path, optional): Output directory path. Default: `<source_stem>.npk`
- `overwrite` (bool): Overwrite existing directory. Default: `False`
- `verify` (bool): Verify CRC32 checksum after extraction. Default: `True`

#### Returns

- `Path`: Path to the extracted NumPack directory

#### Example

```python
from numpack import unpack

# Basic unpack - creates data.npk in same directory
unpack('data.npkg')

# Custom output path
unpack('data.npkg', '/restored/my_data')

# Overwrite existing directory
unpack('data.npkg', 'data.npk', overwrite=True)

# Skip checksum verification (faster but less safe)
unpack('data.npkg', verify=False)
```

---

### `get_package_info(source)`

Get information about a `.npkg` package without extracting it.

#### Parameters

- `source` (str | Path): Path to the `.npkg` file

#### Returns

- `dict`: Package information containing:
  - `version`: Package format version
  - `compression`: Compression type code (0=None, 1=Zstd)
  - `compression_name`: Compression algorithm name
  - `file_count`: Number of files in package
  - `files`: List of file details (name, original_size, compressed_size)
  - `total_original_size`: Total uncompressed size in bytes
  - `total_compressed_size`: Total compressed size in bytes
  - `compression_ratio`: Ratio of compressed to original size

#### Example

```python
from numpack import get_package_info

info = get_package_info('data.npkg')

print(f"Version: {info['version']}")
print(f"Compression: {info['compression_name']}")
print(f"Files: {info['file_count']}")
print(f"Original size: {info['total_original_size'] / 1024 / 1024:.2f} MB")
print(f"Compressed size: {info['total_compressed_size'] / 1024 / 1024:.2f} MB")
print(f"Compression ratio: {info['compression_ratio']:.1%}")

# List all files
for f in info['files']:
    print(f"  {f['name']}: {f['original_size']} bytes")
```

---

### Complete Migration Example

```python
import numpy as np
from numpack import NumPack, pack, unpack, get_package_info

# Create original NumPack with data
with NumPack("original.npk", drop_if_exists=True) as npk:
    npk.save({
        'embeddings': np.random.rand(10000, 128).astype(np.float32),
        'labels': np.arange(10000).astype(np.int64)
    })
    # Simulate some deletions
    npk.drop('embeddings', indexes=[0, 100, 200])

print("Original data created")

# Pack for migration
package_path = pack("original.npk", "backup/data.npkg")
print(f"Packed to: {package_path}")

# View package info
info = get_package_info(package_path)
print(f"Package contains {info['file_count']} files")
print(f"Compression ratio: {info['compression_ratio']:.1%}")

# Simulate migration - unpack to new location
restored_path = unpack(package_path, "restored/data.npk")
print(f"Restored to: {restored_path}")

# Verify data integrity
with NumPack(restored_path) as npk:
    embeddings = npk.load('embeddings')
    labels = npk.load('labels')
    print(f"Embeddings shape: {embeddings.shape}")  # (9997, 128) - 3 rows deleted
    print(f"Labels shape: {labels.shape}")
```

---

### .npkg File Format

The `.npkg` format is a binary format designed for portability and integrity:

```
Header (13 bytes):
  - Magic Number: "NPKG" (4 bytes)
  - Version: uint32 (4 bytes) - currently 1
  - Compression: uint8 (1 byte) - 0=None, 1=Zstd
  - File Count: uint32 (4 bytes)

Per File Entry:
  - Path Length: uint32 (4 bytes)
  - Path: UTF-8 string (variable)
  - Original Size: uint64 (8 bytes)
  - Compressed Size: uint64 (8 bytes) - 0 if not compressed
  - Data: bytes (variable)

Footer:
  - CRC32 Checksum: uint32 (4 bytes)
```

**Features:**
- Cross-platform compatible (little-endian)
- CRC32 checksum for integrity verification
- Zstd compression (with zlib fallback)
- Preserves all NumPack files including deletion bitmaps

