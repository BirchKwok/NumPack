# Performance Optimization Guide

This guide provides comprehensive performance optimization strategies for NumPack, covering benchmarks, best practices, and common pitfalls.

## Table of Contents

- [Performance Overview](#performance-overview)
- [Benchmark Results](#benchmark-results)
- [Optimization Strategies](#optimization-strategies)
- [Common Performance Pitfalls](#common-performance-pitfalls)
- [Platform-Specific Optimizations](#platform-specific-optimizations)
- [Real-World Use Cases](#real-world-use-cases)

---

## Performance Overview

### NumPack's Performance Advantages

NumPack excels in three key areas:

1. **Data Modification** (397-405x faster than NPY)
   - In-place row replacement
   - Incremental data append
   - Efficient partial updates

2. **Lazy Loading** (54x faster than NPY mmap)
   - Near-instant initialization (0.002ms)
   - On-demand data loading
   - Minimal memory footprint

3. **Batch Processing** (25-174x speedup)
   - Memory-cached batch mode
   - Zero-copy writable batch mode
   - Optimized for frequent modifications

### When to Use NumPack

**Strongly Recommended** (90% of use cases):
- Machine learning and deep learning pipelines
- Real-time data stream processing
- Data annotation and correction workflows
- Feature stores with dynamic updates
- Any scenario requiring frequent data modifications
- Fast data loading requirements

**Consider Alternatives** (10% of use cases):
- Write-once, never modify → Use NPY (faster initial write, 2.2x)
- Frequent single-row access → Use NPY mmap
- Extreme compression requirements → Use NPZ (10% smaller, but 1000x slower)

---

## Benchmark Results

### Test Environment

- **Platform**: macOS (Apple Silicon)
- **Backend**: Rust
- **Method**: timeit with multiple repeats, best time selected
- **Data Type**: Float32 arrays
- **Timing**: Pure operation time (excluding file open/close)

### Large Dataset Performance (1M rows × 10 columns, 38.1MB)

| Operation | NumPack | NPY | NPZ | Zarr | HDF5 | Parquet | NumPack Advantage |
|-----------|---------|-----|-----|------|------|---------|-------------------|
| **Full Load** | **8.27ms** 🥇 | 10.51ms | 181.62ms | 41.40ms | 58.39ms | 23.74ms | 1.3x vs NPY |
| **Lazy Load** | **0.002ms** 🥇 | 0.107ms | N/A | 0.397ms | 0.080ms | N/A | 54x vs NPY |
| **Replace 100 rows** | **0.047ms** 🥇 | 18.51ms | 1574ms | 9.08ms | 0.299ms | 187.65ms | 397x vs NPY |
| **Append 100 rows** | **0.067ms** 🥇 | 27.09ms | 1582ms | 9.98ms | 0.212ms | 204.74ms | 405x vs NPY |
| **Random Access (1K)** | 0.051ms | **0.010ms** 🥇 | 183.16ms | 3.46ms | 4.91ms | 22.80ms | 26x vs NPZ |
| **Save** | 16.15ms | **7.19ms** 🥇 | 1378ms | 80.91ms | 55.66ms | 159.14ms | 2.2x slower |

### Medium Dataset Performance (100K rows × 10 columns, 3.8MB)

| Operation | NumPack | NPY | NPZ | Zarr | HDF5 | NumPack Advantage |
|-----------|---------|-----|-----|------|------|-------------------|
| **Full Load** | 0.98ms | **0.66ms** 🥇 | 18.65ms | 6.24ms | 6.35ms | 1.5x slower |
| **Lazy Load** | **0.002ms** 🥇 | 0.103ms | N/A | 0.444ms | 0.085ms | 51x vs NPY |
| **Replace 100 rows** | **0.039ms** 🥇 | 2.13ms | 159.19ms | 4.39ms | 0.208ms | 55x vs NPY |
| **Append 100 rows** | **0.059ms** 🥇 | 3.29ms | 159.19ms | 4.59ms | 0.206ms | 56x vs NPY |
| **Random Access (1K)** | 0.116ms | **0.010ms** 🥇 | 18.73ms | 1.89ms | 4.82ms | 12x vs NPZ |

### Batch Mode Performance (1M rows × 10 columns)

| Mode | Time (100 operations) | Speedup |
|------|----------------------|---------|
| Normal Mode | 856ms | 1.0x |
| **Batch Mode** | 34ms | **25x faster** |
| **Writable Batch Mode** | 4.9ms | **174x faster** |

### File Size Comparison

| Format | Size | Compression |
|--------|------|-------------|
| NumPack | 38.1 MB | None |
| NPY | 38.1 MB | None |
| NPZ | 34.5 MB | gzip |
| Zarr | 36.2 MB | blosc |
| HDF5 | 38.1 MB | None |
| Parquet | 35.8 MB | snappy |

---

## Optimization Strategies

### Strategy 1: Choose the Right Loading Mode

#### Decision Matrix

```python
def choose_optimal_mode(array_info):
    """
    array_info = {
        'size_mb': 100,
        'access_pattern': 'full' | 'partial' | 'sequential' | 'modify',
        'frequency': 'once' | 'frequent',
        'memory_available_mb': 8000
    }
    """
    size = array_info['size_mb']
    pattern = array_info['access_pattern']
    frequency = array_info['frequency']
    memory = array_info['memory_available_mb']
    
    # Modification-heavy workload
    if pattern == 'modify':
        if frequency == 'frequent':
            if size > memory * 0.5:
                return 'writable_batch_mode'  # 174x speedup, low memory
            else:
                return 'batch_mode'  # 25x speedup, supports shape changes
        else:
            return 'normal_replace'  # Still 397x faster than NPY
    
    # Read-heavy workload
    elif pattern == 'full':
        if frequency == 'frequent':
            return 'eager_cached'  # Load once, reuse
        else:
            return 'eager'  # 1.3x faster than NPY
    
    elif pattern == 'partial':
        if size > memory * 0.5:
            return 'lazy'  # 54x faster initialization
        else:
            return 'lazy'  # Still best for partial access
    
    elif pattern == 'sequential':
        return 'stream'  # Memory-efficient
    
    return 'lazy'  # Safe default
```

#### Example Implementation

```python
from numpack import NumPack
import numpy as np

# Get array info
with NumPack("data.npk") as npk:
    shape = npk.get_shape('features')
    size_mb = (shape[0] * shape[1] * 4) / 1024 / 1024  # float32
    
    # Determine optimal mode
    if size_mb > 100:
        # Large array: Use lazy loading
        features = npk.load('features', lazy=True)
        subset = features[1000:2000]
    else:
        # Small array: Eager loading
        features = npk.load('features')
```

### Strategy 2: Batch Operations Effectively

#### Pattern: Multiple Modifications

```python
# Inefficient: Normal mode (856ms for 100 ops)
with NumPack("data.npk") as npk:
    for i in range(100):
        arr = npk.load('features')
        arr *= 1.1
        npk.save({'features': arr})

# Good: Batch mode (34ms, 25x faster)
with NumPack("data.npk") as npk:
    with npk.batch_mode():
        for i in range(100):
            arr = npk.load('features')
            arr *= 1.1
            npk.save({'features': arr})

# Best: Writable batch mode (4.9ms, 174x faster)
with NumPack("data.npk") as npk:
    with npk.writable_batch_mode() as wb:
        arr = wb.load('features')  # Load once
        for i in range(100):
            arr *= 1.1  # Direct modification
```

#### Pattern: Mixed Read/Write

```python
# Optimal: Separate batch contexts for different operations
with NumPack("data.npk") as npk:
    # Read-only analysis: Lazy loading
    lazy_arr = npk.load('features', lazy=True)
    stats = compute_statistics(lazy_arr)
    
    # Frequent modifications: Writable batch mode
    with npk.writable_batch_mode() as wb:
        arr = wb.load('features')
        apply_transformations(arr)
    
    # Shape changes: Batch mode
    with npk.batch_mode():
        arr = npk.load('features')
        npk.append({'features': augment_data(arr)})
```

### Strategy 3: Optimize Data Types

```python
import numpy as np

# File size and load time comparison
dtypes = {
    'float16': 2,  # Half precision (acceptable for many ML tasks)
    'float32': 4,  # Single precision (recommended)
    'float64': 8,  # Double precision (overkill for most cases)
}

# Example: Use appropriate dtype
with NumPack("optimized.npk", drop_if_exists=True) as npk:
    # Original: float64 (8 bytes/element)
    data_f64 = np.random.rand(1000000, 100)  # 800 MB
    
    # Optimized: float32 (4 bytes/element)
    data_f32 = data_f64.astype(np.float32)  # 400 MB, 50% savings
    
    # Trade-off: float16 (2 bytes/element)
    data_f16 = data_f64.astype(np.float16)  # 200 MB, 75% savings
    
    npk.save({'data': data_f32})  # Recommended balance
```

**Performance Impact:**
- Load time is roughly proportional to file size
- float32 → float16: 50% faster load, slight precision loss
- float64 → float32: 50% faster load, minimal precision loss for most ML

### Strategy 4: Minimize File Opens

```python
# Very Inefficient: Open file in every iteration
def process_data_bad():
    for i in range(1000):
        with NumPack("data.npk") as npk:  # File open overhead
            data = npk.load('features')
            process(data)

# Efficient: Open once
def process_data_good():
    with NumPack("data.npk") as npk:  # Single file open
        for i in range(1000):
            data = npk.load('features')
            process(data)

# Best: Open once + cache array
def process_data_best():
    with NumPack("data.npk") as npk:
        data = npk.load('features')  # Load once
        for i in range(1000):
            process(data)
```

**Timing (1000 iterations):**
- Bad: ~15 seconds (file open overhead)
- Good: ~1 second (single open, multiple loads)
- Best: ~0.01 seconds (single open + load)

### Strategy 5: Use Streaming for Large Datasets

```python
import numpy as np

# Memory usage comparison
def compute_statistics_approaches(npk, array_name):
    """Compare different approaches for computing statistics"""
    
    # Memory-intensive: Load full array
    def approach1():
        arr = npk.load(array_name)  # All in memory
        return arr.mean(), arr.std()
    
    # Memory-efficient: Streaming
    def approach2():
        total_sum = 0
        total_sq_sum = 0
        total_count = 0
        
        for batch in npk.stream_load(array_name, buffer_size=10000):
            total_sum += batch.sum()
            total_sq_sum += (batch ** 2).sum()
            total_count += batch.size
        
        mean = total_sum / total_count
        std = np.sqrt(total_sq_sum / total_count - mean ** 2)
        return mean, std
    
    # Compare
    import psutil, os
    process = psutil.Process(os.getpid())
    
    mem1 = process.memory_info().rss / 1024 / 1024
    result1 = approach1()
    mem1_after = process.memory_info().rss / 1024 / 1024
    
    mem2 = process.memory_info().rss / 1024 / 1024
    result2 = approach2()
    mem2_after = process.memory_info().rss / 1024 / 1024
    
    print(f"Approach 1 memory: {mem1_after - mem1:.1f} MB")
    print(f"Approach 2 memory: {mem2_after - mem2:.1f} MB")
    
    return result1, result2
```

### Strategy 6: Optimize Buffer Sizes

```python
def find_optimal_buffer_size(npk, array_name, available_memory_mb=100):
    """Determine optimal buffer size for streaming"""
    shape = npk.get_shape(array_name)
    rows, cols = shape
    
    # Estimate bytes per row
    dtype = npk.load(array_name, lazy=True).dtype
    bytes_per_row = cols * dtype.itemsize
    
    # Calculate optimal buffer size
    max_buffer_size = (available_memory_mb * 1024 * 1024) // bytes_per_row
    
    # Recommended: 10K-100K rows
    optimal = min(max(max_buffer_size, 10000), 100000)
    
    return optimal

# Usage
with NumPack("data.npk") as npk:
    buffer_size = find_optimal_buffer_size(npk, 'features', 
                                           available_memory_mb=100)
    
    for batch in npk.stream_load('features', buffer_size=buffer_size):
        process(batch)
```

---

## Common Performance Pitfalls

### Pitfall 1: Repeated Array Loading

```python
# BAD: Load inside loop (100x overhead)
with NumPack("data.npk") as npk:
    for i in range(100):
        arr = npk.load('features')  # Reloads every time
        value = arr[i]

# GOOD: Load once
with NumPack("data.npk") as npk:
    arr = npk.load('features')  # Load once
    for i in range(100):
        value = arr[i]

# Performance difference: 100x faster
```

### Pitfall 2: Not Using Batch Mode for Modifications

```python
# BAD: Normal mode for frequent modifications
with NumPack("data.npk") as npk:
    for i in range(50):
        arr = npk.load('features')
        arr += 1
        npk.save({'features': arr})
# Time: ~1800ms

# GOOD: Batch mode
with NumPack("data.npk") as npk:
    with npk.batch_mode():
        for i in range(50):
            arr = npk.load('features')
            arr += 1
            npk.save({'features': arr})
# Time: ~48ms (37x faster)

# BEST: Writable batch mode
with NumPack("data.npk") as npk:
    with npk.writable_batch_mode() as wb:
        arr = wb.load('features')
        for i in range(50):
            arr += 1
# Time: ~27ms (67x faster)
```

### Pitfall 3: Wrong Loading Mode for Access Pattern

```python
# BAD: Eager load for partial access
with NumPack("large.npk") as npk:
    arr = npk.load('features')  # Loads 10GB into memory
    subset = arr[1000:2000]     # Only need 1MB
# Memory: 10GB, Time: 8.27ms

# GOOD: Lazy load
with NumPack("large.npk") as npk:
    arr = npk.load('features', lazy=True)  # ~0 memory
    subset = arr[1000:2000]                # Only loads needed portion
# Memory: ~1MB, Time: 0.05ms (165x faster)
```

### Pitfall 4: Inefficient Data Types

```python
# BAD: Using float64 unnecessarily
data = np.random.rand(1000000, 100)  # float64, 800MB
with NumPack("data.npk") as npk:
    npk.save({'features': data})
# Save time: 32ms, File size: 800MB

# GOOD: Use float32
data = np.random.rand(1000000, 100).astype(np.float32)  # 400MB
with NumPack("data.npk") as npk:
    npk.save({'features': data})
# Save time: 16ms (2x faster), File size: 400MB (50% smaller)
```

### Pitfall 5: Not Caching Array Reference in Writable Batch Mode

```python
# BAD: Load in every iteration
with npk.writable_batch_mode() as wb:
    for i in range(1000):
        arr = wb.load('features')  # Creates new mmap
        arr *= 1.1
# Time: ~500ms

# GOOD: Cache reference
with npk.writable_batch_mode() as wb:
    arr = wb.load('features')  # Load once
    for i in range(1000):
        arr *= 1.1
# Time: ~4.9ms (100x faster)
```

### Pitfall 6: Forgetting to Compact After Deletions

```python
# BAD: Many deletions without compaction
with NumPack("data.npk") as npk:
    for i in range(10000):
        npk.drop('features', [i])
# Disk space not reclaimed, file still large

# GOOD: Compact after deletions
with NumPack("data.npk") as npk:
    # Logical deletion
    npk.drop('features', list(range(10000)))
    
    # Physical compaction
    npk.update('features')
# Disk space reclaimed, file size reduced
```

---

## Platform-Specific Optimizations

### macOS / Linux Optimizations

```python
# Leverage mmap efficiently on Unix-like systems
with NumPack("data.npk") as npk:
    # Writable batch mode is very efficient
    with npk.writable_batch_mode() as wb:
        arr = wb.load('features')
        # OS page cache optimization automatic
        arr *= 2.0
```

### Windows Optimizations

```python
# Windows: Always use context manager
with NumPack("data.npk") as npk:  # Recommended on Windows
    # Proper file handle management
    npk.save({'data': array})

# Windows: Force cleanup if needed
from numpack import force_cleanup_windows_handles
force_cleanup_windows_handles()

# Windows: Strict mode for reliability
npk = NumPack("data.npk", strict_context_mode=True)
with npk:
    npk.save({'data': array})
```

### Memory-Constrained Systems

```python
# Optimize for limited RAM
with NumPack("data.npk") as npk:
    # Use streaming instead of batch mode
    for batch in npk.stream_load('features', buffer_size=1000):
        process(batch)
        # Each batch is freed after processing
    
    # Use writable batch mode (zero memory overhead)
    with npk.writable_batch_mode() as wb:
        arr = wb.load('features')  # Virtual memory only
        arr *= 2.0
```

---

## Real-World Use Cases

### Use Case 1: Machine Learning Training Pipeline

```python
from numpack import NumPack
import numpy as np

class DataLoader:
    def __init__(self, npk_file, batch_size=32):
        self.npk_file = npk_file
        self.batch_size = batch_size
        self.npk = None
        self.features_lazy = None
        self.labels_lazy = None
    
    def __enter__(self):
        self.npk = NumPack(self.npk_file).__enter__()
        # Lazy load for efficient batch access
        self.features_lazy = self.npk.load('features', lazy=True)
        self.labels_lazy = self.npk.load('labels', lazy=True)
        return self
    
    def __exit__(self, *args):
        return self.npk.__exit__(*args)
    
    def __iter__(self):
        n_samples = len(self.features_lazy)
        indices = np.random.permutation(n_samples)
        
        for start in range(0, n_samples, self.batch_size):
            end = min(start + self.batch_size, n_samples)
            batch_indices = indices[start:end]
            
            # Efficient batch access
            X = self.features_lazy[batch_indices]
            y = self.labels_lazy[batch_indices]
            
            yield X, y

# Usage
with DataLoader("training_data.npk", batch_size=32) as loader:
    for epoch in range(10):
        for X_batch, y_batch in loader:
            train_step(X_batch, y_batch)

# Performance: 54x faster initialization than NPY mmap
```

### Use Case 2: Real-Time Data Annotation

```python
from numpack import NumPack
import numpy as np

def annotation_workflow(data_file, annotation_file):
    """Interactive data annotation with real-time updates"""
    
    with NumPack(data_file) as data_npk:
        with NumPack(annotation_file) as anno_npk:
            # Initialize annotations if needed
            if not anno_npk.has_array('labels'):
                n_samples = data_npk.get_shape('samples')[0]
                labels = np.full(n_samples, -1, dtype=np.int32)
                anno_npk.save({'labels': labels})
            
            # Use writable batch mode for real-time updates
            with anno_npk.writable_batch_mode() as wb:
                labels = wb.load('labels')
                
                # Lazy load data for efficient access
                samples = data_npk.load('samples', lazy=True)
                
                # Annotation loop
                for i, sample in enumerate(samples):
                    # Display sample
                    display(sample)
                    
                    # Get user annotation
                    label = get_user_input()
                    
                    # Update immediately (no save needed)
                    labels[i] = label
                    
                    # Auto-saved to file
            
            print("Annotation completed and saved")

# Performance: 174x faster than normal mode
# Benefit: Real-time persistence, no data loss on crash
```

### Use Case 3: Feature Engineering Pipeline

```python
from numpack import NumPack
import numpy as np

def feature_engineering_pipeline(input_file, output_file):
    """Memory-efficient feature engineering"""
    
    with NumPack(input_file) as input_npk:
        with NumPack(output_file, drop_if_exists=True) as output_npk:
            # Get input shape
            shape = input_npk.get_shape('raw_features')
            n_samples = shape[0]
            
            # Process in batches (memory-efficient)
            batch_size = 10000
            first_batch = True
            
            for batch in input_npk.stream_load('raw_features', buffer_size=batch_size):
                # Feature engineering
                normalized = (batch - batch.mean()) / batch.std()
                squared = batch ** 2
                interactions = batch[:, :10] * batch[:, 10:20]
                
                # Combine features
                engineered = np.concatenate([
                    normalized,
                    squared,
                    interactions
                ], axis=1)
                
                # Save or append
                if first_batch:
                    output_npk.save({'features': engineered})
                    first_batch = False
                else:
                    output_npk.append({'features': engineered})
                
                print(f"Processed {len(engineered)} samples")

# Performance: Handles TB-scale data with MB-scale memory
```

### Use Case 4: Model Checkpoint Management

```python
from numpack import NumPack
import numpy as np

class ModelCheckpoint:
    def __init__(self, checkpoint_file):
        self.file = checkpoint_file
    
    def save_checkpoint(self, epoch, model_weights, optimizer_state, metrics):
        """Save training checkpoint"""
        with NumPack(self.file) as npk:
            # Use batch mode for atomic update
            with npk.batch_mode():
                # Save weights
                for name, weight in model_weights.items():
                    npk.save({f'weights_{name}': weight})
                
                # Save optimizer state
                for name, state in optimizer_state.items():
                    npk.save({f'optimizer_{name}': state})
                
                # Save metadata
                npk.save({
                    'epoch': np.array([epoch]),
                    'metrics': np.array(list(metrics.values()))
                })
    
    def load_checkpoint(self):
        """Load training checkpoint"""
        with NumPack(self.file) as npk:
            # Lazy load to check what's available
            arrays = npk.get_member_list()
            
            # Load weights
            weights = {}
            optimizer_state = {}
            
            for name in arrays:
                if name.startswith('weights_'):
                    weights[name[8:]] = npk.load(name)
                elif name.startswith('optimizer_'):
                    optimizer_state[name[10:]] = npk.load(name)
            
            epoch = npk.load('epoch')[0]
            metrics = npk.load('metrics')
            
            return epoch, weights, optimizer_state, metrics

# Usage
checkpoint = ModelCheckpoint("model_checkpoint.npk")

# Training loop
for epoch in range(100):
    train_model()
    
    # Save checkpoint
    checkpoint.save_checkpoint(
        epoch=epoch,
        model_weights=model.get_weights(),
        optimizer_state=optimizer.get_state(),
        metrics={'loss': 0.5, 'accuracy': 0.95}
    )

# Resume training
epoch, weights, optimizer_state, metrics = checkpoint.load_checkpoint()
model.set_weights(weights)
optimizer.set_state(optimizer_state)

# Performance: Batch mode ensures atomic checkpoint updates
```

---

## Benchmark Methodology

All benchmarks use:
- `timeit` for precise timing
- Multiple repeats, best time selected
- Pure operation time (excluding file open/close overhead)
- Float32 arrays (4 bytes per element)
- macOS Apple Silicon (results may vary by platform)

### Running Your Own Benchmarks

```python
import timeit
import numpy as np
from numpack import NumPack

def benchmark_operation(operation_func, number=100, repeat=5):
    """Benchmark an operation with precise timing"""
    times = timeit.repeat(
        operation_func,
        number=number,
        repeat=repeat
    )
    best_time = min(times) / number
    return best_time * 1000  # Convert to milliseconds

# Example: Benchmark load operation
def setup():
    with NumPack("benchmark.npk", drop_if_exists=True) as npk:
        npk.save({'data': np.random.rand(100000, 100).astype(np.float32)})

def test_eager_load():
    with NumPack("benchmark.npk") as npk:
        data = npk.load('data')

def test_lazy_load():
    with NumPack("benchmark.npk") as npk:
        data = npk.load('data', lazy=True)

# Run benchmarks
setup()
eager_time = benchmark_operation(test_eager_load)
lazy_time = benchmark_operation(test_lazy_load)

print(f"Eager load: {eager_time:.3f}ms")
print(f"Lazy load: {lazy_time:.3f}ms")
print(f"Speedup: {eager_time/lazy_time:.1f}x")
```

---

## Summary

### Key Performance Takeaways

1. **For Modifications**: Use writable batch mode (174x speedup)
2. **For Partial Access**: Use lazy loading (54x faster initialization)
3. **For Frequent Operations**: Use batch mode (25-37x speedup)
4. **For Large Datasets**: Use streaming (memory-efficient)
5. **For Initial Write**: NumPack is 2.2x slower than NPY (acceptable trade-off for modification speed)

### Quick Reference

| Scenario | Recommended Approach | Speedup |
|----------|---------------------|---------|
| Frequent modifications | Writable batch mode | 174x |
| Batch processing | Batch mode | 25-37x |
| Row replacement | Normal replace | 397x |
| Data append | Normal append | 405x |
| Large array partial access | Lazy loading | 54x init |
| Sequential processing | Streaming | Memory-efficient |
| Full array load | Eager loading | 1.3x |

### Optimization Checklist

- [ ] Use appropriate loading mode (eager/lazy/stream)
- [ ] Use batch mode for frequent modifications
- [ ] Cache array references (avoid repeated loads)
- [ ] Choose optimal data types (float32 vs float64)
- [ ] Use context manager for proper resource management
- [ ] Compact arrays after many deletions
- [ ] Set appropriate buffer sizes for streaming
- [ ] Minimize file open/close operations
- [ ] Profile your specific workload
- [ ] Test on target platform

