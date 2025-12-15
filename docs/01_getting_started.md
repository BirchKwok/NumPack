# Getting Started with NumPack

NumPack is a high-performance array storage library that combines Rust's performance with Python's ease of use. It provides exceptional performance for both reading and writing large NumPy arrays, with special optimizations for in-place modifications.

## Key Features

- **397x faster** row replacement than NPY
- **405x faster** data append than NPY  
- **54x faster** lazy loading than NPY mmap
- **1.3x faster** full data loading than NPY
- **174x speedup** with Writable Batch Mode for frequent modifications
- Zero-copy operations with minimal memory footprint
- Seamless integration with existing NumPy workflows

## Installation

### From PyPI (Recommended)

#### Prerequisites
- Python >= 3.9
- NumPy >= 1.26.0

```bash
pip install numpack
```

### From Source

#### Prerequisites

- Python >= 3.9
- Rust >= 1.70.0
- NumPy >= 1.26.0
- Appropriate C/C++ compiler

#### Build Steps

1. Clone the repository:
```bash
git clone https://github.com/BirchKwok/NumPack.git
cd NumPack
```

2. Install maturin:
```bash
pip install maturin>=1.0,<2.0
```

3. Build and install:
```bash
# Install in development mode
maturin develop

# Or build wheel package
maturin build --release
pip install target/wheels/numpack-*.whl
```

## Quick Start

### Basic Example

```python
import numpy as np
from numpack import NumPack

# Create some sample data
data = {
    'features': np.random.rand(1000, 100).astype(np.float32),
    'labels': np.random.randint(0, 10, size=1000).astype(np.int32)
}

# Using context manager (Recommended)
with NumPack("my_data.npk") as npk:
    # Save arrays
    npk.save(data)
    
    # Load arrays
    features = npk.load("features")
    labels = npk.load("labels")
    
    print(f"Features shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")
```

### Manual File Management

```python
# Create NumPack instance
npk = NumPack('data.npk')

# Manually open the file
npk.open()

# Perform operations
npk.save({'array': np.random.rand(100, 50)})
data = npk.load('array')

# Manually close the file
npk.close()

# Can reopen if needed
npk.open()
data = npk.load('array')
npk.close()
```

### Using Dictionary-Style Access

```python
with NumPack("data.npk") as npk:
    # Save data
    npk.save({'array1': np.random.rand(100, 50)})
    
    # Load using bracket notation
    data = npk['array1']  # Equivalent to npk.load('array1')
    
    # Iterate over array names
    for name in npk:
        print(f"Array: {name}, Shape: {npk.get_shape(name)}")
```

## Understanding the Context Manager

NumPack supports both context manager and manual file management, but using `with` statement is **strongly recommended** for several reasons:

### Why Use Context Manager?

1. **Automatic Resource Management**: Files are automatically opened and closed
2. **Exception Safety**: Resources are properly cleaned up even if errors occur
3. **Best Reliability**: Especially important on Windows for proper file handle management

```python
# Recommended: Context manager
with NumPack("data.npk") as npk:
    npk.save({'array': data})
    result = npk.load('array')
# File automatically closed here

# Manual management (requires explicit close)
npk = NumPack("data.npk")
npk.open()
npk.save({'array': data})
result = npk.load('array')
npk.close()  # Must remember to call close()
```

### Strict Context Mode

For critical applications, you can enforce context manager usage:

```python
# Strict mode: All operations MUST be within 'with' statement
npk = NumPack("data.npk", strict_context_mode=True)

# This will raise RuntimeError
npk.open()  # Error: strict mode requires 'with' statement

# Correct usage
with npk:
    npk.save({'array': data})
```

## File Management Options

### Drop Existing File

```python
# Automatically delete file if it exists
with NumPack("data.npk", drop_if_exists=True) as npk:
    npk.save({'fresh_data': np.random.rand(100, 50)})
```

### Memory Management Options

```python
# Force garbage collection on close (useful for memory-constrained environments)
with NumPack("data.npk", force_gc_on_close=True) as npk:
    npk.save({'data': large_array})
# GC is triggered on exit
```

**Performance Note**: Setting `force_gc_on_close=True` can impact performance. Only use when memory constraints are critical.

## Data Types Support

NumPack supports all standard NumPy data types:

```python
with NumPack("data.npk") as npk:
    arrays = {
        'bool_array': np.random.rand(100, 10) > 0.5,  # Boolean
        'uint8_array': np.random.randint(0, 255, (100, 10), dtype=np.uint8),
        'int32_array': np.random.randint(-1000, 1000, (100, 10), dtype=np.int32),
        'float32_array': np.random.rand(100, 10).astype(np.float32),
        'float64_array': np.random.rand(100, 10).astype(np.float64),
        'complex64_array': np.random.rand(100, 10).astype(np.complex64),
        'complex128_array': np.random.rand(100, 10).astype(np.complex128),
    }
    npk.save(arrays)
```

## Supported Data Types

- **Boolean**: `bool`
- **Unsigned integers**: `uint8`, `uint16`, `uint32`, `uint64`
- **Signed integers**: `int8`, `int16`, `int32`, `int64`
- **Floating point**: `float16`, `float32`, `float64`
- **Complex numbers**: `complex64`, `complex128`

## Basic Metadata Operations

```python
with NumPack("data.npk") as npk:
    # Save some arrays
    npk.save({
        'array1': np.random.rand(100, 50),
        'array2': np.random.rand(200, 30)
    })
    
    # Get list of all arrays
    arrays = npk.get_member_list()
    print(f"Arrays: {arrays}")  # ['array1', 'array2']
    
    # Get array shape
    shape = npk.get_shape('array1')
    print(f"Shape: {shape}")  # (100, 50)
    
    # Check if array exists
    exists = npk.has_array('array1')
    print(f"Exists: {exists}")  # True
    
    # Get modification time
    mod_time = npk.get_modify_time('array1')
    print(f"Last modified: {mod_time}")  # Unix timestamp
    
    # Get metadata
    metadata = npk.get_metadata()
    print(metadata)
```

## Backend Information

```python
from numpack import get_backend_info

# Get backend information
info = get_backend_info()
print(f"Backend: {info['backend_type']}")  # 'rust'
print(f"Platform: {info['platform']}")     # 'Darwin', 'Linux', or 'Windows'
print(f"Version: {info['version']}")       
```

## Performance Tips for Beginners

1. **Use Context Manager**: Always use `with` statement for automatic resource management
2. **Reuse Instances**: Don't create new NumPack instances in loops
3. **Batch Operations**: Use batch mode for frequent modifications (see [Batch Processing Guide](./03_batch_processing.md))
4. **Choose Right Mode**: Use lazy loading for large arrays you don't need to fully load

### Example: Efficient Usage Pattern

```python
# Efficient: Reuse instance
with NumPack("data.npk") as npk:
    for i in range(100):
        data = npk.load('array')
        process(data)

# Inefficient: Create new instance each time
for i in range(100):
    with NumPack("data.npk") as npk:
        data = npk.load('array')
        process(data)
```

## Next Steps

- [Core Operations](./02_core_operations.md): Learn about all basic operations
- [Batch Processing](./03_batch_processing.md): Understand batch modes for 25-174x speedup
- [Advanced Features](./04_advanced_features.md): Lazy arrays, streaming, and more
- [Performance Guide](./05_performance_guide.md): Optimize your usage for maximum performance

## Common Issues

### Windows File Handle Warning

On Windows, you might see a warning about context manager usage. This is because Windows requires strict file handle management:

```python
# This will show a warning on Windows
npk = NumPack("data.npk")

# Suppress warning (if you know what you're doing)
npk = NumPack("data.npk", warn_no_context=False)

# Best practice: Use context manager
with NumPack("data.npk") as npk:
    # All operations here
    pass
```

### File Already Exists

```python
# Option 1: Drop existing file
with NumPack("data.npk", drop_if_exists=True) as npk:
    npk.save({'data': new_data})

# Option 2: Load existing and update
with NumPack("data.npk") as npk:
    existing = npk.load('data')
    npk.save({'data': modified_data})
```

