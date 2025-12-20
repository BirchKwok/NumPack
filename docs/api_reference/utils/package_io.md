# Package I/O API Reference

Functions for packaging and extracting NumPack directories.

## Overview

NumPack directories can be packaged into single `.npkg` files for easier distribution and storage. The package format supports optional compression.

## Import

```python
from numpack import pack, unpack, get_package_info
```

---

## Functions

### `pack(input_path, output_path, compression=True, chunk_size=8*1024*1024)`

Package a NumPack directory into a single `.npkg` file.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_path` | `str` or `Path` | *required* | Input NumPack directory path |
| `output_path` | `str` or `Path` | *required* | Output `.npkg` file path |
| `compression` | `bool` | `True` | Enable Zstd compression |
| `chunk_size` | `int` | `8MB` | Streaming chunk size in bytes |

#### Returns

- `None`

#### Notes

- Uses streaming to minimize memory usage
- Zstd compression provides excellent compression ratio
- Falls back to zlib if zstandard is not installed

#### Example

```python
from numpack import pack

# Package with compression (default)
pack('model.npk', 'model.npkg')

# Package without compression
pack('model.npk', 'model.npkg', compression=False)

# Custom chunk size for very large files
pack('huge.npk', 'huge.npkg', chunk_size=16*1024*1024)
```

---

### `unpack(input_path, output_path)`

Extract a `.npkg` package to a NumPack directory.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_path` | `str` or `Path` | *required* | Input `.npkg` file path |
| `output_path` | `str` or `Path` | *required* | Output NumPack directory path |

#### Returns

- `None`

#### Notes

- Automatically detects compression
- Uses streaming extraction for large packages
- Verifies checksum integrity

#### Example

```python
from numpack import unpack

# Extract package
unpack('model.npkg', 'model.npk')

# Then use normally
from numpack import NumPack
with NumPack('model.npk') as npk:
    data = npk.load('weights')
```

---

### `get_package_info(path)`

Get metadata from a `.npkg` package without extracting.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | `str` or `Path` | *required* | Path to `.npkg` file |

#### Returns

- `Dict[str, Any]`: Package metadata

#### Return Value Structure

```python
{
    'version': int,           # Package format version
    'compression': str,       # 'zstd', 'zlib', or 'none'
    'file_count': int,        # Number of files in package
    'files': [                # List of file info
        {
            'path': str,      # Relative file path
            'original_size': int,
            'compressed_size': int,
        },
        ...
    ],
    'total_original_size': int,
    'total_compressed_size': int,
    'compression_ratio': float,
}
```

#### Example

```python
from numpack import get_package_info

info = get_package_info('model.npkg')
print(f"Files: {info['file_count']}")
print(f"Compression: {info['compression']}")
print(f"Ratio: {info['compression_ratio']:.2f}x")

for f in info['files']:
    print(f"  {f['path']}: {f['original_size']} bytes")
```

---

## Package Format (.npkg v2)

### Structure

```
Header:
  - Magic Number (4 bytes): "NPKG"
  - Version (4 bytes): Format version (2)
  - Compression (1 byte): 0=None, 1=Zstd
  - File Count (4 bytes): Number of files

For each file:
  - Path Length (4 bytes)
  - Path (variable): UTF-8 relative path
  - Original Size (8 bytes)
  - Compressed Size (8 bytes)
  - Data (variable): File data

Footer:
  - Checksum (4 bytes): CRC32 of all file data
```

---

## Usage Examples

### Distribution Workflow

```python
from numpack import NumPack, pack, unpack

# Create model
with NumPack('model.npk', drop_if_exists=True) as npk:
    npk.save({
        'weights': weights,
        'biases': biases,
        'config': config_array
    })

# Package for distribution
pack('model.npk', 'model_v1.0.npkg')

# --- On another machine ---

# Extract and use
unpack('model_v1.0.npkg', 'model.npk')
with NumPack('model.npk') as npk:
    weights = npk.load('weights')
```

### Inspect Before Extract

```python
from numpack import get_package_info, unpack

# Check package contents
info = get_package_info('unknown.npkg')
print(f"Contains {info['file_count']} files")
print(f"Original size: {info['total_original_size'] / 1e6:.1f} MB")
print(f"Compressed: {info['total_compressed_size'] / 1e6:.1f} MB")

# Decide whether to extract
if info['total_original_size'] < 1e9:  # < 1GB
    unpack('unknown.npkg', 'extracted.npk')
```

### Backup and Archive

```python
from numpack import pack
from datetime import datetime

# Create timestamped backup
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
pack('production.npk', f'backups/production_{timestamp}.npkg')
```
