# NumPack File Format Specification v3.0

## Overview

NumPack is a high-performance array storage library with Rust backend that provides cross-platform compatibility. This specification defines the unified high-performance binary file format protocol used by NumPack for persistent storage of multidimensional arrays.

## Key Features

- **Cross-Platform Compatibility**: Files are compatible across all platforms using Rust backend
- **High-Performance Binary Format**: Custom binary format optimized for speed and efficiency
- **Zero-Copy Operations**: Memory-mapped file access for optimal performance
- **Data Type Safety**: Strict type mapping between Python NumPy and Rust data types
- **Compression Support**: Built-in support for zstd compression

## Directory Structure

NumPack stores data as a directory containing multiple files:

```
<numpack_directory>/
├── metadata.npkm               # Metadata file (Binary format)
├── data_<array_name>.npkd      # Raw binary data files (one per array)
└── deleted_<array_name>.npkb   # Deletion bitmap files (one per array, optional)
```

## Binary Format (High-Performance Standard)

NumPack uses a custom high-performance binary format for all metadata storage. This ensures maximum speed and cross-platform compatibility with Rust backend.

### Metadata File Structure (`metadata.npkm`)

The metadata file uses a compact binary serialization with the following structure:

```
Header:
- Magic Number (4 bytes): 0x424B504E (ASCII: "NPKB")
- Version (4 bytes): Format version (currently 1)
- Total Size (8 bytes): Total size of all arrays in bytes
- Array Count (4 bytes): Number of arrays in the file

Array Metadata (repeated for each array):
- Name Length (4 bytes): Length of array name
- Name (variable): Array name in UTF-8
- Shape Length (4 bytes): Number of dimensions
- Shape (8 bytes × dimensions): Array dimensions as uint64
- Data File Length (4 bytes): Length of data file name
- Data File (variable): Data file name in UTF-8
- Last Modified (8 bytes): Timestamp in microseconds since Unix epoch
- Size Bytes (8 bytes): Array size in bytes
- Data Type (1 byte): Data type code (see Data Type Mapping)
- Compression Algorithm (1 byte): Compression algorithm code
- Compression Level (4 bytes): Compression level
- Original Size (8 bytes): Original uncompressed size
- Compressed Size (8 bytes): Compressed size
- Block Compression Flag (1 byte): Whether block compression info follows
- [Optional] Block Compression Info: Block-level compression metadata
```

### Field Specifications

| Field | Type | Description |
|-------|------|-------------|
| Magic Number | uint32 | Format identifier (0x424B504E) |
| Version | uint32 | Format version (currently 1) |
| Total Size | uint64 | Total size of all arrays in bytes |
| Array Count | uint32 | Number of arrays stored |
| Name | string | Array identifier |
| Shape | array[uint64] | Array dimensions |
| Data File | string | Relative path to data file (e.g., "data_array1.npkd") |
| Last Modified | uint64 | Timestamp in microseconds since Unix epoch |
| Size Bytes | uint64 | Array size in bytes |
| Data Type | uint8 | Data type code (see Data Type Mapping) |

### Data Files (`data_<array_name>.npkd`)

Raw binary data stored in little-endian format, containing array elements in C-contiguous (row-major) order.

### Deletion Bitmap Files (`deleted_<array_name>.npkb`)

Bitmap files used for logical deletion tracking. These files are optional and only created when rows are deleted using the `drop` operation.

**File Format:**
- **Bitmap Format**: Compact bit array where each bit represents one row
  - Bit value 0: Row is deleted
  - Bit value 1: Row is active (not deleted)
- **Storage**: Packed 64-bit words (u64) in little-endian format
- **Size Calculation**: `ceil(num_rows / 64) * 8` bytes
- **Initial State**: When created, all bits are set to 1 (all rows active)

**Operations:**
- **Logical Deletion**: `drop` operation sets corresponding bits to 0
- **Index Mapping**: All read/write operations automatically map logical indices (user view) to physical indices (actual storage). Lazy loading (`lazy=True`) must expose only active rows in its reported shape and iteration semantics by accounting for the bitmap. Dense loading and metadata queries follow the same logical view.
- **Physical Compaction**: `update` operation physically removes deleted rows and resets the bitmap

## Data Type Mapping

NumPack uses a standardized data type encoding that maps between Python NumPy and Rust types:

| Code | NumPy Type | Rust Type | Size (bytes) | Description |
|------|------------|-----------|--------------|-------------|
| 0    | np.bool_   | bool      | 1            | Boolean |
| 1    | np.uint8   | u8        | 1            | Unsigned 8-bit integer |
| 2    | np.uint16  | u16       | 2            | Unsigned 16-bit integer |
| 3    | np.uint32  | u32       | 4            | Unsigned 32-bit integer |
| 4    | np.uint64  | u64       | 8            | Unsigned 64-bit integer |
| 5    | np.int8    | i8        | 1            | Signed 8-bit integer |
| 6    | np.int16   | i16       | 2            | Signed 16-bit integer |
| 7    | np.int32   | i32       | 4            | Signed 32-bit integer |
| 8    | np.int64   | i64       | 8            | Signed 64-bit integer |
| 9    | np.float16 | f16       | 2            | Half-precision float |
| 10   | np.float32 | f32       | 4            | Single-precision float |
| 11   | np.float64 | f64       | 8            | Double-precision float |
| 12   | np.complex64 | Complex32 | 8          | Complex (2×f32) |
| 13   | np.complex128 | Complex64 | 16        | Complex (2×f64) |

## Compression Support

NumPack supports optional compression for data arrays:

### Compression Algorithms

| Code | Algorithm | Description |
|------|-----------|-------------|
| 0    | None      | No compression |
| 1    | Zstd      | Zstandard compression |

### Block Compression

For large arrays, NumPack supports block-level compression:

- **Block Size**: Configurable block size (default: 64KB)
- **Independent Blocks**: Each block is compressed independently
- **Random Access**: Allows efficient random access without full decompression

## Byte Order

All multi-byte integers and floating-point numbers are stored in **little-endian** format for cross-platform compatibility.

## Version Control

### Current Version: 1

The version field in metadata indicates format compatibility:

- **Version 1**: High-performance binary format
- **Future versions**: Will maintain backward compatibility where possible

## Cross-Platform Compatibility

### Unified Format Benefits

NumPack uses a custom binary format as the single, unified format across all platforms:

1. **Maximum Performance**: Optimized for speed and minimal overhead
2. **Perfect Compatibility**: 100% cross-platform compatibility with Rust backend
3. **Efficient Storage**: Compact binary representation
4. **Zero Dependencies**: No external serialization libraries required

### Endianness Handling

- **Storage**: Always little-endian
- **Runtime**: Automatic conversion on big-endian systems
- **Arrays**: Converted to native byte order when loaded

## Performance Optimizations

### Memory Mapping

- **Data Files**: Can be memory-mapped for zero-copy access
- **Large Arrays**: Automatic chunked I/O for arrays > 100MB
- **Caching**: LRU cache for recently accessed arrays

### Incremental Updates

- **Metadata**: Only modified arrays trigger metadata updates
- **Data**: Individual array files can be updated independently
- **Atomicity**: Temporary files ensure atomic operations

## Error Handling

### File Corruption Detection

- **Magic Number**: Validates file format during load
- **Graceful Recovery**: Partial data recovery when possible
- **Validation**: Comprehensive metadata and data consistency checks

### Validation

- **Array Shapes**: Validated against data file sizes
- **Data Types**: Strict type checking during load operations
- **File Paths**: Relative path validation for security

## Implementation Notes

### Rust Backend

- **Serialization**: Custom binary serialization for maximum performance
- **Memory Safety**: Compile-time guarantees with explicit lifetimes
- **Performance**: Zero-copy deserialization where possible

## Migration from Legacy Format

### Automatic Conversion

Previous versions of NumPack may have used MessagePack format. The current implementation:

1. **No Backward Compatibility**: Binary format is not compatible with MessagePack
2. **Migration Required**: Users must migrate existing data explicitly
3. **Performance Gain**: Significant performance improvements with new format

## Future Considerations

### Potential Enhancements

- **Additional Compression**: Support for more compression algorithms
- **Checksums**: Data integrity verification
- **Schema Evolution**: Backward-compatible metadata schema updates
- **Indexing**: Built-in indexing for faster array access

### Compatibility Promise

NumPack guarantees:
- **Forward Compatibility**: Newer versions can read older binary formats
- **Cross-Platform**: Files remain compatible across all platforms with Rust backend
- **Migration Path**: Clear upgrade paths for format changes

---

**Document Version**: 3.0  
**Specification Authors**: NumPack Development Team 