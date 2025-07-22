# NumPack File Format Specification v0.2.1

## Overview

NumPack is a high-performance array storage library that provides cross-platform compatibility between Python and Rust implementations. This specification defines the unified MessagePack file format protocol used by NumPack for persistent storage of multidimensional arrays.

## Key Features

- **Cross-Platform Compatibility**: Files created by Python backend can be read by Rust backend and vice versa
- **Unified MessagePack Format**: Single, standardized format for all platforms
- **Zero-Copy Operations**: Memory-mapped file access for optimal performance
- **Data Type Safety**: Strict type mapping between Python NumPy and Rust data types
- **Concurrent Access**: File locking mechanisms for safe multi-process access

## Directory Structure

NumPack stores data as a directory containing multiple files:

```
<numpack_directory>/
├── metadata.npkm           # Metadata file (MessagePack format)
├── data_<array_name>.npkd  # Raw binary data files (one per array)
└── metadata.npkm.lock      # File lock (temporary, for concurrent access)
```

## MessagePack Format (Unified Standard)

NumPack uses MessagePack as the unified format for all metadata storage. This ensures perfect compatibility between Python and Rust implementations.

### Metadata File Structure (`metadata.npkm`)

The metadata file uses MessagePack serialization with the following schema:

```json
{
    "version": <uint32>,
    "total_size": <uint64>, 
    "arrays": {
        "<array_name>": {
            "name": "<string>",
            "shape": [<uint64>, ...],
            "data_file": "<string>",
            "last_modified": <uint64>,
            "size_bytes": <uint64>,
            "dtype": <uint8>
        },
        ...
    }
}
```

### Field Specifications

| Field | Type | Description |
|-------|------|-------------|
| `version` | uint32 | Format version (currently 1) |
| `total_size` | uint64 | Total size of all arrays in bytes |
| `arrays` | object | Dictionary of array metadata |
| `name` | string | Array identifier (matches key) |
| `shape` | array[uint64] | Array dimensions |
| `data_file` | string | Relative path to data file (e.g., "data_array1.npkd") |
| `last_modified` | uint64 | Timestamp in microseconds since Unix epoch |
| `size_bytes` | uint64 | Array size in bytes |
| `dtype` | uint8 | Data type code (see Data Type Mapping) |

### Data Files (`data_<array_name>.npkd`)

Raw binary data stored in little-endian format, containing array elements in C-contiguous (row-major) order.

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

## Byte Order

All multi-byte integers and floating-point numbers are stored in **little-endian** format for cross-platform compatibility.

## File Locking

NumPack implements file locking to ensure safe concurrent access:

- **Lock File**: `<directory>/metadata.npkm.lock`
- **Scope**: Protects metadata read/write operations
- **Type**: Advisory file lock using `filelock` library
- **Timeout**: Operations may timeout if lock cannot be acquired

## Version Control

### Current Version: 1

The version field in metadata indicates format compatibility:

- **Version 1**: Unified MessagePack format
- **Future versions**: Will maintain backward compatibility where possible

## Cross-Platform Compatibility

### Unified Format Benefits

NumPack uses MessagePack as the single, unified format across all platforms:

1. **No Format Conversion**: Eliminates conversion overhead and complexity
2. **Perfect Compatibility**: 100% interoperability between Python and Rust
3. **Simplified Implementation**: Single code path for all platforms
4. **Better Performance**: No conversion delays

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

- **MessagePack**: Built-in validation during deserialization
- **Graceful Recovery**: Partial data recovery when possible
- **Validation**: Comprehensive metadata and data consistency checks

### Validation

- **Array Shapes**: Validated against data file sizes
- **Data Types**: Strict type checking during load operations
- **File Paths**: Relative path validation for security

## Implementation Notes

### Python Backend

- **Format**: Pure MessagePack with `msgpack` library
- **Libraries**: `msgpack`, `numpy`, `filelock`
- **Memory Management**: Automatic garbage collection with weak references

### Rust Backend

- **Serialization**: `rmp-serde` with struct-map configuration
- **Memory Safety**: Compile-time guarantees with explicit lifetimes
- **Performance**: Zero-copy deserialization where possible

## Migration from Legacy Format

### Automatic Conversion

Previous versions of NumPack may have used different formats. The current implementation automatically:

1. **Detects Legacy Files**: Attempts to read old formats
2. **Converts Transparently**: Migrates to MessagePack format
3. **Preserves Data**: Ensures no data loss during conversion
4. **Updates Format**: All new writes use MessagePack format

## Future Considerations

### Potential Enhancements

- **Compression**: Optional compression for data files
- **Checksums**: Data integrity verification
- **Schema Evolution**: Backward-compatible metadata schema updates
- **Indexing**: Built-in indexing for faster array access

### Compatibility Promise

NumPack guarantees:
- **Forward Compatibility**: Newer versions can read older formats
- **Cross-Language**: Files remain compatible between Python and Rust implementations
- **Migration Path**: Clear upgrade paths for format changes

---

**Document Version**: 2.0  
**Last Updated**: January 2025  
**Specification Authors**: NumPack Development Team 