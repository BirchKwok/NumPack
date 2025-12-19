"""
NumPack Package I/O Module

Provides pack/unpack functionality for NumPack directories.
Compresses all NumPack files (metadata, data, deletion bitmaps) into a single .npkg file.

File Format (.npkg v2):
-----------------------
Header:
  - Magic Number (4 bytes): "NPKG"
  - Version (4 bytes): Format version (2 for streaming)
  - Compression (1 byte): 0=None, 1=Zstd
  - File Count (4 bytes): Number of files in the package

For each file:
  - Path Length (4 bytes): Length of relative path
  - Path (variable): Relative path in UTF-8
  - Original Size (8 bytes): Original file size
  - Compressed Size (8 bytes): Compressed size (0 if not compressed)
  - Data (variable): File data (compressed if enabled)

Footer:
  - Checksum (4 bytes): CRC32 of all file data

Streaming Design:
- Files are processed in chunks (default 8MB) to minimize memory usage
- Compression is done per-file with streaming compressor
- Suitable for GB-scale NumPack directories
"""

import io
import os
import struct
import zlib
from pathlib import Path
from typing import Union, Optional, BinaryIO, List, Tuple

# Magic number for .npkg format
NPKG_MAGIC = b'NPKG'
NPKG_VERSION = 2  # Version 2 for streaming support

# Compression types
COMPRESSION_NONE = 0
COMPRESSION_ZSTD = 1

# Streaming chunk size (8MB - balances memory usage and compression efficiency)
CHUNK_SIZE = 8 * 1024 * 1024

# Try to import zstd for better compression
try:
    import zstandard as zstd
    _HAS_ZSTD = True
except ImportError:
    _HAS_ZSTD = False


def _compress_file_streaming(src_file: BinaryIO, dst_file: BinaryIO, 
                              compression: int, chunk_size: int = CHUNK_SIZE) -> Tuple[int, int, int]:
    """Compress a file using streaming to minimize memory usage.
    
    Returns:
        Tuple of (original_size, compressed_size, crc32_checksum)
        compressed_size is 0 if compression was skipped (data not compressible)
    """
    original_size = 0
    checksum = 0
    
    if compression == COMPRESSION_NONE:
        # No compression - stream copy
        while True:
            chunk = src_file.read(chunk_size)
            if not chunk:
                break
            original_size += len(chunk)
            checksum = zlib.crc32(chunk, checksum)
            dst_file.write(chunk)
        return original_size, 0, checksum
    
    # Compress to temporary buffer first to check if compression is beneficial
    compressed_buffer = io.BytesIO()
    
    if _HAS_ZSTD:
        cctx = zstd.ZstdCompressor(level=3)
        compressor = cctx.stream_writer(compressed_buffer, closefd=False)
        while True:
            chunk = src_file.read(chunk_size)
            if not chunk:
                break
            original_size += len(chunk)
            checksum = zlib.crc32(chunk, checksum)
            compressor.write(chunk)
        compressor.close()  # Flush but don't close the buffer
    else:
        # Fallback to zlib streaming
        compressor = zlib.compressobj(level=6)
        while True:
            chunk = src_file.read(chunk_size)
            if not chunk:
                break
            original_size += len(chunk)
            checksum = zlib.crc32(chunk, checksum)
            compressed_buffer.write(compressor.compress(chunk))
        compressed_buffer.write(compressor.flush())
    
    compressed_size = compressed_buffer.tell()
    
    # Check if compression was beneficial
    if compressed_size >= original_size:
        # Compression not beneficial - need to re-read and write uncompressed
        src_file.seek(0)
        while True:
            chunk = src_file.read(chunk_size)
            if not chunk:
                break
            dst_file.write(chunk)
        return original_size, 0, checksum
    
    # Write compressed data
    compressed_buffer.seek(0)
    while True:
        chunk = compressed_buffer.read(chunk_size)
        if not chunk:
            break
        dst_file.write(chunk)
    
    return original_size, compressed_size, checksum


class _LimitedReader:
    """A file-like wrapper that limits reading to a specified number of bytes."""
    def __init__(self, fp: BinaryIO, limit: int):
        self.fp = fp
        self.remaining = limit
    
    def read(self, size: int = -1) -> bytes:
        if self.remaining <= 0:
            return b''
        if size < 0:
            size = self.remaining
        to_read = min(size, self.remaining)
        data = self.fp.read(to_read)
        self.remaining -= len(data)
        return data


def _decompress_streaming(src_file: BinaryIO, dst_file: BinaryIO,
                          original_size: int, compressed_size: int,
                          compression: int, chunk_size: int = CHUNK_SIZE) -> int:
    """Decompress data using streaming to minimize memory usage.
    
    Returns:
        CRC32 checksum of decompressed data
    """
    checksum = 0
    
    if compressed_size == 0:
        # Data was not compressed - stream copy
        remaining = original_size
        while remaining > 0:
            to_read = min(chunk_size, remaining)
            chunk = src_file.read(to_read)
            if not chunk:
                break
            checksum = zlib.crc32(chunk, checksum)
            dst_file.write(chunk)
            remaining -= len(chunk)
        return checksum
    
    if compression == COMPRESSION_ZSTD and _HAS_ZSTD:
        dctx = zstd.ZstdDecompressor()
        # Use limited reader to only read compressed_size bytes
        limited = _LimitedReader(src_file, compressed_size)
        reader = dctx.stream_reader(limited, read_size=chunk_size, closefd=False)
        decompressed_size = 0
        while decompressed_size < original_size:
            chunk = reader.read(chunk_size)
            if not chunk:
                break
            checksum = zlib.crc32(chunk, checksum)
            dst_file.write(chunk)
            decompressed_size += len(chunk)
        # Don't close reader - it would close the underlying file
    else:
        # Fallback to zlib - read compressed data in chunks
        compressed_data = src_file.read(compressed_size)
        decompressor = zlib.decompressobj()
        offset = 0
        while offset < len(compressed_data):
            # Feed chunks to decompressor
            end = min(offset + chunk_size, len(compressed_data))
            decompressed = decompressor.decompress(compressed_data[offset:end])
            if decompressed:
                checksum = zlib.crc32(decompressed, checksum)
                dst_file.write(decompressed)
            offset = end
        # Flush remaining
        remaining = decompressor.flush()
        if remaining:
            checksum = zlib.crc32(remaining, checksum)
            dst_file.write(remaining)
    
    return checksum


def _get_numpack_files(directory: Path) -> List[Path]:
    """Get all files in a NumPack directory that should be packaged.
    
    Returns files matching NumPack patterns:
    - metadata.npkm
    - data_*.npkd
    - deleted_*.npkb
    """
    files = []
    
    if not directory.is_dir():
        raise ValueError(f"'{directory}' is not a directory")
    
    for item in directory.iterdir():
        if item.is_file():
            name = item.name
            # Include all NumPack files
            if (name == 'metadata.npkm' or 
                name.startswith('data_') and name.endswith('.npkd') or
                name.startswith('deleted_') and name.endswith('.npkb')):
                files.append(item)
    
    return sorted(files, key=lambda x: x.name)


def pack(
    source: Union[str, Path],
    target: Optional[Union[str, Path]] = None,
    compression: bool = True,
    overwrite: bool = False
) -> Path:
    """Pack a NumPack directory into a single .npkg file.
    
    This function compresses all NumPack files (metadata, data, deletion bitmaps)
    into a single portable .npkg file for easy migration and sharing.
    
    Parameters
    ----------
    source : str or Path
        Path to the NumPack directory to pack.
    target : str or Path, optional
        Path for the output .npkg file. If None, uses source directory name
        with .npkg extension in the same parent directory.
    compression : bool, default True
        Whether to compress file data. Uses Zstd if available, else zlib.
    overwrite : bool, default False
        Whether to overwrite existing .npkg file.
        
    Returns
    -------
    Path
        Path to the created .npkg file.
        
    Raises
    ------
    ValueError
        If source is not a valid NumPack directory.
    FileExistsError
        If target exists and overwrite is False.
        
    Examples
    --------
    >>> from numpack import pack
    >>> # Pack a NumPack directory
    >>> pack('/path/to/my_data.npk')
    PosixPath('/path/to/my_data.npkg')
    
    >>> # Specify custom output path
    >>> pack('/path/to/my_data.npk', '/backup/my_data.npkg')
    PosixPath('/backup/my_data.npkg')
    """
    source = Path(source)
    
    if not source.exists():
        raise ValueError(f"Source path does not exist: {source}")
    
    if not source.is_dir():
        raise ValueError(f"Source must be a NumPack directory: {source}")
    
    # Check for metadata file to validate it's a NumPack directory
    metadata_file = source / 'metadata.npkm'
    if not metadata_file.exists():
        raise ValueError(
            f"'{source}' does not appear to be a valid NumPack directory. "
            "Missing 'metadata.npkm' file."
        )
    
    # Determine target path
    if target is None:
        target = source.parent / f"{source.name}.npkg"
    else:
        target = Path(target)
        if not target.suffix:
            target = target.with_suffix('.npkg')
    
    # Check if target exists
    if target.exists() and not overwrite:
        raise FileExistsError(
            f"Target file already exists: {target}. "
            "Use overwrite=True to replace it."
        )
    
    # Get all NumPack files
    files = _get_numpack_files(source)
    
    if not files:
        raise ValueError(f"No NumPack files found in '{source}'")
    
    # Determine compression type
    comp_type = COMPRESSION_ZSTD if compression else COMPRESSION_NONE
    
    # Calculate checksum as we go
    checksum = 0
    
    # Create parent directories if needed
    target.parent.mkdir(parents=True, exist_ok=True)
    
    # Use temporary file for atomic write
    temp_target = target.with_suffix('.npkg.tmp')
    
    try:
        with open(temp_target, 'wb') as f:
            # Write header
            f.write(NPKG_MAGIC)  # Magic number
            f.write(struct.pack('<I', NPKG_VERSION))  # Version
            f.write(struct.pack('<B', comp_type))  # Compression type
            f.write(struct.pack('<I', len(files)))  # File count
            
            # Write each file using streaming compression
            for file_path in files:
                # Get relative path
                rel_path = file_path.name
                rel_path_bytes = rel_path.encode('utf-8')
                
                # Write file header (path info)
                f.write(struct.pack('<I', len(rel_path_bytes)))  # Path length
                f.write(rel_path_bytes)  # Path
                
                # Reserve space for sizes (will be updated after compression)
                size_pos = f.tell()
                f.write(struct.pack('<Q', 0))  # Placeholder for original size
                f.write(struct.pack('<Q', 0))  # Placeholder for compressed size
                
                # Stream compress file data
                data_start = f.tell()
                with open(file_path, 'rb') as src:
                    original_size, compressed_size, file_checksum = _compress_file_streaming(
                        src, f, comp_type
                    )
                
                # Update global checksum
                checksum = zlib.crc32(file_checksum.to_bytes(4, 'little'), checksum)
                
                # Go back and write actual sizes
                data_end = f.tell()
                f.seek(size_pos)
                f.write(struct.pack('<Q', original_size))
                f.write(struct.pack('<Q', compressed_size))
                f.seek(data_end)
            
            # Write footer checksum
            f.write(struct.pack('<I', checksum & 0xFFFFFFFF))
        
        # Atomic rename
        if target.exists():
            target.unlink()
        temp_target.rename(target)
        
    except Exception:
        # Clean up temp file on error
        if temp_target.exists():
            temp_target.unlink()
        raise
    
    return target


def unpack(
    source: Union[str, Path],
    target: Optional[Union[str, Path]] = None,
    overwrite: bool = False,
    verify: bool = True
) -> Path:
    """Unpack a .npkg file into a NumPack directory.
    
    This function extracts all files from a .npkg package and recreates
    the original NumPack directory structure.
    
    Parameters
    ----------
    source : str or Path
        Path to the .npkg file to unpack.
    target : str or Path, optional
        Path for the output NumPack directory. If None, uses source filename
        without .npkg extension in the same directory.
    overwrite : bool, default False
        Whether to overwrite existing directory.
    verify : bool, default True
        Whether to verify checksum after unpacking.
        
    Returns
    -------
    Path
        Path to the extracted NumPack directory.
        
    Raises
    ------
    ValueError
        If source is not a valid .npkg file.
    FileExistsError
        If target directory exists and overwrite is False.
    RuntimeError
        If checksum verification fails.
        
    Examples
    --------
    >>> from numpack import unpack
    >>> # Unpack to default location
    >>> unpack('/path/to/my_data.npkg')
    PosixPath('/path/to/my_data.npk')
    
    >>> # Specify custom output directory
    >>> unpack('/path/to/my_data.npkg', '/data/restored')
    PosixPath('/data/restored')
    """
    source = Path(source)
    
    if not source.exists():
        raise ValueError(f"Source file does not exist: {source}")
    
    if not source.is_file():
        raise ValueError(f"Source must be a file: {source}")
    
    # Determine target path
    if target is None:
        # Remove .npkg extension and add .npk if original was .npk
        stem = source.stem
        if stem.endswith('.npk'):
            target = source.parent / stem
        else:
            target = source.parent / f"{stem}.npk"
    else:
        target = Path(target)
    
    # Check if target exists
    if target.exists():
        if not overwrite:
            raise FileExistsError(
                f"Target directory already exists: {target}. "
                "Use overwrite=True to replace it."
            )
        else:
            # Remove existing directory
            import shutil
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
    
    # Read and validate package
    with open(source, 'rb') as f:
        # Read header
        magic = f.read(4)
        if magic != NPKG_MAGIC:
            raise ValueError(
                f"Invalid .npkg file: {source}. "
                f"Expected magic 'NPKG', got {magic!r}"
            )
        
        version = struct.unpack('<I', f.read(4))[0]
        if version > NPKG_VERSION:
            raise ValueError(
                f"Unsupported .npkg version: {version}. "
                f"Maximum supported version is {NPKG_VERSION}."
            )
        
        comp_type = struct.unpack('<B', f.read(1))[0]
        file_count = struct.unpack('<I', f.read(4))[0]
        
        # Create target directory
        target.mkdir(parents=True, exist_ok=True)
        
        # Track checksum for verification
        checksum = 0
        
        # Extract each file using streaming decompression
        for _ in range(file_count):
            # Read file entry header
            path_len = struct.unpack('<I', f.read(4))[0]
            rel_path = f.read(path_len).decode('utf-8')
            original_size = struct.unpack('<Q', f.read(8))[0]
            compressed_size = struct.unpack('<Q', f.read(8))[0]
            
            # Validate relative path (security check)
            if '..' in rel_path or rel_path.startswith('/'):
                raise ValueError(f"Invalid file path in package: {rel_path}")
            
            # Stream decompress directly to file
            file_path = target / rel_path
            
            # For v2 format, use streaming; for v1, use limited streaming
            if version >= 2:
                # Stream decompress to file
                with open(file_path, 'wb') as dst:
                    file_checksum = _decompress_streaming(
                        f, dst, original_size, compressed_size, comp_type
                    )
                # Update global checksum
                checksum = zlib.crc32(file_checksum.to_bytes(4, 'little'), checksum)
            else:
                # Legacy v1 format - still need to read all data but write streaming
                data_size = compressed_size if compressed_size > 0 else original_size
                
                # Use limited memory reader for large files
                with open(file_path, 'wb') as dst:
                    if compressed_size == 0:
                        # Uncompressed - stream copy
                        remaining = original_size
                        while remaining > 0:
                            chunk = f.read(min(CHUNK_SIZE, remaining))
                            if not chunk:
                                break
                            checksum = zlib.crc32(chunk, checksum)
                            dst.write(chunk)
                            remaining -= len(chunk)
                    else:
                        # Compressed - need to decompress
                        # For zstd, we can stream; for zlib, limited streaming
                        if comp_type == COMPRESSION_ZSTD and _HAS_ZSTD:
                            dctx = zstd.ZstdDecompressor()
                            # Create a limited reader that only reads compressed_size bytes
                            class LimitedReader:
                                def __init__(self, fp, limit):
                                    self.fp = fp
                                    self.remaining = limit
                                def read(self, size=-1):
                                    if self.remaining <= 0:
                                        return b''
                                    if size < 0:
                                        size = self.remaining
                                    to_read = min(size, self.remaining)
                                    data = self.fp.read(to_read)
                                    self.remaining -= len(data)
                                    return data
                            
                            limited = LimitedReader(f, compressed_size)
                            reader = dctx.stream_reader(limited, read_size=CHUNK_SIZE)
                            while True:
                                chunk = reader.read(CHUNK_SIZE)
                                if not chunk:
                                    break
                                checksum = zlib.crc32(chunk, checksum)
                                dst.write(chunk)
                            reader.close()
                        else:
                            # zlib fallback - read all compressed data
                            compressed_data = f.read(compressed_size)
                            decompressor = zlib.decompressobj()
                            decompressed = decompressor.decompress(compressed_data)
                            checksum = zlib.crc32(decompressed, checksum)
                            dst.write(decompressed)
                            remaining = decompressor.flush()
                            if remaining:
                                checksum = zlib.crc32(remaining, checksum)
                                dst.write(remaining)
        
        # Verify checksum (note: v1 and v2 have different checksum schemes)
        if verify:
            stored_checksum = struct.unpack('<I', f.read(4))[0]
            if (checksum & 0xFFFFFFFF) != stored_checksum:
                # Clean up on failure
                import shutil
                shutil.rmtree(target)
                raise RuntimeError(
                    f"Checksum verification failed for '{source}'. "
                    "The package may be corrupted."
                )
    
    return target


def get_package_info(source: Union[str, Path]) -> dict:
    """Get information about a .npkg package without extracting it.
    
    Parameters
    ----------
    source : str or Path
        Path to the .npkg file.
        
    Returns
    -------
    dict
        Dictionary containing:
        - version: Package format version
        - compression: Compression type (0=None, 1=Zstd)
        - file_count: Number of files in package
        - files: List of file info dicts with name, original_size, compressed_size
        - total_original_size: Total uncompressed size
        - total_compressed_size: Total compressed size
        - compression_ratio: Compression ratio (compressed/original)
        
    Examples
    --------
    >>> from numpack import get_package_info
    >>> info = get_package_info('/path/to/my_data.npkg')
    >>> print(f"Files: {info['file_count']}, Ratio: {info['compression_ratio']:.2%}")
    """
    source = Path(source)
    
    if not source.exists():
        raise ValueError(f"Source file does not exist: {source}")
    
    with open(source, 'rb') as f:
        # Read header
        magic = f.read(4)
        if magic != NPKG_MAGIC:
            raise ValueError(f"Invalid .npkg file: {source}")
        
        version = struct.unpack('<I', f.read(4))[0]
        comp_type = struct.unpack('<B', f.read(1))[0]
        file_count = struct.unpack('<I', f.read(4))[0]
        
        files = []
        total_original = 0
        total_compressed = 0
        
        for _ in range(file_count):
            path_len = struct.unpack('<I', f.read(4))[0]
            rel_path = f.read(path_len).decode('utf-8')
            original_size = struct.unpack('<Q', f.read(8))[0]
            compressed_size = struct.unpack('<Q', f.read(8))[0]
            
            # Skip data
            data_size = compressed_size if compressed_size > 0 else original_size
            f.seek(data_size, 1)
            
            actual_size = compressed_size if compressed_size > 0 else original_size
            files.append({
                'name': rel_path,
                'original_size': original_size,
                'compressed_size': compressed_size,
                'actual_size': actual_size
            })
            
            total_original += original_size
            total_compressed += actual_size
        
        compression_ratio = total_compressed / total_original if total_original > 0 else 1.0
        
        return {
            'version': version,
            'compression': comp_type,
            'compression_name': 'zstd' if comp_type == COMPRESSION_ZSTD else 'none',
            'file_count': file_count,
            'files': files,
            'total_original_size': total_original,
            'total_compressed_size': total_compressed,
            'compression_ratio': compression_ratio
        }


__all__ = ['pack', 'unpack', 'get_package_info']
