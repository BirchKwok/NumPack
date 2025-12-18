from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional, Tuple, Union

import numpy as np

# Large file threshold: 1GB
LARGE_FILE_THRESHOLD = 1 * 1024 * 1024 * 1024  # 1GB in bytes

# Default chunk size: 100MB (row count is computed based on dtype)
DEFAULT_CHUNK_SIZE = 100 * 1024 * 1024  # 100MB in bytes

# Default batch rows (used for streaming)
DEFAULT_BATCH_ROWS = 100000


# =============================================================================
# Dependency checking utilities
# =============================================================================

class DependencyError(ImportError):
    """Raised when an optional dependency is required but not installed."""

    pass


def _check_dependency(module_name: str, package_name: Optional[str] = None) -> Any:
    """Import an optional dependency (lazy import).

    Parameters
    ----------
    module_name : str
        Module name to import.
    package_name : str, optional
        Package name to install with pip (if different from `module_name`).

    Returns
    -------
    module
        Imported module.

    Raises
    ------
    DependencyError
        If the dependency is not installed.
    """
    import importlib

    if package_name is None:
        package_name = module_name

    try:
        return importlib.import_module(module_name)
    except ImportError:
        raise DependencyError(
            f"The optional dependency '{package_name}' is required to use this feature.\n"
            f"Please run: pip install {package_name}"
        )


def _check_h5py():
    """Import `h5py`."""
    return _check_dependency('h5py')


def _check_zarr():
    """Import `zarr`."""
    return _check_dependency('zarr')


def _check_pyarrow():
    """Import `pyarrow`."""
    return _check_dependency('pyarrow')


def _check_pandas():
    """Import `pandas`."""
    return _check_dependency('pandas')


def _check_torch():
    """Import `torch` (PyTorch)."""
    return _check_dependency('torch', 'torch')


def _check_s3fs():
    """Import `s3fs`."""
    return _check_dependency('s3fs')


def _check_boto3():
    """Import `boto3`."""
    return _check_dependency('boto3')


# =============================================================================
# File size and streaming utilities
# =============================================================================

def get_file_size(path: Union[str, Path]) -> int:
    """Return the file size in bytes.

    Parameters
    ----------
    path : str or Path
        File path. If a directory is provided, the total size of all files under
        the directory is returned.

    Returns
    -------
    int
        File size in bytes.
    """
    path = Path(path)
    if path.is_file():
        return path.stat().st_size
    elif path.is_dir():
        total = 0
        for f in path.rglob('*'):
            if f.is_file():
                total += f.stat().st_size
        return total
    return 0


def is_large_file(path: Union[str, Path], threshold: int = LARGE_FILE_THRESHOLD) -> bool:
    """Return True if the file/directory size exceeds the threshold.

    Parameters
    ----------
    path : str or Path
        File path.
    threshold : int, optional
        Threshold in bytes.

    Returns
    -------
    bool
        True if size is greater than `threshold`.
    """
    return get_file_size(path) > threshold


def estimate_chunk_rows(
    shape: Tuple[int, ...],
    dtype: np.dtype,
    target_chunk_bytes: int = DEFAULT_CHUNK_SIZE,
) -> int:
    """Estimate the number of rows per chunk for streaming I/O.

    Parameters
    ----------
    shape : tuple
        Array shape.
    dtype : numpy.dtype
        Array dtype.
    target_chunk_bytes : int, optional
        Target chunk size in bytes.

    Returns
    -------
    int
        Suggested number of rows per chunk.
    """
    if len(shape) == 0:
        return 1

    # Calculate the number of bytes per row
    row_elements = int(np.prod(shape[1:])) if len(shape) > 1 else 1
    bytes_per_row = row_elements * dtype.itemsize

    if bytes_per_row == 0:
        return DEFAULT_BATCH_ROWS

    # Compute target row count
    target_rows = max(1, target_chunk_bytes // bytes_per_row)

    # Clamp to a reasonable range
    return min(target_rows, shape[0], DEFAULT_BATCH_ROWS * 10)


# =============================================================================
# NumPack helper functions
# =============================================================================

def _get_numpack_class():
    """Get the NumPack class."""
    from numpack import NumPack

    return NumPack


def _open_numpack_for_write(
    output_path: Union[str, Path],
    drop_if_exists: bool = False,
) -> Any:
    """Open a NumPack file for writing.

    Parameters
    ----------
    output_path : str or Path
        Output path.
    drop_if_exists : bool, optional
        If True, delete the existing output directory first.

    Returns
    -------
    NumPack
        NumPack instance.
    """
    NumPack = _get_numpack_class()
    npk = NumPack(str(output_path), drop_if_exists=drop_if_exists)
    npk.open()
    return npk


def _open_numpack_for_read(input_path: Union[str, Path]) -> Any:
    """Open a NumPack file for reading.

    Parameters
    ----------
    input_path : str or Path
        Input path.

    Returns
    -------
    NumPack
        NumPack instance.
    """
    NumPack = _get_numpack_class()
    npk = NumPack(str(input_path))
    npk.open()
    return npk


def _save_array_streaming(
    npk: Any,
    array_name: str,
    arr: np.ndarray,
    chunk_size: int,
) -> None:
    """Stream-save a large array to NumPack."""
    shape = arr.shape
    dtype = arr.dtype
    batch_rows = estimate_chunk_rows(shape, dtype, chunk_size)
    total_rows = shape[0]

    for start_idx in range(0, total_rows, batch_rows):
        end_idx = min(start_idx + batch_rows, total_rows)
        chunk = np.ascontiguousarray(arr[start_idx:end_idx])

        if start_idx == 0:
            npk.save({array_name: chunk})
        else:
            npk.append({array_name: chunk})


def _save_array_with_streaming_check(
    npk: Any,
    array_name: str,
    arr: np.ndarray,
    chunk_size: int,
) -> None:
    """Check array size and decide whether to use streaming writes."""
    if arr.nbytes > LARGE_FILE_THRESHOLD and arr.ndim > 0:
        _save_array_streaming(npk, array_name, arr, chunk_size)
    else:
        npk.save({array_name: arr})


# =============================================================================
# Convenience helpers
# =============================================================================

def _infer_format(path: Path) -> str:
    """Infer format from a file path."""
    suffix = path.suffix.lower()

    format_map = {
        '.npy': 'numpy',
        '.npz': 'numpy',
        '.csv': 'csv',
        '.txt': 'txt',
        '.tsv': 'csv',
        '.h5': 'hdf5',
        '.hdf5': 'hdf5',
        '.hdf': 'hdf5',
        '.zarr': 'zarr',
        '.parquet': 'parquet',
        '.pq': 'parquet',
        '.feather': 'feather',
        '.fea': 'feather',
        '.pt': 'pytorch',
        '.pth': 'pytorch',
        '.npk': 'numpack',
    }

    # Check whether it is a directory (could be NumPack or Zarr)
    if path.is_dir():
        if (path / 'metadata.npkm').exists():
            return 'numpack'
        if (path / '.zarray').exists() or (path / '.zgroup').exists():
            return 'zarr'

    return format_map.get(suffix, 'unknown')


def _safe_unlink(path: Union[str, Path]) -> None:
    tmp_path = str(path)
    if os.path.exists(tmp_path):
        os.unlink(tmp_path)
