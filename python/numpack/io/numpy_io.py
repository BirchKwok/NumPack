from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import numpy as np

from .utils import (
    DEFAULT_CHUNK_SIZE,
    LARGE_FILE_THRESHOLD,
    _open_numpack_for_read,
    _open_numpack_for_write,
    estimate_chunk_rows,
    get_file_size,
    _save_array_streaming,
)


# =============================================================================
# NumPy format conversion (npy/npz)
# =============================================================================

def from_numpy(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    array_name: Optional[str] = None,
    drop_if_exists: bool = False,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> None:
    """Import a NumPy ``.npy``/``.npz`` file into NumPack.

    For large files (by default > 1 GB), this function uses NumPy memory mapping
    and streams data into NumPack in chunks.

    Parameters
    ----------
    input_path : str or Path
        Path to the input ``.npy`` or ``.npz`` file.
    output_path : str or Path
        Output NumPack directory path.
    array_name : str, optional
        Array name used for ``.npy`` input. If None, defaults to the file stem.
        For ``.npz`` input, this parameter is ignored and the keys inside the
        archive are used as array names.
    drop_if_exists : bool, optional
        If True, delete the output path first if it already exists.
    chunk_size : int, optional
        Chunk size in bytes used for streaming write.

    Raises
    ------
    FileNotFoundError
        If `input_path` does not exist.
    ValueError
        If `input_path` suffix is not one of ``.npy`` or ``.npz``.

    Examples
    --------
    >>> from numpack.io import from_numpy
    >>> from_numpy('data.npy', 'output.npk')
    >>> from_numpy('data.npz', 'output.npk')
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")

    suffix = input_path.suffix.lower()

    if suffix == '.npy':
        _from_npy(input_path, output_path, array_name, drop_if_exists, chunk_size)
    elif suffix == '.npz':
        _from_npz(input_path, output_path, drop_if_exists, chunk_size)
    else:
        raise ValueError(f"Unsupported file format: {suffix}. Supported: .npy and .npz")


def _from_npy(
    input_path: Path,
    output_path: Union[str, Path],
    array_name: Optional[str],
    drop_if_exists: bool,
    chunk_size: int,
) -> None:
    """Import from a single ``.npy`` file."""
    if array_name is None:
        array_name = input_path.stem

    file_size = get_file_size(input_path)

    if file_size > LARGE_FILE_THRESHOLD:
        # Large file: memory-map and stream writes
        _from_npy_streaming(input_path, output_path, array_name, drop_if_exists, chunk_size)
    else:
        # Small file: load directly
        arr = np.load(str(input_path))
        npk = _open_numpack_for_write(output_path, drop_if_exists)
        try:
            npk.save({array_name: arr})
        finally:
            npk.close()


def _from_npy_streaming(
    input_path: Path,
    output_path: Union[str, Path],
    array_name: str,
    drop_if_exists: bool,
    chunk_size: int,
) -> None:
    """Stream-import a large ``.npy`` file."""
    # Load with memory mapping
    arr_mmap = np.load(str(input_path), mmap_mode='r')
    shape = arr_mmap.shape
    dtype = arr_mmap.dtype

    # Compute chunk row count
    batch_rows = estimate_chunk_rows(shape, dtype, chunk_size)
    total_rows = shape[0]

    npk = _open_numpack_for_write(output_path, drop_if_exists)
    try:
        # Write in chunks
        for start_idx in range(0, total_rows, batch_rows):
            end_idx = min(start_idx + batch_rows, total_rows)
            chunk = np.ascontiguousarray(arr_mmap[start_idx:end_idx])
            
            if start_idx == 0:
                npk.save({array_name: chunk})
            else:
                npk.append({array_name: chunk})
    finally:
        npk.close()
        del arr_mmap


def _from_npz(
    input_path: Path,
    output_path: Union[str, Path],
    drop_if_exists: bool,
    chunk_size: int,
) -> None:
    """Import from a ``.npz`` file."""
    # Check file size
    file_size = get_file_size(input_path)

    npk = _open_numpack_for_write(output_path, drop_if_exists)
    try:
        if file_size > LARGE_FILE_THRESHOLD:
            # Large file: load arrays one by one
            with np.load(str(input_path), mmap_mode='r') as npz:
                for name in npz.files:
                    arr = npz[name]
                    # For large arrays, stream writes
                    if arr.nbytes > LARGE_FILE_THRESHOLD:
                        _save_array_streaming(npk, name, arr, chunk_size)
                    else:
                        npk.save({name: np.array(arr)})
        else:
            # Small file: load directly
            with np.load(str(input_path)) as npz:
                arrays = {name: npz[name] for name in npz.files}
                npk.save(arrays)
    finally:
        npk.close()


def to_numpy(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    array_names: Optional[List[str]] = None,
    compressed: bool = True,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> None:
    """Export NumPack arrays to NumPy ``.npy``/``.npz``.

    For large arrays (by default > 1 GB), this function streams reads from NumPack
    and writes the output in chunks.

    Parameters
    ----------
    input_path : str or Path
        Input NumPack directory path.
    output_path : str or Path
        Output ``.npy`` or ``.npz`` file path.
    array_names : list of str, optional
        Names of arrays to export. If None, exports all arrays.
        If the output suffix is ``.npy``, exactly one array must be selected.
    compressed : bool, optional
        Whether to compress the ``.npz`` file (uses ``numpy.savez_compressed``).
    chunk_size : int, optional
        Chunk size in bytes used for streaming read/write.

    Raises
    ------
    ValueError
        If `output_path` suffix is not one of ``.npy`` or ``.npz``, or if ``.npy``
        is requested with multiple arrays.

    Examples
    --------
    >>> from numpack.io import to_numpy
    >>> to_numpy('input.npk', 'output.npz')
    >>> to_numpy('input.npk', 'single_array.npy', array_names=['my_array'])
    """
    output_path = Path(output_path)
    suffix = output_path.suffix.lower()

    npk = _open_numpack_for_read(input_path)
    try:
        if array_names is None:
            array_names = npk.get_member_list()

        if suffix == '.npy':
            if len(array_names) != 1:
                raise ValueError(
                    f"The .npy format can only store one array, but {len(array_names)} arrays were provided. "
                    "Use .npz or provide exactly one array name."
                )
            _to_npy(npk, output_path, array_names[0], chunk_size)
        elif suffix == '.npz':
            _to_npz(npk, output_path, array_names, compressed, chunk_size)
        else:
            raise ValueError(f"Unsupported file format: {suffix}. Supported: .npy and .npz")
    finally:
        npk.close()


def _to_npy(
    npk: Any,
    output_path: Path,
    array_name: str,
    chunk_size: int,
) -> None:
    """Export a single array to a ``.npy`` file."""
    shape = npk.get_shape(array_name)

    # Estimate size
    arr_sample = npk.getitem(array_name, [0])
    dtype = arr_sample.dtype
    estimated_size = int(np.prod(shape)) * dtype.itemsize

    if estimated_size > LARGE_FILE_THRESHOLD:
        # Large array: stream reads and writes
        _to_npy_streaming(npk, output_path, array_name, shape, dtype, chunk_size)
    else:
        # Small array: load directly
        arr = npk.load(array_name)
        np.save(str(output_path), arr)


def _to_npy_streaming(
    npk: Any,
    output_path: Path,
    array_name: str,
    shape: Tuple[int, ...],
    dtype: np.dtype,
    chunk_size: int,
) -> None:
    """Stream-export a large array to a ``.npy`` file."""
    batch_rows = estimate_chunk_rows(shape, dtype, chunk_size)
    total_rows = shape[0]

    # Create output file (pre-allocate space)
    # Use NumPy's format module to create a proper .npy header
    from numpy.lib import format as npy_format

    with open(output_path, 'wb') as f:
        # Write .npy header
        npy_format.write_array_header_1_0(
            f, {'descr': dtype.str, 'fortran_order': False, 'shape': shape}
        )
        header_size = f.tell()

    # Write data via memory mapping
    total_bytes = int(np.prod(shape)) * dtype.itemsize
    with open(output_path, 'r+b') as f:
        f.seek(0, 2)  # Seek to end
        f.truncate(header_size + total_bytes)  # Extend file

    # Memory-mapped write
    arr_out = np.memmap(output_path, dtype=dtype, mode='r+', offset=header_size, shape=shape)

    try:
        for start_idx in range(0, total_rows, batch_rows):
            end_idx = min(start_idx + batch_rows, total_rows)
            chunk = npk.getitem(array_name, slice(start_idx, end_idx))
            arr_out[start_idx:end_idx] = chunk
        arr_out.flush()
    finally:
        del arr_out


def _to_npz(
    npk: Any,
    output_path: Path,
    array_names: List[str],
    compressed: bool,
    chunk_size: int,
) -> None:
    """Export multiple arrays to a ``.npz`` file."""
    # The NPZ format does not support true streaming writes; data must be collected.
    # For large datasets, consider using other formats (for example, HDF5 or Zarr).

    arrays = {}
    for name in array_names:
        shape = npk.get_shape(name)
        arr_sample = npk.getitem(name, [0])
        dtype = arr_sample.dtype
        estimated_size = int(np.prod(shape)) * dtype.itemsize

        if estimated_size > LARGE_FILE_THRESHOLD:
            warnings.warn(
                f"Array '{name}' is large (>{estimated_size / 1e9:.1f}GB). "
                "The NPZ format loads all data into memory. "
                "For large datasets, consider using to_hdf5 or to_zarr.",
                UserWarning,
            )

        arrays[name] = npk.load(name)

    if compressed:
        np.savez_compressed(str(output_path), **arrays)
    else:
        np.savez(str(output_path), **arrays)
