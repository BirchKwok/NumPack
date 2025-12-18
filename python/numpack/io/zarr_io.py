from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import numpy as np

from .utils import (
    DEFAULT_CHUNK_SIZE,
    LARGE_FILE_THRESHOLD,
    _check_zarr,
    estimate_chunk_rows,
    _open_numpack_for_read,
    _open_numpack_for_write,
)


# =============================================================================
# Zarr format conversion
# =============================================================================

def from_zarr(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    array_names: Optional[List[str]] = None,
    group: str = '/',
    drop_if_exists: bool = False,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> None:
    """Import arrays from a Zarr store into NumPack.

    Zarr stores are chunked natively. Large arrays are imported in batches and
    streamed into NumPack.

    Parameters
    ----------
    input_path : str or Path
        Path to the input Zarr store.
    output_path : str or Path
        Output NumPack directory path.
    array_names : list of str, optional
        Names of arrays to import. If None, imports all arrays under `group`.
    group : str, optional
        Zarr group path.
    drop_if_exists : bool, optional
        If True, delete the output path first if it already exists.
    chunk_size : int, optional
        Chunk size in bytes used for streaming conversion.

    Raises
    ------
    DependencyError
        If the optional dependency ``zarr`` is not installed.

    Examples
    --------
    >>> from numpack.io import from_zarr
    >>> from_zarr('data.zarr', 'output.npk')
    >>> from_zarr('data.zarr', 'output.npk', array_names=['arr1', 'arr2'])
    """
    zarr = _check_zarr()

    npk = _open_numpack_for_write(output_path, drop_if_exists)

    try:
        store = zarr.open(str(input_path), mode='r')
        if group != '/':
            store = store[group]

        if array_names is None:
            # Collect all arrays
            array_names = [name for name in store.array_keys()]

        for name in array_names:
            arr = store[name]
            shape = arr.shape
            dtype = arr.dtype
            estimated_size = int(np.prod(shape)) * dtype.itemsize

            if estimated_size > LARGE_FILE_THRESHOLD and len(shape) > 0:
                # Large array: streamed reads
                _from_zarr_array_streaming(npk, arr, name, chunk_size)
            else:
                # Small array: load directly
                npk.save({name: arr[...]})
    finally:
        npk.close()


def _from_zarr_array_streaming(
    npk: Any,
    zarr_arr: Any,  # zarr.Array
    array_name: str,
    chunk_size: int,
) -> None:
    """Stream-import a Zarr array."""
    shape = zarr_arr.shape
    dtype = zarr_arr.dtype
    batch_rows = estimate_chunk_rows(shape, dtype, chunk_size)
    total_rows = shape[0]

    for start_idx in range(0, total_rows, batch_rows):
        end_idx = min(start_idx + batch_rows, total_rows)
        chunk = zarr_arr[start_idx:end_idx]

        if start_idx == 0:
            npk.save({array_name: chunk})
        else:
            npk.append({array_name: chunk})


def to_zarr(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    array_names: Optional[List[str]] = None,
    group: str = '/',
    compressor: Optional[str] = 'default',
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> None:
    """Export NumPack arrays to a Zarr store.

    Parameters
    ----------
    input_path : str or Path
        Input NumPack directory path.
    output_path : str or Path
        Output Zarr store path.
    array_names : list of str, optional
        Names of arrays to export. If None, exports all arrays.
    group : str, optional
        Zarr group path.
    compressor : str or None, optional
        Compressor configuration. The default value attempts to use Blosc.
        Use None to disable compression.
    chunk_size : int, optional
        Chunk size in bytes used for streaming export.

    Raises
    ------
    DependencyError
        If the optional dependency ``zarr`` is not installed.

    Examples
    --------
    >>> from numpack.io import to_zarr
    >>> to_zarr('input.npk', 'output.zarr')
    """
    zarr = _check_zarr()

    npk = _open_numpack_for_read(input_path)

    try:
        if array_names is None:
            array_names = npk.get_member_list()

        # Create Zarr store
        store = zarr.open(str(output_path), mode='w')
        if group != '/':
            store = store.require_group(group)

        # Configure compressor
        if compressor == 'default':
            try:
                from zarr.codecs import BloscCodec, BloscCname, BloscShuffle

                compressor_obj = BloscCodec(
                    cname=BloscCname.zstd,
                    clevel=3,
                    shuffle=BloscShuffle.bitshuffle,
                )
            except ImportError:
                compressor_obj = None
        elif compressor is None:
            compressor_obj = None
        else:
            compressor_obj = compressor

        for name in array_names:
            shape = npk.get_shape(name)
            arr_sample = npk.getitem(name, [0])
            dtype = arr_sample.dtype
            estimated_size = int(np.prod(shape)) * dtype.itemsize

            # Compute chunk shape
            if len(shape) > 0:
                batch_rows = estimate_chunk_rows(shape, dtype, chunk_size)
                chunks = (min(batch_rows, shape[0]),) + shape[1:]
            else:
                chunks = shape

            # Create Zarr array
            if hasattr(store, 'create_array'):
                zarr_arr = store.create_array(
                    name,
                    shape=shape,
                    dtype=dtype,
                    chunks=chunks if chunks else None,
                    compressors=compressor_obj,
                    overwrite=True,
                )
            else:
                zarr_arr = store.create_dataset(
                    name,
                    shape=shape,
                    dtype=dtype,
                    chunks=chunks if chunks else None,
                    compressor=compressor_obj,
                )

            if estimated_size > LARGE_FILE_THRESHOLD and len(shape) > 0:
                # Large array: streamed writes
                _to_zarr_array_streaming(npk, zarr_arr, name, shape, dtype, chunk_size)
            else:
                # Small array: write directly
                zarr_arr[...] = npk.load(name)
    finally:
        npk.close()


def _to_zarr_array_streaming(
    npk: Any,
    zarr_arr: Any,  # zarr.Array
    array_name: str,
    shape: Tuple[int, ...],
    dtype: np.dtype,
    chunk_size: int,
) -> None:
    """Stream-export a large array to Zarr."""
    batch_rows = estimate_chunk_rows(shape, dtype, chunk_size)
    total_rows = shape[0]

    for start_idx in range(0, total_rows, batch_rows):
        end_idx = min(start_idx + batch_rows, total_rows)
        chunk = npk.getitem(array_name, slice(start_idx, end_idx))
        zarr_arr[start_idx:end_idx] = chunk
