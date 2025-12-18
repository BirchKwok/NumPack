from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import numpy as np

from .utils import (
    DEFAULT_CHUNK_SIZE,
    LARGE_FILE_THRESHOLD,
    _check_h5py,
    estimate_chunk_rows,
    _open_numpack_for_read,
    _open_numpack_for_write,
)


# =============================================================================
# HDF5 format conversion
# =============================================================================

def from_hdf5(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    dataset_names: Optional[List[str]] = None,
    group: str = '/',
    drop_if_exists: bool = False,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> None:
    """Import datasets from an HDF5 file into NumPack.

    Large datasets (by default > 1 GB) are imported using batched reads and
    streamed into NumPack.

    Parameters
    ----------
    input_path : str or Path
        Path to the input HDF5 file.
    output_path : str or Path
        Output NumPack directory path.
    dataset_names : list of str, optional
        Names of datasets to import. If None, all datasets under `group` are
        imported.
    group : str, optional
        HDF5 group path.
    drop_if_exists : bool, optional
        If True, delete the output path first if it already exists.
    chunk_size : int, optional
        Chunk size in bytes used for streaming conversion.

    Raises
    ------
    DependencyError
        If the optional dependency ``h5py`` is not installed.

    Examples
    --------
    >>> from numpack.io import from_hdf5
    >>> from_hdf5('data.h5', 'output.npk')
    >>> from_hdf5('data.h5', 'output.npk', dataset_names=['dataset1', 'dataset2'])
    >>> from_hdf5('data.h5', 'output.npk', group='/experiments/run1')
    """
    h5py = _check_h5py()

    npk = _open_numpack_for_write(output_path, drop_if_exists)

    try:
        with h5py.File(str(input_path), 'r') as h5f:
            grp = h5f[group]

            if dataset_names is None:
                # Collect all datasets under the group
                dataset_names = [
                    name
                    for name in grp.keys()
                    if isinstance(grp[name], h5py.Dataset)
                ]

            for name in dataset_names:
                dataset = grp[name]
                if not isinstance(dataset, h5py.Dataset):
                    warnings.warn(f"Skipping non-dataset object: {name}")
                    continue

                shape = dataset.shape
                dtype = dataset.dtype
                estimated_size = int(np.prod(shape)) * dtype.itemsize

                if estimated_size > LARGE_FILE_THRESHOLD and len(shape) > 0:
                    # Large dataset: streamed reads
                    _from_hdf5_dataset_streaming(npk, dataset, name, chunk_size)
                else:
                    # Small dataset: load directly
                    arr = dataset[...]
                    npk.save({name: arr})
    finally:
        npk.close()


def _from_hdf5_dataset_streaming(
    npk: Any,
    dataset: Any,  # h5py.Dataset
    array_name: str,
    chunk_size: int,
) -> None:
    """Stream-import an HDF5 dataset."""
    shape = dataset.shape
    dtype = dataset.dtype
    batch_rows = estimate_chunk_rows(shape, dtype, chunk_size)
    total_rows = shape[0]

    for start_idx in range(0, total_rows, batch_rows):
        end_idx = min(start_idx + batch_rows, total_rows)
        chunk = dataset[start_idx:end_idx]

        if start_idx == 0:
            npk.save({array_name: chunk})
        else:
            npk.append({array_name: chunk})


def to_hdf5(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    array_names: Optional[List[str]] = None,
    group: str = '/',
    compression: Optional[str] = 'gzip',
    compression_opts: Optional[int] = 4,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> None:
    """Export NumPack arrays to an HDF5 file.

    Large arrays (by default > 1 GB) are exported via streamed reads from
    NumPack and chunked writes to HDF5.

    Parameters
    ----------
    input_path : str or Path
        Input NumPack directory path.
    output_path : str or Path
        Output HDF5 file path.
    array_names : list of str, optional
        Names of arrays to export. If None, exports all arrays.
    group : str, optional
        HDF5 group path to write into.
    compression : str or None, optional
        Compression codec name (for example, ``"gzip"``). Use None to disable
        compression.
    compression_opts : int, optional
        Compression level/options.
    chunk_size : int, optional
        Chunk size in bytes used for streaming export.

    Raises
    ------
    DependencyError
        If the optional dependency ``h5py`` is not installed.

    Examples
    --------
    >>> from numpack.io import to_hdf5
    >>> to_hdf5('input.npk', 'output.h5')
    >>> to_hdf5('input.npk', 'output.h5', compression='lzf')
    """
    h5py = _check_h5py()

    npk = _open_numpack_for_read(input_path)

    try:
        if array_names is None:
            array_names = npk.get_member_list()

        with h5py.File(str(output_path), 'w') as h5f:
            # Create the group (if needed)
            if group != '/':
                grp = h5f.require_group(group)
            else:
                grp = h5f

            for name in array_names:
                shape = npk.get_shape(name)
                arr_sample = npk.getitem(name, [0])
                dtype = arr_sample.dtype
                estimated_size = int(np.prod(shape)) * dtype.itemsize

                # Compute HDF5 chunk shape
                if len(shape) > 0:
                    batch_rows = estimate_chunk_rows(shape, dtype, chunk_size)
                    chunks = (min(batch_rows, shape[0]),) + shape[1:]
                else:
                    chunks = None

                # Create dataset
                ds = grp.create_dataset(
                    name,
                    shape=shape,
                    dtype=dtype,
                    chunks=chunks if chunks else True,
                    compression=compression,
                    compression_opts=compression_opts if compression else None,
                )

                if estimated_size > LARGE_FILE_THRESHOLD and len(shape) > 0:
                    # Large array: streamed writes
                    _to_hdf5_dataset_streaming(npk, ds, name, shape, dtype, chunk_size)
                else:
                    # Small array: write directly
                    ds[...] = npk.load(name)
    finally:
        npk.close()


def _to_hdf5_dataset_streaming(
    npk: Any,
    dataset: Any,  # h5py.Dataset
    array_name: str,
    shape: Tuple[int, ...],
    dtype: np.dtype,
    chunk_size: int,
) -> None:
    """Stream-export a large array to an HDF5 dataset."""
    batch_rows = estimate_chunk_rows(shape, dtype, chunk_size)
    total_rows = shape[0]

    for start_idx in range(0, total_rows, batch_rows):
        end_idx = min(start_idx + batch_rows, total_rows)
        chunk = npk.getitem(array_name, slice(start_idx, end_idx))
        dataset[start_idx:end_idx] = chunk
