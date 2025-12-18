from __future__ import annotations

import warnings
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from .utils import (
    DEFAULT_CHUNK_SIZE,
    LARGE_FILE_THRESHOLD,
    _check_pyarrow,
    _open_numpack_for_read,
    _open_numpack_for_write,
)


# =============================================================================
# Feather format conversion
# =============================================================================

def from_feather(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    array_name: Optional[str] = None,
    columns: Optional[List[str]] = None,
    drop_if_exists: bool = False,
) -> None:
    """Import a Feather file into NumPack.

    Feather is a fast, lightweight columnar format.

    Parameters
    ----------
    input_path : str or Path
        Path to the input Feather file.
    output_path : str or Path
        Output NumPack directory path.
    array_name : str, optional
        Name of the output array. If None, defaults to the file stem.
    columns : list of str, optional
        Columns to read.
    drop_if_exists : bool, optional
        If True, delete the output path first if it already exists.

    Raises
    ------
    DependencyError
        If the optional dependency ``pyarrow`` is not installed.

    Examples
    --------
    >>> from numpack.io import from_feather
    >>> from_feather('data.feather', 'output.npk')
    """
    _check_pyarrow()
    import pyarrow.feather as feather

    input_path = Path(input_path)

    if array_name is None:
        array_name = input_path.stem

    # Feather requires full materialization (no streaming reads here)
    table = feather.read_table(str(input_path), columns=columns)
    arr = table.to_pandas().values

    npk = _open_numpack_for_write(output_path, drop_if_exists)
    try:
        npk.save({array_name: arr})
    finally:
        npk.close()


def to_feather(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    array_name: Optional[str] = None,
    compression: str = 'zstd',
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> None:
    """Export a NumPack array to Feather.

    Notes
    -----
    Feather does not support true streaming writes here; the array is loaded
    into memory before writing.

    Parameters
    ----------
    input_path : str or Path
        Input NumPack directory path.
    output_path : str or Path
        Output Feather file path.
    array_name : str, optional
        Name of the array to export. If None, the array is inferred only when
        the NumPack file contains exactly one array.
    compression : str, optional
        Feather compression codec.
    chunk_size : int, optional
        Chunk size in bytes (used only for size estimation / warning logic).

    Raises
    ------
    DependencyError
        If the optional dependency ``pyarrow`` is not installed.
    ValueError
        If `array_name` is not provided and the NumPack file contains multiple arrays.

    Examples
    --------
    >>> from numpack.io import to_feather
    >>> to_feather('input.npk', 'output.feather')
    """
    _check_pyarrow()
    import pyarrow as pa
    import pyarrow.feather as feather

    npk = _open_numpack_for_read(input_path)

    try:
        if array_name is None:
            members = npk.get_member_list()
            if len(members) == 1:
                array_name = members[0]
            else:
                raise ValueError(
                    f"NumPack contains multiple arrays {members}; please provide the array_name argument."
                )

        shape = npk.get_shape(array_name)
        estimated_size = int(np.prod(shape)) * npk.getitem(array_name, [0]).dtype.itemsize

        if estimated_size > LARGE_FILE_THRESHOLD:
            warnings.warn(
                f"Array '{array_name}' is large (>{estimated_size / 1e9:.1f}GB). "
                "The Feather format loads all data into memory. "
                "For large datasets, consider using to_parquet or to_zarr.",
                UserWarning,
            )

        arr = npk.load(array_name)

        # Convert to a Table
        if arr.ndim == 1:
            table = pa.table({'data': arr})
        else:
            columns = (
                {f'col{i}': arr[:, i] for i in range(arr.shape[1])}
                if arr.ndim == 2
                else {'data': arr.flatten()}
            )
            table = pa.table(columns)

        feather.write_feather(table, str(output_path), compression=compression)
    finally:
        npk.close()
