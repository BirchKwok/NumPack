from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Union, TYPE_CHECKING

import numpy as np

from .utils import (
    DEFAULT_CHUNK_SIZE,
    LARGE_FILE_THRESHOLD,
    _check_pandas,
    _open_numpack_for_read,
    _open_numpack_for_write,
    _save_array_streaming,
)

if TYPE_CHECKING:
    import pandas as pd


# =============================================================================
# Pandas DataFrame conversion
# =============================================================================

def from_dataframe(
    df: "pd.DataFrame",
    output_path: Union[str, Path],
    array_name: str = 'data',
    drop_if_exists: bool = False,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    return_npk_obj: bool = False,
) -> Any:
    """Save a pandas DataFrame to NumPack format.

    Large DataFrames (by default > 1 GB) are streamed into NumPack in chunks.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.
    output_path : str or Path
        Output NumPack directory path (.npk).
    array_name : str, optional
        Name of the output array.
    drop_if_exists : bool, optional
        If True, delete the output path first if it already exists.
    chunk_size : int, optional
        Chunk size in bytes used for streaming write.
    return_npk_obj : bool, optional
        If True, return an opened NumPack instance for output_path.

    Returns
    -------
    NumPack or None
        The NumPack instance if `return_npk_obj` is True, otherwise None.

    Raises
    ------
    DependencyError
        If the optional dependency ``pandas`` is not installed.

    Examples
    --------
    >>> import pandas as pd
    >>> from numpack.io import from_dataframe
    >>> df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    >>> from_dataframe(df, 'output.npk')
    """
    _check_pandas()

    arr = np.ascontiguousarray(df.values)
    estimated_size = arr.nbytes

    npk = _open_numpack_for_write(output_path, drop_if_exists)

    try:
        if estimated_size > LARGE_FILE_THRESHOLD:
            # Large DataFrame: chunked writes
            _save_array_streaming(npk, array_name, arr, chunk_size)
        else:
            npk.save({array_name: arr})
    finally:
        npk.close()

    if return_npk_obj:
        return _open_numpack_for_read(output_path)
    return None


def to_dataframe(
    input_path: Union[str, Path],
    array_name: Optional[str] = None,
    columns: Optional[List[str]] = None,
) -> "pd.DataFrame":
    """Load a NumPack array as a pandas DataFrame.

    Parameters
    ----------
    input_path : str or Path
        Input NumPack directory path (.npk).
    array_name : str, optional
        Name of the array to export. If None, the array is inferred only when
        the NumPack file contains exactly one array.
    columns : list of str, optional
        Column names. If None and the array is 2D, column names are generated.

    Returns
    -------
    pandas.DataFrame
        A DataFrame view of the exported array.

    Raises
    ------
    DependencyError
        If the optional dependency ``pandas`` is not installed.
    ValueError
        If `array_name` is not provided and the NumPack file contains multiple arrays.

    Examples
    --------
    >>> from numpack.io import to_dataframe
    >>> df = to_dataframe('input.npk')
    """
    pd = _check_pandas()

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

        arr = npk.load(array_name)

        if columns is None and arr.ndim == 2:
            columns = [f'col{i}' for i in range(arr.shape[1])]

        return pd.DataFrame(arr, columns=columns)
    finally:
        npk.close()


# =============================================================================
# Legacy Aliases (deprecated, will be removed in 0.6.0)
# =============================================================================

from .utils import deprecated_alias

from_pandas = deprecated_alias('from_dataframe', from_dataframe)
from_pandas.__name__ = 'from_pandas'

to_pandas = deprecated_alias('to_dataframe', to_dataframe)
to_pandas.__name__ = 'to_pandas'


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Primary names (recommended)
    'from_dataframe',
    'to_dataframe',
    # Legacy aliases (deprecated, will be removed in 0.6.0)
    'from_pandas',
    'to_pandas',
]
