from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Union

import numpy as np

from .utils import (
    DEFAULT_CHUNK_SIZE,
    _check_torch,
    _open_numpack_for_read,
    _open_numpack_for_write,
    _save_array_with_streaming_check,
)


# =============================================================================
# PyTorch tensor conversion
# =============================================================================

def from_pytorch(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    key: Optional[str] = None,
    drop_if_exists: bool = False,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> None:
    """Import tensors from a PyTorch ``.pt``/``.pth`` file into NumPack.

    Parameters
    ----------
    input_path : str or Path
        Path to the input PyTorch file.
    output_path : str or Path
        Output NumPack directory path.
    key : str, optional
        If the file contains a dict, load only this key.
    drop_if_exists : bool, optional
        If True, delete the output path first if it already exists.
    chunk_size : int, optional
        Chunk size in bytes used for streaming conversion.

    Raises
    ------
    DependencyError
        If the optional dependency ``torch`` is not installed.
    KeyError
        If `key` is provided but not present in the loaded dict.
    TypeError
        If the loaded object is not a tensor or a dict of tensors.

    Examples
    --------
    >>> from numpack.io import from_pytorch
    >>> from_pytorch('model.pt', 'output.npk')
    >>> from_pytorch('data.pt', 'output.npk', key='features')
    """
    torch = _check_torch()

    input_path = Path(input_path)

    # Load PyTorch file
    data = torch.load(str(input_path), map_location='cpu', weights_only=False)

    npk = _open_numpack_for_write(output_path, drop_if_exists)

    try:
        if isinstance(data, dict):
            # Dict: save all tensors or a specified key
            if key is not None:
                if key not in data:
                    raise KeyError(f"Key '{key}' was not found in the file. Available keys: {list(data.keys())}")
                tensor = data[key]
                if torch.is_tensor(tensor):
                    arr = tensor.detach().cpu().numpy()
                    _save_array_with_streaming_check(npk, key, arr, chunk_size)
                else:
                    raise TypeError(f"Value for key '{key}' is not a tensor")
            else:
                for name, tensor in data.items():
                    if torch.is_tensor(tensor):
                        arr = tensor.detach().cpu().numpy()
                        _save_array_with_streaming_check(npk, name, arr, chunk_size)
        elif torch.is_tensor(data):
            # Single tensor
            array_name = input_path.stem
            arr = data.detach().cpu().numpy()
            _save_array_with_streaming_check(npk, array_name, arr, chunk_size)
        else:
            raise TypeError(f"Unsupported PyTorch data type: {type(data)}")
    finally:
        npk.close()


def to_pytorch(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    array_names: Optional[List[str]] = None,
    as_dict: bool = True,
) -> None:
    """Export NumPack arrays to a PyTorch ``.pt`` file.

    Parameters
    ----------
    input_path : str or Path
        Input NumPack directory path.
    output_path : str or Path
        Output PyTorch file path.
    array_names : list of str, optional
        Names of arrays to export. If None, exports all arrays.
    as_dict : bool, optional
        If True, save a dict mapping array names to tensors. If False and only
        one array is exported, save the tensor directly.

    Raises
    ------
    DependencyError
        If the optional dependency ``torch`` is not installed.

    Examples
    --------
    >>> from numpack.io import to_pytorch
    >>> to_pytorch('input.npk', 'output.pt')
    """
    torch = _check_torch()

    npk = _open_numpack_for_read(input_path)

    try:
        if array_names is None:
            array_names = npk.get_member_list()

        tensors = {}
        for name in array_names:
            arr = npk.load(name)
            tensors[name] = torch.from_numpy(arr)

        if not as_dict and len(tensors) == 1:
            # Save a single tensor
            torch.save(list(tensors.values())[0], str(output_path))
        else:
            # Save a dict
            torch.save(tensors, str(output_path))
    finally:
        npk.close()
