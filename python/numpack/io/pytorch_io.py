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
# PyTorch Tensor 转换
# =============================================================================

def from_pytorch(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    key: Optional[str] = None,
    drop_if_exists: bool = False,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> None:
    """从 PyTorch .pt/.pth 文件导入为 NumPack 格式

    Parameters
    ----------
    input_path : str or Path
        输入的 PyTorch 文件路径
    output_path : str or Path
        输出的 NumPack 文件路径
    key : str, optional
        如果文件是字典，指定要加载的键名
    drop_if_exists : bool, optional
        如果输出文件存在是否删除，默认 False
    chunk_size : int, optional
        分块大小（字节），默认 100MB

    Examples
    --------
    >>> from numpack.io import from_pytorch
    >>> from_pytorch('model.pt', 'output.npk')
    >>> from_pytorch('data.pt', 'output.npk', key='features')
    """
    torch = _check_torch()

    input_path = Path(input_path)

    # 加载 PyTorch 文件
    data = torch.load(str(input_path), map_location='cpu', weights_only=False)

    npk = _open_numpack_for_write(output_path, drop_if_exists)

    try:
        if isinstance(data, dict):
            # 字典：保存所有张量或指定的键
            if key is not None:
                if key not in data:
                    raise KeyError(f"键 '{key}' 不在文件中。可用的键: {list(data.keys())}")
                tensor = data[key]
                if torch.is_tensor(tensor):
                    arr = tensor.detach().cpu().numpy()
                    _save_array_with_streaming_check(npk, key, arr, chunk_size)
                else:
                    raise TypeError(f"键 '{key}' 的值不是张量类型")
            else:
                for name, tensor in data.items():
                    if torch.is_tensor(tensor):
                        arr = tensor.detach().cpu().numpy()
                        _save_array_with_streaming_check(npk, name, arr, chunk_size)
        elif torch.is_tensor(data):
            # 单个张量
            array_name = input_path.stem
            arr = data.detach().cpu().numpy()
            _save_array_with_streaming_check(npk, array_name, arr, chunk_size)
        else:
            raise TypeError(f"不支持的 PyTorch 数据类型: {type(data)}")
    finally:
        npk.close()


def to_pytorch(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    array_names: Optional[List[str]] = None,
    as_dict: bool = True,
) -> None:
    """从 NumPack 导出为 PyTorch .pt 格式

    Parameters
    ----------
    input_path : str or Path
        输入的 NumPack 文件路径
    output_path : str or Path
        输出的 PyTorch 文件路径
    array_names : list of str, optional
        要导出的数组名列表。如果为 None，导出所有数组。
    as_dict : bool, optional
        是否保存为字典格式，默认 True。如果 False 且只有一个数组，
        则直接保存张量。

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
            # 保存单个张量
            torch.save(list(tensors.values())[0], str(output_path))
        else:
            # 保存字典
            torch.save(tensors, str(output_path))
    finally:
        npk.close()
