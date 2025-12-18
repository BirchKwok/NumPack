from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union, TYPE_CHECKING

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
# Pandas DataFrame 转换
# =============================================================================

def from_pandas(
    df: "pd.DataFrame",
    output_path: Union[str, Path],
    array_name: str = 'data',
    drop_if_exists: bool = False,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> None:
    """从 Pandas DataFrame 导入为 NumPack 格式

    对于大 DataFrame（>1GB），使用流式写入。

    Parameters
    ----------
    df : pandas.DataFrame
        输入的 DataFrame
    output_path : str or Path
        输出的 NumPack 文件路径
    array_name : str, optional
        数组名称，默认 'data'
    drop_if_exists : bool, optional
        如果输出文件存在是否删除，默认 False
    chunk_size : int, optional
        分块大小（字节），默认 100MB

    Examples
    --------
    >>> import pandas as pd
    >>> from numpack.io import from_pandas
    >>> df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    >>> from_pandas(df, 'output.npk')
    """
    _check_pandas()

    arr = np.ascontiguousarray(df.values)
    estimated_size = arr.nbytes

    npk = _open_numpack_for_write(output_path, drop_if_exists)

    try:
        if estimated_size > LARGE_FILE_THRESHOLD:
            # 大 DataFrame：分块写入
            _save_array_streaming(npk, array_name, arr, chunk_size)
        else:
            npk.save({array_name: arr})
    finally:
        npk.close()


def to_pandas(
    input_path: Union[str, Path],
    array_name: Optional[str] = None,
    columns: Optional[List[str]] = None,
) -> "pd.DataFrame":
    """从 NumPack 导出为 Pandas DataFrame

    Parameters
    ----------
    input_path : str or Path
        输入的 NumPack 文件路径
    array_name : str, optional
        要导出的数组名
    columns : list of str, optional
        列名列表，如果为 None 则自动生成

    Returns
    -------
    pandas.DataFrame
        转换后的 DataFrame

    Examples
    --------
    >>> from numpack.io import to_pandas
    >>> df = to_pandas('input.npk')
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
                    f"NumPack 文件包含多个数组 {members}，请指定 array_name 参数"
                )

        arr = npk.load(array_name)

        if columns is None and arr.ndim == 2:
            columns = [f'col{i}' for i in range(arr.shape[1])]

        return pd.DataFrame(arr, columns=columns)
    finally:
        npk.close()
