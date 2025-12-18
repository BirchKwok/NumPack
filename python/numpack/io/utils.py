from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional, Tuple, Union

import numpy as np

# 大文件阈值：1GB
LARGE_FILE_THRESHOLD = 1 * 1024 * 1024 * 1024  # 1GB in bytes

# 默认分块大小：100MB（行数会根据数据类型自动计算）
DEFAULT_CHUNK_SIZE = 100 * 1024 * 1024  # 100MB in bytes

# 默认批处理行数（用于流式处理）
DEFAULT_BATCH_ROWS = 100000


# =============================================================================
# 依赖检查工具
# =============================================================================

class DependencyError(ImportError):
    """可选依赖未安装时抛出的异常"""

    pass


def _check_dependency(module_name: str, package_name: Optional[str] = None) -> Any:
    """检查并导入可选依赖

    Parameters
    ----------
    module_name : str
        要导入的模块名
    package_name : str, optional
        pip 安装时的包名（如果与模块名不同）

    Returns
    -------
    module
        导入的模块

    Raises
    ------
    DependencyError
        如果依赖未安装
    """
    import importlib

    if package_name is None:
        package_name = module_name

    try:
        return importlib.import_module(module_name)
    except ImportError:
        raise DependencyError(
            f"需要安装 '{package_name}' 才能使用此功能。\n"
            f"请运行: pip install {package_name}"
        )


def _check_h5py():
    """检查并导入 h5py"""
    return _check_dependency('h5py')


def _check_zarr():
    """检查并导入 zarr"""
    return _check_dependency('zarr')


def _check_pyarrow():
    """检查并导入 pyarrow"""
    return _check_dependency('pyarrow')


def _check_pandas():
    """检查并导入 pandas"""
    return _check_dependency('pandas')


def _check_torch():
    """检查并导入 torch (PyTorch)"""
    return _check_dependency('torch', 'torch')


def _check_s3fs():
    """检查并导入 s3fs"""
    return _check_dependency('s3fs')


def _check_boto3():
    """检查并导入 boto3"""
    return _check_dependency('boto3')


# =============================================================================
# 文件大小和流式处理工具
# =============================================================================

def get_file_size(path: Union[str, Path]) -> int:
    """获取文件大小（字节）

    Parameters
    ----------
    path : str or Path
        文件路径

    Returns
    -------
    int
        文件大小（字节），如果是目录则返回目录总大小
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
    """检查文件是否为大文件（需要流式处理）

    Parameters
    ----------
    path : str or Path
        文件路径
    threshold : int, optional
        大文件阈值（字节），默认 1GB

    Returns
    -------
    bool
        如果文件大于阈值返回 True
    """
    return get_file_size(path) > threshold


def estimate_chunk_rows(
    shape: Tuple[int, ...],
    dtype: np.dtype,
    target_chunk_bytes: int = DEFAULT_CHUNK_SIZE,
) -> int:
    """估算每个分块应包含的行数

    Parameters
    ----------
    shape : tuple
        数组形状
    dtype : numpy.dtype
        数据类型
    target_chunk_bytes : int, optional
        目标分块大小（字节），默认 100MB

    Returns
    -------
    int
        建议的每批处理行数
    """
    if len(shape) == 0:
        return 1

    # 计算每行的字节数
    row_elements = int(np.prod(shape[1:])) if len(shape) > 1 else 1
    bytes_per_row = row_elements * dtype.itemsize

    if bytes_per_row == 0:
        return DEFAULT_BATCH_ROWS

    # 计算目标行数
    target_rows = max(1, target_chunk_bytes // bytes_per_row)

    # 限制在合理范围内
    return min(target_rows, shape[0], DEFAULT_BATCH_ROWS * 10)


# =============================================================================
# NumPack 工具函数
# =============================================================================

def _get_numpack_class():
    """获取 NumPack 类"""
    from numpack import NumPack

    return NumPack


def _open_numpack_for_write(
    output_path: Union[str, Path],
    drop_if_exists: bool = False,
) -> Any:
    """打开 NumPack 文件用于写入

    Parameters
    ----------
    output_path : str or Path
        输出路径
    drop_if_exists : bool, optional
        如果文件存在是否删除，默认 False

    Returns
    -------
    NumPack
        NumPack 实例
    """
    NumPack = _get_numpack_class()
    npk = NumPack(str(output_path), drop_if_exists=drop_if_exists)
    npk.open()
    return npk


def _open_numpack_for_read(input_path: Union[str, Path]) -> Any:
    """打开 NumPack 文件用于读取

    Parameters
    ----------
    input_path : str or Path
        输入路径

    Returns
    -------
    NumPack
        NumPack 实例
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
    """流式保存大数组到 NumPack"""
    shape = arr.shape
    dtype = arr.dtype
    batch_rows = estimate_chunk_rows(shape, dtype, chunk_size)
    total_rows = shape[0]

    for start_idx in range(0, total_rows, batch_rows):
        end_idx = min(start_idx + batch_rows, total_rows)
        chunk = np.array(arr[start_idx:end_idx])

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
    """检查数组大小并决定是否使用流式保存"""
    if arr.nbytes > LARGE_FILE_THRESHOLD and arr.ndim > 0:
        _save_array_streaming(npk, array_name, arr, chunk_size)
    else:
        npk.save({array_name: arr})


# =============================================================================
# 便捷函数工具
# =============================================================================

def _infer_format(path: Path) -> str:
    """从文件路径推断格式"""
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

    # 检查是否是目录（可能是 NumPack 或 Zarr）
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
