"""
NumPack IO Module - 数据格式转换工具

提供从常见数据科学格式（npy, npz, zarr, hdf5, parquet, feather, pandas, 
csv, txt, pytorch 等）导入为 NumPack 格式，以及从 NumPack 导出为这些格式的功能。

特性：
- 延迟依赖检查：使用时才检查是否安装相关库
- 大文件流式处理：超过 1GB 的文件自动使用分块流式处理
- 高性能：利用并行 I/O 和内存映射优化

Examples
--------
从 NumPy 文件导入：

>>> from numpack.io import from_numpy
>>> from_numpy('data.npy', 'output.npk')

导出为 HDF5：

>>> from numpack.io import to_hdf5
>>> to_hdf5('input.npk', 'output.h5')

流式转换大文件：

>>> from numpack.io import from_csv
>>> from_csv('large_data.csv', 'output.npk')  # 自动检测并使用流式处理
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import (
    Any, 
    Callable, 
    Dict, 
    Iterator, 
    List, 
    Optional, 
    Tuple, 
    Union,
    TYPE_CHECKING
)
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

if TYPE_CHECKING:
    import pandas as pd
    import h5py
    import zarr
    import pyarrow as pa
    import torch

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
    target_chunk_bytes: int = DEFAULT_CHUNK_SIZE
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
    drop_if_exists: bool = False
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


# =============================================================================
# NumPy 格式转换 (npy/npz)
# =============================================================================

def from_numpy(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    array_name: Optional[str] = None,
    drop_if_exists: bool = False,
    chunk_size: int = DEFAULT_CHUNK_SIZE
) -> None:
    """从 NumPy npy/npz 文件导入为 NumPack 格式
    
    对于大文件（>1GB），自动使用内存映射和流式写入。
    
    Parameters
    ----------
    input_path : str or Path
        输入的 .npy 或 .npz 文件路径
    output_path : str or Path
        输出的 NumPack 文件路径
    array_name : str, optional
        数组名称。对于 .npy 文件，默认使用文件名（不含扩展名）。
        对于 .npz 文件，忽略此参数，使用 npz 内的数组名。
    drop_if_exists : bool, optional
        如果输出文件存在是否删除，默认 False
    chunk_size : int, optional
        分块大小（字节），默认 100MB
    
    Examples
    --------
    >>> from numpack.io import from_numpy
    >>> from_numpy('data.npy', 'output.npk')
    >>> from_numpy('data.npz', 'output.npk')  # 保留所有数组名
    """
    input_path = Path(input_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"文件不存在: {input_path}")
    
    suffix = input_path.suffix.lower()
    
    if suffix == '.npy':
        _from_npy(input_path, output_path, array_name, drop_if_exists, chunk_size)
    elif suffix == '.npz':
        _from_npz(input_path, output_path, drop_if_exists, chunk_size)
    else:
        raise ValueError(f"不支持的文件格式: {suffix}，支持 .npy 和 .npz")


def _from_npy(
    input_path: Path,
    output_path: Union[str, Path],
    array_name: Optional[str],
    drop_if_exists: bool,
    chunk_size: int
) -> None:
    """从单个 .npy 文件导入"""
    if array_name is None:
        array_name = input_path.stem
    
    file_size = get_file_size(input_path)
    
    if file_size > LARGE_FILE_THRESHOLD:
        # 大文件：使用内存映射流式写入
        _from_npy_streaming(input_path, output_path, array_name, drop_if_exists, chunk_size)
    else:
        # 小文件：直接加载
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
    chunk_size: int
) -> None:
    """大文件 npy 流式导入"""
    # 使用内存映射加载
    arr_mmap = np.load(str(input_path), mmap_mode='r')
    shape = arr_mmap.shape
    dtype = arr_mmap.dtype
    
    # 计算分块行数
    batch_rows = estimate_chunk_rows(shape, dtype, chunk_size)
    total_rows = shape[0]
    
    npk = _open_numpack_for_write(output_path, drop_if_exists)
    try:
        # 分块写入
        for start_idx in range(0, total_rows, batch_rows):
            end_idx = min(start_idx + batch_rows, total_rows)
            chunk = np.array(arr_mmap[start_idx:end_idx])  # 复制到内存
            
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
    chunk_size: int
) -> None:
    """从 .npz 文件导入"""
    # 检查文件大小
    file_size = get_file_size(input_path)
    
    npk = _open_numpack_for_write(output_path, drop_if_exists)
    try:
        if file_size > LARGE_FILE_THRESHOLD:
            # 大文件：逐个数组加载
            with np.load(str(input_path), mmap_mode='r') as npz:
                for name in npz.files:
                    arr = npz[name]
                    # 对于大数组，流式写入
                    if arr.nbytes > LARGE_FILE_THRESHOLD:
                        _save_array_streaming(npk, name, arr, chunk_size)
                    else:
                        npk.save({name: np.array(arr)})
        else:
            # 小文件：直接加载
            with np.load(str(input_path)) as npz:
                arrays = {name: npz[name] for name in npz.files}
                npk.save(arrays)
    finally:
        npk.close()


def _save_array_streaming(
    npk: Any, 
    array_name: str, 
    arr: np.ndarray, 
    chunk_size: int
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


def to_numpy(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    array_names: Optional[List[str]] = None,
    compressed: bool = True,
    chunk_size: int = DEFAULT_CHUNK_SIZE
) -> None:
    """从 NumPack 导出为 NumPy npy/npz 格式
    
    对于大文件（>1GB），使用流式读取。
    
    Parameters
    ----------
    input_path : str or Path
        输入的 NumPack 文件路径
    output_path : str or Path
        输出的 .npy 或 .npz 文件路径
    array_names : list of str, optional
        要导出的数组名列表。如果为 None，导出所有数组。
        如果只有一个数组且输出为 .npy，则直接保存该数组。
    compressed : bool, optional
        对于 .npz 文件，是否压缩，默认 True
    chunk_size : int, optional
        分块大小（字节），默认 100MB
    
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
                    f".npy 格式只能保存单个数组，但指定了 {len(array_names)} 个数组。"
                    "请使用 .npz 格式或只指定一个数组名。"
                )
            _to_npy(npk, output_path, array_names[0], chunk_size)
        elif suffix == '.npz':
            _to_npz(npk, output_path, array_names, compressed, chunk_size)
        else:
            raise ValueError(f"不支持的文件格式: {suffix}，支持 .npy 和 .npz")
    finally:
        npk.close()


def _to_npy(
    npk: Any, 
    output_path: Path, 
    array_name: str, 
    chunk_size: int
) -> None:
    """导出单个数组为 .npy 文件"""
    shape = npk.get_shape(array_name)
    
    # 估算大小
    arr_sample = npk.getitem(array_name, [0])
    dtype = arr_sample.dtype
    estimated_size = int(np.prod(shape)) * dtype.itemsize
    
    if estimated_size > LARGE_FILE_THRESHOLD:
        # 大数组：流式读取并写入
        _to_npy_streaming(npk, output_path, array_name, shape, dtype, chunk_size)
    else:
        # 小数组：直接加载
        arr = npk.load(array_name)
        np.save(str(output_path), arr)


def _to_npy_streaming(
    npk: Any,
    output_path: Path,
    array_name: str,
    shape: Tuple[int, ...],
    dtype: np.dtype,
    chunk_size: int
) -> None:
    """流式导出大数组为 .npy 文件"""
    batch_rows = estimate_chunk_rows(shape, dtype, chunk_size)
    total_rows = shape[0]
    
    # 创建输出文件（预分配空间）
    # 使用 numpy 的 format 模块创建正确的 npy 文件头
    from numpy.lib import format as npy_format
    
    with open(output_path, 'wb') as f:
        # 写入 npy 文件头
        npy_format.write_array_header_1_0(f, {'descr': dtype.str, 'fortran_order': False, 'shape': shape})
        header_size = f.tell()
    
    # 使用内存映射写入数据
    total_bytes = int(np.prod(shape)) * dtype.itemsize
    with open(output_path, 'r+b') as f:
        f.seek(0, 2)  # 移动到文件末尾
        f.truncate(header_size + total_bytes)  # 扩展文件
    
    # 内存映射写入
    arr_out = np.memmap(output_path, dtype=dtype, mode='r+', offset=header_size, shape=shape)
    
    try:
        for start_idx in range(0, total_rows, batch_rows):
            end_idx = min(start_idx + batch_rows, total_rows)
            indices = list(range(start_idx, end_idx))
            chunk = npk.getitem(array_name, indices)
            arr_out[start_idx:end_idx] = chunk
        arr_out.flush()
    finally:
        del arr_out


def _to_npz(
    npk: Any,
    output_path: Path,
    array_names: List[str],
    compressed: bool,
    chunk_size: int
) -> None:
    """导出多个数组为 .npz 文件"""
    # NPZ 格式不支持真正的流式写入，需要先收集所有数据
    # 对于大数据集，建议使用其他格式（如 HDF5 或 Zarr）
    
    arrays = {}
    for name in array_names:
        shape = npk.get_shape(name)
        arr_sample = npk.getitem(name, [0])
        dtype = arr_sample.dtype
        estimated_size = int(np.prod(shape)) * dtype.itemsize
        
        if estimated_size > LARGE_FILE_THRESHOLD:
            warnings.warn(
                f"数组 '{name}' 较大（>{estimated_size / 1e9:.1f}GB），"
                "NPZ 格式需要将所有数据加载到内存。"
                "对于大数据集，建议使用 to_hdf5 或 to_zarr。",
                UserWarning
            )
        
        arrays[name] = npk.load(name)
    
    if compressed:
        np.savez_compressed(str(output_path), **arrays)
    else:
        np.savez(str(output_path), **arrays)


# =============================================================================
# CSV 格式转换
# =============================================================================

def from_csv(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    array_name: Optional[str] = None,
    drop_if_exists: bool = False,
    dtype: Optional[np.dtype] = None,
    delimiter: str = ',',
    skiprows: int = 0,
    max_rows: Optional[int] = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    **kwargs
) -> None:
    """从 CSV 文件导入为 NumPack 格式
    
    对于大文件（>1GB），自动使用流式处理。
    
    Parameters
    ----------
    input_path : str or Path
        输入的 CSV 文件路径
    output_path : str or Path
        输出的 NumPack 文件路径
    array_name : str, optional
        数组名称，默认使用文件名（不含扩展名）
    drop_if_exists : bool, optional
        如果输出文件存在是否删除，默认 False
    dtype : numpy.dtype, optional
        数据类型，默认自动推断
    delimiter : str, optional
        分隔符，默认 ','
    skiprows : int, optional
        跳过的行数，默认 0
    max_rows : int, optional
        最大读取行数，默认 None（读取全部）
    chunk_size : int, optional
        分块大小（字节），默认 100MB
    **kwargs
        传递给 numpy.loadtxt 或 pandas.read_csv 的其他参数
    
    Examples
    --------
    >>> from numpack.io import from_csv
    >>> from_csv('data.csv', 'output.npk')
    >>> from_csv('data.csv', 'output.npk', dtype=np.float32, delimiter=';')
    """
    input_path = Path(input_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"文件不存在: {input_path}")
    
    if array_name is None:
        array_name = input_path.stem
    
    file_size = get_file_size(input_path)
    
    if file_size > LARGE_FILE_THRESHOLD:
        # 大文件：使用 pandas 分块读取（如果可用）
        _from_csv_streaming(
            input_path, output_path, array_name, drop_if_exists,
            dtype, delimiter, skiprows, chunk_size, **kwargs
        )
    else:
        # 小文件：直接加载
        try:
            # 尝试使用 pandas（更快更灵活）
            pd = _check_pandas()
            if 'header' not in kwargs:
                kwargs['header'] = None
            df = pd.read_csv(
                input_path, 
                delimiter=delimiter, 
                skiprows=skiprows,
                nrows=max_rows,
                dtype=dtype,
                **kwargs
            )
            # 确保数组是 C-contiguous 的（NumPack 要求）
            arr = np.ascontiguousarray(df.values)
        except DependencyError:
            # 回退到 numpy
            arr = np.loadtxt(
                str(input_path),
                delimiter=delimiter,
                skiprows=skiprows,
                max_rows=max_rows,
                dtype=dtype if dtype is not None else np.float64,
                **{k: v for k, v in kwargs.items() if k in ['comments', 'usecols', 'unpack', 'ndmin', 'encoding']}
            )
        
        npk = _open_numpack_for_write(output_path, drop_if_exists)
        try:
            npk.save({array_name: arr})
        finally:
            npk.close()


def _from_csv_streaming(
    input_path: Path,
    output_path: Union[str, Path],
    array_name: str,
    drop_if_exists: bool,
    dtype: Optional[np.dtype],
    delimiter: str,
    skiprows: int,
    chunk_size: int,
    **kwargs
) -> None:
    """大文件 CSV 流式导入"""
    pd = _check_pandas()
    
    # 计算分块行数（估算每行约 100 字节）
    chunk_rows = max(1000, chunk_size // 100)
    
    npk = _open_numpack_for_write(output_path, drop_if_exists)
    first_chunk = True
    
    try:
        if 'header' not in kwargs:
            kwargs['header'] = None
        reader = pd.read_csv(
            input_path,
            delimiter=delimiter,
            skiprows=skiprows,
            dtype=dtype,
            chunksize=chunk_rows,
            **kwargs
        )
        
        for chunk_df in reader:
            # 确保数组是 C-contiguous 的（NumPack 要求）
            arr = np.ascontiguousarray(chunk_df.values)
            if dtype is not None:
                arr = arr.astype(dtype)
            
            if first_chunk:
                npk.save({array_name: arr})
                first_chunk = False
            else:
                npk.append({array_name: arr})
    finally:
        npk.close()


def to_csv(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    array_name: Optional[str] = None,
    delimiter: str = ',',
    header: bool = False,
    fmt: str = '%.18e',
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    **kwargs
) -> None:
    """从 NumPack 导出为 CSV 格式
    
    对于大文件（>1GB），使用流式读取。
    
    Parameters
    ----------
    input_path : str or Path
        输入的 NumPack 文件路径
    output_path : str or Path
        输出的 CSV 文件路径
    array_name : str, optional
        要导出的数组名，如果为 None 且只有一个数组则使用该数组
    delimiter : str, optional
        分隔符，默认 ','
    header : bool, optional
        是否写入列标题，默认 False
    fmt : str, optional
        数值格式，默认 '%.18e'
    chunk_size : int, optional
        分块大小（字节），默认 100MB
    **kwargs
        传递给 numpy.savetxt 的其他参数
    
    Examples
    --------
    >>> from numpack.io import to_csv
    >>> to_csv('input.npk', 'output.csv')
    >>> to_csv('input.npk', 'output.csv', delimiter=';', fmt='%.6f')
    """
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
        
        shape = npk.get_shape(array_name)
        arr_sample = npk.getitem(array_name, [0])
        dtype = arr_sample.dtype
        estimated_size = int(np.prod(shape)) * dtype.itemsize
        
        if estimated_size > LARGE_FILE_THRESHOLD:
            _to_csv_streaming(npk, output_path, array_name, shape, dtype, 
                            delimiter, header, fmt, chunk_size, **kwargs)
        else:
            arr = npk.load(array_name)
            np.savetxt(str(output_path), arr, delimiter=delimiter, fmt=fmt, 
                      header='' if not header else delimiter.join([f'col{i}' for i in range(arr.shape[1] if arr.ndim > 1 else 1)]),
                      **kwargs)
    finally:
        npk.close()


def _to_csv_streaming(
    npk: Any,
    output_path: Union[str, Path],
    array_name: str,
    shape: Tuple[int, ...],
    dtype: np.dtype,
    delimiter: str,
    header: bool,
    fmt: str,
    chunk_size: int,
    **kwargs
) -> None:
    """流式导出大数组为 CSV 文件"""
    batch_rows = estimate_chunk_rows(shape, dtype, chunk_size)
    total_rows = shape[0]
    
    with open(output_path, 'w') as f:
        # 写入标题
        if header and len(shape) > 1:
            header_line = delimiter.join([f'col{i}' for i in range(shape[1])])
            f.write(f"# {header_line}\n")
        
        for start_idx in range(0, total_rows, batch_rows):
            end_idx = min(start_idx + batch_rows, total_rows)
            indices = list(range(start_idx, end_idx))
            chunk = npk.getitem(array_name, indices)
            
            # 写入分块
            for row in chunk:
                if np.isscalar(row) or row.ndim == 0:
                    line = fmt % row
                else:
                    line = delimiter.join([fmt % val for val in np.atleast_1d(row)])
                f.write(line + '\n')


# =============================================================================
# TXT 格式转换（与 CSV 类似，但默认空格分隔）
# =============================================================================

def from_txt(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    array_name: Optional[str] = None,
    drop_if_exists: bool = False,
    dtype: Optional[np.dtype] = None,
    delimiter: Optional[str] = None,
    skiprows: int = 0,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    **kwargs
) -> None:
    """从 TXT 文件导入为 NumPack 格式
    
    与 from_csv 类似，但默认使用空白字符作为分隔符。
    
    Parameters
    ----------
    input_path : str or Path
        输入的 TXT 文件路径
    output_path : str or Path
        输出的 NumPack 文件路径
    array_name : str, optional
        数组名称，默认使用文件名（不含扩展名）
    drop_if_exists : bool, optional
        如果输出文件存在是否删除，默认 False
    dtype : numpy.dtype, optional
        数据类型，默认自动推断
    delimiter : str, optional
        分隔符，默认 None（任意空白字符）
    skiprows : int, optional
        跳过的行数，默认 0
    chunk_size : int, optional
        分块大小（字节），默认 100MB
    **kwargs
        传递给 numpy.loadtxt 的其他参数
    
    Examples
    --------
    >>> from numpack.io import from_txt
    >>> from_txt('data.txt', 'output.npk')
    """
    # 使用 from_csv 但默认分隔符为空白
    from_csv(
        input_path, output_path, array_name, drop_if_exists,
        dtype, delimiter if delimiter else ' ', skiprows, None, chunk_size,
        **kwargs
    )


def to_txt(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    array_name: Optional[str] = None,
    delimiter: str = ' ',
    fmt: str = '%.18e',
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    **kwargs
) -> None:
    """从 NumPack 导出为 TXT 格式
    
    与 to_csv 类似，但默认使用空格作为分隔符。
    
    Parameters
    ----------
    input_path : str or Path
        输入的 NumPack 文件路径
    output_path : str or Path
        输出的 TXT 文件路径
    array_name : str, optional
        要导出的数组名
    delimiter : str, optional
        分隔符，默认 ' '（空格）
    fmt : str, optional
        数值格式，默认 '%.18e'
    chunk_size : int, optional
        分块大小（字节），默认 100MB
    **kwargs
        传递给 numpy.savetxt 的其他参数
    
    Examples
    --------
    >>> from numpack.io import to_txt
    >>> to_txt('input.npk', 'output.txt')
    """
    to_csv(input_path, output_path, array_name, delimiter, False, fmt, chunk_size, **kwargs)


# =============================================================================
# HDF5 格式转换
# =============================================================================

def from_hdf5(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    dataset_names: Optional[List[str]] = None,
    group: str = '/',
    drop_if_exists: bool = False,
    chunk_size: int = DEFAULT_CHUNK_SIZE
) -> None:
    """从 HDF5 文件导入为 NumPack 格式
    
    对于大数据集（>1GB），自动使用流式处理。
    
    Parameters
    ----------
    input_path : str or Path
        输入的 HDF5 文件路径
    output_path : str or Path
        输出的 NumPack 文件路径
    dataset_names : list of str, optional
        要导入的数据集名列表。如果为 None，导入组内所有数据集。
    group : str, optional
        HDF5 组路径，默认 '/'（根组）
    drop_if_exists : bool, optional
        如果输出文件存在是否删除，默认 False
    chunk_size : int, optional
        分块大小（字节），默认 100MB
    
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
                # 获取组内所有数据集
                dataset_names = [name for name in grp.keys() 
                               if isinstance(grp[name], h5py.Dataset)]
            
            for name in dataset_names:
                dataset = grp[name]
                if not isinstance(dataset, h5py.Dataset):
                    warnings.warn(f"跳过非数据集对象: {name}")
                    continue
                
                shape = dataset.shape
                dtype = dataset.dtype
                estimated_size = int(np.prod(shape)) * dtype.itemsize
                
                if estimated_size > LARGE_FILE_THRESHOLD and len(shape) > 0:
                    # 大数据集：流式读取
                    _from_hdf5_dataset_streaming(npk, dataset, name, chunk_size)
                else:
                    # 小数据集：直接加载
                    arr = dataset[...]
                    npk.save({name: arr})
    finally:
        npk.close()


def _from_hdf5_dataset_streaming(
    npk: Any,
    dataset: Any,  # h5py.Dataset
    array_name: str,
    chunk_size: int
) -> None:
    """流式导入 HDF5 数据集"""
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
    chunk_size: int = DEFAULT_CHUNK_SIZE
) -> None:
    """从 NumPack 导出为 HDF5 格式
    
    对于大数组（>1GB），使用流式读取和分块写入。
    
    Parameters
    ----------
    input_path : str or Path
        输入的 NumPack 文件路径
    output_path : str or Path
        输出的 HDF5 文件路径
    array_names : list of str, optional
        要导出的数组名列表。如果为 None，导出所有数组。
    group : str, optional
        HDF5 组路径，默认 '/'（根组）
    compression : str, optional
        压缩算法，默认 'gzip'。设为 None 禁用压缩。
    compression_opts : int, optional
        压缩级别（0-9），默认 4
    chunk_size : int, optional
        分块大小（字节），默认 100MB
    
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
            # 创建组（如果需要）
            if group != '/':
                grp = h5f.require_group(group)
            else:
                grp = h5f
            
            for name in array_names:
                shape = npk.get_shape(name)
                arr_sample = npk.getitem(name, [0])
                dtype = arr_sample.dtype
                estimated_size = int(np.prod(shape)) * dtype.itemsize
                
                # 计算 HDF5 分块大小
                if len(shape) > 0:
                    batch_rows = estimate_chunk_rows(shape, dtype, chunk_size)
                    chunks = (min(batch_rows, shape[0]),) + shape[1:]
                else:
                    chunks = None
                
                # 创建数据集
                ds = grp.create_dataset(
                    name, 
                    shape=shape, 
                    dtype=dtype,
                    chunks=chunks if chunks else True,
                    compression=compression,
                    compression_opts=compression_opts if compression else None
                )
                
                if estimated_size > LARGE_FILE_THRESHOLD and len(shape) > 0:
                    # 大数组：流式写入
                    _to_hdf5_dataset_streaming(npk, ds, name, shape, dtype, chunk_size)
                else:
                    # 小数组：直接写入
                    ds[...] = npk.load(name)
    finally:
        npk.close()


def _to_hdf5_dataset_streaming(
    npk: Any,
    dataset: Any,  # h5py.Dataset
    array_name: str,
    shape: Tuple[int, ...],
    dtype: np.dtype,
    chunk_size: int
) -> None:
    """流式导出大数组到 HDF5 数据集"""
    batch_rows = estimate_chunk_rows(shape, dtype, chunk_size)
    total_rows = shape[0]
    
    for start_idx in range(0, total_rows, batch_rows):
        end_idx = min(start_idx + batch_rows, total_rows)
        indices = list(range(start_idx, end_idx))
        chunk = npk.getitem(array_name, indices)
        dataset[start_idx:end_idx] = chunk


# =============================================================================
# Zarr 格式转换
# =============================================================================

def from_zarr(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    array_names: Optional[List[str]] = None,
    group: str = '/',
    drop_if_exists: bool = False,
    chunk_size: int = DEFAULT_CHUNK_SIZE
) -> None:
    """从 Zarr 存储导入为 NumPack 格式
    
    Zarr 原生支持分块存储，对于大数据集自动使用流式处理。
    
    Parameters
    ----------
    input_path : str or Path
        输入的 Zarr 存储路径（目录或 .zarr 文件）
    output_path : str or Path
        输出的 NumPack 文件路径
    array_names : list of str, optional
        要导入的数组名列表。如果为 None，导入组内所有数组。
    group : str, optional
        Zarr 组路径，默认 '/'（根组）
    drop_if_exists : bool, optional
        如果输出文件存在是否删除，默认 False
    chunk_size : int, optional
        分块大小（字节），默认 100MB
    
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
            # 获取所有数组
            array_names = [name for name in store.array_keys()]
        
        for name in array_names:
            arr = store[name]
            shape = arr.shape
            dtype = arr.dtype
            estimated_size = int(np.prod(shape)) * dtype.itemsize
            
            if estimated_size > LARGE_FILE_THRESHOLD and len(shape) > 0:
                # 大数组：流式读取
                _from_zarr_array_streaming(npk, arr, name, chunk_size)
            else:
                # 小数组：直接加载
                npk.save({name: arr[...]})
    finally:
        npk.close()


def _from_zarr_array_streaming(
    npk: Any,
    zarr_arr: Any,  # zarr.Array
    array_name: str,
    chunk_size: int
) -> None:
    """流式导入 Zarr 数组"""
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
    chunk_size: int = DEFAULT_CHUNK_SIZE
) -> None:
    """从 NumPack 导出为 Zarr 格式
    
    Zarr 原生支持分块存储，适合大数据集。
    
    Parameters
    ----------
    input_path : str or Path
        输入的 NumPack 文件路径
    output_path : str or Path
        输出的 Zarr 存储路径
    array_names : list of str, optional
        要导出的数组名列表。如果为 None，导出所有数组。
    group : str, optional
        Zarr 组路径，默认 '/'（根组）
    compressor : str or None, optional
        压缩器，默认 'default'（使用 Blosc）。设为 None 禁用压缩。
    chunk_size : int, optional
        分块大小（字节），默认 100MB
    
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
        
        # 创建 Zarr 存储
        store = zarr.open(str(output_path), mode='w')
        if group != '/':
            store = store.require_group(group)
        
        # 配置压缩器
        if compressor == 'default':
            try:
                from zarr.codecs import BloscCodec, BloscCname, BloscShuffle
                compressor_obj = BloscCodec(cname=BloscCname.zstd, clevel=3, shuffle=BloscShuffle.bitshuffle)
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
            
            # 计算分块大小
            if len(shape) > 0:
                batch_rows = estimate_chunk_rows(shape, dtype, chunk_size)
                chunks = (min(batch_rows, shape[0]),) + shape[1:]
            else:
                chunks = shape
            
            # 创建 Zarr 数组
            if hasattr(store, 'create_array'):
                zarr_arr = store.create_array(
                    name,
                    shape=shape,
                    dtype=dtype,
                    chunks=chunks if chunks else None,
                    compressors=compressor_obj,
                    overwrite=True
                )
            else:
                zarr_arr = store.create_dataset(
                    name,
                    shape=shape,
                    dtype=dtype,
                    chunks=chunks if chunks else None,
                    compressor=compressor_obj
                )
            
            if estimated_size > LARGE_FILE_THRESHOLD and len(shape) > 0:
                # 大数组：流式写入
                _to_zarr_array_streaming(npk, zarr_arr, name, shape, dtype, chunk_size)
            else:
                # 小数组：直接写入
                zarr_arr[...] = npk.load(name)
    finally:
        npk.close()


def _to_zarr_array_streaming(
    npk: Any,
    zarr_arr: Any,  # zarr.Array
    array_name: str,
    shape: Tuple[int, ...],
    dtype: np.dtype,
    chunk_size: int
) -> None:
    """流式导出大数组到 Zarr"""
    batch_rows = estimate_chunk_rows(shape, dtype, chunk_size)
    total_rows = shape[0]
    
    for start_idx in range(0, total_rows, batch_rows):
        end_idx = min(start_idx + batch_rows, total_rows)
        indices = list(range(start_idx, end_idx))
        chunk = npk.getitem(array_name, indices)
        zarr_arr[start_idx:end_idx] = chunk


# =============================================================================
# Parquet 格式转换
# =============================================================================

def from_parquet(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    array_name: Optional[str] = None,
    columns: Optional[List[str]] = None,
    drop_if_exists: bool = False,
    chunk_size: int = DEFAULT_CHUNK_SIZE
) -> None:
    """从 Parquet 文件导入为 NumPack 格式
    
    对于大文件（>1GB），自动使用流式处理。
    
    Parameters
    ----------
    input_path : str or Path
        输入的 Parquet 文件路径
    output_path : str or Path
        输出的 NumPack 文件路径
    array_name : str, optional
        数组名称，默认使用文件名（不含扩展名）
    columns : list of str, optional
        要读取的列名列表，默认读取全部
    drop_if_exists : bool, optional
        如果输出文件存在是否删除，默认 False
    chunk_size : int, optional
        分块大小（字节），默认 100MB
    
    Examples
    --------
    >>> from numpack.io import from_parquet
    >>> from_parquet('data.parquet', 'output.npk')
    >>> from_parquet('data.parquet', 'output.npk', columns=['col1', 'col2'])
    """
    pa = _check_pyarrow()
    import pyarrow.parquet as pq
    
    input_path = Path(input_path)
    
    if array_name is None:
        array_name = input_path.stem
    
    # 获取文件元信息
    parquet_file = pq.ParquetFile(str(input_path))
    metadata = parquet_file.metadata
    file_size = get_file_size(input_path)
    
    npk = _open_numpack_for_write(output_path, drop_if_exists)
    
    try:
        if file_size > LARGE_FILE_THRESHOLD:
            # 大文件：按行组流式读取
            _from_parquet_streaming(npk, parquet_file, array_name, columns)
        else:
            # 小文件：直接加载
            table = pq.read_table(str(input_path), columns=columns)
            arr = np.ascontiguousarray(table.to_pandas().values)
            npk.save({array_name: arr})
    finally:
        npk.close()


def _from_parquet_streaming(
    npk: Any,
    parquet_file: Any,  # pyarrow.parquet.ParquetFile
    array_name: str,
    columns: Optional[List[str]]
) -> None:
    """流式导入 Parquet 文件"""
    first_batch = True
    
    for batch in parquet_file.iter_batches(columns=columns):
        arr = np.ascontiguousarray(batch.to_pandas().values)
        
        if first_batch:
            npk.save({array_name: arr})
            first_batch = False
        else:
            npk.append({array_name: arr})


def to_parquet(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    array_name: Optional[str] = None,
    compression: str = 'snappy',
    row_group_size: int = 100000,
    chunk_size: int = DEFAULT_CHUNK_SIZE
) -> None:
    """从 NumPack 导出为 Parquet 格式
    
    对于大数组（>1GB），使用流式读取和分批写入。
    
    Parameters
    ----------
    input_path : str or Path
        输入的 NumPack 文件路径
    output_path : str or Path
        输出的 Parquet 文件路径
    array_name : str, optional
        要导出的数组名
    compression : str, optional
        压缩算法，默认 'snappy'
    row_group_size : int, optional
        行组大小，默认 100000
    chunk_size : int, optional
        分块大小（字节），默认 100MB
    
    Examples
    --------
    >>> from numpack.io import to_parquet
    >>> to_parquet('input.npk', 'output.parquet')
    """
    pa = _check_pyarrow()
    import pyarrow.parquet as pq
    
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
        
        shape = npk.get_shape(array_name)
        arr_sample = npk.getitem(array_name, [0])
        dtype = arr_sample.dtype
        estimated_size = int(np.prod(shape)) * dtype.itemsize
        
        if estimated_size > LARGE_FILE_THRESHOLD and len(shape) > 0:
            # 大数组：流式写入
            _to_parquet_streaming(
                npk, output_path, array_name, shape, dtype, 
                compression, row_group_size, chunk_size
            )
        else:
            # 小数组：直接写入
            arr = npk.load(array_name)
            # 转换为 PyArrow Table
            if arr.ndim == 1:
                table = pa.table({'data': arr})
            else:
                # 多维数组转换为列
                columns = {f'col{i}': arr[:, i] for i in range(arr.shape[1])} if arr.ndim == 2 else {'data': arr.flatten()}
                table = pa.table(columns)
            
            pq.write_table(table, str(output_path), compression=compression, 
                          row_group_size=row_group_size)
    finally:
        npk.close()


def _to_parquet_streaming(
    npk: Any,
    output_path: Union[str, Path],
    array_name: str,
    shape: Tuple[int, ...],
    dtype: np.dtype,
    compression: str,
    row_group_size: int,
    chunk_size: int
) -> None:
    """流式导出大数组到 Parquet"""
    pa = _check_pyarrow()
    import pyarrow.parquet as pq
    
    batch_rows = estimate_chunk_rows(shape, dtype, chunk_size)
    total_rows = shape[0]
    
    writer = None
    
    try:
        for start_idx in range(0, total_rows, batch_rows):
            end_idx = min(start_idx + batch_rows, total_rows)
            indices = list(range(start_idx, end_idx))
            chunk = npk.getitem(array_name, indices)
            
            # 转换为 Table
            if chunk.ndim == 1:
                table = pa.table({'data': chunk})
            else:
                columns = {f'col{i}': chunk[:, i] for i in range(chunk.shape[1])} if chunk.ndim == 2 else {'data': chunk.flatten()}
                table = pa.table(columns)
            
            if writer is None:
                writer = pq.ParquetWriter(
                    str(output_path), 
                    table.schema,
                    compression=compression
                )
            
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()


# =============================================================================
# Feather 格式转换
# =============================================================================

def from_feather(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    array_name: Optional[str] = None,
    columns: Optional[List[str]] = None,
    drop_if_exists: bool = False
) -> None:
    """从 Feather 文件导入为 NumPack 格式
    
    Feather 是一种快速、轻量级的列式存储格式。
    
    Parameters
    ----------
    input_path : str or Path
        输入的 Feather 文件路径
    output_path : str or Path
        输出的 NumPack 文件路径
    array_name : str, optional
        数组名称，默认使用文件名（不含扩展名）
    columns : list of str, optional
        要读取的列名列表
    drop_if_exists : bool, optional
        如果输出文件存在是否删除，默认 False
    
    Examples
    --------
    >>> from numpack.io import from_feather
    >>> from_feather('data.feather', 'output.npk')
    """
    pa = _check_pyarrow()
    import pyarrow.feather as feather
    
    input_path = Path(input_path)
    
    if array_name is None:
        array_name = input_path.stem
    
    # Feather 格式需要一次性加载（不支持流式读取）
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
    chunk_size: int = DEFAULT_CHUNK_SIZE
) -> None:
    """从 NumPack 导出为 Feather 格式
    
    Feather 是一种快速、轻量级的列式存储格式。
    注意：Feather 不支持真正的流式写入，大文件会完全加载到内存。
    
    Parameters
    ----------
    input_path : str or Path
        输入的 NumPack 文件路径
    output_path : str or Path
        输出的 Feather 文件路径
    array_name : str, optional
        要导出的数组名
    compression : str, optional
        压缩算法，默认 'zstd'
    chunk_size : int, optional
        分块大小（字节），默认 100MB
    
    Examples
    --------
    >>> from numpack.io import to_feather
    >>> to_feather('input.npk', 'output.feather')
    """
    pa = _check_pyarrow()
    import pyarrow.feather as feather
    
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
        
        shape = npk.get_shape(array_name)
        estimated_size = int(np.prod(shape)) * npk.getitem(array_name, [0]).dtype.itemsize
        
        if estimated_size > LARGE_FILE_THRESHOLD:
            warnings.warn(
                f"数组 '{array_name}' 较大（>{estimated_size / 1e9:.1f}GB），"
                "Feather 格式需要将所有数据加载到内存。"
                "对于大数据集，建议使用 to_parquet 或 to_zarr。",
                UserWarning
            )
        
        arr = npk.load(array_name)
        
        # 转换为 Table
        if arr.ndim == 1:
            table = pa.table({'data': arr})
        else:
            columns = {f'col{i}': arr[:, i] for i in range(arr.shape[1])} if arr.ndim == 2 else {'data': arr.flatten()}
            table = pa.table(columns)
        
        feather.write_feather(table, str(output_path), compression=compression)
    finally:
        npk.close()


# =============================================================================
# Pandas DataFrame 转换
# =============================================================================

def from_pandas(
    df: "pd.DataFrame",
    output_path: Union[str, Path],
    array_name: str = 'data',
    drop_if_exists: bool = False,
    chunk_size: int = DEFAULT_CHUNK_SIZE
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
    pd = _check_pandas()
    
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
    columns: Optional[List[str]] = None
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


# =============================================================================
# PyTorch Tensor 转换
# =============================================================================

def from_pytorch(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    key: Optional[str] = None,
    drop_if_exists: bool = False,
    chunk_size: int = DEFAULT_CHUNK_SIZE
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


def _save_array_with_streaming_check(
    npk: Any,
    array_name: str,
    arr: np.ndarray,
    chunk_size: int
) -> None:
    """检查数组大小并决定是否使用流式保存"""
    if arr.nbytes > LARGE_FILE_THRESHOLD and arr.ndim > 0:
        _save_array_streaming(npk, array_name, arr, chunk_size)
    else:
        npk.save({array_name: arr})


def to_pytorch(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    array_names: Optional[List[str]] = None,
    as_dict: bool = True
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


# =============================================================================
# S3 远程存储支持
# =============================================================================

def from_s3(
    s3_path: str,
    output_path: Union[str, Path],
    format: str = 'auto',
    drop_if_exists: bool = False,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    **s3_kwargs
) -> None:
    """从 S3 下载文件并导入为 NumPack 格式
    
    支持的格式：npy, npz, csv, parquet, feather, hdf5
    
    Parameters
    ----------
    s3_path : str
        S3 路径，格式为 's3://bucket/path/to/file'
    output_path : str or Path
        输出的 NumPack 文件路径
    format : str, optional
        文件格式，默认 'auto'（从扩展名推断）
    drop_if_exists : bool, optional
        如果输出文件存在是否删除，默认 False
    chunk_size : int, optional
        分块大小（字节），默认 100MB
    **s3_kwargs
        传递给 s3fs 的其他参数（如 anon=True 用于公开桶）
    
    Examples
    --------
    >>> from numpack.io import from_s3
    >>> from_s3('s3://my-bucket/data.npy', 'output.npk')
    >>> from_s3('s3://public-bucket/data.csv', 'output.npk', anon=True)
    """
    s3fs = _check_s3fs()
    import tempfile
    
    # 创建 S3 文件系统
    fs = s3fs.S3FileSystem(**s3_kwargs)
    
    # 推断格式
    if format == 'auto':
        suffix = Path(s3_path).suffix.lower()
        format_map = {
            '.npy': 'numpy',
            '.npz': 'numpy',
            '.csv': 'csv',
            '.txt': 'txt',
            '.parquet': 'parquet',
            '.feather': 'feather',
            '.h5': 'hdf5',
            '.hdf5': 'hdf5',
        }
        format = format_map.get(suffix, 'numpy')
    
    # 下载到临时文件并转换
    with tempfile.NamedTemporaryFile(suffix=Path(s3_path).suffix, delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        # 流式下载
        fs.get(s3_path, tmp_path)
        
        # 根据格式调用相应的导入函数
        format_handlers = {
            'numpy': from_numpy,
            'csv': from_csv,
            'txt': from_txt,
            'parquet': from_parquet,
            'feather': from_feather,
            'hdf5': from_hdf5,
        }
        
        handler = format_handlers.get(format)
        if handler is None:
            raise ValueError(f"不支持的格式: {format}")
        
        handler(tmp_path, output_path, drop_if_exists=drop_if_exists, chunk_size=chunk_size)
    finally:
        # 清理临时文件
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def to_s3(
    input_path: Union[str, Path],
    s3_path: str,
    format: str = 'auto',
    array_name: Optional[str] = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    **s3_kwargs
) -> None:
    """从 NumPack 导出并上传到 S3
    
    支持的格式：npy, npz, csv, parquet, feather, hdf5
    
    Parameters
    ----------
    input_path : str or Path
        输入的 NumPack 文件路径
    s3_path : str
        S3 路径，格式为 's3://bucket/path/to/file'
    format : str, optional
        输出文件格式，默认 'auto'（从扩展名推断）
    array_name : str, optional
        要导出的数组名
    chunk_size : int, optional
        分块大小（字节），默认 100MB
    **s3_kwargs
        传递给 s3fs 的其他参数
    
    Examples
    --------
    >>> from numpack.io import to_s3
    >>> to_s3('input.npk', 's3://my-bucket/output.parquet')
    """
    s3fs = _check_s3fs()
    import tempfile
    
    # 创建 S3 文件系统
    fs = s3fs.S3FileSystem(**s3_kwargs)
    
    # 推断格式
    if format == 'auto':
        suffix = Path(s3_path).suffix.lower()
        format_map = {
            '.npy': 'numpy',
            '.npz': 'numpy',
            '.csv': 'csv',
            '.txt': 'txt',
            '.parquet': 'parquet',
            '.feather': 'feather',
            '.h5': 'hdf5',
            '.hdf5': 'hdf5',
        }
        format = format_map.get(suffix, 'numpy')
    
    # 导出到临时文件
    with tempfile.NamedTemporaryFile(suffix=Path(s3_path).suffix, delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        # 根据格式调用相应的导出函数
        format_handlers = {
            'numpy': lambda inp, out, **kw: to_numpy(inp, out, array_names=[array_name] if array_name else None, chunk_size=chunk_size),
            'csv': lambda inp, out, **kw: to_csv(inp, out, array_name=array_name, chunk_size=chunk_size),
            'txt': lambda inp, out, **kw: to_txt(inp, out, array_name=array_name, chunk_size=chunk_size),
            'parquet': lambda inp, out, **kw: to_parquet(inp, out, array_name=array_name, chunk_size=chunk_size),
            'feather': lambda inp, out, **kw: to_feather(inp, out, array_name=array_name, chunk_size=chunk_size),
            'hdf5': lambda inp, out, **kw: to_hdf5(inp, out, array_names=[array_name] if array_name else None, chunk_size=chunk_size),
        }
        
        handler = format_handlers.get(format)
        if handler is None:
            raise ValueError(f"不支持的格式: {format}")
        
        handler(input_path, tmp_path)
        
        # 上传到 S3
        fs.put(tmp_path, s3_path)
    finally:
        # 清理临时文件
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


# =============================================================================
# 便捷函数
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


# =============================================================================
# 导出公共 API
# =============================================================================

__all__ = [
    # 异常
    'DependencyError',
    
    # 工具函数
    'get_file_size',
    'is_large_file',
    'estimate_chunk_rows',
    
    # NumPy 转换
    'from_numpy',
    'to_numpy',
    
    # CSV/TXT 转换
    'from_csv',
    'to_csv',
    'from_txt',
    'to_txt',
    
    # HDF5 转换
    'from_hdf5',
    'to_hdf5',
    
    # Zarr 转换
    'from_zarr',
    'to_zarr',
    
    # Parquet/Feather 转换
    'from_parquet',
    'to_parquet',
    'from_feather',
    'to_feather',
    
    # Pandas 转换
    'from_pandas',
    'to_pandas',
    
    # PyTorch 转换
    'from_pytorch',
    'to_pytorch',
    
    # S3 支持
    'from_s3',
    'to_s3',
    
    # 常量
    'LARGE_FILE_THRESHOLD',
    'DEFAULT_CHUNK_SIZE',
    'DEFAULT_BATCH_ROWS',
]

from .utils import (
    DEFAULT_BATCH_ROWS,
    DEFAULT_CHUNK_SIZE,
    LARGE_FILE_THRESHOLD,
    DependencyError,
    estimate_chunk_rows,
    get_file_size,
    is_large_file,
)

from .csv_io import from_csv, from_txt, to_csv, to_txt
from .feather_io import from_feather, to_feather
from .hdf5_io import from_hdf5, to_hdf5
from .numpy_io import from_numpy, to_numpy
from .pandas_io import from_pandas, to_pandas
from .parquet_io import from_parquet, to_parquet
from .pytorch_io import from_pytorch, to_pytorch
from .s3_io import from_s3, to_s3
from .zarr_io import from_zarr, to_zarr
