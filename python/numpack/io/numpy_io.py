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
# NumPy 格式转换 (npy/npz)
# =============================================================================

def from_numpy(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    array_name: Optional[str] = None,
    drop_if_exists: bool = False,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
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
    chunk_size: int,
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
    chunk_size: int,
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


def to_numpy(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    array_names: Optional[List[str]] = None,
    compressed: bool = True,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
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
    chunk_size: int,
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
    chunk_size: int,
) -> None:
    """流式导出大数组为 .npy 文件"""
    batch_rows = estimate_chunk_rows(shape, dtype, chunk_size)
    total_rows = shape[0]

    # 创建输出文件（预分配空间）
    # 使用 numpy 的 format 模块创建正确的 npy 文件头
    from numpy.lib import format as npy_format

    with open(output_path, 'wb') as f:
        # 写入 npy 文件头
        npy_format.write_array_header_1_0(
            f, {'descr': dtype.str, 'fortran_order': False, 'shape': shape}
        )
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
                UserWarning,
            )

        arrays[name] = npk.load(name)

    if compressed:
        np.savez_compressed(str(output_path), **arrays)
    else:
        np.savez(str(output_path), **arrays)
