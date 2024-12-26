"""
NumPack - A simple package for saving and loading NumPy arrays
"""
from pathlib import Path
from typing import Dict, List, Union, Optional
import numpy as np

from numpack._lib_numpack import (
    save_nnp as _save_nnp,
    load_nnp as _load_nnp,
    replace_arrays as _replace_arrays,
    append_arrays as _append_arrays,
    drop_arrays as _drop_arrays,
    getitem as _getitem
)

def save_nnp(filename: Union[str, Path], arrays: Dict[str, np.ndarray], array_name: Optional[str] = None) -> None:
    """保存数组到 .nnp 文件
    
    Args:
        filename: 文件路径
        arrays: 要保存的数组字典
        array_name: 数组名称前缀，如果为 None，则使用 'array0', 'array1', 'array2', ...
    """
    if not isinstance(arrays, dict):
        raise ValueError("arrays must be a dictionary")
        
    _save_nnp(str(filename), arrays, array_name)

def load_nnp(filename: Union[str, Path], array_names: Optional[Union[List[str], str]] = None, mmap_mode: bool = False) -> Dict[str, np.ndarray]:
    """从 .nnp 文件加载数组
    
    Args:
        filename: 文件路径
        array_names: 要加载的数组名称，如果为 None，则加载所有数组
        mmap_mode: 是否使用内存映射模式
    
    Returns:
        包含加载的数组的字典
    """
    if array_names is not None:
        if isinstance(array_names, str):
            array_names = [array_names]
            
    return _load_nnp(str(filename), array_names, mmap_mode)

def replace_arrays(filename: Union[str, Path], arrays: Dict[str, np.ndarray], indexes: Union[List[int], int, slice, np.ndarray], array_names: Optional[Union[List[str], str]] = None) -> None:
    """替换 .nnp 文件中的数组
    
    Args:
        filename: 文件路径
        arrays: 要替换的数组字典
        indexes: 要替换的索引
        array_names: 要替换的数组名称，如果为 None，则替换所有数组
    """
    if not isinstance(arrays, dict):
        raise ValueError("arrays must be a dictionary")
        
    if array_names is not None:
        if isinstance(array_names, str):
            array_names = [array_names]
            
    _replace_arrays(str(filename), arrays, indexes, array_names)

def drop_arrays(filename: Union[str, Path], indexes: Union[List[int], int, slice, np.ndarray], array_names: Optional[Union[List[str], str]] = None) -> None:
    """从 .nnp 文件中删除数组
    
    Args:
        filename: 文件路径
        indexes: 要删除的索引
        array_names: 要删除的数组名称，如果为 None，则删除所有数组
    """
    if array_names is not None:
        if isinstance(array_names, str):
            array_names = [array_names]
            
    _drop_arrays(str(filename), indexes, array_names)

def append_arrays(filename: Union[str, Path], arrays: Dict[str, np.ndarray], array_names: Optional[Union[List[str], str]] = None) -> None:
    """追加数组�� .nnp 文件
    
    Args:
        filename: 文件路径
        arrays: 要追加的数组字典
        array_names: 要追加的数组名称，如果为 None，则追加所有数组
    """
    if not isinstance(arrays, dict):
        raise ValueError("arrays must be a dictionary")
        
    if array_names is not None:
        if isinstance(array_names, str):
            array_names = [array_names]
            
    _append_arrays(str(filename), arrays, array_names)

def getitem(filename: Union[str, Path], indexes: Union[List[int], int, slice, np.ndarray], array_names: Optional[Union[List[str], str]] = None) -> Dict[str, np.ndarray]:
    """从 .nnp 文件中随机访问指定行的数据
    
    Args:
        filename: 文件路径
        indexes: 要访问的行索引，可以是整数、列表、切片或numpy数组
        array_names: 要访问的数组名称，如果为 None，则访问所有数组
    
    Returns:
        包含指定行数据的字典
    """
    if array_names is not None:
        if isinstance(array_names, str):
            array_names = [array_names]
            
    return _getitem(str(filename), indexes, array_names)

__version__ = "0.1.0"

