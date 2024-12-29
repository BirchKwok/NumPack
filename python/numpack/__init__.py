"""
NumPack - A simple package for saving and loading NumPy arrays
"""
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional
import numpy as np

from numpack._lib_numpack import NumPack as _NumPack

class NumPack:
    def __init__(self, filename: Union[str, Path]):
        self._npk = _NumPack(filename)

    def save_arrays(self, arrays: Dict[str, np.ndarray], array_names: Optional[Union[List[str], str]] = None) -> None:
        """保存数组到 .npk 文件
    
        Args:
            arrays: 要保存的数组字典
            array_names: 要保存的数组名称，如果为 None，则使用字典中的键名
        """
        if not isinstance(arrays, dict):
            raise ValueError("arrays must be a dictionary")
            
        self._npk.save_arrays(arrays, array_names)

    def load_arrays(self, array_names: Optional[Union[List[str], str]] = None, mmap_mode: bool = False) -> Dict[str, np.ndarray]:
        """从 .npk 文件加载数组
    
        Args:
            array_names: 要加载的数组名称，如果为 None，则加载所有数组
            mmap_mode: 是否使用内存映射模式
    
        Returns:
            包含加载的数组的字典
        """
        if array_names is not None and isinstance(array_names, str):
            array_names = [array_names]
        
        return self._npk.load_arrays(array_names, mmap_mode)

    def replace_arrays(self, arrays: Dict[str, np.ndarray], indexes: Union[List[int], int, slice, np.ndarray], array_names: Optional[Union[List[str], str]] = None) -> None:
        """替换 .npk 文件中的数组
    
        Args:
            arrays: 要替换的数组字典
            indexes: 要替换的索引
            array_names: 要替换的数组名称，如果为 None，则替换所有数组
        """
        if not isinstance(arrays, dict):
            raise ValueError("arrays must be a dictionary")
        
        if array_names is not None and isinstance(array_names, str):
            array_names = [array_names]
            
        self._npk.replace_arrays(arrays, indexes, array_names)

    def append_arrays(self, arrays: Dict[str, np.ndarray], array_names: Optional[Union[List[str], str]] = None) -> None:
        """追加数组到 .npk 文件
    
        Args:
            arrays: 要追加的数组字典
            array_names: 要追加的数组名称，如果为 None，则追加所有数组
        """
        if not isinstance(arrays, dict):
            raise ValueError("arrays must be a dictionary")
        
        if array_names is not None and isinstance(array_names, str):
            array_names = [array_names]
            
        self._npk.append_arrays(arrays, array_names)

    def drop_arrays(self, array_names: Optional[Union[List[str], str]] = None) -> None:
        """从 .npk 文件中删除指定数组
    
        Args:
            array_names: 要删除的数组名称，如果为 None，则删除所有数组
        """
        if array_names is not None and isinstance(array_names, str):
            array_names = [array_names]
            
        self._npk.drop_arrays(array_names)


    def getitem(self, indexes: Union[List[int], int, slice, np.ndarray], array_names: Optional[Union[List[str], str]] = None) -> Dict[str, np.ndarray]:
        """从 .npk 文件中随机访问指定行的数据
    
        Args:
            indexes: 要访问的行索引，可以是整数、列表、切片或numpy数组
            array_names: 要访问的数组名称，如果为 None，则访问所有数组
    
        Returns:
            包含指定行数据的字典
        """
        if array_names is not None and isinstance(array_names, str):
            array_names = [array_names]
            
        return self._npk.getitem(indexes, array_names)
    
    def get_shape(self, array_names: Optional[Union[List[str], str]] = None) -> Dict[str, Tuple[int, int]]:
        """获取 .npk 文件中指定数组的形状
    
        Args:
            array_names: 要获取形状的数组名称，如果为 None，则获取所有数组的形状
    
        Returns:
            包含数组形状的字典
        """
        return self._npk.get_shape(array_names)
    
    def get_member_list(self) -> List[str]:
        """获取 .npk 文件中的数组名称列表
    
        Returns:
            包含数组名称的列表
        """
        return self._npk.get_member_list()
    
    def get_modify_time(self, array_name: str) -> Optional[int]:
        """获取 .npk 文件中指定数组的修改时间
    
        Args:
            array_name: 要获取修改时间的数组名称
    
        Returns:
            数组的修改时间，如果数组不存在则返回 None
        """
        return self._npk.get_modify_time(array_name)
    
    def reset(self) -> None:
        """清空 .npk 文件中的所有数组"""
        self._npk.reset()
    
__version__ = "0.1.0"
