from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional
import numpy as np

from numpack._lib_numpack import NumPack as _NumPack


class _LazyArrayDict:
    def __init__(self, npk: _NumPack, mmap_mode: bool):
        self.npk = npk
        self.mmap_mode = mmap_mode

    def keys(self) -> List[str]:
        return self.npk.get_member_list()
    
    def __getitem__(self, key: str) -> np.ndarray:
        if key not in self.npk.get_member_list():
            raise KeyError(f"Key {key} not found in NumPack")
        
        return self.npk.load([key], self.mmap_mode)[key]
    
    def __keys__(self) -> List[str]:
        return self.npk.get_member_list()
    
    def __len__(self) -> int:
        return len(self.npk.get_member_list())
    
    def __contains__(self, key: str) -> bool:
        return key in self.npk.get_member_list()
    

class NumPack:
    def __init__(self, filename: Union[str, Path]):
        self._npk = _NumPack(filename)

    def save(self, arrays: Dict[str, np.ndarray], array_names: Optional[Union[List[str], str]] = None) -> None:
        """保存数组到 .npk 文件
    
        Args:
            arrays: 要保存的数组字典
            array_names: 要保存的数组名称，如果为 None，则使用字典中的键名
        """
        if not isinstance(arrays, dict):
            raise ValueError("arrays must be a dictionary")
            
        self._npk.save(arrays, array_names)

    def load(self, mmap_mode: bool = False) -> Dict[str, np.ndarray]:
        """从 .npk 文件加载数组
    
        Args:
            mmap_mode: 是否使用内存映射模式
    
        Returns:
            一个 LazyArrayDict 对象，可以按需加载数组
        """
        return _LazyArrayDict(self._npk, mmap_mode)

    def replace(self, arrays: Dict[str, np.ndarray], indexes: Union[List[int], int, np.ndarray]) -> None:
        """替换 .npk 文件中的数组
    
        Args:
            arrays: 要替换的数组字典
            indexes: 要替换的索引
        """
        if not isinstance(arrays, dict):
            raise ValueError("arrays must be a dictionary")
            
        self._npk.replace(arrays, indexes)

    def append(self, arrays: Dict[str, np.ndarray]) -> None:
        """追加数组到 .npk 文件
    
        Args:
            arrays: 要追加的数组字典
        """
        if not isinstance(arrays, dict):
            raise ValueError("arrays must be a dictionary")
        
        # 将字典转换为元组列表
        array_tuples = [(name, array) for name, array in arrays.items()]
        self._npk.append(array_tuples)

    def drop(self, array_names: Optional[Union[List[str], str]] = None, indexes: Optional[Union[List[int], int, np.ndarray]] = None) -> None:
        """从 .npk 文件中删除指定数组
    
        Args:
            array_names: 要删除的数组名称，如果为 None，则删除所有数组
            indexes: 要删除的行索引，如果为 None，则删除所有行
        """
        if array_names is not None and isinstance(array_names, str):
            array_names = [array_names]
            
        self._npk.drop(array_names, indexes)


    def getitem(self, indexes: Union[List[int], int, np.ndarray], array_names: Optional[Union[List[str], str]] = None) -> Dict[str, np.ndarray]:
        """从 .npk 文件中随机访问指定行的数据
    
        Args:
            indexes: 要访问的行索引，可以是整数、列表、切片或numpy数组
            array_names: 要访问的数组名称，如果为 None，则访问所有数组
    
        Returns:
            包含指定行数据的字典
        """
        raise NotImplementedError("getitem is not implemented")
    
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
