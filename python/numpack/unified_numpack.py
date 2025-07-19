"""
统一 NumPack 实现 - 使用与 Rust 后端完全兼容的文件格式

这个实现确保 Python 后端和 Rust 后端使用完全相同的文件格式，
实现真正的跨平台文件兼容性。
"""

import os
import shutil
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple, Union, Optional
import numpy as np

from .rust_compatible_format import RustCompatibleManager, RustArrayMetadata


class LazyArray:
    """延迟加载数组 - 使用 Rust 兼容格式"""
    
    def __init__(self, manager: RustCompatibleManager, array_name: str):
        self.manager = manager
        self.array_name = array_name
        self._metadata = None
    
    @property
    def metadata(self) -> RustArrayMetadata:
        """获取元数据"""
        if self._metadata is None:
            self._metadata = self.manager.get_metadata(self.array_name)
        return self._metadata
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """数组形状"""
        return self.metadata.shape
    
    @property
    def dtype(self) -> np.dtype:
        """数据类型"""
        return self.metadata.dtype
    
    @property
    def size(self) -> int:
        """元素总数"""
        return self.metadata.total_elements
    
    def to_numpy(self) -> np.ndarray:
        """转换为 numpy 数组"""
        return self.manager.load(self.array_name)
    
    def __getitem__(self, key) -> np.ndarray:
        """支持索引访问"""
        # 加载完整数组然后切片（简化实现）
        data = self.to_numpy()
        return data[key]
    
    def __array__(self) -> np.ndarray:
        """支持 numpy 函数"""
        return self.to_numpy()
    
    def __len__(self) -> int:
        """支持 len()"""
        return self.shape[0] if self.shape else 0
    
    def __repr__(self) -> str:
        return f"LazyArray(name='{self.array_name}', shape={self.shape}, dtype={self.dtype})"


class ArrayMetadata:
    """数组元数据 - 兼容原 API"""
    
    def __init__(self, rust_metadata: RustArrayMetadata):
        self._rust_metadata = rust_metadata
    
    @property
    def name(self) -> str:
        return self._rust_metadata.name
    
    @property
    def shape(self) -> Tuple[int, ...]:
        return self._rust_metadata.shape
    
    @property
    def dtype(self) -> np.dtype:
        return self._rust_metadata.dtype
    
    @property
    def size(self) -> int:
        return self._rust_metadata.total_elements
    
    @property
    def modify_time(self) -> float:
        return self._rust_metadata.timestamp
    
    def __repr__(self) -> str:
        return f"ArrayMetadata(name='{self.name}', shape={self.shape}, dtype={self.dtype})"


class NumPack:
    """NumPack - 使用统一 Rust 兼容格式的实现"""
    
    def __init__(self, filename: Union[str, Path], drop_if_exists: bool = False):
        """初始化 NumPack 对象
        
        Parameters:
            filename (Union[str, Path]): NumPack 文件路径
            drop_if_exists (bool): 如果存在是否删除
        """
        self.filename = Path(filename)
        
        if drop_if_exists and self.filename.exists():
            if self.filename.is_dir():
                shutil.rmtree(self.filename)
            else:
                self.filename.unlink()
        
        # 创建目录
        self.filename.mkdir(parents=True, exist_ok=True)
        
        # 使用 Rust 兼容管理器
        self.manager = RustCompatibleManager(self.filename)
    
    def save(self, arrays: Dict[str, np.ndarray], **kwargs) -> None:
        """保存数组到 NumPack 文件
        
        Parameters:
            arrays (Dict[str, np.ndarray]): 要保存的数组字典
            **kwargs: 兼容参数（被忽略）
        """
        if not isinstance(arrays, dict):
            raise ValueError("arrays must be a dictionary")
        
        # 获取现有数组（增量保存）
        existing_arrays = {}
        try:
            existing_names = self.manager.list_arrays()
            for name in existing_names:
                existing_arrays[name] = self.manager.load(name)
        except:
            # 如果文件不存在或为空，忽略错误
            pass
        
        # 合并新数组和现有数组
        merged_arrays = existing_arrays.copy()
        
        # 转换为 numpy 数组并验证
        for name, array in arrays.items():
            if not isinstance(name, str):
                raise ValueError("Array names must be strings")
            
            if not isinstance(array, np.ndarray):
                array = np.asarray(array)
            
            merged_arrays[name] = array
        
        # 保存合并后的数组
        self.manager.save(merged_arrays)
    
    def load(self, array_name: str, lazy: bool = False) -> Union[np.ndarray, LazyArray]:
        """加载数组
        
        Parameters:
            array_name (str): 数组名称
            lazy (bool): 是否延迟加载
            
        Returns:
            Union[np.ndarray, LazyArray]: 加载的数组或延迟数组
        """
        if not self.manager.has_array(array_name):
            raise KeyError(f"Array '{array_name}' not found")
        
        if lazy:
            return LazyArray(self.manager, array_name)
        else:
            return self.manager.load(array_name)
    
    def list_arrays(self) -> List[str]:
        """获取所有数组名称"""
        return self.manager.list_arrays()
    
    def has_array(self, array_name: str) -> bool:
        """检查数组是否存在"""
        return self.manager.has_array(array_name)
    
    def get_shape(self, array_name: Optional[str] = None) -> Union[Tuple[int, ...], Dict[str, Tuple[int, ...]]]:
        """获取数组形状"""
        if array_name is not None:
            if not self.manager.has_array(array_name):
                raise KeyError(f"Array '{array_name}' not found")
            metadata = self.manager.get_metadata(array_name)
            return metadata.shape
        else:
            # 获取所有数组的形状
            shapes = {}
            for name in self.manager.list_arrays():
                metadata = self.manager.get_metadata(name)
                shapes[name] = metadata.shape
            return shapes
    
    def get_member_list(self) -> List[str]:
        """获取成员列表（别名）"""
        return self.list_arrays()
    
    def get_modify_time(self, array_name: str) -> float:
        """获取修改时间"""
        if not self.manager.has_array(array_name):
            raise KeyError(f"Array '{array_name}' not found")
        metadata = self.manager.get_metadata(array_name)
        return metadata.timestamp
    
    def get_metadata(self) -> Dict[str, Any]:
        """获取完整元数据"""
        metadata = {}
        for name in self.manager.list_arrays():
            rust_metadata = self.manager.get_metadata(name)
            metadata[name] = {
                'shape': rust_metadata.shape,
                'dtype': str(rust_metadata.dtype),
                'size': rust_metadata.total_elements,
                'modify_time': rust_metadata.timestamp,
            }
        return metadata
    
    def get_array_metadata(self, array_name: str) -> ArrayMetadata:
        """获取数组元数据"""
        if not self.manager.has_array(array_name):
            raise KeyError(f"Array '{array_name}' not found")
        rust_metadata = self.manager.get_metadata(array_name)
        return ArrayMetadata(rust_metadata)
    
    def reset(self) -> None:
        """重置（清除所有数组）"""
        self.manager.reset()
    
    def append(self, arrays: Dict[str, np.ndarray]) -> None:
        """追加数据到现有数组"""
        if not isinstance(arrays, dict):
            raise ValueError("arrays must be a dictionary")
        
        for array_name, array in arrays.items():
            if not self.manager.has_array(array_name):
                # 如果数组不存在，直接保存
                self.save({array_name: array})
            else:
                # 加载现有数组并追加
                existing = self.manager.load(array_name)
                appended = np.concatenate([existing, array], axis=0)
                self.save({array_name: appended})
    
    def stream_load(self, array_name: str, buffer_size: int = 1000) -> Iterator[np.ndarray]:
        """流式加载数组"""
        if not self.manager.has_array(array_name):
            raise KeyError(f"Array '{array_name}' not found")
        
        # 加载完整数组然后分批返回（简化实现）
        array = self.manager.load(array_name)
        
        for i in range(0, len(array), buffer_size):
            yield array[i:i + buffer_size]
    
    # 字典式访问接口
    def __getitem__(self, array_name: str) -> np.ndarray:
        """字典式访问"""
        return self.load(array_name, lazy=False)
    
    def __setitem__(self, array_name: str, array: np.ndarray) -> None:
        """字典式赋值"""
        self.save({array_name: array})
    
    def __contains__(self, array_name: str) -> bool:
        """支持 'in' 操作符"""
        return self.has_array(array_name)
    
    def __iter__(self) -> Iterator[str]:
        """迭代数组名称"""
        return iter(self.list_arrays())
    
    def __len__(self) -> int:
        """数组数量"""
        return len(self.list_arrays())
    
    def __repr__(self) -> str:
        return f"NumPack({self.filename}, arrays={len(self)}, format=rust_compatible)"
    
    # 兼容原 API 的别名
    def create_high_performance_lazy_array(self, array_name: str) -> LazyArray:
        """创建高性能延迟数组（兼容别名）"""
        return self.load(array_name, lazy=True)


def force_cleanup_windows_handles():
    """强制清理 Windows 句柄（兼容函数）"""
    # Python 实现不需要特殊的 Windows 句柄清理
    pass


def test_unified_numpack():
    """测试统一 NumPack 实现"""
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_path = Path(tmp_dir) / "unified_test"
        
        # 创建测试数据
        test_arrays = {
            'matrix': np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32),
            'vector': np.array([1.1, 2.2, 3.3], dtype=np.float64),
            'large': np.arange(1000, dtype=np.int64),
        }
        
        # 测试保存和加载
        npk = NumPack(test_path)
        npk.save(test_arrays)
        
        print(f"✅ 保存成功")
        print(f"数组列表: {npk.list_arrays()}")
        
        # 测试读取
        for name, original in test_arrays.items():
            loaded = npk.load(name)
            assert np.array_equal(loaded, original), f"数据不匹配: {name}"
            print(f"✅ 验证通过: {name}")
        
        # 测试延迟加载
        lazy = npk.load('matrix', lazy=True)
        print(f"✅ 延迟加载: {lazy}")
        lazy_data = lazy.to_numpy()
        assert np.array_equal(lazy_data, test_arrays['matrix'])
        print(f"✅ 延迟加载数据正确")
        
        # 测试增量保存
        new_data = {'extra': np.array([100, 200, 300])}
        npk.save(new_data)
        all_arrays = npk.list_arrays()
        print(f"✅ 增量保存后数组: {all_arrays}")
        assert 'matrix' in all_arrays and 'extra' in all_arrays
        
        # 测试字典访问
        matrix_data = npk['matrix']
        assert np.array_equal(matrix_data, test_arrays['matrix'])
        print(f"✅ 字典访问正确")
        
        print(f"✅ 所有测试通过！使用统一 Rust 兼容格式")


if __name__ == "__main__":
    test_unified_numpack() 