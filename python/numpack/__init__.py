import shutil
import os
import platform
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple, Union, Optional
import numpy as np

__version__ = "0.2.1"

# 平台检测
def _is_windows():
    """Detect if running on Windows platform"""
    return platform.system().lower() == 'windows'

def _should_use_python_backend():
    """Decide whether to use Python backend"""
    # Windows platform forces use of Python backend
    if _is_windows():
        return True
    
    # Allow forcing Python backend via environment variable (for testing)
    if os.environ.get('NUMPACK_FORCE_PYTHON_BACKEND', '').lower() in ['1', 'true', 'yes']:
        return True
    
    return False

# 后端选择和导入
_USE_PYTHON_BACKEND = _should_use_python_backend()

if _USE_PYTHON_BACKEND:
    # Windows platform or forced: unified format Python backend
    from .unified_numpack import NumPack as _NumPack, LazyArray
    _BACKEND_TYPE = "python"
else:
    # Unix/Linux platform: try to use Rust backend
    try:
        import numpack._lib_numpack as rust_backend
        _NumPack = rust_backend.NumPack
        LazyArray = rust_backend.LazyArray
        _BACKEND_TYPE = "rust"
    except ImportError as e:
        print(f"Warning: Cannot import Rust backend, falling back to unified format Python backend: {e}")
        from .unified_numpack import NumPack as _NumPack, LazyArray
        _BACKEND_TYPE = "python"


class NumPack:
    """NumPack - High-performance array storage library (hybrid architecture)
    
    - Windows platform: Use pure Python implementation for compatibility
    - Unix/Linux platform: Use Rust backend for maximum performance, fallback to Python backend on failure
    """
    
    def __init__(self, filename: Union[str, Path], drop_if_exists: bool = False):
        """Initialize NumPack object
        
        Parameters:
            filename (Union[str, Path]): The name of the NumPack file
            drop_if_exists (bool): Whether to drop the file if it already exists
        """
        self._backend_type = _BACKEND_TYPE
        
        if drop_if_exists and Path(filename).exists():
            if Path(filename).is_dir():
                shutil.rmtree(filename)
            else:
                Path(filename).unlink()

        Path(filename).mkdir(parents=True, exist_ok=True)
        
        if self._backend_type == "python":
            # Python 后端支持 drop_if_exists 参数
            self._npk = _NumPack(filename, drop_if_exists)
        else:
            # Rust 后端只接受一个参数
            self._npk = _NumPack(str(filename))

    def save(self, arrays: Dict[str, np.ndarray]) -> None:
        """Save arrays to NumPack file
        
        Parameters:
            arrays (Dict[str, np.ndarray]): The arrays to save
        """
        if not isinstance(arrays, dict):
            raise ValueError("arrays must be a dictionary")
            
        if self._backend_type == "python":
            self._npk.save(arrays)
        else:
            # Rust 后端需要额外的参数
            self._npk.save(arrays, None)

    def load(self, array_name: str, lazy: bool = False) -> Union[np.ndarray, LazyArray]:
        """Load arrays from NumPack file
        
        Parameters:
            array_name (str): The name of the array to load
            lazy (bool): Whether to load the array in lazy mode (memory mapped)
        
        Returns:
            Union[np.ndarray, LazyArray]: The loaded array
        """
        return self._npk.load(array_name, lazy=lazy)

    def replace(self, arrays: Dict[str, np.ndarray], indexes: Union[List[int], int, np.ndarray, slice]) -> None:
        """Replace arrays in NumPack file
        
        Parameters:
            arrays (Dict[str, np.ndarray]): The arrays to replace
            indexes (Union[List[int], int, np.ndarray, slice]): The indexes to replace
        """
        if not isinstance(arrays, dict):
            raise ValueError("arrays must be a dictionary")
        
        if isinstance(indexes, int):
            indexes = [indexes]
        elif isinstance(indexes, np.ndarray):
            indexes = indexes.tolist()
        elif not isinstance(indexes, (list, slice)):
            raise ValueError("The indexes must be int or list or numpy.ndarray or slice.")
            
        if self._backend_type == "python":
            # Python 后端优化的 replace 功能
            self._npk.replace(arrays, indexes)
        else:
            # Rust 后端
            self._npk.replace(arrays, indexes)

    def append(self, arrays: Dict[str, np.ndarray]) -> None:
        """Append arrays to NumPack file
        
        Parameters:
            arrays (Dict[str, np.ndarray]): The arrays to append
        """
        if not isinstance(arrays, dict):
            raise ValueError("arrays must be a dictionary")
        
        # 两个后端现在都期望字典参数
        self._npk.append(arrays)

    def drop(self, array_name: Union[str, List[str]], indexes: Optional[Union[List[int], int, np.ndarray]] = None) -> None:
        """Drop arrays from NumPack file
        
        Parameters:
            array_name (Union[str, List[str]]): The name or names of the arrays to drop
            indexes (Optional[Union[List[int], int, np.ndarray]]): The indexes to drop, if None, drop all rows
        """
        if isinstance(array_name, str):
            array_name = [array_name]
            
        if self._backend_type == "python":
            # Python 后端使用新实现的 drop 方法
            self._npk.drop(array_name, indexes)
        else:
            # Rust 后端
            self._npk.drop(array_name, indexes)

    def getitem(self, array_name: str, indexes: Union[List[int], int, np.ndarray]) -> np.ndarray:
        """Randomly access the data of specified rows from NumPack file
        
        Parameters:
            array_name (str): The name of the array to access
            indexes (Union[List[int], int, np.ndarray]): The indexes to access, can be integers, lists, slices or numpy arrays

        Returns:
            The specified row data
        """
        if isinstance(indexes, int):
            indexes = [indexes]
        elif isinstance(indexes, np.ndarray):
            indexes = indexes.tolist()
        
        if self._backend_type == "python":
            # Python 后端使用优化的随机访问实现
            return self._npk.getitem_optimized(array_name, indexes)
        else:
            # Rust 后端
            return self._npk.getitem(array_name, indexes)
    
    def get_shape(self, array_name: str) -> Tuple[int, int]:
        """Get the shape of specified arrays in NumPack file
        
        Parameters:
            array_name (str): The name of the array to get the shape
        
        Returns:
            tuple: the shape of the array
        """
        if self._backend_type == "python":
            return self._npk.get_shape(array_name)
        else:
            return self._npk.get_shape(array_name)
    
    def get_member_list(self) -> List[str]:
        """Get the list of array names in NumPack file
        
        Returns:
            A list containing the names of the arrays
        """
        if self._backend_type == "python":
            return self._npk.list_arrays()
        else:
            return self._npk.get_member_list()
    
    def get_modify_time(self, array_name: str) -> Optional[int]:
        """Get the modify time of specified array in NumPack file
        
        Parameters:
            array_name (str): The name of the array to get the modify time
        
        Returns:
            The modify time of the array, if the array does not exist, return None
        """
        if self._backend_type == "python":
            return self._npk.get_modify_time(array_name)
        else:
            return self._npk.get_modify_time(array_name)
    
    def reset(self) -> None:
        """Clear all arrays in NumPack file"""
        self._npk.reset()

    def get_metadata(self) -> Dict[str, Any]:
        """Get the metadata of NumPack file"""
        return self._npk.get_metadata()
    
    def __getitem__(self, key: str) -> np.ndarray:
        """Get the array by key"""
        return self.load(key)
    
    def __iter__(self):
        """Iterate over the arrays in NumPack file"""
        return iter(self.get_member_list())
    
    def stream_load(self, array_name: str, buffer_size: Union[int, None] = None) -> Iterator[np.ndarray]:
        """Stream the array by name with buffering support
        
        Parameters:
            array_name (str): The name of the array to stream
            buffer_size (Union[int, None]): Number of rows to load in each batch, if None, load all rows one by one
        
        Returns:
            Iterator yielding numpy arrays of size up to buffer_size
        """
        if buffer_size is not None and buffer_size <= 0:
            raise ValueError("buffer_size must be greater than 0")
        
        if self._backend_type == "python":
            # Python 后端的实现
            # 当buffer_size为None时，逐行返回（buffer_size=1）
            effective_buffer_size = buffer_size if buffer_size is not None else 1
            return self._npk.stream_load(array_name, effective_buffer_size)
        else:
            # Rust 后端：使用新实现的stream_load方法
            effective_buffer_size = buffer_size if buffer_size is not None else 1
            return self._npk.stream_load(array_name, effective_buffer_size)

    def has_array(self, array_name: str) -> bool:
        """Check if array exists
        
        Parameters:
            array_name (str): Name of the array
            
        Returns:
            bool: True if array exists
        """
        if self._backend_type == "python":
            return self._npk.has_array(array_name)
        else:
            return array_name in self._npk.get_member_list()

    @property 
    def backend_type(self) -> str:
        """获取当前使用的后端类型"""
        return self._backend_type
        
    def get_io_stats(self) -> Dict[str, Any]:
        """获取IO性能统计信息 - 内部监控功能
        
        Returns:
            Dict[str, Any]: 性能统计数据
        """
        # 如果是Python后端，直接从Python实例获取
        if self._backend_type == "python" and hasattr(self._npk, "get_io_performance_stats"):
            return self._npk.get_io_performance_stats()
        # 否则返回基本信息
        return {
            "backend_type": self._backend_type,
            "stats_available": False
        }

    def close(self) -> None:
        """显式关闭NumPack实例，释放所有资源"""
        try:
            if hasattr(self._npk, 'close'):
                self._npk.close()
            elif hasattr(self._npk, '_cleanup_windows_handles'):
                self._npk._cleanup_windows_handles()
        except Exception:
            pass
    
    def __del__(self):
        """析构函数"""
        self.close()
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.close()

    def __repr__(self) -> str:
        backend_info = f"backend={self._backend_type}"
        # 尝试获取文件名
        filename = 'unknown'
        if hasattr(self._npk, 'filename'):
            filename = self._npk.filename
        elif hasattr(self._npk, '_filename'):
            filename = self._npk._filename
        elif hasattr(self._npk, 'base_dir'):
            filename = self._npk.base_dir
        
        arrays_count = len(self.get_member_list())
        return f"NumPack({filename}, arrays={arrays_count}, {backend_info})"


# LazyArray类 - 导出到模块级别
# （LazyArray的实际实现来自后端模块）

# 提供向后兼容的空函数
def force_cleanup_windows_handles():
    """强制清理Windows句柄 - 使用统一句柄管理器"""
    try:
        # 尝试使用新的句柄管理器
        from .windows_handle_manager import force_cleanup_windows_handles as _force_cleanup
        return _force_cleanup()
    except ImportError:
        # 如果句柄管理器不可用，使用传统方法
        import gc
        import time
        import os
        
        gc.collect()
        if os.name == 'nt':
            time.sleep(0.1)
            gc.collect()
        return True

# 导出的公共API
__all__ = ['NumPack', 'LazyArray', 'force_cleanup_windows_handles']

# 提供后端信息查询
def get_backend_info():
    """Get current backend information"""
    return {
        'backend_type': _BACKEND_TYPE,
        'platform': platform.system(),
        'is_windows': _is_windows(),
        'use_python_backend': _USE_PYTHON_BACKEND,
        'version': __version__
    }