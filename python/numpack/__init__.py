import shutil
import os
import platform
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple, Union, Optional
import numpy as np

__version__ = "0.4.0"

# 平台检测
def _is_windows():
    """Detect if running on Windows platform"""
    return platform.system().lower() == 'windows'

# 后端选择和导入 - 始终使用Rust后端以获得最高性能
try:
    import numpack._lib_numpack as rust_backend
    _NumPack = rust_backend.NumPack
    LazyArray = rust_backend.LazyArray
    _BACKEND_TYPE = "rust"
except ImportError as e:
    raise ImportError(
        f"无法导入Rust后端: {e}\n"
        "NumPack现在只使用高性能的Rust后端。请确保:\n"
        "1. 已正确编译和安装Rust扩展\n"
        "2. 使用 python build.py 重新构建项目"
    )


class NumPack:
    """NumPack - 高性能数组存储库 (仅使用Rust后端)
    
    使用高性能的Rust后端实现,在所有平台上提供一致的最佳性能。
    """
    
    def __init__(
        self, 
        filename: Union[str, Path], 
        drop_if_exists: bool = False,
        strict_context_mode: bool = False,
        warn_no_context: bool = None,
        force_gc_on_close: bool = False
    ):
        """Initialize NumPack object
        
        文件不会自动打开。用户必须：
        1. 手动调用 open() 方法
        2. 使用 context manager (with 语句)
        
        Parameters:
            filename (Union[str, Path]): The name of the NumPack file
            drop_if_exists (bool): Whether to drop the file if it already exists
            strict_context_mode (bool): If True, requires usage within 'with' statement
            warn_no_context (bool): If True, warns when not using context manager
            force_gc_on_close (bool): 是否在close时强制垃圾回收。默认False以获得最佳性能。
                                    仅在应用有严格内存限制时设置为True。
        """
        self._backend_type = _BACKEND_TYPE  # 始终为 "rust"
        self._strict_context_mode = strict_context_mode
        self._context_entered = False
        self._closed = False
        self._opened = False
        self._filename = Path(filename)
        self._drop_if_exists = drop_if_exists
        self._force_gc_on_close = force_gc_on_close
        
        # Determine warning behavior
        if warn_no_context is None:
            warn_no_context = _is_windows()
        self._warn_no_context = warn_no_context
        
        # Issue warning if not in strict mode and warn_no_context is True
        if not strict_context_mode and warn_no_context:
            import warnings
            warnings.warn(
                f"NumPack instance created for '{filename}' is not using strict context mode. "
                "For best reliability on Windows, please use 'with NumPack(...) as npk:' pattern "
                "or set strict_context_mode=True. "
                "This warning can be suppressed by setting warn_no_context=False.",
                UserWarning,
                stacklevel=2
            )
        
        # 初始化后端实例为None - 不自动打开
        # 用户必须显式调用 open() 或使用 context manager
        self._npk = None
    
    def open(self) -> None:
        """手动打开NumPack文件
        
        如果文件已打开，此方法将不执行任何操作。
        如果文件已关闭，此方法将重新打开文件。
        
        示例:
            ```python
            npk = NumPack('data.npk')
            npk.open()  # 手动打开
            npk.save({'array': data})
            npk.close()  # 手动关闭
            npk.open()  # 重新打开
            data = npk.load('array')
            npk.close()
            ```
        """
        if self._opened and not self._closed:
            # 文件已打开且未关闭，不需要操作
            return
        
        # 处理文件删除（如果需要）
        if self._drop_if_exists and self._filename.exists():
            if self._filename.is_dir():
                shutil.rmtree(self._filename)
            else:
                self._filename.unlink()
        
        # 创建目录
        self._filename.mkdir(parents=True, exist_ok=True)
        
        # 初始化Rust后端 (只接受一个参数)
        self._npk = _NumPack(str(self._filename))
        
        # 更新状态
        self._opened = True
        self._closed = False
        
        # 第一次打开后，不再自动删除文件
        self._drop_if_exists = False
    
    def _check_context_mode(self):
        """Verify context manager usage (if in strict mode)"""
        if not self._opened or self._closed:
            raise RuntimeError(
                f"NumPack instance '{self._filename}' is not opened or has been closed. "
                "Please call open() method first, or use 'with' statement for automatic management."
            )
        
        if self._strict_context_mode and not self._context_entered:
            raise RuntimeError(
                f"NumPack instance '{self._filename}' is in strict context mode. "
                "All operations must be executed within a 'with' statement:\n"
                "  with NumPack(...) as npk:\n"
                "      npk.save(...)\n"
                "      npk.load(...)"
            )

    def save(self, arrays: Dict[str, np.ndarray]) -> None:
        """Save arrays to NumPack file
        
        Parameters:
            arrays (Dict[str, np.ndarray]): The arrays to save
        """
        self._check_context_mode()
        
        if not isinstance(arrays, dict):
            raise ValueError("arrays must be a dictionary")
            
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
        self._check_context_mode()
        return self._npk.load(array_name, lazy=lazy)

    def replace(self, arrays: Dict[str, np.ndarray], indexes: Union[List[int], int, np.ndarray, slice]) -> None:
        """Replace arrays in NumPack file
        
        Parameters:
            arrays (Dict[str, np.ndarray]): The arrays to replace
            indexes (Union[List[int], int, np.ndarray, slice]): The indexes to replace
        """
        self._check_context_mode()
        
        if not isinstance(arrays, dict):
            raise ValueError("arrays must be a dictionary")
        
        if isinstance(indexes, int):
            indexes = [indexes]
        elif isinstance(indexes, np.ndarray):
            indexes = indexes.tolist()
        elif not isinstance(indexes, (list, slice)):
            raise ValueError("The indexes must be int or list or numpy.ndarray or slice.")
            
        # Rust 后端
        self._npk.replace(arrays, indexes)

    def append(self, arrays: Dict[str, np.ndarray]) -> None:
        """Append arrays to NumPack file
        
        Parameters:
            arrays (Dict[str, np.ndarray]): The arrays to append
        """
        self._check_context_mode()
        
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
        self._check_context_mode()
        
        if isinstance(array_name, str):
            array_name = [array_name]
            
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
        self._check_context_mode()
        
        if isinstance(indexes, int):
            indexes = [indexes]
        elif isinstance(indexes, np.ndarray):
            indexes = indexes.tolist()
        
        # Rust 后端
        return self._npk.getitem(array_name, indexes)
    
    def get_shape(self, array_name: str) -> Tuple[int, int]:
        """Get the shape of specified arrays in NumPack file
        
        Parameters:
            array_name (str): The name of the array to get the shape
        
        Returns:
            tuple: the shape of the array
        """
        self._check_context_mode()
        return self._npk.get_shape(array_name)
    
    def get_member_list(self) -> List[str]:
        """Get the list of array names in NumPack file
        
        Returns:
            A list containing the names of the arrays
        """
        self._check_context_mode()
        return self._npk.get_member_list()
    
    def get_modify_time(self, array_name: str) -> Optional[int]:
        """Get the modify time of specified array in NumPack file
        
        Parameters:
            array_name (str): The name of the array to get the modify time
        
        Returns:
            The modify time of the array, if the array does not exist, return None
        """
        self._check_context_mode()
        return self._npk.get_modify_time(array_name)
    
    def reset(self) -> None:
        """Clear all arrays in NumPack file"""
        self._check_context_mode()
        self._npk.reset()

    def get_metadata(self) -> Dict[str, Any]:
        """Get the metadata of NumPack file"""
        self._check_context_mode()
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
        self._check_context_mode()
        
        if buffer_size is not None and buffer_size <= 0:
            raise ValueError("buffer_size must be greater than 0")
        
        # Rust 后端：使用stream_load方法
        effective_buffer_size = buffer_size if buffer_size is not None else 1
        return self._npk.stream_load(array_name, effective_buffer_size)

    def has_array(self, array_name: str) -> bool:
        """Check if array exists
        
        Parameters:
            array_name (str): Name of the array
            
        Returns:
            bool: True if array exists
        """
        self._check_context_mode()
        return array_name in self._npk.get_member_list()

    @property 
    def backend_type(self) -> str:
        """获取当前使用的后端类型"""
        return self._backend_type
    
    @property
    def is_opened(self) -> bool:
        """检查文件是否已打开"""
        return self._opened and not self._closed
    
    @property
    def is_closed(self) -> bool:
        """检查文件是否已关闭"""
        return self._closed or not self._opened
        
    def get_io_stats(self) -> Dict[str, Any]:
        """获取IO性能统计信息 - 内部监控功能
        
        Returns:
            Dict[str, Any]: 性能统计数据
        """
        # Rust后端性能统计
        return {
            "backend_type": self._backend_type,
            "stats_available": False
        }

    def close(self, force_gc: Optional[bool] = None) -> None:
        """显式关闭NumPack实例并释放所有资源
        
        【性能优化】快速close - 确保元数据flush，无额外GC开销
        
        调用close()后，可以通过调用open()重新打开文件。
        多次调用close()是安全的（幂等）。
        
        Parameters:
            force_gc (Optional[bool]): 是否强制执行垃圾回收。默认False以获得最佳性能。
        """
        if self._closed or not self._opened:
            return  # 已关闭或未打开，无需操作
        
        # 【性能优化】调用Rust端close以flush元数据，但不做额外清理
        if self._npk is not None and hasattr(self._npk, 'close'):
            try:
                self._npk.close()
            except:
                pass  # 忽略close错误
        
        # 更新状态
        self._closed = True
        self._opened = False
        self._npk = None  # 释放引用，Rust的Drop会自动清理
        
        # 仅在用户显式请求时才执行GC（通常不需要）
        if force_gc or (force_gc is None and self._force_gc_on_close):
            import gc
            gc.collect()
    
    def _windows_comprehensive_cleanup(self):
        """Windows特定的全面资源清理
        
        注意：由于使用Rust后端，大部分清理工作由Rust的Drop trait自动处理。
        只需要一次GC来清理Python侧的循环引用。
        """
        import gc
        # 只执行一次GC，Rust后端会自动处理其余清理工作
        gc.collect()
    
    def __del__(self):
        """析构函数"""
        self.close()
    
    def __enter__(self):
        """Context manager入口
        
        示例:
            with NumPack('data.npk') as npk:
                npk.save({'array': data})
        """
        # 如果文件未打开或已关闭，自动打开
        if not self._opened or self._closed:
            self.open()
        
        self._context_entered = True
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager出口
        
        即使发生异常也保证清理。
        异常（如果有）会在清理后重新抛出。
        """
        try:
            self.close()
        finally:
            self._context_entered = False
        
        # 不抑制异常
        return False

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

# 提供向后兼容的空函数(Rust后端自动管理内存)
def force_cleanup_windows_handles():
    """强制清理Windows句柄 - Rust后端自动管理,保留此函数以兼容旧代码"""
    import gc
    gc.collect()
    return True

# 导出的公共API
__all__ = ['NumPack', 'LazyArray', 'force_cleanup_windows_handles', 'get_backend_info']

# 提供后端信息查询
def get_backend_info():
    """获取当前后端信息
    
    Returns:
        Dict: 包含后端类型、平台、版本等信息的字典
    """
    return {
        'backend_type': _BACKEND_TYPE,  # 始终为 "rust"
        'platform': platform.system(),
        'is_windows': _is_windows(),
        'version': __version__,
        'description': '高性能Rust后端',
        'use_python_backend': False  # 兼容旧代码
    }