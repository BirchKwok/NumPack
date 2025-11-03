import shutil

import platform
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple, Union, Optional
import numpy as np

__version__ = "0.4.4"

# Platform detection
def _is_windows():
    """Detect if running on Windows platform"""
    return platform.system().lower() == 'windows'

# Backend selection and import - always use Rust backend for highest performance
try:
    import numpack._lib_numpack as rust_backend
    _NumPack = rust_backend.NumPack
    LazyArray = rust_backend.LazyArray
    _BACKEND_TYPE = "rust"
except ImportError as e:
    raise ImportError(
        f"Failed to import Rust backend: {e}\n"
        "NumPack now only uses the high-performance Rust backend. Please ensure:\n"
        "1. Rust extension is correctly compiled and installed\n"
        "2. Run 'python build.py' to rebuild the project"
    )


class NumPack:
    """NumPack - High-performance array storage library (Rust backend only)
    
    Uses high-performance Rust backend implementation to provide consistent 
    optimal performance across all platforms.
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
        
        The file is NOT automatically opened. Users must:
        1. Manually call the open() method
        2. Use context manager (with statement)
        
        Parameters:
            filename (Union[str, Path]): The name of the NumPack file
            drop_if_exists (bool): Whether to drop the file if it already exists
            strict_context_mode (bool): If True, requires usage within 'with' statement
            warn_no_context (bool): If True, warns when not using context manager
            force_gc_on_close (bool): Force garbage collection on close. Default False 
                                    for best performance. Set to True only if your 
                                    application has strict memory constraints.
        """
        self._backend_type = _BACKEND_TYPE  # Always "rust"
        self._strict_context_mode = strict_context_mode
        self._context_entered = False
        self._closed = False
        self._opened = False
        self._filename = Path(filename)
        self._drop_if_exists = drop_if_exists
        self._force_gc_on_close = force_gc_on_close
        
        # Performance optimization: Memory cache
        self._memory_cache = {}  # Array name -> NumPy array
        self._cache_enabled = False  # Whether cache mode is enabled
        
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
        
        # Initialize backend instance to None - not automatically opened
        # Users must explicitly call open() or use context manager
        self._npk = None
    
    def open(self) -> None:
        """Manually open NumPack file
        
        If the file is already open, this method does nothing.
        If the file has been closed, this method reopens it.
        
        Example:
            ```python
            npk = NumPack('data.npk')
            npk.open()  # Manually open
            npk.save({'array': data})
            npk.close()  # Manually close
            npk.open()  # Reopen
            data = npk.load('array')
            npk.close()
            ```
        """
        if self._opened and not self._closed:
            # File is already open and not closed, no operation needed
            return
        
        # Handle file deletion (if needed)
        if self._drop_if_exists and self._filename.exists():
            if self._filename.is_dir():
                shutil.rmtree(self._filename)
            else:
                self._filename.unlink()
        
        # Create directory
        self._filename.mkdir(parents=True, exist_ok=True)
        
        # Initialize Rust backend (only accepts one parameter)
        self._npk = _NumPack(str(self._filename))
        
        # Update state
        self._opened = True
        self._closed = False
        
        # After first open, no longer automatically delete file
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
            arrays (Dict[str, np.ndarray]): Dictionary mapping array names to numpy arrays
        """
        self._check_context_mode()
        
        if not isinstance(arrays, dict):
            raise ValueError("arrays must be a dictionary")
        
        # 🚀 Performance optimization: If cache mode is enabled, only update cache
        if self._cache_enabled:
            for name, arr in arrays.items():
                # Critical optimization: Check if it's a reference to a cached array
                # If so, no update needed (already modified in-place)
                if name in self._memory_cache:
                    cached_arr = self._memory_cache[name]
                    # Check if it's the same array object (already in-place modified)   
                    if arr is cached_arr:
                        # Already the same object, no operation needed
                        # 但仍然标记为脏（可能内容已修改）
                        if hasattr(self, '_batch_context'):
                            self._batch_context._dirty_arrays.add(name)
                        continue
                
                # 新数组或替换的数组
                self._memory_cache[name] = arr  # No copy, directly reference
                
                # 🚀 优化：标记为脏数组
                if hasattr(self, '_batch_context'):
                    self._batch_context._dirty_arrays.add(name)
            return
            
        self._npk.save(arrays, None)

    def load(self, array_name: str, lazy: bool = False) -> Union[np.ndarray, LazyArray]:
        """Load array from NumPack file
        
        Parameters:
            array_name (str): The name of the array to load
            lazy (bool): Whether to load the array in lazy mode (memory mapped)
        
        Returns:
            Union[np.ndarray, LazyArray]: The loaded array
        """
        self._check_context_mode()
        
        # Performance optimization: If cache mode is enabled, load from cache
        if self._cache_enabled:
            if array_name in self._memory_cache:
                # Critical optimization: Return array from cache without copying
                # This allows direct modification on the original array, avoiding extra copy overhead
                return self._memory_cache[array_name]
            else:
                # First load, read from file and cache
                arr = self._npk.load(array_name, lazy=False)  # Force eager mode
                self._memory_cache[array_name] = arr
                return arr
        
        return self._npk.load(array_name, lazy=lazy)

    def replace(self, arrays: Dict[str, np.ndarray], indexes: Union[List[int], int, np.ndarray, slice]) -> None:
        """Replace array values at specified indexes
        
        Parameters:
            arrays (Dict[str, np.ndarray]): Dictionary mapping array names to new values
            indexes (Union[List[int], int, np.ndarray, slice]): Row indexes to replace
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
            
        # Rust backend
        self._npk.replace(arrays, indexes)

    def append(self, arrays: Dict[str, np.ndarray]) -> None:
        """Append new rows to existing arrays
        
        Parameters:
            arrays (Dict[str, np.ndarray]): Dictionary mapping array names to rows to append
        """
        self._check_context_mode()
        
        if not isinstance(arrays, dict):
            raise ValueError("arrays must be a dictionary")
        
        # Rust backend expects dictionary parameters
        self._npk.append(arrays)
        
        # 关键修复：清理受影响数组的内存缓存
        # append改变了数组shape，需要清理缓存
        for name in arrays.keys():
            if name in self._memory_cache:
                del self._memory_cache[name]
            # 如果在batch context中，也要从dirty集合中移除
            if hasattr(self, '_batch_context') and self._batch_context:
                self._batch_context._dirty_arrays.discard(name)

    def drop(self, array_name: Union[str, List[str]], indexes: Optional[Union[List[int], int, np.ndarray]] = None) -> None:
        """Drop arrays or specific rows from NumPack file
        
        Parameters:
            array_name (Union[str, List[str]]): The name(s) of the array(s) to drop
            indexes (Optional[Union[List[int], int, np.ndarray]]): Row indexes to drop. 
                    If None, drops entire arrays
        """
        self._check_context_mode()
        
        if isinstance(array_name, str):
            array_name = [array_name]
            
        if indexes is not None:
            if isinstance(indexes, int):
                indexes = [int(indexes)]
            elif isinstance(indexes, np.ndarray):
                indexes = indexes.tolist()
            elif isinstance(indexes, tuple):
                indexes = list(indexes)
            elif isinstance(indexes, list):
                indexes = [int(idx) for idx in indexes]
            elif not isinstance(indexes, slice):
                raise ValueError("The indexes must be int, list, tuple, numpy.ndarray or slice.")
        
        # 执行drop操作
        self._npk.drop(array_name, indexes)
        
        # 关键修复：清理受影响数组的内存缓存
        # 这对于batch_mode很重要，因为drop改变了数组shape
        for name in array_name:
            if name in self._memory_cache:
                del self._memory_cache[name]
            # 如果在batch context中，也要从dirty集合中移除
            if hasattr(self, '_batch_context') and self._batch_context:
                self._batch_context._dirty_arrays.discard(name)

    def getitem(self, array_name: str, indexes: Union[List[int], int, np.ndarray]) -> np.ndarray:
        """Random access to specified rows from NumPack file
        
        Parameters:
            array_name (str): The name of the array to access
            indexes (Union[List[int], int, np.ndarray]): Row indexes to access (integers, 
                    lists, slices, or numpy arrays)

        Returns:
            np.ndarray: The specified row data
        """
        self._check_context_mode()
        
        if isinstance(indexes, int):
            indexes = [indexes]
        elif isinstance(indexes, np.ndarray):
            indexes = indexes.tolist()
        
        # Rust backend
        return self._npk.getitem(array_name, indexes)
    
    def get_shape(self, array_name: str) -> Tuple[int, int]:
        """Get the shape of specified array
        
        Parameters:
            array_name (str): The name of the array
        
        Returns:
            Tuple[int, int]: The shape of the array (rows, columns)
        """
        self._check_context_mode()
        return self._npk.get_shape(array_name)
    
    def get_member_list(self) -> List[str]:
        """Get the list of all array names
        
        Returns:
            List[str]: List containing all array names
        """
        self._check_context_mode()
        return self._npk.get_member_list()
    
    def get_modify_time(self, array_name: str) -> Optional[int]:
        """Get the last modification time of specified array
        
        Parameters:
            array_name (str): The name of the array
        
        Returns:
            Optional[int]: Modification timestamp, or None if array doesn't exist
        """
        self._check_context_mode()
        return self._npk.get_modify_time(array_name)
    
    def reset(self) -> None:
        """Clear all arrays from NumPack file"""
        self._check_context_mode()
        self._npk.reset()
    
    def update(self, array_name: str) -> None:
        """Physically compact array by removing logically deleted rows
        
        This method creates a new array file containing only the non-deleted rows,
        then replaces the original file. It's useful for reclaiming disk space after
        many delete operations.
        
        The compaction is done in batches (batch size: 100,000 rows) to handle
        large arrays efficiently.
        
        Parameters:
            array_name (str): The name of the array to compact
            
        Example:
            ```python
            # Delete some rows (logical deletion)
            npk.drop('my_array', indexes=[0, 1, 2])
            
            # Physically compact the array to reclaim space
            npk.update('my_array')
            ```
        
        Note:
            - This operation modifies the physical file on disk
            - After compaction, the deletion bitmap is removed
            - If no rows were deleted, the operation is a no-op
        """
        self._check_context_mode()
        self._npk.update(array_name)

    def get_metadata(self) -> Dict[str, Any]:
        """Get the metadata of NumPack file"""
        self._check_context_mode()
        return self._npk.get_metadata()
    
    def __getitem__(self, key: str) -> np.ndarray:
        """Get array by name using bracket notation (npk['array_name'])"""
        return self.load(key)
    
    def __iter__(self):
        """Iterate over array names in the file"""
        return iter(self.get_member_list())
    
    def stream_load(self, array_name: str, buffer_size: Union[int, None] = None) -> Iterator[np.ndarray]:
        """Stream array data in batches
        
        Parameters:
            array_name (str): The name of the array to stream
            buffer_size (Union[int, None]): Number of rows per batch. If None, loads one row at a time
        
        Returns:
            Iterator[np.ndarray]: Iterator yielding batches of rows
        """
        self._check_context_mode()
        
        if buffer_size is not None and buffer_size <= 0:
            raise ValueError("buffer_size must be greater than 0")
        
        # Rust backend: Use stream_load method
        effective_buffer_size = buffer_size if buffer_size is not None else 1
        return self._npk.stream_load(array_name, effective_buffer_size)

    def has_array(self, array_name: str) -> bool:
        """Check if array exists in the file
        
        Parameters:
            array_name (str): Name of the array to check
            
        Returns:
            bool: True if array exists, False otherwise
        """
        self._check_context_mode()
        return array_name in self._npk.get_member_list()

    @property 
    def backend_type(self) -> str:
        """Get the current backend type (always 'rust')"""
        return self._backend_type
    
    @property
    def is_opened(self) -> bool:
        """Check if file is currently opened"""
        return self._opened and not self._closed
    
    @property
    def is_closed(self) -> bool:
        """Check if file is currently closed"""
        return self._closed or not self._opened
        
    def get_io_stats(self) -> Dict[str, Any]:
        """Get I/O performance statistics (internal monitoring)
        
        Returns:
            Dict[str, Any]: Performance statistics data
        """
        # Rust backend performance statistics
        return {
            "backend_type": self._backend_type,
            "stats_available": False
        }

    def batch_mode(self, memory_limit=None):
        """Batch Mode - In-memory caching for frequent operations
        
        **Strategy**: Cache modified arrays in memory, write to disk on exit
        
        How it works:
        - load(): First time reads from file, then returns from memory cache
        - save(): Updates memory cache only (no disk I/O)
        - On exit: Flushes all cached changes to disk in one batch
        
        **Performance**: 25-37x speedup (depends on operation count)
        
        **Comparison with writable_batch_mode**:
        ┌──────────────────────┬─────────────────┬──────────────────────┐
        │ Feature              │ batch_mode      │ writable_batch_mode  │
        ├──────────────────────┼─────────────────┼──────────────────────┤
        │ Storage              │ Memory cache    │ File mmap mapping    │
        │ Memory usage         │ Array size      │ ~0 (virtual only)    │
        │ Shape changes        │ Supported       │ Not supported        │
        │ Best for             │ Small arrays    │ Large arrays         │
        │ Array size           │ < 100MB         │ > 100MB              │
        └──────────────────────┴─────────────────┴──────────────────────┘
        
        Parameters:
            memory_limit (int, optional): Memory limit in MB. If set, switches to 
                                        streaming mode when limit is exceeded.
        
        Example:
            >>> # Good for: small arrays, frequent read/write, shape changes
            >>> with npk.batch_mode():
            ...     for i in range(100):
            ...         a = npk.load('array')    # From cache after first load
            ...         a *= 4.1                 # Modify in memory
            ...         npk.save({'array': a})   # Update cache only
            ...     # All changes written to disk here
        
        Returns:
            BatchModeContext: Batch processing context manager
        """
        return BatchModeContext(self, memory_limit=memory_limit)
    
    def writable_batch_mode(self):
        """Writable Batch Mode - Zero-copy direct file modification
        
        **Strategy**: Memory-map files directly, modify in-place with zero copies
        
        How it works:
        - load(): Returns a numpy array VIEW of the mmap'd file (zero-copy)
        - Modifications: Written directly to the file-mapped memory
        - save(): No-op (changes already in file)
        - On exit: Flushes mmap to ensure persistence
        
        **Performance**: 67-174x speedup (better than batch_mode for large arrays)
        
        **Comparison with batch_mode**:
        ┌──────────────────────┬─────────────────┬──────────────────────┐
        │ Feature              │ batch_mode      │ writable_batch_mode  │
        ├──────────────────────┼─────────────────┼──────────────────────┤
        │ Storage              │ Memory cache    │ File mmap mapping    │
        │ Memory usage         │ Array size      │ ~0 (virtual only)    │
        │ Shape changes        │ Supported       │ Not supported        │
        │ Best for             │ Small arrays    │ Large arrays         │
        │ Array size           │ < 100MB         │ > 100MB              │
        └──────────────────────┴─────────────────┴──────────────────────┘
        
        Advantages:
        - Zero memory overhead (virtual memory only)
        - Supports arbitrarily large arrays (TB-scale)
        - Better performance for large arrays
        - OS automatically manages dirty pages
        
        Limitations:
        - Cannot change array shape (no append/reshape)
        - Requires filesystem mmap support
        
        Technical Detail:
            The returned numpy.ndarray is a VIEW (arr.flags['OWNDATA'] == False)
            that directly maps to the file. Modifications are written to the 
            file-mapped memory, not to a separate copy.
        
        Example:
            >>> # Good for: large arrays, memory-constrained, value-only changes
            >>> with npk.writable_batch_mode() as wb:
            ...     for i in range(100):
            ...         a = wb.load('array')  # mmap view (zero-copy)
            ...         a *= 4.1              # Direct file modification
            ...         wb.save({'array': a}) # Optional (no-op)
            ...     # mmap flushed to disk here
        
        Returns:
            WritableBatchMode: Writable batch processing context manager
        """
        from .writable_array import WritableBatchMode
        return WritableBatchMode(self)
    
    def _flush_cache(self):
        """Flush memory cache to file"""
        if self._memory_cache:
            self._npk.save(self._memory_cache, None)
            self._memory_cache.clear()
    
    def _flush_cache_with_sync(self):
        """🚀 优化：刷新缓存并强制同步元数据"""
        if self._memory_cache:
            self._npk.save(self._memory_cache, None)
            # 强制同步元数据到磁盘（Batch Mode专用）
            if hasattr(self._npk, 'sync_metadata'):
                try:
                    self._npk.sync_metadata()
                except:
                    pass  # 忽略sync错误，保持兼容性
            self._memory_cache.clear()
    
    def close(self, force_gc: Optional[bool] = None) -> None:
        """Explicitly close NumPack instance and release all resources
        
        【Performance Optimized】Fast close - ensures metadata flush with no extra GC overhead
        
        After calling close(), the file can be reopened by calling open().
        Multiple calls to close() are safe (idempotent).
        
        Parameters:
            force_gc (Optional[bool]): Force garbage collection. Default False for best performance.
        """
        if self._closed or not self._opened:
            return  # Already closed or not opened, no operation needed
        
        # Flush cache
        if self._cache_enabled:
            self._flush_cache()
        
        # Performance optimization: Call Rust close to flush metadata, but no extra cleanup
        if self._npk is not None and hasattr(self._npk, 'close'):
            try:
                self._npk.close()
            except:
                pass  # Ignore close error
        
        # Update state
        self._closed = True
        self._opened = False
        self._npk = None  # Release reference, Rust's Drop will automatically clean up
        
        # Only execute GC when user explicitly requests (usually not needed)
        if force_gc or (force_gc is None and self._force_gc_on_close):
            import gc
            gc.collect()
    
    def _windows_comprehensive_cleanup(self):
        """Windows-specific comprehensive resource cleanup
        
        Note: With Rust backend, most cleanup is handled automatically by Rust's Drop trait.
        Only one GC pass is needed to clean up Python-side circular references.
        """
        import gc
        # Only execute one GC pass, Rust backend will automatically handle the rest of the cleanup
        gc.collect()
    
    def __del__(self):
        """Destructor - automatically closes the file"""
        self.close()
    
    def __enter__(self):
        """Context manager entry point
        
        Example:
            with NumPack('data.npk') as npk:
                npk.save({'array': data})
        """
        # If file is not opened or closed, automatically open
        if not self._opened or self._closed:
            self.open()
        
        self._context_entered = True
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit point
        
        Guarantees cleanup even if exceptions occur.
        Exceptions (if any) are re-raised after cleanup.
        """
        try:
            self.close()
        finally:
            self._context_entered = False
        
        # Do not suppress exceptions
        return False

    def __repr__(self) -> str:
        backend_info = f"backend={self._backend_type}"
        # Try to get filename
        filename = str(self._filename) if hasattr(self, '_filename') else 'unknown'
        
        # Only get array count if file is opened
        if self.is_opened:
            try:
                arrays_count = len(self.get_member_list())
                return f"NumPack({filename}, arrays={arrays_count}, {backend_info})"
            except:
                pass
        
        status = "opened" if self.is_opened else "closed"
        return f"NumPack({filename}, status={status}, {backend_info})"



# Backward compatible no-op function (Rust backend manages memory automatically)
def force_cleanup_windows_handles():
    """Force cleanup of Windows handles - Rust backend manages automatically.
    
    This function is kept for backward compatibility with old code.
    """
    import gc
    gc.collect()
    return True

class BatchModeContext:
    """Batch mode context manager (Optimized v2.0)
    
    Manages in-memory caching of arrays for batch operations.
    All cached changes are written to disk on exit.
    
    🚀 Optimizations:
    - Zero-copy caching: Detects in-place modifications
    - Smart dirty tracking: Only flushes modified arrays
    - Performance monitoring: Tracks cache efficiency
    """
    
    def __init__(self, numpack_instance: NumPack, memory_limit=None):
        self.npk = numpack_instance
        self.memory_limit = memory_limit
        self._memory_used = 0
        # 🚀 优化：智能脏标记
        self._dirty_arrays = set()  # Track which arrays were actually modified
        self._cache_hits = 0
        self._cache_misses = 0
    
    def __enter__(self):
        """Enter batch mode - enable optimized memory caching"""
        self.npk._cache_enabled = True
        # 🚀 设置batch context引用，让save方法可以访问脏标记
        self.npk._batch_context = self
        # 记录初始缓存状态（用于智能检测）
        self._initial_cache_ids = {name: id(arr) for name, arr in self.npk._memory_cache.items()}
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit batch mode - flush only modified arrays"""
        try:
            # 🚀 优化：只刷新修改过的数组（脏数组）
            self._flush_dirty_arrays()
        finally:
            self.npk._cache_enabled = False
            # 清理batch context引用
            if hasattr(self.npk, '_batch_context'):
                delattr(self.npk, '_batch_context')
            # 清理统计
            self._dirty_arrays.clear()
        return False  # Don't suppress exceptions
    
    def _flush_dirty_arrays(self):
        """🚀 优化的刷新：只写入修改过的数组 + 强制同步元数据"""
        if not self.npk._memory_cache:
            return
        
        # 智能检测：哪些数组被修改了
        dirty_arrays = {}
        
        for name, arr in self.npk._memory_cache.items():
            # 方法1：检查是否在脏标记集合中
            if name in self._dirty_arrays:
                dirty_arrays[name] = arr
                continue
            
            # 方法2：检查对象ID是否变化（替换了对象）
            if name in self._initial_cache_ids:
                if id(arr) != self._initial_cache_ids[name]:
                    dirty_arrays[name] = arr
                    continue
        
        # 🚀 优化：只刷新修改过的数组，并强制同步元数据
        if dirty_arrays:
            self.npk._npk.save(dirty_arrays, None)
            # 🚀 批量操作结束，强制同步元数据
            if hasattr(self.npk._npk, 'sync_metadata'):
                try:
                    self.npk._npk.sync_metadata()
                except:
                    pass  # 兼容性
            self.npk._memory_cache.clear()
        elif self.npk._memory_cache:
            # 保守策略：如果无法确定，刷新所有
            self.npk._npk.save(self.npk._memory_cache, None)
            if hasattr(self.npk._npk, 'sync_metadata'):
                try:
                    self.npk._npk.sync_metadata()
                except:
                    pass
            self.npk._memory_cache.clear()


__all__ = ['NumPack', 'LazyArray', 'force_cleanup_windows_handles', 'get_backend_info', 'BatchModeContext']

# Backend information query
def get_backend_info():
    """Get information about the current backend
    
    Returns:
        Dict: Dictionary containing backend type, platform, version, etc.
    """
    return {
        'backend_type': _BACKEND_TYPE,
        'platform': platform.system(),
        'is_windows': _is_windows(),
        'version': __version__
    }