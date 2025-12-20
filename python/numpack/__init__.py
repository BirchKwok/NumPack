import shutil

import platform
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple, Union, Optional
import numpy as np

if hasattr(np, "issubdtype"):
    try:
        np.issubdtype(np.float32, (np.integer, np.bool_))  # type: ignore[arg-type]
    except (TypeError, ValueError):  # ValueError for numpy 2.x compatibility
        _orig_issubdtype = np.issubdtype

        def _numpack_issubdtype(arg1, arg2):
            if isinstance(arg2, tuple):
                return any(_orig_issubdtype(arg1, candidate) for candidate in arg2)
            return _orig_issubdtype(arg1, arg2)

        np.issubdtype = _numpack_issubdtype  # type: ignore[assignment]

__version__ = "0.5.0"

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
    """High-performance array storage backed by the Rust implementation.

    This class provides a Python interface to the NumPack on-disk format.

    Notes
    -----
    - The storage backend is provided by the Rust extension module.
    - Files are not opened automatically. Call `open` or use a context manager.
    """
    
    def __init__(
        self, 
        filename: Union[str, Path], 
        drop_if_exists: bool = False,
        strict_context_mode: bool = False,
        warn_no_context: bool = None,
        force_gc_on_close: bool = False
    ):
        """Create a `NumPack` handle.

        The underlying file is **not** opened automatically. You must either:

        - Call `open` explicitly, or
        - Use `NumPack` as a context manager.

        Parameters
        ----------
        filename : str or Path
            NumPack directory path.
        drop_if_exists : bool, optional
            If True, delete the path first if it already exists.
        strict_context_mode : bool, optional
            If True, disallow calling mutating APIs outside of a ``with`` block.
        warn_no_context : bool, optional
            If True, emit a warning when not using a context manager (defaults to
            True on Windows).
        force_gc_on_close : bool, optional
            If True, run garbage collection on close. Default False for best
            performance.
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
        """Open the NumPack file.

        Calling `open` is idempotent. If the file is already opened, this is a
        no-op. If the file has been closed, this reopens it.

        Examples
        --------
        >>> npk = NumPack('data.npk')
        >>> npk.open()
        >>> npk.save({'array': data})
        >>> npk.close()
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
        """Save arrays to the NumPack file.

        Parameters
        ----------
        arrays : dict[str, numpy.ndarray]
            Mapping from array name to data.

        Raises
        ------
        ValueError
            If `arrays` is not a dict.
        RuntimeError
            If the NumPack instance is not opened or (when enabled) not used
            under a context manager.
        """
        self._check_context_mode()
        
        if not isinstance(arrays, dict):
            raise ValueError("arrays must be a dictionary")
        
        # Performance optimization: If cache mode is enabled, only update cache
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
                
                # 优化：标记为脏数组
                if hasattr(self, '_batch_context'):
                    self._batch_context._dirty_arrays.add(name)
            return
            
        self._npk.save(arrays, None)

    def load(self, array_name: str, lazy: bool = False) -> Union[np.ndarray, LazyArray]:
        """Load an array from the NumPack file.

        Parameters
        ----------
        array_name : str
            Array name.
        lazy : bool, optional
            If True, return a `LazyArray` (memory-mapped) instead of loading data
            eagerly.

        Returns
        -------
        numpy.ndarray or LazyArray
            The loaded array.

        Raises
        ------
        RuntimeError
            If the NumPack instance is not opened or (when enabled) not used
            under a context manager.
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
        """Replace values at specific row indexes.

        Parameters
        ----------
        arrays : dict[str, numpy.ndarray]
            Mapping from array name to replacement values.
        indexes : int or list[int] or numpy.ndarray or slice
            Row indexes to replace.

        Raises
        ------
        ValueError
            If `arrays` is not a dict or `indexes` has an unsupported type.
        RuntimeError
            If the NumPack instance is not opened or (when enabled) not used
            under a context manager.
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
        """Append rows to existing arrays.

        Parameters
        ----------
        arrays : dict[str, numpy.ndarray]
            Mapping from array name to the rows to append.

        Raises
        ------
        ValueError
            If `arrays` is not a dict.
        RuntimeError
            If the NumPack instance is not opened or (when enabled) not used
            under a context manager.
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
        """Drop arrays or rows from a NumPack file.

        Parameters
        ----------
        array_name : str or list[str]
            Array name(s) to drop.
        indexes : int or list[int] or numpy.ndarray or slice, optional
            Row indexes to drop. If None, drops entire arrays.

        Raises
        ------
        ValueError
            If `indexes` has an unsupported type.
        RuntimeError
            If the NumPack instance is not opened or (when enabled) not used
            under a context manager.
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

    def getitem(self, array_name: str, indexes: Union[List[int], int, np.ndarray, slice]) -> np.ndarray:
        """Return a subset of rows using random access.

        Parameters
        ----------
        array_name : str
            Array name.
        indexes : int or list[int] or numpy.ndarray or slice
            Row indexes to access.

        Returns
        -------
        numpy.ndarray
            The selected rows.

        Raises
        ------
        RuntimeError
            If the NumPack instance is not opened or (when enabled) not used
            under a context manager.
        """
        self._check_context_mode()
        
        if isinstance(indexes, slice):
            pass
        elif isinstance(indexes, int):
            indexes = [indexes]
        elif isinstance(indexes, np.ndarray):
            indexes = indexes.tolist()
        
        # Rust backend
        return self._npk.getitem(array_name, indexes)
    
    def get_shape(self, array_name: str) -> Tuple[int, int]:
        """Return the shape of an array.

        Parameters
        ----------
        array_name : str
            Array name.

        Returns
        -------
        tuple[int, int]
            Shape of the array.

        Raises
        ------
        RuntimeError
            If the NumPack instance is not opened or (when enabled) not used
            under a context manager.
        """
        self._check_context_mode()
        return self._npk.get_shape(array_name)
    
    def get_member_list(self) -> List[str]:
        """Return the list of array names in the file.

        Returns
        -------
        list[str]
            All array names.

        Raises
        ------
        RuntimeError
            If the NumPack instance is not opened or (when enabled) not used
            under a context manager.
        """
        self._check_context_mode()
        return self._npk.get_member_list()
    
    def get_modify_time(self, array_name: str) -> Optional[int]:
        """Return the last modification time of an array.

        Parameters
        ----------
        array_name : str
            Array name.

        Returns
        -------
        int or None
            Modification timestamp, or None if the array does not exist.

        Raises
        ------
        RuntimeError
            If the NumPack instance is not opened or (when enabled) not used
            under a context manager.
        """
        self._check_context_mode()
        return self._npk.get_modify_time(array_name)
    
    def reset(self) -> None:
        """Remove all arrays from the NumPack file."""
        self._check_context_mode()
        self._npk.reset()
    
    def update(self, array_name: str) -> None:
        """Physically compact an array by removing logically deleted rows.

        This operation rewrites the underlying array file and removes the
        deletion bitmap. It can be used to reclaim disk space after many delete
        operations.

        Parameters
        ----------
        array_name : str
            Name of the array to compact.

        Notes
        -----
        - This operation modifies the physical data on disk.
        - If no rows were deleted, this is a no-op.

        Examples
        --------
        >>> npk.drop('my_array', indexes=[0, 1, 2])
        >>> npk.update('my_array')
        """
        self._check_context_mode()
        self._npk.update(array_name)

    def clone(self, source_name: str, target_name: str) -> None:
        """Clone an existing array to a new array name.

        The cloned array is independent of the original and can be modified
        separately.

        Parameters
        ----------
        source_name : str
            Source array name.
        target_name : str
            Target array name.

        Raises
        ------
        KeyError
            If `source_name` does not exist.
        ValueError
            If `target_name` already exists.

        Examples
        --------
        >>> npk.clone('original_array', 'cloned_array')
        >>> data = npk.load('cloned_array')
        >>> data *= 2
        >>> npk.save({'cloned_array': data})
        """
        self._check_context_mode()
        self._npk.clone(source_name, target_name)

    def get_metadata(self) -> Dict[str, Any]:
        """Return the file metadata.

        Returns
        -------
        dict
            Metadata dictionary returned by the backend.

        Raises
        ------
        RuntimeError
            If the NumPack instance is not opened or (when enabled) not used
            under a context manager.
        """
        self._check_context_mode()
        return self._npk.get_metadata()
    
    def __getitem__(self, key: str) -> np.ndarray:
        """Get array by name using bracket notation (npk['array_name'])"""
        return self.load(key)
    
    def __iter__(self):
        """Iterate over array names in the file"""
        return iter(self.get_member_list())
    
    def stream_load(self, array_name: str, buffer_size: Union[int, None] = None) -> Iterator[np.ndarray]:
        """Iterate over an array in batches.

        Parameters
        ----------
        array_name : str
            Array name.
        buffer_size : int or None, optional
            Number of rows per batch. If None, yields one row per batch.

        Returns
        -------
        iterator of numpy.ndarray
            Iterator yielding batches.

        Raises
        ------
        ValueError
            If `buffer_size` is not None and <= 0.
        RuntimeError
            If the NumPack instance is not opened or (when enabled) not used
            under a context manager.
        """
        self._check_context_mode()
        
        if buffer_size is not None and buffer_size <= 0:
            raise ValueError("buffer_size must be greater than 0")
        
        # Rust backend: Use stream_load method
        effective_buffer_size = buffer_size if buffer_size is not None else 1
        return self._npk.stream_load(array_name, effective_buffer_size)

    def has_array(self, array_name: str) -> bool:
        """Return True if an array exists in the file."""
        self._check_context_mode()
        return array_name in self._npk.get_member_list()

    @property 
    def backend_type(self) -> str:
        """Backend identifier string.

        Notes
        -----
        The current implementation always uses the Rust backend.
        """
        return self._backend_type
    
    @property
    def is_opened(self) -> bool:
        """Whether the NumPack instance is currently opened."""
        return self._opened and not self._closed
    
    @property
    def is_closed(self) -> bool:
        """Whether the NumPack instance is currently closed."""
        return self._closed or not self._opened
        
    def get_io_stats(self) -> Dict[str, Any]:
        """Return I/O statistics, if available.

        Returns
        -------
        dict
            Backend statistics payload. The Rust backend currently does not
            expose detailed per-call statistics via this API.
        """
        # Rust backend performance statistics
        return {
            "backend_type": self._backend_type,
            "stats_available": False
        }

    def batch_mode(self, memory_limit=None):
        """Enable in-memory caching for batch operations.

        In batch mode, repeated `load` / `save` operations are accelerated by
        caching arrays in memory and flushing changes to disk when the context
        exits.

        Parameters
        ----------
        memory_limit : int, optional
            Soft memory limit (in MB). If provided, the context may switch to a
            more conservative strategy when the cache grows beyond the limit.

        Returns
        -------
        BatchModeContext
            Context manager controlling the batch cache lifecycle.

        Notes
        -----
        Compared to `writable_batch_mode`:

        - `batch_mode` caches full arrays in memory (supports shape changes).
        - `writable_batch_mode` uses memory mapping (near-zero RAM, but cannot
          change array shapes).

        Examples
        --------
        >>> with npk.batch_mode():
        ...     a = npk.load('array')
        ...     a *= 4.1
        ...     npk.save({'array': a})
        """
        return BatchModeContext(self, memory_limit=memory_limit)
    
    def writable_batch_mode(self):
        """Enable zero-copy, in-place updates via memory mapping.

        This mode returns writable `numpy.ndarray` views backed directly by the
        array data files. In-place modifications are written to the mapped
        region and flushed on exit.

        Returns
        -------
        WritableBatchMode
            Context manager providing `load`/`save` methods backed by `mmap`.

        Notes
        -----
        - Best for large arrays and memory-constrained workflows.
        - Array shape changes are not supported (append/reshape would require a
          different file layout).

        Examples
        --------
        >>> with npk.writable_batch_mode() as wb:
        ...     a = wb.load('array')
        ...     a *= 4.1
        ...     wb.save({'array': a})
        """
        from .writable_array import WritableBatchMode
        return WritableBatchMode(self)
    
    def _flush_cache(self):
        """Flush memory cache to file"""
        if self._memory_cache:
            self._npk.save(self._memory_cache, None)
            self._memory_cache.clear()
    
    def _flush_cache_with_sync(self):
        """优化：刷新缓存并强制同步元数据"""
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
        """Close the NumPack instance and release resources.

        Calling `close` is idempotent. After closing, the instance can be
        reopened by calling `open`.

        Parameters
        ----------
        force_gc : bool, optional
            If True, force a garbage collection pass after closing.
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
        """Enter the context manager.

        The file is opened automatically if it is not already opened.

        Returns
        -------
        NumPack
            The opened instance.
        """
        # If file is not opened or closed, automatically open
        if not self._opened or self._closed:
            self.open()
        
        self._context_entered = True
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager.

        This always closes the file. Exceptions (if any) are not suppressed.
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
    """Force cleanup of Windows file handles.

    Notes
    -----
    This function is kept for backward compatibility. With the Rust backend,
    most resource cleanup is handled automatically.
    """
    import gc
    gc.collect()
    return True

class BatchModeContext:
    """Context manager implementing `NumPack.batch_mode`.

    This context enables in-memory caching and batches disk writes until exit.
    """
    
    def __init__(self, numpack_instance: NumPack, memory_limit=None):
        self.npk = numpack_instance
        self.memory_limit = memory_limit
        self._memory_used = 0
        # 优化：智能脏标记
        self._dirty_arrays = set()  # Track which arrays were actually modified
        self._cache_hits = 0
        self._cache_misses = 0
    
    def __enter__(self):
        """Enter batch mode and enable caching."""
        self.npk._cache_enabled = True
        # 设置batch context引用，让save方法可以访问脏标记
        self.npk._batch_context = self
        # 记录初始缓存状态（用于智能检测）
        self._initial_cache_ids = {name: id(arr) for name, arr in self.npk._memory_cache.items()}
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit batch mode and flush cached changes."""
        try:
            # 优化：只刷新修改过的数组（脏数组）
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
        """优化的刷新：只写入修改过的数组 + 强制同步元数据"""
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
        
        # 优化：只刷新修改过的数组，并强制同步元数据
        if dirty_arrays:
            # 过滤掉无效的数组对象
            valid_arrays = {}
            for name, arr in dirty_arrays.items():
                if hasattr(arr, 'shape') and hasattr(arr, 'dtype'):
                    valid_arrays[name] = arr
                else:
                    # 调试：记录无效对象
                    print(f"Warning: Skipping invalid array {name}: {type(arr)} - {arr}")

            if valid_arrays:
                try:
                    self.npk._npk.save(valid_arrays, None)
                except Exception as e:
                    print(f"Error saving valid_arrays: {e}")
                    print(f"valid_arrays keys: {list(valid_arrays.keys())}")
                    print(f"valid_arrays types: {[(k, type(v)) for k, v in valid_arrays.items()]}")
                    # 如果保存失败，尝试逐个保存
                    for name, arr in valid_arrays.items():
                        try:
                            self.npk._npk.save({name: arr}, None)
                            print(f"Successfully saved {name}")
                        except Exception as e2:
                            print(f"Failed to save {name}: {e2}")
                    # 无论如何都要清理缓存
                    self.npk._memory_cache.clear()
                    return
                # 批量操作结束，强制同步元数据
                if hasattr(self.npk._npk, 'sync_metadata'):
                    try:
                        self.npk._npk.sync_metadata()
                    except:
                        pass  # 兼容性
            self.npk._memory_cache.clear()
        elif self.npk._memory_cache:
            # 保守策略：如果无法确定，刷新所有
            # 过滤掉无效的数组对象
            valid_cache = {}
            for name, arr in self.npk._memory_cache.items():
                if hasattr(arr, 'shape') and hasattr(arr, 'dtype'):
                    valid_cache[name] = arr

            if valid_cache:
                self.npk._npk.save(valid_cache, None)
                if hasattr(self.npk._npk, 'sync_metadata'):
                    try:
                        self.npk._npk.sync_metadata()
                    except:
                        pass
            self.npk._memory_cache.clear()


__all__ = [
    'NumPack', 
    'LazyArray', 
    'force_cleanup_windows_handles', 
    'get_backend_info', 
    'BatchModeContext',
    'pack',
    'unpack',
    'get_package_info',
]

# Package operations (convenience imports at top level)
from .io.package_io import pack, unpack, get_package_info

# Backend information query
def get_backend_info():
    """Return information about the active backend.

    Returns
    -------
    dict
        Dictionary containing backend type, platform, version, and platform
        flags.
    """
    info = {
        'backend_type': _BACKEND_TYPE,
        'platform': platform.system(),
        'is_windows': _is_windows(),
        'version': __version__,
    }
    
    
    return info