"""
🚀 WritableArray - 可写数组包装器
直接在文件上修改，零内存开销
"""
import numpy as np
import mmap
import os


class WritableArray:
    """可写数组包装器 - 基于mmap的零拷贝方案
    
    核心优化：
    1. 使用可写mmap直接映射文件
    2. 返回NumPy数组视图，直接在文件上操作
    3. 修改自动同步到文件（操作系统管理）
    4. 零内存开销（只是虚拟内存映射）
    """
    
    def __init__(self, file_path, shape, dtype, mode='r+'):
        """
        Args:
            file_path: 数据文件路径
            shape: 数组形状
            dtype: 数据类型
            mode: 'r+'可写，'r'只读
        """
        self.file_path = file_path
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)
        self.mode = mode
        self._mmap = None
        self._array = None
        self._file = None
        
    def __enter__(self):
        """打开文件并创建mmap"""
        # 打开文件
        if self.mode == 'r+':
            self._file = open(self.file_path, 'r+b')
        else:
            self._file = open(self.file_path, 'rb')
        
        # 创建mmap
        if self.mode == 'r+':
            self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_WRITE)
        else:
            self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        
        # 创建NumPy数组视图（零拷贝）
        self._array = np.ndarray(
            shape=self.shape,
            dtype=self.dtype,
            buffer=self._mmap
        )
        
        return self._array
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """关闭mmap和文件"""
        if self._mmap is not None:
            # 🚀 关键：确保修改写入磁盘
            if self.mode == 'r+':
                self._mmap.flush()
            self._mmap.close()
            self._mmap = None
        
        if self._file is not None:
            self._file.close()
            self._file = None
        
        self._array = None
        return False


class WritableBatchMode:
    """可写批处理模式 - 零内存开销
    
    策略：
    1. 使用可写mmap打开所有数组文件
    2. 修改直接在文件上进行（零拷贝）
    3. 操作系统自动管理脏页写回
    4. 退出时统一flush确保持久化
    """
    
    def __init__(self, numpack_instance):
        self.npk = numpack_instance
        self.writable_arrays = {}  # array_name -> WritableArray
        self.array_cache = {}  # array_name -> numpy array view
        
    def __enter__(self):
        return self
    
    def load(self, array_name):
        """加载可写数组视图
        
        Returns:
            numpy array: 直接映射到文件的数组视图（可写）
        """
        if array_name in self.array_cache:
            return self.array_cache[array_name]
        
        # 获取数组元数据（使用Python API）
        try:
            shape_tuple = self.npk.get_shape(array_name)
            shape = list(shape_tuple)
        except Exception as e:
            raise KeyError(f"Array '{array_name}' not found: {e}")
        
        # 构建文件路径
        file_path = os.path.join(str(self.npk._filename), f"data_{array_name}.npkd")
        
        # 推断dtype（从文件大小）
        file_size = os.path.getsize(file_path)
        total_elements = np.prod(shape)
        itemsize = file_size // total_elements
        
        # 根据itemsize推断dtype
        dtype_map = {
            1: np.uint8,
            2: np.float16,
            4: np.float32,
            8: np.float64,
        }
        dtype = dtype_map.get(itemsize, np.float64)
        
        # 打开文件并创建mmap
        file = open(file_path, 'r+b')
        mm = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_WRITE)
        
        # 创建NumPy数组视图
        arr = np.ndarray(
            shape=tuple(shape),
            dtype=dtype,
            buffer=mm
        )
        
        # 保存引用
        self.writable_arrays[array_name] = (file, mm)
        self.array_cache[array_name] = arr
        
        return arr
    
    def save(self, arrays_dict):
        """保存操作变为无操作
        
        因为修改已经直接在文件上进行，无需额外保存
        """
        # 🚀 关键优化：修改已经在文件上，无需操作
        pass
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """关闭所有mmap并flush"""
        for array_name, (file, mm) in self.writable_arrays.items():
            try:
                mm.flush()  # 确保写入磁盘
                mm.close()
                file.close()
            except Exception as e:
                print(f"Warning: Failed to close {array_name}: {e}")
        
        self.writable_arrays.clear()
        self.array_cache.clear()
        return False

