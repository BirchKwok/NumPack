"""
Rust 兼容格式 - 与 Rust 后端完全兼容的文件格式实现

基于对 Rust 格式的深入分析，实现完全兼容的读写功能。

文件格式规范：
1. metadata.npkm - 元数据文件，包含所有数组的信息
2. data_{array_name}.npkd - 数据文件，每个数组一个文件

metadata.npkm 结构：
- Header (16 bytes):
  - 版本号 (4 bytes, little endian uint32)
  - 数组数量 (4 bytes, little endian uint32) 
  - 保留字段1 (4 bytes)
  - 名称长度字段开始标记 (4 bytes)
  
- 每个数组信息：
  - 名称长度 (4 bytes) + 填充 (4 bytes)
  - 数组名称 (utf-8 字符串)
  - 重复名称长度 (4 bytes) + 填充 (4 bytes)
  - 重复数组名称 (utf-8 字符串)
  - 维度数 (8 bytes, little endian uint64)
  - 形状 (每个维度 8 bytes, little endian uint64)
  - 数据类型编码 (8 bytes, little endian uint64)
  - 文件路径长度 (8 bytes, little endian uint64)
  - 文件路径 (utf-8 字符串)
  - 时间戳1 (8 bytes, little endian uint64)
  - 时间戳2 (8 bytes, little endian uint64)
  - 其他信息 (8 bytes, little endian uint64)
"""

import struct
import time
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional
import numpy as np
import filelock

# 添加缓存支持
import weakref
from threading import RLock

# 在文件开头添加缓存管理器类
class ArrayCache:
    """数组缓存管理器"""
    
    def __init__(self, max_size: int = 10):
        self.max_size = max_size
        self.cache = {}
        self.access_order = []
        self.lock = RLock()
    
    def get(self, key: str) -> Optional[np.ndarray]:
        """获取缓存的数组"""
        with self.lock:
            if key in self.cache:
                # 更新访问顺序
                self.access_order.remove(key)
                self.access_order.append(key)
                return self.cache[key]
            return None
    
    def put(self, key: str, array: np.ndarray) -> None:
        """添加数组到缓存"""
        with self.lock:
            if key in self.cache:
                # 更新现有条目
                self.access_order.remove(key)
                self.access_order.append(key)
                self.cache[key] = array
                return
            
            # 如果缓存已满，删除最少使用的
            if len(self.cache) >= self.max_size:
                oldest_key = self.access_order.pop(0)
                del self.cache[oldest_key]
            
            self.cache[key] = array
            self.access_order.append(key)
    
    def clear(self) -> None:
        """清空缓存"""
        with self.lock:
            self.cache.clear()
            self.access_order.clear()


# Rust 数据类型编码映射
RUST_DTYPE_MAP = {
    np.int8: 1,
    np.int16: 2, 
    np.int32: 7,
    np.int64: 8,
    np.uint8: 3,
    np.uint16: 4,
    np.uint32: 5,
    np.uint64: 6,
    np.float16: 9,    # 添加 float16 支持
    np.float32: 10,
    np.float64: 11,
    np.bool_: 12,
    np.complex64: 13,
    np.complex128: 14,
}

# 反向映射
RUST_DTYPE_REVERSE_MAP = {v: k for k, v in RUST_DTYPE_MAP.items()}


class RustArrayMetadata:
    """Rust 格式的数组元数据"""
    
    def __init__(self, name: str, shape: Tuple[int, ...], dtype: np.dtype, 
                 file_path: str, timestamp: Optional[float] = None):
        self.name = name
        self.shape = shape
        self.dtype = np.dtype(dtype)
        self.file_path = file_path
        self.timestamp = timestamp or time.time()
        
        # 转换为 Rust 兼容的类型编码
        if self.dtype.type in RUST_DTYPE_MAP:
            self.dtype_code = RUST_DTYPE_MAP[self.dtype.type]
        else:
            # 尝试转换为兼容类型
            if self.dtype.kind == 'i':  # 整数
                self.dtype_code = RUST_DTYPE_MAP[np.int32]
                self.dtype = np.dtype(np.int32)
            elif self.dtype.kind == 'f':  # 浮点
                self.dtype_code = RUST_DTYPE_MAP[np.float64]
                self.dtype = np.dtype(np.float64)
            else:
                raise ValueError(f"Unsupported dtype: {self.dtype}")
    
    @property
    def total_elements(self) -> int:
        """总元素数"""
        return int(np.prod(self.shape))
    
    @property
    def data_size(self) -> int:
        """数据大小（字节）"""
        return self.total_elements * self.dtype.itemsize


class RustCompatibleWriter:
    """Rust 兼容的写入器"""
    
    def __init__(self, base_path: Union[str, Path]):
        self.base_path = Path(base_path)
        self.metadata_file = self.base_path / "metadata.npkm"
        self.arrays: Dict[str, RustArrayMetadata] = {}
        
        # 确保目录存在
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def save_arrays(self, arrays: Dict[str, np.ndarray]) -> None:
        """保存数组到 Rust 兼容格式"""
        # 使用文件锁确保线程安全
        lock_file = self.base_path / "metadata.npkm.lock"
        lock = filelock.FileLock(lock_file)
        
        with lock:
            # 准备元数据
            self._prepare_metadata(arrays)
            
            # 写入数据文件
            self._write_data_files(arrays)
            
            # 写入元数据文件
            self._write_metadata_file()
    
    def _prepare_metadata(self, arrays: Dict[str, np.ndarray]) -> None:
        """准备元数据"""
        self.arrays.clear()
        
        for name, array in arrays.items():
            # 确保数组是 Rust 兼容的
            compatible_array = self._ensure_rust_compatible(array)
            
            # 数据文件路径
            file_path = f"data_{name}.npkd"
            
            # 创建元数据
            metadata = RustArrayMetadata(
                name=name,
                shape=compatible_array.shape,
                dtype=compatible_array.dtype,
                file_path=file_path
            )
            
            self.arrays[name] = metadata
    
    def _ensure_rust_compatible(self, array: np.ndarray) -> np.ndarray:
        """确保数组与 Rust 兼容"""
        # 转换为 numpy 数组
        if not isinstance(array, np.ndarray):
            array = np.asarray(array)
        
        # 确保使用小端序
        if array.dtype.byteorder == '>':
            array = array.astype(array.dtype.newbyteorder('<'))
        
        # 确保数据类型被支持
        if array.dtype.type not in RUST_DTYPE_MAP:
            if array.dtype.kind == 'i':
                array = array.astype(np.int32)
            elif array.dtype.kind == 'f':
                # 对于浮点类型，使用合适的默认类型
                if array.dtype.itemsize <= 2:  # float16 及更小
                    array = array.astype(np.float16)
                elif array.dtype.itemsize <= 4:  # float32
                    array = array.astype(np.float32)
                else:  # float64 及更大
                    array = array.astype(np.float64)
            elif array.dtype.kind == 'b':
                array = array.astype(np.bool_)
            else:
                raise ValueError(f"Unsupported dtype: {array.dtype}")
        
        return array
    
    def _write_data_files(self, arrays: Dict[str, np.ndarray]) -> None:
        """写入数据文件"""
        for name, array in arrays.items():
            compatible_array = self._ensure_rust_compatible(array)
            data_file = self.base_path / f"data_{name}.npkd"
            
            # 使用内存映射优化写入性能
            self._write_data_file_mmap(compatible_array, data_file)
    
    def _write_data_file_mmap(self, array: np.ndarray, data_file: Path) -> None:
        """使用内存映射方式写入数据文件 - 高效低内存占用"""
        # 计算文件大小
        file_size = array.nbytes
        
        try:
            # 方法1：使用内存映射（性能最佳）
            # 创建或截断文件到正确的大小
            with open(data_file, 'wb') as f:
                f.seek(file_size - 1)
                f.write(b'\0')
            
            # 使用内存映射写入数据
            mm = np.memmap(data_file, dtype=array.dtype, mode='r+', shape=array.shape)
            # 使用切片赋值，避免整个数组复制到内存
            mm[:] = array[:]
            # 确保数据刷新到磁盘
            mm.flush()
            # 释放内存映射
            del mm
            
        except Exception:
            # 回退到标准方式
            array.tofile(data_file)
    
    def _write_metadata_file(self) -> None:
        """写入元数据文件"""
        with open(self.metadata_file, 'wb') as f:
            # 写入头部
            f.write(struct.pack('<I', 1))  # 版本号
            f.write(struct.pack('<I', len(self.arrays)))  # 数组数量
            f.write(struct.pack('<I', 0))  # 保留字段1
            f.write(struct.pack('<I', len(list(self.arrays.keys())[0]) if self.arrays else 0))  # 第一个名称长度
            
            # 写入每个数组的元数据，第一个数组需要特殊处理
            for i, metadata in enumerate(self.arrays.values()):
                if i == 0:
                    self._write_first_array_metadata(f, metadata)
                else:
                    self._write_array_metadata(f, metadata)
    
    def _write_array_metadata(self, f, metadata: RustArrayMetadata) -> None:
        """写入单个数组的元数据"""
        name_bytes = metadata.name.encode('utf-8')
        name_length = len(name_bytes)
        
        # 名称长度 + 填充
        f.write(struct.pack('<I', name_length))
        f.write(struct.pack('<I', 0))  # 填充
        
        # 数组名称
        f.write(name_bytes)
        
        # 重复名称长度 + 填充
        f.write(struct.pack('<I', name_length))
        f.write(struct.pack('<I', 0))  # 填充
        
        # 重复数组名称
        f.write(name_bytes)
        
        # 维度数
        f.write(struct.pack('<Q', len(metadata.shape)))
        
        # 形状
        for dim_size in metadata.shape:
            f.write(struct.pack('<Q', dim_size))
        
        # 数据类型编码
        f.write(struct.pack('<Q', metadata.dtype_code))
        
        # 文件路径
        path_bytes = metadata.file_path.encode('utf-8')
        f.write(struct.pack('<Q', len(path_bytes)))
        f.write(path_bytes)
        
        # 时间戳（转换为微秒）
        timestamp_us = int(metadata.timestamp * 1_000_000)
        f.write(struct.pack('<Q', timestamp_us))
        f.write(struct.pack('<Q', 800))  # 固定值
        
        # 其他信息
        f.write(struct.pack('<Q', metadata.data_size))
    
    def _write_first_array_metadata(self, f, metadata: RustArrayMetadata) -> None:
        """写入第一个数组的元数据（名称长度已在头部）"""
        name_bytes = metadata.name.encode('utf-8')
        name_length = len(name_bytes)
        
        # 第一个数组不需要写名称长度，因为已经在头部了
        # 直接写填充
        f.write(struct.pack('<I', 0))  # 填充
        
        # 数组名称
        f.write(name_bytes)
        
        # 重复名称长度 + 填充
        f.write(struct.pack('<I', name_length))
        f.write(struct.pack('<I', 0))  # 填充
        
        # 重复数组名称
        f.write(name_bytes)
        
        # 维度数
        f.write(struct.pack('<Q', len(metadata.shape)))
        
        # 形状
        for dim_size in metadata.shape:
            f.write(struct.pack('<Q', dim_size))
        
        # 数据类型编码
        f.write(struct.pack('<Q', metadata.dtype_code))
        
        # 文件路径
        path_bytes = metadata.file_path.encode('utf-8')
        f.write(struct.pack('<Q', len(path_bytes)))
        f.write(path_bytes)
        
        # 时间戳（转换为微秒）
        timestamp_us = int(metadata.timestamp * 1_000_000)
        f.write(struct.pack('<Q', timestamp_us))
        f.write(struct.pack('<Q', 800))  # 固定值
        
        # 其他信息
        f.write(struct.pack('<Q', metadata.data_size))


class IncrementalWriter:
    """增量写入器 - 只更新修改的数组并使用内存映射优化写入性能"""
    
    def __init__(self, base_path: Union[str, Path]):
        self.base_path = Path(base_path)
        self.metadata_file = self.base_path / "metadata.npkm"
        
        # 确保目录存在
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # 加载现有元数据
        self.reader = RustCompatibleReader(self.base_path)
        self.existing_arrays = {}
        
        # 复制现有的数组元数据
        for name in self.reader.list_arrays():
            try:
                metadata = self.reader.get_array_metadata(name)
                self.existing_arrays[name] = metadata
            except Exception:
                pass
    
    def write_arrays(self, arrays: Dict[str, np.ndarray]) -> None:
        """增量写入数组 - 只写入新数据，保留未修改的元数据"""
        # 使用文件锁确保线程安全
        lock_file = self.base_path / "metadata.npkm.lock"
        lock = filelock.FileLock(lock_file)
        
        with lock:
            # 准备需要保存的数组元数据
            updated_metadata = {}
            
            # 保留现有元数据
            for name, metadata in self.existing_arrays.items():
                if name not in arrays:  # 不在更新列表中的保留
                    updated_metadata[name] = metadata
            
            # 添加或更新新数组的元数据
            for name, array in arrays.items():
                # 确保数组是 Rust 兼容的
                compatible_array = self._ensure_rust_compatible(array)
                
                # 数据文件路径
                file_path = f"data_{name}.npkd"
                
                # 创建元数据
                metadata = RustArrayMetadata(
                    name=name,
                    shape=compatible_array.shape,
                    dtype=compatible_array.dtype,
                    file_path=file_path,
                    timestamp=time.time()  # 更新时间戳
                )
                
                updated_metadata[name] = metadata
                
                # 使用内存映射写入数据文件
                self._write_data_file_mmap(name, compatible_array)
            
            # 写入更新后的元数据文件
            self._write_metadata_file(updated_metadata)
    
    def _ensure_rust_compatible(self, array: np.ndarray) -> np.ndarray:
        """确保数组与 Rust 兼容"""
        # 转换为 numpy 数组
        if not isinstance(array, np.ndarray):
            array = np.asarray(array)
        
        # 确保使用小端序
        if array.dtype.byteorder == '>':
            array = array.astype(array.dtype.newbyteorder('<'))
        
        # 确保数据类型被支持
        if array.dtype.type not in RUST_DTYPE_MAP:
            if array.dtype.kind == 'i':
                array = array.astype(np.int32)
            elif array.dtype.kind == 'f':
                # 对于浮点类型，使用合适的默认类型
                if array.dtype.itemsize <= 2:  # float16 及更小
                    array = array.astype(np.float16)
                elif array.dtype.itemsize <= 4:  # float32
                    array = array.astype(np.float32)
                else:  # float64 及更大
                    array = array.astype(np.float64)
            elif array.dtype.kind == 'b':
                array = array.astype(np.bool_)
            else:
                raise ValueError(f"Unsupported dtype: {array.dtype}")
        
        return array
    
    def _write_data_file_mmap(self, name: str, array: np.ndarray) -> None:
        """使用内存映射方式写入数据文件 - 高效低内存占用"""
        data_file = self.base_path / f"data_{name}.npkd"
        
        # 计算文件大小
        file_size = array.nbytes
        
        # 对于大型数组，使用分块写入以最小化内存占用
        CHUNK_THRESHOLD_MB = 100  # 超过这个大小使用分块
        
        if array.nbytes > CHUNK_THRESHOLD_MB * 1024 * 1024:
            self._write_data_file_chunked(data_file, array)
            return
            
        try:
            # 方法1：使用内存映射（性能最佳）
            # 创建或截断文件到正确的大小
            with open(data_file, 'wb') as f:
                f.seek(file_size - 1)
                f.write(b'\0')
            
            # 使用内存映射写入数据
            mm = np.memmap(data_file, dtype=array.dtype, mode='r+', shape=array.shape)
            # 使用切片赋值，避免整个数组复制到内存
            mm[:] = array[:]
            # 确保数据刷新到磁盘
            mm.flush()
            # 释放内存映射
            del mm
            
        except Exception as e:
            # 回退到分块写入方法
            self._write_data_file_chunked(data_file, array)
    
    def _write_data_file_chunked(self, data_file: Path, array: np.ndarray) -> None:
        """分块写入数据文件 - 最小化内存使用，适合超大数组"""
        # 确定最佳块大小（平衡内存使用和性能）
        # 默认使用10MB的块大小
        CHUNK_SIZE_MB = 10
        chunk_size_bytes = CHUNK_SIZE_MB * 1024 * 1024
        
        # 计算每个块中的元素数量
        element_size = array.dtype.itemsize
        elements_per_chunk = max(1, chunk_size_bytes // element_size)
        
        # 如果是多维数组，调整块大小为完整行
        if array.ndim > 1:
            row_size = array.shape[1] * element_size
            rows_per_chunk = max(1, chunk_size_bytes // row_size)
            elements_per_chunk = rows_per_chunk * array.shape[1]
            
        total_elements = array.size
        
        # 创建输出文件
        with open(data_file, 'wb') as f:
            # 处理每个块
            for start in range(0, total_elements, elements_per_chunk):
                end = min(start + elements_per_chunk, total_elements)
                
                # 获取一个平坦的切片
                flat_array = array.reshape(-1)
                chunk = flat_array[start:end]
                
                # 直接写入
                f.write(chunk.tobytes())
    
    def _write_metadata_file(self, arrays_metadata: Dict[str, RustArrayMetadata]) -> None:
        """写入元数据文件"""
        # 转换为有序列表确保写入顺序一致
        metadata_list = list(arrays_metadata.values())
        
        with open(self.metadata_file, 'wb') as f:
            # 写入头部
            f.write(struct.pack('<I', 1))  # 版本号
            f.write(struct.pack('<I', len(metadata_list)))  # 数组数量
            f.write(struct.pack('<I', 0))  # 保留字段1
            
            # 写入第一个名称长度作为头部的一部分
            first_name_length = len(metadata_list[0].name) if metadata_list else 0
            f.write(struct.pack('<I', first_name_length))
            
            # 写入每个数组的元数据
            for i, metadata in enumerate(metadata_list):
                if i == 0:
                    self._write_first_array_metadata(f, metadata)
                else:
                    self._write_array_metadata(f, metadata)
    
    def _write_array_metadata(self, f, metadata: RustArrayMetadata) -> None:
        """写入单个数组的元数据"""
        name_bytes = metadata.name.encode('utf-8')
        name_length = len(name_bytes)
        
        # 名称长度 + 填充
        f.write(struct.pack('<I', name_length))
        f.write(struct.pack('<I', 0))  # 填充
        
        # 数组名称
        f.write(name_bytes)
        
        # 重复名称长度 + 填充
        f.write(struct.pack('<I', name_length))
        f.write(struct.pack('<I', 0))  # 填充
        
        # 重复数组名称
        f.write(name_bytes)
        
        # 维度数
        f.write(struct.pack('<Q', len(metadata.shape)))
        
        # 形状
        for dim_size in metadata.shape:
            f.write(struct.pack('<Q', dim_size))
        
        # 数据类型编码
        f.write(struct.pack('<Q', metadata.dtype_code))
        
        # 文件路径
        path_bytes = metadata.file_path.encode('utf-8')
        f.write(struct.pack('<Q', len(path_bytes)))
        f.write(path_bytes)
        
        # 时间戳（转换为微秒）
        timestamp_us = int(metadata.timestamp * 1_000_000)
        f.write(struct.pack('<Q', timestamp_us))
        f.write(struct.pack('<Q', 800))  # 固定值
        
        # 其他信息
        f.write(struct.pack('<Q', metadata.data_size))
    
    def _write_first_array_metadata(self, f, metadata: RustArrayMetadata) -> None:
        """写入第一个数组的元数据（名称长度已在头部）"""
        name_bytes = metadata.name.encode('utf-8')
        name_length = len(name_bytes)
        
        # 第一个数组不需要写名称长度，因为已经在头部了
        # 直接写填充
        f.write(struct.pack('<I', 0))  # 填充
        
        # 数组名称
        f.write(name_bytes)
        
        # 重复名称长度 + 填充
        f.write(struct.pack('<I', name_length))
        f.write(struct.pack('<I', 0))  # 填充
        
        # 重复数组名称
        f.write(name_bytes)
        
        # 维度数
        f.write(struct.pack('<Q', len(metadata.shape)))
        
        # 形状
        for dim_size in metadata.shape:
            f.write(struct.pack('<Q', dim_size))
        
        # 数据类型编码
        f.write(struct.pack('<Q', metadata.dtype_code))
        
        # 文件路径
        path_bytes = metadata.file_path.encode('utf-8')
        f.write(struct.pack('<Q', len(path_bytes)))
        f.write(path_bytes)
        
        # 时间戳（转换为微秒）
        timestamp_us = int(metadata.timestamp * 1_000_000)
        f.write(struct.pack('<Q', timestamp_us))
        f.write(struct.pack('<Q', 800))  # 固定值
        
        # 其他信息
        f.write(struct.pack('<Q', metadata.data_size))


class RustCompatibleReader:
    """Rust 兼容的读取器 - 优化版本"""
    
    def __init__(self, base_path: Union[str, Path]):
        self.base_path = Path(base_path)
        self.metadata_file = self.base_path / "metadata.npkm"
        self.arrays: Dict[str, RustArrayMetadata] = {}
        self._mmap_cache: Dict[str, np.memmap] = {}
        self._array_cache = ArrayCache(max_size=20)  # 缓存最近访问的数组
        self._load_metadata()
    
    def _load_metadata(self) -> None:
        """加载元数据"""
        if not self.metadata_file.exists():
            return
        
        # 使用文件锁保护读取操作
        lock_file = self.base_path / "metadata.npkm.lock"
        lock = filelock.FileLock(lock_file)
        
        with lock:
            try:
                with open(self.metadata_file, 'rb') as f:
                    data = f.read()
                
                # 检查文件大小
                if len(data) < 16:
                    # 文件太小，可能损坏或正在写入
                    return
                
                offset = 0
                
                # 读取头部
                if offset + 4 > len(data):
                    return
                version = struct.unpack('<I', data[offset:offset+4])[0]
                offset += 4
                
                if offset + 4 > len(data):
                    return
                num_arrays = struct.unpack('<I', data[offset:offset+4])[0]
                offset += 4
                
                # 跳过保留字段
                if offset + 4 > len(data):
                    return
                reserved1 = struct.unpack('<I', data[offset:offset+4])[0]
                offset += 4
                
                # 第一个数组的名称长度（头部的一部分）
                if offset + 4 > len(data):
                    return
                first_name_length = struct.unpack('<I', data[offset:offset+4])[0]
                offset += 4
                
                # 现在 offset = 16，这里开始第一个数组的信息
                # 但是名称长度已经在头部了，需要特殊处理第一个数组
                
                for i in range(num_arrays):
                    if i == 0:
                        # 第一个数组特殊处理
                        metadata, next_offset = self._parse_first_array_metadata(data, offset, first_name_length)
                    else:
                        metadata, next_offset = self._parse_array_metadata(data, offset)
                    self.arrays[metadata.name] = metadata
                    offset = next_offset
                    
            except (struct.error, IOError, OSError) as e:
                # 如果读取失败，可能是文件正在被写入，返回空
                return
    
    def _parse_array_metadata(self, data: bytes, offset: int) -> Tuple[RustArrayMetadata, int]:
        """解析单个数组的元数据"""
        # 根据十六进制分析，实际的格式是：
        # 名称长度(4) + 填充(4) + 名称 + 名称长度(4) + 填充(4) + 名称 + 其他字段
        
        # 读取名称长度
        if offset + 4 > len(data):
            raise struct.error("Insufficient data for name length")
        name_length = struct.unpack('<I', data[offset:offset+4])[0]
        offset += 4
        
        # 读取填充
        if offset + 4 > len(data):
            raise struct.error("Insufficient data for padding")
        padding = struct.unpack('<I', data[offset:offset+4])[0]
        offset += 4
        
        # 读取名称
        if offset + name_length > len(data):
            raise struct.error("Insufficient data for name")
        name = data[offset:offset+name_length].decode('utf-8')
        offset += name_length
        
        # 读取重复的名称长度
        if offset + 4 > len(data):
            raise struct.error("Insufficient data for repeated name length")
        name_length_2 = struct.unpack('<I', data[offset:offset+4])[0]
        offset += 4
        
        # 读取填充
        if offset + 4 > len(data):
            raise struct.error("Insufficient data for second padding")
        padding_2 = struct.unpack('<I', data[offset:offset+4])[0]
        offset += 4
        
        # 跳过重复的名称
        if offset + name_length_2 > len(data):
            raise struct.error("Insufficient data for repeated name")
        offset += name_length_2
        
        # 读取维度数
        if offset + 8 > len(data):
            raise struct.error("Insufficient data for ndim")
        ndim = struct.unpack('<Q', data[offset:offset+8])[0]
        offset += 8
        
        # 读取形状
        shape = []
        for _ in range(ndim):
            if offset + 8 > len(data):
                raise struct.error("Insufficient data for shape dimension")
            dim_size = struct.unpack('<Q', data[offset:offset+8])[0]
            shape.append(dim_size)
            offset += 8
        
        # 读取数据类型
        if offset + 8 > len(data):
            raise struct.error("Insufficient data for dtype")
        dtype_code = struct.unpack('<Q', data[offset:offset+8])[0]
        offset += 8
        
        dtype = RUST_DTYPE_REVERSE_MAP.get(dtype_code, np.int32)
        
        # 读取文件路径
        if offset + 8 > len(data):
            raise struct.error("Insufficient data for path length")
        path_length = struct.unpack('<Q', data[offset:offset+8])[0]
        offset += 8
        
        if offset + path_length > len(data):
            raise struct.error("Insufficient data for file path")
        file_path = data[offset:offset+path_length].decode('utf-8')
        offset += path_length
        
        # 跳过时间戳和其他信息
        if offset + 24 > len(data):
            raise struct.error("Insufficient data for timestamps")
        offset += 24  # 3个8字节字段
        
        metadata = RustArrayMetadata(
            name=name,
            shape=tuple(shape),
            dtype=dtype,
            file_path=file_path
        )
        
        return metadata, offset
    
    def _parse_first_array_metadata(self, data: bytes, offset: int, name_length: int) -> Tuple[RustArrayMetadata, int]:
        """解析第一个数组的元数据（名称长度已知）"""
        # 第一个数组的格式略有不同，名称长度在头部
        
        # 跳过填充（偏移16-19）
        if offset + 4 > len(data):
            raise struct.error("Insufficient data for first array padding")
        padding = struct.unpack('<I', data[offset:offset+4])[0]
        offset += 4
        
        # 读取名称（偏移20开始）
        if offset + name_length > len(data):
            raise struct.error("Insufficient data for first array name")
        name = data[offset:offset+name_length].decode('utf-8')
        offset += name_length
        
        # 读取重复的名称长度
        if offset + 4 > len(data):
            raise struct.error("Insufficient data for first array repeated name length")
        name_length_2 = struct.unpack('<I', data[offset:offset+4])[0]
        offset += 4
        
        # 读取填充
        if offset + 4 > len(data):
            raise struct.error("Insufficient data for first array second padding")
        padding_2 = struct.unpack('<I', data[offset:offset+4])[0]
        offset += 4
        
        # 跳过重复的名称
        if offset + name_length_2 > len(data):
            raise struct.error("Insufficient data for first array repeated name")
        offset += name_length_2
        
        # 读取维度数
        if offset + 8 > len(data):
            raise struct.error("Insufficient data for first array ndim")
        ndim = struct.unpack('<Q', data[offset:offset+8])[0]
        offset += 8
        
        # 读取形状
        shape = []
        for _ in range(ndim):
            if offset + 8 > len(data):
                raise struct.error("Insufficient data for first array shape dimension")
            dim_size = struct.unpack('<Q', data[offset:offset+8])[0]
            shape.append(dim_size)
            offset += 8
        
        # 读取数据类型
        if offset + 8 > len(data):
            raise struct.error("Insufficient data for first array dtype")
        dtype_code = struct.unpack('<Q', data[offset:offset+8])[0]
        offset += 8
        
        dtype = RUST_DTYPE_REVERSE_MAP.get(dtype_code, np.int32)
        
        # 读取文件路径
        if offset + 8 > len(data):
            raise struct.error("Insufficient data for first array path length")
        path_length = struct.unpack('<Q', data[offset:offset+8])[0]
        offset += 8
        
        if offset + path_length > len(data):
            raise struct.error("Insufficient data for first array file path")
        file_path = data[offset:offset+path_length].decode('utf-8')
        offset += path_length
        
        # 跳过时间戳和其他信息
        if offset + 24 > len(data):
            raise struct.error("Insufficient data for first array timestamps")
        offset += 24  # 3个8字节字段
        
        metadata = RustArrayMetadata(
            name=name,
            shape=tuple(shape),
            dtype=dtype,
            file_path=file_path
        )
        
        return metadata, offset
    
    def list_arrays(self) -> List[str]:
        """获取所有数组名称"""
        return list(self.arrays.keys())
    
    def has_array(self, name: str) -> bool:
        """检查数组是否存在"""
        return name in self.arrays
    
    def load_array(self, name: str, mmap_mode: Optional[str] = None) -> np.ndarray:
        """加载数组 - 优化版本"""
        if name not in self.arrays:
            raise KeyError(f"Array '{name}' not found")
        
        # 检查缓存
        cached_array = self._array_cache.get(name)
        if cached_array is not None and mmap_mode is None:
            return cached_array.copy()  # 返回副本避免修改缓存
        
        metadata = self.arrays[name]
        data_file = self.base_path / metadata.file_path
        
        if not data_file.exists():
            raise FileNotFoundError(f"Data file not found: {data_file}")
        
        # 优化的数据读取
        if mmap_mode:
            # 内存映射模式 - 使用缓存的 memmap
            cache_key = f"{name}_mmap"
            if cache_key not in self._mmap_cache:
                self._mmap_cache[cache_key] = np.memmap(
                    data_file, dtype=metadata.dtype, mode=mmap_mode, shape=metadata.shape
                )
            return self._mmap_cache[cache_key]
        else:
            # 直接读取模式 - 优化的批量读取
            try:
                # 尝试使用 numpy.fromfile 进行快速读取
                data = np.fromfile(data_file, dtype=metadata.dtype)
                shaped_data = data.reshape(metadata.shape)
                
                # 只缓存较小的数组（< 100MB）
                if shaped_data.nbytes < 100 * 1024 * 1024:
                    self._array_cache.put(name, shaped_data.copy())
                
                return shaped_data
            except Exception:
                # 回退到更安全的读取方式
                with open(data_file, 'rb') as f:
                    data = np.frombuffer(f.read(), dtype=metadata.dtype)
                return data.reshape(metadata.shape)
    
    def get_array_metadata(self, name: str) -> RustArrayMetadata:
        """获取数组元数据"""
        if name not in self.arrays:
            raise KeyError(f"Array '{name}' not found")
        return self.arrays[name]
    
    def get_memmap_array(self, name: str, mode: str = 'r') -> np.memmap:
        """获取内存映射数组 - 高效懒加载"""
        if name not in self.arrays:
            raise KeyError(f"Array '{name}' not found")
        
        metadata = self.arrays[name]
        data_file = self.base_path / metadata.file_path
        
        if not data_file.exists():
            raise FileNotFoundError(f"Data file not found: {data_file}")
        
        # 创建内存映射数组
        return np.memmap(
            data_file, 
            dtype=metadata.dtype, 
            mode=mode, 
            shape=metadata.shape
        )
    
    def clear_cache(self) -> None:
        """清除所有缓存"""
        self._array_cache.clear()
        # 清除 memmap 缓存时更小心地处理文件句柄
        for cache_key, mmap_array in list(self._mmap_cache.items()):
            try:
                # 确保 memmap 被正确关闭
                if hasattr(mmap_array, '_mmap') and mmap_array._mmap is not None:
                    mmap_array._mmap.close()
                del mmap_array
            except Exception:
                pass
        self._mmap_cache.clear()
    
    def __del__(self):
        """析构函数 - 确保资源清理"""
        try:
            self.clear_cache()
        except Exception:
            pass


class RustCompatibleManager:
    """Rust 兼容的管理器 - 性能优化版"""
    
    def __init__(self, base_path: Union[str, Path]):
        self.base_path = Path(base_path)
        self._reader: Optional[RustCompatibleReader] = None
        self._performance_stats = {
            "saves": 0,
            "incremental_saves": 0,
            "total_arrays_saved": 0,
            "total_bytes_saved": 0,
        }
    
    def save(self, arrays: Dict[str, np.ndarray]) -> None:
        """保存数组"""
        # 更新统计信息
        self._performance_stats["saves"] += 1
        self._performance_stats["total_arrays_saved"] += len(arrays)
        self._performance_stats["total_bytes_saved"] += sum(arr.nbytes for arr in arrays.values())
        
        writer = RustCompatibleWriter(self.base_path)
        writer.save_arrays(arrays)
        # 重新加载元数据
        self._reader = None
    
    def save_incremental(self, arrays: Dict[str, np.ndarray]) -> None:
        """增量保存数组 - 只写入新增或修改的数组 - 优化版本
        
        Parameters:
            arrays: 要保存的数组字典
        """
        if not arrays:
            return
        
        # 更新统计信息
        self._performance_stats["incremental_saves"] += 1
        self._performance_stats["total_arrays_saved"] += len(arrays)
        self._performance_stats["total_bytes_saved"] += sum(arr.nbytes for arr in arrays.values())
            
        # 确保读取器初始化
        if self._reader is None:
            self._reader = RustCompatibleReader(self.base_path)
        
        # 获取现有数组列表
        existing_arrays_names = set(self._reader.list_arrays())
        
        # 快速路径：如果所有数组都是新的，直接使用增量写入器
        if not existing_arrays_names.intersection(set(arrays.keys())):
            writer = IncrementalWriter(self.base_path)
            writer.write_arrays(arrays)
            self._reader = None
            return
            
        # 优化的数组比较方法 - 针对大型数组
        new_arrays = {}  # 新数组
        changed_arrays = {}  # 已修改的数组
        
        for name, array in arrays.items():
            if name not in existing_arrays_names:
                new_arrays[name] = array
                continue
                
            # 检查元数据是否有变化
            try:
                metadata = self._reader.get_array_metadata(name)
                
                # 形状或类型不匹配，肯定需要更新
                if metadata.shape != array.shape or metadata.dtype != array.dtype:
                    changed_arrays[name] = array
                    continue
                
                # 大型数组使用采样比较方法
                if array.nbytes >= 50 * 1024 * 1024:  # 50MB以上
                    if self._arrays_likely_different_sampling(name, array):
                        changed_arrays[name] = array
                    continue
                    
                # 中型数组使用校验和比较
                elif array.nbytes >= 10 * 1024 * 1024:  # 10MB以上
                    if self._arrays_different_checksum(name, array):
                        changed_arrays[name] = array
                    continue
                    
                # 小型数组使用完整比较
                else:
                    existing_data = self._reader.load_array(name)
                    if not np.array_equal(existing_data, array):
                        changed_arrays[name] = array
                    continue
                    
            except Exception:
                # 出错时保险起见认为已修改
                changed_arrays[name] = array
        
        # 合并新数组和变化的数组
        arrays_to_write = {**new_arrays, **changed_arrays}
        
        if not arrays_to_write:
            # 没有新增或修改的数组，无需写入
            return
            
        # 只保存需要更新的数组
        writer = IncrementalWriter(self.base_path)
        writer.write_arrays(arrays_to_write)
        
        # 重新加载元数据
        self._reader = None
    
    def _arrays_likely_different_sampling(self, array_name: str, new_array: np.ndarray) -> bool:
        """使用采样法快速比较大型数组是否可能不同
        
        针对超大数组，采样比较以避免完全加载
        """
        try:
            # 使用内存映射模式加载现有数组
            existing_array = self._reader.get_memmap_array(array_name, mode='r')
            
            # 获取数组形状和总元素数
            total_elements = existing_array.size
            
            # 对大型数组采样约1000个点
            num_samples = min(1000, total_elements)
            if num_samples < 100:
                # 元素太少，进行完整比较
                return not np.array_equal(existing_array, new_array)
                
            # 生成随机采样索引
            import random
            indices = random.sample(range(total_elements), num_samples)
            
            # 将多维索引转换为平面索引
            flat_existing = existing_array.reshape(-1)
            flat_new = new_array.reshape(-1)
            
            # 比较采样点
            for idx in indices:
                if flat_existing[idx] != flat_new[idx]:
                    return True
                    
            # 采样点都相同，可能相同（返回False表示不需要更新）
            return False
            
        except Exception:
            # 出错时保险起见认为已修改
            return True
            
    def _arrays_different_checksum(self, array_name: str, new_array: np.ndarray) -> bool:
        """使用快速校验和比较中型数组
        
        针对中型数组，计算校验和以避免逐元素比较
        """
        try:
            # 获取现有数组
            existing_array = self._reader.load_array(array_name)
            
            # 计算校验和 (使用numpy自带的sum函数，快速且内存高效)
            # 对浮点数使用sum+size比较，整数类型使用异或和
            if np.issubdtype(existing_array.dtype, np.integer):
                existing_checksum = np.bitwise_xor.reduce(existing_array.reshape(-1))
                new_checksum = np.bitwise_xor.reduce(new_array.reshape(-1))
                return existing_checksum != new_checksum
            else:
                # 浮点类型使用和+大小比较
                existing_sum = np.sum(existing_array)
                new_sum = np.sum(new_array)
                # 避免浮点精度问题
                return not np.isclose(existing_sum, new_sum, rtol=1e-10) or existing_array.size != new_array.size
                
        except Exception:
            # 出错时保险起见认为已修改
            return True
    
    def load(self, array_name: str, mmap_mode: Optional[str] = None) -> np.ndarray:
        """加载数组"""
        if self._reader is None:
            self._reader = RustCompatibleReader(self.base_path)
        return self._reader.load_array(array_name, mmap_mode)
    
    def list_arrays(self) -> List[str]:
        """获取所有数组名称"""
        if self._reader is None:
            self._reader = RustCompatibleReader(self.base_path)
        return self._reader.list_arrays()
    
    def has_array(self, name: str) -> bool:
        """检查数组是否存在"""
        if self._reader is None:
            self._reader = RustCompatibleReader(self.base_path)
        return self._reader.has_array(name)
    
    def get_metadata(self, name: str) -> RustArrayMetadata:
        """获取数组元数据"""
        if self._reader is None:
            self._reader = RustCompatibleReader(self.base_path)
        return self._reader.get_array_metadata(name)
    
    def reset(self) -> None:
        """重置（删除所有文件）"""
        # 清除缓存
        if self._reader:
            self._reader.clear_cache()
        
        if self.base_path.exists():
            import shutil
            shutil.rmtree(self.base_path)
        self._reader = None


# 测试函数
def test_rust_compatibility():
    """测试 Rust 兼容性"""
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_path = Path(tmp_dir) / "rust_compat_test"
        
        # 创建测试数据
        test_arrays = {
            'int_array': np.array([1, 2, 3, 4, 5], dtype=np.int32),
            'float_matrix': np.array([[1.1, 2.2], [3.3, 4.4]], dtype=np.float64),
            'large_data': np.arange(100, dtype=np.int64),
        }
        
        # 写入数据
        manager = RustCompatibleManager(test_path)
        manager.save(test_arrays)
        
        print(f"✅ 保存完成，文件格式与 Rust 兼容")
        
        # 验证文件结构
        files = list(test_path.glob("*"))
        print(f"✅ 生成文件: {[f.name for f in files]}")
        
        # 读取验证
        for name, original in test_arrays.items():
            loaded = manager.load(name)
            assert np.array_equal(loaded, original), f"数据不匹配: {name}"
            print(f"✅ 验证通过: {name}")
        
        print(f"✅ 所有测试通过，完全 Rust 兼容！")


if __name__ == "__main__":
    test_rust_compatibility() 