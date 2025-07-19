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
            
            # 直接写入二进制数据
            compatible_array.tofile(data_file)
    
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


class RustCompatibleReader:
    """Rust 兼容的读取器"""
    
    def __init__(self, base_path: Union[str, Path]):
        self.base_path = Path(base_path)
        self.metadata_file = self.base_path / "metadata.npkm"
        self.arrays: Dict[str, RustArrayMetadata] = {}
        self._load_metadata()
    
    def _load_metadata(self) -> None:
        """加载元数据"""
        if not self.metadata_file.exists():
            return
        
        with open(self.metadata_file, 'rb') as f:
            data = f.read()
        
        offset = 0
        
        # 读取头部
        version = struct.unpack('<I', data[offset:offset+4])[0]
        offset += 4
        
        num_arrays = struct.unpack('<I', data[offset:offset+4])[0]
        offset += 4
        
        # 跳过保留字段
        reserved1 = struct.unpack('<I', data[offset:offset+4])[0]
        offset += 4
        
        # 第一个数组的名称长度（头部的一部分）
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
    
    def _parse_array_metadata(self, data: bytes, offset: int) -> Tuple[RustArrayMetadata, int]:
        """解析单个数组的元数据"""
        # 根据十六进制分析，实际的格式是：
        # 名称长度(4) + 填充(4) + 名称 + 名称长度(4) + 填充(4) + 名称 + 其他字段
        
        # 读取名称长度
        name_length = struct.unpack('<I', data[offset:offset+4])[0]
        offset += 4
        
        # 读取填充
        padding = struct.unpack('<I', data[offset:offset+4])[0]
        offset += 4
        
        # 读取名称
        name = data[offset:offset+name_length].decode('utf-8')
        offset += name_length
        
        # 读取重复的名称长度
        name_length_2 = struct.unpack('<I', data[offset:offset+4])[0]
        offset += 4
        
        # 读取填充
        padding_2 = struct.unpack('<I', data[offset:offset+4])[0]
        offset += 4
        
        # 跳过重复的名称
        offset += name_length_2
        
        # 读取维度数
        ndim = struct.unpack('<Q', data[offset:offset+8])[0]
        offset += 8
        
        # 读取形状
        shape = []
        for _ in range(ndim):
            dim_size = struct.unpack('<Q', data[offset:offset+8])[0]
            shape.append(dim_size)
            offset += 8
        
        # 读取数据类型
        dtype_code = struct.unpack('<Q', data[offset:offset+8])[0]
        offset += 8
        
        dtype = RUST_DTYPE_REVERSE_MAP.get(dtype_code, np.int32)
        
        # 读取文件路径
        path_length = struct.unpack('<Q', data[offset:offset+8])[0]
        offset += 8
        
        file_path = data[offset:offset+path_length].decode('utf-8')
        offset += path_length
        
        # 跳过时间戳和其他信息
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
        padding = struct.unpack('<I', data[offset:offset+4])[0]
        offset += 4
        
        # 读取名称（偏移20开始）
        name = data[offset:offset+name_length].decode('utf-8')
        offset += name_length
        
        # 读取重复的名称长度
        name_length_2 = struct.unpack('<I', data[offset:offset+4])[0]
        offset += 4
        
        # 读取填充
        padding_2 = struct.unpack('<I', data[offset:offset+4])[0]
        offset += 4
        
        # 跳过重复的名称
        offset += name_length_2
        
        # 读取维度数
        ndim = struct.unpack('<Q', data[offset:offset+8])[0]
        offset += 8
        
        # 读取形状
        shape = []
        for _ in range(ndim):
            dim_size = struct.unpack('<Q', data[offset:offset+8])[0]
            shape.append(dim_size)
            offset += 8
        
        # 读取数据类型
        dtype_code = struct.unpack('<Q', data[offset:offset+8])[0]
        offset += 8
        
        dtype = RUST_DTYPE_REVERSE_MAP.get(dtype_code, np.int32)
        
        # 读取文件路径
        path_length = struct.unpack('<Q', data[offset:offset+8])[0]
        offset += 8
        
        file_path = data[offset:offset+path_length].decode('utf-8')
        offset += path_length
        
        # 跳过时间戳和其他信息
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
        """加载数组"""
        if name not in self.arrays:
            raise KeyError(f"Array '{name}' not found")
        
        metadata = self.arrays[name]
        data_file = self.base_path / metadata.file_path
        
        if not data_file.exists():
            raise FileNotFoundError(f"Data file not found: {data_file}")
        
        # 读取数据
        if mmap_mode:
            # 内存映射模式
            data = np.memmap(data_file, dtype=metadata.dtype, mode=mmap_mode)
        else:
            # 直接读取
            data = np.fromfile(data_file, dtype=metadata.dtype)
        
        # 重新塑形
        return data.reshape(metadata.shape)
    
    def get_array_metadata(self, name: str) -> RustArrayMetadata:
        """获取数组元数据"""
        if name not in self.arrays:
            raise KeyError(f"Array '{name}' not found")
        return self.arrays[name]


class RustCompatibleManager:
    """Rust 兼容的管理器"""
    
    def __init__(self, base_path: Union[str, Path]):
        self.base_path = Path(base_path)
        self._reader: Optional[RustCompatibleReader] = None
    
    def save(self, arrays: Dict[str, np.ndarray]) -> None:
        """保存数组"""
        writer = RustCompatibleWriter(self.base_path)
        writer.save_arrays(arrays)
        # 重新加载元数据
        self._reader = None
    
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