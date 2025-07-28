"""
二进制兼容格式模块

提供与 Rust 后端完全兼容的高性能二进制元数据格式，
完全替代 MessagePack 格式，提供更快的序列化/反序列化性能。
"""

import os
import struct
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from dataclasses import dataclass
from enum import IntEnum


# 二进制格式魔数 (ASCII: "NPKB")
BINARY_MAGIC = 0x424B504E

# 当前二进制格式版本
BINARY_VERSION = 1


class BinaryDataType(IntEnum):
    """二进制格式的数据类型枚举 - 与 Rust 后端保持一致"""
    BOOL = 0
    UINT8 = 1
    UINT16 = 2
    UINT32 = 3
    UINT64 = 4
    INT8 = 5
    INT16 = 6
    INT32 = 7
    INT64 = 8
    FLOAT16 = 9
    FLOAT32 = 10
    FLOAT64 = 11
    COMPLEX64 = 12
    COMPLEX128 = 13
    
    def size_bytes(self) -> int:
        """获取数据类型的字节大小"""
        size_map = {
            BinaryDataType.BOOL: 1,
            BinaryDataType.UINT8: 1,
            BinaryDataType.UINT16: 2,
            BinaryDataType.UINT32: 4,
            BinaryDataType.UINT64: 8,
            BinaryDataType.INT8: 1,
            BinaryDataType.INT16: 2,
            BinaryDataType.INT32: 4,
            BinaryDataType.INT64: 8,
            BinaryDataType.FLOAT16: 2,
            BinaryDataType.FLOAT32: 4,
            BinaryDataType.FLOAT64: 8,
            BinaryDataType.COMPLEX64: 8,
            BinaryDataType.COMPLEX128: 16,
        }
        return size_map[self]
    
    @classmethod
    def from_numpy_dtype(cls, dtype: np.dtype) -> 'BinaryDataType':
        """从 NumPy dtype 转换为 BinaryDataType"""
        dtype_map = {
            np.bool_: cls.BOOL,
            np.uint8: cls.UINT8,
            np.uint16: cls.UINT16,
            np.uint32: cls.UINT32,
            np.uint64: cls.UINT64,
            np.int8: cls.INT8,
            np.int16: cls.INT16,
            np.int32: cls.INT32,
            np.int64: cls.INT64,
            np.float16: cls.FLOAT16,
            np.float32: cls.FLOAT32,
            np.float64: cls.FLOAT64,
            np.complex64: cls.COMPLEX64,
            np.complex128: cls.COMPLEX128,
        }
        return dtype_map.get(dtype.type, cls.INT32)
    
    def to_numpy_dtype(self) -> np.dtype:
        """转换为 NumPy dtype"""
        dtype_map = {
            BinaryDataType.BOOL: np.bool_,
            BinaryDataType.UINT8: np.uint8,
            BinaryDataType.UINT16: np.uint16,
            BinaryDataType.UINT32: np.uint32,
            BinaryDataType.UINT64: np.uint64,
            BinaryDataType.INT8: np.int8,
            BinaryDataType.INT16: np.int16,
            BinaryDataType.INT32: np.int32,
            BinaryDataType.INT64: np.int64,
            BinaryDataType.FLOAT16: np.float16,
            BinaryDataType.FLOAT32: np.float32,
            BinaryDataType.FLOAT64: np.float64,
            BinaryDataType.COMPLEX64: np.complex64,
            BinaryDataType.COMPLEX128: np.complex128,
        }
        return np.dtype(dtype_map[self])


class CompressionAlgorithm(IntEnum):
    """压缩算法枚举"""
    NONE = 0
    ZSTD = 1
    
    @classmethod
    def from_string(cls, s: str) -> 'CompressionAlgorithm':
        """从字符串转换为压缩算法"""
        if s.lower() == "zstd":
            return cls.ZSTD
        return cls.NONE
    
    def to_string(self) -> str:
        """转换为字符串"""
        if self == CompressionAlgorithm.ZSTD:
            return "zstd"
        return "none"


@dataclass
class BinaryBlockInfo:
    """块信息"""
    offset: int
    original_size: int
    compressed_size: int


@dataclass
class BinaryBlockCompressionInfo:
    """块压缩信息"""
    enabled: bool
    block_size: int
    num_blocks: int
    blocks: List[BinaryBlockInfo]


@dataclass
class BinaryCompressionInfo:
    """压缩信息"""
    algorithm: CompressionAlgorithm = CompressionAlgorithm.NONE
    level: int = 0
    original_size: int = 0
    compressed_size: int = 0
    block_compression: Optional[BinaryBlockCompressionInfo] = None


@dataclass
class BinaryArrayMetadata:
    """二进制格式的数组元数据"""
    name: str
    shape: List[int]
    data_file: str
    last_modified: int
    size_bytes: int
    dtype: BinaryDataType
    compression: BinaryCompressionInfo
    
    @property
    def total_elements(self) -> int:
        """计算总元素数量"""
        result = 1
        for dim in self.shape:
            result *= dim
        return result
    
    @classmethod
    def from_numpy_array(cls, name: str, array: np.ndarray, data_file: str) -> 'BinaryArrayMetadata':
        """从 NumPy 数组创建元数据"""
        dtype = BinaryDataType.from_numpy_dtype(array.dtype)
        total_elements = array.size
        size_bytes = total_elements * dtype.size_bytes()
        
        return cls(
            name=name,
            shape=list(array.shape),
            data_file=data_file,
            last_modified=int(time.time() * 1_000_000),  # 微秒
            size_bytes=size_bytes,
            dtype=dtype,
            compression=BinaryCompressionInfo()
        )
    
    def write_to_stream(self, stream) -> None:
        """写入元数据到二进制流"""
        # 写入名称长度和名称
        name_bytes = self.name.encode('utf-8')
        stream.write(struct.pack('<I', len(name_bytes)))
        stream.write(name_bytes)
        
        # 写入形状
        stream.write(struct.pack('<I', len(self.shape)))
        for dim in self.shape:
            stream.write(struct.pack('<Q', dim))
        
        # 写入数据文件名
        data_file_bytes = self.data_file.encode('utf-8')
        stream.write(struct.pack('<I', len(data_file_bytes)))
        stream.write(data_file_bytes)
        
        # 写入基本信息
        stream.write(struct.pack('<Q', self.last_modified))
        stream.write(struct.pack('<Q', self.size_bytes))
        stream.write(struct.pack('<B', self.dtype))
        
        # 写入压缩信息
        stream.write(struct.pack('<B', self.compression.algorithm))
        stream.write(struct.pack('<I', self.compression.level))
        stream.write(struct.pack('<Q', self.compression.original_size))
        stream.write(struct.pack('<Q', self.compression.compressed_size))
        
        # 写入块压缩信息
        if self.compression.block_compression is not None:
            stream.write(struct.pack('<B', 1))  # 有块压缩信息
            block_info = self.compression.block_compression
            stream.write(struct.pack('<B', 1 if block_info.enabled else 0))
            stream.write(struct.pack('<Q', block_info.block_size))
            stream.write(struct.pack('<Q', block_info.num_blocks))
            stream.write(struct.pack('<I', len(block_info.blocks)))
            
            for block in block_info.blocks:
                stream.write(struct.pack('<Q', block.offset))
                stream.write(struct.pack('<Q', block.original_size))
                stream.write(struct.pack('<Q', block.compressed_size))
        else:
            stream.write(struct.pack('<B', 0))  # 没有块压缩信息
    
    @classmethod
    def read_from_stream(cls, stream) -> 'BinaryArrayMetadata':
        """从二进制流读取元数据"""
        # 读取名称
        name_len = struct.unpack('<I', stream.read(4))[0]
        name = stream.read(name_len).decode('utf-8')
        
        # 读取形状
        shape_len = struct.unpack('<I', stream.read(4))[0]
        shape = []
        for _ in range(shape_len):
            dim = struct.unpack('<Q', stream.read(8))[0]
            shape.append(dim)
        
        # 读取数据文件名
        data_file_len = struct.unpack('<I', stream.read(4))[0]
        data_file = stream.read(data_file_len).decode('utf-8')
        
        # 读取基本信息
        last_modified = struct.unpack('<Q', stream.read(8))[0]
        size_bytes = struct.unpack('<Q', stream.read(8))[0]
        dtype = BinaryDataType(struct.unpack('<B', stream.read(1))[0])
        
        # 读取压缩信息
        algorithm = CompressionAlgorithm(struct.unpack('<B', stream.read(1))[0])
        level = struct.unpack('<I', stream.read(4))[0]
        original_size = struct.unpack('<Q', stream.read(8))[0]
        compressed_size = struct.unpack('<Q', stream.read(8))[0]
        
        # 读取块压缩信息
        has_block_info = struct.unpack('<B', stream.read(1))[0] != 0
        
        block_compression = None
        if has_block_info:
            enabled = struct.unpack('<B', stream.read(1))[0] != 0
            block_size = struct.unpack('<Q', stream.read(8))[0]
            num_blocks = struct.unpack('<Q', stream.read(8))[0]
            blocks_len = struct.unpack('<I', stream.read(4))[0]
            
            blocks = []
            for _ in range(blocks_len):
                offset = struct.unpack('<Q', stream.read(8))[0]
                original_size_block = struct.unpack('<Q', stream.read(8))[0]
                compressed_size_block = struct.unpack('<Q', stream.read(8))[0]
                blocks.append(BinaryBlockInfo(offset, original_size_block, compressed_size_block))
            
            block_compression = BinaryBlockCompressionInfo(
                enabled=enabled,
                block_size=block_size,
                num_blocks=num_blocks,
                blocks=blocks
            )
        
        compression = BinaryCompressionInfo(
            algorithm=algorithm,
            level=level,
            original_size=original_size,
            compressed_size=compressed_size,
            block_compression=block_compression
        )
        
        return cls(
            name=name,
            shape=shape,
            data_file=data_file,
            last_modified=last_modified,
            size_bytes=size_bytes,
            dtype=dtype,
            compression=compression
        )


class BinaryMetadataStore:
    """二进制格式的元数据存储"""
    
    def __init__(self):
        self.version = BINARY_VERSION
        self.arrays: Dict[str, BinaryArrayMetadata] = {}
        self.total_size = 0
    
    @classmethod
    def load(cls, path: Path) -> 'BinaryMetadataStore':
        """从文件加载元数据存储"""
        if not path.exists():
            return cls()
        
        with open(path, 'rb') as f:
            # 读取魔数
            magic = struct.unpack('<I', f.read(4))[0]
            if magic != BINARY_MAGIC:
                raise ValueError("Invalid magic number")
            
            # 读取版本
            version = struct.unpack('<I', f.read(4))[0]
            
            # 读取总大小
            total_size = struct.unpack('<Q', f.read(8))[0]
            
            # 读取数组数量
            arrays_count = struct.unpack('<I', f.read(4))[0]
            
            # 读取数组元数据
            arrays = {}
            for _ in range(arrays_count):
                meta = BinaryArrayMetadata.read_from_stream(f)
                arrays[meta.name] = meta
            
            store = cls()
            store.version = version
            store.arrays = arrays
            store.total_size = total_size
            return store
    
    def save(self, path: Path) -> None:
        """保存元数据存储到文件"""
        temp_path = path.with_suffix('.tmp')
        
        with open(temp_path, 'wb') as f:
            # 写入魔数
            f.write(struct.pack('<I', BINARY_MAGIC))
            
            # 写入版本
            f.write(struct.pack('<I', self.version))
            
            # 写入总大小
            f.write(struct.pack('<Q', self.total_size))
            
            # 写入数组数量
            f.write(struct.pack('<I', len(self.arrays)))
            
            # 写入每个数组的元数据
            for meta in self.arrays.values():
                meta.write_to_stream(f)
        
        # 原子性替换
        temp_path.replace(path)
    
    def add_array(self, meta: BinaryArrayMetadata) -> None:
        """添加数组元数据"""
        if meta.name in self.arrays:
            self.total_size -= self.arrays[meta.name].size_bytes
        self.total_size += meta.size_bytes
        self.arrays[meta.name] = meta
    
    def remove_array(self, name: str) -> bool:
        """删除数组元数据"""
        if name in self.arrays:
            self.total_size -= self.arrays[name].size_bytes
            del self.arrays[name]
            return True
        return False
    
    def get_array(self, name: str) -> Optional[BinaryArrayMetadata]:
        """获取数组元数据"""
        return self.arrays.get(name)
    
    def list_arrays(self) -> List[str]:
        """列出所有数组名称"""
        return list(self.arrays.keys())
    
    def has_array(self, name: str) -> bool:
        """检查数组是否存在"""
        return name in self.arrays


class BinaryCompatibleManager:
    """二进制兼容格式管理器 - 与 Rust 后端完全兼容"""
    
    def __init__(self, directory: Union[str, Path]):
        self.directory = Path(directory)
        self.metadata_path = self.directory / "metadata.npkm"
        self._store: Optional[BinaryMetadataStore] = None
        
        # 确保目录存在
        self.directory.mkdir(parents=True, exist_ok=True)
    
    @property
    def store(self) -> BinaryMetadataStore:
        """获取元数据存储（延迟加载）"""
        if self._store is None:
            self._store = BinaryMetadataStore.load(self.metadata_path)
        return self._store
    
    def save_metadata(self) -> None:
        """保存元数据到磁盘"""
        if self._store is not None:
            self._store.save(self.metadata_path)
    
    def add_array(self, name: str, array: np.ndarray) -> str:
        """添加数组到存储"""
        data_file = f"data_{name}.npkd"
        data_path = self.directory / data_file
        
        # 保存原始数据文件
        array.astype(array.dtype, copy=False).tobytes('C')
        with open(data_path, 'wb') as f:
            f.write(array.astype(array.dtype, copy=False).tobytes('C'))
        
        # 创建元数据
        metadata = BinaryArrayMetadata.from_numpy_array(name, array, data_file)
        self.store.add_array(metadata)
        
        # 保存元数据
        self.save_metadata()
        
        return data_file
    
    def get_metadata(self, name: str) -> Optional[BinaryArrayMetadata]:
        """获取数组元数据"""
        return self.store.get_array(name)
    
    def get_data_path(self, name: str) -> Optional[Path]:
        """获取数据文件路径"""
        metadata = self.get_metadata(name)
        if metadata:
            return self.directory / metadata.data_file
        return None
    
    def list_arrays(self) -> List[str]:
        """列出所有数组名称"""
        return self.store.list_arrays()
    
    def has_array(self, name: str) -> bool:
        """检查数组是否存在"""
        return self.store.has_array(name)
    
    def remove_array(self, name: str) -> bool:
        """删除数组"""
        metadata = self.get_metadata(name)
        if metadata:
            # 删除数据文件
            data_path = self.directory / metadata.data_file
            if data_path.exists():
                data_path.unlink()
            
            # 删除元数据
            result = self.store.remove_array(name)
            self.save_metadata()
            return result
        return False
    
    def load_array(self, name: str) -> Optional[np.ndarray]:
        """加载完整数组"""
        metadata = self.get_metadata(name)
        if not metadata:
            return None
        
        data_path = self.directory / metadata.data_file
        if not data_path.exists():
            return None
        
        # 读取数据
        with open(data_path, 'rb') as f:
            data = f.read()
        
        # 转换为 NumPy 数组
        dtype = metadata.dtype.to_numpy_dtype()
        array = np.frombuffer(data, dtype=dtype)
        return array.reshape(metadata.shape)
    
    def save(self, arrays: Dict[str, np.ndarray]) -> None:
        """保存多个数组到存储"""
        for name, array in arrays.items():
            self.add_array(name, array)
    
    def load(self, name: str) -> np.ndarray:
        """加载数组（兼容API）"""
        result = self.load_array(name)
        if result is None:
            raise KeyError(f"Array '{name}' not found")
        return result
    
    def reset(self) -> None:
        """重置存储，删除所有数组和数据文件"""
        # 删除所有数据文件
        for name in self.list_arrays():
            metadata = self.get_metadata(name)
            if metadata:
                data_path = self.directory / metadata.data_file
                if data_path.exists():
                    data_path.unlink()
        
        # 重置元数据存储
        self._store = BinaryMetadataStore()
        self.save_metadata()


class BinaryCompatibleReader:
    """二进制兼容格式读取器 - 用于延迟加载"""
    
    def __init__(self, manager: BinaryCompatibleManager, array_name: str):
        self.manager = manager
        self.array_name = array_name
        self._metadata = None
        self._memmap = None
    
    @property
    def metadata(self) -> Optional[BinaryArrayMetadata]:
        """获取元数据"""
        if self._metadata is None:
            self._metadata = self.manager.get_metadata(self.array_name)
        return self._metadata
    
    @property
    def memmap(self) -> Optional[np.ndarray]:
        """获取内存映射数组"""
        if self._memmap is None and self.metadata:
            data_path = self.manager.get_data_path(self.array_name)
            if data_path and data_path.exists():
                dtype = self.metadata.dtype.to_numpy_dtype()
                self._memmap = np.memmap(
                    data_path, 
                    dtype=dtype, 
                    mode='r', 
                    shape=tuple(self.metadata.shape)
                )
        return self._memmap
    
    def __getitem__(self, key) -> np.ndarray:
        """支持索引访问"""
        if self.memmap is not None:
            return self.memmap[key]
        return None
    
    def __array__(self) -> np.ndarray:
        """支持转换为数组"""
        if self.memmap is not None:
            return np.array(self.memmap)
        return None 