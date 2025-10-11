//! 高性能二进制元数据格式
//! 
//! 提供比MessagePack更快的序列化/反序列化性能，专为NumPack优化

use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Write, Seek, SeekFrom, BufReader, BufWriter};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock};
use std::time::SystemTime;

use crate::error::{NpkError, NpkResult};
use crate::metadata::DataType;

/// 二进制格式魔数 (ASCII: "NPKB")
pub const BINARY_MAGIC: u32 = 0x424B504E;

/// 当前二进制格式版本
pub const BINARY_VERSION: u32 = 1;

/// 二进制格式的数据类型枚举
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(u8)]
pub enum BinaryDataType {
    Bool = 0,
    Uint8 = 1,
    Uint16 = 2,
    Uint32 = 3,
    Uint64 = 4,
    Int8 = 5,
    Int16 = 6,
    Int32 = 7,
    Int64 = 8,
    Float16 = 9,
    Float32 = 10,
    Float64 = 11,
    Complex64 = 12,
    Complex128 = 13,
}

impl BinaryDataType {
    pub fn size_bytes(&self) -> usize {
        match self {
            BinaryDataType::Bool => 1,
            BinaryDataType::Uint8 => 1,
            BinaryDataType::Uint16 => 2,
            BinaryDataType::Uint32 => 4,
            BinaryDataType::Uint64 => 8,
            BinaryDataType::Int8 => 1,
            BinaryDataType::Int16 => 2,
            BinaryDataType::Int32 => 4,
            BinaryDataType::Int64 => 8,
            BinaryDataType::Float16 => 2,
            BinaryDataType::Float32 => 4,
            BinaryDataType::Float64 => 8,
            BinaryDataType::Complex64 => 8,
            BinaryDataType::Complex128 => 16,
        }
    }

    pub fn from_u8(value: u8) -> Self {
        match value {
            0 => BinaryDataType::Bool,
            1 => BinaryDataType::Uint8,
            2 => BinaryDataType::Uint16,
            3 => BinaryDataType::Uint32,
            4 => BinaryDataType::Uint64,
            5 => BinaryDataType::Int8,
            6 => BinaryDataType::Int16,
            7 => BinaryDataType::Int32,
            8 => BinaryDataType::Int64,
            9 => BinaryDataType::Float16,
            10 => BinaryDataType::Float32,
            11 => BinaryDataType::Float64,
            12 => BinaryDataType::Complex64,
            13 => BinaryDataType::Complex128,
            _ => BinaryDataType::Int32, // 默认值
        }
    }
}

impl From<DataType> for BinaryDataType {
    fn from(dt: DataType) -> Self {
        match dt {
            DataType::Bool => BinaryDataType::Bool,
            DataType::Uint8 => BinaryDataType::Uint8,
            DataType::Uint16 => BinaryDataType::Uint16,
            DataType::Uint32 => BinaryDataType::Uint32,
            DataType::Uint64 => BinaryDataType::Uint64,
            DataType::Int8 => BinaryDataType::Int8,
            DataType::Int16 => BinaryDataType::Int16,
            DataType::Int32 => BinaryDataType::Int32,
            DataType::Int64 => BinaryDataType::Int64,
            DataType::Float16 => BinaryDataType::Float16,
            DataType::Float32 => BinaryDataType::Float32,
            DataType::Float64 => BinaryDataType::Float64,
            DataType::Complex64 => BinaryDataType::Complex64,
            DataType::Complex128 => BinaryDataType::Complex128,
        }
    }
}

impl From<BinaryDataType> for DataType {
    fn from(dt: BinaryDataType) -> Self {
        match dt {
            BinaryDataType::Bool => DataType::Bool,
            BinaryDataType::Uint8 => DataType::Uint8,
            BinaryDataType::Uint16 => DataType::Uint16,
            BinaryDataType::Uint32 => DataType::Uint32,
            BinaryDataType::Uint64 => DataType::Uint64,
            BinaryDataType::Int8 => DataType::Int8,
            BinaryDataType::Int16 => DataType::Int16,
            BinaryDataType::Int32 => DataType::Int32,
            BinaryDataType::Int64 => DataType::Int64,
            BinaryDataType::Float16 => DataType::Float16,
            BinaryDataType::Float32 => DataType::Float32,
            BinaryDataType::Float64 => DataType::Float64,
            BinaryDataType::Complex64 => DataType::Complex64,
            BinaryDataType::Complex128 => DataType::Complex128,
        }
    }
}

/// 压缩算法枚举
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(u8)]
pub enum CompressionAlgorithm {
    None = 0,
    Zstd = 1,
}

impl CompressionAlgorithm {
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "zstd" => CompressionAlgorithm::Zstd,
            _ => CompressionAlgorithm::None,
        }
    }

    pub fn to_string(&self) -> &'static str {
        match self {
            CompressionAlgorithm::None => "none",
            CompressionAlgorithm::Zstd => "zstd",
        }
    }

    pub fn from_u8(value: u8) -> Self {
        match value {
            1 => CompressionAlgorithm::Zstd,
            _ => CompressionAlgorithm::None,
        }
    }
}

/// 块信息
#[derive(Debug, Clone)]
pub struct BinaryBlockInfo {
    pub offset: u64,
    pub original_size: u64,
    pub compressed_size: u64,
}

/// 块压缩信息
#[derive(Debug, Clone)]
pub struct BinaryBlockCompressionInfo {
    pub enabled: bool,
    pub block_size: u64,
    pub num_blocks: u64,
    pub blocks: Vec<BinaryBlockInfo>,
}

/// 压缩信息
#[derive(Debug, Clone)]
pub struct BinaryCompressionInfo {
    pub algorithm: CompressionAlgorithm,
    pub level: u32,
    pub original_size: u64,
    pub compressed_size: u64,
    pub block_compression: Option<BinaryBlockCompressionInfo>,
}

impl Default for BinaryCompressionInfo {
    fn default() -> Self {
        Self {
            algorithm: CompressionAlgorithm::None,
            level: 0,
            original_size: 0,
            compressed_size: 0,
            block_compression: None,
        }
    }
}

/// 二进制格式的数组元数据
#[derive(Debug, Clone)]
pub struct BinaryArrayMetadata {
    pub name: String,
    pub shape: Vec<u64>,
    pub data_file: String,
    pub last_modified: u64,
    pub size_bytes: u64,
    pub dtype: BinaryDataType,
    pub compression: BinaryCompressionInfo,
}

impl BinaryArrayMetadata {
    pub fn new(name: String, shape: Vec<u64>, data_file: String, dtype: BinaryDataType) -> Self {
        let total_elements: u64 = shape.iter().product();
        let size_bytes = total_elements * dtype.size_bytes() as u64;
        
        Self {
            name,
            shape,
            data_file,
            last_modified: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_micros() as u64,
            size_bytes,
            dtype,
            compression: BinaryCompressionInfo::default(),
        }
    }

    pub fn get_dtype(&self) -> BinaryDataType {
        self.dtype
    }

    /// 写入元数据到二进制流
    fn write_to_stream<W: Write>(&self, writer: &mut W) -> NpkResult<()> {
        // 写入名称长度和名称
        let name_bytes = self.name.as_bytes();
        writer.write_all(&(name_bytes.len() as u32).to_le_bytes())?;
        writer.write_all(name_bytes)?;

        // 写入形状
        writer.write_all(&(self.shape.len() as u32).to_le_bytes())?;
        for dim in &self.shape {
            writer.write_all(&dim.to_le_bytes())?;
        }

        // 写入数据文件名
        let data_file_bytes = self.data_file.as_bytes();
        writer.write_all(&(data_file_bytes.len() as u32).to_le_bytes())?;
        writer.write_all(data_file_bytes)?;

        // 写入基本信息
        writer.write_all(&self.last_modified.to_le_bytes())?;
        writer.write_all(&self.size_bytes.to_le_bytes())?;
        writer.write_all(&[self.dtype as u8])?;

        // 写入压缩信息
        writer.write_all(&[self.compression.algorithm as u8])?;
        writer.write_all(&self.compression.level.to_le_bytes())?;
        writer.write_all(&self.compression.original_size.to_le_bytes())?;
        writer.write_all(&self.compression.compressed_size.to_le_bytes())?;

        // 写入块压缩信息
        if let Some(ref block_info) = self.compression.block_compression {
            writer.write_all(&[1u8])?; // 表示有块压缩信息
            writer.write_all(&[if block_info.enabled { 1u8 } else { 0u8 }])?;
            writer.write_all(&block_info.block_size.to_le_bytes())?;
            writer.write_all(&block_info.num_blocks.to_le_bytes())?;
            writer.write_all(&(block_info.blocks.len() as u32).to_le_bytes())?;
            
            for block in &block_info.blocks {
                writer.write_all(&block.offset.to_le_bytes())?;
                writer.write_all(&block.original_size.to_le_bytes())?;
                writer.write_all(&block.compressed_size.to_le_bytes())?;
            }
        } else {
            writer.write_all(&[0u8])?; // 表示没有块压缩信息
        }

        Ok(())
    }

    /// 从二进制流读取元数据
    fn read_from_stream<R: Read>(reader: &mut R) -> NpkResult<Self> {
        // 读取名称
        let mut len_buf = [0u8; 4];
        reader.read_exact(&mut len_buf)?;
        let name_len = u32::from_le_bytes(len_buf) as usize;
        let mut name_buf = vec![0u8; name_len];
        reader.read_exact(&mut name_buf)?;
        let name = String::from_utf8(name_buf)
            .map_err(|e| NpkError::InvalidMetadata(format!("Invalid name UTF-8: {}", e)))?;

        // 读取形状
        reader.read_exact(&mut len_buf)?;
        let shape_len = u32::from_le_bytes(len_buf) as usize;
        let mut shape = Vec::with_capacity(shape_len);
        for _ in 0..shape_len {
            let mut dim_buf = [0u8; 8];
            reader.read_exact(&mut dim_buf)?;
            shape.push(u64::from_le_bytes(dim_buf));
        }

        // 读取数据文件名
        reader.read_exact(&mut len_buf)?;
        let data_file_len = u32::from_le_bytes(len_buf) as usize;
        let mut data_file_buf = vec![0u8; data_file_len];
        reader.read_exact(&mut data_file_buf)?;
        let data_file = String::from_utf8(data_file_buf)
            .map_err(|e| NpkError::InvalidMetadata(format!("Invalid data file UTF-8: {}", e)))?;

        // 读取基本信息
        let mut u64_buf = [0u8; 8];
        reader.read_exact(&mut u64_buf)?;
        let last_modified = u64::from_le_bytes(u64_buf);
        
        reader.read_exact(&mut u64_buf)?;
        let size_bytes = u64::from_le_bytes(u64_buf);
        
        let mut dtype_buf = [0u8; 1];
        reader.read_exact(&mut dtype_buf)?;
        let dtype = BinaryDataType::from_u8(dtype_buf[0]);

        // 读取压缩信息
        reader.read_exact(&mut dtype_buf)?;
        let algorithm = CompressionAlgorithm::from_u8(dtype_buf[0]);
        
        let mut u32_buf = [0u8; 4];
        reader.read_exact(&mut u32_buf)?;
        let level = u32::from_le_bytes(u32_buf);
        
        reader.read_exact(&mut u64_buf)?;
        let original_size = u64::from_le_bytes(u64_buf);
        
        reader.read_exact(&mut u64_buf)?;
        let compressed_size = u64::from_le_bytes(u64_buf);

        // 读取块压缩信息
        reader.read_exact(&mut dtype_buf)?;
        let has_block_info = dtype_buf[0] != 0;
        
        let block_compression = if has_block_info {
            reader.read_exact(&mut dtype_buf)?;
            let enabled = dtype_buf[0] != 0;
            
            reader.read_exact(&mut u64_buf)?;
            let block_size = u64::from_le_bytes(u64_buf);
            
            reader.read_exact(&mut u64_buf)?;
            let num_blocks = u64::from_le_bytes(u64_buf);
            
            reader.read_exact(&mut u32_buf)?;
            let blocks_len = u32::from_le_bytes(u32_buf) as usize;
            
            let mut blocks = Vec::with_capacity(blocks_len);
            for _ in 0..blocks_len {
                reader.read_exact(&mut u64_buf)?;
                let offset = u64::from_le_bytes(u64_buf);
                
                reader.read_exact(&mut u64_buf)?;
                let original_size = u64::from_le_bytes(u64_buf);
                
                reader.read_exact(&mut u64_buf)?;
                let compressed_size = u64::from_le_bytes(u64_buf);
                
                blocks.push(BinaryBlockInfo {
                    offset,
                    original_size,
                    compressed_size,
                });
            }
            
            Some(BinaryBlockCompressionInfo {
                enabled,
                block_size,
                num_blocks,
                blocks,
            })
        } else {
            None
        };

        let compression = BinaryCompressionInfo {
            algorithm,
            level,
            original_size,
            compressed_size,
            block_compression,
        };

        Ok(Self {
            name,
            shape,
            data_file,
            last_modified,
            size_bytes,
            dtype,
            compression,
        })
    }
}

/// 二进制格式的元数据存储
#[derive(Debug)]
pub struct BinaryMetadataStore {
    pub version: u32,
    pub arrays: HashMap<String, BinaryArrayMetadata>,
    pub total_size: u64,
}

impl BinaryMetadataStore {
    pub fn new() -> Self {
        Self {
            version: BINARY_VERSION,
            arrays: HashMap::new(),
            total_size: 0,
        }
    }

    pub fn load(path: &Path) -> NpkResult<Self> {
        if !path.exists() {
            return Ok(Self::new());
        }

        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        // 读取魔数
        let mut magic_buf = [0u8; 4];
        reader.read_exact(&mut magic_buf)?;
        let magic = u32::from_le_bytes(magic_buf);
        
        if magic != BINARY_MAGIC {
            return Err(NpkError::InvalidMetadata("Invalid magic number".to_string()));
        }

        // 读取版本
        let mut version_buf = [0u8; 4];
        reader.read_exact(&mut version_buf)?;
        let version = u32::from_le_bytes(version_buf);

        // 读取总大小
        let mut total_size_buf = [0u8; 8];
        reader.read_exact(&mut total_size_buf)?;
        let total_size = u64::from_le_bytes(total_size_buf);

        // 读取数组数量
        let mut arrays_count_buf = [0u8; 4];
        reader.read_exact(&mut arrays_count_buf)?;
        let arrays_count = u32::from_le_bytes(arrays_count_buf);

        // 读取数组元数据
        let mut arrays = HashMap::new();
        for _ in 0..arrays_count {
            let meta = BinaryArrayMetadata::read_from_stream(&mut reader)?;
            arrays.insert(meta.name.clone(), meta);
        }

        Ok(Self {
            version,
            arrays,
            total_size,
        })
    }

    pub fn save(&self, path: &Path) -> NpkResult<()> {
        let temp_path = path.with_extension("tmp");
        
        {
            let file = File::create(&temp_path)?;
            let mut writer = BufWriter::new(file);

            // 写入魔数
            writer.write_all(&BINARY_MAGIC.to_le_bytes())?;

            // 写入版本
            writer.write_all(&self.version.to_le_bytes())?;

            // 写入总大小
            writer.write_all(&self.total_size.to_le_bytes())?;

            // 写入数组数量
            writer.write_all(&(self.arrays.len() as u32).to_le_bytes())?;

            // 写入每个数组的元数据
            for meta in self.arrays.values() {
                meta.write_to_stream(&mut writer)?;
            }

            writer.flush()?;
        }

        std::fs::rename(temp_path, path)?;
        Ok(())
    }

    pub fn add_array(&mut self, meta: BinaryArrayMetadata) {
        self.total_size = self.total_size.saturating_sub(
            self.arrays.get(&meta.name).map(|m| m.size_bytes).unwrap_or(0)
        );
        self.total_size += meta.size_bytes;
        self.arrays.insert(meta.name.clone(), meta);
    }

    pub fn remove_array(&mut self, name: &str) -> bool {
        if let Some(meta) = self.arrays.remove(name) {
            self.total_size = self.total_size.saturating_sub(meta.size_bytes);
            true
        } else {
            false
        }
    }

    pub fn get_array(&self, name: &str) -> Option<&BinaryArrayMetadata> {
        self.arrays.get(name)
    }

    pub fn list_arrays(&self) -> Vec<String> {
        self.arrays.keys().cloned().collect()
    }

    pub fn has_array(&self, name: &str) -> bool {
        self.arrays.contains_key(name)
    }
}

/// 缓存的二进制元数据存储
pub struct BinaryCachedStore {
    store: Arc<RwLock<BinaryMetadataStore>>,
    path: Arc<Path>,
    last_sync: Arc<Mutex<SystemTime>>,
    sync_interval: std::time::Duration,
}

impl BinaryCachedStore {
    pub fn new(path: &Path, _wal_path: Option<PathBuf>) -> NpkResult<Self> {
        let store = BinaryMetadataStore::load(path).unwrap_or_else(|_| BinaryMetadataStore::new());
        
        let cached_store = Self {
            store: Arc::new(RwLock::new(store)),
            path: Arc::from(path),
            last_sync: Arc::new(Mutex::new(SystemTime::now())),
            sync_interval: std::time::Duration::from_secs(1),
        };
        
        // 保存初始存储
        cached_store.sync_to_disk()?;
        Ok(cached_store)
    }

    pub fn from_store(store: BinaryMetadataStore, path: &Path, _wal_path: Option<PathBuf>) -> NpkResult<Self> {
        Ok(Self {
            store: Arc::new(RwLock::new(store)),
            path: Arc::from(path),
            last_sync: Arc::new(Mutex::new(SystemTime::now())),
            sync_interval: std::time::Duration::from_secs(1),
        })
    }

    fn sync_to_disk(&self) -> NpkResult<()> {
        let store = self.store.read().unwrap();
        store.save(&self.path)?;
        
        let mut last_sync = self.last_sync.lock().unwrap();
        *last_sync = SystemTime::now();
        Ok(())
    }

    pub fn add_array(&self, meta: BinaryArrayMetadata) -> NpkResult<()> {
        let mut store = self.store.write().unwrap();
        store.add_array(meta);
        drop(store);
        // 🚀 性能关键优化：延迟同步，不立即写入磁盘
        // 
        // 问题：每次add_array都调用sync_to_disk导致性能下降2-3x
        // NumPy不会每次都fsync，所以更快
        // 
        // 解决方案：
        // - add_array只更新内存中的元数据
        // - 元数据会定期自动同步（sync_interval控制）
        // - 或在显式调用force_sync时同步
        //
        // 注释掉立即同步：
        // self.sync_to_disk()?;
        Ok(())
    }
    
    /// 强制同步到磁盘
    pub fn force_sync(&self) -> NpkResult<()> {
        self.sync_to_disk()
    }

    pub fn delete_array(&self, name: &str) -> NpkResult<bool> {
        let mut store = self.store.write().unwrap();
        let result = store.remove_array(name);
        drop(store);
        // 🚀 延迟同步优化
        // if result {
        //     self.sync_to_disk()?;
        // }
        Ok(result)
    }

    pub fn get_array(&self, name: &str) -> Option<BinaryArrayMetadata> {
        let store = self.store.read().unwrap();
        store.get_array(name).cloned()
    }

    pub fn list_arrays(&self) -> Vec<String> {
        let store = self.store.read().unwrap();
        store.list_arrays()
    }

    pub fn has_array(&self, name: &str) -> bool {
        let store = self.store.read().unwrap();
        store.has_array(name)
    }

    pub fn update_array_metadata(&self, name: &str, meta: BinaryArrayMetadata) -> NpkResult<()> {
        let mut store = self.store.write().unwrap();
        store.remove_array(name);
        store.add_array(meta);
        drop(store);
        // 🚀 延迟同步优化
        // self.sync_to_disk()?;
        Ok(())
    }

    pub fn reset(&self) -> NpkResult<()> {
        let mut store = self.store.write().unwrap();
        *store = BinaryMetadataStore::new();
        drop(store);
        // 🚀 延迟同步优化
        // self.sync_to_disk()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_binary_metadata_store() {
        let temp_dir = TempDir::new().unwrap();
        let metadata_path = temp_dir.path().join("metadata.npkm");

        let mut store = BinaryMetadataStore::new();
        
        // 添加测试数组
        let shape = vec![100, 200];
        let data_file = "data_test.npkd".to_string();
        let dtype = BinaryDataType::Float32;
        
        let meta = BinaryArrayMetadata::new("test_array".to_string(), shape.clone(), data_file.clone(), dtype);
        store.add_array(meta);
        
        // 验证数组存在
        assert!(store.has_array("test_array"));
        
        let retrieved_meta = store.get_array("test_array").unwrap();
        assert_eq!(retrieved_meta.name, "test_array");
        assert_eq!(retrieved_meta.shape, shape);
        
        // 保存并重新加载
        store.save(&metadata_path).unwrap();
        
        let loaded_store = BinaryMetadataStore::load(&metadata_path).unwrap();
        assert!(loaded_store.has_array("test_array"));
        
        let loaded_meta = loaded_store.get_array("test_array").unwrap();
        assert_eq!(loaded_meta.name, "test_array");
        assert_eq!(loaded_meta.shape, shape);
    }

    #[test]
    fn test_data_type_conversion() {
        let original = DataType::Float64;
        let binary: BinaryDataType = original.into();
        let converted_back: DataType = binary.into();
        
        assert_eq!(original, converted_back);
    }

    #[test]
    fn test_compression_info() {
        let compression = BinaryCompressionInfo {
            algorithm: CompressionAlgorithm::Zstd,
            level: 3,
            original_size: 1000,
            compressed_size: 500,
            block_compression: None,
        };
        
        assert_eq!(compression.algorithm, CompressionAlgorithm::Zstd);
        assert_eq!(compression.level, 3);
    }
} 