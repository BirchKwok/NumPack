//! MessagePack兼容的元数据处理模块
//! 
//! 与Python端的MessagePack格式完全兼容

use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock};
use std::time::SystemTime;
use serde::{Deserialize, Serialize, Serializer};
use rmp_serde;

use crate::error::{NpkError, NpkResult};

/// 与Python端完全兼容的数据类型枚举
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[repr(u8)]
pub enum MessagePackDataType {
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

impl MessagePackDataType {
    pub fn size_bytes(&self) -> usize {
        match self {
            MessagePackDataType::Bool => 1,
            MessagePackDataType::Uint8 => 1,
            MessagePackDataType::Uint16 => 2,
            MessagePackDataType::Uint32 => 4,
            MessagePackDataType::Uint64 => 8,
            MessagePackDataType::Int8 => 1,
            MessagePackDataType::Int16 => 2,
            MessagePackDataType::Int32 => 4,
            MessagePackDataType::Int64 => 8,
            MessagePackDataType::Float16 => 2,
            MessagePackDataType::Float32 => 4,
            MessagePackDataType::Float64 => 8,
            MessagePackDataType::Complex64 => 8,
            MessagePackDataType::Complex128 => 16,
        }
    }
}

/// MessagePack格式的数组元数据 - 与Python端完全一致
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessagePackArrayMetadata {
    #[serde(rename = "name")]
    pub name: String,
    #[serde(rename = "shape")]
    pub shape: Vec<u64>,
    #[serde(rename = "data_file")]
    pub data_file: String,
    #[serde(rename = "last_modified")]
    pub last_modified: u64,
    #[serde(rename = "size_bytes")]
    pub size_bytes: u64,
    #[serde(rename = "dtype")]
    pub dtype: u8,  // 使用u8来匹配Python端
}

impl MessagePackArrayMetadata {
    pub fn new(name: String, shape: Vec<u64>, data_file: String, dtype: MessagePackDataType) -> Self {
        let total_elements: u64 = shape.iter().product();
        let size_bytes = total_elements * dtype.size_bytes() as u64;
        
        Self {
            name,
            shape,
            data_file,
            last_modified: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_micros() as u64,  // 微秒，与Python端匹配
            size_bytes,
            dtype: dtype as u8,
        }
    }
    
    pub fn get_dtype(&self) -> MessagePackDataType {
        match self.dtype {
            0 => MessagePackDataType::Bool,
            1 => MessagePackDataType::Uint8,
            2 => MessagePackDataType::Uint16,
            3 => MessagePackDataType::Uint32,
            4 => MessagePackDataType::Uint64,
            5 => MessagePackDataType::Int8,
            6 => MessagePackDataType::Int16,
            7 => MessagePackDataType::Int32,
            8 => MessagePackDataType::Int64,
            9 => MessagePackDataType::Float16,
            10 => MessagePackDataType::Float32,
            11 => MessagePackDataType::Float64,
            12 => MessagePackDataType::Complex64,
            13 => MessagePackDataType::Complex128,
            _ => MessagePackDataType::Int32, // 默认
        }
    }
}

/// MessagePack格式的元数据存储
#[derive(Debug, Serialize, Deserialize)]
pub struct MessagePackMetadataStore {
    #[serde(rename = "version")]
    pub version: u32,
    #[serde(rename = "arrays")]
    pub arrays: HashMap<String, MessagePackArrayMetadata>,
    #[serde(rename = "total_size")]
    pub total_size: u64,
}

impl MessagePackMetadataStore {
    pub fn new() -> Self {
        Self {
            version: 1,
            arrays: HashMap::new(),
            total_size: 0,
        }
    }
    
    pub fn load(path: &Path) -> NpkResult<Self> {
        if !path.exists() {
            return Ok(Self::new());
        }
        
        let mut file = File::open(path)?;
        let mut data = Vec::new();
        file.read_to_end(&mut data)?;
        
        if data.is_empty() {
            return Ok(Self::new());
        }
        
        match rmp_serde::from_slice(&data) {
            Ok(store) => {
                Ok(store)
            },
            Err(e) => {
                Err(NpkError::InvalidMetadata(format!("MessagePack deserialization failed: {}", e)))
            }
        }
    }
    
    pub fn save(&self, path: &Path) -> NpkResult<()> {
        let temp_path = path.with_extension("tmp");
        
        // 使用明确的序列化配置，强制使用映射格式而不是数组格式
        let mut buf = Vec::new();
        let mut serializer = rmp_serde::Serializer::new(&mut buf)
            .with_struct_map(); // 强制结构体序列化为映射而不是数组
        
        self.serialize(&mut serializer)
            .map_err(|e| NpkError::InvalidMetadata(e.to_string()))?;
        
        let mut file = File::create(&temp_path)?;
        file.write_all(&buf)?;
        file.sync_all()?;
        
        std::fs::rename(temp_path, path)?;
        Ok(())
    }
    
    pub fn add_array(&mut self, meta: MessagePackArrayMetadata) {
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
    
    pub fn get_array(&self, name: &str) -> Option<&MessagePackArrayMetadata> {
        self.arrays.get(name)
    }
    
    pub fn list_arrays(&self) -> Vec<String> {
        self.arrays.keys().cloned().collect()
    }
    
    pub fn has_array(&self, name: &str) -> bool {
        self.arrays.contains_key(name)
    }
} 

/// 直接管理MessagePack格式的缓存元数据存储
pub struct MessagePackCachedStore {
    store: Arc<RwLock<MessagePackMetadataStore>>,
    path: Arc<Path>,
    last_sync: Arc<Mutex<SystemTime>>,
    sync_interval: std::time::Duration,
}

impl MessagePackCachedStore {
    pub fn new(path: &Path, _wal_path: Option<PathBuf>) -> NpkResult<Self> {
        let store = MessagePackMetadataStore::new();
        
        let cached_store = Self {
            store: Arc::new(RwLock::new(store)),
            path: Arc::from(path),
            last_sync: Arc::new(Mutex::new(SystemTime::now())),
            sync_interval: std::time::Duration::from_secs(1),
        };
        
        // 保存初始的空存储
        cached_store.sync_to_disk()?;
        Ok(cached_store)
    }
    
    pub fn from_store(store: MessagePackMetadataStore, path: &Path, _wal_path: Option<PathBuf>) -> NpkResult<Self> {
        Ok(Self {
            store: Arc::new(RwLock::new(store)),
            path: Arc::from(path),
            last_sync: Arc::new(Mutex::new(SystemTime::now())),
            sync_interval: std::time::Duration::from_secs(1),
        })
    }
    
    fn should_sync(&self) -> bool {
        let last = self.last_sync.lock().unwrap();
        SystemTime::now()
            .duration_since(*last)
            .map(|duration| duration >= self.sync_interval)
            .unwrap_or(true)
    }
    
    fn sync_to_disk(&self) -> NpkResult<()> {
        let store = self.store.read().unwrap();
        
        // 直接保存为MessagePack格式
        store.save(&self.path)?;
        
        let mut last_sync = self.last_sync.lock().unwrap();
        *last_sync = SystemTime::now();
        Ok(())
    }
    
    pub fn add_array(&self, meta: MessagePackArrayMetadata) -> NpkResult<()> {
        let mut store = self.store.write().unwrap();
        store.add_array(meta);
        drop(store);
        self.sync_to_disk()?;
        Ok(())
    }
    
    pub fn delete_array(&self, name: &str) -> NpkResult<bool> {
        let mut store = self.store.write().unwrap();
        let result = store.remove_array(name);
        drop(store);
        if result {
            self.sync_to_disk()?;
        }
        Ok(result)
    }
    
    pub fn get_array(&self, name: &str) -> Option<MessagePackArrayMetadata> {
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
    
    pub fn update_array_metadata(&self, name: &str, meta: MessagePackArrayMetadata) -> NpkResult<()> {
        let mut store = self.store.write().unwrap();
        // 删除旧的并添加新的元数据
        store.remove_array(name);
        store.add_array(meta);
        drop(store);
        self.sync_to_disk()?;
        Ok(())
    }
    
    pub fn reset(&self) -> NpkResult<()> {
        let mut store = self.store.write().unwrap();
        *store = MessagePackMetadataStore::new();
        drop(store);
        self.sync_to_disk()?;
        Ok(())
    }
} 