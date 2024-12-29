use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io::{BufReader, BufWriter};
use std::path::Path;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};
use serde::{Deserialize, Serialize};

use crate::error::{NpkError, NpkResult};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum DataType {
    Bool,
    Uint8,
    Uint16,
    Uint32,
    Uint64,
    Int8,
    Int16,
    Int32,
    Int64,
    Float16,
    Float32,
    Float64,
}

impl DataType {
    pub fn size_bytes(&self) -> usize {
        match self {
            DataType::Bool => 1,
            DataType::Uint8 => 1,
            DataType::Uint16 => 2,
            DataType::Uint32 => 4,
            DataType::Uint64 => 8,
            DataType::Int8 => 1,
            DataType::Int16 => 2,
            DataType::Int32 => 4,
            DataType::Int64 => 8,
            DataType::Float16 => 2,
            DataType::Float32 => 4,
            DataType::Float64 => 8,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArrayMetadata {
    pub name: String,
    pub rows: u64,
    pub cols: u64,
    pub data_file: String,     // 数据文件名
    pub is_deleted: bool,      // 标记删除
    pub last_modified: u64,    // 最后修改时间
    pub size_bytes: u64,       // 数据大小
    pub dtype: DataType,       // 数据类型
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MetadataStore {
    version: u32,
    arrays: HashMap<String, ArrayMetadata>,
    total_size: u64,           // 所有数据文件总大小
    deleted_size: u64,         // 已删除数据大小
}

impl MetadataStore {
    pub fn new() -> Self {
        Self {
            version: 1,
            arrays: HashMap::new(),
            total_size: 0,
            deleted_size: 0,
        }
    }

    pub fn load(path: &Path) -> NpkResult<Self> {
        let file = OpenOptions::new()
            .read(true)
            .open(path)?;
        let reader = BufReader::new(file);
        let store = serde_json::from_reader(reader)
            .map_err(|e| NpkError::InvalidMetadata(e.to_string()))?;
        Ok(store)
    }

    pub fn save(&self, path: &Path) -> NpkResult<()> {
        let file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, self)
            .map_err(|e| NpkError::InvalidMetadata(e.to_string()))?;
        Ok(())
    }

    pub fn add_array(&mut self, meta: ArrayMetadata) {
        if let Some(old_meta) = self.arrays.get(&meta.name) {
            if !old_meta.is_deleted {
                self.total_size -= old_meta.size_bytes;
            }
        }
        self.total_size += meta.size_bytes;
        self.arrays.insert(meta.name.clone(), meta);
    }

    pub fn mark_deleted(&mut self, name: &str) -> bool {
        if let Some(meta) = self.arrays.get_mut(name) {
            if !meta.is_deleted {
                meta.is_deleted = true;
                self.deleted_size += meta.size_bytes;
                self.total_size -= meta.size_bytes;
                return true;
            }
        }
        false
    }

    pub fn should_compact(&self, threshold: u64) -> bool {
        self.total_size > threshold && (self.deleted_size as f64 / self.total_size as f64) > 0.2
    }

    pub fn get_array(&self, name: &str) -> Option<&ArrayMetadata> {
        self.arrays.get(name)
    }

    pub fn list_arrays(&self) -> Vec<String> {
        self.arrays
            .iter()
            .filter(|(_, meta)| !meta.is_deleted)
            .map(|(name, _)| name.clone())
            .collect()
    }

    pub fn reset_deleted_size(&mut self) {
        self.deleted_size = 0;
    }

    pub fn reset(&mut self) {
        self.arrays.clear();
        self.total_size = 0;
        self.deleted_size = 0;
    }
}

#[derive(Debug)]
pub struct CachedMetadataStore {
    store: Arc<RwLock<MetadataStore>>,
    path: Arc<Path>,
    last_sync: Arc<Mutex<SystemTime>>,
    sync_interval: std::time::Duration,
}

impl CachedMetadataStore {
    pub fn new(path: &Path) -> NpkResult<Self> {
        let store = if path.exists() {
            MetadataStore::load(path)?
        } else {
            MetadataStore::new()
        };
        
        Ok(Self {
            store: Arc::new(RwLock::new(store)),
            path: Arc::from(path),
            last_sync: Arc::new(Mutex::new(SystemTime::now())),
            sync_interval: std::time::Duration::from_secs(5), // 默认5秒同步一次
        })
    }

    pub fn set_sync_interval(&mut self, interval: std::time::Duration) {
        self.sync_interval = interval;
    }

    fn should_sync(&self) -> bool {
        let last_sync = *self.last_sync.lock().unwrap();
        SystemTime::now()
            .duration_since(last_sync)
            .map(|duration| duration >= self.sync_interval)
            .unwrap_or(true)
    }

    fn sync_to_disk(&self) -> NpkResult<()> {
        let store = self.store.read().unwrap();
        store.save(&self.path)?;
        *self.last_sync.lock().unwrap() = SystemTime::now();
        Ok(())
    }

    pub fn add_array(&self, meta: ArrayMetadata) -> NpkResult<()> {
        let mut store = self.store.write().unwrap();
        store.add_array(meta);
        if self.should_sync() {
            drop(store);
            self.sync_to_disk()?;
        }
        Ok(())
    }

    pub fn mark_deleted(&self, name: &str) -> NpkResult<bool> {
        let mut store = self.store.write().unwrap();
        let result = store.mark_deleted(name);
        if result && self.should_sync() {
            drop(store);
            self.sync_to_disk()?;
        }
        Ok(result)
    }

    pub fn get_array(&self, name: &str) -> Option<ArrayMetadata> {
        let store = self.store.read().unwrap();
        store.get_array(name).cloned()
    }

    pub fn list_arrays(&self) -> Vec<String> {
        let store = self.store.read().unwrap();
        store.list_arrays()
    }

    pub fn should_compact(&self, threshold: u64) -> bool {
        let store = self.store.read().unwrap();
        store.should_compact(threshold)
    }

    pub fn reset_deleted_size(&self) -> NpkResult<()> {
        let mut store = self.store.write().unwrap();
        store.reset_deleted_size();
        if self.should_sync() {
            drop(store);
            self.sync_to_disk()?;
        }
        Ok(())
    }

    pub fn reset(&self) -> NpkResult<()> {
        let mut store = self.store.write().unwrap();
        store.reset();
        drop(store);
        self.sync_to_disk()?;
        Ok(())
    }

    pub fn force_sync(&self) -> NpkResult<()> {
        self.sync_to_disk()
    }
} 