use std::collections::HashMap;
use std::fs::{OpenOptions, File};
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};
use serde::{Deserialize, Serialize};
use bitvec::prelude::*;

use crate::error::{NpkError, NpkResult};

// WAL 操作类型
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WalOp {
    BeginTransaction,
    CommitTransaction,
    AddArray(ArrayMetadata),
    DeleteArray(String),
    UpdateArray(String, ArrayMetadata),
    UpdateRows(String, u64),
}

// WAL 记录结构
#[derive(Debug, Serialize, Deserialize)]
struct WalRecord {
    op: WalOp,
    timestamp: u64,
    checksum: u32,
    sequence_number: u64,
}

impl WalRecord {
    fn new(op: WalOp, sequence_number: u64) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let mut record = Self {
            op,
            timestamp,
            checksum: 0,
            sequence_number,
        };
        record.checksum = record.calculate_checksum();
        record
    }

    fn calculate_checksum(&self) -> u32 {
        let mut hasher = crc32fast::Hasher::new();
        let op_bytes = bincode::serialize(&self.op).unwrap_or_default();
        hasher.update(&op_bytes);
        hasher.update(&self.timestamp.to_le_bytes());
        hasher.update(&self.sequence_number.to_le_bytes());
        hasher.finalize()
    }

    fn is_valid(&self) -> bool {
        self.checksum == self.calculate_checksum()
    }
}

// WAL 写入器
#[allow(dead_code)]
#[derive(Debug)]
pub struct WalWriter {
    file: File,
    path: PathBuf,
    current_sequence: u64,
    in_transaction: bool,
}

impl WalWriter {
    const MAX_WAL_SIZE: u64 = 1024 * 1024 * 100; // 100MB

    fn new(path: PathBuf) -> io::Result<Self> {
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)?;
            
        Ok(Self { 
            file,
            path,
            current_sequence: 0,
            in_transaction: false,
        })
    }

    fn rotate_if_needed(&mut self) -> io::Result<()> {
        let metadata = self.file.metadata()?;
        if metadata.len() > Self::MAX_WAL_SIZE {
            // 创建新的WAL文件
            let new_path = self.path.with_extension("wal.new");
            let new_file = OpenOptions::new()
                .create(true)
                .write(true)
                .open(&new_path)?;
                
            // 替换旧文件
            std::fs::rename(&new_path, &self.path)?;
            self.file = new_file;
        }
        Ok(())
    }

    fn append(&mut self, op: WalOp) -> io::Result<()> {
        self.current_sequence += 1;
        let record = WalRecord::new(op, self.current_sequence);
        
        let bytes = bincode::serialize(&record)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
            
        // 原子写入
        let mut temp_buf = Vec::new();
        temp_buf.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
        temp_buf.extend_from_slice(&bytes);
        
        // 一次性写入所有数据
        self.file.write_all(&temp_buf)?;
        self.file.sync_data()?;
        
        self.rotate_if_needed()?;
        Ok(())
    }

    fn batch_append(&mut self, ops: Vec<WalOp>) -> io::Result<()> {
        let mut temp_buf = Vec::new();
        
        for op in ops {
            self.current_sequence += 1;
            let record = WalRecord::new(op, self.current_sequence);
            let bytes = bincode::serialize(&record)
                .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
                
            temp_buf.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
            temp_buf.extend_from_slice(&bytes);
        }
        
        // 一次性写入所有数据
        self.file.write_all(&temp_buf)?;
        self.file.sync_data()?;
        
        self.rotate_if_needed()?;
        Ok(())
    }

    fn begin_transaction(&mut self) -> io::Result<()> {
        if !self.in_transaction {
            self.append(WalOp::BeginTransaction)?;
            self.in_transaction = true;
        }
        Ok(())
    }

    fn commit_transaction(&mut self) -> io::Result<()> {
        if self.in_transaction {
            self.append(WalOp::CommitTransaction)?;
            self.in_transaction = false;
        }
        Ok(())
    }

    fn truncate(&mut self) -> io::Result<()> {
        self.file.set_len(0)?;
        self.file.sync_data()?;
        self.current_sequence = 0;
        self.in_transaction = false;
        Ok(())
    }
}

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
    total_size: u64,
    deleted_size: u64,
    #[serde(skip)]
    bitmap: BitVec,
    #[serde(skip)]
    name_to_index: HashMap<String, usize>,
    #[serde(skip)]
    next_index: usize,
    #[serde(skip)]
    wal: Option<WalWriter>,
}

impl MetadataStore {
    pub fn new(wal_path: Option<PathBuf>) -> Self {
        let mut store = Self {
            version: 1,
            arrays: HashMap::new(),
            total_size: 0,
            deleted_size: 0,
            bitmap: BitVec::new(),
            name_to_index: HashMap::new(),
            next_index: 0,
            wal: None,
        };
        
        if let Some(path) = wal_path {
            store.wal = WalWriter::new(path).ok();
        }
        
        store
    }

    pub fn load(path: &Path, wal_path: Option<PathBuf>) -> NpkResult<Self> {
        let mut store = if path.exists() {
            let file = OpenOptions::new()
                .read(true)
                .open(path)?;
            let reader = BufReader::new(file);
            bincode::deserialize_from(reader)
                .map_err(|e| NpkError::InvalidMetadata(e.to_string()))?
        } else {
            Self::new(None)
        };

        // 重建位图和索引映射
        store.bitmap = BitVec::new();
        store.name_to_index = HashMap::new();
        store.next_index = 0;
        
        for (name, meta) in &store.arrays {
            store.bitmap.push(!meta.is_deleted);
            store.name_to_index.insert(name.clone(), store.next_index);
            store.next_index += 1;
        }

        // 如果有WAL，重放WAL
        if let Some(wal_path) = wal_path {
            if wal_path.exists() {
                store.replay_wal(&wal_path)?;
            }
            store.wal = WalWriter::new(wal_path).ok();
        }
        
        Ok(store)
    }

    fn replay_wal(&mut self, path: &Path) -> NpkResult<()> {
        let mut file = File::open(path)?;
        let mut buffer = Vec::new();
        let mut len_bytes = [0u8; 4];
        let mut last_valid_sequence = 0u64;
        let mut in_transaction = false;
        let mut transaction_ops = Vec::new();
        
        while let Ok(()) = file.read_exact(&mut len_bytes) {
            let len = u32::from_le_bytes(len_bytes);
            if len > 1024 * 1024 * 10 { // 10MB 安全限制
                return Err(NpkError::InvalidMetadata("WAL record too large".to_string()));
            }
            
            buffer.resize(len as usize, 0);
            match file.read_exact(&mut buffer) {
                Ok(()) => {
                    if let Ok(record) = bincode::deserialize::<WalRecord>(&buffer) {
                        if record.is_valid() && record.sequence_number > last_valid_sequence {
                            match record.op {
                                WalOp::BeginTransaction => {
                                    in_transaction = true;
                                    transaction_ops.clear();
                                }
                                WalOp::CommitTransaction => {
                                    if in_transaction {
                                        // 应用事务中的所有操作
                                        for op in transaction_ops.drain(..) {
                                            self.apply_wal_op(op)?;
                                        }
                                        in_transaction = false;
                                    }
                                }
                                op => {
                                    if in_transaction {
                                        transaction_ops.push(op);
                                    } else {
                                        self.apply_wal_op(op)?;
                                    }
                                }
                            }
                            last_valid_sequence = record.sequence_number;
                        }
                    }
                }
                Err(e) => {
                    eprintln!("Warning: WAL replay stopped due to error: {}", e);
                    break;
                }
            }
        }
        
        // 如果还在事务中,说明事务未完成,丢弃未完成的事务
        if in_transaction {
            transaction_ops.clear();
        }
        
        Ok(())
    }

    fn apply_wal_op(&mut self, op: WalOp) -> NpkResult<()> {
        match op {
            WalOp::BeginTransaction => Ok(()), // 事务开始标记,不需要具体操作
            WalOp::CommitTransaction => Ok(()), // 事务结束标记,不需要具体操作
            WalOp::AddArray(meta) => {
                self.add_array(meta);
                Ok(())
            },
            WalOp::DeleteArray(name) => {
                self.mark_deleted(&name)?;
                Ok(())
            },
            WalOp::UpdateArray(name, meta) => {
                self.update_array_metadata(&name, meta);
                Ok(())
            },
            WalOp::UpdateRows(name, rows) => {
                if let Some(meta) = self.arrays.get_mut(&name) {
                    meta.rows = rows;
                }
                Ok(())
            }
        }
    }

    pub fn save(&self, path: &Path) -> NpkResult<()> {
        let file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;
        let writer = BufWriter::new(file);
        bincode::serialize_into(writer, self)
            .map_err(|e| NpkError::InvalidMetadata(e.to_string()))?;
        Ok(())
    }

    // 修改现有方法以支持WAL
    pub fn add_array(&mut self, meta: ArrayMetadata) {
        if let Some(wal) = &mut self.wal {
            let _ = wal.append(WalOp::AddArray(meta.clone()));
        }
        
        let name = meta.name.clone();
        if let Some(&index) = self.name_to_index.get(&name) {
            if self.bitmap[index] {
                if let Some(old_meta) = self.arrays.get(&name) {
                    self.total_size -= old_meta.size_bytes;
                }
            }
            self.bitmap.set(index, true);
        } else {
            self.bitmap.push(true);
            self.name_to_index.insert(name.clone(), self.next_index);
            self.next_index += 1;
        }
        
        self.total_size += meta.size_bytes;
        self.arrays.insert(name, meta);
    }

    pub fn mark_deleted(&mut self, name: &str) -> NpkResult<bool> {
        if let Some(&index) = self.name_to_index.get(name) {
            if self.bitmap[index] {
                if let Some(wal) = &mut self.wal {
                    wal.append(WalOp::DeleteArray(name.to_string()))?;
                }
                
                self.bitmap.set(index, false);
                if let Some(meta) = self.arrays.get(name) {
                    self.deleted_size += meta.size_bytes;
                    self.total_size -= meta.size_bytes;
                }
                return Ok(true);
            }
        }
        Ok(false)
    }

    pub fn update_array_metadata(&mut self, name: &str, meta: ArrayMetadata) {
        if let Some(wal) = &mut self.wal {
            let _ = wal.append(WalOp::UpdateArray(name.to_string(), meta.clone()));
        }
        
        if let Some(old_meta) = self.arrays.get(name) {
            if !old_meta.is_deleted {
                self.total_size -= old_meta.size_bytes;
            }
        }
        self.arrays.insert(name.to_string(), meta.clone());
        if !meta.is_deleted {
            self.total_size += meta.size_bytes;
        }
    }

    pub fn batch_mark_deleted(&mut self, names: &[String]) -> NpkResult<usize> {
        let mut ops = Vec::new();
        let mut deleted_count = 0;

        self.begin_transaction()?;
        
        for name in names {
            if let Some(&index) = self.name_to_index.get(name) {
                if self.bitmap[index] {
                    ops.push(WalOp::DeleteArray(name.clone()));
                    self.bitmap.set(index, false);
                    if let Some(meta) = self.arrays.get(name) {
                        self.deleted_size += meta.size_bytes;
                        self.total_size -= meta.size_bytes;
                    }
                    deleted_count += 1;
                }
            }
        }

        if let Some(wal) = &mut self.wal {
            wal.batch_append(ops)?;
        }

        self.commit_transaction()?;
        
        Ok(deleted_count)
    }

    pub fn should_compact(&self, threshold: u64) -> bool {
        self.total_size > threshold && (self.deleted_size as f64 / self.total_size as f64) > 0.2
    }

    pub fn get_array(&self, name: &str) -> Option<&ArrayMetadata> {
        self.name_to_index.get(name)
            .filter(|&&index| self.bitmap[index])
            .and_then(|_| self.arrays.get(name))
    }

    pub fn list_arrays(&self) -> Vec<String> {
        self.arrays
            .iter()
            .filter(|(name, _)| {
                self.name_to_index
                    .get(*name)
                    .map(|&index| self.bitmap[index])
                    .unwrap_or(false)
            })
            .map(|(name, _)| name.clone())
            .collect()
    }

    pub fn incremental_compact(&mut self, batch_size: usize) -> Vec<String> {
        let mut compacted = Vec::new();
        let mut count = 0;
        
        for (name, _) in self.arrays.iter() {
            if count >= batch_size {
                break;
            }
            
            if let Some(&index) = self.name_to_index.get(name) {
                if !self.bitmap[index] {
                    compacted.push(name.clone());
                    count += 1;
                }
            }
        }
        
        // 移除已标记为删除的数组
        for name in &compacted {
            if let Some(wal) = &mut self.wal {
                let _ = wal.append(WalOp::DeleteArray(name.to_string()));
            }
            self.arrays.remove(name);
            if let Some(index) = self.name_to_index.remove(name) {
                self.bitmap.set(index, false);
            }
        }
        
        compacted
    }

    pub fn reset_deleted_size(&mut self) {
        self.deleted_size = 0;
    }

    pub fn reset(&mut self) {
        self.arrays.clear();
        self.bitmap.clear();
        self.name_to_index.clear();
        self.next_index = 0;
        self.total_size = 0;
        self.deleted_size = 0;
        if let Some(wal) = &mut self.wal {
            let _ = wal.truncate();
        }
    }

    pub fn begin_transaction(&mut self) -> NpkResult<()> {
        if let Some(wal) = &mut self.wal {
            wal.begin_transaction()?;
        }
        Ok(())
    }
    
    pub fn commit_transaction(&mut self) -> NpkResult<()> {
        if let Some(wal) = &mut self.wal {
            wal.commit_transaction()?;
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct CachedMetadataStore {
    store: Arc<RwLock<MetadataStore>>,
    path: Arc<Path>,
    last_sync: Arc<Mutex<SystemTime>>,
    sync_interval: std::time::Duration,
}

#[allow(dead_code)]
impl CachedMetadataStore {
    pub fn new(path: &Path, wal_path: Option<PathBuf>) -> NpkResult<Self> {
        let store = if path.exists() {
            MetadataStore::load(path, wal_path.clone())?
        } else {
            MetadataStore::new(wal_path.clone())
        };
        
        Ok(Self {
            store: Arc::new(RwLock::new(store)),
            path: Arc::from(path),
            last_sync: Arc::new(Mutex::new(SystemTime::now())),
            sync_interval: std::time::Duration::from_secs(5),
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
        let result = store.mark_deleted(name)?;
        if result && self.should_sync() {
            drop(store);
            self.sync_to_disk()?;
        }
        Ok(result)
    }

    pub fn batch_mark_deleted(&self, names: &[String]) -> NpkResult<usize> {
        let mut store = self.store.write().unwrap();
        let deleted_count = store.batch_mark_deleted(names)?;
        if deleted_count > 0 && self.should_sync() {
            drop(store);
            self.sync_to_disk()?;
        }
        Ok(deleted_count)
    }

    pub fn should_compact(&self, threshold: u64) -> bool {
        if let Ok(store) = self.store.read() {
            store.should_compact(threshold)
        } else {
            false
        }
    }

    pub fn incremental_compact(&self, batch_size: usize) -> NpkResult<Vec<String>> {
        let mut store = self.store.write().unwrap();
        let compacted = store.incremental_compact(batch_size);
        if !compacted.is_empty() {
            store.reset_deleted_size();
            drop(store);
            self.sync_to_disk()?;
        }
        Ok(compacted)
    }

    pub fn get_array(&self, name: &str) -> Option<ArrayMetadata> {
        let store = self.store.read().unwrap();
        store.get_array(name).cloned()
    }

    pub fn list_arrays(&self) -> Vec<String> {
        let store = self.store.read().unwrap();
        store.list_arrays()
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

    pub fn update_array_metadata(&self, name: &str, meta: ArrayMetadata) -> NpkResult<()> {
        let mut store = self.store.write().unwrap();
        store.update_array_metadata(name, meta);
        if self.should_sync() {
            drop(store);
            self.sync_to_disk()?;
        }
        Ok(())
    }
} 