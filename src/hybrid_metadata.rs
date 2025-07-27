//! 统一二进制元数据管理器
//! 
//! 提供高性能二进制格式的元数据管理，完全替代MessagePack格式

use std::path::Path;

use crate::error::{NpkError, NpkResult};
use crate::binary_metadata::{BinaryMetadataStore, BinaryArrayMetadata, BinaryDataType, BinaryCompressionInfo};

/// 统一元数据管理器（仅支持二进制格式）
pub struct UnifiedMetadataManager {
    store: BinaryMetadataStore,
}

impl UnifiedMetadataManager {
    /// 创建新的管理器
    pub fn new() -> Self {
        Self {
            store: BinaryMetadataStore::new(),
        }
    }

    /// 从文件加载元数据
    pub fn load(path: &Path) -> NpkResult<Self> {
        let store = BinaryMetadataStore::load(path)?;
        Ok(Self { store })
    }

    /// 保存元数据
    pub fn save(&self, path: &Path) -> NpkResult<()> {
        self.store.save(path)
    }

    /// 添加数组元数据
    pub fn add_array(&mut self, name: String, shape: Vec<u64>, data_file: String, 
                     dtype: crate::metadata::DataType, compression_info: Option<BinaryCompressionInfo>) -> NpkResult<()> {
        let binary_dtype: BinaryDataType = dtype.into();
        let mut meta = BinaryArrayMetadata::new(name, shape, data_file, binary_dtype);
        if let Some(compression) = compression_info {
            meta.compression = compression;
        }
        self.store.add_array(meta);
        Ok(())
    }

    /// 删除数组元数据
    pub fn remove_array(&mut self, name: &str) -> NpkResult<bool> {
        Ok(self.store.remove_array(name))
    }

    /// 获取数组元数据
    pub fn get_array(&self, name: &str) -> Option<BinaryArrayMetadata> {
        self.store.get_array(name).cloned()
    }

    /// 列出所有数组名称
    pub fn list_arrays(&self) -> Vec<String> {
        self.store.list_arrays()
    }

    /// 检查数组是否存在
    pub fn has_array(&self, name: &str) -> bool {
        self.store.has_array(name)
    }

    /// 获取内部存储的引用
    pub fn get_store(&self) -> &BinaryMetadataStore {
        &self.store
    }

    /// 获取内部存储的可变引用
    pub fn get_store_mut(&mut self) -> &mut BinaryMetadataStore {
        &mut self.store
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_unified_metadata_manager() {
        let temp_dir = TempDir::new().unwrap();
        let metadata_path = temp_dir.path().join("metadata.npkm");

        // 创建管理器并添加数组
        let mut manager = UnifiedMetadataManager::new();

        // 添加测试数组
        let shape = vec![100, 200];
        let data_file = "data_test.npkd".to_string();
        let dtype = crate::metadata::DataType::Float32;
        
        manager.add_array("test_array".to_string(), shape.clone(), data_file.clone(), dtype, None).unwrap();
        
        // 验证数组存在
        assert!(manager.has_array("test_array"));
        
        let meta = manager.get_array("test_array").unwrap();
        assert_eq!(meta.name, "test_array");
        assert_eq!(meta.shape, shape);
        
        // 保存
        manager.save(&metadata_path).unwrap();
        
        // 重新加载并验证
        let loaded_manager = UnifiedMetadataManager::load(&metadata_path).unwrap();
        assert!(loaded_manager.has_array("test_array"));
        
        let loaded_meta = loaded_manager.get_array("test_array").unwrap();
        assert_eq!(loaded_meta.name, "test_array");
        assert_eq!(loaded_meta.shape, shape);
    }
} 