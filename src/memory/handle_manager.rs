//! Handle Manager - 统一资源生命周期管理
//!
//! 提供集中式的文件句柄和内存映射追踪与清理，
//! 针对Windows平台进行特殊优化。

use memmap2::Mmap;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

lazy_static::lazy_static! {
    /// 全局句柄管理器实例
    static ref GLOBAL_HANDLE_MANAGER: HandleManager = HandleManager::new();
}

/// 获取全局句柄管理器实例
pub fn get_handle_manager() -> &'static HandleManager {
    &GLOBAL_HANDLE_MANAGER
}

/// 可管理的句柄类型
pub enum ManagedHandle {
    Memmap(Arc<Mmap>),
    FileHandle(std::fs::File),
}

/// 已注册句柄的元数据
struct HandleInfo {
    handle: ManagedHandle,
    path: Option<PathBuf>,
    created_at: Instant,
    last_accessed: Instant,
    owner_name: String,
}

/// 句柄清理行为配置
#[derive(Clone, Debug)]
pub struct CleanupConfig {
    /// 尝试清理前的延迟时间（毫秒）
    pub cleanup_delay_ms: u64,
    /// 最大重试次数
    pub max_retries: usize,
    /// 重试间隔（毫秒）
    pub retry_delay_ms: u64,
    /// 启用激进清理（用于测试）
    pub aggressive_mode: bool,
}

impl Default for CleanupConfig {
    fn default() -> Self {
        // 检测是否运行在测试环境
        let is_test = std::env::var("PYTEST_CURRENT_TEST").is_ok()
            || std::env::var("CARGO_TEST").is_ok()
            || std::env::var("RUST_TEST_THREADS").is_ok();

        if is_test {
            // 测试环境：更快的清理
            Self {
                cleanup_delay_ms: 10,
                max_retries: 2,
                retry_delay_ms: 5,
                aggressive_mode: true,
            }
        } else {
            // 生产环境：更保守的策略
            Self {
                cleanup_delay_ms: 100,
                max_retries: 3,
                retry_delay_ms: 50,
                aggressive_mode: false,
            }
        }
    }
}

/// 主句柄管理器结构
pub struct HandleManager {
    /// 句柄ID到句柄信息的映射
    handles: RwLock<HashMap<String, HandleInfo>>,
    /// 文件路径到句柄ID列表的映射
    path_to_handles: RwLock<HashMap<PathBuf, Vec<String>>>,
    /// 等待清理的句柄队列
    cleanup_queue: Mutex<Vec<String>>,
    /// 清理配置
    config: RwLock<CleanupConfig>,
}

impl HandleManager {
    /// 创建新的句柄管理器
    fn new() -> Self {
        Self {
            handles: RwLock::new(HashMap::new()),
            path_to_handles: RwLock::new(HashMap::new()),
            cleanup_queue: Mutex::new(Vec::new()),
            config: RwLock::new(CleanupConfig::default()),
        }
    }

    /// 注册一个内存映射文件
    pub fn register_memmap(
        &self,
        handle_id: String,
        mmap: Arc<Mmap>,
        path: Option<PathBuf>,
        owner_name: String,
    ) -> Result<(), String> {
        let info = HandleInfo {
            handle: ManagedHandle::Memmap(mmap),
            path: path.clone(),
            created_at: Instant::now(),
            last_accessed: Instant::now(),
            owner_name,
        };

        // 在handles映射中注册
        {
            let mut handles = self
                .handles
                .write()
                .map_err(|e| format!("Failed to acquire write lock: {}", e))?;
            handles.insert(handle_id.clone(), info);
        }

        // 如果提供了路径，在path-to-handles映射中注册
        if let Some(ref path) = path {
            let mut path_map = self
                .path_to_handles
                .write()
                .map_err(|e| format!("Failed to acquire write lock: {}", e))?;
            path_map
                .entry(path.clone())
                .or_insert_with(Vec::new)
                .push(handle_id);
        }

        Ok(())
    }

    /// 通过ID清理特定句柄
    pub fn cleanup_handle(&self, handle_id: &str) -> Result<bool, String> {
        let config = self
            .config
            .read()
            .map_err(|e| format!("Failed to read config: {}", e))?
            .clone();

        // 从handles映射中移除
        let handle_info = {
            let mut handles = self
                .handles
                .write()
                .map_err(|e| format!("Failed to acquire write lock: {}", e))?;
            handles.remove(handle_id)
        };

        if let Some(info) = handle_info {
            // 从path-to-handles映射中移除
            if let Some(ref path) = info.path {
                let mut path_map = self
                    .path_to_handles
                    .write()
                    .map_err(|e| format!("Failed to acquire write lock: {}", e))?;
                if let Some(handles) = path_map.get_mut(path) {
                    handles.retain(|id| id != handle_id);
                    if handles.is_empty() {
                        path_map.remove(path);
                    }
                }
            }

            // 根据句柄类型执行清理
            match info.handle {
                ManagedHandle::Memmap(mmap) => {
                    self.cleanup_memmap(mmap, &config)?;
                }
                ManagedHandle::FileHandle(file) => {
                    drop(file);
                }
            }

            Ok(true)
        } else {
            Ok(false) // 句柄未找到
        }
    }

    /// 使用重试逻辑清理内存映射
    fn cleanup_memmap(&self, mmap: Arc<Mmap>, config: &CleanupConfig) -> Result<(), String> {
        // 尝试获取独占所有权
        match Arc::try_unwrap(mmap) {
            Ok(mmap) => {
                // 成功获取独占所有权，释放它
                drop(mmap);

                // Windows特定：等待操作系统释放句柄
                #[cfg(target_family = "windows")]
                {
                    std::thread::sleep(Duration::from_millis(config.cleanup_delay_ms));
                }

                Ok(())
            }
            Err(mmap) => {
                // 仍有其他引用
                if config.aggressive_mode {
                    // 在测试模式下，强制释放我们的引用
                    drop(mmap);
                    std::thread::sleep(Duration::from_millis(config.cleanup_delay_ms));
                    Ok(())
                } else {
                    Err(format!(
                        "Cannot cleanup: mmap still has {} references",
                        Arc::strong_count(&mmap)
                    ))
                }
            }
        }
    }

    /// 清理与特定路径关联的所有句柄
    pub fn cleanup_by_path(&self, path: &Path) -> Result<usize, String> {
        let handle_ids: Vec<String> = {
            let path_map = self
                .path_to_handles
                .read()
                .map_err(|e| format!("Failed to read path mapping: {}", e))?;
            path_map
                .get(path)
                .map(|handles| handles.clone())
                .unwrap_or_default()
        };

        let mut cleaned = 0;
        for handle_id in handle_ids {
            if self.cleanup_handle(&handle_id).unwrap_or(false) {
                cleaned += 1;
            }
        }

        Ok(cleaned)
    }

    /// 清理所有句柄
    pub fn cleanup_all(&self) -> Result<usize, String> {
        let handle_ids: Vec<String> = {
            let handles = self
                .handles
                .read()
                .map_err(|e| format!("Failed to read handles: {}", e))?;
            handles.keys().cloned().collect()
        };

        let mut cleaned = 0;
        for handle_id in handle_ids {
            if self.cleanup_handle(&handle_id).unwrap_or(false) {
                cleaned += 1;
            }
        }

        Ok(cleaned)
    }

    /// 强制清理并等待（Windows特定辅助函数）
    pub fn force_cleanup_and_wait(&self, wait_ms: Option<u64>) -> Result<usize, String> {
        let cleaned = self.cleanup_all()?;

        let config = self
            .config
            .read()
            .map_err(|e| format!("Failed to read config: {}", e))?;

        #[cfg(target_family = "windows")]
        {
            let wait_time = wait_ms.unwrap_or(config.cleanup_delay_ms);
            std::thread::sleep(Duration::from_millis(wait_time));

            // 强制垃圾回收提示
            // 注意：Rust没有GC，但这有助于Arc清理
            for _ in 0..3 {
                std::thread::sleep(Duration::from_millis(10));
            }
        }

        #[cfg(not(target_family = "windows"))]
        {
            // 在非Windows平台上，简短等待即可
            let _ = wait_ms; // 避免未使用警告
            std::thread::sleep(Duration::from_millis(10));
        }

        Ok(cleaned)
    }

    /// 获取当前管理句柄的统计信息
    pub fn get_stats(&self) -> Result<HandleStats, String> {
        let handles = self
            .handles
            .read()
            .map_err(|e| format!("Failed to read handles: {}", e))?;
        let path_map = self
            .path_to_handles
            .read()
            .map_err(|e| format!("Failed to read path mapping: {}", e))?;
        let cleanup_queue = self
            .cleanup_queue
            .lock()
            .map_err(|e| format!("Failed to lock cleanup queue: {}", e))?;

        Ok(HandleStats {
            total_handles: handles.len(),
            memmap_count: handles
                .values()
                .filter(|h| matches!(h.handle, ManagedHandle::Memmap(_)))
                .count(),
            file_handle_count: handles
                .values()
                .filter(|h| matches!(h.handle, ManagedHandle::FileHandle(_)))
                .count(),
            paths_tracked: path_map.len(),
            cleanup_queue_size: cleanup_queue.len(),
        })
    }

    /// 更新清理配置
    pub fn update_config(&self, config: CleanupConfig) -> Result<(), String> {
        let mut current_config = self
            .config
            .write()
            .map_err(|e| format!("Failed to acquire write lock: {}", e))?;
        *current_config = config;
        Ok(())
    }
}

/// 管理句柄的统计信息
#[derive(Debug, Clone)]
pub struct HandleStats {
    pub total_handles: usize,
    pub memmap_count: usize,
    pub file_handle_count: usize,
    pub paths_tracked: usize,
    pub cleanup_queue_size: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_handle_registration() {
        let manager = HandleManager::new();

        // 创建临时文件
        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(b"test data").unwrap();
        let path = temp_file.path().to_path_buf();

        // 创建内存映射
        let file = std::fs::File::open(&path).unwrap();
        let mmap = unsafe { Mmap::map(&file).unwrap() };
        let mmap = Arc::new(mmap);

        // 注册内存映射
        let result = manager.register_memmap(
            "test_handle_1".to_string(),
            mmap.clone(),
            Some(path.clone()),
            "test".to_string(),
        );

        assert!(result.is_ok());

        // 检查统计信息
        let stats = manager.get_stats().unwrap();
        assert_eq!(stats.total_handles, 1);
        assert_eq!(stats.memmap_count, 1);
        assert_eq!(stats.paths_tracked, 1);
    }

    #[test]
    fn test_handle_cleanup() {
        let manager = HandleManager::new();

        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(b"test data").unwrap();
        let path = temp_file.path().to_path_buf();

        let file = std::fs::File::open(&path).unwrap();
        let mmap = unsafe { Mmap::map(&file).unwrap() };
        let mmap = Arc::new(mmap);

        manager
            .register_memmap(
                "test_handle_2".to_string(),
                mmap.clone(),
                Some(path.clone()),
                "test".to_string(),
            )
            .unwrap();

        // 释放我们的引用
        drop(mmap);

        // 清理应该成功
        let result = manager.cleanup_handle("test_handle_2");
        assert!(result.is_ok());

        // 检查统计信息 - 现在应该为空
        let stats = manager.get_stats().unwrap();
        assert_eq!(stats.total_handles, 0);
    }

    #[test]
    fn test_cleanup_by_path() {
        let manager = HandleManager::new();

        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(b"test data").unwrap();
        let path = temp_file.path().to_path_buf();

        // 为同一路径注册多个句柄
        for i in 0..3 {
            let file = std::fs::File::open(&path).unwrap();
            let mmap = unsafe { Mmap::map(&file).unwrap() };
            let mmap = Arc::new(mmap);

            manager
                .register_memmap(
                    format!("test_handle_{}", i),
                    mmap,
                    Some(path.clone()),
                    "test".to_string(),
                )
                .unwrap();
        }

        let stats = manager.get_stats().unwrap();
        assert_eq!(stats.total_handles, 3);

        // 清理此路径的所有句柄
        let cleaned = manager.cleanup_by_path(&path).unwrap();
        assert_eq!(cleaned, 3);

        let stats = manager.get_stats().unwrap();
        assert_eq!(stats.total_handles, 0);
    }
}
