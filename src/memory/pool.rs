//! 内存池管理
//!
//! 从lazy_array_original.rs中提取的内存池实现

use std::sync::Mutex;

/// 分级内存池，减少内存分配/释放的开销
pub struct MemoryPool {
    small_blocks: Mutex<Vec<Vec<u8>>>,  // <1KB
    medium_blocks: Mutex<Vec<Vec<u8>>>, // 1KB-1MB
    large_blocks: Mutex<Vec<Vec<u8>>>,  // >1MB
}

impl MemoryPool {
    pub fn new() -> Self {
        Self {
            small_blocks: Mutex::new(Vec::new()),
            medium_blocks: Mutex::new(Vec::new()),
            large_blocks: Mutex::new(Vec::new()),
        }
    }

    pub fn get_block(&self, size: usize) -> Vec<u8> {
        if size < 1024 {
            let mut blocks = self.small_blocks.lock().unwrap();
            blocks.pop().unwrap_or_else(|| vec![0u8; size])
        } else if size < 1024 * 1024 {
            let mut blocks = self.medium_blocks.lock().unwrap();
            blocks.pop().unwrap_or_else(|| vec![0u8; size])
        } else {
            let mut blocks = self.large_blocks.lock().unwrap();
            blocks.pop().unwrap_or_else(|| vec![0u8; size])
        }
    }

    pub fn return_block(&self, mut block: Vec<u8>) {
        let size = block.len();
        block.clear();

        if size < 1024 {
            let mut blocks = self.small_blocks.lock().unwrap();
            if blocks.len() < 100 {
                // 限制缓存大小
                blocks.push(block);
            }
        } else if size < 1024 * 1024 {
            let mut blocks = self.medium_blocks.lock().unwrap();
            if blocks.len() < 50 {
                blocks.push(block);
            }
        } else {
            let mut blocks = self.large_blocks.lock().unwrap();
            if blocks.len() < 10 {
                blocks.push(block);
            }
        }
    }
}
