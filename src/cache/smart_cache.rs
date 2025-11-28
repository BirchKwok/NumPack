//! 智能缓存实现
//!
//! 提供自适应缓存管理、访问模式分析和智能缓存块管理

use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::Instant;

// 缓存块大小常量
pub const BASE_CACHE_SIZE: usize = 64 * 1024 * 1024; // 64MB基础缓存大小
pub const CACHE_BLOCK_SIZE: usize = 64 * 1024; // 64KB缓存块大小

// 缓存块结构
#[derive(Debug, Clone)]
pub struct CacheBlock {
    pub data: Vec<u8>,
    last_access: Instant,
    access_count: usize,
    creation_time: Instant,
}

impl CacheBlock {
    pub fn new(data: Vec<u8>) -> Self {
        let now = Instant::now();
        Self {
            data,
            last_access: now,
            access_count: 0,
            creation_time: now,
        }
    }

    pub fn access(&mut self) {
        self.last_access = Instant::now();
        self.access_count += 1;
    }

    pub fn age(&self) -> std::time::Duration {
        self.last_access.elapsed()
    }

    pub fn get_access_frequency(&self) -> f64 {
        let age_seconds = self.creation_time.elapsed().as_secs_f64();
        if age_seconds > 0.0 {
            self.access_count as f64 / age_seconds
        } else {
            0.0
        }
    }
}

// 访问模式分析器
#[derive(Debug)]
pub struct AccessAnalyzer {
    last_access_offset: usize,
    access_stride: usize,
    sequential_count: usize,
    is_sequential: bool,
    recent_hit_rate: f64,
    access_frequency: f64,
    memory_pressure: f64,
}

impl AccessAnalyzer {
    pub fn new() -> Self {
        Self {
            last_access_offset: 0,
            access_stride: 0,
            sequential_count: 0,
            is_sequential: false,
            recent_hit_rate: 0.0,
            access_frequency: 0.0,
            memory_pressure: 0.0,
        }
    }

    pub fn update(&mut self, offset: usize) {
        if self.last_access_offset == 0 {
            self.last_access_offset = offset;
            return;
        }

        let current_stride = if offset > self.last_access_offset {
            offset - self.last_access_offset
        } else {
            self.last_access_offset - offset
        };

        // 检测顺序访问模式
        if current_stride == self.access_stride {
            self.sequential_count += 1;
            if self.sequential_count >= 3 {
                self.is_sequential = true;
            }
        } else {
            self.sequential_count = 0;
            self.is_sequential = false;
        }

        self.access_stride = current_stride;
        self.last_access_offset = offset;
    }

    pub fn is_sequential(&self) -> bool {
        self.is_sequential
    }

    pub fn get_access_stride(&self) -> usize {
        self.access_stride
    }
}

// 智能缓存主结构
pub struct SmartCache {
    blocks: RwLock<HashMap<usize, CacheBlock>>,
    total_size: Arc<Mutex<usize>>,
    access_pattern: Arc<Mutex<AccessAnalyzer>>,
    // 新增：自适应缓存大小
    current_max_size: Arc<Mutex<usize>>,
    last_adjustment: Arc<Mutex<Instant>>,
}

impl SmartCache {
    pub fn new() -> Self {
        Self {
            blocks: RwLock::new(HashMap::new()),
            total_size: Arc::new(Mutex::new(0)),
            access_pattern: Arc::new(Mutex::new(AccessAnalyzer::new())),
            current_max_size: Arc::new(Mutex::new(BASE_CACHE_SIZE)),
            last_adjustment: Arc::new(Mutex::new(Instant::now())),
        }
    }

    pub fn get(&self, block_id: usize) -> Option<Vec<u8>> {
        let mut blocks = self.blocks.write().unwrap();
        if let Some(block) = blocks.get_mut(&block_id) {
            block.access();
            Some(block.data.clone())
        } else {
            None
        }
    }

    pub fn put(&self, block_id: usize, data: Vec<u8>) {
        let data_size = data.len();
        let current_max = *self.current_max_size.lock().unwrap();

        let mut blocks = self.blocks.write().unwrap();
        let mut total_size = self.total_size.lock().unwrap();

        // 检查是否需要清理缓存
        if *total_size + data_size > current_max {
            self.evict_blocks(&mut blocks, &mut total_size, data_size);
        }

        blocks.insert(block_id, CacheBlock::new(data));
        *total_size += data_size;
    }

    // 自适应缓存大小调整
    pub fn adaptive_resize(&self, hit_rate: f64, memory_pressure: f64) {
        let mut current_max = self.current_max_size.lock().unwrap();
        let mut last_adjustment = self.last_adjustment.lock().unwrap();

        // 每30秒最多调整一次
        if last_adjustment.elapsed().as_secs() < 30 {
            return;
        }

        let old_size = *current_max;

        if hit_rate > 0.9 && memory_pressure < 0.7 {
            // 高命中率且内存充足，增加缓存大小
            *current_max = (*current_max as f64 * 1.2) as usize;
            *current_max = (*current_max).min(BASE_CACHE_SIZE * 4); // 最大4倍基础大小
        } else if hit_rate < 0.5 || memory_pressure > 0.8 {
            // 低命中率或内存紧张，减少缓存大小
            *current_max = (*current_max as f64 * 0.8) as usize;
            *current_max = (*current_max).max(BASE_CACHE_SIZE / 4); // 最小1/4基础大小
        }

        if *current_max != old_size {
            *last_adjustment = Instant::now();

            // 如果缓存大小减少，需要立即清理
            if *current_max < old_size {
                let mut blocks = self.blocks.write().unwrap();
                let mut total_size = self.total_size.lock().unwrap();
                self.evict_to_target_size(&mut blocks, &mut total_size, *current_max);
            }
        }
    }

    // 驱逐缓存块以释放空间
    fn evict_blocks(
        &self,
        blocks: &mut HashMap<usize, CacheBlock>,
        total_size: &mut usize,
        needed_size: usize,
    ) {
        let target_size = self.current_max_size.lock().unwrap().clone();

        // 收集所有块的信息用于排序
        let mut block_info: Vec<(usize, Instant, f64)> = blocks
            .iter()
            .map(|(&id, block)| (id, block.last_access, block.get_access_frequency()))
            .collect();

        // 按访问时间和频率排序，优先驱逐老旧且访问频率低的块
        block_info.sort_by(|a, b| {
            let score_a = a.1.elapsed().as_secs_f64() / (a.2 + 1.0);
            let score_b = b.1.elapsed().as_secs_f64() / (b.2 + 1.0);
            score_b
                .partial_cmp(&score_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // 驱逐块直到有足够空间
        for (block_id, _, _) in block_info {
            if *total_size + needed_size <= target_size {
                break;
            }

            if let Some(removed_block) = blocks.remove(&block_id) {
                *total_size -= removed_block.data.len();
            }
        }
    }

    // 驱逐到目标大小
    fn evict_to_target_size(
        &self,
        blocks: &mut HashMap<usize, CacheBlock>,
        total_size: &mut usize,
        target_size: usize,
    ) {
        while *total_size > target_size && !blocks.is_empty() {
            // 找到最老且访问频率最低的块
            let mut worst_block_id = None;
            let mut worst_score = f64::NEG_INFINITY;

            for (&id, block) in blocks.iter() {
                let score = block.last_access.elapsed().as_secs_f64()
                    / (block.get_access_frequency() + 1.0);
                if score > worst_score {
                    worst_score = score;
                    worst_block_id = Some(id);
                }
            }

            if let Some(block_id) = worst_block_id {
                if let Some(removed_block) = blocks.remove(&block_id) {
                    *total_size -= removed_block.data.len();
                }
            } else {
                break;
            }
        }
    }

    pub fn clear(&self) {
        let mut blocks = self.blocks.write().unwrap();
        let mut total_size = self.total_size.lock().unwrap();

        blocks.clear();
        *total_size = 0;
    }

    pub fn get_current_max_size(&self) -> usize {
        *self.current_max_size.lock().unwrap()
    }

    pub fn update_access_pattern(&self, offset: usize) {
        if let Ok(mut analyzer) = self.access_pattern.lock() {
            analyzer.update(offset);
        }
    }

    pub fn get_access_pattern_info(&self) -> (bool, usize) {
        if let Ok(analyzer) = self.access_pattern.lock() {
            (analyzer.is_sequential(), analyzer.get_access_stride())
        } else {
            (false, 0)
        }
    }
}

impl Default for SmartCache {
    fn default() -> Self {
        Self::new()
    }
}
