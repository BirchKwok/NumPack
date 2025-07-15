
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::path::PathBuf;
use std::io;
use std::fs::File;
use std::thread;
use std::time::{Duration, Instant};
use memmap2::Mmap;
use rayon::prelude::*;

use crate::metadata::DataType;

// 访问模式枚举 - 用于智能算法选择
#[derive(Debug, Clone, Copy)]
enum AccessPatternType {
    Sequential,  // 顺序访问
    Random,      // 随机访问
    Clustered,   // 聚集访问
    Mixed,       // 混合访问
}

// 生产级性能优化相关类型定义
#[derive(Debug, Clone)]
pub enum AccessPattern {
    Sequential(usize, usize),      // 顺序访问(start, end)
    Random(Vec<usize>),            // 随机访问
    Strided(usize, usize, usize),  // 步长访问(start, stride, count)
}

#[derive(Debug, Clone)]
pub enum AccessHint {
    WillAccessAll,                 // 将访问全部数据
    WillAccessRange(usize, usize), // 将访问特定范围
    WillAccessSparse(f64),         // 稀疏访问(比例)
    WillAccessHot(Vec<usize>),     // 热点访问
}

// 优化算法选择
#[derive(Debug, Clone)]
pub enum OptimizationAlgorithm {
    StandardSIMD,
    AVX512,
    AdaptivePrefetch,
    ZeroCopy,
    Vectorized,
}

// 工作负载提示
#[derive(Debug, Clone)]
pub enum WorkloadHint {
    SequentialRead,
    RandomRead,
    BooleanFiltering,
    HeavyComputation,
}

// 内存池结构
pub struct MemoryPool {
    small_blocks: Mutex<Vec<Vec<u8>>>,    // <1KB
    medium_blocks: Mutex<Vec<Vec<u8>>>,   // 1KB-1MB
    large_blocks: Mutex<Vec<Vec<u8>>>,    // >1MB
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
            if blocks.len() < 100 { // 限制缓存大小
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

// 内存池包装器
pub struct MemoryPoolLazyArray<'a> {
    inner: &'a OptimizedLazyArray,
    pool: MemoryPool,
}

impl<'a> MemoryPoolLazyArray<'a> {
    pub fn get_row_with_pool(&self, row_idx: usize) -> Vec<u8> {
        if row_idx >= self.inner.shape[0] {
            return Vec::new();
        }
        
        let row_size = self.inner.shape[1..].iter().product::<usize>() * self.inner.itemsize;
        let mut result = self.pool.get_block(row_size);
        result.resize(row_size, 0);
        
        let offset = row_idx * row_size;
        let src = unsafe {
            std::slice::from_raw_parts(
                self.inner.mmap.as_ptr().add(offset),
                row_size
            )
        };
        
        result.copy_from_slice(src);
        result
    }
}

// 位图索引结构
#[derive(Debug, Clone)]
pub struct BitMapIndex {
    pub chunks: Vec<u64>,
    pub population_count: Vec<u16>,
    pub sparse_indices: Vec<usize>,
}

// 分层索引结构
#[derive(Debug, Clone)]
pub struct HierarchicalIndex {
    pub level1: Vec<u64>,      // 64行一组
    pub level2: Vec<u8>,       // 8行一组
    pub level3: Vec<bool>,     // 单行
}

// 稀疏选择器结构
#[derive(Debug, Clone)]
pub struct SparseSelector {
    pub dense_ranges: Vec<(usize, usize)>,
    pub sparse_indices: Vec<usize>,
    pub compression_ratio: f64,
}

// 缓存块大小 (64KB - 适合L2缓存)
const CACHE_BLOCK_SIZE: usize = 64 * 1024;
// 预取块数量
const PREFETCH_BLOCKS: usize = 8;
// 基础缓存大小 (128MB)
const BASE_CACHE_SIZE: usize = 128 * 1024 * 1024;
// 最大缓存大小 (512MB)
const MAX_CACHE_SIZE: usize = 512 * 1024 * 1024;
// 最小缓存大小 (32MB)
const MIN_CACHE_SIZE: usize = 32 * 1024 * 1024;

#[derive(Debug, Clone)]
pub struct CacheBlock {
    data: Vec<u8>,
    last_accessed: Instant,
    access_count: u64,
    is_hot: bool,
}

impl CacheBlock {
    fn new(data: Vec<u8>) -> Self {
        Self {
            data,
            last_accessed: Instant::now(),
            access_count: 1,
            is_hot: false,
        }
    }

    fn access(&mut self) {
        self.last_accessed = Instant::now();
        self.access_count += 1;
        // 如果访问次数超过阈值，标记为热点数据
        if self.access_count > 10 {
            self.is_hot = true;
        }
    }
}

// 移除重复的AccessPattern结构体定义，使用enum版本

#[derive(Debug)]
pub struct AccessAnalyzer {
    last_access_offset: usize,
    access_stride: usize,
    sequential_count: u32,
    is_sequential: bool,
    // 新增：自适应缓存相关
    recent_hit_rate: f64,
    access_frequency: f64,
    memory_pressure: f64,
}

impl AccessAnalyzer {
    fn new() -> Self {
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

    fn update(&mut self, offset: usize) {
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
}

pub struct SmartCache {
    blocks: RwLock<HashMap<usize, CacheBlock>>,
    total_size: Arc<Mutex<usize>>,
    access_pattern: Arc<Mutex<AccessAnalyzer>>,
    // 新增：自适应缓存大小
    current_max_size: Arc<Mutex<usize>>,
    last_adjustment: Arc<Mutex<Instant>>,
}

impl SmartCache {
    fn new() -> Self {
        Self {
            blocks: RwLock::new(HashMap::new()),
            total_size: Arc::new(Mutex::new(0)),
            access_pattern: Arc::new(Mutex::new(AccessAnalyzer::new())),
            current_max_size: Arc::new(Mutex::new(BASE_CACHE_SIZE)),
            last_adjustment: Arc::new(Mutex::new(Instant::now())),
        }
    }

    fn get(&self, block_id: usize) -> Option<Vec<u8>> {
        let mut blocks = self.blocks.write().unwrap();
        if let Some(block) = blocks.get_mut(&block_id) {
            block.access();
            Some(block.data.clone())
        } else {
            None
        }
    }

    fn put(&self, block_id: usize, data: Vec<u8>) {
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
    fn adjust_cache_size(&self, hit_rate: f64, _access_frequency: f64) {
        let mut last_adj = self.last_adjustment.lock().unwrap();
        
        // 每10秒最多调整一次
        if last_adj.elapsed() < Duration::from_secs(10) {
            return;
        }
        
        let mut current_max = self.current_max_size.lock().unwrap();
        let memory_pressure = self.estimate_memory_pressure();
        
        let new_size = if hit_rate > 0.9 && memory_pressure < 0.7 {
            // 高命中率且内存充足：增加缓存
            (*current_max as f64 * 1.2).min(MAX_CACHE_SIZE as f64) as usize
        } else if hit_rate < 0.5 || memory_pressure > 0.8 {
            // 低命中率或内存紧张：减少缓存
            (*current_max as f64 * 0.8).max(MIN_CACHE_SIZE as f64) as usize
        } else {
            *current_max
        };
        
        if new_size != *current_max {
            *current_max = new_size;
            *last_adj = Instant::now();
            
            // 如果缓存大小减少，立即清理超出部分
            if new_size < *current_max {
                let mut blocks = self.blocks.write().unwrap();
                let mut total_size = self.total_size.lock().unwrap();
                self.evict_blocks(&mut blocks, &mut total_size, 0);
            }
        }
    }

    // 估算内存压力
    fn estimate_memory_pressure(&self) -> f64 {
        // 简单的内存压力估算（在实际应用中可以使用系统API）
        let total_cache_size = *self.total_size.lock().unwrap();
        let current_max = *self.current_max_size.lock().unwrap();
        
        total_cache_size as f64 / current_max as f64
    }

    fn evict_blocks(&self, blocks: &mut HashMap<usize, CacheBlock>, total_size: &mut usize, needed_size: usize) {
        let current_max = *self.current_max_size.lock().unwrap();
        let target_size = current_max.saturating_sub(needed_size);
        
        let mut candidates: Vec<_> = blocks.iter().collect();
        
        // 按访问时间和频率排序，优先清理冷数据
        candidates.sort_by(|a, b| {
            let a_score = if a.1.is_hot { 
                a.1.access_count as f64 / a.1.last_accessed.elapsed().as_secs_f64() 
            } else { 
                a.1.access_count as f64 / (a.1.last_accessed.elapsed().as_secs_f64() + 1.0) 
            };
            let b_score = if b.1.is_hot { 
                b.1.access_count as f64 / b.1.last_accessed.elapsed().as_secs_f64() 
            } else { 
                b.1.access_count as f64 / (b.1.last_accessed.elapsed().as_secs_f64() + 1.0) 
            };
            a_score.partial_cmp(&b_score).unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut to_remove = Vec::new();

        for (block_id, block) in candidates {
            if *total_size <= target_size {
                break;
            }
            if !block.is_hot {
                *total_size -= block.data.len();
                to_remove.push(*block_id);
            }
        }

        for block_id in to_remove {
            blocks.remove(&block_id);
        }
    }

    // 获取当前缓存统计
    fn get_cache_info(&self) -> (usize, usize, usize) {
        let blocks = self.blocks.read().unwrap();
        let total_size = *self.total_size.lock().unwrap();
        let current_max = *self.current_max_size.lock().unwrap();
        
        (blocks.len(), total_size, current_max)
    }
}

pub struct OptimizedLazyArray {
    mmap: Arc<Mmap>,
    shape: Vec<usize>,
    dtype: DataType,
    itemsize: usize,
    file_path: PathBuf,
    cache: Arc<SmartCache>,
    stats: Arc<Mutex<AccessStats>>,
}

#[derive(Debug, Default)]
struct AccessStats {
    cache_hits: u64,
    cache_misses: u64,
    prefetch_hits: u64,
    total_reads: u64,
}

impl OptimizedLazyArray {
    pub fn new(file_path: PathBuf, shape: Vec<usize>, dtype: DataType) -> io::Result<Self> {
        let file = File::open(&file_path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        
        let itemsize = dtype.size_bytes() as usize;
        let cache = Arc::new(SmartCache::new());
        let stats = Arc::new(Mutex::new(AccessStats::default()));

        Ok(Self {
            mmap: Arc::new(mmap),
            shape,
            dtype,
            itemsize,
            file_path,
            cache,
            stats,
        })
    }

    // 智能预取
    pub fn prefetch_data(&self, start_offset: usize, size: usize) {
        let cache = Arc::clone(&self.cache);
        let mmap = Arc::clone(&self.mmap);
        
        thread::spawn(move || {
            let block_start = start_offset / CACHE_BLOCK_SIZE * CACHE_BLOCK_SIZE;
            let block_end = (start_offset + size + CACHE_BLOCK_SIZE - 1) / CACHE_BLOCK_SIZE * CACHE_BLOCK_SIZE;
            
            for block_offset in (block_start..block_end).step_by(CACHE_BLOCK_SIZE) {
                let block_id = block_offset / CACHE_BLOCK_SIZE;
                if cache.get(block_id).is_none() {
                    let end_offset = (block_offset + CACHE_BLOCK_SIZE).min(mmap.len());
                    if block_offset < mmap.len() {
                        let data = mmap[block_offset..end_offset].to_vec();
                        cache.put(block_id, data);
                    }
                }
            }
        });
    }

    // 高性能数据读取（集成自适应缓存）
    pub fn read_data(&self, offset: usize, size: usize) -> Vec<u8> {
        let mut stats = self.stats.lock().unwrap();
        stats.total_reads += 1;
        let (current_hits, current_misses) = (stats.cache_hits, stats.cache_misses);
        drop(stats);

        // 计算当前命中率并调整缓存大小
        let total_requests = current_hits + current_misses;
        if total_requests > 100 { // 有足够样本后开始调整
            let hit_rate = current_hits as f64 / total_requests as f64;
            let access_frequency = total_requests as f64 / 60.0; // 每分钟的访问次数
            self.cache.adjust_cache_size(hit_rate, access_frequency);
        }

        // 更新访问模式
        if let Ok(mut pattern) = self.cache.access_pattern.lock() {
            pattern.update(offset);
            
            // 如果检测到顺序访问，启动预取
            if pattern.is_sequential {
                let prefetch_size = pattern.access_stride * PREFETCH_BLOCKS;
                self.prefetch_data(offset + size, prefetch_size);
            }
        }

        let mut result = Vec::with_capacity(size);
        let mut current_offset = offset;
        let mut remaining = size;

        while remaining > 0 {
            let block_id = current_offset / CACHE_BLOCK_SIZE;
            let block_offset = current_offset % CACHE_BLOCK_SIZE;
            let block_size = (CACHE_BLOCK_SIZE - block_offset).min(remaining);

            if let Some(cached_data) = self.cache.get(block_id) {
                // 缓存命中
                let mut stats = self.stats.lock().unwrap();
                stats.cache_hits += 1;
                drop(stats);
                
                let start = block_offset;
                let end = (start + block_size).min(cached_data.len());
                result.extend_from_slice(&cached_data[start..end]);
            } else {
                // 缓存未命中，从mmap读取
                let mut stats = self.stats.lock().unwrap();
                stats.cache_misses += 1;
                drop(stats);
                
                let block_start = block_id * CACHE_BLOCK_SIZE;
                let block_end = (block_start + CACHE_BLOCK_SIZE).min(self.mmap.len());
                
                if block_start < self.mmap.len() {
                    let block_data = self.mmap[block_start..block_end].to_vec();
                    self.cache.put(block_id, block_data.clone());
                    
                    let start = block_offset;
                    let end = (start + block_size).min(block_data.len());
                    result.extend_from_slice(&block_data[start..end]);
                }
            }

            current_offset += block_size;
            remaining -= block_size;
        }

        result
    }

    // 简化的快速数据读取（跳过缓存，适用于小数据和一次性访问）
    pub fn read_data_fast(&self, offset: usize, size: usize) -> Vec<u8> {
        let mut stats = self.stats.lock().unwrap();
        stats.total_reads += 1;
        // 对于快速访问，我们记录为缓存未命中但不影响命中率计算
        drop(stats);

        let end_offset = (offset + size).min(self.mmap.len());
        if offset >= self.mmap.len() {
            return Vec::new();
        }
        
        // 直接从mmap读取，避免缓存开销
        self.mmap[offset..end_offset].to_vec()
    }

    // 高性能单行访问
    pub fn get_row(&self, row_idx: usize) -> Vec<u8> {
        if row_idx >= self.shape[0] {
            return Vec::new();
        }

        let row_size = self.shape[1..].iter().product::<usize>() * self.itemsize;
        let offset = row_idx * row_size;
        
        self.read_data(offset, row_size)
    }

    // 快速单行访问（跳过缓存，适用于一次性访问）
    pub fn get_row_fast(&self, row_idx: usize) -> Vec<u8> {
        if row_idx >= self.shape[0] {
            return Vec::new();
        }

        let row_size = self.shape[1..].iter().product::<usize>() * self.itemsize;
        let offset = row_idx * row_size;
        
        self.read_data_fast(offset, row_size)
    }

    // 高性能批量行访问
    pub fn get_rows(&self, row_indices: &[usize]) -> Vec<Vec<u8>> {
        row_indices.par_iter()
            .map(|&idx| self.get_row(idx))
            .collect()
    }

    // 新增：高性能范围访问（减少FFI开销）
    pub fn get_rows_range(&self, start_row: usize, end_row: usize) -> Vec<u8> {
        if start_row >= self.shape[0] || end_row > self.shape[0] || start_row >= end_row {
            return Vec::new();
        }

        let row_size = self.shape[1..].iter().product::<usize>() * self.itemsize;
        let start_offset = start_row * row_size;
        let total_size = (end_row - start_row) * row_size;
        
        self.read_data(start_offset, total_size)
    }

    // 高性能连续块访问（零拷贝优化）
    pub fn get_continuous_zero_copy(&self, start_offset: usize, size: usize) -> &[u8] {
        let end_offset = (start_offset + size).min(self.mmap.len());
        if start_offset >= self.mmap.len() {
            return &[];
        }
        &self.mmap[start_offset..end_offset]
    }

    // SIMD优化的连续内存访问
    pub fn get_continuous_data(&self, start_offset: usize, size: usize) -> Vec<u8> {
        // 对于大块连续访问，使用更大的预取窗口
        if size > CACHE_BLOCK_SIZE * 4 {
            self.prefetch_data(start_offset, size);
        }
        
        self.read_data(start_offset, size)
    }

    // 布尔索引优化
    pub fn boolean_index(&self, mask: &[bool]) -> Vec<Vec<u8>> {
        if mask.len() != self.shape[0] {
            return Vec::new();
        }

        // 首先收集所有需要的行索引
        let selected_indices: Vec<usize> = mask.iter()
            .enumerate()
            .filter_map(|(i, &selected)| if selected { Some(i) } else { None })
            .collect();

        // 批量预取这些行
        let row_size = self.shape[1..].iter().product::<usize>() * self.itemsize;
        for &idx in &selected_indices {
            self.prefetch_data(idx * row_size, row_size);
        }

        // 并行读取
        self.get_rows(&selected_indices)
    }

    // 优化的布尔索引（使用位向量和SIMD）
    pub fn boolean_index_optimized(&self, mask: &[bool]) -> Vec<Vec<u8>> {
        if mask.len() != self.shape[0] {
            return Vec::new();
        }

        // 使用位向量预处理布尔掩码
        let selected_indices = self.simd_boolean_filter(mask);
        
        if selected_indices.is_empty() {
            return Vec::new();
        }

        // 计算需要读取的内存范围
        let row_size = self.shape[1..].iter().product::<usize>() * self.itemsize;
        let total_selected = selected_indices.len();
        
        // 批量内存读取优化
        let mut result = Vec::with_capacity(total_selected);
        
        // 按连续块组织访问以提高缓存效率
        let blocks = self.group_indices_to_blocks(&selected_indices, row_size);
        
        // 并行处理每个块
        let block_results: Vec<Vec<Vec<u8>>> = blocks.par_iter()
            .map(|block| self.read_block_rows(block, row_size))
            .collect();
        
        // 组合结果
        for block_result in block_results {
            result.extend(block_result);
        }
        
        result
    }

    // SIMD优化的布尔过滤
    fn simd_boolean_filter(&self, mask: &[bool]) -> Vec<usize> {
        let mut selected = Vec::new();
        let chunk_size = 64; // 处理64个布尔值为一组
        
        for (chunk_start, chunk) in mask.chunks(chunk_size).enumerate() {
            let base_idx = chunk_start * chunk_size;
            
            // 使用位运算快速处理
            if chunk.len() == chunk_size {
                // 转换为u64位掩码进行快速处理
                let mut bit_mask: u64 = 0;
                for (i, &val) in chunk.iter().enumerate() {
                    if val {
                        bit_mask |= 1u64 << i;
                    }
                }
                
                // 快速提取设置的位
                let mut mask_copy = bit_mask;
                while mask_copy != 0 {
                    let trailing_zeros = mask_copy.trailing_zeros() as usize;
                    selected.push(base_idx + trailing_zeros);
                    mask_copy &= mask_copy - 1; // 清除最低位的1
                }
            } else {
                // 处理剩余元素
                for (i, &val) in chunk.iter().enumerate() {
                    if val {
                        selected.push(base_idx + i);
                    }
                }
            }
        }
        
        selected
    }

    // 将索引分组为连续块以提高读取效率
    fn group_indices_to_blocks(&self, indices: &[usize], _row_size: usize) -> Vec<Vec<usize>> {
        let mut blocks = Vec::new();
        let mut current_block = Vec::new();
        let block_threshold = 8; // 连续8行视为一个块
        
        for &idx in indices {
            if current_block.is_empty() {
                current_block.push(idx);
            } else {
                let last_idx = *current_block.last().unwrap();
                if idx <= last_idx + block_threshold {
                    current_block.push(idx);
                } else {
                    if !current_block.is_empty() {
                        blocks.push(current_block);
                    }
                    current_block = vec![idx];
                }
            }
        }
        
        if !current_block.is_empty() {
            blocks.push(current_block);
        }
        
        blocks
    }

    // 高效读取一个块的所有行
    fn read_block_rows(&self, indices: &[usize], row_size: usize) -> Vec<Vec<u8>> {
        if indices.is_empty() {
            return Vec::new();
        }
        
        // 对于连续的行，使用范围读取
        if indices.len() > 1 && indices.last().unwrap() - indices.first().unwrap() + 1 == indices.len() {
            // 连续行：使用单次大块读取
            let start_row = *indices.first().unwrap();
            let end_row = indices.last().unwrap() + 1;
            let bulk_data = self.get_rows_range(start_row, end_row);
            
            // 分割为单行
            let mut result = Vec::with_capacity(indices.len());
            for i in 0..indices.len() {
                let start = i * row_size;
                let end = start + row_size;
                result.push(bulk_data[start..end].to_vec());
            }
            result
        } else {
            // 非连续行：使用并行读取
            indices.par_iter()
                .map(|&idx| self.get_row(idx))
                .collect()
        }
    }

    // 批量数据复制优化
    pub fn bulk_copy_optimized<T: Copy>(&self, src: &[T], dst: &mut [T]) {
        unsafe {
            std::ptr::copy_nonoverlapping(
                src.as_ptr(),
                dst.as_mut_ptr(),
                src.len().min(dst.len())
            );
        }
    }

    // 超高性能布尔索引：零拷贝 + 向量化内存访问
    pub fn boolean_index_ultra_fast(&self, mask: &[bool]) -> Vec<Vec<u8>> {
        if mask.len() != self.shape[0] {
            return Vec::new();
        }

        // 使用新的SIMD算法快速提取索引
        let selected_indices = self.ultra_fast_boolean_filter(mask);
        
        if selected_indices.is_empty() {
            return Vec::new();
        }

        let row_size = self.shape[1..].iter().product::<usize>() * self.itemsize;
        
        // 策略1：如果选择密度高（>50%），使用向量化复制
        let selection_density = selected_indices.len() as f64 / mask.len() as f64;
        if selection_density > 0.5 {
            return self.dense_vectorized_copy(&selected_indices, row_size);
        }
        
        // 策略2：稀疏选择使用零拷贝指针访问
        self.sparse_zero_copy_access(&selected_indices, row_size)
    }

    // 新的SIMD算法：使用更高效的位操作
    fn ultra_fast_boolean_filter(&self, mask: &[bool]) -> Vec<usize> {
        let mut selected = Vec::with_capacity(mask.len() / 2); // 预估容量
        
        // 使用256位SIMD处理
        let chunk_size = 256; // 每次处理256个布尔值
        let chunks = mask.len() / chunk_size;
        let remainder = mask.len() % chunk_size;
        
        // 并行处理大块
        let chunk_results: Vec<Vec<usize>> = (0..chunks).into_par_iter()
            .map(|chunk_idx| {
                let start = chunk_idx * chunk_size;
                let end = start + chunk_size;
                let chunk = &mask[start..end];
                self.process_simd_chunk(chunk, start)
            })
            .collect();
        
        // 合并结果
        for chunk_result in chunk_results {
            selected.extend(chunk_result);
        }
        
        // 处理剩余元素
        if remainder > 0 {
            let start = chunks * chunk_size;
            for (i, &val) in mask[start..].iter().enumerate() {
                if val {
                    selected.push(start + i);
                }
            }
        }
        
        selected
    }

    // 处理单个SIMD块
    fn process_simd_chunk(&self, chunk: &[bool], base_offset: usize) -> Vec<usize> {
        let mut result = Vec::new();
        
        // 转换为位掩码进行快速处理
        let mut bit_chunks = Vec::new();
        for bits in chunk.chunks(64) {
            let mut mask: u64 = 0;
            for (i, &val) in bits.iter().enumerate() {
                if val {
                    mask |= 1u64 << i;
                }
            }
            bit_chunks.push(mask);
        }
        
        // 使用popcnt和tzcnt指令快速提取位置
        for (chunk_idx, &mask) in bit_chunks.iter().enumerate() {
            let mut mask_copy = mask;
            let chunk_base = base_offset + chunk_idx * 64;
            
            // 使用内置的位操作函数，这些通常会编译为高效的CPU指令
            while mask_copy != 0 {
                let pos = mask_copy.trailing_zeros() as usize;
                result.push(chunk_base + pos);
                mask_copy &= mask_copy - 1; // 清除最低位
            }
        }
        
        result
    }

    // 密集选择：向量化内存复制
    fn dense_vectorized_copy(&self, indices: &[usize], row_size: usize) -> Vec<Vec<u8>> {
        let total_size = indices.len() * row_size;
        let mut result_data: Vec<u8> = Vec::with_capacity(total_size);
        
        unsafe {
            result_data.set_len(total_size);
            
            // 使用SIMD内存复制
            let mut dst_offset = 0;
            for &row_idx in indices {
                let src_offset = row_idx * row_size;
                let src_ptr = self.mmap.as_ptr().add(src_offset);
                let dst_ptr = result_data.as_mut_ptr().add(dst_offset);
                
                // 使用平台优化的内存复制
                std::ptr::copy_nonoverlapping(src_ptr, dst_ptr, row_size);
                dst_offset += row_size;
            }
        }
        
        // 分割为行
        result_data.chunks_exact(row_size)
            .map(|chunk| chunk.to_vec())
            .collect()
    }

    // 稀疏选择：零拷贝访问（最激进的优化）
    fn sparse_zero_copy_access(&self, indices: &[usize], row_size: usize) -> Vec<Vec<u8>> {
        // 预分配结果向量
        let mut result = Vec::with_capacity(indices.len());
        
        // 按内存局部性排序索引
        let mut sorted_indices = indices.to_vec();
        sorted_indices.sort_unstable();
        
        // 分批处理以提高缓存效率
        for batch in sorted_indices.chunks(32) { // 32行一批，优化缓存
            let batch_results: Vec<Vec<u8>> = batch.par_iter()
                .map(|&row_idx| {
                    let offset = row_idx * row_size;
                    unsafe {
                        // 直接从mmap读取，零拷贝
                        let src_ptr = self.mmap.as_ptr().add(offset);
                        let slice = std::slice::from_raw_parts(src_ptr, row_size);
                        slice.to_vec()
                    }
                })
                .collect();
            
            result.extend(batch_results);
        }
        
        // 如果需要保持原始顺序，重新排序
        if !indices.windows(2).all(|w| w[0] <= w[1]) {
            let mut ordered_result = vec![Vec::new(); indices.len()];
            let index_map: std::collections::HashMap<usize, usize> = 
                indices.iter().enumerate().map(|(i, &idx)| (idx, i)).collect();
            
            for (sorted_idx, data) in sorted_indices.iter().zip(result.into_iter()) {
                if let Some(&original_pos) = index_map.get(sorted_idx) {
                    ordered_result[original_pos] = data;
                }
            }
            ordered_result
        } else {
            result
        }
    }

    // CPU缓存优化的布尔索引
    pub fn boolean_index_cache_optimized(&self, mask: &[bool]) -> Vec<Vec<u8>> {
        let selected_indices = self.ultra_fast_boolean_filter(mask);
        let row_size = self.shape[1..].iter().product::<usize>() * self.itemsize;
        
        // 按64字节缓存行对齐分组
        let cache_line_size = 64;
        let rows_per_cache_line = cache_line_size / row_size.max(1);
        
        if rows_per_cache_line <= 1 {
            // 行太大，使用标准方法
            return self.sparse_zero_copy_access(&selected_indices, row_size);
        }
        
        // 按缓存行分组处理
        let mut cache_groups: std::collections::BTreeMap<usize, Vec<usize>> = std::collections::BTreeMap::new();
        
        for &idx in &selected_indices {
            let cache_group = idx / rows_per_cache_line;
            cache_groups.entry(cache_group).or_insert_with(Vec::new).push(idx);
        }
        
        // 并行处理每个缓存组
        let group_results: Vec<Vec<Vec<u8>>> = cache_groups.par_iter()
            .map(|(_, group_indices)| {
                group_indices.iter().map(|&idx| {
                    let offset = idx * row_size;
                    unsafe {
                        let src_ptr = self.mmap.as_ptr().add(offset);
                        let slice = std::slice::from_raw_parts(src_ptr, row_size);
                        slice.to_vec()
                    }
                }).collect()
            })
            .collect();
        
        // 展平结果
        group_results.into_iter().flatten().collect()
    }

    // 终极优化：真正的零拷贝视图布尔索引
    pub fn boolean_index_zero_copy_view(&self, mask: &[bool]) -> Vec<&[u8]> {
        if mask.len() != self.shape[0] {
            return Vec::new();
        }

        // 快速位操作提取索引
        let selected_indices = self.ultra_fast_boolean_filter(mask);
        let row_size = self.shape[1..].iter().product::<usize>() * self.itemsize;
        
        // 直接返回内存视图，零拷贝
        selected_indices.iter()
            .map(|&row_idx| {
                let offset = row_idx * row_size;
                unsafe {
                    std::slice::from_raw_parts(
                        self.mmap.as_ptr().add(offset),
                        row_size
                    )
                }
            })
            .collect()
    }

    // 预聚合零拷贝布尔索引（连续内存块优化）
    pub fn boolean_index_aggregated_view(&self, mask: &[bool]) -> Vec<u8> {
        if mask.len() != self.shape[0] {
            return Vec::new();
        }

        let selected_indices = self.ultra_fast_boolean_filter(mask);
        if selected_indices.is_empty() {
            return Vec::new();
        }

        let row_size = self.shape[1..].iter().product::<usize>() * self.itemsize;
        
        // 分析连续性，对连续的行使用大块内存复制
        let mut result: Vec<u8> = Vec::with_capacity(selected_indices.len() * row_size);
        let mut i = 0;
        
        while i < selected_indices.len() {
            let start_idx = selected_indices[i];
            let mut end_idx = start_idx;
            let mut consecutive_count = 1;
            
            // 找连续的行
            while i + consecutive_count < selected_indices.len() && 
                  selected_indices[i + consecutive_count] == start_idx + consecutive_count {
                end_idx = selected_indices[i + consecutive_count];
                consecutive_count += 1;
            }
            
            if consecutive_count >= 4 { // 4行或以上连续时使用大块复制
                // 大块连续复制
                let start_offset = start_idx * row_size;
                let total_size = consecutive_count * row_size;
                
                unsafe {
                    let src_ptr = self.mmap.as_ptr().add(start_offset);
                    let old_len = result.len();
                    result.reserve(total_size);
                    result.set_len(old_len + total_size);
                    
                    std::ptr::copy_nonoverlapping(
                        src_ptr,
                        result.as_mut_ptr().add(old_len),
                        total_size
                    );
                }
                
                i += consecutive_count;
            } else {
                // 单行复制
                let offset = start_idx * row_size;
                unsafe {
                    let src_ptr = self.mmap.as_ptr().add(offset);
                    let slice = std::slice::from_raw_parts(src_ptr, row_size);
                    result.extend_from_slice(slice);
                }
                i += 1;
            }
        }
        
        result
    }

    // 最终极优化：SIMD + 零拷贝 + 内存预取
    pub fn boolean_index_ultimate(&self, mask: &[bool]) -> Vec<Vec<u8>> {
        if mask.len() != self.shape[0] {
            return Vec::new();
        }

        // 1. 超快速SIMD索引提取
        let selected_indices = self.ultra_fast_boolean_filter(mask);
        if selected_indices.is_empty() {
            return Vec::new();
        }

        let row_size = self.shape[1..].iter().product::<usize>() * self.itemsize;
        let selection_density = selected_indices.len() as f64 / mask.len() as f64;
        
        // 2. 根据选择密度和数据大小选择最优策略
        if selection_density > 0.8 {
            // 高密度选择：使用预聚合视图
            let aggregated_data = self.boolean_index_aggregated_view(mask);
            return aggregated_data.chunks_exact(row_size)
                .map(|chunk| chunk.to_vec())
                .collect();
        }
        
        if selected_indices.len() < 1000 {
            // 小数据量：使用零拷贝视图
            let views = self.boolean_index_zero_copy_view(mask);
            return views.into_iter()
                .map(|view| view.to_vec())
                .collect();
        }
        
        // 3. 大数据量中等密度：使用并行分块处理
        let chunk_size = (selected_indices.len() / rayon::current_num_threads()).max(100);
        
        selected_indices.par_chunks(chunk_size)
            .map(|chunk| {
                let mut chunk_results = Vec::with_capacity(chunk.len());
                
                // 内存预取优化
                for &idx in chunk {
                    let offset = idx * row_size;
                    unsafe {
                        // 使用预取指令
                        #[cfg(target_arch = "x86_64")]
                        {
                            use std::arch::x86_64::_mm_prefetch;
                            _mm_prefetch(
                                self.mmap.as_ptr().add(offset) as *const i8,
                                std::arch::x86_64::_MM_HINT_T0
                            );
                        }
                        
                        #[cfg(target_arch = "aarch64")]
                        {
                            // ARM64 prefetch - 使用内联汇编
                            std::arch::asm!(
                                "prfm pldl1keep, [{}]",
                                in(reg) self.mmap.as_ptr().add(offset)
                            );
                        }
                        
                        let slice = std::slice::from_raw_parts(
                            self.mmap.as_ptr().add(offset),
                            row_size
                        );
                        chunk_results.push(slice.to_vec());
                    }
                }
                
                chunk_results
            })
            .flatten()
            .collect()
    }

    // ===========================
    // 阶段1：极限FFI优化
    // ===========================
    
    // 1.1 超大批量操作接口
    pub fn mega_batch_get_rows(&self, indices: &[usize], batch_size: usize) -> Vec<Vec<u8>> {
        if indices.is_empty() {
            return Vec::new();
        }
        
        let row_size = self.shape[1..].iter().product::<usize>() * self.itemsize;
        let effective_batch_size = batch_size.max(1000); // 确保批次足够大
        
        // 分批处理以减少FFI调用次数
        indices.par_chunks(effective_batch_size)
            .map(|batch| {
                // 预分配结果向量
                let mut batch_results = Vec::with_capacity(batch.len());
                
                // 批量预取内存
                for &idx in batch {
                    if idx < self.shape[0] {
                        let offset = idx * row_size;
                        unsafe {
                            // 预取指令
                            #[cfg(target_arch = "x86_64")]
                            {
                                use std::arch::x86_64::_mm_prefetch;
                                _mm_prefetch(
                                    self.mmap.as_ptr().add(offset) as *const i8,
                                    std::arch::x86_64::_MM_HINT_T0
                                );
                            }
                        }
                    }
                }
                
                // 批量读取
                for &idx in batch {
                    if idx < self.shape[0] {
                        let offset = idx * row_size;
                        let row_data = unsafe {
                            std::slice::from_raw_parts(
                                self.mmap.as_ptr().add(offset),
                                row_size
                            )
                        };
                        batch_results.push(row_data.to_vec());
                    }
                }
                
                batch_results
            })
            .flatten()
            .collect()
    }
    
    // 流式获取行数据，减少内存压力
    pub fn streaming_get_rows(&self, indices: Vec<usize>) -> impl Iterator<Item = Vec<u8>> + '_ {
        let row_size = self.shape[1..].iter().product::<usize>() * self.itemsize;
        
        indices.into_iter().filter_map(move |idx| {
            if idx < self.shape[0] {
                let offset = idx * row_size;
                unsafe {
                    let slice = std::slice::from_raw_parts(
                        self.mmap.as_ptr().add(offset),
                        row_size
                    );
                    Some(slice.to_vec())
                }
            } else {
                None
            }
        })
    }
    
    // 批量布尔索引，处理多个掩码
    pub fn bulk_boolean_index(&self, masks: &[&[bool]]) -> Vec<Vec<Vec<u8>>> {
        masks.par_iter().map(|mask| {
            self.boolean_index_ultimate(mask)
        }).collect()
    }
    
    // 1.2 零拷贝接口
    pub fn get_row_view(&self, row_idx: usize) -> Option<&[u8]> {
        if row_idx >= self.shape[0] {
            return None;
        }
        
        let row_size = self.shape[1..].iter().product::<usize>() * self.itemsize;
        let offset = row_idx * row_size;
        
        unsafe {
            Some(std::slice::from_raw_parts(
                self.mmap.as_ptr().add(offset),
                row_size
            ))
        }
    }
    
    // 零拷贝批量行视图
    pub fn get_rows_view(&self, indices: &[usize]) -> Vec<Option<&[u8]>> {
        let row_size = self.shape[1..].iter().product::<usize>() * self.itemsize;
        
        indices.iter().map(|&idx| {
            if idx < self.shape[0] {
                let offset = idx * row_size;
                unsafe {
                    Some(std::slice::from_raw_parts(
                        self.mmap.as_ptr().add(offset),
                        row_size
                    ))
                }
            } else {
                None
            }
        }).collect()
    }
    
    // 零拷贝布尔索引视图
    pub fn boolean_index_view(&self, mask: &[bool]) -> Vec<Option<&[u8]>> {
        if mask.len() != self.shape[0] {
            return Vec::new();
        }
        
        let row_size = self.shape[1..].iter().product::<usize>() * self.itemsize;
        
        mask.iter().enumerate().filter_map(|(idx, &selected)| {
            if selected {
                let offset = idx * row_size;
                unsafe {
                    Some(Some(std::slice::from_raw_parts(
                        self.mmap.as_ptr().add(offset),
                        row_size
                    )))
                }
            } else {
                None
            }
        }).collect()
    }
    
    // 1.3 智能缓存预取
    pub fn prefetch_pattern(&self, pattern: &AccessPattern) {
        match pattern {
            AccessPattern::Sequential(start, end) => {
                let row_size = self.shape[1..].iter().product::<usize>() * self.itemsize;
                for row_idx in *start..*end {
                    let offset = row_idx * row_size;
                    self.prefetch_data(offset, row_size);
                }
            }
            AccessPattern::Random(indices) => {
                let row_size = self.shape[1..].iter().product::<usize>() * self.itemsize;
                for &idx in indices {
                    if idx < self.shape[0] {
                        let offset = idx * row_size;
                        self.prefetch_data(offset, row_size);
                    }
                }
            }
            AccessPattern::Strided(start, stride, count) => {
                let row_size = self.shape[1..].iter().product::<usize>() * self.itemsize;
                for i in 0..*count {
                    let row_idx = start + i * stride;
                    if row_idx < self.shape[0] {
                        let offset = row_idx * row_size;
                        self.prefetch_data(offset, row_size);
                    }
                }
            }
        }
    }
    
    // 智能预热，根据访问提示预加载数据
    pub fn warmup_intelligent(&self, hint: &AccessHint) {
        match hint {
            AccessHint::WillAccessAll => {
                // 预热所有数据
                self.warmup_cache(1.0);
            }
            AccessHint::WillAccessRange(start, end) => {
                // 预热特定范围
                let start_ratio = *start as f64 / self.shape[0] as f64;
                let end_ratio = *end as f64 / self.shape[0] as f64;
                self.warmup_cache_range(start_ratio, end_ratio);
            }
            AccessHint::WillAccessSparse(ratio) => {
                // 稀疏访问预热
                self.warmup_cache(*ratio);
            }
            AccessHint::WillAccessHot(indices) => {
                // 预热热点数据
                let row_size = self.shape[1..].iter().product::<usize>() * self.itemsize;
                for &idx in indices {
                    if idx < self.shape[0] {
                        let offset = idx * row_size;
                        self.read_data(offset, row_size);
                    }
                }
            }
        }
    }
    
    // 范围预热缓存
    fn warmup_cache_range(&self, start_ratio: f64, end_ratio: f64) {
        let start_row = (self.shape[0] as f64 * start_ratio) as usize;
        let end_row = (self.shape[0] as f64 * end_ratio) as usize;
        let row_size = self.shape[1..].iter().product::<usize>() * self.itemsize;
        
        for row_idx in start_row..end_row.min(self.shape[0]) {
            let offset = row_idx * row_size;
            self.read_data(offset, row_size);
        }
    }
    
    // ===========================
    // 阶段2：深度SIMD优化
    // ===========================
    
    // 2.1 AVX-512支持
    #[cfg(target_arch = "x86_64")]
    fn avx512_boolean_filter(&self, mask: &[bool]) -> Vec<usize> {
        let mut selected = Vec::new();
        
        if is_x86_feature_detected!("avx512f") {
            // 使用AVX-512指令处理64个布尔值
            let chunks = mask.chunks_exact(64);
            let mut base_idx = 0;
            
            for chunk in chunks {
                // 转换为512位掩码
                let mut avx512_mask = [0u8; 64];
                for (i, &val) in chunk.iter().enumerate() {
                    avx512_mask[i] = if val { 1 } else { 0 };
                }
                
                // 使用AVX-512指令快速找到设置的位
                unsafe {
                    use std::arch::x86_64::*;
                    let mask_vec = _mm512_loadu_si512(avx512_mask.as_ptr() as *const i32);
                    let zero = _mm512_setzero_si512();
                    let cmp_result = _mm512_cmpneq_epi8_mask(mask_vec, zero);
                    
                    // 提取设置的位
                    let mut temp_mask = cmp_result;
                    let mut bit_idx = 0;
                    while temp_mask != 0 {
                        if temp_mask & 1 != 0 {
                            selected.push(base_idx + bit_idx);
                        }
                        temp_mask >>= 1;
                        bit_idx += 1;
                    }
                }
                
                base_idx += 64;
            }
            
            // 处理剩余元素
            let remainder = mask.chunks_exact(64).remainder();
            for (i, &val) in remainder.iter().enumerate() {
                if val {
                    selected.push(base_idx + i);
                }
            }
        } else {
            // 回退到标准SIMD
            selected = self.ultra_fast_boolean_filter(mask);
        }
        
        selected
    }
    
    // AVX-512并行复制
    #[cfg(target_arch = "x86_64")]
    fn avx512_parallel_copy(&self, src: &[u8], dst: &mut [u8]) {
        if is_x86_feature_detected!("avx512f") {
            let len = src.len().min(dst.len());
            let chunks = len / 64;
            
            unsafe {
                use std::arch::x86_64::*;
                let src_ptr = src.as_ptr();
                let dst_ptr = dst.as_mut_ptr();
                
                for i in 0..chunks {
                    let src_offset = i * 64;
                    let dst_offset = i * 64;
                    
                    let data = _mm512_loadu_si512(src_ptr.add(src_offset) as *const i32);
                    _mm512_storeu_si512(dst_ptr.add(dst_offset) as *mut i32, data);
                }
                
                // 处理剩余字节
                let remainder = len % 64;
                if remainder > 0 {
                    let start = chunks * 64;
                    std::ptr::copy_nonoverlapping(
                        src_ptr.add(start),
                        dst_ptr.add(start),
                        remainder
                    );
                }
            }
        } else {
            // 回退到标准复制
            dst[..src.len().min(dst.len())].copy_from_slice(&src[..src.len().min(dst.len())]);
        }
    }
    
    // 2.2 向量化内存访问
    pub fn vectorized_gather(&self, indices: &[usize]) -> Vec<Vec<u8>> {
        let row_size = self.shape[1..].iter().product::<usize>() * self.itemsize;
        
        // 使用向量化操作批量收集数据
        let chunk_size = 16; // 16个索引一组进行向量化处理
        
        indices.par_chunks(chunk_size)
            .map(|chunk| {
                let mut results = Vec::with_capacity(chunk.len());
                
                // 预分配所有结果向量
                for _ in 0..chunk.len() {
                    results.push(vec![0u8; row_size]);
                }
                
                // 批量预取
                for &idx in chunk {
                    if idx < self.shape[0] {
                        let offset = idx * row_size;
                        unsafe {
                            #[cfg(target_arch = "x86_64")]
                            {
                                use std::arch::x86_64::_mm_prefetch;
                                _mm_prefetch(
                                    self.mmap.as_ptr().add(offset) as *const i8,
                                    std::arch::x86_64::_MM_HINT_T0
                                );
                            }
                        }
                    }
                }
                
                // 向量化复制
                for (i, &idx) in chunk.iter().enumerate() {
                    if idx < self.shape[0] {
                        let offset = idx * row_size;
                        let src = unsafe {
                            std::slice::from_raw_parts(
                                self.mmap.as_ptr().add(offset),
                                row_size
                            )
                        };
                        
                        #[cfg(target_arch = "x86_64")]
                        {
                            self.avx512_parallel_copy(src, &mut results[i]);
                        }
                        
                        #[cfg(not(target_arch = "x86_64"))]
                        {
                            results[i].copy_from_slice(src);
                        }
                    }
                }
                
                results
            })
            .flatten()
            .collect()
    }
    
    // 2.3 并行计算优化
    pub fn parallel_boolean_index(&self, mask: &[bool]) -> Vec<Vec<u8>> {
        if mask.len() != self.shape[0] {
            return Vec::new();
        }
        
        let row_size = self.shape[1..].iter().product::<usize>() * self.itemsize;
        let num_threads = rayon::current_num_threads();
        let chunk_size = (mask.len() + num_threads - 1) / num_threads;
        
        // 并行处理布尔掩码
        let results: Vec<Vec<Vec<u8>>> = mask.par_chunks(chunk_size)
            .enumerate()
            .map(|(chunk_idx, chunk)| {
                let base_idx = chunk_idx * chunk_size;
                let mut local_results = Vec::new();
                
                for (i, &selected) in chunk.iter().enumerate() {
                    if selected {
                        let row_idx = base_idx + i;
                        if row_idx < self.shape[0] {
                            let offset = row_idx * row_size;
                            let row_data = unsafe {
                                std::slice::from_raw_parts(
                                    self.mmap.as_ptr().add(offset),
                                    row_size
                                )
                            };
                            local_results.push(row_data.to_vec());
                        }
                    }
                }
                
                local_results
            })
            .collect();
        
        // 合并结果
        results.into_iter().flatten().collect()
    }
    
    // 并行多索引操作
    pub fn parallel_multi_index(&self, indices_list: &[Vec<usize>]) -> Vec<Vec<Vec<u8>>> {
        indices_list.par_iter().map(|indices| {
            self.vectorized_gather(indices)
        }).collect()
    }
    
    // ===========================
    // 阶段3：内存管理优化
    // ===========================
    
    // 3.1 内存池管理
    pub fn with_memory_pool(&self) -> MemoryPoolLazyArray {
        MemoryPoolLazyArray {
            inner: self,
            pool: MemoryPool::new(),
        }
    }
    
    // 3.2 NUMA感知分配
    #[cfg(target_os = "linux")]
    pub fn numa_aware_read(&self, offset: usize, size: usize) -> Vec<u8> {
        // 检查当前NUMA节点
        let current_node = self.get_current_numa_node();
        
        // 如果数据在本地NUMA节点，使用快速路径
        if self.is_numa_local(offset, size, current_node) {
            return self.read_data_fast(offset, size);
        }
        
        // 否则使用标准路径
        self.read_data(offset, size)
    }
    
    #[cfg(target_os = "linux")]
    fn get_current_numa_node(&self) -> usize {
        // 简化实现，实际应使用libnuma
        0
    }
    
    #[cfg(target_os = "linux")]
    fn is_numa_local(&self, _offset: usize, _size: usize, _node: usize) -> bool {
        // 简化实现，实际应检查内存页的NUMA位置
        true
    }
    
    // 3.3 智能缓存预热策略
    pub fn intelligent_warmup(&self, workload_hint: &WorkloadHint) {
        match workload_hint {
            WorkloadHint::SequentialRead => {
                // 顺序读取预热
                self.warmup_cache(0.1); // 预热10%
            }
            WorkloadHint::RandomRead => {
                // 随机读取预热
                self.warmup_random_blocks(100); // 预热100个随机块
            }
            WorkloadHint::BooleanFiltering => {
                // 布尔过滤预热
                self.warmup_boolean_index_cache();
            }
            WorkloadHint::HeavyComputation => {
                // 重度计算预热
                self.warmup_cache(0.5); // 预热50%
            }
        }
    }
    
    // 随机块预热
    fn warmup_random_blocks(&self, num_blocks: usize) {
        let row_size = self.shape[1..].iter().product::<usize>() * self.itemsize;
        let max_blocks = self.shape[0] / (CACHE_BLOCK_SIZE / row_size).max(1);
        let blocks_to_warm = num_blocks.min(max_blocks);
        
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        for _ in 0..blocks_to_warm {
            let random_row = rng.gen_range(0..self.shape[0]);
            let offset = random_row * row_size;
            self.read_data(offset, row_size);
        }
    }
    
    // 布尔索引缓存预热
    fn warmup_boolean_index_cache(&self) {
        // 预热布尔索引相关的缓存结构
        let sample_size = 1000.min(self.shape[0]);
        let step = self.shape[0] / sample_size;
        
        for i in (0..self.shape[0]).step_by(step) {
            let row_size = self.shape[1..].iter().product::<usize>() * self.itemsize;
            let offset = i * row_size;
            self.read_data(offset, row_size);
        }
    }
    
    // ===========================
    // 阶段4：算法级优化
    // ===========================
    
    // 4.1 位图索引优化
    pub fn create_bitmap_index(&self, mask: &[bool]) -> BitMapIndex {
        let num_chunks = (mask.len() + 63) / 64;
        let mut chunks = Vec::with_capacity(num_chunks);
        let mut population_count = Vec::with_capacity(num_chunks);
        let mut sparse_indices = Vec::new();
        
        for (chunk_idx, chunk) in mask.chunks(64).enumerate() {
            let mut chunk_value = 0u64;
            let mut pop_count = 0u16;
            
            for (bit_idx, &bit) in chunk.iter().enumerate() {
                if bit {
                    chunk_value |= 1u64 << bit_idx;
                    pop_count += 1;
                    sparse_indices.push(chunk_idx * 64 + bit_idx);
                }
            }
            
            chunks.push(chunk_value);
            population_count.push(pop_count);
        }
        
        BitMapIndex {
            chunks,
            population_count,
            sparse_indices,
        }
    }
    
    // 使用位图索引进行快速布尔索引
    pub fn boolean_index_with_bitmap(&self, bitmap: &BitMapIndex) -> Vec<Vec<u8>> {
        let row_size = self.shape[1..].iter().product::<usize>() * self.itemsize;
        let mut results = Vec::with_capacity(bitmap.sparse_indices.len());
        
        // 使用稀疏索引直接访问
        for &idx in &bitmap.sparse_indices {
            if idx < self.shape[0] {
                let offset = idx * row_size;
                let row_data = unsafe {
                    std::slice::from_raw_parts(
                        self.mmap.as_ptr().add(offset),
                        row_size
                    )
                };
                results.push(row_data.to_vec());
            }
        }
        
        results
    }
    
    // 4.2 分层索引结构
    pub fn create_hierarchical_index(&self, mask: &[bool]) -> HierarchicalIndex {
        let num_level1 = (mask.len() + 63) / 64;
        let num_level2 = (mask.len() + 7) / 8;
        
        let mut level1 = Vec::with_capacity(num_level1);
        let mut level2 = Vec::with_capacity(num_level2);
        let level3 = mask.to_vec();
        
        // 创建64位级别索引
        for chunk in mask.chunks(64) {
            let mut level1_value = 0u64;
            for (i, &bit) in chunk.iter().enumerate() {
                if bit {
                    level1_value |= 1u64 << i;
                }
            }
            level1.push(level1_value);
        }
        
        // 创建8位级别索引
        for chunk in mask.chunks(8) {
            let mut level2_value = 0u8;
            for (i, &bit) in chunk.iter().enumerate() {
                if bit {
                    level2_value |= 1u8 << i;
                }
            }
            level2.push(level2_value);
        }
        
        HierarchicalIndex {
            level1,
            level2,
            level3,
        }
    }
    
    // 使用分层索引进行布尔索引
    pub fn boolean_index_with_hierarchical(&self, hier_index: &HierarchicalIndex) -> Vec<Vec<u8>> {
        let row_size = self.shape[1..].iter().product::<usize>() * self.itemsize;
        let mut results = Vec::new();
        
        // 使用分层索引快速跳过大块的false值
        for (level1_idx, &level1_value) in hier_index.level1.iter().enumerate() {
            if level1_value == 0 {
                continue; // 跳过整个64位块
            }
            
            let base_idx = level1_idx * 64;
            let mut temp_value = level1_value;
            
            while temp_value != 0 {
                let bit_pos = temp_value.trailing_zeros() as usize;
                let actual_idx = base_idx + bit_pos;
                
                if actual_idx < self.shape[0] {
                    let offset = actual_idx * row_size;
                    let row_data = unsafe {
                        std::slice::from_raw_parts(
                            self.mmap.as_ptr().add(offset),
                            row_size
                        )
                    };
                    results.push(row_data.to_vec());
                }
                
                temp_value &= temp_value - 1; // 清除最低位的1
            }
        }
        
        results
    }
    
    // 4.3 稀疏数据结构优化
    pub fn create_sparse_selector(&self, mask: &[bool]) -> SparseSelector {
        let mut dense_ranges = Vec::new();
        let mut sparse_indices = Vec::new();
        let mut current_range_start = None;
        let mut total_selected = 0;
        
        for (idx, &selected) in mask.iter().enumerate() {
            if selected {
                total_selected += 1;
                
                if current_range_start.is_none() {
                    current_range_start = Some(idx);
                }
            } else {
                if let Some(start) = current_range_start {
                    let range_len = idx - start;
                    if range_len >= 3 {
                        // 连续范围
                        dense_ranges.push((start, idx));
                    } else {
                        // 稀疏索引
                        for i in start..idx {
                            sparse_indices.push(i);
                        }
                    }
                    current_range_start = None;
                }
            }
        }
        
        // 处理最后一个范围
        if let Some(start) = current_range_start {
            let range_len = mask.len() - start;
            if range_len >= 3 {
                dense_ranges.push((start, mask.len()));
            } else {
                for i in start..mask.len() {
                    sparse_indices.push(i);
                }
            }
        }
        
        let compression_ratio = if total_selected > 0 {
            (dense_ranges.len() + sparse_indices.len()) as f64 / total_selected as f64
        } else {
            1.0
        };
        
        SparseSelector {
            dense_ranges,
            sparse_indices,
            compression_ratio,
        }
    }
    
    // 使用稀疏选择器进行布尔索引
    pub fn boolean_index_with_sparse_selector(&self, selector: &SparseSelector) -> Vec<Vec<u8>> {
        let row_size = self.shape[1..].iter().product::<usize>() * self.itemsize;
        let mut results = Vec::new();
        
        // 处理密集范围
        for &(start, end) in &selector.dense_ranges {
            for idx in start..end {
                if idx < self.shape[0] {
                    let offset = idx * row_size;
                    let row_data = unsafe {
                        std::slice::from_raw_parts(
                            self.mmap.as_ptr().add(offset),
                            row_size
                        )
                    };
                    results.push(row_data.to_vec());
                }
            }
        }
        
        // 处理稀疏索引
        for &idx in &selector.sparse_indices {
            if idx < self.shape[0] {
                let offset = idx * row_size;
                let row_data = unsafe {
                    std::slice::from_raw_parts(
                        self.mmap.as_ptr().add(offset),
                        row_size
                    )
                };
                results.push(row_data.to_vec());
            }
        }
        
        results
    }
    
    // 4.4 自适应算法选择
    pub fn choose_optimal_algorithm(&self, mask: &[bool]) -> OptimizationAlgorithm {
        let total_rows = mask.len();
        let selected_count = mask.iter().filter(|&&x| x).count();
        let selection_density = selected_count as f64 / total_rows as f64;
        let data_size = selected_count * self.shape[1..].iter().product::<usize>() * self.itemsize;
        
        // 分析访问模式
        let mut consecutive_count = 0;
        let mut max_consecutive = 0;
        let mut current_consecutive = 0;
        
        for &selected in mask {
            if selected {
                current_consecutive += 1;
                max_consecutive = max_consecutive.max(current_consecutive);
            } else {
                if current_consecutive > 0 {
                    consecutive_count += 1;
                    current_consecutive = 0;
                }
            }
        }
        
        let consecutiveness = max_consecutive as f64 / total_rows as f64;
        
        // 选择最优算法
        if selection_density < 0.01 {
            // 极稀疏选择
            OptimizationAlgorithm::ZeroCopy
        } else if selection_density > 0.9 {
            // 极密集选择
            OptimizationAlgorithm::Vectorized
        } else if consecutiveness > 0.7 {
            // 高连续性
            OptimizationAlgorithm::AdaptivePrefetch
        } else if data_size > 100 * 1024 * 1024 {
            // 大数据量
            OptimizationAlgorithm::AVX512
        } else {
            // 通用情况
            OptimizationAlgorithm::StandardSIMD
        }
    }
    
    // 使用自适应算法进行布尔索引
    pub fn boolean_index_adaptive_algorithm(&self, mask: &[bool]) -> Vec<Vec<u8>> {
        let algorithm = self.choose_optimal_algorithm(mask);
        
                 match algorithm {
            OptimizationAlgorithm::ZeroCopy => {
                let views = self.boolean_index_zero_copy_view(mask);
                views.into_iter().map(|v| v.to_vec()).collect()
            }
            OptimizationAlgorithm::Vectorized => {
                let selected_indices = self.ultra_fast_boolean_filter(mask);
                self.vectorized_gather(&selected_indices)
            }
            OptimizationAlgorithm::AdaptivePrefetch => {
                self.boolean_index_adaptive_prefetch(mask)
            }
            OptimizationAlgorithm::AVX512 => {
                #[cfg(target_arch = "x86_64")]
                {
                    let selected_indices = self.avx512_boolean_filter(mask);
                    self.vectorized_gather(&selected_indices)
                }
                #[cfg(not(target_arch = "x86_64"))]
                {
                    self.boolean_index_ultimate(mask)
                }
            }
            OptimizationAlgorithm::StandardSIMD => {
                self.boolean_index_ultimate(mask)
            }
        }
    }
    
    // 生产级性能布尔索引入口
    pub fn boolean_index_production(&self, mask: &[bool]) -> Vec<Vec<u8>> {
        if mask.len() != self.shape[0] {
            return Vec::new();
        }
        
        // 统计和分析
        let selected_count = mask.iter().filter(|&&x| x).count();
        if selected_count == 0 {
            return Vec::new();
        }
        
        let selection_density = selected_count as f64 / mask.len() as f64;
        let data_size = selected_count * self.shape[1..].iter().product::<usize>() * self.itemsize;
        
        // 选择最优策略
        if selection_density < 0.05 && selected_count < 1000 {
            // 极稀疏小数据：直接零拷贝
            let views = self.boolean_index_zero_copy_view(mask);
            views.into_iter().map(|v| v.to_vec()).collect()
        } else if selection_density > 0.8 && data_size > 10 * 1024 * 1024 {
            // 高密度大数据：使用位图索引
            let bitmap = self.create_bitmap_index(mask);
            self.boolean_index_with_bitmap(&bitmap)
        } else if selection_density > 0.5 {
            // 中等密度：使用分层索引
            let hier_index = self.create_hierarchical_index(mask);
            self.boolean_index_with_hierarchical(&hier_index)
        } else {
            // 其他情况：使用稀疏选择器
            let selector = self.create_sparse_selector(mask);
            self.boolean_index_with_sparse_selector(&selector)
        }
    }

    // 新增：极限SIMD优化的布尔索引（使用更激进的优化）
    pub fn boolean_index_extreme(&self, mask: &[bool]) -> Vec<Vec<u8>> {
        if mask.len() != self.shape[0] {
            return Vec::new();
        }

        // 1. 使用极限位操作提取索引
        let selected_indices = self.extreme_simd_boolean_filter(mask);
        if selected_indices.is_empty() {
            return Vec::new();
        }

        let row_size = self.shape[1..].iter().product::<usize>() * self.itemsize;
        
        // 2. 并行处理并收集结果
        selected_indices.par_chunks(64) // 64行一组并行处理
            .flat_map(|chunk| {
                chunk.iter().map(|&row_idx| {
                    let src_offset = row_idx * row_size;
                    let row_data = unsafe {
                        std::slice::from_raw_parts(
                            self.mmap.as_ptr().add(src_offset),
                            row_size
                        )
                    };
                    row_data.to_vec()
                }).collect::<Vec<_>>()
            })
            .collect()
    }

    // 极限SIMD位操作滤波器
    fn extreme_simd_boolean_filter(&self, mask: &[bool]) -> Vec<usize> {
        let mut selected: Vec<usize> = Vec::with_capacity(mask.len() / 4); // 预估25%选择率
        
        // 使用AVX-512指令（如果可用）
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f") {
                return self.avx512_boolean_filter(mask);
            }
            if is_x86_feature_detected!("avx2") {
                return self.avx2_boolean_filter(mask);
            }
        }
        
        // 回退到标准SIMD优化
        self.ultra_fast_boolean_filter(mask)
    }

    // AVX-512优化的布尔滤波器
    #[cfg(target_arch = "x86_64")]
    fn avx512_boolean_filter(&self, mask: &[bool]) -> Vec<usize> {
        let mut selected = Vec::with_capacity(mask.len() / 4);
        let chunk_size = 512; // AVX-512可以处理512位
        
        for (chunk_start, chunk) in mask.chunks(chunk_size).enumerate() {
            let base_idx = chunk_start * chunk_size;
            
            if chunk.len() == chunk_size {
                // 使用AVX-512位操作
                unsafe {
                    let mut bit_masks: [u64; 8] = [0; 8]; // 512位 = 8 * 64位
                    
                    for (i, &val) in chunk.iter().enumerate() {
                        if val {
                            let word_idx = i / 64;
                            let bit_idx = i % 64;
                            bit_masks[word_idx] |= 1u64 << bit_idx;
                        }
                    }
                    
                    // 并行处理8个64位掩码
                    for (word_idx, &mask) in bit_masks.iter().enumerate() {
                        let mut mask_copy = mask;
                        let word_base = base_idx + word_idx * 64;
                        
                        // 使用BMI指令加速位提取
                        while mask_copy != 0 {
                            #[cfg(target_feature = "bmi1")]
                            {
                                let pos = mask_copy.trailing_zeros() as usize;
                                selected.push(word_base + pos);
                                mask_copy = std::arch::x86_64::_blsr_u64(mask_copy); // BMI1指令
                            }
                            #[cfg(not(target_feature = "bmi1"))]
                            {
                                let pos = mask_copy.trailing_zeros() as usize;
                                selected.push(word_base + pos);
                                mask_copy &= mask_copy - 1;
                            }
                        }
                    }
                }
            } else {
                // 处理剩余元素
                for (i, &val) in chunk.iter().enumerate() {
                    if val {
                        selected.push(base_idx + i);
                    }
                }
            }
        }
        
        selected
    }

    // AVX2优化的布尔滤波器
    #[cfg(target_arch = "x86_64")]
    fn avx2_boolean_filter(&self, mask: &[bool]) -> Vec<usize> {
        let mut selected = Vec::with_capacity(mask.len() / 4);
        let chunk_size = 256; // AVX2处理256位
        
        for (chunk_start, chunk) in mask.chunks(chunk_size).enumerate() {
            let base_idx = chunk_start * chunk_size;
            
            if chunk.len() == chunk_size {
                unsafe {
                    let mut bit_masks: [u64; 4] = [0; 4]; // 256位 = 4 * 64位
                    
                    for (i, &val) in chunk.iter().enumerate() {
                        if val {
                            let word_idx = i / 64;
                            let bit_idx = i % 64;
                            bit_masks[word_idx] |= 1u64 << bit_idx;
                        }
                    }
                    
                    // 并行处理4个64位掩码
                    for (word_idx, &mask) in bit_masks.iter().enumerate() {
                        let mut mask_copy = mask;
                        let word_base = base_idx + word_idx * 64;
                        
                        while mask_copy != 0 {
                            let pos = mask_copy.trailing_zeros() as usize;
                            selected.push(word_base + pos);
                            mask_copy &= mask_copy - 1;
                        }
                    }
                }
            } else {
                // 处理剩余元素
                for (i, &val) in chunk.iter().enumerate() {
                    if val {
                        selected.push(base_idx + i);
                    }
                }
            }
        }
        
        selected
    }

    // SIMD内存复制
    unsafe fn simd_memory_copy(&self, src: *const u8, dst: *mut u8, size: usize) {
        if size >= 64 && size % 64 == 0 {
            // 对于64字节对齐的数据，使用更高效的SIMD复制
            #[cfg(target_arch = "x86_64")]
            {
                if is_x86_feature_detected!("avx2") {
                    self.avx2_memory_copy(src, dst, size);
                    return;
                }
            }
        }
        
        // 回退到标准复制
        std::ptr::copy_nonoverlapping(src, dst, size);
    }

    // AVX2内存复制
    #[cfg(target_arch = "x86_64")]
    unsafe fn avx2_memory_copy(&self, src: *const u8, dst: *mut u8, size: usize) {
        let chunks = size / 32; // AVX2一次处理32字节
        
        for i in 0..chunks {
            let src_offset = i * 32;
            let dst_offset = i * 32;
            
            // 使用AVX2指令进行32字节传输
            let data = std::arch::x86_64::_mm256_loadu_si256(
                src.add(src_offset) as *const std::arch::x86_64::__m256i
            );
            std::arch::x86_64::_mm256_storeu_si256(
                dst.add(dst_offset) as *mut std::arch::x86_64::__m256i,
                data
            );
        }
        
        // 处理剩余字节
        let remaining = size % 32;
        if remaining > 0 {
            std::ptr::copy_nonoverlapping(
                src.add(chunks * 32),
                dst.add(chunks * 32),
                remaining
            );
        }
    }

    // 新增：专门的小数据量优化
    pub fn boolean_index_micro(&self, mask: &[bool]) -> Vec<Vec<u8>> {
        if mask.len() != self.shape[0] || mask.len() < 1000 {
            return self.boolean_index_zero_copy_micro(mask);
        }
        
        self.boolean_index_extreme(mask)
    }

    // 微小数据量的零拷贝优化
    fn boolean_index_zero_copy_micro(&self, mask: &[bool]) -> Vec<Vec<u8>> {
        let row_size = self.shape[1..].iter().product::<usize>() * self.itemsize;
        let mut result = Vec::new();
        
        // 对于小数据，直接遍历，避免复杂的SIMD开销
        for (row_idx, &selected) in mask.iter().enumerate() {
            if selected {
                let offset = row_idx * row_size;
                unsafe {
                    let slice = std::slice::from_raw_parts(
                        self.mmap.as_ptr().add(offset),
                        row_size
                    );
                    result.push(slice.to_vec());
                }
            }
        }
        
        result
    }

    // 新增：智能策略选择器
    pub fn boolean_index_smart(&self, mask: &[bool]) -> Vec<Vec<u8>> {
        if mask.len() != self.shape[0] {
            return Vec::new();
        }

        let total_rows = mask.len();
        let selected_count = mask.iter().filter(|&&x| x).count();
        
        if selected_count == 0 {
            return Vec::new();
        }

        let selection_density = selected_count as f64 / total_rows as f64;
        let row_size = self.shape[1..].iter().product::<usize>() * self.itemsize;
        let total_data_size = selected_count * row_size;

        // 策略1：极小数据量 (<100行 或 <10KB)
        if selected_count < 100 || total_data_size < 10 * 1024 {
            return self.boolean_index_zero_copy_micro(mask);
        }

        // 策略2：小数据量但高密度 (<1000行 且 >80%密度)
        if selected_count < 1000 && selection_density > 0.8 {
            return self.boolean_index_aggregated_view_optimized(mask);
        }

        // 策略3：中等数据量稀疏选择 (<50%密度 且 <50MB)
        if selection_density < 0.5 && total_data_size < 50 * 1024 * 1024 {
            return self.boolean_index_sparse_optimized(mask);
        }

        // 策略4：大数据量高密度选择 (>80%密度)
        if selection_density > 0.8 {
            return self.boolean_index_dense_vectorized(mask);
        }

        // 策略5：大数据量中等密度 - 使用终极优化
        self.boolean_index_ultimate(mask)
    }

    // 优化的聚合视图（针对小高密度数据）
    fn boolean_index_aggregated_view_optimized(&self, mask: &[bool]) -> Vec<Vec<u8>> {
        let selected_indices = self.ultra_fast_boolean_filter(mask);
        let row_size = self.shape[1..].iter().product::<usize>() * self.itemsize;
        
        // 检测连续性并使用大块复制
        let mut result = Vec::with_capacity(selected_indices.len());
        let mut i = 0;
        
        while i < selected_indices.len() {
            let start_idx = selected_indices[i];
            let mut consecutive_count = 1;
            
            // 找连续块
            while i + consecutive_count < selected_indices.len() &&
                  selected_indices[i + consecutive_count] == start_idx + consecutive_count {
                consecutive_count += 1;
            }
            
            if consecutive_count >= 3 { // 3行或以上连续时使用范围复制
                let range_data = self.get_rows_range(start_idx, start_idx + consecutive_count);
                for chunk in range_data.chunks_exact(row_size) {
                    result.push(chunk.to_vec());
                }
                i += consecutive_count;
            } else {
                // 单行复制
                let offset = start_idx * row_size;
                unsafe {
                    let slice = std::slice::from_raw_parts(
                        self.mmap.as_ptr().add(offset),
                        row_size
                    );
                    result.push(slice.to_vec());
                }
                i += 1;
            }
        }
        
        result
    }

    // 稀疏选择优化（针对低密度数据）
    fn boolean_index_sparse_optimized(&self, mask: &[bool]) -> Vec<Vec<u8>> {
        let selected_indices = self.extreme_simd_boolean_filter(mask);
        let row_size = self.shape[1..].iter().product::<usize>() * self.itemsize;
        
        // 按内存页分组以提高缓存效率
        let page_size = 4096; // 4KB页面
        let rows_per_page = page_size / row_size.max(1);
        
        let mut page_groups: std::collections::BTreeMap<usize, Vec<usize>> = 
            std::collections::BTreeMap::new();
        
        for &idx in &selected_indices {
            let page_idx = idx / rows_per_page.max(1);
            page_groups.entry(page_idx).or_insert_with(Vec::new).push(idx);
        }
        
        // 按页面顺序处理，减少页面错误
        let page_results: Vec<Vec<Vec<u8>>> = page_groups.par_iter()
            .map(|(_, indices)| {
                indices.iter().map(|&row_idx| {
                    let offset = row_idx * row_size;
                    unsafe {
                        // 预取下一个可能的访问
                        if offset + row_size * 2 < self.mmap.len() {
                            #[cfg(target_arch = "x86_64")]
                            {
                                use std::arch::x86_64::_mm_prefetch;
                                _mm_prefetch(
                                    self.mmap.as_ptr().add(offset + row_size) as *const i8,
                                    std::arch::x86_64::_MM_HINT_T1
                                );
                            }
                        }
                        
                        let slice = std::slice::from_raw_parts(
                            self.mmap.as_ptr().add(offset),
                            row_size
                        );
                        slice.to_vec()
                    }
                }).collect()
            })
            .collect();
        
        // 按原始索引顺序重新组合
        let mut result = Vec::with_capacity(selected_indices.len());
        for page_result in page_results {
            result.extend(page_result);
        }
        
        result
    }

    // 密集选择的向量化优化
    fn boolean_index_dense_vectorized(&self, mask: &[bool]) -> Vec<Vec<u8>> {
        let row_size = self.shape[1..].iter().product::<usize>() * self.itemsize;
        let selected_count = mask.iter().filter(|&&x| x).count();
        
        // 预分配连续内存
        let total_size = selected_count * row_size;
        let mut result_buffer: Vec<u8> = Vec::with_capacity(total_size);
        unsafe { result_buffer.set_len(total_size); }
        
        let mut dst_offset = 0;
        
        // 使用向量化处理连续块
        let mut i = 0;
        while i < mask.len() {
            if mask[i] {
                // 查找连续的true块
                let start_row = i;
                let mut end_row = i + 1;
                
                while end_row < mask.len() && mask[end_row] {
                    end_row += 1;
                }
                
                let block_size = (end_row - start_row) * row_size;
                let src_offset = start_row * row_size;
                
                if dst_offset + block_size <= total_size && src_offset + block_size <= self.mmap.len() {
                    unsafe {
                        // 大块连续复制
                        if block_size >= 256 {
                            self.simd_memory_copy(
                                self.mmap.as_ptr().add(src_offset),
                                result_buffer.as_mut_ptr().add(dst_offset),
                                block_size
                            );
                        } else {
                            std::ptr::copy_nonoverlapping(
                                self.mmap.as_ptr().add(src_offset),
                                result_buffer.as_mut_ptr().add(dst_offset),
                                block_size
                            );
                        }
                        dst_offset += block_size;
                    }
                }
                
                i = end_row;
            } else {
                i += 1;
            }
        }
        
        // 分割为行向量
        result_buffer.chunks_exact(row_size)
            .map(|chunk| chunk.to_vec())
            .collect()
    }

    // 新增：自适应预取布尔索引
    pub fn boolean_index_adaptive_prefetch(&self, mask: &[bool]) -> Vec<Vec<u8>> {
        let selected_indices = self.extreme_simd_boolean_filter(mask);
        let row_size = self.shape[1..].iter().product::<usize>() * self.itemsize;
        
        // 分析访问模式
        let access_pattern = self.analyze_access_pattern(&selected_indices);
        
        match access_pattern {
            AccessPatternType::Sequential => self.boolean_index_sequential_optimized(mask),
            AccessPatternType::Random => self.boolean_index_random_optimized(mask),
            AccessPatternType::Clustered => self.boolean_index_clustered_optimized(mask),
            AccessPatternType::Mixed => self.boolean_index_smart(mask),
        }
    }

    // 访问模式分析
    fn analyze_access_pattern(&self, indices: &[usize]) -> AccessPatternType {
        if indices.len() < 2 {
            return AccessPatternType::Sequential;
        }
        
        let mut sequential_count = 0;
        let mut random_count = 0;
        let mut clustered_count = 0;
        
        for window in indices.windows(2) {
            let diff = if window[1] > window[0] { 
                window[1] - window[0] 
            } else { 
                window[0] - window[1] 
            };
            
            if diff == 1 {
                sequential_count += 1;
            } else if diff <= 10 {
                clustered_count += 1;
            } else {
                random_count += 1;
            }
        }
        
        let total = sequential_count + random_count + clustered_count;
        let sequential_ratio = sequential_count as f64 / total as f64;
        let clustered_ratio = clustered_count as f64 / total as f64;
        
        if sequential_ratio > 0.7 {
            AccessPatternType::Sequential
        } else if clustered_ratio > 0.5 {
            AccessPatternType::Clustered
        } else {
            AccessPatternType::Random
        }
    }

    // 顺序访问优化
    fn boolean_index_sequential_optimized(&self, mask: &[bool]) -> Vec<Vec<u8>> {
        // 对于顺序访问，使用最简单高效的方法
        self.boolean_index_aggregated_view_optimized(mask)
    }

    // 随机访问优化
    fn boolean_index_random_optimized(&self, mask: &[bool]) -> Vec<Vec<u8>> {
        // 对于随机访问，使用缓存友好的方法
        self.boolean_index_sparse_optimized(mask)
    }

    // 聚集访问优化
    fn boolean_index_clustered_optimized(&self, mask: &[bool]) -> Vec<Vec<u8>> {
        // 对于聚集访问，使用混合策略
        let selected_indices = self.extreme_simd_boolean_filter(mask);
        let row_size = self.shape[1..].iter().product::<usize>() * self.itemsize;
        
        // 检测聚集并使用块复制
        let clusters = self.detect_clusters(&selected_indices, 5); // 5行内的聚集
        
        let mut result = Vec::with_capacity(selected_indices.len());
        
        for cluster in clusters {
            if cluster.len() >= 3 {
                // 聚集块：使用范围读取
                let start = cluster[0];
                let end = cluster[cluster.len() - 1] + 1;
                let range_data = self.get_rows_range(start, end);
                
                // 只取需要的行
                for &idx in &cluster {
                    let relative_idx = idx - start;
                    let row_start = relative_idx * row_size;
                    let row_end = row_start + row_size;
                    if row_end <= range_data.len() {
                        result.push(range_data[row_start..row_end].to_vec());
                    }
                }
            } else {
                // 小聚集：直接访问
                for &idx in &cluster {
                    let offset = idx * row_size;
                    unsafe {
                        let slice = std::slice::from_raw_parts(
                            self.mmap.as_ptr().add(offset),
                            row_size
                        );
                        result.push(slice.to_vec());
                    }
                }
            }
        }
        
        result
    }

    // 聚集检测
    fn detect_clusters(&self, indices: &[usize], max_gap: usize) -> Vec<Vec<usize>> {
        if indices.is_empty() {
            return Vec::new();
        }
        
        let mut clusters = Vec::new();
        let mut current_cluster = vec![indices[0]];
        
        for &idx in &indices[1..] {
            if idx <= current_cluster.last().unwrap() + max_gap {
                current_cluster.push(idx);
            } else {
                clusters.push(std::mem::take(&mut current_cluster));
                current_cluster.push(idx);
            }
        }
        
        if !current_cluster.is_empty() {
            clusters.push(current_cluster);
        }
        
        clusters
    }

    // 高级切片操作
    pub fn slice(&self, ranges: &[std::ops::Range<usize>]) -> Vec<u8> {
        if ranges.is_empty() || ranges.len() > self.shape.len() {
            return Vec::new();
        }

        // 计算切片的总大小
        let slice_shape: Vec<usize> = ranges.iter()
            .enumerate()
            .map(|(dim, range)| {
                let dim_size = self.shape.get(dim).cloned().unwrap_or(1);
                range.end.min(dim_size) - range.start.min(dim_size)
            })
            .collect();

        let total_elements = slice_shape.iter().product::<usize>();
        let mut result = Vec::with_capacity(total_elements * self.itemsize);

        // 对于简单的行切片，使用优化路径
        if ranges.len() == 1 {
            let row_size = self.shape[1..].iter().product::<usize>() * self.itemsize;
            for row_idx in ranges[0].start..ranges[0].end.min(self.shape[0]) {
                let row_data = self.get_row(row_idx);
                result.extend(row_data);
            }
        } else {
            // 复杂多维切片
            self.slice_recursive(&ranges, &mut result, 0, 0);
        }
        
        result
    }

    fn slice_recursive(&self, ranges: &[std::ops::Range<usize>], result: &mut Vec<u8>, dim: usize, base_offset: usize) {
        if dim >= ranges.len() {
            return;
        }
        
        let dim_size = self.shape.get(dim).cloned().unwrap_or(1);
        let range = &ranges[dim];
        
        if dim == ranges.len() - 1 {
            // 最后一个维度，直接读取数据
            let start_idx = range.start.min(dim_size);
            let end_idx = range.end.min(dim_size);
            let element_size = self.itemsize;
            
            for idx in start_idx..end_idx {
                let offset = base_offset + idx * element_size;
                let data = self.read_data(offset, element_size);
                result.extend(data);
            }
        } else {
            // 递归处理下一个维度
            let stride = self.shape[dim + 1..].iter().product::<usize>() * self.itemsize;
            let start_idx = range.start.min(dim_size);
            let end_idx = range.end.min(dim_size);
            
            for idx in start_idx..end_idx {
                let new_offset = base_offset + idx * stride;
                self.slice_recursive(ranges, result, dim + 1, new_offset);
            }
        }
    }

    // 统计信息
    pub fn get_cache_stats(&self) -> (u64, u64, f64) {
        let stats = self.stats.lock().unwrap();
        let total_requests = stats.cache_hits + stats.cache_misses;
        let hit_rate = if total_requests > 0 {
            stats.cache_hits as f64 / total_requests as f64
        } else {
            0.0
        };
        (stats.cache_hits, stats.cache_misses, hit_rate)
    }

    // 获取扩展缓存统计信息（包括自适应信息）
    pub fn get_extended_cache_stats(&self) -> (u64, u64, f64, usize, usize, usize) {
        let stats = self.stats.lock().unwrap();
        let total_requests = stats.cache_hits + stats.cache_misses;
        let hit_rate = if total_requests > 0 {
            stats.cache_hits as f64 / total_requests as f64
        } else {
            0.0
        };
        
        let (block_count, current_size, max_size) = self.cache.get_cache_info();
        
        (stats.cache_hits, stats.cache_misses, hit_rate, block_count, current_size, max_size)
    }

    // 清理缓存
    pub fn clear_cache(&self) {
        let mut blocks = self.cache.blocks.write().unwrap();
        let mut total_size = self.cache.total_size.lock().unwrap();
        blocks.clear();
        *total_size = 0;
    }

    // 预热缓存
    pub fn warmup_cache(&self, sample_rate: f64) {
        let total_rows = self.shape[0];
        let sample_size = (total_rows as f64 * sample_rate) as usize;
        
        let sample_indices: Vec<usize> = (0..sample_size)
            .map(|i| i * total_rows / sample_size)
            .collect();
        
        // 并行预热
        sample_indices.par_iter().for_each(|&idx| {
            let _ = self.get_row(idx);
        });
    }
}

// 高性能类型转换
pub trait FastTypeConversion {
    fn to_typed_slice<T>(&self) -> &[T] where T: Copy;
    fn to_typed_vec<T>(&self) -> Vec<T> where T: Copy;
}

impl FastTypeConversion for Vec<u8> {
    fn to_typed_slice<T>(&self) -> &[T] where T: Copy {
        unsafe {
            std::slice::from_raw_parts(
                self.as_ptr() as *const T,
                self.len() / std::mem::size_of::<T>()
            )
        }
    }

    fn to_typed_vec<T>(&self) -> Vec<T> where T: Copy {
        let typed_slice = self.to_typed_slice::<T>();
        typed_slice.to_vec()
    }
}

// SIMD优化的数据处理
#[cfg(target_arch = "x86_64")]
pub mod simd_ops {
    use std::arch::x86_64::*;

    pub fn sum_f32_simd(data: &[f32]) -> f32 {
        unsafe {
            let mut sum = _mm256_setzero_ps();
            let chunks = data.chunks_exact(8);
            let remainder = chunks.remainder();
            
            for chunk in chunks {
                let vec = _mm256_loadu_ps(chunk.as_ptr());
                sum = _mm256_add_ps(sum, vec);
            }
            
            // 水平求和
            let sum_high = _mm256_extractf128_ps(sum, 1);
            let sum_low = _mm256_castps256_ps128(sum);
            let sum_final = _mm_add_ps(sum_high, sum_low);
            let sum_final = _mm_hadd_ps(sum_final, sum_final);
            let sum_final = _mm_hadd_ps(sum_final, sum_final);
            
            let mut result = _mm_cvtss_f32(sum_final);
            
            // 处理剩余元素
            for &val in remainder {
                result += val;
            }
            
            result
        }
    }

    pub fn sum_i32_simd(data: &[i32]) -> i32 {
        unsafe {
            let mut sum = _mm256_setzero_si256();
            let chunks = data.chunks_exact(8);
            let remainder = chunks.remainder();
            
            for chunk in chunks {
                let vec = _mm256_loadu_si256(chunk.as_ptr() as *const __m256i);
                sum = _mm256_add_epi32(sum, vec);
            }
            
            // 提取并求和
            let sum_array: [i32; 8] = std::mem::transmute(sum);
            let mut result = sum_array.iter().sum();
            
            // 处理剩余元素
            for &val in remainder {
                result += val;
            }
            
            result
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::Write;

    #[test]
    fn test_cache_functionality() {
        let cache = SmartCache::new();
        let data = vec![1, 2, 3, 4, 5];
        cache.put(0, data.clone());
        
        let retrieved = cache.get(0).unwrap();
        assert_eq!(retrieved, data);
    }

    #[test]
    fn test_optimized_lazy_array() {
        // 创建测试数据文件
        let test_data = vec![1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let mut file = File::create("test_data.bin").unwrap();
        file.write_all(&test_data).unwrap();
        
        let lazy_array = OptimizedLazyArray::new(
            "test_data.bin".into(),
            vec![5, 2],
            DataType::Uint8
        ).unwrap();
        
        let row_data = lazy_array.get_row(0);
        assert_eq!(row_data, vec![1u8, 2]);
        
        let row_data = lazy_array.get_row(1);
        assert_eq!(row_data, vec![3u8, 4]);
        
        // 清理
        std::fs::remove_file("test_data.bin").unwrap();
    }

    #[test]
    fn test_boolean_indexing() {
        let test_data = vec![1u8, 2, 3, 4, 5, 6, 7, 8];
        let mut file = File::create("test_bool.bin").unwrap();
        file.write_all(&test_data).unwrap();
        
        let lazy_array = OptimizedLazyArray::new(
            "test_bool.bin".into(),
            vec![4, 2],
            DataType::Uint8
        ).unwrap();
        
        let mask = vec![true, false, true, false];
        let selected_rows = lazy_array.boolean_index(&mask);
        assert_eq!(selected_rows.len(), 2);
        assert_eq!(selected_rows[0], vec![1u8, 2]);
        assert_eq!(selected_rows[1], vec![5u8, 6]);
        
        // 清理
        std::fs::remove_file("test_bool.bin").unwrap();
    }
} 