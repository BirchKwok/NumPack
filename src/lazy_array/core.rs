//! 核心OptimizedLazyArray实现
//!
//! 这个文件将在Task 7中从lazy_array_original.rs中提取核心实现

use crate::access_pattern::AccessHint;
use crate::cache::smart_cache::SmartCache;
use crate::core::metadata::DataType;
use crate::io::batch_access_engine::BatchAccessEngine;
use crate::memory::simd_processor::SIMDProcessor;
use memmap2::Mmap;
use std::fs::File;
use std::path::PathBuf;
use std::sync::Arc;

// 访问模式分析相关类型
#[derive(Debug, Clone)]
enum AccessPatternType {
    Sequential,
    Strided,
    Random,
    Clustered,
}

pub struct OptimizedLazyArray {
    mmap: Arc<Mmap>,
    pub shape: Vec<usize>,
    dtype: DataType,
    pub itemsize: usize,
    file_path: PathBuf,
    cache: Arc<SmartCache>,
    batch_engine: Arc<BatchAccessEngine>,
    simd_processor: SIMDProcessor,
}

impl OptimizedLazyArray {
    pub fn new(file_path: PathBuf, shape: Vec<usize>, dtype: DataType) -> std::io::Result<Self> {
        let file = File::open(&file_path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        let itemsize = dtype.size_bytes() as usize;
        let cache = Arc::new(SmartCache::new());
        let simd_processor = SIMDProcessor::new();

        Ok(Self {
            mmap: Arc::new(mmap),
            shape,
            dtype,
            itemsize,
            file_path,
            cache,
            batch_engine: Arc::new(BatchAccessEngine::new()),
            simd_processor,
        })
    }

    pub fn from_file(file_path: &str, shape: Vec<usize>, itemsize: usize) -> std::io::Result<Self> {
        let path = PathBuf::from(file_path);
        let dtype = match itemsize {
            1 => DataType::Uint8,
            2 => DataType::Uint16,
            4 => DataType::Uint32,
            8 => DataType::Uint64,
            _ => DataType::Uint8,
        };
        Self::new(path, shape, dtype)
    }

    pub fn read_data(&self, offset: usize, size: usize) -> Vec<u8> {
        if offset + size <= self.mmap.len() {
            self.mmap[offset..offset + size].to_vec()
        } else {
            vec![]
        }
    }

    // 核心数据访问方法的实际实现
    pub fn get_row(&self, row_idx: usize) -> Vec<u8> {
        if row_idx >= self.shape[0] {
            return vec![];
        }

        let row_size = self.shape[1..].iter().product::<usize>() * self.itemsize;
        let offset = row_idx * row_size;

        if offset + row_size <= self.mmap.len() {
            self.mmap[offset..offset + row_size].to_vec()
        } else {
            vec![]
        }
    }

    pub fn get_row_fast(&self, row_idx: usize) -> Vec<u8> {
        // 快速版本跳过一些边界检查
        if row_idx >= self.shape[0] {
            return vec![];
        }

        let row_size = self.shape[1..].iter().product::<usize>() * self.itemsize;
        let offset = row_idx * row_size;

        if offset + row_size <= self.mmap.len() {
            // 使用unsafe提升性能
            unsafe {
                let ptr = self.mmap.as_ptr().add(offset);
                std::slice::from_raw_parts(ptr, row_size).to_vec()
            }
        } else {
            vec![]
        }
    }

    pub fn get_rows(&self, row_indices: &[usize]) -> Vec<Vec<u8>> {
        let mut results = Vec::with_capacity(row_indices.len());

        for &row_idx in row_indices {
            results.push(self.get_row(row_idx));
        }

        results
    }

    pub fn get_rows_range(&self, start_row: usize, end_row: usize) -> Vec<u8> {
        if start_row >= self.shape[0] || end_row > self.shape[0] || start_row >= end_row {
            return vec![];
        }

        let row_size = self.shape[1..].iter().product::<usize>() * self.itemsize;
        let start_offset = start_row * row_size;
        let total_size = (end_row - start_row) * row_size;

        if start_offset + total_size <= self.mmap.len() {
            self.mmap[start_offset..start_offset + total_size].to_vec()
        } else {
            vec![]
        }
    }

    pub fn get_continuous_data(&self, start_offset: usize, size: usize) -> Vec<u8> {
        if start_offset + size <= self.mmap.len() {
            self.mmap[start_offset..start_offset + size].to_vec()
        } else {
            vec![]
        }
    }
    pub fn get_continuous_zero_copy(&self, start_offset: usize, size: usize) -> &[u8] {
        if start_offset + size <= self.mmap.len() {
            &self.mmap[start_offset..start_offset + size]
        } else {
            &[]
        }
    }

    /// Selects the appropriate boolean indexing strategy based on mask selectivity.
    pub fn boolean_index_smart(&self, mask: &[bool]) -> Vec<Vec<u8>> {
        if mask.len() != self.shape[0] {
            return vec![];
        }

        let selectivity = mask.iter().filter(|&&x| x).count() as f64 / mask.len() as f64;

        if selectivity < 0.1 {
            self.boolean_index_sparse(mask)
        } else if selectivity > 0.8 {
            self.boolean_index_dense(mask)
        } else {
            self.boolean_index_standard(mask)
        }
    }

    /// Highly optimized boolean indexing variant that leverages parallel execution.
    pub fn boolean_index_extreme(&self, mask: &[bool]) -> Vec<Vec<u8>> {
        if mask.len() != self.shape[0] {
            return vec![];
        }

        use rayon::prelude::*;

        let selected_indices: Vec<usize> = mask
            .par_iter()
            .enumerate()
            .filter_map(|(i, &selected)| if selected { Some(i) } else { None })
            .collect();

        selected_indices
            .par_iter()
            .map(|&idx| self.get_row(idx))
            .collect()
    }

    /// Micro-optimized boolean indexing using preallocation and fast row access.
    pub fn boolean_index_micro(&self, mask: &[bool]) -> Vec<Vec<u8>> {
        if mask.len() != self.shape[0] {
            return vec![];
        }

        let selected_count = mask.iter().filter(|&&x| x).count();
        let mut results = Vec::with_capacity(selected_count);

        for (idx, &selected) in mask.iter().enumerate() {
            if selected {
                results.push(self.get_row_fast(idx));
            }
        }

        results
    }

    pub fn slice(&self, ranges: &[std::ops::Range<usize>]) -> Vec<u8> {
        if ranges.is_empty() || ranges.len() != self.shape.len() {
            return vec![];
        }

        // 简化实现：只处理一维切片
        if self.shape.len() == 1 {
            let range = &ranges[0];
            let start = range.start.min(self.shape[0]);
            let end = range.end.min(self.shape[0]);

            if start >= end {
                return vec![];
            }

            let item_size = self.itemsize;
            let start_offset = start * item_size;
            let size = (end - start) * item_size;

            if start_offset + size <= self.mmap.len() {
                self.mmap[start_offset..start_offset + size].to_vec()
            } else {
                vec![]
            }
        } else {
            // 多维切片的简化实现
            vec![]
        }
    }
    /// Performs cache warm-up by sampling rows based on the given rate.
    pub fn warmup_cache(&self, sample_rate: f64) {
        let sample_size = (self.shape[0] as f64 * sample_rate.clamp(0.0, 1.0)) as usize;
        let step = if sample_size > 0 {
            self.shape[0] / sample_size
        } else {
            1
        };

        for i in (0..self.shape[0]).step_by(step) {
            let _ = self.get_row(i);
        }
    }

    pub fn clear_cache(&self) {
        // 清空缓存
        self.cache.clear();
    }

    pub fn boolean_index_adaptive_prefetch(&self, mask: &[bool]) -> Vec<Vec<u8>> {
        if mask.len() != self.shape[0] {
            return vec![];
        }

        // Adaptive prefetch: select strategy based on sampling density
        let selected_indices: Vec<usize> = mask
            .iter()
            .enumerate()
            .filter_map(|(i, &selected)| if selected { Some(i) } else { None })
            .collect();

        if selected_indices.len() < 100 {
            // 小数据集：直接读取
            self.get_rows(&selected_indices)
        } else {
            // 大数据集：使用预取优化
            use rayon::prelude::*;
            selected_indices
                .par_chunks(50)
                .flat_map(|chunk| self.get_rows(chunk))
                .collect()
        }
    }

    pub fn mega_batch_get_rows(&self, indices: &[usize], batch_size: usize) -> Vec<Vec<u8>> {
        if indices.is_empty() {
            return vec![];
        }

        let effective_batch_size = batch_size.max(100).min(1000);
        let mut results = Vec::with_capacity(indices.len());

        // 按批次处理
        for chunk in indices.chunks(effective_batch_size) {
            // 检查是否为连续索引，如果是则优化读取
            let is_continuous = chunk.windows(2).all(|w| w[1] == w[0] + 1);

            if is_continuous && chunk.len() > 10 {
                // 连续索引：使用范围读取
                let start_row = chunk[0];
                let end_row = chunk[chunk.len() - 1] + 1;
                let range_data = self.get_rows_range(start_row, end_row);

                let row_size = self.shape[1..].iter().product::<usize>() * self.itemsize;
                for i in 0..chunk.len() {
                    let start = i * row_size;
                    let end = start + row_size;
                    if end <= range_data.len() {
                        results.push(range_data[start..end].to_vec());
                    }
                }
            } else {
                // 非连续索引：逐个读取
                for &idx in chunk {
                    results.push(self.get_row(idx));
                }
            }
        }

        results
    }

    pub fn get_row_view(&self, row_idx: usize) -> Option<&[u8]> {
        if row_idx >= self.shape[0] {
            return None;
        }

        let row_size = self.shape[1..].iter().product::<usize>() * self.itemsize;
        let offset = row_idx * row_size;

        if offset + row_size <= self.mmap.len() {
            Some(&self.mmap[offset..offset + row_size])
        } else {
            None
        }
    }

    pub fn vectorized_gather(&self, indices: &[usize]) -> Vec<Vec<u8>> {
        if indices.is_empty() {
            return vec![];
        }

        use rayon::prelude::*;

        if indices.len() < 50 {
            self.get_rows(indices)
        } else {
            indices.par_iter().map(|&idx| self.get_row(idx)).collect()
        }
    }

    pub fn parallel_boolean_index(&self, mask: &[bool]) -> Vec<Vec<u8>> {
        if mask.len() != self.shape[0] {
            return vec![];
        }

        use rayon::prelude::*;

        let chunk_size = (mask.len() / rayon::current_num_threads()).max(1000);

        mask.par_chunks(chunk_size)
            .enumerate()
            .flat_map(|(chunk_idx, chunk)| {
                let offset = chunk_idx * chunk_size;
                chunk
                    .iter()
                    .enumerate()
                    .filter_map(|(local_idx, &selected)| {
                        if selected {
                            let global_idx = offset + local_idx;
                            if global_idx < self.shape[0] {
                                Some(self.get_row(global_idx))
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect()
    }
    pub fn warmup_intelligent(&self, hint: &AccessHint) {
        match hint {
            AccessHint::WillAccessAll => {
                // 预热所有数据
                self.warmup_cache(1.0);
            }
            AccessHint::WillAccessRange(start, end) => {
                // 预热指定范围
                let start_idx = *start;
                let end_idx = (*end).min(self.shape[0]);
                for i in start_idx..end_idx {
                    let _ = self.get_row(i);
                }
            }
            AccessHint::WillAccessSparse(ratio) => {
                // 稀疏访问预热
                self.warmup_cache(*ratio);
            }
            AccessHint::WillAccessHot(indices) => {
                // 预热热点数据
                for &idx in indices {
                    if idx < self.shape[0] {
                        let _ = self.get_row(idx);
                    }
                }
            }
        }
    }

    pub fn boolean_index_production(&self, mask: &[bool]) -> Vec<Vec<u8>> {
        // 生产级优化：根据数据规模和选择性自动选择最佳策略
        if mask.len() != self.shape[0] {
            return vec![];
        }

        let selectivity = mask.iter().filter(|&&x| x).count() as f64 / mask.len() as f64;
        let data_size = self.shape[0];

        if data_size < 1000 {
            // 小数据集：使用微优化版本
            self.boolean_index_micro(mask)
        } else if selectivity < 0.05 {
            // 高选择性：使用稀疏优化
            self.boolean_index_sparse(mask)
        } else if selectivity > 0.9 {
            // 低选择性：使用密集优化
            self.boolean_index_dense(mask)
        } else if data_size > 10000 {
            // 大数据集：使用并行版本
            self.boolean_index_extreme(mask)
        } else {
            // 中等数据集：使用智能版本
            self.boolean_index_smart(mask)
        }
    }

    pub fn boolean_index_adaptive_algorithm(&self, mask: &[bool]) -> Vec<Vec<u8>> {
        // Adaptive algorithm: dynamically switch between implementations
        self.boolean_index_production(mask)
    }

    pub fn boolean_index(&self, mask: &[bool]) -> Vec<Vec<u8>> {
        self.boolean_index_standard(mask)
    }

    pub fn boolean_index_optimized(&self, mask: &[bool]) -> Vec<Vec<u8>> {
        self.boolean_index_smart(mask)
    }

    pub fn boolean_index_ultra_fast(&self, mask: &[bool]) -> Vec<Vec<u8>> {
        self.boolean_index_micro(mask)
    }

    pub fn boolean_index_ultimate(&self, mask: &[bool]) -> Vec<Vec<u8>> {
        self.boolean_index_extreme(mask)
    }

    // 高级功能方法（用于high_performance.rs）
    pub fn get_column(&self, col_idx: usize) -> Vec<u8> {
        if self.shape.len() < 2 || col_idx >= self.shape[1] {
            return vec![];
        }

        let num_rows = self.shape[0];
        let row_size = self.shape[1..].iter().product::<usize>() * self.itemsize;
        let col_offset = col_idx * self.itemsize;
        let mut column_data = Vec::with_capacity(num_rows * self.itemsize);

        for row in 0..num_rows {
            let row_offset = row * row_size + col_offset;
            if row_offset + self.itemsize <= self.mmap.len() {
                column_data.extend(&self.mmap[row_offset..row_offset + self.itemsize]);
            }
        }

        column_data
    }

    pub fn get_columns(&self, col_indices: &[usize]) -> Vec<Vec<u8>> {
        if self.shape.len() < 2 {
            return vec![];
        }

        let mut results = Vec::with_capacity(col_indices.len());
        for &col_idx in col_indices {
            results.push(self.get_column(col_idx));
        }
        results
    }

    pub fn simd_parallel_gather(&self, indices: &[usize]) -> Vec<Vec<u8>> {
        // SIMD并行聚合：使用向量化和并行处理
        if indices.is_empty() {
            return vec![];
        }

        use rayon::prelude::*;

        // 按SIMD友好的块大小分组
        const SIMD_CHUNK_SIZE: usize = 8; // AVX2 can process 8 elements at once

        if indices.len() < SIMD_CHUNK_SIZE * 4 {
            // 数据太小，不值得SIMD优化
            return self.get_rows(indices);
        }

        // 分组并行处理
        indices
            .par_chunks(SIMD_CHUNK_SIZE * 2)
            .flat_map(|chunk| {
                // 检查是否连续，如果是则批量读取
                let is_continuous = chunk.len() > 1 && chunk.windows(2).all(|w| w[1] == w[0] + 1);

                if is_continuous {
                    let start_row = chunk[0];
                    let end_row = chunk[chunk.len() - 1] + 1;
                    let range_data = self.get_rows_range(start_row, end_row);
                    let row_size = self.shape[1..].iter().product::<usize>() * self.itemsize;

                    (0..chunk.len())
                        .map(|i| {
                            let start = i * row_size;
                            let end = start + row_size;
                            range_data.get(start..end).unwrap_or(&[]).to_vec()
                        })
                        .collect::<Vec<_>>()
                } else {
                    self.get_rows(chunk)
                }
            })
            .collect()
    }

    pub fn adaptive_gather(&self, indices: &[usize]) -> Vec<Vec<u8>> {
        // Adaptive aggregation: pick the most suitable strategy based on access pattern
        if indices.is_empty() {
            return vec![];
        }

        let data_size = indices.len();
        let locality_score = self.calculate_locality_score(indices);

        if data_size < 100 {
            // 小数据集：直接访问
            self.get_rows(indices)
        } else if locality_score > 0.8 {
            // 高局部性：使用连续读取优化
            self.simd_parallel_gather(indices)
        } else if data_size > 1000 {
            // 大数据集，低局部性：使用并行处理
            self.vectorized_gather(indices)
        } else {
            // 中等数据集：使用智能预取
            self.intelligent_prefetch_gather(indices)
        }
    }

    pub fn hierarchical_memory_prefetch(&self, indices: &[usize]) -> Vec<Vec<u8>> {
        // 分层内存预取：优化内存访问模式
        if indices.is_empty() {
            return vec![];
        }

        // 按缓存行大小（通常64字节）对齐的预取
        const CACHE_LINE_SIZE: usize = 64;
        let row_size = self.shape[1..].iter().product::<usize>() * self.itemsize;
        let rows_per_cache_line = CACHE_LINE_SIZE / row_size.max(1);

        if rows_per_cache_line <= 1 {
            // 行太大，每行占用多个缓存行
            return self.get_rows(indices);
        }

        // 按缓存行分组预取
        let mut results = Vec::with_capacity(indices.len());
        let mut sorted_indices: Vec<(usize, usize)> = indices
            .iter()
            .enumerate()
            .map(|(original_idx, &array_idx)| (array_idx, original_idx))
            .collect();
        sorted_indices.sort_by_key(|&(array_idx, _)| array_idx);

        for group in sorted_indices.chunks(rows_per_cache_line) {
            let group_indices: Vec<usize> = group.iter().map(|&(array_idx, _)| array_idx).collect();
            let group_data = self.get_rows(&group_indices);

            for (i, &(_, original_idx)) in group.iter().enumerate() {
                if i < group_data.len() {
                    if results.len() <= original_idx {
                        results.resize(original_idx + 1, vec![]);
                    }
                    results[original_idx] = group_data[i].clone();
                }
            }
        }

        results
    }

    pub fn numa_aware_gather(&self, indices: &[usize]) -> Vec<Vec<u8>> {
        // NUMA感知聚合：根据NUMA拓扑优化内存访问
        if indices.is_empty() {
            return vec![];
        }

        #[cfg(target_os = "linux")]
        {
            // 在Linux上尝试NUMA优化
            self.numa_optimized_gather(indices)
        }

        #[cfg(not(target_os = "linux"))]
        {
            // 非Linux系统：降级到并行版本
            self.vectorized_gather(indices)
        }
    }

    pub fn intelligent_prefetch_gather(&self, indices: &[usize]) -> Vec<Vec<u8>> {
        // 智能预取聚合：预测访问模式并预取数据
        if indices.is_empty() {
            return vec![];
        }

        // 分析访问模式
        let pattern = self.analyze_access_pattern(indices);

        match pattern {
            AccessPatternType::Sequential => {
                // 顺序访问：使用大块读取
                self.sequential_prefetch_gather(indices)
            }
            AccessPatternType::Strided => {
                // 步长访问：预测下一个访问位置
                self.strided_prefetch_gather(indices)
            }
            AccessPatternType::Random => {
                // 随机访问：使用并行处理
                self.vectorized_gather(indices)
            }
            AccessPatternType::Clustered => {
                // 聚簇访问：按簇分组处理
                self.clustered_prefetch_gather(indices)
            }
        }
    }

    pub fn cpu_accelerated_gather(&self, indices: &[usize]) -> Vec<Vec<u8>> {
        // CPU加速聚合：使用并行处理
        if indices.is_empty() {
            return vec![];
        }

        use rayon::prelude::*;

        // CPU并行处理
        const CPU_THREAD_BLOCK_SIZE: usize = 256;

        indices
            .par_chunks(CPU_THREAD_BLOCK_SIZE)
            .flat_map(|chunk| {
                // 本地缓存优化
                let mut local_cache: std::collections::HashMap<usize, Vec<u8>> =
                    std::collections::HashMap::new();

                chunk
                    .iter()
                    .map(|&idx| {
                        if let Some(cached_data) = local_cache.get(&idx) {
                            cached_data.clone()
                        } else {
                            let data = self.get_row(idx);
                            local_cache.insert(idx, data.clone());
                            data
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    pub fn numa_aware_read(&self, offset: usize, size: usize) -> Vec<u8> {
        // TODO: 实现NUMA感知读取 - 目前使用基础read_data实现
        self.read_data(offset, size)
    }

    // NUMA优化方法（仅Linux）
    #[cfg(target_os = "linux")]
    fn numa_optimized_gather(&self, indices: &[usize]) -> Vec<Vec<u8>> {
        // 在Linux上使用libnuma优化（简化实现）
        self.vectorized_gather(indices)
    }

    // 局部性分数计算
    fn calculate_locality_score(&self, indices: &[usize]) -> f64 {
        if indices.len() < 2 {
            return 1.0;
        }

        let mut consecutive_pairs = 0;
        let mut total_pairs = 0;

        for window in indices.windows(2) {
            let diff = if window[1] > window[0] {
                window[1] - window[0]
            } else {
                window[0] - window[1]
            };

            if diff <= 1 {
                consecutive_pairs += 1;
            }
            total_pairs += 1;
        }

        if total_pairs > 0 {
            consecutive_pairs as f64 / total_pairs as f64
        } else {
            0.0
        }
    }

    // 访问模式分析
    fn analyze_access_pattern(&self, indices: &[usize]) -> AccessPatternType {
        if indices.len() < 3 {
            return AccessPatternType::Random;
        }

        // 检查是否为顺序访问
        let is_sequential = indices.windows(2).all(|w| w[1] == w[0] + 1);
        if is_sequential {
            return AccessPatternType::Sequential;
        }

        // 检查是否为固定步长访问
        let differences: Vec<i64> = indices
            .windows(2)
            .map(|w| w[1] as i64 - w[0] as i64)
            .collect();

        let first_diff = differences[0];
        let is_strided = differences.iter().all(|&diff| diff == first_diff);
        if is_strided {
            return AccessPatternType::Strided;
        }

        // 检查是否为聚簇访问
        let mut clusters = 0;
        let mut in_cluster = false;

        for window in indices.windows(2) {
            let diff = if window[1] > window[0] {
                window[1] - window[0]
            } else {
                window[0] - window[1]
            };

            if diff <= 5 {
                // 定义聚簇阈值
                if !in_cluster {
                    clusters += 1;
                    in_cluster = true;
                }
            } else {
                in_cluster = false;
            }
        }

        if clusters > indices.len() / 10 {
            AccessPatternType::Clustered
        } else {
            AccessPatternType::Random
        }
    }

    // 各种预取策略的实现
    fn sequential_prefetch_gather(&self, indices: &[usize]) -> Vec<Vec<u8>> {
        // 顺序预取：使用大块读取优化
        if let (Some(&first), Some(&last)) = (indices.first(), indices.last()) {
            if last > first && (last - first) == indices.len() - 1 {
                // 真正的顺序访问，使用范围读取
                return self.split_range_data(self.get_rows_range(first, last + 1), indices.len());
            }
        }

        // 非严格顺序，使用标准方法
        self.get_rows(indices)
    }

    fn strided_prefetch_gather(&self, indices: &[usize]) -> Vec<Vec<u8>> {
        // 步长预取：预测并预取下一个位置
        let mut results = Vec::with_capacity(indices.len());
        let mut prefetch_cache: std::collections::HashMap<usize, Vec<u8>> =
            std::collections::HashMap::new();

        for (i, &idx) in indices.iter().enumerate() {
            // 如果已经预取过，直接使用
            if let Some(data) = prefetch_cache.remove(&idx) {
                results.push(data);
            } else {
                results.push(self.get_row(idx));
            }

            // 预测下一个访问位置并预取
            if i < indices.len() - 1 {
                let next_idx = indices[i + 1];
                if !prefetch_cache.contains_key(&next_idx) {
                    prefetch_cache.insert(next_idx, self.get_row(next_idx));
                }
            }
        }

        results
    }

    fn clustered_prefetch_gather(&self, indices: &[usize]) -> Vec<Vec<u8>> {
        // 聚簇预取：按簇分组处理
        let mut clusters = Vec::new();
        let mut current_cluster = Vec::new();

        for (i, &idx) in indices.iter().enumerate() {
            if i == 0 {
                current_cluster.push((idx, i));
            } else {
                let prev_idx = indices[i - 1];
                if idx.abs_diff(prev_idx) <= 5 {
                    // 聚簇阈值
                    current_cluster.push((idx, i));
                } else {
                    if !current_cluster.is_empty() {
                        clusters.push(current_cluster.clone());
                        current_cluster.clear();
                    }
                    current_cluster.push((idx, i));
                }
            }
        }

        if !current_cluster.is_empty() {
            clusters.push(current_cluster);
        }

        // 按簇并行处理
        use rayon::prelude::*;
        let cluster_results: Vec<Vec<(Vec<u8>, usize)>> = clusters
            .par_iter()
            .map(|cluster| {
                let cluster_indices: Vec<usize> = cluster.iter().map(|&(idx, _)| idx).collect();
                let cluster_data = self.get_rows(&cluster_indices);
                cluster_data
                    .into_iter()
                    .zip(cluster.iter().map(|&(_, orig_pos)| orig_pos))
                    .collect()
            })
            .collect();

        // 重新排序结果
        let mut results = vec![vec![]; indices.len()];
        for cluster_result in cluster_results {
            for (data, orig_pos) in cluster_result {
                results[orig_pos] = data;
            }
        }

        results
    }

    fn split_range_data(&self, range_data: Vec<u8>, count: usize) -> Vec<Vec<u8>> {
        let row_size = self.shape[1..].iter().product::<usize>() * self.itemsize;
        let mut results = Vec::with_capacity(count);

        for i in 0..count {
            let start = i * row_size;
            let end = start + row_size;
            if end <= range_data.len() {
                results.push(range_data[start..end].to_vec());
            } else {
                results.push(vec![]);
            }
        }

        results
    }

    // Boolean索引的辅助方法
    fn boolean_index_sparse(&self, mask: &[bool]) -> Vec<Vec<u8>> {
        // 稀疏策略：先收集所有true的索引，然后批量读取
        let indices: Vec<usize> = mask
            .iter()
            .enumerate()
            .filter_map(|(i, &b)| if b { Some(i) } else { None })
            .collect();

        self.get_rows(&indices)
    }

    fn boolean_index_dense(&self, mask: &[bool]) -> Vec<Vec<u8>> {
        // 密集策略：使用范围读取然后过滤
        let mut results = Vec::new();
        let mut current_start = None;

        for (idx, &selected) in mask.iter().enumerate() {
            if selected {
                if current_start.is_none() {
                    current_start = Some(idx);
                }
            } else if let Some(start) = current_start {
                // 读取连续范围
                for i in start..idx {
                    if mask[i] {
                        results.push(self.get_row(i));
                    }
                }
                current_start = None;
            }
        }

        // 处理末尾的连续范围
        if let Some(start) = current_start {
            for i in start..mask.len() {
                if mask[i] {
                    results.push(self.get_row(i));
                }
            }
        }

        results
    }

    fn boolean_index_standard(&self, mask: &[bool]) -> Vec<Vec<u8>> {
        // 标准策略：直接遍历
        let mut results = Vec::new();

        for (idx, &selected) in mask.iter().enumerate() {
            if selected {
                results.push(self.get_row(idx));
            }
        }

        results
    }
}
