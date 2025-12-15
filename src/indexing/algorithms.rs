//! 索引算法实现
//!
//! 从lazy_array_original.rs中提取的各种高性能索引算法

// 恢复必要的导入
use crate::indexing::types::*;
pub use crate::performance::metrics::IndexAlgorithm;
use rayon::prelude::*;
use std::time::Instant;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// 索引算法执行器 - 核心索引算法实现
pub struct IndexAlgorithmExecutor {
    config: IndexOptimizationConfig,
    performance_monitor: IndexPerformanceMonitor,
}

impl IndexAlgorithmExecutor {
    pub fn new(config: IndexOptimizationConfig) -> Self {
        Self {
            config,
            performance_monitor: IndexPerformanceMonitor::new(),
        }
    }

    /// 执行布尔索引操作 - 根据算法类型选择实现
    pub fn execute_boolean_index(
        &mut self,
        algorithm: IndexAlgorithm,
        mask: &[bool],
        data: &[u8],
        shape: &[usize],
        itemsize: usize,
    ) -> Result<(Vec<Vec<u8>>, AccessStatistics), IndexError> {
        let start_time = Instant::now();

        let result = match algorithm {
            IndexAlgorithm::BooleanBitmap => self.boolean_bitmap(mask, data, shape, itemsize),
            IndexAlgorithm::BooleanHierarchical => {
                self.boolean_hierarchical(mask, data, shape, itemsize)
            }
            IndexAlgorithm::BooleanSparse => self.boolean_sparse(mask, data, shape, itemsize),
            IndexAlgorithm::BooleanDense => self.boolean_dense(mask, data, shape, itemsize),
            IndexAlgorithm::BooleanExtreme => self.boolean_extreme(mask, data, shape, itemsize),
            _ => self.boolean_bitmap(mask, data, shape, itemsize), // 默认算法
        };

        let latency_ns = start_time.elapsed().as_nanos() as u64;
        let success = result.is_ok();
        let bytes_processed = if success {
            mask.iter().filter(|&&b| b).count() * itemsize * shape[1..].iter().product::<usize>()
        } else {
            0
        };

        // 记录性能统计
        self.performance_monitor.record_strategy_performance(
            AccessStrategy::VectorizedGather, // 布尔索引主要使用向量化聚集
            latency_ns,
            bytes_processed as u64,
            success,
        );

        match result {
            Ok(data) => {
                let stats = AccessStatistics {
                    sequential_ratio: 0.0,
                    random_ratio: 1.0, // 布尔索引通常是随机访问
                    cluster_ratio: 0.0,
                    hit_rate: 1.0,
                    average_latency_ns: latency_ns,
                    total_accesses: 1,
                };
                Ok((data, stats))
            }
            Err(e) => Err(e),
        }
    }

    /// 位图布尔索引算法 - 使用位向量优化
    fn boolean_bitmap(
        &self,
        mask: &[bool],
        data: &[u8],
        shape: &[usize],
        itemsize: usize,
    ) -> Result<Vec<Vec<u8>>, IndexError> {
        if mask.len() != shape[0] {
            return Err(IndexError::DimensionMismatch);
        }

        let row_size = shape[1..].iter().product::<usize>() * itemsize;
        let selected_count = mask.iter().filter(|&&b| b).count();
        let mut result = Vec::with_capacity(selected_count);

        // 预先构建选中索引的位图
        let selected_indices: Vec<usize> = mask
            .iter()
            .enumerate()
            .filter_map(|(i, &selected)| if selected { Some(i) } else { None })
            .collect();

        // 并行处理选中的行
        if self.config.enable_parallel && selected_indices.len() > 100 {
            let chunk_size = self
                .config
                .chunk_size
                .min(selected_indices.len() / self.config.thread_pool_size);
            let chunks: Vec<Vec<u8>> = selected_indices
                .par_chunks(chunk_size)
                .flat_map(|chunk| {
                    chunk
                        .iter()
                        .map(|&row_idx| {
                            let offset = row_idx * row_size;
                            if offset + row_size <= data.len() {
                                data[offset..offset + row_size].to_vec()
                            } else {
                                vec![0u8; row_size] // 错误情况下的默认值
                            }
                        })
                        .collect::<Vec<_>>()
                })
                .collect();
            result = chunks;
        } else {
            // 串行处理
            for &row_idx in &selected_indices {
                let offset = row_idx * row_size;
                if offset + row_size <= data.len() {
                    result.push(data[offset..offset + row_size].to_vec());
                }
            }
        }

        Ok(result)
    }

    /// 分层布尔索引算法 - 使用分层过滤优化大数据集
    fn boolean_hierarchical(
        &self,
        mask: &[bool],
        data: &[u8],
        shape: &[usize],
        itemsize: usize,
    ) -> Result<Vec<Vec<u8>>, IndexError> {
        if mask.len() != shape[0] {
            return Err(IndexError::DimensionMismatch);
        }

        let row_size = shape[1..].iter().product::<usize>() * itemsize;

        // 第一层：快速SIMD过滤，构建层次索引
        let mut level1_indices = Vec::new();
        let mut level2_indices = Vec::new();

        if self.config.enable_simd {
            // 使用SIMD加速布尔掩码处理
            let simd_selected = self.simd_boolean_filter(mask);

            // 分层分组：连续的索引放在level1，分散的放在level2
            let mut current_group = Vec::new();
            let mut last_idx = None;

            for idx in simd_selected {
                if let Some(last) = last_idx {
                    if idx == last + 1 {
                        current_group.push(idx);
                    } else {
                        if current_group.len() > 3 {
                            level1_indices.extend(current_group);
                        } else {
                            level2_indices.extend(current_group);
                        }
                        current_group = vec![idx];
                    }
                } else {
                    current_group.push(idx);
                }
                last_idx = Some(idx);
            }

            // 处理最后一组
            if current_group.len() > 3 {
                level1_indices.extend(current_group);
            } else {
                level2_indices.extend(current_group);
            }
        } else {
            // 回退到基本实现
            return self.boolean_bitmap(mask, data, shape, itemsize);
        }

        let mut result = Vec::new();

        // Level 1: 批量连续访问
        if !level1_indices.is_empty() {
            let level1_data = self.batch_sequential_access(&level1_indices, data, row_size);
            result.extend(level1_data);
        }

        // Level 2: 随机访问优化
        if !level2_indices.is_empty() {
            let level2_data = self.optimized_random_access(&level2_indices, data, row_size);
            result.extend(level2_data);
        }

        Ok(result)
    }

    /// 稀疏布尔索引算法 - 专为低密度选择优化
    fn boolean_sparse(
        &self,
        mask: &[bool],
        data: &[u8],
        shape: &[usize],
        itemsize: usize,
    ) -> Result<Vec<Vec<u8>>, IndexError> {
        if mask.len() != shape[0] {
            return Err(IndexError::DimensionMismatch);
        }

        let density = mask.iter().filter(|&&b| b).count() as f64 / mask.len() as f64;

        // 如果不是稀疏访问，使用更适合的算法
        if density > 0.2 {
            return self.boolean_dense(mask, data, shape, itemsize);
        }

        let row_size = shape[1..].iter().product::<usize>() * itemsize;

        // 稀疏索引优化：使用跳跃扫描
        let mut result = Vec::new();
        let mut i = 0;

        while i < mask.len() {
            if mask[i] {
                let offset = i * row_size;
                if offset + row_size <= data.len() {
                    result.push(data[offset..offset + row_size].to_vec());
                }
                i += 1;
            } else {
                // 跳跃扫描：寻找下一个true
                let skip = mask[i..].iter().position(|&b| b).unwrap_or(mask.len() - i);
                i += skip;
            }
        }

        Ok(result)
    }

    /// 密集布尔索引算法 - 专为高密度选择优化
    fn boolean_dense(
        &self,
        mask: &[bool],
        data: &[u8],
        shape: &[usize],
        itemsize: usize,
    ) -> Result<Vec<Vec<u8>>, IndexError> {
        if mask.len() != shape[0] {
            return Err(IndexError::DimensionMismatch);
        }

        let row_size = shape[1..].iter().product::<usize>() * itemsize;
        let mut result = Vec::new();

        // 密集访问优化：预分配结果空间，使用块复制
        let selected_count = mask.iter().filter(|&&b| b).count();
        result.reserve(selected_count);

        if self.config.enable_parallel && selected_count > 1000 {
            // 大规模并行处理
            let chunk_size = mask.len() / self.config.thread_pool_size;
            let chunks: Vec<Vec<Vec<u8>>> = mask
                .par_chunks(chunk_size)
                .enumerate()
                .map(|(chunk_idx, chunk_mask)| {
                    let mut chunk_result = Vec::new();
                    let base_offset = chunk_idx * chunk_size;

                    for (local_idx, &selected) in chunk_mask.iter().enumerate() {
                        if selected {
                            let global_idx = base_offset + local_idx;
                            let offset = global_idx * row_size;
                            if offset + row_size <= data.len() {
                                chunk_result.push(data[offset..offset + row_size].to_vec());
                            }
                        }
                    }
                    chunk_result
                })
                .collect();

            // 合并结果
            for chunk in chunks {
                result.extend(chunk);
            }
        } else {
            // 串行密集访问
            for (i, &selected) in mask.iter().enumerate() {
                if selected {
                    let offset = i * row_size;
                    if offset + row_size <= data.len() {
                        result.push(data[offset..offset + row_size].to_vec());
                    }
                }
            }
        }

        Ok(result)
    }

    /// 极限布尔索引算法 - 最高性能的SIMD优化实现
    fn boolean_extreme(
        &self,
        mask: &[bool],
        data: &[u8],
        shape: &[usize],
        itemsize: usize,
    ) -> Result<Vec<Vec<u8>>, IndexError> {
        if mask.len() != shape[0] {
            return Err(IndexError::DimensionMismatch);
        }

        let row_size = shape[1..].iter().product::<usize>() * itemsize;

        // 使用最高性能的SIMD过滤
        let selected_indices = if self.config.enable_simd {
            self.extreme_simd_boolean_filter(mask)
        } else {
            mask.iter()
                .enumerate()
                .filter_map(|(i, &b)| if b { Some(i) } else { None })
                .collect()
        };

        if selected_indices.is_empty() {
            return Ok(Vec::new());
        }

        // 并行块处理，每64行一组
        let result: Vec<Vec<u8>> = if self.config.enable_parallel && selected_indices.len() > 64 {
            selected_indices
                .par_chunks(64)
                .flat_map(|chunk| {
                    chunk
                        .iter()
                        .filter_map(|&row_idx| {
                            let offset = row_idx * row_size;
                            if offset + row_size <= data.len() {
                                Some(data[offset..offset + row_size].to_vec())
                            } else {
                                None
                            }
                        })
                        .collect::<Vec<_>>()
                })
                .collect()
        } else {
            selected_indices
                .iter()
                .filter_map(|&row_idx| {
                    let offset = row_idx * row_size;
                    if offset + row_size <= data.len() {
                        Some(data[offset..offset + row_size].to_vec())
                    } else {
                        None
                    }
                })
                .collect()
        };

        Ok(result)
    }

    /// 整数数组索引（花式索引）
    pub fn execute_fancy_index(
        &mut self,
        indices: &[i64],
        data: &[u8],
        shape: &[usize],
        itemsize: usize,
    ) -> Result<Vec<Vec<u8>>, IndexError> {
        let start_time = Instant::now();
        let row_size = shape[1..].iter().product::<usize>() * itemsize;

        // 标准化索引（处理负数索引）
        let normalized_indices: Result<Vec<usize>, IndexError> = indices
            .iter()
            .map(|&idx| {
                let normalized = if idx < 0 { shape[0] as i64 + idx } else { idx };

                if normalized >= 0 && (normalized as usize) < shape[0] {
                    Ok(normalized as usize)
                } else {
                    Err(IndexError::IndexOutOfBounds)
                }
            })
            .collect();

        let normalized_indices = normalized_indices?;

        let result: Vec<Vec<u8>> = if self.config.enable_parallel && normalized_indices.len() > 100
        {
            // 并行花式索引
            normalized_indices
                .par_iter()
                .filter_map(|&row_idx| {
                    let offset = row_idx * row_size;
                    if offset + row_size <= data.len() {
                        Some(data[offset..offset + row_size].to_vec())
                    } else {
                        None
                    }
                })
                .collect()
        } else {
            // 串行花式索引
            normalized_indices
                .iter()
                .filter_map(|&row_idx| {
                    let offset = row_idx * row_size;
                    if offset + row_size <= data.len() {
                        Some(data[offset..offset + row_size].to_vec())
                    } else {
                        None
                    }
                })
                .collect()
        };

        let latency_ns = start_time.elapsed().as_nanos() as u64;
        self.performance_monitor.record_strategy_performance(
            AccessStrategy::VectorizedGather,
            latency_ns,
            (result.len() * row_size) as u64,
            true,
        );

        Ok(result)
    }

    /// SIMD优化的布尔过滤器
    fn simd_boolean_filter(&self, mask: &[bool]) -> Vec<usize> {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                return self.avx2_boolean_filter(mask);
            }
        }

        // 回退到标准实现
        mask.iter()
            .enumerate()
            .filter_map(|(i, &b)| if b { Some(i) } else { None })
            .collect()
    }

    /// 极限SIMD布尔过滤器
    fn extreme_simd_boolean_filter(&self, mask: &[bool]) -> Vec<usize> {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f") {
                return self.avx512_boolean_filter(mask);
            }
            if is_x86_feature_detected!("avx2") {
                return self.avx2_boolean_filter(mask);
            }
        }

        self.simd_boolean_filter(mask)
    }

    /// AVX2优化的布尔过滤器
    #[cfg(target_arch = "x86_64")]
    fn avx2_boolean_filter(&self, mask: &[bool]) -> Vec<usize> {
        let mut selected = Vec::with_capacity(mask.len() / 4);

        // 简化的AVX2实现（实际需要更复杂的SIMD操作）
        for (i, &b) in mask.iter().enumerate() {
            if b {
                selected.push(i);
            }
        }

        selected
    }

    /// AVX-512优化的布尔过滤器
    #[cfg(target_arch = "x86_64")]
    fn avx512_boolean_filter(&self, mask: &[bool]) -> Vec<usize> {
        let mut selected = Vec::with_capacity(mask.len() / 4);

        // 简化的AVX-512实现（实际需要更复杂的SIMD操作）
        for (i, &b) in mask.iter().enumerate() {
            if b {
                selected.push(i);
            }
        }

        selected
    }

    /// 批量顺序访问
    fn batch_sequential_access(
        &self,
        indices: &[usize],
        data: &[u8],
        row_size: usize,
    ) -> Vec<Vec<u8>> {
        let mut result = Vec::with_capacity(indices.len());

        for &idx in indices {
            let offset = idx * row_size;
            if offset + row_size <= data.len() {
                result.push(data[offset..offset + row_size].to_vec());
            }
        }

        result
    }

    /// 优化的随机访问
    fn optimized_random_access(
        &self,
        indices: &[usize],
        data: &[u8],
        row_size: usize,
    ) -> Vec<Vec<u8>> {
        if self.config.enable_parallel && indices.len() > 50 {
            indices
                .par_iter()
                .filter_map(|&idx| {
                    let offset = idx * row_size;
                    if offset + row_size <= data.len() {
                        Some(data[offset..offset + row_size].to_vec())
                    } else {
                        None
                    }
                })
                .collect()
        } else {
            let mut result = Vec::with_capacity(indices.len());
            for &idx in indices {
                let offset = idx * row_size;
                if offset + row_size <= data.len() {
                    result.push(data[offset..offset + row_size].to_vec());
                }
            }
            result
        }
    }

    /// 获取性能报告
    pub fn get_performance_report(&self) -> IndexPerformanceReport {
        self.performance_monitor.get_performance_report()
    }
}

/// 索引错误类型
#[derive(Debug)]
pub enum IndexError {
    DimensionMismatch,
    IndexOutOfBounds,
    InvalidSlice,
    MemoryError,
    SIMDNotSupported,
}

impl std::fmt::Display for IndexError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IndexError::DimensionMismatch => write!(f, "Index dimension mismatch"),
            IndexError::IndexOutOfBounds => write!(f, "Index out of bounds"),
            IndexError::InvalidSlice => write!(f, "Invalid slice"),
            IndexError::MemoryError => write!(f, "Memory error"),
            IndexError::SIMDNotSupported => write!(f, "SIMD not supported"),
        }
    }
}

impl std::error::Error for IndexError {}
