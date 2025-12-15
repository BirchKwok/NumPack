//! 访问模式感知的SIMD调度器
//!
//! 本模块根据NumPack的实际访问模式智能选择最优的SIMD策略
//! 包括随机访问、批量访问、流式访问等不同场景的专门优化

use super::{DataType, NumPackSIMD, SIMDStrategy};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// 访问模式类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AccessPattern {
    /// 单次随机访问 - 用户明确的单次访问意图
    SingleRandom,
    /// 批量随机访问 - 多个不连续的索引
    BatchRandom,
    /// 顺序访问 - 连续的内存区域
    Sequential,
    /// 步长访问 - 固定间隔的访问
    Strided,
    /// 聚集访问 - 部分连续的数据块
    Clustered,
    /// 流式访问 - 大规模顺序处理
    Streaming,
}

/// 访问统计信息
#[derive(Debug, Clone)]
pub struct AccessStats {
    pub pattern: AccessPattern,
    pub data_size: usize,
    pub access_count: usize,
    pub avg_latency_ns: u64,
    pub cache_hit_rate: f64,
    pub simd_efficiency: f64,
}

/// 性能历史记录
#[derive(Debug, Clone)]
struct PerformanceRecord {
    strategy: SIMDStrategy,
    latency_ns: u64,
    throughput_mb_s: f64,
    efficiency_score: f64,
    timestamp: Instant,
}

/// 访问模式感知的SIMD调度器
pub struct AccessPatternSIMD {
    simd: NumPackSIMD,
    // 性能历史记录
    performance_history: HashMap<(AccessPattern, DataType, usize), Vec<PerformanceRecord>>,
    // 当前访问统计
    current_stats: HashMap<AccessPattern, AccessStats>,
    // 自适应学习参数
    learning_rate: f64,
    history_window: usize,
}

impl AccessPatternSIMD {
    pub fn new() -> Self {
        Self {
            simd: NumPackSIMD::new(),
            performance_history: HashMap::new(),
            current_stats: HashMap::new(),
            learning_rate: 0.1,
            history_window: 100,
        }
    }

    /// 智能选择SIMD策略 - 基于访问模式和历史性能
    pub fn select_optimal_strategy(
        &mut self,
        pattern: AccessPattern,
        dtype: DataType,
        data_size: usize,
    ) -> SIMDStrategy {
        let key = (pattern, dtype, Self::size_bucket(data_size));

        // 检查历史性能记录
        if let Some(records) = self.performance_history.get(&key) {
            if !records.is_empty() {
                // 基于历史性能选择最优策略
                return self.select_from_history(records, pattern, dtype, data_size);
            }
        }

        // 没有历史记录时，使用启发式规则
        self.heuristic_strategy_selection(pattern, dtype, data_size)
    }

    /// 基于历史性能选择策略
    fn select_from_history(
        &self,
        records: &[PerformanceRecord],
        _pattern: AccessPattern,
        dtype: DataType,
        data_size: usize,
    ) -> SIMDStrategy {
        // 计算每种策略的加权效率分数
        let mut strategy_scores: HashMap<SIMDStrategy, f64> = HashMap::new();

        for record in records.iter().rev().take(self.history_window) {
            let age_weight = self.calculate_age_weight(record.timestamp);
            let size_weight = self.calculate_size_weight(data_size, record);
            let total_weight = age_weight * size_weight;

            *strategy_scores.entry(record.strategy).or_insert(0.0) +=
                record.efficiency_score * total_weight;
        }

        // 选择得分最高的策略
        strategy_scores
            .into_iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(strategy, _)| strategy)
            .unwrap_or_else(|| self.simd.get_optimal_strategy(dtype, data_size))
    }

    /// 启发式策略选择 - 基于访问模式特征
    fn heuristic_strategy_selection(
        &self,
        pattern: AccessPattern,
        dtype: DataType,
        data_size: usize,
    ) -> SIMDStrategy {
        match pattern {
            AccessPattern::SingleRandom => {
                // 单次随机访问：优先使用标量版本，避免SIMD开销
                if data_size < 64 {
                    SIMDStrategy::Scalar
                } else {
                    self.simd.get_optimal_strategy(dtype, data_size)
                }
            }

            AccessPattern::BatchRandom => {
                // 批量随机访问：中等SIMD优化
                if data_size >= 256 {
                    self.select_batch_strategy(dtype, data_size)
                } else {
                    SIMDStrategy::Scalar
                }
            }

            AccessPattern::Sequential => {
                // 顺序访问：最积极的SIMD优化
                self.select_aggressive_simd_strategy(dtype, data_size)
            }

            AccessPattern::Strided => {
                // 步长访问：需要特殊优化
                self.select_strided_strategy(dtype, data_size)
            }

            AccessPattern::Clustered => {
                // 聚集访问：平衡的SIMD策略
                self.select_clustered_strategy(dtype, data_size)
            }

            AccessPattern::Streaming => {
                // 流式访问：最高性能SIMD策略
                self.select_streaming_strategy(dtype, data_size)
            }
        }
    }

    /// 选择批量访问策略
    fn select_batch_strategy(&self, dtype: DataType, data_size: usize) -> SIMDStrategy {
        let base_strategy = self.simd.get_optimal_strategy(dtype, data_size);

        // 根据数据类型调整策略
        match dtype {
            DataType::Int64 | DataType::Uint64 | DataType::Float64 => {
                // 64位数据类型：适度使用SIMD
                if self.simd.capabilities.avx2 && data_size >= 512 {
                    match dtype {
                        DataType::Int64 => SIMDStrategy::AVX2QWord,
                        DataType::Uint64 => SIMDStrategy::AVX2QWord,
                        DataType::Float64 => SIMDStrategy::AVX2QWord,
                        _ => base_strategy,
                    }
                } else {
                    SIMDStrategy::SSE2QWord
                }
            }

            DataType::Int32 | DataType::Uint32 | DataType::Float32 => {
                // 32位数据类型：积极使用SIMD
                if self.simd.capabilities.avx512f && data_size >= 1024 {
                    SIMDStrategy::AVX512DWord
                } else if self.simd.capabilities.avx2 && data_size >= 256 {
                    SIMDStrategy::AVX2DWord
                } else {
                    SIMDStrategy::SSE2DWord
                }
            }

            _ => base_strategy,
        }
    }

    /// 选择最积极的SIMD策略（顺序访问）
    fn select_aggressive_simd_strategy(&self, dtype: DataType, data_size: usize) -> SIMDStrategy {
        // 顺序访问时可以使用最高级的SIMD指令集
        if self.simd.capabilities.avx512f && data_size >= 512 {
            match dtype {
                DataType::Bool => SIMDStrategy::AVX512Bool,
                DataType::Uint8 | DataType::Int8 => SIMDStrategy::AVX512Byte,
                DataType::Uint16 | DataType::Int16 | DataType::Float16 => SIMDStrategy::AVX512Word,
                DataType::Uint32 | DataType::Int32 | DataType::Float32 => SIMDStrategy::AVX512DWord,
                DataType::Uint64 | DataType::Int64 | DataType::Float64 => SIMDStrategy::AVX512QWord,
                DataType::Complex64 => SIMDStrategy::AVX2Complex64,
                DataType::Complex128 => SIMDStrategy::AVX2Complex128,
            }
        } else if self.simd.capabilities.avx2 && data_size >= 128 {
            match dtype {
                DataType::Bool => SIMDStrategy::AVX2Bool,
                DataType::Uint8 | DataType::Int8 => SIMDStrategy::AVX2Byte,
                DataType::Uint16 | DataType::Int16 | DataType::Float16 => SIMDStrategy::AVX2Word,
                DataType::Uint32 | DataType::Int32 | DataType::Float32 => SIMDStrategy::AVX2DWord,
                DataType::Uint64 | DataType::Int64 | DataType::Float64 => SIMDStrategy::AVX2QWord,
                DataType::Complex64 => SIMDStrategy::AVX2Complex64,
                DataType::Complex128 => SIMDStrategy::AVX2Complex128,
            }
        } else {
            self.simd.get_optimal_strategy(dtype, data_size)
        }
    }

    /// 选择步长访问策略
    fn select_strided_strategy(&self, dtype: DataType, data_size: usize) -> SIMDStrategy {
        // 步长访问需要特殊处理，通常SIMD效果不如顺序访问
        if data_size >= 1024 {
            // 大数据集时使用中等SIMD优化
            if self.simd.capabilities.avx2 {
                match dtype {
                    DataType::Float32 => SIMDStrategy::AVX2DWord,
                    DataType::Float64 => SIMDStrategy::AVX2QWord,
                    DataType::Int32 => SIMDStrategy::AVX2DWord,
                    DataType::Int64 => SIMDStrategy::AVX2QWord,
                    _ => self.simd.get_optimal_strategy(dtype, data_size),
                }
            } else {
                SIMDStrategy::SSE2DWord
            }
        } else {
            // 小数据集使用标量版本
            SIMDStrategy::Scalar
        }
    }

    /// 选择聚集访问策略
    fn select_clustered_strategy(&self, dtype: DataType, data_size: usize) -> SIMDStrategy {
        // 聚集访问介于随机和顺序之间
        if data_size >= 512 && self.simd.capabilities.avx2 {
            match dtype {
                DataType::Float32 => SIMDStrategy::AVX2DWord,
                DataType::Float64 => SIMDStrategy::AVX2QWord,
                DataType::Int32 => SIMDStrategy::AVX2DWord,
                DataType::Int64 => SIMDStrategy::AVX2QWord,
                DataType::Uint8 => SIMDStrategy::AVX2Byte,
                DataType::Uint16 => SIMDStrategy::AVX2Word,
                _ => self.simd.get_optimal_strategy(dtype, data_size),
            }
        } else if self.simd.capabilities.sse2 && data_size >= 128 {
            SIMDStrategy::SSE2DWord
        } else {
            SIMDStrategy::Scalar
        }
    }

    /// 选择流式访问策略
    fn select_streaming_strategy(&self, dtype: DataType, data_size: usize) -> SIMDStrategy {
        // 流式访问：使用最高性能的SIMD策略
        if data_size >= 2048 {
            self.select_aggressive_simd_strategy(dtype, data_size)
        } else {
            self.select_batch_strategy(dtype, data_size)
        }
    }

    /// 记录性能数据用于学习
    pub fn record_performance(
        &mut self,
        pattern: AccessPattern,
        dtype: DataType,
        data_size: usize,
        strategy: SIMDStrategy,
        latency: Duration,
        throughput_mb_s: f64,
    ) {
        let key = (pattern, dtype, Self::size_bucket(data_size));
        let efficiency_score = self.calculate_efficiency_score(latency, throughput_mb_s, data_size);

        let record = PerformanceRecord {
            strategy,
            latency_ns: latency.as_nanos() as u64,
            throughput_mb_s,
            efficiency_score,
            timestamp: Instant::now(),
        };

        self.performance_history
            .entry(key)
            .or_insert_with(Vec::new)
            .push(record);

        // 限制历史记录数量
        let history = self.performance_history.get_mut(&key).unwrap();
        if history.len() > self.history_window * 2 {
            history.drain(0..self.history_window);
        }
    }

    /// 自动检测访问模式
    pub fn detect_access_pattern(&self, indices: &[usize], data_size: usize) -> AccessPattern {
        if indices.len() == 1 {
            return AccessPattern::SingleRandom;
        }

        if indices.is_empty() {
            return AccessPattern::Sequential;
        }

        // 分析索引的分布特征
        let is_sequential = self.is_sequential_access(indices);
        let is_strided = self.is_strided_access(indices);
        let clustering_factor = self.calculate_clustering_factor(indices);

        match (is_sequential, is_strided, clustering_factor) {
            (true, _, _) => AccessPattern::Sequential,
            (false, true, _) => AccessPattern::Strided,
            (false, false, factor) if factor > 0.7 => AccessPattern::Clustered,
            (false, false, _) if data_size > 1024 * 1024 => AccessPattern::Streaming,
            _ => AccessPattern::BatchRandom,
        }
    }

    /// 检查是否为顺序访问
    fn is_sequential_access(&self, indices: &[usize]) -> bool {
        if indices.len() < 2 {
            return false;
        }

        for i in 1..indices.len() {
            if indices[i] != indices[i - 1] + 1 {
                return false;
            }
        }
        true
    }

    /// 检查是否为步长访问
    fn is_strided_access(&self, indices: &[usize]) -> bool {
        if indices.len() < 3 {
            return false;
        }

        let stride = indices[1] - indices[0];
        if stride <= 1 {
            return false;
        }

        for i in 2..indices.len() {
            if indices[i] - indices[i - 1] != stride {
                return false;
            }
        }
        true
    }

    /// 计算聚集因子
    fn calculate_clustering_factor(&self, indices: &[usize]) -> f64 {
        if indices.len() < 2 {
            return 0.0;
        }

        let mut sorted_indices = indices.to_vec();
        sorted_indices.sort_unstable();

        let mut cluster_count = 0;
        let mut current_cluster_size = 1;

        for i in 1..sorted_indices.len() {
            let gap = sorted_indices[i] - sorted_indices[i - 1];
            if gap <= 16 {
                // 认为间隔16以内为聚集
                current_cluster_size += 1;
            } else {
                if current_cluster_size >= 3 {
                    cluster_count += 1;
                }
                current_cluster_size = 1;
            }
        }

        if current_cluster_size >= 3 {
            cluster_count += 1;
        }

        cluster_count as f64 / (indices.len() as f64 / 3.0).max(1.0)
    }

    /// 计算效率分数
    fn calculate_efficiency_score(
        &self,
        latency: Duration,
        throughput_mb_s: f64,
        data_size: usize,
    ) -> f64 {
        let latency_score = 1.0 / (latency.as_secs_f64() * 1000.0 + 1.0); // 延迟越低分数越高
        let throughput_score = throughput_mb_s / 1000.0; // 吞吐量分数
        let size_factor = (data_size as f64 / 1024.0).ln().max(1.0); // 数据大小因子

        (latency_score + throughput_score) * size_factor
    }

    /// 计算时间权重（较新的记录权重更高）
    fn calculate_age_weight(&self, timestamp: Instant) -> f64 {
        let age = timestamp.elapsed().as_secs_f64() / 3600.0; // 小时
        (-age * self.learning_rate).exp() // 指数衰减
    }

    /// 计算数据大小权重
    fn calculate_size_weight(&self, current_size: usize, record: &PerformanceRecord) -> f64 {
        let size_ratio = (current_size as f64 / (record.throughput_mb_s * 1024.0 * 1024.0)).ln();
        (-size_ratio.abs() * 0.1).exp() // 大小越接近权重越高
    }

    /// 将数据大小归类到桶中，减少历史记录的维度
    fn size_bucket(size: usize) -> usize {
        match size {
            0..=64 => 64,
            65..=256 => 256,
            257..=1024 => 1024,
            1025..=4096 => 4096,
            4097..=16384 => 16384,
            16385..=65536 => 65536,
            _ => usize::MAX,
        }
    }

    /// 清理过期的性能记录
    pub fn cleanup_expired_records(&mut self, max_age: Duration) {
        let cutoff = Instant::now() - max_age;

        for records in self.performance_history.values_mut() {
            records.retain(|record| record.timestamp > cutoff);
        }

        // 移除空的记录
        self.performance_history
            .retain(|_, records| !records.is_empty());
    }
}
