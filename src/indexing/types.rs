//! 索引系统类型定义
//!
//! 从lazy_array_original.rs中提取和重构的索引相关类型

use std::collections::HashMap;
use std::time::Instant;

/// 索引类型枚举 - 支持多种NumPy风格的索引方式
#[derive(Debug, Clone)]
pub enum IndexType {
    Integer(i64),           // 单个整数索引
    Slice(SliceInfo),       // 切片索引 [start:stop:step]
    BooleanMask(Vec<bool>), // 布尔掩码索引
    IntegerArray(Vec<i64>), // 整数数组索引（花式索引）
    Ellipsis,               // 省略号 ...
    NewAxis,                // 新轴 np.newaxis
}

/// 切片信息结构
#[derive(Debug, Clone)]
pub struct SliceInfo {
    pub start: Option<i64>,
    pub stop: Option<i64>,
    pub step: Option<i64>,
}

impl SliceInfo {
    pub fn new(start: Option<i64>, stop: Option<i64>, step: Option<i64>) -> Self {
        Self { start, stop, step }
    }
}

/// 索引解析结果
#[derive(Debug, Clone)]
pub struct IndexResult {
    pub indices: Vec<Vec<usize>>,      // 每个维度的具体索引值
    pub result_shape: Vec<usize>,      // 结果数组的形状
    pub needs_broadcasting: bool,      // 是否需要广播
    pub access_pattern: AccessPattern, // 访问模式
    pub estimated_size: usize,         // 预估数据大小
}

/// 访问模式枚举 - 描述数据访问的特征
#[derive(Debug, Clone, PartialEq)]
pub enum AccessPattern {
    Sequential, // 顺序访问：连续的内存位置
    Random,     // 随机访问：分散的内存位置
    Clustered,  // 聚集访问：部分连续的块
    Mixed,      // 混合访问：顺序和随机的组合
    Strided,    // 步长访问：等间距访问
    Sparse,     // 稀疏访问：大部分位置不访问
}

/// 访问策略枚举 - 决定如何执行索引操作
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum AccessStrategy {
    DirectMemory,        // 直接内存访问
    BlockCopy,           // 块复制访问
    ParallelPointAccess, // 并行点访问
    PrefetchOptimized,   // 预取优化访问
    ZeroCopy,            // 零拷贝访问
    VectorizedGather,    // 向量化聚集
    Adaptive,            // 自适应访问
}

/// 访问模式分析器
#[derive(Debug)]
pub struct AccessPatternAnalyzer {
    // 历史访问数据
    access_history: Vec<AccessInfo>,
    // 统计信息
    sequential_ratio: f64,
    random_ratio: f64,
    cluster_ratio: f64,
    // 性能指标
    hit_rate: f64,
    average_latency_ns: u64,
}

/// 单次访问信息
#[derive(Debug, Clone)]
pub struct AccessInfo {
    pub indices: Vec<usize>,
    pub pattern: AccessPattern,
    pub size: usize,
    pub timestamp: Instant,
    pub latency_ns: u64,
}

impl AccessPatternAnalyzer {
    pub fn new() -> Self {
        Self {
            access_history: Vec::new(),
            sequential_ratio: 0.0,
            random_ratio: 0.0,
            cluster_ratio: 0.0,
            hit_rate: 0.0,
            average_latency_ns: 0,
        }
    }

    /// 分析索引模式并返回访问模式类型
    pub fn analyze_indices(&mut self, indices: &[Vec<usize>]) -> AccessPattern {
        if indices.is_empty() {
            return AccessPattern::Random;
        }

        // 分析第一个维度的访问模式
        let first_dim = &indices[0];
        if first_dim.is_empty() {
            return AccessPattern::Random;
        }

        let pattern = if self.is_sequential(first_dim) {
            AccessPattern::Sequential
        } else if self.is_strided(first_dim) {
            AccessPattern::Strided
        } else if self.is_clustered(first_dim) {
            AccessPattern::Clustered
        } else if self.is_sparse(first_dim, 1000) {
            // 假设总大小1000
            AccessPattern::Sparse
        } else {
            AccessPattern::Random
        };

        // 记录访问信息
        let access_info = AccessInfo {
            indices: first_dim.clone(),
            pattern: pattern.clone(),
            size: first_dim.len(),
            timestamp: Instant::now(),
            latency_ns: 0, // 将在后续填充
        };

        self.access_history.push(access_info);
        self.update_statistics();

        pattern
    }

    /// 检查是否为顺序访问
    fn is_sequential(&self, indices: &[usize]) -> bool {
        if indices.len() < 2 {
            return true;
        }

        for i in 1..indices.len() {
            if indices[i] != indices[i - 1] + 1 {
                return false;
            }
        }
        true
    }

    /// 检查是否为步长访问
    fn is_strided(&self, indices: &[usize]) -> bool {
        if indices.len() < 3 {
            return false;
        }

        let stride = indices[1] as i64 - indices[0] as i64;
        for i in 2..indices.len() {
            if indices[i] as i64 - indices[i - 1] as i64 != stride {
                return false;
            }
        }
        stride > 1
    }

    /// 检查是否为聚集访问
    fn is_clustered(&self, indices: &[usize]) -> bool {
        if indices.len() < 2 {
            return false;
        }

        let mut sorted_indices = indices.to_vec();
        sorted_indices.sort_unstable();

        let mut clusters = 0;
        let mut in_cluster = false;

        for i in 1..sorted_indices.len() {
            let gap = sorted_indices[i] - sorted_indices[i - 1];
            if gap <= 10 {
                // 间隔小于等于10认为是聚集
                if !in_cluster {
                    clusters += 1;
                    in_cluster = true;
                }
            } else {
                in_cluster = false;
            }
        }

        clusters > 0 && clusters < indices.len() / 3
    }

    /// 检查是否为稀疏访问
    fn is_sparse(&self, indices: &[usize], total_size: usize) -> bool {
        let density = indices.len() as f64 / total_size as f64;
        density < 0.1 // 密度小于10%认为是稀疏
    }

    /// 更新统计信息
    fn update_statistics(&mut self) {
        if self.access_history.is_empty() {
            return;
        }

        let recent_window = 100; // 只考虑最近100次访问
        let start_idx = if self.access_history.len() > recent_window {
            self.access_history.len() - recent_window
        } else {
            0
        };

        let recent_accesses = &self.access_history[start_idx..];
        let total = recent_accesses.len() as f64;

        let sequential_count = recent_accesses
            .iter()
            .filter(|a| a.pattern == AccessPattern::Sequential)
            .count() as f64;
        let random_count = recent_accesses
            .iter()
            .filter(|a| a.pattern == AccessPattern::Random)
            .count() as f64;
        let cluster_count = recent_accesses
            .iter()
            .filter(|a| a.pattern == AccessPattern::Clustered)
            .count() as f64;

        self.sequential_ratio = sequential_count / total;
        self.random_ratio = random_count / total;
        self.cluster_ratio = cluster_count / total;

        // 计算平均延迟
        let total_latency: u64 = recent_accesses.iter().map(|a| a.latency_ns).sum();
        self.average_latency_ns = if recent_accesses.is_empty() {
            0
        } else {
            total_latency / recent_accesses.len() as u64
        };
    }

    /// 记录访问性能
    pub fn record_access_performance(&mut self, latency_ns: u64) {
        if let Some(last_access) = self.access_history.last_mut() {
            last_access.latency_ns = latency_ns;
        }
        self.update_statistics();
    }

    /// 获取当前统计信息
    pub fn get_statistics(&self) -> AccessStatistics {
        AccessStatistics {
            sequential_ratio: self.sequential_ratio,
            random_ratio: self.random_ratio,
            cluster_ratio: self.cluster_ratio,
            hit_rate: self.hit_rate,
            average_latency_ns: self.average_latency_ns,
            total_accesses: self.access_history.len(),
        }
    }

    /// 预测最优访问策略
    pub fn predict_optimal_strategy(&self, estimated_size: usize) -> AccessStrategy {
        // 小数据量直接访问
        if estimated_size < 1024 {
            return AccessStrategy::DirectMemory;
        }

        // 根据历史模式预测
        if self.sequential_ratio > 0.7 {
            if estimated_size > 1024 * 1024 {
                AccessStrategy::BlockCopy
            } else {
                AccessStrategy::DirectMemory
            }
        } else if self.random_ratio > 0.7 {
            if estimated_size > 100 * 1024 {
                AccessStrategy::ParallelPointAccess
            } else {
                AccessStrategy::VectorizedGather
            }
        } else if self.cluster_ratio > 0.5 {
            AccessStrategy::PrefetchOptimized
        } else {
            AccessStrategy::Adaptive
        }
    }
}

/// 访问统计信息
#[derive(Debug, Clone)]
pub struct AccessStatistics {
    pub sequential_ratio: f64,
    pub random_ratio: f64,
    pub cluster_ratio: f64,
    pub hit_rate: f64,
    pub average_latency_ns: u64,
    pub total_accesses: usize,
}

/// 索引优化配置
#[derive(Debug, Clone)]
pub struct IndexOptimizationConfig {
    pub enable_simd: bool,
    pub enable_parallel: bool,
    pub enable_prefetch: bool,
    pub enable_zero_copy: bool,
    pub cache_size: usize,
    pub chunk_size: usize,
    pub thread_pool_size: usize,
}

impl Default for IndexOptimizationConfig {
    fn default() -> Self {
        Self {
            enable_simd: true,
            enable_parallel: true,
            enable_prefetch: true,
            enable_zero_copy: true,
            cache_size: 64 * 1024 * 1024, // 64MB
            chunk_size: 4096,
            thread_pool_size: rayon::current_num_threads(),
        }
    }
}

/// 索引性能监控器
#[derive(Debug)]
pub struct IndexPerformanceMonitor {
    strategy_stats: HashMap<AccessStrategy, StrategyStats>,
    global_stats: GlobalIndexStats,
}

#[derive(Debug, Clone)]
pub struct StrategyStats {
    pub total_operations: u64,
    pub total_latency_ns: u64,
    pub total_bytes: u64,
    pub error_count: u64,
    pub last_used: Instant,
}

#[derive(Debug, Clone)]
pub struct GlobalIndexStats {
    pub total_index_operations: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub average_throughput_mbps: f64,
    pub peak_memory_usage: usize,
}

impl IndexPerformanceMonitor {
    pub fn new() -> Self {
        Self {
            strategy_stats: HashMap::new(),
            global_stats: GlobalIndexStats {
                total_index_operations: 0,
                cache_hits: 0,
                cache_misses: 0,
                average_throughput_mbps: 0.0,
                peak_memory_usage: 0,
            },
        }
    }

    /// 记录策略性能
    pub fn record_strategy_performance(
        &mut self,
        strategy: AccessStrategy,
        latency_ns: u64,
        bytes_processed: u64,
        success: bool,
    ) {
        let stats = self
            .strategy_stats
            .entry(strategy)
            .or_insert(StrategyStats {
                total_operations: 0,
                total_latency_ns: 0,
                total_bytes: 0,
                error_count: 0,
                last_used: Instant::now(),
            });

        stats.total_operations += 1;
        stats.total_latency_ns += latency_ns;
        stats.total_bytes += bytes_processed;
        stats.last_used = Instant::now();

        if !success {
            stats.error_count += 1;
        }

        self.global_stats.total_index_operations += 1;
    }

    /// 获取最佳策略建议
    pub fn get_best_strategy_recommendation(&self, pattern: &AccessPattern) -> AccessStrategy {
        // 根据模式和历史性能数据推荐最佳策略
        match pattern {
            AccessPattern::Sequential => {
                if let Some(block_stats) = self.strategy_stats.get(&AccessStrategy::BlockCopy) {
                    if let Some(direct_stats) =
                        self.strategy_stats.get(&AccessStrategy::DirectMemory)
                    {
                        if block_stats.total_operations > 10 && direct_stats.total_operations > 10 {
                            let block_avg_latency =
                                block_stats.total_latency_ns / block_stats.total_operations;
                            let direct_avg_latency =
                                direct_stats.total_latency_ns / direct_stats.total_operations;

                            if block_avg_latency < direct_avg_latency {
                                return AccessStrategy::BlockCopy;
                            }
                        }
                    }
                }
                AccessStrategy::BlockCopy
            }
            AccessPattern::Random => AccessStrategy::VectorizedGather,
            AccessPattern::Clustered => AccessStrategy::PrefetchOptimized,
            AccessPattern::Strided => AccessStrategy::VectorizedGather,
            AccessPattern::Sparse => AccessStrategy::ParallelPointAccess,
            AccessPattern::Mixed => AccessStrategy::Adaptive,
        }
    }

    /// 获取性能报告
    pub fn get_performance_report(&self) -> IndexPerformanceReport {
        IndexPerformanceReport {
            strategy_performance: self.strategy_stats.clone(),
            global_stats: self.global_stats.clone(),
            recommendation_accuracy: self.calculate_recommendation_accuracy(),
        }
    }

    fn calculate_recommendation_accuracy(&self) -> f64 {
        // 简化的准确性计算
        if self.global_stats.total_index_operations == 0 {
            return 0.0;
        }

        let total_errors: u64 = self
            .strategy_stats
            .values()
            .map(|stats| stats.error_count)
            .sum();

        1.0 - (total_errors as f64 / self.global_stats.total_index_operations as f64)
    }
}

#[derive(Debug, Clone)]
pub struct IndexPerformanceReport {
    pub strategy_performance: HashMap<AccessStrategy, StrategyStats>,
    pub global_stats: GlobalIndexStats,
    pub recommendation_accuracy: f64,
}
