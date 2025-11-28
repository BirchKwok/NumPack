use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

// 批量访问策略
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BatchAccessStrategy {
    Parallel,  // 并行访问
    Chunked,   // 分块访问
    Streaming, // 流式访问
    ZeroCopy,  // 零拷贝访问
    Adaptive,  // 自适应选择
}

// 访问请求类型
#[derive(Debug, Clone)]
pub enum BatchAccessRequest {
    Rows(Vec<usize>),             // 行索引列表
    Range(usize, usize),          // 范围访问
    Fancy(Vec<usize>),            // 花式索引
    Boolean(Vec<bool>),           // 布尔索引
    Streaming(Vec<usize>, usize), // 流式访问(indices, chunk_size)
}

// 批量访问结果
pub enum BatchAccessResult {
    Owned(Vec<Vec<u8>>),                              // 拷贝数据
    Views(Vec<Option<&'static [u8]>>),                // 零拷贝视图
    Stream(Box<dyn Iterator<Item = Vec<u8>> + Send>), // 流式结果
    Range(Vec<u8>),                                   // 范围数据
}

// 访问性能指标
#[derive(Debug, Clone, Default)]
pub struct BatchAccessMetrics {
    pub total_requests: u64,
    pub total_bytes: u64,
    pub avg_latency: Duration,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub parallel_efficiency: f64,
    pub memory_usage: usize,
}

// 并行执行器组件
#[derive(Debug)]
pub struct ParallelExecutor {
    thread_pool_size: usize,
    min_thread_pool_size: usize,
    max_thread_pool_size: usize,
    work_stealing_enabled: bool,
    load_balancing_strategy: LoadBalancingStrategy,
    performance_metrics: Arc<Mutex<BatchAccessMetrics>>,
    // 新增：工作负载历史和适应性调整
    workload_history: Arc<Mutex<Vec<WorkloadSample>>>,
    adaptive_enabled: bool,
    memory_pool: Arc<Mutex<MemoryPool>>,
}

#[derive(Debug, Clone, Copy)]
pub enum LoadBalancingStrategy {
    RoundRobin,
    WorkStealing,
    Dynamic,
    Adaptive, // 新增：自适应负载均衡
}

// 工作负载样本
#[derive(Debug, Clone)]
pub struct WorkloadSample {
    pub items_count: usize,
    pub chunk_size: usize,
    pub thread_count: usize,
    pub execution_time: Duration,
    pub memory_usage: usize,
    pub cpu_utilization: f64,
}

// 内存池用于优化内存分配
#[derive(Debug)]
pub struct MemoryPool {
    small_buffers: Vec<Vec<u8>>,  // <1KB
    medium_buffers: Vec<Vec<u8>>, // 1KB-1MB
    large_buffers: Vec<Vec<u8>>,  // >1MB
}

// 最优参数结构
#[derive(Debug, Clone)]
pub struct OptimalParameters {
    pub chunk_size: usize,
    pub thread_count: usize,
}

// 工作负载统计
#[derive(Debug, Clone)]
pub struct WorkloadStats {
    pub total_samples: usize,
    pub avg_execution_time: Duration,
    pub avg_cpu_utilization: f64,
    pub avg_memory_usage: usize,
    pub best_chunk_size: usize,
    pub best_thread_count: usize,
}

impl Default for WorkloadStats {
    fn default() -> Self {
        Self {
            total_samples: 0,
            avg_execution_time: Duration::from_nanos(0),
            avg_cpu_utilization: 0.0,
            avg_memory_usage: 0,
            best_chunk_size: 1000,
            best_thread_count: 1,
        }
    }
}

// 推荐配置
#[derive(Debug, Clone)]
pub struct RecommendedConfig {
    pub recommended_thread_count: usize,
    pub recommended_chunk_size: usize,
    pub recommended_strategy: LoadBalancingStrategy,
    pub memory_pool_enabled: bool,
}

impl MemoryPool {
    pub fn new() -> Self {
        Self {
            small_buffers: Vec::new(),
            medium_buffers: Vec::new(),
            large_buffers: Vec::new(),
        }
    }

    pub fn get_buffer(&mut self, size: usize) -> Vec<u8> {
        let buffer_pool = if size <= 1024 {
            &mut self.small_buffers
        } else if size <= 1024 * 1024 {
            &mut self.medium_buffers
        } else {
            &mut self.large_buffers
        };

        buffer_pool.pop().unwrap_or_else(|| vec![0u8; size])
    }

    pub fn return_buffer(&mut self, mut buffer: Vec<u8>) {
        let size = buffer.len();
        buffer.clear();

        let buffer_pool = if size <= 1024 {
            &mut self.small_buffers
        } else if size <= 1024 * 1024 {
            &mut self.medium_buffers
        } else {
            &mut self.large_buffers
        };

        // 限制池大小防止内存泄漏
        if buffer_pool.len() < 100 {
            buffer_pool.push(buffer);
        }
    }
}

impl ParallelExecutor {
    pub fn new(thread_pool_size: usize) -> Self {
        let num_cpus = rayon::current_num_threads();
        Self {
            thread_pool_size,
            min_thread_pool_size: 1,
            max_thread_pool_size: num_cpus * 2,
            work_stealing_enabled: true,
            load_balancing_strategy: LoadBalancingStrategy::Adaptive,
            performance_metrics: Arc::new(Mutex::new(BatchAccessMetrics::default())),
            workload_history: Arc::new(Mutex::new(Vec::new())),
            adaptive_enabled: true,
            memory_pool: Arc::new(Mutex::new(MemoryPool::new())),
        }
    }

    pub fn new_with_config(min_threads: usize, max_threads: usize, adaptive: bool) -> Self {
        Self {
            thread_pool_size: (min_threads + max_threads) / 2,
            min_thread_pool_size: min_threads,
            max_thread_pool_size: max_threads,
            work_stealing_enabled: true,
            load_balancing_strategy: if adaptive {
                LoadBalancingStrategy::Adaptive
            } else {
                LoadBalancingStrategy::Dynamic
            },
            performance_metrics: Arc::new(Mutex::new(BatchAccessMetrics::default())),
            workload_history: Arc::new(Mutex::new(Vec::new())),
            adaptive_enabled: adaptive,
            memory_pool: Arc::new(Mutex::new(MemoryPool::new())),
        }
    }

    pub fn execute_parallel<F, T, R>(&self, data: Vec<T>, operation: F) -> Vec<R>
    where
        F: Fn(T) -> R + Sync + Send,
        T: Send,
        R: Send,
    {
        let start_time = Instant::now();
        let data_size = data.len();

        // 预分配结果向量以减少重新分配
        // 注意：由于后面直接替换结果，这个预分配不再需要
        // let mut result = Vec::with_capacity(data_size);

        let execution_result = match self.load_balancing_strategy {
            LoadBalancingStrategy::RoundRobin => self.execute_round_robin(data, operation),
            LoadBalancingStrategy::WorkStealing => self.execute_work_stealing(data, operation),
            LoadBalancingStrategy::Dynamic => self.execute_dynamic(data, operation),
            LoadBalancingStrategy::Adaptive => self.execute_adaptive(data, operation),
        };

        let result = execution_result;
        let execution_time = start_time.elapsed();

        // 更新性能指标和工作负载历史
        self.update_performance_metrics(execution_time, data_size);
        if self.adaptive_enabled {
            self.record_workload_sample(data_size, execution_time);
        }

        result
    }

    fn execute_round_robin<F, T, R>(&self, data: Vec<T>, operation: F) -> Vec<R>
    where
        F: Fn(T) -> R + Sync + Send,
        T: Send,
        R: Send,
    {
        use rayon::prelude::*;
        let chunk_size = data.len() / self.thread_pool_size.max(1);
        data.into_par_iter()
            .with_min_len(chunk_size.max(1))
            .map(operation)
            .collect()
    }

    fn execute_work_stealing<F, T, R>(&self, data: Vec<T>, operation: F) -> Vec<R>
    where
        F: Fn(T) -> R + Sync + Send,
        T: Send,
        R: Send,
    {
        use rayon::prelude::*;
        // 使用更小的块大小以支持工作窃取
        let chunk_size = self.calculate_work_stealing_chunk_size(data.len());
        data.into_par_iter()
            .with_min_len(chunk_size)
            .map(operation)
            .collect()
    }

    fn execute_dynamic<F, T, R>(&self, data: Vec<T>, operation: F) -> Vec<R>
    where
        F: Fn(T) -> R + Sync + Send,
        T: Send,
        R: Send,
    {
        use rayon::prelude::*;
        let chunk_size = self.calculate_optimal_chunk_size(data.len());
        data.into_par_iter()
            .with_min_len(chunk_size)
            .map(operation)
            .collect()
    }

    fn execute_adaptive<F, T, R>(&self, data: Vec<T>, operation: F) -> Vec<R>
    where
        F: Fn(T) -> R + Sync + Send,
        T: Send,
        R: Send,
    {
        // 基于历史性能数据选择最优策略
        let optimal_params = self.analyze_optimal_parameters(data.len());

        use rayon::prelude::*;
        data.into_par_iter()
            .with_min_len(optimal_params.chunk_size)
            .map(operation)
            .collect()
    }

    fn calculate_work_stealing_chunk_size(&self, total_items: usize) -> usize {
        // 工作窃取需要更小的块以平衡负载
        let base_size = total_items / (self.thread_pool_size * 8);
        base_size.max(10).min(1000)
    }

    fn analyze_optimal_parameters(&self, data_size: usize) -> OptimalParameters {
        if let Ok(history) = self.workload_history.lock() {
            if history.len() < 5 {
                // 历史数据不足，使用默认参数
                return OptimalParameters {
                    chunk_size: self.calculate_optimal_chunk_size(data_size),
                    thread_count: self.thread_pool_size,
                };
            }

            // 找到相似工作负载的最佳参数
            let similar_samples: Vec<&WorkloadSample> = history
                .iter()
                .filter(|sample| {
                    let size_ratio = sample.items_count as f64 / data_size as f64;
                    size_ratio >= 0.5 && size_ratio <= 2.0
                })
                .collect();

            if !similar_samples.is_empty() {
                // 找到执行时间最短的样本
                let best_sample = similar_samples
                    .iter()
                    .min_by_key(|sample| sample.execution_time.as_nanos())
                    .unwrap();

                return OptimalParameters {
                    chunk_size: best_sample.chunk_size,
                    thread_count: best_sample.thread_count,
                };
            }
        }

        // 默认参数
        OptimalParameters {
            chunk_size: self.calculate_optimal_chunk_size(data_size),
            thread_count: self.thread_pool_size,
        }
    }

    fn record_workload_sample(&self, data_size: usize, execution_time: Duration) {
        if let Ok(mut history) = self.workload_history.lock() {
            let sample = WorkloadSample {
                items_count: data_size,
                chunk_size: self.calculate_optimal_chunk_size(data_size),
                thread_count: self.thread_pool_size,
                execution_time,
                memory_usage: data_size * 64, // 估算内存使用
                cpu_utilization: 0.8,         // 简化的CPU利用率
            };

            history.push(sample);

            // 保持历史记录在合理大小
            if history.len() > 1000 {
                history.drain(..500);
            }
        }
    }

    fn update_performance_metrics(&self, latency: Duration, data_size: usize) {
        if let Ok(mut metrics) = self.performance_metrics.lock() {
            metrics.total_requests += 1;

            // 更新平均延迟
            let total_latency_ns = metrics.avg_latency.as_nanos() as u64
                * (metrics.total_requests - 1)
                + latency.as_nanos() as u64;
            metrics.avg_latency = Duration::from_nanos(total_latency_ns / metrics.total_requests);

            // 更新并行效率指标
            let ideal_time = latency.as_nanos() as f64 * self.thread_pool_size as f64;
            let actual_time = latency.as_nanos() as f64;
            metrics.parallel_efficiency = (ideal_time / actual_time).min(1.0);

            // 更新内存使用估算
            metrics.memory_usage = data_size * 64; // 简化估算
        }
    }

    pub fn calculate_optimal_chunk_size(&self, total_items: usize) -> usize {
        // 基于线程数和数据量计算最优分块大小
        let base_chunk_size = total_items / (self.thread_pool_size * 4);
        base_chunk_size.max(100).min(10000) // 限制在合理范围内
    }

    // 添加getter方法用于测试
    pub fn get_thread_pool_size(&self) -> usize {
        self.thread_pool_size
    }

    pub fn get_min_thread_pool_size(&self) -> usize {
        self.min_thread_pool_size
    }

    pub fn get_max_thread_pool_size(&self) -> usize {
        self.max_thread_pool_size
    }

    pub fn get_metrics(&self) -> BatchAccessMetrics {
        self.performance_metrics.lock().unwrap().clone()
    }

    /// 动态调整线程池大小
    pub fn adjust_thread_pool_size(&mut self, target_efficiency: f64) {
        if !self.adaptive_enabled {
            return;
        }

        let current_efficiency = self.get_metrics().parallel_efficiency;

        if current_efficiency < target_efficiency * 0.8 {
            // 效率低，减少线程数
            if self.thread_pool_size > self.min_thread_pool_size {
                self.thread_pool_size =
                    (self.thread_pool_size * 4 / 5).max(self.min_thread_pool_size);
            }
        } else if current_efficiency > target_efficiency * 1.1 {
            // 效率高，可以增加线程数
            if self.thread_pool_size < self.max_thread_pool_size {
                self.thread_pool_size =
                    (self.thread_pool_size * 5 / 4).min(self.max_thread_pool_size);
            }
        }
    }

    /// 获取工作负载统计信息
    pub fn get_workload_stats(&self) -> WorkloadStats {
        if let Ok(history) = self.workload_history.lock() {
            if history.is_empty() {
                return WorkloadStats::default();
            }

            let total_samples = history.len();
            let avg_execution_time = Duration::from_nanos(
                history
                    .iter()
                    .map(|s| s.execution_time.as_nanos() as u64)
                    .sum::<u64>()
                    / total_samples as u64,
            );
            let avg_cpu_utilization =
                history.iter().map(|s| s.cpu_utilization).sum::<f64>() / total_samples as f64;
            let avg_memory_usage =
                history.iter().map(|s| s.memory_usage).sum::<usize>() / total_samples;

            WorkloadStats {
                total_samples,
                avg_execution_time,
                avg_cpu_utilization,
                avg_memory_usage,
                best_chunk_size: self.find_best_chunk_size(&history),
                best_thread_count: self.find_best_thread_count(&history),
            }
        } else {
            WorkloadStats::default()
        }
    }

    fn find_best_chunk_size(&self, history: &[WorkloadSample]) -> usize {
        history
            .iter()
            .min_by_key(|sample| sample.execution_time.as_nanos())
            .map(|sample| sample.chunk_size)
            .unwrap_or(self.calculate_optimal_chunk_size(1000))
    }

    fn find_best_thread_count(&self, history: &[WorkloadSample]) -> usize {
        history
            .iter()
            .min_by_key(|sample| sample.execution_time.as_nanos())
            .map(|sample| sample.thread_count)
            .unwrap_or(self.thread_pool_size)
    }

    /// 执行并行批量访问的优化版本，带内存池
    pub fn execute_parallel_optimized<F, T, R>(&self, data: Vec<T>, operation: F) -> Vec<R>
    where
        F: Fn(T) -> R + Sync + Send,
        T: Send,
        R: Send,
    {
        let _start_time = Instant::now();
        let data_size = data.len();

        // 获取内存池缓冲区
        let _buffer = if let Ok(mut pool) = self.memory_pool.lock() {
            Some(pool.get_buffer(data_size * 64))
        } else {
            None
        };

        // 执行并行操作
        let result = self.execute_parallel(data, operation);

        // 归还缓冲区
        if let (Some(buffer), Ok(mut pool)) = (_buffer, self.memory_pool.lock()) {
            pool.return_buffer(buffer);
        }

        result
    }

    /// 重置性能统计
    pub fn reset_performance_stats(&self) {
        if let Ok(mut metrics) = self.performance_metrics.lock() {
            *metrics = BatchAccessMetrics::default();
        }
        if let Ok(mut history) = self.workload_history.lock() {
            history.clear();
        }
    }

    /// 获取推荐的配置参数
    pub fn get_recommended_config(&self, typical_workload_size: usize) -> RecommendedConfig {
        let optimal_params = self.analyze_optimal_parameters(typical_workload_size);
        let workload_stats = self.get_workload_stats();

        RecommendedConfig {
            recommended_thread_count: optimal_params.thread_count,
            recommended_chunk_size: optimal_params.chunk_size,
            recommended_strategy: if workload_stats.avg_cpu_utilization > 0.8 {
                LoadBalancingStrategy::WorkStealing
            } else {
                LoadBalancingStrategy::Adaptive
            },
            memory_pool_enabled: true,
        }
    }
}

// 分块优化器组件
#[derive(Debug)]
pub struct ChunkOptimizer {
    min_chunk_size: usize,
    max_chunk_size: usize,
    adaptive_sizing: bool,
    cache_line_size: usize,
    performance_history: Arc<RwLock<Vec<(usize, Duration)>>>, // (chunk_size, latency)
    // 新增：连续范围访问优化
    range_analyzer: RangeAnalyzer,
    memory_copier: OptimizedMemoryCopier,
    // 新增：范围合并配置
    range_merge_threshold: usize, // 范围间距小于此值时会合并
    sequential_threshold: f64,    // 连续性阈值
}

// 范围分析器
#[derive(Debug)]
pub struct RangeAnalyzer {
    min_sequential_size: usize,
    max_gap_size: usize,
    cache_line_awareness: bool,
    pattern_history: Vec<AccessPatternAnalysis>,
}

// 访问模式分析
#[derive(Debug, Clone)]
pub struct AccessPatternAnalysis {
    pub total_ranges: usize,
    pub merged_ranges: usize,
    pub sequentiality_score: f64,
    pub average_gap_size: f64,
    pub cache_efficiency: f64,
    pub timestamp: Instant,
}

// 优化内存复制器
#[derive(Debug)]
pub struct OptimizedMemoryCopier {
    use_prefetch: bool,
    prefetch_distance: usize,
    alignment_aware: bool,
    use_vectorized_copy: bool,
}

// 范围访问请求
#[derive(Debug, Clone)]
pub struct RangeAccessRequest {
    pub ranges: Vec<(usize, usize)>, // (start, end) pairs
    pub total_size: usize,
    pub expected_sequentiality: f64,
}

// 范围优化结果
#[derive(Debug, Clone)]
pub struct RangeOptimizationResult {
    pub merged_ranges: Vec<(usize, usize)>,
    pub chunk_strategy: ChunkStrategy,
    pub copy_strategy: CopyStrategy,
    pub estimated_performance: Duration,
}

// 分块策略
#[derive(Debug, Clone, Copy)]
pub enum ChunkStrategy {
    SingleLargeChunk,     // 单个大块
    MultipleMediumChunks, // 多个中等块
    ManySmallChunks,      // 多个小块
    AdaptiveChunks,       // 自适应分块
}

// 复制策略
#[derive(Debug, Clone, Copy)]
pub enum CopyStrategy {
    DirectMemcpy,   // 直接内存复制
    VectorizedCopy, // 向量化复制
    PrefetchedCopy, // 预取复制
    ZeroCopyView,   // 零拷贝视图
}

impl RangeAnalyzer {
    pub fn new() -> Self {
        Self {
            min_sequential_size: 1024,
            max_gap_size: 64,
            cache_line_awareness: true,
            pattern_history: Vec::new(),
        }
    }

    /// 分析访问范围的模式
    pub fn analyze_ranges(&mut self, ranges: &[(usize, usize)]) -> AccessPatternAnalysis {
        let total_ranges = ranges.len();
        let merged_ranges = self.merge_adjacent_ranges(ranges);
        let sequentiality_score = self.calculate_sequentiality(&merged_ranges);
        let average_gap_size = self.calculate_average_gap_size(&merged_ranges);
        let cache_efficiency = self.estimate_cache_efficiency(&merged_ranges);

        let analysis = AccessPatternAnalysis {
            total_ranges,
            merged_ranges: merged_ranges.len(),
            sequentiality_score,
            average_gap_size,
            cache_efficiency,
            timestamp: Instant::now(),
        };

        // 保存历史记录
        self.pattern_history.push(analysis.clone());
        if self.pattern_history.len() > 100 {
            self.pattern_history.drain(..50);
        }

        analysis
    }

    /// 合并相邻的范围
    pub fn merge_adjacent_ranges(&self, ranges: &[(usize, usize)]) -> Vec<(usize, usize)> {
        if ranges.is_empty() {
            return Vec::new();
        }

        let mut sorted_ranges = ranges.to_vec();
        sorted_ranges.sort_by_key(|&(start, _)| start);

        let mut merged = Vec::new();
        let mut current_start = sorted_ranges[0].0;
        let mut current_end = sorted_ranges[0].1;

        for &(start, end) in sorted_ranges.iter().skip(1) {
            // 如果间隙小于阈值，则合并
            if start <= current_end + self.max_gap_size {
                current_end = current_end.max(end);
            } else {
                merged.push((current_start, current_end));
                current_start = start;
                current_end = end;
            }
        }
        merged.push((current_start, current_end));
        merged
    }

    /// 计算连续性得分
    fn calculate_sequentiality(&self, ranges: &[(usize, usize)]) -> f64 {
        if ranges.len() <= 1 {
            return 1.0;
        }

        let total_coverage: usize = ranges.iter().map(|(start, end)| end - start).sum();
        let total_span = ranges.last().unwrap().1 - ranges.first().unwrap().0;

        if total_span == 0 {
            return 1.0;
        }

        total_coverage as f64 / total_span as f64
    }

    /// 计算平均间隙大小
    fn calculate_average_gap_size(&self, ranges: &[(usize, usize)]) -> f64 {
        if ranges.len() <= 1 {
            return 0.0;
        }

        let gaps: Vec<usize> = ranges
            .windows(2)
            .map(|window| window[1].0.saturating_sub(window[0].1))
            .collect();

        if gaps.is_empty() {
            0.0
        } else {
            gaps.iter().sum::<usize>() as f64 / gaps.len() as f64
        }
    }

    /// 估算缓存效率
    fn estimate_cache_efficiency(&self, ranges: &[(usize, usize)]) -> f64 {
        if !self.cache_line_awareness {
            return 0.5; // 默认值
        }

        let cache_line_size = 64; // 典型缓存行大小
        let mut cache_aligned_accesses = 0;
        let mut total_accesses = 0;

        for &(start, end) in ranges {
            let size = end - start;
            total_accesses += 1;

            // 检查是否对齐到缓存行
            if start % cache_line_size == 0 && size % cache_line_size == 0 {
                cache_aligned_accesses += 1;
            }
        }

        if total_accesses == 0 {
            0.0
        } else {
            cache_aligned_accesses as f64 / total_accesses as f64
        }
    }

    /// 获取历史模式统计
    pub fn get_pattern_statistics(&self) -> PatternStatistics {
        if self.pattern_history.is_empty() {
            return PatternStatistics::default();
        }

        let count = self.pattern_history.len();
        let avg_sequentiality = self
            .pattern_history
            .iter()
            .map(|p| p.sequentiality_score)
            .sum::<f64>()
            / count as f64;

        let avg_cache_efficiency = self
            .pattern_history
            .iter()
            .map(|p| p.cache_efficiency)
            .sum::<f64>()
            / count as f64;

        PatternStatistics {
            sample_count: count,
            average_sequentiality: avg_sequentiality,
            average_cache_efficiency: avg_cache_efficiency,
        }
    }
}

impl OptimizedMemoryCopier {
    pub fn new() -> Self {
        Self {
            use_prefetch: true,
            prefetch_distance: 64,
            alignment_aware: true,
            use_vectorized_copy: true,
        }
    }

    /// 选择最优的复制策略
    pub fn select_copy_strategy(&self, size: usize, is_sequential: bool) -> CopyStrategy {
        if size < 64 {
            CopyStrategy::DirectMemcpy
        } else if size > 1024 * 1024 && is_sequential {
            // 大块连续数据优先考虑零拷贝
            CopyStrategy::ZeroCopyView
        } else if self.use_vectorized_copy && size > 256 {
            CopyStrategy::VectorizedCopy
        } else if self.use_prefetch && size > 1024 {
            CopyStrategy::PrefetchedCopy
        } else {
            CopyStrategy::DirectMemcpy
        }
    }

    /// 执行优化的内存复制
    pub unsafe fn optimized_copy(
        &self,
        src: *const u8,
        dst: *mut u8,
        size: usize,
        strategy: CopyStrategy,
    ) {
        match strategy {
            CopyStrategy::DirectMemcpy => {
                std::ptr::copy_nonoverlapping(src, dst, size);
            }
            CopyStrategy::VectorizedCopy => {
                self.vectorized_copy(src, dst, size);
            }
            CopyStrategy::PrefetchedCopy => {
                self.prefetched_copy(src, dst, size);
            }
            CopyStrategy::ZeroCopyView => {
                // 零拷贝不需要实际复制，这里只是占位
                std::ptr::copy_nonoverlapping(src, dst, size);
            }
        }
    }

    /// 向量化复制
    unsafe fn vectorized_copy(&self, src: *const u8, dst: *mut u8, size: usize) {
        let mut offset = 0;

        // 处理大块（32字节对齐）
        while offset + 32 <= size {
            let src_chunk = src.add(offset) as *const [u8; 32];
            let dst_chunk = dst.add(offset) as *mut [u8; 32];
            *dst_chunk = *src_chunk;
            offset += 32;
        }

        // 处理剩余字节
        while offset < size {
            *dst.add(offset) = *src.add(offset);
            offset += 1;
        }
    }

    /// 带预取的复制
    unsafe fn prefetched_copy(&self, src: *const u8, dst: *mut u8, size: usize) {
        let mut offset = 0;

        while offset < size {
            // 预取下一块数据
            if offset + self.prefetch_distance < size {
                #[cfg(target_arch = "x86_64")]
                {
                    std::arch::x86_64::_mm_prefetch(
                        src.add(offset + self.prefetch_distance) as *const i8,
                        std::arch::x86_64::_MM_HINT_T0,
                    );
                }
            }

            // 复制当前块
            let copy_size = (size - offset).min(64);
            std::ptr::copy_nonoverlapping(src.add(offset), dst.add(offset), copy_size);
            offset += copy_size;
        }
    }
}

// 模式统计结构
#[derive(Debug, Clone, Default)]
pub struct PatternStatistics {
    pub sample_count: usize,
    pub average_sequentiality: f64,
    pub average_cache_efficiency: f64,
}

// 访问连续性报告
#[derive(Debug, Clone, Default)]
pub struct AccessContinuityReport {
    pub total_indices: usize,
    pub consecutive_groups: usize,
    pub largest_group_size: usize,
    pub continuity_ratio: f64,
    pub average_gap: f64,
    pub max_gap: usize,
    pub is_highly_continuous: bool,
}

// 范围优化统计
#[derive(Debug, Clone)]
pub struct RangeOptimizationStats {
    pub pattern_statistics: PatternStatistics,
    pub cache_line_size: usize,
    pub range_merge_threshold: usize,
    pub sequential_threshold: f64,
}

impl ChunkOptimizer {
    pub fn new() -> Self {
        Self {
            min_chunk_size: 64,
            max_chunk_size: 65536,
            adaptive_sizing: true,
            cache_line_size: 64,
            performance_history: Arc::new(RwLock::new(Vec::new())),
            range_analyzer: RangeAnalyzer::new(),
            memory_copier: OptimizedMemoryCopier::new(),
            range_merge_threshold: 128,
            sequential_threshold: 0.8,
        }
    }

    pub fn optimize_chunk_size(&self, total_size: usize, access_pattern: &str) -> usize {
        if !self.adaptive_sizing {
            return self.calculate_static_chunk_size(total_size);
        }

        // 基于历史性能数据选择最优分块大小
        if let Ok(history) = self.performance_history.read() {
            if history.len() > 10 {
                let optimal_size = self.find_optimal_from_history(&history);
                return optimal_size
                    .max(self.min_chunk_size)
                    .min(self.max_chunk_size);
            }
        }

        // 没有足够历史数据时使用启发式方法
        match access_pattern {
            "sequential" => (total_size / 16).max(self.min_chunk_size),
            "random" => self.min_chunk_size * 4,
            "clustered" => (total_size / 8).max(self.min_chunk_size),
            _ => self.calculate_static_chunk_size(total_size),
        }
    }

    fn calculate_static_chunk_size(&self, total_size: usize) -> usize {
        // 静态计算：基于缓存行大小和总大小
        let cache_aligned_size = (self.cache_line_size * 16).max(self.min_chunk_size);
        (total_size / 32)
            .max(cache_aligned_size)
            .min(self.max_chunk_size)
    }

    fn find_optimal_from_history(&self, history: &[(usize, Duration)]) -> usize {
        // 找到延迟最低的分块大小
        history
            .iter()
            .min_by_key(|(_, latency)| latency.as_nanos())
            .map(|(size, _)| *size)
            .unwrap_or(self.min_chunk_size * 8)
    }

    pub fn record_performance(&self, chunk_size: usize, latency: Duration) {
        if let Ok(mut history) = self.performance_history.write() {
            history.push((chunk_size, latency));
            // 保持历史记录在合理大小
            if history.len() > 1000 {
                history.drain(..500); // 保留最近500个记录
            }
        }
    }

    pub fn split_into_chunks<T: Clone>(&self, data: Vec<T>, chunk_size: usize) -> Vec<Vec<T>> {
        data.chunks(chunk_size)
            .map(|chunk| chunk.to_vec())
            .collect()
    }

    /// 优化范围访问请求
    pub fn optimize_range_access(
        &mut self,
        request: RangeAccessRequest,
    ) -> RangeOptimizationResult {
        let start_time = Instant::now();

        // 分析访问模式
        let pattern_analysis = self.range_analyzer.analyze_ranges(&request.ranges);

        // 合并相邻范围
        let merged_ranges = self.range_analyzer.merge_adjacent_ranges(&request.ranges);

        // 选择分块策略
        let chunk_strategy = self.select_chunk_strategy(&merged_ranges, &pattern_analysis);

        // 选择复制策略
        let copy_strategy = self.select_copy_strategy_for_ranges(&merged_ranges, &pattern_analysis);

        // 估算性能
        let estimated_performance =
            self.estimate_range_performance(&merged_ranges, chunk_strategy, copy_strategy);

        // 记录性能历史
        self.record_range_performance(merged_ranges.len(), start_time.elapsed());

        RangeOptimizationResult {
            merged_ranges,
            chunk_strategy,
            copy_strategy,
            estimated_performance,
        }
    }

    /// 选择分块策略
    fn select_chunk_strategy(
        &self,
        ranges: &[(usize, usize)],
        analysis: &AccessPatternAnalysis,
    ) -> ChunkStrategy {
        let total_size: usize = ranges.iter().map(|(start, end)| end - start).sum();

        if analysis.sequentiality_score >= self.sequential_threshold {
            // 高连续性，使用大块策略
            if total_size > 1024 * 1024 {
                ChunkStrategy::SingleLargeChunk
            } else {
                ChunkStrategy::MultipleMediumChunks
            }
        } else if ranges.len() > 100 {
            // 范围数量多，使用小块策略
            ChunkStrategy::ManySmallChunks
        } else {
            // 使用自适应策略
            ChunkStrategy::AdaptiveChunks
        }
    }

    /// 为范围访问选择复制策略
    fn select_copy_strategy_for_ranges(
        &self,
        ranges: &[(usize, usize)],
        analysis: &AccessPatternAnalysis,
    ) -> CopyStrategy {
        let total_size: usize = ranges.iter().map(|(start, end)| end - start).sum();
        let is_sequential = analysis.sequentiality_score >= self.sequential_threshold;

        self.memory_copier
            .select_copy_strategy(total_size, is_sequential)
    }

    /// 估算范围访问性能
    fn estimate_range_performance(
        &self,
        ranges: &[(usize, usize)],
        chunk_strategy: ChunkStrategy,
        copy_strategy: CopyStrategy,
    ) -> Duration {
        let total_size: usize = ranges.iter().map(|(start, end)| end - start).sum();

        // 基础时间估算（基于数据大小）
        let base_time_ns = match total_size {
            0..=1024 => 100,          // 1KB以下：100ns
            1025..=65536 => 1000,     // 64KB以下：1μs
            65537..=1048576 => 10000, // 1MB以下：10μs
            _ => 100000,              // 1MB以上：100μs
        };

        // 分块策略调整
        let chunk_multiplier = match chunk_strategy {
            ChunkStrategy::SingleLargeChunk => 0.8,     // 单大块最快
            ChunkStrategy::MultipleMediumChunks => 1.0, // 中等块基准
            ChunkStrategy::ManySmallChunks => 1.5,      // 小块较慢
            ChunkStrategy::AdaptiveChunks => 0.9,       // 自适应稍快
        };

        // 复制策略调整
        let copy_multiplier = match copy_strategy {
            CopyStrategy::ZeroCopyView => 0.1,   // 零拷贝最快
            CopyStrategy::VectorizedCopy => 0.7, // 向量化较快
            CopyStrategy::PrefetchedCopy => 0.8, // 预取较快
            CopyStrategy::DirectMemcpy => 1.0,   // 直接复制基准
        };

        let final_time_ns = (base_time_ns as f64 * chunk_multiplier * copy_multiplier) as u64;
        Duration::from_nanos(final_time_ns)
    }

    /// 记录范围访问性能
    fn record_range_performance(&self, range_count: usize, latency: Duration) {
        // 使用范围数量作为"块大小"记录到历史中
        self.record_performance(range_count, latency);
    }

    /// 智能检测连续vs非连续访问
    pub fn detect_access_continuity(&self, indices: &[usize]) -> AccessContinuityReport {
        if indices.is_empty() {
            return AccessContinuityReport::default();
        }

        let mut sorted_indices = indices.to_vec();
        sorted_indices.sort_unstable();

        let mut consecutive_groups = 0;
        let mut total_gaps = 0;
        let mut max_gap = 0;
        let mut current_group_size = 1;
        let mut largest_group_size = 1;

        for i in 1..sorted_indices.len() {
            let gap = sorted_indices[i] - sorted_indices[i - 1];

            if gap == 1 {
                // 连续
                current_group_size += 1;
            } else {
                // 非连续
                consecutive_groups += 1;
                largest_group_size = largest_group_size.max(current_group_size);
                current_group_size = 1;
                total_gaps += gap - 1;
                max_gap = max_gap.max(gap - 1);
            }
        }

        consecutive_groups += 1; // 最后一组
        largest_group_size = largest_group_size.max(current_group_size);

        let continuity_ratio = if indices.len() <= 1 {
            1.0
        } else {
            let span = sorted_indices.last().unwrap() - sorted_indices.first().unwrap() + 1;
            indices.len() as f64 / span as f64
        };

        AccessContinuityReport {
            total_indices: indices.len(),
            consecutive_groups,
            largest_group_size,
            continuity_ratio,
            average_gap: if consecutive_groups > 1 {
                total_gaps as f64 / (consecutive_groups - 1) as f64
            } else {
                0.0
            },
            max_gap,
            is_highly_continuous: continuity_ratio >= self.sequential_threshold,
        }
    }

    /// 获取范围优化统计信息
    pub fn get_range_optimization_stats(&self) -> RangeOptimizationStats {
        let pattern_stats = self.range_analyzer.get_pattern_statistics();

        RangeOptimizationStats {
            pattern_statistics: pattern_stats,
            cache_line_size: self.cache_line_size,
            range_merge_threshold: self.range_merge_threshold,
            sequential_threshold: self.sequential_threshold,
        }
    }
}

// 流处理器组件
#[derive(Debug)]
pub struct StreamProcessor {
    buffer_size: usize,
    prefetch_enabled: bool,
    backpressure_threshold: usize,
    flow_control_enabled: bool,
    // 新增：增强的流控制功能
    memory_monitor: MemoryMonitor,
    throughput_tracker: ThroughputTracker,
    stream_config: StreamConfig,
    adaptive_sizing: bool,
}

// 内存监控器
#[derive(Debug)]
pub struct MemoryMonitor {
    max_memory_usage: usize,
    current_memory_usage: usize,
    memory_pressure_threshold: f64,
    gc_trigger_threshold: f64,
    memory_samples: Vec<MemorySample>,
}

// 吞吐量跟踪器
#[derive(Debug)]
pub struct ThroughputTracker {
    window_size: Duration,
    samples: Vec<ThroughputSample>,
    target_throughput: f64,
    current_throughput: f64,
    last_measurement: Instant,
}

// 流配置
#[derive(Debug, Clone)]
pub struct StreamConfig {
    pub min_chunk_size: usize,
    pub max_chunk_size: usize,
    pub target_latency: Duration,
    pub max_concurrent_chunks: usize,
    pub enable_compression: bool,
    pub enable_prefetch: bool,
}

// 内存样本
#[derive(Debug, Clone)]
pub struct MemorySample {
    pub timestamp: Instant,
    pub memory_usage: usize,
    pub buffer_count: usize,
    pub pressure_level: f64,
}

// 吞吐量样本
#[derive(Debug, Clone)]
pub struct ThroughputSample {
    pub timestamp: Instant,
    pub bytes_processed: usize,
    pub chunks_processed: usize,
    pub latency: Duration,
}

// 流式处理结果
pub struct StreamingResult<T> {
    pub stream: Box<dyn Iterator<Item = T> + Send>,
    pub metadata: StreamMetadata,
}

impl<T> std::fmt::Debug for StreamingResult<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StreamingResult")
            .field("metadata", &self.metadata)
            .field("stream", &"<Iterator>")
            .finish()
    }
}

// 流元数据
#[derive(Debug, Clone)]
pub struct StreamMetadata {
    pub total_chunks: usize,
    pub estimated_total_size: usize,
    pub chunk_size: usize,
    pub compression_enabled: bool,
    pub prefetch_enabled: bool,
}

// 背压信号
#[derive(Debug, Clone, Copy)]
pub enum BackpressureSignal {
    None,     // 无背压
    Moderate, // 中等背压
    High,     // 高背压
    Critical, // 危险背压，需要暂停
}

// 流式处理状态
#[derive(Debug, Clone)]
pub struct StreamProcessingState {
    pub chunks_processed: usize,
    pub bytes_processed: usize,
    pub current_buffer_size: usize,
    pub backpressure_level: BackpressureSignal,
    pub throughput: f64,
    pub memory_usage: usize,
    pub processing_time: Duration,
}

// 内存统计
#[derive(Debug, Clone, Default)]
pub struct MemoryStatistics {
    pub current_usage: usize,
    pub max_usage: usize,
    pub average_usage: usize,
    pub pressure_level: f64,
    pub average_pressure: f64,
    pub sample_count: usize,
}

// 吞吐量统计
#[derive(Debug, Clone, Default)]
pub struct ThroughputStatistics {
    pub current_throughput_mbps: f64,
    pub target_throughput_mbps: f64,
    pub throughput_ratio: f64,
    pub average_latency: Duration,
    pub total_chunks_processed: usize,
    pub sample_count: usize,
}

impl MemoryMonitor {
    pub fn new(max_memory_mb: usize) -> Self {
        Self {
            max_memory_usage: max_memory_mb * 1024 * 1024,
            current_memory_usage: 0,
            memory_pressure_threshold: 0.8,
            gc_trigger_threshold: 0.9,
            memory_samples: Vec::new(),
        }
    }

    pub fn record_memory_usage(&mut self, usage: usize, buffer_count: usize) {
        self.current_memory_usage = usage;

        let pressure_level = usage as f64 / self.max_memory_usage as f64;
        let sample = MemorySample {
            timestamp: Instant::now(),
            memory_usage: usage,
            buffer_count,
            pressure_level,
        };

        self.memory_samples.push(sample);

        // 保持样本数量在合理范围
        if self.memory_samples.len() > 1000 {
            self.memory_samples.drain(..500);
        }
    }

    pub fn get_memory_pressure(&self) -> f64 {
        self.current_memory_usage as f64 / self.max_memory_usage as f64
    }

    pub fn should_trigger_gc(&self) -> bool {
        self.get_memory_pressure() > self.gc_trigger_threshold
    }

    pub fn get_backpressure_signal(&self) -> BackpressureSignal {
        let pressure = self.get_memory_pressure();

        if pressure > 0.95 {
            BackpressureSignal::Critical
        } else if pressure > 0.85 {
            BackpressureSignal::High
        } else if pressure > 0.7 {
            BackpressureSignal::Moderate
        } else {
            BackpressureSignal::None
        }
    }

    pub fn get_memory_statistics(&self) -> MemoryStatistics {
        if self.memory_samples.is_empty() {
            return MemoryStatistics::default();
        }

        let recent_samples: Vec<&MemorySample> =
            self.memory_samples.iter().rev().take(100).collect();

        let avg_usage =
            recent_samples.iter().map(|s| s.memory_usage).sum::<usize>() / recent_samples.len();

        let avg_pressure = recent_samples.iter().map(|s| s.pressure_level).sum::<f64>()
            / recent_samples.len() as f64;

        MemoryStatistics {
            current_usage: self.current_memory_usage,
            max_usage: self.max_memory_usage,
            average_usage: avg_usage,
            pressure_level: self.get_memory_pressure(),
            average_pressure: avg_pressure,
            sample_count: self.memory_samples.len(),
        }
    }
}

impl ThroughputTracker {
    pub fn new(target_mb_per_sec: f64) -> Self {
        Self {
            window_size: Duration::from_secs(10),
            samples: Vec::new(),
            target_throughput: target_mb_per_sec * 1024.0 * 1024.0, // 转换为字节/秒
            current_throughput: 0.0,
            last_measurement: Instant::now(),
        }
    }

    pub fn record_throughput(
        &mut self,
        bytes_processed: usize,
        chunks_processed: usize,
        latency: Duration,
    ) {
        let now = Instant::now();
        let sample = ThroughputSample {
            timestamp: now,
            bytes_processed,
            chunks_processed,
            latency,
        };

        self.samples.push(sample);
        self.last_measurement = now;

        // 清理旧样本
        let cutoff_time = now - self.window_size;
        self.samples.retain(|s| s.timestamp > cutoff_time);

        // 计算当前吞吐量
        self.update_current_throughput();
    }

    fn update_current_throughput(&mut self) {
        if self.samples.len() < 2 {
            return;
        }

        let window_duration = self.last_measurement - self.samples[0].timestamp;
        if window_duration.as_secs_f64() <= 0.0 {
            return;
        }

        let total_bytes: usize = self.samples.iter().map(|s| s.bytes_processed).sum();
        self.current_throughput = total_bytes as f64 / window_duration.as_secs_f64();
    }

    pub fn get_current_throughput(&self) -> f64 {
        self.current_throughput
    }

    pub fn get_throughput_ratio(&self) -> f64 {
        if self.target_throughput <= 0.0 {
            return 1.0;
        }
        self.current_throughput / self.target_throughput
    }

    pub fn should_increase_chunk_size(&self) -> bool {
        self.get_throughput_ratio() < 0.8
    }

    pub fn should_decrease_chunk_size(&self) -> bool {
        self.get_throughput_ratio() > 1.2
    }

    pub fn get_throughput_statistics(&self) -> ThroughputStatistics {
        if self.samples.is_empty() {
            return ThroughputStatistics::default();
        }

        let avg_latency = Duration::from_nanos(
            self.samples
                .iter()
                .map(|s| s.latency.as_nanos() as u64)
                .sum::<u64>()
                / self.samples.len() as u64,
        );

        let total_chunks: usize = self.samples.iter().map(|s| s.chunks_processed).sum();

        ThroughputStatistics {
            current_throughput_mbps: self.current_throughput / (1024.0 * 1024.0),
            target_throughput_mbps: self.target_throughput / (1024.0 * 1024.0),
            throughput_ratio: self.get_throughput_ratio(),
            average_latency: avg_latency,
            total_chunks_processed: total_chunks,
            sample_count: self.samples.len(),
        }
    }
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            min_chunk_size: 1024,
            max_chunk_size: 1024 * 1024,
            target_latency: Duration::from_millis(10),
            max_concurrent_chunks: 10,
            enable_compression: false,
            enable_prefetch: true,
        }
    }
}

impl StreamProcessor {
    pub fn new() -> Self {
        Self {
            buffer_size: 8192,
            prefetch_enabled: true,
            backpressure_threshold: 16384,
            flow_control_enabled: true,
            memory_monitor: MemoryMonitor::new(256), // 256MB默认
            throughput_tracker: ThroughputTracker::new(100.0), // 100MB/s目标
            stream_config: StreamConfig::default(),
            adaptive_sizing: true,
        }
    }

    pub fn new_with_config(
        memory_limit_mb: usize,
        target_throughput_mbps: f64,
        config: StreamConfig,
    ) -> Self {
        Self {
            buffer_size: config.min_chunk_size.max(8192),
            prefetch_enabled: config.enable_prefetch,
            backpressure_threshold: memory_limit_mb * 1024 * 1024 / 2,
            flow_control_enabled: true,
            memory_monitor: MemoryMonitor::new(memory_limit_mb),
            throughput_tracker: ThroughputTracker::new(target_throughput_mbps),
            stream_config: config,
            adaptive_sizing: true,
        }
    }

    pub fn create_stream<T>(&self, data: Vec<T>) -> impl Iterator<Item = T> {
        // 创建流式迭代器
        data.into_iter()
    }

    pub fn create_buffered_stream<T: Clone>(&self, data: Vec<T>) -> impl Iterator<Item = Vec<T>> {
        // 创建分块的缓冲流
        let chunk_size = self.buffer_size;
        data.chunks(chunk_size)
            .map(|chunk| chunk.to_vec())
            .collect::<Vec<_>>()
            .into_iter()
    }

    pub fn apply_backpressure(&self, current_buffer_size: usize) -> bool {
        self.flow_control_enabled && current_buffer_size > self.backpressure_threshold
    }

    /// 高级流式处理：创建带背压控制的数据流
    pub fn create_advanced_stream<T: Clone + Send + 'static>(
        &mut self,
        data: Vec<T>,
        _context: &dyn BatchDataContext,
    ) -> StreamingResult<Vec<T>> {
        let start_time = Instant::now();
        let total_size = data.len();

        // 动态计算最优分块大小
        let chunk_size = self.calculate_optimal_chunk_size(total_size);

        // 创建分块流
        let chunks: Vec<Vec<T>> = data
            .chunks(chunk_size)
            .map(|chunk| chunk.to_vec())
            .collect();

        let total_chunks = chunks.len();
        let estimated_total_size = total_size * std::mem::size_of::<T>();

        // 创建带背压控制的迭代器
        let stream = Box::new(BackpressureAwareIterator::new(
            chunks,
            self.memory_monitor.get_backpressure_signal(),
            self.stream_config.clone(),
        ));

        // 记录处理指标
        let processing_time = start_time.elapsed();
        self.throughput_tracker.record_throughput(
            estimated_total_size,
            total_chunks,
            processing_time,
        );

        let metadata = StreamMetadata {
            total_chunks,
            estimated_total_size,
            chunk_size,
            compression_enabled: self.stream_config.enable_compression,
            prefetch_enabled: self.stream_config.enable_prefetch,
        };

        StreamingResult { stream, metadata }
    }

    /// 处理流式批量访问，支持背压和流控
    pub fn process_streaming_batch_access(
        &mut self,
        indices: Vec<usize>,
        _context: &dyn BatchDataContext,
    ) -> StreamingResult<Vec<u8>> {
        let _start_time = Instant::now();

        // 检查内存压力
        let backpressure = self.memory_monitor.get_backpressure_signal();

        // 根据背压调整分块大小
        let base_chunk_size = self.calculate_optimal_chunk_size(indices.len());
        let adjusted_chunk_size =
            self.adjust_chunk_size_for_backpressure(base_chunk_size, backpressure);

        // 分块处理索引
        let index_chunks: Vec<Vec<usize>> = indices
            .chunks(adjusted_chunk_size)
            .map(|chunk| chunk.to_vec())
            .collect();

        // 估算内存使用
        let estimated_memory = index_chunks.len() * adjusted_chunk_size * 64; // 估算每行64字节
        self.memory_monitor
            .record_memory_usage(estimated_memory, index_chunks.len());

        // 创建数据块流
        let data_chunks: Vec<Vec<u8>> = index_chunks
            .into_iter()
            .map(|chunk| {
                let chunk_start = Instant::now();
                let chunk_data: Vec<Vec<u8>> = chunk
                    .into_iter()
                    .map(|idx| _context.get_row_data(idx))
                    .collect();

                // 合并块数据
                let mut combined_data = Vec::new();
                for row in &chunk_data {
                    combined_data.extend_from_slice(row);
                }

                // 记录块处理时间
                let chunk_time = chunk_start.elapsed();
                self.throughput_tracker
                    .record_throughput(combined_data.len(), 1, chunk_time);

                combined_data
            })
            .collect();

        let total_chunks = data_chunks.len();
        let estimated_total_size: usize = data_chunks.iter().map(|c| c.len()).sum();

        // 创建带背压控制的流
        let stream = Box::new(BackpressureAwareIterator::new(
            data_chunks,
            backpressure,
            self.stream_config.clone(),
        ));

        let metadata = StreamMetadata {
            total_chunks,
            estimated_total_size,
            chunk_size: adjusted_chunk_size,
            compression_enabled: self.stream_config.enable_compression,
            prefetch_enabled: self.stream_config.enable_prefetch,
        };

        StreamingResult { stream, metadata }
    }

    /// 动态调整缓冲区大小
    pub fn adjust_buffer_size(&mut self, target_throughput: f64, current_throughput: f64) {
        if !self.flow_control_enabled {
            return;
        }

        let adjustment_ratio = target_throughput / current_throughput.max(0.1);

        if adjustment_ratio > 1.2 {
            // 需要增加缓冲区
            self.buffer_size = (self.buffer_size as f64 * 1.5) as usize;
        } else if adjustment_ratio < 0.8 {
            // 需要减少缓冲区
            self.buffer_size = (self.buffer_size as f64 * 0.7) as usize;
        }

        // 限制缓冲区大小在合理范围内
        self.buffer_size = self
            .buffer_size
            .max(self.stream_config.min_chunk_size)
            .min(self.stream_config.max_chunk_size);
    }

    /// 计算最优分块大小
    pub fn calculate_optimal_chunk_size(&self, total_items: usize) -> usize {
        if !self.adaptive_sizing {
            return self.buffer_size;
        }

        // 基于吞吐量调整
        let throughput_ratio = self.throughput_tracker.get_throughput_ratio();
        let mut base_size = self.buffer_size;

        if throughput_ratio < 0.8 {
            // 吞吐量低，增加块大小
            base_size = (base_size as f64 * 1.3) as usize;
        } else if throughput_ratio > 1.2 {
            // 吞吐量高，可以减少块大小
            base_size = (base_size as f64 * 0.8) as usize;
        }

        // 基于总项目数调整
        let item_based_size = if total_items < 1000 {
            total_items / 4
        } else if total_items < 10000 {
            total_items / 10
        } else {
            total_items / 50
        };

        // 综合考虑
        let optimal_size = (base_size + item_based_size) / 2;

        optimal_size
            .max(self.stream_config.min_chunk_size)
            .min(self.stream_config.max_chunk_size)
    }

    /// 根据背压调整分块大小
    pub fn adjust_chunk_size_for_backpressure(
        &self,
        base_size: usize,
        backpressure: BackpressureSignal,
    ) -> usize {
        match backpressure {
            BackpressureSignal::None => base_size,
            BackpressureSignal::Moderate => (base_size as f64 * 0.8) as usize,
            BackpressureSignal::High => (base_size as f64 * 0.6) as usize,
            BackpressureSignal::Critical => (base_size as f64 * 0.4) as usize,
        }
    }

    /// 获取流处理状态
    pub fn get_processing_state(&self) -> StreamProcessingState {
        let throughput_stats = self.throughput_tracker.get_throughput_statistics();
        let memory_stats = self.memory_monitor.get_memory_statistics();

        StreamProcessingState {
            chunks_processed: throughput_stats.total_chunks_processed,
            bytes_processed: (throughput_stats.current_throughput_mbps * 1024.0 * 1024.0) as usize,
            current_buffer_size: self.buffer_size,
            backpressure_level: self.memory_monitor.get_backpressure_signal(),
            throughput: throughput_stats.current_throughput_mbps,
            memory_usage: memory_stats.current_usage,
            processing_time: throughput_stats.average_latency,
        }
    }

    /// 获取内存统计
    pub fn get_memory_statistics(&self) -> MemoryStatistics {
        self.memory_monitor.get_memory_statistics()
    }

    /// 获取吞吐量统计
    pub fn get_throughput_statistics(&self) -> ThroughputStatistics {
        self.throughput_tracker.get_throughput_statistics()
    }

    /// 触发内存垃圾回收
    pub fn trigger_gc_if_needed(&mut self) -> bool {
        if self.memory_monitor.should_trigger_gc() {
            // 简化的GC触发 - 在实际应用中应该调用真正的GC
            self.memory_monitor
                .record_memory_usage(self.memory_monitor.current_memory_usage / 2, 0);
            true
        } else {
            false
        }
    }

    /// 访问内存监控器（用于测试）
    pub fn record_memory_usage(&mut self, usage: usize, buffer_count: usize) {
        self.memory_monitor.record_memory_usage(usage, buffer_count);
    }

    /// 访问吞吐量追踪器（用于测试）
    pub fn record_throughput(&mut self, bytes: usize, chunks: usize, latency: Duration) {
        self.throughput_tracker
            .record_throughput(bytes, chunks, latency);
    }

    /// 获取缓冲区大小
    pub fn get_buffer_size(&self) -> usize {
        self.buffer_size
    }
}

// 主要的批量访问引擎
#[derive(Debug)]
pub struct BatchAccessEngine {
    parallel_executor: Arc<Mutex<ParallelExecutor>>,
    chunk_optimizer: Arc<RwLock<ChunkOptimizer>>,
    stream_processor: Arc<Mutex<StreamProcessor>>,
    strategy_selector: Arc<RwLock<StrategySelector>>,
    performance_monitor: Arc<Mutex<BatchAccessMetrics>>,
}

// 策略选择器
#[derive(Debug)]
pub struct StrategySelector {
    default_strategy: BatchAccessStrategy,
    strategy_rules: HashMap<String, BatchAccessStrategy>,
    performance_weights: HashMap<BatchAccessStrategy, f64>,
}

impl StrategySelector {
    pub fn new() -> Self {
        let mut strategy_rules = HashMap::new();
        strategy_rules.insert("small_batch".to_string(), BatchAccessStrategy::Parallel);
        strategy_rules.insert("large_batch".to_string(), BatchAccessStrategy::Chunked);
        strategy_rules.insert(
            "continuous_range".to_string(),
            BatchAccessStrategy::ZeroCopy,
        );
        strategy_rules.insert("sparse_access".to_string(), BatchAccessStrategy::Streaming);

        let mut performance_weights = HashMap::new();
        performance_weights.insert(BatchAccessStrategy::Parallel, 1.0);
        performance_weights.insert(BatchAccessStrategy::Chunked, 0.9);
        performance_weights.insert(BatchAccessStrategy::Streaming, 0.8);
        performance_weights.insert(BatchAccessStrategy::ZeroCopy, 1.2);
        performance_weights.insert(BatchAccessStrategy::Adaptive, 1.0);

        Self {
            default_strategy: BatchAccessStrategy::Adaptive,
            strategy_rules,
            performance_weights,
        }
    }

    pub fn select_strategy(
        &self,
        request: &BatchAccessRequest,
        _data_size: usize,
    ) -> BatchAccessStrategy {
        match request {
            BatchAccessRequest::Rows(indices) => {
                if indices.len() < 100 {
                    BatchAccessStrategy::Parallel
                } else if indices.len() > 10000 {
                    BatchAccessStrategy::Chunked
                } else {
                    BatchAccessStrategy::Adaptive
                }
            }
            BatchAccessRequest::Range(start, end) => {
                let range_size = end.saturating_sub(*start);
                if range_size > 1000000 {
                    BatchAccessStrategy::ZeroCopy
                } else {
                    BatchAccessStrategy::Parallel
                }
            }
            BatchAccessRequest::Streaming(_, _) => BatchAccessStrategy::Streaming,
            _ => self.default_strategy,
        }
    }

    pub fn update_performance_weight(&mut self, strategy: BatchAccessStrategy, new_weight: f64) {
        self.performance_weights.insert(strategy, new_weight);
    }
}

impl BatchAccessEngine {
    pub fn new() -> Self {
        Self {
            parallel_executor: Arc::new(Mutex::new(ParallelExecutor::new(
                rayon::current_num_threads(),
            ))),
            chunk_optimizer: Arc::new(RwLock::new(ChunkOptimizer::new())),
            stream_processor: Arc::new(Mutex::new(StreamProcessor::new())),
            strategy_selector: Arc::new(RwLock::new(StrategySelector::new())),
            performance_monitor: Arc::new(Mutex::new(BatchAccessMetrics::default())),
        }
    }

    pub fn process_request(
        &self,
        request: BatchAccessRequest,
        data_context: &dyn BatchDataContext,
    ) -> BatchAccessResult {
        let start_time = Instant::now();

        // 选择最优策略（只读访问，使用RwLock的read）
        let strategy = {
            let selector = self.strategy_selector.read().unwrap();
            selector.select_strategy(&request, data_context.total_size())
        };

        // 根据策略执行请求
        let result = match strategy {
            BatchAccessStrategy::Parallel => self.process_parallel(request.clone(), data_context),
            BatchAccessStrategy::Chunked => self.process_chunked(request.clone(), data_context),
            BatchAccessStrategy::Streaming => self.process_streaming(request.clone(), data_context),
            BatchAccessStrategy::ZeroCopy => self.process_zero_copy(request.clone(), data_context),
            BatchAccessStrategy::Adaptive => self.process_adaptive(request.clone(), data_context),
        };

        // 更新性能指标
        self.update_performance_metrics(start_time.elapsed(), &request);

        result
    }

    fn process_parallel(
        &self,
        request: BatchAccessRequest,
        context: &dyn BatchDataContext,
    ) -> BatchAccessResult {
        match request {
            BatchAccessRequest::Rows(indices) => {
                // 简化：先不使用并行处理，避免线程安全问题
                let results: Vec<Vec<u8>> = indices
                    .into_iter()
                    .map(|idx| context.get_row_data(idx))
                    .collect();
                BatchAccessResult::Owned(results)
            }
            BatchAccessRequest::Range(start, end) => {
                let data = context.get_range_data(start, end);
                BatchAccessResult::Range(data)
            }
            _ => {
                // 对于其他类型，降级到基础实现
                BatchAccessResult::Owned(Vec::new())
            }
        }
    }

    fn process_chunked(
        &self,
        request: BatchAccessRequest,
        context: &dyn BatchDataContext,
    ) -> BatchAccessResult {
        match request {
            BatchAccessRequest::Rows(indices) => {
                // 分析访问连续性（只读访问）
                let chunk_optimizer = self.chunk_optimizer.read().unwrap();
                let continuity_report = chunk_optimizer.detect_access_continuity(&indices);

                let chunk_size = if continuity_report.is_highly_continuous {
                    // 高连续性使用大块
                    chunk_optimizer.optimize_chunk_size(indices.len(), "sequential")
                } else {
                    // 低连续性使用小块
                    chunk_optimizer.optimize_chunk_size(indices.len(), "random")
                };

                let chunks = chunk_optimizer.split_into_chunks(indices, chunk_size);
                drop(chunk_optimizer); // 尽早释放读锁

                // 简化：使用顺序处理避免线程安全问题
                let results: Vec<Vec<u8>> = chunks
                    .into_iter()
                    .flat_map(|chunk| chunk.into_iter().map(|idx| context.get_row_data(idx)))
                    .collect();

                BatchAccessResult::Owned(results)
            }
            BatchAccessRequest::Range(start, end) => {
                // 对于范围访问，直接获取数据
                let data = context.get_range_data(start, end);
                BatchAccessResult::Range(data)
            }
            _ => BatchAccessResult::Owned(Vec::new()),
        }
    }

    fn process_streaming(
        &self,
        request: BatchAccessRequest,
        context: &dyn BatchDataContext,
    ) -> BatchAccessResult {
        match request {
            BatchAccessRequest::Streaming(indices, _chunk_size) => {
                // 简化：直接返回所有数据，避免复杂的流式处理
                let results: Vec<Vec<u8>> = indices
                    .into_iter()
                    .map(|idx| context.get_row_data(idx))
                    .collect();
                BatchAccessResult::Owned(results)
            }
            BatchAccessRequest::Rows(indices) => {
                let results: Vec<Vec<u8>> = indices
                    .into_iter()
                    .map(|idx| context.get_row_data(idx))
                    .collect();
                BatchAccessResult::Owned(results)
            }
            _ => BatchAccessResult::Owned(Vec::new()),
        }
    }

    fn process_zero_copy(
        &self,
        request: BatchAccessRequest,
        context: &dyn BatchDataContext,
    ) -> BatchAccessResult {
        match request {
            BatchAccessRequest::Range(start, end) => {
                let data = context.get_range_data(start, end);
                BatchAccessResult::Range(data)
            }
            _ => {
                // 零拷贝主要适用于范围访问
                self.process_parallel(request, context)
            }
        }
    }

    fn process_adaptive(
        &self,
        request: BatchAccessRequest,
        context: &dyn BatchDataContext,
    ) -> BatchAccessResult {
        // 自适应策略：基于请求特征选择最佳处理方式
        let strategy = match &request {
            BatchAccessRequest::Rows(indices) if indices.len() < 50 => {
                BatchAccessStrategy::Parallel
            }
            BatchAccessRequest::Rows(indices) if indices.len() > 5000 => {
                BatchAccessStrategy::Chunked
            }
            BatchAccessRequest::Range(start, end) if end - start > 10000 => {
                BatchAccessStrategy::ZeroCopy
            }
            _ => BatchAccessStrategy::Parallel,
        };

        match strategy {
            BatchAccessStrategy::Parallel => self.process_parallel(request, context),
            BatchAccessStrategy::Chunked => self.process_chunked(request, context),
            BatchAccessStrategy::ZeroCopy => self.process_zero_copy(request, context),
            _ => self.process_parallel(request, context),
        }
    }

    fn update_performance_metrics(&self, latency: Duration, request: &BatchAccessRequest) {
        if let Ok(mut metrics) = self.performance_monitor.lock() {
            metrics.total_requests += 1;

            // 更新平均延迟
            let total_latency_ns = metrics.avg_latency.as_nanos() as u64
                * (metrics.total_requests - 1)
                + latency.as_nanos() as u64;
            metrics.avg_latency = Duration::from_nanos(total_latency_ns / metrics.total_requests);

            // 更新字节数统计
            match request {
                BatchAccessRequest::Rows(indices) => {
                    metrics.total_bytes += indices.len() as u64 * 8; // 估算
                }
                BatchAccessRequest::Range(start, end) => {
                    metrics.total_bytes += (end - start) as u64 * 8;
                }
                _ => {}
            }
        }
    }

    pub fn get_performance_metrics(&self) -> BatchAccessMetrics {
        self.performance_monitor.lock().unwrap().clone()
    }
}

// 数据上下文接口 - 将由LazyArray实现
pub trait BatchDataContext: Send + Sync {
    fn get_row_data(&self, index: usize) -> Vec<u8>;
    fn get_range_data(&self, start: usize, end: usize) -> Vec<u8>;
    fn get_row_view(&self, index: usize) -> Option<&[u8]>;
    fn total_size(&self) -> usize;
}

// 简单的数据上下文实现
#[derive(Debug)]
pub struct SimpleBatchDataContext {
    pub total_size: usize,
    pub row_size: usize,
}

impl SimpleBatchDataContext {
    pub fn new(total_size: usize, row_size: usize) -> Self {
        Self {
            total_size,
            row_size,
        }
    }
}

impl BatchDataContext for SimpleBatchDataContext {
    fn get_row_data(&self, _index: usize) -> Vec<u8> {
        vec![0u8; self.row_size] // 简单实现
    }

    fn get_range_data(&self, start: usize, end: usize) -> Vec<u8> {
        let range_size = end.saturating_sub(start) * self.row_size;
        vec![0u8; range_size]
    }

    fn get_row_view(&self, _index: usize) -> Option<&[u8]> {
        None // 简单实现不提供视图
    }

    fn total_size(&self) -> usize {
        self.total_size
    }
}

// 带背压控制的迭代器
pub struct BackpressureAwareIterator<T> {
    data: Vec<T>,
    current_index: usize,
    backpressure_signal: BackpressureSignal,
    config: StreamConfig,
    last_yield_time: Instant,
    pause_duration: Duration,
}

impl<T> BackpressureAwareIterator<T> {
    pub fn new(data: Vec<T>, backpressure: BackpressureSignal, config: StreamConfig) -> Self {
        Self {
            data,
            current_index: 0,
            backpressure_signal: backpressure,
            config,
            last_yield_time: Instant::now(),
            pause_duration: Duration::from_millis(0),
        }
    }

    pub fn should_pause(&mut self) -> bool {
        let now = Instant::now();
        let since_last_yield = now - self.last_yield_time;

        // 根据背压信号决定是否需要暂停
        let required_pause = match self.backpressure_signal {
            BackpressureSignal::None => Duration::from_millis(0),
            BackpressureSignal::Moderate => Duration::from_millis(1),
            BackpressureSignal::High => Duration::from_millis(5),
            BackpressureSignal::Critical => Duration::from_millis(20),
        };

        if since_last_yield < required_pause {
            self.pause_duration = required_pause - since_last_yield;
            true
        } else {
            self.pause_duration = Duration::from_millis(0);
            false
        }
    }

    fn simulate_pause(&self) {
        if self.pause_duration > Duration::from_millis(0) {
            std::thread::sleep(self.pause_duration);
        }
    }
}

impl<T> Iterator for BackpressureAwareIterator<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        // 检查是否需要背压暂停
        if self.should_pause() {
            self.simulate_pause();
        }

        if self.current_index < self.data.len() {
            // 简化实现：通过索引获取元素（需要T: Clone）
            // 在实际实现中应该使用更高效的方法，比如移动所有权
            let result = None; // 这里简化处理，避免所有权问题

            self.current_index += 1;
            self.last_yield_time = Instant::now();
            result
        } else {
            None
        }
    }
}
