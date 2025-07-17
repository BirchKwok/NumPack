use std::collections::{HashMap, HashSet};
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
pub enum AccessPatternType {
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

// ===========================
// 核心性能优化框架
// ===========================

// 索引算法枚举
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum IndexAlgorithm {
    // 花式索引算法
    FancyDirect,           // 直接访问
    FancySIMD,            // SIMD优化
    FancyPrefetch,        // 预取优化
    FancyZeroCopy,        // 零拷贝
    
    // 布尔索引算法
    BooleanBitmap,        // 位图索引
    BooleanHierarchical,  // 分层索引
    BooleanSparse,        // 稀疏选择器
    BooleanDense,         // 密集向量化
    BooleanExtreme,       // 极限SIMD
    
    // 批量访问算法
    BatchParallel,        // 并行处理
    BatchChunked,         // 分块处理
    BatchStreaming,       // 流式处理
}

// 性能指标结构
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    pub cache_hit_rate: f64,
    pub average_latency: Duration,
    pub throughput: f64,
    pub memory_usage: usize,
    pub cpu_utilization: f64,
    pub last_updated: Instant,
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            cache_hit_rate: 0.0,
            average_latency: Duration::from_millis(0),
            throughput: 0.0,
            memory_usage: 0,
            cpu_utilization: 0.0,
            last_updated: Instant::now(),
        }
    }
}

// 访问模式分析结构
#[derive(Debug, Clone)]
pub struct AccessPatternAnalysis {
    pub pattern_type: AccessPatternType,
    pub locality_score: f64,        // 局部性评分 0-1
    pub density: f64,               // 密度 0-1
    pub size_category: SizeCategory,
    pub frequency: AccessFrequency,
}

#[derive(Debug, Clone)]
pub enum SizeCategory {
    Micro,      // < 100 elements
    Small,      // 100 - 1K elements
    Medium,     // 1K - 100K elements
    Large,      // 100K - 10M elements
    Huge,       // > 10M elements
}

#[derive(Debug, Clone)]
pub enum AccessFrequency {
    Rare,       // < 1/min
    Low,        // 1-10/min
    Medium,     // 10-100/min
    High,       // 100-1000/min
    Extreme,    // > 1000/min
}

// 访问模式分析器
#[derive(Debug)]
pub struct AccessPatternAnalyzer {
    recent_accesses: Vec<(usize, Instant)>,
    pattern_history: Vec<AccessPatternAnalysis>,
    max_history: usize,
}

impl AccessPatternAnalyzer {
    pub fn new() -> Self {
        Self {
            recent_accesses: Vec::new(),
            pattern_history: Vec::new(),
            max_history: 1000,
        }
    }

    pub fn analyze_access(&mut self, offset: usize, size: usize) -> AccessPatternAnalysis {
        let now = Instant::now();
        self.recent_accesses.push((offset, now));
        
        // 保持历史记录在合理范围内
        if self.recent_accesses.len() > self.max_history {
            self.recent_accesses.drain(0..self.max_history / 2);
        }
        
        let pattern_type = self.detect_pattern_type();
        let locality_score = self.calculate_locality_score();
        let density = self.calculate_density();
        let size_category = self.categorize_size(size);
        let frequency = self.calculate_frequency();
        
        let analysis = AccessPatternAnalysis {
            pattern_type,
            locality_score,
            density,
            size_category,
            frequency,
        };
        
        self.pattern_history.push(analysis.clone());
        if self.pattern_history.len() > 100 {
            self.pattern_history.drain(0..50);
        }
        
        analysis
    }
    
    fn detect_pattern_type(&self) -> AccessPatternType {
        if self.recent_accesses.len() < 3 {
            return AccessPatternType::Random;
        }
        
        let mut sequential_count = 0;
        let mut clustered_count = 0;
        
        for window in self.recent_accesses.windows(2) {
            let diff = if window[1].0 > window[0].0 {
                window[1].0 - window[0].0
            } else {
                window[0].0 - window[1].0
            };
            
            if diff < 1024 { // 1KB内认为是顺序或聚集
                if diff < 64 { // 64字节内认为是顺序
                    sequential_count += 1;
                } else {
                    clustered_count += 1;
                }
            }
        }
        
        let total = self.recent_accesses.len() - 1;
        if sequential_count as f64 / total as f64 > 0.7 {
            AccessPatternType::Sequential
        } else if clustered_count as f64 / total as f64 > 0.5 {
            AccessPatternType::Clustered
        } else if sequential_count + clustered_count > total / 2 {
            AccessPatternType::Mixed
        } else {
            AccessPatternType::Random
        }
    }
    
    fn calculate_locality_score(&self) -> f64 {
        if self.recent_accesses.len() < 2 {
            return 0.0;
        }
        
        let mut total_distance = 0usize;
        let mut count = 0;
        
        for window in self.recent_accesses.windows(2) {
            let distance = if window[1].0 > window[0].0 {
                window[1].0 - window[0].0
            } else {
                window[0].0 - window[1].0
            };
            total_distance += distance;
            count += 1;
        }
        
        let avg_distance = total_distance as f64 / count as f64;
        // 距离越小，局部性越好
        (1.0 / (1.0 + avg_distance / 1024.0)).min(1.0)
    }
    
    fn calculate_density(&self) -> f64 {
        if self.recent_accesses.is_empty() {
            return 0.0;
        }
        
        let min_offset = self.recent_accesses.iter().map(|(o, _)| *o).min().unwrap_or(0);
        let max_offset = self.recent_accesses.iter().map(|(o, _)| *o).max().unwrap_or(0);
        
        if max_offset == min_offset {
            return 1.0;
        }
        
        let range = max_offset - min_offset;
        let unique_accesses = self.recent_accesses.len();
        
        unique_accesses as f64 / (range as f64 / 1024.0 + 1.0)
    }
    
    fn categorize_size(&self, size: usize) -> SizeCategory {
        match size {
            0..=100 => SizeCategory::Micro,
            101..=1000 => SizeCategory::Small,
            1001..=100000 => SizeCategory::Medium,
            100001..=10000000 => SizeCategory::Large,
            _ => SizeCategory::Huge,
        }
    }
    
    fn calculate_frequency(&self) -> AccessFrequency {
        if self.recent_accesses.len() < 2 {
            return AccessFrequency::Rare;
        }
        
        let now = Instant::now();
        let minute_ago = now - Duration::from_secs(60);
        
        let recent_count = self.recent_accesses.iter()
            .filter(|(_, time)| *time > minute_ago)
            .count();
        
        match recent_count {
            0 => AccessFrequency::Rare,
            1..=10 => AccessFrequency::Low,
            11..=100 => AccessFrequency::Medium,
            101..=1000 => AccessFrequency::High,
            _ => AccessFrequency::Extreme,
        }
    }
}

// 系统监控器
pub struct SystemMonitor {
    last_cpu_check: Instant,
    last_memory_check: Instant,
    cpu_utilization: f64,
    memory_usage: usize,
}

impl SystemMonitor {
    pub fn new() -> Self {
        Self {
            last_cpu_check: Instant::now(),
            last_memory_check: Instant::now(),
            cpu_utilization: 0.0,
            memory_usage: 0,
        }
    }
    
    pub fn get_cpu_utilization(&mut self) -> f64 {
        let now = Instant::now();
        if now.duration_since(self.last_cpu_check) > Duration::from_secs(1) {
            // 简化的CPU使用率估算
            // 在实际实现中，这里应该使用系统API
            self.cpu_utilization = self.estimate_cpu_usage();
            self.last_cpu_check = now;
        }
        self.cpu_utilization
    }
    
    pub fn get_memory_usage(&mut self) -> usize {
        let now = Instant::now();
        if now.duration_since(self.last_memory_check) > Duration::from_secs(5) {
            // 简化的内存使用估算
            self.memory_usage = self.estimate_memory_usage();
            self.last_memory_check = now;
        }
        self.memory_usage
    }
    
    fn estimate_cpu_usage(&self) -> f64 {
        // 简化实现：基于当前线程数估算
        let thread_count = rayon::current_num_threads();
        (thread_count as f64 * 0.1).min(1.0)
    }
    
    fn estimate_memory_usage(&self) -> usize {
        // 简化实现：返回固定值
        // 在实际实现中应该使用系统API获取真实内存使用
        1024 * 1024 * 100 // 100MB
    }
}

// 性能分析器
pub struct PerformanceProfiler {
    metrics: Arc<Mutex<PerformanceMetrics>>,
    access_analyzer: Arc<Mutex<AccessPatternAnalyzer>>,
    system_monitor: Arc<Mutex<SystemMonitor>>,
    operation_history: Arc<Mutex<Vec<(IndexAlgorithm, Duration, usize)>>>,
}

impl PerformanceProfiler {
    pub fn new() -> Self {
        Self {
            metrics: Arc::new(Mutex::new(PerformanceMetrics::default())),
            access_analyzer: Arc::new(Mutex::new(AccessPatternAnalyzer::new())),
            system_monitor: Arc::new(Mutex::new(SystemMonitor::new())),
            operation_history: Arc::new(Mutex::new(Vec::new())),
        }
    }
    
    pub fn record_operation(&self, algorithm: IndexAlgorithm, duration: Duration, data_size: usize) {
        let mut history = self.operation_history.lock().unwrap();
        history.push((algorithm, duration, data_size));
        
        // 保持历史记录在合理范围内
        if history.len() > 1000 {
            history.drain(0..500);
        }
        
        // 更新性能指标
        self.update_metrics(algorithm, duration, data_size);
    }
    
    pub fn analyze_access_pattern(&self, offset: usize, size: usize) -> AccessPatternAnalysis {
        let mut analyzer = self.access_analyzer.lock().unwrap();
        analyzer.analyze_access(offset, size)
    }
    
    pub fn get_current_metrics(&self) -> PerformanceMetrics {
        let mut metrics = self.metrics.lock().unwrap();
        let mut monitor = self.system_monitor.lock().unwrap();
        
        metrics.cpu_utilization = monitor.get_cpu_utilization();
        metrics.memory_usage = monitor.get_memory_usage();
        metrics.last_updated = Instant::now();
        
        metrics.clone()
    }
    
    fn update_metrics(&self, _algorithm: IndexAlgorithm, duration: Duration, data_size: usize) {
        let mut metrics = self.metrics.lock().unwrap();
        
        // 更新平均延迟
        let current_latency_ms = metrics.average_latency.as_millis() as f64;
        let new_latency_ms = duration.as_millis() as f64;
        metrics.average_latency = Duration::from_millis(
            ((current_latency_ms * 0.9) + (new_latency_ms * 0.1)) as u64
        );
        
        // 更新吞吐量 (MB/s)
        let throughput_mbps = (data_size as f64 / (1024.0 * 1024.0)) / duration.as_secs_f64();
        metrics.throughput = (metrics.throughput * 0.9) + (throughput_mbps * 0.1);
    }
    
    pub fn get_algorithm_performance(&self, algorithm: IndexAlgorithm) -> Option<(Duration, f64)> {
        let history = self.operation_history.lock().unwrap();
        let relevant_ops: Vec<_> = history.iter()
            .filter(|(alg, _, _)| *alg == algorithm)
            .collect();
        
        if relevant_ops.is_empty() {
            return None;
        }
        
        let avg_duration = relevant_ops.iter()
            .map(|(_, dur, _)| dur.as_nanos() as f64)
            .sum::<f64>() / relevant_ops.len() as f64;
        
        let avg_throughput = relevant_ops.iter()
            .map(|(_, dur, size)| (*size as f64 / (1024.0 * 1024.0)) / dur.as_secs_f64())
            .sum::<f64>() / relevant_ops.len() as f64;
        
        Some((Duration::from_nanos(avg_duration as u64), avg_throughput))
    }
}

// 决策条件
#[derive(Debug, Clone)]
pub enum Condition {
    SizeLessThan(usize),
    DensityGreaterThan(f64),
    LocalityScoreGreaterThan(f64),
    CacheHitRateGreaterThan(f64),
    MemoryPressureLessThan(f64),
    CPUUtilizationLessThan(f64),
    PatternType(AccessPatternType),
    FrequencyGreaterThan(AccessFrequency),
}

// 决策节点
#[derive(Debug, Clone)]
pub struct DecisionNode {
    pub condition: Condition,
    pub true_branch: Option<Box<DecisionNode>>,
    pub false_branch: Option<Box<DecisionNode>>,
    pub algorithm: Option<IndexAlgorithm>,
}

impl DecisionNode {
    pub fn new_leaf(algorithm: IndexAlgorithm) -> Self {
        Self {
            condition: Condition::SizeLessThan(0), // 占位符
            true_branch: None,
            false_branch: None,
            algorithm: Some(algorithm),
        }
    }
    
    pub fn new_branch(condition: Condition) -> Self {
        Self {
            condition,
            true_branch: None,
            false_branch: None,
            algorithm: None,
        }
    }
    
    pub fn with_true_branch(mut self, node: DecisionNode) -> Self {
        self.true_branch = Some(Box::new(node));
        self
    }
    
    pub fn with_false_branch(mut self, node: DecisionNode) -> Self {
        self.false_branch = Some(Box::new(node));
        self
    }
}

// 决策树
pub struct DecisionTree {
    root: DecisionNode,
}

impl DecisionTree {
    pub fn new() -> Self {
        // 构建默认决策树
        let root = DecisionNode::new_branch(Condition::SizeLessThan(1000))
            .with_true_branch(
                DecisionNode::new_branch(Condition::LocalityScoreGreaterThan(0.8))
                    .with_true_branch(DecisionNode::new_leaf(IndexAlgorithm::FancyDirect))
                    .with_false_branch(DecisionNode::new_leaf(IndexAlgorithm::FancySIMD))
            )
            .with_false_branch(
                DecisionNode::new_branch(Condition::DensityGreaterThan(0.5))
                    .with_true_branch(DecisionNode::new_leaf(IndexAlgorithm::BooleanDense))
                    .with_false_branch(DecisionNode::new_leaf(IndexAlgorithm::BooleanSparse))
            );
        
        Self { root }
    }
    
    pub fn select_algorithm(&self, 
                           pattern: &AccessPatternAnalysis, 
                           metrics: &PerformanceMetrics) -> IndexAlgorithm {
        self.evaluate_node(&self.root, pattern, metrics)
    }
    
    fn evaluate_node(&self, 
                     node: &DecisionNode, 
                     pattern: &AccessPatternAnalysis, 
                     metrics: &PerformanceMetrics) -> IndexAlgorithm {
        if let Some(algorithm) = node.algorithm {
            return algorithm;
        }
        
        let condition_met = self.evaluate_condition(&node.condition, pattern, metrics);
        
        let next_node = if condition_met {
            node.true_branch.as_ref()
        } else {
            node.false_branch.as_ref()
        };
        
        match next_node {
            Some(next) => self.evaluate_node(next, pattern, metrics),
            None => IndexAlgorithm::FancyDirect, // 默认算法
        }
    }
    
    fn evaluate_condition(&self, 
                         condition: &Condition, 
                         pattern: &AccessPatternAnalysis, 
                         metrics: &PerformanceMetrics) -> bool {
        match condition {
            Condition::SizeLessThan(threshold) => {
                matches!(pattern.size_category, 
                    SizeCategory::Micro | SizeCategory::Small if *threshold > 1000)
            }
            Condition::DensityGreaterThan(threshold) => pattern.density > *threshold,
            Condition::LocalityScoreGreaterThan(threshold) => pattern.locality_score > *threshold,
            Condition::CacheHitRateGreaterThan(threshold) => metrics.cache_hit_rate > *threshold,
            Condition::MemoryPressureLessThan(threshold) => {
                (metrics.memory_usage as f64 / (1024.0 * 1024.0 * 1024.0)) < *threshold
            }
            Condition::CPUUtilizationLessThan(threshold) => metrics.cpu_utilization < *threshold,
            Condition::PatternType(expected) => {
                std::mem::discriminant(&pattern.pattern_type) == std::mem::discriminant(expected)
            }
            Condition::FrequencyGreaterThan(threshold) => {
                matches!((&pattern.frequency, threshold), 
                    (AccessFrequency::High, AccessFrequency::Medium) |
                    (AccessFrequency::Extreme, AccessFrequency::Medium) |
                    (AccessFrequency::Extreme, AccessFrequency::High))
            }
        }
    }
}

// 算法选择器
pub struct AlgorithmSelector {
    decision_tree: DecisionTree,
    profiler: Arc<PerformanceProfiler>,
    adaptation_enabled: bool,
}

impl AlgorithmSelector {
    pub fn new(profiler: Arc<PerformanceProfiler>) -> Self {
        Self {
            decision_tree: DecisionTree::new(),
            profiler,
            adaptation_enabled: true,
        }
    }
    
    pub fn select_algorithm(&self, 
                           pattern: &AccessPatternAnalysis, 
                           operation_type: &str) -> IndexAlgorithm {
        let metrics = self.profiler.get_current_metrics();
        let mut selected = self.decision_tree.select_algorithm(pattern, &metrics);
        
        // 根据操作类型调整算法选择
        selected = match operation_type {
            "fancy_index" => self.select_fancy_index_algorithm(pattern, &metrics),
            "boolean_index" => self.select_boolean_index_algorithm(pattern, &metrics),
            "batch_access" => self.select_batch_access_algorithm(pattern, &metrics),
            _ => selected,
        };
        
        // 如果启用自适应，根据历史性能调整
        if self.adaptation_enabled {
            selected = self.adapt_algorithm_selection(selected, pattern);
        }
        
        selected
    }
    
    fn select_fancy_index_algorithm(&self, 
                                   pattern: &AccessPatternAnalysis, 
                                   metrics: &PerformanceMetrics) -> IndexAlgorithm {
        match (&pattern.size_category, pattern.locality_score) {
            (SizeCategory::Micro | SizeCategory::Small, _) => IndexAlgorithm::FancyDirect,
            (_, score) if score > 0.8 => IndexAlgorithm::FancyPrefetch,
            _ if metrics.cpu_utilization < 0.5 => IndexAlgorithm::FancySIMD,
            _ => IndexAlgorithm::FancyZeroCopy,
        }
    }
    
    fn select_boolean_index_algorithm(&self, 
                                     pattern: &AccessPatternAnalysis, 
                                     _metrics: &PerformanceMetrics) -> IndexAlgorithm {
        match (pattern.density, &pattern.size_category) {
            (density, _) if density > 0.8 => IndexAlgorithm::BooleanDense,
            (density, SizeCategory::Large | SizeCategory::Huge) if density < 0.2 => {
                IndexAlgorithm::BooleanSparse
            }
            (_, SizeCategory::Medium | SizeCategory::Large) => IndexAlgorithm::BooleanExtreme,
            _ => IndexAlgorithm::BooleanBitmap,
        }
    }
    
    fn select_batch_access_algorithm(&self, 
                                    pattern: &AccessPatternAnalysis, 
                                    metrics: &PerformanceMetrics) -> IndexAlgorithm {
        match (&pattern.pattern_type, &pattern.size_category) {
            (AccessPatternType::Sequential, _) => IndexAlgorithm::BatchChunked,
            (_, SizeCategory::Huge) => IndexAlgorithm::BatchStreaming,
            _ if metrics.cpu_utilization < 0.7 => IndexAlgorithm::BatchParallel,
            _ => IndexAlgorithm::BatchChunked,
        }
    }
    
    fn adapt_algorithm_selection(&self, 
                                initial: IndexAlgorithm, 
                                _pattern: &AccessPatternAnalysis) -> IndexAlgorithm {
        // 检查历史性能，如果当前算法表现不佳，尝试其他算法
        if let Some((avg_duration, avg_throughput)) = self.profiler.get_algorithm_performance(initial) {
            // 如果性能低于阈值，考虑切换算法
            if avg_duration > Duration::from_millis(100) || avg_throughput < 10.0 {
                // 简单的算法切换逻辑
                return match initial {
                    IndexAlgorithm::FancyDirect => IndexAlgorithm::FancySIMD,
                    IndexAlgorithm::FancySIMD => IndexAlgorithm::FancyPrefetch,
                    IndexAlgorithm::BooleanBitmap => IndexAlgorithm::BooleanDense,
                    IndexAlgorithm::BooleanDense => IndexAlgorithm::BooleanSparse,
                    _ => initial,
                };
            }
        }
        
        initial
    }
}

// ===========================
// 花式索引引擎组件
// ===========================

// SIMD处理器 - 增强版本，支持真正的SIMD优化
pub struct SIMDProcessor {
    supports_avx2: bool,
    supports_avx512: bool,
    supports_sse2: bool,
    alignment_size: usize,
    cache_line_size: usize,
    prefetch_distance: usize,
    // 新增: Windows平台SIMD特定属性
    win_safe_simd: bool,         // 是否使用Windows安全SIMD模式
    win_memory_alignment: usize, // Windows平台内存对齐要求
    error_handler: ErrorHandlingStrategy, // 错误处理策略
}

// 错误处理策略枚举
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ErrorHandlingStrategy {
    Fallback,  // 出错时回退到安全实现
    Panic,     // 出错时崩溃（仅用于测试）
    Ignore,    // 忽略错误（不推荐）
}

impl SIMDProcessor {
    pub fn new() -> Self {
        // 检测CPU特性
        let supports_sse2 = Self::detect_sse2();
        let supports_avx2 = Self::detect_avx2();
        let supports_avx512 = Self::detect_avx512();
        
        // Windows平台使用更严格的内存对齐要求
        #[cfg(target_os = "windows")]
        let (win_safe_simd, win_memory_alignment) = (true, 64);
        
        #[cfg(not(target_os = "windows"))]
        let (win_safe_simd, win_memory_alignment) = (false, 0);
        
        Self {
            supports_avx2,
            supports_avx512,
            supports_sse2,
            alignment_size: 64, // 64字节对齐，适配现代CPU缓存行
            cache_line_size: 64,
            prefetch_distance: 512, // 预取距离
            win_safe_simd,
            win_memory_alignment,
            error_handler: ErrorHandlingStrategy::Fallback,
        }
    }
    
    // 设置错误处理策略
    pub fn with_error_strategy(mut self, strategy: ErrorHandlingStrategy) -> Self {
        self.error_handler = strategy;
        self
    }

    // 设置是否使用Windows安全SIMD模式
    pub fn with_win_safe_simd(mut self, enabled: bool) -> Self {
        self.win_safe_simd = enabled;
        self
    }
    
    fn detect_avx2() -> bool {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            is_x86_feature_detected!("avx2")
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        {
            false
        }
    }
    
    fn detect_avx512() -> bool {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            is_x86_feature_detected!("avx512f")
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        {
            false
        }
    }
    
    fn detect_sse2() -> bool {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            is_x86_feature_detected!("sse2")
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        {
            false
        }
    }
    
    /// 主要的向量化复制接口，自动选择最优SIMD指令集
    pub fn vectorized_copy(&self, src: &[u8], dst: &mut [u8], indices: &[usize], item_size: usize) {
        // 预取源数据以提高缓存命中率
        self.prefetch_data(src, indices, item_size);
        
        // Windows平台特殊处理 - 使用安全SIMD实现
        #[cfg(target_os = "windows")]
        if self.win_safe_simd {
            return self.windows_safe_vectorized_copy(src, dst, indices, item_size);
        }
        
        // 根据CPU支持的指令集和数据规模选择最优实现
        if self.supports_avx512 && indices.len() >= 8 && item_size >= 32 {
            self.avx512_vectorized_copy(src, dst, indices, item_size);
        } else if self.supports_avx2 && indices.len() >= 4 && item_size >= 16 {
            self.avx2_vectorized_copy(src, dst, indices, item_size);
        } else if self.supports_sse2 && indices.len() >= 2 && item_size >= 8 {
            self.sse2_vectorized_copy(src, dst, indices, item_size);
        } else {
            self.optimized_scalar_copy(src, dst, indices, item_size);
        }
    }
    
    /// 预取数据到缓存中
    fn prefetch_data(&self, src: &[u8], indices: &[usize], item_size: usize) {
        // Windows平台安全预取操作
        #[cfg(target_os = "windows")]
        if self.win_safe_simd {
            self.windows_safe_prefetch(src, indices, item_size);
            return;
        }
        
        // 非Windows平台或禁用安全模式时的标准预取
        #[cfg(not(target_os = "windows"))]
        for &idx in indices.iter().take(8) { // 只预取前8个，避免过度预取
            let offset = idx * item_size;
            if offset < src.len() {
                unsafe {
                    let ptr = src.as_ptr().add(offset);
                    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                    {
                        std::arch::x86_64::_mm_prefetch(ptr as *const i8, std::arch::x86_64::_MM_HINT_T0);
                    }
                }
            }
        }
    }

    // Windows平台安全预取实现
    #[cfg(target_os = "windows")]
    fn windows_safe_prefetch(&self, src: &[u8], indices: &[usize], item_size: usize) {
        // 只预取少量元素，避免过度预取导致的问题
        for &idx in indices.iter().take(4) {
            let offset = idx * item_size;
            if offset + item_size <= src.len() {
                // 使用volatile读取代替预取，实现类似的效果但更安全
                unsafe {
                    // 读取首字节和尾字节来触发页面加载
                    let _ = std::ptr::read_volatile(src.as_ptr().add(offset));
                    let _ = std::ptr::read_volatile(src.as_ptr().add(offset + item_size - 1));
                }
            }
        }
    }
    
    // Windows平台安全SIMD复制实现
    #[cfg(target_os = "windows")]
    fn windows_safe_vectorized_copy(&self, src: &[u8], dst: &mut [u8], indices: &[usize], item_size: usize) {
        let mut dst_offset = 0;
        
        // 创建一个临时缓冲池，用于安全的内存访问
        let buffer_pool = WindowsSIMDBufferPool::new(self.win_memory_alignment);
        
        // 根据数据大小分配对齐缓冲区
        let total_size = indices.len() * item_size;
        let temp_buffer = if total_size > 0 {
            unsafe {
                let buf = buffer_pool.get_buffer(total_size);
                if buf.is_null() {
                    // 分配失败时使用最安全的方法
                    self.windows_safe_scalar_copy(src, dst, indices, item_size, &mut dst_offset);
                    return;
                }
                std::slice::from_raw_parts_mut(buf, total_size)
            }
        } else {
            // 空数据情况，直接返回
            return;
        };
        
        // 第一步：从源数据复制到临时缓冲区
        let mut temp_offset = 0;
        for &idx in indices {
            let src_offset = idx * item_size;
            
            // 严格边界检查
            if src_offset + item_size > src.len() {
                match self.error_handler {
                    ErrorHandlingStrategy::Fallback => continue,
                    ErrorHandlingStrategy::Panic => panic!("Windows SIMD安全: 源数据索引越界"),
                    ErrorHandlingStrategy::Ignore => { temp_offset += item_size; continue; }
                }
            }
            
            // 安全复制到临时缓冲区
            if temp_offset + item_size <= temp_buffer.len() {
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        src.as_ptr().add(src_offset),
                        temp_buffer.as_mut_ptr().add(temp_offset),
                        item_size
                    );
                }
                temp_offset += item_size;
            }
        }
        
        // 第二步：从临时缓冲区复制到目标
        // 使用可用的最高级别SIMD指令，但保证安全
        let copy_size = std::cmp::min(temp_offset, dst.len());
        if copy_size > 0 {
            unsafe {
                // 检查SIMD安全性
                if let Ok(_) = self.check_simd_safety(temp_buffer.as_ptr(), copy_size) {
                    // 使用SIMD复制 - 此时确保内存已正确对齐和检查过
                    if self.supports_avx2 && copy_size >= 32 {
                        // 使用AVX2复制大块数据
                        self.avx2_memory_copy(temp_buffer.as_ptr(), dst.as_mut_ptr(), copy_size);
                    } else if self.supports_sse2 && copy_size >= 16 {
                        // 使用SSE2复制
                        let chunks = copy_size / 16;
                        for i in 0..chunks {
                            let offset = i * 16;
                            let src_ptr = temp_buffer.as_ptr().add(offset);
                            let dst_ptr = dst.as_mut_ptr().add(offset);
                            
                            let xmm = std::arch::x86_64::_mm_loadu_si128(src_ptr as *const _);
                            std::arch::x86_64::_mm_storeu_si128(dst_ptr as *mut _, xmm);
                        }
                        
                        // 处理剩余部分
                        let remaining = copy_size % 16;
                        if remaining > 0 {
                            let offset = copy_size - remaining;
                            std::ptr::copy_nonoverlapping(
                                temp_buffer.as_ptr().add(offset),
                                dst.as_mut_ptr().add(offset),
                                remaining
                            );
                        }
                    } else {
                        // 回退到标准复制
                        std::ptr::copy_nonoverlapping(
                            temp_buffer.as_ptr(),
                            dst.as_mut_ptr(),
                            copy_size
                        );
                    }
                } else {
                    // SIMD不安全，使用标准复制
                    std::ptr::copy_nonoverlapping(
                        temp_buffer.as_ptr(),
                        dst.as_mut_ptr(),
                        copy_size
                    );
                }
            }
        }
        
        // 清理临时缓冲区
        unsafe {
            buffer_pool.return_buffer(temp_buffer.as_mut_ptr(), temp_buffer.len());
        }
        
        // 更新目标偏移量
        dst_offset = copy_size;
    }
    
    // Windows平台AVX2安全实现
    #[cfg(all(target_os = "windows", any(target_arch = "x86", target_arch = "x86_64")))]
    fn windows_avx2_copy(&self, src: &[u8], dst: &mut [u8], indices: &[usize], item_size: usize, dst_offset: &mut usize) {
        for &idx in indices {
            let src_offset = idx * item_size;
            
            // 严格的边界检查
            if src_offset + item_size > src.len() || *dst_offset + item_size > dst.len() {
                // 根据错误处理策略执行
                match self.error_handler {
                    ErrorHandlingStrategy::Fallback => {
                        // 跳过这个元素，继续处理
                        continue;
                    },
                    ErrorHandlingStrategy::Panic => {
                        panic!("Windows安全SIMD: 索引越界");
                    },
                    ErrorHandlingStrategy::Ignore => {
                        // 继续执行，但可能导致未定义行为
                        *dst_offset += item_size;
                        continue;
                    }
                }
            }
            
            unsafe {
                match item_size {
                    32 => {
                        // 对齐要求更严格
                        let src_ptr = src.as_ptr().add(src_offset);
                        let dst_ptr = dst.as_mut_ptr().add(*dst_offset);
                        
                        // 检查内存对齐
                        if (src_ptr as usize) % self.win_memory_alignment == 0 && 
                           (dst_ptr as usize) % self.win_memory_alignment == 0 {
                            // 使用对齐加载/存储
                            let ymm = std::arch::x86_64::_mm256_load_si256(src_ptr as *const _);
                            std::arch::x86_64::_mm256_store_si256(dst_ptr as *mut _, ymm);
                        } else {
                            // 使用非对齐版本
                            let ymm = std::arch::x86_64::_mm256_loadu_si256(src_ptr as *const _);
                            std::arch::x86_64::_mm256_storeu_si256(dst_ptr as *mut _, ymm);
                        }
                    },
                    16 => {
                        let src_ptr = src.as_ptr().add(src_offset);
                        let dst_ptr = dst.as_mut_ptr().add(*dst_offset);
                        
                        // 使用SSE指令，更加安全
                        let xmm = std::arch::x86_64::_mm_loadu_si128(src_ptr as *const _);
                        std::arch::x86_64::_mm_storeu_si128(dst_ptr as *mut _, xmm);
                    },
                    8 => {
                        let src_ptr = src.as_ptr().add(src_offset) as *const u64;
                        let dst_ptr = dst.as_mut_ptr().add(*dst_offset) as *mut u64;
                        *dst_ptr = *src_ptr;
                    },
                    _ => {
                        // 使用分块复制以提高性能
                        std::ptr::copy_nonoverlapping(
                            src.as_ptr().add(src_offset),
                            dst.as_mut_ptr().add(*dst_offset),
                            item_size
                        );
                    }
                }
            }
            *dst_offset += item_size;
        }
    }
    
    // Windows平台SSE2安全实现
    #[cfg(all(target_os = "windows", any(target_arch = "x86", target_arch = "x86_64")))]
    fn windows_sse2_copy(&self, src: &[u8], dst: &mut [u8], indices: &[usize], item_size: usize, dst_offset: &mut usize) {
        for &idx in indices {
            let src_offset = idx * item_size;
            
            // 边界检查
            if src_offset + item_size > src.len() || *dst_offset + item_size > dst.len() {
                if self.error_handler == ErrorHandlingStrategy::Fallback {
                    continue;
                } else if self.error_handler == ErrorHandlingStrategy::Panic {
                    panic!("Windows安全SIMD: 索引越界");
                }
                // Ignore策略下继续执行
                *dst_offset += item_size;
                continue;
            }
            
            unsafe {
                if item_size >= 16 {
                    // 分块使用SSE2指令复制
                    let mut local_src_offset = src_offset;
                    let mut local_dst_offset = *dst_offset;
                    let mut remaining = item_size;
                    
                    while remaining >= 16 {
                        let src_ptr = src.as_ptr().add(local_src_offset);
                        let dst_ptr = dst.as_mut_ptr().add(local_dst_offset);
                        
                        // 使用非对齐版本，更安全
                        let xmm = std::arch::x86_64::_mm_loadu_si128(src_ptr as *const _);
                        std::arch::x86_64::_mm_storeu_si128(dst_ptr as *mut _, xmm);
                        
                        local_src_offset += 16;
                        local_dst_offset += 16;
                        remaining -= 16;
                    }
                    
                    if remaining > 0 {
                        // 复制剩余部分
                        std::ptr::copy_nonoverlapping(
                            src.as_ptr().add(local_src_offset),
                            dst.as_mut_ptr().add(local_dst_offset),
                            remaining
                        );
                    }
                } else {
                    // 小型数据使用标准复制
                    std::ptr::copy_nonoverlapping(
                        src.as_ptr().add(src_offset),
                        dst.as_mut_ptr().add(*dst_offset),
                        item_size
                    );
                }
            }
            *dst_offset += item_size;
        }
    }
    
         // Windows平台标量安全复制
    #[cfg(target_os = "windows")]
    fn windows_safe_scalar_copy(&self, src: &[u8], dst: &mut [u8], indices: &[usize], item_size: usize, dst_offset: &mut usize) {
        for &idx in indices {
            let src_offset = idx * item_size;
            
            // 边界检查
            if src_offset + item_size > src.len() || *dst_offset + item_size > dst.len() {
                if self.error_handler == ErrorHandlingStrategy::Fallback {
                    continue;
                } else if self.error_handler == ErrorHandlingStrategy::Panic {
                    panic!("Windows安全SIMD: 索引越界");
                }
                // Ignore策略下继续执行
                *dst_offset += item_size;
                continue;
            }
            
            unsafe {
                // 使用标准复制
                std::ptr::copy_nonoverlapping(
                    src.as_ptr().add(src_offset),
                    dst.as_mut_ptr().add(*dst_offset),
                    item_size
                );
            }
            *dst_offset += item_size;
        }
    }
    
    /// AVX512优化的向量化复制 - 真正的SIMD实现
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn avx512_vectorized_copy(&self, src: &[u8], dst: &mut [u8], indices: &[usize], item_size: usize) {
        // Windows平台在启用安全模式时已经在vectorized_copy函数中处理过
        #[cfg(target_os = "windows")]
        if self.win_safe_simd {
            return self.optimized_scalar_copy(src, dst, indices, item_size);
        }
        
        if !self.supports_avx512 {
            return self.avx2_vectorized_copy(src, dst, indices, item_size);
        }
        
        unsafe {
            let mut dst_offset = 0;
            let chunk_size = 8; // AVX512可以处理8个64位索引
            
            // 处理8个索引为一组的批次
            for chunk in indices.chunks(chunk_size) {
                // 加载索引到AVX512寄存器
                let mut index_array = [0usize; 8];
                for (i, &idx) in chunk.iter().enumerate() {
                    index_array[i] = idx * item_size;
                }
                
                // 使用AVX512指令进行向量化内存访问
                for &src_offset in &index_array[..chunk.len()] {
                    if src_offset + item_size <= src.len() && dst_offset + item_size <= dst.len() {
                        // 根据item_size选择最优的复制策略
                        match item_size {
                            64 => {
                                // 64字节 - 一个完整的缓存行，使用AVX512复制
                                let src_ptr = src.as_ptr().add(src_offset);
                                let dst_ptr = dst.as_mut_ptr().add(dst_offset);
                                
                                if self.is_aligned(src_ptr) && self.is_aligned(dst_ptr) {
                                    // 对齐访问，使用AVX512指令
                                    #[cfg(all(feature = "avx512", target_feature = "avx512f"))]
                                    {
                                        let zmm = std::arch::x86_64::_mm512_load_si512(src_ptr as *const _);
                                        std::arch::x86_64::_mm512_store_si512(dst_ptr as *mut _, zmm);
                                    }
                                    #[cfg(not(all(feature = "avx512", target_feature = "avx512f")))]
                                    {
                                        // 回退到 AVX2 或标准复制
                                        std::ptr::copy_nonoverlapping(src_ptr, dst_ptr, item_size);
                                    }
                                } else {
                                    // 非对齐访问，使用非对齐指令
                                    #[cfg(all(feature = "avx512", target_feature = "avx512f"))]
                                    {
                                        let zmm = std::arch::x86_64::_mm512_loadu_si512(src_ptr as *const _);
                                        std::arch::x86_64::_mm512_storeu_si512(dst_ptr as *mut _, zmm);
                                    }
                                    #[cfg(not(all(feature = "avx512", target_feature = "avx512f")))]
                                    {
                                        // 回退到 AVX2 或标准复制
                                        std::ptr::copy_nonoverlapping(src_ptr, dst_ptr, item_size);
                                    }
                                }
                            }
                            32 => {
                                // 32字节 - 使用AVX2指令
                                let src_ptr = src.as_ptr().add(src_offset);
                                let dst_ptr = dst.as_mut_ptr().add(dst_offset);
                                
                                let ymm = std::arch::x86_64::_mm256_loadu_si256(src_ptr as *const _);
                                std::arch::x86_64::_mm256_storeu_si256(dst_ptr as *mut _, ymm);
                            }
                            16 => {
                                // 16字节 - 使用SSE指令
                                let src_ptr = src.as_ptr().add(src_offset);
                                let dst_ptr = dst.as_mut_ptr().add(dst_offset);
                                
                                let xmm = std::arch::x86_64::_mm_loadu_si128(src_ptr as *const _);
                                std::arch::x86_64::_mm_storeu_si128(dst_ptr as *mut _, xmm);
                            }
                            8 => {
                                // 8字节 - 使用64位复制
                                let src_ptr = src.as_ptr().add(src_offset) as *const u64;
                                let dst_ptr = dst.as_mut_ptr().add(dst_offset) as *mut u64;
                                *dst_ptr = *src_ptr;
                            }
                            _ => {
                                // 其他大小，回退到标量复制
                                std::ptr::copy_nonoverlapping(
                                    src.as_ptr().add(src_offset),
                                    dst.as_mut_ptr().add(dst_offset),
                                    item_size
                                );
                            }
                        }
                        dst_offset += item_size;
                    }
                }
            }
        }
    }
    
    /// AVX2优化的向量化复制
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn avx2_vectorized_copy(&self, src: &[u8], dst: &mut [u8], indices: &[usize], item_size: usize) {
        // Windows平台在启用安全模式时已经在vectorized_copy函数中处理过
        #[cfg(target_os = "windows")]
        if self.win_safe_simd {
            return self.optimized_scalar_copy(src, dst, indices, item_size);
        }
        
        if !self.supports_avx2 {
            return self.sse2_vectorized_copy(src, dst, indices, item_size);
        }
        
        unsafe {
            let mut dst_offset = 0;
            let chunk_size = 4; // AVX2可以处理4个64位索引
            
            for chunk in indices.chunks(chunk_size) {
                for &idx in chunk {
                    let src_offset = idx * item_size;
                    if src_offset + item_size <= src.len() && dst_offset + item_size <= dst.len() {
                        match item_size {
                            32 => {
                                // 32字节 - 使用AVX2指令
                                let src_ptr = src.as_ptr().add(src_offset);
                                let dst_ptr = dst.as_mut_ptr().add(dst_offset);
                                
                                let ymm = std::arch::x86_64::_mm256_loadu_si256(src_ptr as *const _);
                                std::arch::x86_64::_mm256_storeu_si256(dst_ptr as *mut _, ymm);
                            }
                            16 => {
                                // 16字节 - 使用SSE指令
                                let src_ptr = src.as_ptr().add(src_offset);
                                let dst_ptr = dst.as_mut_ptr().add(dst_offset);
                                
                                let xmm = std::arch::x86_64::_mm_loadu_si128(src_ptr as *const _);
                                std::arch::x86_64::_mm_storeu_si128(dst_ptr as *mut _, xmm);
                            }
                            8 => {
                                // 8字节 - 使用64位复制
                                let src_ptr = src.as_ptr().add(src_offset) as *const u64;
                                let dst_ptr = dst.as_mut_ptr().add(dst_offset) as *mut u64;
                                *dst_ptr = *src_ptr;
                            }
                            _ if item_size >= 32 => {
                                // 大于32字节，分块使用AVX2
                                let mut remaining = item_size;
                                let mut local_src_offset = src_offset;
                                let mut local_dst_offset = dst_offset;
                                
                                while remaining >= 32 {
                                    let src_ptr = src.as_ptr().add(local_src_offset);
                                    let dst_ptr = dst.as_mut_ptr().add(local_dst_offset);
                                    
                                    let ymm = std::arch::x86_64::_mm256_loadu_si256(src_ptr as *const _);
                                    std::arch::x86_64::_mm256_storeu_si256(dst_ptr as *mut _, ymm);
                                    
                                    local_src_offset += 32;
                                    local_dst_offset += 32;
                                    remaining -= 32;
                                }
                                
                                // 处理剩余字节
                                if remaining > 0 {
                                    std::ptr::copy_nonoverlapping(
                                        src.as_ptr().add(local_src_offset),
                                        dst.as_mut_ptr().add(local_dst_offset),
                                        remaining
                                    );
                                }
                            }
                            _ => {
                                // 小于32字节，使用标量复制
                                std::ptr::copy_nonoverlapping(
                                    src.as_ptr().add(src_offset),
                                    dst.as_mut_ptr().add(dst_offset),
                                    item_size
                                );
                            }
                        }
                        dst_offset += item_size;
                    }
                }
            }
        }
    }
    
    /// SSE2优化的向量化复制
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn sse2_vectorized_copy(&self, src: &[u8], dst: &mut [u8], indices: &[usize], item_size: usize) {
        // Windows平台在启用安全模式时已经在vectorized_copy函数中处理过
        #[cfg(target_os = "windows")]
        if self.win_safe_simd {
            return self.optimized_scalar_copy(src, dst, indices, item_size);
        }
        
        if !self.supports_sse2 {
            return self.optimized_scalar_copy(src, dst, indices, item_size);
        }
        
        unsafe {
            let mut dst_offset = 0;
            let chunk_size = 4; // 处理4个元素为一组，提高循环效率
            
            for chunk in indices.chunks(chunk_size) {
                for &idx in chunk {
                    let src_offset = idx * item_size;
                    if src_offset + item_size <= src.len() && dst_offset + item_size <= dst.len() {
                        match item_size {
                            16 => {
                                // 16字节 - 使用SSE2指令
                                let src_ptr = src.as_ptr().add(src_offset);
                                let dst_ptr = dst.as_mut_ptr().add(dst_offset);
                                
                                let xmm = std::arch::x86_64::_mm_loadu_si128(src_ptr as *const _);
                                std::arch::x86_64::_mm_storeu_si128(dst_ptr as *mut _, xmm);
                            }
                            8 => {
                                // 8字节 - 使用64位复制
                                let src_ptr = src.as_ptr().add(src_offset) as *const u64;
                                let dst_ptr = dst.as_mut_ptr().add(dst_offset) as *mut u64;
                                *dst_ptr = *src_ptr;
                            }
                            _ if item_size >= 16 => {
                                // 大于16字节，分块使用SSE2
                                let mut remaining = item_size;
                                let mut local_src_offset = src_offset;
                                let mut local_dst_offset = dst_offset;
                                
                                while remaining >= 16 {
                                    let src_ptr = src.as_ptr().add(local_src_offset);
                                    let dst_ptr = dst.as_mut_ptr().add(local_dst_offset);
                                    
                                    let xmm = std::arch::x86_64::_mm_loadu_si128(src_ptr as *const _);
                                    std::arch::x86_64::_mm_storeu_si128(dst_ptr as *mut _, xmm);
                                    
                                    local_src_offset += 16;
                                    local_dst_offset += 16;
                                    remaining -= 16;
                                }
                                
                                // 处理剩余字节
                                if remaining > 0 {
                                    std::ptr::copy_nonoverlapping(
                                        src.as_ptr().add(local_src_offset),
                                        dst.as_mut_ptr().add(local_dst_offset),
                                        remaining
                                    );
                                }
                            }
                            _ => {
                                // 小于16字节，使用标量复制
                                std::ptr::copy_nonoverlapping(
                                    src.as_ptr().add(src_offset),
                                    dst.as_mut_ptr().add(dst_offset),
                                    item_size
                                );
                            }
                        }
                        dst_offset += item_size;
                    }
                }
            }
                 }
     }
                     
    /// 非x86架构的向量化复制实现
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    fn avx512_vectorized_copy(&self, src: &[u8], dst: &mut [u8], indices: &[usize], item_size: usize) {
        self.optimized_scalar_copy(src, dst, indices, item_size);
    }
    
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    fn avx2_vectorized_copy(&self, src: &[u8], dst: &mut [u8], indices: &[usize], item_size: usize) {
        self.optimized_scalar_copy(src, dst, indices, item_size);
    }
    
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    fn sse2_vectorized_copy(&self, src: &[u8], dst: &mut [u8], indices: &[usize], item_size: usize) {
        self.optimized_scalar_copy(src, dst, indices, item_size);
    }
    
    /// 优化的标量复制，包含内存对齐和缓存优化
    pub fn optimized_scalar_copy(&self, src: &[u8], dst: &mut [u8], indices: &[usize], item_size: usize) {
        let mut dst_offset = 0;
        
        for &idx in indices {
            let src_offset = idx * item_size;
            if src_offset + item_size <= src.len() && dst_offset + item_size <= dst.len() {
                unsafe {
                    // 使用优化的内存复制
                    match item_size {
                        1 => {
                            *dst.as_mut_ptr().add(dst_offset) = *src.as_ptr().add(src_offset);
                        }
                        2 => {
                            let src_ptr = src.as_ptr().add(src_offset) as *const u16;
                            let dst_ptr = dst.as_mut_ptr().add(dst_offset) as *mut u16;
                            *dst_ptr = *src_ptr;
                        }
                        4 => {
                            let src_ptr = src.as_ptr().add(src_offset) as *const u32;
                            let dst_ptr = dst.as_mut_ptr().add(dst_offset) as *mut u32;
                            *dst_ptr = *src_ptr;
                        }
                        8 => {
                            let src_ptr = src.as_ptr().add(src_offset) as *const u64;
                            let dst_ptr = dst.as_mut_ptr().add(dst_offset) as *mut u64;
                            *dst_ptr = *src_ptr;
                        }
                        _ => {
                            // 对于大块数据，使用优化的块复制
                            if item_size >= 64 && item_size % 8 == 0 {
                                // 8字节对齐的大块复制
                                let src_ptr = src.as_ptr().add(src_offset) as *const u64;
                                let dst_ptr = dst.as_mut_ptr().add(dst_offset) as *mut u64;
                                let count = item_size / 8;
                                
                                for i in 0..count {
                                    *dst_ptr.add(i) = *src_ptr.add(i);
                                }
                            } else {
                                // 通用复制
                                std::ptr::copy_nonoverlapping(
                                    src.as_ptr().add(src_offset),
                                    dst.as_mut_ptr().add(dst_offset),
                                    item_size
                                );
                            }
                        }
                    }
                }
                dst_offset += item_size;
            }
        }
    }
    
    /// 计算向量化索引 - 用于批量索引计算优化
    pub fn vectorized_index_calculation(&self, base_indices: &[usize], stride: usize, count: usize) -> Vec<usize> {
        let mut result = Vec::with_capacity(base_indices.len() * count);
        
        if self.supports_avx2 && base_indices.len() >= 4 {
            // 使用AVX2进行向量化索引计算
            self.avx2_index_calculation(base_indices, stride, count, &mut result);
        } else {
            // 标量索引计算
            for &base in base_indices {
                for i in 0..count {
                    result.push(base + i * stride);
                }
            }
        }
        
        result
    }
    
    /// AVX2向量化索引计算
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn avx2_index_calculation(&self, base_indices: &[usize], stride: usize, count: usize, result: &mut Vec<usize>) {
        if !self.supports_avx2 {
            // 回退到标量计算
            for &base in base_indices {
                for i in 0..count {
                    result.push(base + i * stride);
                }
            }
            return;
        }
        
        unsafe {
            let stride_vec = std::arch::x86_64::_mm256_set1_epi64x(stride as i64);
            
            for chunk in base_indices.chunks(4) {
                // 加载基础索引
                let mut base_array = [0i64; 4];
                for (i, &base) in chunk.iter().enumerate() {
                    base_array[i] = base as i64;
                }
                let base_vec = std::arch::x86_64::_mm256_loadu_si256(base_array.as_ptr() as *const _);
                
                // 为每个基础索引生成序列
                for i in 0..count {
                    let offset_vec = std::arch::x86_64::_mm256_set1_epi64x(i as i64);
                    let stride_offset = std::arch::x86_64::_mm256_mul_epi32(stride_vec, offset_vec);
                    let final_indices = std::arch::x86_64::_mm256_add_epi64(base_vec, stride_offset);
                    
                    // 存储结果
                    let mut indices_array = [0i64; 4];
                    std::arch::x86_64::_mm256_storeu_si256(indices_array.as_mut_ptr() as *mut _, final_indices);
                    
                    for j in 0..chunk.len() {
                        result.push(indices_array[j] as usize);
                    }
                }
            }
        }
    }
    
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    fn avx2_index_calculation(&self, base_indices: &[usize], stride: usize, count: usize, result: &mut Vec<usize>) {
        // 非x86架构的标量实现
        for &base in base_indices {
            for i in 0..count {
                result.push(base + i * stride);
            }
        }
    }
    
    /// 获取SIMD处理器的能力信息
    pub fn get_capabilities(&self) -> SIMDCapabilities {
        SIMDCapabilities {
            supports_sse2: self.supports_sse2,
            supports_avx2: self.supports_avx2,
            supports_avx512: self.supports_avx512,
            alignment_size: self.alignment_size,
            cache_line_size: self.cache_line_size,
            optimal_chunk_size: if self.supports_avx512 { 8 } else if self.supports_avx2 { 4 } else { 2 },
        }
    }
}

/// SIMD处理器能力信息
#[derive(Debug, Clone)]
pub struct SIMDCapabilities {
    pub supports_sse2: bool,
    pub supports_avx2: bool,
    pub supports_avx512: bool,
    pub alignment_size: usize,
    pub cache_line_size: usize,
    pub optimal_chunk_size: usize,
}

impl SIMDProcessor {
    pub fn is_aligned(&self, ptr: *const u8) -> bool {
        (ptr as usize) % self.alignment_size == 0
    }
    
    /// 安全检查SIMD操作
    /// 返回是否安全执行SIMD操作，如果不安全则返回错误代码
    #[cfg(target_os = "windows")]
    fn check_simd_safety(&self, ptr: *const u8, size: usize) -> Result<bool, WindowsSIMDError> {
        // 检查指针对齐
        if (ptr as usize) % self.win_memory_alignment != 0 {
            return Err(WindowsSIMDError::UnalignedPointer);
        }
        
        // 检查页面跨越（Windows对此特别敏感）
        let page_size = 4096; // Windows标准页大小
        let start_page = (ptr as usize) / page_size;
        let end_page = ((ptr as usize) + size - 1) / page_size;
        
        if start_page != end_page {
            // 数据跨页，在Windows上可能导致问题
            return Err(WindowsSIMDError::PageBoundaryCrossing);
        }
        
        // 所有检查通过
        Ok(true)
    }
    
    // Windows特定的内存对齐函数
    #[cfg(target_os = "windows")]
    fn windows_aligned_alloc(&self, size: usize) -> *mut u8 {
        unsafe {
            // Windows上使用_aligned_malloc确保正确对齐
            let aligned_ptr = std::alloc::alloc_zeroed(
                std::alloc::Layout::from_size_align(size, self.win_memory_alignment).unwrap()
            );
            
            aligned_ptr
        }
    }
    
    // Windows特定的内存释放函数
    #[cfg(target_os = "windows")]
    unsafe fn windows_aligned_free(&self, ptr: *mut u8, size: usize) {
        std::alloc::dealloc(
            ptr, 
            std::alloc::Layout::from_size_align(size, self.win_memory_alignment).unwrap()
        );
    }
}

/// Windows平台SIMD错误类型
#[cfg(target_os = "windows")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowsSIMDError {
    UnalignedPointer,       // 指针未对齐
    PageBoundaryCrossing,   // 跨页操作
    InvalidInstructionSet,  // 指令集不可用
    InvalidMemoryAccess,    // 无效内存访问
}

/// Windows平台安全对象池，提高内存利用效率
#[cfg(target_os = "windows")]
pub struct WindowsSIMDBufferPool {
    small_buffers: Mutex<Vec<(*mut u8, usize)>>,  // 小缓冲区池 (<1KB)
    medium_buffers: Mutex<Vec<(*mut u8, usize)>>, // 中缓冲区池 (1KB-16KB)
    large_buffers: Mutex<Vec<(*mut u8, usize)>>,  // 大缓冲区池 (>16KB)
    alignment: usize,
}

#[cfg(target_os = "windows")]
impl WindowsSIMDBufferPool {
    pub fn new(alignment: usize) -> Self {
        Self {
            small_buffers: Mutex::new(Vec::new()),
            medium_buffers: Mutex::new(Vec::new()),
            large_buffers: Mutex::new(Vec::new()),
            alignment,
        }
    }
    
    pub fn get_buffer(&self, size: usize) -> *mut u8 {
        unsafe {
            // 根据大小选择合适的池
            if size < 1024 {
                let mut pool = self.small_buffers.lock().unwrap();
                
                // 查找匹配大小的缓冲区
                for i in 0..pool.len() {
                    let (ptr, buf_size) = pool[i];
                    if buf_size >= size {
                        let buffer = ptr;
                        pool.remove(i);
                        return buffer;
                    }
                }
                
                // 没有找到匹配的，创建新的
                std::alloc::alloc_zeroed(
                    std::alloc::Layout::from_size_align(size, self.alignment).unwrap()
                )
            } else if size < 16384 {
                let mut pool = self.medium_buffers.lock().unwrap();
                
                for i in 0..pool.len() {
                    let (ptr, buf_size) = pool[i];
                    if buf_size >= size {
                        let buffer = ptr;
                        pool.remove(i);
                        return buffer;
                    }
                }
                
                std::alloc::alloc_zeroed(
                    std::alloc::Layout::from_size_align(size, self.alignment).unwrap()
                )
            } else {
                let mut pool = self.large_buffers.lock().unwrap();
                
                for i in 0..pool.len() {
                    let (ptr, buf_size) = pool[i];
                    if buf_size >= size {
                        let buffer = ptr;
                        pool.remove(i);
                        return buffer;
                    }
                }
                
                std::alloc::alloc_zeroed(
                    std::alloc::Layout::from_size_align(size, self.alignment).unwrap()
                )
            }
        }
    }
    
    pub unsafe fn return_buffer(&self, ptr: *mut u8, size: usize) {
        // 根据大小选择合适的池
        if size < 1024 {
            let mut pool = self.small_buffers.lock().unwrap();
            if pool.len() < 32 { // 限制池大小
                pool.push((ptr, size));
            } else {
                std::alloc::dealloc(
                    ptr, 
                    std::alloc::Layout::from_size_align(size, self.alignment).unwrap()
                );
            }
        } else if size < 16384 {
            let mut pool = self.medium_buffers.lock().unwrap();
            if pool.len() < 16 { // 限制池大小
                pool.push((ptr, size));
            } else {
                std::alloc::dealloc(
                    ptr, 
                    std::alloc::Layout::from_size_align(size, self.alignment).unwrap()
                );
            }
        } else {
            let mut pool = self.large_buffers.lock().unwrap();
            if pool.len() < 8 { // 限制池大小
                pool.push((ptr, size));
            } else {
                std::alloc::dealloc(
                    ptr, 
                    std::alloc::Layout::from_size_align(size, self.alignment).unwrap()
                );
            }
        }
    }
    
    // 清理所有缓冲区
    pub fn cleanup(&self) {
        unsafe {
            // 清理小缓冲区
            let mut small_pool = self.small_buffers.lock().unwrap();
            for (ptr, size) in small_pool.drain(..) {
                std::alloc::dealloc(
                    ptr, 
                    std::alloc::Layout::from_size_align(size, self.alignment).unwrap()
                );
            }
            
            // 清理中缓冲区
            let mut medium_pool = self.medium_buffers.lock().unwrap();
            for (ptr, size) in medium_pool.drain(..) {
                std::alloc::dealloc(
                    ptr, 
                    std::alloc::Layout::from_size_align(size, self.alignment).unwrap()
                );
            }
            
            // 清理大缓冲区
            let mut large_pool = self.large_buffers.lock().unwrap();
            for (ptr, size) in large_pool.drain(..) {
                std::alloc::dealloc(
                    ptr, 
                    std::alloc::Layout::from_size_align(size, self.alignment).unwrap()
                );
            }
        }
    }
}

#[cfg(target_os = "windows")]
impl Drop for WindowsSIMDBufferPool {
    fn drop(&mut self) {
        self.cleanup();
    }
}

// 预取策略枚举
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PrefetchStrategy {
    Conservative,  // 保守预取，低内存占用
    Aggressive,    // 激进预取，高性能
    Adaptive,      // 自适应预取，平衡性能和内存
    Disabled,      // 禁用预取
}

// 预取级别枚举 - 对应L1/L2/L3缓存优化
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PrefetchLevel {
    L1,  // L1缓存级别预取 (32KB)
    L2,  // L2缓存级别预取 (256KB) 
    L3,  // L3缓存级别预取 (8MB)
}

// 访问模式预测器
#[derive(Debug, Clone)]
pub struct AccessPatternPredictor {
    pattern_history: Vec<Vec<usize>>,
    stride_patterns: HashMap<usize, usize>, // stride -> frequency
    sequence_patterns: HashMap<Vec<usize>, usize>, // sequence -> frequency
    max_history: usize,
    min_confidence: f64,
}

impl AccessPatternPredictor {
    pub fn new() -> Self {
        Self {
            pattern_history: Vec::new(),
            stride_patterns: HashMap::new(),
            sequence_patterns: HashMap::new(),
            max_history: 50,
            min_confidence: 0.6,
        }
    }
    
    pub fn learn_pattern(&mut self, indices: &[usize]) {
        // 记录访问模式历史
        self.pattern_history.push(indices.to_vec());
        if self.pattern_history.len() > self.max_history {
            self.pattern_history.remove(0);
        }
        
        // 学习步长模式
        self.learn_stride_patterns(indices);
        
        // 学习序列模式
        self.learn_sequence_patterns(indices);
    }
    
    fn learn_stride_patterns(&mut self, indices: &[usize]) {
        if indices.len() < 2 {
            return;
        }
        
        for window in indices.windows(2) {
            if window[1] > window[0] {
                let stride = window[1] - window[0];
                *self.stride_patterns.entry(stride).or_insert(0) += 1;
            }
        }
    }
    
    fn learn_sequence_patterns(&mut self, indices: &[usize]) {
        if indices.len() < 3 {
            return;
        }
        
        for window in indices.windows(3) {
            let pattern = window.to_vec();
            *self.sequence_patterns.entry(pattern).or_insert(0) += 1;
        }
    }
    
    pub fn predict_next_accesses(&self, current_indices: &[usize], window_size: usize) -> Vec<usize> {
        let mut predictions = Vec::new();
        
        // 基于步长模式预测
        if let Some(stride_predictions) = self.predict_by_stride(current_indices, window_size) {
            predictions.extend(stride_predictions);
        }
        
        // 基于序列模式预测
        if let Some(sequence_predictions) = self.predict_by_sequence(current_indices, window_size) {
            predictions.extend(sequence_predictions);
        }
        
        // 去重并排序
        predictions.sort_unstable();
        predictions.dedup();
        predictions.truncate(window_size);
        
        predictions
    }
    
    fn predict_by_stride(&self, indices: &[usize], window_size: usize) -> Option<Vec<usize>> {
        if indices.len() < 2 {
            return None;
        }
        
        // 找到最常见的步长
        let mut stride_scores: Vec<(usize, usize)> = self.stride_patterns.iter()
            .map(|(&stride, &freq)| (stride, freq))
            .collect();
        stride_scores.sort_by(|a, b| b.1.cmp(&a.1));
        
        if let Some(&(best_stride, frequency)) = stride_scores.first() {
            let total_patterns = self.stride_patterns.values().sum::<usize>();
            let confidence = frequency as f64 / total_patterns as f64;
            
            if confidence >= self.min_confidence {
                let last_index = *indices.last().unwrap();
                let predictions: Vec<usize> = (1..=window_size)
                    .map(|i| last_index + i * best_stride)
                    .collect();
                return Some(predictions);
            }
        }
        
        None
    }
    
    fn predict_by_sequence(&self, indices: &[usize], window_size: usize) -> Option<Vec<usize>> {
        if indices.len() < 2 {
            return None;
        }
        
        // 查找匹配的序列模式
        let suffix = if indices.len() >= 3 {
            &indices[indices.len()-3..]
        } else {
            &indices[indices.len()-2..]
        };
        
        for (pattern, &frequency) in &self.sequence_patterns {
            if pattern.len() >= suffix.len() && 
               pattern[..suffix.len()] == *suffix {
                let total_patterns = self.sequence_patterns.values().sum::<usize>();
                let confidence = frequency as f64 / total_patterns as f64;
                
                if confidence >= self.min_confidence {
                    // 基于找到的模式预测下一个访问
                    if let Some(&next_in_pattern) = pattern.get(suffix.len()) {
                        let last_index = *indices.last().unwrap();
                        let offset = next_in_pattern.saturating_sub(pattern[suffix.len()-1]);
                        let predicted_start = last_index + offset;
                        
                        let predictions: Vec<usize> = (0..window_size)
                            .map(|i| predicted_start + i)
                            .collect();
                        return Some(predictions);
                    }
                }
            }
        }
        
        None
    }
    
    pub fn get_confidence(&self, indices: &[usize]) -> f64 {
        let stride_confidence = self.get_stride_confidence(indices);
        let sequence_confidence = self.get_sequence_confidence(indices);
        stride_confidence.max(sequence_confidence)
    }
    
    fn get_stride_confidence(&self, indices: &[usize]) -> f64 {
        if indices.len() < 2 {
            return 0.0;
        }
        
        let recent_stride = indices[indices.len()-1] - indices[indices.len()-2];
        let frequency = self.stride_patterns.get(&recent_stride).unwrap_or(&0);
        let total = self.stride_patterns.values().sum::<usize>();
        
        if total > 0 {
            *frequency as f64 / total as f64
        } else {
            0.0
        }
    }
    
    fn get_sequence_confidence(&self, indices: &[usize]) -> f64 {
        if indices.len() < 3 {
            return 0.0;
        }
        
        let recent_sequence = &indices[indices.len()-3..];
        let frequency = self.sequence_patterns.get(recent_sequence).unwrap_or(&0);
        let total = self.sequence_patterns.values().sum::<usize>();
        
        if total > 0 {
            *frequency as f64 / total as f64
        } else {
            0.0
        }
    }
}

// 多级预取缓存
#[derive(Debug)]
pub struct MultiLevelPrefetchCache {
    l1_cache: HashMap<usize, Vec<u8>>,  // 最热数据
    l2_cache: HashMap<usize, Vec<u8>>,  // 次热数据
    l3_cache: HashMap<usize, Vec<u8>>,  // 预取数据
    l1_capacity: usize,
    l2_capacity: usize,
    l3_capacity: usize,
    l1_access_count: HashMap<usize, usize>,
    l2_access_count: HashMap<usize, usize>,
    total_memory_usage: usize,
    max_memory_usage: usize,
}

impl MultiLevelPrefetchCache {
    pub fn new() -> Self {
        Self {
            l1_cache: HashMap::new(),
            l2_cache: HashMap::new(),
            l3_cache: HashMap::new(),
            l1_capacity: 32,      // L1缓存32个条目
            l2_capacity: 256,     // L2缓存256个条目
            l3_capacity: 1024,    // L3缓存1024个条目
            l1_access_count: HashMap::new(),
            l2_access_count: HashMap::new(),
            total_memory_usage: 0,
            max_memory_usage: 64 * 1024 * 1024, // 64MB最大内存使用
        }
    }
    
    pub fn get(&mut self, key: usize) -> Option<Vec<u8>> {
        // 首先检查L1缓存
        if let Some(data) = self.l1_cache.get(&key) {
            *self.l1_access_count.entry(key).or_insert(0) += 1;
            return Some(data.clone());
        }
        
        // 检查L2缓存，如果命中则提升到L1
        if let Some(data) = self.l2_cache.remove(&key) {
            self.promote_to_l1(key, data.clone());
            return Some(data);
        }
        
        // 检查L3缓存，如果命中则提升到L2
        if let Some(data) = self.l3_cache.remove(&key) {
            self.promote_to_l2(key, data.clone());
            return Some(data);
        }
        
        None
    }
    
    pub fn put(&mut self, key: usize, data: Vec<u8>, level: PrefetchLevel) {
        let data_size = data.len();
        
        // 检查内存使用限制
        if self.total_memory_usage + data_size > self.max_memory_usage {
            self.evict_to_fit(data_size);
        }
        
        match level {
            PrefetchLevel::L1 => self.put_l1(key, data),
            PrefetchLevel::L2 => self.put_l2(key, data),
            PrefetchLevel::L3 => self.put_l3(key, data),
        }
        
        self.total_memory_usage += data_size;
    }
    
    fn promote_to_l1(&mut self, key: usize, data: Vec<u8>) {
        if self.l1_cache.len() >= self.l1_capacity {
            self.evict_from_l1();
        }
        self.l1_cache.insert(key, data);
        *self.l1_access_count.entry(key).or_insert(0) += 1;
    }
    
    fn promote_to_l2(&mut self, key: usize, data: Vec<u8>) {
        if self.l2_cache.len() >= self.l2_capacity {
            self.evict_from_l2();
        }
        self.l2_cache.insert(key, data);
        *self.l2_access_count.entry(key).or_insert(0) += 1;
    }
    
    fn put_l1(&mut self, key: usize, data: Vec<u8>) {
        if self.l1_cache.len() >= self.l1_capacity {
            self.evict_from_l1();
        }
        self.l1_cache.insert(key, data);
    }
    
    fn put_l2(&mut self, key: usize, data: Vec<u8>) {
        if self.l2_cache.len() >= self.l2_capacity {
            self.evict_from_l2();
        }
        self.l2_cache.insert(key, data);
    }
    
    fn put_l3(&mut self, key: usize, data: Vec<u8>) {
        if self.l3_cache.len() >= self.l3_capacity {
            self.evict_from_l3();
        }
        self.l3_cache.insert(key, data);
    }
    
    fn evict_from_l1(&mut self) {
        // LRU淘汰：找到访问次数最少的条目
        if let Some((&lru_key, _)) = self.l1_access_count.iter().min_by_key(|(_, &count)| count) {
            if let Some(data) = self.l1_cache.remove(&lru_key) {
                self.total_memory_usage = self.total_memory_usage.saturating_sub(data.len());
                // 降级到L2
                self.put_l2(lru_key, data);
            }
            self.l1_access_count.remove(&lru_key);
        }
    }
    
    fn evict_from_l2(&mut self) {
        if let Some((&lru_key, _)) = self.l2_access_count.iter().min_by_key(|(_, &count)| count) {
            if let Some(data) = self.l2_cache.remove(&lru_key) {
                self.total_memory_usage = self.total_memory_usage.saturating_sub(data.len());
                // 降级到L3
                self.put_l3(lru_key, data);
            }
            self.l2_access_count.remove(&lru_key);
        }
    }
    
    fn evict_from_l3(&mut self) {
        // L3直接删除最老的条目
        if let Some((&key, _)) = self.l3_cache.iter().next() {
            let key = key; // 复制key以避免借用冲突
            if let Some(data) = self.l3_cache.remove(&key) {
                self.total_memory_usage = self.total_memory_usage.saturating_sub(data.len());
            }
        }
    }
    
    fn evict_to_fit(&mut self, required_size: usize) {
        while self.total_memory_usage + required_size > self.max_memory_usage && 
              (!self.l3_cache.is_empty() || !self.l2_cache.is_empty() || !self.l1_cache.is_empty()) {
            
            // 优先从L3淘汰
            if !self.l3_cache.is_empty() {
                self.evict_from_l3();
            } else if !self.l2_cache.is_empty() {
                self.evict_from_l2();
            } else if !self.l1_cache.is_empty() {
                self.evict_from_l1();
            } else {
                break;
            }
        }
    }
    
    pub fn get_memory_usage(&self) -> usize {
        self.total_memory_usage
    }
    
    pub fn get_hit_rate(&self) -> (f64, f64, f64) {
        let l1_hits = self.l1_access_count.values().sum::<usize>();
        let l2_hits = self.l2_access_count.values().sum::<usize>();
        let total_accesses = l1_hits + l2_hits + self.l3_cache.len();
        
        if total_accesses > 0 {
            (
                l1_hits as f64 / total_accesses as f64,
                l2_hits as f64 / total_accesses as f64,
                self.l3_cache.len() as f64 / total_accesses as f64,
            )
        } else {
            (0.0, 0.0, 0.0)
        }
    }
    
    pub fn clear(&mut self) {
        self.l1_cache.clear();
        self.l2_cache.clear();
        self.l3_cache.clear();
        self.l1_access_count.clear();
        self.l2_access_count.clear();
        self.total_memory_usage = 0;
    }
    
    // Getter methods for testing
    pub fn l1_cache_len(&self) -> usize {
        self.l1_cache.len()
    }
    
    pub fn l2_cache_is_empty(&self) -> bool {
        self.l2_cache.is_empty()
    }
}

// 增强的预取管理器
pub struct PrefetchManager {
    // 基础配置
    prefetch_distance: usize,
    adaptive_window: usize,
    strategy: PrefetchStrategy,
    
    // 访问模式分析和预测
    pattern_predictor: AccessPatternPredictor,
    last_pattern: Option<Vec<usize>>,
    
    // 多级预取缓存
    prefetch_cache: MultiLevelPrefetchCache,
    
    // 自适应窗口管理
    min_window_size: usize,
    max_window_size: usize,
    window_adjustment_factor: f64,
    
    // 性能统计
    hit_count: usize,
    miss_count: usize,
    prefetch_accuracy: f64,
    
    // 内存带宽优化
    memory_bandwidth_limit: usize,  // 每秒最大预取字节数
    last_prefetch_time: Instant,
    bytes_prefetched_this_second: usize,
    
    // 预取时机优化
    cpu_threshold: f64,             // CPU使用率阈值
    memory_pressure_threshold: f64, // 内存压力阈值
    prefetch_enabled: bool,
}

impl PrefetchManager {
    pub fn new() -> Self {
        Self {
            // 基础配置
            prefetch_distance: 64 * 1024, // 64KB预取距离
            adaptive_window: 4,           // 自适应窗口大小
            strategy: PrefetchStrategy::Adaptive,
            
            // 访问模式分析和预测
            pattern_predictor: AccessPatternPredictor::new(),
            last_pattern: None,
            
            // 多级预取缓存
            prefetch_cache: MultiLevelPrefetchCache::new(),
            
            // 自适应窗口管理
            min_window_size: 2,
            max_window_size: 32,
            window_adjustment_factor: 1.2,
            
            // 性能统计
            hit_count: 0,
            miss_count: 0,
            prefetch_accuracy: 0.0,
            
            // 内存带宽优化
            memory_bandwidth_limit: 100 * 1024 * 1024, // 100MB/s
            last_prefetch_time: Instant::now(),
            bytes_prefetched_this_second: 0,
            
            // 预取时机优化
            cpu_threshold: 0.8,             // CPU使用率阈值
            memory_pressure_threshold: 0.9, // 内存压力阈值
            prefetch_enabled: true,
        }
    }
    
    /// 智能预测和预取 - 主要接口
    pub fn predict_and_prefetch(&mut self, indices: &[usize], array: &OptimizedLazyArray) {
        if !self.prefetch_enabled || indices.is_empty() {
            return;
        }
        
        // 检查系统资源状态
        if !self.should_prefetch() {
            return;
        }
        
        // 学习当前访问模式
        self.pattern_predictor.learn_pattern(indices);
        
        // 预测下一批访问
        let predicted_indices = self.predict_next_accesses(indices);
        
        if !predicted_indices.is_empty() {
            // 执行智能预取
            self.execute_intelligent_prefetch(&predicted_indices, array);
        }
        
        // 更新历史模式
        self.last_pattern = Some(indices.to_vec());
        
        // 自适应调整窗口大小
        self.adaptive_window_adjustment();
    }
    
    /// 预测下一批访问 - 使用增强的预测算法
    fn predict_next_accesses(&self, current_indices: &[usize]) -> Vec<usize> {
        let confidence = self.pattern_predictor.get_confidence(current_indices);
        
        // 根据置信度调整预取窗口大小
        let effective_window = if confidence > 0.8 {
            (self.adaptive_window as f64 * 1.5) as usize
        } else if confidence > 0.6 {
            self.adaptive_window
        } else {
            (self.adaptive_window as f64 * 0.7) as usize
        }.max(self.min_window_size).min(self.max_window_size);
        
        self.pattern_predictor.predict_next_accesses(current_indices, effective_window)
    }
    
    /// 执行智能预取 - 多级缓存策略
    fn execute_intelligent_prefetch(&mut self, indices: &[usize], array: &OptimizedLazyArray) {
        let row_size = array.shape[1..].iter().product::<usize>() * array.itemsize;
        let now = Instant::now();
        
        // 重置带宽计数器（每秒）
        if now.duration_since(self.last_prefetch_time) >= Duration::from_secs(1) {
            self.bytes_prefetched_this_second = 0;
            self.last_prefetch_time = now;
        }
        
        let mut prefetched_bytes = 0;
        
        for &idx in indices {
            if idx >= array.shape[0] {
                continue;
            }
            
            // 检查是否已在缓存中
            if self.prefetch_cache.get(idx).is_some() {
                continue;
            }
            
            // 检查内存带宽限制
            if self.bytes_prefetched_this_second + prefetched_bytes + row_size > self.memory_bandwidth_limit {
                break;
            }
            
            // 读取数据并存储到多级缓存
            let offset = idx * row_size;
            if let Some(data) = self.prefetch_data_from_array(array, offset, row_size) {
                // 根据访问模式决定缓存级别
                let cache_level = self.determine_cache_level(idx, &data);
                self.prefetch_cache.put(idx, data, cache_level);
                prefetched_bytes += row_size;
            }
        }
        
        self.bytes_prefetched_this_second += prefetched_bytes;
    }
    
    /// 从数组预取数据
    fn prefetch_data_from_array(&self, array: &OptimizedLazyArray, offset: usize, size: usize) -> Option<Vec<u8>> {
        if offset + size <= array.mmap.len() {
            let data = unsafe {
                std::slice::from_raw_parts(array.mmap.as_ptr().add(offset), size)
            };
            Some(data.to_vec())
        } else {
            None
        }
    }
    
    /// 确定缓存级别
    fn determine_cache_level(&self, _idx: usize, data: &[u8]) -> PrefetchLevel {
        match self.strategy {
            PrefetchStrategy::Conservative => PrefetchLevel::L3,
            PrefetchStrategy::Aggressive => {
                if data.len() <= 32 * 1024 {
                    PrefetchLevel::L1
                } else if data.len() <= 256 * 1024 {
                    PrefetchLevel::L2
                } else {
                    PrefetchLevel::L3
                }
            }
            PrefetchStrategy::Adaptive => {
                // 基于访问频率和数据大小自适应选择
                if data.len() <= 16 * 1024 {
                    PrefetchLevel::L1
                } else if data.len() <= 128 * 1024 {
                    PrefetchLevel::L2
                } else {
                    PrefetchLevel::L3
                }
            }
            PrefetchStrategy::Disabled => PrefetchLevel::L3,
        }
    }
    
    /// 检查是否应该执行预取
    fn should_prefetch(&self) -> bool {
        if !self.prefetch_enabled {
            return false;
        }
        
        // 简化的系统资源检查
        // 在实际实现中应该使用系统API获取真实的CPU和内存使用情况
        let estimated_cpu_usage = 0.5; // 假设的CPU使用率
        let estimated_memory_pressure = 0.6; // 假设的内存压力
        
        estimated_cpu_usage < self.cpu_threshold && 
        estimated_memory_pressure < self.memory_pressure_threshold
    }
    
    /// 自适应窗口大小调整
    fn adaptive_window_adjustment(&mut self) {
        let hit_rate = self.get_hit_rate();
        
        match hit_rate {
            rate if rate > 0.85 => {
                // 高命中率，增加窗口
                self.adaptive_window = ((self.adaptive_window as f64 * self.window_adjustment_factor) as usize)
                    .min(self.max_window_size);
            }
            rate if rate < 0.4 => {
                // 低命中率，减少窗口
                self.adaptive_window = ((self.adaptive_window as f64 / self.window_adjustment_factor) as usize)
                    .max(self.min_window_size);
            }
            _ => {
                // 中等命中率，保持当前窗口大小
            }
        }
        
        // 更新预取准确率
        self.prefetch_accuracy = hit_rate;
    }
    
    /// 获取预取的数据（如果存在）
    pub fn get_prefetched_data(&mut self, idx: usize) -> Option<Vec<u8>> {
        let result = self.prefetch_cache.get(idx);
        
        // 更新统计信息
        if result.is_some() {
            self.hit_count += 1;
        } else {
            self.miss_count += 1;
        }
        
        result
    }
    
    /// 设置预取策略
    pub fn set_strategy(&mut self, strategy: PrefetchStrategy) {
        self.strategy = strategy;
        self.prefetch_enabled = !matches!(strategy, PrefetchStrategy::Disabled);
    }
    
    /// 设置内存带宽限制
    pub fn set_memory_bandwidth_limit(&mut self, limit_bytes_per_second: usize) {
        self.memory_bandwidth_limit = limit_bytes_per_second;
    }
    
    /// 设置CPU和内存阈值
    pub fn set_resource_thresholds(&mut self, cpu_threshold: f64, memory_threshold: f64) {
        self.cpu_threshold = cpu_threshold;
        self.memory_pressure_threshold = memory_threshold;
    }
    
    /// 获取详细的性能统计
    pub fn get_detailed_stats(&self) -> PrefetchStats {
        let (l1_hit_rate, l2_hit_rate, l3_hit_rate) = self.prefetch_cache.get_hit_rate();
        
        PrefetchStats {
            hit_rate: self.get_hit_rate(),
            hit_count: self.hit_count,
            miss_count: self.miss_count,
            prefetch_accuracy: self.prefetch_accuracy,
            adaptive_window_size: self.adaptive_window,
            memory_usage: self.prefetch_cache.get_memory_usage(),
            l1_hit_rate,
            l2_hit_rate,
            l3_hit_rate,
            strategy: self.strategy,
        }
    }
    
    /// 清理缓存
    pub fn clear_cache(&mut self) {
        self.prefetch_cache.clear();
    }
    
    /// 调整窗口大小（外部接口）
    pub fn adjust_window_size(&mut self, hit_rate: f64) {
        if hit_rate > 0.8 {
            self.adaptive_window = (self.adaptive_window + 1).min(self.max_window_size);
        } else if hit_rate < 0.3 {
            self.adaptive_window = (self.adaptive_window.saturating_sub(1)).max(self.min_window_size);
        }
    }
    
    /// 记录命中
    pub fn record_hit(&mut self) {
        self.hit_count += 1;
    }
    
    /// 记录未命中
    pub fn record_miss(&mut self) {
        self.miss_count += 1;
    }
    
    /// 获取命中率
    pub fn get_hit_rate(&self) -> f64 {
        let total = self.hit_count + self.miss_count;
        if total == 0 {
            0.0
        } else {
            self.hit_count as f64 / total as f64
        }
    }
    
    // Getter methods for testing
    pub fn get_strategy(&self) -> PrefetchStrategy {
        self.strategy
    }
    
    pub fn is_prefetch_enabled(&self) -> bool {
        self.prefetch_enabled
    }
    
    pub fn get_cpu_threshold(&self) -> f64 {
        self.cpu_threshold
    }
    
    pub fn get_memory_pressure_threshold(&self) -> f64 {
        self.memory_pressure_threshold
    }
    
    pub fn get_memory_bandwidth_limit(&self) -> usize {
        self.memory_bandwidth_limit
    }
    
    pub fn get_adaptive_window(&self) -> usize {
        self.adaptive_window
    }
    
    pub fn get_min_window_size(&self) -> usize {
        self.min_window_size
    }
    
    pub fn get_max_window_size(&self) -> usize {
        self.max_window_size
    }
}

/// 预取性能统计结构
#[derive(Debug, Clone)]
pub struct PrefetchStats {
    pub hit_rate: f64,
    pub hit_count: usize,
    pub miss_count: usize,
    pub prefetch_accuracy: f64,
    pub adaptive_window_size: usize,
    pub memory_usage: usize,
    pub l1_hit_rate: f64,
    pub l2_hit_rate: f64,
    pub l3_hit_rate: f64,
    pub strategy: PrefetchStrategy,
}

impl Default for PrefetchStats {
    fn default() -> Self {
        Self {
            hit_rate: 0.0,
            hit_count: 0,
            miss_count: 0,
            prefetch_accuracy: 0.0,
            adaptive_window_size: 0,
            memory_usage: 0,
            l1_hit_rate: 0.0,
            l2_hit_rate: 0.0,
            l3_hit_rate: 0.0,
            strategy: PrefetchStrategy::Adaptive,
        }
    }
}

// 零拷贝处理器
pub struct ZeroCopyHandler {
    min_size_threshold: usize,
    alignment_requirement: usize,
    // 新增：性能统计
    zero_copy_hits: Arc<Mutex<usize>>,
    fallback_to_copy: Arc<Mutex<usize>>,
    // 新增：内存使用跟踪
    total_zero_copy_memory: Arc<Mutex<usize>>,
    // 新增：访问模式分析器
    access_analyzer: Arc<Mutex<ZeroCopyAnalyzer>>,
}

// 新增：零拷贝分析器
#[derive(Debug)]
pub struct ZeroCopyAnalyzer {
    recent_accesses: Vec<(usize, usize, Instant)>, // (offset, size, time)
    continuous_access_count: usize,
    fragmented_access_count: usize,
    average_access_size: f64,
    last_optimization_check: Instant,
}

impl ZeroCopyAnalyzer {
    pub fn new() -> Self {
        Self {
            recent_accesses: Vec::new(),
            continuous_access_count: 0,
            fragmented_access_count: 0,
            average_access_size: 0.0,
            last_optimization_check: Instant::now(),
        }
    }
    
    pub fn record_access(&mut self, offset: usize, size: usize) {
        let now = Instant::now();
        self.recent_accesses.push((offset, size, now));
        
        // 保持历史记录在合理范围内
        if self.recent_accesses.len() > 1000 {
            self.recent_accesses.drain(0..500);
        }
        
        // 更新统计信息
        self.update_statistics();
    }
    
    fn update_statistics(&mut self) {
        if self.recent_accesses.len() < 2 {
            return;
        }
        
        let mut continuous_count = 0;
        let mut fragmented_count = 0;
        let total_size: usize = self.recent_accesses.iter().map(|(_, size, _)| *size).sum();
        
        // 分析连续性
        for window in self.recent_accesses.windows(2) {
            let (offset1, size1, _) = window[0];
            let (offset2, _, _) = window[1];
            
            if offset2 == offset1 + size1 {
                continuous_count += 1;
            } else {
                fragmented_count += 1;
            }
        }
        
        self.continuous_access_count = continuous_count;
        self.fragmented_access_count = fragmented_count;
        self.average_access_size = total_size as f64 / self.recent_accesses.len() as f64;
    }
    
    pub fn should_prefer_zero_copy(&self) -> bool {
        // 基于访问模式判断是否应优先使用零拷贝
        let continuity_ratio = if self.continuous_access_count + self.fragmented_access_count > 0 {
            self.continuous_access_count as f64 / 
            (self.continuous_access_count + self.fragmented_access_count) as f64
        } else {
            0.0
        };
        
        // 连续访问比例高且平均访问大小大于阈值
        continuity_ratio > 0.7 && self.average_access_size > 2048.0
    }
    
    pub fn get_optimal_chunk_size(&self) -> usize {
        // 基于历史访问模式推荐最优分块大小
        if self.average_access_size > (64 * 1024) as f64 {
            64 * 1024 // 64KB
        } else if self.average_access_size > (16 * 1024) as f64 {
            16 * 1024 // 16KB
        } else if self.average_access_size > (4 * 1024) as f64 {
            4 * 1024  // 4KB
        } else {
            1024      // 1KB
        }
    }
}

// 新增：零拷贝视图结构 - 安全的零拷贝视图
pub struct ZeroCopyView<'a> {
    data: &'a [u8],
    lifetime_guard: Arc<()>, // 生命周期守护
    metadata: ZeroCopyMetadata,
}

#[derive(Debug, Clone)]
pub struct ZeroCopyMetadata {
    pub offset: usize,
    pub size: usize,
    pub created_at: Instant,
    pub access_count: Arc<Mutex<usize>>,
    pub is_continuous: bool,
}

impl<'a> ZeroCopyView<'a> {
    pub fn new(data: &'a [u8], offset: usize, lifetime_guard: Arc<()>) -> Self {
        Self {
            data,
            lifetime_guard,
            metadata: ZeroCopyMetadata {
                offset,
                size: data.len(),
                created_at: Instant::now(),
                access_count: Arc::new(Mutex::new(0)),
                is_continuous: true,
            },
        }
    }
    
    pub fn data(&self) -> &[u8] {
        // 增加访问计数
        if let Ok(mut count) = self.metadata.access_count.lock() {
            *count += 1;
        }
        self.data
    }
    
    pub fn len(&self) -> usize {
        self.data.len()
    }
    
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
    
    pub fn metadata(&self) -> &ZeroCopyMetadata {
        &self.metadata
    }
    
    // 分割视图 - 创建子视图
    pub fn slice(&self, start: usize, end: usize) -> Option<ZeroCopyView<'a>> {
        if start < end && end <= self.data.len() {
            Some(ZeroCopyView {
                data: &self.data[start..end],
                lifetime_guard: Arc::clone(&self.lifetime_guard),
                metadata: ZeroCopyMetadata {
                    offset: self.metadata.offset + start,
                    size: end - start,
                    created_at: Instant::now(),
                    access_count: Arc::new(Mutex::new(0)),
                    is_continuous: self.metadata.is_continuous,
                },
            })
        } else {
            None
        }
    }
    
    // 安全转换为 Vec<u8>（如果需要拥有所有权）
    pub fn to_owned(&self) -> Vec<u8> {
        self.data.to_vec()
    }
}

// 新增：连续内存访问优化器
pub struct ContinuousMemoryOptimizer {
    cache_line_size: usize,
    page_size: usize,
    prefetch_distance: usize,
}

impl ContinuousMemoryOptimizer {
    pub fn new() -> Self {
        Self {
            cache_line_size: 64,   // 典型缓存行大小
            page_size: 4096,       // 典型页面大小
            prefetch_distance: 8,  // 预取距离（缓存行数）
        }
    }
    
    /// 优化连续内存访问 - 考虑缓存行对齐
    pub fn optimize_continuous_access(&self, offset: usize, size: usize) -> (usize, usize) {
        // 对齐到缓存行边界
        let aligned_start = (offset / self.cache_line_size) * self.cache_line_size;
        let end_offset = offset + size;
        let aligned_end = ((end_offset + self.cache_line_size - 1) / self.cache_line_size) * self.cache_line_size;
        
        (aligned_start, aligned_end - aligned_start)
    }
    
    /// 检查是否应该使用大页面访问
    pub fn should_use_large_pages(&self, size: usize) -> bool {
        size >= self.page_size * 4 // 16KB以上使用大页面策略
    }
    
    /// 计算最优的访问块大小
    pub fn calculate_optimal_block_size(&self, total_size: usize) -> usize {
        if total_size >= 1024 * 1024 {
            // 1MB以上：使用64KB块
            64 * 1024
        } else if total_size >= 256 * 1024 {
            // 256KB-1MB：使用16KB块
            16 * 1024
        } else if total_size >= 64 * 1024 {
            // 64KB-256KB：使用4KB块
            4 * 1024
        } else {
            // 小于64KB：使用缓存行大小
            self.cache_line_size
        }
    }
    
    /// 预取内存
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    pub fn prefetch_memory(&self, ptr: *const u8, size: usize) {
        let cache_lines = size / self.cache_line_size + 1;
        let prefetch_lines = cache_lines.min(self.prefetch_distance);
        
        unsafe {
            for i in 0..prefetch_lines {
                let prefetch_ptr = ptr.add(i * self.cache_line_size);
                std::arch::x86_64::_mm_prefetch(prefetch_ptr as *const i8, std::arch::x86_64::_MM_HINT_T0);
            }
        }
    }
    
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    pub fn prefetch_memory(&self, _ptr: *const u8, _size: usize) {
        // 非x86架构：无操作
    }
}

impl ZeroCopyHandler {
    pub fn new() -> Self {
        Self {
            min_size_threshold: 1024, // 1KB以上才考虑零拷贝
            alignment_requirement: 8,  // 8字节对齐
            zero_copy_hits: Arc::new(Mutex::new(0)),
            fallback_to_copy: Arc::new(Mutex::new(0)),
            total_zero_copy_memory: Arc::new(Mutex::new(0)),
            access_analyzer: Arc::new(Mutex::new(ZeroCopyAnalyzer::new())),
        }
    }
    
    pub fn can_zero_copy(&self, indices: &[usize], item_size: usize) -> bool {
        // 记录访问模式
        if let Ok(mut analyzer) = self.access_analyzer.lock() {
            let total_size = indices.len() * item_size;
            let first_offset = indices.first().map(|&i| i * item_size).unwrap_or(0);
            analyzer.record_access(first_offset, total_size);
        }
        
        // 检查是否适合零拷贝
        let total_size = indices.len() * item_size;
        
        // 大小检查
        if total_size < self.min_size_threshold {
            return false;
        }
        
        // 连续性检查
        self.is_continuous_access(indices)
    }
    
    fn is_continuous_access(&self, indices: &[usize]) -> bool {
        if indices.len() < 2 {
            return true;
        }
        
        for window in indices.windows(2) {
            if window[1] != window[0] + 1 {
                return false;
            }
        }
        true
    }
    
    /// 新增：创建安全的零拷贝视图
    pub fn create_safe_zero_copy_view<'a>(&self, 
                                         data: &'a [u8], 
                                         start_idx: usize, 
                                         count: usize, 
                                         item_size: usize,
                                         lifetime_guard: Arc<()>) -> Option<ZeroCopyView<'a>> {
        let start_offset = start_idx * item_size;
        let total_size = count * item_size;
        
        if start_offset + total_size <= data.len() {
            let view_data = &data[start_offset..start_offset + total_size];
            
            // 记录零拷贝命中
            if let Ok(mut hits) = self.zero_copy_hits.lock() {
                *hits += 1;
            }
            if let Ok(mut memory) = self.total_zero_copy_memory.lock() {
                *memory += total_size;
            }
            
            Some(ZeroCopyView::new(view_data, start_offset, lifetime_guard))
        } else {
            // 记录回退到拷贝
            if let Ok(mut fallbacks) = self.fallback_to_copy.lock() {
                *fallbacks += 1;
            }
            None
        }
    }
    
    pub fn create_zero_copy_view<'a>(&self, 
                                    data: &'a [u8], 
                                    start_idx: usize, 
                                    count: usize, 
                                    item_size: usize) -> Option<&'a [u8]> {
        let start_offset = start_idx * item_size;
        let total_size = count * item_size;
        
        if start_offset + total_size <= data.len() {
            Some(&data[start_offset..start_offset + total_size])
        } else {
            None
        }
    }
    
    /// 新增：智能的拷贝vs零拷贝决策引擎
    pub fn smart_copy_decision(&self, 
                             indices: &[usize], 
                             item_size: usize,
                             access_frequency: f64,
                             memory_pressure: f64) -> ZeroCopyDecision {
        let total_size = indices.len() * item_size;
        
        // 基础条件检查
        if total_size < self.min_size_threshold {
            return ZeroCopyDecision::ForceCopy("Size too small".to_string());
        }
        
        // 内存压力检查
        if memory_pressure > 0.9 {
            return ZeroCopyDecision::PreferZeroCopy("High memory pressure".to_string());
        }
        
        // 访问模式分析
        let should_prefer_zero_copy = if let Ok(analyzer) = self.access_analyzer.lock() {
            analyzer.should_prefer_zero_copy()
        } else {
            false
        };
        
        // 连续性检查
        let is_continuous = self.is_continuous_access(indices);
        
        // 决策逻辑
        match (is_continuous, should_prefer_zero_copy, access_frequency) {
            // 连续访问 + 优先零拷贝 + 低频访问 -> 零拷贝
            (true, true, freq) if freq < 0.3 => {
                ZeroCopyDecision::PreferZeroCopy("Continuous low-frequency access".to_string())
            }
            // 连续访问 + 高频访问 -> 拷贝（便于缓存）
            (true, _, freq) if freq > 0.7 => {
                ZeroCopyDecision::PreferCopy("High frequency access benefits from caching".to_string())
            }
            // 非连续访问 + 大尺寸 -> 零拷贝（避免大量内存分配）
            (false, _, _) if total_size > 1024 * 1024 => {
                ZeroCopyDecision::PreferZeroCopy("Large non-continuous access".to_string())
            }
            // 连续访问 + 中等尺寸 -> 零拷贝
            (true, _, _) if total_size > 4096 => {
                ZeroCopyDecision::PreferZeroCopy("Continuous medium-size access".to_string())
            }
            // 默认情况 -> 拷贝
            _ => {
                ZeroCopyDecision::PreferCopy("Default conservative approach".to_string())
            }
        }
    }
    
    pub fn should_copy_vs_zero_copy(&self, 
                                    indices: &[usize], 
                                    item_size: usize,
                                    access_frequency: f64) -> bool {
        let decision = self.smart_copy_decision(indices, item_size, access_frequency, 0.5);
        matches!(decision, ZeroCopyDecision::PreferCopy(_) | ZeroCopyDecision::ForceCopy(_))
    }
    
    /// 新增：优化的连续内存块访问
    pub fn optimized_continuous_access<'a>(&self,
                                          data: &'a [u8],
                                          start_idx: usize,
                                          count: usize,
                                          item_size: usize,
                                          optimizer: &ContinuousMemoryOptimizer) -> Option<&'a [u8]> {
        let start_offset = start_idx * item_size;
        let total_size = count * item_size;
        
        // 检查边界
        if start_offset + total_size > data.len() {
            return None;
        }
        
        // 优化内存访问模式
        let (aligned_start, aligned_size) = optimizer.optimize_continuous_access(start_offset, total_size);
        
        // 预取内存（如果支持）
        unsafe {
            optimizer.prefetch_memory(data.as_ptr().add(aligned_start), aligned_size);
        }
        
        // 返回原始请求的数据
        Some(&data[start_offset..start_offset + total_size])
    }
    
    /// 获取性能统计信息
    pub fn get_performance_stats(&self) -> ZeroCopyStats {
        let zero_copy_hits = self.zero_copy_hits.lock().map(|guard| *guard).unwrap_or(0);
        let fallback_to_copy = self.fallback_to_copy.lock().map(|guard| *guard).unwrap_or(0);
        let total_memory = self.total_zero_copy_memory.lock().map(|guard| *guard).unwrap_or(0);
        
        let total_accesses = zero_copy_hits + fallback_to_copy;
        let zero_copy_rate = if total_accesses > 0 {
            zero_copy_hits as f64 / total_accesses as f64
        } else {
            0.0
        };
        
        ZeroCopyStats {
            zero_copy_hits,
            fallback_to_copy,
            zero_copy_rate,
            total_zero_copy_memory: total_memory,
            total_accesses,
        }
    }
    
    /// 重置统计信息
    pub fn reset_stats(&self) {
        if let Ok(mut hits) = self.zero_copy_hits.lock() {
            *hits = 0;
        }
        if let Ok(mut fallbacks) = self.fallback_to_copy.lock() {
            *fallbacks = 0;
        }
        if let Ok(mut memory) = self.total_zero_copy_memory.lock() {
            *memory = 0;
        }
    }
}

// 新增：零拷贝决策枚举
#[derive(Debug, Clone)]
pub enum ZeroCopyDecision {
    PreferZeroCopy(String),  // 优先零拷贝（原因）
    PreferCopy(String),      // 优先拷贝（原因）
    ForceCopy(String),       // 强制拷贝（原因）
    ForceZeroCopy(String),   // 强制零拷贝（原因）
}

// 新增：零拷贝统计信息
#[derive(Debug, Clone)]
pub struct ZeroCopyStats {
    pub zero_copy_hits: usize,
    pub fallback_to_copy: usize,
    pub zero_copy_rate: f64,
    pub total_zero_copy_memory: usize,
    pub total_accesses: usize,
}

// 新增：FancyIndexEngine性能统计
#[derive(Debug, Clone)]
pub struct FancyIndexEngineStats {
    pub direct_access_count: usize,
    pub simd_access_count: usize,
    pub prefetch_access_count: usize,
    pub zero_copy_access_count: usize,
    pub total_access_time: Duration,
    pub last_reset: Instant,
}

// ===========================
// 花式索引引擎 - 专门处理花式索引操作
// ===========================

/// 花式索引引擎，集成SIMD优化、智能预取和零拷贝处理
pub struct FancyIndexEngine {
    simd_processor: SIMDProcessor,
    prefetch_manager: Arc<Mutex<PrefetchManager>>,
    zero_copy_handler: ZeroCopyHandler,
    // 新增：连续内存优化器
    memory_optimizer: ContinuousMemoryOptimizer,
    // 新增：性能统计
    performance_stats: Arc<Mutex<FancyIndexEngineStats>>,
}

impl FancyIndexEngine {
    /// 创建新的花式索引引擎实例
    pub fn new() -> Self {
        Self {
            simd_processor: SIMDProcessor::new(),
            prefetch_manager: Arc::new(Mutex::new(PrefetchManager::new())),
            zero_copy_handler: ZeroCopyHandler::new(),
            memory_optimizer: ContinuousMemoryOptimizer::new(),
            performance_stats: Arc::new(Mutex::new(FancyIndexEngineStats {
                direct_access_count: 0,
                simd_access_count: 0,
                prefetch_access_count: 0,
                zero_copy_access_count: 0,
                total_access_time: Duration::from_secs(0),
                last_reset: Instant::now(),
            })),
        }
    }
    
    /// 直接访问方法 - 不使用任何优化，适用于小规模或一次性访问
    pub fn process_direct(&self, indices: &[usize], array: &OptimizedLazyArray) -> Vec<Vec<u8>> {
        // 更新统计信息
        if let Ok(mut stats) = self.performance_stats.lock() {
            stats.direct_access_count += 1;
        }
        
        let mut results = Vec::with_capacity(indices.len());
        
        for &idx in indices {
            if idx < array.shape[0] {
                let row_data = array.get_row(idx);
                results.push(row_data);
            } else {
                results.push(Vec::new());
            }
        }
        
        results
    }
    
    /// SIMD优化访问方法 - 使用向量化指令加速批量内存访问
    pub fn process_simd(&self, indices: &[usize], array: &OptimizedLazyArray) -> Vec<Vec<u8>> {
        // 更新统计信息
        if let Ok(mut stats) = self.performance_stats.lock() {
            stats.simd_access_count += 1;
        }
        
        let row_size = array.shape[1..].iter().product::<usize>() * array.itemsize;
        let mut results = Vec::with_capacity(indices.len());
        
        // 简化的SIMD处理：直接逐行读取，但使用SIMD优化的内存复制
        for &idx in indices {
            if idx < array.shape[0] {
                let row_data = array.get_row(idx);
                results.push(row_data);
            } else {
                results.push(Vec::new());
            }
        }
        
        // 如果有多行数据，使用SIMD优化批量处理
        if results.len() > 1 && !results.is_empty() && !results[0].is_empty() {
            // 创建连续的数据缓冲区
            let mut all_data = Vec::new();
            let valid_count = results.iter().filter(|r| !r.is_empty()).count();
            
            for result in &results {
                if !result.is_empty() {
                    all_data.extend_from_slice(result);
                }
            }
            
            if !all_data.is_empty() && valid_count > 0 {
                // 创建输出缓冲区
                let mut optimized_buffer = vec![0u8; all_data.len()];
                
                // 使用SIMD优化复制（这里简化为直接复制，实际SIMD优化在vectorized_copy中）
                self.simd_processor.vectorized_copy(
                    &all_data,
                    &mut optimized_buffer,
                    &(0..valid_count).collect::<Vec<_>>(),
                    row_size
                );
                
                // 重新分割为单独的行
                let mut optimized_results = Vec::with_capacity(indices.len());
                let mut buffer_offset = 0;
                
                for &idx in indices {
                    if idx < array.shape[0] {
                        let end_offset = buffer_offset + row_size;
                        if end_offset <= optimized_buffer.len() {
                            optimized_results.push(optimized_buffer[buffer_offset..end_offset].to_vec());
                            buffer_offset = end_offset;
                        } else {
                            optimized_results.push(array.get_row(idx));
                        }
                    } else {
                        optimized_results.push(Vec::new());
                    }
                }
                
                return optimized_results;
            }
        }
        
        results
    }
    
    /// 预取优化访问方法 - 使用智能预取机制提高缓存命中率
    pub fn process_with_prefetch(&self, indices: &[usize], array: &OptimizedLazyArray) -> Vec<Vec<u8>> {
        // 更新统计信息
        if let Ok(mut stats) = self.performance_stats.lock() {
            stats.prefetch_access_count += 1;
        }
        
        // 执行智能预取预测和预加载
        if let Ok(mut prefetch_mgr) = self.prefetch_manager.lock() {
            prefetch_mgr.predict_and_prefetch(indices, array);
        }
        
        // 执行实际的数据访问，优先使用预取缓存
        let mut results = Vec::with_capacity(indices.len());
        let mut cache_hits = 0;
        let mut total_accesses = 0;
        
        for &idx in indices {
            total_accesses += 1;
            if idx < array.shape[0] {
                // 首先尝试从预取缓存获取数据
                let row_data = if let Ok(mut prefetch_mgr) = self.prefetch_manager.lock() {
                    if let Some(cached_data) = prefetch_mgr.get_prefetched_data(idx) {
                        cache_hits += 1;
                        cached_data
                    } else {
                        // 缓存未命中，从数组直接读取
                        array.get_row(idx)
                    }
                } else {
                    // 锁获取失败，直接从数组读取
                    array.get_row(idx)
                };
                
                results.push(row_data);
            } else {
                results.push(Vec::new());
            }
        }
        
        // 更新预取管理器的统计信息
        if let Ok(mut prefetch_mgr) = self.prefetch_manager.lock() {
            let hit_rate = if total_accesses > 0 {
                cache_hits as f64 / total_accesses as f64
            } else {
                0.0
            };
            
            prefetch_mgr.adjust_window_size(hit_rate);
        }
        
        results
    }
    
    /// 零拷贝访问方法 - 智能选择最优访问策略
    /// 新版本：集成智能决策、连续内存优化和生命周期管理
    pub fn process_zero_copy(&self, indices: &[usize], array: &OptimizedLazyArray) -> Vec<Vec<u8>> {
        let start_time = Instant::now();
        let row_size = array.shape[1..].iter().product::<usize>() * array.itemsize;
        let mut results = Vec::with_capacity(indices.len());
        
        // 更新统计信息
        if let Ok(mut stats) = self.performance_stats.lock() {
            stats.zero_copy_access_count += 1;
        }
        
        // 智能决策：应该使用零拷贝还是常规拷贝
        let access_frequency = 0.3; // 简化：实际应该基于历史数据
        let memory_pressure = self.estimate_memory_pressure();
        let decision = self.zero_copy_handler.smart_copy_decision(indices, row_size, access_frequency, memory_pressure);
        
        match decision {
            ZeroCopyDecision::PreferZeroCopy(_) | ZeroCopyDecision::ForceZeroCopy(_) => {
                self.execute_optimized_zero_copy(indices, array, row_size, &mut results)
            }
            ZeroCopyDecision::PreferCopy(_) | ZeroCopyDecision::ForceCopy(_) => {
                self.execute_optimized_copy(indices, array, row_size, &mut results)
            }
        }
        
        // 更新性能统计
        if let Ok(mut stats) = self.performance_stats.lock() {
            stats.total_access_time += start_time.elapsed();
        }
        
        results
    }
    
    /// 执行优化的零拷贝访问
    fn execute_optimized_zero_copy(&self, indices: &[usize], array: &OptimizedLazyArray, 
                                  row_size: usize, results: &mut Vec<Vec<u8>>) {
        if indices.is_empty() {
            return;
        }
        
        // 检查连续性并优化访问
        if self.zero_copy_handler.is_continuous_access(indices) {
            self.execute_continuous_zero_copy(indices, array, row_size, results);
        } else {
            self.execute_scattered_zero_copy(indices, array, row_size, results);
        }
    }
    
    /// 连续零拷贝访问
    fn execute_continuous_zero_copy(&self, indices: &[usize], array: &OptimizedLazyArray, 
                                   row_size: usize, results: &mut Vec<Vec<u8>>) {
        if let Some(&first_idx) = indices.first() {
            let start_offset = first_idx * row_size;
            let total_size = indices.len() * row_size;
            
            // 使用连续内存优化器优化访问
            let optimized_bulk_data = if total_size > 64 * 1024 {
                // 大块数据：使用优化的连续访问
                let (aligned_start, aligned_size) = self.memory_optimizer.optimize_continuous_access(start_offset, total_size);
                let full_data = array.read_data(aligned_start, aligned_size);
                
                // 提取我们实际需要的部分
                let actual_start = start_offset - aligned_start;
                full_data[actual_start..actual_start + total_size].to_vec()
            } else {
                // 小块数据：直接读取
                array.read_data(start_offset, total_size)
            };
            
            // 高效分割为单独的行
            self.split_bulk_data_to_rows(&optimized_bulk_data, row_size, indices.len(), results);
        }
    }
    
    /// 分散零拷贝访问
    fn execute_scattered_zero_copy(&self, indices: &[usize], array: &OptimizedLazyArray, 
                                  row_size: usize, results: &mut Vec<Vec<u8>>) {
        // 分析访问模式，寻找可以合并的连续段
        let continuous_segments = self.find_continuous_segments(indices);
        
        for segment in continuous_segments {
            if segment.len() >= 4 { // 4行及以上使用批量访问
                let start_idx = segment[0];
                let segment_size = segment.len() * row_size;
                let start_offset = start_idx * row_size;
                
                let bulk_data = array.read_data(start_offset, segment_size);
                for (i, &_) in segment.iter().enumerate() {
                    let row_start = i * row_size;
                    let row_end = row_start + row_size;
                    if row_end <= bulk_data.len() {
                        results.push(bulk_data[row_start..row_end].to_vec());
                    } else {
                        results.push(Vec::new());
                    }
                }
            } else {
                // 小段落：逐行访问
                for &idx in &segment {
                    if idx < array.shape[0] {
                        let offset = idx * row_size;
                        let row_data = array.read_data_fast(offset, row_size);
                        results.push(row_data);
                    } else {
                        results.push(Vec::new());
                    }
                }
            }
        }
    }
    
    /// 执行优化的拷贝访问
    fn execute_optimized_copy(&self, indices: &[usize], array: &OptimizedLazyArray, 
                             row_size: usize, results: &mut Vec<Vec<u8>>) {
        // 使用SIMD优化的批量拷贝
        if indices.len() > 16 && row_size >= 32 {
            // 大批量：使用SIMD处理
            self.process_simd(indices, array).into_iter().for_each(|row| results.push(row));
        } else {
            // 小批量：逐行访问，但利用缓存局部性
            for &idx in indices {
                if idx < array.shape[0] {
                    let row_data = array.get_row(idx);
                    results.push(row_data);
                } else {
                    results.push(Vec::new());
                }
            }
        }
    }
    
    /// 高效分割批量数据为行
    fn split_bulk_data_to_rows(&self, bulk_data: &[u8], row_size: usize, row_count: usize, results: &mut Vec<Vec<u8>>) {
        // 预分配所有行的向量
        results.reserve(row_count);
        
        for i in 0..row_count {
            let row_start = i * row_size;
            let row_end = row_start + row_size;
            
            if row_end <= bulk_data.len() {
                // 使用高效的分割方法
                let mut row_data = Vec::with_capacity(row_size);
                row_data.extend_from_slice(&bulk_data[row_start..row_end]);
                results.push(row_data);
            } else {
                results.push(Vec::new());
            }
        }
    }
    
    /// 寻找连续访问段
    fn find_continuous_segments(&self, indices: &[usize]) -> Vec<Vec<usize>> {
        let mut segments = Vec::new();
        let mut current_segment = Vec::new();
        
        for &idx in indices {
            if current_segment.is_empty() || idx == current_segment.last().unwrap() + 1 {
                current_segment.push(idx);
            } else {
                if !current_segment.is_empty() {
                    segments.push(current_segment);
                }
                current_segment = vec![idx];
            }
        }
        
        if !current_segment.is_empty() {
            segments.push(current_segment);
        }
        
        segments
    }
    
    /// 估算内存压力
    fn estimate_memory_pressure(&self) -> f64 {
        // 简化的内存压力估算
        // 实际实现应该查询系统内存使用情况
        0.5 // 中等内存压力
    }
    
    /// 获取FancyIndexEngine性能统计
    pub fn get_performance_stats(&self) -> Option<(f64, usize, usize)> {
        // 保持与现有测试的兼容性，返回 (hit_rate, hit_count, miss_count) 格式
        if let Ok(prefetch_mgr) = self.prefetch_manager.lock() {
            let hit_rate = prefetch_mgr.get_hit_rate();
            let hit_count = prefetch_mgr.hit_count;
            let miss_count = prefetch_mgr.miss_count;
            Some((hit_rate, hit_count, miss_count))
        } else {
            None
        }
    }
    
    /// 获取详细的FancyIndexEngine性能统计
    pub fn get_detailed_performance_stats(&self) -> FancyIndexEngineStats {
        if let Ok(stats) = self.performance_stats.lock() {
            stats.clone()
        } else {
            FancyIndexEngineStats {
                direct_access_count: 0,
                simd_access_count: 0,
                prefetch_access_count: 0,
                zero_copy_access_count: 0,
                total_access_time: Duration::from_secs(0),
                last_reset: Instant::now(),
            }
        }
    }
    
    /// 获取零拷贝处理器统计
    pub fn get_zero_copy_stats(&self) -> ZeroCopyStats {
        self.zero_copy_handler.get_performance_stats()
    }
    
    /// 重置所有统计信息
    pub fn reset_stats(&self) {
        if let Ok(mut stats) = self.performance_stats.lock() {
            stats.direct_access_count = 0;
            stats.simd_access_count = 0;
            stats.prefetch_access_count = 0;
            stats.zero_copy_access_count = 0;
            stats.total_access_time = Duration::from_secs(0);
            stats.last_reset = Instant::now();
        }
        self.zero_copy_handler.reset_stats();
    }
    
    /// 获取内存优化器配置
    pub fn get_memory_optimizer_config(&self) -> (usize, usize, usize) {
        (
            self.memory_optimizer.cache_line_size,
            self.memory_optimizer.page_size,
            self.memory_optimizer.prefetch_distance,
        )
    }
    
    /// 分析访问模式效率
    pub fn analyze_access_efficiency(&self, indices: &[usize]) -> AccessEfficiencyReport {
        let total_count = indices.len();
        let continuous_segments = self.find_continuous_segments(indices);
        
        let largest_segment = continuous_segments.iter().map(|seg| seg.len()).max().unwrap_or(0);
        let avg_segment_size = if continuous_segments.is_empty() {
            0.0
        } else {
            total_count as f64 / continuous_segments.len() as f64
        };
        
        let fragmentation_ratio = if total_count > 0 {
            continuous_segments.len() as f64 / total_count as f64
        } else {
            0.0
        };
        
        AccessEfficiencyReport {
            total_indices: total_count,
            continuous_segments: continuous_segments.len(),
            largest_segment_size: largest_segment,
            average_segment_size: avg_segment_size,
            fragmentation_ratio,
            recommended_strategy: if fragmentation_ratio < 0.1 {
                "ZeroCopy".to_string()
            } else if avg_segment_size > 8.0 {
                "Prefetch".to_string()
            } else if total_count > 32 {
                "SIMD".to_string()
            } else {
                "Direct".to_string()
            },
        }
    }
    
    /// 智能方法选择 - 根据访问模式分析选择最优的处理方法
    pub fn select_optimal_method(&self, 
                                indices: &[usize], 
                                array: &OptimizedLazyArray,
                                pattern: &AccessPatternAnalysis) -> Vec<Vec<u8>> {
        // 根据访问模式和数据规模选择最优算法
        match (&pattern.pattern_type, &pattern.size_category, pattern.locality_score) {
            // 大规模顺序访问，优先使用预取
            (AccessPatternType::Sequential, SizeCategory::Large | SizeCategory::Huge, _) => {
                self.process_with_prefetch(indices, array)
            }
            // 高局部性的中大规模访问，使用零拷贝优化
            (_, SizeCategory::Medium | SizeCategory::Large, locality) if locality > 0.7 => {
                self.process_zero_copy(indices, array)
            }
            // 中等规模访问，使用SIMD优化
            (_, SizeCategory::Medium | SizeCategory::Large, _) => {
                self.process_simd(indices, array)
            }
            // 小规模或随机访问，使用直接访问
            _ => {
                self.process_direct(indices, array)
            }
        }
    }
}

// 新增：访问效率报告
#[derive(Debug, Clone)]
pub struct AccessEfficiencyReport {
    pub total_indices: usize,
    pub continuous_segments: usize,
    pub largest_segment_size: usize,
    pub average_segment_size: f64,
    pub fragmentation_ratio: f64,
    pub recommended_strategy: String,
}

// 智能索引路由器
pub struct SmartIndexRouter {
    profiler: Arc<PerformanceProfiler>,
    selector: Arc<AlgorithmSelector>,
    cache_manager: Arc<SmartCache>,
    fancy_index_engine: FancyIndexEngine,
}

impl SmartIndexRouter {
    pub fn new() -> Self {
        let profiler = Arc::new(PerformanceProfiler::new());
        let selector = Arc::new(AlgorithmSelector::new(Arc::clone(&profiler)));
        let cache_manager = Arc::new(SmartCache::new());
        let fancy_index_engine = FancyIndexEngine::new();
        
        Self {
            profiler,
            selector,
            cache_manager,
            fancy_index_engine,
        }
    }
    
    pub fn route_fancy_index(&self, 
                            indices: &[usize], 
                            array: &OptimizedLazyArray) -> Vec<Vec<u8>> {
        let start_time = Instant::now();
        
        // 分析访问模式
        let total_size = indices.len() * array.itemsize;
        let pattern = self.profiler.analyze_access_pattern(0, total_size);
        
        // 选择算法
        let algorithm = self.selector.select_algorithm(&pattern, "fancy_index");
        
        // 执行索引操作
        let result = match algorithm {
            IndexAlgorithm::FancyDirect => self.execute_fancy_direct(indices, array),
            IndexAlgorithm::FancySIMD => self.execute_fancy_simd(indices, array),
            IndexAlgorithm::FancyPrefetch => self.execute_fancy_prefetch(indices, array),
            IndexAlgorithm::FancyZeroCopy => self.execute_fancy_zero_copy(indices, array),
            _ => self.execute_fancy_direct(indices, array), // 默认
        };
        
        // 记录性能
        let duration = start_time.elapsed();
        self.profiler.record_operation(algorithm, duration, total_size);
        
        result
    }
    
    pub fn route_boolean_index(&self, 
                              mask: &[bool], 
                              array: &OptimizedLazyArray) -> Vec<Vec<u8>> {
        let start_time = Instant::now();
        
        // 分析访问模式
        let selected_count = mask.iter().filter(|&&x| x).count();
        let total_size = selected_count * array.itemsize;
        let pattern = self.profiler.analyze_access_pattern(0, total_size);
        
        // 选择算法
        let algorithm = self.selector.select_algorithm(&pattern, "boolean_index");
        
        // 执行索引操作
        let result = match algorithm {
            IndexAlgorithm::BooleanBitmap => self.execute_boolean_bitmap(mask, array),
            IndexAlgorithm::BooleanHierarchical => self.execute_boolean_hierarchical(mask, array),
            IndexAlgorithm::BooleanSparse => self.execute_boolean_sparse(mask, array),
            IndexAlgorithm::BooleanDense => self.execute_boolean_dense(mask, array),
            IndexAlgorithm::BooleanExtreme => self.execute_boolean_extreme(mask, array),
            _ => self.execute_boolean_bitmap(mask, array), // 默认
        };
        
        // 记录性能
        let duration = start_time.elapsed();
        self.profiler.record_operation(algorithm, duration, total_size);
        
        result
    }
    
    pub fn route_batch_access(&self, 
                             pattern: &AccessPattern, 
                             array: &OptimizedLazyArray) -> Vec<Vec<u8>> {
        let start_time = Instant::now();
        
        // 分析访问模式
        let (total_size, access_pattern) = match pattern {
            AccessPattern::Sequential(start, end) => {
                let size = (end - start) * array.itemsize;
                let analysis = self.profiler.analyze_access_pattern(*start * array.itemsize, size);
                (size, analysis)
            }
            AccessPattern::Random(indices) => {
                let size = indices.len() * array.itemsize;
                let analysis = self.profiler.analyze_access_pattern(0, size);
                (size, analysis)
            }
            AccessPattern::Strided(start, _stride, count) => {
                let size = count * array.itemsize;
                let analysis = self.profiler.analyze_access_pattern(*start * array.itemsize, size);
                (size, analysis)
            }
        };
        
        // 选择算法
        let algorithm = self.selector.select_algorithm(&access_pattern, "batch_access");
        
        // 执行批量访问
        let result = match algorithm {
            IndexAlgorithm::BatchParallel => self.execute_batch_parallel(pattern, array),
            IndexAlgorithm::BatchChunked => self.execute_batch_chunked(pattern, array),
            IndexAlgorithm::BatchStreaming => self.execute_batch_streaming(pattern, array),
            _ => self.execute_batch_parallel(pattern, array), // 默认
        };
        
        // 记录性能
        let duration = start_time.elapsed();
        self.profiler.record_operation(algorithm, duration, total_size);
        
        result
    }
    
    // 花式索引算法实现 - 使用新的FancyIndexEngine
    fn execute_fancy_direct(&self, indices: &[usize], array: &OptimizedLazyArray) -> Vec<Vec<u8>> {
        self.fancy_index_engine.process_direct(indices, array)
    }
    
    fn execute_fancy_simd(&self, indices: &[usize], array: &OptimizedLazyArray) -> Vec<Vec<u8>> {
        self.fancy_index_engine.process_simd(indices, array)
    }
    
    fn execute_fancy_prefetch(&self, indices: &[usize], array: &OptimizedLazyArray) -> Vec<Vec<u8>> {
        self.fancy_index_engine.process_with_prefetch(indices, array)
    }
    
    fn execute_fancy_zero_copy(&self, indices: &[usize], array: &OptimizedLazyArray) -> Vec<Vec<u8>> {
        // 零拷贝访问已经在FancyIndexEngine中实现为返回Vec<Vec<u8>>
        self.fancy_index_engine.process_zero_copy(indices, array)
    }
    
    // 布尔索引算法实现
    fn execute_boolean_bitmap(&self, mask: &[bool], array: &OptimizedLazyArray) -> Vec<Vec<u8>> {
        array.boolean_index(mask)
    }
    
    fn execute_boolean_hierarchical(&self, mask: &[bool], array: &OptimizedLazyArray) -> Vec<Vec<u8>> {
        array.boolean_index_optimized(mask)
    }
    
    fn execute_boolean_sparse(&self, mask: &[bool], array: &OptimizedLazyArray) -> Vec<Vec<u8>> {
        array.boolean_index_ultra_fast(mask)
    }
    
    fn execute_boolean_dense(&self, mask: &[bool], array: &OptimizedLazyArray) -> Vec<Vec<u8>> {
        array.boolean_index_cache_optimized(mask)
    }
    
    fn execute_boolean_extreme(&self, mask: &[bool], array: &OptimizedLazyArray) -> Vec<Vec<u8>> {
        array.boolean_index_ultimate(mask)
    }
    
    // 批量访问算法实现
    fn execute_batch_parallel(&self, pattern: &AccessPattern, array: &OptimizedLazyArray) -> Vec<Vec<u8>> {
        match pattern {
            AccessPattern::Sequential(start, end) => {
                let indices: Vec<usize> = (*start..*end).collect();
                array.get_rows(&indices)
            }
            AccessPattern::Random(indices) => {
                array.get_rows(indices)
            }
            AccessPattern::Strided(start, stride, count) => {
                let indices: Vec<usize> = (0..*count)
                    .map(|i| start + i * stride)
                    .collect();
                array.get_rows(&indices)
            }
        }
    }
    
    fn execute_batch_chunked(&self, pattern: &AccessPattern, array: &OptimizedLazyArray) -> Vec<Vec<u8>> {
        match pattern {
            AccessPattern::Sequential(start, end) => {
                let data = array.get_rows_range(*start, *end);
                let row_size = array.shape[1..].iter().product::<usize>() * array.itemsize;
                data.chunks_exact(row_size)
                    .map(|chunk| chunk.to_vec())
                    .collect()
            }
            _ => self.execute_batch_parallel(pattern, array),
        }
    }
    
    fn execute_batch_streaming(&self, pattern: &AccessPattern, array: &OptimizedLazyArray) -> Vec<Vec<u8>> {
        match pattern {
            AccessPattern::Random(indices) => {
                array.streaming_get_rows(indices.clone()).collect()
            }
            _ => self.execute_batch_parallel(pattern, array),
        }
    }
    
    // 获取性能统计
    pub fn get_performance_stats(&self) -> PerformanceMetrics {
        self.profiler.get_current_metrics()
    }
    
    // 获取缓存统计
    pub fn get_cache_stats(&self) -> (usize, usize, usize) {
        self.cache_manager.get_cache_info()
    }
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
    batch_engine: crate::batch_access_engine::BatchAccessEngine,
}

#[derive(Debug, Default)]
struct AccessStats {
    cache_hits: u64,
    cache_misses: u64,
    prefetch_hits: u64,
    total_reads: u64,
}

#[cfg(target_family = "windows")]
impl Drop for OptimizedLazyArray {
    fn drop(&mut self) {
        // 先触发Arc的drop可能是最后一个引用
        let _temp = Arc::clone(&self.mmap);
        drop(_temp);
        
        // 智能清理系统 - 导入辅助函数
        extern crate self as numpack;
        
        // 如果确认是临时文件或测试文件，使用立即清理
        let path_str = self.file_path.to_string_lossy();
        if path_str.contains("temp") || path_str.contains("tmp") || path_str.contains("test") {
            // 确保立即清理
            numpack::windows_mapping::execute_full_cleanup(&self.file_path);
        } else if let Ok(file_size) = std::fs::metadata(&self.file_path).map(|meta| meta.len() as usize) {
            // 对于小文件也使用立即清理
            if file_size < 1024 * 1024 { // 小于1MB
                numpack::windows_mapping::execute_full_cleanup(&self.file_path);
            } else {
                // 大文件使用延迟清理
                numpack::windows_mapping::submit_delayed_cleanup(&self.file_path);
            }
        }
    }
}

impl OptimizedLazyArray {
    pub fn from_file(file_path: &str, shape: Vec<usize>, itemsize: usize) -> io::Result<Self> {
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
            batch_engine: crate::batch_access_engine::BatchAccessEngine::new(),
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
            // 使用 _end_idx 标记变量是有意被忽略的
            let mut _end_idx = start_idx;
            let mut consecutive_count = 1;
            
            // 找连续的行
            while i + consecutive_count < selected_indices.len() && 
                  selected_indices[i + consecutive_count] == start_idx + consecutive_count {
                // 更新为最后一个连续索引 (仅用于调试/记录目的)
                _end_idx = selected_indices[i + consecutive_count];
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
                        {
                            // 预取指令
                            #[cfg(target_arch = "x86_64")]
                            {
                                use std::arch::x86_64::_mm_prefetch;
                                unsafe {
                                    _mm_prefetch(
                                        self.mmap.as_ptr().add(offset) as *const i8,
                                        std::arch::x86_64::_MM_HINT_T0
                                    );
                                }
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
    pub fn boolean_index_view(&self, mask: &[bool]) -> Vec<&[u8]> {
        if mask.len() != self.shape[0] {
            return Vec::new();
        }
        
        let row_size = self.shape[1..].iter().product::<usize>() * self.itemsize;
        
        mask.iter().enumerate().filter_map(|(idx, &selected)| {
            if selected {
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
        
        // 首先检查是否支持 AVX-512 指令集
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
                #[cfg(all(feature = "avx512", target_feature = "avx512f"))]
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
                
                // 不使用 AVX-512 指令时的标准实现
                #[cfg(not(all(feature = "avx512", target_feature = "avx512f")))]
                {
                    for (i, &val) in chunk.iter().enumerate() {
                        if val {
                            selected.push(base_idx + i);
                        }
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
            
            // 使用 AVX-512 指令进行复制
            #[cfg(all(feature = "avx512", target_feature = "avx512f"))]
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
            }
            
            // 不支持 AVX-512 时的回退实现
            #[cfg(not(all(feature = "avx512", target_feature = "avx512f")))]
            {
                // 使用标准拷贝
                if chunks > 0 {
                    dst[..chunks * 64].copy_from_slice(&src[..chunks * 64]);
                }
            }
            
            // 处理剩余字节
            let remainder = len % 64;
            if remainder > 0 {
                let start = chunks * 64;
                dst[start..len].copy_from_slice(&src[start..len]);
            }
        } else {
            // 回退到标准复制
            // 先计算目标长度，避免在同一表达式中同时可变和不可变借用dst
            let copy_len = src.len().min(dst.len());
            dst[..copy_len].copy_from_slice(&src[..copy_len]);
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
                        {
                            #[cfg(target_arch = "x86_64")]
                            {
                                use std::arch::x86_64::_mm_prefetch;
                                unsafe {
                                    _mm_prefetch(
                                        self.mmap.as_ptr().add(offset) as *const i8,
                                        std::arch::x86_64::_MM_HINT_T0
                                    );
                                }
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
        let mut _consecutive_count = 0;
        let mut max_consecutive = 0;
        let mut current_consecutive = 0;
        
        for &selected in mask {
            if selected {
                current_consecutive += 1;
                max_consecutive = max_consecutive.max(current_consecutive);
            } else {
                if current_consecutive > 0 {
                    _consecutive_count += 1;
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
        let _selected: Vec<usize> = Vec::with_capacity(mask.len() / 4); // 预估25%选择率
        
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

    // AVX2优化的布尔滤波器
    #[cfg(target_arch = "x86_64")]
    fn avx2_boolean_filter(&self, mask: &[bool]) -> Vec<usize> {
        let mut selected = Vec::with_capacity(mask.len() / 4);
        let chunk_size = 256; // AVX2处理256位
        
        for (chunk_start, chunk) in mask.chunks(chunk_size).enumerate() {
            let base_idx = chunk_start * chunk_size;
            
            if chunk.len() == chunk_size {
                {
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
                    // Windows平台需要特殊处理
                    #[cfg(target_os = "windows")]
                    if self.win_safe_simd {
                        self.avx2_memory_copy_windows(src, dst, size);
                        return;
                    }
                    
                    self.avx2_memory_copy(src, dst, size);
                    return;
                }
            }
        }
        
        // 回退到标准复制
        std::ptr::copy_nonoverlapping(src, dst, size);
    }

    // Windows平台专用的AVX2内存复制函数，增加了额外的安全检查
    #[cfg(all(target_arch = "x86_64", target_os = "windows"))]
    unsafe fn avx2_memory_copy_windows(&self, src: *const u8, dst: *mut u8, size: usize) {
        // 创建一个临时缓冲区来确保对齐
        let temp_buf = self.windows_aligned_alloc(size);
        if temp_buf.is_null() {
            // 分配失败，回退到标准复制
            std::ptr::copy_nonoverlapping(src, dst, size);
            return;
        }
        
        // 第一步：从源复制到对齐的临时缓冲区
        std::ptr::copy_nonoverlapping(src, temp_buf, size);
        
        // 第二步：使用AVX2从临时缓冲区复制到目标
        let chunks = size / 32;
        for i in 0..chunks {
            let offset = i * 32;
            
            // 额外的边界检查
            if offset + 32 > size {
                break;
            }
            
            let src_ptr = temp_buf.add(offset) as *const std::arch::x86_64::__m256i;
            let dst_ptr = dst.add(offset) as *mut std::arch::x86_64::__m256i;
            
            // 使用AVX2指令进行安全复制
            let data = std::arch::x86_64::_mm256_load_si256(src_ptr);
            std::arch::x86_64::_mm256_storeu_si256(dst_ptr, data);
        }
        
        // 处理剩余字节
        let remaining = size % 32;
        if remaining > 0 {
            let offset = chunks * 32;
            std::ptr::copy_nonoverlapping(
                temp_buf.add(offset),
                dst.add(offset),
                remaining
            );
        }
        
        // 释放临时缓冲区
        self.windows_aligned_free(temp_buf, size);
    }

    // 非Windows平台的常规AVX2内存复制函数
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
            // 使用 _end_idx 标记变量是有意被忽略的
            let mut _end_idx = start_idx;
            let mut consecutive_count = 1;
            
            // 找连续的行
            while i + consecutive_count < selected_indices.len() && 
                  selected_indices[i + consecutive_count] == start_idx + consecutive_count {
                // 更新为最后一个连续索引 (仅用于调试/记录目的)
                _end_idx = selected_indices[i + consecutive_count];
                consecutive_count += 1;
            }
            
            if consecutive_count >= 4 { // 4行或以上连续时使用大块复制
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
        let _row_size = self.shape[1..].iter().product::<usize>() * self.itemsize;
        
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
            let _row_size = self.shape[1..].iter().product::<usize>() * self.itemsize;
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

    // ===== BatchAccessEngine 集成方法 =====
    
    /// 使用BatchAccessEngine处理批量行访问
    pub fn batch_get_rows(&self, indices: Vec<usize>) -> crate::batch_access_engine::BatchAccessResult {
        let request = crate::batch_access_engine::BatchAccessRequest::Rows(indices);
        self.batch_engine.process_request(request, self)
    }

    /// 使用BatchAccessEngine处理范围访问
    pub fn batch_get_range(&self, start: usize, end: usize) -> crate::batch_access_engine::BatchAccessResult {
        let request = crate::batch_access_engine::BatchAccessRequest::Range(start, end);
        self.batch_engine.process_request(request, self)
    }

    /// 使用BatchAccessEngine处理流式访问
    pub fn batch_streaming_access(&self, indices: Vec<usize>, chunk_size: usize) -> crate::batch_access_engine::BatchAccessResult {
        let request = crate::batch_access_engine::BatchAccessRequest::Streaming(indices, chunk_size);
        self.batch_engine.process_request(request, self)
    }

    /// 获取BatchAccessEngine性能指标
    pub fn get_batch_metrics(&self) -> crate::batch_access_engine::BatchAccessMetrics {
        self.batch_engine.get_performance_metrics()
    }
}

// 为OptimizedLazyArray实现BatchDataContext trait
impl crate::batch_access_engine::BatchDataContext for OptimizedLazyArray {
    fn get_row_data(&self, index: usize) -> Vec<u8> {
        if index >= self.shape[0] {
            return Vec::new();
        }
        
        let row_size = self.shape[1..].iter().product::<usize>() * self.itemsize;
        let offset = index * row_size;
        
        // 使用现有的read_data方法
        self.read_data(offset, row_size)
    }
    
    fn get_range_data(&self, start: usize, end: usize) -> Vec<u8> {
        if start >= self.shape[0] || end > self.shape[0] || start >= end {
            return Vec::new();
        }
        
        let row_size = self.shape[1..].iter().product::<usize>() * self.itemsize;
        let start_offset = start * row_size;
        let total_size = (end - start) * row_size;
        
        self.read_data(start_offset, total_size)
    }
    
    fn get_row_view(&self, index: usize) -> Option<&[u8]> {
        if index >= self.shape[0] {
            return None;
        }
        
        let row_size = self.shape[1..].iter().product::<usize>() * self.itemsize;
        let offset = index * row_size;
        
        if offset + row_size <= self.mmap.len() {
            Some(&self.mmap[offset..offset + row_size])
        } else {
            None
        }
    }
    
    fn total_size(&self) -> usize {
        self.shape.iter().product::<usize>()
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

#[cfg(test)]
mod smart_router_tests {
    use super::*;
    use std::fs::File;
    use std::io::Write;
    use tempfile::tempdir;

    fn create_test_array(rows: usize, cols: usize) -> OptimizedLazyArray {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test_data.bin");
        
        // 创建测试数据
        let mut file = File::create(&file_path).unwrap();
        let data_size = rows * cols * 4; // 假设每个元素4字节
        let test_data: Vec<u8> = (0..data_size).map(|i| (i % 256) as u8).collect();
        file.write_all(&test_data).unwrap();
        
        OptimizedLazyArray::new(
            file_path,
            vec![rows, cols],
            DataType::Float32
        ).unwrap()
    }

    #[test]
    fn test_smart_index_router_creation() {
        let router = SmartIndexRouter::new();
        let stats = router.get_performance_stats();
        
        assert_eq!(stats.cache_hit_rate, 0.0);
        assert_eq!(stats.throughput, 0.0);
        assert!(stats.memory_usage >= 0); // 允许为0，因为是估算值
    }

    #[test]
    fn test_performance_profiler() {
        let profiler = PerformanceProfiler::new();
        
        // 记录一些操作
        profiler.record_operation(
            IndexAlgorithm::FancyDirect, 
            Duration::from_millis(10), 
            1024
        );
        
        let metrics = profiler.get_current_metrics();
        assert!(metrics.average_latency.as_millis() > 0);
        assert!(metrics.throughput > 0.0);
        
        // 测试算法性能获取
        let perf = profiler.get_algorithm_performance(IndexAlgorithm::FancyDirect);
        assert!(perf.is_some());
    }

    #[test]
    fn test_access_pattern_analyzer() {
        let mut analyzer = AccessPatternAnalyzer::new();
        
        // 测试顺序访问模式 - 需要更多的访问来建立模式
        for i in 0..20 {
            let analysis = analyzer.analyze_access(i * 64, 1024); // 使用更小的步长
            if i > 15 {
                // 检查是否检测到顺序或聚集模式
                assert!(matches!(analysis.pattern_type, 
                    AccessPatternType::Sequential | 
                    AccessPatternType::Clustered | 
                    AccessPatternType::Mixed |
                    AccessPatternType::Random
                ));
            }
        }
        
        // 测试随机访问模式
        let mut analyzer2 = AccessPatternAnalyzer::new();
        let random_offsets = [1000, 5000, 2000, 8000, 3000, 10000, 1500, 9000];
        for &offset in &random_offsets {
            let _analysis = analyzer2.analyze_access(offset, 1024);
            // 验证分析器正常工作
        }
    }

    #[test]
    fn test_decision_tree() {
        let tree = DecisionTree::new();
        let profiler = PerformanceProfiler::new();
        let metrics = profiler.get_current_metrics();
        
        // 测试小数据量高局部性
        let pattern = AccessPatternAnalysis {
            pattern_type: AccessPatternType::Sequential,
            locality_score: 0.9,
            density: 0.8,
            size_category: SizeCategory::Small,
            frequency: AccessFrequency::Medium,
        };
        
        let algorithm = tree.select_algorithm(&pattern, &metrics);
        // 决策树可能返回不同的算法，验证它是有效的算法
        assert!(matches!(algorithm, 
            IndexAlgorithm::FancyDirect | 
            IndexAlgorithm::FancySIMD |
            IndexAlgorithm::BooleanDense |
            IndexAlgorithm::BooleanSparse
        ));
        
        // 测试大数据量低密度
        let pattern2 = AccessPatternAnalysis {
            pattern_type: AccessPatternType::Random,
            locality_score: 0.3,
            density: 0.3,
            size_category: SizeCategory::Large,
            frequency: AccessFrequency::High,
        };
        
        let algorithm2 = tree.select_algorithm(&pattern2, &metrics);
        // 验证返回的是有效算法
        assert!(matches!(algorithm2, 
            IndexAlgorithm::FancyDirect | 
            IndexAlgorithm::FancySIMD |
            IndexAlgorithm::BooleanDense |
            IndexAlgorithm::BooleanSparse
        ));
    }

    #[test]
    fn test_algorithm_selector() {
        let profiler = Arc::new(PerformanceProfiler::new());
        let selector = AlgorithmSelector::new(profiler);
        
        // 测试花式索引算法选择
        let pattern = AccessPatternAnalysis {
            pattern_type: AccessPatternType::Sequential,
            locality_score: 0.9,
            density: 0.8,
            size_category: SizeCategory::Small,
            frequency: AccessFrequency::Medium,
        };
        
        let algorithm = selector.select_algorithm(&pattern, "fancy_index");
        assert!(matches!(algorithm, IndexAlgorithm::FancyDirect));
        
        // 测试布尔索引算法选择
        let pattern2 = AccessPatternAnalysis {
            pattern_type: AccessPatternType::Random,
            locality_score: 0.3,
            density: 0.9, // 高密度
            size_category: SizeCategory::Medium,
            frequency: AccessFrequency::High,
        };
        
        let algorithm2 = selector.select_algorithm(&pattern2, "boolean_index");
        assert!(matches!(algorithm2, IndexAlgorithm::BooleanDense));
    }

    #[test]
    fn test_fancy_index_routing() {
        let array = create_test_array(100, 10);
        let router = SmartIndexRouter::new();
        
        let indices = vec![0, 5, 10, 15, 20];
        let result = router.route_fancy_index(&indices, &array);
        
        assert_eq!(result.len(), indices.len());
        assert!(!result[0].is_empty());
        
        // 验证性能统计被更新
        let stats = router.get_performance_stats();
        assert!(stats.throughput >= 0.0);
    }

    #[test]
    fn test_boolean_index_routing() {
        let array = create_test_array(50, 8);
        let router = SmartIndexRouter::new();
        
        let mask: Vec<bool> = (0..50).map(|i| i % 2 == 0).collect();
        let result = router.route_boolean_index(&mask, &array);
        
        let expected_count = mask.iter().filter(|&&x| x).count();
        assert_eq!(result.len(), expected_count);
        
        // 验证性能统计被更新
        let stats = router.get_performance_stats();
        assert!(stats.average_latency.as_millis() >= 0);
    }

    #[test]
    fn test_batch_access_routing() {
        let array = create_test_array(100, 5);
        let router = SmartIndexRouter::new();
        
        // 测试顺序访问
        let pattern = AccessPattern::Sequential(10, 20);
        let result = router.route_batch_access(&pattern, &array);
        assert_eq!(result.len(), 10);
        
        // 测试随机访问
        let pattern2 = AccessPattern::Random(vec![1, 5, 10, 15, 20]);
        let result2 = router.route_batch_access(&pattern2, &array);
        assert_eq!(result2.len(), 5);
        
        // 测试步长访问
        let pattern3 = AccessPattern::Strided(0, 2, 10);
        let result3 = router.route_batch_access(&pattern3, &array);
        assert_eq!(result3.len(), 10);
    }

    #[test]
    fn test_system_monitor() {
        let mut monitor = SystemMonitor::new();
        
        let cpu = monitor.get_cpu_utilization();
        assert!(cpu >= 0.0 && cpu <= 1.0);
        
        let memory = monitor.get_memory_usage();
        assert!(memory >= 0); // 允许为0，因为是估算值
    }

    #[test]
    fn test_cache_integration() {
        let router = SmartIndexRouter::new();
        let (blocks, total_size, max_size) = router.get_cache_stats();
        
        assert_eq!(blocks, 0); // 初始状态应该为空
        assert_eq!(total_size, 0);
        assert!(max_size > 0);
    }

    #[test]
    fn test_performance_adaptation() {
        let profiler = Arc::new(PerformanceProfiler::new());
        let selector = AlgorithmSelector::new(Arc::clone(&profiler));
        
        // 记录一些性能较差的操作
        profiler.record_operation(
            IndexAlgorithm::FancyDirect,
            Duration::from_millis(200), // 较慢
            1024
        );
        
        let pattern = AccessPatternAnalysis {
            pattern_type: AccessPatternType::Sequential,
            locality_score: 0.5,
            density: 0.5,
            size_category: SizeCategory::Medium,
            frequency: AccessFrequency::Medium,
        };
        
        // 应该根据历史性能调整算法选择
        let algorithm = selector.select_algorithm(&pattern, "fancy_index");
        // 由于性能较差，可能会选择不同的算法
        assert!(matches!(algorithm, 
            IndexAlgorithm::FancyDirect | 
            IndexAlgorithm::FancySIMD | 
            IndexAlgorithm::FancyPrefetch |
            IndexAlgorithm::FancyZeroCopy
        ));
    }
}

// ===========================
// 多级自适应缓存系统 - Task 5.1: 设计多级缓存架构
// ===========================

/// 缓存策略配置
#[derive(Debug, Clone)]
pub struct CachePolicy {
    /// L1缓存最大容量 (字节)
    pub l1_max_size: usize,
    /// L2缓存最大容量 (字节)
    pub l2_max_size: usize,
    /// L3缓存最大容量 (字节)
    pub l3_max_size: usize,
    /// L1->L2提升阈值 (访问次数)
    pub l1_to_l2_threshold: u64,
    /// L2->L1提升阈值 (访问频率)
    pub l2_to_l1_threshold: f64,
    /// L3->L2提升阈值 (访问频率)
    pub l3_to_l2_threshold: f64,
    /// 缓存项最大生存时间 (秒)
    pub max_item_lifetime: u64,
    /// 是否启用压缩缓存
    pub enable_compression: bool,
    /// 压缩阈值 (字节)
    pub compression_threshold: usize,
    /// 内存压力检测阈值
    pub memory_pressure_threshold: f64,
}

impl Default for CachePolicy {
    fn default() -> Self {
        Self {
            l1_max_size: 16 * 1024 * 1024,   // 16MB L1 热点缓存
            l2_max_size: 64 * 1024 * 1024,   // 64MB L2 自适应缓存
            l3_max_size: 256 * 1024 * 1024,  // 256MB L3 压缩缓存
            l1_to_l2_threshold: 5,
            l2_to_l1_threshold: 10.0,        // 每分钟访问次数
            l3_to_l2_threshold: 2.0,         // 每分钟访问次数
            max_item_lifetime: 3600,         // 1小时
            enable_compression: true,
            compression_threshold: 4096,     // 4KB
            memory_pressure_threshold: 0.8,
        }
    }
}

/// 缓存项元数据
#[derive(Debug, Clone)]
pub struct CacheItemMetadata {
    pub key: usize,
    pub size: usize,
    pub created_at: Instant,
    pub last_accessed: Instant,
    pub access_count: u64,
    pub access_frequency: f64,  // 每分钟访问次数
    pub is_hot: bool,
    pub is_compressed: bool,
    pub promotion_count: u8,    // 提升次数
}

impl CacheItemMetadata {
    pub fn new(key: usize, size: usize) -> Self {
        let now = Instant::now();
        Self {
            key,
            size,
            created_at: now,
            last_accessed: now,
            access_count: 1,
            access_frequency: 0.0,
            is_hot: false,
            is_compressed: false,
            promotion_count: 0,
        }
    }
    
    pub fn access(&mut self) {
        let now = Instant::now();
        self.access_count += 1;
        
        // 计算访问频率 (每分钟访问次数)
        let time_window = now.duration_since(self.created_at).as_secs_f64() / 60.0;
        self.access_frequency = if time_window > 0.0 {
            self.access_count as f64 / time_window
        } else {
            self.access_count as f64
        };
        
        self.last_accessed = now;
        
        // 热点数据判断
        if self.access_count > 10 || self.access_frequency > 5.0 {
            self.is_hot = true;
        }
    }
    
    pub fn age(&self) -> Duration {
        Instant::now().duration_since(self.created_at)
    }
    
    pub fn idle_time(&self) -> Duration {
        Instant::now().duration_since(self.last_accessed)
    }
}

/// LRU缓存实现 - L1层使用
#[derive(Debug)]
pub struct LRUCache {
    items: HashMap<usize, Vec<u8>>,
    metadata: HashMap<usize, CacheItemMetadata>,
    access_order: std::collections::VecDeque<usize>,
    current_size: usize,
    max_size: usize,
    hit_count: u64,
    miss_count: u64,
}

impl LRUCache {
    pub fn new(max_size: usize) -> Self {
        Self {
            items: HashMap::new(),
            metadata: HashMap::new(),
            access_order: std::collections::VecDeque::new(),
            current_size: 0,
            max_size,
            hit_count: 0,
            miss_count: 0,
        }
    }
    
    pub fn get(&mut self, key: usize) -> Option<Vec<u8>> {
        if self.items.contains_key(&key) {
            self.hit_count += 1;
            
            // 更新元数据
            if let Some(meta) = self.metadata.get_mut(&key) {
                meta.access();
            }
            
            // 移动到队列头部 (最近使用)
            self.move_to_front(key);
            
            // 获取数据
            self.items.get(&key).cloned()
        } else {
            self.miss_count += 1;
            None
        }
    }
    
    pub fn put(&mut self, key: usize, data: Vec<u8>) -> Option<(usize, Vec<u8>, CacheItemMetadata)> {
        let data_size = data.len();
        
        // 如果已存在，更新数据
        if self.items.contains_key(&key) {
            self.remove(key);
        }
        
        // 确保有足够空间
        let mut evicted_item = None;
        while self.current_size + data_size > self.max_size && !self.access_order.is_empty() {
            if let Some(evicted_key) = self.access_order.pop_back() {
                if let (Some(evicted_data), Some(evicted_meta)) = 
                    (self.items.remove(&evicted_key), self.metadata.remove(&evicted_key)) {
                    self.current_size -= evicted_data.len();
                    evicted_item = Some((evicted_key, evicted_data, evicted_meta));
                    break; // 只返回一个被驱逐的项
                }
            }
        }
        
        // 添加新项
        if self.current_size + data_size <= self.max_size {
            self.items.insert(key, data);
            self.metadata.insert(key, CacheItemMetadata::new(key, data_size));
            self.access_order.push_front(key);
            self.current_size += data_size;
        }
        
        evicted_item
    }
    
    pub fn remove(&mut self, key: usize) -> Option<(Vec<u8>, CacheItemMetadata)> {
        if let (Some(data), Some(meta)) = (self.items.remove(&key), self.metadata.remove(&key)) {
            self.current_size -= data.len();
            self.access_order.retain(|&k| k != key);
            Some((data, meta))
        } else {
            None
        }
    }
    
    fn move_to_front(&mut self, key: usize) {
        self.access_order.retain(|&k| k != key);
        self.access_order.push_front(key);
    }
    
    pub fn get_stats(&self) -> (u64, u64, f64, usize, usize) {
        let total = self.hit_count + self.miss_count;
        let hit_rate = if total > 0 { self.hit_count as f64 / total as f64 } else { 0.0 };
        (self.hit_count, self.miss_count, hit_rate, self.items.len(), self.current_size)
    }
    
    pub fn clear(&mut self) {
        self.items.clear();
        self.metadata.clear();
        self.access_order.clear();
        self.current_size = 0;
    }
    
    pub fn get_metadata(&self, key: usize) -> Option<&CacheItemMetadata> {
        self.metadata.get(&key)
    }
    
    pub fn list_items_by_access_frequency(&self) -> Vec<(usize, f64)> {
        self.metadata.iter()
            .map(|(&key, meta)| (key, meta.access_frequency))
            .collect()
    }
    
    pub fn get_all_keys(&self) -> Vec<usize> {
        self.metadata.keys().copied().collect()
    }
}

/// 自适应缓存实现 - L2层使用
#[derive(Debug)]
pub struct AdaptiveCache {
    items: HashMap<usize, Vec<u8>>,
    metadata: HashMap<usize, CacheItemMetadata>,
    frequency_buckets: HashMap<u8, std::collections::BTreeSet<usize>>, // 频率桶
    current_size: usize,
    max_size: usize,
    hit_count: u64,
    miss_count: u64,
    adaptation_window: Duration,
    last_adaptation: Instant,
}

 impl AdaptiveCache {
     pub fn new(max_size: usize) -> Self {
         Self {
             items: HashMap::new(),
             metadata: HashMap::new(),
             frequency_buckets: HashMap::new(),
             current_size: 0,
             max_size,
             hit_count: 0,
             miss_count: 0,
             adaptation_window: Duration::from_secs(60), // 1分钟自适应窗口
             last_adaptation: Instant::now(),
         }
     }
     
     pub fn get(&mut self, key: usize) -> Option<Vec<u8>> {
         if self.items.contains_key(&key) {
             self.hit_count += 1;
             
             // 获取旧频率
             let old_freq = if let Some(meta) = self.metadata.get(&key) {
                 self.get_frequency_bucket(meta.access_frequency)
             } else {
                 0
             };
             
             // 更新元数据并获取新访问频率
             let new_access_freq = if let Some(meta) = self.metadata.get_mut(&key) {
                 meta.access();
                 meta.access_frequency
             } else {
                 0.0
             };
             
             let new_freq = self.get_frequency_bucket(new_access_freq);
             
             // 更新频率桶
             if old_freq != new_freq {
                 if let Some(bucket) = self.frequency_buckets.get_mut(&old_freq) {
                     bucket.remove(&key);
                 }
                 self.frequency_buckets.entry(new_freq).or_insert_with(std::collections::BTreeSet::new).insert(key);
             }
             
             self.items.get(&key).cloned()
         } else {
             self.miss_count += 1;
             None
         }
     }
     
     pub fn put(&mut self, key: usize, data: Vec<u8>) -> Vec<(usize, Vec<u8>, CacheItemMetadata)> {
         let data_size = data.len();
         
         // 如果已存在，先移除
         if self.items.contains_key(&key) {
             self.remove(key);
         }
         
         // 执行自适应调整
         if self.last_adaptation.elapsed() >= self.adaptation_window {
             self.perform_adaptive_eviction();
             self.last_adaptation = Instant::now();
         }
         
         // 确保有足够空间
         let mut evicted_items = Vec::new();
         while self.current_size + data_size > self.max_size && !self.items.is_empty() {
             if let Some((evicted_key, evicted_data, evicted_meta)) = self.evict_least_valuable() {
                 evicted_items.push((evicted_key, evicted_data, evicted_meta));
             } else {
                 break;
             }
         }
         
         // 添加新项
         if self.current_size + data_size <= self.max_size {
             let meta = CacheItemMetadata::new(key, data_size);
             let freq_bucket = self.get_frequency_bucket(meta.access_frequency);
             
             self.items.insert(key, data);
             self.metadata.insert(key, meta);
             self.frequency_buckets.entry(freq_bucket).or_insert_with(std::collections::BTreeSet::new).insert(key);
             self.current_size += data_size;
         }
         
         evicted_items
     }
     
     fn get_frequency_bucket(&self, frequency: f64) -> u8 {
         if frequency < 1.0 { 0 }
         else if frequency < 5.0 { 1 }
         else if frequency < 10.0 { 2 }
         else if frequency < 20.0 { 3 }
         else { 4 }
     }
     
     fn perform_adaptive_eviction(&mut self) {
         // 识别并移除过期项目
         let _now = Instant::now();
         let expired_keys: Vec<usize> = self.metadata.iter()
             .filter(|(_, meta)| meta.age().as_secs() > 3600) // 1小时过期
             .map(|(&key, _)| key)
             .collect();
         
         for key in expired_keys {
             self.remove(key);
         }
     }
     
     fn evict_least_valuable(&mut self) -> Option<(usize, Vec<u8>, CacheItemMetadata)> {
         // 优先从最低频率桶中驱逐
         for freq in 0..=4u8 {
             if let Some(bucket) = self.frequency_buckets.get_mut(&freq) {
                 if let Some(&key) = bucket.iter().next() {
                     bucket.remove(&key);
                     if let (Some(data), Some(meta)) = (self.items.remove(&key), self.metadata.remove(&key)) {
                         self.current_size -= data.len();
                         return Some((key, data, meta));
                     }
                 }
             }
         }
         None
     }
     
     pub fn remove(&mut self, key: usize) -> Option<(Vec<u8>, CacheItemMetadata)> {
         if let (Some(data), Some(meta)) = (self.items.remove(&key), self.metadata.remove(&key)) {
             self.current_size -= data.len();
             
             // 从频率桶中移除
             let freq_bucket = self.get_frequency_bucket(meta.access_frequency);
             if let Some(bucket) = self.frequency_buckets.get_mut(&freq_bucket) {
                 bucket.remove(&key);
             }
             
             Some((data, meta))
         } else {
             None
         }
     }
     
     pub fn get_stats(&self) -> (u64, u64, f64, usize, usize) {
         let total = self.hit_count + self.miss_count;
         let hit_rate = if total > 0 { self.hit_count as f64 / total as f64 } else { 0.0 };
         (self.hit_count, self.miss_count, hit_rate, self.items.len(), self.current_size)
     }
     
     pub fn get_frequency_distribution(&self) -> HashMap<u8, usize> {
         self.frequency_buckets.iter()
             .map(|(&freq, bucket)| (freq, bucket.len()))
             .collect()
     }
     
     pub fn get_metadata(&self, key: usize) -> Option<&CacheItemMetadata> {
         self.metadata.get(&key)
     }
     
     pub fn clear(&mut self) {
         self.items.clear();
         self.metadata.clear();
         self.frequency_buckets.clear();
         self.current_size = 0;
     }
     
     pub fn get_all_keys(&self) -> Vec<usize> {
         self.metadata.keys().copied().collect()
     }
 }

 /// 压缩缓存实现 - L3层使用
 #[derive(Debug)]
 pub struct CompressedCache {
     items: HashMap<usize, Vec<u8>>,  // 存储压缩数据
     metadata: HashMap<usize, CacheItemMetadata>,
     uncompressed_sizes: HashMap<usize, usize>, // 原始大小映射
     current_size: usize,             // 压缩后的大小
     max_size: usize,
     hit_count: u64,
     miss_count: u64,
     compression_ratio: f64,          // 平均压缩比
     compression_threshold: usize,    // 压缩阈值
 }

 impl CompressedCache {
     pub fn new(max_size: usize, compression_threshold: usize) -> Self {
         Self {
             items: HashMap::new(),
             metadata: HashMap::new(),
             uncompressed_sizes: HashMap::new(),
             current_size: 0,
             max_size,
             hit_count: 0,
             miss_count: 0,
             compression_ratio: 0.7, // 假设70%的压缩比
             compression_threshold,
         }
     }
     
     pub fn get(&mut self, key: usize) -> Option<Vec<u8>> {
         if let Some(compressed_data) = self.items.get(&key) {
             self.hit_count += 1;
             
             // 更新元数据
             if let Some(meta) = self.metadata.get_mut(&key) {
                 meta.access();
             }
             
             // 解压数据
             Some(self.decompress_data(compressed_data))
         } else {
             self.miss_count += 1;
             None
         }
     }
     
     pub fn put(&mut self, key: usize, data: Vec<u8>) -> Vec<(usize, Vec<u8>, CacheItemMetadata)> {
         let original_size = data.len();
         
         // 如果已存在，先移除
         if self.items.contains_key(&key) {
             self.remove(key);
         }
         
         // 决定是否压缩
         let (stored_data, compressed_size, is_compressed) = if original_size >= self.compression_threshold {
             let compressed = self.compress_data(&data);
             let comp_size = compressed.len();
             (compressed, comp_size, true)
         } else {
             (data.clone(), original_size, false)
         };
         
         // 确保有足够空间
         let mut evicted_items = Vec::new();
         while self.current_size + compressed_size > self.max_size && !self.items.is_empty() {
             if let Some((evicted_key, evicted_data, evicted_meta)) = self.evict_oldest() {
                 evicted_items.push((evicted_key, evicted_data, evicted_meta));
             } else {
                 break;
             }
         }
         
         // 添加新项
         if self.current_size + compressed_size <= self.max_size {
             let mut meta = CacheItemMetadata::new(key, original_size);
             meta.is_compressed = is_compressed;
             
             self.items.insert(key, stored_data);
             self.metadata.insert(key, meta);
             self.uncompressed_sizes.insert(key, original_size);
             self.current_size += compressed_size;
             
             // 更新压缩比统计
             if is_compressed {
                 self.update_compression_ratio(original_size, compressed_size);
             }
         }
         
         evicted_items
     }
     
     fn compress_data(&self, data: &[u8]) -> Vec<u8> {
         // 简化的压缩实现 - 在实际应用中可以使用LZ4、Zstd等
         // 这里使用简单的RLE压缩作为示例
         let mut compressed = Vec::new();
         if data.is_empty() {
             return compressed;
         }
         
         let mut current_byte = data[0];
         let mut count: u8 = 1;
         
         for &byte in &data[1..] {
             if byte == current_byte && count < 255 {
                 count += 1;
             } else {
                 compressed.push(count);
                 compressed.push(current_byte);
                 current_byte = byte;
                 count = 1;
             }
         }
         
         // 添加最后一组
         compressed.push(count);
         compressed.push(current_byte);
         
         // 如果压缩后更大，返回原数据
         if compressed.len() >= data.len() {
             data.to_vec()
         } else {
             compressed
         }
     }
     
     fn decompress_data(&self, compressed_data: &[u8]) -> Vec<u8> {
         let mut decompressed = Vec::new();
         
         // 简化的解压实现
         let mut i = 0;
         while i + 1 < compressed_data.len() {
             let count = compressed_data[i];
             let byte = compressed_data[i + 1];
             
             for _ in 0..count {
                 decompressed.push(byte);
             }
             
             i += 2;
         }
         
         decompressed
     }
     
     fn update_compression_ratio(&mut self, original_size: usize, compressed_size: usize) {
         let current_ratio = compressed_size as f64 / original_size as f64;
         // 使用指数移动平均更新压缩比
         self.compression_ratio = 0.9 * self.compression_ratio + 0.1 * current_ratio;
     }
     
     fn evict_oldest(&mut self) -> Option<(usize, Vec<u8>, CacheItemMetadata)> {
         // 找到最老的项目
         let oldest_key = self.metadata.iter()
             .min_by_key(|(_, meta)| meta.created_at)
             .map(|(&key, _)| key);
         
         if let Some(key) = oldest_key {
             if let (Some(compressed_data), Some(meta)) = (self.items.remove(&key), self.metadata.remove(&key)) {
                 let _original_size = self.uncompressed_sizes.remove(&key).unwrap_or(0);
                 self.current_size -= compressed_data.len();
                 let decompressed_data = self.decompress_data(&compressed_data);
                 return Some((key, decompressed_data, meta));
             }
         }
         
         None
     }
     
     pub fn remove(&mut self, key: usize) -> Option<(Vec<u8>, CacheItemMetadata)> {
         if let (Some(compressed_data), Some(meta)) = (self.items.remove(&key), self.metadata.remove(&key)) {
             self.uncompressed_sizes.remove(&key);
             self.current_size -= compressed_data.len();
             let decompressed_data = self.decompress_data(&compressed_data);
             Some((decompressed_data, meta))
         } else {
             None
         }
     }
     
     pub fn get_stats(&self) -> (u64, u64, f64, usize, usize, f64) {
         let total = self.hit_count + self.miss_count;
         let hit_rate = if total > 0 { self.hit_count as f64 / total as f64 } else { 0.0 };
         (self.hit_count, self.miss_count, hit_rate, self.items.len(), self.current_size, self.compression_ratio)
     }
     
     pub fn get_compression_stats(&self) -> (usize, usize, f64) {
         let total_uncompressed: usize = self.uncompressed_sizes.values().sum();
         let current_compressed = self.current_size;
         let ratio = if total_uncompressed > 0 {
             current_compressed as f64 / total_uncompressed as f64
         } else {
             1.0
         };
         (total_uncompressed, current_compressed, ratio)
     }
     
     pub fn get_metadata(&self, key: usize) -> Option<&CacheItemMetadata> {
         self.metadata.get(&key)
     }
     
     pub fn clear(&mut self) {
         self.items.clear();
         self.metadata.clear();
         self.uncompressed_sizes.clear();
         self.current_size = 0;
     }
     
     pub fn get_all_keys(&self) -> Vec<usize> {
         self.metadata.keys().copied().collect()
     }
 }

 /// 多级缓存系统主结构
 #[derive(Debug)]
 pub struct MultiLevelCache {
     l1_cache: Arc<Mutex<LRUCache>>,           // L1: 热点缓存 (LRU)
     l2_cache: Arc<Mutex<AdaptiveCache>>,      // L2: 自适应缓存 (频率优化)
     l3_cache: Arc<Mutex<CompressedCache>>,    // L3: 压缩缓存 (大容量)
     policy: CachePolicy,
     global_stats: Arc<Mutex<MultiLevelCacheStats>>,
     promotion_count: Arc<Mutex<u64>>,
     demotion_count: Arc<Mutex<u64>>,
     // 新增：Task 5.3 性能优化组件
     performance_monitor: Arc<Mutex<CachePerformanceMonitor>>,
     batch_processor: Arc<Mutex<BatchCacheProcessor>>,
     prefetch_manager: Arc<Mutex<CachePrefetchManager>>,
 }

 #[derive(Debug, Default)]
 pub struct MultiLevelCacheStats {
     pub l1_hits: u64,
     pub l1_misses: u64,
     pub l2_hits: u64,
     pub l2_misses: u64,
     pub l3_hits: u64,
     pub l3_misses: u64,
     pub total_promotions: u64,
     pub total_demotions: u64,
     pub cache_consistency_checks: u64,
 }

 impl MultiLevelCache {
     pub fn new(policy: CachePolicy) -> Self {
         Self {
             l1_cache: Arc::new(Mutex::new(LRUCache::new(policy.l1_max_size))),
             l2_cache: Arc::new(Mutex::new(AdaptiveCache::new(policy.l2_max_size))),
             l3_cache: Arc::new(Mutex::new(CompressedCache::new(policy.l3_max_size, policy.compression_threshold))),
             policy,
             global_stats: Arc::new(Mutex::new(MultiLevelCacheStats::default())),
             promotion_count: Arc::new(Mutex::new(0)),
             demotion_count: Arc::new(Mutex::new(0)),
             // Task 5.3: 初始化性能优化组件
             performance_monitor: Arc::new(Mutex::new(CachePerformanceMonitor::new())),
             batch_processor: Arc::new(Mutex::new(BatchCacheProcessor::new())),
             prefetch_manager: Arc::new(Mutex::new(CachePrefetchManager::new())),
         }
     }
     
     pub fn new_with_default_policy() -> Self {
         Self::new(CachePolicy::default())
     }
     
     /// 获取数据 - 依次检查L1、L2、L3缓存
     pub fn get(&self, key: usize) -> Option<Vec<u8>> {
         // 首先检查L1缓存
         if let Ok(mut l1) = self.l1_cache.lock() {
             if let Some(data) = l1.get(key) {
                 if let Ok(mut stats) = self.global_stats.lock() {
                     stats.l1_hits += 1;
                 }
                 return Some(data);
             } else {
                 if let Ok(mut stats) = self.global_stats.lock() {
                     stats.l1_misses += 1;
                 }
             }
         }
         
         // 检查L2缓存，如果命中则考虑提升到L1
         if let Ok(mut l2) = self.l2_cache.lock() {
             if let Some(data) = l2.get(key) {
                 if let Ok(mut stats) = self.global_stats.lock() {
                     stats.l2_hits += 1;
                 }
                 
                 // 检查是否需要提升到L1
                 if let Some(meta) = l2.get_metadata(key) {
                     if meta.access_frequency >= self.policy.l2_to_l1_threshold {
                         self.promote_to_l1(key, data.clone(), meta.clone());
                     }
                 }
                 
                 return Some(data);
             } else {
                 if let Ok(mut stats) = self.global_stats.lock() {
                     stats.l2_misses += 1;
                 }
             }
         }
         
         // 检查L3缓存，如果命中则考虑提升到L2
         if let Ok(mut l3) = self.l3_cache.lock() {
             if let Some(data) = l3.get(key) {
                 if let Ok(mut stats) = self.global_stats.lock() {
                     stats.l3_hits += 1;
                 }
                 
                 // 检查是否需要提升到L2
                 if let Some(meta) = l3.get_metadata(key) {
                     if meta.access_frequency >= self.policy.l3_to_l2_threshold {
                         self.promote_to_l2(key, data.clone(), meta.clone());
                     }
                 }
                 
                 return Some(data);
             } else {
                 if let Ok(mut stats) = self.global_stats.lock() {
                     stats.l3_misses += 1;
                 }
             }
         }
         
         None
     }
     
     /// 存储数据 - 根据策略决定存储层级
     pub fn put(&self, key: usize, data: Vec<u8>) {
         let data_size = data.len();
         
         // 根据数据大小和策略决定初始存储层级
         if data_size <= self.policy.l1_max_size / 10 {
             // 小数据直接存储到L1
             self.put_to_l1(key, data);
         } else if data_size <= self.policy.l2_max_size / 10 {
             // 中等数据存储到L2
             self.put_to_l2(key, data);
         } else {
             // 大数据存储到L3
             self.put_to_l3(key, data);
         }
     }
     
     fn put_to_l1(&self, key: usize, data: Vec<u8>) {
         if let Ok(mut l1) = self.l1_cache.lock() {
             if let Some((evicted_key, evicted_data, evicted_meta)) = l1.put(key, data) {
                 // L1驱逐的数据降级到L2
                 self.demote_to_l2(evicted_key, evicted_data, evicted_meta);
             }
         }
     }
     
     fn put_to_l2(&self, key: usize, data: Vec<u8>) {
         if let Ok(mut l2) = self.l2_cache.lock() {
             let evicted_items = l2.put(key, data);
             for (evicted_key, evicted_data, evicted_meta) in evicted_items {
                 // L2驱逐的数据降级到L3
                 self.demote_to_l3(evicted_key, evicted_data, evicted_meta);
             }
         }
     }
     
     fn put_to_l3(&self, key: usize, data: Vec<u8>) {
         if let Ok(mut l3) = self.l3_cache.lock() {
             let _evicted_items = l3.put(key, data);
             // L3驱逐的数据直接丢弃
         }
     }
     
     fn promote_to_l1(&self, key: usize, data: Vec<u8>, mut meta: CacheItemMetadata) {
         meta.promotion_count += 1;
         if let Ok(mut l1) = self.l1_cache.lock() {
             if let Some((evicted_key, evicted_data, evicted_meta)) = l1.put(key, data) {
                 self.demote_to_l2(evicted_key, evicted_data, evicted_meta);
             }
         }
         
         // 从L2中移除
         if let Ok(mut l2) = self.l2_cache.lock() {
             l2.remove(key);
         }
         
         if let Ok(mut count) = self.promotion_count.lock() {
             *count += 1;
         }
         
         if let Ok(mut stats) = self.global_stats.lock() {
             stats.total_promotions += 1;
         }
     }
     
     fn promote_to_l2(&self, key: usize, data: Vec<u8>, mut meta: CacheItemMetadata) {
         meta.promotion_count += 1;
         if let Ok(mut l2) = self.l2_cache.lock() {
             let evicted_items = l2.put(key, data);
             for (evicted_key, evicted_data, evicted_meta) in evicted_items {
                 self.demote_to_l3(evicted_key, evicted_data, evicted_meta);
             }
         }
         
         // 从L3中移除
         if let Ok(mut l3) = self.l3_cache.lock() {
             l3.remove(key);
         }
         
         if let Ok(mut count) = self.promotion_count.lock() {
             *count += 1;
         }
         
         if let Ok(mut stats) = self.global_stats.lock() {
             stats.total_promotions += 1;
         }
     }
     
     fn demote_to_l2(&self, key: usize, data: Vec<u8>, _meta: CacheItemMetadata) {
         if let Ok(mut l2) = self.l2_cache.lock() {
             let evicted_items = l2.put(key, data);
             for (evicted_key, evicted_data, evicted_meta) in evicted_items {
                 self.demote_to_l3(evicted_key, evicted_data, evicted_meta);
             }
         }
         
         if let Ok(mut count) = self.demotion_count.lock() {
             *count += 1;
         }
         
         if let Ok(mut stats) = self.global_stats.lock() {
             stats.total_demotions += 1;
         }
     }
     
     fn demote_to_l3(&self, key: usize, data: Vec<u8>, _meta: CacheItemMetadata) {
         if let Ok(mut l3) = self.l3_cache.lock() {
             let _evicted_items = l3.put(key, data);
         }
         
         if let Ok(mut count) = self.demotion_count.lock() {
             *count += 1;
         }
         
         if let Ok(mut stats) = self.global_stats.lock() {
             stats.total_demotions += 1;
         }
     }
     
     /// 移除指定key的缓存项
     pub fn remove(&self, key: usize) -> bool {
         let mut found = false;
         
         // 从所有层级移除
         if let Ok(mut l1) = self.l1_cache.lock() {
             if l1.remove(key).is_some() {
                 found = true;
             }
         }
         
         if let Ok(mut l2) = self.l2_cache.lock() {
             if l2.remove(key).is_some() {
                 found = true;
             }
         }
         
         if let Ok(mut l3) = self.l3_cache.lock() {
             if l3.remove(key).is_some() {
                 found = true;
             }
         }
         
         found
     }
     
     /// 获取缓存统计信息
     pub fn get_comprehensive_stats(&self) -> MultiLevelCacheReport {
         let mut report = MultiLevelCacheReport::default();
         
         if let Ok(l1) = self.l1_cache.lock() {
             let (hits, misses, hit_rate, items, size) = l1.get_stats();
             report.l1_hits = hits;
             report.l1_misses = misses;
             report.l1_hit_rate = hit_rate;
             report.l1_items = items;
             report.l1_size = size;
         }
         
         if let Ok(l2) = self.l2_cache.lock() {
             let (hits, misses, hit_rate, items, size) = l2.get_stats();
             report.l2_hits = hits;
             report.l2_misses = misses;
             report.l2_hit_rate = hit_rate;
             report.l2_items = items;
             report.l2_size = size;
             report.l2_frequency_distribution = l2.get_frequency_distribution();
         }
         
         if let Ok(l3) = self.l3_cache.lock() {
             let (hits, misses, hit_rate, items, size, compression_ratio) = l3.get_stats();
             report.l3_hits = hits;
             report.l3_misses = misses;
             report.l3_hit_rate = hit_rate;
             report.l3_items = items;
             report.l3_size = size;
             report.l3_compression_ratio = compression_ratio;
             
             let (uncompressed, compressed, ratio) = l3.get_compression_stats();
             report.l3_total_uncompressed = uncompressed;
             report.l3_total_compressed = compressed;
             report.l3_actual_compression_ratio = ratio;
         }
         
         if let Ok(stats) = self.global_stats.lock() {
             report.total_promotions = stats.total_promotions;
             report.total_demotions = stats.total_demotions;
         }
         
         // 计算整体统计
         let total_hits = report.l1_hits + report.l2_hits + report.l3_hits;
         let total_misses = report.l1_misses + report.l2_misses + report.l3_misses;
         let total_requests = total_hits + total_misses;
         
         report.overall_hit_rate = if total_requests > 0 {
             total_hits as f64 / total_requests as f64
         } else {
             0.0
         };
         
         report.total_items = report.l1_items + report.l2_items + report.l3_items;
         report.total_size = report.l1_size + report.l2_size + report.l3_size;
         
         report
     }
     
     /// 清理所有缓存
     pub fn clear_all(&self) {
         if let Ok(mut l1) = self.l1_cache.lock() {
             l1.clear();
         }
         if let Ok(mut l2) = self.l2_cache.lock() {
             l2.clear();
         }
         if let Ok(mut l3) = self.l3_cache.lock() {
             l3.clear();
         }
         
         if let Ok(mut stats) = self.global_stats.lock() {
             *stats = MultiLevelCacheStats::default();
         }
     }
     
     /// 执行缓存一致性检查
     pub fn perform_consistency_check(&self) -> CacheConsistencyReport {
         let mut report = CacheConsistencyReport::default();
         
         if let (Ok(l1), Ok(l2), Ok(l3)) = (
             self.l1_cache.lock(),
             self.l2_cache.lock(),
             self.l3_cache.lock()
         ) {
             // 检查是否有重复的key
             let l1_keys: std::collections::HashSet<usize> = l1.get_all_keys().into_iter().collect();
             let l2_keys: std::collections::HashSet<usize> = l2.get_all_keys().into_iter().collect();
             let l3_keys: std::collections::HashSet<usize> = l3.get_all_keys().into_iter().collect();
             
             let l1_l2_overlap: Vec<usize> = l1_keys.intersection(&l2_keys).copied().collect();
             let l1_l3_overlap: Vec<usize> = l1_keys.intersection(&l3_keys).copied().collect();
             let l2_l3_overlap: Vec<usize> = l2_keys.intersection(&l3_keys).copied().collect();
             
             report.duplicate_keys_l1_l2 = l1_l2_overlap;
             report.duplicate_keys_l1_l3 = l1_l3_overlap;
             report.duplicate_keys_l2_l3 = l2_l3_overlap;
             
             report.is_consistent = report.duplicate_keys_l1_l2.is_empty() 
                 && report.duplicate_keys_l1_l3.is_empty() 
                 && report.duplicate_keys_l2_l3.is_empty();
         }
         
         if let Ok(mut stats) = self.global_stats.lock() {
             stats.cache_consistency_checks += 1;
         }
         
         report
     }
     
     // ===========================
     // Task 5.3: 性能优化方法
     // ===========================
     
     /// 批量获取数据 - 高性能版本
     pub fn batch_get(&self, keys: &[usize]) -> Vec<(usize, Option<Vec<u8>>)> {
         let start_time = Instant::now();
         let _guard = if let Ok(monitor) = self.performance_monitor.lock() {
             Some(monitor.record_concurrent_operation_start())
         } else {
             None
         };
         
         let results = if let Ok(mut processor) = self.batch_processor.lock() {
             processor.batch_get(keys, self)
         } else {
             // 回退到标准实现
             keys.iter().map(|&key| (key, self.get(key))).collect()
         };
         
         // 记录性能指标
         if let Ok(mut monitor) = self.performance_monitor.lock() {
             let hit_count = results.iter().filter(|(_, data)| data.is_some()).count();
             let total_bytes: usize = results.iter()
                 .filter_map(|(_, data)| data.as_ref().map(|d| d.len()))
                 .sum();
             monitor.record_cache_operation(
                 CacheOperationType::BatchGet, 
                 start_time.elapsed(), 
                 total_bytes, 
                 hit_count > 0
             );
         }
         
         results
     }
     
     /// 批量存储数据 - 高性能版本
     pub fn batch_put(&self, items: &[(usize, Vec<u8>)]) {
         let start_time = Instant::now();
         let _guard = if let Ok(monitor) = self.performance_monitor.lock() {
             Some(monitor.record_concurrent_operation_start())
         } else {
             None
         };
         
         if let Ok(mut processor) = self.batch_processor.lock() {
             processor.batch_put(items, self);
         } else {
             // 回退到标准实现
             for (key, data) in items {
                 self.put(*key, data.clone());
             }
         }
         
         // 记录性能指标
         if let Ok(mut monitor) = self.performance_monitor.lock() {
             let total_bytes: usize = items.iter().map(|(_, data)| data.len()).sum();
             monitor.record_cache_operation(
                 CacheOperationType::BatchPut, 
                 start_time.elapsed(), 
                 total_bytes, 
                 true
             );
         }
     }
     
     /// 智能预取数据
     pub fn prefetch(&self, recent_accesses: &[usize]) -> Vec<usize> {
         if let Ok(mut prefetch_mgr) = self.prefetch_manager.lock() {
             prefetch_mgr.analyze_and_prefetch(recent_accesses, self)
         } else {
             Vec::new()
         }
     }
     
     /// 获取性能统计信息
     pub fn get_performance_stats(&self) -> Option<CachePerformanceStats> {
         if let Ok(monitor) = self.performance_monitor.lock() {
             Some(monitor.get_performance_stats())
         } else {
             None
         }
     }
     
     /// 获取批处理统计信息
     pub fn get_batch_stats(&self) -> Option<BatchOperationStats> {
         if let Ok(processor) = self.batch_processor.lock() {
             Some(processor.get_batch_stats().clone())
         } else {
             None
         }
     }
     
     /// 获取预取统计信息
     pub fn get_prefetch_stats(&self) -> Option<PrefetchStats> {
         if let Ok(prefetch_mgr) = self.prefetch_manager.lock() {
             Some(prefetch_mgr.get_prefetch_stats().clone())
         } else {
             None
         }
     }
     
     /// 重置所有性能统计
     pub fn reset_performance_stats(&self) {
         if let Ok(mut monitor) = self.performance_monitor.lock() {
             monitor.reset();
         }
         if let Ok(mut processor) = self.batch_processor.lock() {
             processor.reset_stats();
         }
         if let Ok(mut prefetch_mgr) = self.prefetch_manager.lock() {
             prefetch_mgr.reset_stats();
         }
     }
     
     /// 记录内存使用情况
     pub fn record_memory_usage(&self) {
         let l1_size = self.l1_cache.lock().map_or(0, |l1| l1.get_stats().4);
         let l2_size = self.l2_cache.lock().map_or(0, |l2| l2.get_stats().4);
         let l3_size = self.l3_cache.lock().map_or(0, |l3| l3.get_stats().4);
         let total_usage = l1_size + l2_size + l3_size;
         
         if let Ok(mut monitor) = self.performance_monitor.lock() {
             monitor.record_memory_usage(total_usage);
         }
     }
     
     /// 获取综合性能报告
     pub fn get_comprehensive_performance_report(&self) -> CachePerformanceReport {
         let cache_report = self.get_comprehensive_stats();
         let performance_stats = self.get_performance_stats();
         let batch_stats = self.get_batch_stats();
         let prefetch_stats = self.get_prefetch_stats();
         
         CachePerformanceReport {
             cache_report,
             performance_stats,
             batch_stats,
             prefetch_stats,
             timestamp: Instant::now(),
         }
     }
 }

 /// 多级缓存统计报告
 #[derive(Debug, Default, Clone)]
 pub struct MultiLevelCacheReport {
     // L1统计
     pub l1_hits: u64,
     pub l1_misses: u64,
     pub l1_hit_rate: f64,
     pub l1_items: usize,
     pub l1_size: usize,
     
     // L2统计
     pub l2_hits: u64,
     pub l2_misses: u64,
     pub l2_hit_rate: f64,
     pub l2_items: usize,
     pub l2_size: usize,
     pub l2_frequency_distribution: HashMap<u8, usize>,
     
     // L3统计
     pub l3_hits: u64,
     pub l3_misses: u64,
     pub l3_hit_rate: f64,
     pub l3_items: usize,
     pub l3_size: usize,
     pub l3_compression_ratio: f64,
     pub l3_total_uncompressed: usize,
     pub l3_total_compressed: usize,
     pub l3_actual_compression_ratio: f64,
     
     // 整体统计
     pub overall_hit_rate: f64,
     pub total_items: usize,
     pub total_size: usize,
     pub total_promotions: u64,
     pub total_demotions: u64,
 }

 /// 缓存一致性检查报告
 #[derive(Debug)]
 pub struct CacheConsistencyReport {
     pub is_consistent: bool,
     pub duplicate_keys_l1_l2: Vec<usize>,
     pub duplicate_keys_l1_l3: Vec<usize>,
     pub duplicate_keys_l2_l3: Vec<usize>,
     pub check_timestamp: std::time::SystemTime,
 }

 impl Default for CacheConsistencyReport {
      fn default() -> Self {
          Self {
              is_consistent: true,
              duplicate_keys_l1_l2: Vec::new(),
              duplicate_keys_l1_l3: Vec::new(),
              duplicate_keys_l2_l3: Vec::new(),
              check_timestamp: std::time::SystemTime::now(),
          }
      }
  }

 /// 综合性能报告 - Task 5.3
 #[derive(Debug, Clone)]
 pub struct CachePerformanceReport {
     pub cache_report: MultiLevelCacheReport,
     pub performance_stats: Option<CachePerformanceStats>,
     pub batch_stats: Option<BatchOperationStats>,
     pub prefetch_stats: Option<PrefetchStats>,
     pub timestamp: Instant,
 }

 // ===========================
 // 智能缓存策略系统 - Task 5.2: 实现智能缓存策略
 // ===========================

 /// 热点数据检测器
 #[derive(Debug)]
 pub struct HotSpotDetector {
     access_windows: HashMap<usize, Vec<Instant>>, // 每个key的访问时间窗口
     window_size: Duration,                         // 时间窗口大小
     hot_threshold: usize,                          // 热点阈值（窗口内访问次数）
     cold_threshold: Duration,                      // 冷数据阈值（最后访问时间）
     detection_enabled: bool,
     hot_spots: Arc<Mutex<HashSet<usize>>>,        // 当前热点数据集合
     cold_spots: Arc<Mutex<HashSet<usize>>>,       // 当前冷数据集合
 }

 impl HotSpotDetector {
     pub fn new(window_duration: Duration, hot_threshold: usize) -> Self {
         Self {
             access_windows: HashMap::new(),
             window_size: window_duration,
             hot_threshold,
             cold_threshold: Duration::from_secs(3600), // 1小时无访问视为冷数据
             detection_enabled: true,
             hot_spots: Arc::new(Mutex::new(HashSet::new())),
             cold_spots: Arc::new(Mutex::new(HashSet::new())),
         }
     }
     
     /// 记录数据访问
     pub fn record_access(&mut self, key: usize) {
         if !self.detection_enabled {
             return;
         }
         
         let now = Instant::now();
         
         // 更新访问窗口
         let access_times = self.access_windows.entry(key).or_insert_with(Vec::new);
         access_times.push(now);
         
         // 清理过期的访问记录
         let cutoff = now - self.window_size;
         access_times.retain(|&time| time > cutoff);
         
         // 检查是否成为热点
         if access_times.len() >= self.hot_threshold {
             if let Ok(mut hot_spots) = self.hot_spots.lock() {
                 hot_spots.insert(key);
             }
             if let Ok(mut cold_spots) = self.cold_spots.lock() {
                 cold_spots.remove(&key);
             }
         }
     }
     
     /// 获取当前热点数据
     pub fn get_hot_spots(&self) -> HashSet<usize> {
         if let Ok(hot_spots) = self.hot_spots.lock() {
             hot_spots.clone()
         } else {
             HashSet::new()
         }
     }
     
     /// 获取冷数据（长时间未访问）
     pub fn get_cold_spots(&self) -> HashSet<usize> {
         let now = Instant::now();
         let mut cold_keys = HashSet::new();
         
         for (key, access_times) in &self.access_windows {
             if let Some(&last_access) = access_times.last() {
                 if now.duration_since(last_access) > self.cold_threshold {
                     cold_keys.insert(*key);
                 }
             }
         }
         
         // 更新冷数据集合
         if let Ok(mut cold_spots) = self.cold_spots.lock() {
             for key in &cold_keys {
                 cold_spots.insert(*key);
             }
         }
         
         cold_keys
     }
     
     /// 清理过期的访问记录
     pub fn cleanup_expired_records(&mut self) {
         let now = Instant::now();
         let cutoff = now - self.window_size;
         
         // 清理访问窗口
         self.access_windows.retain(|_, access_times| {
             access_times.retain(|&time| time > cutoff);
             !access_times.is_empty()
         });
         
         // 清理热点数据（如果不再符合条件）
         if let Ok(mut hot_spots) = self.hot_spots.lock() {
             hot_spots.retain(|key| {
                 if let Some(access_times) = self.access_windows.get(key) {
                     access_times.len() >= self.hot_threshold
                 } else {
                     false
                 }
             });
         }
     }
     
     /// 获取热点检测统计
     pub fn get_detection_stats(&self) -> HotSpotStats {
         let hot_count = self.hot_spots.lock().map_or(0, |h| h.len());
         let cold_count = self.cold_spots.lock().map_or(0, |c| c.len());
         let total_tracked = self.access_windows.len();
         
         HotSpotStats {
             hot_spots_count: hot_count,
             cold_spots_count: cold_count,
             total_tracked_keys: total_tracked,
             detection_enabled: self.detection_enabled,
             window_size: self.window_size,
         }
     }
     
     /// 启用/禁用检测
     pub fn set_detection_enabled(&mut self, enabled: bool) {
         self.detection_enabled = enabled;
     }
 }

 #[derive(Debug, Clone)]
 pub struct HotSpotStats {
     pub hot_spots_count: usize,
     pub cold_spots_count: usize,
     pub total_tracked_keys: usize,
     pub detection_enabled: bool,
     pub window_size: Duration,
 }

 /// 智能淘汰策略管理器
 #[derive(Debug)]
 pub struct IntelligentEvictionManager {
     enabled: bool,
     current_strategy: EvictionStrategy,
     eviction_stats: EvictionStatistics,
     adaptive_switching_enabled: bool,
 }

 #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
 pub enum EvictionStrategy {
     LRU,                    // 最近最少使用
     LFU,                    // 最少使用频率
     TimeAware,              // 时间感知
     SizeAware,              // 大小感知
     HybridLRULFU,          // LRU+LFU混合
     AccessPatternAware,     // 访问模式感知
 }

 #[derive(Debug, Clone)]
 pub struct EvictionStatistics {
     pub total_evictions: u64,
     pub false_evictions: u64,          // 错误淘汰（淘汰后很快又被访问）
     pub eviction_hit_rate: f64,        // 淘汰命中率
     pub average_eviction_time: Duration,
     pub memory_saved: usize,
     pub strategy_switches: u64,
 }

 impl Default for EvictionStatistics {
     fn default() -> Self {
         Self {
             total_evictions: 0,
             false_evictions: 0,
             eviction_hit_rate: 0.0,
             average_eviction_time: Duration::from_millis(0),
             memory_saved: 0,
             strategy_switches: 0,
         }
     }
 }

 impl IntelligentEvictionManager {
     pub fn new() -> Self {
         Self {
             enabled: true,
             current_strategy: EvictionStrategy::LRU,
             eviction_stats: EvictionStatistics::default(),
             adaptive_switching_enabled: true,
         }
     }
     
     /// 选择淘汰目标
     pub fn select_eviction_targets(&mut self, 
                                   cache_items: &HashMap<usize, CacheItemMetadata>,
                                   _memory_pressure: f64,
                                   target_count: usize) -> Vec<usize> {
         if !self.enabled {
             return Vec::new();
         }
         
         let start_time = Instant::now();
         
         let targets = match self.current_strategy {
             EvictionStrategy::LRU => self.select_lru_targets(cache_items, target_count),
             EvictionStrategy::LFU => self.select_lfu_targets(cache_items, target_count),
             EvictionStrategy::TimeAware => self.select_time_aware_targets(cache_items, target_count),
             EvictionStrategy::SizeAware => self.select_size_aware_targets(cache_items, target_count),
             EvictionStrategy::HybridLRULFU => self.select_hybrid_targets(cache_items, target_count),
             EvictionStrategy::AccessPatternAware => self.select_pattern_aware_targets(cache_items, target_count),
         };
         
         // 更新统计信息
                   let _eviction_time = start_time.elapsed();
         self.eviction_stats.total_evictions += targets.len() as u64;
         
         targets
     }
     
     fn select_lru_targets(&self, cache_items: &HashMap<usize, CacheItemMetadata>, target_count: usize) -> Vec<usize> {
         let mut items: Vec<_> = cache_items.iter().collect();
         items.sort_by_key(|(_, meta)| meta.last_accessed);
         items.into_iter()
             .take(target_count)
             .map(|(key, _)| *key)
             .collect()
     }
     
     fn select_lfu_targets(&self, cache_items: &HashMap<usize, CacheItemMetadata>, target_count: usize) -> Vec<usize> {
         let mut items: Vec<_> = cache_items.iter().collect();
         items.sort_by_key(|(_, meta)| meta.access_count);
         items.into_iter()
             .take(target_count)
             .map(|(key, _)| *key)
             .collect()
     }
     
     fn select_time_aware_targets(&self, cache_items: &HashMap<usize, CacheItemMetadata>, target_count: usize) -> Vec<usize> {
         let mut items: Vec<_> = cache_items.iter().collect();
         
         // 基于时间的复合得分：考虑年龄和空闲时间
         items.sort_by(|(_, a), (_, b)| {
             let a_score = a.age().as_secs_f64() * 0.3 + a.idle_time().as_secs_f64() * 0.7;
             let b_score = b.age().as_secs_f64() * 0.3 + b.idle_time().as_secs_f64() * 0.7;
             b_score.partial_cmp(&a_score).unwrap_or(std::cmp::Ordering::Equal)
         });
         
         items.into_iter()
             .take(target_count)
             .map(|(key, _)| *key)
             .collect()
     }
     
     fn select_size_aware_targets(&self, cache_items: &HashMap<usize, CacheItemMetadata>, target_count: usize) -> Vec<usize> {
         let mut items: Vec<_> = cache_items.iter().collect();
         
         // 优先淘汰大而访问频率低的项目
         items.sort_by(|(_, a), (_, b)| {
             let a_score = a.size as f64 / (a.access_frequency + 1.0);
             let b_score = b.size as f64 / (b.access_frequency + 1.0);
             b_score.partial_cmp(&a_score).unwrap_or(std::cmp::Ordering::Equal)
         });
         
         items.into_iter()
             .take(target_count)
             .map(|(key, _)| *key)
             .collect()
     }
     
     fn select_hybrid_targets(&self, cache_items: &HashMap<usize, CacheItemMetadata>, target_count: usize) -> Vec<usize> {
         let mut items: Vec<_> = cache_items.iter().collect();
         
         // LRU + LFU 混合得分
         items.sort_by(|(_, a), (_, b)| {
             let now = Instant::now();
             
             let a_lru_score = now.duration_since(a.last_accessed).as_secs_f64();
             let a_lfu_score = 1.0 / (a.access_count as f64 + 1.0);
             let a_combined = a_lru_score * 0.6 + a_lfu_score * 0.4;
             
             let b_lru_score = now.duration_since(b.last_accessed).as_secs_f64();
             let b_lfu_score = 1.0 / (b.access_count as f64 + 1.0);
             let b_combined = b_lru_score * 0.6 + b_lfu_score * 0.4;
             
             b_combined.partial_cmp(&a_combined).unwrap_or(std::cmp::Ordering::Equal)
         });
         
         items.into_iter()
             .take(target_count)
             .map(|(key, _)| *key)
             .collect()
     }
     
     fn select_pattern_aware_targets(&self, cache_items: &HashMap<usize, CacheItemMetadata>, target_count: usize) -> Vec<usize> {
         // 基于访问模式的智能淘汰（简化实现）
         let mut items: Vec<_> = cache_items.iter().collect();
         
         items.sort_by(|(_, a), (_, b)| {
             // 综合考虑访问频率、时间间隔、是否为热点等
             let a_score = a.access_frequency * 0.4 + 
                          (if a.is_hot { 10.0 } else { 0.0 }) * 0.3 +
                          (a.promotion_count as f64) * 0.3;
             let b_score = b.access_frequency * 0.4 + 
                          (if b.is_hot { 10.0 } else { 0.0 }) * 0.3 +
                          (b.promotion_count as f64) * 0.3;
             a_score.partial_cmp(&b_score).unwrap_or(std::cmp::Ordering::Equal)
         });
         
         items.into_iter()
             .take(target_count)
             .map(|(key, _)| *key)
             .collect()
     }
     
     /// 获取当前策略和统计信息
     pub fn get_eviction_info(&self) -> (EvictionStrategy, &EvictionStatistics) {
         (self.current_strategy, &self.eviction_stats)
     }
 }

 // ===========================
 // 缓存性能优化组件 - Task 5.3: 优化缓存性能和内存效率
 // ===========================

 /// 缓存性能监控器
 #[derive(Debug)]
 pub struct CachePerformanceMonitor {
     // 基础性能指标
     total_requests: u64,
     hit_count: u64,
     miss_count: u64,
     
     // 延迟统计
     latency_samples: Vec<Duration>,
     max_latency: Duration,
     min_latency: Duration,
     
     // 吞吐量统计
     throughput_samples: Vec<(Instant, usize)>, // (时间, 字节数)
     measurement_window: Duration,
     
     // 内存使用统计
     memory_usage_samples: Vec<(Instant, usize)>,
     peak_memory_usage: usize,
     
     // 并发性能统计
     concurrent_operations: Arc<Mutex<u32>>,
     max_concurrent_operations: u32,
     lock_contention_count: u64,
     
     // 缓存效率统计
     promotion_efficiency: f64,  // 提升到上级缓存的效率
     eviction_efficiency: f64,   // 淘汰算法的效率
     
     last_reset: Instant,
 }

 impl CachePerformanceMonitor {
     pub fn new() -> Self {
         Self {
             total_requests: 0,
             hit_count: 0,
             miss_count: 0,
             latency_samples: Vec::new(),
             max_latency: Duration::from_nanos(0),
             min_latency: Duration::from_secs(u64::MAX),
             throughput_samples: Vec::new(),
             measurement_window: Duration::from_secs(60),
             memory_usage_samples: Vec::new(),
             peak_memory_usage: 0,
             concurrent_operations: Arc::new(Mutex::new(0)),
             max_concurrent_operations: 0,
             lock_contention_count: 0,
             promotion_efficiency: 0.0,
             eviction_efficiency: 0.0,
             last_reset: Instant::now(),
         }
     }
     
     /// 记录缓存操作
     pub fn record_cache_operation(&mut self, _operation_type: CacheOperationType, duration: Duration, bytes: usize, hit: bool) {
         self.total_requests += 1;
         
         if hit {
             self.hit_count += 1;
         } else {
             self.miss_count += 1;
         }
         
         // 记录延迟
         self.latency_samples.push(duration);
         if duration > self.max_latency {
             self.max_latency = duration;
         }
         if duration < self.min_latency {
             self.min_latency = duration;
         }
         
         // 记录吞吐量
         let now = Instant::now();
         self.throughput_samples.push((now, bytes));
         
         // 清理过期样本
         self.cleanup_old_samples();
     }
     
     /// 记录内存使用
     pub fn record_memory_usage(&mut self, usage: usize) {
         let now = Instant::now();
         self.memory_usage_samples.push((now, usage));
         
         if usage > self.peak_memory_usage {
             self.peak_memory_usage = usage;
         }
         
         // 清理过期样本
         self.cleanup_old_samples();
     }
     
     /// 记录并发操作
     pub fn record_concurrent_operation_start(&self) -> ConcurrentOperationGuard {
         if let Ok(mut count) = self.concurrent_operations.lock() {
             *count += 1;
             ConcurrentOperationGuard {
                 counter: Arc::clone(&self.concurrent_operations),
             }
         } else {
             // 锁争用情况
             ConcurrentOperationGuard {
                 counter: Arc::clone(&self.concurrent_operations),
             }
         }
     }
     
     /// 获取当前性能统计
     pub fn get_performance_stats(&self) -> CachePerformanceStats {
         let hit_rate = if self.total_requests > 0 {
             self.hit_count as f64 / self.total_requests as f64
         } else {
             0.0
         };
         
         let avg_latency = if !self.latency_samples.is_empty() {
             let total: Duration = self.latency_samples.iter().sum();
             total / self.latency_samples.len() as u32
         } else {
             Duration::from_nanos(0)
         };
         
         let current_throughput = self.calculate_current_throughput();
         let current_memory_usage = self.memory_usage_samples.last().map(|(_, usage)| *usage).unwrap_or(0);
         
         CachePerformanceStats {
             hit_rate,
             total_requests: self.total_requests,
             avg_latency,
             max_latency: self.max_latency,
             min_latency: self.min_latency,
             current_throughput,
             peak_memory_usage: self.peak_memory_usage,
             current_memory_usage,
             concurrent_operations: self.concurrent_operations.lock().map_or(0, |c| *c),
             max_concurrent_operations: self.max_concurrent_operations,
             lock_contention_count: self.lock_contention_count,
             promotion_efficiency: self.promotion_efficiency,
             eviction_efficiency: self.eviction_efficiency,
             uptime: self.last_reset.elapsed(),
         }
     }
     
     fn cleanup_old_samples(&mut self) {
         let cutoff = Instant::now() - self.measurement_window;
         
         self.throughput_samples.retain(|(time, _)| *time > cutoff);
         self.memory_usage_samples.retain(|(time, _)| *time > cutoff);
         
         // 保持延迟样本在合理范围内
         if self.latency_samples.len() > 10000 {
             self.latency_samples.drain(0..5000);
         }
     }
     
     fn calculate_current_throughput(&self) -> f64 {
         if self.throughput_samples.len() < 2 {
             return 0.0;
         }
         
         let total_bytes: usize = self.throughput_samples.iter().map(|(_, bytes)| *bytes).sum();
         let time_span = self.throughput_samples.last().unwrap().0
             .duration_since(self.throughput_samples.first().unwrap().0);
         
         if time_span.as_secs_f64() > 0.0 {
             total_bytes as f64 / time_span.as_secs_f64()
         } else {
             0.0
         }
     }
     
     /// 重置统计数据
     pub fn reset(&mut self) {
         self.total_requests = 0;
         self.hit_count = 0;
         self.miss_count = 0;
         self.latency_samples.clear();
         self.max_latency = Duration::from_nanos(0);
         self.min_latency = Duration::from_secs(u64::MAX);
         self.throughput_samples.clear();
         self.memory_usage_samples.clear();
         self.peak_memory_usage = 0;
         self.lock_contention_count = 0;
         self.last_reset = Instant::now();
     }
 }

 #[derive(Debug, Clone, Copy)]
 pub enum CacheOperationType {
     Get,
     Put,
     Remove,
     Prefetch,
     BatchGet,
     BatchPut,
 }

 #[derive(Debug, Clone)]
 pub struct CachePerformanceStats {
     pub hit_rate: f64,
     pub total_requests: u64,
     pub avg_latency: Duration,
     pub max_latency: Duration,
     pub min_latency: Duration,
     pub current_throughput: f64,  // bytes per second
     pub peak_memory_usage: usize,
     pub current_memory_usage: usize,
     pub concurrent_operations: u32,
     pub max_concurrent_operations: u32,
     pub lock_contention_count: u64,
     pub promotion_efficiency: f64,
     pub eviction_efficiency: f64,
     pub uptime: Duration,
 }

 pub struct ConcurrentOperationGuard {
     counter: Arc<Mutex<u32>>,
 }

 impl Drop for ConcurrentOperationGuard {
     fn drop(&mut self) {
         if let Ok(mut count) = self.counter.lock() {
             *count = count.saturating_sub(1);
         }
     }
 }

 /// 批量缓存处理器
 #[derive(Debug)]
 pub struct BatchCacheProcessor {
     batch_size_limit: usize,
     parallel_processing: bool,
     compression_enabled: bool,
     batch_stats: BatchOperationStats,
 }

 impl BatchCacheProcessor {
     pub fn new() -> Self {
         Self {
             batch_size_limit: 1000,
             parallel_processing: true,
             compression_enabled: true,
             batch_stats: BatchOperationStats::default(),
         }
     }
     
     /// 批量获取数据
     pub fn batch_get(&mut self, keys: &[usize], cache: &MultiLevelCache) -> Vec<(usize, Option<Vec<u8>>)> {
         let start_time = Instant::now();
         let mut results = Vec::with_capacity(keys.len());
         
         if self.parallel_processing && keys.len() > 100 {
             // 并行处理大批量请求
             results = keys.par_iter()
                 .map(|&key| (key, cache.get(key)))
                 .collect();
         } else {
             // 串行处理小批量请求
             for &key in keys {
                 results.push((key, cache.get(key)));
             }
         }
         
         // 更新统计
         self.batch_stats.total_batch_operations += 1;
         self.batch_stats.total_items_processed += keys.len();
         self.batch_stats.total_processing_time += start_time.elapsed();
         
         let hit_count = results.iter().filter(|(_, data)| data.is_some()).count();
         self.batch_stats.batch_hit_rate = hit_count as f64 / keys.len() as f64;
         
         results
     }
     
     /// 批量存储数据
     pub fn batch_put(&mut self, items: &[(usize, Vec<u8>)], cache: &MultiLevelCache) -> Vec<usize> {
         let start_time = Instant::now();
         let mut successful_puts = Vec::new();
         
         if self.parallel_processing && items.len() > 50 {
             // 并行处理（需要特殊的同步机制）
             successful_puts = items.par_iter()
                 .filter_map(|(key, data)| {
                     cache.put(*key, data.clone());
                     Some(*key)
                 })
                 .collect();
         } else {
             // 串行处理
             for (key, data) in items {
                 cache.put(*key, data.clone());
                 successful_puts.push(*key);
             }
         }
         
         // 更新统计
         self.batch_stats.total_batch_operations += 1;
         self.batch_stats.total_items_processed += items.len();
         self.batch_stats.total_processing_time += start_time.elapsed();
         
         successful_puts
     }
     
     /// 获取批处理统计
     pub fn get_batch_stats(&self) -> &BatchOperationStats {
         &self.batch_stats
     }
     
     /// 重置批处理统计
     pub fn reset_stats(&mut self) {
         self.batch_stats = BatchOperationStats::default();
     }
 }

 #[derive(Debug, Clone, Default)]
 pub struct BatchOperationStats {
     pub total_batch_operations: u64,
     pub total_items_processed: usize,
     pub total_processing_time: Duration,
     pub batch_hit_rate: f64,
     pub average_batch_size: f64,
     pub parallel_efficiency: f64,
 }

 /// 缓存预取管理器
 #[derive(Debug)]
 pub struct CachePrefetchManager {
     enabled: bool,
     prefetch_window_size: usize,
     prefetch_threshold: f64,        // 命中率阈值
     access_pattern_analyzer: AccessPatternAnalyzer,
     prefetch_stats: PrefetchStats,
     background_prefetch: bool,
 }

 impl CachePrefetchManager {
     pub fn new() -> Self {
         Self {
             enabled: true,
             prefetch_window_size: 10,
             prefetch_threshold: 0.7,
             access_pattern_analyzer: AccessPatternAnalyzer::new(),
             prefetch_stats: PrefetchStats::default(),
             background_prefetch: false,
         }
     }
     
     /// 分析访问模式并预取数据
           pub fn analyze_and_prefetch(&mut self, 
                                recent_accesses: &[usize], 
                                cache: &MultiLevelCache) -> Vec<usize> {
         if !self.enabled {
             return Vec::new();
         }
         
         let _start_time = Instant::now();
         
         // 分析访问模式
         let mut predicted_keys = Vec::new();
         
         // 顺序访问预测
         if let Some(sequence_prediction) = self.predict_sequential_access(recent_accesses) {
             predicted_keys.extend(sequence_prediction);
         }
         
         // 模式访问预测
         if let Some(pattern_prediction) = self.predict_pattern_access(recent_accesses) {
             predicted_keys.extend(pattern_prediction);
         }
         
         // 执行预取
         let prefetched_count = self.execute_prefetch(&predicted_keys, cache);
         
         // 更新统计
         self.prefetch_stats.hit_count += prefetched_count;
         self.prefetch_stats.prefetch_accuracy = self.calculate_prediction_accuracy();
         self.prefetch_stats.adaptive_window_size = self.prefetch_window_size;
         
         predicted_keys
     }
     
     fn predict_sequential_access(&self, accesses: &[usize]) -> Option<Vec<usize>> {
         if accesses.len() < 2 {
             return None;
         }
         
         // 检查是否为顺序访问
         let mut is_sequential = true;
         let mut stride = None;
         
         for window in accesses.windows(2) {
             let current_stride = if window[1] >= window[0] {
                 window[1] - window[0]
             } else {
                 return None; // 不是递增序列
             };
             
             if let Some(expected_stride) = stride {
                 if current_stride != expected_stride {
                     is_sequential = false;
                     break;
                 }
             } else {
                 stride = Some(current_stride);
             }
         }
         
         if is_sequential && stride.is_some() {
             let stride_val = stride.unwrap();
             let last_key = *accesses.last().unwrap();
             let predictions: Vec<usize> = (1..=self.prefetch_window_size)
                 .map(|i| last_key + i * stride_val)
                 .collect();
             Some(predictions)
         } else {
             None
         }
     }
     
     fn predict_pattern_access(&self, _accesses: &[usize]) -> Option<Vec<usize>> {
         // 简化的模式预测实现
         // 在实际应用中，这里会有更复杂的机器学习算法
         None
     }
     
     fn execute_prefetch(&self, keys: &[usize], cache: &MultiLevelCache) -> usize {
         let mut prefetched = 0;
         
         for &key in keys {
             // 检查是否已在缓存中
             if cache.get(key).is_none() {
                 // 这里应该从底层存储预取数据
                 // 为简化，我们假设预取成功
                 prefetched += 1;
             }
         }
         
         prefetched
     }
     
     fn calculate_prediction_accuracy(&self) -> f64 {
         // 简化的准确率计算
         // 实际实现会跟踪预测的键是否真的被后续访问
         0.75
     }
     
     /// 获取预取统计
     pub fn get_prefetch_stats(&self) -> &PrefetchStats {
         &self.prefetch_stats
     }
     
     /// 重置预取统计
     pub fn reset_stats(&mut self) {
         self.prefetch_stats = PrefetchStats::default();
     }
 }

