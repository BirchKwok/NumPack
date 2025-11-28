//! 性能分析器
//!
//! 从lazy_array_original.rs中提取的性能分析和监控功能

use super::metrics::{IndexAlgorithm, PerformanceMetrics};
use super::monitor::SystemMonitor;
use crate::access_pattern::{AccessPatternAnalysis, AccessPatternAnalyzer};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// 性能分析器，收集和分析系统性能数据
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

    /// 记录操作性能
    pub fn record_operation(
        &self,
        algorithm: IndexAlgorithm,
        duration: Duration,
        data_size: usize,
    ) {
        let mut history = self.operation_history.lock().unwrap();
        history.push((algorithm, duration, data_size));

        // 保持历史记录在合理范围内
        if history.len() > 1000 {
            history.drain(0..500);
        }

        // 更新性能指标
        self.update_metrics(algorithm, duration, data_size);
    }

    /// 分析访问模式
    pub fn analyze_access_pattern(&self, offset: usize, size: usize) -> AccessPatternAnalysis {
        let mut analyzer = self.access_analyzer.lock().unwrap();
        analyzer.analyze_access(offset, size)
    }

    /// 获取当前性能指标
    pub fn get_current_metrics(&self) -> PerformanceMetrics {
        let mut metrics = self.metrics.lock().unwrap();
        let mut monitor = self.system_monitor.lock().unwrap();

        // 更新系统监控数据
        metrics.cpu_utilization = monitor.get_cpu_utilization();
        metrics.memory_usage = monitor.get_memory_usage();
        metrics.last_updated = Instant::now();

        metrics.clone()
    }

    /// 更新性能指标
    fn update_metrics(&self, _algorithm: IndexAlgorithm, duration: Duration, data_size: usize) {
        let mut metrics = self.metrics.lock().unwrap();

        // 更新平均延迟
        metrics.update_latency(duration);

        // 更新吞吐量
        metrics.update_throughput(data_size, duration);
    }

    /// 获取特定算法的性能统计
    pub fn get_algorithm_performance(&self, algorithm: IndexAlgorithm) -> Option<(Duration, f64)> {
        let history = self.operation_history.lock().unwrap();
        let relevant_ops: Vec<_> = history
            .iter()
            .filter(|(alg, _, _)| *alg == algorithm)
            .collect();

        if relevant_ops.is_empty() {
            return None;
        }

        let avg_duration = relevant_ops
            .iter()
            .map(|(_, dur, _)| dur.as_nanos() as f64)
            .sum::<f64>()
            / relevant_ops.len() as f64;

        let avg_throughput = relevant_ops
            .iter()
            .map(|(_, dur, size)| (*size as f64 / (1024.0 * 1024.0)) / dur.as_secs_f64())
            .sum::<f64>()
            / relevant_ops.len() as f64;

        Some((Duration::from_nanos(avg_duration as u64), avg_throughput))
    }

    /// 获取所有算法的性能对比
    pub fn get_all_algorithms_performance(&self) -> Vec<(IndexAlgorithm, Duration, f64)> {
        let mut results = Vec::new();

        let algorithms = [
            IndexAlgorithm::FancyDirect,
            IndexAlgorithm::FancySIMD,
            IndexAlgorithm::FancyPrefetch,
            IndexAlgorithm::FancyZeroCopy,
            IndexAlgorithm::BooleanBitmap,
            IndexAlgorithm::BooleanHierarchical,
            IndexAlgorithm::BooleanSparse,
            IndexAlgorithm::BooleanDense,
            IndexAlgorithm::BooleanExtreme,
            IndexAlgorithm::BatchParallel,
            IndexAlgorithm::BatchChunked,
            IndexAlgorithm::BatchStreaming,
        ];

        for algorithm in algorithms {
            if let Some((duration, throughput)) = self.get_algorithm_performance(algorithm) {
                results.push((algorithm, duration, throughput));
            }
        }

        results
    }

    /// 清除历史记录
    pub fn clear_history(&self) {
        let mut history = self.operation_history.lock().unwrap();
        history.clear();

        let mut metrics = self.metrics.lock().unwrap();
        *metrics = PerformanceMetrics::default();
    }

    /// 获取性能分析器统计信息
    pub fn get_profiler_stats(&self) -> ProfilerStats {
        let history = self.operation_history.lock().unwrap();
        let metrics = self.metrics.lock().unwrap();

        ProfilerStats {
            total_operations: history.len(),
            total_data_processed: history.iter().map(|(_, _, size)| size).sum(),
            avg_latency: metrics.average_latency,
            avg_throughput: metrics.throughput,
            current_performance_score: metrics.get_performance_score(),
        }
    }

    /// 设置缓存命中率
    pub fn update_cache_hit_rate(&self, hit_rate: f64) {
        let mut metrics = self.metrics.lock().unwrap();
        metrics.update_cache_hit_rate(hit_rate);
    }
}

/// 性能分析器统计信息
#[derive(Debug, Clone)]
pub struct ProfilerStats {
    pub total_operations: usize,
    pub total_data_processed: usize,
    pub avg_latency: Duration,
    pub avg_throughput: f64,
    pub current_performance_score: f64,
}
