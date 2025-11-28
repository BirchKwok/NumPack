//! 性能指标定义
//!
//! 从lazy_array_original.rs中提取的性能监控相关结构

use std::time::{Duration, Instant};

/// 算法类型枚举
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum IndexAlgorithm {
    // 花式索引算法
    FancyDirect,   // 直接访问
    FancySIMD,     // SIMD优化
    FancyPrefetch, // 预取优化
    FancyZeroCopy, // 零拷贝

    // 布尔索引算法
    BooleanBitmap,       // 位图索引
    BooleanHierarchical, // 分层索引
    BooleanSparse,       // 稀疏选择器
    BooleanDense,        // 密集向量化
    BooleanExtreme,      // 极限SIMD

    // 批量访问算法
    BatchParallel,  // 并行处理
    BatchChunked,   // 分块处理
    BatchStreaming, // 流式处理
}

/// 性能指标结构
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

impl PerformanceMetrics {
    pub fn new() -> Self {
        Self::default()
    }

    /// 更新平均延迟（使用指数移动平均）
    pub fn update_latency(&mut self, new_latency: Duration) {
        let current_latency_ms = self.average_latency.as_millis() as f64;
        let new_latency_ms = new_latency.as_millis() as f64;
        self.average_latency =
            Duration::from_millis(((current_latency_ms * 0.9) + (new_latency_ms * 0.1)) as u64);
        self.last_updated = Instant::now();
    }

    /// 更新吞吐量（使用指数移动平均）
    pub fn update_throughput(&mut self, data_size: usize, duration: Duration) {
        let throughput_mbps = (data_size as f64 / (1024.0 * 1024.0)) / duration.as_secs_f64();
        self.throughput = (self.throughput * 0.9) + (throughput_mbps * 0.1);
        self.last_updated = Instant::now();
    }

    /// 更新缓存命中率
    pub fn update_cache_hit_rate(&mut self, hit_rate: f64) {
        self.cache_hit_rate = hit_rate.clamp(0.0, 1.0);
        self.last_updated = Instant::now();
    }

    /// 更新CPU利用率
    pub fn update_cpu_utilization(&mut self, cpu_usage: f64) {
        self.cpu_utilization = cpu_usage.clamp(0.0, 1.0);
        self.last_updated = Instant::now();
    }

    /// 更新内存使用量
    pub fn update_memory_usage(&mut self, memory_bytes: usize) {
        self.memory_usage = memory_bytes;
        self.last_updated = Instant::now();
    }

    /// 检查指标是否需要更新
    pub fn needs_update(&self, max_age: Duration) -> bool {
        self.last_updated.elapsed() > max_age
    }

    /// 获取性能评分（0-1，1表示最佳性能）
    pub fn get_performance_score(&self) -> f64 {
        let latency_score =
            (1.0 / (1.0 + self.average_latency.as_millis() as f64 / 100.0)).clamp(0.0, 1.0);
        let throughput_score = (self.throughput / 100.0).clamp(0.0, 1.0); // 假设100MB/s为最大
        let cache_score = self.cache_hit_rate;
        let cpu_score = 1.0 - self.cpu_utilization.clamp(0.0, 1.0);
        let memory_score =
            1.0 - (self.memory_usage as f64 / (1024.0 * 1024.0 * 1024.0)).clamp(0.0, 1.0); // 假设1GB为最大

        (latency_score + throughput_score + cache_score + cpu_score + memory_score) / 5.0
    }
}
