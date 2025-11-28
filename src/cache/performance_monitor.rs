//! 缓存性能监控模块
//!
//! 提供缓存系统的性能指标收集、分析和报告功能

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// 缓存操作类型
#[derive(Debug, Clone, Copy)]
pub enum CacheOperationType {
    Get,
    Put,
    Remove,
    Prefetch,
    BatchGet,
    BatchPut,
}

/// 缓存性能统计数据
#[derive(Debug, Clone)]
pub struct CachePerformanceStats {
    pub hit_rate: f64,
    pub total_requests: u64,
    pub avg_latency: Duration,
    pub max_latency: Duration,
    pub min_latency: Duration,
    pub current_throughput: f64, // bytes per second
    pub peak_memory_usage: usize,
    pub current_memory_usage: usize,
    pub concurrent_operations: u32,
    pub max_concurrent_operations: u32,
    pub lock_contention_count: u64,
    pub promotion_efficiency: f64,
    pub eviction_efficiency: f64,
    pub uptime: Duration,
}

/// 并发操作守卫
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
    promotion_efficiency: f64, // 提升到上级缓存的效率
    eviction_efficiency: f64,  // 淘汰算法的效率

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
    pub fn record_cache_operation(
        &mut self,
        _operation_type: CacheOperationType,
        duration: Duration,
        bytes: usize,
        hit: bool,
    ) {
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

    /// 获取当前性能统计 (已弃用，保留空实现以兼容)
    #[deprecated(note = "性能统计功能已移除")]
    pub fn get_performance_stats(&self) -> CachePerformanceStats {
        CachePerformanceStats {
            hit_rate: 0.0,
            total_requests: 0,
            avg_latency: Duration::from_nanos(0),
            max_latency: Duration::from_nanos(0),
            min_latency: Duration::from_nanos(0),
            current_throughput: 0.0,
            peak_memory_usage: 0,
            current_memory_usage: 0,
            concurrent_operations: 0,
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

        let total_bytes: usize = self
            .throughput_samples
            .iter()
            .map(|(_, bytes)| *bytes)
            .sum();
        let time_span = self
            .throughput_samples
            .last()
            .unwrap()
            .0
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
