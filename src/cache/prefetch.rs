//! 智能预取系统
//!
//! 提供访问模式预测、多级预取缓存和自适应预取管理

use crate::access_pattern::AccessPatternPredictor;
use std::collections::HashMap;
use std::time::{Duration, Instant};
// use crate::access_pattern::AccessPatternAnalysis;  // 暂时注释掉未使用的导入

// 预取策略枚举
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PrefetchStrategy {
    Conservative, // 保守预取，低内存占用
    Aggressive,   // 激进预取，高性能
    Adaptive,     // 自适应预取，平衡性能和内存
    Disabled,     // 禁用预取
}

// 预取级别枚举 - 对应L1/L2/L3缓存优化
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PrefetchLevel {
    L1, // L1缓存级别预取 (32KB)
    L2, // L2缓存级别预取 (256KB)
    L3, // L3缓存级别预取 (8MB)
}

// 多级预取缓存
#[derive(Debug)]
pub struct MultiLevelPrefetchCache {
    l1_cache: HashMap<usize, Vec<u8>>, // 最热数据
    l2_cache: HashMap<usize, Vec<u8>>, // 次热数据
    l3_cache: HashMap<usize, Vec<u8>>, // 预取数据
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
            l1_capacity: 32,   // L1缓存32个条目
            l2_capacity: 256,  // L2缓存256个条目
            l3_capacity: 1024, // L3缓存1024个条目
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
    }

    fn promote_to_l2(&mut self, key: usize, data: Vec<u8>) {
        if self.l2_cache.len() >= self.l2_capacity {
            self.evict_from_l2();
        }
        self.l2_cache.insert(key, data);
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
        // 找到访问次数最少的项
        if let Some((&least_used_key, _)) =
            self.l1_access_count.iter().min_by_key(|(_, &count)| count)
        {
            if let Some(data) = self.l1_cache.remove(&least_used_key) {
                self.l1_access_count.remove(&least_used_key);
                // 降级到L2
                self.put_l2(least_used_key, data);
            }
        } else if let Some((&first_key, _)) = self.l1_cache.iter().next() {
            let data = self.l1_cache.remove(&first_key).unwrap();
            self.put_l2(first_key, data);
        }
    }

    fn evict_from_l2(&mut self) {
        // 找到访问次数最少的项
        if let Some((&least_used_key, _)) =
            self.l2_access_count.iter().min_by_key(|(_, &count)| count)
        {
            if let Some(data) = self.l2_cache.remove(&least_used_key) {
                self.l2_access_count.remove(&least_used_key);
                // 降级到L3
                self.put_l3(least_used_key, data);
            }
        } else if let Some((&first_key, _)) = self.l2_cache.iter().next() {
            let data = self.l2_cache.remove(&first_key).unwrap();
            self.put_l3(first_key, data);
        }
    }

    fn evict_from_l3(&mut self) {
        // 简单的FIFO驱逐
        if let Some((&first_key, _)) = self.l3_cache.iter().next() {
            if let Some(data) = self.l3_cache.remove(&first_key) {
                self.total_memory_usage -= data.len();
            }
        }
    }

    fn evict_to_fit(&mut self, needed_size: usize) {
        while self.total_memory_usage + needed_size > self.max_memory_usage
            && (!self.l1_cache.is_empty() || !self.l2_cache.is_empty() || !self.l3_cache.is_empty())
        {
            // 优先从L3驱逐
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

    pub fn get_memory_usage(&self) -> usize {
        self.total_memory_usage
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
    memory_bandwidth_limit: usize, // 每秒最大预取字节数
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
    pub fn predict_and_prefetch(&mut self, indices: &[usize]) {
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
            self.execute_intelligent_prefetch(&predicted_indices);
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
        }
        .max(self.min_window_size)
        .min(self.max_window_size);

        self.pattern_predictor
            .predict_next_accesses(current_indices, effective_window)
    }

    /// 执行智能预取 - 多级缓存策略
    fn execute_intelligent_prefetch(&mut self, indices: &[usize]) {
        let now = Instant::now();

        // 重置带宽计数器（每秒）
        if now.duration_since(self.last_prefetch_time) >= Duration::from_secs(1) {
            self.bytes_prefetched_this_second = 0;
            self.last_prefetch_time = now;
        }

        let mut prefetched_bytes = 0;
        let estimate_item_size = 1024; // 估计的项目大小

        for &idx in indices {
            // 检查是否已在缓存中
            if self.prefetch_cache.get(idx).is_some() {
                continue;
            }

            // 检查内存带宽限制
            if self.bytes_prefetched_this_second + prefetched_bytes + estimate_item_size
                > self.memory_bandwidth_limit
            {
                break;
            }

            // 模拟预取数据（在实际实现中这里会从文件系统读取）
            let simulated_data = vec![0u8; estimate_item_size];

            // 根据访问模式决定缓存级别
            let cache_level = self.determine_cache_level(idx, &simulated_data);
            self.prefetch_cache.put(idx, simulated_data, cache_level);
            prefetched_bytes += estimate_item_size;
        }

        self.bytes_prefetched_this_second += prefetched_bytes;
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

        estimated_cpu_usage < self.cpu_threshold
            && estimated_memory_pressure < self.memory_pressure_threshold
    }

    /// 自适应窗口大小调整
    fn adaptive_window_adjustment(&mut self) {
        let hit_rate = if self.hit_count + self.miss_count > 0 {
            self.hit_count as f64 / (self.hit_count + self.miss_count) as f64
        } else {
            0.5
        };

        if hit_rate > 0.8 {
            // 高命中率，增加窗口大小
            self.adaptive_window = ((self.adaptive_window as f64 * self.window_adjustment_factor)
                as usize)
                .min(self.max_window_size);
        } else if hit_rate < 0.4 {
            // 低命中率，减少窗口大小
            self.adaptive_window = ((self.adaptive_window as f64 / self.window_adjustment_factor)
                as usize)
                .max(self.min_window_size);
        }
    }

    /// 获取预取数据
    pub fn get_prefetched_data(&mut self, key: usize) -> Option<Vec<u8>> {
        if let Some(data) = self.prefetch_cache.get(key) {
            self.hit_count += 1;
            Some(data)
        } else {
            self.miss_count += 1;
            None
        }
    }

    /// 调整窗口大小
    pub fn adjust_window_size(&mut self, hit_rate: f64) {
        if hit_rate > 0.8 {
            self.adaptive_window = (self.adaptive_window + 1).min(self.max_window_size);
        } else if hit_rate < 0.4 {
            self.adaptive_window =
                (self.adaptive_window.saturating_sub(1)).max(self.min_window_size);
        }
    }

    /// 获取详细统计信息
    pub fn get_detailed_stats(&self) -> PrefetchStats {
        let hit_rate = if self.hit_count + self.miss_count > 0 {
            self.hit_count as f64 / (self.hit_count + self.miss_count) as f64
        } else {
            0.0
        };

        PrefetchStats {
            hit_count: self.hit_count,
            miss_count: self.miss_count,
            hit_rate,
            prefetch_accuracy: self.prefetch_accuracy,
            adaptive_window_size: self.adaptive_window,
            memory_usage: self.prefetch_cache.get_memory_usage(),
        }
    }

    /// 清理预取缓存
    pub fn clear_cache(&mut self) {
        self.prefetch_cache.clear();
        self.hit_count = 0;
        self.miss_count = 0;
    }

    pub fn get_hit_rate(&self) -> f64 {
        let total = self.hit_count + self.miss_count;
        if total == 0 {
            0.0
        } else {
            self.hit_count as f64 / total as f64
        }
    }

    pub fn get_hit_count(&self) -> u64 {
        self.hit_count as u64
    }

    pub fn get_miss_count(&self) -> u64 {
        self.miss_count as u64
    }
}

// 预取统计信息
#[derive(Debug, Clone, Default)]
pub struct PrefetchStats {
    pub hit_count: usize,
    pub miss_count: usize,
    pub hit_rate: f64,
    pub prefetch_accuracy: f64,
    pub adaptive_window_size: usize,
    pub memory_usage: usize,
}
