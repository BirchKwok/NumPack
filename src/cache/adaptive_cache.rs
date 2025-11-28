//! 自适应缓存策略
//!
//! 提供动态缓存大小调整、智能驱逐策略和运行时优化

use crate::cache::multilevel_cache::{CachePolicy, MultiLevelCache};
use crate::performance::SystemMonitor;
use std::collections::HashMap;
use std::time::{Duration, Instant};
// use crate::performance::PerformanceMetrics;  // 暂时注释掉未使用的导入

// 自适应缓存控制器
#[derive(Debug)]
pub struct AdaptiveCacheController {
    target_cache: MultiLevelCache,
    performance_monitor: SystemMonitor,
    adaptation_history: Vec<AdaptationEvent>,
    adaptation_interval: Duration,
    last_adaptation: Instant,

    // 动态调整参数
    min_l1_size: usize,
    max_l1_size: usize,
    min_l2_size: usize,
    max_l2_size: usize,
    min_l3_size: usize,
    max_l3_size: usize,

    // 性能阈值
    hit_rate_threshold: f64,
    memory_pressure_threshold: f64,
    cpu_usage_threshold: f64,
}

#[derive(Debug, Clone)]
pub struct AdaptationEvent {
    pub timestamp: Instant,
    pub event_type: AdaptationType,
    pub old_sizes: (usize, usize, usize), // L1, L2, L3
    pub new_sizes: (usize, usize, usize),
    pub trigger_reason: String,
    pub performance_impact: f64,
}

#[derive(Debug, Clone)]
pub enum AdaptationType {
    SizeIncrease,
    SizeDecrease,
    LayerRebalance,
    PolicyChange,
    CompressionAdjustment,
}

impl AdaptiveCacheController {
    pub fn new(initial_policy: CachePolicy) -> Self {
        Self {
            target_cache: MultiLevelCache::new(initial_policy.clone()),
            performance_monitor: SystemMonitor::new(),
            adaptation_history: Vec::new(),
            adaptation_interval: Duration::from_secs(30),
            last_adaptation: Instant::now(),

            // 设置动态调整范围
            min_l1_size: initial_policy.l1_max_size / 4,
            max_l1_size: initial_policy.l1_max_size * 4,
            min_l2_size: initial_policy.l2_max_size / 4,
            max_l2_size: initial_policy.l2_max_size * 4,
            min_l3_size: initial_policy.l3_max_size / 2,
            max_l3_size: initial_policy.l3_max_size * 2,

            // 性能阈值
            hit_rate_threshold: 0.8,
            memory_pressure_threshold: 0.8,
            cpu_usage_threshold: 0.7,
        }
    }

    /// 主要的自适应调整入口点
    pub fn perform_adaptive_optimization(&mut self) -> AdaptationResult {
        if self.last_adaptation.elapsed() < self.adaptation_interval {
            return AdaptationResult::NoActionNeeded;
        }

        // 性能统计功能已移除，自适应优化功能暂时禁用
        self.last_adaptation = Instant::now();
        AdaptationResult::NoActionNeeded
    }

    /// 调整L1缓存大小
    fn adjust_l1_size(&mut self, factor: f64, reason: &str) {
        // 注意：这里是简化实现，实际中需要重建缓存实例
        // 记录调整事件
        let event = AdaptationEvent {
            timestamp: Instant::now(),
            event_type: if factor > 1.0 {
                AdaptationType::SizeIncrease
            } else {
                AdaptationType::SizeDecrease
            },
            old_sizes: (0, 0, 0), // 简化，实际需要获取当前大小
            new_sizes: (0, 0, 0), // 简化，实际需要计算新大小
            trigger_reason: reason.to_string(),
            performance_impact: 0.0, // 需要后续测量
        };

        self.adaptation_history.push(event);
        self.last_adaptation = Instant::now();
    }

    /// 重新平衡缓存层级
    fn rebalance_cache_layers(&mut self) {
        // 性能统计功能已移除，重新平衡功能暂时禁用
        self.last_adaptation = Instant::now();
    }

    /// 调整压缩阈值
    fn adjust_compression_threshold(&mut self, new_threshold: usize) {
        // 记录压缩调整事件
        let event = AdaptationEvent {
            timestamp: Instant::now(),
            event_type: AdaptationType::CompressionAdjustment,
            old_sizes: (0, 0, 0),
            new_sizes: (0, 0, new_threshold),
            trigger_reason: "Compression threshold adjustment for CPU optimization".to_string(),
            performance_impact: 0.0,
        };

        self.adaptation_history.push(event);
        self.last_adaptation = Instant::now();
    }

    /// 获取自适应历史记录
    pub fn get_adaptation_history(&self) -> &[AdaptationEvent] {
        &self.adaptation_history
    }

    /// 清除旧的适应历史记录
    pub fn cleanup_old_history(&mut self, retention_duration: Duration) {
        let cutoff_time = Instant::now() - retention_duration;
        self.adaptation_history
            .retain(|event| event.timestamp > cutoff_time);
    }

    /// 获取自适应性能报告
    pub fn get_adaptation_report(&self) -> AdaptationReport {
        let recent_adaptations = self
            .adaptation_history
            .iter()
            .filter(|event| event.timestamp.elapsed() < Duration::from_secs(3600))
            .count();

        let avg_adaptation_interval = if self.adaptation_history.len() > 1 {
            let total_duration = self
                .adaptation_history
                .last()
                .unwrap()
                .timestamp
                .duration_since(self.adaptation_history.first().unwrap().timestamp);
            total_duration.as_secs_f64() / (self.adaptation_history.len() - 1) as f64
        } else {
            0.0
        };

        AdaptationReport {
            total_adaptations: self.adaptation_history.len(),
            recent_adaptations_1h: recent_adaptations,
            average_adaptation_interval_secs: avg_adaptation_interval,
            last_adaptation: self.last_adaptation,
            adaptation_types_count: self.count_adaptation_types(),
        }
    }

    fn count_adaptation_types(&self) -> HashMap<String, usize> {
        let mut counts = HashMap::new();

        for event in &self.adaptation_history {
            let type_name = match event.event_type {
                AdaptationType::SizeIncrease => "size_increase",
                AdaptationType::SizeDecrease => "size_decrease",
                AdaptationType::LayerRebalance => "layer_rebalance",
                AdaptationType::PolicyChange => "policy_change",
                AdaptationType::CompressionAdjustment => "compression_adjustment",
            };

            *counts.entry(type_name.to_string()).or_insert(0) += 1;
        }

        counts
    }

    /// 代理访问底层缓存
    pub fn get(&self, key: usize) -> Option<Vec<u8>> {
        self.target_cache.get(key)
    }

    pub fn put(&self, key: usize, data: Vec<u8>) {
        self.target_cache.put(key, data);
    }

    pub fn remove(&self, key: usize) -> bool {
        self.target_cache.remove(key)
    }

    pub fn clear(&self) {
        self.target_cache.clear();
    }
}

// 自适应策略枚举
#[derive(Debug, Clone)]
enum AdaptationStrategy {
    None,
    IncreaseL1Size(f64),
    DecreaseL1Size(f64),
    RebalanceLayers,
    AdjustCompression(usize),
}

// 自适应结果枚举
#[derive(Debug, Clone)]
pub enum AdaptationResult {
    NoActionNeeded,
    L1SizeIncreased(f64),
    L1SizeDecreased(f64),
    LayersRebalanced,
    CompressionAdjusted(usize),
}

// 自适应报告
#[derive(Debug, Clone)]
pub struct AdaptationReport {
    pub total_adaptations: usize,
    pub recent_adaptations_1h: usize,
    pub average_adaptation_interval_secs: f64,
    pub last_adaptation: Instant,
    pub adaptation_types_count: HashMap<String, usize>,
}

// 智能缓存预热器
#[derive(Debug)]
pub struct IntelligentCacheWarmer {
    target_cache: MultiLevelCache,
    warmup_strategies: Vec<WarmupStrategy>,
    warmup_history: Vec<WarmupEvent>,
}

#[derive(Debug, Clone)]
pub enum WarmupStrategy {
    SequentialPrefetch {
        start: usize,
        end: usize,
        step: usize,
    },
    PopularItemsPrefetch {
        items: Vec<usize>,
    },
    PatternBasedPrefetch {
        pattern: AccessPattern,
    },
    HeuristicPrefetch {
        heuristic: String,
    },
}

#[derive(Debug, Clone)]
pub enum AccessPattern {
    Sequential,
    Random,
    Hotspot(Vec<usize>),
    Temporal(Duration),
}

#[derive(Debug, Clone)]
pub struct WarmupEvent {
    pub timestamp: Instant,
    pub strategy: WarmupStrategy,
    pub items_prefetched: usize,
    pub time_taken: Duration,
    pub hit_rate_improvement: f64,
}

impl IntelligentCacheWarmer {
    pub fn new(cache: MultiLevelCache) -> Self {
        Self {
            target_cache: cache,
            warmup_strategies: Vec::new(),
            warmup_history: Vec::new(),
        }
    }

    /// 添加预热策略
    pub fn add_warmup_strategy(&mut self, strategy: WarmupStrategy) {
        self.warmup_strategies.push(strategy);
    }

    /// 执行智能预热
    pub fn perform_intelligent_warmup(&mut self) -> WarmupResult {
        let start_time = Instant::now();
        let mut total_items_prefetched = 0;

        // 执行所有预热策略
        for strategy in &self.warmup_strategies.clone() {
            let items_count = self.execute_warmup_strategy(strategy);
            total_items_prefetched += items_count;
        }

        let time_taken = start_time.elapsed();

        // 记录预热事件
        if let Some(strategy) = self.warmup_strategies.first() {
            let event = WarmupEvent {
                timestamp: start_time,
                strategy: strategy.clone(),
                items_prefetched: total_items_prefetched,
                time_taken,
                hit_rate_improvement: 0.0, // 需要后续测量
            };
            self.warmup_history.push(event);
        }

        WarmupResult {
            success: true,
            items_prefetched: total_items_prefetched,
            time_taken,
            strategies_executed: self.warmup_strategies.len(),
        }
    }

    fn execute_warmup_strategy(&mut self, strategy: &WarmupStrategy) -> usize {
        match strategy {
            WarmupStrategy::SequentialPrefetch { start, end, step } => {
                let mut count = 0;
                for i in (*start..*end).step_by(*step) {
                    // 模拟预取数据
                    let dummy_data = vec![0u8; 1024]; // 1KB dummy data
                    self.target_cache.put(i, dummy_data);
                    count += 1;
                }
                count
            }
            WarmupStrategy::PopularItemsPrefetch { items } => {
                for &item in items {
                    let dummy_data = vec![0u8; 2048]; // 2KB dummy data for popular items
                    self.target_cache.put(item, dummy_data);
                }
                items.len()
            }
            WarmupStrategy::PatternBasedPrefetch { pattern } => {
                match pattern {
                    AccessPattern::Sequential => {
                        // 顺序预取前100项
                        for i in 0..100 {
                            let dummy_data = vec![0u8; 1024];
                            self.target_cache.put(i, dummy_data);
                        }
                        100
                    }
                    AccessPattern::Random => {
                        // 随机预取50项
                        for i in 0..50 {
                            let random_key = (i * 17) % 1000; // 简单的伪随机
                            let dummy_data = vec![0u8; 1024];
                            self.target_cache.put(random_key, dummy_data);
                        }
                        50
                    }
                    AccessPattern::Hotspot(hotspots) => {
                        for &hotspot in hotspots {
                            let dummy_data = vec![0u8; 4096]; // 4KB for hotspots
                            self.target_cache.put(hotspot, dummy_data);
                        }
                        hotspots.len()
                    }
                    AccessPattern::Temporal(_duration) => {
                        // 基于时间的预取策略（简化实现）
                        for i in 0..20 {
                            let dummy_data = vec![0u8; 1024];
                            self.target_cache.put(i, dummy_data);
                        }
                        20
                    }
                }
            }
            WarmupStrategy::HeuristicPrefetch { heuristic: _ } => {
                // 启发式预取（简化实现）
                for i in 0..30 {
                    let key = i * 3; // 每隔3个预取一个
                    let dummy_data = vec![0u8; 1024];
                    self.target_cache.put(key, dummy_data);
                }
                30
            }
        }
    }

    /// 获取预热历史
    pub fn get_warmup_history(&self) -> &[WarmupEvent] {
        &self.warmup_history
    }

    /// 清理旧的预热历史
    pub fn cleanup_warmup_history(&mut self, retention_duration: Duration) {
        let cutoff_time = Instant::now() - retention_duration;
        self.warmup_history
            .retain(|event| event.timestamp > cutoff_time);
    }
}

#[derive(Debug, Clone)]
pub struct WarmupResult {
    pub success: bool,
    pub items_prefetched: usize,
    pub time_taken: Duration,
    pub strategies_executed: usize,
}
