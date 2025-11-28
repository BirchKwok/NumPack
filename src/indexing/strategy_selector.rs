//! 访问策略选择器
//!
//! 智能选择最优的索引访问策略，基于访问模式分析和性能历史

use crate::indexing::types::*;
use crate::performance::metrics::IndexAlgorithm;
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// 策略选择器 - 智能选择最优访问策略
pub struct AccessStrategySelector {
    pattern_analyzer: AccessPatternAnalyzer,
    performance_history: HashMap<AccessStrategy, StrategyPerformanceHistory>,
    algorithm_selector: AlgorithmSelector,
    adaptation_config: AdaptationConfig,
    last_adaptation: Instant,
}

/// 策略性能历史记录
#[derive(Debug, Clone)]
struct StrategyPerformanceHistory {
    total_operations: u64,
    success_count: u64,
    total_latency_ns: u64,
    total_throughput_mbps: f64,
    last_used: Instant,
    confidence_score: f64,
}

impl StrategyPerformanceHistory {
    fn new() -> Self {
        Self {
            total_operations: 0,
            success_count: 0,
            total_latency_ns: 0,
            total_throughput_mbps: 0.0,
            last_used: Instant::now(),
            confidence_score: 0.5, // 初始中等信心
        }
    }

    fn average_latency_ns(&self) -> u64 {
        if self.total_operations > 0 {
            self.total_latency_ns / self.total_operations
        } else {
            u64::MAX
        }
    }

    fn success_rate(&self) -> f64 {
        if self.total_operations > 0 {
            self.success_count as f64 / self.total_operations as f64
        } else {
            0.0
        }
    }

    fn average_throughput(&self) -> f64 {
        if self.total_operations > 0 {
            self.total_throughput_mbps / self.total_operations as f64
        } else {
            0.0
        }
    }
}

/// 自适应配置
#[derive(Debug, Clone)]
struct AdaptationConfig {
    adaptation_threshold: f64,         // 适应阈值
    confidence_decay_rate: f64,        // 信心衰减率
    learning_rate: f64,                // 学习率
    performance_window: usize,         // 性能窗口大小
    min_samples_for_adaptation: usize, // 适应所需的最小样本数
}

impl Default for AdaptationConfig {
    fn default() -> Self {
        Self {
            adaptation_threshold: 0.1,
            confidence_decay_rate: 0.95,
            learning_rate: 0.1,
            performance_window: 100,
            min_samples_for_adaptation: 10,
        }
    }
}

impl AccessStrategySelector {
    pub fn new() -> Self {
        Self {
            pattern_analyzer: AccessPatternAnalyzer::new(),
            performance_history: HashMap::new(),
            algorithm_selector: AlgorithmSelector::new(),
            adaptation_config: AdaptationConfig::default(),
            last_adaptation: Instant::now(),
        }
    }

    /// 选择最优访问策略
    pub fn select_strategy(
        &mut self,
        indices: &[Vec<usize>],
        data_size: usize,
        cache_pressure: f64,
        memory_pressure: f64,
    ) -> StrategySelection {
        // 1. 分析访问模式
        let access_pattern = self.pattern_analyzer.analyze_indices(indices);

        // 2. 评估数据特征
        let data_characteristics = self.evaluate_data_characteristics(indices, data_size);

        // 3. 考虑系统压力
        let system_context = SystemContext {
            cache_pressure,
            memory_pressure,
            available_threads: rayon::current_num_threads(),
        };

        // 4. 基于历史性能选择策略
        let candidate_strategies =
            self.get_candidate_strategies(&access_pattern, &data_characteristics);
        let selected_strategy =
            self.rank_and_select_strategy(candidate_strategies, &system_context);

        // 5. 选择具体算法
        let algorithm = self
            .algorithm_selector
            .select_algorithm(&access_pattern, &data_characteristics);

        StrategySelection {
            strategy: selected_strategy,
            algorithm,
            confidence: self.calculate_selection_confidence(&selected_strategy),
            expected_performance: self
                .estimate_performance(&selected_strategy, &data_characteristics),
        }
    }

    /// 记录策略执行结果
    pub fn record_execution_result(
        &mut self,
        strategy: AccessStrategy,
        latency_ns: u64,
        throughput_mbps: f64,
        success: bool,
    ) {
        let history = self
            .performance_history
            .entry(strategy)
            .or_insert_with(StrategyPerformanceHistory::new);

        history.total_operations += 1;
        history.total_latency_ns += latency_ns;
        history.total_throughput_mbps += throughput_mbps;
        history.last_used = Instant::now();

        if success {
            history.success_count += 1;
            // 成功时提升信心
            history.confidence_score = (history.confidence_score + 0.1).min(1.0);
        } else {
            // 失败时降低信心
            history.confidence_score = (history.confidence_score - 0.2).max(0.0);
        }

        // 检查是否需要自适应调整
        if self.should_adapt() {
            self.perform_adaptation();
        }
    }

    /// 评估数据特征
    fn evaluate_data_characteristics(
        &self,
        indices: &[Vec<usize>],
        data_size: usize,
    ) -> DataCharacteristics {
        let total_elements = indices.iter().map(|dim| dim.len()).product::<usize>();
        let density = if indices.is_empty() {
            0.0
        } else {
            total_elements as f64 / indices[0].len() as f64
        };

        let locality_score = self.calculate_locality_score(indices);
        let size_category = self.categorize_data_size(data_size);

        DataCharacteristics {
            total_elements,
            density,
            locality_score,
            size_category,
            estimated_memory_footprint: total_elements * 8, // 假设8字节每元素
        }
    }

    /// 计算局部性评分
    fn calculate_locality_score(&self, indices: &[Vec<usize>]) -> f64 {
        if indices.is_empty() || indices[0].is_empty() {
            return 0.0;
        }

        let first_dim = &indices[0];
        if first_dim.len() < 2 {
            return 1.0;
        }

        // 计算相邻元素的平均间距
        let mut total_distance = 0;
        for i in 1..first_dim.len() {
            total_distance += (first_dim[i] as i64 - first_dim[i - 1] as i64).abs();
        }

        let average_distance = total_distance as f64 / (first_dim.len() - 1) as f64;

        // 局部性评分：间距越小，局部性越好
        (10.0 / (1.0 + average_distance)).min(1.0)
    }

    /// 数据大小分类
    fn categorize_data_size(&self, data_size: usize) -> DataSizeCategory {
        if data_size < 64 * 1024 {
            DataSizeCategory::Small
        } else if data_size < 16 * 1024 * 1024 {
            DataSizeCategory::Medium
        } else if data_size < 1024 * 1024 * 1024 {
            DataSizeCategory::Large
        } else {
            DataSizeCategory::ExtraLarge
        }
    }

    /// 获取候选策略
    fn get_candidate_strategies(
        &self,
        pattern: &AccessPattern,
        characteristics: &DataCharacteristics,
    ) -> Vec<AccessStrategy> {
        let mut candidates = Vec::new();

        match pattern {
            AccessPattern::Sequential => {
                candidates.push(AccessStrategy::BlockCopy);
                candidates.push(AccessStrategy::DirectMemory);
                if characteristics.size_category >= DataSizeCategory::Large {
                    candidates.push(AccessStrategy::PrefetchOptimized);
                }
            }
            AccessPattern::Random => {
                candidates.push(AccessStrategy::VectorizedGather);
                candidates.push(AccessStrategy::ParallelPointAccess);
                if characteristics.locality_score > 0.5 {
                    candidates.push(AccessStrategy::PrefetchOptimized);
                }
            }
            AccessPattern::Clustered => {
                candidates.push(AccessStrategy::PrefetchOptimized);
                candidates.push(AccessStrategy::BlockCopy);
                candidates.push(AccessStrategy::VectorizedGather);
            }
            AccessPattern::Strided => {
                candidates.push(AccessStrategy::VectorizedGather);
                candidates.push(AccessStrategy::PrefetchOptimized);
            }
            AccessPattern::Sparse => {
                candidates.push(AccessStrategy::ParallelPointAccess);
                candidates.push(AccessStrategy::VectorizedGather);
                if characteristics.density < 0.1 {
                    candidates.push(AccessStrategy::ZeroCopy);
                }
            }
            AccessPattern::Mixed => {
                candidates.push(AccessStrategy::Adaptive);
                candidates.push(AccessStrategy::VectorizedGather);
            }
        }

        // 根据数据大小添加额外候选策略
        match characteristics.size_category {
            DataSizeCategory::Small => {
                candidates.push(AccessStrategy::DirectMemory);
            }
            DataSizeCategory::Large | DataSizeCategory::ExtraLarge => {
                candidates.push(AccessStrategy::ParallelPointAccess);
                candidates.push(AccessStrategy::PrefetchOptimized);
            }
            _ => {}
        }

        candidates.sort();
        candidates.dedup();
        candidates
    }

    /// 排序和选择策略
    fn rank_and_select_strategy(
        &self,
        candidates: Vec<AccessStrategy>,
        context: &SystemContext,
    ) -> AccessStrategy {
        if candidates.is_empty() {
            return AccessStrategy::DirectMemory;
        }

        let mut scored_candidates: Vec<(AccessStrategy, f64)> = candidates
            .into_iter()
            .map(|strategy| {
                let score = self.calculate_strategy_score(&strategy, context);
                (strategy, score)
            })
            .collect();

        // 按评分排序
        scored_candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        scored_candidates[0].0
    }

    /// 计算策略评分
    fn calculate_strategy_score(&self, strategy: &AccessStrategy, context: &SystemContext) -> f64 {
        let mut score = 0.5; // 基础分数

        // 历史性能评分
        if let Some(history) = self.performance_history.get(strategy) {
            let performance_score = self.calculate_performance_score(history);
            score += performance_score * 0.4; // 性能权重40%

            let confidence_score = history.confidence_score;
            score += confidence_score * 0.2; // 信心权重20%
        }

        // 系统上下文适应性评分
        let context_score = self.calculate_context_score(strategy, context);
        score += context_score * 0.3; // 上下文权重30%

        // 策略特性评分
        let strategy_score = self.calculate_strategy_characteristics_score(strategy);
        score += strategy_score * 0.1; // 特性权重10%

        score.clamp(0.0, 1.0)
    }

    /// 计算性能评分
    fn calculate_performance_score(&self, history: &StrategyPerformanceHistory) -> f64 {
        if history.total_operations < 3 {
            return 0.5; // 数据不足，给予中等评分
        }

        let latency_score = 1.0 / (1.0 + history.average_latency_ns() as f64 / 1_000_000.0); // 延迟越低越好
        let success_score = history.success_rate();
        let throughput_score = (history.average_throughput() / 1000.0).min(1.0); // 归一化吞吐量

        (latency_score * 0.4 + success_score * 0.4 + throughput_score * 0.2).clamp(0.0, 1.0)
    }

    /// 计算上下文评分
    fn calculate_context_score(&self, strategy: &AccessStrategy, context: &SystemContext) -> f64 {
        let mut score = 0.5;

        match strategy {
            AccessStrategy::ParallelPointAccess => {
                // 并行策略在多线程环境下表现更好
                if context.available_threads > 4 {
                    score += 0.3;
                }
                // 内存压力下表现较差
                score -= context.memory_pressure * 0.2;
            }
            AccessStrategy::PrefetchOptimized => {
                // 缓存压力下表现较差
                score -= context.cache_pressure * 0.3;
                // 低内存压力下表现更好
                score += (1.0 - context.memory_pressure) * 0.2;
            }
            AccessStrategy::ZeroCopy => {
                // 内存压力下表现更好
                score += (1.0 - context.memory_pressure) * 0.4;
            }
            AccessStrategy::BlockCopy => {
                // 缓存友好策略
                score += (1.0 - context.cache_pressure) * 0.3;
            }
            _ => {}
        }

        score.clamp(0.0, 1.0)
    }

    /// 计算策略特性评分
    fn calculate_strategy_characteristics_score(&self, strategy: &AccessStrategy) -> f64 {
        match strategy {
            AccessStrategy::DirectMemory => 0.7,        // 简单可靠
            AccessStrategy::BlockCopy => 0.8,           // 高效的顺序访问
            AccessStrategy::VectorizedGather => 0.9,    // 高性能随机访问
            AccessStrategy::ParallelPointAccess => 0.8, // 高并发性能
            AccessStrategy::PrefetchOptimized => 0.85,  // 智能预取
            AccessStrategy::ZeroCopy => 0.95,           // 极致内存效率
            AccessStrategy::Adaptive => 0.75,           // 灵活适应
        }
    }

    /// 计算选择信心度
    fn calculate_selection_confidence(&self, strategy: &AccessStrategy) -> f64 {
        if let Some(history) = self.performance_history.get(strategy) {
            let data_confidence = if history.total_operations >= 10 {
                0.9
            } else if history.total_operations >= 5 {
                0.7
            } else {
                0.5
            };

            (history.confidence_score + data_confidence) / 2.0
        } else {
            0.5 // 新策略，中等信心
        }
    }

    /// 估算性能
    fn estimate_performance(
        &self,
        strategy: &AccessStrategy,
        characteristics: &DataCharacteristics,
    ) -> EstimatedPerformance {
        let base_latency_ns = match strategy {
            AccessStrategy::DirectMemory => 1000,
            AccessStrategy::BlockCopy => 500,
            AccessStrategy::VectorizedGather => 800,
            AccessStrategy::ParallelPointAccess => 1200,
            AccessStrategy::PrefetchOptimized => 600,
            AccessStrategy::ZeroCopy => 200,
            AccessStrategy::Adaptive => 900,
        };

        // 基于数据特征调整估算
        let size_factor = match characteristics.size_category {
            DataSizeCategory::Small => 0.8,
            DataSizeCategory::Medium => 1.0,
            DataSizeCategory::Large => 1.5,
            DataSizeCategory::ExtraLarge => 2.0,
        };

        let locality_factor = 0.5 + characteristics.locality_score * 0.5;

        let estimated_latency_ns = (base_latency_ns as f64 * size_factor / locality_factor) as u64;
        let estimated_throughput_mbps = characteristics.estimated_memory_footprint as f64
            / (estimated_latency_ns as f64 / 1_000_000_000.0)
            / (1024.0 * 1024.0);

        EstimatedPerformance {
            latency_ns: estimated_latency_ns,
            throughput_mbps: estimated_throughput_mbps,
            memory_usage_mb: characteristics.estimated_memory_footprint / (1024 * 1024),
        }
    }

    /// 检查是否应该进行自适应调整
    fn should_adapt(&self) -> bool {
        self.last_adaptation.elapsed() > Duration::from_secs(60) && // 至少1分钟间隔
        self.performance_history.values()
            .any(|h| h.total_operations >= self.adaptation_config.min_samples_for_adaptation as u64)
    }

    /// 执行自适应调整
    fn perform_adaptation(&mut self) {
        // 衰减所有策略的信心度
        for history in self.performance_history.values_mut() {
            history.confidence_score *= self.adaptation_config.confidence_decay_rate;
        }

        self.last_adaptation = Instant::now();
    }

    /// 获取性能统计报告
    pub fn get_performance_report(&self) -> StrategySelectionReport {
        StrategySelectionReport {
            strategy_performance: self.performance_history.clone(),
            total_selections: self
                .performance_history
                .values()
                .map(|h| h.total_operations)
                .sum(),
            best_performing_strategy: self.get_best_performing_strategy(),
            adaptation_stats: self.get_adaptation_stats(),
        }
    }

    fn get_best_performing_strategy(&self) -> Option<AccessStrategy> {
        self.performance_history
            .iter()
            .filter(|(_, h)| h.total_operations >= 5) // 至少5次样本
            .max_by(|(_, a), (_, b)| {
                let score_a = self.calculate_performance_score(a);
                let score_b = self.calculate_performance_score(b);
                score_a.partial_cmp(&score_b).unwrap()
            })
            .map(|(strategy, _)| *strategy)
    }

    fn get_adaptation_stats(&self) -> AdaptationStats {
        AdaptationStats {
            last_adaptation: self.last_adaptation,
            adaptation_count: 0, // 简化实现
            average_confidence: self
                .performance_history
                .values()
                .map(|h| h.confidence_score)
                .sum::<f64>()
                / self.performance_history.len().max(1) as f64,
        }
    }
}

/// 算法选择器
struct AlgorithmSelector {
    algorithm_performance: HashMap<IndexAlgorithm, f64>,
}

impl AlgorithmSelector {
    fn new() -> Self {
        Self {
            algorithm_performance: HashMap::new(),
        }
    }

    fn select_algorithm(
        &self,
        pattern: &AccessPattern,
        characteristics: &DataCharacteristics,
    ) -> IndexAlgorithm {
        match pattern {
            AccessPattern::Sequential => {
                if characteristics.density > 0.8 {
                    IndexAlgorithm::BooleanDense
                } else {
                    IndexAlgorithm::BooleanBitmap
                }
            }
            AccessPattern::Random => {
                if characteristics.size_category >= DataSizeCategory::Large {
                    IndexAlgorithm::BooleanExtreme
                } else {
                    IndexAlgorithm::BooleanHierarchical
                }
            }
            AccessPattern::Sparse => IndexAlgorithm::BooleanSparse,
            AccessPattern::Clustered => IndexAlgorithm::BooleanHierarchical,
            _ => IndexAlgorithm::BooleanBitmap,
        }
    }
}

// 支持类型定义

#[derive(Debug, Clone)]
pub struct StrategySelection {
    pub strategy: AccessStrategy,
    pub algorithm: IndexAlgorithm,
    pub confidence: f64,
    pub expected_performance: EstimatedPerformance,
}

#[derive(Debug, Clone)]
struct DataCharacteristics {
    total_elements: usize,
    density: f64,
    locality_score: f64,
    size_category: DataSizeCategory,
    estimated_memory_footprint: usize,
}

#[derive(Debug, Clone, PartialEq, PartialOrd)]
enum DataSizeCategory {
    Small,
    Medium,
    Large,
    ExtraLarge,
}

#[derive(Debug, Clone)]
struct SystemContext {
    cache_pressure: f64,
    memory_pressure: f64,
    available_threads: usize,
}

#[derive(Debug, Clone)]
pub struct EstimatedPerformance {
    pub latency_ns: u64,
    pub throughput_mbps: f64,
    pub memory_usage_mb: usize,
}

#[derive(Debug, Clone)]
pub struct StrategySelectionReport {
    pub strategy_performance: HashMap<AccessStrategy, StrategyPerformanceHistory>,
    pub total_selections: u64,
    pub best_performing_strategy: Option<AccessStrategy>,
    pub adaptation_stats: AdaptationStats,
}

#[derive(Debug, Clone)]
pub struct AdaptationStats {
    pub last_adaptation: Instant,
    pub adaptation_count: u64,
    pub average_confidence: f64,
}
