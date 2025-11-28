//! 算法选择器
//!
//! 从lazy_array_original.rs中提取的智能算法选择功能

use super::metrics::{IndexAlgorithm, PerformanceMetrics};
use super::profiler::PerformanceProfiler;
use crate::access_pattern::{
    AccessFrequency, AccessPatternAnalysis, AccessPatternType, SizeCategory,
};
use std::sync::Arc;
use std::time::Duration;

/// 决策条件
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

/// 决策节点
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

/// 决策树
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
                    .with_false_branch(DecisionNode::new_leaf(IndexAlgorithm::FancySIMD)),
            )
            .with_false_branch(
                DecisionNode::new_branch(Condition::DensityGreaterThan(0.5))
                    .with_true_branch(DecisionNode::new_leaf(IndexAlgorithm::BooleanDense))
                    .with_false_branch(DecisionNode::new_leaf(IndexAlgorithm::BooleanSparse)),
            );

        Self { root }
    }

    /// 基于决策树选择算法
    pub fn select_algorithm(
        &self,
        pattern: &AccessPatternAnalysis,
        metrics: &PerformanceMetrics,
    ) -> IndexAlgorithm {
        self.evaluate_node(&self.root, pattern, metrics)
    }

    fn evaluate_node(
        &self,
        node: &DecisionNode,
        pattern: &AccessPatternAnalysis,
        metrics: &PerformanceMetrics,
    ) -> IndexAlgorithm {
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

    fn evaluate_condition(
        &self,
        condition: &Condition,
        pattern: &AccessPatternAnalysis,
        metrics: &PerformanceMetrics,
    ) -> bool {
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
                matches!(
                    (&pattern.frequency, threshold),
                    (AccessFrequency::High, AccessFrequency::Medium)
                        | (AccessFrequency::Extreme, AccessFrequency::Medium)
                        | (AccessFrequency::Extreme, AccessFrequency::High)
                )
            }
        }
    }
}

/// 智能算法选择器
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

    /// 启用或禁用自适应算法选择
    pub fn set_adaptation_enabled(&mut self, enabled: bool) {
        self.adaptation_enabled = enabled;
    }

    /// 主要的算法选择接口
    pub fn select_algorithm(
        &self,
        pattern: &AccessPatternAnalysis,
        operation_type: &str,
    ) -> IndexAlgorithm {
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

    /// 选择花式索引算法
    fn select_fancy_index_algorithm(
        &self,
        pattern: &AccessPatternAnalysis,
        metrics: &PerformanceMetrics,
    ) -> IndexAlgorithm {
        match (&pattern.size_category, pattern.locality_score) {
            (SizeCategory::Micro | SizeCategory::Small, _) => IndexAlgorithm::FancyDirect,
            (_, score) if score > 0.8 => IndexAlgorithm::FancyPrefetch,
            _ if metrics.cpu_utilization < 0.5 => IndexAlgorithm::FancySIMD,
            _ => IndexAlgorithm::FancyZeroCopy,
        }
    }

    /// 选择布尔索引算法
    fn select_boolean_index_algorithm(
        &self,
        pattern: &AccessPatternAnalysis,
        _metrics: &PerformanceMetrics,
    ) -> IndexAlgorithm {
        match (pattern.density, &pattern.size_category) {
            (density, _) if density > 0.8 => IndexAlgorithm::BooleanDense,
            (density, SizeCategory::Large | SizeCategory::Huge) if density < 0.2 => {
                IndexAlgorithm::BooleanSparse
            }
            (_, SizeCategory::Medium | SizeCategory::Large) => IndexAlgorithm::BooleanExtreme,
            _ => IndexAlgorithm::BooleanBitmap,
        }
    }

    /// 选择批量访问算法
    fn select_batch_access_algorithm(
        &self,
        pattern: &AccessPatternAnalysis,
        metrics: &PerformanceMetrics,
    ) -> IndexAlgorithm {
        match (&pattern.pattern_type, &pattern.size_category) {
            (AccessPatternType::Sequential, _) => IndexAlgorithm::BatchChunked,
            (_, SizeCategory::Huge) => IndexAlgorithm::BatchStreaming,
            _ if metrics.cpu_utilization < 0.7 => IndexAlgorithm::BatchParallel,
            _ => IndexAlgorithm::BatchChunked,
        }
    }

    /// 自适应算法选择（基于历史性能）
    fn adapt_algorithm_selection(
        &self,
        initial: IndexAlgorithm,
        _pattern: &AccessPatternAnalysis,
    ) -> IndexAlgorithm {
        // 检查历史性能，如果当前算法表现不佳，尝试其他算法
        if let Some((avg_duration, avg_throughput)) =
            self.profiler.get_algorithm_performance(initial)
        {
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

    /// 获取算法选择的详细分析
    pub fn get_algorithm_analysis(
        &self,
        pattern: &AccessPatternAnalysis,
        operation_type: &str,
    ) -> AlgorithmAnalysis {
        let metrics = self.profiler.get_current_metrics();
        let recommended = self.select_algorithm(pattern, operation_type);
        let alternatives = self.get_alternative_algorithms(pattern, operation_type);

        AlgorithmAnalysis {
            recommended,
            alternatives,
            decision_factors: self.get_decision_factors(pattern, &metrics),
            confidence_score: self.calculate_confidence_score(pattern, &metrics),
        }
    }

    /// 获取备选算法
    fn get_alternative_algorithms(
        &self,
        pattern: &AccessPatternAnalysis,
        operation_type: &str,
    ) -> Vec<IndexAlgorithm> {
        match operation_type {
            "fancy_index" => vec![
                IndexAlgorithm::FancyDirect,
                IndexAlgorithm::FancySIMD,
                IndexAlgorithm::FancyPrefetch,
                IndexAlgorithm::FancyZeroCopy,
            ],
            "boolean_index" => vec![
                IndexAlgorithm::BooleanBitmap,
                IndexAlgorithm::BooleanDense,
                IndexAlgorithm::BooleanSparse,
                IndexAlgorithm::BooleanExtreme,
            ],
            "batch_access" => vec![
                IndexAlgorithm::BatchParallel,
                IndexAlgorithm::BatchChunked,
                IndexAlgorithm::BatchStreaming,
            ],
            _ => vec![IndexAlgorithm::FancyDirect],
        }
    }

    /// 获取决策因子
    fn get_decision_factors(
        &self,
        pattern: &AccessPatternAnalysis,
        metrics: &PerformanceMetrics,
    ) -> Vec<String> {
        let mut factors = Vec::new();

        factors.push(format!("Size category: {:?}", pattern.size_category));
        factors.push(format!("Access pattern: {:?}", pattern.pattern_type));
        factors.push(format!("Density: {:.2}", pattern.density));
        factors.push(format!("Locality score: {:.2}", pattern.locality_score));
        factors.push(format!("CPU utilization: {:.2}", metrics.cpu_utilization));
        factors.push(format!("Cache hit rate: {:.2}", metrics.cache_hit_rate));

        factors
    }

    /// 计算置信度分数
    fn calculate_confidence_score(
        &self,
        pattern: &AccessPatternAnalysis,
        metrics: &PerformanceMetrics,
    ) -> f64 {
        let mut score = 0.0;
        let mut weight_sum = 0.0;

        // 基于数据质量的置信度
        if pattern.locality_score > 0.5 {
            score += 0.8;
            weight_sum += 1.0;
        }

        // 基于访问模式明确性的置信度
        match pattern.pattern_type {
            AccessPatternType::Sequential => {
                score += 0.9;
                weight_sum += 1.0;
            }
            AccessPatternType::Random => {
                score += 0.6;
                weight_sum += 1.0;
            }
            _ => {
                score += 0.7;
                weight_sum += 1.0;
            }
        }

        // 基于系统状态稳定性的置信度
        if metrics.cpu_utilization < 0.8 {
            score += 0.8;
            weight_sum += 1.0;
        }

        if weight_sum > 0.0 {
            score / weight_sum
        } else {
            0.5
        }
    }
}

/// 算法分析结果
#[derive(Debug)]
pub struct AlgorithmAnalysis {
    pub recommended: IndexAlgorithm,
    pub alternatives: Vec<IndexAlgorithm>,
    pub decision_factors: Vec<String>,
    pub confidence_score: f64,
}
