//! 索引系统模块
//!
//! 提供高性能的数组索引操作，包括多种索引算法、访问策略选择和优化技术

pub mod algorithms;
pub mod boolean_index;
pub mod fancy_index;
pub mod optimizations;
pub mod smart_router;
pub mod strategy_selector;
pub mod types;

// 重新导出核心类型（保持API稳定，未被内部使用时忽略未用警告）
#[allow(unused_imports)]
pub use types::{
    AccessInfo, AccessPattern, AccessPatternAnalyzer, AccessStatistics, AccessStrategy,
    IndexOptimizationConfig, IndexPerformanceMonitor, IndexPerformanceReport, IndexResult,
    IndexType, SliceInfo,
};

#[allow(unused_imports)]
pub use algorithms::{IndexAlgorithmExecutor, IndexError};

#[allow(unused_imports)]
pub use strategy_selector::{
    AccessStrategySelector, AdaptationStats, EstimatedPerformance, StrategySelection,
    StrategySelectionReport,
};

#[allow(unused_imports)]
pub use optimizations::{IndexOptimizationManager, OptimizationReport};

/// 索引系统主控制器
pub struct IndexingSystem {
    algorithm_executor: IndexAlgorithmExecutor,
    strategy_selector: AccessStrategySelector,
    optimization_manager: IndexOptimizationManager,
    config: IndexOptimizationConfig,
}

impl IndexingSystem {
    /// 创建新的索引系统
    pub fn new() -> Self {
        let config = IndexOptimizationConfig::default();
        Self::with_config(config)
    }

    /// 使用指定配置创建索引系统
    pub fn with_config(config: IndexOptimizationConfig) -> Self {
        Self {
            algorithm_executor: IndexAlgorithmExecutor::new(config.clone()),
            strategy_selector: AccessStrategySelector::new(),
            optimization_manager: IndexOptimizationManager::new(config.clone()),
            config,
        }
    }

    /// 执行布尔索引操作
    pub fn execute_boolean_index(
        &mut self,
        mask: &[bool],
        data: &[u8],
        shape: &[usize],
        itemsize: usize,
        cache_pressure: f64,
        memory_pressure: f64,
    ) -> Result<(Vec<Vec<u8>>, IndexingPerformanceReport), IndexError> {
        // 1. 构建索引向量
        let indices = vec![mask
            .iter()
            .enumerate()
            .filter_map(|(i, &b)| if b { Some(i) } else { None })
            .collect()];

        // 2. 选择最优策略和算法
        let selection = self.strategy_selector.select_strategy(
            &indices,
            data.len(),
            cache_pressure,
            memory_pressure,
        );

        // 3. 执行索引操作
        let result = if self.config.enable_parallel
            || selection.strategy == AccessStrategy::ParallelPointAccess
        {
            // 使用优化管理器执行
            self.optimization_manager.execute_optimized_index(
                selection.strategy,
                &indices,
                data,
                shape,
                itemsize,
            )
        } else {
            // 使用算法执行器执行
            let (data_result, _stats) = self.algorithm_executor.execute_boolean_index(
                selection.algorithm,
                mask,
                data,
                shape,
                itemsize,
            )?;
            Ok(data_result)
        };

        match result {
            Ok(data_result) => {
                // 记录性能
                let performance_report = IndexingPerformanceReport {
                    strategy_used: selection.strategy,
                    algorithm_used: selection.algorithm,
                    confidence: selection.confidence,
                    expected_performance: selection.expected_performance,
                    actual_elements_processed: data_result.len(),
                };

                Ok((data_result, performance_report))
            }
            Err(e) => Err(e),
        }
    }

    /// 执行花式索引操作
    pub fn execute_fancy_index(
        &mut self,
        indices: &[i64],
        data: &[u8],
        shape: &[usize],
        itemsize: usize,
        cache_pressure: f64,
        memory_pressure: f64,
    ) -> Result<(Vec<Vec<u8>>, IndexingPerformanceReport), IndexError> {
        // 标准化索引
        let normalized_indices: Result<Vec<_>, _> = indices
            .iter()
            .map(|&idx| {
                let normalized = if idx < 0 { shape[0] as i64 + idx } else { idx };

                if normalized >= 0 && (normalized as usize) < shape[0] {
                    Ok(normalized as usize)
                } else {
                    Err(IndexError::IndexOutOfBounds)
                }
            })
            .collect();

        let normalized_indices = normalized_indices?;
        let index_vectors = vec![normalized_indices];

        // 选择策略
        let selection = self.strategy_selector.select_strategy(
            &index_vectors,
            data.len(),
            cache_pressure,
            memory_pressure,
        );

        // 执行索引操作
        let result = if self.config.enable_parallel
            || selection.strategy == AccessStrategy::ParallelPointAccess
        {
            self.optimization_manager.execute_optimized_index(
                selection.strategy,
                &index_vectors,
                data,
                shape,
                itemsize,
            )
        } else {
            self.algorithm_executor
                .execute_fancy_index(indices, data, shape, itemsize)
        };

        match result {
            Ok(data_result) => {
                let performance_report = IndexingPerformanceReport {
                    strategy_used: selection.strategy,
                    algorithm_used: selection.algorithm,
                    confidence: selection.confidence,
                    expected_performance: selection.expected_performance,
                    actual_elements_processed: data_result.len(),
                };

                Ok((data_result, performance_report))
            }
            Err(e) => Err(e),
        }
    }

    /// 执行切片索引操作
    pub fn execute_slice_index(
        &mut self,
        start: Option<i64>,
        stop: Option<i64>,
        step: Option<i64>,
        data: &[u8],
        shape: &[usize],
        itemsize: usize,
    ) -> Result<(Vec<Vec<u8>>, IndexingPerformanceReport), IndexError> {
        // 解析切片参数
        let step = step.unwrap_or(1);
        if step == 0 {
            return Err(IndexError::InvalidSlice);
        }

        let start = start.unwrap_or(if step > 0 { 0 } else { shape[0] as i64 - 1 });
        let stop = stop.unwrap_or(if step > 0 { shape[0] as i64 } else { -1 });

        // 生成索引序列
        let mut indices = Vec::new();
        let mut i = start;

        if step > 0 {
            while i < stop && i < shape[0] as i64 {
                if i >= 0 {
                    indices.push(i as usize);
                }
                i += step;
            }
        } else {
            while i > stop && i >= 0 {
                if i < shape[0] as i64 {
                    indices.push(i as usize);
                }
                i += step;
            }
        }

        let index_vectors = vec![indices];

        // 选择策略 - 切片通常是顺序访问
        let selection = self.strategy_selector.select_strategy(
            &index_vectors,
            data.len(),
            0.0, // 低缓存压力
            0.0, // 低内存压力
        );

        // 执行索引操作
        let result = self.optimization_manager.execute_optimized_index(
            selection.strategy,
            &index_vectors,
            data,
            shape,
            itemsize,
        );

        match result {
            Ok(data_result) => {
                let performance_report = IndexingPerformanceReport {
                    strategy_used: selection.strategy,
                    algorithm_used: selection.algorithm,
                    confidence: selection.confidence,
                    expected_performance: selection.expected_performance,
                    actual_elements_processed: data_result.len(),
                };

                Ok((data_result, performance_report))
            }
            Err(e) => Err(e),
        }
    }

    /// 获取综合性能报告
    pub fn get_comprehensive_performance_report(&self) -> ComprehensivePerformanceReport {
        ComprehensivePerformanceReport {
            algorithm_performance: self.algorithm_executor.get_performance_report(),
            strategy_selection: self.strategy_selector.get_performance_report(),
            optimization_report: self.optimization_manager.get_optimization_report(),
            system_config: self.config.clone(),
        }
    }

    /// 更新配置
    pub fn update_config(&mut self, new_config: IndexOptimizationConfig) {
        self.config = new_config;
        // 注意：这里可能需要重新创建子组件以应用新配置
    }

    /// 记录执行结果以改进策略选择
    pub fn record_execution_feedback(
        &mut self,
        strategy: AccessStrategy,
        latency_ns: u64,
        throughput_mbps: f64,
        success: bool,
    ) {
        self.strategy_selector.record_execution_result(
            strategy,
            latency_ns,
            throughput_mbps,
            success,
        );
    }
}

impl Default for IndexingSystem {
    fn default() -> Self {
        Self::new()
    }
}

/// 索引性能报告
#[derive(Debug, Clone)]
pub struct IndexingPerformanceReport {
    pub strategy_used: AccessStrategy,
    pub algorithm_used: crate::performance::metrics::IndexAlgorithm,
    pub confidence: f64,
    pub expected_performance: EstimatedPerformance,
    pub actual_elements_processed: usize,
}

/// 综合性能报告
#[derive(Debug, Clone)]
pub struct ComprehensivePerformanceReport {
    pub algorithm_performance: IndexPerformanceReport,
    pub strategy_selection: StrategySelectionReport,
    pub optimization_report: OptimizationReport,
    pub system_config: IndexOptimizationConfig,
}
