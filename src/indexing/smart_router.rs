//! 智能索引路由器实现
//!
//! 提供智能的索引路由和算法选择功能

use std::sync::{Arc, Mutex};
use std::time::Instant;

use crate::access_pattern::types::AccessPattern;
use crate::lazy_array::core::OptimizedLazyArray;
// use crate::cache::prefetch::PrefetchManager; // 暂时未使用
use crate::cache::smart_cache::SmartCache;
use crate::indexing::fancy_index::FancyIndexEngine;
// 暂时注释掉未使用的导入，避免警告
// use crate::indexing::boolean_index::BooleanIndexEngine;
use crate::performance::algorithm_selector::AlgorithmSelector;
use crate::performance::metrics::IndexAlgorithm;
use crate::performance::profiler::PerformanceProfiler;

/// 智能索引路由器
pub struct SmartIndexRouter {
    profiler: Arc<PerformanceProfiler>,
    selector: Arc<AlgorithmSelector>,
    cache_manager: Arc<SmartCache>,
    fancy_index_engine: Arc<Mutex<FancyIndexEngine>>,
}

impl SmartIndexRouter {
    pub fn new() -> Self {
        let profiler = Arc::new(PerformanceProfiler::new());
        let selector = Arc::new(AlgorithmSelector::new(Arc::clone(&profiler)));
        let cache_manager = Arc::new(SmartCache::new());
        let fancy_index_engine = Arc::new(Mutex::new(FancyIndexEngine::new()));

        Self {
            profiler,
            selector,
            cache_manager,
            fancy_index_engine,
        }
    }

    /// 路由花式索引操作
    pub fn route_fancy_index(&self, indices: &[usize], array: &OptimizedLazyArray) -> Vec<Vec<u8>> {
        let start_time = Instant::now();

        // 分析访问模式
        let total_size = indices.len() * array.itemsize;
        let pattern = self
            .profiler
            .analyze_access_pattern(0, indices.len() * array.itemsize);

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
        self.profiler
            .record_operation(algorithm, duration, total_size);

        result
    }

    /// 路由布尔索引操作
    pub fn route_boolean_index(&self, mask: &[bool], array: &OptimizedLazyArray) -> Vec<Vec<u8>> {
        let start_time = Instant::now();

        // 分析访问模式
        let selected_count = mask.iter().filter(|&&x| x).count();
        let total_size = selected_count * array.itemsize;
        let selected_count = mask.iter().filter(|&&x| x).count();
        let pattern = self
            .profiler
            .analyze_access_pattern(0, selected_count * array.itemsize);

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
        self.profiler
            .record_operation(algorithm, duration, total_size);

        result
    }

    /// 路由批量访问操作
    pub fn route_batch_access(
        &self,
        pattern: &AccessPattern,
        array: &OptimizedLazyArray,
    ) -> Vec<Vec<u8>> {
        let start_time = Instant::now();

        // 分析访问模式
        let (total_size, access_pattern) = match pattern {
            AccessPattern::Sequential(start, end) => {
                let size = (end - start) * array.itemsize;
                let analysis = self
                    .profiler
                    .analyze_access_pattern(*start * array.itemsize, size);
                (size, analysis)
            }
            AccessPattern::Random(indices) => {
                let size = indices.len() * array.itemsize;
                let analysis = self.profiler.analyze_access_pattern(0, size);
                (size, analysis)
            }
            AccessPattern::Strided(start, stride, count) => {
                let size = count * array.itemsize;
                let analysis = self
                    .profiler
                    .analyze_access_pattern(*start * array.itemsize, size);
                (size, analysis)
            }
        };

        // 选择算法
        let algorithm = self
            .selector
            .select_algorithm(&access_pattern, "batch_access");

        // 执行批量访问
        let result = match algorithm {
            IndexAlgorithm::BatchParallel => self.execute_batch_parallel(pattern, array),
            IndexAlgorithm::BatchChunked => self.execute_batch_chunked(pattern, array),
            IndexAlgorithm::BatchStreaming => self.execute_batch_streaming(pattern, array),
            _ => self.execute_batch_parallel(pattern, array), // 默认
        };

        // 记录性能
        let duration = start_time.elapsed();
        self.profiler
            .record_operation(algorithm, duration, total_size);

        result
    }

    // 花式索引算法实现
    fn execute_fancy_direct(&self, indices: &[usize], array: &OptimizedLazyArray) -> Vec<Vec<u8>> {
        if let Ok(engine) = self.fancy_index_engine.lock() {
            engine.process_direct(indices, array)
        } else {
            vec![]
        }
    }

    fn execute_fancy_simd(&self, indices: &[usize], array: &OptimizedLazyArray) -> Vec<Vec<u8>> {
        if let Ok(mut engine) = self.fancy_index_engine.lock() {
            engine.process_simd(indices, array)
        } else {
            vec![]
        }
    }

    fn execute_fancy_prefetch(
        &self,
        indices: &[usize],
        array: &OptimizedLazyArray,
    ) -> Vec<Vec<u8>> {
        if let Ok(engine) = self.fancy_index_engine.lock() {
            engine.process_with_prefetch(indices, array)
        } else {
            vec![]
        }
    }

    fn execute_fancy_zero_copy(
        &self,
        indices: &[usize],
        array: &OptimizedLazyArray,
    ) -> Vec<Vec<u8>> {
        if let Ok(engine) = self.fancy_index_engine.lock() {
            engine.process_zero_copy(indices, array)
        } else {
            vec![]
        }
    }

    // 布尔索引算法实现
    fn execute_boolean_bitmap(&self, mask: &[bool], array: &OptimizedLazyArray) -> Vec<Vec<u8>> {
        array.boolean_index(mask)
    }

    fn execute_boolean_hierarchical(
        &self,
        mask: &[bool],
        array: &OptimizedLazyArray,
    ) -> Vec<Vec<u8>> {
        array.boolean_index_optimized(mask)
    }

    fn execute_boolean_sparse(&self, mask: &[bool], array: &OptimizedLazyArray) -> Vec<Vec<u8>> {
        // 对于稀疏布尔索引，首先转换为索引数组
        let indices: Vec<usize> = mask
            .iter()
            .enumerate()
            .filter_map(|(i, &b)| if b { Some(i) } else { None })
            .collect();

        if let Ok(engine) = self.fancy_index_engine.lock() {
            engine.process_direct(&indices, array)
        } else {
            vec![]
        }
    }

    fn execute_boolean_dense(&self, mask: &[bool], array: &OptimizedLazyArray) -> Vec<Vec<u8>> {
        // 对于密集布尔索引，使用优化的批量处理
        array.boolean_index_optimized(mask)
    }

    fn execute_boolean_extreme(&self, mask: &[bool], array: &OptimizedLazyArray) -> Vec<Vec<u8>> {
        // 对于极端情况，使用最保守的方法
        self.execute_boolean_bitmap(mask, array)
    }

    // 批量访问算法实现
    fn execute_batch_parallel(
        &self,
        pattern: &AccessPattern,
        array: &OptimizedLazyArray,
    ) -> Vec<Vec<u8>> {
        if let Ok(mut engine) = self.fancy_index_engine.lock() {
            match pattern {
                AccessPattern::Sequential(start, end) => {
                    let indices: Vec<usize> = (*start..*end).collect();
                    engine.process_simd(&indices, array)
                }
                AccessPattern::Random(indices) => engine.process_with_prefetch(indices, array),
                AccessPattern::Strided(start, stride, count) => {
                    let indices: Vec<usize> = (0..*count).map(|i| start + i * stride).collect();
                    engine.process_simd(&indices, array)
                }
            }
        } else {
            vec![]
        }
    }

    fn execute_batch_chunked(
        &self,
        pattern: &AccessPattern,
        array: &OptimizedLazyArray,
    ) -> Vec<Vec<u8>> {
        // 分块处理大批量访问
        self.execute_batch_parallel(pattern, array)
    }

    fn execute_batch_streaming(
        &self,
        pattern: &AccessPattern,
        array: &OptimizedLazyArray,
    ) -> Vec<Vec<u8>> {
        // 流式处理超大批量访问
        self.execute_batch_parallel(pattern, array)
    }
}
