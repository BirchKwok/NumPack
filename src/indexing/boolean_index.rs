//! 布尔索引引擎实现
//!
//! 提供高效的布尔索引操作，支持多种优化策略

use crate::lazy_array::OptimizedLazyArray;

/// 布尔索引引擎
pub struct BooleanIndexEngine {
    // 可以添加配置和优化相关的字段
}

impl BooleanIndexEngine {
    /// 创建新的布尔索引引擎
    pub fn new() -> Self {
        Self {}
    }

    /// 基础位图布尔索引
    pub fn bitmap_index(&self, mask: &[bool], array: &OptimizedLazyArray) -> Vec<Vec<u8>> {
        let mut results = Vec::new();

        for (idx, &should_include) in mask.iter().enumerate() {
            if should_include && idx < array.shape[0] {
                let row_data = array.get_row(idx);
                results.push(row_data);
            }
        }

        results
    }

    /// 分层布尔索引（针对大型稀疏掩码优化）
    pub fn hierarchical_index(&self, mask: &[bool], array: &OptimizedLazyArray) -> Vec<Vec<u8>> {
        // 首先统计true值的数量
        let true_count = mask.iter().filter(|&&x| x).count();

        if true_count == 0 {
            return Vec::new();
        }

        // 如果true值很少，转换为索引数组更高效
        if true_count < mask.len() / 4 {
            return self.sparse_index(mask, array);
        }

        // 否则使用直接遍历
        self.bitmap_index(mask, array)
    }

    /// 稀疏布尔索引（转换为索引数组）
    pub fn sparse_index(&self, mask: &[bool], array: &OptimizedLazyArray) -> Vec<Vec<u8>> {
        // 收集所有true的索引
        let indices: Vec<usize> = mask
            .iter()
            .enumerate()
            .filter_map(|(i, &b)| if b { Some(i) } else { None })
            .collect();

        // 使用花式索引处理
        let mut results = Vec::with_capacity(indices.len());
        for &idx in &indices {
            if idx < array.shape[0] {
                let row_data = array.get_row(idx);
                results.push(row_data);
            }
        }

        results
    }

    /// 密集布尔索引（针对大部分为true的掩码优化）
    pub fn dense_index(&self, mask: &[bool], array: &OptimizedLazyArray) -> Vec<Vec<u8>> {
        let true_count = mask.iter().filter(|&&x| x).count();

        // 如果大部分都是true，可以考虑批量读取然后过滤
        if true_count > mask.len() * 3 / 4 {
            return self.bulk_then_filter(mask, array);
        }

        // 否则使用标准方法
        self.bitmap_index(mask, array)
    }

    /// 批量读取然后过滤（适用于密集掩码）
    fn bulk_then_filter(&self, mask: &[bool], array: &OptimizedLazyArray) -> Vec<Vec<u8>> {
        let mut results = Vec::new();

        // 分块处理以控制内存使用
        const CHUNK_SIZE: usize = 1000;

        for chunk_start in (0..mask.len()).step_by(CHUNK_SIZE) {
            let chunk_end = (chunk_start + CHUNK_SIZE).min(mask.len());
            let chunk_mask = &mask[chunk_start..chunk_end];

            // 检查这个块中是否有true值
            if !chunk_mask.iter().any(|&x| x) {
                continue;
            }

            // 读取这个块的所有数据
            for (local_idx, &should_include) in chunk_mask.iter().enumerate() {
                let global_idx = chunk_start + local_idx;
                if should_include && global_idx < array.shape[0] {
                    let row_data = array.get_row(global_idx);
                    results.push(row_data);
                }
            }
        }

        results
    }

    /// 极端情况布尔索引（容错性最高）
    pub fn extreme_index(&self, mask: &[bool], array: &OptimizedLazyArray) -> Vec<Vec<u8>> {
        // 最保守的实现，确保在任何情况下都能工作
        let mut results = Vec::new();

        for (idx, &should_include) in mask.iter().enumerate() {
            if should_include {
                // 额外的边界检查
                if idx < array.shape[0] && idx < mask.len() {
                    let row_data = array.get_row(idx);
                    if !row_data.is_empty() {
                        results.push(row_data);
                    }
                }
            }
        }

        results
    }

    /// SIMD优化的布尔索引
    pub fn simd_boolean_filter(&self, mask: &[bool]) -> Vec<usize> {
        // 简化的SIMD布尔过滤，返回索引
        let mut indices = Vec::new();

        // 处理8个布尔值为一组（简化的SIMD模拟）
        let chunks = mask.chunks(8);
        let mut current_idx = 0;

        for chunk in chunks {
            for (local_idx, &value) in chunk.iter().enumerate() {
                if value {
                    indices.push(current_idx + local_idx);
                }
            }
            current_idx += chunk.len();
        }

        indices
    }

    /// 选择最佳的布尔索引策略
    pub fn select_best_strategy(&self, mask: &[bool], array: &OptimizedLazyArray) -> Vec<Vec<u8>> {
        let mask_len = mask.len();
        let true_count = mask.iter().filter(|&&x| x).count();
        let sparsity = true_count as f64 / mask_len as f64;

        match sparsity {
            s if s < 0.1 => self.sparse_index(mask, array), // 非常稀疏
            s if s < 0.25 => self.hierarchical_index(mask, array), // 稀疏
            s if s > 0.75 => self.dense_index(mask, array), // 密集
            _ => self.bitmap_index(mask, array),            // 中等密度
        }
    }
}

impl Default for BooleanIndexEngine {
    fn default() -> Self {
        Self::new()
    }
}
