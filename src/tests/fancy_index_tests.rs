//! 花式索引引擎测试模块
//!
//! 这个模块包含了FancyIndexEngine的全面测试，验证各种访问方法的正确性和性能

use crate::core::metadata::DataType;
use crate::{
    access_pattern::{AccessFrequency, AccessPatternAnalysis, AccessPatternType, SizeCategory},
    indexing::fancy_index::FancyIndexEngine,
    lazy_array::core::OptimizedLazyArray,
};
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;

/// 创建测试用的临时数据文件
fn create_test_data_file(rows: usize, cols: usize, item_size: usize) -> (PathBuf, Vec<u8>) {
    let test_data: Vec<u8> = (0..rows * cols * item_size)
        .map(|i| (i % 256) as u8)
        .collect();

    // 使用系统临时目录而不是硬编码的/tmp
    let temp_dir = std::env::temp_dir();
    let temp_path = temp_dir.join(format!("test_fancy_index_{}.dat", std::process::id()));

    let mut file = File::create(&temp_path).expect("Failed to create test file");
    file.write_all(&test_data)
        .expect("Failed to write test data");

    (temp_path, test_data)
}

/// 创建测试用的访问模式分析
fn create_test_pattern(
    pattern_type: AccessPatternType,
    size_category: SizeCategory,
) -> AccessPatternAnalysis {
    AccessPatternAnalysis {
        pattern_type,
        locality_score: 0.8,
        density: 0.6,
        size_category,
        frequency: AccessFrequency::Medium,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fancy_index_engine_creation() {
        let engine = FancyIndexEngine::new();

        // 验证引擎创建成功
        assert!(engine.get_performance_stats().is_some());

        // 验证初始统计为零
        if let Some((hit_rate, hits, misses)) = engine.get_performance_stats() {
            assert_eq!(hit_rate, 0.0);
            assert_eq!(hits, 0);
            assert_eq!(misses, 0);
        }
    }

    #[test]
    fn test_process_direct_basic() {
        // 创建测试数据
        let (temp_path, expected_data) = create_test_data_file(10, 5, 4);
        let shape = vec![10, 5];
        let array = OptimizedLazyArray::new(temp_path.clone(), shape, DataType::Float32)
            .expect("Failed to create OptimizedLazyArray");

        let engine = FancyIndexEngine::new();
        let indices = vec![0, 2, 4];

        // 测试直接访问
        let results = engine.process_direct(&indices, &array);

        // 验证结果
        assert_eq!(results.len(), 3);
        for (i, result) in results.iter().enumerate() {
            let expected_row_size = 5 * 4; // 5 columns * 4 bytes
            assert_eq!(result.len(), expected_row_size);

            // 验证数据正确性
            let row_idx = indices[i];
            let expected_start = row_idx * expected_row_size;
            let expected_row = &expected_data[expected_start..expected_start + expected_row_size];
            assert_eq!(result, expected_row);
        }

        // 清理
        std::fs::remove_file(temp_path).ok();
    }

    #[test]
    fn test_process_direct_invalid_indices() {
        let (temp_path, _) = create_test_data_file(5, 3, 4);
        let shape = vec![5, 3];
        let array = OptimizedLazyArray::new(temp_path.clone(), shape, DataType::Float32)
            .expect("Failed to create OptimizedLazyArray");

        let engine = FancyIndexEngine::new();
        let indices = vec![0, 10, 2]; // 索引10超出范围

        let results = engine.process_direct(&indices, &array);

        // 验证结果
        assert_eq!(results.len(), 3);
        assert!(!results[0].is_empty()); // 有效索引
        assert!(results[1].is_empty()); // 无效索引返回空
        assert!(!results[2].is_empty()); // 有效索引

        std::fs::remove_file(temp_path).ok();
    }

    #[test]
    fn test_process_simd_basic() {
        let (temp_path, expected_data) = create_test_data_file(8, 4, 4);
        let shape = vec![8, 4];
        let array = OptimizedLazyArray::new(temp_path.clone(), shape, DataType::Float32)
            .expect("Failed to create OptimizedLazyArray");

        let engine = FancyIndexEngine::new();
        let indices = vec![1, 3, 5, 7];

        let results = engine.process_simd(&indices, &array);

        // 验证结果
        assert_eq!(results.len(), 4);
        let row_size = 4 * 4; // 4 columns * 4 bytes

        for (i, result) in results.iter().enumerate() {
            assert_eq!(result.len(), row_size);

            // 验证数据正确性
            let row_idx = indices[i];
            let expected_start = row_idx * row_size;
            let expected_row = &expected_data[expected_start..expected_start + row_size];
            assert_eq!(result, expected_row);
        }

        std::fs::remove_file(temp_path).ok();
    }

    #[test]
    fn test_process_with_prefetch() {
        let (temp_path, expected_data) = create_test_data_file(12, 6, 4);
        let shape = vec![12, 6];
        let array = OptimizedLazyArray::new(temp_path.clone(), shape, DataType::Float32)
            .expect("Failed to create OptimizedLazyArray");

        let engine = FancyIndexEngine::new();
        let indices = vec![0, 1, 2, 3]; // 顺序访问，适合预取

        let results = engine.process_with_prefetch(&indices, &array);

        // 验证结果
        assert_eq!(results.len(), 4);
        let row_size = 6 * 4;

        for (i, result) in results.iter().enumerate() {
            assert_eq!(result.len(), row_size);

            let row_idx = indices[i];
            let expected_start = row_idx * row_size;
            let expected_row = &expected_data[expected_start..expected_start + row_size];
            assert_eq!(result, expected_row);
        }

        // 验证预取统计有更新
        if let Some((hit_rate, hits, misses)) = engine.get_performance_stats() {
            assert!(hits + misses > 0); // 应该有访问记录
        }

        std::fs::remove_file(temp_path).ok();
    }

    #[test]
    fn test_process_zero_copy_continuous() {
        let (temp_path, expected_data) = create_test_data_file(10, 3, 4);
        let shape = vec![10, 3];
        let array = OptimizedLazyArray::new(temp_path.clone(), shape, DataType::Float32)
            .expect("Failed to create OptimizedLazyArray");

        let engine = FancyIndexEngine::new();
        let indices = vec![2, 3, 4, 5]; // 连续索引，适合零拷贝优化

        let results = engine.process_zero_copy(&indices, &array);

        // 验证结果
        assert_eq!(results.len(), 4);
        let row_size = 3 * 4;

        for (i, result) in results.iter().enumerate() {
            assert_eq!(result.len(), row_size);

            let row_idx = indices[i];
            let expected_start = row_idx * row_size;
            let expected_row = &expected_data[expected_start..expected_start + row_size];
            assert_eq!(result, expected_row);
        }

        std::fs::remove_file(temp_path).ok();
    }

    #[test]
    fn test_process_zero_copy_non_continuous() {
        let (temp_path, expected_data) = create_test_data_file(10, 3, 4);
        let shape = vec![10, 3];
        let array = OptimizedLazyArray::new(temp_path.clone(), shape, DataType::Float32)
            .expect("Failed to create OptimizedLazyArray");

        let engine = FancyIndexEngine::new();
        let indices = vec![1, 4, 7]; // 非连续索引

        let results = engine.process_zero_copy(&indices, &array);

        // 验证结果
        assert_eq!(results.len(), 3);
        let row_size = 3 * 4;

        for (i, result) in results.iter().enumerate() {
            assert_eq!(result.len(), row_size);

            let row_idx = indices[i];
            let expected_start = row_idx * row_size;
            let expected_row = &expected_data[expected_start..expected_start + row_size];
            assert_eq!(result, expected_row);
        }

        std::fs::remove_file(temp_path).ok();
    }

    #[test]
    fn test_select_optimal_method_sequential_large() {
        let (temp_path, expected_data) = create_test_data_file(100, 10, 4);
        let shape = vec![100, 10];
        let array = OptimizedLazyArray::new(temp_path.clone(), shape, DataType::Float32)
            .expect("Failed to create OptimizedLazyArray");

        let engine = FancyIndexEngine::new();
        let indices: Vec<usize> = (10..20).collect(); // 大规模顺序访问
        let pattern = create_test_pattern(AccessPatternType::Sequential, SizeCategory::Large);

        let results = engine.select_optimal_method(&indices, &array, &pattern);

        // 验证结果正确性
        assert_eq!(results.len(), 10);
        let row_size = 10 * 4;

        for (i, result) in results.iter().enumerate() {
            assert_eq!(result.len(), row_size);

            let row_idx = indices[i];
            let expected_start = row_idx * row_size;
            let expected_row = &expected_data[expected_start..expected_start + row_size];
            assert_eq!(result, expected_row);
        }

        std::fs::remove_file(temp_path).ok();
    }

    #[test]
    fn test_select_optimal_method_random_small() {
        let (temp_path, expected_data) = create_test_data_file(20, 5, 4);
        let shape = vec![20, 5];
        let array = OptimizedLazyArray::new(temp_path.clone(), shape, DataType::Float32)
            .expect("Failed to create OptimizedLazyArray");

        let engine = FancyIndexEngine::new();
        let indices = vec![1, 15, 3]; // 小规模随机访问
        let pattern = create_test_pattern(AccessPatternType::Random, SizeCategory::Small);

        let results = engine.select_optimal_method(&indices, &array, &pattern);

        // 验证结果正确性
        assert_eq!(results.len(), 3);
        let row_size = 5 * 4;

        for (i, result) in results.iter().enumerate() {
            assert_eq!(result.len(), row_size);

            let row_idx = indices[i];
            let expected_start = row_idx * row_size;
            let expected_row = &expected_data[expected_start..expected_start + row_size];
            assert_eq!(result, expected_row);
        }

        std::fs::remove_file(temp_path).ok();
    }

    #[test]
    fn test_performance_stats_tracking() {
        let (temp_path, _) = create_test_data_file(10, 3, 4);
        let shape = vec![10, 3];
        let array = OptimizedLazyArray::new(temp_path.clone(), shape, DataType::Float32)
            .expect("Failed to create OptimizedLazyArray");

        let engine = FancyIndexEngine::new();
        let indices = vec![0, 1, 2];

        // 初始统计应该为零
        if let Some((hit_rate, hits, misses)) = engine.get_performance_stats() {
            assert_eq!(hit_rate, 0.0);
            assert_eq!(hits, 0);
            assert_eq!(misses, 0);
        }

        // 执行预取访问以更新统计
        let _results = engine.process_with_prefetch(&indices, &array);

        // 统计应该有更新
        if let Some((_, hits, misses)) = engine.get_performance_stats() {
            assert!(hits + misses > 0);
        }

        // 重置统计
        engine.reset_stats();

        // 统计应该重置为零
        if let Some((hit_rate, hits, misses)) = engine.get_performance_stats() {
            assert_eq!(hit_rate, 0.0);
            assert_eq!(hits, 0);
            assert_eq!(misses, 0);
        }

        std::fs::remove_file(temp_path).ok();
    }

    #[test]
    fn test_empty_indices() {
        let (temp_path, _) = create_test_data_file(5, 3, 4);
        let shape = vec![5, 3];
        let array = OptimizedLazyArray::new(temp_path.clone(), shape, DataType::Float32)
            .expect("Failed to create OptimizedLazyArray");

        let engine = FancyIndexEngine::new();
        let indices: Vec<usize> = vec![];

        // 测试所有方法都能正确处理空索引
        let results_direct = engine.process_direct(&indices, &array);
        let results_simd = engine.process_simd(&indices, &array);
        let results_prefetch = engine.process_with_prefetch(&indices, &array);
        let results_zero_copy = engine.process_zero_copy(&indices, &array);

        assert!(results_direct.is_empty());
        assert!(results_simd.is_empty());
        assert!(results_prefetch.is_empty());
        assert!(results_zero_copy.is_empty());

        std::fs::remove_file(temp_path).ok();
    }

    #[test]
    fn test_large_batch_processing() {
        let (temp_path, expected_data) = create_test_data_file(1000, 8, 4);
        let shape = vec![1000, 8];
        let array = OptimizedLazyArray::new(temp_path.clone(), shape, DataType::Float32)
            .expect("Failed to create OptimizedLazyArray");

        let engine = FancyIndexEngine::new();
        let indices: Vec<usize> = (0..100).step_by(10).collect(); // 每10个取一个，共10个

        let results = engine.process_simd(&indices, &array);

        // 验证大批量处理的正确性
        assert_eq!(results.len(), 10);
        let row_size = 8 * 4;

        for (i, result) in results.iter().enumerate() {
            assert_eq!(result.len(), row_size);

            let row_idx = indices[i];
            let expected_start = row_idx * row_size;
            let expected_row = &expected_data[expected_start..expected_start + row_size];
            assert_eq!(result, expected_row);
        }

        std::fs::remove_file(temp_path).ok();
    }
}
