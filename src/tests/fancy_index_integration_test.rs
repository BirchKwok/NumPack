//! 花式索引引擎集成测试
//!
//! 这个模块包含了FancyIndexEngine与OptimizedLazyArray的集成测试

use crate::core::metadata::DataType;
use crate::{indexing::fancy_index::FancyIndexEngine, lazy_array::core::OptimizedLazyArray};
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;

#[cfg(test)]
mod integration_tests {
    use super::*;

    /// 创建简单的测试数据文件
    fn create_simple_test_file() -> (PathBuf, Vec<u8>) {
        let test_data: Vec<u8> = (0..200).map(|i| (i % 256) as u8).collect();
        let temp_dir = std::env::temp_dir();
        let temp_path = temp_dir.join(format!("integration_test_{}.dat", std::process::id()));

        let mut file = File::create(&temp_path).expect("Failed to create test file");
        file.write_all(&test_data)
            .expect("Failed to write test data");

        (temp_path, test_data)
    }

    #[test]
    fn test_fancy_index_engine_integration() {
        // 创建测试数据：10行，每行5个元素，每个元素4字节
        let (temp_path, expected_data) = create_simple_test_file();
        let shape = vec![10, 5];

        // 创建OptimizedLazyArray
        let array = OptimizedLazyArray::new(temp_path.clone(), shape, DataType::Float32)
            .expect("Failed to create OptimizedLazyArray");

        // 创建FancyIndexEngine
        let engine = FancyIndexEngine::new();

        // 测试直接访问
        let indices = vec![0, 2, 4];
        let results = engine.process_direct(&indices, &array);

        // 验证结果
        assert_eq!(results.len(), 3);

        let row_size = 5 * 4; // 5 columns * 4 bytes per element
        for (i, result) in results.iter().enumerate() {
            assert_eq!(result.len(), row_size);

            // 验证数据正确性
            let row_idx = indices[i];
            let expected_start = row_idx * row_size;
            let expected_row = &expected_data[expected_start..expected_start + row_size];
            assert_eq!(result, expected_row);
        }

        // 清理
        std::fs::remove_file(temp_path).ok();
    }

    #[test]
    fn test_fancy_index_engine_empty_indices() {
        let (temp_path, _) = create_simple_test_file();
        let shape = vec![10, 5];

        let array = OptimizedLazyArray::new(temp_path.clone(), shape, DataType::Float32)
            .expect("Failed to create OptimizedLazyArray");

        let engine = FancyIndexEngine::new();

        // 测试空索引
        let indices: Vec<usize> = vec![];
        let results = engine.process_direct(&indices, &array);

        assert!(results.is_empty());

        std::fs::remove_file(temp_path).ok();
    }

    #[test]
    fn test_fancy_index_engine_invalid_indices() {
        let (temp_path, _) = create_simple_test_file();
        let shape = vec![10, 5];

        let array = OptimizedLazyArray::new(temp_path.clone(), shape, DataType::Float32)
            .expect("Failed to create OptimizedLazyArray");

        let engine = FancyIndexEngine::new();

        // 测试包含无效索引的情况
        let indices = vec![0, 15, 2]; // 索引15超出范围
        let results = engine.process_direct(&indices, &array);

        assert_eq!(results.len(), 3);
        assert!(!results[0].is_empty()); // 有效索引
        assert!(results[1].is_empty()); // 无效索引返回空
        assert!(!results[2].is_empty()); // 有效索引

        std::fs::remove_file(temp_path).ok();
    }

    #[test]
    fn test_fancy_index_engine_performance_stats() {
        let (temp_path, _) = create_simple_test_file();
        let shape = vec![10, 5];

        let array = OptimizedLazyArray::new(temp_path.clone(), shape, DataType::Float32)
            .expect("Failed to create OptimizedLazyArray");

        let engine = FancyIndexEngine::new();

        // 初始统计应该为零
        if let Some((hit_rate, hits, misses)) = engine.get_performance_stats() {
            assert_eq!(hit_rate, 0.0);
            assert_eq!(hits, 0);
            assert_eq!(misses, 0);
        }

        // 执行一些操作
        let indices = vec![0, 1, 2];
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
}
