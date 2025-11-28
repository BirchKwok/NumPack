//! 索引系统测试
//! 测试花式索引、布尔索引和智能路由功能

#[cfg(test)]
mod tests {
    use crate::core::metadata::DataType;
    use crate::indexing::{
        boolean_index::BooleanIndexEngine, fancy_index::FancyIndexEngine,
        smart_router::SmartIndexRouter,
    };
    use crate::lazy_array::core::OptimizedLazyArray;
    use std::io::Write;
    use std::path::PathBuf;
    use tempfile::NamedTempFile;

    fn create_test_array(rows: usize, cols: usize) -> OptimizedLazyArray {
        let mut temp_file = NamedTempFile::new().unwrap();
        let total_size = rows * cols * 8; // 8字节每个元素
        let test_data: Vec<u8> = (0..total_size).map(|i| (i % 256) as u8).collect();
        temp_file.write_all(&test_data).unwrap();

        OptimizedLazyArray::new(
            temp_file.path().to_path_buf(),
            vec![rows, cols],
            DataType::Uint64,
        )
        .unwrap()
    }

    #[test]
    fn test_fancy_index_basic() {
        let engine = FancyIndexEngine::new();
        let array = create_test_array(100, 10);
        let indices = vec![0, 5, 10, 15, 20];

        let result = engine.process_direct(&indices, &array);
        assert_eq!(result.len(), indices.len());

        // 验证每个结果都不为空
        for row in &result {
            assert!(!row.is_empty());
        }
    }

    #[test]
    fn test_fancy_index_simd() {
        let engine = FancyIndexEngine::new();
        let array = create_test_array(100, 10);
        let indices = vec![1, 3, 5, 7, 9, 11, 13, 15];

        let result = engine.process_simd(&indices, &array);
        assert_eq!(result.len(), indices.len());
    }

    #[test]
    fn test_boolean_index_basic() {
        let engine = BooleanIndexEngine::new();
        let array = create_test_array(10, 10);
        let mask = vec![
            true, false, true, false, true, false, true, false, true, false,
        ];

        let result = engine.process_bitmap(&mask, &array);

        // 应该返回5行数据（mask中有5个true）
        assert_eq!(result.len(), 5);

        for row in &result {
            assert!(!row.is_empty());
        }
    }

    #[test]
    fn test_boolean_index_sparse() {
        let engine = BooleanIndexEngine::new();
        let array = create_test_array(100, 10);

        // 创建稀疏掩码（只有少数true）
        let mut mask = vec![false; 100];
        mask[5] = true;
        mask[25] = true;
        mask[75] = true;

        let result = engine.process_sparse(&mask, &array);
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_smart_router_basic() {
        let router = SmartIndexRouter::new();
        let array = create_test_array(50, 10);
        let indices = vec![0, 1, 2, 3, 4]; // 顺序访问

        let result = router.route_fancy_index(&indices, &array);
        assert_eq!(result.len(), indices.len());
    }

    #[test]
    fn test_smart_router_boolean() {
        let router = SmartIndexRouter::new();
        let array = create_test_array(20, 10);
        let mask = vec![true; 10]; // 前10个为true

        let result = router.route_boolean_index(&mask, &array);
        assert_eq!(result.len(), 10);
    }

    #[test]
    fn test_performance_stats() {
        let engine = FancyIndexEngine::new();
        let array = create_test_array(100, 10);
        let indices = vec![0, 10, 20, 30, 40];

        // 执行一些操作
        let _result1 = engine.process_direct(&indices, &array);
        let _result2 = engine.process_simd(&indices, &array);

        // 获取性能统计
        let stats = engine.get_performance_stats();
        assert!(stats.total_operations > 0);
    }
}
