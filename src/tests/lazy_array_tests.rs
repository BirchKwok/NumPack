//! LazyArray核心功能测试
//! 测试OptimizedLazyArray的基本功能和性能

#[cfg(test)]
mod tests {
    use crate::core::metadata::DataType;
    use crate::lazy_array::core::OptimizedLazyArray;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn create_test_file(rows: usize, cols: usize, item_size: usize) -> (NamedTempFile, Vec<u8>) {
        let mut temp_file = NamedTempFile::new().unwrap();
        let total_size = rows * cols * item_size;
        let test_data: Vec<u8> = (0..total_size).map(|i| (i % 256) as u8).collect();
        temp_file.write_all(&test_data).unwrap();
        (temp_file, test_data)
    }

    #[test]
    fn test_lazy_array_creation() {
        let (temp_file, _) = create_test_file(10, 5, 8);

        let lazy_array = OptimizedLazyArray::new(
            temp_file.path().to_path_buf(),
            vec![10, 5],
            DataType::Uint64,
        );

        assert!(lazy_array.is_ok());
        let array = lazy_array.unwrap();
        assert_eq!(array.shape, vec![10, 5]);
        assert_eq!(array.itemsize, 8);
    }

    #[test]
    fn test_from_file_constructor() {
        let (temp_file, _) = create_test_file(20, 3, 4);
        let file_path = temp_file.path().to_str().unwrap();

        let lazy_array = OptimizedLazyArray::from_file(file_path, vec![20, 3], 4);

        assert!(lazy_array.is_ok());
        let array = lazy_array.unwrap();
        assert_eq!(array.shape, vec![20, 3]);
        assert_eq!(array.itemsize, 4);
    }

    #[test]
    fn test_read_data_basic() {
        let (temp_file, original_data) = create_test_file(5, 4, 2);

        let array =
            OptimizedLazyArray::new(temp_file.path().to_path_buf(), vec![5, 4], DataType::Uint16)
                .unwrap();

        // 读取前10个字节
        let read_data = array.read_data(0, 10);
        assert_eq!(read_data.len(), 10);
        assert_eq!(read_data, &original_data[0..10]);

        // 读取中间部分
        let read_data = array.read_data(10, 10);
        assert_eq!(read_data.len(), 10);
        assert_eq!(read_data, &original_data[10..20]);
    }

    #[test]
    fn test_read_data_bounds() {
        let (temp_file, _) = create_test_file(3, 3, 1);

        let array =
            OptimizedLazyArray::new(temp_file.path().to_path_buf(), vec![3, 3], DataType::Uint8)
                .unwrap();

        // 测试越界读取
        let read_data = array.read_data(100, 10);
        assert!(read_data.is_empty());

        // 测试部分越界
        let read_data = array.read_data(5, 10);
        assert_eq!(read_data.len(), 4); // 只能读取4个字节 (9 - 5)
    }

    #[test]
    fn test_shape_and_itemsize_access() {
        let (temp_file, _) = create_test_file(7, 8, 4);

        let array =
            OptimizedLazyArray::new(temp_file.path().to_path_buf(), vec![7, 8], DataType::Uint32)
                .unwrap();

        // 测试公共字段访问
        assert_eq!(array.shape[0], 7);
        assert_eq!(array.shape[1], 8);
        assert_eq!(array.itemsize, 4);
    }

    #[test]
    fn test_placeholder_methods() {
        let (temp_file, _) = create_test_file(5, 5, 8);

        let array =
            OptimizedLazyArray::new(temp_file.path().to_path_buf(), vec![5, 5], DataType::Uint64)
                .unwrap();

        // 测试占位符方法不会崩溃
        let _row = array.get_row(0);
        let _rows = array.get_rows(&[0, 1, 2]);
        let _range = array.get_rows_range(0, 3);

        // 测试布尔索引方法
        let mask = vec![true, false, true, false, true];
        let _result = array.boolean_index(&mask);
        let _result = array.boolean_index_optimized(&mask);

        // 测试缓存方法
        array.clear_cache();
        let _stats = array.get_cache_stats();
        array.warmup_cache(0.1);
    }

    #[test]
    fn test_different_data_types() {
        // 测试不同数据类型的构造
        let test_cases = vec![
            (DataType::Uint8, 1),
            (DataType::Uint16, 2),
            (DataType::Uint32, 4),
            (DataType::Uint64, 8),
        ];

        for (dtype, expected_size) in test_cases {
            let (temp_file, _) = create_test_file(10, 10, expected_size);

            let array =
                OptimizedLazyArray::new(temp_file.path().to_path_buf(), vec![10, 10], dtype)
                    .unwrap();

            assert_eq!(array.itemsize, expected_size);
        }
    }
}
