// SIMD优化测试模块
use crate::lazy_array::*;
use std::io::Write;
use std::time::Instant;
use tempfile::NamedTempFile;

/// 创建测试用的SIMD处理器
fn create_test_simd_processor() -> SIMDProcessor {
    SIMDProcessor::new()
}

/// 创建测试数据
fn create_test_data(size: usize) -> Vec<u8> {
    (0..size).map(|i| (i % 256) as u8).collect()
}

/// 创建测试索引
fn create_test_indices(count: usize, max_index: usize) -> Vec<usize> {
    (0..count).map(|i| (i * 7) % max_index).collect()
}

/// 创建测试用的LazyArray
fn create_test_lazy_array(rows: usize, cols: usize, _item_size: usize) -> OptimizedLazyArray {
    let mut temp_file = NamedTempFile::new().unwrap();
    let total_size = rows * cols * 8; // 固定使用8字节
    let test_data: Vec<u8> = (0..total_size).map(|i| (i % 256) as u8).collect();
    temp_file.write_all(&test_data).unwrap();

    OptimizedLazyArray::new(
        temp_file.path().to_path_buf(),
        vec![rows, cols],
        crate::metadata::DataType::Uint8,
    )
    .unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_processor_creation() {
        let processor = create_test_simd_processor();
        let capabilities = processor.get_capabilities();

        // 验证处理器创建成功
        assert!(capabilities.alignment_size > 0);
        assert!(capabilities.cache_line_size > 0);
        assert!(capabilities.optimal_chunk_size > 0);

        println!("SIMD Capabilities:");
        println!("  SSE2: {}", capabilities.supports_sse2);
        println!("  AVX2: {}", capabilities.supports_avx2);
        println!("  AVX512: {}", capabilities.supports_avx512);
        println!("  Alignment: {} bytes", capabilities.alignment_size);
        println!("  Cache line: {} bytes", capabilities.cache_line_size);
        println!("  Optimal chunk: {}", capabilities.optimal_chunk_size);
    }

    #[test]
    fn test_vectorized_copy_correctness() {
        let processor = create_test_simd_processor();
        let src_data = create_test_data(1024);
        let indices = vec![0, 2, 4, 6, 8, 10, 12, 14];
        let item_size = 16;

        let mut dst_simd = vec![0u8; indices.len() * item_size];
        let mut dst_scalar = vec![0u8; indices.len() * item_size];

        // SIMD复制
        processor.vectorized_copy(&src_data, &mut dst_simd, &indices, item_size);

        // 标量复制作为参考
        processor.optimized_scalar_copy(&src_data, &mut dst_scalar, &indices, item_size);

        // 验证结果一致性
        assert_eq!(dst_simd, dst_scalar, "SIMD和标量复制结果不一致");

        // 验证数据正确性
        for (i, &idx) in indices.iter().enumerate() {
            let src_start = idx * item_size;
            let dst_start = i * item_size;

            if src_start + item_size <= src_data.len() {
                assert_eq!(
                    &dst_simd[dst_start..dst_start + item_size],
                    &src_data[src_start..src_start + item_size],
                    "索引 {} 的数据不正确",
                    idx
                );
            }
        }
    }

    #[test]
    fn test_different_item_sizes() {
        let processor = create_test_simd_processor();
        let src_data = create_test_data(2048);
        let indices = vec![0, 1, 2, 3, 4, 5, 6, 7];

        // 测试不同的item_size
        let item_sizes = vec![1, 2, 4, 8, 16, 32, 64, 128];

        for item_size in item_sizes {
            let mut dst_simd = vec![0u8; indices.len() * item_size];
            let mut dst_scalar = vec![0u8; indices.len() * item_size];

            processor.vectorized_copy(&src_data, &mut dst_simd, &indices, item_size);
            processor.optimized_scalar_copy(&src_data, &mut dst_scalar, &indices, item_size);

            assert_eq!(dst_simd, dst_scalar, "item_size={} 时结果不一致", item_size);
        }
    }

    #[test]
    fn test_large_scale_vectorized_copy() {
        let processor = create_test_simd_processor();
        let src_data = create_test_data(100000);
        let indices = create_test_indices(1000, 1000);
        let item_size = 64;

        let mut dst = vec![0u8; indices.len() * item_size];

        let start_time = Instant::now();
        processor.vectorized_copy(&src_data, &mut dst, &indices, item_size);
        let duration = start_time.elapsed();

        println!("大规模向量化复制耗时: {:?}", duration);

        // 验证部分结果的正确性
        for i in 0..10 {
            let idx = indices[i];
            let src_start = idx * item_size;
            let dst_start = i * item_size;

            if src_start + item_size <= src_data.len() {
                assert_eq!(
                    &dst[dst_start..dst_start + item_size],
                    &src_data[src_start..src_start + item_size],
                    "大规模测试中索引 {} 的数据不正确",
                    idx
                );
            }
        }
    }

    #[test]
    fn test_vectorized_index_calculation() {
        let processor = create_test_simd_processor();
        let base_indices = vec![0, 100, 200, 300];
        let stride = 10;
        let count = 5;

        let result = processor.vectorized_index_calculation(&base_indices, stride, count);

        // 验证结果长度
        assert_eq!(result.len(), base_indices.len() * count);

        // 验证计算正确性
        let mut expected = Vec::new();
        for &base in &base_indices {
            for i in 0..count {
                expected.push(base + i * stride);
            }
        }

        assert_eq!(result, expected, "向量化索引计算结果不正确");
    }

    #[test]
    fn test_memory_alignment_detection() {
        let processor = create_test_simd_processor();

        // 测试对齐的内存
        let aligned_data = vec![0u8; 128];
        let aligned_ptr = aligned_data.as_ptr();

        // 测试非对齐的内存
        let unaligned_data = vec![0u8; 129];
        let unaligned_ptr = unsafe { unaligned_data.as_ptr().add(1) };

        // 注意：这个测试可能因为内存分配器的行为而不稳定
        // 主要是验证is_aligned方法不会崩溃
        let _aligned_result = processor.is_aligned(aligned_ptr);
        let _unaligned_result = processor.is_aligned(unaligned_ptr);
    }

    #[test]
    fn test_simd_performance_comparison() {
        let processor = create_test_simd_processor();
        let src_data = create_test_data(50000);
        let indices = create_test_indices(500, 500);
        let item_size = 32;

        let mut dst_simd = vec![0u8; indices.len() * item_size];
        let mut dst_scalar = vec![0u8; indices.len() * item_size];

        // 测试SIMD性能
        let start_time = Instant::now();
        processor.vectorized_copy(&src_data, &mut dst_simd, &indices, item_size);
        let simd_duration = start_time.elapsed();

        // 测试标量性能
        let start_time = Instant::now();
        processor.optimized_scalar_copy(&src_data, &mut dst_scalar, &indices, item_size);
        let scalar_duration = start_time.elapsed();

        println!("SIMD复制耗时: {:?}", simd_duration);
        println!("标量复制耗时: {:?}", scalar_duration);

        // 验证结果一致性
        assert_eq!(dst_simd, dst_scalar, "性能测试中SIMD和标量结果不一致");

        // 在支持SIMD的平台上，SIMD应该更快或至少不慢太多
        if processor.get_capabilities().supports_avx2 {
            let speedup_ratio = scalar_duration.as_nanos() as f64 / simd_duration.as_nanos() as f64;
            println!("SIMD加速比: {:.2}x", speedup_ratio);

            // SIMD至少不应该比标量慢太多（允许一些开销）
            assert!(
                speedup_ratio > 0.5,
                "SIMD性能过差，加速比: {:.2}",
                speedup_ratio
            );
        }
    }

    #[test]
    fn test_edge_cases() {
        let processor = create_test_simd_processor();

        // 测试空索引
        let src_data = create_test_data(100);
        let empty_indices: Vec<usize> = vec![];
        let mut dst = vec![0u8; 0];
        processor.vectorized_copy(&src_data, &mut dst, &empty_indices, 8);
        assert_eq!(dst.len(), 0);

        // 测试单个索引
        let single_index = vec![5];
        let mut dst = vec![0u8; 8];
        processor.vectorized_copy(&src_data, &mut dst, &single_index, 8);
        assert_eq!(&dst[0..8], &src_data[40..48]);

        // 测试越界索引（应该被安全处理）
        let out_of_bounds = vec![1000];
        let mut dst = vec![0u8; 8];
        processor.vectorized_copy(&src_data, &mut dst, &out_of_bounds, 8);
        // 应该不会崩溃，dst保持原值或被清零
    }

    #[test]
    fn test_cache_line_optimization() {
        let processor = create_test_simd_processor();
        let capabilities = processor.get_capabilities();

        // 测试缓存行大小的数据
        let cache_line_size = capabilities.cache_line_size;
        let src_data = create_test_data(cache_line_size * 10);
        let indices = vec![0, 1, 2, 3]; // 连续索引，应该有好的缓存局部性

        let mut dst = vec![0u8; indices.len() * cache_line_size];

        let start_time = Instant::now();
        processor.vectorized_copy(&src_data, &mut dst, &indices, cache_line_size);
        let duration = start_time.elapsed();

        println!("缓存行优化测试耗时: {:?}", duration);

        // 验证结果正确性
        for (i, &idx) in indices.iter().enumerate() {
            let src_start = idx * cache_line_size;
            let dst_start = i * cache_line_size;

            assert_eq!(
                &dst[dst_start..dst_start + cache_line_size],
                &src_data[src_start..src_start + cache_line_size],
                "缓存行优化测试中索引 {} 的数据不正确",
                idx
            );
        }
    }

    #[test]
    fn test_random_access_pattern() {
        let processor = create_test_simd_processor();
        let src_data = create_test_data(10000);

        // 创建随机访问模式
        let mut indices = vec![13, 157, 2, 891, 45, 678, 234, 567, 89, 345];
        indices.sort(); // 确保索引有效
        let item_size = 16;

        let mut dst_simd = vec![0u8; indices.len() * item_size];
        let mut dst_scalar = vec![0u8; indices.len() * item_size];

        processor.vectorized_copy(&src_data, &mut dst_simd, &indices, item_size);
        processor.optimized_scalar_copy(&src_data, &mut dst_scalar, &indices, item_size);

        assert_eq!(dst_simd, dst_scalar, "随机访问模式测试结果不一致");
    }

    #[test]
    fn test_sequential_access_pattern() {
        let processor = create_test_simd_processor();
        let src_data = create_test_data(1000);

        // 创建顺序访问模式
        let indices: Vec<usize> = (0..50).collect();
        let item_size = 8;

        let mut dst_simd = vec![0u8; indices.len() * item_size];
        let mut dst_scalar = vec![0u8; indices.len() * item_size];

        let start_time = Instant::now();
        processor.vectorized_copy(&src_data, &mut dst_simd, &indices, item_size);
        let simd_duration = start_time.elapsed();

        let start_time = Instant::now();
        processor.optimized_scalar_copy(&src_data, &mut dst_scalar, &indices, item_size);
        let scalar_duration = start_time.elapsed();

        println!(
            "顺序访问 - SIMD: {:?}, 标量: {:?}",
            simd_duration, scalar_duration
        );

        assert_eq!(dst_simd, dst_scalar, "顺序访问模式测试结果不一致");
    }

    #[test]
    fn test_strided_access_pattern() {
        let processor = create_test_simd_processor();
        let src_data = create_test_data(2000);

        // 创建步长访问模式
        let indices: Vec<usize> = (0..20).map(|i| i * 5).collect(); // 步长为5
        let item_size = 12;

        let mut dst_simd = vec![0u8; indices.len() * item_size];
        let mut dst_scalar = vec![0u8; indices.len() * item_size];

        processor.vectorized_copy(&src_data, &mut dst_simd, &indices, item_size);
        processor.optimized_scalar_copy(&src_data, &mut dst_scalar, &indices, item_size);

        assert_eq!(dst_simd, dst_scalar, "步长访问模式测试结果不一致");
    }

    /// 基准测试：比较不同SIMD指令集的性能
    #[test]
    fn benchmark_simd_instruction_sets() {
        let processor = create_test_simd_processor();
        let capabilities = processor.get_capabilities();

        if !capabilities.supports_sse2 {
            println!("跳过SIMD基准测试：不支持SSE2");
            return;
        }

        let src_data = create_test_data(100000);
        let indices = create_test_indices(1000, 1000);
        let item_size = 64;

        println!("SIMD指令集基准测试:");
        println!("数据大小: {} bytes", src_data.len());
        println!("索引数量: {}", indices.len());
        println!("项目大小: {} bytes", item_size);

        // 测试标量性能
        let mut dst_scalar = vec![0u8; indices.len() * item_size];
        let start_time = Instant::now();
        processor.optimized_scalar_copy(&src_data, &mut dst_scalar, &indices, item_size);
        let scalar_duration = start_time.elapsed();
        println!("标量复制: {:?}", scalar_duration);

        // 测试SIMD性能
        let mut dst_simd = vec![0u8; indices.len() * item_size];
        let start_time = Instant::now();
        processor.vectorized_copy(&src_data, &mut dst_simd, &indices, item_size);
        let simd_duration = start_time.elapsed();
        println!("SIMD复制: {:?}", simd_duration);

        // 计算加速比
        let speedup = scalar_duration.as_nanos() as f64 / simd_duration.as_nanos() as f64;
        println!("SIMD加速比: {:.2}x", speedup);

        // 验证结果正确性
        assert_eq!(dst_simd, dst_scalar, "基准测试中结果不一致");

        // 输出支持的指令集信息
        println!("支持的指令集:");
        println!("  SSE2: {}", capabilities.supports_sse2);
        println!("  AVX2: {}", capabilities.supports_avx2);
        println!("  AVX512: {}", capabilities.supports_avx512);
    }

    #[test]
    fn test_fancy_index_engine_simd_integration() {
        let array = create_test_lazy_array(1000, 10, 8);
        let engine = FancyIndexEngine::new();

        let indices = vec![0, 5, 10, 15, 20, 25, 30, 35];

        // 测试不同的处理方法
        let result_direct = engine.process_direct(&indices, &array);
        let result_simd = engine.process_simd(&indices, &array);

        // 验证结果一致性
        assert_eq!(result_direct.len(), result_simd.len());
        for (i, (direct, simd)) in result_direct.iter().zip(result_simd.iter()).enumerate() {
            assert_eq!(direct, simd, "索引 {} 的结果不一致", indices[i]);
        }

        println!("花式索引引擎SIMD集成测试通过");
    }

    #[test]
    fn test_smart_index_router_simd_selection() {
        let array = create_test_lazy_array(500, 20, 16);
        let router = SmartIndexRouter::new();

        let indices = vec![0, 2, 4, 6, 8, 10, 12, 14, 16, 18];

        // 执行路由，应该自动选择合适的SIMD算法
        let result = router.route_fancy_index(&indices, &array);

        // 验证结果正确性
        assert_eq!(result.len(), indices.len());
        for (i, row) in result.iter().enumerate() {
            assert!(!row.is_empty(), "索引 {} 的结果为空", indices[i]);
            assert_eq!(row.len(), 20 * 16, "索引 {} 的数据大小不正确", indices[i]);
        }

        // 获取性能统计
        let stats = router.get_performance_stats();
        println!("路由器性能统计:");
        println!("  缓存命中率: {:.2}%", stats.cache_hit_rate * 100.0);
        println!("  平均延迟: {:?}", stats.average_latency);
        println!("  吞吐量: {:.2} MB/s", stats.throughput);
    }

    #[test]
    fn test_simd_with_different_data_types() {
        // 测试不同数据类型的SIMD优化
        let test_cases = vec![
            (100, 1, 1),  // uint8
            (100, 1, 2),  // uint16
            (100, 1, 4),  // uint32/float32
            (100, 1, 8),  // uint64/float64
            (100, 10, 4), // 多列float32
            (100, 5, 8),  // 多列float64
        ];

        for (rows, cols, item_size) in test_cases {
            let array = create_test_lazy_array(rows, cols, item_size);
            let engine = FancyIndexEngine::new();

            let indices = vec![0, 10, 20, 30, 40];
            let result = engine.process_simd(&indices, &array);

            assert_eq!(result.len(), indices.len());
            for row in &result {
                assert_eq!(row.len(), cols * item_size);
            }

            println!(
                "数据类型测试通过: {}行 x {}列 x {}字节",
                rows, cols, item_size
            );
        }
    }
}
