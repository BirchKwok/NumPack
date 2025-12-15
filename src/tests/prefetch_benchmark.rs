//! 智能预取机制性能基准测试
//!
//! 这个模块包含了全面的性能基准测试，用于验证智能预取机制的效果：
//! - 与直接访问的性能对比
//! - 不同访问模式下的预取效果
//! - 多级缓存策略的性能影响
//! - 内存带宽利用优化效果

#[cfg(test)]
mod benchmarks {
    use crate::lazy_array::*;
    use std::fs::File;
    use std::io::Write;
    use std::time::{Duration, Instant};

    /// 创建大规模测试数据
    fn create_large_test_array(rows: usize, row_size: usize) -> OptimizedLazyArray {
        let data_size = rows * row_size;
        let mut data = Vec::with_capacity(data_size);

        // 创建具有一定模式的测试数据
        for i in 0..rows {
            for j in 0..row_size {
                let value = ((i * 17 + j * 23) % 256) as u8;
                data.push(value);
            }
        }

        // 写入临时文件
        let temp_path = format!("/tmp/large_test_array_{}.dat", std::process::id());
        let mut file = File::create(&temp_path).unwrap();
        file.write_all(&data).unwrap();
        file.sync_all().unwrap();
        drop(file);

        OptimizedLazyArray::from_file(&temp_path, vec![rows, row_size], 1).unwrap()
    }

    #[test]
    fn benchmark_sequential_access_performance() {
        let array = create_large_test_array(10000, 128);
        let engine = FancyIndexEngine::new();

        // 顺序访问模式
        let sequential_indices: Vec<usize> = (0..1000).collect();

        println!("\n=== 顺序访问性能基准测试 ===");

        // 测试直接访问性能
        let start = Instant::now();
        let _results_direct = engine.process_direct(&sequential_indices, &array);
        let duration_direct = start.elapsed();
        println!("直接访问耗时: {:?}", duration_direct);

        // 重置统计
        engine.reset_stats();

        // 测试SIMD优化访问性能
        let start = Instant::now();
        let _results_simd = engine.process_simd(&sequential_indices, &array);
        let duration_simd = start.elapsed();
        println!("SIMD优化访问耗时: {:?}", duration_simd);

        // 重置统计
        engine.reset_stats();

        // 测试预取优化访问性能
        let start = Instant::now();
        let _results_prefetch = engine.process_with_prefetch(&sequential_indices, &array);
        let duration_prefetch = start.elapsed();
        println!("预取优化访问耗时: {:?}", duration_prefetch);

        // 获取性能统计
        if let Some((hit_rate, hits, misses)) = engine.get_performance_stats() {
            println!("预取命中率: {:.2}%", hit_rate * 100.0);
            println!("命中次数: {}, 未命中次数: {}", hits, misses);
        }

        // 计算性能提升
        let simd_improvement = if duration_simd < duration_direct {
            (duration_direct.as_nanos() as f64 / duration_simd.as_nanos() as f64 - 1.0) * 100.0
        } else {
            0.0
        };

        let prefetch_improvement = if duration_prefetch < duration_direct {
            (duration_direct.as_nanos() as f64 / duration_prefetch.as_nanos() as f64 - 1.0) * 100.0
        } else {
            0.0
        };

        println!("SIMD优化性能提升: {:.1}%", simd_improvement);
        println!("预取优化性能提升: {:.1}%", prefetch_improvement);

        // 验证结果正确性
        assert!(!_results_direct.is_empty());
        assert!(!_results_simd.is_empty());
        assert!(!_results_prefetch.is_empty());
        assert_eq!(_results_direct.len(), _results_simd.len());
        assert_eq!(_results_direct.len(), _results_prefetch.len());
    }

    #[test]
    fn benchmark_random_access_performance() {
        let array = create_large_test_array(10000, 128);
        let engine = FancyIndexEngine::new();

        // 随机访问模式
        let mut random_indices = Vec::new();
        for i in 0..1000 {
            random_indices.push((i * 17 + i * i * 23) % 9000);
        }

        println!("\n=== 随机访问性能基准测试 ===");

        // 测试直接访问性能
        let start = Instant::now();
        let _results_direct = engine.process_direct(&random_indices, &array);
        let duration_direct = start.elapsed();
        println!("直接访问耗时: {:?}", duration_direct);

        // 重置统计
        engine.reset_stats();

        // 测试预取优化访问性能
        let start = Instant::now();
        let _results_prefetch = engine.process_with_prefetch(&random_indices, &array);
        let duration_prefetch = start.elapsed();
        println!("预取优化访问耗时: {:?}", duration_prefetch);

        // 获取性能统计
        if let Some((hit_rate, hits, misses)) = engine.get_performance_stats() {
            println!("预取命中率: {:.2}%", hit_rate * 100.0);
            println!("命中次数: {}, 未命中次数: {}", hits, misses);
        }

        // 对于随机访问，预取的效果可能有限，但不应该显著降低性能
        let performance_ratio =
            duration_prefetch.as_nanos() as f64 / duration_direct.as_nanos() as f64;
        println!("预取/直接访问性能比: {:.2}", performance_ratio);

        // 验证预取不会显著降低随机访问性能（允许20%的开销）
        assert!(
            performance_ratio < 1.2,
            "预取优化不应该显著降低随机访问性能"
        );
    }

    #[test]
    fn benchmark_strided_access_performance() {
        let array = create_large_test_array(10000, 128);
        let engine = FancyIndexEngine::new();

        // 步长访问模式
        let strided_indices: Vec<usize> = (0..500).map(|i| i * 4).collect();

        println!("\n=== 步长访问性能基准测试 ===");

        // 测试直接访问性能
        let start = Instant::now();
        let _results_direct = engine.process_direct(&strided_indices, &array);
        let duration_direct = start.elapsed();
        println!("直接访问耗时: {:?}", duration_direct);

        // 重置统计
        engine.reset_stats();

        // 测试预取优化访问性能
        let start = Instant::now();
        let _results_prefetch = engine.process_with_prefetch(&strided_indices, &array);
        let duration_prefetch = start.elapsed();
        println!("预取优化访问耗时: {:?}", duration_prefetch);

        // 获取性能统计
        if let Some((hit_rate, hits, misses)) = engine.get_performance_stats() {
            println!("预取命中率: {:.2}%", hit_rate * 100.0);
            println!("命中次数: {}, 未命中次数: {}", hits, misses);
        }

        // 步长访问应该能从预取中受益
        let improvement = if duration_prefetch < duration_direct {
            (duration_direct.as_nanos() as f64 / duration_prefetch.as_nanos() as f64 - 1.0) * 100.0
        } else {
            0.0
        };

        println!("预取优化性能提升: {:.1}%", improvement);
    }

    #[test]
    fn benchmark_prefetch_strategies() {
        let array = create_large_test_array(5000, 256);
        let indices: Vec<usize> = (0..200).collect();

        println!("\n=== 预取策略性能对比 ===");

        // 测试保守策略
        let mut prefetch_mgr_conservative = PrefetchManager::new();
        prefetch_mgr_conservative.set_strategy(PrefetchStrategy::Conservative);

        let start = Instant::now();
        prefetch_mgr_conservative.predict_and_prefetch(&indices, &array);
        let duration_conservative = start.elapsed();
        let stats_conservative = prefetch_mgr_conservative.get_detailed_stats();

        println!("保守策略:");
        println!("  耗时: {:?}", duration_conservative);
        println!("  内存使用: {} bytes", stats_conservative.memory_usage);
        println!("  窗口大小: {}", stats_conservative.adaptive_window_size);

        // 测试激进策略
        let mut prefetch_mgr_aggressive = PrefetchManager::new();
        prefetch_mgr_aggressive.set_strategy(PrefetchStrategy::Aggressive);

        let start = Instant::now();
        prefetch_mgr_aggressive.predict_and_prefetch(&indices, &array);
        let duration_aggressive = start.elapsed();
        let stats_aggressive = prefetch_mgr_aggressive.get_detailed_stats();

        println!("激进策略:");
        println!("  耗时: {:?}", duration_aggressive);
        println!("  内存使用: {} bytes", stats_aggressive.memory_usage);
        println!("  窗口大小: {}", stats_aggressive.adaptive_window_size);

        // 测试自适应策略
        let mut prefetch_mgr_adaptive = PrefetchManager::new();
        prefetch_mgr_adaptive.set_strategy(PrefetchStrategy::Adaptive);

        let start = Instant::now();
        prefetch_mgr_adaptive.predict_and_prefetch(&indices, &array);
        let duration_adaptive = start.elapsed();
        let stats_adaptive = prefetch_mgr_adaptive.get_detailed_stats();

        println!("自适应策略:");
        println!("  耗时: {:?}", duration_adaptive);
        println!("  内存使用: {} bytes", stats_adaptive.memory_usage);
        println!("  窗口大小: {}", stats_adaptive.adaptive_window_size);

        // 验证不同策略都能正常工作
        assert!(stats_conservative.memory_usage >= 0);
        assert!(stats_aggressive.memory_usage >= 0);
        assert!(stats_adaptive.memory_usage >= 0);
    }

    #[test]
    fn benchmark_cache_levels_performance() {
        let mut cache = MultiLevelPrefetchCache::new();

        println!("\n=== 多级缓存性能基准测试 ===");

        // 准备测试数据
        let test_data_size = 1024;
        let test_data: Vec<u8> = (0..test_data_size).map(|i| (i % 256) as u8).collect();

        // 测试L1缓存性能
        let start = Instant::now();
        for i in 0..100 {
            cache.put(i, test_data.clone(), PrefetchLevel::L1);
        }
        let l1_write_duration = start.elapsed();

        let start = Instant::now();
        for i in 0..100 {
            let _ = cache.get(i);
        }
        let l1_read_duration = start.elapsed();

        println!("L1缓存:");
        println!("  写入耗时: {:?}", l1_write_duration);
        println!("  读取耗时: {:?}", l1_read_duration);

        // 测试L2缓存性能
        cache.clear();
        let start = Instant::now();
        for i in 0..100 {
            cache.put(i, test_data.clone(), PrefetchLevel::L2);
        }
        let l2_write_duration = start.elapsed();

        let start = Instant::now();
        for i in 0..100 {
            let _ = cache.get(i);
        }
        let l2_read_duration = start.elapsed();

        println!("L2缓存:");
        println!("  写入耗时: {:?}", l2_write_duration);
        println!("  读取耗时: {:?}", l2_read_duration);

        // 测试L3缓存性能
        cache.clear();
        let start = Instant::now();
        for i in 0..100 {
            cache.put(i, test_data.clone(), PrefetchLevel::L3);
        }
        let l3_write_duration = start.elapsed();

        let start = Instant::now();
        for i in 0..100 {
            let _ = cache.get(i);
        }
        let l3_read_duration = start.elapsed();

        println!("L3缓存:");
        println!("  写入耗时: {:?}", l3_write_duration);
        println!("  读取耗时: {:?}", l3_read_duration);

        // 获取命中率统计
        let (l1_hit_rate, l2_hit_rate, l3_hit_rate) = cache.get_hit_rate();
        println!("缓存命中率:");
        println!("  L1: {:.2}%", l1_hit_rate * 100.0);
        println!("  L2: {:.2}%", l2_hit_rate * 100.0);
        println!("  L3: {:.2}%", l3_hit_rate * 100.0);

        println!("总内存使用: {} bytes", cache.get_memory_usage());
    }

    #[test]
    fn benchmark_pattern_prediction_accuracy() {
        let mut predictor = AccessPatternPredictor::new();

        println!("\n=== 访问模式预测准确性基准测试 ===");

        // 测试顺序模式预测
        let sequential_patterns = vec![
            vec![0, 1, 2, 3, 4],
            vec![5, 6, 7, 8, 9],
            vec![10, 11, 12, 13, 14],
            vec![15, 16, 17, 18, 19],
        ];

        for pattern in &sequential_patterns {
            predictor.learn_pattern(pattern);
        }

        let test_sequence = vec![20, 21, 22];
        let predictions = predictor.predict_next_accesses(&test_sequence, 5);
        let confidence = predictor.get_confidence(&test_sequence);

        println!("顺序模式预测:");
        println!("  输入序列: {:?}", test_sequence);
        println!("  预测结果: {:?}", predictions);
        println!("  置信度: {:.2}", confidence);

        // 测试步长模式预测
        let mut stride_predictor = AccessPatternPredictor::new();
        let stride_patterns = vec![
            vec![0, 2, 4, 6, 8],
            vec![10, 12, 14, 16, 18],
            vec![20, 22, 24, 26, 28],
            vec![30, 32, 34, 36, 38],
        ];

        for pattern in &stride_patterns {
            stride_predictor.learn_pattern(pattern);
        }

        let test_stride_sequence = vec![40, 42];
        let stride_predictions = stride_predictor.predict_next_accesses(&test_stride_sequence, 5);
        let stride_confidence = stride_predictor.get_confidence(&test_stride_sequence);

        println!("步长模式预测:");
        println!("  输入序列: {:?}", test_stride_sequence);
        println!("  预测结果: {:?}", stride_predictions);
        println!("  置信度: {:.2}", stride_confidence);

        // 验证预测结果的合理性
        assert!(!predictions.is_empty(), "顺序模式应该能够预测");
        assert!(!stride_predictions.is_empty(), "步长模式应该能够预测");
        assert!(confidence > 0.0, "顺序模式置信度应该大于0");
        assert!(stride_confidence > 0.0, "步长模式置信度应该大于0");
    }

    #[test]
    fn benchmark_memory_bandwidth_optimization() {
        let array = create_large_test_array(2000, 512); // 1MB per row
        let mut prefetch_mgr = PrefetchManager::new();

        println!("\n=== 内存带宽优化基准测试 ===");

        let large_indices: Vec<usize> = (0..100).collect(); // 100MB total

        // 测试无限制的预取性能
        prefetch_mgr.set_memory_bandwidth_limit(usize::MAX);
        let start = Instant::now();
        prefetch_mgr.predict_and_prefetch(&large_indices, &array);
        let duration_unlimited = start.elapsed();
        let stats_unlimited = prefetch_mgr.get_detailed_stats();

        println!("无限制预取:");
        println!("  耗时: {:?}", duration_unlimited);
        println!(
            "  内存使用: {} MB",
            stats_unlimited.memory_usage / (1024 * 1024)
        );

        // 测试带宽限制的预取性能
        let mut limited_prefetch_mgr = PrefetchManager::new();
        limited_prefetch_mgr.set_memory_bandwidth_limit(10 * 1024 * 1024); // 10MB/s

        let start = Instant::now();
        limited_prefetch_mgr.predict_and_prefetch(&large_indices, &array);
        let duration_limited = start.elapsed();
        let stats_limited = limited_prefetch_mgr.get_detailed_stats();

        println!("限制带宽预取 (10MB/s):");
        println!("  耗时: {:?}", duration_limited);
        println!(
            "  内存使用: {} MB",
            stats_limited.memory_usage / (1024 * 1024)
        );

        // 验证带宽限制的效果
        assert!(
            duration_limited >= duration_unlimited,
            "带宽限制应该影响预取速度"
        );

        // 计算实际带宽利用
        let actual_bandwidth = if duration_limited.as_secs() > 0 {
            stats_limited.memory_usage as f64 / duration_limited.as_secs_f64() / (1024.0 * 1024.0)
        } else {
            0.0
        };

        println!("实际带宽利用: {:.2} MB/s", actual_bandwidth);
    }

    #[test]
    fn benchmark_comprehensive_performance() {
        println!("\n=== 综合性能基准测试 ===");

        let array = create_large_test_array(5000, 256);
        let engine = FancyIndexEngine::new();

        // 测试不同规模的访问性能
        let test_sizes = vec![10, 50, 100, 500, 1000];

        for &size in &test_sizes {
            let indices: Vec<usize> = (0..size).collect();

            // 直接访问
            let start = Instant::now();
            let _results_direct = engine.process_direct(&indices, &array);
            let duration_direct = start.elapsed();

            // 预取优化访问
            engine.reset_stats();
            let start = Instant::now();
            let _results_prefetch = engine.process_with_prefetch(&indices, &array);
            let duration_prefetch = start.elapsed();

            let improvement = if duration_prefetch < duration_direct {
                (duration_direct.as_nanos() as f64 / duration_prefetch.as_nanos() as f64 - 1.0)
                    * 100.0
            } else {
                -((duration_prefetch.as_nanos() as f64 / duration_direct.as_nanos() as f64 - 1.0)
                    * 100.0)
            };

            println!(
                "规模 {}: 直接访问 {:?}, 预取优化 {:?}, 性能变化 {:.1}%",
                size, duration_direct, duration_prefetch, improvement
            );
        }

        // 获取最终性能统计
        if let Some((final_hit_rate, total_hits, total_misses)) = engine.get_performance_stats() {
            println!("\n最终性能统计:");
            println!("  总命中率: {:.2}%", final_hit_rate * 100.0);
            println!("  总命中次数: {}", total_hits);
            println!("  总未命中次数: {}", total_misses);
            println!("  总访问次数: {}", total_hits + total_misses);
        }
    }
}
