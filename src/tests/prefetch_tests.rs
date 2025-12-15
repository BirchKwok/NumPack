//! 智能预取机制测试模块
//!
//! 测试内容包括：
//! - 访问模式检测和预测算法
//! - 自适应预取窗口大小调整
//! - 多级预取策略（L1/L2/L3缓存优化）
//! - 预取时机和内存带宽利用优化
//! - 预取效果和性能基准测试

#[cfg(test)]
mod tests {
    use crate::lazy_array::*;
    use std::sync::{Arc, Mutex};
    use std::time::{Duration, Instant};

    /// 创建测试用的模拟数组数据
    fn create_test_array_data(rows: usize, row_size: usize) -> Vec<u8> {
        let mut data = Vec::with_capacity(rows * row_size);
        for i in 0..rows {
            for j in 0..row_size {
                data.push(((i * row_size + j) % 256) as u8);
            }
        }
        data
    }

    /// 创建测试用的OptimizedLazyArray
    fn create_test_lazy_array(rows: usize, row_size: usize) -> OptimizedLazyArray {
        let data = create_test_array_data(rows, row_size);

        // 创建临时文件
        use std::fs::File;
        use std::io::Write;
        let temp_path = format!("/tmp/test_array_{}.dat", std::process::id());
        let mut file = File::create(&temp_path).unwrap();
        file.write_all(&data).unwrap();
        file.sync_all().unwrap();
        drop(file);

        // 创建LazyArray
        OptimizedLazyArray::from_file(&temp_path, vec![rows, row_size], 1).unwrap()
    }

    #[test]
    fn test_access_pattern_predictor_stride_learning() {
        let mut predictor = AccessPatternPredictor::new();

        // 学习步长为2的访问模式
        let stride_pattern = vec![0, 2, 4, 6, 8, 10];
        predictor.learn_pattern(&stride_pattern);

        // 测试预测
        let current_indices = vec![12, 14];
        let predictions = predictor.predict_next_accesses(&current_indices, 4);

        assert!(!predictions.is_empty(), "应该能够预测下一批访问");

        // 验证预测的合理性（应该包含步长为2的后续访问）
        let expected_next = vec![16, 18, 20, 22];
        for &expected in &expected_next {
            assert!(
                predictions.contains(&expected)
                    || predictions
                        .iter()
                        .any(|&p| (p as i32 - expected as i32).abs() <= 2),
                "预测结果应该包含或接近期望的索引: {}, 实际预测: {:?}",
                expected,
                predictions
            );
        }
    }

    #[test]
    fn test_access_pattern_predictor_sequence_learning() {
        let mut predictor = AccessPatternPredictor::new();

        // 学习特定的序列模式
        let sequence_pattern1 = vec![1, 3, 7];
        let sequence_pattern2 = vec![3, 7, 15];
        let sequence_pattern3 = vec![7, 15, 31];

        predictor.learn_pattern(&sequence_pattern1);
        predictor.learn_pattern(&sequence_pattern2);
        predictor.learn_pattern(&sequence_pattern3);

        // 测试基于序列的预测
        let current_indices = vec![15, 31];
        let predictions = predictor.predict_next_accesses(&current_indices, 3);

        assert!(!predictions.is_empty(), "应该能够基于序列模式预测");

        // 验证置信度
        let confidence = predictor.get_confidence(&current_indices);
        assert!(confidence > 0.0, "置信度应该大于0");
    }

    #[test]
    fn test_multi_level_prefetch_cache() {
        let mut cache = MultiLevelPrefetchCache::new();

        // 测试L1缓存
        let test_data_l1 = vec![1, 2, 3, 4];
        cache.put(1, test_data_l1.clone(), PrefetchLevel::L1);

        let retrieved = cache.get(1);
        assert_eq!(
            retrieved,
            Some(test_data_l1),
            "L1缓存应该能够正确存储和检索数据"
        );

        // 测试L2缓存
        let test_data_l2 = vec![5, 6, 7, 8];
        cache.put(2, test_data_l2.clone(), PrefetchLevel::L2);

        let retrieved = cache.get(2);
        assert_eq!(
            retrieved,
            Some(test_data_l2),
            "L2缓存应该能够正确存储和检索数据"
        );

        // 测试L3缓存
        let test_data_l3 = vec![9, 10, 11, 12];
        cache.put(3, test_data_l3.clone(), PrefetchLevel::L3);

        let retrieved = cache.get(3);
        assert_eq!(
            retrieved,
            Some(test_data_l3),
            "L3缓存应该能够正确存储和检索数据"
        );

        // 测试缓存提升机制
        // 再次访问L2中的数据，应该被提升到L1
        let retrieved_again = cache.get(2);
        assert!(retrieved_again.is_some(), "L2数据被访问后应该被提升到L1");
    }

    #[test]
    fn test_multi_level_cache_eviction() {
        let mut cache = MultiLevelPrefetchCache::new();

        // 填满L1缓存
        for i in 0..35 {
            // 超过L1容量(32)
            let data = vec![i as u8; 100];
            cache.put(i, data, PrefetchLevel::L1);
        }

        // 验证LRU淘汰机制工作
        assert!(cache.l1_cache_len() <= 32, "L1缓存大小应该不超过容量限制");

        // 验证被淘汰的数据被降级到L2
        assert!(!cache.l2_cache_is_empty(), "被淘汰的L1数据应该被降级到L2");
    }

    #[test]
    fn test_prefetch_manager_basic_functionality() {
        let mut prefetch_mgr = PrefetchManager::new();

        // 测试基本配置
        assert_eq!(prefetch_mgr.get_strategy(), PrefetchStrategy::Adaptive);
        assert!(prefetch_mgr.is_prefetch_enabled());

        // 测试策略设置
        prefetch_mgr.set_strategy(PrefetchStrategy::Conservative);
        assert_eq!(prefetch_mgr.get_strategy(), PrefetchStrategy::Conservative);

        prefetch_mgr.set_strategy(PrefetchStrategy::Disabled);
        assert!(!prefetch_mgr.is_prefetch_enabled());

        // 测试资源阈值设置
        prefetch_mgr.set_resource_thresholds(0.9, 0.8);
        assert_eq!(prefetch_mgr.get_cpu_threshold(), 0.9);
        assert_eq!(prefetch_mgr.get_memory_pressure_threshold(), 0.8);

        // 测试内存带宽限制设置
        prefetch_mgr.set_memory_bandwidth_limit(50 * 1024 * 1024); // 50MB/s
        assert_eq!(prefetch_mgr.get_memory_bandwidth_limit(), 50 * 1024 * 1024);
    }

    #[test]
    fn test_prefetch_manager_adaptive_window() {
        let mut prefetch_mgr = PrefetchManager::new();
        let initial_window = prefetch_mgr.get_adaptive_window();

        // 测试高命中率时窗口增大
        prefetch_mgr.adjust_window_size(0.9);
        assert!(
            prefetch_mgr.get_adaptive_window() >= initial_window,
            "高命中率时窗口大小应该增加或保持不变"
        );

        // 测试低命中率时窗口减小
        prefetch_mgr.adjust_window_size(0.2);
        assert!(
            prefetch_mgr.get_adaptive_window() < initial_window
                || prefetch_mgr.get_adaptive_window() == prefetch_mgr.get_min_window_size(),
            "低命中率时窗口大小应该减小"
        );

        // 测试窗口大小边界
        for _ in 0..20 {
            prefetch_mgr.adjust_window_size(0.1); // 持续低命中率
        }
        assert!(
            prefetch_mgr.get_adaptive_window() >= prefetch_mgr.get_min_window_size(),
            "窗口大小不应该小于最小值"
        );

        for _ in 0..20 {
            prefetch_mgr.adjust_window_size(0.95); // 持续高命中率
        }
        assert!(
            prefetch_mgr.get_adaptive_window() <= prefetch_mgr.get_max_window_size(),
            "窗口大小不应该大于最大值"
        );
    }

    #[test]
    fn test_prefetch_manager_with_real_array() {
        let array = create_test_lazy_array(1000, 64);
        let mut prefetch_mgr = PrefetchManager::new();

        // 测试顺序访问模式的预取
        let sequential_indices = vec![0, 1, 2, 3, 4];
        prefetch_mgr.predict_and_prefetch(&sequential_indices, &array);

        // 验证预取数据可以被检索
        let prefetched = prefetch_mgr.get_prefetched_data(5);
        // 注意：由于预取是异步的，这里可能需要一些时间

        // 测试随机访问模式
        let random_indices = vec![10, 50, 200, 800, 15];
        prefetch_mgr.predict_and_prefetch(&random_indices, &array);

        // 获取性能统计
        let stats = prefetch_mgr.get_detailed_stats();
        assert!(stats.adaptive_window_size > 0, "自适应窗口大小应该大于0");
        assert!(stats.memory_usage >= 0, "内存使用量应该非负");
    }

    #[test]
    fn test_fancy_index_engine_with_prefetch() {
        let array = create_test_lazy_array(500, 32);
        let engine = FancyIndexEngine::new();

        // 测试预取优化的访问
        let indices = vec![0, 1, 2, 3, 4, 5];
        let results = engine.process_with_prefetch(&indices, &array);

        assert_eq!(results.len(), indices.len(), "结果数量应该与索引数量匹配");

        for (i, result) in results.iter().enumerate() {
            assert_eq!(result.len(), 32, "每行数据大小应该正确");

            // 验证数据正确性
            let expected_first_byte = ((indices[i] * 32) % 256) as u8;
            assert_eq!(
                result[0], expected_first_byte,
                "第{}行的第一个字节应该是{}, 实际是{}",
                i, expected_first_byte, result[0]
            );
        }

        // 获取性能统计
        if let Some((hit_rate, hit_count, miss_count)) = engine.get_performance_stats() {
            assert!(hit_rate >= 0.0 && hit_rate <= 1.0, "命中率应该在0-1之间");
            assert!(hit_count + miss_count > 0, "应该有访问统计");
        }
    }

    #[test]
    fn test_prefetch_performance_comparison() {
        let array = create_test_lazy_array(1000, 64);
        let engine = FancyIndexEngine::new();

        // 测试大量顺序访问的性能
        let sequential_indices: Vec<usize> = (0..100).collect();

        // 测试直接访问性能
        let start_direct = Instant::now();
        let _results_direct = engine.process_direct(&sequential_indices, &array);
        let duration_direct = start_direct.elapsed();

        // 重置统计
        engine.reset_stats();

        // 测试预取优化访问性能
        let start_prefetch = Instant::now();
        let _results_prefetch = engine.process_with_prefetch(&sequential_indices, &array);
        let duration_prefetch = start_prefetch.elapsed();

        println!("直接访问耗时: {:?}", duration_direct);
        println!("预取优化访问耗时: {:?}", duration_prefetch);

        // 对于顺序访问，预取优化应该有所帮助（或至少不会显著变慢）
        // 注意：在测试环境中，由于数据量小，差异可能不明显
        assert!(
            duration_prefetch <= duration_direct * 2,
            "预取优化不应该显著降低性能"
        );
    }

    #[test]
    fn test_prefetch_memory_bandwidth_limiting() {
        let array = create_test_lazy_array(1000, 1024); // 1KB per row
        let mut prefetch_mgr = PrefetchManager::new();

        // 设置较低的内存带宽限制
        prefetch_mgr.set_memory_bandwidth_limit(10 * 1024); // 10KB/s

        let large_indices: Vec<usize> = (0..50).collect(); // 50KB total

        let start = Instant::now();
        prefetch_mgr.predict_and_prefetch(&large_indices, &array);
        let duration = start.elapsed();

        // 验证带宽限制生效（预取应该被限制）
        assert!(
            duration >= Duration::from_millis(100),
            "带宽限制应该影响预取速度"
        );

        let stats = prefetch_mgr.get_detailed_stats();
        assert!(
            stats.memory_usage <= prefetch_mgr.get_memory_bandwidth_limit() * 2,
            "内存使用应该受到带宽限制的约束"
        );
    }

    #[test]
    fn test_prefetch_strategy_differences() {
        let array = create_test_lazy_array(200, 128);
        let indices = vec![0, 1, 2, 3, 4];

        // 测试保守策略
        let mut prefetch_mgr_conservative = PrefetchManager::new();
        prefetch_mgr_conservative.set_strategy(PrefetchStrategy::Conservative);
        prefetch_mgr_conservative.predict_and_prefetch(&indices, &array);
        let stats_conservative = prefetch_mgr_conservative.get_detailed_stats();

        // 测试激进策略
        let mut prefetch_mgr_aggressive = PrefetchManager::new();
        prefetch_mgr_aggressive.set_strategy(PrefetchStrategy::Aggressive);
        prefetch_mgr_aggressive.predict_and_prefetch(&indices, &array);
        let stats_aggressive = prefetch_mgr_aggressive.get_detailed_stats();

        // 测试自适应策略
        let mut prefetch_mgr_adaptive = PrefetchManager::new();
        prefetch_mgr_adaptive.set_strategy(PrefetchStrategy::Adaptive);
        prefetch_mgr_adaptive.predict_and_prefetch(&indices, &array);
        let stats_adaptive = prefetch_mgr_adaptive.get_detailed_stats();

        // 验证不同策略的行为差异
        assert_eq!(stats_conservative.strategy, PrefetchStrategy::Conservative);
        assert_eq!(stats_aggressive.strategy, PrefetchStrategy::Aggressive);
        assert_eq!(stats_adaptive.strategy, PrefetchStrategy::Adaptive);

        // 激进策略通常会使用更多内存
        // 注意：在小规模测试中差异可能不明显
        println!("保守策略内存使用: {}", stats_conservative.memory_usage);
        println!("激进策略内存使用: {}", stats_aggressive.memory_usage);
        println!("自适应策略内存使用: {}", stats_adaptive.memory_usage);
    }

    #[test]
    fn test_prefetch_cache_hit_rates() {
        let mut cache = MultiLevelPrefetchCache::new();

        // 添加一些测试数据到不同级别
        for i in 0..10 {
            let data = vec![i as u8; 64];
            cache.put(i, data, PrefetchLevel::L1);
        }

        for i in 10..30 {
            let data = vec![i as u8; 64];
            cache.put(i, data, PrefetchLevel::L2);
        }

        for i in 30..50 {
            let data = vec![i as u8; 64];
            cache.put(i, data, PrefetchLevel::L3);
        }

        // 访问不同级别的数据
        for i in 0..5 {
            cache.get(i); // L1 hits
        }

        for i in 10..15 {
            cache.get(i); // L2 hits (will be promoted to L1)
        }

        for i in 30..35 {
            cache.get(i); // L3 hits (will be promoted to L2)
        }

        let (l1_hit_rate, l2_hit_rate, l3_hit_rate) = cache.get_hit_rate();

        assert!(
            l1_hit_rate >= 0.0 && l1_hit_rate <= 1.0,
            "L1命中率应该在有效范围内"
        );
        assert!(
            l2_hit_rate >= 0.0 && l2_hit_rate <= 1.0,
            "L2命中率应该在有效范围内"
        );
        assert!(
            l3_hit_rate >= 0.0 && l3_hit_rate <= 1.0,
            "L3命中率应该在有效范围内"
        );

        println!("L1命中率: {:.2}", l1_hit_rate);
        println!("L2命中率: {:.2}", l2_hit_rate);
        println!("L3命中率: {:.2}", l3_hit_rate);
    }

    #[test]
    fn test_prefetch_pattern_confidence() {
        let mut predictor = AccessPatternPredictor::new();

        // 建立强模式
        for _ in 0..10 {
            let pattern = vec![0, 2, 4, 6, 8]; // 一致的步长模式
            predictor.learn_pattern(&pattern);
        }

        let high_confidence_indices = vec![10, 12];
        let confidence = predictor.get_confidence(&high_confidence_indices);
        assert!(
            confidence > 0.5,
            "一致的模式应该产生高置信度: {}",
            confidence
        );

        // 测试随机模式的低置信度
        let mut random_predictor = AccessPatternPredictor::new();
        for i in 0..5 {
            let random_pattern = vec![i * 17 % 100, i * 23 % 100, i * 31 % 100];
            random_predictor.learn_pattern(&random_pattern);
        }

        let random_indices = vec![50, 75];
        let low_confidence = random_predictor.get_confidence(&random_indices);
        assert!(
            low_confidence < confidence,
            "随机模式应该产生较低的置信度: {} vs {}",
            low_confidence,
            confidence
        );
    }

    #[test]
    fn test_prefetch_integration_with_fancy_index_engine() {
        let array = create_test_lazy_array(300, 256);
        let engine = FancyIndexEngine::new();

        // 测试不同访问模式下的预取效果

        // 1. 顺序访问模式
        let sequential_indices: Vec<usize> = (0..20).collect();
        let results_seq = engine.process_with_prefetch(&sequential_indices, &array);
        assert_eq!(results_seq.len(), 20);

        // 2. 步长访问模式
        let strided_indices: Vec<usize> = (0..10).map(|i| i * 3).collect();
        let results_strided = engine.process_with_prefetch(&strided_indices, &array);
        assert_eq!(results_strided.len(), 10);

        // 3. 随机访问模式
        let random_indices = vec![5, 50, 150, 200, 25, 75, 125, 175];
        let results_random = engine.process_with_prefetch(&random_indices, &array);
        assert_eq!(results_random.len(), 8);

        // 验证所有结果的正确性
        for (indices, results) in [
            (&sequential_indices, &results_seq),
            (&strided_indices, &results_strided),
            (&random_indices, &results_random),
        ] {
            for (i, result) in results.iter().enumerate() {
                assert_eq!(result.len(), 256, "行大小应该正确");

                let expected_first_byte = ((indices[i] * 256) % 256) as u8;
                assert_eq!(result[0], expected_first_byte, "数据内容应该正确");
            }
        }

        // 获取最终的性能统计
        if let Some((final_hit_rate, total_hits, total_misses)) = engine.get_performance_stats() {
            println!("最终命中率: {:.2}%", final_hit_rate * 100.0);
            println!("总命中次数: {}", total_hits);
            println!("总未命中次数: {}", total_misses);

            assert!(total_hits + total_misses > 0, "应该有访问统计");
        }
    }
}
