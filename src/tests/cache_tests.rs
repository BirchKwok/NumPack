//! 缓存系统测试
//! 测试各种缓存组件的功能

#[cfg(test)]
mod tests {
    use crate::cache::{
        compressed_cache::CompressedCache, lru_cache::LRUCache,
        performance_monitor::CachePerformanceMonitor, smart_cache::SmartCache,
    };
    use std::num::NonZeroUsize;

    #[test]
    fn test_smart_cache_basic() {
        let cache = SmartCache::new();

        // 测试基本缓存操作
        let key = 1;
        let data = vec![1, 2, 3, 4, 5];

        // 缓存miss测试
        assert!(cache.get(key).is_none());

        // 缓存写入测试
        cache.put(key, data.clone());

        // 缓存hit测试
        if let Some(cached_data) = cache.get(key) {
            assert_eq!(cached_data, &data);
        } else {
            panic!("Cache should contain the data");
        }
    }

    #[test]
    fn test_lru_cache() {
        let mut cache = LRUCache::new(NonZeroUsize::new(2).unwrap());

        // 测试LRU逻辑
        cache.put(1, vec![1]);
        cache.put(2, vec![2]);
        cache.put(3, vec![3]); // 应该驱逐键1

        assert!(cache.get(&1).is_none());
        assert!(cache.get(&2).is_some());
        assert!(cache.get(&3).is_some());
    }

    #[test]
    fn test_compressed_cache() {
        let mut cache = CompressedCache::new(1024, 100); // 1KB阈值

        // 测试小数据（不压缩）
        let small_data = vec![1, 2, 3, 4];
        cache.put(1, small_data.clone());
        assert_eq!(cache.get(&1), Some(&small_data));

        // 测试大数据（应该压缩）
        let large_data = vec![1u8; 2048]; // 2KB数据
        cache.put(2, large_data.clone());
        assert_eq!(cache.get(&2), Some(&large_data));
    }

    #[test]
    fn test_performance_monitor() {
        let mut monitor = CachePerformanceMonitor::new();

        // 记录一些操作
        monitor.record_hit(1000); // 1微秒
        monitor.record_miss(5000); // 5微秒
        monitor.record_hit(2000); // 2微秒

        let stats = monitor.get_statistics();
        assert_eq!(stats.total_hits, 2);
        assert_eq!(stats.total_misses, 1);
        assert_eq!(stats.hit_rate, 2.0 / 3.0);

        // 验证平均延迟
        assert_eq!(stats.avg_hit_latency_nanos, 1500); // (1000 + 2000) / 2
        assert_eq!(stats.avg_miss_latency_nanos, 5000);
    }
}
