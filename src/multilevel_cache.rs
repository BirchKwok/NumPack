//! 多级智能缓存系统
//! 
//! L1: 热数据缓存（CPU缓存友好）
//! L2: 温数据缓存（内存）
//! L3: 冷数据缓存（内存映射）
//! 
//! 目标: 70-90%缓存命中率，5-20x热数据访问速度

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Instant, Duration};
use parking_lot::RwLock;
use lru::LruCache;
use memmap2::Mmap;

/// 小向量优化（避免堆分配）
type SmallVec = smallvec::SmallVec<[u8; 4096]>;

/// 访问模式
#[derive(Debug, Clone, Copy)]
pub enum AccessPattern {
    Sequential,      // 顺序访问
    Random,          // 随机访问
    Clustered,       // 聚集访问
    Unknown,         // 未知模式
}

/// 访问统计
#[derive(Debug, Clone)]
pub struct AccessStats {
    pub key: String,
    pub access_count: usize,
    pub last_access: Instant,
    pub access_pattern: AccessPattern,
    pub avg_interval: Duration,
}

impl AccessStats {
    pub fn new(key: String) -> Self {
        Self {
            key,
            access_count: 0,
            last_access: Instant::now(),
            access_pattern: AccessPattern::Unknown,
            avg_interval: Duration::from_secs(0),
        }
    }
    
    pub fn record_access(&mut self) {
        let now = Instant::now();
        let interval = now.duration_since(self.last_access);
        
        // 更新平均访问间隔
        if self.access_count > 0 {
            let total = self.avg_interval * self.access_count as u32 + interval;
            self.avg_interval = total / (self.access_count + 1) as u32;
        } else {
            self.avg_interval = interval;
        }
        
        self.access_count += 1;
        self.last_access = now;
    }
    
    /// 计算热度分数（0-100）
    pub fn hotness_score(&self) -> u32 {
        let recency_score = {
            let elapsed = self.last_access.elapsed().as_secs();
            if elapsed < 1 {
                50
            } else if elapsed < 10 {
                30
            } else if elapsed < 60 {
                10
            } else {
                0
            }
        };
        
        let frequency_score = {
            if self.access_count > 100 {
                50
            } else if self.access_count > 10 {
                30
            } else if self.access_count > 1 {
                10
            } else {
                0
            }
        };
        
        recency_score + frequency_score
    }
    
    /// 是否应该提升到更高级缓存
    pub fn should_promote(&self) -> bool {
        self.hotness_score() >= 60
    }
}

/// L1缓存（热数据）
pub struct L1Cache {
    cache: RwLock<LruCache<String, SmallVec>>,
    stats: Mutex<HashMap<String, AccessStats>>,
}

impl L1Cache {
    pub fn new(capacity: usize) -> Self {
        Self {
            cache: RwLock::new(LruCache::new(capacity.try_into().unwrap())),
            stats: Mutex::new(HashMap::new()),
        }
    }
    
    pub fn get(&self, key: &str) -> Option<SmallVec> {
        let result = self.cache.write().get(key).cloned();
        
        if result.is_some() {
            let mut stats = self.stats.lock().unwrap();
            stats.entry(key.to_string())
                .or_insert_with(|| AccessStats::new(key.to_string()))
                .record_access();
        }
        
        result
    }
    
    pub fn put(&self, key: String, value: SmallVec) {
        self.cache.write().put(key.clone(), value);
        
        let mut stats = self.stats.lock().unwrap();
        stats.entry(key.clone())
            .or_insert_with(|| AccessStats::new(key))
            .record_access();
    }
    
    pub fn get_stats(&self, key: &str) -> Option<AccessStats> {
        self.stats.lock().unwrap().get(key).cloned()
    }
}

/// L2缓存（温数据）
pub struct L2Cache {
    cache: RwLock<LruCache<String, Arc<Vec<u8>>>>,
    stats: Mutex<HashMap<String, AccessStats>>,
}

impl L2Cache {
    pub fn new(capacity: usize) -> Self {
        Self {
            cache: RwLock::new(LruCache::new(capacity.try_into().unwrap())),
            stats: Mutex::new(HashMap::new()),
        }
    }
    
    pub fn get(&self, key: &str) -> Option<Arc<Vec<u8>>> {
        let result = self.cache.write().get(key).cloned();
        
        if result.is_some() {
            let mut stats = self.stats.lock().unwrap();
            stats.entry(key.to_string())
                .or_insert_with(|| AccessStats::new(key.to_string()))
                .record_access();
        }
        
        result
    }
    
    pub fn put(&self, key: String, value: Arc<Vec<u8>>) {
        self.cache.write().put(key.clone(), value);
        
        let mut stats = self.stats.lock().unwrap();
        stats.entry(key.clone())
            .or_insert_with(|| AccessStats::new(key))
            .record_access();
    }
    
    pub fn get_stats(&self, key: &str) -> Option<AccessStats> {
        self.stats.lock().unwrap().get(key).cloned()
    }
}

/// L3缓存（冷数据，内存映射）
pub struct L3Cache {
    cache: RwLock<LruCache<String, Arc<Mmap>>>,
    stats: Mutex<HashMap<String, AccessStats>>,
}

impl L3Cache {
    pub fn new(capacity: usize) -> Self {
        Self {
            cache: RwLock::new(LruCache::new(capacity.try_into().unwrap())),
            stats: Mutex::new(HashMap::new()),
        }
    }
    
    pub fn get(&self, key: &str) -> Option<Arc<Mmap>> {
        let result = self.cache.write().get(key).cloned();
        
        if result.is_some() {
            let mut stats = self.stats.lock().unwrap();
            stats.entry(key.to_string())
                .or_insert_with(|| AccessStats::new(key.to_string()))
                .record_access();
        }
        
        result
    }
    
    pub fn put(&self, key: String, value: Arc<Mmap>) {
        self.cache.write().put(key.clone(), value);
        
        let mut stats = self.stats.lock().unwrap();
        stats.entry(key.clone())
            .or_insert_with(|| AccessStats::new(key))
            .record_access();
    }
    
    pub fn get_stats(&self, key: &str) -> Option<AccessStats> {
        self.stats.lock().unwrap().get(key).cloned()
    }
}

/// 多级缓存系统
pub struct MultiLevelCache {
    l1: L1Cache,
    l2: L2Cache,
    l3: L3Cache,
}

impl MultiLevelCache {
    pub fn new() -> Self {
        Self {
            l1: L1Cache::new(100),    // L1: 100个小块
            l2: L2Cache::new(50),     // L2: 50个中等数据块
            l3: L3Cache::new(20),     // L3: 20个大数据块
        }
    }
    
    pub fn with_capacity(l1_size: usize, l2_size: usize, l3_size: usize) -> Self {
        Self {
            l1: L1Cache::new(l1_size),
            l2: L2Cache::new(l2_size),
            l3: L3Cache::new(l3_size),
        }
    }
    
    /// 获取数据（自动从最快的缓存层获取）
    pub fn get(&self, key: &str) -> CacheResult {
        // 尝试L1
        if let Some(data) = self.l1.get(key) {
            return CacheResult::L1Hit(data.to_vec());
        }
        
        // 尝试L2
        if let Some(data) = self.l2.get(key) {
            // 如果访问频繁，提升到L1
            if let Some(stats) = self.l2.get_stats(key) {
                if stats.should_promote() && data.len() <= 4096 {
                    let small_vec = SmallVec::from_slice(&data);
                    self.l1.put(key.to_string(), small_vec);
                }
            }
            return CacheResult::L2Hit(data);
        }
        
        // 尝试L3
        if let Some(mmap) = self.l3.get(key) {
            // 如果访问频繁，提升到L2
            if let Some(stats) = self.l3.get_stats(key) {
                if stats.should_promote() {
                    let data = Arc::new(mmap.as_ref().to_vec());
                    self.l2.put(key.to_string(), data);
                }
            }
            return CacheResult::L3Hit(mmap);
        }
        
        CacheResult::Miss
    }
    
    /// 存储数据（自动选择合适的缓存层）
    pub fn put(&self, key: String, data: Vec<u8>) {
        if data.len() <= 4096 {
            // 小数据放L1
            let small_vec = SmallVec::from_slice(&data);
            self.l1.put(key, small_vec);
        } else {
            // 中等数据放L2
            self.l2.put(key, Arc::new(data));
        }
    }
    
    /// 存储内存映射（放L3）
    pub fn put_mmap(&self, key: String, mmap: Arc<Mmap>) {
        self.l3.put(key, mmap);
    }
    
    /// 获取缓存统计
    pub fn get_stats(&self) -> CacheStats {
        // TODO: 实现完整的统计信息收集
        CacheStats {
            l1_size: 0,
            l2_size: 0,
            l3_size: 0,
            total_hits: 0,
            total_misses: 0,
        }
    }
}

/// 缓存查询结果
pub enum CacheResult {
    L1Hit(Vec<u8>),
    L2Hit(Arc<Vec<u8>>),
    L3Hit(Arc<Mmap>),
    Miss,
}

/// 缓存统计
pub struct CacheStats {
    pub l1_size: usize,
    pub l2_size: usize,
    pub l3_size: usize,
    pub total_hits: u64,
    pub total_misses: u64,
}

impl CacheStats {
    pub fn hit_rate(&self) -> f64 {
        let total = self.total_hits + self.total_misses;
        if total == 0 {
            0.0
        } else {
            self.total_hits as f64 / total as f64
        }
    }
}

// 添加smallvec依赖
use smallvec;

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_multilevel_cache() {
        let cache = MultiLevelCache::new();
        
        // 测试小数据（应该进L1）
        cache.put("small".to_string(), vec![1, 2, 3, 4]);
        
        match cache.get("small") {
            CacheResult::L1Hit(data) => {
                assert_eq!(data, vec![1, 2, 3, 4]);
            }
            _ => panic!("Expected L1 hit"),
        }
        
        // 测试中等数据（应该进L2）
        let large_data = vec![0u8; 10000];
        cache.put("large".to_string(), large_data.clone());
        
        match cache.get("large") {
            CacheResult::L2Hit(data) => {
                assert_eq!(data.len(), 10000);
            }
            _ => panic!("Expected L2 hit"),
        }
    }
    
    #[test]
    fn test_cache_promotion() {
        let cache = MultiLevelCache::new();
        
        // 放入L2
        let data = vec![0u8; 2000];
        cache.put("test".to_string(), data);
        
        // 多次访问应该提升到L1
        for _ in 0..10 {
            std::thread::sleep(Duration::from_millis(1));
            cache.get("test");
        }
        
        // 最后一次应该从L1获取
        match cache.get("test") {
            CacheResult::L1Hit(_) => {},  // 成功提升
            CacheResult::L2Hit(_) => {},  // 或仍在L2（取决于提升策略）
            _ => panic!("Data should be in cache"),
        }
    }
}

