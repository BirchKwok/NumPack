//! LRU缓存实现
//!
//! 提供Least Recently Used缓存机制，支持热点数据识别和访问频率统计

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

/// 缓存项元数据
#[derive(Debug, Clone)]
pub struct CacheItemMetadata {
    pub key: usize,
    pub size: usize,
    pub created_at: Instant,
    pub last_accessed: Instant,
    pub access_count: u64,
    pub access_frequency: f64, // 每分钟访问次数
    pub is_hot: bool,
    pub is_compressed: bool,
    pub promotion_count: u8, // 提升次数
}

impl CacheItemMetadata {
    pub fn new(key: usize, size: usize) -> Self {
        let now = Instant::now();
        Self {
            key,
            size,
            created_at: now,
            last_accessed: now,
            access_count: 1,
            access_frequency: 0.0,
            is_hot: false,
            is_compressed: false,
            promotion_count: 0,
        }
    }

    pub fn access(&mut self) {
        let now = Instant::now();
        self.access_count += 1;

        // 计算访问频率 (每分钟访问次数)
        let time_window = now.duration_since(self.created_at).as_secs_f64() / 60.0;
        self.access_frequency = if time_window > 0.0 {
            self.access_count as f64 / time_window
        } else {
            self.access_count as f64
        };

        self.last_accessed = now;

        // 热点数据判断
        if self.access_count > 10 || self.access_frequency > 5.0 {
            self.is_hot = true;
        }
    }

    pub fn age(&self) -> Duration {
        Instant::now().duration_since(self.created_at)
    }

    pub fn idle_time(&self) -> Duration {
        Instant::now().duration_since(self.last_accessed)
    }
}

/// LRU缓存实现 - L1层使用
#[derive(Debug)]
pub struct LRUCache {
    items: HashMap<usize, Vec<u8>>,
    metadata: HashMap<usize, CacheItemMetadata>,
    access_order: VecDeque<usize>,
    current_size: usize,
    max_size: usize,
}

impl LRUCache {
    pub fn new(max_size: usize) -> Self {
        Self {
            items: HashMap::new(),
            metadata: HashMap::new(),
            access_order: VecDeque::new(),
            current_size: 0,
            max_size,
        }
    }

    pub fn get(&mut self, key: usize) -> Option<Vec<u8>> {
        if self.items.contains_key(&key) {
            // 更新元数据
            if let Some(meta) = self.metadata.get_mut(&key) {
                meta.access();
            }

            // 移动到队列头部 (最近使用)
            self.move_to_front(key);

            // 获取数据
            self.items.get(&key).cloned()
        } else {
            None
        }
    }

    pub fn put(
        &mut self,
        key: usize,
        data: Vec<u8>,
    ) -> Option<(usize, Vec<u8>, CacheItemMetadata)> {
        let data_size = data.len();

        // 如果已存在，更新数据
        if self.items.contains_key(&key) {
            self.remove(key);
        }

        // 确保有足够空间
        let mut evicted_item = None;
        while self.current_size + data_size > self.max_size && !self.access_order.is_empty() {
            if let Some(evicted_key) = self.access_order.pop_back() {
                if let (Some(evicted_data), Some(evicted_meta)) = (
                    self.items.remove(&evicted_key),
                    self.metadata.remove(&evicted_key),
                ) {
                    self.current_size -= evicted_data.len();
                    evicted_item = Some((evicted_key, evicted_data, evicted_meta));
                    break; // 只返回一个被驱逐的项
                }
            }
        }

        // 添加新项
        if self.current_size + data_size <= self.max_size {
            self.items.insert(key, data);
            self.metadata
                .insert(key, CacheItemMetadata::new(key, data_size));
            self.access_order.push_front(key);
            self.current_size += data_size;
        }

        evicted_item
    }

    pub fn remove(&mut self, key: usize) -> Option<(Vec<u8>, CacheItemMetadata)> {
        if let (Some(data), Some(meta)) = (self.items.remove(&key), self.metadata.remove(&key)) {
            self.current_size -= data.len();
            self.access_order.retain(|&k| k != key);
            Some((data, meta))
        } else {
            None
        }
    }

    fn move_to_front(&mut self, key: usize) {
        self.access_order.retain(|&k| k != key);
        self.access_order.push_front(key);
    }

    pub fn clear(&mut self) {
        self.items.clear();
        self.metadata.clear();
        self.access_order.clear();
        self.current_size = 0;
    }

    pub fn get_metadata(&self, key: usize) -> Option<&CacheItemMetadata> {
        self.metadata.get(&key)
    }

    pub fn list_items_by_access_frequency(&self) -> Vec<(usize, f64)> {
        self.metadata
            .iter()
            .map(|(&key, meta)| (key, meta.access_frequency))
            .collect()
    }

    pub fn get_all_keys(&self) -> Vec<usize> {
        self.metadata.keys().copied().collect()
    }
}
