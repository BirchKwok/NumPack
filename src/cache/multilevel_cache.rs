//! 多级缓存系统
//!
//! 提供L1/L2/L3三级缓存架构，包括LRU、自适应和压缩缓存策略

use std::collections::{BTreeSet, HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

// 缓存项元数据
#[derive(Debug, Clone)]
pub struct CacheItemMetadata {
    pub key: usize,
    pub size: usize,
    pub last_access: Instant,
    pub access_count: usize,
    pub access_frequency: f64,
    pub creation_time: Instant,
    pub is_compressed: bool,
}

impl CacheItemMetadata {
    pub fn new(key: usize, size: usize) -> Self {
        let now = Instant::now();
        Self {
            key,
            size,
            last_access: now,
            access_count: 1,
            access_frequency: 1.0,
            creation_time: now,
            is_compressed: false,
        }
    }

    pub fn access(&mut self) {
        self.last_access = Instant::now();
        self.access_count += 1;

        // 更新访问频率（基于时间衰减）
        let age = self.creation_time.elapsed().as_secs_f64();
        if age > 0.0 {
            self.access_frequency = self.access_count as f64 / age;
        }
    }
}

// 缓存策略配置
#[derive(Debug, Clone)]
pub struct CachePolicy {
    pub l1_max_size: usize,
    pub l2_max_size: usize,
    pub l3_max_size: usize,
    pub compression_threshold: usize,
    pub l2_to_l1_threshold: f64,
    pub l3_to_l2_threshold: f64,
    pub eviction_batch_size: usize,
}

impl Default for CachePolicy {
    fn default() -> Self {
        Self {
            l1_max_size: 16 * 1024 * 1024,   // 16MB L1缓存
            l2_max_size: 64 * 1024 * 1024,   // 64MB L2缓存
            l3_max_size: 256 * 1024 * 1024,  // 256MB L3缓存
            compression_threshold: 8 * 1024, // 8KB以上压缩
            l2_to_l1_threshold: 5.0,         // 访问频率阈值
            l3_to_l2_threshold: 2.0,         // 访问频率阈值
            eviction_batch_size: 10,         // 批量驱逐大小
        }
    }
}

// LRU缓存实现
#[derive(Debug)]
pub struct LRUCache {
    items: HashMap<usize, Vec<u8>>,
    metadata: HashMap<usize, CacheItemMetadata>,
    access_order: VecDeque<usize>,
    max_size: usize,
    current_size: usize,
}

impl LRUCache {
    pub fn new(max_size: usize) -> Self {
        Self {
            items: HashMap::new(),
            metadata: HashMap::new(),
            access_order: VecDeque::new(),
            max_size,
            current_size: 0,
        }
    }

    pub fn get(&mut self, key: usize) -> Option<Vec<u8>> {
        if self.items.contains_key(&key) {
            // 更新元数据
            if let Some(meta) = self.metadata.get_mut(&key) {
                meta.access();
            }

            // 移动到前面
            self.move_to_front(key);

            // 现在获取数据
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
}

// 自适应缓存实现
#[derive(Debug)]
pub struct AdaptiveCache {
    items: HashMap<usize, Vec<u8>>,
    metadata: HashMap<usize, CacheItemMetadata>,
    frequency_buckets: HashMap<u8, BTreeSet<usize>>, // 频率分桶
    max_size: usize,
    current_size: usize,
    adaptation_window: Duration,
    last_adaptation: Instant,
}

impl AdaptiveCache {
    pub fn new(max_size: usize) -> Self {
        Self {
            items: HashMap::new(),
            metadata: HashMap::new(),
            frequency_buckets: HashMap::new(),
            max_size,
            current_size: 0,
            adaptation_window: Duration::from_secs(60),
            last_adaptation: Instant::now(),
        }
    }

    pub fn get(&mut self, key: usize) -> Option<Vec<u8>> {
        if self.items.contains_key(&key) {
            // 先获取旧的频率桶ID
            let old_bucket = if let Some(meta) = self.metadata.get(&key) {
                self.get_frequency_bucket(meta.access_frequency)
            } else {
                0
            };

            // 更新元数据
            let new_bucket = if let Some(meta) = self.metadata.get_mut(&key) {
                meta.access();
                let frequency = meta.access_frequency;
                self.get_frequency_bucket(frequency)
            } else {
                old_bucket
            };

            // 如果频率桶改变，移动到新桶
            if old_bucket != new_bucket {
                if let Some(bucket) = self.frequency_buckets.get_mut(&old_bucket) {
                    bucket.remove(&key);
                }
                self.frequency_buckets
                    .entry(new_bucket)
                    .or_insert_with(BTreeSet::new)
                    .insert(key);
            }

            // 获取数据
            self.items.get(&key).cloned()
        } else {
            None
        }
    }

    pub fn put(&mut self, key: usize, data: Vec<u8>) -> Vec<(usize, Vec<u8>, CacheItemMetadata)> {
        let data_size = data.len();

        // 如果已存在，先移除
        if self.items.contains_key(&key) {
            self.remove(key);
        }

        // 执行自适应调整
        if self.last_adaptation.elapsed() >= self.adaptation_window {
            self.perform_adaptive_eviction();
            self.last_adaptation = Instant::now();
        }

        // 确保有足够空间
        let mut evicted_items = Vec::new();
        while self.current_size + data_size > self.max_size && !self.items.is_empty() {
            if let Some((evicted_key, evicted_data, evicted_meta)) = self.evict_least_valuable() {
                evicted_items.push((evicted_key, evicted_data, evicted_meta));
            } else {
                break;
            }
        }

        // 添加新项
        if self.current_size + data_size <= self.max_size {
            let meta = CacheItemMetadata::new(key, data_size);
            let freq_bucket = self.get_frequency_bucket(meta.access_frequency);

            self.items.insert(key, data);
            self.metadata.insert(key, meta);
            self.frequency_buckets
                .entry(freq_bucket)
                .or_insert_with(BTreeSet::new)
                .insert(key);
            self.current_size += data_size;
        }

        evicted_items
    }

    fn get_frequency_bucket(&self, frequency: f64) -> u8 {
        if frequency < 1.0 {
            0
        } else if frequency < 5.0 {
            1
        } else if frequency < 10.0 {
            2
        } else if frequency < 20.0 {
            3
        } else {
            4
        }
    }

    fn evict_least_valuable(&mut self) -> Option<(usize, Vec<u8>, CacheItemMetadata)> {
        // 从最低频率桶开始驱逐
        for bucket_id in 0..5u8 {
            if let Some(bucket) = self.frequency_buckets.get_mut(&bucket_id) {
                if let Some(&key) = bucket.iter().next() {
                    bucket.remove(&key);
                    if let (Some(data), Some(meta)) =
                        (self.items.remove(&key), self.metadata.remove(&key))
                    {
                        self.current_size -= data.len();
                        return Some((key, data, meta));
                    }
                }
            }
        }
        None
    }

    fn perform_adaptive_eviction(&mut self) {
        // 执行自适应驱逐策略，移除过时的低频项目
        let _current_time = Instant::now();
        let mut to_remove = Vec::new();

        for (&key, meta) in &self.metadata {
            // 如果项目超过5分钟未访问且频率很低，标记为移除
            if meta.last_access.elapsed() > Duration::from_secs(300) && meta.access_frequency < 0.1
            {
                to_remove.push(key);
            }
        }

        for key in to_remove {
            self.remove(key);
        }
    }

    pub fn remove(&mut self, key: usize) -> Option<(Vec<u8>, CacheItemMetadata)> {
        if let (Some(data), Some(meta)) = (self.items.remove(&key), self.metadata.remove(&key)) {
            self.current_size -= data.len();

            // 从频率桶中移除
            let freq_bucket = self.get_frequency_bucket(meta.access_frequency);
            if let Some(bucket) = self.frequency_buckets.get_mut(&freq_bucket) {
                bucket.remove(&key);
            }

            Some((data, meta))
        } else {
            None
        }
    }

    pub fn clear(&mut self) {
        self.items.clear();
        self.metadata.clear();
        self.frequency_buckets.clear();
        self.current_size = 0;
    }

    pub fn get_frequency_distribution(&self) -> HashMap<u8, usize> {
        self.frequency_buckets
            .iter()
            .map(|(&k, v)| (k, v.len()))
            .collect()
    }

    pub fn get_metadata(&self, key: usize) -> Option<&CacheItemMetadata> {
        self.metadata.get(&key)
    }
}

// 压缩缓存实现
#[derive(Debug)]
pub struct CompressedCache {
    items: HashMap<usize, Vec<u8>>, // 存储压缩后的数据
    metadata: HashMap<usize, CacheItemMetadata>,
    uncompressed_sizes: HashMap<usize, usize>, // 原始大小映射
    max_size: usize,
    current_size: usize,
    compression_threshold: usize,
    compression_ratio_history: VecDeque<f64>,
}

impl CompressedCache {
    pub fn new(max_size: usize, compression_threshold: usize) -> Self {
        Self {
            items: HashMap::new(),
            metadata: HashMap::new(),
            uncompressed_sizes: HashMap::new(),
            max_size,
            current_size: 0,
            compression_threshold,
            compression_ratio_history: VecDeque::new(),
        }
    }

    pub fn get(&mut self, key: usize) -> Option<Vec<u8>> {
        if let Some(compressed_data) = self.items.get(&key) {
            // 更新元数据
            if let Some(meta) = self.metadata.get_mut(&key) {
                meta.access();
            }

            // 解压缩数据
            if let Some(meta) = self.metadata.get(&key) {
                if meta.is_compressed {
                    self.decompress_data(compressed_data)
                } else {
                    Some(compressed_data.clone())
                }
            } else {
                Some(compressed_data.clone())
            }
        } else {
            None
        }
    }

    pub fn put(&mut self, key: usize, data: Vec<u8>) -> Vec<(usize, Vec<u8>, CacheItemMetadata)> {
        let original_size = data.len();

        // 如果已存在，先移除
        if self.items.contains_key(&key) {
            self.remove(key);
        }

        // 决定是否压缩
        let (stored_data, compressed_size, is_compressed) =
            if original_size >= self.compression_threshold {
                let compressed = self.compress_data(&data);
                let comp_size = compressed.len();
                (compressed, comp_size, true)
            } else {
                (data.clone(), original_size, false)
            };

        // 确保有足够空间
        let mut evicted_items = Vec::new();
        while self.current_size + compressed_size > self.max_size && !self.items.is_empty() {
            if let Some((evicted_key, evicted_data, evicted_meta)) = self.evict_oldest() {
                evicted_items.push((evicted_key, evicted_data, evicted_meta));
            } else {
                break;
            }
        }

        // 添加新项
        if self.current_size + compressed_size <= self.max_size {
            let mut meta = CacheItemMetadata::new(key, original_size);
            meta.is_compressed = is_compressed;

            self.items.insert(key, stored_data);
            self.metadata.insert(key, meta);
            self.uncompressed_sizes.insert(key, original_size);
            self.current_size += compressed_size;

            // 更新压缩比统计
            if is_compressed {
                self.update_compression_ratio(original_size, compressed_size);
            }
        }

        evicted_items
    }

    fn compress_data(&self, data: &[u8]) -> Vec<u8> {
        // 简化的压缩实现（实际中应使用真正的压缩算法如zstd, lz4等）
        // 这里只是模拟压缩，实际压缩率约为70%
        let compressed_size = (data.len() as f64 * 0.7) as usize;
        let mut compressed = Vec::with_capacity(compressed_size + 8);

        // 添加原始大小作为头部
        compressed.extend_from_slice(&data.len().to_le_bytes());

        // 简化压缩：取样本数据
        if data.len() > compressed_size {
            compressed.extend_from_slice(&data[..compressed_size]);
        } else {
            compressed.extend_from_slice(data);
        }

        compressed
    }

    fn decompress_data(&self, compressed_data: &[u8]) -> Option<Vec<u8>> {
        if compressed_data.len() < 8 {
            return None;
        }

        // 读取原始大小
        let original_size = usize::from_le_bytes([
            compressed_data[0],
            compressed_data[1],
            compressed_data[2],
            compressed_data[3],
            compressed_data[4],
            compressed_data[5],
            compressed_data[6],
            compressed_data[7],
        ]);

        // 简化解压缩：重复数据填充到原始大小
        let mut decompressed = Vec::with_capacity(original_size);
        let payload = &compressed_data[8..];

        while decompressed.len() < original_size {
            let remaining = original_size - decompressed.len();
            let copy_size = remaining.min(payload.len());
            decompressed.extend_from_slice(&payload[..copy_size]);
        }

        decompressed.truncate(original_size);
        Some(decompressed)
    }

    fn evict_oldest(&mut self) -> Option<(usize, Vec<u8>, CacheItemMetadata)> {
        // 找到最老的项目
        let oldest_key = self
            .metadata
            .iter()
            .min_by_key(|(_, meta)| meta.last_access)
            .map(|(k, _)| *k);

        if let Some(key) = oldest_key {
            if let (Some(data), Some(meta)) = (self.items.remove(&key), self.metadata.remove(&key))
            {
                self.current_size -= data.len();
                self.uncompressed_sizes.remove(&key);

                // 如果是压缩数据，需要解压缩后返回
                let return_data = if meta.is_compressed {
                    self.decompress_data(&data).unwrap_or(data)
                } else {
                    data
                };

                return Some((key, return_data, meta));
            }
        }

        None
    }

    fn update_compression_ratio(&mut self, original_size: usize, compressed_size: usize) {
        let ratio = compressed_size as f64 / original_size as f64;
        self.compression_ratio_history.push_back(ratio);

        // 保持最近100个压缩比记录
        if self.compression_ratio_history.len() > 100 {
            self.compression_ratio_history.pop_front();
        }
    }

    pub fn remove(&mut self, key: usize) -> Option<(Vec<u8>, CacheItemMetadata)> {
        if let (Some(data), Some(meta)) = (self.items.remove(&key), self.metadata.remove(&key)) {
            self.current_size -= data.len();
            self.uncompressed_sizes.remove(&key);

            // 如果是压缩数据，解压缩后返回
            let return_data = if meta.is_compressed {
                self.decompress_data(&data).unwrap_or(data)
            } else {
                data
            };

            Some((return_data, meta))
        } else {
            None
        }
    }

    pub fn clear(&mut self) {
        self.items.clear();
        self.metadata.clear();
        self.uncompressed_sizes.clear();
        self.current_size = 0;
        self.compression_ratio_history.clear();
    }

    pub fn get_metadata(&self, key: usize) -> Option<&CacheItemMetadata> {
        self.metadata.get(&key)
    }

    pub fn get_all_keys(&self) -> Vec<usize> {
        self.metadata.keys().copied().collect()
    }
}

// 多级缓存系统主结构
#[derive(Debug)]
pub struct MultiLevelCache {
    l1_cache: Arc<Mutex<LRUCache>>,        // L1: 热点缓存 (LRU)
    l2_cache: Arc<Mutex<AdaptiveCache>>,   // L2: 自适应缓存 (频率优化)
    l3_cache: Arc<Mutex<CompressedCache>>, // L3: 压缩缓存 (大容量)
    policy: CachePolicy,
}

impl MultiLevelCache {
    pub fn new(policy: CachePolicy) -> Self {
        Self {
            l1_cache: Arc::new(Mutex::new(LRUCache::new(policy.l1_max_size))),
            l2_cache: Arc::new(Mutex::new(AdaptiveCache::new(policy.l2_max_size))),
            l3_cache: Arc::new(Mutex::new(CompressedCache::new(
                policy.l3_max_size,
                policy.compression_threshold,
            ))),
            policy,
        }
    }

    pub fn new_with_default_policy() -> Self {
        Self::new(CachePolicy::default())
    }

    /// 获取数据 - 依次检查L1、L2、L3缓存
    pub fn get(&self, key: usize) -> Option<Vec<u8>> {
        // 首先检查L1缓存
        if let Ok(mut l1) = self.l1_cache.lock() {
            if let Some(data) = l1.get(key) {
                return Some(data);
            }
        }

        // 检查L2缓存，如果命中则考虑提升到L1
        if let Ok(mut l2) = self.l2_cache.lock() {
            if let Some(data) = l2.get(key) {
                // 检查是否需要提升到L1
                if let Some(meta) = l2.get_metadata(key) {
                    if meta.access_frequency >= self.policy.l2_to_l1_threshold {
                        self.promote_to_l1(key, data.clone(), meta.clone());
                    }
                }

                return Some(data);
            }
        }

        // 检查L3缓存，如果命中则考虑提升到L2
        if let Ok(mut l3) = self.l3_cache.lock() {
            if let Some(data) = l3.get(key) {
                // 检查是否需要提升到L2
                if let Some(meta) = l3.get_metadata(key) {
                    if meta.access_frequency >= self.policy.l3_to_l2_threshold {
                        self.promote_to_l2(key, data.clone(), meta.clone());
                    }
                }

                return Some(data);
            }
        }

        None
    }

    /// 存储数据 - 根据策略决定存储层级
    pub fn put(&self, key: usize, data: Vec<u8>) {
        let data_size = data.len();

        // 根据数据大小和策略决定初始存储层级
        if data_size <= self.policy.l1_max_size / 10 {
            // 小数据直接存储到L1
            self.put_to_l1(key, data);
        } else if data_size <= self.policy.l2_max_size / 10 {
            // 中等数据存储到L2
            self.put_to_l2(key, data);
        } else {
            // 大数据存储到L3
            self.put_to_l3(key, data);
        }
    }

    fn put_to_l1(&self, key: usize, data: Vec<u8>) {
        if let Ok(mut l1) = self.l1_cache.lock() {
            if let Some((evicted_key, evicted_data, evicted_meta)) = l1.put(key, data) {
                // L1驱逐的数据降级到L2
                self.demote_to_l2(evicted_key, evicted_data, evicted_meta);
            }
        }
    }

    fn put_to_l2(&self, key: usize, data: Vec<u8>) {
        if let Ok(mut l2) = self.l2_cache.lock() {
            let evicted_items = l2.put(key, data);
            // L2驱逐的数据降级到L3
            for (evicted_key, evicted_data, evicted_meta) in evicted_items {
                self.demote_to_l3(evicted_key, evicted_data, evicted_meta);
            }
        }
    }

    fn put_to_l3(&self, key: usize, data: Vec<u8>) {
        if let Ok(mut l3) = self.l3_cache.lock() {
            let _evicted_items = l3.put(key, data);
            // L3驱逐的数据直接丢弃
        }
    }

    fn promote_to_l1(&self, key: usize, data: Vec<u8>, _meta: CacheItemMetadata) {
        self.put_to_l1(key, data);
    }

    fn promote_to_l2(&self, key: usize, data: Vec<u8>, _meta: CacheItemMetadata) {
        self.put_to_l2(key, data);
    }

    fn demote_to_l2(&self, key: usize, data: Vec<u8>, _meta: CacheItemMetadata) {
        self.put_to_l2(key, data);
    }

    fn demote_to_l3(&self, key: usize, data: Vec<u8>, _meta: CacheItemMetadata) {
        self.put_to_l3(key, data);
    }

    /// 移除数据 - 从所有层级移除
    pub fn remove(&self, key: usize) -> bool {
        let mut found = false;

        // 从所有层级移除
        if let Ok(mut l1) = self.l1_cache.lock() {
            if l1.remove(key).is_some() {
                found = true;
            }
        }

        if let Ok(mut l2) = self.l2_cache.lock() {
            if l2.remove(key).is_some() {
                found = true;
            }
        }

        if let Ok(mut l3) = self.l3_cache.lock() {
            if l3.remove(key).is_some() {
                found = true;
            }
        }

        found
    }

    /// 清空所有缓存
    pub fn clear(&self) {
        if let Ok(mut l1) = self.l1_cache.lock() {
            l1.clear();
        }

        if let Ok(mut l2) = self.l2_cache.lock() {
            l2.clear();
        }

        if let Ok(mut l3) = self.l3_cache.lock() {
            l3.clear();
        }
    }
}
