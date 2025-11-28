//! 压缩缓存实现
//!
//! 提供带压缩功能的缓存机制，自动压缩大于阈值的数据

use crate::cache::lru_cache::CacheItemMetadata;
use std::collections::HashMap;

/// 压缩缓存实现 - L3层使用
#[derive(Debug)]
pub struct CompressedCache {
    items: HashMap<usize, Vec<u8>>, // 存储压缩数据
    metadata: HashMap<usize, CacheItemMetadata>,
    uncompressed_sizes: HashMap<usize, usize>, // 原始大小映射
    current_size: usize,                       // 压缩后的大小
    max_size: usize,
    compression_ratio: f64,       // 平均压缩比
    compression_threshold: usize, // 压缩阈值
}

impl CompressedCache {
    pub fn new(max_size: usize, compression_threshold: usize) -> Self {
        Self {
            items: HashMap::new(),
            metadata: HashMap::new(),
            uncompressed_sizes: HashMap::new(),
            current_size: 0,
            max_size,
            compression_ratio: 0.7, // 假设70%的压缩比
            compression_threshold,
        }
    }

    pub fn get(&mut self, key: usize) -> Option<Vec<u8>> {
        if let Some(compressed_data) = self.items.get(&key) {
            // 更新元数据
            if let Some(meta) = self.metadata.get_mut(&key) {
                meta.access();
            }

            // 解压数据
            Some(self.decompress_data(compressed_data))
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
        // 简化的压缩实现 - 在实际应用中可以使用LZ4、Zstd等
        // 这里使用简单的RLE压缩作为示例
        let mut compressed = Vec::new();
        if data.is_empty() {
            return compressed;
        }

        let mut current_byte = data[0];
        let mut count: u8 = 1;

        for &byte in &data[1..] {
            if byte == current_byte && count < 255 {
                count += 1;
            } else {
                compressed.push(count);
                compressed.push(current_byte);
                current_byte = byte;
                count = 1;
            }
        }

        // 添加最后一组
        compressed.push(count);
        compressed.push(current_byte);

        // 如果压缩后更大，返回原数据
        if compressed.len() >= data.len() {
            data.to_vec()
        } else {
            compressed
        }
    }

    fn decompress_data(&self, compressed_data: &[u8]) -> Vec<u8> {
        let mut decompressed = Vec::new();

        // 简化的解压实现
        let mut i = 0;
        while i + 1 < compressed_data.len() {
            let count = compressed_data[i];
            let byte = compressed_data[i + 1];

            for _ in 0..count {
                decompressed.push(byte);
            }

            i += 2;
        }

        decompressed
    }

    fn update_compression_ratio(&mut self, original_size: usize, compressed_size: usize) {
        let current_ratio = compressed_size as f64 / original_size as f64;
        // 使用指数移动平均更新压缩比
        self.compression_ratio = 0.9 * self.compression_ratio + 0.1 * current_ratio;
    }

    fn evict_oldest(&mut self) -> Option<(usize, Vec<u8>, CacheItemMetadata)> {
        // 找到最老的项目
        let oldest_key = self
            .metadata
            .iter()
            .min_by_key(|(_, meta)| meta.created_at)
            .map(|(&key, _)| key);

        if let Some(key) = oldest_key {
            if let (Some(compressed_data), Some(meta)) =
                (self.items.remove(&key), self.metadata.remove(&key))
            {
                let _original_size = self.uncompressed_sizes.remove(&key).unwrap_or(0);
                self.current_size -= compressed_data.len();
                let decompressed_data = self.decompress_data(&compressed_data);
                return Some((key, decompressed_data, meta));
            }
        }

        None
    }

    pub fn remove(&mut self, key: usize) -> Option<(Vec<u8>, CacheItemMetadata)> {
        if let (Some(compressed_data), Some(meta)) =
            (self.items.remove(&key), self.metadata.remove(&key))
        {
            self.uncompressed_sizes.remove(&key);
            self.current_size -= compressed_data.len();
            let decompressed_data = self.decompress_data(&compressed_data);
            Some((decompressed_data, meta))
        } else {
            None
        }
    }

    pub fn get_metadata(&self, key: usize) -> Option<&CacheItemMetadata> {
        self.metadata.get(&key)
    }

    pub fn clear(&mut self) {
        self.items.clear();
        self.metadata.clear();
        self.uncompressed_sizes.clear();
        self.current_size = 0;
    }

    pub fn get_all_keys(&self) -> Vec<usize> {
        self.metadata.keys().copied().collect()
    }
}
