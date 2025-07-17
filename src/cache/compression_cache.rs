//! 高级压缩缓存系统
//! 
//! 提供多种压缩算法、自适应压缩策略和压缩性能优化

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

// 压缩算法类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CompressionAlgorithm {
    None,           // 无压缩
    Simple,         // 简单压缩（用于测试）
    LZ4,           // 快速压缩
    Zstd,          // 平衡压缩
    Brotli,        // 高压缩比
}

// 压缩策略
#[derive(Debug, Clone)]
pub enum CompressionStrategy {
    Fixed(CompressionAlgorithm),                    // 固定算法
    SizeBased(Vec<(usize, CompressionAlgorithm)>), // 基于数据大小选择
    Adaptive,                                       // 自适应选择
    Performance,                                    // 性能优先
    SpaceEfficient,                                // 空间优先
}

// 压缩统计信息
#[derive(Debug, Clone, Default)]
pub struct CompressionStats {
    pub total_compressions: u64,
    pub total_decompressions: u64,
    pub total_original_bytes: u64,
    pub total_compressed_bytes: u64,
    pub compression_time: Duration,
    pub decompression_time: Duration,
    pub average_compression_ratio: f64,
    pub algorithm_usage: HashMap<CompressionAlgorithm, u64>,
}

impl CompressionStats {
    pub fn record_compression(&mut self, 
                            algorithm: CompressionAlgorithm,
                            original_size: usize, 
                            compressed_size: usize, 
                            time_taken: Duration) {
        self.total_compressions += 1;
        self.total_original_bytes += original_size as u64;
        self.total_compressed_bytes += compressed_size as u64;
        self.compression_time += time_taken;
        
        *self.algorithm_usage.entry(algorithm).or_insert(0) += 1;
        
        // 更新平均压缩比
        if self.total_original_bytes > 0 {
            self.average_compression_ratio = self.total_compressed_bytes as f64 / self.total_original_bytes as f64;
        }
    }
    
    pub fn record_decompression(&mut self, time_taken: Duration) {
        self.total_decompressions += 1;
        self.decompression_time += time_taken;
    }
    
    pub fn get_compression_ratio(&self) -> f64 {
        self.average_compression_ratio
    }
    
    pub fn get_average_compression_time(&self) -> Duration {
        if self.total_compressions > 0 {
            self.compression_time / self.total_compressions as u32
        } else {
            Duration::from_secs(0)
        }
    }
    
    pub fn get_average_decompression_time(&self) -> Duration {
        if self.total_decompressions > 0 {
            self.decompression_time / self.total_decompressions as u32
        } else {
            Duration::from_secs(0)
        }
    }
}

// 高级压缩缓存
#[derive(Debug)]
pub struct AdvancedCompressionCache {
    // 核心存储
    compressed_data: HashMap<usize, CompressedItem>,
    metadata: HashMap<usize, CompressionMetadata>,
    
    // 配置参数
    max_size: usize,
    current_size: usize,
    compression_strategy: CompressionStrategy,
    min_compression_size: usize,
    
    // 性能统计
    stats: CompressionStats,
    
    // 自适应算法选择
    algorithm_performance: HashMap<CompressionAlgorithm, AlgorithmPerformance>,
    recent_performance_samples: VecDeque<PerformanceSample>,
    
    // 驱逐策略
    access_order: VecDeque<usize>,
    compression_efficiency_threshold: f64,
}

#[derive(Debug, Clone)]
struct CompressedItem {
    data: Vec<u8>,
    algorithm: CompressionAlgorithm,
    original_size: usize,
    compressed_size: usize,
    compression_time: Duration,
}

#[derive(Debug, Clone)]
struct CompressionMetadata {
    key: usize,
    last_access: Instant,
    access_count: usize,
    compression_ratio: f64,
    algorithm_used: CompressionAlgorithm,
    creation_time: Instant,
}

#[derive(Debug, Clone, Default)]
struct AlgorithmPerformance {
    total_compressions: u64,
    total_compression_time: Duration,
    total_original_bytes: u64,
    total_compressed_bytes: u64,
    average_ratio: f64,
    average_speed: f64, // bytes per second
}

#[derive(Debug, Clone)]
struct PerformanceSample {
    algorithm: CompressionAlgorithm,
    original_size: usize,
    compressed_size: usize,
    compression_time: Duration,
    timestamp: Instant,
}

impl AdvancedCompressionCache {
    pub fn new(max_size: usize, strategy: CompressionStrategy) -> Self {
        Self {
            compressed_data: HashMap::new(),
            metadata: HashMap::new(),
            max_size,
            current_size: 0,
            compression_strategy: strategy,
            min_compression_size: 1024, // 1KB最小压缩大小
            stats: CompressionStats::default(),
            algorithm_performance: HashMap::new(),
            recent_performance_samples: VecDeque::new(),
            access_order: VecDeque::new(),
            compression_efficiency_threshold: 0.8, // 80%压缩效率阈值
        }
    }
    
    /// 获取数据（自动解压缩）
    pub fn get(&mut self, key: usize) -> Option<Vec<u8>> {
        if self.compressed_data.contains_key(&key) {
            // 获取解压缩所需的信息
            let (data, algorithm, original_size) = if let Some(item) = self.compressed_data.get(&key) {
                (item.data.clone(), item.algorithm, item.original_size)
            } else {
                return None;
            };
            
            // 更新访问记录
            if let Some(meta) = self.metadata.get_mut(&key) {
                meta.last_access = Instant::now();
                meta.access_count += 1;
            }
            
            // 移动到访问队列前端
            self.move_to_front(key);
            
            // 解压缩数据
            let start_time = Instant::now();
            let decompressed = self.decompress_data(&data, algorithm, original_size);
            let decompression_time = start_time.elapsed();
            
            // 记录解压缩性能
            self.stats.record_decompression(decompression_time);
            
            decompressed
        } else {
            None
        }
    }
    
    /// 存储数据（自动压缩）
    pub fn put(&mut self, key: usize, data: Vec<u8>) -> Vec<(usize, Vec<u8>)> {
        let original_size = data.len();
        
        // 移除已存在的项
        if self.compressed_data.contains_key(&key) {
            self.remove(key);
        }
        
        // 选择压缩算法
        let algorithm = self.select_compression_algorithm(&data);
        
        // 执行压缩
        let start_time = Instant::now();
        let (compressed_data, compressed_size) = if original_size >= self.min_compression_size && algorithm != CompressionAlgorithm::None {
            let compressed = self.compress_data(&data, algorithm);
            let compressed_len = compressed.len();
            (compressed, compressed_len)
        } else {
            (data.clone(), original_size)
        };
        let compression_time = start_time.elapsed();
        
        // 记录性能统计
        self.stats.record_compression(algorithm, original_size, compressed_size, compression_time);
        self.record_algorithm_performance(algorithm, original_size, compressed_size, compression_time);
        
        // 确保有足够空间
        let mut evicted_items = Vec::new();
        while self.current_size + compressed_size > self.max_size && !self.compressed_data.is_empty() {
            if let Some((evicted_key, evicted_data)) = self.evict_least_efficient() {
                evicted_items.push((evicted_key, evicted_data));
            } else {
                break;
            }
        }
        
        // 存储压缩项
        if self.current_size + compressed_size <= self.max_size {
            let item = CompressedItem {
                data: compressed_data,
                algorithm,
                original_size,
                compressed_size,
                compression_time,
            };
            
            let metadata = CompressionMetadata {
                key,
                last_access: Instant::now(),
                access_count: 1,
                compression_ratio: compressed_size as f64 / original_size as f64,
                algorithm_used: algorithm,
                creation_time: Instant::now(),
            };
            
            self.compressed_data.insert(key, item);
            self.metadata.insert(key, metadata);
            self.access_order.push_front(key);
            self.current_size += compressed_size;
        }
        
        evicted_items
    }
    
    /// 移除项
    pub fn remove(&mut self, key: usize) -> Option<Vec<u8>> {
        if let (Some(item), Some(_meta)) = (self.compressed_data.remove(&key), self.metadata.remove(&key)) {
            self.current_size -= item.compressed_size;
            self.access_order.retain(|&k| k != key);
            
            // 返回解压缩的原始数据
            self.decompress_data(&item.data, item.algorithm, item.original_size)
        } else {
            None
        }
    }
    
    /// 选择压缩算法
    fn select_compression_algorithm(&self, data: &[u8]) -> CompressionAlgorithm {
        match &self.compression_strategy {
            CompressionStrategy::Fixed(algorithm) => *algorithm,
            CompressionStrategy::SizeBased(size_rules) => {
                let data_size = data.len();
                for (threshold, algorithm) in size_rules {
                    if data_size >= *threshold {
                        return *algorithm;
                    }
                }
                CompressionAlgorithm::None
            }
            CompressionStrategy::Adaptive => {
                self.adaptive_algorithm_selection(data)
            }
            CompressionStrategy::Performance => {
                // 选择最快的算法
                CompressionAlgorithm::LZ4
            }
            CompressionStrategy::SpaceEfficient => {
                // 选择压缩比最高的算法
                CompressionAlgorithm::Brotli
            }
        }
    }
    
    /// 自适应算法选择
    fn adaptive_algorithm_selection(&self, data: &[u8]) -> CompressionAlgorithm {
        let data_size = data.len();
        
        // 小文件使用快速算法
        if data_size < 4096 {
            return CompressionAlgorithm::LZ4;
        }
        
        // 基于历史性能选择最佳算法
        let mut best_algorithm = CompressionAlgorithm::Simple;
        let mut best_score = 0.0;
        
        for (&algorithm, performance) in &self.algorithm_performance {
            if performance.total_compressions > 10 { // 需要足够的样本
                // 综合评分：考虑压缩比和速度
                let compression_score = 1.0 - performance.average_ratio; // 压缩比越高分数越高
                let speed_score = performance.average_speed / 1_000_000.0; // 速度归一化
                let total_score = compression_score * 0.6 + speed_score * 0.4;
                
                if total_score > best_score {
                    best_score = total_score;
                    best_algorithm = algorithm;
                }
            }
        }
        
        best_algorithm
    }
    
    /// 压缩数据
    fn compress_data(&self, data: &[u8], algorithm: CompressionAlgorithm) -> Vec<u8> {
        match algorithm {
            CompressionAlgorithm::None => data.to_vec(),
            CompressionAlgorithm::Simple => {
                // 简化的压缩实现（仅用于演示）
                let mut compressed = Vec::new();
                compressed.push(algorithm as u8); // 算法标识
                compressed.extend_from_slice(&data.len().to_le_bytes()); // 原始大小
                
                // 简单的游程编码模拟
                let mut i = 0;
                while i < data.len() {
                    let byte = data[i];
                    let mut count = 1;
                    
                    while i + count < data.len() && data[i + count] == byte && count < 255 {
                        count += 1;
                    }
                    
                    if count > 1 {
                        compressed.push(255); // 重复标记
                        compressed.push(count as u8);
                        compressed.push(byte);
                    } else {
                        if byte == 255 {
                            compressed.push(255);
                            compressed.push(0); // 转义
                        }
                        compressed.push(byte);
                    }
                    
                    i += count;
                }
                
                compressed
            }
            CompressionAlgorithm::LZ4 => {
                // 模拟LZ4压缩（70%压缩比）
                self.simulate_compression(data, 0.7, algorithm)
            }
            CompressionAlgorithm::Zstd => {
                // 模拟Zstd压缩（60%压缩比）
                self.simulate_compression(data, 0.6, algorithm)
            }
            CompressionAlgorithm::Brotli => {
                // 模拟Brotli压缩（50%压缩比）
                self.simulate_compression(data, 0.5, algorithm)
            }
        }
    }
    
    /// 模拟压缩（用于演示）
    fn simulate_compression(&self, data: &[u8], ratio: f64, algorithm: CompressionAlgorithm) -> Vec<u8> {
        let compressed_size = (data.len() as f64 * ratio) as usize;
        let mut compressed = Vec::with_capacity(compressed_size + 9);
        
        // 添加头部信息
        compressed.push(algorithm as u8);
        compressed.extend_from_slice(&data.len().to_le_bytes());
        
        // 添加压缩数据（取样）
        let step = if data.len() > compressed_size {
            data.len() / compressed_size
        } else {
            1
        };
        
        for i in (0..data.len()).step_by(step) {
            if compressed.len() < compressed_size + 9 {
                compressed.push(data[i]);
            }
        }
        
        compressed
    }
    
    /// 解压缩数据
    fn decompress_data(&self, compressed: &[u8], algorithm: CompressionAlgorithm, original_size: usize) -> Option<Vec<u8>> {
        if compressed.len() < 9 {
            return None;
        }
        
        match algorithm {
            CompressionAlgorithm::None => Some(compressed.to_vec()),
            CompressionAlgorithm::Simple => {
                // 简单解压缩
                let mut decompressed = Vec::with_capacity(original_size);
                let mut i = 9; // 跳过头部
                
                while i < compressed.len() {
                    if compressed[i] == 255 && i + 1 < compressed.len() {
                        if compressed[i + 1] == 0 {
                            // 转义的255
                            decompressed.push(255);
                            i += 2;
                        } else {
                            // 重复序列
                            let count = compressed[i + 1] as usize;
                            let byte = compressed[i + 2];
                            for _ in 0..count {
                                decompressed.push(byte);
                            }
                            i += 3;
                        }
                    } else {
                        decompressed.push(compressed[i]);
                        i += 1;
                    }
                }
                
                Some(decompressed)
            }
            _ => {
                // 模拟解压缩：重复数据填充到原始大小
                let mut decompressed = Vec::with_capacity(original_size);
                let payload = &compressed[9..];
                
                while decompressed.len() < original_size {
                    let remaining = original_size - decompressed.len();
                    let copy_size = remaining.min(payload.len());
                    decompressed.extend_from_slice(&payload[..copy_size]);
                }
                
                decompressed.truncate(original_size);
                Some(decompressed)
            }
        }
    }
    
    /// 记录算法性能
    fn record_algorithm_performance(&mut self, 
                                  algorithm: CompressionAlgorithm,
                                  original_size: usize,
                                  compressed_size: usize,
                                  time_taken: Duration) {
        // 更新算法性能统计
        let performance = self.algorithm_performance.entry(algorithm).or_default();
        performance.total_compressions += 1;
        performance.total_compression_time += time_taken;
        performance.total_original_bytes += original_size as u64;
        performance.total_compressed_bytes += compressed_size as u64;
        
        // 更新平均值
        if performance.total_original_bytes > 0 {
            performance.average_ratio = performance.total_compressed_bytes as f64 / performance.total_original_bytes as f64;
        }
        
        if time_taken.as_secs_f64() > 0.0 {
            performance.average_speed = original_size as f64 / time_taken.as_secs_f64();
        }
        
        // 记录性能样本
        let sample = PerformanceSample {
            algorithm,
            original_size,
            compressed_size,
            compression_time: time_taken,
            timestamp: Instant::now(),
        };
        
        self.recent_performance_samples.push_back(sample);
        
        // 保持最近1000个样本
        if self.recent_performance_samples.len() > 1000 {
            self.recent_performance_samples.pop_front();
        }
    }
    
    /// 驱逐效率最低的项
    fn evict_least_efficient(&mut self) -> Option<(usize, Vec<u8>)> {
        // 找到压缩效率最低且访问频率最低的项
        let mut worst_key = None;
        let mut worst_score = f64::INFINITY;
        
        for (&key, meta) in &self.metadata {
            // 综合评分：压缩效率 + 访问频率 + 时间因素
            let efficiency_score = meta.compression_ratio; // 越低越好
            let access_score = 1.0 / (meta.access_count as f64 + 1.0); // 越低越好
            let age_score = meta.last_access.elapsed().as_secs_f64() / 3600.0; // 小时为单位
            
            let total_score = efficiency_score * 0.4 + access_score * 0.3 + age_score * 0.3;
            
            if total_score < worst_score {
                worst_score = total_score;
                worst_key = Some(key);
            }
        }
        
        if let Some(key) = worst_key {
            self.remove(key).map(|data| (key, data))
        } else {
            None
        }
    }
    
    /// 移动到访问队列前端
    fn move_to_front(&mut self, key: usize) {
        self.access_order.retain(|&k| k != key);
        self.access_order.push_front(key);
    }
    
    /// 清空缓存
    pub fn clear(&mut self) {
        self.compressed_data.clear();
        self.metadata.clear();
        self.access_order.clear();
        self.current_size = 0;
    }
    
    /// 获取统计信息
    pub fn get_stats(&self) -> &CompressionStats {
        &self.stats
    }
    
    /// 获取详细统计
    pub fn get_detailed_stats(&self) -> CompressionCacheStats {
        CompressionCacheStats {
            total_items: self.compressed_data.len(),
            total_compressed_size: self.current_size,
            total_original_size: self.metadata.values().map(|m| self.compressed_data.get(&m.key).map(|i| i.original_size).unwrap_or(0)).sum(),
            average_compression_ratio: self.stats.average_compression_ratio,
            algorithm_performance: self.algorithm_performance.clone(),
            current_strategy: format!("{:?}", self.compression_strategy),
            cache_utilization: self.current_size as f64 / self.max_size as f64,
        }
    }
    
    /// 优化压缩策略
    pub fn optimize_compression_strategy(&mut self) {
        // 基于性能历史优化策略
        if self.recent_performance_samples.len() >= 100 {
            let mut algorithm_scores: HashMap<CompressionAlgorithm, f64> = HashMap::new();
            
            for sample in &self.recent_performance_samples {
                let ratio_score = 1.0 - (sample.compressed_size as f64 / sample.original_size as f64);
                let speed_score = if sample.compression_time.as_secs_f64() > 0.0 {
                    sample.original_size as f64 / sample.compression_time.as_secs_f64() / 1_000_000.0
                } else {
                    0.0
                };
                let total_score = ratio_score * 0.6 + speed_score * 0.4;
                
                let current_score = algorithm_scores.get(&sample.algorithm).unwrap_or(&0.0);
                algorithm_scores.insert(sample.algorithm, current_score + total_score);
            }
            
            // 找到最佳算法
            let best_algorithm = algorithm_scores.iter()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(algo, _)| *algo)
                .unwrap_or(CompressionAlgorithm::Simple);
            
            // 更新策略
            self.compression_strategy = CompressionStrategy::Fixed(best_algorithm);
        }
    }
}

#[derive(Debug, Clone)]
pub struct CompressionCacheStats {
    pub total_items: usize,
    pub total_compressed_size: usize,
    pub total_original_size: usize,
    pub average_compression_ratio: f64,
    pub algorithm_performance: HashMap<CompressionAlgorithm, AlgorithmPerformance>,
    pub current_strategy: String,
    pub cache_utilization: f64,
} 