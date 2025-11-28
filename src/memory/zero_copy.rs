//! 零拷贝内存管理
//!
//! 从lazy_array_original.rs中提取的零拷贝处理器实现

use std::sync::{Arc, Mutex};
use std::time::Instant;

/// 零拷贝处理器，提供高效的内存访问模式
pub struct ZeroCopyHandler {
    min_size_threshold: usize,
    alignment_requirement: usize,
    // 性能统计
    zero_copy_hits: Arc<Mutex<usize>>,
    fallback_to_copy: Arc<Mutex<usize>>,
    // 内存使用跟踪
    total_zero_copy_memory: Arc<Mutex<usize>>,
    // 访问模式分析器
    access_analyzer: Arc<Mutex<ZeroCopyAnalyzer>>,
}

/// 零拷贝分析器
#[derive(Debug)]
pub struct ZeroCopyAnalyzer {
    recent_accesses: Vec<(usize, usize, Instant)>, // (offset, size, time)
    continuous_access_count: usize,
    fragmented_access_count: usize,
    average_access_size: f64,
    last_optimization_check: Instant,
}

impl ZeroCopyAnalyzer {
    pub fn new() -> Self {
        Self {
            recent_accesses: Vec::new(),
            continuous_access_count: 0,
            fragmented_access_count: 0,
            average_access_size: 0.0,
            last_optimization_check: Instant::now(),
        }
    }

    pub fn record_access(&mut self, offset: usize, size: usize) {
        let now = Instant::now();
        self.recent_accesses.push((offset, size, now));

        // 保持历史记录在合理范围内
        if self.recent_accesses.len() > 1000 {
            self.recent_accesses.drain(0..500);
        }

        // 更新统计信息
        self.update_statistics();
    }

    fn update_statistics(&mut self) {
        if self.recent_accesses.len() < 2 {
            return;
        }

        let mut continuous_count = 0;
        let mut fragmented_count = 0;
        let total_size: usize = self.recent_accesses.iter().map(|(_, size, _)| *size).sum();

        // 分析连续性
        for window in self.recent_accesses.windows(2) {
            let (offset1, size1, _) = window[0];
            let (offset2, _, _) = window[1];

            if offset2 == offset1 + size1 {
                continuous_count += 1;
            } else {
                fragmented_count += 1;
            }
        }

        self.continuous_access_count = continuous_count;
        self.fragmented_access_count = fragmented_count;
        self.average_access_size = total_size as f64 / self.recent_accesses.len() as f64;
    }

    pub fn should_prefer_zero_copy(&self) -> bool {
        // 基于访问模式判断是否应优先使用零拷贝
        let continuity_ratio = if self.continuous_access_count + self.fragmented_access_count > 0 {
            self.continuous_access_count as f64
                / (self.continuous_access_count + self.fragmented_access_count) as f64
        } else {
            0.0
        };

        // 连续访问比例高且平均访问大小大于阈值
        continuity_ratio > 0.7 && self.average_access_size > 2048.0
    }

    pub fn get_optimal_chunk_size(&self) -> usize {
        // 基于历史访问模式推荐最优分块大小
        if self.average_access_size > (64 * 1024) as f64 {
            64 * 1024 // 64KB
        } else if self.average_access_size > (16 * 1024) as f64 {
            16 * 1024 // 16KB
        } else if self.average_access_size > (4 * 1024) as f64 {
            4 * 1024 // 4KB
        } else {
            1024 // 1KB
        }
    }
}

/// 零拷贝视图结构 - 安全的零拷贝视图
pub struct ZeroCopyView<'a> {
    data: &'a [u8],
    lifetime_guard: Arc<()>, // 生命周期守护
    metadata: ZeroCopyMetadata,
}

#[derive(Debug, Clone)]
pub struct ZeroCopyMetadata {
    pub offset: usize,
    pub size: usize,
    pub created_at: Instant,
    pub access_count: Arc<Mutex<usize>>,
    pub is_continuous: bool,
}

impl<'a> ZeroCopyView<'a> {
    pub fn new(data: &'a [u8], offset: usize, lifetime_guard: Arc<()>) -> Self {
        Self {
            data,
            lifetime_guard,
            metadata: ZeroCopyMetadata {
                offset,
                size: data.len(),
                created_at: Instant::now(),
                access_count: Arc::new(Mutex::new(0)),
                is_continuous: true,
            },
        }
    }

    pub fn data(&self) -> &[u8] {
        // 增加访问计数
        if let Ok(mut count) = self.metadata.access_count.lock() {
            *count += 1;
        }
        self.data
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn metadata(&self) -> &ZeroCopyMetadata {
        &self.metadata
    }

    // 分割视图 - 创建子视图
    pub fn slice(&self, start: usize, end: usize) -> Option<ZeroCopyView<'a>> {
        if start < end && end <= self.data.len() {
            Some(ZeroCopyView {
                data: &self.data[start..end],
                lifetime_guard: Arc::clone(&self.lifetime_guard),
                metadata: ZeroCopyMetadata {
                    offset: self.metadata.offset + start,
                    size: end - start,
                    created_at: Instant::now(),
                    access_count: Arc::new(Mutex::new(0)),
                    is_continuous: self.metadata.is_continuous,
                },
            })
        } else {
            None
        }
    }

    // 安全转换为 Vec<u8>（如果需要拥有所有权）
    pub fn to_owned(&self) -> Vec<u8> {
        self.data.to_vec()
    }
}

impl ZeroCopyHandler {
    pub fn new() -> Self {
        Self {
            min_size_threshold: 4096,  // 4KB最小阈值
            alignment_requirement: 64, // 64字节对齐
            zero_copy_hits: Arc::new(Mutex::new(0)),
            fallback_to_copy: Arc::new(Mutex::new(0)),
            total_zero_copy_memory: Arc::new(Mutex::new(0)),
            access_analyzer: Arc::new(Mutex::new(ZeroCopyAnalyzer::new())),
        }
    }

    /// 判断是否可以使用零拷贝
    pub fn can_zero_copy(&self, indices: &[usize], item_size: usize) -> bool {
        // 检查数据大小是否达到阈值
        let total_size = indices.len() * item_size;
        if total_size < self.min_size_threshold {
            return false;
        }

        // 检查访问是否连续
        self.is_continuous_access(indices)
    }

    fn is_continuous_access(&self, indices: &[usize]) -> bool {
        if indices.len() < 2 {
            return true;
        }

        for window in indices.windows(2) {
            if window[1] != window[0] + 1 {
                return false;
            }
        }
        true
    }
}

/// 零拷贝性能统计
#[derive(Debug, Clone)]
pub struct ZeroCopyStats {
    pub zero_copy_hits: usize,
    pub fallback_to_copy: usize,
    pub zero_copy_rate: f64,
    pub total_zero_copy_memory: usize,
    pub total_accesses: usize,
}
