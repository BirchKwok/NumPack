//! SIMD处理器
//!
//! 从lazy_array_original.rs中提取的SIMD处理器实现
//! **特别关注Windows平台兼容性和内存安全**

// use std::sync::{Arc, Mutex}; // 暂时未使用

/// SIMD错误类型
#[derive(Debug)]
pub enum SIMDError {
    NotSupported,
    ProcessingFailed,
    InvalidInput,
    IndexOutOfBounds,
}

/// 错误处理策略枚举
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ErrorHandlingStrategy {
    Fallback, // 出错时回退到安全实现
    Panic,    // 出错时崩溃（仅用于测试）
    Ignore,   // 忽略错误（不推荐）
}

/// SIMD处理器，提供高性能向量化操作
/// **增强Windows平台安全性**
pub struct SIMDProcessor {
    supports_avx2: bool,
    supports_avx512: bool,
    supports_sse2: bool,
    alignment_size: usize,
    cache_line_size: usize,
    prefetch_distance: usize,
    // 新增: Windows平台SIMD特定属性
    win_safe_simd: bool,                  // 是否使用Windows安全SIMD模式
    win_memory_alignment: usize,          // Windows平台内存对齐要求
    error_handler: ErrorHandlingStrategy, // 错误处理策略
}

impl SIMDProcessor {
    pub fn new() -> Self {
        // 检测CPU特性
        let supports_sse2 = Self::detect_sse2();
        let supports_avx2 = Self::detect_avx2();
        let supports_avx512 = Self::detect_avx512();

        // Windows平台使用更严格的内存对齐要求
        #[cfg(target_os = "windows")]
        let (win_safe_simd, win_memory_alignment) = (true, 64);

        #[cfg(not(target_os = "windows"))]
        let (win_safe_simd, win_memory_alignment) = (false, 0);

        Self {
            supports_avx2,
            supports_avx512,
            supports_sse2,
            alignment_size: 64, // 64字节对齐，适配现代CPU缓存行
            cache_line_size: 64,
            prefetch_distance: 512, // 预取距离
            win_safe_simd,
            win_memory_alignment,
            error_handler: ErrorHandlingStrategy::Fallback,
        }
    }

    // 设置错误处理策略
    pub fn with_error_strategy(mut self, strategy: ErrorHandlingStrategy) -> Self {
        self.error_handler = strategy;
        self
    }

    // 设置是否使用Windows安全SIMD模式
    pub fn with_win_safe_simd(mut self, enabled: bool) -> Self {
        self.win_safe_simd = enabled;
        self
    }

    fn detect_avx2() -> bool {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            is_x86_feature_detected!("avx2")
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        {
            false
        }
    }

    fn detect_avx512() -> bool {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            is_x86_feature_detected!("avx512f")
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        {
            false
        }
    }

    fn detect_sse2() -> bool {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            is_x86_feature_detected!("sse2")
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        {
            false
        }
    }

    /// **主要的向量化复制接口 - Windows内存错误修复重点**
    pub fn vectorized_copy(&self, src: &[u8], dst: &mut [u8], indices: &[usize], item_size: usize) {
        // 预取源数据以提高缓存命中率
        self.prefetch_data(src, indices, item_size);

        // **Windows平台特殊处理 - 使用安全SIMD实现**
        #[cfg(target_os = "windows")]
        if self.win_safe_simd {
            return self.windows_safe_vectorized_copy(src, dst, indices, item_size);
        }

        // 根据CPU支持的指令集和数据规模选择最优实现
        if self.supports_avx512 && indices.len() >= 8 && item_size >= 32 {
            // AVX512实现（占位符）
            self.optimized_scalar_copy(src, dst, indices, item_size);
        } else if self.supports_avx2 && indices.len() >= 4 && item_size >= 16 {
            // AVX2实现（占位符）
            self.optimized_scalar_copy(src, dst, indices, item_size);
        } else if self.supports_sse2 && indices.len() >= 2 && item_size >= 8 {
            // SSE2实现（占位符）
            self.optimized_scalar_copy(src, dst, indices, item_size);
        } else {
            self.optimized_scalar_copy(src, dst, indices, item_size);
        }
    }

    /// 预取数据到缓存中
    fn prefetch_data(&self, src: &[u8], indices: &[usize], item_size: usize) {
        // **Windows平台安全预取操作**
        #[cfg(target_os = "windows")]
        if self.win_safe_simd {
            self.windows_safe_prefetch(src, indices, item_size);
            return;
        }

        // 非Windows平台或禁用安全模式时的标准预取
        #[cfg(not(target_os = "windows"))]
        for &idx in indices.iter().take(8) {
            // 只预取前8个，避免过度预取
            let offset = idx * item_size;
            if offset < src.len() {
                unsafe {
                    let ptr = src.as_ptr().add(offset);
                    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                    {
                        std::arch::x86_64::_mm_prefetch(
                            ptr as *const i8,
                            std::arch::x86_64::_MM_HINT_T0,
                        );
                    }
                }
            }
        }
    }

    // **Windows平台安全预取实现**
    #[cfg(target_os = "windows")]
    fn windows_safe_prefetch(&self, src: &[u8], indices: &[usize], item_size: usize) {
        // 只预取少量元素，避免过度预取导致的问题
        for &idx in indices.iter().take(4) {
            let offset = idx * item_size;
            if offset + item_size <= src.len() {
                // 使用volatile读取代替预取，实现类似的效果但更安全
                unsafe {
                    // 读取首字节和尾字节来触发页面加载
                    let _ = std::ptr::read_volatile(src.as_ptr().add(offset));
                    if item_size > 1 {
                        let _ = std::ptr::read_volatile(src.as_ptr().add(offset + item_size - 1));
                    }
                }
            }
        }
    }

    // **Windows平台安全SIMD复制实现 - 核心错误修复逻辑**
    #[cfg(target_os = "windows")]
    fn windows_safe_vectorized_copy(
        &self,
        src: &[u8],
        dst: &mut [u8],
        indices: &[usize],
        item_size: usize,
    ) {
        let mut dst_offset = 0;

        // **使用安全的标量复制避免SIMD内存访问违规**
        for &idx in indices {
            let src_offset = idx * item_size;

            // **严格边界检查 - 防止内存访问违规**
            if src_offset + item_size > src.len() || dst_offset + item_size > dst.len() {
                match self.error_handler {
                    ErrorHandlingStrategy::Fallback => continue,
                    ErrorHandlingStrategy::Panic => {
                        panic!("Windows safe SIMD: Index out of bounds")
                    }
                    ErrorHandlingStrategy::Ignore => {
                        dst_offset += item_size;
                        continue;
                    }
                }
            }

            // **使用最安全的内存复制方式**
            unsafe {
                std::ptr::copy_nonoverlapping(
                    src.as_ptr().add(src_offset),
                    dst.as_mut_ptr().add(dst_offset),
                    item_size,
                );
            }
            dst_offset += item_size;
        }
    }

    /// 优化的标量复制实现
    pub fn optimized_scalar_copy(
        &self,
        src: &[u8],
        dst: &mut [u8],
        indices: &[usize],
        item_size: usize,
    ) {
        let mut dst_offset = 0;

        for &idx in indices {
            let src_offset = idx * item_size;

            // 边界检查
            if src_offset + item_size > src.len() || dst_offset + item_size > dst.len() {
                if self.error_handler == ErrorHandlingStrategy::Fallback {
                    continue;
                } else if self.error_handler == ErrorHandlingStrategy::Panic {
                    panic!("SIMD processor: Index out of bounds");
                }
                dst_offset += item_size;
                continue;
            }

            unsafe {
                std::ptr::copy_nonoverlapping(
                    src.as_ptr().add(src_offset),
                    dst.as_mut_ptr().add(dst_offset),
                    item_size,
                );
            }
            dst_offset += item_size;
        }
    }

    pub fn apply_simd_optimization(&mut self, _data: Vec<u8>) {
        // TODO: 实现SIMD优化逻辑
    }

    /// 获取SIMD处理器能力
    pub fn get_capabilities(&self) -> SIMDCapabilities {
        SIMDCapabilities {
            supports_sse2: self.supports_sse2,
            supports_avx2: self.supports_avx2,
            supports_avx512: self.supports_avx512,
            alignment_size: self.alignment_size,
            cache_line_size: self.cache_line_size,
            optimal_chunk_size: if self.supports_avx512 {
                64
            } else if self.supports_avx2 {
                32
            } else {
                16
            },
        }
    }

    /// 聚集行数据
    pub fn gather_rows(
        &mut self,
        indices: &[usize],
        data: &[u8],
        row_size: usize,
    ) -> Result<Vec<Vec<u8>>, SIMDError> {
        let mut result = Vec::with_capacity(indices.len());

        for &idx in indices {
            let offset = idx * row_size;
            if offset + row_size <= data.len() {
                result.push(data[offset..offset + row_size].to_vec());
            } else {
                return Err(SIMDError::IndexOutOfBounds);
            }
        }

        Ok(result)
    }
}

/// SIMD能力信息
#[derive(Debug, Clone)]
pub struct SIMDCapabilities {
    pub supports_sse2: bool,
    pub supports_avx2: bool,
    pub supports_avx512: bool,
    pub alignment_size: usize,
    pub cache_line_size: usize,
    pub optimal_chunk_size: usize,
}
