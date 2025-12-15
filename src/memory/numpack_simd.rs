//! NumPack专用多平台SIMD优化模块
//!
//! 本模块专门针对NumPack的使用场景设计，提供：
//! 1. 多平台支持：x86_64 (SSE2/AVX2/AVX512) 和 ARM (NEON)
//! 2. 数据类型特异化：针对NumPack支持的14种数据类型专门优化
//! 3. 访问模式感知：针对NPK文件格式的行主序布局优化
//! 4. FFI友好：批量处理减少Python-Rust边界开销

use crate::core::metadata::DataType;

/// CPU特性支持信息
#[derive(Debug, Clone)]
pub struct CpuCapabilities {
    // x86_64 特性
    pub sse2: bool,
    pub avx2: bool,
    pub avx512f: bool,
    pub avx512vl: bool,
    // ARM 特性
    pub neon: bool,
    // 通用特性
    pub cache_line_size: usize,
    pub preferred_vector_width: usize,
}

impl CpuCapabilities {
    pub fn detect() -> Self {
        Self {
            sse2: Self::detect_sse2(),
            avx2: Self::detect_avx2(),
            avx512f: Self::detect_avx512f(),
            avx512vl: Self::detect_avx512vl(),
            neon: Self::detect_neon(),
            cache_line_size: Self::detect_cache_line_size(),
            preferred_vector_width: Self::detect_preferred_vector_width(),
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn detect_sse2() -> bool {
        is_x86_feature_detected!("sse2")
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn detect_avx2() -> bool {
        is_x86_feature_detected!("avx2")
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn detect_avx512f() -> bool {
        is_x86_feature_detected!("avx512f")
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn detect_avx512vl() -> bool {
        is_x86_feature_detected!("avx512vl")
    }

    #[cfg(target_arch = "aarch64")]
    fn detect_neon() -> bool {
        std::arch::is_aarch64_feature_detected!("neon")
    }

    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    fn detect_sse2() -> bool {
        false
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    fn detect_avx2() -> bool {
        false
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    fn detect_avx512f() -> bool {
        false
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    fn detect_avx512vl() -> bool {
        false
    }
    #[cfg(not(target_arch = "aarch64"))]
    fn detect_neon() -> bool {
        false
    }

    fn detect_cache_line_size() -> usize {
        // 大多数现代CPU的缓存行大小为64字节
        64
    }

    fn detect_preferred_vector_width() -> usize {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("avx512f") {
                64
            } else if is_x86_feature_detected!("avx2") {
                32
            } else if is_x86_feature_detected!("sse2") {
                16
            } else {
                8
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                16
            } else {
                8
            }
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
        {
            8
        }
    }
}

/// NumPack数据类型专用的SIMD操作集合
#[derive(Debug)]
pub struct NumPackSIMD {
    pub capabilities: CpuCapabilities,
}

impl NumPackSIMD {
    pub fn new() -> Self {
        Self {
            capabilities: CpuCapabilities::detect(),
        }
    }

    /// 获取指定数据类型的最优SIMD策略
    pub fn get_optimal_strategy(&self, dtype: DataType, data_size: usize) -> SIMDStrategy {
        let element_size = dtype.size_bytes() as usize;
        let vector_width = self.capabilities.preferred_vector_width;
        let elements_per_vector = vector_width / element_size;

        // 根据数据类型、数据大小和CPU能力选择最优策略
        match dtype {
            DataType::Bool => self.choose_bool_strategy(data_size, elements_per_vector),
            DataType::Uint8 | DataType::Int8 => {
                self.choose_byte_strategy(data_size, elements_per_vector)
            }
            DataType::Uint16 | DataType::Int16 | DataType::Float16 => {
                self.choose_word_strategy(data_size, elements_per_vector)
            }
            DataType::Uint32 | DataType::Int32 | DataType::Float32 => {
                self.choose_dword_strategy(data_size, elements_per_vector)
            }
            DataType::Uint64 | DataType::Int64 | DataType::Float64 => {
                self.choose_qword_strategy(data_size, elements_per_vector)
            }
            DataType::Complex64 | DataType::Complex128 => {
                self.choose_complex_strategy(dtype, data_size)
            }
        }
    }

    fn choose_bool_strategy(&self, data_size: usize, elements_per_vector: usize) -> SIMDStrategy {
        if data_size >= 64 && self.capabilities.avx512f {
            SIMDStrategy::AVX512Bool
        } else if data_size >= 32 && self.capabilities.avx2 {
            SIMDStrategy::AVX2Bool
        } else if data_size >= 16 && (self.capabilities.sse2 || self.capabilities.neon) {
            SIMDStrategy::PackedBool
        } else {
            SIMDStrategy::Scalar
        }
    }

    fn choose_byte_strategy(&self, data_size: usize, elements_per_vector: usize) -> SIMDStrategy {
        if data_size >= elements_per_vector * 4 {
            if self.capabilities.avx512f {
                SIMDStrategy::AVX512Byte
            } else if self.capabilities.avx2 {
                SIMDStrategy::AVX2Byte
            } else if self.capabilities.sse2 {
                SIMDStrategy::SSE2Byte
            } else if self.capabilities.neon {
                SIMDStrategy::NEONByte
            } else {
                SIMDStrategy::Scalar
            }
        } else {
            SIMDStrategy::Scalar
        }
    }

    fn choose_word_strategy(&self, data_size: usize, elements_per_vector: usize) -> SIMDStrategy {
        if data_size >= elements_per_vector * 4 {
            if self.capabilities.avx512f {
                SIMDStrategy::AVX512Word
            } else if self.capabilities.avx2 {
                SIMDStrategy::AVX2Word
            } else if self.capabilities.sse2 {
                SIMDStrategy::SSE2Word
            } else if self.capabilities.neon {
                SIMDStrategy::NEONWord
            } else {
                SIMDStrategy::Scalar
            }
        } else {
            SIMDStrategy::Scalar
        }
    }

    fn choose_dword_strategy(&self, data_size: usize, elements_per_vector: usize) -> SIMDStrategy {
        if data_size >= elements_per_vector * 4 {
            if self.capabilities.avx512f {
                SIMDStrategy::AVX512DWord
            } else if self.capabilities.avx2 {
                SIMDStrategy::AVX2DWord
            } else if self.capabilities.sse2 {
                SIMDStrategy::SSE2DWord
            } else if self.capabilities.neon {
                SIMDStrategy::NEONDWord
            } else {
                SIMDStrategy::Scalar
            }
        } else {
            SIMDStrategy::Scalar
        }
    }

    fn choose_qword_strategy(&self, data_size: usize, elements_per_vector: usize) -> SIMDStrategy {
        if data_size >= elements_per_vector * 4 {
            if self.capabilities.avx512f {
                SIMDStrategy::AVX512QWord
            } else if self.capabilities.avx2 {
                SIMDStrategy::AVX2QWord
            } else if self.capabilities.sse2 {
                SIMDStrategy::SSE2QWord
            } else if self.capabilities.neon {
                SIMDStrategy::NEONQWord
            } else {
                SIMDStrategy::Scalar
            }
        } else {
            SIMDStrategy::Scalar
        }
    }

    fn choose_complex_strategy(&self, dtype: DataType, data_size: usize) -> SIMDStrategy {
        match dtype {
            DataType::Complex64 => {
                if data_size >= 8 && self.capabilities.avx2 {
                    SIMDStrategy::AVX2Complex64
                } else if data_size >= 4 && self.capabilities.sse2 {
                    SIMDStrategy::SSE2Complex64
                } else {
                    SIMDStrategy::Scalar
                }
            }
            DataType::Complex128 => {
                if data_size >= 4 && self.capabilities.avx2 {
                    SIMDStrategy::AVX2Complex128
                } else if data_size >= 2 && self.capabilities.sse2 {
                    SIMDStrategy::SSE2Complex128
                } else {
                    SIMDStrategy::Scalar
                }
            }
            _ => SIMDStrategy::Scalar,
        }
    }

    /// 高性能行拷贝 - 专门针对NPK文件格式的行主序布局
    pub fn copy_rows(
        &self,
        src: &[u8],
        dst: &mut [u8],
        indices: &[usize],
        row_size: usize,
        dtype: DataType,
    ) -> Result<(), SIMDError> {
        let strategy = self.get_optimal_strategy(dtype, indices.len() * row_size);

        match strategy {
            SIMDStrategy::AVX512Byte
            | SIMDStrategy::AVX512Word
            | SIMDStrategy::AVX512DWord
            | SIMDStrategy::AVX512QWord => self.avx512_copy_rows(src, dst, indices, row_size),
            SIMDStrategy::AVX2Byte
            | SIMDStrategy::AVX2Word
            | SIMDStrategy::AVX2DWord
            | SIMDStrategy::AVX2QWord => self.avx2_copy_rows(src, dst, indices, row_size),
            SIMDStrategy::SSE2Byte
            | SIMDStrategy::SSE2Word
            | SIMDStrategy::SSE2DWord
            | SIMDStrategy::SSE2QWord => self.sse2_copy_rows(src, dst, indices, row_size),
            SIMDStrategy::NEONByte
            | SIMDStrategy::NEONWord
            | SIMDStrategy::NEONDWord
            | SIMDStrategy::NEONQWord => self.neon_copy_rows(src, dst, indices, row_size),
            _ => self.scalar_copy_rows(src, dst, indices, row_size),
        }
    }

    /// 批量数据类型转换 - 支持NPK文件格式的14种数据类型
    pub fn batch_convert(
        &self,
        src: &[u8],
        dst: &mut [u8],
        src_dtype: DataType,
        dst_dtype: DataType,
    ) -> Result<(), SIMDError> {
        if src_dtype == dst_dtype {
            // 相同类型，直接拷贝
            if src.len() != dst.len() {
                return Err(SIMDError::SizeMismatch);
            }
            dst.copy_from_slice(src);
            return Ok(());
        }

        // 选择合适的转换策略
        let strategy = self.get_conversion_strategy(src_dtype, dst_dtype, src.len());

        match strategy {
            ConversionStrategy::SIMDIntToFloat => {
                self.simd_int_to_float_convert(src, dst, src_dtype, dst_dtype)
            }
            ConversionStrategy::SIMDFloatToInt => {
                self.simd_float_to_int_convert(src, dst, src_dtype, dst_dtype)
            }
            ConversionStrategy::SIMDWidening => {
                self.simd_widening_convert(src, dst, src_dtype, dst_dtype)
            }
            ConversionStrategy::SIMDNarrowing => {
                self.simd_narrowing_convert(src, dst, src_dtype, dst_dtype)
            }
            ConversionStrategy::Scalar => self.scalar_convert(src, dst, src_dtype, dst_dtype),
        }
    }

    fn get_conversion_strategy(
        &self,
        src_dtype: DataType,
        dst_dtype: DataType,
        size: usize,
    ) -> ConversionStrategy {
        // 如果数据量太小，使用标量版本
        if size < 64 {
            return ConversionStrategy::Scalar;
        }

        use DataType::*;
        match (src_dtype, dst_dtype) {
            // 整数到浮点
            (
                Int8 | Int16 | Int32 | Int64 | Uint8 | Uint16 | Uint32 | Uint64,
                Float32 | Float64,
            ) => {
                if self.capabilities.avx2 {
                    ConversionStrategy::SIMDIntToFloat
                } else {
                    ConversionStrategy::Scalar
                }
            }
            // 浮点到整数
            (
                Float32 | Float64,
                Int8 | Int16 | Int32 | Int64 | Uint8 | Uint16 | Uint32 | Uint64,
            ) => {
                if self.capabilities.avx2 {
                    ConversionStrategy::SIMDFloatToInt
                } else {
                    ConversionStrategy::Scalar
                }
            }
            // 类型扩展（如int8到int32）
            (Int8, Int16 | Int32 | Int64)
            | (Int16, Int32 | Int64)
            | (Int32, Int64)
            | (Uint8, Uint16 | Uint32 | Uint64)
            | (Uint16, Uint32 | Uint64)
            | (Uint32, Uint64) => {
                if self.capabilities.avx2 {
                    ConversionStrategy::SIMDWidening
                } else {
                    ConversionStrategy::Scalar
                }
            }
            // 类型收窄（如int32到int8）
            (Int16 | Int32 | Int64, Int8)
            | (Int32 | Int64, Int16)
            | (Int64, Int32)
            | (Uint16 | Uint32 | Uint64, Uint8)
            | (Uint32 | Uint64, Uint16)
            | (Uint64, Uint32) => {
                if self.capabilities.avx2 {
                    ConversionStrategy::SIMDNarrowing
                } else {
                    ConversionStrategy::Scalar
                }
            }
            _ => ConversionStrategy::Scalar,
        }
    }
}

/// SIMD策略枚举 - 精确匹配不同的指令集和数据类型组合
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SIMDStrategy {
    // 标量回退
    Scalar,

    // 布尔类型专用
    AVX512Bool,
    AVX2Bool,
    PackedBool,

    // 字节类型 (u8/i8)
    AVX512Byte,
    AVX2Byte,
    SSE2Byte,
    NEONByte,

    // 字类型 (u16/i16/f16)
    AVX512Word,
    AVX2Word,
    SSE2Word,
    NEONWord,

    // 双字类型 (u32/i32/f32)
    AVX512DWord,
    AVX2DWord,
    SSE2DWord,
    NEONDWord,

    // 四字类型 (u64/i64/f64)
    AVX512QWord,
    AVX2QWord,
    SSE2QWord,
    NEONQWord,

    // 复数类型
    AVX2Complex64,
    SSE2Complex64,
    AVX2Complex128,
    SSE2Complex128,
}

/// 数据类型转换策略
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConversionStrategy {
    Scalar,
    SIMDIntToFloat,
    SIMDFloatToInt,
    SIMDWidening,
    SIMDNarrowing,
}

/// SIMD错误类型
#[derive(Debug)]
pub enum SIMDError {
    UnsupportedOperation,
    SizeMismatch,
    AlignmentError,
    IndexOutOfBounds,
    ConversionError,
}

impl std::fmt::Display for SIMDError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            SIMDError::UnsupportedOperation => write!(f, "Unsupported SIMD operation"),
            SIMDError::SizeMismatch => write!(f, "Data size mismatch"),
            SIMDError::AlignmentError => write!(f, "Memory alignment error"),
            SIMDError::IndexOutOfBounds => write!(f, "Index out of bounds"),
            SIMDError::ConversionError => write!(f, "Data type conversion error"),
        }
    }
}

impl std::error::Error for SIMDError {}

// 平台特定实现模块
mod access_pattern_simd;
mod arm_impl;
mod benchmark;
mod scalar_impl;
mod x86_impl;

// 测试模块
#[cfg(test)]
mod test_simd;

// 重新导出实现
#[allow(unused_imports)]
pub use access_pattern_simd::*;
#[allow(unused_imports)]
pub use arm_impl::*;
#[allow(unused_imports)]
pub use benchmark::*;
#[allow(unused_imports)]
pub use scalar_impl::*;
#[allow(unused_imports)]
pub use x86_impl::*;
