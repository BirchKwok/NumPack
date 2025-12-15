//! x86_64 SIMD实现
//!
//! 专门针对NumPack的x86_64 SIMD优化实现
//! 支持SSE2、AVX2、AVX512指令集

use super::{DataType, NumPackSIMD, SIMDError};

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use std::arch::x86_64::*;

impl NumPackSIMD {
    /// AVX512行拷贝实现 - 针对大数据集的高性能优化
    /// 注意：AVX512在stable Rust中是不稳定特性，暂时禁用
    #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
    feature = "avx512"
    ))]
    pub fn avx512_copy_rows(
        &self,
        src: &[u8],
        dst: &mut [u8],
        indices: &[usize],
        row_size: usize,
    ) -> Result<(), SIMDError> {
        if !self.capabilities.avx512f {
            return self.avx2_copy_rows(src, dst, indices, row_size);
        }

        // 检查边界
        for &idx in indices {
            if idx * row_size + row_size > src.len() {
                return Err(SIMDError::IndexOutOfBounds);
            }
        }

        let mut dst_offset = 0;

        unsafe {
            // AVX512可以处理64字节的数据块
            let avx512_chunk_size = 64;

            for &idx in indices {
                let src_offset = idx * row_size;
                let mut copied = 0;

                // 主循环：使用AVX512处理64字节块
                while copied + avx512_chunk_size <= row_size {
                    let src_ptr = src.as_ptr().add(src_offset + copied);
                    let dst_ptr = dst.as_mut_ptr().add(dst_offset + copied);

                    // 使用AVX512加载和存储（需要nightly Rust）
                    #[cfg(feature = "avx512")]
                    {
                        let data = _mm512_loadu_si512(src_ptr as *const __m512i);
                        _mm512_storeu_si512(dst_ptr as *mut __m512i, data);
                    }

                    copied += avx512_chunk_size;
                }

                // 处理剩余的字节（使用AVX2）
                while copied + 32 <= row_size {
                    let src_ptr = src.as_ptr().add(src_offset + copied);
                    let dst_ptr = dst.as_mut_ptr().add(dst_offset + copied);

                    let data = _mm256_loadu_si256(src_ptr as *const __m256i);
                    _mm256_storeu_si256(dst_ptr as *mut __m256i, data);

                    copied += 32;
                }

                // 处理最后的字节
                while copied < row_size {
                    dst[dst_offset + copied] = src[src_offset + copied];
                    copied += 1;
                }

                dst_offset += row_size;
            }
        }

        Ok(())
    }

    /// AVX512行拷贝实现的回退版本 - 当AVX512不可用时使用AVX2
    #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
    not(feature = "avx512")
    ))]
    pub fn avx512_copy_rows(
        &self,
        src: &[u8],
        dst: &mut [u8],
        indices: &[usize],
        row_size: usize,
    ) -> Result<(), SIMDError> {
        // 回退到AVX2实现
        self.avx2_copy_rows(src, dst, indices, row_size)
    }

    /// AVX2行拷贝实现 - 平衡性能和兼容性
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    pub fn avx2_copy_rows(
        &self,
        src: &[u8],
        dst: &mut [u8],
        indices: &[usize],
        row_size: usize,
    ) -> Result<(), SIMDError> {
        if !self.capabilities.avx2 {
            return self.sse2_copy_rows(src, dst, indices, row_size);
        }

        // 检查边界
        for &idx in indices {
            if idx * row_size + row_size > src.len() {
                return Err(SIMDError::IndexOutOfBounds);
            }
        }

        let mut dst_offset = 0;

        unsafe {
            // AVX2可以处理32字节的数据块
            let avx2_chunk_size = 32;

            for &idx in indices {
                let src_offset = idx * row_size;
                let mut copied = 0;

                // 主循环：使用AVX2处理32字节块
                while copied + avx2_chunk_size <= row_size {
                    let src_ptr = src.as_ptr().add(src_offset + copied);
                    let dst_ptr = dst.as_mut_ptr().add(dst_offset + copied);

                    let data = _mm256_loadu_si256(src_ptr as *const __m256i);
                    _mm256_storeu_si256(dst_ptr as *mut __m256i, data);

                    copied += avx2_chunk_size;
                }

                // 处理剩余的字节（使用SSE2）
                while copied + 16 <= row_size {
                    let src_ptr = src.as_ptr().add(src_offset + copied);
                    let dst_ptr = dst.as_mut_ptr().add(dst_offset + copied);

                    let data = _mm_loadu_si128(src_ptr as *const __m128i);
                    _mm_storeu_si128(dst_ptr as *mut __m128i, data);

                    copied += 16;
                }

                // 处理最后的字节
                while copied < row_size {
                    dst[dst_offset + copied] = src[src_offset + copied];
                    copied += 1;
                }

                dst_offset += row_size;
            }
        }

        Ok(())
    }

    /// SSE2行拷贝实现 - 基础SIMD支持
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    pub fn sse2_copy_rows(
        &self,
        src: &[u8],
        dst: &mut [u8],
        indices: &[usize],
        row_size: usize,
    ) -> Result<(), SIMDError> {
        if !self.capabilities.sse2 {
            return self.scalar_copy_rows(src, dst, indices, row_size);
        }

        // 检查边界
        for &idx in indices {
            if idx * row_size + row_size > src.len() {
                return Err(SIMDError::IndexOutOfBounds);
            }
        }

        let mut dst_offset = 0;

        unsafe {
            // SSE2可以处理16字节的数据块
            let sse2_chunk_size = 16;

            for &idx in indices {
                let src_offset = idx * row_size;
                let mut copied = 0;

                // 主循环：使用SSE2处理16字节块
                while copied + sse2_chunk_size <= row_size {
                    let src_ptr = src.as_ptr().add(src_offset + copied);
                    let dst_ptr = dst.as_mut_ptr().add(dst_offset + copied);

                    let data = _mm_loadu_si128(src_ptr as *const __m128i);
                    _mm_storeu_si128(dst_ptr as *mut __m128i, data);

                    copied += sse2_chunk_size;
                }

                // 处理最后的字节
                while copied < row_size {
                    dst[dst_offset + copied] = src[src_offset + copied];
                    copied += 1;
                }

                dst_offset += row_size;
            }
        }

        Ok(())
    }

    /// SIMD整数到浮点转换 - 专门针对NumPack数据类型优化
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    pub fn simd_int_to_float_convert(
        &self,
        src: &[u8],
        dst: &mut [u8],
        src_dtype: DataType,
        dst_dtype: DataType,
    ) -> Result<(), SIMDError> {
        if !self.capabilities.avx2 {
            return self.scalar_convert(src, dst, src_dtype, dst_dtype);
        }

        use DataType::*;
        match (src_dtype, dst_dtype) {
            (Int32, Float32) => self.avx2_i32_to_f32(src, dst),
            (Int64, Float64) => self.avx2_i64_to_f64(src, dst),
            (Uint32, Float32) => self.avx2_u32_to_f32(src, dst),
            (Int16, Float32) => self.avx2_i16_to_f32(src, dst),
            (Int8, Float32) => self.avx2_i8_to_f32(src, dst),
            _ => self.scalar_convert(src, dst, src_dtype, dst_dtype),
        }
    }

    /// AVX2: int32 到 float32 转换
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn avx2_i32_to_f32(&self, src: &[u8], dst: &mut [u8]) -> Result<(), SIMDError> {
        let src_i32 =
            unsafe { std::slice::from_raw_parts(src.as_ptr() as *const i32, src.len() / 4) };
        let dst_f32 =
            unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut f32, dst.len() / 4) };

        if src_i32.len() != dst_f32.len() {
            return Err(SIMDError::SizeMismatch);
        }

        unsafe {
            let chunks = src_i32.len() / 8; // AVX2处理8个int32
            let remainder = src_i32.len() % 8;

            for i in 0..chunks {
                let offset = i * 8;

                // 加载8个int32
                let int_data = _mm256_loadu_si256(src_i32.as_ptr().add(offset) as *const __m256i);

                // 转换为float32
                let float_data = _mm256_cvtepi32_ps(int_data);

                // 存储结果
                _mm256_storeu_ps(dst_f32.as_mut_ptr().add(offset), float_data);
            }

            // 处理剩余元素
            for i in (chunks * 8)..(chunks * 8 + remainder) {
                dst_f32[i] = src_i32[i] as f32;
            }
        }

        Ok(())
    }

    /// AVX2: int64 到 float64 转换
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn avx2_i64_to_f64(&self, src: &[u8], dst: &mut [u8]) -> Result<(), SIMDError> {
        let src_i64 =
            unsafe { std::slice::from_raw_parts(src.as_ptr() as *const i64, src.len() / 8) };
        let dst_f64 =
            unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut f64, dst.len() / 8) };

        if src_i64.len() != dst_f64.len() {
            return Err(SIMDError::SizeMismatch);
        }

        unsafe {
            let chunks = src_i64.len() / 4; // AVX2处理4个int64
            let remainder = src_i64.len() % 4;

            for i in 0..chunks {
                let offset = i * 4;

                // 对于int64到float64，需要特殊处理，因为AVX2没有直接的指令
                // 我们需要分别处理每对int64
                let low_data = _mm_loadu_si128(src_i64.as_ptr().add(offset) as *const __m128i);
                let high_data = _mm_loadu_si128(src_i64.as_ptr().add(offset + 2) as *const __m128i);

                // 提取并转换
                let val0 = _mm_extract_epi64(low_data, 0) as f64;
                let val1 = _mm_extract_epi64(low_data, 1) as f64;
                let val2 = _mm_extract_epi64(high_data, 0) as f64;
                let val3 = _mm_extract_epi64(high_data, 1) as f64;

                // 组合并存储
                let result = _mm256_set_pd(val3, val2, val1, val0);
                _mm256_storeu_pd(dst_f64.as_mut_ptr().add(offset), result);
            }

            // 处理剩余元素
            for i in (chunks * 4)..(chunks * 4 + remainder) {
                dst_f64[i] = src_i64[i] as f64;
            }
        }

        Ok(())
    }

    /// AVX2: uint32 到 float32 转换
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn avx2_u32_to_f32(&self, src: &[u8], dst: &mut [u8]) -> Result<(), SIMDError> {
        let src_u32 =
            unsafe { std::slice::from_raw_parts(src.as_ptr() as *const u32, src.len() / 4) };
        let dst_f32 =
            unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut f32, dst.len() / 4) };

        if src_u32.len() != dst_f32.len() {
            return Err(SIMDError::SizeMismatch);
        }

        unsafe {
            let chunks = src_u32.len() / 8;
            let remainder = src_u32.len() % 8;

            for i in 0..chunks {
                let offset = i * 8;

                // 加载8个uint32
                let uint_data = _mm256_loadu_si256(src_u32.as_ptr().add(offset) as *const __m256i);

                // uint32到float32需要特殊处理，因为AVX2没有直接指令
                // 使用浮点常数进行转换
                let magic = _mm256_set1_ps(1.0f32 / (1u64 << 32) as f32);

                // 分别处理高位和低位
                let low_part =
                    _mm256_cvtepi32_ps(_mm256_and_si256(uint_data, _mm256_set1_epi32(0x0000FFFF)));
                let high_part = _mm256_cvtepi32_ps(_mm256_srli_epi32(uint_data, 16));

                // 组合结果
                let result = _mm256_add_ps(
                    low_part,
                    _mm256_mul_ps(high_part, _mm256_set1_ps(65536.0f32)),
                );

                _mm256_storeu_ps(dst_f32.as_mut_ptr().add(offset), result);
            }

            // 处理剩余元素
            for i in (chunks * 8)..(chunks * 8 + remainder) {
                dst_f32[i] = src_u32[i] as f32;
            }
        }

        Ok(())
    }

    /// AVX2: int16 到 float32 转换
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn avx2_i16_to_f32(&self, src: &[u8], dst: &mut [u8]) -> Result<(), SIMDError> {
        let src_i16 =
            unsafe { std::slice::from_raw_parts(src.as_ptr() as *const i16, src.len() / 2) };
        let dst_f32 =
            unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut f32, dst.len() / 4) };

        if src_i16.len() != dst_f32.len() {
            return Err(SIMDError::SizeMismatch);
        }

        unsafe {
            let chunks = src_i16.len() / 8;
            let remainder = src_i16.len() % 8;

            for i in 0..chunks {
                let offset = i * 8;

                // 加载8个int16
                let int16_data = _mm_loadu_si128(src_i16.as_ptr().add(offset) as *const __m128i);

                // 扩展为int32
                let int32_data = _mm256_cvtepi16_epi32(int16_data);

                // 转换为float32
                let float_data = _mm256_cvtepi32_ps(int32_data);

                // 存储结果
                _mm256_storeu_ps(dst_f32.as_mut_ptr().add(offset), float_data);
            }

            // 处理剩余元素
            for i in (chunks * 8)..(chunks * 8 + remainder) {
                dst_f32[i] = src_i16[i] as f32;
            }
        }

        Ok(())
    }

    /// AVX2: int8 到 float32 转换
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn avx2_i8_to_f32(&self, src: &[u8], dst: &mut [u8]) -> Result<(), SIMDError> {
        let src_i8 = unsafe { std::slice::from_raw_parts(src.as_ptr() as *const i8, src.len()) };
        let dst_f32 =
            unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut f32, dst.len() / 4) };

        if src_i8.len() != dst_f32.len() {
            return Err(SIMDError::SizeMismatch);
        }

        unsafe {
            let chunks = src_i8.len() / 8;
            let remainder = src_i8.len() % 8;

            for i in 0..chunks {
                let offset = i * 8;

                // 加载8个int8（作为64位）
                let int8_data = _mm_loadl_epi64(src_i8.as_ptr().add(offset) as *const __m128i);

                // 扩展为int16，然后为int32
                let int16_data = _mm_cvtepi8_epi16(int8_data);
                let int32_data = _mm256_cvtepi16_epi32(int16_data);

                // 转换为float32
                let float_data = _mm256_cvtepi32_ps(int32_data);

                // 存储结果
                _mm256_storeu_ps(dst_f32.as_mut_ptr().add(offset), float_data);
            }

            // 处理剩余元素
            for i in (chunks * 8)..(chunks * 8 + remainder) {
                dst_f32[i] = src_i8[i] as f32;
            }
        }

        Ok(())
    }

    /// SIMD浮点到整数转换
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    pub fn simd_float_to_int_convert(
        &self,
        src: &[u8],
        dst: &mut [u8],
        src_dtype: DataType,
        dst_dtype: DataType,
    ) -> Result<(), SIMDError> {
        if !self.capabilities.avx2 {
            return self.scalar_convert(src, dst, src_dtype, dst_dtype);
        }

        use DataType::*;
        match (src_dtype, dst_dtype) {
            (Float32, Int32) => self.avx2_f32_to_i32(src, dst),
            (Float64, Int64) => self.avx2_f64_to_i64(src, dst),
            (Float32, Int16) => self.avx2_f32_to_i16(src, dst),
            (Float32, Int8) => self.avx2_f32_to_i8(src, dst),
            _ => self.scalar_convert(src, dst, src_dtype, dst_dtype),
        }
    }

    /// AVX2: float32 到 int32 转换
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn avx2_f32_to_i32(&self, src: &[u8], dst: &mut [u8]) -> Result<(), SIMDError> {
        let src_f32 =
            unsafe { std::slice::from_raw_parts(src.as_ptr() as *const f32, src.len() / 4) };
        let dst_i32 =
            unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut i32, dst.len() / 4) };

        if src_f32.len() != dst_i32.len() {
            return Err(SIMDError::SizeMismatch);
        }

        unsafe {
            let chunks = src_f32.len() / 8;
            let remainder = src_f32.len() % 8;

            for i in 0..chunks {
                let offset = i * 8;

                // 加载8个float32
                let float_data = _mm256_loadu_ps(src_f32.as_ptr().add(offset));

                // 转换为int32（带截断）
                let int_data = _mm256_cvttps_epi32(float_data);

                // 存储结果
                _mm256_storeu_si256(dst_i32.as_mut_ptr().add(offset) as *mut __m256i, int_data);
            }

            // 处理剩余元素
            for i in (chunks * 8)..(chunks * 8 + remainder) {
                dst_i32[i] = src_f32[i] as i32;
            }
        }

        Ok(())
    }

    /// AVX2: float64 到 int64 转换
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn avx2_f64_to_i64(&self, src: &[u8], dst: &mut [u8]) -> Result<(), SIMDError> {
        let src_f64 =
            unsafe { std::slice::from_raw_parts(src.as_ptr() as *const f64, src.len() / 8) };
        let dst_i64 =
            unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut i64, dst.len() / 8) };

        if src_f64.len() != dst_i64.len() {
            return Err(SIMDError::SizeMismatch);
        }

        // AVX2没有直接的float64到int64转换，需要逐个处理
        for i in 0..src_f64.len() {
            dst_i64[i] = src_f64[i] as i64;
        }

        Ok(())
    }

    /// AVX2: float32 到 int16 转换
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn avx2_f32_to_i16(&self, src: &[u8], dst: &mut [u8]) -> Result<(), SIMDError> {
        let src_f32 =
            unsafe { std::slice::from_raw_parts(src.as_ptr() as *const f32, src.len() / 4) };
        let dst_i16 =
            unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut i16, dst.len() / 2) };

        if src_f32.len() != dst_i16.len() {
            return Err(SIMDError::SizeMismatch);
        }

        unsafe {
            let chunks = src_f32.len() / 8;
            let remainder = src_f32.len() % 8;

            for i in 0..chunks {
                let offset = i * 8;

                // 加载8个float32
                let float_data = _mm256_loadu_ps(src_f32.as_ptr().add(offset));

                // 转换为int32，然后打包为int16
                let int32_data = _mm256_cvttps_epi32(float_data);
                let int16_data = _mm256_packs_epi32(int32_data, int32_data);

                // 提取并存储
                _mm_storeu_si128(
                    dst_i16.as_mut_ptr().add(offset) as *mut __m128i,
                    _mm256_extracti128_si256(int16_data, 0),
                );
            }

            // 处理剩余元素
            for i in (chunks * 8)..(chunks * 8 + remainder) {
                dst_i16[i] = src_f32[i] as i16;
            }
        }

        Ok(())
    }

    /// AVX2: float32 到 int8 转换
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn avx2_f32_to_i8(&self, src: &[u8], dst: &mut [u8]) -> Result<(), SIMDError> {
        let src_f32 =
            unsafe { std::slice::from_raw_parts(src.as_ptr() as *const f32, src.len() / 4) };
        let dst_i8 =
            unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut i8, dst.len()) };

        if src_f32.len() != dst_i8.len() {
            return Err(SIMDError::SizeMismatch);
        }

        unsafe {
            let chunks = src_f32.len() / 8;
            let remainder = src_f32.len() % 8;

            for i in 0..chunks {
                let offset = i * 8;

                // 加载8个float32
                let float_data = _mm256_loadu_ps(src_f32.as_ptr().add(offset));

                // 转换为int32，然后打包为int16，再打包为int8
                let int32_data = _mm256_cvttps_epi32(float_data);
                let int16_data = _mm256_packs_epi32(int32_data, int32_data);
                let int8_data = _mm_packs_epi16(
                    _mm256_extracti128_si256(int16_data, 0),
                    _mm256_extracti128_si256(int16_data, 0),
                );

                // 存储8个字节
                _mm_storel_epi64(dst_i8.as_mut_ptr().add(offset) as *mut __m128i, int8_data);
            }

            // 处理剩余元素
            for i in (chunks * 8)..(chunks * 8 + remainder) {
                dst_i8[i] = src_f32[i] as i8;
            }
        }

        Ok(())
    }

    /// SIMD类型扩展转换（如int8到int32）
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    pub fn simd_widening_convert(
        &self,
        src: &[u8],
        dst: &mut [u8],
        src_dtype: DataType,
        dst_dtype: DataType,
    ) -> Result<(), SIMDError> {
        // 实现类型扩展转换的SIMD版本
        // 这里为了简化，先使用标量版本
        self.scalar_convert(src, dst, src_dtype, dst_dtype)
    }

    /// SIMD类型收窄转换（如int32到int8）
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    pub fn simd_narrowing_convert(
        &self,
        src: &[u8],
        dst: &mut [u8],
        src_dtype: DataType,
        dst_dtype: DataType,
    ) -> Result<(), SIMDError> {
        // 实现类型收窄转换的SIMD版本
        // 这里为了简化，先使用标量版本
        self.scalar_convert(src, dst, src_dtype, dst_dtype)
    }
}

// 为非x86架构提供空实现
#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
impl NumPackSIMD {
    pub fn avx512_copy_rows(
        &self,
        src: &[u8],
        dst: &mut [u8],
        indices: &[usize],
        row_size: usize,
    ) -> Result<(), SIMDError> {
        self.scalar_copy_rows(src, dst, indices, row_size)
    }

    pub fn avx2_copy_rows(
        &self,
        src: &[u8],
        dst: &mut [u8],
        indices: &[usize],
        row_size: usize,
    ) -> Result<(), SIMDError> {
        self.scalar_copy_rows(src, dst, indices, row_size)
    }

    pub fn sse2_copy_rows(
        &self,
        src: &[u8],
        dst: &mut [u8],
        indices: &[usize],
        row_size: usize,
    ) -> Result<(), SIMDError> {
        self.scalar_copy_rows(src, dst, indices, row_size)
    }

    pub fn simd_int_to_float_convert(
        &self,
        src: &[u8],
        dst: &mut [u8],
        src_dtype: DataType,
        dst_dtype: DataType,
    ) -> Result<(), SIMDError> {
        self.scalar_convert(src, dst, src_dtype, dst_dtype)
    }

    pub fn simd_float_to_int_convert(
        &self,
        src: &[u8],
        dst: &mut [u8],
        src_dtype: DataType,
        dst_dtype: DataType,
    ) -> Result<(), SIMDError> {
        self.scalar_convert(src, dst, src_dtype, dst_dtype)
    }

    pub fn simd_widening_convert(
        &self,
        src: &[u8],
        dst: &mut [u8],
        src_dtype: DataType,
        dst_dtype: DataType,
    ) -> Result<(), SIMDError> {
        self.scalar_convert(src, dst, src_dtype, dst_dtype)
    }

    pub fn simd_narrowing_convert(
        &self,
        src: &[u8],
        dst: &mut [u8],
        src_dtype: DataType,
        dst_dtype: DataType,
    ) -> Result<(), SIMDError> {
        self.scalar_convert(src, dst, src_dtype, dst_dtype)
    }
}
