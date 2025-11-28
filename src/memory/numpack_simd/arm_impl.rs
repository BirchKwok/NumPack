//! ARM NEON SIMD实现
//!
//! 专门针对NumPack的ARM NEON SIMD优化实现
//! 支持AArch64 NEON指令集

use super::{DataType, NumPackSIMD, SIMDError};

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

impl NumPackSIMD {
    /// NEON行拷贝实现 - 针对ARM架构优化
    #[cfg(target_arch = "aarch64")]
    pub fn neon_copy_rows(
        &self,
        src: &[u8],
        dst: &mut [u8],
        indices: &[usize],
        row_size: usize,
    ) -> Result<(), SIMDError> {
        if !self.capabilities.neon {
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
            // NEON可以处理16字节的数据块
            let neon_chunk_size = 16;

            for &idx in indices {
                let src_offset = idx * row_size;
                let mut copied = 0;

                // 主循环：使用NEON处理16字节块
                while copied + neon_chunk_size <= row_size {
                    let src_ptr = src.as_ptr().add(src_offset + copied);
                    let dst_ptr = dst.as_mut_ptr().add(dst_offset + copied);

                    // 使用NEON加载和存储
                    let data = vld1q_u8(src_ptr);
                    vst1q_u8(dst_ptr, data);

                    copied += neon_chunk_size;
                }

                // 处理剩余的字节（8字节块）
                while copied + 8 <= row_size {
                    let src_ptr = src.as_ptr().add(src_offset + copied);
                    let dst_ptr = dst.as_mut_ptr().add(dst_offset + copied);

                    let data = vld1_u8(src_ptr);
                    vst1_u8(dst_ptr, data);

                    copied += 8;
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

    /// NEON整数到浮点转换
    #[cfg(target_arch = "aarch64")]
    pub fn neon_int_to_float_convert(
        &self,
        src: &[u8],
        dst: &mut [u8],
        src_dtype: DataType,
        dst_dtype: DataType,
    ) -> Result<(), SIMDError> {
        if !self.capabilities.neon {
            return self.scalar_convert(src, dst, src_dtype, dst_dtype);
        }

        use DataType::*;
        match (src_dtype, dst_dtype) {
            (Int32, Float32) => self.neon_i32_to_f32(src, dst),
            (Int16, Float32) => self.neon_i16_to_f32(src, dst),
            (Int8, Float32) => self.neon_i8_to_f32(src, dst),
            _ => self.scalar_convert(src, dst, src_dtype, dst_dtype),
        }
    }

    /// NEON: int32 到 float32 转换
    #[cfg(target_arch = "aarch64")]
    fn neon_i32_to_f32(&self, src: &[u8], dst: &mut [u8]) -> Result<(), SIMDError> {
        let src_i32 =
            unsafe { std::slice::from_raw_parts(src.as_ptr() as *const i32, src.len() / 4) };
        let dst_f32 =
            unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut f32, dst.len() / 4) };

        if src_i32.len() != dst_f32.len() {
            return Err(SIMDError::SizeMismatch);
        }

        unsafe {
            let chunks = src_i32.len() / 4; // NEON处理4个int32
            let remainder = src_i32.len() % 4;

            for i in 0..chunks {
                let offset = i * 4;

                // 加载4个int32
                let int_data = vld1q_s32(src_i32.as_ptr().add(offset));

                // 转换为float32
                let float_data = vcvtq_f32_s32(int_data);

                // 存储结果
                vst1q_f32(dst_f32.as_mut_ptr().add(offset), float_data);
            }

            // 处理剩余元素
            for i in (chunks * 4)..(chunks * 4 + remainder) {
                dst_f32[i] = src_i32[i] as f32;
            }
        }

        Ok(())
    }

    /// NEON: int16 到 float32 转换
    #[cfg(target_arch = "aarch64")]
    fn neon_i16_to_f32(&self, src: &[u8], dst: &mut [u8]) -> Result<(), SIMDError> {
        let src_i16 =
            unsafe { std::slice::from_raw_parts(src.as_ptr() as *const i16, src.len() / 2) };
        let dst_f32 =
            unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut f32, dst.len() / 4) };

        if src_i16.len() != dst_f32.len() {
            return Err(SIMDError::SizeMismatch);
        }

        unsafe {
            let chunks = src_i16.len() / 4;
            let remainder = src_i16.len() % 4;

            for i in 0..chunks {
                let offset = i * 4;

                // 加载4个int16
                let int16_data = vld1_s16(src_i16.as_ptr().add(offset));

                // 扩展为int32
                let int32_data = vmovl_s16(int16_data);

                // 转换为float32
                let float_data = vcvtq_f32_s32(int32_data);

                // 存储结果
                vst1q_f32(dst_f32.as_mut_ptr().add(offset), float_data);
            }

            // 处理剩余元素
            for i in (chunks * 4)..(chunks * 4 + remainder) {
                dst_f32[i] = src_i16[i] as f32;
            }
        }

        Ok(())
    }

    /// NEON: int8 到 float32 转换
    #[cfg(target_arch = "aarch64")]
    fn neon_i8_to_f32(&self, src: &[u8], dst: &mut [u8]) -> Result<(), SIMDError> {
        let src_i8 = unsafe { std::slice::from_raw_parts(src.as_ptr() as *const i8, src.len()) };
        let dst_f32 =
            unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut f32, dst.len() / 4) };

        if src_i8.len() != dst_f32.len() {
            return Err(SIMDError::SizeMismatch);
        }

        unsafe {
            let chunks = src_i8.len() / 4;
            let remainder = src_i8.len() % 4;

            for i in 0..chunks {
                let offset = i * 4;

                // 手动构建int8x8_t，只使用前4个元素
                let mut temp = [0i8; 8];
                for j in 0..4 {
                    temp[j] = src_i8[offset + j];
                }
                let int8_data = vld1_s8(temp.as_ptr());

                // 扩展为int16，然后为int32
                let int16_data = vmovl_s8(int8_data);
                let int32_data = vmovl_s16(vget_low_s16(int16_data));

                // 转换为float32
                let float_data = vcvtq_f32_s32(int32_data);

                // 存储结果
                vst1q_f32(dst_f32.as_mut_ptr().add(offset), float_data);
            }

            // 处理剩余元素
            for i in (chunks * 4)..(chunks * 4 + remainder) {
                dst_f32[i] = src_i8[i] as f32;
            }
        }

        Ok(())
    }

    /// NEON浮点到整数转换
    #[cfg(target_arch = "aarch64")]
    pub fn neon_float_to_int_convert(
        &self,
        src: &[u8],
        dst: &mut [u8],
        src_dtype: DataType,
        dst_dtype: DataType,
    ) -> Result<(), SIMDError> {
        if !self.capabilities.neon {
            return self.scalar_convert(src, dst, src_dtype, dst_dtype);
        }

        use DataType::*;
        match (src_dtype, dst_dtype) {
            (Float32, Int32) => self.neon_f32_to_i32(src, dst),
            (Float32, Int16) => self.neon_f32_to_i16(src, dst),
            (Float32, Int8) => self.neon_f32_to_i8(src, dst),
            _ => self.scalar_convert(src, dst, src_dtype, dst_dtype),
        }
    }

    /// NEON: float32 到 int32 转换
    #[cfg(target_arch = "aarch64")]
    fn neon_f32_to_i32(&self, src: &[u8], dst: &mut [u8]) -> Result<(), SIMDError> {
        let src_f32 =
            unsafe { std::slice::from_raw_parts(src.as_ptr() as *const f32, src.len() / 4) };
        let dst_i32 =
            unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut i32, dst.len() / 4) };

        if src_f32.len() != dst_i32.len() {
            return Err(SIMDError::SizeMismatch);
        }

        unsafe {
            let chunks = src_f32.len() / 4;
            let remainder = src_f32.len() % 4;

            for i in 0..chunks {
                let offset = i * 4;

                // 加载4个float32
                let float_data = vld1q_f32(src_f32.as_ptr().add(offset));

                // 转换为int32（带截断）
                let int_data = vcvtq_s32_f32(float_data);

                // 存储结果
                vst1q_s32(dst_i32.as_mut_ptr().add(offset), int_data);
            }

            // 处理剩余元素
            for i in (chunks * 4)..(chunks * 4 + remainder) {
                dst_i32[i] = src_f32[i] as i32;
            }
        }

        Ok(())
    }

    /// NEON: float32 到 int16 转换
    #[cfg(target_arch = "aarch64")]
    fn neon_f32_to_i16(&self, src: &[u8], dst: &mut [u8]) -> Result<(), SIMDError> {
        let src_f32 =
            unsafe { std::slice::from_raw_parts(src.as_ptr() as *const f32, src.len() / 4) };
        let dst_i16 =
            unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut i16, dst.len() / 2) };

        if src_f32.len() != dst_i16.len() {
            return Err(SIMDError::SizeMismatch);
        }

        unsafe {
            let chunks = src_f32.len() / 4;
            let remainder = src_f32.len() % 4;

            for i in 0..chunks {
                let offset = i * 4;

                // 加载4个float32
                let float_data = vld1q_f32(src_f32.as_ptr().add(offset));

                // 转换为int32，然后收窄为int16
                let int32_data = vcvtq_s32_f32(float_data);
                let int16_data = vmovn_s32(int32_data);

                // 存储结果
                vst1_s16(dst_i16.as_mut_ptr().add(offset), int16_data);
            }

            // 处理剩余元素
            for i in (chunks * 4)..(chunks * 4 + remainder) {
                dst_i16[i] = src_f32[i] as i16;
            }
        }

        Ok(())
    }

    /// NEON: float32 到 int8 转换
    #[cfg(target_arch = "aarch64")]
    fn neon_f32_to_i8(&self, src: &[u8], dst: &mut [u8]) -> Result<(), SIMDError> {
        let src_f32 =
            unsafe { std::slice::from_raw_parts(src.as_ptr() as *const f32, src.len() / 4) };
        let dst_i8 =
            unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut i8, dst.len()) };

        if src_f32.len() != dst_i8.len() {
            return Err(SIMDError::SizeMismatch);
        }

        unsafe {
            let chunks = src_f32.len() / 4;
            let remainder = src_f32.len() % 4;

            for i in 0..chunks {
                let offset = i * 4;

                // 加载4个float32
                let float_data = vld1q_f32(src_f32.as_ptr().add(offset));

                // 转换为int32，然后收窄为int16，再收窄为int8
                let int32_data = vcvtq_s32_f32(float_data);
                let int16_data = vmovn_s32(int32_data);
                let int8_data = vmovn_s16(vcombine_s16(int16_data, int16_data));

                // 手动存储前4个字节
                let temp: [i8; 8] = std::mem::transmute(int8_data);
                for j in 0..4 {
                    dst_i8[offset + j] = temp[j];
                }
            }

            // 处理剩余元素
            for i in (chunks * 4)..(chunks * 4 + remainder) {
                dst_i8[i] = src_f32[i] as i8;
            }
        }

        Ok(())
    }

    /// NEON数学运算 - 向量加法
    #[cfg(target_arch = "aarch64")]
    pub fn neon_vector_add_f32(
        &self,
        a: &[f32],
        b: &[f32],
        result: &mut [f32],
    ) -> Result<(), SIMDError> {
        if a.len() != b.len() || a.len() != result.len() {
            return Err(SIMDError::SizeMismatch);
        }

        if !self.capabilities.neon {
            for i in 0..a.len() {
                result[i] = a[i] + b[i];
            }
            return Ok(());
        }

        unsafe {
            let chunks = a.len() / 4;
            let remainder = a.len() % 4;

            for i in 0..chunks {
                let offset = i * 4;

                let vec_a = vld1q_f32(a.as_ptr().add(offset));
                let vec_b = vld1q_f32(b.as_ptr().add(offset));
                let sum = vaddq_f32(vec_a, vec_b);

                vst1q_f32(result.as_mut_ptr().add(offset), sum);
            }

            // 处理剩余元素
            for i in (chunks * 4)..(chunks * 4 + remainder) {
                result[i] = a[i] + b[i];
            }
        }

        Ok(())
    }

    /// NEON数学运算 - 向量乘法
    #[cfg(target_arch = "aarch64")]
    pub fn neon_vector_mul_f32(
        &self,
        a: &[f32],
        b: &[f32],
        result: &mut [f32],
    ) -> Result<(), SIMDError> {
        if a.len() != b.len() || a.len() != result.len() {
            return Err(SIMDError::SizeMismatch);
        }

        if !self.capabilities.neon {
            for i in 0..a.len() {
                result[i] = a[i] * b[i];
            }
            return Ok(());
        }

        unsafe {
            let chunks = a.len() / 4;
            let remainder = a.len() % 4;

            for i in 0..chunks {
                let offset = i * 4;

                let vec_a = vld1q_f32(a.as_ptr().add(offset));
                let vec_b = vld1q_f32(b.as_ptr().add(offset));
                let product = vmulq_f32(vec_a, vec_b);

                vst1q_f32(result.as_mut_ptr().add(offset), product);
            }

            // 处理剩余元素
            for i in (chunks * 4)..(chunks * 4 + remainder) {
                result[i] = a[i] * b[i];
            }
        }

        Ok(())
    }

    /// NEON数学运算 - 向量点积
    #[cfg(target_arch = "aarch64")]
    pub fn neon_dot_product_f32(&self, a: &[f32], b: &[f32]) -> Result<f32, SIMDError> {
        if a.len() != b.len() {
            return Err(SIMDError::SizeMismatch);
        }

        if !self.capabilities.neon {
            let mut sum = 0.0f32;
            for i in 0..a.len() {
                sum += a[i] * b[i];
            }
            return Ok(sum);
        }

        unsafe {
            let chunks = a.len() / 4;
            let remainder = a.len() % 4;

            let mut sum_vec = vdupq_n_f32(0.0);

            for i in 0..chunks {
                let offset = i * 4;

                let vec_a = vld1q_f32(a.as_ptr().add(offset));
                let vec_b = vld1q_f32(b.as_ptr().add(offset));
                let product = vmulq_f32(vec_a, vec_b);
                sum_vec = vaddq_f32(sum_vec, product);
            }

            // 水平求和
            let sum_low = vget_low_f32(sum_vec);
            let sum_high = vget_high_f32(sum_vec);
            let sum_pair = vadd_f32(sum_low, sum_high);
            let sum_final = vpadd_f32(sum_pair, sum_pair);

            let mut result = vget_lane_f32(sum_final, 0);

            // 处理剩余元素
            for i in (chunks * 4)..(chunks * 4 + remainder) {
                result += a[i] * b[i];
            }

            Ok(result)
        }
    }
}

// 为非ARM架构提供空实现
#[cfg(not(target_arch = "aarch64"))]
impl NumPackSIMD {
    pub fn neon_copy_rows(
        &self,
        src: &[u8],
        dst: &mut [u8],
        indices: &[usize],
        row_size: usize,
    ) -> Result<(), SIMDError> {
        self.scalar_copy_rows(src, dst, indices, row_size)
    }

    pub fn neon_int_to_float_convert(
        &self,
        src: &[u8],
        dst: &mut [u8],
        src_dtype: DataType,
        dst_dtype: DataType,
    ) -> Result<(), SIMDError> {
        self.scalar_convert(src, dst, src_dtype, dst_dtype)
    }

    pub fn neon_float_to_int_convert(
        &self,
        src: &[u8],
        dst: &mut [u8],
        src_dtype: DataType,
        dst_dtype: DataType,
    ) -> Result<(), SIMDError> {
        self.scalar_convert(src, dst, src_dtype, dst_dtype)
    }

    pub fn neon_vector_add_f32(
        &self,
        a: &[f32],
        b: &[f32],
        result: &mut [f32],
    ) -> Result<(), SIMDError> {
        if a.len() != b.len() || a.len() != result.len() {
            return Err(SIMDError::SizeMismatch);
        }
        for i in 0..a.len() {
            result[i] = a[i] + b[i];
        }
        Ok(())
    }

    pub fn neon_vector_mul_f32(
        &self,
        a: &[f32],
        b: &[f32],
        result: &mut [f32],
    ) -> Result<(), SIMDError> {
        if a.len() != b.len() || a.len() != result.len() {
            return Err(SIMDError::SizeMismatch);
        }
        for i in 0..a.len() {
            result[i] = a[i] * b[i];
        }
        Ok(())
    }

    pub fn neon_dot_product_f32(&self, a: &[f32], b: &[f32]) -> Result<f32, SIMDError> {
        if a.len() != b.len() {
            return Err(SIMDError::SizeMismatch);
        }
        let mut sum = 0.0f32;
        for i in 0..a.len() {
            sum += a[i] * b[i];
        }
        Ok(sum)
    }
}
