//! 标量回退实现
//!
//! 为所有平台提供统一的标量版本，用作SIMD优化的回退选项
//! 确保在不支持SIMD或数据量太小时仍能正常工作

use super::{DataType, NumPackSIMD, SIMDError};

impl NumPackSIMD {
    /// 标量行拷贝实现 - 通用回退版本
    pub fn scalar_copy_rows(
        &self,
        src: &[u8],
        dst: &mut [u8],
        indices: &[usize],
        row_size: usize,
    ) -> Result<(), SIMDError> {
        // 检查边界
        for &idx in indices {
            if idx * row_size + row_size > src.len() {
                return Err(SIMDError::IndexOutOfBounds);
            }
        }

        if dst.len() < indices.len() * row_size {
            return Err(SIMDError::SizeMismatch);
        }

        let mut dst_offset = 0;

        for &idx in indices {
            let src_offset = idx * row_size;

            // 直接内存拷贝
            unsafe {
                std::ptr::copy_nonoverlapping(
                    src.as_ptr().add(src_offset),
                    dst.as_mut_ptr().add(dst_offset),
                    row_size,
                );
            }

            dst_offset += row_size;
        }

        Ok(())
    }

    /// 标量数据类型转换实现 - 支持NumPack的所有14种数据类型
    pub fn scalar_convert(
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

        // 分发到具体的转换函数
        match (src_dtype, dst_dtype) {
            // 整数到浮点转换
            (DataType::Int8, DataType::Float32) => self.scalar_i8_to_f32(src, dst),
            (DataType::Int16, DataType::Float32) => self.scalar_i16_to_f32(src, dst),
            (DataType::Int32, DataType::Float32) => self.scalar_i32_to_f32(src, dst),
            (DataType::Int64, DataType::Float64) => self.scalar_i64_to_f64(src, dst),
            (DataType::Uint8, DataType::Float32) => self.scalar_u8_to_f32(src, dst),
            (DataType::Uint16, DataType::Float32) => self.scalar_u16_to_f32(src, dst),
            (DataType::Uint32, DataType::Float32) => self.scalar_u32_to_f32(src, dst),
            (DataType::Uint64, DataType::Float64) => self.scalar_u64_to_f64(src, dst),

            // 浮点到整数转换
            (DataType::Float32, DataType::Int8) => self.scalar_f32_to_i8(src, dst),
            (DataType::Float32, DataType::Int16) => self.scalar_f32_to_i16(src, dst),
            (DataType::Float32, DataType::Int32) => self.scalar_f32_to_i32(src, dst),
            (DataType::Float64, DataType::Int64) => self.scalar_f64_to_i64(src, dst),
            (DataType::Float32, DataType::Uint8) => self.scalar_f32_to_u8(src, dst),
            (DataType::Float32, DataType::Uint16) => self.scalar_f32_to_u16(src, dst),
            (DataType::Float32, DataType::Uint32) => self.scalar_f32_to_u32(src, dst),
            (DataType::Float64, DataType::Uint64) => self.scalar_f64_to_u64(src, dst),

            // 整数类型间转换
            (DataType::Int8, DataType::Int16) => self.scalar_i8_to_i16(src, dst),
            (DataType::Int8, DataType::Int32) => self.scalar_i8_to_i32(src, dst),
            (DataType::Int8, DataType::Int64) => self.scalar_i8_to_i64(src, dst),
            (DataType::Int16, DataType::Int32) => self.scalar_i16_to_i32(src, dst),
            (DataType::Int16, DataType::Int64) => self.scalar_i16_to_i64(src, dst),
            (DataType::Int32, DataType::Int64) => self.scalar_i32_to_i64(src, dst),

            // 类型收窄转换
            (DataType::Int16, DataType::Int8) => self.scalar_i16_to_i8(src, dst),
            (DataType::Int32, DataType::Int8) => self.scalar_i32_to_i8(src, dst),
            (DataType::Int32, DataType::Int16) => self.scalar_i32_to_i16(src, dst),
            (DataType::Int64, DataType::Int8) => self.scalar_i64_to_i8(src, dst),
            (DataType::Int64, DataType::Int16) => self.scalar_i64_to_i16(src, dst),
            (DataType::Int64, DataType::Int32) => self.scalar_i64_to_i32(src, dst),

            // 浮点类型间转换
            (DataType::Float32, DataType::Float64) => self.scalar_f32_to_f64(src, dst),
            (DataType::Float64, DataType::Float32) => self.scalar_f64_to_f32(src, dst),

            // Float16的特殊处理
            (DataType::Float16, DataType::Float32) => self.scalar_f16_to_f32(src, dst),
            (DataType::Float32, DataType::Float16) => self.scalar_f32_to_f16(src, dst),

            // 布尔类型转换
            (DataType::Bool, DataType::Int8) => self.scalar_bool_to_i8(src, dst),
            (DataType::Bool, DataType::Float32) => self.scalar_bool_to_f32(src, dst),
            (DataType::Int8, DataType::Bool) => self.scalar_i8_to_bool(src, dst),
            (DataType::Float32, DataType::Bool) => self.scalar_f32_to_bool(src, dst),

            // 复数类型转换
            (DataType::Complex64, DataType::Float32) => self.scalar_complex64_to_f32(src, dst),
            (DataType::Complex128, DataType::Float64) => self.scalar_complex128_to_f64(src, dst),
            (DataType::Float32, DataType::Complex64) => self.scalar_f32_to_complex64(src, dst),
            (DataType::Float64, DataType::Complex128) => self.scalar_f64_to_complex128(src, dst),

            _ => Err(SIMDError::UnsupportedOperation),
        }
    }

    // === 整数到浮点转换实现 ===

    fn scalar_i8_to_f32(&self, src: &[u8], dst: &mut [u8]) -> Result<(), SIMDError> {
        let src_data = unsafe { std::slice::from_raw_parts(src.as_ptr() as *const i8, src.len()) };
        let dst_data =
            unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut f32, dst.len() / 4) };

        if src_data.len() != dst_data.len() {
            return Err(SIMDError::SizeMismatch);
        }

        for i in 0..src_data.len() {
            dst_data[i] = src_data[i] as f32;
        }
        Ok(())
    }

    fn scalar_i16_to_f32(&self, src: &[u8], dst: &mut [u8]) -> Result<(), SIMDError> {
        let src_data =
            unsafe { std::slice::from_raw_parts(src.as_ptr() as *const i16, src.len() / 2) };
        let dst_data =
            unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut f32, dst.len() / 4) };

        if src_data.len() != dst_data.len() {
            return Err(SIMDError::SizeMismatch);
        }

        for i in 0..src_data.len() {
            dst_data[i] = src_data[i] as f32;
        }
        Ok(())
    }

    fn scalar_i32_to_f32(&self, src: &[u8], dst: &mut [u8]) -> Result<(), SIMDError> {
        let src_data =
            unsafe { std::slice::from_raw_parts(src.as_ptr() as *const i32, src.len() / 4) };
        let dst_data =
            unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut f32, dst.len() / 4) };

        if src_data.len() != dst_data.len() {
            return Err(SIMDError::SizeMismatch);
        }

        for i in 0..src_data.len() {
            dst_data[i] = src_data[i] as f32;
        }
        Ok(())
    }

    fn scalar_i64_to_f64(&self, src: &[u8], dst: &mut [u8]) -> Result<(), SIMDError> {
        let src_data =
            unsafe { std::slice::from_raw_parts(src.as_ptr() as *const i64, src.len() / 8) };
        let dst_data =
            unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut f64, dst.len() / 8) };

        if src_data.len() != dst_data.len() {
            return Err(SIMDError::SizeMismatch);
        }

        for i in 0..src_data.len() {
            dst_data[i] = src_data[i] as f64;
        }
        Ok(())
    }

    fn scalar_u8_to_f32(&self, src: &[u8], dst: &mut [u8]) -> Result<(), SIMDError> {
        let dst_data =
            unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut f32, dst.len() / 4) };

        if src.len() != dst_data.len() {
            return Err(SIMDError::SizeMismatch);
        }

        for i in 0..src.len() {
            dst_data[i] = src[i] as f32;
        }
        Ok(())
    }

    fn scalar_u16_to_f32(&self, src: &[u8], dst: &mut [u8]) -> Result<(), SIMDError> {
        let src_data =
            unsafe { std::slice::from_raw_parts(src.as_ptr() as *const u16, src.len() / 2) };
        let dst_data =
            unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut f32, dst.len() / 4) };

        if src_data.len() != dst_data.len() {
            return Err(SIMDError::SizeMismatch);
        }

        for i in 0..src_data.len() {
            dst_data[i] = src_data[i] as f32;
        }
        Ok(())
    }

    fn scalar_u32_to_f32(&self, src: &[u8], dst: &mut [u8]) -> Result<(), SIMDError> {
        let src_data =
            unsafe { std::slice::from_raw_parts(src.as_ptr() as *const u32, src.len() / 4) };
        let dst_data =
            unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut f32, dst.len() / 4) };

        if src_data.len() != dst_data.len() {
            return Err(SIMDError::SizeMismatch);
        }

        for i in 0..src_data.len() {
            dst_data[i] = src_data[i] as f32;
        }
        Ok(())
    }

    fn scalar_u64_to_f64(&self, src: &[u8], dst: &mut [u8]) -> Result<(), SIMDError> {
        let src_data =
            unsafe { std::slice::from_raw_parts(src.as_ptr() as *const u64, src.len() / 8) };
        let dst_data =
            unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut f64, dst.len() / 8) };

        if src_data.len() != dst_data.len() {
            return Err(SIMDError::SizeMismatch);
        }

        for i in 0..src_data.len() {
            dst_data[i] = src_data[i] as f64;
        }
        Ok(())
    }

    // === 浮点到整数转换实现 ===

    fn scalar_f32_to_i8(&self, src: &[u8], dst: &mut [u8]) -> Result<(), SIMDError> {
        let src_data =
            unsafe { std::slice::from_raw_parts(src.as_ptr() as *const f32, src.len() / 4) };
        let dst_data =
            unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut i8, dst.len()) };

        if src_data.len() != dst_data.len() {
            return Err(SIMDError::SizeMismatch);
        }

        for i in 0..src_data.len() {
            dst_data[i] = src_data[i] as i8;
        }
        Ok(())
    }

    fn scalar_f32_to_i16(&self, src: &[u8], dst: &mut [u8]) -> Result<(), SIMDError> {
        let src_data =
            unsafe { std::slice::from_raw_parts(src.as_ptr() as *const f32, src.len() / 4) };
        let dst_data =
            unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut i16, dst.len() / 2) };

        if src_data.len() != dst_data.len() {
            return Err(SIMDError::SizeMismatch);
        }

        for i in 0..src_data.len() {
            dst_data[i] = src_data[i] as i16;
        }
        Ok(())
    }

    fn scalar_f32_to_i32(&self, src: &[u8], dst: &mut [u8]) -> Result<(), SIMDError> {
        let src_data =
            unsafe { std::slice::from_raw_parts(src.as_ptr() as *const f32, src.len() / 4) };
        let dst_data =
            unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut i32, dst.len() / 4) };

        if src_data.len() != dst_data.len() {
            return Err(SIMDError::SizeMismatch);
        }

        for i in 0..src_data.len() {
            dst_data[i] = src_data[i] as i32;
        }
        Ok(())
    }

    fn scalar_f64_to_i64(&self, src: &[u8], dst: &mut [u8]) -> Result<(), SIMDError> {
        let src_data =
            unsafe { std::slice::from_raw_parts(src.as_ptr() as *const f64, src.len() / 8) };
        let dst_data =
            unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut i64, dst.len() / 8) };

        if src_data.len() != dst_data.len() {
            return Err(SIMDError::SizeMismatch);
        }

        for i in 0..src_data.len() {
            dst_data[i] = src_data[i] as i64;
        }
        Ok(())
    }

    // === 无符号整数转换实现 ===

    fn scalar_f32_to_u8(&self, src: &[u8], dst: &mut [u8]) -> Result<(), SIMDError> {
        let src_data =
            unsafe { std::slice::from_raw_parts(src.as_ptr() as *const f32, src.len() / 4) };

        if src_data.len() != dst.len() {
            return Err(SIMDError::SizeMismatch);
        }

        for i in 0..src_data.len() {
            dst[i] = src_data[i] as u8;
        }
        Ok(())
    }

    fn scalar_f32_to_u16(&self, src: &[u8], dst: &mut [u8]) -> Result<(), SIMDError> {
        let src_data =
            unsafe { std::slice::from_raw_parts(src.as_ptr() as *const f32, src.len() / 4) };
        let dst_data =
            unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut u16, dst.len() / 2) };

        if src_data.len() != dst_data.len() {
            return Err(SIMDError::SizeMismatch);
        }

        for i in 0..src_data.len() {
            dst_data[i] = src_data[i] as u16;
        }
        Ok(())
    }

    fn scalar_f32_to_u32(&self, src: &[u8], dst: &mut [u8]) -> Result<(), SIMDError> {
        let src_data =
            unsafe { std::slice::from_raw_parts(src.as_ptr() as *const f32, src.len() / 4) };
        let dst_data =
            unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut u32, dst.len() / 4) };

        if src_data.len() != dst_data.len() {
            return Err(SIMDError::SizeMismatch);
        }

        for i in 0..src_data.len() {
            dst_data[i] = src_data[i] as u32;
        }
        Ok(())
    }

    fn scalar_f64_to_u64(&self, src: &[u8], dst: &mut [u8]) -> Result<(), SIMDError> {
        let src_data =
            unsafe { std::slice::from_raw_parts(src.as_ptr() as *const f64, src.len() / 8) };
        let dst_data =
            unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut u64, dst.len() / 8) };

        if src_data.len() != dst_data.len() {
            return Err(SIMDError::SizeMismatch);
        }

        for i in 0..src_data.len() {
            dst_data[i] = src_data[i] as u64;
        }
        Ok(())
    }

    // === 整数类型间转换（类型扩展） ===

    fn scalar_i8_to_i16(&self, src: &[u8], dst: &mut [u8]) -> Result<(), SIMDError> {
        let src_data = unsafe { std::slice::from_raw_parts(src.as_ptr() as *const i8, src.len()) };
        let dst_data =
            unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut i16, dst.len() / 2) };

        if src_data.len() != dst_data.len() {
            return Err(SIMDError::SizeMismatch);
        }

        for i in 0..src_data.len() {
            dst_data[i] = src_data[i] as i16;
        }
        Ok(())
    }

    fn scalar_i8_to_i32(&self, src: &[u8], dst: &mut [u8]) -> Result<(), SIMDError> {
        let src_data = unsafe { std::slice::from_raw_parts(src.as_ptr() as *const i8, src.len()) };
        let dst_data =
            unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut i32, dst.len() / 4) };

        if src_data.len() != dst_data.len() {
            return Err(SIMDError::SizeMismatch);
        }

        for i in 0..src_data.len() {
            dst_data[i] = src_data[i] as i32;
        }
        Ok(())
    }

    fn scalar_i8_to_i64(&self, src: &[u8], dst: &mut [u8]) -> Result<(), SIMDError> {
        let src_data = unsafe { std::slice::from_raw_parts(src.as_ptr() as *const i8, src.len()) };
        let dst_data =
            unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut i64, dst.len() / 8) };

        if src_data.len() != dst_data.len() {
            return Err(SIMDError::SizeMismatch);
        }

        for i in 0..src_data.len() {
            dst_data[i] = src_data[i] as i64;
        }
        Ok(())
    }

    fn scalar_i16_to_i32(&self, src: &[u8], dst: &mut [u8]) -> Result<(), SIMDError> {
        let src_data =
            unsafe { std::slice::from_raw_parts(src.as_ptr() as *const i16, src.len() / 2) };
        let dst_data =
            unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut i32, dst.len() / 4) };

        if src_data.len() != dst_data.len() {
            return Err(SIMDError::SizeMismatch);
        }

        for i in 0..src_data.len() {
            dst_data[i] = src_data[i] as i32;
        }
        Ok(())
    }

    fn scalar_i16_to_i64(&self, src: &[u8], dst: &mut [u8]) -> Result<(), SIMDError> {
        let src_data =
            unsafe { std::slice::from_raw_parts(src.as_ptr() as *const i16, src.len() / 2) };
        let dst_data =
            unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut i64, dst.len() / 8) };

        if src_data.len() != dst_data.len() {
            return Err(SIMDError::SizeMismatch);
        }

        for i in 0..src_data.len() {
            dst_data[i] = src_data[i] as i64;
        }
        Ok(())
    }

    fn scalar_i32_to_i64(&self, src: &[u8], dst: &mut [u8]) -> Result<(), SIMDError> {
        let src_data =
            unsafe { std::slice::from_raw_parts(src.as_ptr() as *const i32, src.len() / 4) };
        let dst_data =
            unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut i64, dst.len() / 8) };

        if src_data.len() != dst_data.len() {
            return Err(SIMDError::SizeMismatch);
        }

        for i in 0..src_data.len() {
            dst_data[i] = src_data[i] as i64;
        }
        Ok(())
    }

    // === 整数类型间转换（类型收窄） ===

    fn scalar_i16_to_i8(&self, src: &[u8], dst: &mut [u8]) -> Result<(), SIMDError> {
        let src_data =
            unsafe { std::slice::from_raw_parts(src.as_ptr() as *const i16, src.len() / 2) };
        let dst_data =
            unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut i8, dst.len()) };

        if src_data.len() != dst_data.len() {
            return Err(SIMDError::SizeMismatch);
        }

        for i in 0..src_data.len() {
            dst_data[i] = src_data[i] as i8;
        }
        Ok(())
    }

    fn scalar_i32_to_i8(&self, src: &[u8], dst: &mut [u8]) -> Result<(), SIMDError> {
        let src_data =
            unsafe { std::slice::from_raw_parts(src.as_ptr() as *const i32, src.len() / 4) };
        let dst_data =
            unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut i8, dst.len()) };

        if src_data.len() != dst_data.len() {
            return Err(SIMDError::SizeMismatch);
        }

        for i in 0..src_data.len() {
            dst_data[i] = src_data[i] as i8;
        }
        Ok(())
    }

    fn scalar_i32_to_i16(&self, src: &[u8], dst: &mut [u8]) -> Result<(), SIMDError> {
        let src_data =
            unsafe { std::slice::from_raw_parts(src.as_ptr() as *const i32, src.len() / 4) };
        let dst_data =
            unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut i16, dst.len() / 2) };

        if src_data.len() != dst_data.len() {
            return Err(SIMDError::SizeMismatch);
        }

        for i in 0..src_data.len() {
            dst_data[i] = src_data[i] as i16;
        }
        Ok(())
    }

    fn scalar_i64_to_i8(&self, src: &[u8], dst: &mut [u8]) -> Result<(), SIMDError> {
        let src_data =
            unsafe { std::slice::from_raw_parts(src.as_ptr() as *const i64, src.len() / 8) };
        let dst_data =
            unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut i8, dst.len()) };

        if src_data.len() != dst_data.len() {
            return Err(SIMDError::SizeMismatch);
        }

        for i in 0..src_data.len() {
            dst_data[i] = src_data[i] as i8;
        }
        Ok(())
    }

    fn scalar_i64_to_i16(&self, src: &[u8], dst: &mut [u8]) -> Result<(), SIMDError> {
        let src_data =
            unsafe { std::slice::from_raw_parts(src.as_ptr() as *const i64, src.len() / 8) };
        let dst_data =
            unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut i16, dst.len() / 2) };

        if src_data.len() != dst_data.len() {
            return Err(SIMDError::SizeMismatch);
        }

        for i in 0..src_data.len() {
            dst_data[i] = src_data[i] as i16;
        }
        Ok(())
    }

    fn scalar_i64_to_i32(&self, src: &[u8], dst: &mut [u8]) -> Result<(), SIMDError> {
        let src_data =
            unsafe { std::slice::from_raw_parts(src.as_ptr() as *const i64, src.len() / 8) };
        let dst_data =
            unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut i32, dst.len() / 4) };

        if src_data.len() != dst_data.len() {
            return Err(SIMDError::SizeMismatch);
        }

        for i in 0..src_data.len() {
            dst_data[i] = src_data[i] as i32;
        }
        Ok(())
    }

    // === 浮点类型间转换 ===

    fn scalar_f32_to_f64(&self, src: &[u8], dst: &mut [u8]) -> Result<(), SIMDError> {
        let src_data =
            unsafe { std::slice::from_raw_parts(src.as_ptr() as *const f32, src.len() / 4) };
        let dst_data =
            unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut f64, dst.len() / 8) };

        if src_data.len() != dst_data.len() {
            return Err(SIMDError::SizeMismatch);
        }

        for i in 0..src_data.len() {
            dst_data[i] = src_data[i] as f64;
        }
        Ok(())
    }

    fn scalar_f64_to_f32(&self, src: &[u8], dst: &mut [u8]) -> Result<(), SIMDError> {
        let src_data =
            unsafe { std::slice::from_raw_parts(src.as_ptr() as *const f64, src.len() / 8) };
        let dst_data =
            unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut f32, dst.len() / 4) };

        if src_data.len() != dst_data.len() {
            return Err(SIMDError::SizeMismatch);
        }

        for i in 0..src_data.len() {
            dst_data[i] = src_data[i] as f32;
        }
        Ok(())
    }

    // === Float16特殊处理 ===

    fn scalar_f16_to_f32(&self, src: &[u8], dst: &mut [u8]) -> Result<(), SIMDError> {
        let src_data =
            unsafe { std::slice::from_raw_parts(src.as_ptr() as *const u16, src.len() / 2) };
        let dst_data =
            unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut f32, dst.len() / 4) };

        if src_data.len() != dst_data.len() {
            return Err(SIMDError::SizeMismatch);
        }

        for i in 0..src_data.len() {
            let f16_val = half::f16::from_bits(src_data[i]);
            dst_data[i] = f16_val.to_f32();
        }
        Ok(())
    }

    fn scalar_f32_to_f16(&self, src: &[u8], dst: &mut [u8]) -> Result<(), SIMDError> {
        let src_data =
            unsafe { std::slice::from_raw_parts(src.as_ptr() as *const f32, src.len() / 4) };
        let dst_data =
            unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut u16, dst.len() / 2) };

        if src_data.len() != dst_data.len() {
            return Err(SIMDError::SizeMismatch);
        }

        for i in 0..src_data.len() {
            let f16_val = half::f16::from_f32(src_data[i]);
            dst_data[i] = f16_val.to_bits();
        }
        Ok(())
    }

    // === 布尔类型转换 ===

    fn scalar_bool_to_i8(&self, src: &[u8], dst: &mut [u8]) -> Result<(), SIMDError> {
        if src.len() != dst.len() {
            return Err(SIMDError::SizeMismatch);
        }

        for i in 0..src.len() {
            dst[i] = if src[i] != 0 { 1 } else { 0 };
        }
        Ok(())
    }

    fn scalar_bool_to_f32(&self, src: &[u8], dst: &mut [u8]) -> Result<(), SIMDError> {
        let dst_data =
            unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut f32, dst.len() / 4) };

        if src.len() != dst_data.len() {
            return Err(SIMDError::SizeMismatch);
        }

        for i in 0..src.len() {
            dst_data[i] = if src[i] != 0 { 1.0 } else { 0.0 };
        }
        Ok(())
    }

    fn scalar_i8_to_bool(&self, src: &[u8], dst: &mut [u8]) -> Result<(), SIMDError> {
        if src.len() != dst.len() {
            return Err(SIMDError::SizeMismatch);
        }

        for i in 0..src.len() {
            dst[i] = if src[i] != 0 { 1 } else { 0 };
        }
        Ok(())
    }

    fn scalar_f32_to_bool(&self, src: &[u8], dst: &mut [u8]) -> Result<(), SIMDError> {
        let src_data =
            unsafe { std::slice::from_raw_parts(src.as_ptr() as *const f32, src.len() / 4) };

        if src_data.len() != dst.len() {
            return Err(SIMDError::SizeMismatch);
        }

        for i in 0..src_data.len() {
            dst[i] = if src_data[i] != 0.0 { 1 } else { 0 };
        }
        Ok(())
    }

    // === 复数类型转换 ===

    fn scalar_complex64_to_f32(&self, src: &[u8], dst: &mut [u8]) -> Result<(), SIMDError> {
        // Complex64 = 2×f32，取实部
        let src_data =
            unsafe { std::slice::from_raw_parts(src.as_ptr() as *const f32, src.len() / 4) };
        let dst_data =
            unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut f32, dst.len() / 4) };

        if src_data.len() / 2 != dst_data.len() {
            return Err(SIMDError::SizeMismatch);
        }

        for i in 0..dst_data.len() {
            dst_data[i] = src_data[i * 2]; // 取实部
        }
        Ok(())
    }

    fn scalar_complex128_to_f64(&self, src: &[u8], dst: &mut [u8]) -> Result<(), SIMDError> {
        // Complex128 = 2×f64，取实部
        let src_data =
            unsafe { std::slice::from_raw_parts(src.as_ptr() as *const f64, src.len() / 8) };
        let dst_data =
            unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut f64, dst.len() / 8) };

        if src_data.len() / 2 != dst_data.len() {
            return Err(SIMDError::SizeMismatch);
        }

        for i in 0..dst_data.len() {
            dst_data[i] = src_data[i * 2]; // 取实部
        }
        Ok(())
    }

    fn scalar_f32_to_complex64(&self, src: &[u8], dst: &mut [u8]) -> Result<(), SIMDError> {
        // f32到Complex64，虚部为0
        let src_data =
            unsafe { std::slice::from_raw_parts(src.as_ptr() as *const f32, src.len() / 4) };
        let dst_data =
            unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut f32, dst.len() / 4) };

        if src_data.len() * 2 != dst_data.len() {
            return Err(SIMDError::SizeMismatch);
        }

        for i in 0..src_data.len() {
            dst_data[i * 2] = src_data[i]; // 实部
            dst_data[i * 2 + 1] = 0.0; // 虚部
        }
        Ok(())
    }

    fn scalar_f64_to_complex128(&self, src: &[u8], dst: &mut [u8]) -> Result<(), SIMDError> {
        // f64到Complex128，虚部为0
        let src_data =
            unsafe { std::slice::from_raw_parts(src.as_ptr() as *const f64, src.len() / 8) };
        let dst_data =
            unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut f64, dst.len() / 8) };

        if src_data.len() * 2 != dst_data.len() {
            return Err(SIMDError::SizeMismatch);
        }

        for i in 0..src_data.len() {
            dst_data[i * 2] = src_data[i]; // 实部
            dst_data[i * 2 + 1] = 0.0; // 虚部
        }
        Ok(())
    }
}
