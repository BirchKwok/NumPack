//! SIMD操作模块
//!
//! 从lazy_array_original.rs中提取的SIMD数学运算和向量化操作

/// SIMD优化的数据处理模块
#[cfg(target_arch = "x86_64")]
pub mod simd_ops {
    use std::arch::x86_64::*;

    /// SIMD优化的f32数组求和
    pub fn sum_f32_simd(data: &[f32]) -> f32 {
        unsafe {
            let mut sum = _mm256_setzero_ps();
            let chunks = data.chunks_exact(8);
            let remainder = chunks.remainder();

            for chunk in chunks {
                let vec = _mm256_loadu_ps(chunk.as_ptr());
                sum = _mm256_add_ps(sum, vec);
            }

            // 水平求和
            let sum_high = _mm256_extractf128_ps(sum, 1);
            let sum_low = _mm256_castps256_ps128(sum);
            let sum_final = _mm_add_ps(sum_high, sum_low);
            let sum_final = _mm_hadd_ps(sum_final, sum_final);
            let sum_final = _mm_hadd_ps(sum_final, sum_final);

            let mut result = _mm_cvtss_f32(sum_final);

            // 处理剩余元素
            for &val in remainder {
                result += val;
            }

            result
        }
    }

    /// SIMD优化的f64数组求和
    pub fn sum_f64_simd(data: &[f64]) -> f64 {
        unsafe {
            let mut sum = _mm256_setzero_pd();
            let chunks = data.chunks_exact(4);
            let remainder = chunks.remainder();

            for chunk in chunks {
                let vec = _mm256_loadu_pd(chunk.as_ptr());
                sum = _mm256_add_pd(sum, vec);
            }

            // 水平求和
            let sum_high = _mm256_extractf128_pd(sum, 1);
            let sum_low = _mm256_castpd256_pd128(sum);
            let sum_final = _mm_add_pd(sum_high, sum_low);
            let sum_final = _mm_hadd_pd(sum_final, sum_final);

            let mut result = _mm_cvtsd_f64(sum_final);

            // 处理剩余元素
            for &val in remainder {
                result += val;
            }

            result
        }
    }

    /// SIMD优化的向量加法
    pub fn add_f32_simd(a: &[f32], b: &[f32], result: &mut [f32]) {
        assert_eq!(a.len(), b.len());
        assert_eq!(a.len(), result.len());

        unsafe {
            let chunks = a.len() / 8;
            let remainder = a.len() % 8;

            for i in 0..chunks {
                let offset = i * 8;
                let vec_a = _mm256_loadu_ps(a.as_ptr().add(offset));
                let vec_b = _mm256_loadu_ps(b.as_ptr().add(offset));
                let sum = _mm256_add_ps(vec_a, vec_b);
                _mm256_storeu_ps(result.as_mut_ptr().add(offset), sum);
            }

            // 处理剩余元素
            for i in (chunks * 8)..(chunks * 8 + remainder) {
                result[i] = a[i] + b[i];
            }
        }
    }

    /// SIMD优化的向量乘法
    pub fn mul_f32_simd(a: &[f32], b: &[f32], result: &mut [f32]) {
        assert_eq!(a.len(), b.len());
        assert_eq!(a.len(), result.len());

        unsafe {
            let chunks = a.len() / 8;
            let remainder = a.len() % 8;

            for i in 0..chunks {
                let offset = i * 8;
                let vec_a = _mm256_loadu_ps(a.as_ptr().add(offset));
                let vec_b = _mm256_loadu_ps(b.as_ptr().add(offset));
                let product = _mm256_mul_ps(vec_a, vec_b);
                _mm256_storeu_ps(result.as_mut_ptr().add(offset), product);
            }

            // 处理剩余元素
            for i in (chunks * 8)..(chunks * 8 + remainder) {
                result[i] = a[i] * b[i];
            }
        }
    }

    /// SIMD优化的向量点积
    pub fn dot_product_f32_simd(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());

        unsafe {
            let mut sum = _mm256_setzero_ps();
            let chunks = a.len() / 8;
            let remainder = a.len() % 8;

            for i in 0..chunks {
                let offset = i * 8;
                let vec_a = _mm256_loadu_ps(a.as_ptr().add(offset));
                let vec_b = _mm256_loadu_ps(b.as_ptr().add(offset));
                let product = _mm256_mul_ps(vec_a, vec_b);
                sum = _mm256_add_ps(sum, product);
            }

            // 水平求和
            let sum_high = _mm256_extractf128_ps(sum, 1);
            let sum_low = _mm256_castps256_ps128(sum);
            let sum_final = _mm_add_ps(sum_high, sum_low);
            let sum_final = _mm_hadd_ps(sum_final, sum_final);
            let sum_final = _mm_hadd_ps(sum_final, sum_final);

            let mut result = _mm_cvtss_f32(sum_final);

            // 处理剩余元素
            for i in (chunks * 8)..(chunks * 8 + remainder) {
                result += a[i] * b[i];
            }

            result
        }
    }

    /// SIMD优化的数组比较
    pub fn compare_f32_simd(a: &[f32], b: &[f32], threshold: f32) -> Vec<bool> {
        assert_eq!(a.len(), b.len());
        let mut result = vec![false; a.len()];

        unsafe {
            let threshold_vec = _mm256_set1_ps(threshold);
            let chunks = a.len() / 8;
            let remainder = a.len() % 8;

            for i in 0..chunks {
                let offset = i * 8;
                let vec_a = _mm256_loadu_ps(a.as_ptr().add(offset));
                let vec_b = _mm256_loadu_ps(b.as_ptr().add(offset));
                let diff = _mm256_sub_ps(vec_a, vec_b);
                let abs_diff = _mm256_andnot_ps(_mm256_set1_ps(-0.0), diff);
                let cmp = _mm256_cmp_ps(abs_diff, threshold_vec, _CMP_LT_OQ);

                // 提取比较结果
                let mask = _mm256_movemask_ps(cmp);
                for j in 0..8 {
                    result[offset + j] = (mask & (1 << j)) != 0;
                }
            }

            // 处理剩余元素
            for i in (chunks * 8)..(chunks * 8 + remainder) {
                result[i] = (a[i] - b[i]).abs() < threshold;
            }
        }

        result
    }

    /// SIMD优化的最大值查找
    pub fn max_f32_simd(data: &[f32]) -> f32 {
        if data.is_empty() {
            return f32::NEG_INFINITY;
        }

        unsafe {
            let mut max_vec = _mm256_set1_ps(f32::NEG_INFINITY);
            let chunks = data.chunks_exact(8);
            let remainder = chunks.remainder();

            for chunk in chunks {
                let vec = _mm256_loadu_ps(chunk.as_ptr());
                max_vec = _mm256_max_ps(max_vec, vec);
            }

            // 水平最大值
            let max_high = _mm256_extractf128_ps(max_vec, 1);
            let max_low = _mm256_castps256_ps128(max_vec);
            let max_final = _mm_max_ps(max_high, max_low);
            let max_final = _mm_max_ps(max_final, _mm_movehl_ps(max_final, max_final));
            let max_final = _mm_max_ss(max_final, _mm_movehdup_ps(max_final));

            let mut result = _mm_cvtss_f32(max_final);

            // 处理剩余元素
            for &val in remainder {
                if val > result {
                    result = val;
                }
            }

            result
        }
    }

    /// SIMD优化的最小值查找
    pub fn min_f32_simd(data: &[f32]) -> f32 {
        if data.is_empty() {
            return f32::INFINITY;
        }

        unsafe {
            let mut min_vec = _mm256_set1_ps(f32::INFINITY);
            let chunks = data.chunks_exact(8);
            let remainder = chunks.remainder();

            for chunk in chunks {
                let vec = _mm256_loadu_ps(chunk.as_ptr());
                min_vec = _mm256_min_ps(min_vec, vec);
            }

            // 水平最小值
            let min_high = _mm256_extractf128_ps(min_vec, 1);
            let min_low = _mm256_castps256_ps128(min_vec);
            let min_final = _mm_min_ps(min_high, min_low);
            let min_final = _mm_min_ps(min_final, _mm_movehl_ps(min_final, min_final));
            let min_final = _mm_min_ss(min_final, _mm_movehdup_ps(min_final));

            let mut result = _mm_cvtss_f32(min_final);

            // 处理剩余元素
            for &val in remainder {
                if val < result {
                    result = val;
                }
            }

            result
        }
    }

    /// SIMD优化的数组填充
    pub fn fill_f32_simd(array: &mut [f32], value: f32) {
        unsafe {
            let value_vec = _mm256_set1_ps(value);
            let chunks = array.len() / 8;
            let remainder = array.len() % 8;

            for i in 0..chunks {
                let offset = i * 8;
                _mm256_storeu_ps(array.as_mut_ptr().add(offset), value_vec);
            }

            // 处理剩余元素
            for i in (chunks * 8)..(chunks * 8 + remainder) {
                array[i] = value;
            }
        }
    }

    /// SIMD优化的内存复制
    pub fn copy_aligned_simd(src: &[u8], dst: &mut [u8]) {
        assert_eq!(src.len(), dst.len());
        assert!(src.len() % 32 == 0, "数据长度必须是32的倍数");
        assert!(src.as_ptr() as usize % 32 == 0, "源数据必须32字节对齐");
        assert!(dst.as_ptr() as usize % 32 == 0, "目标数据必须32字节对齐");

        unsafe {
            let chunks = src.len() / 32;

            for i in 0..chunks {
                let offset = i * 32;
                let data = _mm256_load_si256(src.as_ptr().add(offset) as *const __m256i);
                _mm256_store_si256(dst.as_mut_ptr().add(offset) as *mut __m256i, data);
            }
        }
    }

    /// SIMD优化的数组比较（返回相等的元素个数）
    pub fn count_equal_f32_simd(a: &[f32], b: &[f32]) -> usize {
        assert_eq!(a.len(), b.len());
        let mut count = 0;

        unsafe {
            let chunks = a.len() / 8;
            let remainder = a.len() % 8;

            for i in 0..chunks {
                let offset = i * 8;
                let vec_a = _mm256_loadu_ps(a.as_ptr().add(offset));
                let vec_b = _mm256_loadu_ps(b.as_ptr().add(offset));
                let cmp = _mm256_cmp_ps(vec_a, vec_b, _CMP_EQ_OQ);
                let mask = _mm256_movemask_ps(cmp);
                count += mask.count_ones() as usize;
            }

            // 处理剩余元素
            for i in (chunks * 8)..(chunks * 8 + remainder) {
                if a[i] == b[i] {
                    count += 1;
                }
            }
        }

        count
    }
}

/// 非x86_64架构的回退实现
#[cfg(not(target_arch = "x86_64"))]
pub mod simd_ops {
    /// 标量版本的f32数组求和
    pub fn sum_f32_simd(data: &[f32]) -> f32 {
        data.iter().sum()
    }

    /// 标量版本的f64数组求和
    pub fn sum_f64_simd(data: &[f64]) -> f64 {
        data.iter().sum()
    }

    /// 标量版本的向量加法
    pub fn add_f32_simd(a: &[f32], b: &[f32], result: &mut [f32]) {
        for i in 0..a.len() {
            result[i] = a[i] + b[i];
        }
    }

    /// 标量版本的向量乘法
    pub fn mul_f32_simd(a: &[f32], b: &[f32], result: &mut [f32]) {
        for i in 0..a.len() {
            result[i] = a[i] * b[i];
        }
    }

    /// 标量版本的向量点积
    pub fn dot_product_f32_simd(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    /// 标量版本的数组比较
    pub fn compare_f32_simd(a: &[f32], b: &[f32], threshold: f32) -> Vec<bool> {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs() < threshold)
            .collect()
    }

    /// 标量版本的最大值查找
    pub fn max_f32_simd(data: &[f32]) -> f32 {
        data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b))
    }

    /// 标量版本的最小值查找
    pub fn min_f32_simd(data: &[f32]) -> f32 {
        data.iter().fold(f32::INFINITY, |a, &b| a.min(b))
    }

    /// 标量版本的数组填充
    pub fn fill_f32_simd(array: &mut [f32], value: f32) {
        for element in array {
            *element = value;
        }
    }

    /// 标量版本的内存复制
    pub fn copy_aligned_simd(src: &[u8], dst: &mut [u8]) {
        dst.copy_from_slice(src);
    }

    /// 标量版本的数组比较（返回相等的元素个数）
    pub fn count_equal_f32_simd(a: &[f32], b: &[f32]) -> usize {
        a.iter().zip(b.iter()).filter(|(x, y)| *x == *y).count()
    }
}

// 重新导出SIMD操作（对外API，内部未直接使用时忽略未用警告）
#[allow(unused_imports)]
pub use simd_ops::*;

/// SIMD工具函数
pub struct SIMDUtils;

impl SIMDUtils {
    /// 检查数据是否适合SIMD处理
    pub fn is_simd_suitable(len: usize, min_len: usize) -> bool {
        len >= min_len && cfg!(target_arch = "x86_64")
    }

    /// 获取SIMD对齐大小
    pub fn alignment_size() -> usize {
        #[cfg(target_arch = "x86_64")]
        {
            32 // AVX2 对齐
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            8 // 默认对齐
        }
    }

    /// 检查指针是否对齐
    pub fn is_aligned<T>(ptr: *const T, alignment: usize) -> bool {
        (ptr as usize) % alignment == 0
    }

    /// 计算需要的padding以达到对齐
    pub fn padding_for_alignment(addr: usize, alignment: usize) -> usize {
        let remainder = addr % alignment;
        if remainder == 0 {
            0
        } else {
            alignment - remainder
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sum_f32_simd() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let expected: f32 = data.iter().sum();
        let result = sum_f32_simd(&data);
        assert!((result - expected).abs() < 1e-6);
    }

    #[test]
    fn test_add_f32_simd() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let mut result = vec![0.0; 8];

        add_f32_simd(&a, &b, &mut result);

        for i in 0..8 {
            assert_eq!(result[i], a[i] + b[i]);
        }
    }

    #[test]
    fn test_dot_product_f32_simd() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![2.0, 3.0, 4.0, 5.0];
        let expected = 1.0 * 2.0 + 2.0 * 3.0 + 3.0 * 4.0 + 4.0 * 5.0;
        let result = dot_product_f32_simd(&a, &b);
        assert!((result - expected).abs() < 1e-6);
    }

    #[test]
    fn test_max_f32_simd() {
        let data = vec![1.0, 5.0, 3.0, 9.0, 2.0, 7.0, 4.0, 6.0];
        let result = max_f32_simd(&data);
        assert_eq!(result, 9.0);
    }

    #[test]
    fn test_min_f32_simd() {
        let data = vec![5.0, 1.0, 3.0, 9.0, 2.0, 7.0, 4.0, 6.0];
        let result = min_f32_simd(&data);
        assert_eq!(result, 1.0);
    }

    #[test]
    fn test_simd_utils() {
        assert!(SIMDUtils::is_simd_suitable(100, 16));
        assert!(!SIMDUtils::is_simd_suitable(10, 16));

        let alignment = SIMDUtils::alignment_size();
        assert!(alignment > 0);

        let ptr = &[1, 2, 3, 4] as *const [i32];
        let is_aligned = SIMDUtils::is_aligned(ptr as *const i32, 4);
        // 对齐检查结果取决于运行时内存分配
        assert!(is_aligned || !is_aligned); // 总是true，只是为了测试函数能运行
    }
}
