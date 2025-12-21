//! SimSIMD 后端集成层
//!
//! 封装 SimSIMD 库的 FFI 调用，提供类型安全的 Rust 接口
//! 混合策略：dot product 使用 BLAS，其他 metric 使用 SimSIMD

use crate::vector_engine::metrics::MetricType;
use std::fmt;

// BLAS support for matrix multiplication (Accelerate on macOS, OpenBLAS on Linux)
// Windows uses matrixmultiply fallback, no blas_src needed
#[cfg(not(target_os = "windows"))]
extern crate blas_src;

/// SimSIMD 计算错误
#[derive(Debug, Clone)]
pub enum SimdError {
    /// 向量长度不匹配
    LengthMismatch { expected: usize, got: usize },
    /// 不支持的度量类型
    UnsupportedMetric {
        metric: MetricType,
        dtype: &'static str,
    },
    /// 零向量错误
    ZeroVector,
    /// 其他错误
    Other(String),
}

impl fmt::Display for SimdError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SimdError::LengthMismatch { expected, got } => {
                write!(
                    f,
                    "Vector length mismatch: expected {}, got {}",
                    expected, got
                )
            }
            SimdError::UnsupportedMetric { metric, dtype } => {
                write!(f, "Unsupported metric {} for dtype {}", metric, dtype)
            }
            SimdError::ZeroVector => {
                write!(f, "Cannot compute metric for zero vector")
            }
            SimdError::Other(msg) => write!(f, "{}", msg),
        }
    }
}

impl std::error::Error for SimdError {}

pub type Result<T> = std::result::Result<T, SimdError>;

/// SimSIMD 后端
pub struct SimdBackend {
    /// 检测到的 SIMD 能力
    capabilities: SIMDCapabilities,
}

/// SIMD 能力检测
#[derive(Debug, Clone)]
pub struct SIMDCapabilities {
    pub has_avx2: bool,
    pub has_avx512: bool,
    pub has_neon: bool,
    pub has_sve: bool,
}

impl SimdBackend {
    /// 创建后端实例，自动检测 SIMD 能力
    pub fn new() -> Self {
        let capabilities = Self::detect_capabilities();
        Self { capabilities }
    }

    /// 检测 CPU SIMD 能力
    fn detect_capabilities() -> SIMDCapabilities {
        #[cfg(target_arch = "x86_64")]
        {
            SIMDCapabilities {
                has_avx2: is_x86_feature_detected!("avx2"),
                has_avx512: is_x86_feature_detected!("avx512f"),
                has_neon: false,
                has_sve: false,
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            SIMDCapabilities {
                has_avx2: false,
                has_avx512: false,
                has_neon: true, // ARM64 总是有 NEON
                has_sve: false, // SVE 检测较复杂，暂时设为 false
            }
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            SIMDCapabilities {
                has_avx2: false,
                has_avx512: false,
                has_neon: false,
                has_sve: false,
            }
        }
    }

    /// 获取 SIMD 能力
    pub fn capabilities(&self) -> &SIMDCapabilities {
        &self.capabilities
    }

    /// 计算两个 f64 向量的度量
    pub fn compute_f64(&self, a: &[f64], b: &[f64], metric: MetricType) -> Result<f64> {
        // 检查长度
        if a.len() != b.len() {
            return Err(SimdError::LengthMismatch {
                expected: a.len(),
                got: b.len(),
            });
        }

        // 调用 SimSIMD
        match metric {
            MetricType::DotProduct | MetricType::InnerProduct => {
                Ok(simsimd::SpatialSimilarity::dot(a, b).expect("SimSIMD dot failed"))
            }
            MetricType::Cosine => {
                // SimSIMD 返回的是余弦距离 (1 - cosine_similarity)
                // 我们需要转换为余弦相似度
                let distance =
                    simsimd::SpatialSimilarity::cosine(a, b).expect("SimSIMD cosine failed");
                Ok(1.0 - distance) // 转换为相似度
            }
            MetricType::L2Distance => {
                let sq = simsimd::SpatialSimilarity::sqeuclidean(a, b)
                    .expect("SimSIMD sqeuclidean failed");
                Ok(sq.sqrt())
            }
            MetricType::L2Squared => {
                Ok(simsimd::SpatialSimilarity::sqeuclidean(a, b)
                    .expect("SimSIMD sqeuclidean failed"))
            }
            MetricType::KL => {
                // KL 散度 (Kullback-Leibler Divergence)
                Ok(simsimd::ProbabilitySimilarity::kullbackleibler(a, b)
                    .expect("SimSIMD KL failed"))
            }
            MetricType::JS => {
                // JS 散度 (Jensen-Shannon Divergence)
                Ok(simsimd::ProbabilitySimilarity::jensenshannon(a, b).expect("SimSIMD JS failed"))
            }
            MetricType::Hamming | MetricType::Jaccard => Err(SimdError::UnsupportedMetric {
                metric,
                dtype: "f64 (requires binary/uint8 vectors)",
            }),
        }
    }

    /// 计算两个 f32 向量的度量
    pub fn compute_f32(&self, a: &[f32], b: &[f32], metric: MetricType) -> Result<f32> {
        // 检查长度
        if a.len() != b.len() {
            return Err(SimdError::LengthMismatch {
                expected: a.len(),
                got: b.len(),
            });
        }

        // 调用 SimSIMD（注意：SimSIMD 返回 f64，需要转换为 f32）
        match metric {
            MetricType::DotProduct | MetricType::InnerProduct => {
                let result: f64 =
                    simsimd::SpatialSimilarity::dot(a, b).expect("SimSIMD dot failed");
                Ok(result as f32)
            }
            MetricType::Cosine => {
                // SimSIMD 返回的是余弦距离 (1 - cosine_similarity)
                // 我们需要转换为余弦相似度
                let distance: f64 =
                    simsimd::SpatialSimilarity::cosine(a, b).expect("SimSIMD cosine failed");
                Ok((1.0 - distance) as f32) // 转换为相似度
            }
            MetricType::L2Distance => {
                let sq: f64 = simsimd::SpatialSimilarity::sqeuclidean(a, b)
                    .expect("SimSIMD sqeuclidean failed");
                Ok(sq.sqrt() as f32)
            }
            MetricType::L2Squared => {
                let result: f64 = simsimd::SpatialSimilarity::sqeuclidean(a, b)
                    .expect("SimSIMD sqeuclidean failed");
                Ok(result as f32)
            }
            MetricType::KL => {
                // KL 散度
                let result: f64 = simsimd::ProbabilitySimilarity::kullbackleibler(a, b)
                    .expect("SimSIMD KL failed");
                Ok(result as f32)
            }
            MetricType::JS => {
                // JS 散度
                let result: f64 =
                    simsimd::ProbabilitySimilarity::jensenshannon(a, b).expect("SimSIMD JS failed");
                Ok(result as f32)
            }
            MetricType::Hamming | MetricType::Jaccard => Err(SimdError::UnsupportedMetric {
                metric,
                dtype: "f32 (requires binary/uint8 vectors)",
            }),
        }
    }

    /// 批量计算：query 向量 vs 多个候选向量
    ///
    /// 关键优化：使用 Rayon 并行计算以充分利用多核 CPU
    pub fn batch_compute_f64(
        &self,
        query: &[f64],
        candidates: &[&[f64]],
        metric: MetricType,
    ) -> Result<Vec<f64>> {
        #[cfg(feature = "rayon")]
        {
            use rayon::prelude::*;

            // 并行计算：利用多核CPU加速
            // 这是性能提升的关键！单核无法匹敌 NumPy 的优化
            candidates
                .par_iter()
                .map(|candidate| self.compute_f64(query, candidate, metric))
                .collect()
        }

        #[cfg(not(feature = "rayon"))]
        {
            candidates
                .iter()
                .map(|candidate| self.compute_f64(query, candidate, metric))
                .collect()
        }
    }

    /// 批量计算：query 向量 vs 多个候选向量 (f32)
    ///
    /// 关键优化：使用 Rayon 并行计算
    pub fn batch_compute_f32(
        &self,
        query: &[f32],
        candidates: &[&[f32]],
        metric: MetricType,
    ) -> Result<Vec<f32>> {
        #[cfg(feature = "rayon")]
        {
            use rayon::prelude::*;

            candidates
                .par_iter()
                .map(|candidate| self.compute_f32(query, candidate, metric))
                .collect()
        }

        #[cfg(not(feature = "rayon"))]
        {
            candidates
                .iter()
                .map(|candidate| self.compute_f32(query, candidate, metric))
                .collect()
        }
    }

    /// 计算两个 i8 向量的度量
    pub fn compute_i8(&self, a: &[i8], b: &[i8], metric: MetricType) -> Result<f64> {
        if a.len() != b.len() {
            return Err(SimdError::LengthMismatch {
                expected: a.len(),
                got: b.len(),
            });
        }

        match metric {
            MetricType::DotProduct | MetricType::InnerProduct => {
                Ok(simsimd::SpatialSimilarity::dot(a, b).expect("SimSIMD i8 dot failed"))
            }
            MetricType::Cosine => {
                let distance =
                    simsimd::SpatialSimilarity::cosine(a, b).expect("SimSIMD i8 cosine failed");
                Ok(1.0 - distance)
            }
            MetricType::L2Distance => {
                let sq = simsimd::SpatialSimilarity::sqeuclidean(a, b)
                    .expect("SimSIMD i8 sqeuclidean failed");
                Ok(sq.sqrt())
            }
            MetricType::L2Squared => Ok(simsimd::SpatialSimilarity::sqeuclidean(a, b)
                .expect("SimSIMD i8 sqeuclidean failed")),
            _ => Err(SimdError::UnsupportedMetric {
                metric,
                dtype: "i8",
            }),
        }
    }

    /// 批量计算 (i8)
    pub fn batch_compute_i8(
        &self,
        query: &[i8],
        candidates: &[&[i8]],
        metric: MetricType,
    ) -> Result<Vec<f64>> {
        #[cfg(feature = "rayon")]
        {
            use rayon::prelude::*;
            candidates
                .par_iter()
                .map(|candidate| self.compute_i8(query, candidate, metric))
                .collect()
        }

        #[cfg(not(feature = "rayon"))]
        {
            candidates
                .iter()
                .map(|candidate| self.compute_i8(query, candidate, metric))
                .collect()
        }
    }

    /// 计算两个 u8 向量的度量（二进制向量）
    pub fn compute_u8(&self, a: &[u8], b: &[u8], metric: MetricType) -> Result<f64> {
        if a.len() != b.len() {
            return Err(SimdError::LengthMismatch {
                expected: a.len(),
                got: b.len(),
            });
        }

        match metric {
            MetricType::Hamming => {
                Ok(simsimd::BinarySimilarity::hamming(a, b).expect("SimSIMD hamming failed"))
            }
            MetricType::Jaccard => {
                Ok(simsimd::BinarySimilarity::jaccard(a, b).expect("SimSIMD jaccard failed"))
            }
            _ => Err(SimdError::UnsupportedMetric {
                metric,
                dtype: "u8",
            }),
        }
    }

    /// 批量计算 (u8)
    pub fn batch_compute_u8(
        &self,
        query: &[u8],
        candidates: &[&[u8]],
        metric: MetricType,
    ) -> Result<Vec<f64>> {
        #[cfg(feature = "rayon")]
        {
            use rayon::prelude::*;
            candidates
                .par_iter()
                .map(|candidate| self.compute_u8(query, candidate, metric))
                .collect()
        }

        #[cfg(not(feature = "rayon"))]
        {
            candidates
                .iter()
                .map(|candidate| self.compute_u8(query, candidate, metric))
                .collect()
        }
    }

    // ============== int16 support ==============
    
    /// 计算两个 i16 向量的度量 (通过转换为 f64 计算)
    pub fn compute_i16(&self, a: &[i16], b: &[i16], metric: MetricType) -> Result<f64> {
        if a.len() != b.len() {
            return Err(SimdError::LengthMismatch {
                expected: a.len(),
                got: b.len(),
            });
        }
        // Convert to f64 for computation
        let a_f64: Vec<f64> = a.iter().map(|&x| x as f64).collect();
        let b_f64: Vec<f64> = b.iter().map(|&x| x as f64).collect();
        self.compute_f64(&a_f64, &b_f64, metric)
    }

    /// 批量计算 (i16)
    pub fn batch_compute_i16(
        &self,
        query: &[i16],
        candidates: &[&[i16]],
        metric: MetricType,
    ) -> Result<Vec<f64>> {
        #[cfg(feature = "rayon")]
        {
            use rayon::prelude::*;
            candidates
                .par_iter()
                .map(|candidate| self.compute_i16(query, candidate, metric))
                .collect()
        }

        #[cfg(not(feature = "rayon"))]
        {
            candidates
                .iter()
                .map(|candidate| self.compute_i16(query, candidate, metric))
                .collect()
        }
    }

    // ============== int32 support ==============
    
    /// 计算两个 i32 向量的度量 (通过转换为 f64 计算)
    pub fn compute_i32(&self, a: &[i32], b: &[i32], metric: MetricType) -> Result<f64> {
        if a.len() != b.len() {
            return Err(SimdError::LengthMismatch {
                expected: a.len(),
                got: b.len(),
            });
        }
        let a_f64: Vec<f64> = a.iter().map(|&x| x as f64).collect();
        let b_f64: Vec<f64> = b.iter().map(|&x| x as f64).collect();
        self.compute_f64(&a_f64, &b_f64, metric)
    }

    /// 批量计算 (i32)
    pub fn batch_compute_i32(
        &self,
        query: &[i32],
        candidates: &[&[i32]],
        metric: MetricType,
    ) -> Result<Vec<f64>> {
        #[cfg(feature = "rayon")]
        {
            use rayon::prelude::*;
            candidates
                .par_iter()
                .map(|candidate| self.compute_i32(query, candidate, metric))
                .collect()
        }

        #[cfg(not(feature = "rayon"))]
        {
            candidates
                .iter()
                .map(|candidate| self.compute_i32(query, candidate, metric))
                .collect()
        }
    }

    // ============== int64 support ==============
    
    /// 计算两个 i64 向量的度量 (通过转换为 f64 计算)
    pub fn compute_i64(&self, a: &[i64], b: &[i64], metric: MetricType) -> Result<f64> {
        if a.len() != b.len() {
            return Err(SimdError::LengthMismatch {
                expected: a.len(),
                got: b.len(),
            });
        }
        let a_f64: Vec<f64> = a.iter().map(|&x| x as f64).collect();
        let b_f64: Vec<f64> = b.iter().map(|&x| x as f64).collect();
        self.compute_f64(&a_f64, &b_f64, metric)
    }

    /// 批量计算 (i64)
    pub fn batch_compute_i64(
        &self,
        query: &[i64],
        candidates: &[&[i64]],
        metric: MetricType,
    ) -> Result<Vec<f64>> {
        #[cfg(feature = "rayon")]
        {
            use rayon::prelude::*;
            candidates
                .par_iter()
                .map(|candidate| self.compute_i64(query, candidate, metric))
                .collect()
        }

        #[cfg(not(feature = "rayon"))]
        {
            candidates
                .iter()
                .map(|candidate| self.compute_i64(query, candidate, metric))
                .collect()
        }
    }

    // ============== uint16 support ==============
    
    /// 计算两个 u16 向量的度量 (通过转换为 f64 计算)
    pub fn compute_u16(&self, a: &[u16], b: &[u16], metric: MetricType) -> Result<f64> {
        if a.len() != b.len() {
            return Err(SimdError::LengthMismatch {
                expected: a.len(),
                got: b.len(),
            });
        }
        let a_f64: Vec<f64> = a.iter().map(|&x| x as f64).collect();
        let b_f64: Vec<f64> = b.iter().map(|&x| x as f64).collect();
        self.compute_f64(&a_f64, &b_f64, metric)
    }

    /// 批量计算 (u16)
    pub fn batch_compute_u16(
        &self,
        query: &[u16],
        candidates: &[&[u16]],
        metric: MetricType,
    ) -> Result<Vec<f64>> {
        #[cfg(feature = "rayon")]
        {
            use rayon::prelude::*;
            candidates
                .par_iter()
                .map(|candidate| self.compute_u16(query, candidate, metric))
                .collect()
        }

        #[cfg(not(feature = "rayon"))]
        {
            candidates
                .iter()
                .map(|candidate| self.compute_u16(query, candidate, metric))
                .collect()
        }
    }

    // ============== uint32 support ==============
    
    /// 计算两个 u32 向量的度量 (通过转换为 f64 计算)
    pub fn compute_u32(&self, a: &[u32], b: &[u32], metric: MetricType) -> Result<f64> {
        if a.len() != b.len() {
            return Err(SimdError::LengthMismatch {
                expected: a.len(),
                got: b.len(),
            });
        }
        let a_f64: Vec<f64> = a.iter().map(|&x| x as f64).collect();
        let b_f64: Vec<f64> = b.iter().map(|&x| x as f64).collect();
        self.compute_f64(&a_f64, &b_f64, metric)
    }

    /// 批量计算 (u32)
    pub fn batch_compute_u32(
        &self,
        query: &[u32],
        candidates: &[&[u32]],
        metric: MetricType,
    ) -> Result<Vec<f64>> {
        #[cfg(feature = "rayon")]
        {
            use rayon::prelude::*;
            candidates
                .par_iter()
                .map(|candidate| self.compute_u32(query, candidate, metric))
                .collect()
        }

        #[cfg(not(feature = "rayon"))]
        {
            candidates
                .iter()
                .map(|candidate| self.compute_u32(query, candidate, metric))
                .collect()
        }
    }

    // ============== uint64 support ==============
    
    /// 计算两个 u64 向量的度量 (通过转换为 f64 计算)
    pub fn compute_u64(&self, a: &[u64], b: &[u64], metric: MetricType) -> Result<f64> {
        if a.len() != b.len() {
            return Err(SimdError::LengthMismatch {
                expected: a.len(),
                got: b.len(),
            });
        }
        let a_f64: Vec<f64> = a.iter().map(|&x| x as f64).collect();
        let b_f64: Vec<f64> = b.iter().map(|&x| x as f64).collect();
        self.compute_f64(&a_f64, &b_f64, metric)
    }

    /// 批量计算 (u64)
    pub fn batch_compute_u64(
        &self,
        query: &[u64],
        candidates: &[&[u64]],
        metric: MetricType,
    ) -> Result<Vec<f64>> {
        #[cfg(feature = "rayon")]
        {
            use rayon::prelude::*;
            candidates
                .par_iter()
                .map(|candidate| self.compute_u64(query, candidate, metric))
                .collect()
        }

        #[cfg(not(feature = "rayon"))]
        {
            candidates
                .iter()
                .map(|candidate| self.compute_u64(query, candidate, metric))
                .collect()
        }
    }

    // ============== float16 support ==============
    
    /// 计算两个 f16 向量的度量 (通过转换为 f32 计算)
    pub fn compute_f16(&self, a: &[half::f16], b: &[half::f16], metric: MetricType) -> Result<f64> {
        if a.len() != b.len() {
            return Err(SimdError::LengthMismatch {
                expected: a.len(),
                got: b.len(),
            });
        }
        // Convert to f32 for computation (f16 -> f32 is lossless)
        let a_f32: Vec<f32> = a.iter().map(|&x| x.to_f32()).collect();
        let b_f32: Vec<f32> = b.iter().map(|&x| x.to_f32()).collect();
        let result = self.compute_f32(&a_f32, &b_f32, metric)?;
        Ok(result as f64)
    }

    /// 批量计算 (f16)
    pub fn batch_compute_f16(
        &self,
        query: &[half::f16],
        candidates: &[&[half::f16]],
        metric: MetricType,
    ) -> Result<Vec<f64>> {
        #[cfg(feature = "rayon")]
        {
            use rayon::prelude::*;
            candidates
                .par_iter()
                .map(|candidate| self.compute_f16(query, candidate, metric))
                .collect()
        }

        #[cfg(not(feature = "rayon"))]
        {
            candidates
                .iter()
                .map(|candidate| self.compute_f16(query, candidate, metric))
                .collect()
        }
    }
}

// ============== cdist / matmul optimized implementations ==============

impl SimdBackend {
    /// Compute pairwise distances/similarities between two matrices (cdist)
    /// 
    /// This is optimized for computing all pairwise metrics between rows of matrix A
    /// and rows of matrix B, returning a result matrix of shape [M, N].
    /// 
    /// Uses parallel computation over the 2D output grid for better performance
    /// compared to row-by-row computation.
    /// 
    /// # Arguments
    /// * `a` - First matrix as contiguous slice, shape [M, D]
    /// * `b` - Second matrix as contiguous slice, shape [N, D]  
    /// * `m` - Number of rows in A
    /// * `n` - Number of rows in B
    /// * `d` - Dimension (number of columns)
    /// * `metric` - The distance/similarity metric to compute
    /// 
    /// # Returns
    /// Result matrix as flat Vec<f64> in row-major order, shape [M, N]
    pub fn cdist_f64(
        &self,
        a: &[f64],
        b: &[f64],
        m: usize,
        n: usize,
        d: usize,
        metric: MetricType,
    ) -> Result<Vec<f64>> {
        // Validate input sizes
        if a.len() != m * d {
            return Err(SimdError::Other(format!(
                "Matrix A size mismatch: expected {}, got {}",
                m * d,
                a.len()
            )));
        }
        if b.len() != n * d {
            return Err(SimdError::Other(format!(
                "Matrix B size mismatch: expected {}, got {}",
                n * d,
                b.len()
            )));
        }

        // Use BLAS for dot product (matrix multiplication) - much faster than SimSIMD for this case
        if matches!(metric, MetricType::DotProduct | MetricType::InnerProduct) {
            return self.cdist_f64_dot_blas(a, b, m, n, d);
        }

        #[cfg(feature = "rayon")]
        {
            use rayon::prelude::*;

            // Pre-allocate output buffer
            let mut result = vec![0.0f64; m * n];
            
            // Parallel iteration over output matrix elements
            // Using chunks to improve cache locality
            result
                .par_chunks_mut(n)
                .enumerate()
                .for_each(|(i, row)| {
                    let a_row = &a[i * d..(i + 1) * d];
                    for (j, out) in row.iter_mut().enumerate() {
                        let b_row = &b[j * d..(j + 1) * d];
                        *out = self.compute_f64(a_row, b_row, metric).unwrap_or(f64::NAN);
                    }
                });

            Ok(result)
        }

        #[cfg(not(feature = "rayon"))]
        {
            let mut result = Vec::with_capacity(m * n);
            for i in 0..m {
                let a_row = &a[i * d..(i + 1) * d];
                for j in 0..n {
                    let b_row = &b[j * d..(j + 1) * d];
                    result.push(self.compute_f64(a_row, b_row, metric).unwrap_or(f64::NAN));
                }
            }
            Ok(result)
        }
    }

    /// matrixmultiply-accelerated dot product cdist for f64
    /// Uses SIMD-optimized GEMM
    #[inline]
    fn cdist_f64_dot_blas(&self, a: &[f64], b: &[f64], m: usize, n: usize, d: usize) -> Result<Vec<f64>> {
        use matrixmultiply::dgemm;
        
        // Allocate output buffer
        let mut result = vec![0.0f64; m * n];
        
        // dgemm computes C = alpha * A * B + beta * C
        // We want C = A @ B.T
        unsafe {
            dgemm(
                m, d, n,           // dimensions: m, k, n
                1.0,               // alpha
                a.as_ptr(), d as isize, 1,  // A: ptr, row_stride, col_stride
                b.as_ptr(), 1, d as isize,  // B.T: ptr, row_stride, col_stride
                0.0,               // beta
                result.as_mut_ptr(), n as isize, 1,  // C: ptr, row_stride, col_stride
            );
        }
        
        Ok(result)
    }

    /// Compute pairwise distances/similarities between two f32 matrices (cdist)
    /// 
    /// Optimized implementation with:
    /// - Direct SimSIMD calls without wrapper overhead
    /// - Metric-specific fast paths to avoid match in hot loop
    /// - Parallel computation with Rayon
    pub fn cdist_f32(
        &self,
        a: &[f32],
        b: &[f32],
        m: usize,
        n: usize,
        d: usize,
        metric: MetricType,
    ) -> Result<Vec<f32>> {
        if a.len() != m * d {
            return Err(SimdError::Other(format!(
                "Matrix A size mismatch: expected {}, got {}",
                m * d,
                a.len()
            )));
        }
        if b.len() != n * d {
            return Err(SimdError::Other(format!(
                "Matrix B size mismatch: expected {}, got {}",
                n * d,
                b.len()
            )));
        }

        // Use metric-specific optimized paths to avoid match overhead in hot loop
        match metric {
            MetricType::DotProduct | MetricType::InnerProduct => {
                self.cdist_f32_dot(a, b, m, n, d)
            }
            MetricType::Cosine => {
                self.cdist_f32_cosine(a, b, m, n, d)
            }
            MetricType::L2Squared => {
                self.cdist_f32_l2sq(a, b, m, n, d)
            }
            MetricType::L2Distance => {
                self.cdist_f32_l2(a, b, m, n, d)
            }
            _ => {
                // Fallback for other metrics
                self.cdist_f32_generic(a, b, m, n, d, metric)
            }
        }
    }

    /// Optimized dot product cdist - uses parallel SimSIMD for best performance
    /// This computes C = A @ B.T using SIMD-accelerated dot products
    #[inline]
    fn cdist_f32_dot(&self, a: &[f32], b: &[f32], m: usize, n: usize, d: usize) -> Result<Vec<f32>> {
        #[cfg(feature = "rayon")]
        {
            use rayon::prelude::*;
            
            // Pre-allocate output buffer
            let mut result = vec![0.0f32; m * n];
            
            // Parallel iteration over output rows
            result
                .par_chunks_mut(n)
                .enumerate()
                .for_each(|(i, row)| {
                    let a_row = &a[i * d..(i + 1) * d];
                    for (j, out) in row.iter_mut().enumerate() {
                        let b_row = &b[j * d..(j + 1) * d];
                        *out = simsimd::SpatialSimilarity::dot(a_row, b_row)
                            .map(|v| v as f32)
                            .unwrap_or(0.0);
                    }
                });
            
            Ok(result)
        }
        
        #[cfg(not(feature = "rayon"))]
        {
            let mut result = Vec::with_capacity(m * n);
            for i in 0..m {
                let a_row = &a[i * d..(i + 1) * d];
                for j in 0..n {
                    let b_row = &b[j * d..(j + 1) * d];
                    result.push(
                        simsimd::SpatialSimilarity::dot(a_row, b_row)
                            .map(|v| v as f32)
                            .unwrap_or(0.0)
                    );
                }
            }
            Ok(result)
        }
    }

    /// Optimized cosine cdist - precompute norms + use dot product for similarity
    /// cosine(a, b) = dot(a, b) / (norm(a) * norm(b))
    #[inline]
    fn cdist_f32_cosine(&self, a: &[f32], b: &[f32], m: usize, n: usize, d: usize) -> Result<Vec<f32>> {
        // On non-Windows with rayon: use BLAS-accelerated ndarray
        #[cfg(all(feature = "rayon", not(target_os = "windows")))]
        {
            use rayon::prelude::*;
            use ndarray::ArrayView2;
            
            // Step 1: Precompute norms in parallel (O(m+n) instead of O(m*n))
            let a_norms: Vec<f32> = (0..m)
                .into_par_iter()
                .map(|i| {
                    let row = &a[i * d..(i + 1) * d];
                    row.iter().map(|x| x * x).sum::<f32>().sqrt()
                })
                .collect();
            
            let b_norms: Vec<f32> = (0..n)
                .into_par_iter()
                .map(|j| {
                    let row = &b[j * d..(j + 1) * d];
                    row.iter().map(|x| x * x).sum::<f32>().sqrt()
                })
                .collect();
            
            // Step 2: Compute dot products using BLAS
            let a_view = ArrayView2::from_shape((m, d), a)
                .map_err(|e| SimdError::Other(format!("Failed to create array view A: {}", e)))?;
            let b_view = ArrayView2::from_shape((n, d), b)
                .map_err(|e| SimdError::Other(format!("Failed to create array view B: {}", e)))?;
            
            #[allow(deprecated)]
            let dots = a_view.dot(&b_view.t());
            #[allow(deprecated)]
            let mut result = dots.into_raw_vec();
            
            // Step 3: Normalize by precomputed norms in parallel
            result
                .par_chunks_mut(n)
                .enumerate()
                .for_each(|(i, row)| {
                    let a_norm = a_norms[i];
                    for (j, val) in row.iter_mut().enumerate() {
                        let b_norm = b_norms[j];
                        let denom = a_norm * b_norm;
                        *val = if denom > 1e-10 { *val / denom } else { 0.0 };
                    }
                });
            
            Ok(result)
        }

        // On Windows with rayon: use parallel SimSIMD (no BLAS)
        #[cfg(all(feature = "rayon", target_os = "windows"))]
        {
            use rayon::prelude::*;
            let mut result = vec![0.0f32; m * n];
            
            result
                .par_chunks_mut(n)
                .enumerate()
                .for_each(|(i, row)| {
                    let a_row = &a[i * d..(i + 1) * d];
                    for (j, val) in row.iter_mut().enumerate() {
                        let b_row = &b[j * d..(j + 1) * d];
                        *val = simsimd::SpatialSimilarity::cosine(a_row, b_row)
                            .map(|dist| (1.0 - dist) as f32)
                            .unwrap_or(f32::NAN);
                    }
                });
            
            Ok(result)
        }

        #[cfg(not(feature = "rayon"))]
        {
            let mut result = Vec::with_capacity(m * n);
            for i in 0..m {
                let a_row = &a[i * d..(i + 1) * d];
                for j in 0..n {
                    let b_row = &b[j * d..(j + 1) * d];
                    result.push(
                        simsimd::SpatialSimilarity::cosine(a_row, b_row)
                            .map(|dist| (1.0 - dist) as f32)
                            .unwrap_or(f32::NAN)
                    );
                }
            }
            Ok(result)
        }
    }

    /// Optimized L2 squared cdist
    #[inline]
    fn cdist_f32_l2sq(&self, a: &[f32], b: &[f32], m: usize, n: usize, d: usize) -> Result<Vec<f32>> {
        #[cfg(feature = "rayon")]
        {
            use rayon::prelude::*;
            let mut result = vec![0.0f32; m * n];
            
            result
                .par_chunks_mut(n)
                .enumerate()
                .for_each(|(i, row)| {
                    let a_row = &a[i * d..(i + 1) * d];
                    for (j, out) in row.iter_mut().enumerate() {
                        let b_row = &b[j * d..(j + 1) * d];
                        *out = simsimd::SpatialSimilarity::sqeuclidean(a_row, b_row)
                            .map(|v| v as f32)
                            .unwrap_or(f32::NAN);
                    }
                });
            Ok(result)
        }

        #[cfg(not(feature = "rayon"))]
        {
            let mut result = Vec::with_capacity(m * n);
            for i in 0..m {
                let a_row = &a[i * d..(i + 1) * d];
                for j in 0..n {
                    let b_row = &b[j * d..(j + 1) * d];
                    result.push(
                        simsimd::SpatialSimilarity::sqeuclidean(a_row, b_row)
                            .map(|v| v as f32)
                            .unwrap_or(f32::NAN)
                    );
                }
            }
            Ok(result)
        }
    }

    /// Optimized L2 distance cdist
    #[inline]
    fn cdist_f32_l2(&self, a: &[f32], b: &[f32], m: usize, n: usize, d: usize) -> Result<Vec<f32>> {
        #[cfg(feature = "rayon")]
        {
            use rayon::prelude::*;
            let mut result = vec![0.0f32; m * n];
            
            result
                .par_chunks_mut(n)
                .enumerate()
                .for_each(|(i, row)| {
                    let a_row = &a[i * d..(i + 1) * d];
                    for (j, out) in row.iter_mut().enumerate() {
                        let b_row = &b[j * d..(j + 1) * d];
                        *out = simsimd::SpatialSimilarity::sqeuclidean(a_row, b_row)
                            .map(|v| (v as f32).sqrt())
                            .unwrap_or(f32::NAN);
                    }
                });
            Ok(result)
        }

        #[cfg(not(feature = "rayon"))]
        {
            let mut result = Vec::with_capacity(m * n);
            for i in 0..m {
                let a_row = &a[i * d..(i + 1) * d];
                for j in 0..n {
                    let b_row = &b[j * d..(j + 1) * d];
                    result.push(
                        simsimd::SpatialSimilarity::sqeuclidean(a_row, b_row)
                            .map(|v| (v as f32).sqrt())
                            .unwrap_or(f32::NAN)
                    );
                }
            }
            Ok(result)
        }
    }

    /// Generic cdist fallback for less common metrics
    fn cdist_f32_generic(&self, a: &[f32], b: &[f32], m: usize, n: usize, d: usize, metric: MetricType) -> Result<Vec<f32>> {
        #[cfg(feature = "rayon")]
        {
            use rayon::prelude::*;
            let mut result = vec![0.0f32; m * n];
            
            result
                .par_chunks_mut(n)
                .enumerate()
                .for_each(|(i, row)| {
                    let a_row = &a[i * d..(i + 1) * d];
                    for (j, out) in row.iter_mut().enumerate() {
                        let b_row = &b[j * d..(j + 1) * d];
                        *out = self.compute_f32(a_row, b_row, metric).unwrap_or(f32::NAN);
                    }
                });
            Ok(result)
        }

        #[cfg(not(feature = "rayon"))]
        {
            let mut result = Vec::with_capacity(m * n);
            for i in 0..m {
                let a_row = &a[i * d..(i + 1) * d];
                for j in 0..n {
                    let b_row = &b[j * d..(j + 1) * d];
                    result.push(self.compute_f32(a_row, b_row, metric).unwrap_or(f32::NAN));
                }
            }
            Ok(result)
        }
    }

    /// Matrix multiplication using SimSIMD dot products: C = A @ B.T
    /// 
    /// Computes the matrix product where each element C[i,j] = dot(A[i,:], B[j,:])
    /// This is equivalent to A @ B.T in NumPy.
    /// 
    /// Optimized with:
    /// - Parallel computation over output rows
    /// - Cache-friendly row-major access pattern
    /// - SimSIMD SIMD-accelerated dot products
    /// 
    /// # Arguments
    /// * `a` - First matrix as contiguous slice, shape [M, K]
    /// * `b` - Second matrix as contiguous slice, shape [N, K] (will compute A @ B.T)
    /// * `m` - Number of rows in A
    /// * `n` - Number of rows in B (columns in result)
    /// * `k` - Shared dimension (columns in A, columns in B)
    /// 
    /// # Returns
    /// Result matrix as flat Vec<f64> in row-major order, shape [M, N]
    pub fn matmul_f64(
        &self,
        a: &[f64],
        b: &[f64],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<Vec<f64>> {
        // matmul is just cdist with DotProduct metric
        self.cdist_f64(a, b, m, n, k, MetricType::DotProduct)
    }

    /// Matrix multiplication for f32: C = A @ B.T
    pub fn matmul_f32(
        &self,
        a: &[f32],
        b: &[f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<Vec<f32>> {
        self.cdist_f32(a, b, m, n, k, MetricType::DotProduct)
    }

    /// Optimized cdist for i8 vectors
    pub fn cdist_i8(
        &self,
        a: &[i8],
        b: &[i8],
        m: usize,
        n: usize,
        d: usize,
        metric: MetricType,
    ) -> Result<Vec<f64>> {
        if a.len() != m * d || b.len() != n * d {
            return Err(SimdError::Other("Matrix size mismatch".to_string()));
        }

        #[cfg(feature = "rayon")]
        {
            use rayon::prelude::*;
            let mut result = vec![0.0f64; m * n];
            result
                .par_chunks_mut(n)
                .enumerate()
                .for_each(|(i, row)| {
                    let a_row = &a[i * d..(i + 1) * d];
                    for (j, out) in row.iter_mut().enumerate() {
                        let b_row = &b[j * d..(j + 1) * d];
                        *out = self.compute_i8(a_row, b_row, metric).unwrap_or(f64::NAN);
                    }
                });
            Ok(result)
        }

        #[cfg(not(feature = "rayon"))]
        {
            let mut result = Vec::with_capacity(m * n);
            for i in 0..m {
                let a_row = &a[i * d..(i + 1) * d];
                for j in 0..n {
                    let b_row = &b[j * d..(j + 1) * d];
                    result.push(self.compute_i8(a_row, b_row, metric).unwrap_or(f64::NAN));
                }
            }
            Ok(result)
        }
    }

    /// Optimized cdist for u8 binary vectors
    pub fn cdist_u8(
        &self,
        a: &[u8],
        b: &[u8],
        m: usize,
        n: usize,
        d: usize,
        metric: MetricType,
    ) -> Result<Vec<f64>> {
        if a.len() != m * d || b.len() != n * d {
            return Err(SimdError::Other("Matrix size mismatch".to_string()));
        }

        #[cfg(feature = "rayon")]
        {
            use rayon::prelude::*;
            let mut result = vec![0.0f64; m * n];
            result
                .par_chunks_mut(n)
                .enumerate()
                .for_each(|(i, row)| {
                    let a_row = &a[i * d..(i + 1) * d];
                    for (j, out) in row.iter_mut().enumerate() {
                        let b_row = &b[j * d..(j + 1) * d];
                        *out = self.compute_u8(a_row, b_row, metric).unwrap_or(f64::NAN);
                    }
                });
            Ok(result)
        }

        #[cfg(not(feature = "rayon"))]
        {
            let mut result = Vec::with_capacity(m * n);
            for i in 0..m {
                let a_row = &a[i * d..(i + 1) * d];
                for j in 0..n {
                    let b_row = &b[j * d..(j + 1) * d];
                    result.push(self.compute_u8(a_row, b_row, metric).unwrap_or(f64::NAN));
                }
            }
            Ok(result)
        }
    }
}

impl Default for SimdBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_capabilities() {
        let backend = SimdBackend::new();
        let caps = backend.capabilities();

        // 至少应该检测到某种 SIMD 支持
        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
        {
            let has_any = caps.has_avx2 || caps.has_avx512 || caps.has_neon || caps.has_sve;
            assert!(has_any, "Should detect at least one SIMD instruction set");
        }
    }

    #[test]
    fn test_dot_product_f64() {
        let backend = SimdBackend::new();
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];

        let result = backend.compute_f64(&a, &b, MetricType::DotProduct).unwrap();
        assert!((result - 32.0).abs() < 1e-10); // 1*4 + 2*5 + 3*6 = 32
    }

    #[test]
    fn test_cosine_similarity_f64() {
        let backend = SimdBackend::new();
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];

        let result = backend.compute_f64(&a, &b, MetricType::Cosine).unwrap();
        assert!((result - 1.0).abs() < 1e-10);

        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let result = backend.compute_f64(&a, &b, MetricType::Cosine).unwrap();
        assert!((result - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_l2_distance_f64() {
        let backend = SimdBackend::new();
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];

        let result = backend.compute_f64(&a, &b, MetricType::L2Distance).unwrap();
        assert!((result - 5.0).abs() < 1e-10); // sqrt(3^2 + 4^2) = 5
    }

    #[test]
    fn test_l2_squared_f64() {
        let backend = SimdBackend::new();
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];

        let result = backend.compute_f64(&a, &b, MetricType::L2Squared).unwrap();
        assert!((result - 25.0).abs() < 1e-10); // 3^2 + 4^2 = 25
    }

    #[test]
    fn test_batch_compute_f64() {
        let backend = SimdBackend::new();
        let query = vec![1.0, 2.0, 3.0];
        let candidates = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![1.0, 1.0, 1.0],
        ];
        let candidate_refs: Vec<&[f64]> = candidates.iter().map(|v| v.as_slice()).collect();

        let results = backend
            .batch_compute_f64(&query, &candidate_refs, MetricType::DotProduct)
            .unwrap();

        assert_eq!(results.len(), 3);
        assert!((results[0] - 1.0).abs() < 1e-10);
        assert!((results[1] - 2.0).abs() < 1e-10);
        assert!((results[2] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_length_mismatch() {
        let backend = SimdBackend::new();
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0, 3.0];

        let result = backend.compute_f64(&a, &b, MetricType::DotProduct);
        assert!(matches!(result, Err(SimdError::LengthMismatch { .. })));
    }
}
