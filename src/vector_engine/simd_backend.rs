//! SimSIMD 后端集成层
//!
//! 封装 SimSIMD 库的 FFI 调用，提供类型安全的 Rust 接口

use crate::vector_engine::metrics::MetricType;
use std::fmt;

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
    /// 🚀 关键优化：使用 Rayon 并行计算以充分利用多核 CPU
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
    /// 🚀 关键优化：使用 Rayon 并行计算
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
