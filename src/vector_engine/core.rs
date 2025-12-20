//! 向量引擎核心实现

use crate::vector_engine::metrics::MetricType;
use crate::vector_engine::simd_backend::{Result, SimdBackend};

/// 向量计算引擎
pub struct VectorEngine {
    /// SimSIMD 后端（CPU）- 公开以允许直接访问
    pub(crate) cpu_backend: SimdBackend,
}

impl VectorEngine {
    /// 创建新的向量引擎实例
    ///
    /// 自动检测 CPU SIMD 能力
    pub fn new() -> Self {
        Self {
            cpu_backend: SimdBackend::new(),
        }
    }

    /// 获取 CPU SIMD 能力信息
    pub fn capabilities(&self) -> String {
        let caps = self.cpu_backend.capabilities();
        let mut features = Vec::new();

        if caps.has_avx512 {
            features.push("AVX-512");
        }
        if caps.has_avx2 {
            features.push("AVX2");
        }
        if caps.has_neon {
            features.push("NEON");
        }
        if caps.has_sve {
            features.push("SVE");
        }

        if features.is_empty() {
            "CPU: scalar (no SIMD)".to_string()
        } else {
            format!("CPU: {}", features.join(", "))
        }
    }

    /// 计算两个向量的度量值
    ///
    /// # Arguments
    ///
    /// * `a` - 第一个向量
    /// * `b` - 第二个向量
    /// * `metric` - 度量类型
    ///
    /// # Returns
    ///
    /// 度量值（距离或相似度）
    ///
    /// # Note
    ///
    /// 单次计算使用 CPU
    ///
    /// # Example
    ///
    /// ```
    /// use numpack::vector_engine::{VectorEngine, MetricType};
    ///
    /// let engine = VectorEngine::new();
    /// let a = vec![1.0, 2.0, 3.0];
    /// let b = vec![4.0, 5.0, 6.0];
    ///
    /// let similarity = engine.compute_metric(&a, &b, MetricType::Cosine).unwrap();
    /// ```
    pub fn compute_metric(&self, a: &[f64], b: &[f64], metric: MetricType) -> Result<f64> {
        // 单次计算使用 CPU
        self.cpu_backend.compute_f64(a, b, metric)
    }

    /// 计算两个 f32 向量的度量值
    pub fn compute_metric_f32(&self, a: &[f32], b: &[f32], metric: MetricType) -> Result<f32> {
        self.cpu_backend.compute_f32(a, b, metric)
    }

    /// 批量计算：query 向量与多个候选向量的度量
    ///
    /// # Arguments
    ///
    /// * `query` - 查询向量
    /// * `candidates` - 候选向量列表
    /// * `metric` - 度量类型
    ///
    /// # Returns
    ///
    /// 度量值列表
    ///
    /// # Example
    ///
    /// ```
    /// use numpack::vector_engine::{VectorEngine, MetricType};
    ///
    /// let engine = VectorEngine::new();
    /// let query = vec![1.0, 2.0, 3.0];
    /// let candidates = vec![
    ///     vec![1.0, 0.0, 0.0],
    ///     vec![0.0, 1.0, 0.0],
    ///     vec![1.0, 1.0, 1.0],
    /// ];
    /// let candidate_refs: Vec<&[f64]> = candidates.iter().map(|v| v.as_slice()).collect();
    ///
    /// // CPU 计算
    /// let scores = engine.batch_compute(&query, &candidate_refs, MetricType::Cosine).unwrap();
    /// ```
    pub fn batch_compute(
        &self,
        query: &[f64],
        candidates: &[&[f64]],
        metric: MetricType,
    ) -> Result<Vec<f64>> {
        // 使用 CPU 计算
        self.cpu_backend
            .batch_compute_f64(query, candidates, metric)
    }

    /// 批量计算 (f32)
    pub fn batch_compute_f32(
        &self,
        query: &[f32],
        candidates: &[&[f32]],
        metric: MetricType,
    ) -> Result<Vec<f32>> {
        // f32 版本使用 CPU
        self.cpu_backend
            .batch_compute_f32(query, candidates, metric)
    }

    /// 批量计算 (i8 - 整数向量)
    pub fn batch_compute_i8(
        &self,
        query: &[i8],
        candidates: &[&[i8]],
        metric: MetricType,
    ) -> Result<Vec<f64>> {
        // i8 使用 CPU SimSIMD 加速
        self.cpu_backend.batch_compute_i8(query, candidates, metric)
    }

    /// 批量计算 (u8 - 二进制向量)
    pub fn batch_compute_u8(
        &self,
        query: &[u8],
        candidates: &[&[u8]],
        metric: MetricType,
    ) -> Result<Vec<f64>> {
        // u8 使用 CPU SimSIMD 加速（hamming/jaccard）
        self.cpu_backend.batch_compute_u8(query, candidates, metric)
    }

    /// 批量计算 (i16 - 16位整数向量)
    pub fn batch_compute_i16(
        &self,
        query: &[i16],
        candidates: &[&[i16]],
        metric: MetricType,
    ) -> Result<Vec<f64>> {
        self.cpu_backend.batch_compute_i16(query, candidates, metric)
    }

    /// 批量计算 (i32 - 32位整数向量)
    pub fn batch_compute_i32(
        &self,
        query: &[i32],
        candidates: &[&[i32]],
        metric: MetricType,
    ) -> Result<Vec<f64>> {
        self.cpu_backend.batch_compute_i32(query, candidates, metric)
    }

    /// 批量计算 (i64 - 64位整数向量)
    pub fn batch_compute_i64(
        &self,
        query: &[i64],
        candidates: &[&[i64]],
        metric: MetricType,
    ) -> Result<Vec<f64>> {
        self.cpu_backend.batch_compute_i64(query, candidates, metric)
    }

    /// 批量计算 (u16 - 16位无符号整数向量)
    pub fn batch_compute_u16(
        &self,
        query: &[u16],
        candidates: &[&[u16]],
        metric: MetricType,
    ) -> Result<Vec<f64>> {
        self.cpu_backend.batch_compute_u16(query, candidates, metric)
    }

    /// 批量计算 (u32 - 32位无符号整数向量)
    pub fn batch_compute_u32(
        &self,
        query: &[u32],
        candidates: &[&[u32]],
        metric: MetricType,
    ) -> Result<Vec<f64>> {
        self.cpu_backend.batch_compute_u32(query, candidates, metric)
    }

    /// 批量计算 (u64 - 64位无符号整数向量)
    pub fn batch_compute_u64(
        &self,
        query: &[u64],
        candidates: &[&[u64]],
        metric: MetricType,
    ) -> Result<Vec<f64>> {
        self.cpu_backend.batch_compute_u64(query, candidates, metric)
    }

    /// 批量计算 (f16 - 半精度浮点)
    pub fn batch_compute_f16(
        &self,
        query: &[half::f16],
        candidates: &[&[half::f16]],
        metric: MetricType,
    ) -> Result<Vec<f64>> {
        self.cpu_backend.batch_compute_f16(query, candidates, metric)
    }
}

// ============== cdist / matmul optimized API ==============

impl VectorEngine {
    /// Compute pairwise distances/similarities between two matrices (cdist)
    /// 
    /// Equivalent to `scipy.spatial.distance.cdist`
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
        self.cpu_backend.cdist_f64(a, b, m, n, d, metric)
    }

    /// Compute pairwise distances/similarities between two f32 matrices
    pub fn cdist_f32(
        &self,
        a: &[f32],
        b: &[f32],
        m: usize,
        n: usize,
        d: usize,
        metric: MetricType,
    ) -> Result<Vec<f32>> {
        self.cpu_backend.cdist_f32(a, b, m, n, d, metric)
    }

    /// Matrix multiplication using SimSIMD: C = A @ B.T
    /// 
    /// Computes the matrix product where each element C[i,j] = dot(A[i,:], B[j,:])
    /// 
    /// # Arguments
    /// * `a` - First matrix as contiguous slice, shape [M, K]
    /// * `b` - Second matrix as contiguous slice, shape [N, K]
    /// * `m` - Number of rows in A
    /// * `n` - Number of rows in B
    /// * `k` - Shared dimension
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
        self.cpu_backend.matmul_f64(a, b, m, n, k)
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
        self.cpu_backend.matmul_f32(a, b, m, n, k)
    }

    /// cdist for i8 vectors
    pub fn cdist_i8(
        &self,
        a: &[i8],
        b: &[i8],
        m: usize,
        n: usize,
        d: usize,
        metric: MetricType,
    ) -> Result<Vec<f64>> {
        self.cpu_backend.cdist_i8(a, b, m, n, d, metric)
    }

    /// cdist for u8 binary vectors
    pub fn cdist_u8(
        &self,
        a: &[u8],
        b: &[u8],
        m: usize,
        n: usize,
        d: usize,
        metric: MetricType,
    ) -> Result<Vec<f64>> {
        self.cpu_backend.cdist_u8(a, b, m, n, d, metric)
    }
}

impl Default for VectorEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_creation() {
        let engine = VectorEngine::new();
        let caps = engine.capabilities();
        println!("SIMD capabilities: {}", caps);
        assert!(!caps.is_empty());
    }

    #[test]
    fn test_compute_metric() {
        let engine = VectorEngine::new();
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];

        let result = engine
            .compute_metric(&a, &b, MetricType::DotProduct)
            .unwrap();
        assert!((result - 32.0).abs() < 1e-10);
    }

    #[test]
    fn test_batch_compute() {
        let engine = VectorEngine::new();
        let query = vec![1.0, 2.0, 3.0];
        let candidates = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![1.0, 1.0, 1.0],
        ];
        let candidate_refs: Vec<&[f64]> = candidates.iter().map(|v| v.as_slice()).collect();

        let results = engine
            .batch_compute(&query, &candidate_refs, MetricType::DotProduct)
            .unwrap();

        assert_eq!(results.len(), 3);
        assert!((results[0] - 1.0).abs() < 1e-10);
        assert!((results[1] - 2.0).abs() < 1e-10);
        assert!((results[2] - 6.0).abs() < 1e-10);
    }
}
