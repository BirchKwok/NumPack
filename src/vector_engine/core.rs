//! 向量引擎核心实现

use crate::vector_engine::metrics::MetricType;
use crate::vector_engine::simd_backend::{SimdBackend, SimdError, Result};
use std::sync::{Arc, Mutex};

#[cfg(feature = "gpu")]
use crate::gpu::{GpuEngine, GpuBackendType};

/// 向量计算引擎
pub struct VectorEngine {
    /// SimSIMD 后端（CPU）- 公开以允许直接访问
    pub(crate) cpu_backend: SimdBackend,
    /// GPU 后端（可选）
    #[cfg(feature = "gpu")]
    gpu_backend: Option<Arc<Mutex<GpuEngine>>>,
}

impl VectorEngine {
    /// 创建新的向量引擎实例
    /// 
    /// 自动检测 CPU SIMD 能力和 GPU 硬件
    pub fn new() -> Self {
        #[cfg(feature = "gpu")]
        let gpu_backend = {
            match GpuEngine::new() {
                Ok(engine) if engine.is_gpu_available() => {
                    Some(Arc::new(Mutex::new(engine)))
                }
                _ => None,
            }
        };
        
        Self {
            cpu_backend: SimdBackend::new(),
            #[cfg(feature = "gpu")]
            gpu_backend,
        }
    }
    
    /// 检查 GPU 是否可用
    #[cfg(feature = "gpu")]
    pub fn is_gpu_available(&self) -> bool {
        self.gpu_backend.is_some()
    }
    
    /// 检查 GPU 是否可用（无 GPU 特性时）
    #[cfg(not(feature = "gpu"))]
    pub fn is_gpu_available(&self) -> bool {
        false
    }
    
    /// 获取当前可用的后端类型
    #[cfg(feature = "gpu")]
    pub fn get_backend_type(&self, use_gpu: bool) -> String {
        if use_gpu && self.gpu_backend.is_some() {
            if let Some(gpu) = &self.gpu_backend {
                let gpu_lock = gpu.lock().unwrap();
                format!("{}", gpu_lock.get_backend_type())
            } else {
                "CPU (SimSIMD)".to_string()
            }
        } else {
            "CPU (SimSIMD)".to_string()
        }
    }
    
    /// 获取当前可用的后端类型（无 GPU 特性时）
    #[cfg(not(feature = "gpu"))]
    pub fn get_backend_type(&self, _use_gpu: bool) -> String {
        "CPU (SimSIMD)".to_string()
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
        
        let cpu_info = if features.is_empty() {
            "CPU: scalar (no SIMD)".to_string()
        } else {
            format!("CPU: {}", features.join(", "))
        };
        
        // 添加 GPU 信息
        #[cfg(feature = "gpu")]
        {
            if let Some(gpu) = &self.gpu_backend {
                let gpu_lock = gpu.lock().unwrap();
                let gpu_type = gpu_lock.get_backend_type();
                return format!("{}, GPU: {}", cpu_info, gpu_type);
            }
        }
        
        cpu_info
    }
    
    /// 获取 GPU 设备信息
    #[cfg(feature = "gpu")]
    pub fn get_gpu_info(&self) -> Vec<crate::gpu::GpuDevice> {
        if let Some(gpu) = &self.gpu_backend {
            let gpu_lock = gpu.lock().unwrap();
            gpu_lock.get_gpu_info()
        } else {
            Vec::new()
        }
    }
    
    /// 获取 GPU 设备信息（无 GPU 特性时）
    #[cfg(not(feature = "gpu"))]
    pub fn get_gpu_info(&self) -> Vec<String> {
        Vec::new()
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
    /// 单次计算不使用 GPU（GPU 启动开销大）
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
        // 单次计算总是使用 CPU（GPU 开销大）
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
    /// * `use_gpu` - 是否使用 GPU（如果可用）
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
    /// let scores = engine.batch_compute(&query, &candidate_refs, MetricType::Cosine, false).unwrap();
    /// 
    /// // GPU 计算（如果可用）
    /// let scores = engine.batch_compute(&query, &candidate_refs, MetricType::Cosine, true).unwrap();
    /// ```
    pub fn batch_compute(
        &self,
        query: &[f64],
        candidates: &[&[f64]],
        metric: MetricType,
        use_gpu: bool,
    ) -> Result<Vec<f64>> {
        // 回退链：专用 GPU > WebGPU > CPU
        
        #[cfg(feature = "gpu")]
        {
            if use_gpu && self.gpu_backend.is_some() {
                if let Some(gpu) = &self.gpu_backend {
                    let mut gpu_lock = gpu.lock().unwrap();
                    
                    // 尝试 GPU 计算（可能是 MPS/CUDA/ROCm/WebGPU）
                    match gpu_lock.batch_compute(query, candidates, metric) {
                        Ok(scores) => {
                            // GPU 计算成功
                            return Ok(scores);
                        }
                        Err(e) => {
                            // GPU 失败，自动回退到 CPU
                            eprintln!("⚠️  GPU 计算失败: {}, 回退到 CPU", e);
                        }
                    }
                }
            }
        }
        
        // 使用 CPU（默认或 GPU 失败后回退）
        self.cpu_backend.batch_compute_f64(query, candidates, metric)
    }
    
    /// 批量计算 (f32)
    pub fn batch_compute_f32(
        &self,
        query: &[f32],
        candidates: &[&[f32]],
        metric: MetricType,
    ) -> Result<Vec<f32>> {
        // f32 版本只使用 CPU（GPU 内部已转换为 f32）
        self.cpu_backend.batch_compute_f32(query, candidates, metric)
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
        
        let result = engine.compute_metric(&a, &b, MetricType::DotProduct).unwrap();
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

