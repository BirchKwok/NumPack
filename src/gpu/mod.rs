//! GPU 加速模块
//! 
//! 提供多种 GPU 后端支持：
//! - Metal Performance Shaders (MPS) - Apple Silicon
//! - WebGPU - 通用跨平台 GPU
//! - CUDA - NVIDIA GPU (未来支持)

use std::fmt;
use thiserror::Error;

use crate::vector_engine::metrics::MetricType;

/// GPU 错误类型
#[derive(Error, Debug)]
pub enum GpuError {
    #[error("GPU 初始化失败: {0}")]
    InitializationError(String),
    
    #[error("GPU 不可用")]
    NotAvailable,
    
    #[error("不支持的操作: {0}")]
    UnsupportedOperation(String),
    
    #[error("计算错误: {0}")]
    ComputeError(String),
    
    #[error("内存错误: {0}")]
    MemoryError(String),
}

pub type GpuResult<T> = Result<T, GpuError>;

/// GPU 后端类型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuBackendType {
    /// Metal Performance Shaders (Apple Silicon)
    MPS,
    /// WebGPU (通用跨平台)
    WebGPU,
    /// NVIDIA CUDA
    CUDA,
    /// AMD ROCm
    ROCm,
}

impl fmt::Display for GpuBackendType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GpuBackendType::MPS => write!(f, "Metal Performance Shaders (MPS)"),
            GpuBackendType::WebGPU => write!(f, "WebGPU"),
            GpuBackendType::CUDA => write!(f, "CUDA"),
            GpuBackendType::ROCm => write!(f, "ROCm"),
        }
    }
}

/// GPU 设备信息
#[derive(Debug, Clone)]
pub struct GpuDevice {
    /// 设备名称
    pub name: String,
    /// 后端类型
    pub backend_type: GpuBackendType,
    /// 显存大小（MB）
    pub memory_mb: usize,
    /// 是否可用
    pub available: bool,
}

// 条件编译：Metal Performance Shaders 后端
#[cfg(feature = "gpu-mps")]
pub mod metal_backend;

// 条件编译：WebGPU 后端
#[cfg(feature = "gpu-wgpu")]
pub mod wgpu_backend;

// 条件编译：CUDA 后端（仅在非 macOS 平台）
#[cfg(all(feature = "gpu-cuda", not(target_os = "macos")))]
pub mod cuda_backend;

// 条件编译：ROCm 后端（仅在 Linux 平台）
#[cfg(all(feature = "gpu-rocm", target_os = "linux"))]
pub mod rocm_backend;

/// GPU 计算引擎
/// 
/// 自动检测并选择最佳可用的 GPU 后端
pub struct GpuEngine {
    backend: Box<dyn GpuBackend + Send + Sync>,
}

/// GPU 后端特征
/// 
/// 所有 GPU 后端必须实现此特征
pub trait GpuBackend {
    /// 获取后端类型
    fn get_backend_type(&self) -> GpuBackendType;
    
    /// 检查是否可用
    fn is_available(&self) -> bool;
    
    /// 获取设备信息
    fn get_device_info(&self) -> Vec<GpuDevice>;
    
    /// 批量计算向量度量
    /// 
    /// # Arguments
    /// 
    /// * `query` - 查询向量
    /// * `candidates` - 候选向量列表
    /// * `metric` - 度量类型
    /// 
    /// # Returns
    /// 
    /// 每个候选向量与查询向量的度量值
    fn batch_compute(
        &mut self,
        query: &[f64],
        candidates: &[&[f64]],
        metric: MetricType,
    ) -> GpuResult<Vec<f64>>;
}

impl GpuEngine {
    /// 创建新的 GPU 引擎
    /// 
    /// 自动检测并选择最佳可用的 GPU 后端
    /// 优先级：MPS > WebGPU > CUDA > ROCm
    pub fn new() -> GpuResult<Self> {
        // 用于记录尝试过的后端（调试用）
        let mut attempted = Vec::new();
        
        // 优先级调整：专用 GPU > WebGPU > 无
        
        // 1. 尝试 Metal Performance Shaders (Apple Silicon) - 最高优先级
        #[cfg(all(feature = "gpu-mps", target_os = "macos"))]
        {
            attempted.push("MPS");
            match metal_backend::MetalBackend::new() {
                Ok(backend) => {
                    if backend.is_available() {
                        eprintln!("✓ 使用 Metal Performance Shaders (MPS)");
                        return Ok(Self {
                            backend: Box::new(backend),
                        });
                    }
                }
                Err(e) => {
                    eprintln!("  MPS 不可用: {}", e);
                }
            }
        }
        
        // 2. 尝试 CUDA (NVIDIA GPU) - 高优先级
        #[cfg(all(feature = "gpu-cuda", not(target_os = "macos")))]
        {
            attempted.push("CUDA");
            match cuda_backend::CudaBackend::new() {
                Ok(backend) => {
                    if backend.is_available() {
                        eprintln!("✓ 使用 CUDA 后端");
                        return Ok(Self { backend: Box::new(backend) });
                    }
                }
                Err(e) => {
                    eprintln!("  CUDA 不可用: {}", e);
                }
            }
        }
        
        // 3. 尝试 ROCm (AMD GPU) - 高优先级
        #[cfg(all(feature = "gpu-rocm", target_os = "linux"))]
        {
            attempted.push("ROCm");
            match rocm_backend::RocmBackend::new() {
                Ok(backend) => {
                    if backend.is_available() {
                        eprintln!("✓ 使用 ROCm 后端");
                        return Ok(Self { backend: Box::new(backend) });
                    }
                }
                Err(e) => {
                    eprintln!("  ROCm 不可用: {}", e);
                }
            }
        }
        
        // 4. 尝试 WebGPU（通用回退） - 低优先级但优于 CPU
        #[cfg(feature = "gpu-wgpu")]
        {
            attempted.push("WebGPU");
            match wgpu_backend::WebGpuBackend::new() {
                Ok(backend) => {
                    if backend.is_available() {
                        eprintln!("✓ 使用 WebGPU 后端（通用 GPU 回退）");
                        return Ok(Self {
                            backend: Box::new(backend),
                        });
                    }
                }
                Err(e) => {
                    eprintln!("  WebGPU 不可用: {}", e);
                }
            }
        }
        
        // 所有 GPU 都不可用 - 将回退到 CPU
        if !attempted.is_empty() {
            eprintln!("ℹ️  GPU 后端都不可用（尝试过: {:?}），将使用 CPU", attempted);
        } else {
            eprintln!("ℹ️  未编译 GPU 支持，将使用 CPU");
        }
        Err(GpuError::NotAvailable)
    }
    
    /// 检查 GPU 是否可用
    pub fn is_gpu_available(&self) -> bool {
        self.backend.is_available()
    }
    
    /// 获取后端类型
    pub fn get_backend_type(&self) -> GpuBackendType {
        self.backend.get_backend_type()
    }
    
    /// 获取 GPU 设备信息
    pub fn get_gpu_info(&self) -> Vec<GpuDevice> {
        self.backend.get_device_info()
    }
    
    /// 批量计算
    pub fn batch_compute(
        &mut self,
        query: &[f64],
        candidates: &[&[f64]],
        metric: MetricType,
    ) -> GpuResult<Vec<f64>> {
        self.backend.batch_compute(query, candidates, metric)
    }
}

impl Default for GpuEngine {
    fn default() -> Self {
        Self::new().unwrap_or_else(|_| {
            panic!("无法初始化 GPU 引擎")
        })
    }
}

