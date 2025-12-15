//! 向量计算引擎模块
//!
//! 提供高性能向量相似度计算，支持：
//! - SimSIMD SIMD 加速（CPU）
//! - 多种数据类型（f64, f32, i8, u8, f16）
//! - 多种度量（dot, cosine, l2, l2sq, kl, js, hamming, jaccard）

pub mod core;
pub mod metrics;
pub mod python_bindings;
pub mod simd_backend;

// 重新导出主要类型
#[allow(unused_imports)]
pub use core::VectorEngine;
#[allow(unused_imports)]
pub use metrics::MetricType;
#[allow(unused_imports)]
pub use simd_backend::{SIMDCapabilities, SimdBackend, SimdError};
