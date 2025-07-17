//! 内存管理模块
//! 
//! 提供内存池、SIMD处理和零拷贝优化功能
//! 特别关注Windows平台兼容性

pub mod pool;
pub mod simd_processor;
pub mod zero_copy;

pub mod windows_simd;

pub use zero_copy::*;
pub use simd_processor::*;
pub use pool::*;

#[cfg(target_family = "windows")]
pub use windows_simd::*;

// 为非Windows平台提供类型别名
#[cfg(not(target_family = "windows"))]
pub use windows_simd::{WindowsSIMDError, WindowsSafeMemoryAccess}; 