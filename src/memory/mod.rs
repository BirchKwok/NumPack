//! 内存管理模块
//! 
//! 提供内存池、SIMD处理和零拷贝优化功能

pub mod pool;
pub mod simd_processor;
pub mod zero_copy;

pub use zero_copy::*;
pub use simd_processor::*;
// 暂时注释掉未使用的导入，避免警告
// pub use pool::*; 