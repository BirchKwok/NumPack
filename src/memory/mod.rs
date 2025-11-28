//! 内存管理模块
//!
//! 提供高性能的内存操作、零拷贝访问和SIMD优化

pub mod handle_manager;
pub mod numpack_simd;
pub mod pool;
pub mod simd_optimized;
pub mod simd_processor;
pub mod zero_copy;

// 重新导出主要组件
pub use handle_manager::{get_handle_manager, CleanupConfig, HandleManager, HandleStats};
pub use numpack_simd::*;
pub use pool::*;
pub use simd_processor::*;
pub use zero_copy::*;
