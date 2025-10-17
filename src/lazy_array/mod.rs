//! 核心LazyArray模块
//! 提供优化的懒加载数组实现

// 核心功能模块
pub mod core;
pub mod traits;
pub mod simd_ops;

// 不同LazyArray实现
pub mod standard;
pub mod high_performance;
pub mod iterator;

// 索引处理
pub mod indexing;

// FFI通信优化
pub mod ffi_optimization;

// Python绑定
pub mod python_bindings;

// 重新导出主要类型
pub use core::*;
pub use traits::*;
pub use simd_ops::*;
pub use standard::LazyArray;
pub use high_performance::HighPerformanceLazyArray;
pub use iterator::LazyArrayIterator;
pub use indexing::{IndexType, SliceInfo, IndexResult, AccessPattern, AccessStrategy};
pub use standard::LogicalRowMap;
pub use ffi_optimization::{
    FFIOptimizationConfig,
    BatchDataCollector,
    ZeroCopyArrayBuilder,
    BatchIndexOptimizer,
    MetadataCache,
    FFIOptimizationStats,
};

// Python绑定不重新导出，仅供lib.rs使用 