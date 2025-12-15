//! 核心LazyArray模块
//! 提供优化的懒加载数组实现

// 核心功能模块
pub mod core;
pub mod simd_ops;
pub mod traits;

// 不同LazyArray实现
pub mod iterator;
pub mod standard;

// 索引处理
pub mod indexing;

// FFI通信优化
pub mod ffi_optimization;

// Python绑定
pub mod python_bindings;

// 重新导出主要类型（对外API，内部未直接使用时忽略未用警告）
#[allow(unused_imports)]
pub use core::*;
#[allow(unused_imports)]
pub use ffi_optimization::{
    BatchDataCollector, BatchIndexOptimizer, FFIOptimizationConfig, FFIOptimizationStats,
    MetadataCache, ZeroCopyArrayBuilder,
};
#[allow(unused_imports)]
pub use indexing::{AccessPattern, AccessStrategy, IndexResult, IndexType, SliceInfo};
#[allow(unused_imports)]
pub use iterator::LazyArrayIterator;
#[allow(unused_imports)]
pub use simd_ops::*;
#[allow(unused_imports)]
pub use standard::LazyArray;
#[allow(unused_imports)]
pub use standard::LogicalRowMap;
#[allow(unused_imports)]
pub use traits::*;

// Python绑定不重新导出，仅供lib.rs使用
