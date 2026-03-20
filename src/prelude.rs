//! Convenience re-exports for common NumPack types.
//!
//! ```rust
//! use numpack::prelude::*;
//! ```

pub use crate::core::error::{NpkError, NpkResult};
pub use crate::core::metadata::{DataType, NpkElement};
pub use crate::io::parallel_io::{ParallelIO, StreamIterator};
