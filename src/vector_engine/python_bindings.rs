//! Python FFI bindings
//!
//! Exposes vector engine functionality to Python
//!
//! This module provides two main classes:
//! - `VectorSearch`: Pure in-memory vector similarity computation
//! - `StreamingVectorSearch`: Streaming vector search from files (memory-efficient)

use memmap2::MmapOptions;
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArrayDyn};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use std::path::Path;

use crate::storage::binary_metadata::{BinaryDataType, BinaryMetadataStore};
use crate::vector_engine::core::VectorEngine;
use crate::vector_engine::metrics::MetricType;

/// Python wrapper for in-memory vector search operations
///
/// This is a Python binding for the SimSIMD Rust library, providing high-performance
/// vector similarity computation with SIMD acceleration (AVX2, AVX-512, NEON, SVE).
///
/// VectorSearch is designed for pure in-memory operations where all data is loaded
/// into memory. For large datasets that don't fit in memory, use StreamingVectorSearch.
///
/// Supported data types:
/// - float64 (f64): Double precision floating point
/// - float32 (f32): Single precision floating point
/// - float16 (f16): Half precision floating point (not yet implemented)
/// - int8 (i8): 8-bit signed integers
/// - uint8 (u8): Binary vectors (for hamming/jaccard metrics)
///
/// Supported metrics:
/// - "dot", "dot_product", "dotproduct": Dot product (similarity, higher is better)
/// - "cos", "cosine", "cosine_similarity": Cosine similarity (similarity, range [-1, 1], higher is better)
/// - "l2", "euclidean", "l2_distance": L2/Euclidean distance (distance, lower is better)
/// - "l2sq", "l2_squared", "squared_euclidean": Squared L2 distance (distance, lower is better, faster than l2)
/// - "hamming": Hamming distance for binary vectors (distance, lower is better)
/// - "jaccard": Jaccard distance for binary vectors (distance, lower is better)
/// - "kl", "kl_divergence": Kullback-Leibler divergence (distance, lower is better)
/// - "js", "js_divergence": Jensen-Shannon divergence (distance, lower is better)
/// - "inner", "inner_product": Inner product (similarity, higher is better, same as dot)
#[pyclass(module = "numpack.vector_engine", name = "VectorSearch")]
pub struct PyVectorSearch {
    engine: VectorEngine,
}

#[pymethods]
impl PyVectorSearch {
    /// Create a new VectorSearch instance for in-memory vector operations
    ///
    /// Automatically detects CPU SIMD capabilities (AVX2, AVX-512, NEON, SVE).
    #[new]
    pub fn new() -> Self {
        Self {
            engine: VectorEngine::new(),
        }
    }

    /// Get SIMD capabilities information
    ///
    /// Returns:
    ///     str: A string describing detected SIMD features (e.g., "CPU: AVX2, AVX-512")
    pub fn capabilities(&self) -> String {
        self.engine.capabilities()
    }

    /// Compute the metric value between two vectors
    ///
    /// Supports multiple data types, automatically selects the optimal computation path
    /// based on input dtype:
    /// - int8 (i8): Integer vectors (supports: dot, cosine, l2, l2sq)
    /// - float32 (f32): Single precision floating point (all metrics)
    /// - float64 (f64): Double precision floating point (all metrics)
    /// - uint8 (u8): Binary vectors (supports: hamming, jaccard only)
    ///
    /// Args:
    ///     a: First vector (1D numpy array, any supported dtype)
    ///     b: Second vector (1D numpy array, same dtype and length as a)
    ///     metric: Metric type string. Supported values:
    ///         - "dot", "dot_product", "dotproduct": Dot product
    ///         - "cos", "cosine", "cosine_similarity": Cosine similarity
    ///         - "l2", "euclidean", "l2_distance": L2/Euclidean distance
    ///         - "l2sq", "l2_squared", "squared_euclidean": Squared L2 distance
    ///         - "hamming": Hamming distance (for binary vectors)
    ///         - "jaccard": Jaccard distance (for binary vectors)
    ///         - "kl", "kl_divergence": Kullback-Leibler divergence
    ///         - "js", "js_divergence": Jensen-Shannon divergence
    ///         - "inner", "inner_product": Inner product
    ///
    /// Returns:
    ///     float: The computed metric value (float64)
    ///
    /// Raises:
    ///     TypeError: If dtype is not supported or a/b dtypes don't match
    ///     ValueError: If the metric is unknown or computation fails
    ///     ValueError: If vector dimensions don't match
    #[pyo3(signature = (a, b, metric))]
    pub fn compute_metric(
        &self,
        py: Python,
        a: &Bound<'_, PyAny>,
        b: &Bound<'_, PyAny>,
        metric: &str,
    ) -> PyResult<f64> {
        // Parse metric type
        let metric_type = MetricType::from_str(metric)
            .ok_or_else(|| PyValueError::new_err(format!("Unknown metric: {}", metric)))?;

        // Get array dtypes
        let a_dtype = a.getattr("dtype")?.str()?.to_string();
        let b_dtype = b.getattr("dtype")?.str()?.to_string();

        // If dtypes are different, convert both to float64 for computation
        if a_dtype != b_dtype {
            let numpy = py.import("numpy")?;
            let a_f64 = a.call_method1("astype", (numpy.getattr("float64")?,))?;
            let b_f64 = b.call_method1("astype", (numpy.getattr("float64")?,))?;
            return self.compute_metric_f64(py, &a_f64, &b_f64, metric_type);
        }

        // Dispatch to different computation paths based on dtype
        match a_dtype.as_str() {
            "float64" => self.compute_metric_f64(py, a, b, metric_type),
            "float32" => self.compute_metric_f32(py, a, b, metric_type),
            "float16" => self.compute_metric_f16(py, a, b, metric_type),
            "int8" => self.compute_metric_i8(py, a, b, metric_type),
            "int16" => self.compute_metric_i16(py, a, b, metric_type),
            "int32" => self.compute_metric_i32(py, a, b, metric_type),
            "int64" => self.compute_metric_i64(py, a, b, metric_type),
            "uint8" => self.compute_metric_u8(py, a, b, metric_type),
            "uint16" => self.compute_metric_u16(py, a, b, metric_type),
            "uint32" => self.compute_metric_u32(py, a, b, metric_type),
            "uint64" => self.compute_metric_u64(py, a, b, metric_type),
            _ => Err(PyTypeError::new_err(format!(
                "Unsupported dtype: {}. Supported: float64, float32, float16, int8, int16, int32, int64, uint8, uint16, uint32, uint64",
                a_dtype
            ))),
        }
    }

    /// Batch compute metrics between a query vector and multiple candidate vectors
    ///
    /// Supports multiple data types, automatically selects the optimal computation path
    /// based on input dtype:
    /// - int8 (i8): Integer vectors (supports: dot, cosine, l2, l2sq)
    /// - float16 (f16): Half precision floating point (all metrics, not yet implemented)
    /// - float32 (f32): Single precision floating point (all metrics)
    /// - float64 (f64): Double precision floating point (all metrics)
    /// - uint8 (u8): Binary vectors (supports: hamming, jaccard only)
    ///
    /// Args:
    ///     query: Query vector (1D numpy array, any supported dtype)
    ///     candidates: Candidate vectors matrix (2D numpy array, shape: [N, D], same dtype as query)
    ///     metric: Metric type string. Supported values:
    ///         - "dot", "dot_product", "dotproduct": Dot product
    ///         - "cos", "cosine", "cosine_similarity": Cosine similarity
    ///         - "l2", "euclidean", "l2_distance": L2/Euclidean distance
    ///         - "l2sq", "l2_squared", "squared_euclidean": Squared L2 distance
    ///         - "hamming": Hamming distance (for uint8 binary vectors only)
    ///         - "jaccard": Jaccard distance (for uint8 binary vectors only)
    ///         - "kl", "kl_divergence": Kullback-Leibler divergence
    ///         - "js", "js_divergence": Jensen-Shannon divergence
    ///         - "inner", "inner_product": Inner product
    ///
    /// Returns:
    ///     numpy.ndarray: Metric values array (1D numpy array of float64, shape: [N])
    ///
    /// Raises:
    ///     TypeError: If dtype is not supported or query/candidates dtypes don't match
    ///     ValueError: If metric is unknown, computation fails, or dimensions don't match
    ///
    /// Note:
    ///     For large batches (>= 500 candidates), computation is automatically parallelized
    ///     using multiple CPU cores. Smaller batches use serial computation to avoid
    ///     thread pool overhead.
    #[pyo3(signature = (query, candidates, metric))]
    pub fn batch_compute(
        &self,
        py: Python,
        query: &Bound<'_, PyAny>,
        candidates: &Bound<'_, PyAny>,
        metric: &str,
    ) -> PyResult<Py<PyArray1<f64>>> {
        // Parse metric type
        let metric_type = MetricType::from_str(metric)
            .ok_or_else(|| PyValueError::new_err(format!("Unknown metric: {}", metric)))?;

        // Get array dtypes
        let query_dtype = query.getattr("dtype")?.str()?.to_string();
        let candidates_dtype = candidates.getattr("dtype")?.str()?.to_string();

        // If dtypes are different, convert both to float64 for computation
        if query_dtype != candidates_dtype {
            let numpy = py.import("numpy")?;
            let query_f64 = query.call_method1("astype", (numpy.getattr("float64")?,))?;
            let candidates_f64 = candidates.call_method1("astype", (numpy.getattr("float64")?,))?;
            return self.batch_compute_f64(py, &query_f64, &candidates_f64, metric_type);
        }

        // Dispatch to different computation paths based on dtype
        // This avoids unnecessary type conversions and directly uses SimSIMD's native support
        match query_dtype.as_str() {
            "float64" => self.batch_compute_f64(py, query, candidates, metric_type),
            "float32" => self.batch_compute_f32(py, query, candidates, metric_type),
            "float16" => self.batch_compute_f16_impl(py, query, candidates, metric_type),
            "int8" => self.batch_compute_i8(py, query, candidates, metric_type),
            "int16" => self.batch_compute_i16(py, query, candidates, metric_type),
            "int32" => self.batch_compute_i32(py, query, candidates, metric_type),
            "int64" => self.batch_compute_i64(py, query, candidates, metric_type),
            "uint8" => self.batch_compute_u8(py, query, candidates, metric_type),
            "uint16" => self.batch_compute_u16(py, query, candidates, metric_type),
            "uint32" => self.batch_compute_u32(py, query, candidates, metric_type),
            "uint64" => self.batch_compute_u64(py, query, candidates, metric_type),
            _ => Err(PyTypeError::new_err(format!(
                "Unsupported dtype: {}. Supported: float64, float32, float16, int8, int16, int32, int64, uint8, uint16, uint32, uint64",
                query_dtype
            ))),
        }
    }

    /// Segmented Top-K search: Process large datasets in segments with incremental merging
    ///
    /// This method is designed for streaming/segmented processing of large candidate sets.
    /// It computes scores for a batch of candidates and merges results with the current top-k
    /// in a single FFI call, eliminating intermediate Python array allocation.
    ///
    /// Args:
    ///     query: Query vector (1D numpy array)
    ///     candidates: Candidate vectors batch (2D numpy array, shape: [N, D])
    ///     metric: Metric type string
    ///     global_offset: Starting global index for this batch
    ///     current_indices: Current top-k indices (1D numpy array of uint64, can be empty)
    ///     current_scores: Current top-k scores (1D numpy array of float64, can be empty)
    ///     k: Number of top results to maintain
    ///     is_similarity: true = higher is better, false = lower is better
    ///
    /// Returns:
    ///     tuple: (new_indices, new_scores)
    ///         - new_indices: Updated top-k indices (1D numpy array of uint64, shape: [<=k])
    ///         - new_scores: Updated top-k scores (1D numpy array of float64, shape: [<=k])
    #[pyo3(signature = (query, candidates, metric, global_offset, current_indices, current_scores, k, is_similarity))]
    pub fn segmented_top_k_search(
        &self,
        py: Python,
        query: &Bound<'_, PyAny>,
        candidates: &Bound<'_, PyAny>,
        metric: &str,
        global_offset: u64,
        current_indices: &Bound<'_, PyAny>,
        current_scores: &Bound<'_, PyAny>,
        k: usize,
        is_similarity: bool,
    ) -> PyResult<(Py<PyArray1<u64>>, Py<PyArray1<f64>>)> {
        // Parse metric type
        let metric_type = MetricType::from_str(metric)
            .ok_or_else(|| PyValueError::new_err(format!("Unknown metric: {}", metric)))?;

        // Get array dtypes
        let query_dtype = query.getattr("dtype")?.str()?.to_string();
        let candidates_dtype = candidates.getattr("dtype")?.str()?.to_string();

        // Extract current top-k (may be empty on first call)
        let curr_idx_arr: PyReadonlyArrayDyn<u64> = current_indices.extract()?;
        let curr_scores_arr: PyReadonlyArrayDyn<f64> = current_scores.extract()?;
        let curr_idx_slice = curr_idx_arr.as_slice()?;
        let curr_scores_slice = curr_scores_arr.as_slice()?;

        // If dtypes are different, convert both to float64 for computation
        if query_dtype != candidates_dtype {
            let numpy = py.import("numpy")?;
            let query_f64 = query.call_method1("astype", (numpy.getattr("float64")?,))?;
            let candidates_f64 = candidates.call_method1("astype", (numpy.getattr("float64")?,))?;
            return self.segmented_top_k_search_f64(
                py,
                &query_f64,
                &candidates_f64,
                metric_type,
                global_offset,
                curr_idx_slice,
                curr_scores_slice,
                k,
                is_similarity,
            );
        }

        // Dispatch based on dtype
        match query_dtype.as_str() {
            "float32" => self.segmented_top_k_search_f32(
                py,
                query,
                candidates,
                metric_type,
                global_offset,
                curr_idx_slice,
                curr_scores_slice,
                k,
                is_similarity,
            ),
            "float64" => self.segmented_top_k_search_f64(
                py,
                query,
                candidates,
                metric_type,
                global_offset,
                curr_idx_slice,
                curr_scores_slice,
                k,
                is_similarity,
            ),
            "float16" => self.segmented_top_k_search_f16(
                py,
                query,
                candidates,
                metric_type,
                global_offset,
                curr_idx_slice,
                curr_scores_slice,
                k,
                is_similarity,
            ),
            "int8" => self.segmented_top_k_search_i8(
                py,
                query,
                candidates,
                metric_type,
                global_offset,
                curr_idx_slice,
                curr_scores_slice,
                k,
                is_similarity,
            ),
            "int16" => self.segmented_top_k_search_i16(
                py,
                query,
                candidates,
                metric_type,
                global_offset,
                curr_idx_slice,
                curr_scores_slice,
                k,
                is_similarity,
            ),
            "int32" => self.segmented_top_k_search_i32(
                py,
                query,
                candidates,
                metric_type,
                global_offset,
                curr_idx_slice,
                curr_scores_slice,
                k,
                is_similarity,
            ),
            "int64" => self.segmented_top_k_search_i64(
                py,
                query,
                candidates,
                metric_type,
                global_offset,
                curr_idx_slice,
                curr_scores_slice,
                k,
                is_similarity,
            ),
            "uint8" => self.segmented_top_k_search_u8(
                py,
                query,
                candidates,
                metric_type,
                global_offset,
                curr_idx_slice,
                curr_scores_slice,
                k,
                is_similarity,
            ),
            "uint16" => self.segmented_top_k_search_u16(
                py,
                query,
                candidates,
                metric_type,
                global_offset,
                curr_idx_slice,
                curr_scores_slice,
                k,
                is_similarity,
            ),
            "uint32" => self.segmented_top_k_search_u32(
                py,
                query,
                candidates,
                metric_type,
                global_offset,
                curr_idx_slice,
                curr_scores_slice,
                k,
                is_similarity,
            ),
            "uint64" => self.segmented_top_k_search_u64(
                py,
                query,
                candidates,
                metric_type,
                global_offset,
                curr_idx_slice,
                curr_scores_slice,
                k,
                is_similarity,
            ),
            _ => Err(PyTypeError::new_err(format!(
                "Unsupported dtype: {}. Supported: float64, float32, float16, int8, int16, int32, int64, uint8, uint16, uint32, uint64",
                query_dtype
            ))),
        }
    }

    /// Top-K search: Find the k most similar/closest vectors
    ///
    /// Supports multiple data types (automatically detects dtype):
    /// - int8 (i8), float32 (f32), float64 (f64), uint8 (u8)
    ///   (same as batch_compute)
    ///
    /// Args:
    ///     query: Query vector (1D numpy array, any supported dtype)
    ///     candidates: Candidate vectors matrix (2D numpy array, same dtype as query)
    ///     metric: Metric type string. Supported values:
    ///         - "dot", "dot_product", "dotproduct": Dot product
    ///         - "cos", "cosine", "cosine_similarity": Cosine similarity
    ///         - "l2", "euclidean", "l2_distance": L2/Euclidean distance
    ///         - "l2sq", "l2_squared", "squared_euclidean": Squared L2 distance
    ///         - "hamming": Hamming distance (for uint8 binary vectors only)
    ///         - "jaccard": Jaccard distance (for uint8 binary vectors only)
    ///         - "kl", "kl_divergence": Kullback-Leibler divergence
    ///         - "js", "js_divergence": Jensen-Shannon divergence
    ///         - "inner", "inner_product": Inner product
    ///     k: Number of results to return
    ///
    /// Returns:
    ///     tuple: (indices, scores)
    ///         - indices: Array of candidate indices (1D numpy array of uint64, shape: [k])
    ///         - scores: Array of metric scores (1D numpy array of float64, shape: [k])
    ///
    ///         For similarity metrics (dot, cosine, inner): returns k highest scores
    ///         For distance metrics (l2, l2sq, hamming, jaccard, kl, js): returns k lowest scores
    ///
    /// Raises:
    ///     TypeError: If dtype is not supported or query/candidates dtypes don't match
    ///     ValueError: If metric is unknown, computation fails, or dimensions don't match
    #[pyo3(signature = (query, candidates, metric, k))]
    pub fn top_k_search(
        &self,
        py: Python,
        query: &Bound<'_, PyAny>,
        candidates: &Bound<'_, PyAny>,
        metric: &str,
        k: usize,
    ) -> PyResult<(Py<PyArray1<usize>>, Py<PyArray1<f64>>)> {
        // Parse metric type
        let metric_type = MetricType::from_str(metric)
            .ok_or_else(|| PyValueError::new_err(format!("Unknown metric: {}", metric)))?;

        // Get array dtypes
        let query_dtype = query.getattr("dtype")?.str()?.to_string();
        let candidates_dtype = candidates.getattr("dtype")?.str()?.to_string();

        // If dtypes are different, convert both to float64 for computation
        if query_dtype != candidates_dtype {
            let numpy = py.import("numpy")?;
            let query_f64 = query.call_method1("astype", (numpy.getattr("float64")?,))?;
            let candidates_f64 = candidates.call_method1("astype", (numpy.getattr("float64")?,))?;
            return self.top_k_search_f64(py, &query_f64, &candidates_f64, metric_type, k);
        }

        // Dispatch based on dtype
        match query_dtype.as_str() {
            "float64" => self.top_k_search_f64(py, query, candidates, metric_type, k),
            "float32" => self.top_k_search_f32(py, query, candidates, metric_type, k),
            "float16" => self.top_k_search_f16(py, query, candidates, metric_type, k),
            "int8" => self.top_k_search_i8(py, query, candidates, metric_type, k),
            "int16" => self.top_k_search_i16(py, query, candidates, metric_type, k),
            "int32" => self.top_k_search_i32(py, query, candidates, metric_type, k),
            "int64" => self.top_k_search_i64(py, query, candidates, metric_type, k),
            "uint8" => self.top_k_search_u8(py, query, candidates, metric_type, k),
            "uint16" => self.top_k_search_u16(py, query, candidates, metric_type, k),
            "uint32" => self.top_k_search_u32(py, query, candidates, metric_type, k),
            "uint64" => self.top_k_search_u64(py, query, candidates, metric_type, k),
            _ => Err(PyTypeError::new_err(format!(
                "Unsupported dtype: {}. Supported: float64, float32, float16, int8, int16, int32, int64, uint8, uint16, uint32, uint64",
                query_dtype
            ))),
        }
    }

    /// Batch multi-query Top-K search (optimized for multiple queries)
    ///
    /// This method processes multiple queries in a single FFI call, significantly
    /// reducing Python-Rust boundary overhead compared to calling top_k_search repeatedly.
    ///
    /// Performance: ~30-50% faster than calling top_k_search in a loop.
    ///
    /// Args:
    ///     queries: Multiple query vectors (2D numpy array, shape: [N, D])
    ///     candidates: Candidate vectors matrix (2D numpy array, shape: [M, D])
    ///     metric: Metric type string ('cosine', 'dot', 'l2', etc.)
    ///     k: Number of top results to return per query
    ///
    /// Returns:
    ///     tuple: (all_indices, all_scores)
    ///         - all_indices: 1D array of shape [N*k], can reshape to [N, k]
    ///         - all_scores: 1D array of shape [N*k], can reshape to [N, k]
    #[pyo3(signature = (queries, candidates, metric, k))]
    pub fn multi_query_top_k(
        &self,
        py: Python,
        queries: &Bound<'_, PyAny>,
        candidates: &Bound<'_, PyAny>,
        metric: &str,
        k: usize,
    ) -> PyResult<(Py<PyArray1<u64>>, Py<PyArray1<f64>>)> {
        // Parse metric type
        let metric_type = MetricType::from_str(metric)
            .ok_or_else(|| PyValueError::new_err(format!("Unknown metric: {}", metric)))?;

        // Get array dtypes
        let queries_dtype = queries.getattr("dtype")?.str()?.to_string();
        let candidates_dtype = candidates.getattr("dtype")?.str()?.to_string();

        // If dtypes are different, convert both to float64 for computation
        if queries_dtype != candidates_dtype {
            let numpy = py.import("numpy")?;
            let queries_f64 = queries.call_method1("astype", (numpy.getattr("float64")?,))?;
            let candidates_f64 = candidates.call_method1("astype", (numpy.getattr("float64")?,))?;
            return self.multi_query_top_k_f64(py, &queries_f64, &candidates_f64, metric_type, k);
        }

        // Dispatch based on dtype
        match queries_dtype.as_str() {
            "float32" => self.multi_query_top_k_f32(py, queries, candidates, metric_type, k),
            "float64" => self.multi_query_top_k_f64(py, queries, candidates, metric_type, k),
            "float16" => self.multi_query_top_k_f16(py, queries, candidates, metric_type, k),
            "int8" => self.multi_query_top_k_i8(py, queries, candidates, metric_type, k),
            "int16" => self.multi_query_top_k_i16(py, queries, candidates, metric_type, k),
            "int32" => self.multi_query_top_k_i32(py, queries, candidates, metric_type, k),
            "int64" => self.multi_query_top_k_i64(py, queries, candidates, metric_type, k),
            "uint8" => self.multi_query_top_k_u8(py, queries, candidates, metric_type, k),
            "uint16" => self.multi_query_top_k_u16(py, queries, candidates, metric_type, k),
            "uint32" => self.multi_query_top_k_u32(py, queries, candidates, metric_type, k),
            "uint64" => self.multi_query_top_k_u64(py, queries, candidates, metric_type, k),
            _ => Err(PyTypeError::new_err(format!(
                "Unsupported dtype: {}. Supported: float64, float32, float16, int8, int16, int32, int64, uint8, uint16, uint32, uint64",
                queries_dtype
            ))),
        }
    }

}

// ========================================================================
// Type-specialized implementations: Zero-copy computation paths for each data type
// These are private helper methods, not exposed to Python
// ========================================================================

impl PyVectorSearch {
    /// f64 single vector computation (double precision floating point)
    fn compute_metric_f64(
        &self,
        _py: Python,
        a: &Bound<'_, PyAny>,
        b: &Bound<'_, PyAny>,
        metric_type: MetricType,
    ) -> PyResult<f64> {
        use numpy::PyArrayMethods;

        let a_arr: PyReadonlyArrayDyn<f64> = a.extract()?;
        let b_arr: PyReadonlyArrayDyn<f64> = b.extract()?;

        // Check dimensions
        let readonly_a = a_arr.readonly();
        let a_array = readonly_a.as_array();
        let readonly_b = b_arr.readonly();
        let b_array = readonly_b.as_array();
        let a_shape = a_array.shape();
        let b_shape = b_array.shape();

        if a_shape.len() != 1 || b_shape.len() != 1 {
            return Err(PyTypeError::new_err("Both arrays must be 1D vectors"));
        }

        if a_shape[0] != b_shape[0] {
            return Err(PyValueError::new_err(format!(
                "Vector dimensions don't match: {} vs {}",
                a_shape[0], b_shape[0]
            )));
        }

        let a_slice = a_arr.as_slice()?;
        let b_slice = b_arr.as_slice()?;

        self.engine
            .compute_metric(a_slice, b_slice, metric_type)
            .map_err(|e| PyValueError::new_err(format!("Compute error: {}", e)))
    }

    /// f32 single vector computation (single precision floating point)
    fn compute_metric_f32(
        &self,
        _py: Python,
        a: &Bound<'_, PyAny>,
        b: &Bound<'_, PyAny>,
        metric_type: MetricType,
    ) -> PyResult<f64> {
        use numpy::PyArrayMethods;

        let a_arr: PyReadonlyArrayDyn<f32> = a.extract()?;
        let b_arr: PyReadonlyArrayDyn<f32> = b.extract()?;

        // Check dimensions
        let readonly_a = a_arr.readonly();
        let a_array = readonly_a.as_array();
        let readonly_b = b_arr.readonly();
        let b_array = readonly_b.as_array();
        let a_shape = a_array.shape();
        let b_shape = b_array.shape();

        if a_shape.len() != 1 || b_shape.len() != 1 {
            return Err(PyTypeError::new_err("Both arrays must be 1D vectors"));
        }

        if a_shape[0] != b_shape[0] {
            return Err(PyValueError::new_err(format!(
                "Vector dimensions don't match: {} vs {}",
                a_shape[0], b_shape[0]
            )));
        }

        let a_slice = a_arr.as_slice()?;
        let b_slice = b_arr.as_slice()?;

        let result = self
            .engine
            .compute_metric_f32(a_slice, b_slice, metric_type)
            .map_err(|e| PyValueError::new_err(format!("Compute error: {}", e)))?;

        Ok(result as f64)
    }

    /// i8 single vector computation (integer vectors)
    fn compute_metric_i8(
        &self,
        _py: Python,
        a: &Bound<'_, PyAny>,
        b: &Bound<'_, PyAny>,
        metric_type: MetricType,
    ) -> PyResult<f64> {
        use numpy::PyArrayMethods;

        let a_arr: PyReadonlyArrayDyn<i8> = a.extract()?;
        let b_arr: PyReadonlyArrayDyn<i8> = b.extract()?;

        // Check dimensions
        let readonly_a = a_arr.readonly();
        let a_array = readonly_a.as_array();
        let readonly_b = b_arr.readonly();
        let b_array = readonly_b.as_array();
        let a_shape = a_array.shape();
        let b_shape = b_array.shape();

        if a_shape.len() != 1 || b_shape.len() != 1 {
            return Err(PyTypeError::new_err("Both arrays must be 1D vectors"));
        }

        if a_shape[0] != b_shape[0] {
            return Err(PyValueError::new_err(format!(
                "Vector dimensions don't match: {} vs {}",
                a_shape[0], b_shape[0]
            )));
        }

        let a_slice = a_arr.as_slice()?;
        let b_slice = b_arr.as_slice()?;

        let result = self
            .engine
            .cpu_backend
            .compute_i8(a_slice, b_slice, metric_type)
            .map_err(|e| PyValueError::new_err(format!("Compute error: {}", e)))?;

        Ok(result)
    }

    /// u8 single vector computation (binary vectors - hamming/jaccard)
    fn compute_metric_u8(
        &self,
        _py: Python,
        a: &Bound<'_, PyAny>,
        b: &Bound<'_, PyAny>,
        metric_type: MetricType,
    ) -> PyResult<f64> {
        use numpy::PyArrayMethods;

        // u8 only supports Hamming and Jaccard
        if !matches!(metric_type, MetricType::Hamming | MetricType::Jaccard) {
            return Err(PyValueError::new_err(format!(
                "uint8 arrays only support 'hamming' and 'jaccard' metrics, got: {}",
                metric_type.as_str()
            )));
        }

        let a_arr: PyReadonlyArrayDyn<u8> = a.extract()?;
        let b_arr: PyReadonlyArrayDyn<u8> = b.extract()?;

        // Check dimensions
        let readonly_a = a_arr.readonly();
        let a_array = readonly_a.as_array();
        let readonly_b = b_arr.readonly();
        let b_array = readonly_b.as_array();
        let a_shape = a_array.shape();
        let b_shape = b_array.shape();

        if a_shape.len() != 1 || b_shape.len() != 1 {
            return Err(PyTypeError::new_err("Both arrays must be 1D vectors"));
        }

        if a_shape[0] != b_shape[0] {
            return Err(PyValueError::new_err(format!(
                "Vector dimensions don't match: {} vs {}",
                a_shape[0], b_shape[0]
            )));
        }

        let a_slice = a_arr.as_slice()?;
        let b_slice = b_arr.as_slice()?;

        let result = self
            .engine
            .cpu_backend
            .compute_u8(a_slice, b_slice, metric_type)
            .map_err(|e| PyValueError::new_err(format!("Compute error: {}", e)))?;

        Ok(result)
    }

    /// f64 batch computation (double precision floating point)
    ///
    /// Optimization: Reduces FFI overhead by directly passing contiguous memory
    fn batch_compute_f64(
        &self,
        py: Python,
        query: &Bound<'_, PyAny>,
        candidates: &Bound<'_, PyAny>,
        metric_type: MetricType,
    ) -> PyResult<Py<PyArray1<f64>>> {
        use numpy::PyArrayMethods;

        let query_arr: PyReadonlyArrayDyn<f64> = query.extract()?;
        let candidates_arr: PyReadonlyArrayDyn<f64> = candidates.extract()?;

        let query_slice = query_arr.as_slice()?;
        let readonly_candidates = candidates_arr.readonly();
        let candidates_array = readonly_candidates.as_array();
        let shape = candidates_array.shape();

        if shape.len() != 2 {
            return Err(PyTypeError::new_err("Candidates must be a 2D array"));
        }

        let n_candidates = shape[0];
        let dim = shape[1];

        if query_slice.len() != dim {
            return Err(PyValueError::new_err(format!(
                "Query dimension {} does not match candidates dimension {}",
                query_slice.len(),
                dim
            )));
        }

        let candidates_slice = candidates_arr.as_slice()?;

        // Key optimization: Use usize to pass addresses (can cross threads)
        let query_addr = query_slice.as_ptr() as usize;
        let candidates_addr = candidates_slice.as_ptr() as usize;

        // Release GIL for parallel computation
        let scores = py
            .allow_threads(|| {
                // Smart batching strategy: serial for small batches, parallel for large batches
                // Avoids Rayon thread pool overhead for small batches
                const PARALLEL_THRESHOLD: usize = 500;

                if n_candidates < PARALLEL_THRESHOLD {
                    // Serial: avoid thread pool overhead
                    let mut scores = Vec::with_capacity(n_candidates);
                    for i in 0..n_candidates {
                        unsafe {
                            let query = std::slice::from_raw_parts(query_addr as *const f64, dim);
                            let candidate = std::slice::from_raw_parts(
                                (candidates_addr + i * dim * std::mem::size_of::<f64>())
                                    as *const f64,
                                dim,
                            );
                            scores.push(self.engine.cpu_backend.compute_f64(
                                query,
                                candidate,
                                metric_type,
                            )?);
                        }
                    }
                    Ok(scores)
                } else {
                    // Parallel: use multiple cores for large batches
                    #[cfg(feature = "rayon")]
                    {
                        use rayon::prelude::*;

                        (0..n_candidates)
                            .into_par_iter()
                            .map(|i| unsafe {
                                let query =
                                    std::slice::from_raw_parts(query_addr as *const f64, dim);
                                let candidate = std::slice::from_raw_parts(
                                    (candidates_addr + i * dim * std::mem::size_of::<f64>())
                                        as *const f64,
                                    dim,
                                );
                                self.engine
                                    .cpu_backend
                                    .compute_f64(query, candidate, metric_type)
                            })
                            .collect::<Result<Vec<_>, _>>()
                    }

                    #[cfg(not(feature = "rayon"))]
                    {
                        let mut scores = Vec::with_capacity(n_candidates);
                        for i in 0..n_candidates {
                            unsafe {
                                let query =
                                    std::slice::from_raw_parts(query_addr as *const f64, dim);
                                let candidate = std::slice::from_raw_parts(
                                    (candidates_addr + i * dim * std::mem::size_of::<f64>())
                                        as *const f64,
                                    dim,
                                );
                                scores.push(self.engine.cpu_backend.compute_f64(
                                    query,
                                    candidate,
                                    metric_type,
                                )?);
                            }
                        }
                        Ok(scores)
                    }
                }
            })
            .map_err(|e| PyValueError::new_err(format!("Compute error: {}", e)))?;

        Ok(PyArray1::from_vec(py, scores).into())
    }

    /// f32 batch computation (single precision floating point)
    fn batch_compute_f32(
        &self,
        py: Python,
        query: &Bound<'_, PyAny>,
        candidates: &Bound<'_, PyAny>,
        metric_type: MetricType,
    ) -> PyResult<Py<PyArray1<f64>>> {
        use numpy::PyArrayMethods;

        let query_arr: PyReadonlyArrayDyn<f32> = query.extract()?;
        let candidates_arr: PyReadonlyArrayDyn<f32> = candidates.extract()?;

        let query_slice = query_arr.as_slice()?;
        let readonly_candidates = candidates_arr.readonly();
        let candidates_array = readonly_candidates.as_array();
        let shape = candidates_array.shape();

        if shape.len() != 2 {
            return Err(PyTypeError::new_err("Candidates must be a 2D array"));
        }

        let n_candidates = shape[0];
        let dim = shape[1];

        if query_slice.len() != dim {
            return Err(PyValueError::new_err(format!(
                "Query dimension {} does not match candidates dimension {}",
                query_slice.len(),
                dim
            )));
        }

        let candidates_slice = candidates_arr.as_slice()?;

        // Optimization: use usize to pass addresses
        let query_addr = query_slice.as_ptr() as usize;
        let candidates_addr = candidates_slice.as_ptr() as usize;

        let scores = py
            .allow_threads(|| {
                #[cfg(feature = "rayon")]
                {
                    use rayon::prelude::*;

                    (0..n_candidates)
                        .into_par_iter()
                        .map(|i| unsafe {
                            let query = std::slice::from_raw_parts(query_addr as *const f32, dim);
                            let candidate = std::slice::from_raw_parts(
                                (candidates_addr + i * dim * std::mem::size_of::<f32>())
                                    as *const f32,
                                dim,
                            );
                            self.engine
                                .cpu_backend
                                .compute_f32(query, candidate, metric_type)
                        })
                        .collect::<Result<Vec<_>, _>>()
                }

                #[cfg(not(feature = "rayon"))]
                {
                    (0..n_candidates)
                        .map(|i| unsafe {
                            let query = std::slice::from_raw_parts(query_addr as *const f32, dim);
                            let candidate = std::slice::from_raw_parts(
                                (candidates_addr + i * dim * std::mem::size_of::<f32>())
                                    as *const f32,
                                dim,
                            );
                            self.engine
                                .cpu_backend
                                .compute_f32(query, candidate, metric_type)
                        })
                        .collect::<Result<Vec<_>, _>>()
                }
            })
            .map_err(|e| PyValueError::new_err(format!("Compute error: {}", e)))?;

        // Convert f32 results to f64 (unified output type)
        let scores_f64: Vec<f64> = scores.into_iter().map(|x| x as f64).collect();
        Ok(PyArray1::from_vec(py, scores_f64).into())
    }

    /// f16 batch computation (half precision floating point)
    fn batch_compute_f16(
        &self,
        py: Python,
        _query: &Bound<'_, PyAny>,
        _candidates: &Bound<'_, PyAny>,
        _metric_type: MetricType,
    ) -> PyResult<Py<PyArray1<f64>>> {
        // TODO: Implement f16 support (requires half crate integration)
        Err(PyTypeError::new_err(
            "float16 support not yet implemented. Please use float32 or float64.",
        ))
    }

    /// i8 batch computation (integer vectors)
    fn batch_compute_i8(
        &self,
        py: Python,
        query: &Bound<'_, PyAny>,
        candidates: &Bound<'_, PyAny>,
        metric_type: MetricType,
    ) -> PyResult<Py<PyArray1<f64>>> {
        use numpy::PyArrayMethods;

        let query_arr: PyReadonlyArrayDyn<i8> = query.extract()?;
        let candidates_arr: PyReadonlyArrayDyn<i8> = candidates.extract()?;

        let query_slice = query_arr.as_slice()?;
        let readonly_candidates = candidates_arr.readonly();
        let candidates_array = readonly_candidates.as_array();
        let shape = candidates_array.shape();

        if shape.len() != 2 {
            return Err(PyTypeError::new_err("Candidates must be a 2D array"));
        }

        let n_candidates = shape[0];
        let dim = shape[1];

        if query_slice.len() != dim {
            return Err(PyValueError::new_err(format!(
                "Query dimension {} does not match candidates dimension {}",
                query_slice.len(),
                dim
            )));
        }

        let candidates_slice = candidates_arr.as_slice()?;

        // Optimization: use usize to pass addresses
        let query_addr = query_slice.as_ptr() as usize;
        let candidates_addr = candidates_slice.as_ptr() as usize;

        let scores = py
            .allow_threads(|| {
                #[cfg(feature = "rayon")]
                {
                    use rayon::prelude::*;

                    (0..n_candidates)
                        .into_par_iter()
                        .map(|i| unsafe {
                            let query = std::slice::from_raw_parts(query_addr as *const i8, dim);
                            let candidate = std::slice::from_raw_parts(
                                (candidates_addr + i * dim * std::mem::size_of::<i8>())
                                    as *const i8,
                                dim,
                            );
                            self.engine
                                .cpu_backend
                                .compute_i8(query, candidate, metric_type)
                        })
                        .collect::<Result<Vec<_>, _>>()
                }

                #[cfg(not(feature = "rayon"))]
                {
                    (0..n_candidates)
                        .map(|i| unsafe {
                            let query = std::slice::from_raw_parts(query_addr as *const i8, dim);
                            let candidate = std::slice::from_raw_parts(
                                (candidates_addr + i * dim * std::mem::size_of::<i8>())
                                    as *const i8,
                                dim,
                            );
                            self.engine
                                .cpu_backend
                                .compute_i8(query, candidate, metric_type)
                        })
                        .collect::<Result<Vec<_>, _>>()
                }
            })
            .map_err(|e| PyValueError::new_err(format!("Compute error: {}", e)))?;

        Ok(PyArray1::from_vec(py, scores).into())
    }

    /// u8 batch computation (binary vectors - hamming/jaccard)
    fn batch_compute_u8(
        &self,
        py: Python,
        query: &Bound<'_, PyAny>,
        candidates: &Bound<'_, PyAny>,
        metric_type: MetricType,
    ) -> PyResult<Py<PyArray1<f64>>> {
        use numpy::PyArrayMethods;

        // u8 only supports Hamming and Jaccard
        if !matches!(metric_type, MetricType::Hamming | MetricType::Jaccard) {
            return Err(PyValueError::new_err(format!(
                "uint8 arrays only support 'hamming' and 'jaccard' metrics, got: {}",
                metric_type.as_str()
            )));
        }

        let query_arr: PyReadonlyArrayDyn<u8> = query.extract()?;
        let candidates_arr: PyReadonlyArrayDyn<u8> = candidates.extract()?;

        let query_slice = query_arr.as_slice()?;
        let readonly_candidates = candidates_arr.readonly();
        let candidates_array = readonly_candidates.as_array();
        let shape = candidates_array.shape();

        if shape.len() != 2 {
            return Err(PyTypeError::new_err("Candidates must be a 2D array"));
        }

        let n_candidates = shape[0];
        let dim = shape[1];

        if query_slice.len() != dim {
            return Err(PyValueError::new_err(format!(
                "Query dimension {} does not match candidates dimension {}",
                query_slice.len(),
                dim
            )));
        }

        let candidates_slice = candidates_arr.as_slice()?;

        // Optimization: use usize to pass addresses
        let query_addr = query_slice.as_ptr() as usize;
        let candidates_addr = candidates_slice.as_ptr() as usize;

        let scores = py
            .allow_threads(|| {
                #[cfg(feature = "rayon")]
                {
                    use rayon::prelude::*;

                    (0..n_candidates)
                        .into_par_iter()
                        .map(|i| unsafe {
                            let query = std::slice::from_raw_parts(query_addr as *const u8, dim);
                            let candidate = std::slice::from_raw_parts(
                                (candidates_addr + i * dim * std::mem::size_of::<u8>())
                                    as *const u8,
                                dim,
                            );
                            self.engine
                                .cpu_backend
                                .compute_u8(query, candidate, metric_type)
                        })
                        .collect::<Result<Vec<_>, _>>()
                }

                #[cfg(not(feature = "rayon"))]
                {
                    (0..n_candidates)
                        .map(|i| unsafe {
                            let query = std::slice::from_raw_parts(query_addr as *const u8, dim);
                            let candidate = std::slice::from_raw_parts(
                                (candidates_addr + i * dim * std::mem::size_of::<u8>())
                                    as *const u8,
                                dim,
                            );
                            self.engine
                                .cpu_backend
                                .compute_u8(query, candidate, metric_type)
                        })
                        .collect::<Result<Vec<_>, _>>()
                }
            })
            .map_err(|e| PyValueError::new_err(format!("Compute error: {}", e)))?;

        Ok(PyArray1::from_vec(py, scores).into())
    }

    // ========================================================================
    // New type implementations: compute_metric and batch_compute for additional types
    // ========================================================================

    /// f16 single vector computation (half precision floating point)
    fn compute_metric_f16(
        &self,
        _py: Python,
        a: &Bound<'_, PyAny>,
        b: &Bound<'_, PyAny>,
        metric_type: MetricType,
    ) -> PyResult<f64> {
        use numpy::PyArrayMethods;

        let a_arr: PyReadonlyArrayDyn<half::f16> = a.extract()?;
        let b_arr: PyReadonlyArrayDyn<half::f16> = b.extract()?;

        let readonly_a = a_arr.readonly();
        let a_array = readonly_a.as_array();
        let readonly_b = b_arr.readonly();
        let b_array = readonly_b.as_array();
        let a_shape = a_array.shape();
        let b_shape = b_array.shape();

        if a_shape.len() != 1 || b_shape.len() != 1 {
            return Err(PyTypeError::new_err("Both arrays must be 1D vectors"));
        }

        if a_shape[0] != b_shape[0] {
            return Err(PyValueError::new_err(format!(
                "Vector dimensions don't match: {} vs {}",
                a_shape[0], b_shape[0]
            )));
        }

        let a_slice = a_arr.as_slice()?;
        let b_slice = b_arr.as_slice()?;

        self.engine
            .cpu_backend
            .compute_f16(a_slice, b_slice, metric_type)
            .map_err(|e| PyValueError::new_err(format!("Compute error: {}", e)))
    }

    /// i16 single vector computation
    fn compute_metric_i16(
        &self,
        _py: Python,
        a: &Bound<'_, PyAny>,
        b: &Bound<'_, PyAny>,
        metric_type: MetricType,
    ) -> PyResult<f64> {
        use numpy::PyArrayMethods;

        let a_arr: PyReadonlyArrayDyn<i16> = a.extract()?;
        let b_arr: PyReadonlyArrayDyn<i16> = b.extract()?;

        let readonly_a = a_arr.readonly();
        let a_array = readonly_a.as_array();
        let readonly_b = b_arr.readonly();
        let b_array = readonly_b.as_array();
        let a_shape = a_array.shape();
        let b_shape = b_array.shape();

        if a_shape.len() != 1 || b_shape.len() != 1 {
            return Err(PyTypeError::new_err("Both arrays must be 1D vectors"));
        }

        if a_shape[0] != b_shape[0] {
            return Err(PyValueError::new_err(format!(
                "Vector dimensions don't match: {} vs {}",
                a_shape[0], b_shape[0]
            )));
        }

        let a_slice = a_arr.as_slice()?;
        let b_slice = b_arr.as_slice()?;

        self.engine
            .cpu_backend
            .compute_i16(a_slice, b_slice, metric_type)
            .map_err(|e| PyValueError::new_err(format!("Compute error: {}", e)))
    }

    /// i32 single vector computation
    fn compute_metric_i32(
        &self,
        _py: Python,
        a: &Bound<'_, PyAny>,
        b: &Bound<'_, PyAny>,
        metric_type: MetricType,
    ) -> PyResult<f64> {
        use numpy::PyArrayMethods;

        let a_arr: PyReadonlyArrayDyn<i32> = a.extract()?;
        let b_arr: PyReadonlyArrayDyn<i32> = b.extract()?;

        let readonly_a = a_arr.readonly();
        let a_array = readonly_a.as_array();
        let readonly_b = b_arr.readonly();
        let b_array = readonly_b.as_array();
        let a_shape = a_array.shape();
        let b_shape = b_array.shape();

        if a_shape.len() != 1 || b_shape.len() != 1 {
            return Err(PyTypeError::new_err("Both arrays must be 1D vectors"));
        }

        if a_shape[0] != b_shape[0] {
            return Err(PyValueError::new_err(format!(
                "Vector dimensions don't match: {} vs {}",
                a_shape[0], b_shape[0]
            )));
        }

        let a_slice = a_arr.as_slice()?;
        let b_slice = b_arr.as_slice()?;

        self.engine
            .cpu_backend
            .compute_i32(a_slice, b_slice, metric_type)
            .map_err(|e| PyValueError::new_err(format!("Compute error: {}", e)))
    }

    /// i64 single vector computation
    fn compute_metric_i64(
        &self,
        _py: Python,
        a: &Bound<'_, PyAny>,
        b: &Bound<'_, PyAny>,
        metric_type: MetricType,
    ) -> PyResult<f64> {
        use numpy::PyArrayMethods;

        let a_arr: PyReadonlyArrayDyn<i64> = a.extract()?;
        let b_arr: PyReadonlyArrayDyn<i64> = b.extract()?;

        let readonly_a = a_arr.readonly();
        let a_array = readonly_a.as_array();
        let readonly_b = b_arr.readonly();
        let b_array = readonly_b.as_array();
        let a_shape = a_array.shape();
        let b_shape = b_array.shape();

        if a_shape.len() != 1 || b_shape.len() != 1 {
            return Err(PyTypeError::new_err("Both arrays must be 1D vectors"));
        }

        if a_shape[0] != b_shape[0] {
            return Err(PyValueError::new_err(format!(
                "Vector dimensions don't match: {} vs {}",
                a_shape[0], b_shape[0]
            )));
        }

        let a_slice = a_arr.as_slice()?;
        let b_slice = b_arr.as_slice()?;

        self.engine
            .cpu_backend
            .compute_i64(a_slice, b_slice, metric_type)
            .map_err(|e| PyValueError::new_err(format!("Compute error: {}", e)))
    }

    /// u16 single vector computation
    fn compute_metric_u16(
        &self,
        _py: Python,
        a: &Bound<'_, PyAny>,
        b: &Bound<'_, PyAny>,
        metric_type: MetricType,
    ) -> PyResult<f64> {
        use numpy::PyArrayMethods;

        let a_arr: PyReadonlyArrayDyn<u16> = a.extract()?;
        let b_arr: PyReadonlyArrayDyn<u16> = b.extract()?;

        let readonly_a = a_arr.readonly();
        let a_array = readonly_a.as_array();
        let readonly_b = b_arr.readonly();
        let b_array = readonly_b.as_array();
        let a_shape = a_array.shape();
        let b_shape = b_array.shape();

        if a_shape.len() != 1 || b_shape.len() != 1 {
            return Err(PyTypeError::new_err("Both arrays must be 1D vectors"));
        }

        if a_shape[0] != b_shape[0] {
            return Err(PyValueError::new_err(format!(
                "Vector dimensions don't match: {} vs {}",
                a_shape[0], b_shape[0]
            )));
        }

        let a_slice = a_arr.as_slice()?;
        let b_slice = b_arr.as_slice()?;

        self.engine
            .cpu_backend
            .compute_u16(a_slice, b_slice, metric_type)
            .map_err(|e| PyValueError::new_err(format!("Compute error: {}", e)))
    }

    /// u32 single vector computation
    fn compute_metric_u32(
        &self,
        _py: Python,
        a: &Bound<'_, PyAny>,
        b: &Bound<'_, PyAny>,
        metric_type: MetricType,
    ) -> PyResult<f64> {
        use numpy::PyArrayMethods;

        let a_arr: PyReadonlyArrayDyn<u32> = a.extract()?;
        let b_arr: PyReadonlyArrayDyn<u32> = b.extract()?;

        let readonly_a = a_arr.readonly();
        let a_array = readonly_a.as_array();
        let readonly_b = b_arr.readonly();
        let b_array = readonly_b.as_array();
        let a_shape = a_array.shape();
        let b_shape = b_array.shape();

        if a_shape.len() != 1 || b_shape.len() != 1 {
            return Err(PyTypeError::new_err("Both arrays must be 1D vectors"));
        }

        if a_shape[0] != b_shape[0] {
            return Err(PyValueError::new_err(format!(
                "Vector dimensions don't match: {} vs {}",
                a_shape[0], b_shape[0]
            )));
        }

        let a_slice = a_arr.as_slice()?;
        let b_slice = b_arr.as_slice()?;

        self.engine
            .cpu_backend
            .compute_u32(a_slice, b_slice, metric_type)
            .map_err(|e| PyValueError::new_err(format!("Compute error: {}", e)))
    }

    /// u64 single vector computation
    fn compute_metric_u64(
        &self,
        _py: Python,
        a: &Bound<'_, PyAny>,
        b: &Bound<'_, PyAny>,
        metric_type: MetricType,
    ) -> PyResult<f64> {
        use numpy::PyArrayMethods;

        let a_arr: PyReadonlyArrayDyn<u64> = a.extract()?;
        let b_arr: PyReadonlyArrayDyn<u64> = b.extract()?;

        let readonly_a = a_arr.readonly();
        let a_array = readonly_a.as_array();
        let readonly_b = b_arr.readonly();
        let b_array = readonly_b.as_array();
        let a_shape = a_array.shape();
        let b_shape = b_array.shape();

        if a_shape.len() != 1 || b_shape.len() != 1 {
            return Err(PyTypeError::new_err("Both arrays must be 1D vectors"));
        }

        if a_shape[0] != b_shape[0] {
            return Err(PyValueError::new_err(format!(
                "Vector dimensions don't match: {} vs {}",
                a_shape[0], b_shape[0]
            )));
        }

        let a_slice = a_arr.as_slice()?;
        let b_slice = b_arr.as_slice()?;

        self.engine
            .cpu_backend
            .compute_u64(a_slice, b_slice, metric_type)
            .map_err(|e| PyValueError::new_err(format!("Compute error: {}", e)))
    }

    // ========================================================================
    // New batch_compute implementations for additional types
    // ========================================================================

    /// f16 batch computation (half precision floating point) - actual implementation
    fn batch_compute_f16_impl(
        &self,
        py: Python,
        query: &Bound<'_, PyAny>,
        candidates: &Bound<'_, PyAny>,
        metric_type: MetricType,
    ) -> PyResult<Py<PyArray1<f64>>> {
        use numpy::PyArrayMethods;

        let query_arr: PyReadonlyArrayDyn<half::f16> = query.extract()?;
        let candidates_arr: PyReadonlyArrayDyn<half::f16> = candidates.extract()?;

        let query_slice = query_arr.as_slice()?;
        let readonly_candidates = candidates_arr.readonly();
        let candidates_array = readonly_candidates.as_array();
        let shape = candidates_array.shape();

        if shape.len() != 2 {
            return Err(PyTypeError::new_err("Candidates must be a 2D array"));
        }

        let n_candidates = shape[0];
        let dim = shape[1];

        if query_slice.len() != dim {
            return Err(PyValueError::new_err(format!(
                "Query dimension {} does not match candidates dimension {}",
                query_slice.len(),
                dim
            )));
        }

        let candidates_slice = candidates_arr.as_slice()?;
        let query_addr = query_slice.as_ptr() as usize;
        let candidates_addr = candidates_slice.as_ptr() as usize;

        let scores = py
            .allow_threads(|| {
                #[cfg(feature = "rayon")]
                {
                    use rayon::prelude::*;
                    (0..n_candidates)
                        .into_par_iter()
                        .map(|i| unsafe {
                            let query = std::slice::from_raw_parts(query_addr as *const half::f16, dim);
                            let candidate = std::slice::from_raw_parts(
                                (candidates_addr + i * dim * std::mem::size_of::<half::f16>())
                                    as *const half::f16,
                                dim,
                            );
                            self.engine.cpu_backend.compute_f16(query, candidate, metric_type)
                        })
                        .collect::<Result<Vec<_>, _>>()
                }

                #[cfg(not(feature = "rayon"))]
                {
                    (0..n_candidates)
                        .map(|i| unsafe {
                            let query = std::slice::from_raw_parts(query_addr as *const half::f16, dim);
                            let candidate = std::slice::from_raw_parts(
                                (candidates_addr + i * dim * std::mem::size_of::<half::f16>())
                                    as *const half::f16,
                                dim,
                            );
                            self.engine.cpu_backend.compute_f16(query, candidate, metric_type)
                        })
                        .collect::<Result<Vec<_>, _>>()
                }
            })
            .map_err(|e| PyValueError::new_err(format!("Compute error: {}", e)))?;

        Ok(PyArray1::from_vec(py, scores).into())
    }

    /// i16 batch computation
    fn batch_compute_i16(
        &self,
        py: Python,
        query: &Bound<'_, PyAny>,
        candidates: &Bound<'_, PyAny>,
        metric_type: MetricType,
    ) -> PyResult<Py<PyArray1<f64>>> {
        use numpy::PyArrayMethods;

        let query_arr: PyReadonlyArrayDyn<i16> = query.extract()?;
        let candidates_arr: PyReadonlyArrayDyn<i16> = candidates.extract()?;

        let query_slice = query_arr.as_slice()?;
        let readonly_candidates = candidates_arr.readonly();
        let candidates_array = readonly_candidates.as_array();
        let shape = candidates_array.shape();

        if shape.len() != 2 {
            return Err(PyTypeError::new_err("Candidates must be a 2D array"));
        }

        let n_candidates = shape[0];
        let dim = shape[1];

        if query_slice.len() != dim {
            return Err(PyValueError::new_err(format!(
                "Query dimension {} does not match candidates dimension {}",
                query_slice.len(),
                dim
            )));
        }

        let candidates_slice = candidates_arr.as_slice()?;
        let query_addr = query_slice.as_ptr() as usize;
        let candidates_addr = candidates_slice.as_ptr() as usize;

        let scores = py
            .allow_threads(|| {
                #[cfg(feature = "rayon")]
                {
                    use rayon::prelude::*;
                    (0..n_candidates)
                        .into_par_iter()
                        .map(|i| unsafe {
                            let query = std::slice::from_raw_parts(query_addr as *const i16, dim);
                            let candidate = std::slice::from_raw_parts(
                                (candidates_addr + i * dim * std::mem::size_of::<i16>())
                                    as *const i16,
                                dim,
                            );
                            self.engine.cpu_backend.compute_i16(query, candidate, metric_type)
                        })
                        .collect::<Result<Vec<_>, _>>()
                }

                #[cfg(not(feature = "rayon"))]
                {
                    (0..n_candidates)
                        .map(|i| unsafe {
                            let query = std::slice::from_raw_parts(query_addr as *const i16, dim);
                            let candidate = std::slice::from_raw_parts(
                                (candidates_addr + i * dim * std::mem::size_of::<i16>())
                                    as *const i16,
                                dim,
                            );
                            self.engine.cpu_backend.compute_i16(query, candidate, metric_type)
                        })
                        .collect::<Result<Vec<_>, _>>()
                }
            })
            .map_err(|e| PyValueError::new_err(format!("Compute error: {}", e)))?;

        Ok(PyArray1::from_vec(py, scores).into())
    }

    /// i32 batch computation
    fn batch_compute_i32(
        &self,
        py: Python,
        query: &Bound<'_, PyAny>,
        candidates: &Bound<'_, PyAny>,
        metric_type: MetricType,
    ) -> PyResult<Py<PyArray1<f64>>> {
        use numpy::PyArrayMethods;

        let query_arr: PyReadonlyArrayDyn<i32> = query.extract()?;
        let candidates_arr: PyReadonlyArrayDyn<i32> = candidates.extract()?;

        let query_slice = query_arr.as_slice()?;
        let readonly_candidates = candidates_arr.readonly();
        let candidates_array = readonly_candidates.as_array();
        let shape = candidates_array.shape();

        if shape.len() != 2 {
            return Err(PyTypeError::new_err("Candidates must be a 2D array"));
        }

        let n_candidates = shape[0];
        let dim = shape[1];

        if query_slice.len() != dim {
            return Err(PyValueError::new_err(format!(
                "Query dimension {} does not match candidates dimension {}",
                query_slice.len(),
                dim
            )));
        }

        let candidates_slice = candidates_arr.as_slice()?;
        let query_addr = query_slice.as_ptr() as usize;
        let candidates_addr = candidates_slice.as_ptr() as usize;

        let scores = py
            .allow_threads(|| {
                #[cfg(feature = "rayon")]
                {
                    use rayon::prelude::*;
                    (0..n_candidates)
                        .into_par_iter()
                        .map(|i| unsafe {
                            let query = std::slice::from_raw_parts(query_addr as *const i32, dim);
                            let candidate = std::slice::from_raw_parts(
                                (candidates_addr + i * dim * std::mem::size_of::<i32>())
                                    as *const i32,
                                dim,
                            );
                            self.engine.cpu_backend.compute_i32(query, candidate, metric_type)
                        })
                        .collect::<Result<Vec<_>, _>>()
                }

                #[cfg(not(feature = "rayon"))]
                {
                    (0..n_candidates)
                        .map(|i| unsafe {
                            let query = std::slice::from_raw_parts(query_addr as *const i32, dim);
                            let candidate = std::slice::from_raw_parts(
                                (candidates_addr + i * dim * std::mem::size_of::<i32>())
                                    as *const i32,
                                dim,
                            );
                            self.engine.cpu_backend.compute_i32(query, candidate, metric_type)
                        })
                        .collect::<Result<Vec<_>, _>>()
                }
            })
            .map_err(|e| PyValueError::new_err(format!("Compute error: {}", e)))?;

        Ok(PyArray1::from_vec(py, scores).into())
    }

    /// i64 batch computation
    fn batch_compute_i64(
        &self,
        py: Python,
        query: &Bound<'_, PyAny>,
        candidates: &Bound<'_, PyAny>,
        metric_type: MetricType,
    ) -> PyResult<Py<PyArray1<f64>>> {
        use numpy::PyArrayMethods;

        let query_arr: PyReadonlyArrayDyn<i64> = query.extract()?;
        let candidates_arr: PyReadonlyArrayDyn<i64> = candidates.extract()?;

        let query_slice = query_arr.as_slice()?;
        let readonly_candidates = candidates_arr.readonly();
        let candidates_array = readonly_candidates.as_array();
        let shape = candidates_array.shape();

        if shape.len() != 2 {
            return Err(PyTypeError::new_err("Candidates must be a 2D array"));
        }

        let n_candidates = shape[0];
        let dim = shape[1];

        if query_slice.len() != dim {
            return Err(PyValueError::new_err(format!(
                "Query dimension {} does not match candidates dimension {}",
                query_slice.len(),
                dim
            )));
        }

        let candidates_slice = candidates_arr.as_slice()?;
        let query_addr = query_slice.as_ptr() as usize;
        let candidates_addr = candidates_slice.as_ptr() as usize;

        let scores = py
            .allow_threads(|| {
                #[cfg(feature = "rayon")]
                {
                    use rayon::prelude::*;
                    (0..n_candidates)
                        .into_par_iter()
                        .map(|i| unsafe {
                            let query = std::slice::from_raw_parts(query_addr as *const i64, dim);
                            let candidate = std::slice::from_raw_parts(
                                (candidates_addr + i * dim * std::mem::size_of::<i64>())
                                    as *const i64,
                                dim,
                            );
                            self.engine.cpu_backend.compute_i64(query, candidate, metric_type)
                        })
                        .collect::<Result<Vec<_>, _>>()
                }

                #[cfg(not(feature = "rayon"))]
                {
                    (0..n_candidates)
                        .map(|i| unsafe {
                            let query = std::slice::from_raw_parts(query_addr as *const i64, dim);
                            let candidate = std::slice::from_raw_parts(
                                (candidates_addr + i * dim * std::mem::size_of::<i64>())
                                    as *const i64,
                                dim,
                            );
                            self.engine.cpu_backend.compute_i64(query, candidate, metric_type)
                        })
                        .collect::<Result<Vec<_>, _>>()
                }
            })
            .map_err(|e| PyValueError::new_err(format!("Compute error: {}", e)))?;

        Ok(PyArray1::from_vec(py, scores).into())
    }

    /// u16 batch computation
    fn batch_compute_u16(
        &self,
        py: Python,
        query: &Bound<'_, PyAny>,
        candidates: &Bound<'_, PyAny>,
        metric_type: MetricType,
    ) -> PyResult<Py<PyArray1<f64>>> {
        use numpy::PyArrayMethods;

        let query_arr: PyReadonlyArrayDyn<u16> = query.extract()?;
        let candidates_arr: PyReadonlyArrayDyn<u16> = candidates.extract()?;

        let query_slice = query_arr.as_slice()?;
        let readonly_candidates = candidates_arr.readonly();
        let candidates_array = readonly_candidates.as_array();
        let shape = candidates_array.shape();

        if shape.len() != 2 {
            return Err(PyTypeError::new_err("Candidates must be a 2D array"));
        }

        let n_candidates = shape[0];
        let dim = shape[1];

        if query_slice.len() != dim {
            return Err(PyValueError::new_err(format!(
                "Query dimension {} does not match candidates dimension {}",
                query_slice.len(),
                dim
            )));
        }

        let candidates_slice = candidates_arr.as_slice()?;
        let query_addr = query_slice.as_ptr() as usize;
        let candidates_addr = candidates_slice.as_ptr() as usize;

        let scores = py
            .allow_threads(|| {
                #[cfg(feature = "rayon")]
                {
                    use rayon::prelude::*;
                    (0..n_candidates)
                        .into_par_iter()
                        .map(|i| unsafe {
                            let query = std::slice::from_raw_parts(query_addr as *const u16, dim);
                            let candidate = std::slice::from_raw_parts(
                                (candidates_addr + i * dim * std::mem::size_of::<u16>())
                                    as *const u16,
                                dim,
                            );
                            self.engine.cpu_backend.compute_u16(query, candidate, metric_type)
                        })
                        .collect::<Result<Vec<_>, _>>()
                }

                #[cfg(not(feature = "rayon"))]
                {
                    (0..n_candidates)
                        .map(|i| unsafe {
                            let query = std::slice::from_raw_parts(query_addr as *const u16, dim);
                            let candidate = std::slice::from_raw_parts(
                                (candidates_addr + i * dim * std::mem::size_of::<u16>())
                                    as *const u16,
                                dim,
                            );
                            self.engine.cpu_backend.compute_u16(query, candidate, metric_type)
                        })
                        .collect::<Result<Vec<_>, _>>()
                }
            })
            .map_err(|e| PyValueError::new_err(format!("Compute error: {}", e)))?;

        Ok(PyArray1::from_vec(py, scores).into())
    }

    /// u32 batch computation
    fn batch_compute_u32(
        &self,
        py: Python,
        query: &Bound<'_, PyAny>,
        candidates: &Bound<'_, PyAny>,
        metric_type: MetricType,
    ) -> PyResult<Py<PyArray1<f64>>> {
        use numpy::PyArrayMethods;

        let query_arr: PyReadonlyArrayDyn<u32> = query.extract()?;
        let candidates_arr: PyReadonlyArrayDyn<u32> = candidates.extract()?;

        let query_slice = query_arr.as_slice()?;
        let readonly_candidates = candidates_arr.readonly();
        let candidates_array = readonly_candidates.as_array();
        let shape = candidates_array.shape();

        if shape.len() != 2 {
            return Err(PyTypeError::new_err("Candidates must be a 2D array"));
        }

        let n_candidates = shape[0];
        let dim = shape[1];

        if query_slice.len() != dim {
            return Err(PyValueError::new_err(format!(
                "Query dimension {} does not match candidates dimension {}",
                query_slice.len(),
                dim
            )));
        }

        let candidates_slice = candidates_arr.as_slice()?;
        let query_addr = query_slice.as_ptr() as usize;
        let candidates_addr = candidates_slice.as_ptr() as usize;

        let scores = py
            .allow_threads(|| {
                #[cfg(feature = "rayon")]
                {
                    use rayon::prelude::*;
                    (0..n_candidates)
                        .into_par_iter()
                        .map(|i| unsafe {
                            let query = std::slice::from_raw_parts(query_addr as *const u32, dim);
                            let candidate = std::slice::from_raw_parts(
                                (candidates_addr + i * dim * std::mem::size_of::<u32>())
                                    as *const u32,
                                dim,
                            );
                            self.engine.cpu_backend.compute_u32(query, candidate, metric_type)
                        })
                        .collect::<Result<Vec<_>, _>>()
                }

                #[cfg(not(feature = "rayon"))]
                {
                    (0..n_candidates)
                        .map(|i| unsafe {
                            let query = std::slice::from_raw_parts(query_addr as *const u32, dim);
                            let candidate = std::slice::from_raw_parts(
                                (candidates_addr + i * dim * std::mem::size_of::<u32>())
                                    as *const u32,
                                dim,
                            );
                            self.engine.cpu_backend.compute_u32(query, candidate, metric_type)
                        })
                        .collect::<Result<Vec<_>, _>>()
                }
            })
            .map_err(|e| PyValueError::new_err(format!("Compute error: {}", e)))?;

        Ok(PyArray1::from_vec(py, scores).into())
    }

    /// u64 batch computation
    fn batch_compute_u64(
        &self,
        py: Python,
        query: &Bound<'_, PyAny>,
        candidates: &Bound<'_, PyAny>,
        metric_type: MetricType,
    ) -> PyResult<Py<PyArray1<f64>>> {
        use numpy::PyArrayMethods;

        let query_arr: PyReadonlyArrayDyn<u64> = query.extract()?;
        let candidates_arr: PyReadonlyArrayDyn<u64> = candidates.extract()?;

        let query_slice = query_arr.as_slice()?;
        let readonly_candidates = candidates_arr.readonly();
        let candidates_array = readonly_candidates.as_array();
        let shape = candidates_array.shape();

        if shape.len() != 2 {
            return Err(PyTypeError::new_err("Candidates must be a 2D array"));
        }

        let n_candidates = shape[0];
        let dim = shape[1];

        if query_slice.len() != dim {
            return Err(PyValueError::new_err(format!(
                "Query dimension {} does not match candidates dimension {}",
                query_slice.len(),
                dim
            )));
        }

        let candidates_slice = candidates_arr.as_slice()?;
        let query_addr = query_slice.as_ptr() as usize;
        let candidates_addr = candidates_slice.as_ptr() as usize;

        let scores = py
            .allow_threads(|| {
                #[cfg(feature = "rayon")]
                {
                    use rayon::prelude::*;
                    (0..n_candidates)
                        .into_par_iter()
                        .map(|i| unsafe {
                            let query = std::slice::from_raw_parts(query_addr as *const u64, dim);
                            let candidate = std::slice::from_raw_parts(
                                (candidates_addr + i * dim * std::mem::size_of::<u64>())
                                    as *const u64,
                                dim,
                            );
                            self.engine.cpu_backend.compute_u64(query, candidate, metric_type)
                        })
                        .collect::<Result<Vec<_>, _>>()
                }

                #[cfg(not(feature = "rayon"))]
                {
                    (0..n_candidates)
                        .map(|i| unsafe {
                            let query = std::slice::from_raw_parts(query_addr as *const u64, dim);
                            let candidate = std::slice::from_raw_parts(
                                (candidates_addr + i * dim * std::mem::size_of::<u64>())
                                    as *const u64,
                                dim,
                            );
                            self.engine.cpu_backend.compute_u64(query, candidate, metric_type)
                        })
                        .collect::<Result<Vec<_>, _>>()
                }
            })
            .map_err(|e| PyValueError::new_err(format!("Compute error: {}", e)))?;

        Ok(PyArray1::from_vec(py, scores).into())
    }

    // ========================================================================
    // Top-K search implementations: Optimized Top-K search for each data type
    // ========================================================================

    /// Top-K search (f64)
    fn top_k_search_f64(
        &self,
        py: Python,
        query: &Bound<'_, PyAny>,
        candidates: &Bound<'_, PyAny>,
        metric_type: MetricType,
        k: usize,
    ) -> PyResult<(Py<PyArray1<usize>>, Py<PyArray1<f64>>)> {
        // First compute all scores
        let scores_array = self.batch_compute_f64(py, query, candidates, metric_type)?;

        // Extract scores
        let scores = scores_array.bind(py).readonly();
        let scores_slice = scores.as_slice()?;

        // Top-K selection
        let (indices, top_scores) =
            Self::select_top_k(scores_slice, k, metric_type.is_similarity());

        Ok((
            PyArray1::from_vec(py, indices).into(),
            PyArray1::from_vec(py, top_scores).into(),
        ))
    }

    /// Top-K search (f32)
    fn top_k_search_f32(
        &self,
        py: Python,
        query: &Bound<'_, PyAny>,
        candidates: &Bound<'_, PyAny>,
        metric_type: MetricType,
        k: usize,
    ) -> PyResult<(Py<PyArray1<usize>>, Py<PyArray1<f64>>)> {
        let scores_array = self.batch_compute_f32(py, query, candidates, metric_type)?;
        let scores = scores_array.bind(py).readonly();
        let scores_slice = scores.as_slice()?;
        let (indices, top_scores) =
            Self::select_top_k(scores_slice, k, metric_type.is_similarity());

        Ok((
            PyArray1::from_vec(py, indices).into(),
            PyArray1::from_vec(py, top_scores).into(),
        ))
    }

    /// Top-K search (i8)
    fn top_k_search_i8(
        &self,
        py: Python,
        query: &Bound<'_, PyAny>,
        candidates: &Bound<'_, PyAny>,
        metric_type: MetricType,
        k: usize,
    ) -> PyResult<(Py<PyArray1<usize>>, Py<PyArray1<f64>>)> {
        let scores_array = self.batch_compute_i8(py, query, candidates, metric_type)?;
        let scores = scores_array.bind(py).readonly();
        let scores_slice = scores.as_slice()?;
        let (indices, top_scores) =
            Self::select_top_k(scores_slice, k, metric_type.is_similarity());

        Ok((
            PyArray1::from_vec(py, indices).into(),
            PyArray1::from_vec(py, top_scores).into(),
        ))
    }

    /// Top-K search (u8)
    fn top_k_search_u8(
        &self,
        py: Python,
        query: &Bound<'_, PyAny>,
        candidates: &Bound<'_, PyAny>,
        metric_type: MetricType,
        k: usize,
    ) -> PyResult<(Py<PyArray1<usize>>, Py<PyArray1<f64>>)> {
        let scores_array = self.batch_compute_u8(py, query, candidates, metric_type)?;
        let scores = scores_array.bind(py).readonly();
        let scores_slice = scores.as_slice()?;
        // u8 metrics are all distances (lower is better)
        let (indices, top_scores) = Self::select_top_k(scores_slice, k, false);

        Ok((
            PyArray1::from_vec(py, indices).into(),
            PyArray1::from_vec(py, top_scores).into(),
        ))
    }

    /// Top-K search (f16)
    fn top_k_search_f16(
        &self,
        py: Python,
        query: &Bound<'_, PyAny>,
        candidates: &Bound<'_, PyAny>,
        metric_type: MetricType,
        k: usize,
    ) -> PyResult<(Py<PyArray1<usize>>, Py<PyArray1<f64>>)> {
        let scores_array = self.batch_compute_f16_impl(py, query, candidates, metric_type)?;
        let scores = scores_array.bind(py).readonly();
        let scores_slice = scores.as_slice()?;
        let (indices, top_scores) =
            Self::select_top_k(scores_slice, k, metric_type.is_similarity());
        Ok((
            PyArray1::from_vec(py, indices).into(),
            PyArray1::from_vec(py, top_scores).into(),
        ))
    }

    /// Top-K search (i16)
    fn top_k_search_i16(
        &self,
        py: Python,
        query: &Bound<'_, PyAny>,
        candidates: &Bound<'_, PyAny>,
        metric_type: MetricType,
        k: usize,
    ) -> PyResult<(Py<PyArray1<usize>>, Py<PyArray1<f64>>)> {
        let scores_array = self.batch_compute_i16(py, query, candidates, metric_type)?;
        let scores = scores_array.bind(py).readonly();
        let scores_slice = scores.as_slice()?;
        let (indices, top_scores) =
            Self::select_top_k(scores_slice, k, metric_type.is_similarity());
        Ok((
            PyArray1::from_vec(py, indices).into(),
            PyArray1::from_vec(py, top_scores).into(),
        ))
    }

    /// Top-K search (i32)
    fn top_k_search_i32(
        &self,
        py: Python,
        query: &Bound<'_, PyAny>,
        candidates: &Bound<'_, PyAny>,
        metric_type: MetricType,
        k: usize,
    ) -> PyResult<(Py<PyArray1<usize>>, Py<PyArray1<f64>>)> {
        let scores_array = self.batch_compute_i32(py, query, candidates, metric_type)?;
        let scores = scores_array.bind(py).readonly();
        let scores_slice = scores.as_slice()?;
        let (indices, top_scores) =
            Self::select_top_k(scores_slice, k, metric_type.is_similarity());
        Ok((
            PyArray1::from_vec(py, indices).into(),
            PyArray1::from_vec(py, top_scores).into(),
        ))
    }

    /// Top-K search (i64)
    fn top_k_search_i64(
        &self,
        py: Python,
        query: &Bound<'_, PyAny>,
        candidates: &Bound<'_, PyAny>,
        metric_type: MetricType,
        k: usize,
    ) -> PyResult<(Py<PyArray1<usize>>, Py<PyArray1<f64>>)> {
        let scores_array = self.batch_compute_i64(py, query, candidates, metric_type)?;
        let scores = scores_array.bind(py).readonly();
        let scores_slice = scores.as_slice()?;
        let (indices, top_scores) =
            Self::select_top_k(scores_slice, k, metric_type.is_similarity());
        Ok((
            PyArray1::from_vec(py, indices).into(),
            PyArray1::from_vec(py, top_scores).into(),
        ))
    }

    /// Top-K search (u16)
    fn top_k_search_u16(
        &self,
        py: Python,
        query: &Bound<'_, PyAny>,
        candidates: &Bound<'_, PyAny>,
        metric_type: MetricType,
        k: usize,
    ) -> PyResult<(Py<PyArray1<usize>>, Py<PyArray1<f64>>)> {
        let scores_array = self.batch_compute_u16(py, query, candidates, metric_type)?;
        let scores = scores_array.bind(py).readonly();
        let scores_slice = scores.as_slice()?;
        let (indices, top_scores) =
            Self::select_top_k(scores_slice, k, metric_type.is_similarity());
        Ok((
            PyArray1::from_vec(py, indices).into(),
            PyArray1::from_vec(py, top_scores).into(),
        ))
    }

    /// Top-K search (u32)
    fn top_k_search_u32(
        &self,
        py: Python,
        query: &Bound<'_, PyAny>,
        candidates: &Bound<'_, PyAny>,
        metric_type: MetricType,
        k: usize,
    ) -> PyResult<(Py<PyArray1<usize>>, Py<PyArray1<f64>>)> {
        let scores_array = self.batch_compute_u32(py, query, candidates, metric_type)?;
        let scores = scores_array.bind(py).readonly();
        let scores_slice = scores.as_slice()?;
        let (indices, top_scores) =
            Self::select_top_k(scores_slice, k, metric_type.is_similarity());
        Ok((
            PyArray1::from_vec(py, indices).into(),
            PyArray1::from_vec(py, top_scores).into(),
        ))
    }

    /// Top-K search (u64)
    fn top_k_search_u64(
        &self,
        py: Python,
        query: &Bound<'_, PyAny>,
        candidates: &Bound<'_, PyAny>,
        metric_type: MetricType,
        k: usize,
    ) -> PyResult<(Py<PyArray1<usize>>, Py<PyArray1<f64>>)> {
        let scores_array = self.batch_compute_u64(py, query, candidates, metric_type)?;
        let scores = scores_array.bind(py).readonly();
        let scores_slice = scores.as_slice()?;
        let (indices, top_scores) =
            Self::select_top_k(scores_slice, k, metric_type.is_similarity());
        Ok((
            PyArray1::from_vec(py, indices).into(),
            PyArray1::from_vec(py, top_scores).into(),
        ))
    }

    /// Select Top-K from a scores array
    ///
    /// Args:
    ///     scores: Array of scores
    ///     k: Number of results to return
    ///     is_similarity: true = higher is better (similarity), false = lower is better (distance)
    ///
    /// Returns:
    ///     (indices, top_scores): Indices and corresponding scores
    fn select_top_k(scores: &[f64], k: usize, is_similarity: bool) -> (Vec<usize>, Vec<f64>) {
        let n = scores.len();
        let k = k.min(n); // k cannot exceed total count

        // Create (index, score) pairs
        let mut indexed_scores: Vec<(usize, f64)> = scores
            .iter()
            .enumerate()
            .map(|(i, &score)| (i, score))
            .collect();

        // Partial sort: only sort top k
        // Similarity: descending (high to low), Distance: ascending (low to high)
        if is_similarity {
            // Use select_nth_unstable for O(n) partial sort
            indexed_scores.select_nth_unstable_by(k - 1, |a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            // Sort top k again
            indexed_scores[..k].sort_unstable_by(|a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
        } else {
            // Distance: ascending
            indexed_scores.select_nth_unstable_by(k - 1, |a, b| {
                a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            indexed_scores[..k].sort_unstable_by(|a, b| {
                a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        // Extract indices and scores of top k
        let indices: Vec<usize> = indexed_scores[..k].iter().map(|(i, _)| *i).collect();
        let top_scores: Vec<f64> = indexed_scores[..k].iter().map(|(_, s)| *s).collect();

        (indices, top_scores)
    }

    /// Multi-query Top-K search (f32) - processes all queries in a single FFI call
    fn multi_query_top_k_f32(
        &self,
        py: Python,
        queries: &Bound<'_, PyAny>,
        candidates: &Bound<'_, PyAny>,
        metric_type: MetricType,
        k: usize,
    ) -> PyResult<(Py<PyArray1<u64>>, Py<PyArray1<f64>>)> {
        use numpy::PyArrayMethods;

        let queries_arr: PyReadonlyArrayDyn<f32> = queries.extract()?;
        let candidates_arr: PyReadonlyArrayDyn<f32> = candidates.extract()?;

        let readonly_queries = queries_arr.readonly();
        let queries_array = readonly_queries.as_array();
        let queries_shape = queries_array.shape();

        if queries_shape.len() != 2 {
            return Err(PyTypeError::new_err("Queries must be a 2D array"));
        }

        let n_queries = queries_shape[0];
        let dim = queries_shape[1];

        let readonly_candidates = candidates_arr.readonly();
        let candidates_array = readonly_candidates.as_array();
        let candidates_shape = candidates_array.shape();

        if candidates_shape.len() != 2 {
            return Err(PyTypeError::new_err("Candidates must be a 2D array"));
        }

        let n_candidates = candidates_shape[0];
        let candidates_dim = candidates_shape[1];

        if dim != candidates_dim {
            return Err(PyValueError::new_err(format!(
                "Query dimension {} does not match candidates dimension {}",
                dim, candidates_dim
            )));
        }

        let queries_slice = queries_arr.as_slice()?;
        let candidates_slice = candidates_arr.as_slice()?;

        // Pass addresses for thread safety
        let queries_addr = queries_slice.as_ptr() as usize;
        let candidates_addr = candidates_slice.as_ptr() as usize;
        let is_similarity = metric_type.is_similarity();

        // Release GIL and process all queries
        let (all_indices, all_scores) = py.allow_threads(|| {
            let mut result_indices: Vec<u64> = Vec::with_capacity(n_queries * k);
            let mut result_scores: Vec<f64> = Vec::with_capacity(n_queries * k);

            for q_idx in 0..n_queries {
                // Get query vector
                let query = unsafe {
                    std::slice::from_raw_parts(
                        (queries_addr + q_idx * dim * std::mem::size_of::<f32>()) as *const f32,
                        dim,
                    )
                };

                // Parallel compute scores for all candidates
                #[cfg(feature = "rayon")]
                let scores: Vec<f64> = {
                    use rayon::prelude::*;

                    (0..n_candidates)
                        .into_par_iter()
                        .map(|i| unsafe {
                            let candidate = std::slice::from_raw_parts(
                                (candidates_addr + i * dim * std::mem::size_of::<f32>()) as *const f32,
                                dim,
                            );
                            self.engine
                                .cpu_backend
                                .compute_f32(query, candidate, metric_type)
                                .unwrap_or(f32::NAN) as f64
                        })
                        .collect()
                };

                #[cfg(not(feature = "rayon"))]
                let scores: Vec<f64> = {
                    (0..n_candidates)
                        .map(|i| unsafe {
                            let candidate = std::slice::from_raw_parts(
                                (candidates_addr + i * dim * std::mem::size_of::<f32>()) as *const f32,
                                dim,
                            );
                            self.engine
                                .cpu_backend
                                .compute_f32(query, candidate, metric_type)
                                .unwrap_or(f32::NAN) as f64
                        })
                        .collect()
                };

                // Select top-k for this query
                let (indices, top_scores) = Self::select_top_k(&scores, k, is_similarity);

                // Append to results (convert usize to u64)
                for i in 0..k {
                    if i < indices.len() {
                        result_indices.push(indices[i] as u64);
                        result_scores.push(top_scores[i]);
                    } else {
                        result_indices.push(0);
                        result_scores.push(f64::NAN);
                    }
                }
            }

            (result_indices, result_scores)
        });

        Ok((
            PyArray1::from_vec(py, all_indices).into(),
            PyArray1::from_vec(py, all_scores).into(),
        ))
    }

    /// Multi-query Top-K search (f64) - processes all queries in a single FFI call
    fn multi_query_top_k_f64(
        &self,
        py: Python,
        queries: &Bound<'_, PyAny>,
        candidates: &Bound<'_, PyAny>,
        metric_type: MetricType,
        k: usize,
    ) -> PyResult<(Py<PyArray1<u64>>, Py<PyArray1<f64>>)> {
        use numpy::PyArrayMethods;

        let queries_arr: PyReadonlyArrayDyn<f64> = queries.extract()?;
        let candidates_arr: PyReadonlyArrayDyn<f64> = candidates.extract()?;

        let readonly_queries = queries_arr.readonly();
        let queries_array = readonly_queries.as_array();
        let queries_shape = queries_array.shape();

        if queries_shape.len() != 2 {
            return Err(PyTypeError::new_err("Queries must be a 2D array"));
        }

        let n_queries = queries_shape[0];
        let dim = queries_shape[1];

        let readonly_candidates = candidates_arr.readonly();
        let candidates_array = readonly_candidates.as_array();
        let candidates_shape = candidates_array.shape();

        if candidates_shape.len() != 2 {
            return Err(PyTypeError::new_err("Candidates must be a 2D array"));
        }

        let n_candidates = candidates_shape[0];
        let candidates_dim = candidates_shape[1];

        if dim != candidates_dim {
            return Err(PyValueError::new_err(format!(
                "Query dimension {} does not match candidates dimension {}",
                dim, candidates_dim
            )));
        }

        let queries_slice = queries_arr.as_slice()?;
        let candidates_slice = candidates_arr.as_slice()?;

        // Pass addresses for thread safety
        let queries_addr = queries_slice.as_ptr() as usize;
        let candidates_addr = candidates_slice.as_ptr() as usize;
        let is_similarity = metric_type.is_similarity();

        // Release GIL and process all queries
        let (all_indices, all_scores) = py.allow_threads(|| {
            let mut result_indices: Vec<u64> = Vec::with_capacity(n_queries * k);
            let mut result_scores: Vec<f64> = Vec::with_capacity(n_queries * k);

            for q_idx in 0..n_queries {
                // Get query vector
                let query = unsafe {
                    std::slice::from_raw_parts(
                        (queries_addr + q_idx * dim * std::mem::size_of::<f64>()) as *const f64,
                        dim,
                    )
                };

                // Parallel compute scores for all candidates
                #[cfg(feature = "rayon")]
                let scores: Vec<f64> = {
                    use rayon::prelude::*;

                    (0..n_candidates)
                        .into_par_iter()
                        .map(|i| unsafe {
                            let candidate = std::slice::from_raw_parts(
                                (candidates_addr + i * dim * std::mem::size_of::<f64>()) as *const f64,
                                dim,
                            );
                            self.engine
                                .cpu_backend
                                .compute_f64(query, candidate, metric_type)
                                .unwrap_or(f64::NAN)
                        })
                        .collect()
                };

                #[cfg(not(feature = "rayon"))]
                let scores: Vec<f64> = {
                    (0..n_candidates)
                        .map(|i| unsafe {
                            let candidate = std::slice::from_raw_parts(
                                (candidates_addr + i * dim * std::mem::size_of::<f64>()) as *const f64,
                                dim,
                            );
                            self.engine
                                .cpu_backend
                                .compute_f64(query, candidate, metric_type)
                                .unwrap_or(f64::NAN)
                        })
                        .collect()
                };

                // Select top-k for this query
                let (indices, top_scores) = Self::select_top_k(&scores, k, is_similarity);

                // Append to results
                for i in 0..k {
                    if i < indices.len() {
                        result_indices.push(indices[i] as u64);
                        result_scores.push(top_scores[i]);
                    } else {
                        result_indices.push(0);
                        result_scores.push(f64::NAN);
                    }
                }
            }

            (result_indices, result_scores)
        });

        Ok((
            PyArray1::from_vec(py, all_indices).into(),
            PyArray1::from_vec(py, all_scores).into(),
        ))
    }

    /// f32 batch compute and merge top-k (single FFI call optimization)
    fn segmented_top_k_search_f32(
        &self,
        py: Python,
        query: &Bound<'_, PyAny>,
        candidates: &Bound<'_, PyAny>,
        metric_type: MetricType,
        global_offset: u64,
        current_indices: &[u64],
        current_scores: &[f64],
        k: usize,
        is_similarity: bool,
    ) -> PyResult<(Py<PyArray1<u64>>, Py<PyArray1<f64>>)> {
        use numpy::PyArrayMethods;

        let query_arr: PyReadonlyArrayDyn<f32> = query.extract()?;
        let candidates_arr: PyReadonlyArrayDyn<f32> = candidates.extract()?;

        let query_slice = query_arr.as_slice()?;
        let readonly_candidates = candidates_arr.readonly();
        let candidates_array = readonly_candidates.as_array();
        let shape = candidates_array.shape();

        if shape.len() != 2 {
            return Err(PyTypeError::new_err("Candidates must be a 2D array"));
        }

        let n_candidates = shape[0];
        let dim = shape[1];

        if query_slice.len() != dim {
            return Err(PyValueError::new_err(format!(
                "Query dimension {} does not match candidates dimension {}",
                query_slice.len(),
                dim
            )));
        }

        let candidates_slice = candidates_arr.as_slice()?;

        // Optimization: use usize to pass addresses
        let query_addr = query_slice.as_ptr() as usize;
        let candidates_addr = candidates_slice.as_ptr() as usize;

        // Copy current top-k for thread safety
        let curr_idx_vec: Vec<u64> = current_indices.to_vec();
        let curr_scores_vec: Vec<f64> = current_scores.to_vec();

        // Release GIL for parallel computation AND merge
        let (new_indices, new_scores) = py
            .allow_threads(|| {
                // Step 1: Compute all scores
                let scores: Vec<f64>;

                #[cfg(feature = "rayon")]
                {
                    use rayon::prelude::*;

                    scores = (0..n_candidates)
                        .into_par_iter()
                        .map(|i| unsafe {
                            let query = std::slice::from_raw_parts(query_addr as *const f32, dim);
                            let candidate = std::slice::from_raw_parts(
                                (candidates_addr + i * dim * std::mem::size_of::<f32>())
                                    as *const f32,
                                dim,
                            );
                            self.engine
                                .cpu_backend
                                .compute_f32(query, candidate, metric_type)
                                .map(|v| v as f64)
                                .unwrap_or(f64::NAN)
                        })
                        .collect();
                }

                #[cfg(not(feature = "rayon"))]
                {
                    scores = (0..n_candidates)
                        .map(|i| unsafe {
                            let query = std::slice::from_raw_parts(query_addr as *const f32, dim);
                            let candidate = std::slice::from_raw_parts(
                                (candidates_addr + i * dim * std::mem::size_of::<f32>())
                                    as *const f32,
                                dim,
                            );
                            self.engine
                                .cpu_backend
                                .compute_f32(query, candidate, metric_type)
                                .map(|v| v as f64)
                                .unwrap_or(f64::NAN)
                        })
                        .collect();
                }

                // Step 2: Merge with current top-k (all in Rust, no FFI overhead)
                Self::merge_top_k_impl(
                    &scores,
                    global_offset,
                    &curr_idx_vec,
                    &curr_scores_vec,
                    k,
                    is_similarity,
                )
            });

        Ok((
            PyArray1::from_vec(py, new_indices).into(),
            PyArray1::from_vec(py, new_scores).into(),
        ))
    }

    /// f64 segmented top-k search (single FFI call optimization)
    fn segmented_top_k_search_f64(
        &self,
        py: Python,
        query: &Bound<'_, PyAny>,
        candidates: &Bound<'_, PyAny>,
        metric_type: MetricType,
        global_offset: u64,
        current_indices: &[u64],
        current_scores: &[f64],
        k: usize,
        is_similarity: bool,
    ) -> PyResult<(Py<PyArray1<u64>>, Py<PyArray1<f64>>)> {
        use numpy::PyArrayMethods;

        let query_arr: PyReadonlyArrayDyn<f64> = query.extract()?;
        let candidates_arr: PyReadonlyArrayDyn<f64> = candidates.extract()?;

        let query_slice = query_arr.as_slice()?;
        let readonly_candidates = candidates_arr.readonly();
        let candidates_array = readonly_candidates.as_array();
        let shape = candidates_array.shape();

        if shape.len() != 2 {
            return Err(PyTypeError::new_err("Candidates must be a 2D array"));
        }

        let n_candidates = shape[0];
        let dim = shape[1];

        if query_slice.len() != dim {
            return Err(PyValueError::new_err(format!(
                "Query dimension {} does not match candidates dimension {}",
                query_slice.len(),
                dim
            )));
        }

        let candidates_slice = candidates_arr.as_slice()?;

        // Optimization: use usize to pass addresses
        let query_addr = query_slice.as_ptr() as usize;
        let candidates_addr = candidates_slice.as_ptr() as usize;

        // Copy current top-k for thread safety
        let curr_idx_vec: Vec<u64> = current_indices.to_vec();
        let curr_scores_vec: Vec<f64> = current_scores.to_vec();

        // Release GIL for parallel computation AND merge
        let (new_indices, new_scores) = py
            .allow_threads(|| {
                // Step 1: Compute all scores
                let scores: Vec<f64>;

                const PARALLEL_THRESHOLD: usize = 500;

                if n_candidates < PARALLEL_THRESHOLD {
                    // Serial for small batches
                    scores = (0..n_candidates)
                        .map(|i| unsafe {
                            let query = std::slice::from_raw_parts(query_addr as *const f64, dim);
                            let candidate = std::slice::from_raw_parts(
                                (candidates_addr + i * dim * std::mem::size_of::<f64>())
                                    as *const f64,
                                dim,
                            );
                            self.engine
                                .cpu_backend
                                .compute_f64(query, candidate, metric_type)
                                .unwrap_or(f64::NAN)
                        })
                        .collect();
                } else {
                    #[cfg(feature = "rayon")]
                    {
                        use rayon::prelude::*;

                        scores = (0..n_candidates)
                            .into_par_iter()
                            .map(|i| unsafe {
                                let query =
                                    std::slice::from_raw_parts(query_addr as *const f64, dim);
                                let candidate = std::slice::from_raw_parts(
                                    (candidates_addr + i * dim * std::mem::size_of::<f64>())
                                        as *const f64,
                                    dim,
                                );
                                self.engine
                                    .cpu_backend
                                    .compute_f64(query, candidate, metric_type)
                                    .unwrap_or(f64::NAN)
                            })
                            .collect();
                    }

                    #[cfg(not(feature = "rayon"))]
                    {
                        scores = (0..n_candidates)
                            .map(|i| unsafe {
                                let query =
                                    std::slice::from_raw_parts(query_addr as *const f64, dim);
                                let candidate = std::slice::from_raw_parts(
                                    (candidates_addr + i * dim * std::mem::size_of::<f64>())
                                        as *const f64,
                                    dim,
                                );
                                self.engine
                                    .cpu_backend
                                    .compute_f64(query, candidate, metric_type)
                                    .unwrap_or(f64::NAN)
                            })
                            .collect();
                    }
                }

                // Step 2: Merge with current top-k (all in Rust, no FFI overhead)
                Self::merge_top_k_impl(
                    &scores,
                    global_offset,
                    &curr_idx_vec,
                    &curr_scores_vec,
                    k,
                    is_similarity,
                )
            });

        Ok((
            PyArray1::from_vec(py, new_indices).into(),
            PyArray1::from_vec(py, new_scores).into(),
        ))
    }

    /// Merge batch scores with current top-k (efficient Rust implementation)
    ///
    /// This replaces the Python heapq loop with a single efficient Rust operation.
    /// Algorithm:
    /// 1. Combine current top-k with batch scores
    /// 2. Use partial sort (select_nth_unstable) to find new top-k
    /// 3. Sort the final top-k results
    fn merge_top_k_impl(
        batch_scores: &[f64],
        global_offset: u64,
        current_indices: &[u64],
        current_scores: &[f64],
        k: usize,
        is_similarity: bool,
    ) -> (Vec<u64>, Vec<f64>) {
        // Early return for empty batch
        if batch_scores.is_empty() {
            return (current_indices.to_vec(), current_scores.to_vec());
        }

        // Calculate total candidates
        let current_count = current_indices.len();
        let batch_count = batch_scores.len();
        let total = current_count + batch_count;
        let k = k.min(total);

        if k == 0 {
            return (Vec::new(), Vec::new());
        }

        // Combine all candidates: (global_index, score)
        let mut all_candidates: Vec<(u64, f64)> = Vec::with_capacity(total);

        // Add current top-k
        for i in 0..current_count {
            all_candidates.push((current_indices[i], current_scores[i]));
        }

        // Add batch scores with global indices
        for (local_idx, &score) in batch_scores.iter().enumerate() {
            let global_idx = global_offset + local_idx as u64;
            all_candidates.push((global_idx, score));
        }

        // Use partial sort to find top-k efficiently
        if is_similarity {
            // Higher is better - descending order
            if all_candidates.len() > k {
                all_candidates.select_nth_unstable_by(k - 1, |a, b| {
                    b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                });
                all_candidates.truncate(k);
            }
            // Sort top-k
            all_candidates.sort_unstable_by(|a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
        } else {
            // Lower is better - ascending order
            if all_candidates.len() > k {
                all_candidates.select_nth_unstable_by(k - 1, |a, b| {
                    a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
                });
                all_candidates.truncate(k);
            }
            // Sort top-k
            all_candidates.sort_unstable_by(|a, b| {
                a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        // Extract indices and scores
        let new_indices: Vec<u64> = all_candidates.iter().map(|(idx, _)| *idx).collect();
        let new_scores: Vec<f64> = all_candidates.iter().map(|(_, score)| *score).collect();

        (new_indices, new_scores)
    }

    // ========================================================================
    // multi_query_top_k implementations for additional types (using conversion)
    // ========================================================================

    /// Multi-query Top-K search (f16) - uses f32 conversion internally
    fn multi_query_top_k_f16(
        &self,
        py: Python,
        queries: &Bound<'_, PyAny>,
        candidates: &Bound<'_, PyAny>,
        metric_type: MetricType,
        k: usize,
    ) -> PyResult<(Py<PyArray1<u64>>, Py<PyArray1<f64>>)> {
        let numpy = py.import("numpy")?;
        let queries_f32 = queries.call_method1("astype", (numpy.getattr("float32")?,))?;
        let candidates_f32 = candidates.call_method1("astype", (numpy.getattr("float32")?,))?;
        self.multi_query_top_k_f32(py, &queries_f32, &candidates_f32, metric_type, k)
    }

    /// Multi-query Top-K search (i8)
    fn multi_query_top_k_i8(
        &self,
        py: Python,
        queries: &Bound<'_, PyAny>,
        candidates: &Bound<'_, PyAny>,
        metric_type: MetricType,
        k: usize,
    ) -> PyResult<(Py<PyArray1<u64>>, Py<PyArray1<f64>>)> {
        let numpy = py.import("numpy")?;
        let queries_f64 = queries.call_method1("astype", (numpy.getattr("float64")?,))?;
        let candidates_f64 = candidates.call_method1("astype", (numpy.getattr("float64")?,))?;
        self.multi_query_top_k_f64(py, &queries_f64, &candidates_f64, metric_type, k)
    }

    /// Multi-query Top-K search (i16)
    fn multi_query_top_k_i16(
        &self,
        py: Python,
        queries: &Bound<'_, PyAny>,
        candidates: &Bound<'_, PyAny>,
        metric_type: MetricType,
        k: usize,
    ) -> PyResult<(Py<PyArray1<u64>>, Py<PyArray1<f64>>)> {
        let numpy = py.import("numpy")?;
        let queries_f64 = queries.call_method1("astype", (numpy.getattr("float64")?,))?;
        let candidates_f64 = candidates.call_method1("astype", (numpy.getattr("float64")?,))?;
        self.multi_query_top_k_f64(py, &queries_f64, &candidates_f64, metric_type, k)
    }

    /// Multi-query Top-K search (i32)
    fn multi_query_top_k_i32(
        &self,
        py: Python,
        queries: &Bound<'_, PyAny>,
        candidates: &Bound<'_, PyAny>,
        metric_type: MetricType,
        k: usize,
    ) -> PyResult<(Py<PyArray1<u64>>, Py<PyArray1<f64>>)> {
        let numpy = py.import("numpy")?;
        let queries_f64 = queries.call_method1("astype", (numpy.getattr("float64")?,))?;
        let candidates_f64 = candidates.call_method1("astype", (numpy.getattr("float64")?,))?;
        self.multi_query_top_k_f64(py, &queries_f64, &candidates_f64, metric_type, k)
    }

    /// Multi-query Top-K search (i64)
    fn multi_query_top_k_i64(
        &self,
        py: Python,
        queries: &Bound<'_, PyAny>,
        candidates: &Bound<'_, PyAny>,
        metric_type: MetricType,
        k: usize,
    ) -> PyResult<(Py<PyArray1<u64>>, Py<PyArray1<f64>>)> {
        let numpy = py.import("numpy")?;
        let queries_f64 = queries.call_method1("astype", (numpy.getattr("float64")?,))?;
        let candidates_f64 = candidates.call_method1("astype", (numpy.getattr("float64")?,))?;
        self.multi_query_top_k_f64(py, &queries_f64, &candidates_f64, metric_type, k)
    }

    /// Multi-query Top-K search (u8)
    fn multi_query_top_k_u8(
        &self,
        py: Python,
        queries: &Bound<'_, PyAny>,
        candidates: &Bound<'_, PyAny>,
        metric_type: MetricType,
        k: usize,
    ) -> PyResult<(Py<PyArray1<u64>>, Py<PyArray1<f64>>)> {
        let numpy = py.import("numpy")?;
        let queries_f64 = queries.call_method1("astype", (numpy.getattr("float64")?,))?;
        let candidates_f64 = candidates.call_method1("astype", (numpy.getattr("float64")?,))?;
        self.multi_query_top_k_f64(py, &queries_f64, &candidates_f64, metric_type, k)
    }

    /// Multi-query Top-K search (u16)
    fn multi_query_top_k_u16(
        &self,
        py: Python,
        queries: &Bound<'_, PyAny>,
        candidates: &Bound<'_, PyAny>,
        metric_type: MetricType,
        k: usize,
    ) -> PyResult<(Py<PyArray1<u64>>, Py<PyArray1<f64>>)> {
        let numpy = py.import("numpy")?;
        let queries_f64 = queries.call_method1("astype", (numpy.getattr("float64")?,))?;
        let candidates_f64 = candidates.call_method1("astype", (numpy.getattr("float64")?,))?;
        self.multi_query_top_k_f64(py, &queries_f64, &candidates_f64, metric_type, k)
    }

    /// Multi-query Top-K search (u32)
    fn multi_query_top_k_u32(
        &self,
        py: Python,
        queries: &Bound<'_, PyAny>,
        candidates: &Bound<'_, PyAny>,
        metric_type: MetricType,
        k: usize,
    ) -> PyResult<(Py<PyArray1<u64>>, Py<PyArray1<f64>>)> {
        let numpy = py.import("numpy")?;
        let queries_f64 = queries.call_method1("astype", (numpy.getattr("float64")?,))?;
        let candidates_f64 = candidates.call_method1("astype", (numpy.getattr("float64")?,))?;
        self.multi_query_top_k_f64(py, &queries_f64, &candidates_f64, metric_type, k)
    }

    /// Multi-query Top-K search (u64)
    fn multi_query_top_k_u64(
        &self,
        py: Python,
        queries: &Bound<'_, PyAny>,
        candidates: &Bound<'_, PyAny>,
        metric_type: MetricType,
        k: usize,
    ) -> PyResult<(Py<PyArray1<u64>>, Py<PyArray1<f64>>)> {
        let numpy = py.import("numpy")?;
        let queries_f64 = queries.call_method1("astype", (numpy.getattr("float64")?,))?;
        let candidates_f64 = candidates.call_method1("astype", (numpy.getattr("float64")?,))?;
        self.multi_query_top_k_f64(py, &queries_f64, &candidates_f64, metric_type, k)
    }

    // ========================================================================
    // segmented_top_k_search implementations for additional types
    // ========================================================================

    /// segmented_top_k_search (f16)
    fn segmented_top_k_search_f16(
        &self,
        py: Python,
        query: &Bound<'_, PyAny>,
        candidates: &Bound<'_, PyAny>,
        metric_type: MetricType,
        global_offset: u64,
        current_indices: &[u64],
        current_scores: &[f64],
        k: usize,
        is_similarity: bool,
    ) -> PyResult<(Py<PyArray1<u64>>, Py<PyArray1<f64>>)> {
        // Compute batch scores using f16 batch_compute
        let scores_array = self.batch_compute_f16_impl(py, query, candidates, metric_type)?;
        let scores = scores_array.bind(py).readonly();
        let scores_slice = scores.as_slice()?;

        let (new_indices, new_scores) = Self::merge_top_k_impl(
            scores_slice, global_offset, current_indices, current_scores, k, is_similarity,
        );

        Ok((
            PyArray1::from_vec(py, new_indices).into(),
            PyArray1::from_vec(py, new_scores).into(),
        ))
    }

    /// segmented_top_k_search (i8)
    fn segmented_top_k_search_i8(
        &self,
        py: Python,
        query: &Bound<'_, PyAny>,
        candidates: &Bound<'_, PyAny>,
        metric_type: MetricType,
        global_offset: u64,
        current_indices: &[u64],
        current_scores: &[f64],
        k: usize,
        is_similarity: bool,
    ) -> PyResult<(Py<PyArray1<u64>>, Py<PyArray1<f64>>)> {
        let scores_array = self.batch_compute_i8(py, query, candidates, metric_type)?;
        let scores = scores_array.bind(py).readonly();
        let scores_slice = scores.as_slice()?;

        let (new_indices, new_scores) = Self::merge_top_k_impl(
            scores_slice, global_offset, current_indices, current_scores, k, is_similarity,
        );

        Ok((
            PyArray1::from_vec(py, new_indices).into(),
            PyArray1::from_vec(py, new_scores).into(),
        ))
    }

    /// segmented_top_k_search (i16)
    fn segmented_top_k_search_i16(
        &self,
        py: Python,
        query: &Bound<'_, PyAny>,
        candidates: &Bound<'_, PyAny>,
        metric_type: MetricType,
        global_offset: u64,
        current_indices: &[u64],
        current_scores: &[f64],
        k: usize,
        is_similarity: bool,
    ) -> PyResult<(Py<PyArray1<u64>>, Py<PyArray1<f64>>)> {
        let scores_array = self.batch_compute_i16(py, query, candidates, metric_type)?;
        let scores = scores_array.bind(py).readonly();
        let scores_slice = scores.as_slice()?;

        let (new_indices, new_scores) = Self::merge_top_k_impl(
            scores_slice, global_offset, current_indices, current_scores, k, is_similarity,
        );

        Ok((
            PyArray1::from_vec(py, new_indices).into(),
            PyArray1::from_vec(py, new_scores).into(),
        ))
    }

    /// segmented_top_k_search (i32)
    fn segmented_top_k_search_i32(
        &self,
        py: Python,
        query: &Bound<'_, PyAny>,
        candidates: &Bound<'_, PyAny>,
        metric_type: MetricType,
        global_offset: u64,
        current_indices: &[u64],
        current_scores: &[f64],
        k: usize,
        is_similarity: bool,
    ) -> PyResult<(Py<PyArray1<u64>>, Py<PyArray1<f64>>)> {
        let scores_array = self.batch_compute_i32(py, query, candidates, metric_type)?;
        let scores = scores_array.bind(py).readonly();
        let scores_slice = scores.as_slice()?;

        let (new_indices, new_scores) = Self::merge_top_k_impl(
            scores_slice, global_offset, current_indices, current_scores, k, is_similarity,
        );

        Ok((
            PyArray1::from_vec(py, new_indices).into(),
            PyArray1::from_vec(py, new_scores).into(),
        ))
    }

    /// segmented_top_k_search (i64)
    fn segmented_top_k_search_i64(
        &self,
        py: Python,
        query: &Bound<'_, PyAny>,
        candidates: &Bound<'_, PyAny>,
        metric_type: MetricType,
        global_offset: u64,
        current_indices: &[u64],
        current_scores: &[f64],
        k: usize,
        is_similarity: bool,
    ) -> PyResult<(Py<PyArray1<u64>>, Py<PyArray1<f64>>)> {
        let scores_array = self.batch_compute_i64(py, query, candidates, metric_type)?;
        let scores = scores_array.bind(py).readonly();
        let scores_slice = scores.as_slice()?;

        let (new_indices, new_scores) = Self::merge_top_k_impl(
            scores_slice, global_offset, current_indices, current_scores, k, is_similarity,
        );

        Ok((
            PyArray1::from_vec(py, new_indices).into(),
            PyArray1::from_vec(py, new_scores).into(),
        ))
    }

    /// segmented_top_k_search (u8)
    fn segmented_top_k_search_u8(
        &self,
        py: Python,
        query: &Bound<'_, PyAny>,
        candidates: &Bound<'_, PyAny>,
        metric_type: MetricType,
        global_offset: u64,
        current_indices: &[u64],
        current_scores: &[f64],
        k: usize,
        is_similarity: bool,
    ) -> PyResult<(Py<PyArray1<u64>>, Py<PyArray1<f64>>)> {
        let scores_array = self.batch_compute_u8(py, query, candidates, metric_type)?;
        let scores = scores_array.bind(py).readonly();
        let scores_slice = scores.as_slice()?;

        let (new_indices, new_scores) = Self::merge_top_k_impl(
            scores_slice, global_offset, current_indices, current_scores, k, is_similarity,
        );

        Ok((
            PyArray1::from_vec(py, new_indices).into(),
            PyArray1::from_vec(py, new_scores).into(),
        ))
    }

    /// segmented_top_k_search (u16)
    fn segmented_top_k_search_u16(
        &self,
        py: Python,
        query: &Bound<'_, PyAny>,
        candidates: &Bound<'_, PyAny>,
        metric_type: MetricType,
        global_offset: u64,
        current_indices: &[u64],
        current_scores: &[f64],
        k: usize,
        is_similarity: bool,
    ) -> PyResult<(Py<PyArray1<u64>>, Py<PyArray1<f64>>)> {
        let scores_array = self.batch_compute_u16(py, query, candidates, metric_type)?;
        let scores = scores_array.bind(py).readonly();
        let scores_slice = scores.as_slice()?;

        let (new_indices, new_scores) = Self::merge_top_k_impl(
            scores_slice, global_offset, current_indices, current_scores, k, is_similarity,
        );

        Ok((
            PyArray1::from_vec(py, new_indices).into(),
            PyArray1::from_vec(py, new_scores).into(),
        ))
    }

    /// segmented_top_k_search (u32)
    fn segmented_top_k_search_u32(
        &self,
        py: Python,
        query: &Bound<'_, PyAny>,
        candidates: &Bound<'_, PyAny>,
        metric_type: MetricType,
        global_offset: u64,
        current_indices: &[u64],
        current_scores: &[f64],
        k: usize,
        is_similarity: bool,
    ) -> PyResult<(Py<PyArray1<u64>>, Py<PyArray1<f64>>)> {
        let scores_array = self.batch_compute_u32(py, query, candidates, metric_type)?;
        let scores = scores_array.bind(py).readonly();
        let scores_slice = scores.as_slice()?;

        let (new_indices, new_scores) = Self::merge_top_k_impl(
            scores_slice, global_offset, current_indices, current_scores, k, is_similarity,
        );

        Ok((
            PyArray1::from_vec(py, new_indices).into(),
            PyArray1::from_vec(py, new_scores).into(),
        ))
    }

    /// segmented_top_k_search (u64)
    fn segmented_top_k_search_u64(
        &self,
        py: Python,
        query: &Bound<'_, PyAny>,
        candidates: &Bound<'_, PyAny>,
        metric_type: MetricType,
        global_offset: u64,
        current_indices: &[u64],
        current_scores: &[f64],
        k: usize,
        is_similarity: bool,
    ) -> PyResult<(Py<PyArray1<u64>>, Py<PyArray1<f64>>)> {
        let scores_array = self.batch_compute_u64(py, query, candidates, metric_type)?;
        let scores = scores_array.bind(py).readonly();
        let scores_slice = scores.as_slice()?;

        let (new_indices, new_scores) = Self::merge_top_k_impl(
            scores_slice, global_offset, current_indices, current_scores, k, is_similarity,
        );

        Ok((
            PyArray1::from_vec(py, new_indices).into(),
            PyArray1::from_vec(py, new_scores).into(),
        ))
    }
}

// ========================================================================
// StreamingVectorSearch: Memory-efficient streaming vector search from files
// ========================================================================

/// Python wrapper for streaming vector search operations
///
/// StreamingVectorSearch is designed for memory-efficient vector similarity search
/// on large datasets that don't fit in memory. It reads data directly from NumPack
/// files using memory mapping, processing data in batches.
///
/// Features:
/// - Single FFI call for entire search operation
/// - Zero-copy mmap file access
/// - No intermediate Python array allocation per batch
/// - ~10-30x faster than Python-based streaming
///
/// For in-memory operations where all data fits in memory, use VectorSearch instead.
#[pyclass(module = "numpack.vector_engine", name = "StreamingVectorSearch")]
pub struct PyStreamingVectorSearch {
    engine: VectorEngine,
}

#[pymethods]
impl PyStreamingVectorSearch {
    /// Create a new StreamingVectorSearch instance
    ///
    /// Automatically detects CPU SIMD capabilities (AVX2, AVX-512, NEON, SVE).
    #[new]
    pub fn new() -> Self {
        Self {
            engine: VectorEngine::new(),
        }
    }

    /// Get SIMD capabilities information
    ///
    /// Returns:
    ///     str: A string describing detected SIMD features (e.g., "CPU: AVX2, AVX-512")
    pub fn capabilities(&self) -> String {
        self.engine.capabilities()
    }

    /// Streaming Top-K search directly from NumPack file (Rust-native, single FFI call)
    ///
    /// This is a highly optimized streaming search that:
    /// 1. Reads candidates directly from NumPack file using memory mapping
    /// 2. Processes data in batches entirely within Rust (no Python-Rust data transfer per batch)
    /// 3. Maintains global Top-K using efficient partial sort
    ///
    /// Performance: ~10-30x faster than Python-based streaming due to:
    /// - Single FFI call instead of N calls for N batches
    /// - Zero-copy mmap file access
    /// - No intermediate Python array allocation
    ///
    /// Args:
    ///     query: Query vector (1D numpy array, float32 or float64)
    ///     npk_dir: NumPack directory path (string)
    ///     array_name: Name of candidates array in NumPack file
    ///     metric: Metric type string ('cosine', 'dot', 'l2', etc.)
    ///     k: Number of top results to return
    ///     batch_size: Number of rows to process per batch (default: 10000)
    ///
    /// Returns:
    ///     tuple: (indices, scores)
    ///         - indices: Global indices of top-k candidates (uint64)
    ///         - scores: Corresponding metric scores (float64)
    #[pyo3(signature = (query, npk_dir, array_name, metric, k, batch_size=10000))]
    pub fn streaming_top_k_from_file(
        &self,
        py: Python,
        query: &Bound<'_, PyAny>,
        npk_dir: &str,
        array_name: &str,
        metric: &str,
        k: usize,
        batch_size: usize,
    ) -> PyResult<(Py<PyArray1<u64>>, Py<PyArray1<f64>>)> {
        // Parse metric type
        let metric_type = MetricType::from_str(metric)
            .ok_or_else(|| PyValueError::new_err(format!("Unknown metric: {}", metric)))?;

        // Get query dtype
        let query_dtype = query.getattr("dtype")?.str()?.to_string();

        // Dispatch based on dtype - convert non-float types to f64
        match query_dtype.as_str() {
            "float32" => self.streaming_top_k_from_file_f32(
                py,
                query,
                npk_dir,
                array_name,
                metric_type,
                k,
                batch_size,
            ),
            "float64" => self.streaming_top_k_from_file_f64(
                py,
                query,
                npk_dir,
                array_name,
                metric_type,
                k,
                batch_size,
            ),
            "float16" => {
                // Convert f16 query to f32
                let numpy = py.import("numpy")?;
                let query_f32 = query.call_method1("astype", (numpy.getattr("float32")?,))?;
                self.streaming_top_k_from_file_f32(
                    py, &query_f32, npk_dir, array_name, metric_type, k, batch_size,
                )
            }
            "int8" | "int16" | "int32" | "int64" | "uint8" | "uint16" | "uint32" | "uint64" => {
                // Convert integer query to f32 (most common file format)
                let numpy = py.import("numpy")?;
                let query_f32 = query.call_method1("astype", (numpy.getattr("float32")?,))?;
                self.streaming_top_k_from_file_f32(
                    py, &query_f32, npk_dir, array_name, metric_type, k, batch_size,
                )
            }
            _ => Err(PyTypeError::new_err(format!(
                "Unsupported dtype: {}. Supported: float64, float32, float16, int8, int16, int32, int64, uint8, uint16, uint32, uint64",
                query_dtype
            ))),
        }
    }

    /// Streaming batch compute directly from NumPack file (Rust-native, single FFI call)
    ///
    /// Computes metric values between query and all candidates without loading
    /// all data into Python memory.
    ///
    /// Args:
    ///     query: Query vector (1D numpy array, float32 or float64)
    ///     npk_dir: NumPack directory path (string)
    ///     array_name: Name of candidates array in NumPack file
    ///     metric: Metric type string ('cosine', 'dot', 'l2', etc.)
    ///     batch_size: Number of rows to process per batch (default: 10000)
    ///
    /// Returns:
    ///     numpy.ndarray: All computed metric values (1D array of float64)
    #[pyo3(signature = (query, npk_dir, array_name, metric, batch_size=10000))]
    pub fn streaming_batch_compute(
        &self,
        py: Python,
        query: &Bound<'_, PyAny>,
        npk_dir: &str,
        array_name: &str,
        metric: &str,
        batch_size: usize,
    ) -> PyResult<Py<PyArray1<f64>>> {
        // Parse metric type
        let metric_type = MetricType::from_str(metric)
            .ok_or_else(|| PyValueError::new_err(format!("Unknown metric: {}", metric)))?;

        // Get query dtype
        let query_dtype = query.getattr("dtype")?.str()?.to_string();

        // Dispatch based on dtype - convert non-float types to f64
        match query_dtype.as_str() {
            "float32" => self.streaming_batch_compute_f32(
                py,
                query,
                npk_dir,
                array_name,
                metric_type,
                batch_size,
            ),
            "float64" => self.streaming_batch_compute_f64(
                py,
                query,
                npk_dir,
                array_name,
                metric_type,
                batch_size,
            ),
            "float16" => {
                // Convert f16 query to f32
                let numpy = py.import("numpy")?;
                let query_f32 = query.call_method1("astype", (numpy.getattr("float32")?,))?;
                self.streaming_batch_compute_f32(
                    py, &query_f32, npk_dir, array_name, metric_type, batch_size,
                )
            }
            "int8" | "int16" | "int32" | "int64" | "uint8" | "uint16" | "uint32" | "uint64" => {
                // Convert integer query to f32 (most common file format)
                let numpy = py.import("numpy")?;
                let query_f32 = query.call_method1("astype", (numpy.getattr("float32")?,))?;
                self.streaming_batch_compute_f32(
                    py, &query_f32, npk_dir, array_name, metric_type, batch_size,
                )
            }
            _ => Err(PyTypeError::new_err(format!(
                "Unsupported dtype: {}. Supported: float64, float32, float16, int8, int16, int32, int64, uint8, uint16, uint32, uint64",
                query_dtype
            ))),
        }
    }

    /// Batch multi-query Top-K search from file (optimized for multiple queries)
    ///
    /// This method opens the file once and executes multiple queries, amortizing
    /// the file open and metadata loading overhead across all queries.
    ///
    /// Performance: ~2x faster than calling streaming_top_k_from_file repeatedly
    /// for multi-query scenarios.
    ///
    /// Args:
    ///     queries: Multiple query vectors (2D numpy array, shape: [N, D], float32 or float64)
    ///     npk_dir: NumPack directory path (string)
    ///     array_name: Name of candidates array in NumPack file
    ///     metric: Metric type string ('cosine', 'dot', 'l2', etc.)
    ///     k: Number of top results to return per query
    ///     batch_size: Number of rows to process per batch (default: 10000)
    ///
    /// Returns:
    ///     tuple: (all_indices, all_scores)
    ///         - all_indices: 2D array of shape [N, k] with top-k indices per query
    ///         - all_scores: 2D array of shape [N, k] with top-k scores per query
    #[pyo3(signature = (queries, npk_dir, array_name, metric, k, batch_size=10000))]
    pub fn streaming_multi_query_top_k(
        &self,
        py: Python,
        queries: &Bound<'_, PyAny>,
        npk_dir: &str,
        array_name: &str,
        metric: &str,
        k: usize,
        batch_size: usize,
    ) -> PyResult<(Py<PyArray1<u64>>, Py<PyArray1<f64>>)> {
        // Parse metric type
        let metric_type = MetricType::from_str(metric)
            .ok_or_else(|| PyValueError::new_err(format!("Unknown metric: {}", metric)))?;

        // Get queries dtype and shape - convert to f32 if necessary
        let queries_dtype = queries.getattr("dtype")?.str()?.to_string();
        
        // Convert non-f32 types to f32
        let queries_bound: std::borrow::Cow<'_, Bound<'_, PyAny>>;
        let queries_ref: &Bound<'_, PyAny>;
        
        match queries_dtype.as_str() {
            "float32" => {
                queries_ref = queries;
            }
            "float64" | "float16" | "int8" | "int16" | "int32" | "int64" | "uint8" | "uint16" | "uint32" | "uint64" => {
                let numpy = py.import("numpy")?;
                queries_bound = std::borrow::Cow::Owned(
                    queries.call_method1("astype", (numpy.getattr("float32")?,))?
                );
                queries_ref = queries_bound.as_ref();
            }
            _ => {
                return Err(PyTypeError::new_err(format!(
                    "Unsupported dtype: {}. Supported: float64, float32, float16, int8, int16, int32, int64, uint8, uint16, uint32, uint64",
                    queries_dtype
                )));
            }
        }
        
        let queries_arr: PyReadonlyArrayDyn<f32> = queries_ref.extract()?;

        let readonly_queries = queries_arr.readonly();
        let queries_array = readonly_queries.as_array();
        let queries_shape = queries_array.shape();
        if queries_shape.len() != 2 {
            return Err(PyValueError::new_err("Queries must be a 2D array"));
        }
        let n_queries = queries_shape[0];
        let dim = queries_shape[1];

        // Load metadata once
        let base_dir = Path::new(npk_dir);
        let metadata_path = base_dir.join("metadata.npkm");
        let metadata_store = BinaryMetadataStore::load(&metadata_path)
            .map_err(|e| PyValueError::new_err(format!("Failed to load metadata: {}", e)))?;

        let array_meta = metadata_store.get_array(array_name).ok_or_else(|| {
            PyValueError::new_err(format!("Array '{}' not found in NumPack", array_name))
        })?;

        // Verify dtype
        if array_meta.dtype != BinaryDataType::Float32 {
            return Err(PyTypeError::new_err(format!(
                "Array dtype mismatch: queries are float32 but array is {:?}",
                array_meta.dtype
            )));
        }

        // Verify dimensions
        let array_dim = if array_meta.shape.len() > 1 {
            array_meta.shape[1] as usize
        } else {
            1
        };
        if array_dim != dim {
            return Err(PyValueError::new_err(format!(
                "Dimension mismatch: query dim {} vs array dim {}",
                dim, array_dim
            )));
        }

        let total_rows = array_meta.shape[0] as usize;
        let data_path = base_dir.join(format!("data_{}.npkd", array_name));
        let is_similarity = metric_type.is_similarity();

        // Copy queries to Vec for thread safety
        let queries_slice = queries_arr.as_slice()?;
        let queries_vec: Vec<f32> = queries_slice.to_vec();

        // Release GIL for file I/O and computation
        let (all_indices, all_scores) = py.allow_threads(|| {
            // Open file and mmap ONCE
            let file = std::fs::File::open(&data_path)
                .map_err(|e| format!("Failed to open data file: {}", e))?;
            let mmap = unsafe {
                MmapOptions::new()
                    .map(&file)
                    .map_err(|e| format!("Failed to mmap: {}", e))?
            };

            // Advise kernel for sequential access
            #[cfg(unix)]
            {
                unsafe {
                    libc::madvise(
                        mmap.as_ptr() as *mut libc::c_void,
                        mmap.len(),
                        libc::MADV_SEQUENTIAL,
                    );
                }
            }

            let row_bytes = dim * std::mem::size_of::<f32>();
            let mmap_ptr = mmap.as_ptr() as usize;
            let mmap_len = mmap.len();

            // Process all queries
            let mut all_indices: Vec<Vec<u64>> = Vec::with_capacity(n_queries);
            let mut all_scores: Vec<Vec<f64>> = Vec::with_capacity(n_queries);

            for q_idx in 0..n_queries {
                let query_start = q_idx * dim;
                let query_vec: Vec<f32> = queries_vec[query_start..query_start + dim].to_vec();

                // Single-pass parallel processing for this query
                #[cfg(feature = "rayon")]
                let scores: Vec<f64> = {
                    use rayon::prelude::*;
                    
                    (0..total_rows)
                        .into_par_iter()
                        .map(|row_idx| {
                            let offset = row_idx * row_bytes;

                            if offset + row_bytes > mmap_len {
                                return f64::NAN;
                            }

                            let candidate = unsafe {
                                std::slice::from_raw_parts(
                                    (mmap_ptr + offset) as *const f32,
                                    dim,
                                )
                            };

                            self.engine
                                .cpu_backend
                                .compute_f32(&query_vec, candidate, metric_type)
                                .unwrap_or(f32::NAN) as f64
                        })
                        .collect()
                };

                #[cfg(not(feature = "rayon"))]
                let scores: Vec<f64> = {
                    (0..total_rows)
                        .map(|row_idx| {
                            let offset = row_idx * row_bytes;

                            if offset + row_bytes > mmap_len {
                                return f64::NAN;
                            }

                            let candidate = unsafe {
                                std::slice::from_raw_parts(
                                    (mmap_ptr + offset) as *const f32,
                                    dim,
                                )
                            };

                            self.engine
                                .cpu_backend
                                .compute_f32(&query_vec, candidate, metric_type)
                                .unwrap_or(f32::NAN) as f64
                        })
                        .collect()
                };

                // Merge to get top-k for this query
                let (indices, query_scores) = Self::merge_top_k_impl(
                    &scores,
                    0,
                    &[],
                    &[],
                    k,
                    is_similarity,
                );

                all_indices.push(indices);
                all_scores.push(query_scores);
            }

            Ok::<_, String>((all_indices, all_scores))
        })
        .map_err(|e| PyValueError::new_err(e))?;

        // Convert to flat 1D numpy arrays (will be reshaped in Python if needed)
        // Format: [q0_idx0, q0_idx1, ..., q0_idxK, q1_idx0, q1_idx1, ..., qN_idxK]
        let mut indices_flat: Vec<u64> = Vec::with_capacity(n_queries * k);
        let mut scores_flat: Vec<f64> = Vec::with_capacity(n_queries * k);

        for (indices, scores) in all_indices.iter().zip(all_scores.iter()) {
            // Pad if needed
            for i in 0..k {
                if i < indices.len() {
                    indices_flat.push(indices[i]);
                    scores_flat.push(scores[i]);
                } else {
                    indices_flat.push(0);
                    scores_flat.push(f64::NAN);
                }
            }
        }

        Ok((
            PyArray1::from_vec(py, indices_flat).into(),
            PyArray1::from_vec(py, scores_flat).into(),
        ))
    }
}

// Private streaming implementations for PyStreamingVectorSearch
impl PyStreamingVectorSearch {
    /// Streaming Top-K from file (f32)
    fn streaming_top_k_from_file_f32(
        &self,
        py: Python,
        query: &Bound<'_, PyAny>,
        npk_dir: &str,
        array_name: &str,
        metric_type: MetricType,
        k: usize,
        batch_size: usize,
    ) -> PyResult<(Py<PyArray1<u64>>, Py<PyArray1<f64>>)> {
        // Extract query
        let query_arr: PyReadonlyArrayDyn<f32> = query.extract()?;
        let query_slice = query_arr.as_slice()?;
        let query_vec: Vec<f32> = query_slice.to_vec();
        let dim = query_vec.len();

        // Load metadata and get array info
        let base_dir = Path::new(npk_dir);
        let metadata_path = base_dir.join("metadata.npkm");
        let metadata_store = BinaryMetadataStore::load(&metadata_path)
            .map_err(|e| PyValueError::new_err(format!("Failed to load metadata: {}", e)))?;

        let array_meta = metadata_store.get_array(array_name).ok_or_else(|| {
            PyValueError::new_err(format!("Array '{}' not found in NumPack", array_name))
        })?;

        // Verify dtype
        if array_meta.dtype != BinaryDataType::Float32 {
            return Err(PyTypeError::new_err(format!(
                "Array dtype mismatch: query is float32 but array is {:?}",
                array_meta.dtype
            )));
        }

        // Verify dimensions
        let array_dim = if array_meta.shape.len() > 1 {
            array_meta.shape[1] as usize
        } else {
            1
        };
        if array_dim != dim {
            return Err(PyValueError::new_err(format!(
                "Dimension mismatch: query dim {} vs array dim {}",
                dim, array_dim
            )));
        }

        let total_rows = array_meta.shape[0] as usize;
        let data_path = base_dir.join(format!("data_{}.npkd", array_name));

        // Determine if higher is better
        let is_similarity = metric_type.is_similarity();

        // Release GIL for file I/O and computation
        let (indices, scores) = py.allow_threads(|| {
            // Memory map the data file
            let file = std::fs::File::open(&data_path)
                .map_err(|e| format!("Failed to open data file: {}", e))?;
            let mmap = unsafe {
                MmapOptions::new()
                    .map(&file)
                    .map_err(|e| format!("Failed to mmap: {}", e))?
            };

            // Advise kernel for sequential access (improves read-ahead)
            #[cfg(unix)]
            {
                unsafe {
                    libc::madvise(
                        mmap.as_ptr() as *mut libc::c_void,
                        mmap.len(),
                        libc::MADV_SEQUENTIAL,
                    );
                }
            }

            let row_bytes = dim * std::mem::size_of::<f32>();
            let mmap_ptr = mmap.as_ptr() as usize;
            let mmap_len = mmap.len();

            // Optimization: if data is small enough, process all at once to avoid batch overhead
            // This eliminates multiple merge_top_k calls
            const SINGLE_PASS_THRESHOLD: usize = 200000;  // ~100MB for dim=128

            if total_rows <= SINGLE_PASS_THRESHOLD {
                // Single-pass parallel processing for better cache utilization
                #[cfg(feature = "rayon")]
                let all_scores: Vec<f64> = {
                    use rayon::prelude::*;
                    
                    (0..total_rows)
                        .into_par_iter()
                        .map(|row_idx| {
                            let offset = row_idx * row_bytes;

                            if offset + row_bytes > mmap_len {
                                return f64::NAN;
                            }

                            let candidate = unsafe {
                                std::slice::from_raw_parts(
                                    (mmap_ptr + offset) as *const f32,
                                    dim,
                                )
                            };

                            self.engine
                                .cpu_backend
                                .compute_f32(&query_vec, candidate, metric_type)
                                .unwrap_or(f32::NAN) as f64
                        })
                        .collect()
                };

                #[cfg(not(feature = "rayon"))]
                let all_scores: Vec<f64> = {
                    (0..total_rows)
                        .map(|row_idx| {
                            let offset = row_idx * row_bytes;

                            if offset + row_bytes > mmap_len {
                                return f64::NAN;
                            }

                            let candidate = unsafe {
                                std::slice::from_raw_parts(
                                    (mmap_ptr + offset) as *const f32,
                                    dim,
                                )
                            };

                            self.engine
                                .cpu_backend
                                .compute_f32(&query_vec, candidate, metric_type)
                                .unwrap_or(f32::NAN) as f64
                        })
                        .collect()
                };

                // Single merge call for all scores
                let (indices, scores) = Self::merge_top_k_impl(
                    &all_scores,
                    0,
                    &[],
                    &[],
                    k,
                    is_similarity,
                );

                return Ok((indices, scores));
            }

            // For very large datasets, use batched processing
            let mut current_indices: Vec<u64> = Vec::new();
            let mut current_scores: Vec<f64> = Vec::new();

            for batch_start in (0..total_rows).step_by(batch_size) {
                let batch_end = (batch_start + batch_size).min(total_rows);
                let batch_count = batch_end - batch_start;

                // Always use parallel for streaming (data is large)
                #[cfg(feature = "rayon")]
                let batch_scores: Vec<f64> = {
                    use rayon::prelude::*;
                    
                    (0..batch_count)
                        .into_par_iter()
                        .map(|i| {
                            let row_idx = batch_start + i;
                            let offset = row_idx * row_bytes;

                            if offset + row_bytes > mmap_len {
                                return f64::NAN;
                            }

                            let candidate = unsafe {
                                std::slice::from_raw_parts(
                                    (mmap_ptr + offset) as *const f32,
                                    dim,
                                )
                            };

                            self.engine
                                .cpu_backend
                                .compute_f32(&query_vec, candidate, metric_type)
                                .unwrap_or(f32::NAN) as f64
                        })
                        .collect()
                };

                #[cfg(not(feature = "rayon"))]
                let batch_scores: Vec<f64> = {
                    (0..batch_count)
                        .map(|i| {
                            let row_idx = batch_start + i;
                            let offset = row_idx * row_bytes;

                            if offset + row_bytes > mmap_len {
                                return f64::NAN;
                            }

                            let candidate = unsafe {
                                std::slice::from_raw_parts(
                                    (mmap_ptr + offset) as *const f32,
                                    dim,
                                )
                            };

                            self.engine
                                .cpu_backend
                                .compute_f32(&query_vec, candidate, metric_type)
                                .unwrap_or(f32::NAN) as f64
                        })
                        .collect()
                };

                // Merge with current top-k
                let (new_indices, new_scores) = Self::merge_top_k_impl(
                    &batch_scores,
                    batch_start as u64,
                    &current_indices,
                    &current_scores,
                    k,
                    is_similarity,
                );

                current_indices = new_indices;
                current_scores = new_scores;
            }

            Ok::<_, String>((current_indices, current_scores))
        })
        .map_err(|e| PyValueError::new_err(e))?;

        Ok((
            PyArray1::from_vec(py, indices).into(),
            PyArray1::from_vec(py, scores).into(),
        ))
    }

    /// Streaming Top-K from file (f64)
    fn streaming_top_k_from_file_f64(
        &self,
        py: Python,
        query: &Bound<'_, PyAny>,
        npk_dir: &str,
        array_name: &str,
        metric_type: MetricType,
        k: usize,
        batch_size: usize,
    ) -> PyResult<(Py<PyArray1<u64>>, Py<PyArray1<f64>>)> {
        // Extract query
        let query_arr: PyReadonlyArrayDyn<f64> = query.extract()?;
        let query_slice = query_arr.as_slice()?;
        let query_vec: Vec<f64> = query_slice.to_vec();
        let dim = query_vec.len();

        // Load metadata
        let base_dir = Path::new(npk_dir);
        let metadata_path = base_dir.join("metadata.npkm");
        let metadata_store = BinaryMetadataStore::load(&metadata_path)
            .map_err(|e| PyValueError::new_err(format!("Failed to load metadata: {}", e)))?;

        let array_meta = metadata_store.get_array(array_name).ok_or_else(|| {
            PyValueError::new_err(format!("Array '{}' not found in NumPack", array_name))
        })?;

        // Verify dtype
        if array_meta.dtype != BinaryDataType::Float64 {
            return Err(PyTypeError::new_err(format!(
                "Array dtype mismatch: query is float64 but array is {:?}",
                array_meta.dtype
            )));
        }

        let array_dim = if array_meta.shape.len() > 1 {
            array_meta.shape[1] as usize
        } else {
            1
        };
        if array_dim != dim {
            return Err(PyValueError::new_err(format!(
                "Dimension mismatch: query dim {} vs array dim {}",
                dim, array_dim
            )));
        }

        let total_rows = array_meta.shape[0] as usize;
        let data_path = base_dir.join(format!("data_{}.npkd", array_name));
        let is_similarity = metric_type.is_similarity();

        // Release GIL
        let (indices, scores) = py.allow_threads(|| {
            let file = std::fs::File::open(&data_path)
                .map_err(|e| format!("Failed to open data file: {}", e))?;
            let mmap = unsafe {
                MmapOptions::new()
                    .map(&file)
                    .map_err(|e| format!("Failed to mmap: {}", e))?
            };

            // Advise kernel for sequential access (improves read-ahead)
            #[cfg(unix)]
            {
                unsafe {
                    libc::madvise(
                        mmap.as_ptr() as *mut libc::c_void,
                        mmap.len(),
                        libc::MADV_SEQUENTIAL,
                    );
                }
            }

            let row_bytes = dim * std::mem::size_of::<f64>();
            let mmap_ptr = mmap.as_ptr() as usize;
            let mmap_len = mmap.len();

            // Optimization: if data is small enough, process all at once to avoid batch overhead
            const SINGLE_PASS_THRESHOLD: usize = 200000;

            if total_rows <= SINGLE_PASS_THRESHOLD {
                // Single-pass parallel processing
                #[cfg(feature = "rayon")]
                let all_scores: Vec<f64> = {
                    use rayon::prelude::*;
                    
                    (0..total_rows)
                        .into_par_iter()
                        .map(|row_idx| {
                            let offset = row_idx * row_bytes;

                            if offset + row_bytes > mmap_len {
                                return f64::NAN;
                            }

                            let candidate = unsafe {
                                std::slice::from_raw_parts(
                                    (mmap_ptr + offset) as *const f64,
                                    dim,
                                )
                            };

                            self.engine
                                .cpu_backend
                                .compute_f64(&query_vec, candidate, metric_type)
                                .unwrap_or(f64::NAN)
                        })
                        .collect()
                };

                #[cfg(not(feature = "rayon"))]
                let all_scores: Vec<f64> = {
                    (0..total_rows)
                        .map(|row_idx| {
                            let offset = row_idx * row_bytes;

                            if offset + row_bytes > mmap_len {
                                return f64::NAN;
                            }

                            let candidate = unsafe {
                                std::slice::from_raw_parts(
                                    (mmap_ptr + offset) as *const f64,
                                    dim,
                                )
                            };

                            self.engine
                                .cpu_backend
                                .compute_f64(&query_vec, candidate, metric_type)
                                .unwrap_or(f64::NAN)
                        })
                        .collect()
                };

                // Single merge call for all scores
                let (indices, scores) = Self::merge_top_k_impl(
                    &all_scores,
                    0,
                    &[],
                    &[],
                    k,
                    is_similarity,
                );

                return Ok((indices, scores));
            }

            // For very large datasets, use batched processing
            let mut current_indices: Vec<u64> = Vec::new();
            let mut current_scores: Vec<f64> = Vec::new();

            for batch_start in (0..total_rows).step_by(batch_size) {
                let batch_end = (batch_start + batch_size).min(total_rows);
                let batch_count = batch_end - batch_start;

                // Always use parallel for streaming (data is large)
                #[cfg(feature = "rayon")]
                let batch_scores: Vec<f64> = {
                    use rayon::prelude::*;
                    
                    (0..batch_count)
                        .into_par_iter()
                        .map(|i| {
                            let row_idx = batch_start + i;
                            let offset = row_idx * row_bytes;

                            if offset + row_bytes > mmap_len {
                                return f64::NAN;
                            }

                            let candidate = unsafe {
                                std::slice::from_raw_parts(
                                    (mmap_ptr + offset) as *const f64,
                                    dim,
                                )
                            };

                            self.engine
                                .cpu_backend
                                .compute_f64(&query_vec, candidate, metric_type)
                                .unwrap_or(f64::NAN)
                        })
                        .collect()
                };

                #[cfg(not(feature = "rayon"))]
                let batch_scores: Vec<f64> = {
                    (0..batch_count)
                        .map(|i| {
                            let row_idx = batch_start + i;
                            let offset = row_idx * row_bytes;

                            if offset + row_bytes > mmap_len {
                                return f64::NAN;
                            }

                            let candidate = unsafe {
                                std::slice::from_raw_parts(
                                    (mmap_ptr + offset) as *const f64,
                                    dim,
                                )
                            };

                            self.engine
                                .cpu_backend
                                .compute_f64(&query_vec, candidate, metric_type)
                                .unwrap_or(f64::NAN)
                        })
                        .collect()
                };

                let (new_indices, new_scores) = Self::merge_top_k_impl(
                    &batch_scores,
                    batch_start as u64,
                    &current_indices,
                    &current_scores,
                    k,
                    is_similarity,
                );

                current_indices = new_indices;
                current_scores = new_scores;
            }

            Ok::<_, String>((current_indices, current_scores))
        })
        .map_err(|e| PyValueError::new_err(e))?;

        Ok((
            PyArray1::from_vec(py, indices).into(),
            PyArray1::from_vec(py, scores).into(),
        ))
    }

    /// Streaming batch compute from file (f32)
    fn streaming_batch_compute_f32(
        &self,
        py: Python,
        query: &Bound<'_, PyAny>,
        npk_dir: &str,
        array_name: &str,
        metric_type: MetricType,
        batch_size: usize,
    ) -> PyResult<Py<PyArray1<f64>>> {
        // Extract query
        let query_arr: PyReadonlyArrayDyn<f32> = query.extract()?;
        let query_slice = query_arr.as_slice()?;
        let query_vec: Vec<f32> = query_slice.to_vec();
        let dim = query_vec.len();

        // Load metadata
        let base_dir = Path::new(npk_dir);
        let metadata_path = base_dir.join("metadata.npkm");
        let metadata_store = BinaryMetadataStore::load(&metadata_path)
            .map_err(|e| PyValueError::new_err(format!("Failed to load metadata: {}", e)))?;

        let array_meta = metadata_store.get_array(array_name).ok_or_else(|| {
            PyValueError::new_err(format!("Array '{}' not found in NumPack", array_name))
        })?;

        if array_meta.dtype != BinaryDataType::Float32 {
            return Err(PyTypeError::new_err(format!(
                "Array dtype mismatch: query is float32 but array is {:?}",
                array_meta.dtype
            )));
        }

        let array_dim = if array_meta.shape.len() > 1 {
            array_meta.shape[1] as usize
        } else {
            1
        };
        if array_dim != dim {
            return Err(PyValueError::new_err(format!(
                "Dimension mismatch: query dim {} vs array dim {}",
                dim, array_dim
            )));
        }

        let total_rows = array_meta.shape[0] as usize;
        let data_path = base_dir.join(format!("data_{}.npkd", array_name));

        // Release GIL
        let scores = py.allow_threads(|| {
            let file = std::fs::File::open(&data_path)
                .map_err(|e| format!("Failed to open data file: {}", e))?;
            let mmap = unsafe {
                MmapOptions::new()
                    .map(&file)
                    .map_err(|e| format!("Failed to mmap: {}", e))?
            };

            // Advise kernel for sequential access
            #[cfg(unix)]
            {
                unsafe {
                    libc::madvise(
                        mmap.as_ptr() as *mut libc::c_void,
                        mmap.len(),
                        libc::MADV_SEQUENTIAL,
                    );
                }
            }

            let row_bytes = dim * std::mem::size_of::<f32>();
            let mmap_ptr = mmap.as_ptr() as usize;
            let mmap_len = mmap.len();

            // Single-pass parallel processing for batch compute (returns all scores)
            #[cfg(feature = "rayon")]
            let all_scores: Vec<f64> = {
                use rayon::prelude::*;
                
                (0..total_rows)
                    .into_par_iter()
                    .map(|row_idx| {
                        let offset = row_idx * row_bytes;

                        if offset + row_bytes > mmap_len {
                            return f64::NAN;
                        }

                        let candidate = unsafe {
                            std::slice::from_raw_parts(
                                (mmap_ptr + offset) as *const f32,
                                dim,
                            )
                        };

                        self.engine
                            .cpu_backend
                            .compute_f32(&query_vec, candidate, metric_type)
                            .unwrap_or(f32::NAN) as f64
                    })
                    .collect()
            };

            #[cfg(not(feature = "rayon"))]
            let all_scores: Vec<f64> = {
                (0..total_rows)
                    .map(|row_idx| {
                        let offset = row_idx * row_bytes;

                        if offset + row_bytes > mmap_len {
                            return f64::NAN;
                        }

                        let candidate = unsafe {
                            std::slice::from_raw_parts(
                                (mmap_ptr + offset) as *const f32,
                                dim,
                            )
                        };

                        self.engine
                            .cpu_backend
                            .compute_f32(&query_vec, candidate, metric_type)
                            .unwrap_or(f32::NAN) as f64
                    })
                    .collect()
            };

            Ok::<_, String>(all_scores)
        })
        .map_err(|e| PyValueError::new_err(e))?;

        Ok(PyArray1::from_vec(py, scores).into())
    }

    /// Streaming batch compute from file (f64)
    fn streaming_batch_compute_f64(
        &self,
        py: Python,
        query: &Bound<'_, PyAny>,
        npk_dir: &str,
        array_name: &str,
        metric_type: MetricType,
        batch_size: usize,
    ) -> PyResult<Py<PyArray1<f64>>> {
        // Extract query
        let query_arr: PyReadonlyArrayDyn<f64> = query.extract()?;
        let query_slice = query_arr.as_slice()?;
        let query_vec: Vec<f64> = query_slice.to_vec();
        let dim = query_vec.len();

        // Load metadata
        let base_dir = Path::new(npk_dir);
        let metadata_path = base_dir.join("metadata.npkm");
        let metadata_store = BinaryMetadataStore::load(&metadata_path)
            .map_err(|e| PyValueError::new_err(format!("Failed to load metadata: {}", e)))?;

        let array_meta = metadata_store.get_array(array_name).ok_or_else(|| {
            PyValueError::new_err(format!("Array '{}' not found in NumPack", array_name))
        })?;

        if array_meta.dtype != BinaryDataType::Float64 {
            return Err(PyTypeError::new_err(format!(
                "Array dtype mismatch: query is float64 but array is {:?}",
                array_meta.dtype
            )));
        }

        let array_dim = if array_meta.shape.len() > 1 {
            array_meta.shape[1] as usize
        } else {
            1
        };
        if array_dim != dim {
            return Err(PyValueError::new_err(format!(
                "Dimension mismatch: query dim {} vs array dim {}",
                dim, array_dim
            )));
        }

        let total_rows = array_meta.shape[0] as usize;
        let data_path = base_dir.join(format!("data_{}.npkd", array_name));

        // Release GIL
        let scores = py.allow_threads(|| {
            let file = std::fs::File::open(&data_path)
                .map_err(|e| format!("Failed to open data file: {}", e))?;
            let mmap = unsafe {
                MmapOptions::new()
                    .map(&file)
                    .map_err(|e| format!("Failed to mmap: {}", e))?
            };

            // Advise kernel for sequential access
            #[cfg(unix)]
            {
                unsafe {
                    libc::madvise(
                        mmap.as_ptr() as *mut libc::c_void,
                        mmap.len(),
                        libc::MADV_SEQUENTIAL,
                    );
                }
            }

            let row_bytes = dim * std::mem::size_of::<f64>();
            let mmap_ptr = mmap.as_ptr() as usize;
            let mmap_len = mmap.len();

            // Single-pass parallel processing
            #[cfg(feature = "rayon")]
            let all_scores: Vec<f64> = {
                use rayon::prelude::*;
                
                (0..total_rows)
                    .into_par_iter()
                    .map(|row_idx| {
                        let offset = row_idx * row_bytes;

                        if offset + row_bytes > mmap_len {
                            return f64::NAN;
                        }

                        let candidate = unsafe {
                            std::slice::from_raw_parts(
                                (mmap_ptr + offset) as *const f64,
                                dim,
                            )
                        };

                        self.engine
                            .cpu_backend
                            .compute_f64(&query_vec, candidate, metric_type)
                            .unwrap_or(f64::NAN)
                    })
                    .collect()
            };

            #[cfg(not(feature = "rayon"))]
            let all_scores: Vec<f64> = {
                (0..total_rows)
                    .map(|row_idx| {
                        let offset = row_idx * row_bytes;

                        if offset + row_bytes > mmap_len {
                            return f64::NAN;
                        }

                        let candidate = unsafe {
                            std::slice::from_raw_parts(
                                (mmap_ptr + offset) as *const f64,
                                dim,
                            )
                        };

                        self.engine
                            .cpu_backend
                            .compute_f64(&query_vec, candidate, metric_type)
                            .unwrap_or(f64::NAN)
                    })
                    .collect()
            };

            Ok::<_, String>(all_scores)
        })
        .map_err(|e| PyValueError::new_err(e))?;

        Ok(PyArray1::from_vec(py, scores).into())
    }

    /// Merge batch scores with current top-k (efficient Rust implementation)
    ///
    /// This replaces the Python heapq loop with a single efficient Rust operation.
    /// Algorithm:
    /// 1. Combine current top-k with batch scores
    /// 2. Use partial sort (select_nth_unstable) to find new top-k
    /// 3. Sort the final top-k results
    fn merge_top_k_impl(
        batch_scores: &[f64],
        global_offset: u64,
        current_indices: &[u64],
        current_scores: &[f64],
        k: usize,
        is_similarity: bool,
    ) -> (Vec<u64>, Vec<f64>) {
        // Early return for empty batch
        if batch_scores.is_empty() {
            return (current_indices.to_vec(), current_scores.to_vec());
        }

        // Calculate total candidates
        let current_count = current_indices.len();
        let batch_count = batch_scores.len();
        let total = current_count + batch_count;
        let k = k.min(total);

        if k == 0 {
            return (Vec::new(), Vec::new());
        }

        // Combine all candidates: (global_index, score)
        let mut all_candidates: Vec<(u64, f64)> = Vec::with_capacity(total);

        // Add current top-k
        for i in 0..current_count {
            all_candidates.push((current_indices[i], current_scores[i]));
        }

        // Add batch scores with global indices
        for (local_idx, &score) in batch_scores.iter().enumerate() {
            let global_idx = global_offset + local_idx as u64;
            all_candidates.push((global_idx, score));
        }

        // Use partial sort to find top-k efficiently
        if is_similarity {
            // Higher is better - descending order
            if all_candidates.len() > k {
                all_candidates.select_nth_unstable_by(k - 1, |a, b| {
                    b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                });
                all_candidates.truncate(k);
            }
            // Sort top-k
            all_candidates.sort_unstable_by(|a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
        } else {
            // Lower is better - ascending order
            if all_candidates.len() > k {
                all_candidates.select_nth_unstable_by(k - 1, |a, b| {
                    a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
                });
                all_candidates.truncate(k);
            }
            // Sort top-k
            all_candidates.sort_unstable_by(|a, b| {
                a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        // Extract indices and scores
        let new_indices: Vec<u64> = all_candidates.iter().map(|(idx, _)| *idx).collect();
        let new_scores: Vec<f64> = all_candidates.iter().map(|(_, score)| *score).collect();

        (new_indices, new_scores)
    }
}

/// Register vector engine module to Python
pub fn register_vector_engine_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    // Create vector_engine submodule
    let vector_engine_module = PyModule::new(parent_module.py(), "vector_engine")?;
    
    // Register classes in submodule
    vector_engine_module.add_class::<PyVectorSearch>()?;
    vector_engine_module.add_class::<PyStreamingVectorSearch>()?;
    
    // Also register classes directly in parent module for backward compatibility
    parent_module.add_class::<PyVectorSearch>()?;
    parent_module.add_class::<PyStreamingVectorSearch>()?;
    
    // Add submodule to parent
    parent_module.add_submodule(&vector_engine_module)?;
    
    Ok(())
}