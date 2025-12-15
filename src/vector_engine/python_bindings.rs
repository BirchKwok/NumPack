//! Python FFI bindings
//!
//! Exposes vector engine functionality to Python

use numpy::{PyArray1, PyArrayMethods, PyReadonlyArrayDyn};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

use crate::vector_engine::core::VectorEngine;
use crate::vector_engine::metrics::MetricType;

/// Python wrapper for the vector engine
///
/// This is a Python binding for the SimSIMD Rust library, providing high-performance
/// vector similarity computation with SIMD acceleration (AVX2, AVX-512, NEON, SVE).
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
#[pyclass(module = "numpack", name = "VectorEngine")]
pub struct PyVectorEngine {
    engine: VectorEngine,
}

#[pymethods]
impl PyVectorEngine {
    /// Create a new vector engine instance
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

        // Ensure both arrays have the same dtype
        if a_dtype != b_dtype {
            return Err(PyTypeError::new_err(format!(
                "Array a dtype ({}) must match array b dtype ({})",
                a_dtype, b_dtype
            )));
        }

        // Dispatch to different computation paths based on dtype
        match a_dtype.as_str() {
            "float64" => self.compute_metric_f64(py, a, b, metric_type),
            "float32" => self.compute_metric_f32(py, a, b, metric_type),
            "int8" => self.compute_metric_i8(py, a, b, metric_type),
            "uint8" => self.compute_metric_u8(py, a, b, metric_type),
            _ => Err(PyTypeError::new_err(format!(
                "Unsupported dtype: {}. Supported: float64, float32, int8, uint8",
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

        // Ensure both arrays have the same dtype
        if query_dtype != candidates_dtype {
            return Err(PyTypeError::new_err(format!(
                "Query dtype ({}) must match candidates dtype ({})",
                query_dtype, candidates_dtype
            )));
        }

        // Dispatch to different computation paths based on dtype
        // This avoids unnecessary type conversions and directly uses SimSIMD's native support
        match query_dtype.as_str() {
            "float64" => self.batch_compute_f64(py, query, candidates, metric_type),
            "float32" => self.batch_compute_f32(py, query, candidates, metric_type),
            "float16" => self.batch_compute_f16(py, query, candidates, metric_type),
            "int8" => self.batch_compute_i8(py, query, candidates, metric_type),
            "uint8" => self.batch_compute_u8(py, query, candidates, metric_type),
            _ => Err(PyTypeError::new_err(format!(
                "Unsupported dtype: {}. Supported: float64, float32, float16, int8, uint8",
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

        // Ensure both arrays have the same dtype
        if query_dtype != candidates_dtype {
            return Err(PyTypeError::new_err(format!(
                "Query dtype ({}) must match candidates dtype ({})",
                query_dtype, candidates_dtype
            )));
        }

        // Dispatch based on dtype
        match query_dtype.as_str() {
            "float64" => self.top_k_search_f64(py, query, candidates, metric_type, k),
            "float32" => self.top_k_search_f32(py, query, candidates, metric_type, k),
            "int8" => self.top_k_search_i8(py, query, candidates, metric_type, k),
            "uint8" => self.top_k_search_u8(py, query, candidates, metric_type, k),
            _ => Err(PyTypeError::new_err(format!(
                "Unsupported dtype: {}. Supported: float64, float32, int8, uint8",
                query_dtype
            ))),
        }
    }
}

// ========================================================================
// Type-specialized implementations: Zero-copy computation paths for each data type
// These are private helper methods, not exposed to Python
// ========================================================================

impl PyVectorEngine {
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
    /// 🚀 Optimization: Reduces FFI overhead by directly passing contiguous memory
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

        // 🚀 Key optimization: Use usize to pass addresses (can cross threads)
        let query_addr = query_slice.as_ptr() as usize;
        let candidates_addr = candidates_slice.as_ptr() as usize;

        // Release GIL for parallel computation
        let scores = py
            .allow_threads(|| {
                // 🚀 Smart batching strategy: serial for small batches, parallel for large batches
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

        // 🚀 Optimization: use usize to pass addresses
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

        // 🚀 Optimization: use usize to pass addresses
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
}

/// Register vector engine module to Python
pub fn register_vector_engine_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    // Register class directly in parent module
    parent_module.add_class::<PyVectorEngine>()?;
    Ok(())
}
