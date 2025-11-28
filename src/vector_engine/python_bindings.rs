//! Python FFI 绑定
//!
//! 将向量引擎的功能暴露给 Python

use numpy::{PyArray1, PyArrayDyn, PyArrayMethods, PyReadonlyArrayDyn};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

use crate::vector_engine::core::VectorEngine;
use crate::vector_engine::metrics::MetricType;

/// Python 侧的向量引擎包装
#[pyclass(module = "numpack", name = "VectorEngine")]
pub struct PyVectorEngine {
    engine: VectorEngine,
}

#[pymethods]
impl PyVectorEngine {
    /// 创建新的向量引擎实例
    #[new]
    pub fn new() -> Self {
        Self {
            engine: VectorEngine::new(),
        }
    }

    /// 获取 SIMD 能力信息
    pub fn capabilities(&self) -> String {
        self.engine.capabilities()
    }

    /// 计算两个向量的度量值
    ///
    /// Args:
    ///     a: 第一个向量 (numpy array)
    ///     b: 第二个向量 (numpy array)
    ///     metric: 度量类型字符串 ('dot', 'cosine', 'l2', etc.)
    ///
    /// Returns:
    ///     度量值 (float)
    #[pyo3(signature = (a, b, metric))]
    pub fn compute_metric(
        &self,
        py: Python,
        a: PyReadonlyArrayDyn<f64>,
        b: PyReadonlyArrayDyn<f64>,
        metric: &str,
    ) -> PyResult<f64> {
        // 解析度量类型
        let metric_type = MetricType::from_str(metric)
            .ok_or_else(|| PyValueError::new_err(format!("Unknown metric: {}", metric)))?;

        // 提取数据（零拷贝）
        let a_slice = a.as_slice()?;
        let b_slice = b.as_slice()?;

        // 计算
        self.engine
            .compute_metric(a_slice, b_slice, metric_type)
            .map_err(|e| PyValueError::new_err(format!("Compute error: {}", e)))
    }

    /// 批量计算：query 向量与多个候选向量的度量
    ///
    /// 支持多种数据类型，自动根据输入 dtype 选择最优计算路径：
    /// - i8: 整数向量（dot, cosine, l2, l2sq）
    /// - f16: 半精度浮点（所有度量）
    /// - f32: 单精度浮点（所有度量）
    /// - f64: 双精度浮点（所有度量）
    /// - u8: 二进制向量（hamming, jaccard）
    ///
    /// Args:
    ///     query: 查询向量 (1D numpy array, any supported dtype)
    ///     candidates: 候选向量矩阵 (2D numpy array, shape: [N, D], same dtype as query)
    ///     metric: 度量类型字符串
    ///
    /// Returns:
    ///     度量值数组 (1D numpy array, shape: [N], always f64)
    #[pyo3(signature = (query, candidates, metric))]
    pub fn batch_compute(
        &self,
        py: Python,
        query: &Bound<'_, PyAny>,
        candidates: &Bound<'_, PyAny>,
        metric: &str,
    ) -> PyResult<Py<PyArray1<f64>>> {
        // 解析度量类型
        let metric_type = MetricType::from_str(metric)
            .ok_or_else(|| PyValueError::new_err(format!("Unknown metric: {}", metric)))?;

        // 获取数组的 dtype
        let query_dtype = query.getattr("dtype")?.str()?.to_string();
        let candidates_dtype = candidates.getattr("dtype")?.str()?.to_string();

        // 确保两个数组类型一致
        if query_dtype != candidates_dtype {
            return Err(PyTypeError::new_err(format!(
                "Query dtype ({}) must match candidates dtype ({})",
                query_dtype, candidates_dtype
            )));
        }

        // 根据 dtype 分派到不同的计算路径
        // 这样可以避免不必要的类型转换，直接使用 SimSIMD 的原生支持
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

    /// Top-K 搜索：找到最相似/最近的 k 个向量
    ///
    /// 支持多种数据类型（自动识别 dtype）：
    /// - i8, f32, f64, u8（与 batch_compute 相同）
    ///
    /// Args:
    ///     query: 查询向量 (1D numpy array, any supported dtype)
    ///     candidates: 候选向量矩阵 (2D numpy array, same dtype as query)
    ///     metric: 度量类型字符串
    ///     k: 返回的结果数量
    ///
    /// Returns:
    ///     (indices, scores):
    ///         - indices: 索引数组 (shape: [k])
    ///         - scores: 分数数组 (shape: [k])
    ///         
    ///     对于相似度度量（dot, cosine），返回最高的 k 个
    ///     对于距离度量（l2, l2sq, hamming, jaccard, kl, js），返回最低的 k 个
    #[pyo3(signature = (query, candidates, metric, k))]
    pub fn top_k_search(
        &self,
        py: Python,
        query: &Bound<'_, PyAny>,
        candidates: &Bound<'_, PyAny>,
        metric: &str,
        k: usize,
    ) -> PyResult<(Py<PyArray1<usize>>, Py<PyArray1<f64>>)> {
        // 解析度量类型
        let metric_type = MetricType::from_str(metric)
            .ok_or_else(|| PyValueError::new_err(format!("Unknown metric: {}", metric)))?;

        // 获取数组的 dtype
        let query_dtype = query.getattr("dtype")?.str()?.to_string();
        let candidates_dtype = candidates.getattr("dtype")?.str()?.to_string();

        // 确保两个数组类型一致
        if query_dtype != candidates_dtype {
            return Err(PyTypeError::new_err(format!(
                "Query dtype ({}) must match candidates dtype ({})",
                query_dtype, candidates_dtype
            )));
        }

        // 根据 dtype 分派
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
// 类型特化实现：为每种数据类型提供零拷贝的计算路径
// 这些是私有辅助方法，不暴露给 Python
// ========================================================================

impl PyVectorEngine {
    /// f64 批量计算（双精度浮点）
    ///
    /// 🚀 优化：减少 FFI 开销，直接传递连续内存
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
        let candidates_array = candidates_arr.as_array();
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

        // 🚀 关键优化：使用 usize 传递地址（可以跨线程）
        let query_addr = query_slice.as_ptr() as usize;
        let candidates_addr = candidates_slice.as_ptr() as usize;

        // 释放 GIL 执行并行计算
        let scores = py
            .allow_threads(|| {
                // 🚀 智能批处理策略：小批量串行，大批量并行
                // 避免小批量时 Rayon 线程池的开销
                const PARALLEL_THRESHOLD: usize = 500;

                if n_candidates < PARALLEL_THRESHOLD {
                    // 串行：避免线程池开销
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
                    // 并行：大批量使用多核
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

    /// f32 批量计算（单精度浮点）
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
        let candidates_array = candidates_arr.as_array();
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

        // 优化：使用 usize 传递地址
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

        // 转换 f32 结果为 f64（统一输出类型）
        let scores_f64: Vec<f64> = scores.into_iter().map(|x| x as f64).collect();
        Ok(PyArray1::from_vec(py, scores_f64).into())
    }

    /// f16 批量计算（半精度浮点）
    fn batch_compute_f16(
        &self,
        py: Python,
        _query: &Bound<'_, PyAny>,
        _candidates: &Bound<'_, PyAny>,
        _metric_type: MetricType,
    ) -> PyResult<Py<PyArray1<f64>>> {
        // TODO: 实现 f16 支持（需要 half crate 集成）
        Err(PyTypeError::new_err(
            "float16 support not yet implemented. Please use float32 or float64.",
        ))
    }

    /// i8 批量计算（整数向量）
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
        let candidates_array = candidates_arr.as_array();
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

        // 🚀 优化：使用 usize 传递地址
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

    /// u8 批量计算（二进制向量 - hamming/jaccard）
    fn batch_compute_u8(
        &self,
        py: Python,
        query: &Bound<'_, PyAny>,
        candidates: &Bound<'_, PyAny>,
        metric_type: MetricType,
    ) -> PyResult<Py<PyArray1<f64>>> {
        use numpy::PyArrayMethods;

        // u8 只支持 Hamming 和 Jaccard
        if !matches!(metric_type, MetricType::Hamming | MetricType::Jaccard) {
            return Err(PyValueError::new_err(format!(
                "uint8 arrays only support 'hamming' and 'jaccard' metrics, got: {}",
                metric_type.as_str()
            )));
        }

        let query_arr: PyReadonlyArrayDyn<u8> = query.extract()?;
        let candidates_arr: PyReadonlyArrayDyn<u8> = candidates.extract()?;

        let query_slice = query_arr.as_slice()?;
        let candidates_array = candidates_arr.as_array();
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

        // 🚀 优化：使用 usize 传递地址
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
    // Top-K 搜索实现：为每种数据类型提供优化的 Top-K 搜索
    // ========================================================================

    /// Top-K 搜索 (f64)
    fn top_k_search_f64(
        &self,
        py: Python,
        query: &Bound<'_, PyAny>,
        candidates: &Bound<'_, PyAny>,
        metric_type: MetricType,
        k: usize,
    ) -> PyResult<(Py<PyArray1<usize>>, Py<PyArray1<f64>>)> {
        // 先计算所有分数
        let scores_array = self.batch_compute_f64(py, query, candidates, metric_type)?;

        // 提取分数
        let scores = scores_array.bind(py).readonly();
        let scores_slice = scores.as_slice()?;

        // Top-K 选择
        let (indices, top_scores) =
            Self::select_top_k(scores_slice, k, metric_type.is_similarity());

        Ok((
            PyArray1::from_vec(py, indices).into(),
            PyArray1::from_vec(py, top_scores).into(),
        ))
    }

    /// Top-K 搜索 (f32)
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

    /// Top-K 搜索 (i8)
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

    /// Top-K 搜索 (u8)
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
        // u8 的度量都是距离（越小越好）
        let (indices, top_scores) = Self::select_top_k(scores_slice, k, false);

        Ok((
            PyArray1::from_vec(py, indices).into(),
            PyArray1::from_vec(py, top_scores).into(),
        ))
    }

    /// 从分数数组中选择 Top-K
    ///
    /// Args:
    ///     scores: 分数数组
    ///     k: 返回数量
    ///     is_similarity: true = 越大越好（相似度），false = 越小越好（距离）
    ///
    /// Returns:
    ///     (indices, top_scores): 索引和对应的分数
    fn select_top_k(scores: &[f64], k: usize, is_similarity: bool) -> (Vec<usize>, Vec<f64>) {
        let n = scores.len();
        let k = k.min(n); // k 不能超过总数

        // 创建 (index, score) 对
        let mut indexed_scores: Vec<(usize, f64)> = scores
            .iter()
            .enumerate()
            .map(|(i, &score)| (i, score))
            .collect();

        // 部分排序：只排序前 k 个
        // 相似度：降序（大到小），距离：升序（小到大）
        if is_similarity {
            // 使用 select_nth_unstable 进行 O(n) 部分排序
            indexed_scores.select_nth_unstable_by(k - 1, |a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            // 对前 k 个再排序
            indexed_scores[..k].sort_unstable_by(|a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
        } else {
            // 距离：升序
            indexed_scores.select_nth_unstable_by(k - 1, |a, b| {
                a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            indexed_scores[..k].sort_unstable_by(|a, b| {
                a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        // 提取前 k 个的索引和分数
        let indices: Vec<usize> = indexed_scores[..k].iter().map(|(i, _)| *i).collect();
        let top_scores: Vec<f64> = indexed_scores[..k].iter().map(|(_, s)| *s).collect();

        (indices, top_scores)
    }
}

/// 注册向量引擎模块到 Python
pub fn register_vector_engine_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    // 直接在父模块中注册类
    parent_module.add_class::<PyVectorEngine>()?;
    Ok(())
}
