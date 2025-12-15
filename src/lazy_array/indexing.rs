//! LazyArray索引处理模块
//!
//! 从lib.rs中提取的索引解析和处理逻辑

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyList, PySlice, PyTuple};

/// 索引类型枚举
#[derive(Debug, Clone)]
pub enum IndexType {
    Integer(i64),
    Slice(SliceInfo),
    BooleanMask(Vec<bool>),
    IntegerArray(Vec<i64>),
    Ellipsis,
    NewAxis,
}

/// 切片信息结构体
#[derive(Debug, Clone)]
pub struct SliceInfo {
    pub start: Option<i64>,
    pub stop: Option<i64>,
    pub step: Option<i64>,
}

/// 索引解析结果
#[derive(Debug, Clone)]
pub struct IndexResult {
    pub indices: Vec<Vec<usize>>, // 每个维度的索引
    pub result_shape: Vec<usize>,
    pub needs_broadcasting: bool,
    pub access_pattern: AccessPattern,
}

/// 访问模式枚举
#[derive(Debug, Clone)]
pub enum AccessPattern {
    Sequential,
    Random,
    Clustered,
    Mixed,
}

/// 访问策略枚举
#[derive(Debug, Clone)]
pub enum AccessStrategy {
    DirectMemory,
    BlockCopy,
    ParallelPointAccess,
    PrefetchOptimized,
    Adaptive,
}

impl SliceInfo {
    /// 创建新的SliceInfo
    pub fn new(start: Option<i64>, stop: Option<i64>, step: Option<i64>) -> Self {
        Self { start, stop, step }
    }

    /// 从PySlice对象解析
    pub fn from_pyslice(slice: &Bound<'_, PySlice>, length: usize) -> PyResult<Self> {
        let indices = slice.indices(length as isize)?;
        Ok(Self {
            start: Some(indices.start as i64),
            stop: Some(indices.stop as i64),
            step: Some(indices.step as i64),
        })
    }

    /// 规范化切片参数
    pub fn normalize(&self, length: usize) -> (usize, usize, usize) {
        let len = length as i64;

        let step = self.step.unwrap_or(1);
        if step == 0 {
            panic!("slice step cannot be zero");
        }

        let (start, stop) = if step > 0 {
            let start = self.start.unwrap_or(0);
            let stop = self.stop.unwrap_or(len);

            let norm_start = if start < 0 {
                (len + start).max(0) as usize
            } else {
                (start.min(len)) as usize
            };

            let norm_stop = if stop < 0 {
                (len + stop).max(0) as usize
            } else {
                (stop.min(len)) as usize
            };

            (norm_start, norm_stop)
        } else {
            let start = self.start.unwrap_or(len - 1);
            let stop = self.stop.unwrap_or(-1);

            let norm_start = if start < 0 {
                (len + start).max(-1) as usize
            } else {
                (start.min(len - 1)) as usize
            };

            let norm_stop = if stop < 0 {
                if stop == -1 {
                    0
                } else {
                    (len + stop).max(-1) as usize
                }
            } else {
                stop as usize
            };

            (norm_start, norm_stop)
        };

        (start, stop, step.abs() as usize)
    }

    /// 计算切片结果的长度
    pub fn result_length(&self, length: usize) -> usize {
        let (start, stop, step) = self.normalize(length);

        if step == 1 {
            if stop > start {
                stop - start
            } else {
                0
            }
        } else if stop > start {
            (stop - start + step - 1) / step
        } else {
            0
        }
    }

    /// 生成切片索引
    pub fn generate_indices(&self, length: usize) -> PyResult<Vec<usize>> {
        resolve_slice(self, length)
    }
}

/// 将SliceInfo解析为实际的索引序列
#[inline]
pub fn resolve_slice(slice_info: &SliceInfo, dim_size: usize) -> PyResult<Vec<usize>> {
    let start = slice_info.start.unwrap_or(0);
    let stop = slice_info.stop.unwrap_or(dim_size as i64);
    let step = slice_info.step.unwrap_or(1);

    if step == 0 {
        return Err(PyErr::new::<PyValueError, _>("Slice step cannot be zero"));
    }

    let mut indices = Vec::new();

    if step > 0 {
        let mut i = if start < 0 {
            dim_size as i64 + start
        } else {
            start
        };
        let end = if stop < 0 {
            dim_size as i64 + stop
        } else {
            stop
        };

        while i < end && i < dim_size as i64 {
            if i >= 0 {
                indices.push(i as usize);
            }
            i += step;
        }
    } else {
        let mut i = if start < 0 {
            dim_size as i64 + start
        } else {
            start.min(dim_size as i64 - 1)
        };
        let end = if stop < 0 {
            dim_size as i64 + stop
        } else {
            stop
        };

        while i > end && i >= 0 {
            if i < dim_size as i64 {
                indices.push(i as usize);
            }
            i += step;
        }
    }

    Ok(indices)
}

impl IndexResult {
    /// 创建新的IndexResult
    pub fn new(
        indices: Vec<Vec<usize>>,
        result_shape: Vec<usize>,
        needs_broadcasting: bool,
        access_pattern: AccessPattern,
    ) -> Self {
        Self {
            indices,
            result_shape,
            needs_broadcasting,
            access_pattern,
        }
    }

    /// 是否为单个元素索引
    pub fn is_single_element(&self) -> bool {
        self.result_shape.iter().all(|&x| x == 1) || self.result_shape.is_empty()
    }

    /// 是否为连续索引
    pub fn is_contiguous(&self) -> bool {
        matches!(self.access_pattern, AccessPattern::Sequential)
    }

    /// 计算总元素数量
    pub fn total_elements(&self) -> usize {
        if self.result_shape.is_empty() {
            1
        } else {
            self.result_shape.iter().product()
        }
    }
}

/// 索引解析器
pub struct IndexParser;

impl IndexParser {
    /// 解析单个索引
    pub fn parse_single_index(key: &Bound<'_, PyAny>) -> PyResult<IndexType> {
        // 处理整数索引
        if let Ok(idx) = key.extract::<i64>() {
            return Ok(IndexType::Integer(idx));
        }

        // 处理切片索引
        if let Ok(slice) = key.downcast::<PySlice>() {
            let start = slice.getattr("start")?.extract::<Option<i64>>()?;
            let stop = slice.getattr("stop")?.extract::<Option<i64>>()?;
            let step = slice.getattr("step")?.extract::<Option<i64>>()?;
            return Ok(IndexType::Slice(SliceInfo::new(start, stop, step)));
        }

        // 处理省略号
        if key.is_instance_of::<pyo3::types::PyEllipsis>() {
            return Ok(IndexType::Ellipsis);
        }

        // 处理numpy.newaxis (None类型)
        if key.is_none() {
            return Ok(IndexType::NewAxis);
        }

        // 处理列表/数组索引
        if let Ok(list) = key.downcast::<PyList>() {
            if list.is_empty() {
                return Ok(IndexType::IntegerArray(Vec::new()));
            }

            // 检查第一个元素类型
            let first_item = list.get_item(0)?;
            if first_item.extract::<bool>().is_ok() {
                // 布尔掩码
                let mask: Vec<bool> = list.extract()?;
                return Ok(IndexType::BooleanMask(mask));
            } else {
                // 整数数组
                let indices: Vec<i64> = list.extract()?;
                return Ok(IndexType::IntegerArray(indices));
            }
        }

        // 处理numpy数组
        if let Ok(_) = key.getattr("__array__") {
            // 尝试获取数组的dtype
            if let Ok(dtype) = key.getattr("dtype") {
                if let Ok(dtype_name) = dtype.getattr("name") {
                    let dtype_str: String = dtype_name.extract()?;
                    if dtype_str == "bool" {
                        // 布尔掩码数组
                        let array = key.call_method0("__array__")?;
                        let mask: Vec<bool> = array.extract()?;
                        return Ok(IndexType::BooleanMask(mask));
                    }
                }
            }

            // 整数数组
            let array = key.call_method0("__array__")?;
            if let Ok(indices) = array.extract::<Vec<i64>>() {
                return Ok(IndexType::IntegerArray(indices));
            }
        }

        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "Unsupported index type",
        ))
    }

    /// 解析多维索引
    pub fn parse_tuple_index(tuple: &Bound<'_, PyTuple>) -> PyResult<Vec<IndexType>> {
        let mut index_types = Vec::new();

        for i in 0..tuple.len() {
            let item = tuple.get_item(i)?;
            index_types.push(Self::parse_single_index(&item)?);
        }

        Ok(index_types)
    }

    /// 处理省略号扩展
    pub fn expand_ellipsis(index_types: Vec<IndexType>, ndim: usize) -> Vec<IndexType> {
        let mut result = Vec::new();
        let mut ellipsis_found = false;

        for index_type in &index_types {
            if matches!(index_type, IndexType::Ellipsis) {
                if ellipsis_found {
                    // 多个省略号，跳过后续的
                    continue;
                }
                ellipsis_found = true;

                // 计算需要插入多少个完整切片
                let explicit_dims = index_types
                    .iter()
                    .filter(|t| !matches!(t, IndexType::Ellipsis | IndexType::NewAxis))
                    .count();
                let missing_dims = ndim.saturating_sub(explicit_dims);

                // 插入完整切片
                for _ in 0..missing_dims {
                    result.push(IndexType::Slice(SliceInfo::new(None, None, None)));
                }
            } else {
                result.push(index_type.clone());
            }
        }

        result
    }

    /// 规范化索引
    pub fn normalize_indices(
        index_types: Vec<IndexType>,
        shape: &[usize],
    ) -> PyResult<IndexResult> {
        let mut indices = Vec::new();
        let mut result_shape = Vec::new();
        let mut access_pattern = AccessPattern::Sequential;

        for (dim, index_type) in index_types.iter().enumerate() {
            if dim >= shape.len() {
                return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                    "Too many indices for array",
                ));
            }

            let dim_size = shape[dim];

            match index_type {
                IndexType::Integer(idx) => {
                    let normalized_idx = if *idx < 0 {
                        (dim_size as i64 + idx) as usize
                    } else {
                        *idx as usize
                    };

                    if normalized_idx >= dim_size {
                        return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                            "Index {} is out of bounds for axis {} with size {}",
                            idx, dim, dim_size
                        )));
                    }

                    indices.push(vec![normalized_idx]);
                    // 整数索引不增加结果维度
                }

                IndexType::Slice(slice_info) => {
                    let slice_indices = slice_info.generate_indices(dim_size)?;
                    let slice_len = slice_indices.len();

                    indices.push(slice_indices);
                    result_shape.push(slice_len);

                    // 非连续切片改变访问模式
                    if slice_info.step.unwrap_or(1) != 1 {
                        access_pattern = AccessPattern::Clustered;
                    }
                }

                IndexType::BooleanMask(mask) => {
                    if mask.len() != dim_size {
                        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                            "Boolean mask length {} doesn't match dimension size {}",
                            mask.len(),
                            dim_size
                        )));
                    }

                    let selected_indices: Vec<usize> = mask
                        .iter()
                        .enumerate()
                        .filter_map(|(i, &selected)| if selected { Some(i) } else { None })
                        .collect();

                    let selected_count = selected_indices.len();
                    indices.push(selected_indices);
                    result_shape.push(selected_count);
                    access_pattern = AccessPattern::Random;
                }

                IndexType::IntegerArray(int_array) => {
                    let mut normalized_indices = Vec::new();

                    for &idx in int_array {
                        let normalized_idx = if idx < 0 {
                            (dim_size as i64 + idx) as usize
                        } else {
                            idx as usize
                        };

                        if normalized_idx >= dim_size {
                            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                                "Index {} is out of bounds for axis {} with size {}",
                                idx, dim, dim_size
                            )));
                        }

                        normalized_indices.push(normalized_idx);
                    }

                    let array_len = normalized_indices.len();
                    indices.push(normalized_indices);
                    result_shape.push(array_len);
                    access_pattern = AccessPattern::Random;
                }

                IndexType::Ellipsis | IndexType::NewAxis => {
                    // 这些应该在之前的处理中被展开或处理
                    continue;
                }
            }
        }

        // 添加剩余维度
        for dim in index_types.len()..shape.len() {
            let full_range: Vec<usize> = (0..shape[dim]).collect();
            indices.push(full_range);
            result_shape.push(shape[dim]);
        }

        Ok(IndexResult::new(
            indices,
            result_shape,
            false,
            access_pattern,
        ))
    }

    /// 检测访问模式
    pub fn detect_access_pattern(indices: &[Vec<usize>]) -> AccessPattern {
        if indices.is_empty() {
            return AccessPattern::Sequential;
        }

        // 检查第一个维度的模式
        let first_dim = &indices[0];

        if first_dim.is_empty() {
            return AccessPattern::Sequential;
        }

        if first_dim.len() == 1 {
            return AccessPattern::Sequential;
        }

        // 检查是否为连续序列
        let mut is_sequential = true;
        for i in 1..first_dim.len() {
            if first_dim[i] != first_dim[i - 1] + 1 {
                is_sequential = false;
                break;
            }
        }

        if is_sequential {
            return AccessPattern::Sequential;
        }

        // 检查是否为聚簇访问（相对局部的）
        let mut gaps = Vec::new();
        for i in 1..first_dim.len() {
            gaps.push(first_dim[i].saturating_sub(first_dim[i - 1]));
        }

        let avg_gap = gaps.iter().sum::<usize>() as f64 / gaps.len() as f64;
        let gap_variance = gaps
            .iter()
            .map(|&gap| (gap as f64 - avg_gap).powi(2))
            .sum::<f64>()
            / gaps.len() as f64;

        if gap_variance < avg_gap {
            AccessPattern::Clustered
        } else {
            AccessPattern::Random
        }
    }
}

/// 策略选择器
pub struct StrategySelector;

impl StrategySelector {
    /// 根据索引结果选择最佳访问策略
    pub fn choose_strategy(index_result: &IndexResult, array_shape: &[usize]) -> AccessStrategy {
        let total_elements = index_result.total_elements();
        let array_size = array_shape.iter().product::<usize>();

        // 单个元素直接访问
        if index_result.is_single_element() {
            return AccessStrategy::DirectMemory;
        }

        // 小数组使用直接内存访问
        if array_size < 10000 {
            return AccessStrategy::DirectMemory;
        }

        // 根据访问模式选择策略
        match index_result.access_pattern {
            AccessPattern::Sequential => {
                if index_result.is_contiguous() && total_elements > 1000 {
                    AccessStrategy::BlockCopy
                } else {
                    AccessStrategy::DirectMemory
                }
            }

            AccessPattern::Clustered => {
                if total_elements > 100 {
                    AccessStrategy::PrefetchOptimized
                } else {
                    AccessStrategy::ParallelPointAccess
                }
            }

            AccessPattern::Random => {
                if total_elements > 1000 {
                    AccessStrategy::Adaptive
                } else {
                    AccessStrategy::ParallelPointAccess
                }
            }

            AccessPattern::Mixed => AccessStrategy::Adaptive,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slice_info_normalize() {
        let slice = SliceInfo::new(Some(1), Some(5), Some(1));
        let (start, stop, step) = slice.normalize(10);
        assert_eq!((start, stop, step), (1, 5, 1));

        let slice = SliceInfo::new(Some(-3), None, Some(1));
        let (start, stop, step) = slice.normalize(10);
        assert_eq!((start, stop, step), (7, 10, 1));
    }

    #[test]
    fn test_slice_info_result_length() {
        let slice = SliceInfo::new(Some(1), Some(5), Some(1));
        assert_eq!(slice.result_length(10), 4);

        let slice = SliceInfo::new(Some(0), Some(10), Some(2));
        assert_eq!(slice.result_length(10), 5);
    }

    #[test]
    fn test_access_pattern_detection() {
        let sequential = vec![vec![0, 1, 2, 3, 4]];
        assert!(matches!(
            IndexParser::detect_access_pattern(&sequential),
            AccessPattern::Sequential
        ));

        let random = vec![vec![0, 5, 2, 8, 1]];
        assert!(matches!(
            IndexParser::detect_access_pattern(&random),
            AccessPattern::Random
        ));
    }
}
