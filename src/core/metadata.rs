// NumPack元数据定义
//
// 本文件仅保留核心数据类型定义
// 实际元数据存储使用 binary_metadata.rs 中的 BinaryMetadataStore

use serde::{Deserialize, Serialize};
use serde_bytes::ByteBuf;

/// NumPack数据类型，对应NumPy的数据类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum DataType {
    Bool = 0,
    Uint8 = 1,
    Uint16 = 2,
    Uint32 = 3,
    Uint64 = 4,
    Int8 = 5,
    Int16 = 6,
    Int32 = 7,
    Int64 = 8,
    Float16 = 9,
    Float32 = 10,
    Float64 = 11,
    Complex64 = 12,
    Complex128 = 13,
}

impl DataType {
    pub fn size_bytes(&self) -> usize {
        match self {
            DataType::Bool => 1,
            DataType::Uint8 => 1,
            DataType::Uint16 => 2,
            DataType::Uint32 => 4,
            DataType::Uint64 => 8,
            DataType::Int8 => 1,
            DataType::Int16 => 2,
            DataType::Int32 => 4,
            DataType::Int64 => 8,
            DataType::Float16 => 2,
            DataType::Float32 => 4,
            DataType::Float64 => 8,
            DataType::Complex64 => 8,   // 2 * float32
            DataType::Complex128 => 16, // 2 * float64
        }
    }
}

/// 数组元数据（用于内部类型转换）
///
/// 注意：这个类型主要用于 parallel_io.rs 中的内部转换
/// 实际元数据存储使用 BinaryArrayMetadata
#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArrayMetadata {
    pub name: String,
    pub shape: Vec<u64>,    // Data shape
    pub data_file: String,  // Data file name
    pub last_modified: u64, // Last modified time in microseconds
    pub size_bytes: u64,    // Data size
    pub dtype: u8,          // Data type as u8 to match Python
    #[serde(skip)]
    pub raw_data: Option<ByteBuf>, // For zero-copy serialization
}

impl ArrayMetadata {
    pub fn new(name: String, shape: Vec<u64>, data_file: String, dtype: DataType) -> Self {
        let total_elements: u64 = shape.iter().product();
        Self {
            name,
            shape,
            data_file,
            last_modified: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_micros() as u64, // Use microseconds to match Python
            size_bytes: total_elements * dtype.size_bytes() as u64,
            dtype: dtype as u8, // Convert DataType enum to u8
            raw_data: None,
        }
    }

    // Helper method to get DataType from u8
    pub fn get_dtype(&self) -> DataType {
        match self.dtype {
            0 => DataType::Bool,
            1 => DataType::Uint8,
            2 => DataType::Uint16,
            3 => DataType::Uint32,
            4 => DataType::Uint64,
            5 => DataType::Int8,
            6 => DataType::Int16,
            7 => DataType::Int32,
            8 => DataType::Int64,
            9 => DataType::Float16,
            10 => DataType::Float32,
            11 => DataType::Float64,
            12 => DataType::Complex64,
            13 => DataType::Complex128,
            _ => DataType::Int32, // Default fallback
        }
    }

    pub fn total_elements(&self) -> u64 {
        self.shape.iter().product()
    }

    pub fn _ndim(&self) -> usize {
        self.shape.len()
    }
}
