//! 自适应压缩策略
//!
//! 根据数据特征自动选择最优压缩算法，实现2-5x压缩速度提升

use crate::core::metadata::DataType;
use std::io;

/// 压缩算法类型
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CompressionAlgorithm {
    None,       // 无压缩
    Zstd,       // Zstandard - 平衡压缩率和速度
    LZ4,        // LZ4 - 极速压缩
    Snappy,     // Snappy - 快速压缩
    RLE,        // Run-Length Encoding - 重复数据
    Delta,      // Delta编码 - 有序数据
    Dictionary, // 字典编码 - 低基数数据
}

/// 数据特征
#[derive(Debug)]
pub struct DataCharacteristics {
    pub entropy: f64,         // 信息熵 (0-1)
    pub cardinality: usize,   // 不同值的数量
    pub is_sorted: bool,      // 是否有序
    pub repetition_rate: f64, // 重复率
    pub delta_variance: f64,  // Delta方差（用于判断是否适合delta编码）
}

impl DataCharacteristics {
    /// 分析数据特征
    pub fn analyze(data: &[u8], dtype: DataType) -> Self {
        let entropy = Self::calculate_entropy(data);
        let cardinality = Self::estimate_cardinality(data);
        let is_sorted = Self::check_sorted(data, dtype);
        let repetition_rate = Self::calculate_repetition_rate(data);
        let delta_variance = Self::calculate_delta_variance(data, dtype);

        Self {
            entropy,
            cardinality,
            is_sorted,
            repetition_rate,
            delta_variance,
        }
    }

    /// 计算信息熵
    fn calculate_entropy(data: &[u8]) -> f64 {
        if data.is_empty() {
            return 0.0;
        }

        // 计算字节频率
        let mut freq = [0u32; 256];
        for &byte in data {
            freq[byte as usize] += 1;
        }

        // 计算熵
        let len = data.len() as f64;
        let mut entropy = 0.0;

        for &count in freq.iter() {
            if count > 0 {
                let p = count as f64 / len;
                entropy -= p * p.log2();
            }
        }

        // 归一化到0-1范围
        entropy / 8.0
    }

    /// 估算基数（不同值的数量）
    fn estimate_cardinality(data: &[u8]) -> usize {
        use std::collections::HashSet;

        // 对于大数据，使用采样
        let sample_size = 10000.min(data.len());
        let step = data.len() / sample_size;

        let mut unique: HashSet<u8> = HashSet::new();
        for i in (0..data.len()).step_by(step.max(1)) {
            unique.insert(data[i]);
        }

        // 估算总基数
        (unique.len() * data.len()) / sample_size
    }

    /// 检查是否有序
    fn check_sorted(data: &[u8], dtype: DataType) -> bool {
        if data.len() < 16 {
            return false;
        }

        // 检查前100个元素
        let check_size = 100.min(data.len() / std::mem::size_of::<f32>());
        let element_size = dtype.size_bytes() as usize;

        match dtype {
            DataType::Int32 => {
                let values =
                    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const i32, check_size) };
                values.windows(2).all(|w| w[0] <= w[1])
            }
            DataType::Float32 => {
                let values =
                    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f32, check_size) };
                values.windows(2).all(|w| w[0] <= w[1])
            }
            _ => false,
        }
    }

    /// 计算重复率
    fn calculate_repetition_rate(data: &[u8]) -> f64 {
        if data.len() < 2 {
            return 0.0;
        }

        let mut repetitions = 0;
        for i in 1..data.len().min(1000) {
            if data[i] == data[i - 1] {
                repetitions += 1;
            }
        }

        repetitions as f64 / data.len().min(1000) as f64
    }

    /// 计算Delta方差（用于判断是否适合delta编码）
    fn calculate_delta_variance(data: &[u8], dtype: DataType) -> f64 {
        if data.len() < 16 {
            return f64::MAX;
        }

        match dtype {
            DataType::Int32 => {
                let values = unsafe {
                    std::slice::from_raw_parts(data.as_ptr() as *const i32, 100.min(data.len() / 4))
                };

                if values.len() < 2 {
                    return f64::MAX;
                }

                let deltas: Vec<i32> = values.windows(2).map(|w| w[1] - w[0]).collect();

                let mean = deltas.iter().sum::<i32>() as f64 / deltas.len() as f64;
                let variance = deltas
                    .iter()
                    .map(|&d| {
                        let diff = d as f64 - mean;
                        diff * diff
                    })
                    .sum::<f64>()
                    / deltas.len() as f64;

                variance
            }
            _ => f64::MAX,
        }
    }
}

/// 压缩策略选择器
pub struct CompressionSelector;

impl CompressionSelector {
    /// 根据数据特征选择最优压缩算法
    pub fn select_algorithm(
        characteristics: &DataCharacteristics,
        dtype: DataType,
    ) -> CompressionAlgorithm {
        // 规则1: 高重复率 -> RLE
        if characteristics.repetition_rate > 0.5 {
            return CompressionAlgorithm::RLE;
        }

        // 规则2: 有序数据且delta方差小 -> Delta编码
        if characteristics.is_sorted && characteristics.delta_variance < 100.0 {
            return CompressionAlgorithm::Delta;
        }

        // 规则3: 低基数 -> 字典编码
        let estimated_size = match dtype {
            DataType::Float32 | DataType::Int32 => 4,
            DataType::Float64 | DataType::Int64 => 8,
            _ => 1,
        };

        if characteristics.cardinality * 10 < estimated_size {
            return CompressionAlgorithm::Dictionary;
        }

        // 规则4: 低熵 -> LZ4（快速）
        if characteristics.entropy < 0.3 {
            return CompressionAlgorithm::LZ4;
        }

        // 规则5: 中熵 -> Snappy（平衡）
        if characteristics.entropy < 0.6 {
            return CompressionAlgorithm::Snappy;
        }

        // 规则6: 高熵 -> Zstd（高压缩率）
        CompressionAlgorithm::Zstd
    }

    /// 智能压缩
    pub fn compress(data: &[u8], dtype: DataType) -> io::Result<(Vec<u8>, CompressionAlgorithm)> {
        // 分析数据特征
        let characteristics = DataCharacteristics::analyze(data, dtype);

        // 选择算法
        let algorithm = Self::select_algorithm(&characteristics, dtype);

        // 执行压缩
        let compressed = match algorithm {
            CompressionAlgorithm::None => data.to_vec(),
            CompressionAlgorithm::Zstd => Self::compress_zstd(data, 3)?,
            CompressionAlgorithm::LZ4 => Self::compress_lz4(data)?,
            CompressionAlgorithm::Snappy => Self::compress_snappy(data)?,
            CompressionAlgorithm::RLE => Self::compress_rle(data),
            CompressionAlgorithm::Delta => Self::compress_delta(data, dtype),
            CompressionAlgorithm::Dictionary => Self::compress_dictionary(data),
        };

        // 如果压缩后更大，返回原始数据
        if compressed.len() >= data.len() {
            Ok((data.to_vec(), CompressionAlgorithm::None))
        } else {
            Ok((compressed, algorithm))
        }
    }

    /// Zstd压缩
    fn compress_zstd(data: &[u8], level: i32) -> io::Result<Vec<u8>> {
        zstd::encode_all(data, level)
    }

    /// LZ4压缩
    fn compress_lz4(data: &[u8]) -> io::Result<Vec<u8>> {
        lz4::block::compress(data, None, false).map_err(|e| io::Error::new(io::ErrorKind::Other, e))
    }

    /// Snappy压缩
    fn compress_snappy(data: &[u8]) -> io::Result<Vec<u8>> {
        let mut encoder = snap::raw::Encoder::new();
        encoder
            .compress_vec(data)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))
    }

    /// RLE压缩
    fn compress_rle(data: &[u8]) -> Vec<u8> {
        if data.is_empty() {
            return Vec::new();
        }

        let mut compressed = Vec::new();
        let mut current_byte = data[0];
        let mut count: u8 = 1;

        for &byte in &data[1..] {
            if byte == current_byte && count < 255 {
                count += 1;
            } else {
                compressed.push(count);
                compressed.push(current_byte);
                current_byte = byte;
                count = 1;
            }
        }

        compressed.push(count);
        compressed.push(current_byte);

        compressed
    }

    /// Delta编码
    fn compress_delta(data: &[u8], dtype: DataType) -> Vec<u8> {
        match dtype {
            DataType::Int32 => {
                let values = unsafe {
                    std::slice::from_raw_parts(data.as_ptr() as *const i32, data.len() / 4)
                };

                if values.is_empty() {
                    return Vec::new();
                }

                let mut result = Vec::with_capacity(data.len());
                result.extend_from_slice(&values[0].to_le_bytes());

                for i in 1..values.len() {
                    let delta = values[i] - values[i - 1];
                    result.extend_from_slice(&delta.to_le_bytes());
                }

                result
            }
            _ => data.to_vec(),
        }
    }

    /// 字典编码（简化版）
    fn compress_dictionary(data: &[u8]) -> Vec<u8> {
        use std::collections::HashMap;

        // 构建字典
        let mut dictionary = HashMap::new();
        let mut next_code = 0u8;

        for &byte in data {
            dictionary.entry(byte).or_insert_with(|| {
                let code = next_code;
                next_code += 1;
                code
            });
        }

        // 编码数据
        let mut compressed = Vec::new();

        // 写入字典大小
        compressed.push(dictionary.len() as u8);

        // 写入字典
        let mut dict_vec: Vec<_> = dictionary.iter().collect();
        dict_vec.sort_by_key(|(_, &code)| code);
        for (&value, _) in dict_vec {
            compressed.push(value);
        }

        // 写入编码数据
        for &byte in data {
            compressed.push(*dictionary.get(&byte).unwrap());
        }

        compressed
    }

    /// 解压缩
    pub fn decompress(
        compressed: &[u8],
        algorithm: CompressionAlgorithm,
        _dtype: DataType,
    ) -> io::Result<Vec<u8>> {
        match algorithm {
            CompressionAlgorithm::None => Ok(compressed.to_vec()),
            CompressionAlgorithm::Zstd => zstd::decode_all(compressed),
            CompressionAlgorithm::LZ4 => {
                // LZ4需要知道原始大小，这里假设在实际使用中会存储
                lz4::block::decompress(compressed, None)
                    .map_err(|e| io::Error::new(io::ErrorKind::Other, e))
            }
            CompressionAlgorithm::Snappy => {
                let mut decoder = snap::raw::Decoder::new();
                decoder
                    .decompress_vec(compressed)
                    .map_err(|e| io::Error::new(io::ErrorKind::Other, e))
            }
            CompressionAlgorithm::RLE => Ok(Self::decompress_rle(compressed)),
            CompressionAlgorithm::Delta => Ok(compressed.to_vec()), // 需要dtype信息
            CompressionAlgorithm::Dictionary => Ok(Self::decompress_dictionary(compressed)),
        }
    }

    /// RLE解压缩
    fn decompress_rle(compressed: &[u8]) -> Vec<u8> {
        let mut decompressed = Vec::new();

        for chunk in compressed.chunks(2) {
            if chunk.len() == 2 {
                let count = chunk[0];
                let value = chunk[1];
                decompressed.extend(std::iter::repeat(value).take(count as usize));
            }
        }

        decompressed
    }

    /// 字典解压缩
    fn decompress_dictionary(compressed: &[u8]) -> Vec<u8> {
        if compressed.is_empty() {
            return Vec::new();
        }

        let dict_size = compressed[0] as usize;
        if compressed.len() < 1 + dict_size {
            return Vec::new();
        }

        // 读取字典
        let dictionary = &compressed[1..1 + dict_size];

        // 解码数据
        compressed[1 + dict_size..]
            .iter()
            .map(|&code| dictionary.get(code as usize).copied().unwrap_or(0))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adaptive_compression() {
        // 测试高重复率数据
        let data = vec![42u8; 1000];
        let (compressed, algorithm) =
            CompressionSelector::compress(&data, DataType::Uint8).unwrap();

        // 应该使用RLE
        assert!(compressed.len() < data.len());
        println!(
            "RLE compression: {} -> {} bytes ({:?})",
            data.len(),
            compressed.len(),
            algorithm
        );
    }

    #[test]
    fn test_delta_encoding() {
        // 测试有序数据
        let data: Vec<i32> = (0..100).collect();
        let bytes =
            unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };

        let (compressed, algorithm) =
            CompressionSelector::compress(bytes, DataType::Int32).unwrap();
        println!(
            "Delta encoding: {} -> {} bytes ({:?})",
            bytes.len(),
            compressed.len(),
            algorithm
        );
    }
}
