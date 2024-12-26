use std::io::{Read, Write, BufReader, Seek, SeekFrom};
use std::fs::{File, OpenOptions};
use std::path::Path;
use std::collections::HashMap;
use numpy::{PyArray2};
use ndarray::Array2;
use pyo3::{Py, Python};
use fs2::FileExt;
use crate::error::{NnpResult, NnpError};
use crate::types::ArrayMeta;

const MAGIC_NUMBER: &[u8; 4] = b"NNPK";
const VERSION: u32 = 1;

struct FileLock {
    file: File,
    is_exclusive: bool,
}

impl FileLock {
    fn new_exclusive(file: File) -> NnpResult<Self> {
        // 尝试获取独占锁，如果失败则等待
        file.lock_exclusive()
            .map_err(|e| NnpError::IoError(format!("Failed to acquire exclusive file lock: {}", e)))?;
        Ok(FileLock { file, is_exclusive: true })
    }

    fn new_shared(file: File) -> NnpResult<Self> {
        // 尝试获取共享锁，如果失败则等待
        file.lock_shared()
            .map_err(|e| NnpError::IoError(format!("Failed to acquire shared file lock: {}", e)))?;
        Ok(FileLock { file, is_exclusive: false })
    }

    fn as_file(&mut self) -> &mut File {
        &mut self.file
    }
}

impl Drop for FileLock {
    fn drop(&mut self) {
        // 释放锁
        let _ = self.file.unlock();
    }
}

pub fn save_arrays(path: &Path, arrays: &HashMap<String, Py<PyArray2<f32>>>) -> NnpResult<()> {
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)?;
    
    // 获取独占锁
    let mut locked_file = FileLock::new_exclusive(file)?;
    let file = locked_file.as_file();
    
    // 写入魔数和版本号
    file.write_all(MAGIC_NUMBER)?;
    file.write_all(&VERSION.to_le_bytes())?;
    
    // 写入数组数量
    let array_count = arrays.len() as u32;
    file.write_all(&array_count.to_le_bytes())?;
    
    // 计算头部大小
    let mut headers = Vec::new();
    let mut total_header_size = 12u64; // 魔数(4) + 版本号(4) + 数组数量(4)
    
    // 计算每个数组的头部大小
    Python::with_gil(|py| {
        for (name, array) in arrays {
            let array_ref = unsafe { array.as_ref(py).as_array() };
            let shape = array_ref.shape();
            
            // 头部大小：名称长度(4) + 名称 + 行数(8) + 列数(8) + 偏移量(8)
            total_header_size += 4 + name.len() as u64 + 24;
            
            let header = ArrayMeta {
                name: name.clone(),
                rows: shape[0] as u64,
                cols: shape[1] as u64,
                data_offset: 0, // 暂时设为0，后面再更新
            };
            headers.push(header);
        }
        Ok::<_, NnpError>(())
    })?;
    
    // 计算每个数组的数据偏移量
    let mut current_offset = total_header_size;
    for header in &mut headers {
        header.data_offset = current_offset;
        current_offset += header.rows * header.cols * 4;
    }

    // 写入所有头部信息
    for header in &headers {
        let name_bytes = header.name.as_bytes();
        file.write_all(&(name_bytes.len() as u32).to_le_bytes())?;
        file.write_all(name_bytes)?;
        file.write_all(&header.rows.to_le_bytes())?;
        file.write_all(&header.cols.to_le_bytes())?;
        file.write_all(&header.data_offset.to_le_bytes())?;
    }

    // 写入数组数据
    for (_, array) in arrays {
        write_array_data(file, array)?;
    }

    // 确保所有数据都写入磁盘
    file.sync_all()?;

    Ok(())
}

pub fn load_arrays(path: &Path) -> NnpResult<HashMap<String, Array2<f32>>> {
    let file = OpenOptions::new()
        .read(true)
        .open(path)?;
    
    // 获取共享锁
    let mut locked_file = FileLock::new_shared(file)?;
    let file = locked_file.as_file();
    let mut reader = BufReader::new(file);
    let mut arrays = HashMap::new();

    // 读取并验证魔数
    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic)?;
    if magic != *MAGIC_NUMBER {
        return Err(NnpError::InvalidArrayData("Invalid file format".to_string()));
    }

    // 读取并验证版本号
    let mut version_bytes = [0u8; 4];
    reader.read_exact(&mut version_bytes)?;
    let version = u32::from_le_bytes(version_bytes);
    if version != VERSION {
        return Err(NnpError::InvalidArrayData(format!("Unsupported version: {}", version)));
    }

    // 读取数组数量
    let mut count_bytes = [0u8; 4];
    reader.read_exact(&mut count_bytes)?;
    let array_count = u32::from_le_bytes(count_bytes);

    // 读取头部信息
    let mut headers = Vec::new();
    for _ in 0..array_count {
        // 读取名称长度
        let mut name_len_bytes = [0u8; 4];
        reader.read_exact(&mut name_len_bytes)?;
        let name_len = u32::from_le_bytes(name_len_bytes) as usize;

        // 读取名称
        let mut name_bytes = vec![0u8; name_len];
        reader.read_exact(&mut name_bytes)?;
        let name = String::from_utf8(name_bytes)
            .map_err(|e| NnpError::InvalidArrayData(format!("Invalid array name: {}", e)))?;

        // 读取维度和偏移量
        let mut rows_bytes = [0u8; 8];
        let mut cols_bytes = [0u8; 8];
        let mut offset_bytes = [0u8; 8];
        reader.read_exact(&mut rows_bytes)?;
        reader.read_exact(&mut cols_bytes)?;
        reader.read_exact(&mut offset_bytes)?;

        let header = ArrayMeta {
            name: name.clone(),
            rows: u64::from_le_bytes(rows_bytes),
            cols: u64::from_le_bytes(cols_bytes),
            data_offset: u64::from_le_bytes(offset_bytes),
        };
        headers.push(header);
    }

    // 读取数组数据
    for header in headers {
        let array = read_array_data(&mut reader, &header)?;
        arrays.insert(header.name, array);
    }

    Ok(arrays)
}

// 写入单个数组数据
fn write_array_data(writer: &mut impl Write, array: &Py<PyArray2<f32>>) -> NnpResult<usize> {
    Python::with_gil(|py| {
        let array = unsafe { array.as_ref(py).as_array() };
        if !array.is_standard_layout() {
            return Err(NnpError::InvalidArrayData("Array must be C-contiguous".to_string()));
        }
        
        let data = array.as_slice().unwrap();
        let mut bytes_written = 0;
        
        // 将每个浮点数转换为字节并写入
        for &value in data {
            let bytes = value.to_le_bytes();
            writer.write_all(&bytes)?;
            bytes_written += bytes.len();
        }
        
        Ok(bytes_written)
    })
}

// 读取单个数组数据
fn read_array_data(reader: &mut (impl Read + Seek), header: &ArrayMeta) -> NnpResult<Array2<f32>> {
    // 跳转到数据位置
    reader.seek(SeekFrom::Start(header.data_offset))?;
    
    let total_size = (header.rows * header.cols) as usize;
    let mut data = Vec::with_capacity(total_size);
    
    // 读取每个浮点数
    for _ in 0..total_size {
        let mut bytes = [0u8; 4];
        reader.read_exact(&mut bytes)?;
        let value = f32::from_le_bytes(bytes);
        data.push(value);
    }
    
    // 创建新的数组
    Array2::from_shape_vec((header.rows as usize, header.cols as usize), data)
        .map_err(|e| NnpError::InvalidArrayData(format!("Failed to create array: {}", e)))
} 