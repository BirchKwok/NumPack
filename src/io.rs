use std::io::{Read, Write, BufReader, BufWriter, Seek, SeekFrom};
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
const BUFFER_SIZE: usize = 8 * 1024 * 1024; // 8MB 缓冲区

struct FileLock {
    file: File,
    is_exclusive: bool,
}

impl FileLock {
    fn new_exclusive(file: File) -> NnpResult<Self> {
        file.lock_exclusive()
            .map_err(|e| NnpError::IoError(format!("Failed to acquire exclusive file lock: {}", e)))?;
        Ok(FileLock { file, is_exclusive: true })
    }

    fn new_shared(file: File) -> NnpResult<Self> {
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
        let _ = self.file.unlock();
    }
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
        let mut buffer = Vec::with_capacity(BUFFER_SIZE);
        
        // 批量处理数据
        for chunk in data.chunks(BUFFER_SIZE / 4) {
            buffer.clear();
            buffer.reserve(chunk.len() * 4);
            
            // 将浮点数转换为字节
            for &value in chunk {
                buffer.extend_from_slice(&value.to_le_bytes());
            }
            
            // 批量写入
            writer.write_all(&buffer)?;
            bytes_written += buffer.len();
        }
        
        Ok(bytes_written)
    })
}

// 读取单个数组数据
fn read_array_data(reader: &mut (impl Read + Seek), header: &ArrayMeta) -> NnpResult<Array2<f32>> {
    reader.seek(SeekFrom::Start(header.data_offset))?;
    
    let total_size = (header.rows * header.cols) as usize;
    let mut data = Vec::with_capacity(total_size);
    let mut buffer = vec![0u8; BUFFER_SIZE];
    let mut remaining = total_size * 4;
    
    while remaining > 0 {
        let to_read = remaining.min(BUFFER_SIZE);
        let bytes_read = reader.read(&mut buffer[..to_read])?;
        if bytes_read == 0 {
            break;
        }
        
        // 批量转换字节为浮点数
        for chunk in buffer[..bytes_read].chunks_exact(4) {
            let value = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            data.push(value);
        }
        
        remaining -= bytes_read;
    }
    
    if data.len() != total_size {
        return Err(NnpError::InvalidArrayData("Incomplete array data".to_string()));
    }
    
    Array2::from_shape_vec((header.rows as usize, header.cols as usize), data)
        .map_err(|e| NnpError::InvalidArrayData(format!("Failed to create array: {}", e)))
}

pub fn save_arrays(path: &Path, arrays: &HashMap<String, Py<PyArray2<f32>>>) -> NnpResult<()> {
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)?;
    
    let mut locked_file = FileLock::new_exclusive(file)?;
    let file = locked_file.as_file();
    let mut writer = BufWriter::with_capacity(BUFFER_SIZE, file);
    
    // 写入文件头
    writer.write_all(MAGIC_NUMBER)?;
    writer.write_all(&VERSION.to_le_bytes())?;
    
    let array_count = arrays.len() as u32;
    writer.write_all(&array_count.to_le_bytes())?;
    
    // 计算头部信息
    let mut headers = Vec::new();
    let mut total_header_size = 12u64;
    
    Python::with_gil(|py| {
        for (name, array) in arrays {
            let array_ref = unsafe { array.as_ref(py).as_array() };
            let shape = array_ref.shape();
            
            total_header_size += 4 + name.len() as u64 + 24;
            
            let header = ArrayMeta {
                name: name.clone(),
                rows: shape[0] as u64,
                cols: shape[1] as u64,
                data_offset: 0,
            };
            headers.push(header);
        }
        Ok::<_, NnpError>(())
    })?;
    
    // 计算数据偏移量
    let mut current_offset = total_header_size;
    for header in &mut headers {
        header.data_offset = current_offset;
        current_offset += header.rows * header.cols * 4;
    }

    // 写入头部信息
    for header in &headers {
        let name_bytes = header.name.as_bytes();
        writer.write_all(&(name_bytes.len() as u32).to_le_bytes())?;
        writer.write_all(name_bytes)?;
        writer.write_all(&header.rows.to_le_bytes())?;
        writer.write_all(&header.cols.to_le_bytes())?;
        writer.write_all(&header.data_offset.to_le_bytes())?;
    }

    // 写入数组数据
    for (_, array) in arrays {
        write_array_data(&mut writer, array)?;
    }

    // 刷新缓冲区并同步到磁盘
    writer.flush()?;
    writer.get_ref().sync_all()?;

    Ok(())
}

pub fn load_arrays(path: &Path) -> NnpResult<HashMap<String, Array2<f32>>> {
    let file = OpenOptions::new()
        .read(true)
        .open(path)?;
    
    let mut locked_file = FileLock::new_shared(file)?;
    let file = locked_file.as_file();
    let mut reader = BufReader::with_capacity(BUFFER_SIZE, file);
    let mut arrays = HashMap::new();

    // 读取文件头
    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic)?;
    if magic != *MAGIC_NUMBER {
        return Err(NnpError::InvalidArrayData("Invalid file format".to_string()));
    }

    let mut version_bytes = [0u8; 4];
    reader.read_exact(&mut version_bytes)?;
    let version = u32::from_le_bytes(version_bytes);
    if version != VERSION {
        return Err(NnpError::InvalidArrayData(format!("Unsupported version: {}", version)));
    }

    let mut count_bytes = [0u8; 4];
    reader.read_exact(&mut count_bytes)?;
    let array_count = u32::from_le_bytes(count_bytes);

    // 读取头部信息
    let mut headers = Vec::with_capacity(array_count as usize);
    for _ in 0..array_count {
        let mut name_len_bytes = [0u8; 4];
        reader.read_exact(&mut name_len_bytes)?;
        let name_len = u32::from_le_bytes(name_len_bytes) as usize;

        let mut name_bytes = vec![0u8; name_len];
        reader.read_exact(&mut name_bytes)?;
        let name = String::from_utf8(name_bytes)
            .map_err(|e| NnpError::InvalidArrayData(format!("Invalid array name: {}", e)))?;

        let mut rows_bytes = [0u8; 8];
        let mut cols_bytes = [0u8; 8];
        let mut offset_bytes = [0u8; 8];
        reader.read_exact(&mut rows_bytes)?;
        reader.read_exact(&mut cols_bytes)?;
        reader.read_exact(&mut offset_bytes)?;

        headers.push(ArrayMeta {
            name,
            rows: u64::from_le_bytes(rows_bytes),
            cols: u64::from_le_bytes(cols_bytes),
            data_offset: u64::from_le_bytes(offset_bytes),
        });
    }

    // 读取数组数据
    for header in headers {
        let array = read_array_data(&mut reader, &header)?;
        arrays.insert(header.name, array);
    }

    Ok(arrays)
} 