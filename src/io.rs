use std::io::{Read, Write, BufReader, BufWriter, Seek, SeekFrom};
use std::fs::{File, OpenOptions};
use std::path::Path;
use std::collections::HashMap;
use numpy::{PyArray2};
use ndarray::{Array2, ArrayView2};
use pyo3::{Py, Python};
use fs2::FileExt;
use memmap2::{MmapOptions, MmapMut, Mmap};
use crate::error::{NnpResult, NnpError};
use crate::types::ArrayMeta;

const MAGIC_NUMBER: &[u8; 4] = b"NNPK";
const VERSION: u32 = 1;
const BUFFER_SIZE: usize = 32 * 1024 * 1024; // 32MB 缓冲区

// 内存映射数组结构体
pub struct MmapArray {
    mmap: MmapMut,
    rows: usize,
    cols: usize,
    offset: usize,
}

impl MmapArray {
    fn new(mmap: MmapMut, rows: usize, cols: usize, offset: usize) -> Self {
        Self {
            mmap,
            rows,
            cols,
            offset,
        }
    }

    // 获取数组视图
    pub fn view(&self) -> ArrayView2<f32> {
        let data = unsafe {
            std::slice::from_raw_parts(
                self.mmap[self.offset..].as_ptr() as *const f32,
                self.rows * self.cols,
            )
        };
        ArrayView2::from_shape((self.rows, self.cols), data).unwrap()
    }

    // 写入数据
    pub fn write(&mut self, data: &[f32]) {
        let bytes = bytemuck::cast_slice(data);
        self.mmap[self.offset..self.offset + bytes.len()].copy_from_slice(bytes);
    }
}

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

// 缓存文件头部信息的结构体
#[derive(Debug, Clone)]
struct FileHeader {
    version: u32,
    array_count: u32,
    headers: Vec<ArrayMeta>,
}

impl FileHeader {
    fn read(reader: &mut impl Read) -> NnpResult<Self> {
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

        let mut headers = Vec::with_capacity(array_count as usize);
        for _ in 0..array_count {
            let mut name_len_bytes = [0u8; 4];
            reader.read_exact(&mut name_len_bytes)?;
            let name_len = u32::from_le_bytes(name_len_bytes) as usize;

            let mut name_bytes = vec![0u8; name_len];
            reader.read_exact(&mut name_bytes)?;
            let name = String::from_utf8(name_bytes)?;

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

        Ok(FileHeader {
            version,
            array_count,
            headers,
        })
    }
}

// 优化的数组读取函数
fn read_array_data_optimized(reader: &mut (impl Read + Seek), header: &ArrayMeta) -> NnpResult<Array2<f32>> {
    reader.seek(SeekFrom::Start(header.data_offset))?;
    
    let total_size = (header.rows * header.cols) as usize;
    let total_bytes = total_size * 4;
    
    // 使用预分配的缓冲区
    let mut buffer = vec![0u8; total_bytes];
    reader.read_exact(&mut buffer)?;
    
    // 直接将字节切片转换为f32切片，避免逐个转换
    let data = unsafe {
        std::slice::from_raw_parts(buffer.as_ptr() as *const f32, total_size)
    }.to_vec();
    
    Array2::from_shape_vec((header.rows as usize, header.cols as usize), data)
        .map_err(|e| NnpError::InvalidArrayData(format!("Failed to create array: {}", e)))
}

// 优化的数组写入函数
fn write_array_data_optimized(writer: &mut impl Write, array: &Py<PyArray2<f32>>) -> NnpResult<usize> {
    Python::with_gil(|py| {
        let array = unsafe { array.as_ref(py).as_array() };
        if !array.is_standard_layout() {
            return Err(NnpError::InvalidArrayData("Array must be C-contiguous".to_string()));
        }
        
        let data = array.as_slice().unwrap();
        
        // 直接将f32切片转换为字节切片
        let bytes = unsafe {
            std::slice::from_raw_parts(
                data.as_ptr() as *const u8,
                data.len() * 4
            )
        };
        
        // 分块写入
        for chunk in bytes.chunks(BUFFER_SIZE) {
            writer.write_all(chunk)?;
        }
        
        Ok(bytes.len())
    })
}

// 优化的内存映射读取函数
fn mmap_array_data_optimized(file: &File, header: &ArrayMeta, reuse_mmap: Option<&Mmap>) -> NnpResult<Array2<f32>> {
    let total_size = (header.rows * header.cols) as usize;
    let total_bytes = total_size * 4;
    
    let data = if let Some(mmap) = reuse_mmap {
        // 重用现有的内存映射
        unsafe {
            std::slice::from_raw_parts(
                mmap[header.data_offset as usize..].as_ptr() as *const f32,
                total_size
            )
        }
    } else {
        // 创建新的内存映射
        unsafe {
            let mmap = MmapOptions::new()
                .offset(header.data_offset)
                .len(total_bytes)
                .map(file)
                .map_err(|e| NnpError::IoError(format!("Failed to create memory map: {}", e)))?;
                
            std::slice::from_raw_parts(mmap.as_ptr() as *const f32, total_size)
        }
    };
    
    let array = Array2::from_shape_vec((header.rows as usize, header.cols as usize), data.to_vec())
        .map_err(|e| NnpError::InvalidArrayData(format!("Failed to create array: {}", e)))?;
        
    Ok(array)
}

// 优化的内存映射写入函数
fn mmap_write_array_data_optimized(file: &File, array: &Py<PyArray2<f32>>, offset: u64) -> NnpResult<()> {
    Python::with_gil(|py| {
        let array = unsafe { array.as_ref(py).as_array() };
        if !array.is_standard_layout() {
            return Err(NnpError::InvalidArrayData("Array must be C-contiguous".to_string()));
        }
        
        let data = array.as_slice().unwrap();
        let total_bytes = data.len() * 4;
        
        unsafe {
            let mut mmap = MmapOptions::new()
                .offset(offset)
                .len(total_bytes)
                .map_mut(file)
                .map_err(|e| NnpError::IoError(format!("Failed to create memory map: {}", e)))?;
                
            let bytes = std::slice::from_raw_parts(
                data.as_ptr() as *const u8,
                total_bytes
            );
            mmap.copy_from_slice(bytes);
            mmap.flush()
                .map_err(|e| NnpError::IoError(format!("Failed to flush memory map: {}", e)))?;
        }
        
        Ok(())
    })
}

// 优化的加载函数
pub fn load_arrays(path: &Path, mmap_mode: bool) -> NnpResult<HashMap<String, Array2<f32>>> {
    let file = OpenOptions::new()
        .read(true)
        .open(path)?;
    
    let mut reader = BufReader::with_capacity(BUFFER_SIZE, file);
    
    // 读取文件头部信息
    let header = FileHeader::read(&mut reader)?;
    
    let mut arrays = HashMap::new();
    
    if mmap_mode {
        // 对于内存映射模式，创建一个共享的内存映射
        let file = OpenOptions::new()
            .read(true)
            .open(path)?;
            
        let mmap = unsafe {
            MmapOptions::new()
                .map(&file)
                .map_err(|e| NnpError::IoError(format!("Failed to create memory map: {}", e)))?
        };
        
        // 使用共享的内存映射读取所有数组
        for meta in header.headers {
            let array = mmap_array_data_optimized(&file, &meta, Some(&mmap))?;
            arrays.insert(meta.name, array);
        }
    } else {
        // 对于普通模式，使用优化的读取函数
        for meta in header.headers {
            let array = read_array_data_optimized(&mut reader, &meta)?;
            arrays.insert(meta.name, array);
        }
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
        
        // 预分配一次性缓冲区，避免频繁分配
        let mut buffer = Vec::with_capacity(data.len() * 4);
        
        // 一次性转换所有数据
        for &value in data {
            buffer.extend_from_slice(&value.to_le_bytes());
        }
        
        // 分块写入大缓冲区
        for chunk in buffer.chunks(BUFFER_SIZE) {
            writer.write_all(chunk)?;
            bytes_written += chunk.len();
        }
        
        Ok(bytes_written)
    })
}

// 读取单个数组数据
fn read_array_data(reader: &mut (impl Read + Seek), header: &ArrayMeta) -> NnpResult<Array2<f32>> {
    reader.seek(SeekFrom::Start(header.data_offset))?;
    
    let total_size = (header.rows * header.cols) as usize;
    let total_bytes = total_size * 4;
    
    // 预分配完整缓冲区
    let mut buffer = vec![0u8; total_bytes];
    
    // 一次性读取所有数据
    reader.read_exact(&mut buffer)?;
    
    // 批量转换字节为浮点数
    let mut data = Vec::with_capacity(total_size);
    for chunk in buffer.chunks_exact(4) {
        let value = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        data.push(value);
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
    
    // 计算文件总大小
    let mut total_size = 12u64; // 魔数(4) + 版本(4) + 数组数量(4)
    let mut headers = Vec::new();
    
    Python::with_gil(|py| {
        for (name, array) in arrays {
            let array_ref = unsafe { array.as_ref(py).as_array() };
            let shape = array_ref.shape();
            
            total_size += 4 + name.len() as u64 + 24; // 名称长度(4) + 名称 + 行数(8) + 列数(8) + 偏移量(8)
            
            let header = ArrayMeta {
                name: name.clone(),
                rows: shape[0] as u64,
                cols: shape[1] as u64,
                data_offset: 0,
            };
            headers.push(header);
        }
        
        // 计算数据偏移量
        let mut current_offset = total_size;
        for header in &mut headers {
            header.data_offset = current_offset;
            current_offset += header.rows * header.cols * 4;
        }
        
        // 设置文件大小
        file.set_len(current_offset)?;
        
        // 写入文件头和头部信息
        {
            let mut writer = BufWriter::with_capacity(BUFFER_SIZE, file);
            writer.write_all(MAGIC_NUMBER)?;
            writer.write_all(&VERSION.to_le_bytes())?;
            writer.write_all(&(arrays.len() as u32).to_le_bytes())?;
            
            // 写入头部信息
            for header in &headers {
                let name_bytes = header.name.as_bytes();
                writer.write_all(&(name_bytes.len() as u32).to_le_bytes())?;
                writer.write_all(name_bytes)?;
                writer.write_all(&header.rows.to_le_bytes())?;
                writer.write_all(&header.cols.to_le_bytes())?;
                writer.write_all(&header.data_offset.to_le_bytes())?;
            }
            writer.flush()?;
        }
        
        // 使用内存映射写入数组数据
        for (name, array) in arrays {
            if let Some(header) = headers.iter().find(|h| h.name == *name) {
                // 重新打开文件以获取新的文件句柄用于内存映射
                let mmap_file = OpenOptions::new()
                    .read(true)
                    .write(true)
                    .open(path)?;
                mmap_write_array_data_optimized(&mmap_file, array, header.data_offset)?;
            }
        }
        
        Ok(())
    })
}

// 优化的切片访问实现
fn mmap_getitem_optimized(file: &File, header: &ArrayMeta, indexes: &[usize]) -> NnpResult<Array2<f32>> {
    let cols = header.cols as usize;
    
    // 计算连续区间，减少内存映射次数
    let mut ranges = Vec::new();
    let mut current_start = indexes[0];
    let mut current_end = indexes[0];
    
    for &idx in &indexes[1..] {
        if idx == current_end + 1 {
            current_end = idx;
        } else {
            ranges.push((current_start, current_end));
            current_start = idx;
            current_end = idx;
        }
    }
    ranges.push((current_start, current_end));
    
    // 预分配结果向量
    let mut result = Vec::with_capacity(indexes.len() * cols);
    
    // 为每个连续区间创建一次内存映射
    for (start, end) in ranges {
        let range_size = end - start + 1;
        let offset = header.data_offset + (start * cols * 4) as u64;
        let map_size = range_size * cols * 4;
        
        unsafe {
            let mmap = MmapOptions::new()
                .offset(offset)
                .len(map_size)
                .map(file)
                .map_err(|e| NnpError::IoError(format!("Failed to create memory map: {}", e)))?;
            
            // 直接将整个区间的数据转换为f32切片
            let data = std::slice::from_raw_parts(
                mmap.as_ptr() as *const f32,
                range_size * cols
            );
            
            // 将数据添加到结果向量
            result.extend_from_slice(data);
        }
    }
    
    Array2::from_shape_vec((indexes.len(), cols), result)
        .map_err(|e| NnpError::InvalidArrayData(format!("Failed to create array: {}", e)))
}

// 优化的随机访问函数
pub fn getitem(path: &Path, indexes: &[usize], array_names: Option<&[String]>) -> NnpResult<HashMap<String, Array2<f32>>> {
    let file = OpenOptions::new()
        .read(true)
        .open(path)?;
    
    let mut reader = BufReader::with_capacity(BUFFER_SIZE, file);
    
    // 读取文件头部信息
    let header = FileHeader::read(&mut reader)?;
    let mut arrays = HashMap::new();
    
    // 重新打开文件以获取新的文件句柄用于内存映射
    let mmap_file = OpenOptions::new()
        .read(true)
        .open(path)?;
        
    // 读取指定数组的数据
    for meta in header.headers {
        if let Some(names) = array_names {
            if !names.contains(&meta.name) {
                continue;
            }
        }
        
        let array = mmap_getitem_optimized(&mmap_file, &meta, indexes)?;
        arrays.insert(meta.name, array);
    }
    
    Ok(arrays)
} 