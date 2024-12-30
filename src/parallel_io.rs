use std::fs::File;
use std::io::{self, BufWriter, Read, Write, Seek};
use std::path::PathBuf;
use std::sync::Arc;
use rayon::prelude::*;
use ndarray::{Array2, ArrayView2};
use memmap2::MmapOptions;
use numpy::Element;
use pyo3::{Python, PyResult, PyObject, types::PySlice, IntoPy};
use std::collections::HashSet;

use crate::error::NpkResult;
use crate::metadata::{CachedMetadataStore, ArrayMetadata, DataType};

const BUFFER_SIZE: usize = 8 * 1024 * 1024;  // 8MB buffer

#[allow(dead_code)]
pub struct ArrayView {
    pub meta: ArrayMetadata,
    file: File,
    file_path: PathBuf,
    mmap_mode: bool,
}

impl ArrayView {
    fn new(meta: ArrayMetadata, file: File, file_path: PathBuf, mmap_mode: bool) -> Self {
        Self {
            meta,
            file,
            file_path,
            mmap_mode,
        }
    }

    fn get_retained_indices(&self, excluded_indices: Option<&[i64]>) -> Vec<usize> {
        let mut excluded_set = HashSet::new();
        let original_rows = self.meta.rows as i64;
        
        // 合并元数据中的 deleted_indices 和传入的 excluded_indices
        if let Some(deleted) = &self.meta.deleted_indices {
            for &idx in deleted {
                let normalized_idx = if idx < 0 { original_rows + idx } else { idx };
                if normalized_idx >= 0 && normalized_idx < original_rows {
                    excluded_set.insert(normalized_idx);
                }
            }
        }
        
        if let Some(excluded) = excluded_indices {
            for &idx in excluded {
                let normalized_idx = if idx < 0 { original_rows + idx } else { idx };
                if normalized_idx >= 0 && normalized_idx < original_rows {
                    excluded_set.insert(normalized_idx);
                }
            }
        }

        // 如果没有需要排除的索引，返回所有行
        if excluded_set.is_empty() {
            return (0..self.meta.rows as usize).collect();
        }
        
        // 创建一个有序的排除索引列表
        let mut excluded_vec: Vec<i64> = excluded_set.into_iter().collect();
        excluded_vec.sort_unstable();
        
        // 计算保留的索引
        let mut retained = Vec::with_capacity((original_rows - excluded_vec.len() as i64) as usize);
        
        // 遍历所有行，保留不在排除列表中的行
        for i in 0..original_rows {
            if !excluded_vec.contains(&i) {
                retained.push(i as usize);
            }
        }
        
        retained
    }

    pub fn get_mmap_array(&mut self, py: Python, excluded_indices: Option<&[i64]>) -> PyResult<PyObject> {
        let np = py.import("numpy")?;
        let mmap = unsafe { MmapOptions::new().map(&self.file)? };
        
        // 如果没有要排除的行，使用直接内存映射
        if excluded_indices.map_or(true, |v| v.is_empty()) && self.meta.deleted_indices.is_none() {
            let dtype = match self.meta.dtype {
                DataType::Bool => np.getattr("bool")?,
                DataType::Uint8 => np.getattr("uint8")?,
                DataType::Uint16 => np.getattr("uint16")?,
                DataType::Uint32 => np.getattr("uint32")?,
                DataType::Uint64 => np.getattr("uint64")?,
                DataType::Int8 => np.getattr("int8")?,
                DataType::Int16 => np.getattr("int16")?,
                DataType::Int32 => np.getattr("int32")?,
                DataType::Int64 => np.getattr("int64")?,
                DataType::Float32 => np.getattr("float32")?,
                DataType::Float64 => np.getattr("float64")?,
            };

            // 直接使用内存映射
            let array = np.call_method1(
                "frombuffer",
                (
                    &mmap[..],
                    dtype,
                )
            )?;
            
            return Ok(array.call_method1("reshape", ((self.meta.rows as usize, self.meta.cols as usize),))?.into_py(py));
        }
        
        // 如果有要排除的行，使用复制方式
        let retained = self.get_retained_indices(excluded_indices);
        let shape = (retained.len(), self.meta.cols as usize);
        let element_size = self.meta.dtype.size_bytes() as usize;
        let row_size = self.meta.cols as usize * element_size;
        
        // 创建一个新的numpy数组
        let dtype = match self.meta.dtype {
            DataType::Bool => np.getattr("bool")?,
            DataType::Uint8 => np.getattr("uint8")?,
            DataType::Uint16 => np.getattr("uint16")?,
            DataType::Uint32 => np.getattr("uint32")?,
            DataType::Uint64 => np.getattr("uint64")?,
            DataType::Int8 => np.getattr("int8")?,
            DataType::Int16 => np.getattr("int16")?,
            DataType::Int32 => np.getattr("int32")?,
            DataType::Int64 => np.getattr("int64")?,
            DataType::Float32 => np.getattr("float32")?,
            DataType::Float64 => np.getattr("float64")?,
        };

        // 创建新数组
        let array = np.call_method1("zeros", (shape,))?.call_method1("astype", (dtype,))?;
        
        // 复制需要保留的行
        for (new_row, &old_row) in retained.iter().enumerate() {
            let src_offset = old_row * row_size;
            let dst_slice = array.call_method1("__getitem__", (new_row,))?;
            
            let src_data = &mmap[src_offset..src_offset + row_size];
            let slice = PySlice::new(py, 0, self.meta.cols as isize, 1);
            let row_data = np.call_method1("frombuffer", (src_data, dtype))?;
            dst_slice.call_method1("__setitem__", (slice, row_data))?;
        }
        
        Ok(array.into_py(py))
    }

    pub fn into_array<T: Element + Copy>(&mut self, excluded_indices: Option<&[i64]>) -> NpkResult<Array2<T>> {
        let element_size = std::mem::size_of::<T>();
        let row_size = (self.meta.cols as usize) * element_size;
        
        if self.mmap_mode {
            // 使用内存映射
            let mmap = unsafe { MmapOptions::new().map(&self.file)? };
            
            if let Some(excluded) = excluded_indices {
                // 计算需要保留的行
                let retained = self.get_retained_indices(Some(excluded));
                let new_rows = retained.len();
                let cols = self.meta.cols as usize;
                
                // 创建新数组
                let mut raw_data = vec![0u8; new_rows * cols * element_size];
                
                // 复制需要保留的行
                for (new_row, &old_row) in retained.iter().enumerate() {
                    let src_offset = old_row * row_size;
                    let dst_offset = new_row * row_size;
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            mmap.as_ptr().add(src_offset),
                            raw_data.as_mut_ptr().add(dst_offset),
                            row_size
                        );
                    }
                }
                
                // 转换为最终数组
                let array = unsafe {
                    Array2::from_shape_vec_unchecked(
                        (new_rows, cols),
                        std::slice::from_raw_parts(
                            raw_data.as_ptr() as *const T,
                            new_rows * cols
                        ).to_vec()
                    )
                };
                Ok(array)
            } else {
                // 如果没有要排除的行，直接返回整个数组
                let ptr = mmap.as_ptr() as *const T;
                let shape = (self.meta.rows as usize, self.meta.cols as usize);
                unsafe {
                    Ok(ArrayView2::from_shape_ptr(shape, ptr).to_owned())
                }
            }
        } else {
            if let Some(excluded) = excluded_indices {
                // 计算需要保留的行
                let retained = self.get_retained_indices(Some(excluded));
                let new_rows = retained.len();
                let cols = self.meta.cols as usize;
                
                // 创建新数组
                let mut raw_data = vec![0u8; new_rows * cols * element_size];
                let mut row_buffer = vec![0u8; row_size];
                
                // 读取并复制需要保留的行
                for (new_row, &old_row) in retained.iter().enumerate() {
                    // 定位到正确的文件位置
                    self.file.seek(io::SeekFrom::Start((old_row * row_size) as u64))?;
                    self.file.read_exact(&mut row_buffer)?;
                    
                    // 复制数据到新数组
                    let dst_offset = new_row * row_size;
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            row_buffer.as_ptr(),
                            raw_data.as_mut_ptr().add(dst_offset),
                            row_size
                        );
                    }
                }
                
                // 转换为最终数组
                let array = unsafe {
                    Array2::from_shape_vec_unchecked(
                        (new_rows, cols),
                        std::slice::from_raw_parts(
                            raw_data.as_ptr() as *const T,
                            new_rows * cols
                        ).to_vec()
                    )
                };
                Ok(array)
            } else {
                // 如果没有要排除的行，直接读取整个数组
                let size = (self.meta.rows * self.meta.cols * self.meta.dtype.size_bytes() as u64) as usize;
                let mut data = vec![0u8; size];
                self.file.read_exact(&mut data)?;
                
                let array = unsafe {
                    Array2::from_shape_vec_unchecked(
                        (self.meta.rows as usize, self.meta.cols as usize),
                        std::slice::from_raw_parts(
                            data.as_ptr() as *const T,
                            (self.meta.rows * self.meta.cols) as usize
                        ).to_vec()
                    )
                };
                Ok(array)
            }
        }
    }

    pub fn physical_delete(&mut self, excluded_indices: &[i64]) -> NpkResult<()> {
        let element_size = self.meta.dtype.size_bytes() as usize;
        let row_size = (self.meta.cols as usize) * element_size;
        
        // 计算需要保留的行
        let retained = self.get_retained_indices(Some(excluded_indices));
        let new_rows = retained.len();
        
        // 创建临时文件
        let temp_path = self.file_path.with_extension("tmp");
        let mut temp_file = BufWriter::with_capacity(
            BUFFER_SIZE,
            File::create(&temp_path)?
        );
        
        // 创建读取缓冲区
        let mut row_buffer = vec![0u8; row_size];
        
        // 复制需要保留的行到临时文件
        for &old_row in &retained {
            // 定位到正确的文件位置
            self.file.seek(io::SeekFrom::Start((old_row * row_size) as u64))?;
            self.file.read_exact(&mut row_buffer)?;
            temp_file.write_all(&row_buffer)?;
        }
        
        // 确保所有数据都写入磁盘
        temp_file.flush()?;
        
        // 关闭原文件和临时文件
        drop(temp_file);
        drop(&mut self.file);
        
        // 原子性地用临时文件替换原文件
        std::fs::rename(&temp_path, &self.file_path)?;
        
        // 重新打开文件
        self.file = File::open(&self.file_path)?;
        
        // 更新元数据
        self.meta.rows = new_rows as u64;
        self.meta.size_bytes = (new_rows * self.meta.cols as usize * element_size) as u64;
        self.meta.deleted_indices = None; // 清除已删除的索引，因为已经物理删除了
        
        Ok(())
    }
}

#[allow(dead_code)]
pub struct ParallelIO {
    base_dir: PathBuf,
    metadata: Arc<CachedMetadataStore>,
    metadata_path: PathBuf,
}

impl ParallelIO {
    pub fn new(base_dir: PathBuf) -> NpkResult<Self> {
        let metadata_path = base_dir.join("metadata.npkm");
        let wal_path = Some(base_dir.join("metadata.wal"));
        let metadata = CachedMetadataStore::new(&metadata_path, wal_path)?;
        
        Ok(Self {
            base_dir,
            metadata: Arc::new(metadata),
            metadata_path,
        })
    }

    pub fn save_arrays<T: Element + Copy + Send + Sync>(&self, arrays: &[(String, Array2<T>, DataType)]) -> NpkResult<()> {
        // 顺序保存数组
        for (name, array, dtype) in arrays {
            let data_file = format!("data_{}.npkd", name);
            let data_path = self.base_dir.join(&data_file);
            
            let meta = ArrayMetadata::new(
                name.clone(),
                array.shape()[0] as u64,
                array.shape()[1] as u64,
                data_file,
                *dtype,
            );

            // 写入数据文件
            let mut file = BufWriter::with_capacity(BUFFER_SIZE, File::create(&data_path)?);
            let data = unsafe {
                std::slice::from_raw_parts(
                    array.as_ptr() as *const u8,
                    array.shape()[0] * array.shape()[1] * std::mem::size_of::<T>()
                )
            };
            file.write_all(data)?;
            file.flush()?;

            // 更新元数据
            self.metadata.add_array(meta)?;
        }
        
        Ok(())
    }

    pub fn get_array_views(&self, names: Option<&[String]>, mmap_mode: bool) -> NpkResult<Vec<(String, ArrayView)>> {
        let arrays_to_load: Vec<_> = if let Some(names) = names {
            names.iter()
                .filter_map(|name| {
                    self.metadata.get_array(name)
                        .map(|meta| (name.clone(), meta))
                })
                .collect()
        } else {
            self.metadata.list_arrays()
                .into_iter()
                .filter_map(|name| {
                    self.metadata.get_array(&name)
                        .map(|meta| (name.clone(), meta))
                })
                .collect()
        };

        // 并行创建视图
        arrays_to_load.into_par_iter()
            .map(|(name, meta)| {
                let data_path = self.base_dir.join(&meta.data_file);
                let file = File::open(&data_path)?;
                let view = ArrayView::new(meta, file, data_path.clone(), mmap_mode);
                Ok((name, view))
            })
            .collect()
    }

    pub fn mark_deleted(&self, names: &[String]) -> NpkResult<()> {
        let mut any_deleted = false;
        
        for name in names {
            if self.metadata.mark_deleted(name)? {
                any_deleted = true;
            }
        }
        
        // 如果有删除操作且满足压缩条件，则进行压缩
        if any_deleted && self.metadata.should_compact(4 * 1024 * 1024 * 1024) { // 4GB
            self.compact()?;
        }
        
        Ok(())
    }

    pub fn get_array_meta(&self, name: &str) -> Option<ArrayMetadata> {
        self.metadata.get_array(name)
    }

    pub fn list_arrays(&self) -> Vec<String> {
        self.metadata.list_arrays()
    }

    fn compact(&self) -> NpkResult<()> {
        let active_arrays: Vec<_> = self.metadata.list_arrays();
        
        // 创建临时目录
        let temp_dir = self.base_dir.join(".temp_compact");
        std::fs::create_dir_all(&temp_dir)?;
        
        // 复制未删除的数组到新文件
        for name in &active_arrays {
            if let Some(meta) = self.metadata.get_array(name) {
                let old_path = self.base_dir.join(&meta.data_file);
                let new_path = temp_dir.join(&meta.data_file);
                
                // 复制数据文件
                std::fs::copy(old_path, new_path)?;
            }
        }
        
        // 删除所有旧文件
        for entry in std::fs::read_dir(&self.base_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().map_or(false, |ext| ext == "npkd") {
                std::fs::remove_file(path)?;
            }
        }
        
        // 移动新文件到原目录
        for entry in std::fs::read_dir(&temp_dir)? {
            let entry = entry?;
            let old_path = entry.path();
            let new_path = self.base_dir.join(entry.file_name());
            std::fs::rename(old_path, new_path)?;
        }
        
        // 删除临时目录
        std::fs::remove_dir(&temp_dir)?;
        
        // 更新元数据
        self.metadata.reset_deleted_size()?;
        self.metadata.force_sync()?;
        
        Ok(())
    }

    pub fn reset(&self) -> NpkResult<()> {
        self.metadata.reset()?;
        // 删除数组文件
        for entry in std::fs::read_dir(&self.base_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().map_or(false, |ext| ext == "npkd") {
                std::fs::remove_file(path)?;
            }
        }
        Ok(())
    }

    pub fn batch_mark_deleted(&self, names: &[String]) -> NpkResult<usize> {
        let deleted_count = self.metadata.batch_mark_deleted(names)?;
        
        // 如果删除的数组数量达到阈值，执行增量压缩
        if deleted_count > 0 && self.metadata.should_compact(4 * 1024 * 1024 * 1024) { // 4GB阈值
            // 每次处理100个数组
            let compacted = self.metadata.incremental_compact(100)?;
            if !compacted.is_empty() {
                // 删除已压缩的数组文件
                for name in compacted {
                    if let Some(meta) = self.metadata.get_array(&name) {
                        let file_path = self.base_dir.join(&meta.data_file);
                        if let Err(e) = std::fs::remove_file(file_path) {
                            eprintln!("Warning: Failed to remove file for array {}: {}", name, e);
                        }
                    }
                }
            }
        }
        
        Ok(deleted_count)
    }

    pub fn should_compact(&self, threshold: u64) -> bool {
        self.metadata.should_compact(threshold)
    }

    pub fn incremental_compact(&self, batch_size: usize) -> NpkResult<Vec<String>> {
        self.metadata.incremental_compact(batch_size)
    }

    pub fn update_array_metadata(&self, name: &str, meta: ArrayMetadata) -> NpkResult<()> {
        self.metadata.update_array_metadata(name, meta)
    }

    pub fn has_array(&self, name: &str) -> bool {
        self.metadata.has_array(name)
    }

    pub fn drop_arrays(&self, name: &str, excluded_indices: Option<&[i64]>) -> NpkResult<()> {
        if let Some(meta) = self.metadata.get_array(name) {
            let data_path = self.base_dir.join(&meta.data_file);
            let file = File::open(&data_path)?;
            let mut view = ArrayView::new(meta, file, data_path.clone(), false);
            
            if let Some(indices) = excluded_indices {
                // 物理删除指定的行
                view.physical_delete(indices)?;
                // 更新元数据
                self.metadata.update_array_metadata(name, view.meta)?;
            } else {
                // 如果没有指定行索引，删除整个数组
                std::fs::remove_file(&data_path)?;
                self.metadata.mark_deleted(name)?;
            }
        }
        Ok(())
    }

    pub fn batch_drop_arrays(&self, names: &[String], excluded_indices: Option<&[i64]>) -> NpkResult<()> {
        for name in names {
            self.drop_arrays(name, excluded_indices)?;
        }
        Ok(())
    }
} 