use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use rayon::prelude::*;
use ndarray::{Array2, ArrayView2};
use memmap2::MmapOptions;
use numpy::Element;
use pyo3::{Python, PyResult, PyObject};

use crate::error::NpkResult;
use crate::metadata::{MetadataStore, ArrayMetadata, DataType};

const BUFFER_SIZE: usize = 4 * 1024 * 1024;  // 4MB buffer

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

    pub fn get_mmap_array(&mut self, py: Python) -> PyResult<PyObject> {
        let np = py.import("numpy")?;
        let path = self.file_path.to_str().ok_or_else(|| 
            pyo3::exceptions::PyValueError::new_err("Invalid file path")
        )?;
        
        let dtype = match self.meta.dtype {
            DataType::Bool => "bool",
            DataType::Uint8 => "uint8",
            DataType::Uint16 => "uint16",
            DataType::Uint32 => "uint32",
            DataType::Uint64 => "uint64",
            DataType::Int8 => "int8",
            DataType::Int16 => "int16",
            DataType::Int32 => "int32",
            DataType::Int64 => "int64",
            DataType::Float16 => "float16",
            DataType::Float32 => "float32",
            DataType::Float64 => "float64",
        };
        
        let shape = (self.meta.rows as i64, self.meta.cols as i64);
        let kwargs = pyo3::types::PyDict::new(py);
        kwargs.set_item("mode", "r")?;
        kwargs.set_item("dtype", dtype)?;
        kwargs.set_item("shape", shape)?;
        
        let memmap = np.getattr("memmap")?.call(
            (path,), 
            Some(kwargs)
        )?;
        Ok(memmap.into())
    }

    pub fn into_array<T: Element + Copy>(&mut self) -> NpkResult<Array2<T>> {
        let size = (self.meta.rows * self.meta.cols * self.meta.dtype.size_bytes() as u64) as usize;
        
        if self.mmap_mode {
            // 使用内存映射
            let mmap = unsafe { MmapOptions::new().map(&self.file)? };
            let ptr = mmap.as_ptr() as *const T;
            let shape = (self.meta.rows as usize, self.meta.cols as usize);
            
            // 使用内存映射创建数组视图
            unsafe {
                Ok(ArrayView2::from_shape_ptr(shape, ptr).to_owned())
            }
        } else {
            // 普通读取
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

pub struct ParallelIO {
    base_dir: PathBuf,
    metadata: Arc<Mutex<MetadataStore>>,
    metadata_path: PathBuf,
}

impl ParallelIO {
    pub fn new(base_dir: PathBuf) -> NpkResult<Self> {
        let metadata_path = base_dir.join("metadata.npkm");
        let metadata = if metadata_path.exists() {
            MetadataStore::load(&metadata_path)?
        } else {
            MetadataStore::new()
        };
        
        Ok(Self {
            base_dir,
            metadata: Arc::new(Mutex::new(metadata)),
            metadata_path,
        })
    }

    pub fn save_arrays<T: Element + Copy + Send + Sync>(&self, arrays: &[(String, Array2<T>, DataType)]) -> NpkResult<()> {
        // 获取元数据锁
        let mut metadata = self.metadata.lock().unwrap();
        
        // 顺序保存数组
        for (name, array, dtype) in arrays {
            let data_file = format!("data_{}.npkd", name);
            let data_path = self.base_dir.join(&data_file);
            
            // 创建元数据
            let meta = ArrayMetadata {
                name: name.clone(),
                rows: array.shape()[0] as u64,
                cols: array.shape()[1] as u64,
                data_file,
                is_deleted: false,
                last_modified: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
                size_bytes: (array.shape()[0] * array.shape()[1] * dtype.size_bytes()) as u64,
                dtype: *dtype,
            };

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
            metadata.add_array(meta);
        }
        
        // 保存元数据
        metadata.save(&self.metadata_path)?;
        
        Ok(())
    }

    pub fn get_array_views(&self, names: Option<&[String]>, mmap_mode: bool) -> NpkResult<Vec<(String, ArrayView)>> {
        let metadata = self.metadata.lock().unwrap();
        let arrays_to_load: Vec<_> = if let Some(names) = names {
            names.iter()
                .filter_map(|name| {
                    let meta = metadata.get_array(name)?;
                    if !meta.is_deleted {
                        Some((name.clone(), meta.clone()))
                    } else {
                        None
                    }
                })
                .collect()
        } else {
            metadata.list_arrays()
                .into_iter()
                .filter_map(|name| {
                    let meta = metadata.get_array(&name)?;
                    if !meta.is_deleted {
                        Some((name.clone(), meta.clone()))
                    } else {
                        None
                    }
                })
                .collect()
        };
        drop(metadata);

        // 并行创建视图
        arrays_to_load.into_par_iter()
            .map(|(name, meta)| {
                let data_path = self.base_dir.join(&meta.data_file);
                let file = File::open(&data_path)?;
                let view = ArrayView::new(meta, file, data_path, mmap_mode);
                Ok((name, view))
            })
            .collect()
    }

    pub fn mark_deleted(&self, names: &[String]) -> NpkResult<()> {
        let mut metadata = self.metadata.lock().unwrap();
        let mut any_deleted = false;
        
        for name in names {
            if metadata.mark_deleted(name) {
                any_deleted = true;
            }
        }
        
        // 保存元数据
        metadata.save(&self.metadata_path)?;
        
        // 如果有删除操作且满足压缩条件，则进行压缩
        if any_deleted && metadata.should_compact(4 * 1024 * 1024 * 1024) { // 4GB
            drop(metadata); // 释放锁
            self.compact()?;
        }
        
        Ok(())
    }

    pub fn get_array_meta(&self, name: &str) -> Option<ArrayMetadata> {
        let metadata = self.metadata.lock().unwrap();
        metadata.get_array(name).cloned()
    }

    pub fn list_arrays(&self) -> Vec<String> {
        let metadata = self.metadata.lock().unwrap();
        metadata.list_arrays()
    }

    fn compact(&self) -> NpkResult<()> {
        let mut metadata = self.metadata.lock().unwrap();
        let active_arrays: Vec<_> = metadata.list_arrays();
        
        // 创建临时目录
        let temp_dir = self.base_dir.join(".temp_compact");
        std::fs::create_dir_all(&temp_dir)?;
        
        // 复制未删除的数组到新文件
        for name in &active_arrays {
            if let Some(meta) = metadata.get_array(name) {
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
        metadata.reset_deleted_size();
        metadata.save(&self.metadata_path)?;
        
        Ok(())
    }

    pub fn reset(&self) -> NpkResult<()> {
        let mut metadata = self.metadata.lock().unwrap();
        metadata.reset();
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
} 