use std::fs::{File, OpenOptions};
use std::io::{self, BufWriter, Read, Write, Seek};
use std::path::PathBuf;
use std::sync::Arc;
use rayon::prelude::*;
use ndarray::{Array2, ArrayView2};
use memmap2::MmapOptions;
use numpy::Element;
use pyo3::{Python, PyResult, PyObject, IntoPy};
use std::collections::HashSet;
use std::os::unix::fs::FileExt;

use crate::error::{NpkResult, NpkError};
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
                
        if let Some(excluded) = excluded_indices {
            for &idx in excluded {
                let normalized_idx = if idx < 0 { original_rows + idx } else { idx };
                if normalized_idx >= 0 && normalized_idx < original_rows {
                    excluded_set.insert(normalized_idx);
                }
            }
        }

        // If there are no indices to exclude, return all rows
        if excluded_set.is_empty() {
            return (0..self.meta.rows as usize).collect();
        }
        
        // Create a sorted list of excluded indices
        let mut excluded_vec: Vec<i64> = excluded_set.into_iter().collect();
        excluded_vec.sort_unstable();
        
        // Calculate retained indices
        let mut retained = Vec::with_capacity((original_rows - excluded_vec.len() as i64) as usize);
        
        // Iterate over all rows, keeping rows not in the exclusion list
        for i in 0..original_rows {
            if !excluded_vec.contains(&i) {
                retained.push(i as usize);
            }
        }
        
        retained
    }

    pub fn get_mmap_array(&mut self, py: Python, excluded_indices: Option<&[i64]>) -> PyResult<PyObject> {
        let np = py.import("numpy")?;
        
        // If there are no indices to exclude, use direct memory mapping
        if excluded_indices.map_or(true, |v| v.is_empty()) {
            // Create memory mapping
            let mmap = unsafe { 
                MmapOptions::new()
                    .len(self.meta.size_bytes as usize)
                    .map(&self.file)?
            };

            // Get data type and shape
            let dtype = match self.meta.dtype {
                DataType::Bool => "bool",
                DataType::Uint16 => "uint16",
                DataType::Uint32 => "uint32",
                DataType::Uint64 => "uint64",
                DataType::Int8 => "int8",
                DataType::Int16 => "int16",
                DataType::Int32 => "int32",
                DataType::Int64 => "int64",
                DataType::Float32 => "float32",
                DataType::Float64 => "float64",
                DataType::Uint8 => "uint8",
            };

            // Create numpy array from memory directly
            let array = unsafe {
                let data = std::slice::from_raw_parts(
                    mmap.as_ptr(),
                    self.meta.size_bytes as usize
                );

                // Create array directly using frombuffer
                np.call_method1(
                    "frombuffer",
                    (data, dtype),
                )?.call_method1(
                    "reshape",
                    ((self.meta.rows as usize, self.meta.cols as usize),)
                )?
            };

            // Keep memory mapping alive
            std::mem::forget(mmap);
            
            return Ok(array.into_py(py));
        }
        
        // If there are rows to exclude, use optimized copying
        let retained = self.get_retained_indices(excluded_indices);
        let shape = (retained.len(), self.meta.cols as usize);
        
        // Create memory mapping
        let mmap = unsafe { 
            MmapOptions::new()
                .len(self.meta.size_bytes as usize)
                .map(&self.file)?
        };

        // Get data type
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
            DataType::Float32 => "float32",
            DataType::Float64 => "float64",
        };

        // Calculate row size
        let element_size = self.meta.dtype.size_bytes() as usize;
        let row_size = self.meta.cols as usize * element_size;
        
        // Create target buffer
        let buffer_size = shape.0 * shape.1 * element_size;
        let mut buffer = Vec::with_capacity(buffer_size);
        
        // Collect all needed rows
        for &idx in &retained {
            let start = idx * row_size;
            let end = start + row_size;
            buffer.extend_from_slice(&mmap[start..end]);
        }

        // Create numpy array directly from buffer
        let array = {
            np.call_method1(
                "frombuffer",
                (&buffer[..], dtype),
            )?.call_method1(
                "reshape",
                shape,
            )?
        };

        // Keep buffer alive
        std::mem::forget(buffer);
        
        Ok(array.into_py(py))
    }

    pub fn into_array<T: Element + Copy>(&mut self, excluded_indices: Option<&[i64]>) -> NpkResult<Array2<T>> {
        let element_size = std::mem::size_of::<T>();
        let row_size = (self.meta.cols as usize) * element_size;
        
        if self.mmap_mode {
            // Use memory mapping
            let mmap = unsafe { MmapOptions::new().map(&self.file)? };
            
            if let Some(excluded) = excluded_indices {
                // Calculate rows to retain
                let retained = self.get_retained_indices(Some(excluded));
                let new_rows = retained.len();
                let cols = self.meta.cols as usize;
                
                // Create new array
                let mut raw_data = vec![0u8; new_rows * cols * element_size];
                
                // Copy needed rows
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
                
                // Convert to final array
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
                // If there are no rows to exclude, return the entire array
                let ptr = mmap.as_ptr() as *const T;
                let shape = (self.meta.rows as usize, self.meta.cols as usize);
                unsafe {
                    Ok(ArrayView2::from_shape_ptr(shape, ptr).to_owned())
                }
            }
        } else {
            if let Some(excluded) = excluded_indices {
                // Calculate rows to retain
                let retained = self.get_retained_indices(Some(excluded));
                let new_rows = retained.len();
                let cols = self.meta.cols as usize;
                
                // Create new array
                let mut raw_data = vec![0u8; new_rows * cols * element_size];
                let mut row_buffer = vec![0u8; row_size];
                
                // Read and copy needed rows
                for (new_row, &old_row) in retained.iter().enumerate() {
                    // Locate to correct file position
                    self.file.seek(io::SeekFrom::Start((old_row * row_size) as u64))?;
                    self.file.read_exact(&mut row_buffer)?;
                    
                    // Copy data to new array
                    let dst_offset = new_row * row_size;
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            row_buffer.as_ptr(),
                            raw_data.as_mut_ptr().add(dst_offset),
                            row_size
                        );
                    }
                }
                
                // Convert to final array
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
                // If there are no rows to exclude, read the entire array directly
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
        
        // Calculate rows to retain
        let retained = self.get_retained_indices(Some(excluded_indices));
        let new_rows = retained.len();
        
        // Create temporary file
        let temp_path = self.file_path.with_extension("tmp");
        let mut temp_file = BufWriter::with_capacity(
            BUFFER_SIZE,
            File::create(&temp_path)?
        );
        
        // Create read buffer
        let mut row_buffer = vec![0u8; row_size];
        
        // Copy needed rows to temporary file
        for &old_row in &retained {
            // Locate to correct file position
            self.file.seek(io::SeekFrom::Start((old_row * row_size) as u64))?;
            self.file.read_exact(&mut row_buffer)?;
            temp_file.write_all(&row_buffer)?;
        }
        
        // Ensure all data is written to disk
        temp_file.flush()?;
        
        // Close original file and temporary file
        drop(temp_file);
        self.file = File::create(&self.file_path)?;
        
        // Atomically replace original file with temporary file
        std::fs::rename(&temp_path, &self.file_path)?;
        
        // Reopen file
        self.file = File::open(&self.file_path)?;
        
        // Update metadata
        self.meta.rows = new_rows as u64;
        self.meta.size_bytes = (new_rows * self.meta.cols as usize * element_size) as u64;
        
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

    const WRITE_CHUNK_SIZE: usize = 8 * 1024 * 1024;  // 8MB write chunk size

    pub fn save_arrays<T: Element + Copy + Send + Sync>(&self, arrays: &[(String, Array2<T>, DataType)]) -> NpkResult<()> {
        // Parallel process array writing and collect metadata
        let metadata_updates: Vec<_> = arrays.par_iter()
            .map(|(name, array, dtype)| -> NpkResult<(String, ArrayMetadata)> {
                let data_file = format!("data_{}.npkd", name);
                let data_path = self.base_dir.join(&data_file);
                
                // Calculate total size
                let total_size = array.shape()[0] * array.shape()[1] * std::mem::size_of::<T>();
                
                // Create and preallocate file
                let file = OpenOptions::new()
                    .read(true)
                    .write(true)
                    .create(true)
                    .open(&data_path)?;
                file.set_len(total_size as u64)?;
                
                // Use direct write instead of memory mapping
                let mut writer = BufWriter::with_capacity(Self::WRITE_CHUNK_SIZE, &file);
                
                // Get data pointer
                let data_ptr = array.as_ptr() as *const u8;
                let mut offset = 0;
                
                // Write data in chunks
                while offset < total_size {
                    let chunk_size = std::cmp::min(Self::WRITE_CHUNK_SIZE, total_size - offset);
                    let chunk = unsafe {
                        std::slice::from_raw_parts(data_ptr.add(offset), chunk_size)
                    };
                    
                    writer.write_all(chunk)?;
                    
                    offset += chunk_size;
                }
                
                // Ensure data is written to disk
                writer.flush()?;
                
                // Create metadata
                let meta = ArrayMetadata::new(
                    name.clone(),
                    array.shape()[0] as u64,
                    array.shape()[1] as u64,
                    data_file,
                    *dtype,
                );
                
                Ok((name.clone(), meta))
            })
            .collect::<Result<Vec<_>, _>>()?;
        
        // Batch update metadata
        for (_name, meta) in metadata_updates {
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

        // Parallel create views
        arrays_to_load.into_par_iter()
            .map(|(name, meta)| {
                let data_path = self.base_dir.join(&meta.data_file);
                let file = File::open(&data_path)?;
                let view = ArrayView::new(meta, file, data_path.clone(), mmap_mode);
                Ok((name, view))
            })
            .collect()
    }

    pub fn get_array_meta(&self, name: &str) -> Option<ArrayMetadata> {
        self.metadata.get_array(name)
    }

    pub fn list_arrays(&self) -> Vec<String> {
        self.metadata.list_arrays()
    }

    #[allow(dead_code)]
    fn compact(&self) -> NpkResult<()> {
        let active_arrays: Vec<_> = self.metadata.list_arrays();
        
        // Create temporary directory
        let temp_dir = self.base_dir.join(".temp_compact");
        std::fs::create_dir_all(&temp_dir)?;
        
        // Copy active arrays to new files
        for name in &active_arrays {
            if let Some(meta) = self.metadata.get_array(name) {
                let old_path = self.base_dir.join(&meta.data_file);
                let new_path = temp_dir.join(&meta.data_file);
                
                // Copy data file
                std::fs::copy(old_path, new_path)?;
            }
        }
        
        // Delete all old files
        for entry in std::fs::read_dir(&self.base_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().map_or(false, |ext| ext == "npkd") {
                std::fs::remove_file(path)?;
            }
        }
        
        // Move new files to original directory
        for entry in std::fs::read_dir(&temp_dir)? {
            let entry = entry?;
            let old_path = entry.path();
            let new_path = self.base_dir.join(entry.file_name());
            std::fs::rename(old_path, new_path)?;
        }
        
        // Delete temporary directory
        std::fs::remove_dir(&temp_dir)?;
        
        // Update metadata
        self.metadata.force_sync()?;
        
        Ok(())
    }

    pub fn reset(&self) -> NpkResult<()> {
        self.metadata.reset()?;
        // Delete array files
        for entry in std::fs::read_dir(&self.base_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().map_or(false, |ext| ext == "npkd") {
                std::fs::remove_file(path)?;
            }
        }
        Ok(())
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
                // Physical delete specified rows
                view.physical_delete(indices)?;
                // Update metadata
                self.metadata.update_array_metadata(name, view.meta)?;
            } else {
                // If no specified row indices, delete the entire array
                std::fs::remove_file(&data_path)?;
                // Delete array from metadata
                self.metadata.delete_array(name)?;
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

    // Check if indices are continuous
    fn is_continuous_indices(indices: &[i64]) -> Option<(i64, usize)> {
        if indices.is_empty() {
            return None;
        }
        
        let start = indices[0];
        let len = indices.len();
        
        for (i, &idx) in indices.iter().enumerate() {
            if idx != start + i as i64 {
                return None;
            }
        }
        
        Some((start, len))
    }

    const REPLACE_CHUNK_SIZE: usize = 1024;  // Process 1024 rows per batch
    const REPLACE_DISTANCE_THRESHOLD: u64 = 1024 * 1024;  // 1MB distance threshold

    // Group indices, putting nearby indices together
    fn group_indices(indices: &[i64], row_size: usize, rows: u64) -> Vec<Vec<(usize, i64)>> {
        let mut indexed: Vec<_> = indices.iter()
            .enumerate()
            .map(|(i, &idx)| {
                let normalized_idx = if idx < 0 {
                    (rows as i64 + idx) as u64
                } else {
                    idx as u64
                };
                (i, normalized_idx)
            })
            .collect();

        // Sort by file offset
        indexed.sort_by_key(|&(_, idx)| idx);

        let mut groups = Vec::new();
        let mut current_group = Vec::new();
        let mut last_offset = None;

        for (i, idx) in indexed {
            match last_offset {
                Some(last) => {
                    let distance = (idx as u64 - last) * row_size as u64;
                    if distance > Self::REPLACE_DISTANCE_THRESHOLD || current_group.len() >= Self::REPLACE_CHUNK_SIZE {
                        if !current_group.is_empty() {
                            groups.push(std::mem::take(&mut current_group));
                        }
                    }
                }
                None => {}
            }
            current_group.push((i, idx as i64));
            last_offset = Some(idx);
        }

        if !current_group.is_empty() {
            groups.push(current_group);
        }

        groups
    }

    pub fn replace_rows<T: Element + Copy + Send + Sync>(
        &self,
        name: &str,
        data: &Array2<T>,
        indices: &[i64]
    ) -> NpkResult<()> {
        let meta = self.get_array_meta(name).ok_or_else(|| {
            io::Error::new(io::ErrorKind::NotFound, format!("Array {} not found", name))
        })?;
        
        let element_size = std::mem::size_of::<T>();
        let row_size = meta.cols as usize * element_size;
        
        // Open file for writing
        let file_path = self.base_dir.join(&meta.data_file);
        let file = OpenOptions::new()
            .write(true)
            .open(&file_path)?;
            
        // Check if indices are continuous
        if let Some((start, len)) = Self::is_continuous_indices(indices) {
            let normalized_start = if start < 0 {
                (meta.rows as i64 + start) as u64
            } else {
                start as u64
            };
            
            if normalized_start + len as u64 > meta.rows {
                return Err(NpkError::IoError(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("Index range {}:{} is out of bounds", start, start + len as i64)
                )));
            }
            
            // For continuous indices, use one-time write
            let offset = normalized_start * row_size as u64;
            let data_slice = unsafe {
                std::slice::from_raw_parts(
                    data.as_ptr() as *const u8,
                    len * row_size
                )
            };
            file.write_at(data_slice, offset)?;
        } else {
            // For non-continuous indices, use grouped batch processing
            let groups = Self::group_indices(indices, row_size, meta.rows);
            
            // Parallel process each group
            groups.par_iter().try_for_each(|group| {
                let local_file = file.try_clone()?;
                let mut buffer = Vec::with_capacity(group.len() * row_size);
                
                // Collect data for all rows in the group
                for &(i, idx) in group {
                    let normalized_idx = if idx < 0 {
                        (meta.rows as i64 + idx) as u64
                    } else {
                        idx as u64
                    };
                    
                    if normalized_idx >= meta.rows {
                        return Err(NpkError::IoError(io::Error::new(
                            io::ErrorKind::InvalidInput,
                            format!("Index {} is out of bounds", idx)
                        )));
                    }
                    
                    // Get row data
                    let row_data = unsafe {
                        std::slice::from_raw_parts(
                            data.as_ptr().add(i * meta.cols as usize) as *const u8,
                            row_size
                        )
                    };
                    buffer.extend_from_slice(row_data);
                }
                
                // Write all data in the group at once
                let first_idx = if group[0].1 < 0 {
                    (meta.rows as i64 + group[0].1) as u64
                } else {
                    group[0].1 as u64
                };
                let offset = first_idx * row_size as u64;
                local_file.write_at(&buffer, offset)?;
                
                Ok(())
            })?;
        }
        
        Ok(())
    }
}
