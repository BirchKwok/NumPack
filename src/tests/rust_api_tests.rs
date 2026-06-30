//! Tests for the pure Rust API methods on ParallelIO
//! that mirror the Python NumPack data-operation API.

use ndarray::{Array2, ArrayD};
use std::fs;
use tempfile::TempDir;

use crate::core::error::NpkResult;
use crate::core::metadata::DataType;
use crate::io::ParallelIO;
use crate::storage::deletion_bitmap::DeletionBitmap;

fn create_test_io() -> (TempDir, ParallelIO) {
    let tmp = TempDir::new().unwrap();
    let io = ParallelIO::new(tmp.path().to_path_buf()).unwrap();
    (tmp, io)
}

fn save_f32_array(io: &ParallelIO, name: &str, rows: usize, cols: usize) -> ArrayD<f32> {
    let data =
        Array2::<f32>::from_shape_fn((rows, cols), |(r, c)| (r * cols + c) as f32).into_dyn();
    io.save_arrays(&[(name.to_string(), data.clone(), DataType::Float32)])
        .unwrap();
    io.sync_metadata().unwrap();
    data
}

// =========================================================================
// append_rows
// =========================================================================

#[test]
fn test_append_rows_basic() {
    let (_tmp, io) = create_test_io();
    let _ = save_f32_array(&io, "arr", 10, 5);

    let extra = Array2::<f32>::ones((3, 5)).into_dyn();
    io.append_rows("arr", &extra).unwrap();

    let shape = io.get_shape("arr").unwrap();
    assert_eq!(shape, vec![13, 5]);

    let loaded: ArrayD<f32> = io.load_array("arr").unwrap();
    assert_eq!(loaded.shape(), &[13, 5]);
}

#[test]
fn test_append_rows_shape_mismatch() {
    let (_tmp, io) = create_test_io();
    let _ = save_f32_array(&io, "arr", 10, 5);

    let bad = Array2::<f32>::ones((3, 7)).into_dyn();
    let result = io.append_rows("arr", &bad);
    assert!(result.is_err());
}

#[test]
fn test_append_rows_not_found() {
    let (_tmp, io) = create_test_io();
    let data = Array2::<f32>::ones((3, 5)).into_dyn();
    let result = io.append_rows("nonexistent", &data);
    assert!(result.is_err());
}

#[test]
fn test_append_rows_with_bitmap() {
    let (_tmp, io) = create_test_io();
    let _ = save_f32_array(&io, "arr", 10, 5);

    // Drop row 0 (creates bitmap)
    io.drop_arrays("arr", Some(&[0])).unwrap();
    let shape_after_drop = io.get_shape("arr").unwrap();
    assert_eq!(shape_after_drop, vec![9, 5]);

    // Append
    let extra = Array2::<f32>::ones((5, 5)).into_dyn();
    io.append_rows("arr", &extra).unwrap();

    let shape = io.get_shape("arr").unwrap();
    assert_eq!(shape, vec![14, 5]);
}

// =========================================================================
// load_array
// =========================================================================

#[test]
fn test_load_array_basic() {
    let (_tmp, io) = create_test_io();
    let original = save_f32_array(&io, "arr", 20, 4);

    let loaded: ArrayD<f32> = io.load_array("arr").unwrap();
    assert_eq!(loaded.shape(), original.shape());
    assert_eq!(loaded, original);
}

#[test]
fn test_load_array_with_deletions() {
    let (_tmp, io) = create_test_io();
    let _ = save_f32_array(&io, "arr", 10, 3);

    io.drop_arrays("arr", Some(&[0, 5])).unwrap();

    let loaded: ArrayD<f32> = io.load_array("arr").unwrap();
    assert_eq!(loaded.shape(), &[8, 3]);
}

#[test]
fn test_load_array_not_found() {
    let (_tmp, io) = create_test_io();
    let result: NpkResult<ArrayD<f32>> = io.load_array("nonexistent");
    assert!(result.is_err());
}

// =========================================================================
// getitem
// =========================================================================

#[test]
fn test_getitem_basic() {
    let (_tmp, io) = create_test_io();
    let original = save_f32_array(&io, "arr", 10, 5);

    let rows: ArrayD<f32> = io.getitem("arr", &[0, 3, 9]).unwrap();
    assert_eq!(rows.shape(), &[3, 5]);

    // Verify content
    for c in 0..5 {
        assert_eq!(rows[[0, c]], original[[0, c]]);
        assert_eq!(rows[[1, c]], original[[3, c]]);
        assert_eq!(rows[[2, c]], original[[9, c]]);
    }
}

#[test]
fn test_getitem_negative_index() {
    let (_tmp, io) = create_test_io();
    let original = save_f32_array(&io, "arr", 10, 5);

    let rows: ArrayD<f32> = io.getitem("arr", &[-1]).unwrap();
    assert_eq!(rows.shape(), &[1, 5]);
    for c in 0..5 {
        assert_eq!(rows[[0, c]], original[[9, c]]);
    }
}

// =========================================================================
// get_shape
// =========================================================================

#[test]
fn test_get_shape_basic() {
    let (_tmp, io) = create_test_io();
    let _ = save_f32_array(&io, "arr", 100, 8);

    let shape = io.get_shape("arr").unwrap();
    assert_eq!(shape, vec![100, 8]);
}

#[test]
fn test_get_shape_with_deletions() {
    let (_tmp, io) = create_test_io();
    let _ = save_f32_array(&io, "arr", 100, 8);

    io.drop_arrays("arr", Some(&[0, 1, 2])).unwrap();

    let shape = io.get_shape("arr").unwrap();
    assert_eq!(shape, vec![97, 8]);
}

// =========================================================================
// get_modify_time
// =========================================================================

#[test]
fn test_get_modify_time() {
    let (_tmp, io) = create_test_io();
    let _ = save_f32_array(&io, "arr", 10, 5);

    let time = io.get_modify_time("arr");
    assert!(time.is_some());
    assert!(time.unwrap() > 0);
}

#[test]
fn test_get_modify_time_not_found() {
    let (_tmp, io) = create_test_io();
    let time = io.get_modify_time("nonexistent");
    assert!(time.is_none());
}

// =========================================================================
// clone_array
// =========================================================================

#[test]
fn test_clone_array_basic() {
    let (_tmp, io) = create_test_io();
    let original = save_f32_array(&io, "source", 10, 5);

    io.clone_array("source", "target").unwrap();
    io.sync_metadata().unwrap();

    assert!(io.has_array("target"));

    let loaded: ArrayD<f32> = io.load_array("target").unwrap();
    assert_eq!(loaded, original);
}

#[test]
fn test_clone_array_independent() {
    let (_tmp, io) = create_test_io();
    let _ = save_f32_array(&io, "source", 10, 5);

    io.clone_array("source", "target").unwrap();
    io.sync_metadata().unwrap();

    // Modify target
    let new_data = Array2::<f32>::zeros((10, 5)).into_dyn();
    let indices: Vec<i64> = (0..10).collect();
    io.replace_rows("target", &new_data, &indices).unwrap();

    // Source should be unchanged
    let source_loaded: ArrayD<f32> = io.load_array("source").unwrap();
    let target_loaded: ArrayD<f32> = io.load_array("target").unwrap();
    assert_ne!(source_loaded, target_loaded);
}

#[test]
fn test_clone_array_source_not_found() {
    let (_tmp, io) = create_test_io();
    let result = io.clone_array("nonexistent", "target");
    assert!(result.is_err());
}

#[test]
fn test_clone_array_target_exists() {
    let (_tmp, io) = create_test_io();
    let _ = save_f32_array(&io, "source", 10, 5);
    let _ = save_f32_array(&io, "target", 5, 5);

    let result = io.clone_array("source", "target");
    assert!(result.is_err());
}

// =========================================================================
// get_member_list / update aliases
// =========================================================================

#[test]
fn test_get_member_list() {
    let (_tmp, io) = create_test_io();
    let _ = save_f32_array(&io, "a", 5, 3);
    let _ = save_f32_array(&io, "b", 5, 3);

    let mut members = io.get_member_list();
    members.sort();
    assert_eq!(members, vec!["a".to_string(), "b".to_string()]);
}

#[test]
fn test_update_alias() {
    let (_tmp, io) = create_test_io();
    let _ = save_f32_array(&io, "arr", 10, 5);

    io.drop_arrays("arr", Some(&[0, 1])).unwrap();
    io.update("arr").unwrap();

    let shape = io.get_shape("arr").unwrap();
    assert_eq!(shape, vec![8, 5]);
}

#[test]
fn test_drop_entire_array_is_logical_until_compact() {
    let (_tmp, io) = create_test_io();
    let _ = save_f32_array(&io, "arr", 10, 3);
    let meta = io.get_array_meta("arr").unwrap();
    let data_path = io.get_base_dir().join(&meta.data_file);
    let original_size = fs::metadata(&data_path).unwrap().len();

    io.drop_arrays("arr", None).unwrap();

    assert!(io.has_array("arr"));
    assert_eq!(io.get_shape("arr").unwrap(), vec![0, 3]);
    assert!(data_path.exists());
    assert_eq!(fs::metadata(&data_path).unwrap().len(), original_size);

    let loaded: ArrayD<f32> = io.load_array("arr").unwrap();
    assert_eq!(loaded.shape(), &[0, 3]);

    io.update("arr").unwrap();

    assert!(io.has_array("arr"));
    assert_eq!(io.get_shape("arr").unwrap(), vec![0, 3]);
    assert!(data_path.exists());
    assert_eq!(fs::metadata(&data_path).unwrap().len(), 0);
    assert!(!DeletionBitmap::exists(io.get_base_dir(), "arr"));
}

#[test]
fn test_segmented_storage_roundtrip_and_logical_compact() {
    let tmp = TempDir::new().unwrap();
    let io = ParallelIO::new(tmp.path().to_path_buf())
        .unwrap()
        .with_segment_target_bytes(64);
    let data = Array2::<f32>::from_shape_fn((20, 4), |(r, c)| (r * 4 + c) as f32).into_dyn();
    io.save_arrays(&[("seg".to_string(), data.clone(), DataType::Float32)])
        .unwrap();
    io.sync_metadata().unwrap();

    let meta = io.get_array_meta("seg").unwrap();
    assert!(meta.data_file.ends_with(".npkseg"));
    assert!(tmp.path().join("data_seg_segments").exists());

    let loaded: ArrayD<f32> = io.load_array("seg").unwrap();
    assert_eq!(loaded, data);

    io.clone_array("seg", "seg_clone").unwrap();
    assert!(io
        .get_array_meta("seg_clone")
        .unwrap()
        .data_file
        .ends_with(".npkseg"));
    let cloned: ArrayD<f32> = io.load_array("seg_clone").unwrap();
    assert_eq!(cloned, data);

    let rows: ArrayD<f32> = io.getitem("seg", &[0, 7, 19]).unwrap();
    assert_eq!(rows.shape(), &[3, 4]);
    assert_eq!(rows[[1, 0]], 28.0);
    assert_eq!(rows[[2, 3]], 79.0);

    let extra = Array2::<f32>::from_elem((3, 4), -5.0).into_dyn();
    io.append_rows("seg", &extra).unwrap();
    assert_eq!(io.get_shape("seg").unwrap(), vec![23, 4]);

    let data_file_before_drop = io.get_array_meta("seg").unwrap().data_file;
    io.drop_arrays("seg", Some(&[0, 1, 22])).unwrap();

    assert_eq!(io.get_shape("seg").unwrap(), vec![20, 4]);
    assert_eq!(
        io.get_array_meta("seg").unwrap().data_file,
        data_file_before_drop
    );

    let before_compact: ArrayD<f32> = io.load_array("seg").unwrap();
    assert_eq!(before_compact.shape(), &[20, 4]);
    assert_eq!(before_compact[[0, 0]], 8.0);

    io.update("seg").unwrap();

    assert_eq!(io.get_shape("seg").unwrap(), vec![20, 4]);
    assert!(io
        .get_array_meta("seg")
        .unwrap()
        .data_file
        .ends_with(".npkseg"));
    assert!(!DeletionBitmap::exists(io.get_base_dir(), "seg"));
    let after_compact: ArrayD<f32> = io.load_array("seg").unwrap();
    assert_eq!(after_compact, before_compact);

    io.reset().unwrap();
    assert!(io.list_arrays().is_empty());
    assert!(!tmp.path().join("data_seg_segments").exists());
    assert!(!tmp.path().join("data_seg_clone_segments").exists());
}

// =========================================================================
// stream_load
// =========================================================================

#[test]
fn test_stream_load_basic() {
    let (_tmp, io) = create_test_io();
    let _ = save_f32_array(&io, "arr", 25, 4);

    let iter = io.stream_load::<f32>("arr", 10).unwrap();
    let batches: Vec<_> = iter.collect::<Result<Vec<_>, _>>().unwrap();

    assert_eq!(batches.len(), 3); // 10 + 10 + 5
    assert_eq!(batches[0].shape(), &[10, 4]);
    assert_eq!(batches[1].shape(), &[10, 4]);
    assert_eq!(batches[2].shape(), &[5, 4]);

    // Total rows
    let total: usize = batches.iter().map(|b| b.shape()[0]).sum();
    assert_eq!(total, 25);
}

#[test]
fn test_stream_load_single_batch() {
    let (_tmp, io) = create_test_io();
    let original = save_f32_array(&io, "arr", 5, 3);

    let iter = io.stream_load::<f32>("arr", 100).unwrap();
    let batches: Vec<_> = iter.collect::<Result<Vec<_>, _>>().unwrap();

    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0], original);
}

// =========================================================================
// has_array
// =========================================================================

#[test]
fn test_has_array() {
    let (_tmp, io) = create_test_io();
    assert!(!io.has_array("arr"));

    let _ = save_f32_array(&io, "arr", 5, 3);
    assert!(io.has_array("arr"));
}

// =========================================================================
// reset
// =========================================================================

#[test]
fn test_reset() {
    let (_tmp, io) = create_test_io();
    let _ = save_f32_array(&io, "a", 5, 3);
    let _ = save_f32_array(&io, "b", 5, 3);

    assert_eq!(io.list_arrays().len(), 2);

    io.reset().unwrap();
    assert_eq!(io.list_arrays().len(), 0);
}

// =========================================================================
// replace_rows
// =========================================================================

#[test]
fn test_replace_rows_basic() {
    let (_tmp, io) = create_test_io();
    let _ = save_f32_array(&io, "arr", 10, 5);

    let new_data = Array2::<f32>::from_elem((2, 5), 999.0).into_dyn();
    io.replace_rows("arr", &new_data, &[0, 5]).unwrap();

    let loaded: ArrayD<f32> = io.load_array("arr").unwrap();
    assert_eq!(loaded[[0, 0]], 999.0);
    assert_eq!(loaded[[5, 0]], 999.0);
    // Untouched row
    assert_eq!(loaded[[1, 0]], 5.0); // row 1, col 0 = 1*5+0 = 5.0
}

// =========================================================================
// Integration: full workflow
// =========================================================================

#[test]
fn test_full_workflow() {
    let (_tmp, io) = create_test_io();

    // save
    let _data = save_f32_array(&io, "embeddings", 100, 16);
    assert!(io.has_array("embeddings"));
    assert_eq!(io.get_shape("embeddings").unwrap(), vec![100, 16]);

    // append
    let extra = Array2::<f32>::ones((20, 16)).into_dyn();
    io.append_rows("embeddings", &extra).unwrap();
    assert_eq!(io.get_shape("embeddings").unwrap(), vec![120, 16]);

    // getitem
    let rows: ArrayD<f32> = io.getitem("embeddings", &[0, 50, 110]).unwrap();
    assert_eq!(rows.shape(), &[3, 16]);

    // replace
    let replacement = Array2::<f32>::from_elem((1, 16), -1.0).into_dyn();
    io.replace_rows("embeddings", &replacement, &[0]).unwrap();
    let row0: ArrayD<f32> = io.getitem("embeddings", &[0]).unwrap();
    assert_eq!(row0[[0, 0]], -1.0);

    // drop rows
    io.drop_arrays("embeddings", Some(&[0, 1])).unwrap();
    assert_eq!(io.get_shape("embeddings").unwrap(), vec![118, 16]);

    // update (compact)
    io.update("embeddings").unwrap();
    assert_eq!(io.get_shape("embeddings").unwrap(), vec![118, 16]);

    // clone
    io.clone_array("embeddings", "backup").unwrap();
    io.sync_metadata().unwrap();
    assert!(io.has_array("backup"));
    assert_eq!(io.get_shape("backup").unwrap(), vec![118, 16]);

    // stream_load
    let iter = io.stream_load::<f32>("embeddings", 50).unwrap();
    let total: usize = iter
        .collect::<Result<Vec<_>, _>>()
        .unwrap()
        .iter()
        .map(|b| b.shape()[0])
        .sum();
    assert_eq!(total, 118);

    // get_member_list
    let mut members = io.get_member_list();
    members.sort();
    assert_eq!(members, vec!["backup", "embeddings"]);

    // get_modify_time
    assert!(io.get_modify_time("embeddings").is_some());

    // drop entire array
    io.drop_arrays("backup", None).unwrap();
    assert!(io.has_array("backup"));
    assert_eq!(io.get_shape("backup").unwrap(), vec![0, 16]);
    io.update("backup").unwrap();
    assert!(io.has_array("backup"));
    assert_eq!(io.get_shape("backup").unwrap(), vec![0, 16]);

    // reset
    io.reset().unwrap();
    assert_eq!(io.list_arrays().len(), 0);
}
