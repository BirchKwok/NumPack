use pyo3::exceptions::PyRuntimeError;
use pyo3::PyErr;
use std::io;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum NpkError {
    #[error("IO error: {0}")]
    IoError(#[from] io::Error),

    #[error("Invalid array name: {0}")]
    InvalidArrayName(String),

    #[error("Array not found: {0}")]
    ArrayNotFound(String),

    #[error("Invalid shape: {0}")]
    InvalidShape(String),

    #[error("Invalid dtype: {0}")]
    InvalidDtype(String),

    #[error("Invalid metadata: {0}")]
    InvalidMetadata(String),

    #[error("Invalid operation: {0}")]
    InvalidOperation(String),

    #[error("Index {0} out of bounds (rows: {1})")]
    IndexOutOfBounds(i64, u64),
}

impl From<NpkError> for PyErr {
    fn from(err: NpkError) -> PyErr {
        PyRuntimeError::new_err(err.to_string())
    }
}

pub type NpkResult<T> = Result<T, NpkError>;
