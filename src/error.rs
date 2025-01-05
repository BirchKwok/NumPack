use std::fmt;
use std::io;
use std::string::FromUtf8Error;
use numpy::NotContiguousError;
use pyo3::PyErr;
use tempfile::PersistError;
use ndarray::ShapeError;

#[derive(Debug)]
pub enum NpkError {
    IoError(io::Error),
    PyError(PyErr),
    ShapeError(ShapeError),
    ArrayNotFound(String),
    Utf8Error(FromUtf8Error),
    PersistError(PersistError),
    InvalidMetadata(String),
    IndexOutOfBounds(i64, u64),
    NotContiguousError(NotContiguousError),
    Other(String),
}

impl fmt::Display for NpkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NpkError::IoError(err) => write!(f, "IO error: {}", err),
            NpkError::PyError(err) => write!(f, "Python error: {}", err),
            NpkError::ShapeError(err) => write!(f, "Shape error: {}", err),
            NpkError::ArrayNotFound(name) => write!(f, "Array not found: {}", name),
            NpkError::Utf8Error(err) => write!(f, "UTF-8 error: {}", err),
            NpkError::PersistError(err) => write!(f, "File persist error: {}", err),
            NpkError::InvalidMetadata(msg) => write!(f, "Invalid metadata: {}", msg),
            NpkError::NotContiguousError(err) => write!(f, "Array not contiguous: {}", err),
            NpkError::IndexOutOfBounds(idx, rows) => write!(f, "Index out of bounds: {} (rows: {})", idx, rows),
            NpkError::Other(msg) => write!(f, "Other error: {}", msg),
        }
    }
}

impl std::error::Error for NpkError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            NpkError::IoError(err) => Some(err),
            NpkError::PyError(err) => Some(err),
            NpkError::ShapeError(err) => Some(err),
            NpkError::Utf8Error(err) => Some(err),
            NpkError::PersistError(err) => Some(err),
            NpkError::NotContiguousError(err) => Some(err),
            _ => None,
        }
    }
}

impl From<io::Error> for NpkError {
    fn from(err: io::Error) -> Self {
        NpkError::IoError(err)
    }
}

impl From<PyErr> for NpkError {
    fn from(err: PyErr) -> Self {
        NpkError::PyError(err)
    }
}

impl From<ShapeError> for NpkError {
    fn from(err: ShapeError) -> Self {
        NpkError::ShapeError(err)
    }
}

impl From<FromUtf8Error> for NpkError {
    fn from(err: FromUtf8Error) -> Self {
        NpkError::Utf8Error(err)
    }
}

impl From<PersistError> for NpkError {
    fn from(err: PersistError) -> Self {
        NpkError::PersistError(err)
    }
}

impl From<NotContiguousError> for NpkError {
    fn from(err: NotContiguousError) -> Self {
        NpkError::NotContiguousError(err)
    }
}

impl From<NpkError> for PyErr {
    fn from(err: NpkError) -> Self {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(err.to_string())
    }
}

pub type NpkResult<T> = Result<T, NpkError>; 