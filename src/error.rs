use std::fmt;
use std::io;
use std::string::FromUtf8Error;
use numpy::NotContiguousError;
use pyo3::PyErr;

#[derive(Debug)]
pub enum NnpError {
    IoError(String),
    InvalidArrayName(String),
    InvalidDtype(String),
    InvalidArrayData(String),
    InvalidArrayHeader(String),
    InvalidArrayCount(String),
    InvalidArraySize(String),
    InvalidArrayOffset(String),
    InvalidArrayType(String),
    InvalidArrayDtype(String),
    InvalidArrayDataSize(String),
    InvalidArrayDataOffset(String),
    InvalidArrayDataType(String),
    InvalidArrayDataDtype(String),
    InvalidArrayDataShape(String),
    InvalidArrayDataCount(String),
    InvalidArrayDataHeader(String),
    InvalidArrayDataFormat(String),
    SystemError(String),
    DuplicateArrayName(String),
    Utf8Error(String),
    NotContiguousError(String),
    PyError(String),
}

impl fmt::Display for NnpError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            NnpError::IoError(msg) => write!(f, "IO error: {}", msg),
            NnpError::InvalidArrayName(msg) => write!(f, "Invalid array name: {}", msg),
            NnpError::InvalidDtype(msg) => write!(f, "Invalid dtype: {}", msg),
            NnpError::InvalidArrayData(msg) => write!(f, "Invalid array data: {}", msg),
            NnpError::InvalidArrayHeader(msg) => write!(f, "Invalid array header: {}", msg),
            NnpError::InvalidArrayCount(msg) => write!(f, "Invalid array count: {}", msg),
            NnpError::InvalidArraySize(msg) => write!(f, "Invalid array size: {}", msg),
            NnpError::InvalidArrayOffset(msg) => write!(f, "Invalid array offset: {}", msg),
            NnpError::InvalidArrayType(msg) => write!(f, "Invalid array type: {}", msg),
            NnpError::InvalidArrayDtype(msg) => write!(f, "Invalid array dtype: {}", msg),
            NnpError::InvalidArrayDataSize(msg) => write!(f, "Invalid array data size: {}", msg),
            NnpError::InvalidArrayDataOffset(msg) => write!(f, "Invalid array data offset: {}", msg),
            NnpError::InvalidArrayDataType(msg) => write!(f, "Invalid array data type: {}", msg),
            NnpError::InvalidArrayDataDtype(msg) => write!(f, "Invalid array data dtype: {}", msg),
            NnpError::InvalidArrayDataShape(msg) => write!(f, "Invalid array data shape: {}", msg),
            NnpError::InvalidArrayDataCount(msg) => write!(f, "Invalid array data count: {}", msg),
            NnpError::InvalidArrayDataHeader(msg) => write!(f, "Invalid array data header: {}", msg),
            NnpError::InvalidArrayDataFormat(msg) => write!(f, "Invalid array data format: {}", msg),
            NnpError::SystemError(msg) => write!(f, "System error: {}", msg),
            NnpError::DuplicateArrayName(msg) => write!(f, "Duplicate array name: {}", msg),
            NnpError::Utf8Error(msg) => write!(f, "UTF-8 error: {}", msg),
            NnpError::NotContiguousError(msg) => write!(f, "Array not contiguous: {}", msg),
            NnpError::PyError(msg) => write!(f, "Python error: {}", msg),
        }
    }
}

impl std::error::Error for NnpError {}

impl From<io::Error> for NnpError {
    fn from(err: io::Error) -> Self {
        NnpError::IoError(err.to_string())
    }
}

impl From<FromUtf8Error> for NnpError {
    fn from(err: FromUtf8Error) -> Self {
        NnpError::Utf8Error(err.to_string())
    }
}

impl From<NotContiguousError> for NnpError {
    fn from(err: NotContiguousError) -> Self {
        NnpError::NotContiguousError(err.to_string())
    }
}

impl From<PyErr> for NnpError {
    fn from(err: PyErr) -> Self {
        NnpError::PyError(err.to_string())
    }
}

pub type NnpResult<T> = Result<T, NnpError>; 