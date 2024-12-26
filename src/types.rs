use numpy::PyArray2;
use std::result;

pub type NnpResult<T> = result::Result<T, crate::error::NnpError>;

#[derive(Debug, Clone)]
pub struct ArrayHeader {
    pub name: String,
    pub rows: u64,
    pub cols: u64,
    pub data_offset: u64,
}

#[derive(Debug, Clone)]
pub struct ArrayMeta {
    pub name: String,
    pub rows: u64,
    pub cols: u64,
    pub data_offset: u64,
}

#[derive(Debug)]
pub struct BufferManager {
    pub buffer: Vec<u8>,
    pub position: usize,
    pub capacity: usize,
}

impl BufferManager {
    pub fn new(capacity: usize) -> Self {
        BufferManager {
            buffer: vec![0; capacity],
            position: 0,
            capacity,
        }
    }

    pub fn reset(&mut self) {
        self.position = 0;
    }

    pub fn remaining(&self) -> usize {
        self.capacity - self.position
    }

    pub fn is_full(&self) -> bool {
        self.position >= self.capacity
    }

    pub fn write(&mut self, data: &[u8]) -> usize {
        let remaining = self.remaining();
        let write_size = remaining.min(data.len());
        self.buffer[self.position..self.position + write_size]
            .copy_from_slice(&data[..write_size]);
        self.position += write_size;
        write_size
    }

    pub fn read(&mut self, data: &mut [u8]) -> usize {
        let read_size = self.position.min(data.len());
        data[..read_size].copy_from_slice(&self.buffer[..read_size]);
        self.position = read_size;
        read_size
    }
}

