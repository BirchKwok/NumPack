use std::fs::File;
use std::io::Read;
use std::collections::HashMap;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessagePackArrayMetadata {
    pub name: String,
    pub shape: Vec<u64>,
    pub data_file: String,
    pub last_modified: u64,
    pub size_bytes: u64,
    pub dtype: u8,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessagePackMetadataStore {
    pub version: u32,
    pub arrays: HashMap<String, MessagePackArrayMetadata>,
    pub total_size: u64,
}

fn main() {
    let file_path = std::env::args().nth(1).unwrap_or_else(|| {
        "test_msgpack_data/metadata.npkm".to_string()
    });
    
    println!("读取文件: {}", file_path);
    
    match File::open(&file_path) {
        Ok(mut file) => {
            let mut data = Vec::new();
            match file.read_to_end(&mut data) {
                Ok(_) => {
                    println!("文件大小: {} bytes", data.len());
                    
                    match rmp_serde::from_slice::<MessagePackMetadataStore>(&data) {
                        Ok(metadata) => {
                            println!("✅ 解析成功!");
                            println!("版本: {}", metadata.version);
                            println!("总大小: {}", metadata.total_size);
                            println!("数组数量: {}", metadata.arrays.len());
                            
                            for (name, array) in &metadata.arrays {
                                println!("数组 '{}':", name);
                                println!("  shape: {:?}", array.shape);
                                println!("  shape长度: {}", array.shape.len());
                                
                                // 计算shape乘积
                                let shape_product: u64 = array.shape.iter().product();
                                println!("  shape乘积: {}", shape_product);
                                
                                // 计算总大小 (shape_product * dtype_size)
                                let dtype_size = match array.dtype {
                                    7 => 4, // Int32 = 4 bytes
                                    _ => 1,
                                };
                                let total_size = shape_product * dtype_size;
                                println!("  计算总大小: {} * {} = {}", shape_product, dtype_size, total_size);
                                
                                println!("  dtype: {}", array.dtype);
                                println!("  size_bytes: {}", array.size_bytes);
                                println!("  data_file: {}", array.data_file);
                                println!("  last_modified: {}", array.last_modified);
                                
                                // 检查是否有异常大的值
                                for (i, &dim) in array.shape.iter().enumerate() {
                                    if dim > 1_000_000 {
                                        println!("  ⚠️  警告: shape[{}] = {} 异常大!", i, dim);
                                    }
                                }
                                
                                if array.size_bytes > 1_000_000 {
                                    println!("  ⚠️  警告: size_bytes = {} 异常大!", array.size_bytes);
                                }
                            }
                        }
                        Err(e) => {
                            println!("❌ 解析失败: {}", e);
                        }
                    }
                }
                Err(e) => {
                    println!("无法读取文件: {}", e);
                }
            }
        }
        Err(e) => {
            println!("无法打开文件: {}", e);
        }
    }
} 