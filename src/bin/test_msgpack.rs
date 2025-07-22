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
                    println!("前20字节: {:?}", &data[..20.min(data.len())]);
                    
                    match rmp_serde::from_slice::<MessagePackMetadataStore>(&data) {
                        Ok(metadata) => {
                            println!("✅ 解析成功!");
                            println!("版本: {}", metadata.version);
                            println!("总大小: {}", metadata.total_size);
                            println!("数组数量: {}", metadata.arrays.len());
                            
                            for (name, array) in &metadata.arrays {
                                println!("数组 '{}': shape={:?}, dtype={}, size={}", 
                                    name, array.shape, array.dtype, array.size_bytes);
                            }
                        }
                        Err(e) => {
                            println!("❌ 解析失败: {}", e);
                            
                            // 输出更多调试信息
                            println!("调试: 文件前100字节内容");
                            let debug_len = 100.min(data.len());
                            for chunk in data[..debug_len].chunks(16) {
                                println!("  {:02x?}", chunk);
                            }
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