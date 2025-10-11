@0xf6a8b3c2d4e5f601;
# Cap'n Proto Schema for NumPack Metadata
# High-performance zero-copy serialization

struct ArrayMetadata {
    name @0 :Text;
    # Array identifier
    
    shape @1 :List(UInt64);
    # Array dimensions
    
    dataFile @2 :Text;
    # Relative path to data file
    
    lastModified @3 :UInt64;
    # Timestamp in microseconds since Unix epoch
    
    sizeBytes @4 :UInt64;
    # Total array size in bytes
    
    dtype @5 :UInt8;
    # Data type code (0-13)
    
    compression @6 :CompressionInfo;
    # Compression metadata
}

struct CompressionInfo {
    algorithm @0 :UInt8;
    # Compression algorithm: 0=None, 1=Zstd, 2=LZ4, 3=Snappy, 4=Gorilla, 5=FastPFOR
    
    level @1 :UInt32;
    # Compression level
    
    originalSize @2 :UInt64;
    # Uncompressed size
    
    compressedSize @3 :UInt64;
    # Compressed size
    
    blockCompression @4 :BlockCompressionInfo;
    # Optional block-level compression info
}

struct BlockCompressionInfo {
    enabled @0 :Bool;
    # Whether block compression is enabled
    
    blockSize @1 :UInt64;
    # Size of each compression block
    
    numBlocks @2 :UInt64;
    # Total number of blocks
    
    blocks @3 :List(BlockInfo);
    # Per-block metadata
}

struct BlockInfo {
    offset @0 :UInt64;
    # Offset in file
    
    originalSize @1 :UInt64;
    # Uncompressed block size
    
    compressedSize @2 :UInt64;
    # Compressed block size
}

struct MetadataStore {
    version @0 :UInt32;
    # Format version
    
    magic @1 :UInt32;
    # Magic number (0x424B504E = "NPKB")
    
    totalSize @2 :UInt64;
    # Total size of all arrays
    
    arrays @3 :List(ArrayMetadata);
    # Array metadata list
    
    timestamp @4 :UInt64;
    # Last modification time
    
    checksum @5 :UInt64;
    # Optional integrity checksum
}

