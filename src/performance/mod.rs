//! High-performance optimizations and scalability features

pub mod parallel;
pub mod cache;

pub use parallel::{
    ParallelExecutor, ParallelConfig, LoadBalancingStrategy, MemoryPool
};
pub use cache::{
    PhotonicCache, CacheConfig, CacheKey, CachedResult, EvictionPolicy,
    CompressionType, PersistenceConfig, CacheStats
};