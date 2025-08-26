//! High-performance optimizations and scalability features

pub mod parallel;
pub mod cache;
pub mod hyperscale_optimization;

pub use parallel::{
    ParallelExecutor, ParallelConfig, LoadBalancingStrategy, MemoryPool
};
pub use cache::{
    PhotonicCache, CacheConfig, CacheKey, CachedResult, EvictionPolicy,
    CompressionType, PersistenceConfig, CacheStats
};
pub use hyperscale_optimization::{
    HyperscaleEngine, AutoScaler, AutoScalingConfig, IntelligentCache,
    ResourcePool, BatchProcessor, PerformanceMetrics, HyperscaleStats
};