//! Simulation engine and photonic array implementations

pub mod array;
pub mod result;
pub mod solver;
pub mod cache;
pub mod streaming;

pub use array::{PhotonicArray, ArrayTopology};
pub use result::{SimulationResult, SimulationMetrics};
pub use solver::{SimulationSolver, SolverConfig};
pub use cache::{
    MemoryCache, DiskCache, TieredCache, SimulationCacheManager,
    CacheKey, CachedResult, CacheStats, cache_keys
};
pub use streaming::{
    StreamProcessor, BatchProcessor, DataChunk, ProcessedChunk,
    StreamConfig, StreamStats, chunk_utils
};