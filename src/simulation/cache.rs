//! High-performance caching system for photonic simulation data

use crate::core::{Result, PhotonicError, OpticalField};
use nalgebra::{DVector, DMatrix};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, SystemTime};
use serde::{Serialize, Deserialize};
use bincode;

/// Cache key for simulation results
#[derive(Debug, Clone, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub struct CacheKey {
    /// Hash of input parameters
    pub param_hash: u64,
    /// Simulation type identifier
    pub sim_type: String,
    /// Version for cache invalidation
    pub version: u32,
}

/// Cached simulation result with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedResult<T> {
    /// The cached data
    pub data: T,
    /// Timestamp when cached
    pub timestamp: SystemTime,
    /// Number of times accessed
    pub access_count: u64,
    /// Time taken to compute originally
    pub compute_time: Duration,
    /// Memory size estimate in bytes
    pub size_bytes: usize,
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub total_entries: usize,
    pub memory_usage: usize,
    pub hit_rate: f64,
}

/// Memory-based LRU cache for simulation results
pub struct MemoryCache<T> {
    /// Cache storage
    cache: Arc<RwLock<HashMap<CacheKey, CachedResult<T>>>>,
    /// Access order for LRU eviction
    access_order: Arc<RwLock<Vec<CacheKey>>>,
    /// Maximum number of entries
    max_entries: usize,
    /// Maximum memory usage in bytes
    max_memory: usize,
    /// Cache statistics
    stats: Arc<RwLock<CacheStats>>,
    /// Default TTL for cache entries
    default_ttl: Duration,
}

impl<T> MemoryCache<T> 
where 
    T: Clone + Serialize + for<'de> Deserialize<'de>
{
    /// Create new memory cache
    pub fn new(max_entries: usize, max_memory: usize, default_ttl: Duration) -> Self {
        Self {
            cache: Arc::new(RwLock::new(HashMap::new())),
            access_order: Arc::new(RwLock::new(Vec::new())),
            max_entries,
            max_memory,
            stats: Arc::new(RwLock::new(CacheStats {
                hits: 0,
                misses: 0,
                evictions: 0,
                total_entries: 0,
                memory_usage: 0,
                hit_rate: 0.0,
            })),
            default_ttl,
        }
    }
    
    /// Get value from cache
    pub fn get(&self, key: &CacheKey) -> Option<T> {
        let mut cache = self.cache.write().unwrap();
        let mut access_order = self.access_order.write().unwrap();
        let mut stats = self.stats.write().unwrap();
        
        if let Some(cached) = cache.get_mut(key) {
            // Check if expired
            if cached.timestamp.elapsed().unwrap_or(Duration::MAX) > self.default_ttl {
                cache.remove(key);
                if let Some(pos) = access_order.iter().position(|k| k == key) {
                    access_order.remove(pos);
                }
                stats.misses += 1;
                return None;
            }
            
            // Update access information
            cached.access_count += 1;
            
            // Move to end of access order (most recently used)
            if let Some(pos) = access_order.iter().position(|k| k == key) {
                access_order.remove(pos);
            }
            access_order.push(key.clone());
            
            stats.hits += 1;
            stats.hit_rate = stats.hits as f64 / (stats.hits + stats.misses) as f64;
            
            Some(cached.data.clone())
        } else {
            stats.misses += 1;
            stats.hit_rate = stats.hits as f64 / (stats.hits + stats.misses) as f64;
            None
        }
    }
    
    /// Put value into cache
    pub fn put(&self, key: CacheKey, value: T, compute_time: Duration) -> Result<()> {
        let serialized_size = bincode::serialized_size(&value).unwrap_or(1024) as usize;
        
        let mut cache = self.cache.write().unwrap();
        let mut access_order = self.access_order.write().unwrap();
        let mut stats = self.stats.write().unwrap();
        
        // Check if we need to evict entries
        self.evict_if_needed(&mut cache, &mut access_order, &mut stats, serialized_size)?;
        
        let cached_result = CachedResult {
            data: value,
            timestamp: SystemTime::now(),
            access_count: 0,
            compute_time,
            size_bytes: serialized_size,
        };
        
        // Remove from access order if already exists
        if let Some(pos) = access_order.iter().position(|k| k == &key) {
            access_order.remove(pos);
        }
        
        // Add to cache and access order
        cache.insert(key.clone(), cached_result);
        access_order.push(key);
        
        stats.total_entries = cache.len();
        stats.memory_usage += serialized_size;
        
        Ok(())
    }
    
    /// Evict entries if cache limits exceeded
    fn evict_if_needed(
        &self,
        cache: &mut HashMap<CacheKey, CachedResult<T>>,
        access_order: &mut Vec<CacheKey>,
        stats: &mut CacheStats,
        new_entry_size: usize,
    ) -> Result<()> {
        // Evict based on memory limit
        while stats.memory_usage + new_entry_size > self.max_memory && !access_order.is_empty() {
            let lru_key = access_order.remove(0);
            if let Some(removed) = cache.remove(&lru_key) {
                stats.memory_usage = stats.memory_usage.saturating_sub(removed.size_bytes);
                stats.evictions += 1;
            }
        }
        
        // Evict based on entry count limit
        while cache.len() >= self.max_entries && !access_order.is_empty() {
            let lru_key = access_order.remove(0);
            if let Some(removed) = cache.remove(&lru_key) {
                stats.memory_usage = stats.memory_usage.saturating_sub(removed.size_bytes);
                stats.evictions += 1;
            }
        }
        
        stats.total_entries = cache.len();
        Ok(())
    }
    
    /// Clear all cache entries
    pub fn clear(&self) {
        let mut cache = self.cache.write().unwrap();
        let mut access_order = self.access_order.write().unwrap();
        let mut stats = self.stats.write().unwrap();
        
        cache.clear();
        access_order.clear();
        stats.total_entries = 0;
        stats.memory_usage = 0;
    }
    
    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        self.stats.read().unwrap().clone()
    }
}

/// Persistent disk cache for large simulation results
pub struct DiskCache {
    /// Cache directory path
    cache_dir: std::path::PathBuf,
    /// Index of cached files
    index: Arc<RwLock<HashMap<CacheKey, CacheMetadata>>>,
    /// Maximum disk usage in bytes
    max_disk_usage: u64,
    /// Current disk usage
    current_usage: Arc<RwLock<u64>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CacheMetadata {
    file_path: std::path::PathBuf,
    size_bytes: u64,
    timestamp: SystemTime,
    access_count: u64,
}

impl DiskCache {
    /// Create new disk cache
    pub fn new<P: AsRef<std::path::Path>>(cache_dir: P, max_disk_usage: u64) -> Result<Self> {
        let cache_dir = cache_dir.as_ref().to_path_buf();
        
        // Create cache directory if it doesn't exist
        std::fs::create_dir_all(&cache_dir).map_err(|e| {
            PhotonicError::simulation_error(&format!("Failed to create cache directory: {}", e))
        })?;
        
        let cache = Self {
            cache_dir,
            index: Arc::new(RwLock::new(HashMap::new())),
            max_disk_usage,
            current_usage: Arc::new(RwLock::new(0)),
        };
        
        // Load existing index
        cache.load_index()?;
        
        Ok(cache)
    }
    
    /// Load cache index from disk
    fn load_index(&self) -> Result<()> {
        let index_path = self.cache_dir.join("index.json");
        
        if index_path.exists() {
            let index_data = std::fs::read_to_string(&index_path).map_err(|e| {
                PhotonicError::simulation_error(&format!("Failed to read cache index: {}", e))
            })?;
            
            let index: HashMap<CacheKey, CacheMetadata> = serde_json::from_str(&index_data)
                .map_err(|e| {
                    PhotonicError::simulation_error(&format!("Failed to parse cache index: {}", e))
                })?;
            
            // Calculate current disk usage
            let total_usage: u64 = index.values().map(|meta| meta.size_bytes).sum();
            
            *self.index.write().unwrap() = index;
            *self.current_usage.write().unwrap() = total_usage;
        }
        
        Ok(())
    }
    
    /// Save cache index to disk
    fn save_index(&self) -> Result<()> {
        let index_path = self.cache_dir.join("index.json");
        let index = self.index.read().unwrap();
        
        let index_data = serde_json::to_string_pretty(&*index).map_err(|e| {
            PhotonicError::simulation_error(&format!("Failed to serialize cache index: {}", e))
        })?;
        
        std::fs::write(&index_path, index_data).map_err(|e| {
            PhotonicError::simulation_error(&format!("Failed to write cache index: {}", e))
        })?;
        
        Ok(())
    }
    
    /// Get value from disk cache
    pub fn get<T>(&self, key: &CacheKey) -> Result<Option<T>>
    where
        T: for<'de> Deserialize<'de>
    {
        let mut index = self.index.write().unwrap();
        
        if let Some(metadata) = index.get_mut(key) {
            // Check if file still exists
            if !metadata.file_path.exists() {
                index.remove(key);
                return Ok(None);
            }
            
            // Read and deserialize data
            let data = std::fs::read(&metadata.file_path).map_err(|e| {
                PhotonicError::simulation_error(&format!("Failed to read cached file: {}", e))
            })?;
            
            let value: T = bincode::deserialize(&data).map_err(|e| {
                PhotonicError::simulation_error(&format!("Failed to deserialize cached data: {}", e))
            })?;
            
            // Update access count
            metadata.access_count += 1;
            
            Ok(Some(value))
        } else {
            Ok(None)
        }
    }
    
    /// Put value into disk cache
    pub fn put<T>(&self, key: CacheKey, value: &T) -> Result<()>
    where
        T: Serialize
    {
        // Serialize data
        let data = bincode::serialize(value).map_err(|e| {
            PhotonicError::simulation_error(&format!("Failed to serialize data: {}", e))
        })?;
        
        let data_size = data.len() as u64;
        
        // Evict if necessary
        self.evict_if_needed(data_size)?;
        
        // Generate unique filename
        let filename = format!("cache_{:016x}.bin", self.hash_key(&key));
        let file_path = self.cache_dir.join(filename);
        
        // Write data to file
        std::fs::write(&file_path, &data).map_err(|e| {
            PhotonicError::simulation_error(&format!("Failed to write cache file: {}", e))
        })?;
        
        // Update index
        let metadata = CacheMetadata {
            file_path,
            size_bytes: data_size,
            timestamp: SystemTime::now(),
            access_count: 0,
        };
        
        let mut index = self.index.write().unwrap();
        index.insert(key, metadata);
        
        *self.current_usage.write().unwrap() += data_size;
        
        // Save index
        drop(index);
        self.save_index()?;
        
        Ok(())
    }
    
    /// Evict entries if disk usage exceeds limit
    fn evict_if_needed(&self, new_size: u64) -> Result<()> {
        let mut current_usage = self.current_usage.write().unwrap();
        
        if *current_usage + new_size <= self.max_disk_usage {
            return Ok(());
        }
        
        let mut index = self.index.write().unwrap();
        
        // Sort by access count and timestamp (LRU with access frequency)
        let mut entries: Vec<_> = index.iter().collect();
        entries.sort_by_key(|(_, meta)| (meta.access_count, meta.timestamp));
        
        // Remove entries until we have enough space
        for (key, metadata) in entries {
            if *current_usage + new_size <= self.max_disk_usage {
                break;
            }
            
            // Remove file
            if metadata.file_path.exists() {
                std::fs::remove_file(&metadata.file_path).ok();
            }
            
            *current_usage = current_usage.saturating_sub(metadata.size_bytes);
            index.remove(key);
        }
        
        Ok(())
    }
    
    /// Hash cache key to generate filename
    fn hash_key(&self, key: &CacheKey) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        key.hash(&mut hasher);
        hasher.finish()
    }
    
    /// Clear all cached data
    pub fn clear(&self) -> Result<()> {
        let mut index = self.index.write().unwrap();
        
        // Remove all cache files
        for metadata in index.values() {
            if metadata.file_path.exists() {
                std::fs::remove_file(&metadata.file_path).ok();
            }
        }
        
        index.clear();
        *self.current_usage.write().unwrap() = 0;
        
        // Save empty index
        drop(index);
        self.save_index()?;
        
        Ok(())
    }
}

/// Two-tier cache system combining memory and disk caches
pub struct TieredCache<T> {
    memory_cache: MemoryCache<T>,
    disk_cache: DiskCache,
}

impl<T> TieredCache<T> 
where
    T: Clone + Serialize + for<'de> Deserialize<'de>
{
    /// Create new tiered cache
    pub fn new(
        memory_entries: usize,
        memory_size: usize,
        memory_ttl: Duration,
        disk_dir: &std::path::Path,
        disk_size: u64,
    ) -> Result<Self> {
        Ok(Self {
            memory_cache: MemoryCache::new(memory_entries, memory_size, memory_ttl),
            disk_cache: DiskCache::new(disk_dir, disk_size)?,
        })
    }
    
    /// Get value from cache (memory first, then disk)
    pub fn get(&self, key: &CacheKey) -> Result<Option<T>> {
        // Try memory cache first
        if let Some(value) = self.memory_cache.get(key) {
            return Ok(Some(value));
        }
        
        // Try disk cache
        if let Some(value) = self.disk_cache.get(key)? {
            // Promote to memory cache
            self.memory_cache.put(key.clone(), value.clone(), Duration::from_millis(0))?;
            return Ok(Some(value));
        }
        
        Ok(None)
    }
    
    /// Put value into cache (both memory and disk)
    pub fn put(&self, key: CacheKey, value: T, compute_time: Duration) -> Result<()> {
        // Store in memory cache
        self.memory_cache.put(key.clone(), value.clone(), compute_time)?;
        
        // Store in disk cache for persistence
        self.disk_cache.put(key, &value)?;
        
        Ok(())
    }
    
    /// Clear both caches
    pub fn clear(&self) -> Result<()> {
        self.memory_cache.clear();
        self.disk_cache.clear()?;
        Ok(())
    }
    
    /// Get combined cache statistics
    pub fn stats(&self) -> CacheStats {
        self.memory_cache.stats()
    }
}

/// Cache manager for different types of simulation results
pub struct SimulationCacheManager {
    /// Cache for optical field results
    optical_field_cache: TieredCache<OpticalField>,
    /// Cache for device parameter optimization results
    parameter_cache: TieredCache<DVector<f64>>,
    /// Cache for neural network weights
    weights_cache: TieredCache<DMatrix<f64>>,
    /// Cache for simulation metadata
    metadata_cache: MemoryCache<HashMap<String, String>>,
}

impl SimulationCacheManager {
    /// Create new cache manager
    pub fn new(cache_dir: &std::path::Path) -> Result<Self> {
        let optical_field_dir = cache_dir.join("optical_fields");
        let parameter_dir = cache_dir.join("parameters");
        let weights_dir = cache_dir.join("weights");
        
        Ok(Self {
            optical_field_cache: TieredCache::new(
                1000,                           // 1000 entries in memory
                100 * 1024 * 1024,             // 100MB memory limit
                Duration::from_secs(3600),     // 1 hour TTL
                &optical_field_dir,
                1024 * 1024 * 1024,            // 1GB disk limit
            )?,
            parameter_cache: TieredCache::new(
                500,                            // 500 entries in memory
                50 * 1024 * 1024,              // 50MB memory limit
                Duration::from_secs(7200),     // 2 hour TTL
                &parameter_dir,
                512 * 1024 * 1024,             // 512MB disk limit
            )?,
            weights_cache: TieredCache::new(
                100,                            // 100 entries in memory
                200 * 1024 * 1024,             // 200MB memory limit
                Duration::from_secs(1800),     // 30 minute TTL
                &weights_dir,
                2048 * 1024 * 1024,            // 2GB disk limit
            )?,
            metadata_cache: MemoryCache::new(
                10000,                          // 10000 entries
                10 * 1024 * 1024,              // 10MB memory limit
                Duration::from_secs(600),      // 10 minute TTL
            ),
        })
    }
    
    /// Cache optical field simulation result
    pub fn cache_optical_field(&self, key: CacheKey, field: OpticalField, compute_time: Duration) -> Result<()> {
        self.optical_field_cache.put(key, field, compute_time)
    }
    
    /// Get cached optical field
    pub fn get_optical_field(&self, key: &CacheKey) -> Result<Option<OpticalField>> {
        self.optical_field_cache.get(key)
    }
    
    /// Cache device parameters
    pub fn cache_parameters(&self, key: CacheKey, params: DVector<f64>, compute_time: Duration) -> Result<()> {
        self.parameter_cache.put(key, params, compute_time)
    }
    
    /// Get cached parameters
    pub fn get_parameters(&self, key: &CacheKey) -> Result<Option<DVector<f64>>> {
        self.parameter_cache.get(key)
    }
    
    /// Cache neural network weights
    pub fn cache_weights(&self, key: CacheKey, weights: DMatrix<f64>, compute_time: Duration) -> Result<()> {
        self.weights_cache.put(key, weights, compute_time)
    }
    
    /// Get cached weights
    pub fn get_weights(&self, key: &CacheKey) -> Result<Option<DMatrix<f64>>> {
        self.weights_cache.get(key)
    }
    
    /// Cache simulation metadata
    pub fn cache_metadata(&self, key: CacheKey, metadata: HashMap<String, String>) -> Result<()> {
        self.metadata_cache.put(key, metadata, Duration::from_millis(0))
    }
    
    /// Get cached metadata
    pub fn get_metadata(&self, key: &CacheKey) -> Option<HashMap<String, String>> {
        self.metadata_cache.get(key)
    }
    
    /// Clear all caches
    pub fn clear_all(&self) -> Result<()> {
        self.optical_field_cache.clear()?;
        self.parameter_cache.clear()?;
        self.weights_cache.clear()?;
        self.metadata_cache.clear();
        Ok(())
    }
    
    /// Get comprehensive cache statistics
    pub fn get_all_stats(&self) -> HashMap<String, CacheStats> {
        let mut stats = HashMap::new();
        stats.insert("optical_fields".to_string(), self.optical_field_cache.stats());
        stats.insert("parameters".to_string(), self.parameter_cache.stats());
        stats.insert("weights".to_string(), self.weights_cache.stats());
        stats.insert("metadata".to_string(), self.metadata_cache.stats());
        stats
    }
}

/// Utility functions for generating cache keys
pub mod cache_keys {
    use super::*;
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    
    /// Generate cache key for optical field simulation
    pub fn optical_field_key(
        input_hash: u64,
        device_params: &[f64],
        wavelength: f64,
        version: u32,
    ) -> CacheKey {
        let mut hasher = DefaultHasher::new();
        input_hash.hash(&mut hasher);
        device_params.hash(&mut hasher);
        wavelength.to_bits().hash(&mut hasher);
        
        CacheKey {
            param_hash: hasher.finish(),
            sim_type: "optical_field".to_string(),
            version,
        }
    }
    
    /// Generate cache key for device optimization
    pub fn optimization_key(
        objective_hash: u64,
        constraints: &[f64],
        algorithm: &str,
        version: u32,
    ) -> CacheKey {
        let mut hasher = DefaultHasher::new();
        objective_hash.hash(&mut hasher);
        constraints.hash(&mut hasher);
        algorithm.hash(&mut hasher);
        
        CacheKey {
            param_hash: hasher.finish(),
            sim_type: "optimization".to_string(),
            version,
        }
    }
    
    /// Generate cache key for neural network training
    pub fn training_key(
        network_arch: &[usize],
        data_hash: u64,
        hyperparams: &[f64],
        version: u32,
    ) -> CacheKey {
        let mut hasher = DefaultHasher::new();
        network_arch.hash(&mut hasher);
        data_hash.hash(&mut hasher);
        hyperparams.hash(&mut hasher);
        
        CacheKey {
            param_hash: hasher.finish(),
            sim_type: "training".to_string(),
            version,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    
    #[test]
    fn test_memory_cache() {
        let cache = MemoryCache::new(10, 1024, Duration::from_secs(60));
        
        let key = CacheKey {
            param_hash: 12345,
            sim_type: "test".to_string(),
            version: 1,
        };
        
        // Test miss
        assert!(cache.get(&key).is_none());
        
        // Test put and hit
        cache.put(key.clone(), "test_value".to_string(), Duration::from_millis(100)).unwrap();
        assert_eq!(cache.get(&key), Some("test_value".to_string()));
        
        // Test stats
        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
    }
    
    #[test]
    fn test_cache_key_generation() {
        let key1 = cache_keys::optical_field_key(123, &[1.0, 2.0, 3.0], 1550e-9, 1);
        let key2 = cache_keys::optical_field_key(123, &[1.0, 2.0, 3.0], 1550e-9, 1);
        let key3 = cache_keys::optical_field_key(123, &[1.0, 2.0, 4.0], 1550e-9, 1);
        
        assert_eq!(key1, key2);
        assert_ne!(key1, key3);
    }
}