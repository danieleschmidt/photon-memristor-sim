//! High-performance caching system for photonic simulations

use crate::core::{Result, PhotonicError, Logger, OpticalField};
use std::collections::{HashMap, BTreeMap};
use std::sync::{Arc, RwLock, Mutex};
use std::time::{Instant, Duration, SystemTime};
use std::hash::{Hash, Hasher};
use serde::{Serialize, Deserialize};
use std::fmt;

/// Cache key for photonic simulation results
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CacheKey {
    /// Operation type identifier
    pub operation: String,
    /// Parameters hash
    pub parameters_hash: u64,
    /// Input data hash
    pub input_hash: u64,
    /// Precision level
    pub precision: u32,
}

impl CacheKey {
    /// Create cache key from parameters
    pub fn new(operation: &str, parameters: &[f64], input_data: &[u8], precision: u32) -> Self {
        use std::collections::hash_map::DefaultHasher;
        
        let mut param_hasher = DefaultHasher::new();
        for &param in parameters {
            param.to_bits().hash(&mut param_hasher);
        }
        let parameters_hash = param_hasher.finish();
        
        let mut input_hasher = DefaultHasher::new();
        input_data.hash(&mut input_hasher);
        let input_hash = input_hasher.finish();
        
        Self {
            operation: operation.to_string(),
            parameters_hash,
            input_hash,
            precision,
        }
    }
    
    /// Create key for optical field operation
    pub fn from_optical_field(operation: &str, field: &OpticalField, parameters: &[f64]) -> Self {
        use std::collections::hash_map::DefaultHasher;
        
        // Hash field data
        let mut field_hasher = DefaultHasher::new();
        field.wavelength.to_bits().hash(&mut field_hasher);
        field.power.to_bits().hash(&mut field_hasher);
        
        // Hash amplitude matrix (sample approach)
        let (rows, cols) = field.amplitude.shape();
        rows.hash(&mut field_hasher);
        cols.hash(&mut field_hasher);
        
        // Sample some elements for hash (not all for performance)
        let step = (rows * cols / 100).max(1); // Sample ~100 elements
        for i in (0..rows * cols).step_by(step) {
            let row = i / cols;
            let col = i % cols;
            if let Some(value) = field.amplitude.get((row, col)) {
                value.re.to_bits().hash(&mut field_hasher);
                value.im.to_bits().hash(&mut field_hasher);
            }
        }
        
        let input_hash = field_hasher.finish();
        
        // Hash parameters
        let mut param_hasher = DefaultHasher::new();
        for &param in parameters {
            param.to_bits().hash(&mut param_hasher);
        }
        let parameters_hash = param_hasher.finish();
        
        Self {
            operation: operation.to_string(),
            parameters_hash,
            input_hash,
            precision: 32, // Default precision
        }
    }
}

impl fmt::Display for CacheKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}:{}:{}", self.operation, self.parameters_hash, self.input_hash, self.precision)
    }
}

/// Cached result with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedResult<T> {
    /// Cached data
    pub data: T,
    /// Creation timestamp
    pub created_at: SystemTime,
    /// Last access timestamp
    pub last_accessed: SystemTime,
    /// Access count
    pub access_count: u64,
    /// Computation time that was saved (ms)
    pub computation_time_saved: f64,
    /// Size estimate in bytes
    pub size_bytes: usize,
}

impl<T> CachedResult<T> {
    /// Create new cached result
    pub fn new(data: T, computation_time_ms: f64, size_bytes: usize) -> Self {
        let now = SystemTime::now();
        Self {
            data,
            created_at: now,
            last_accessed: now,
            access_count: 0,
            computation_time_saved: computation_time_ms,
            size_bytes,
        }
    }
    
    /// Access the cached data (updates metadata)
    pub fn access(&mut self) -> &T {
        self.last_accessed = SystemTime::now();
        self.access_count += 1;
        &self.data
    }
    
    /// Get age of cached result
    pub fn age(&self) -> Duration {
        SystemTime::now().duration_since(self.created_at).unwrap_or_default()
    }
    
    /// Get time since last access
    pub fn time_since_last_access(&self) -> Duration {
        SystemTime::now().duration_since(self.last_accessed).unwrap_or_default()
    }
}

/// Cache eviction policies
#[derive(Debug, Clone, PartialEq)]
pub enum EvictionPolicy {
    /// Least Recently Used
    LRU,
    /// Least Frequently Used
    LFU,
    /// Time-based expiration
    TTL(Duration),
    /// Size-based with priority
    SizeBased,
    /// Adaptive based on access patterns
    Adaptive,
}

/// High-performance cache configuration
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Maximum cache size in bytes
    pub max_size_bytes: usize,
    /// Maximum number of entries
    pub max_entries: usize,
    /// Eviction policy
    pub eviction_policy: EvictionPolicy,
    /// Enable statistics collection
    pub enable_stats: bool,
    /// Compression for large objects
    pub compression: CompressionType,
    /// Persistence to disk
    pub persistence: PersistenceConfig,
}

/// Compression types for cache entries
#[derive(Debug, Clone, PartialEq)]
pub enum CompressionType {
    None,
    LZ4,
    Zstd,
    Adaptive, // Choose based on data characteristics
}

/// Cache persistence configuration
#[derive(Debug, Clone)]
pub struct PersistenceConfig {
    pub enabled: bool,
    pub file_path: String,
    pub sync_interval: Duration,
    pub compression: bool,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_size_bytes: 1024 * 1024 * 1024, // 1GB
            max_entries: 100_000,
            eviction_policy: EvictionPolicy::LRU,
            enable_stats: true,
            compression: CompressionType::None,
            persistence: PersistenceConfig {
                enabled: false,
                file_path: "/tmp/photonic_cache.dat".to_string(),
                sync_interval: Duration::from_secs(300), // 5 minutes
                compression: true,
            },
        }
    }
}

/// Cache statistics
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub total_size_bytes: usize,
    pub entry_count: usize,
    pub hit_rate: f64,
    pub average_access_time_ns: f64,
    pub total_computation_time_saved_ms: f64,
}

impl CacheStats {
    /// Calculate hit rate percentage
    pub fn hit_rate(&self) -> f64 {
        if self.hits + self.misses == 0 {
            0.0
        } else {
            (self.hits as f64) / ((self.hits + self.misses) as f64) * 100.0
        }
    }
    
    /// Get formatted statistics summary
    pub fn summary(&self) -> String {
        format!(
            "Cache Stats: {:.1}% hit rate, {} hits, {} misses, {} entries, {:.1}MB used, {:.1}s saved",
            self.hit_rate(),
            self.hits,
            self.misses,
            self.entry_count,
            self.total_size_bytes as f64 / (1024.0 * 1024.0),
            self.total_computation_time_saved_ms / 1000.0
        )
    }
}

/// High-performance cache implementation
pub struct PhotonicCache<T> {
    /// Cache data storage
    data: Arc<RwLock<HashMap<CacheKey, CachedResult<T>>>>,
    /// LRU tracking
    lru_order: Arc<RwLock<BTreeMap<SystemTime, CacheKey>>>,
    /// Configuration
    config: CacheConfig,
    /// Statistics
    stats: Arc<RwLock<CacheStats>>,
    /// Logger
    logger: Arc<Logger>,
    /// Current size in bytes
    current_size: Arc<RwLock<usize>>,
}

impl<T: Clone + Serialize + for<'de> Deserialize<'de> + Send + Sync + 'static> PhotonicCache<T> {
    /// Create new cache
    pub fn new(config: CacheConfig) -> Self {
        Self {
            data: Arc::new(RwLock::new(HashMap::new())),
            lru_order: Arc::new(RwLock::new(BTreeMap::new())),
            config,
            stats: Arc::new(RwLock::new(CacheStats::default())),
            logger: Arc::new(Logger::new("photonic_cache")),
            current_size: Arc::new(RwLock::new(0)),
        }
    }
    
    /// Get item from cache
    pub fn get(&self, key: &CacheKey) -> Option<T> {
        let start_time = Instant::now();
        
        let mut data = self.data.write().unwrap();
        if let Some(cached_result) = data.get_mut(key) {
            // Update access metadata
            let result = cached_result.access().clone();
            
            // Update LRU order
            {
                let mut lru = self.lru_order.write().unwrap();
                // Remove old entry
                lru.retain(|_, k| k != key);
                // Add new entry with current time
                lru.insert(SystemTime::now(), key.clone());
            }
            
            // Update statistics
            {
                let mut stats = self.stats.write().unwrap();
                stats.hits += 1;
                stats.total_computation_time_saved_ms += cached_result.computation_time_saved;
                
                let access_time = start_time.elapsed().as_nanos() as f64;
                stats.average_access_time_ns = (stats.average_access_time_ns * (stats.hits - 1) as f64 + access_time) / stats.hits as f64;
            }
            
            self.logger.debug(&format!("Cache hit for key: {}", key));
            Some(result)
        } else {
            // Cache miss
            {
                let mut stats = self.stats.write().unwrap();
                stats.misses += 1;
            }
            
            self.logger.debug(&format!("Cache miss for key: {}", key));
            None
        }
    }
    
    /// Put item into cache
    pub fn put(&self, key: CacheKey, value: T, computation_time_ms: f64) -> Result<()> {
        let size_estimate = self.estimate_size(&value);
        
        // Check if we need to evict entries
        self.ensure_capacity(size_estimate)?;
        
        let cached_result = CachedResult::new(value, computation_time_ms, size_estimate);
        
        {
            let mut data = self.data.write().unwrap();
            let mut size = self.current_size.write().unwrap();
            
            // Remove old entry if exists
            if let Some(old_result) = data.get(&key) {
                *size -= old_result.size_bytes;
            }
            
            data.insert(key.clone(), cached_result);
            *size += size_estimate;
        }
        
        // Update LRU order
        {
            let mut lru = self.lru_order.write().unwrap();
            lru.insert(SystemTime::now(), key.clone());
        }
        
        // Update statistics
        {
            let mut stats = self.stats.write().unwrap();
            stats.entry_count = self.data.read().unwrap().len();
            stats.total_size_bytes = *self.current_size.read().unwrap();
        }
        
        self.logger.debug(&format!("Cached result for key: {} ({} bytes)", key, size_estimate));
        Ok(())
    }
    
    /// Ensure cache has capacity for new entry
    fn ensure_capacity(&self, required_size: usize) -> Result<()> {
        let current_size = *self.current_size.read().unwrap();
        let current_entries = self.data.read().unwrap().len();
        
        // Check if we need to evict
        if current_size + required_size > self.config.max_size_bytes || 
           current_entries >= self.config.max_entries {
            self.evict_entries(required_size)?;
        }
        
        Ok(())
    }
    
    /// Evict entries based on policy
    fn evict_entries(&self, required_size: usize) -> Result<()> {
        match self.config.eviction_policy {
            EvictionPolicy::LRU => self.evict_lru(required_size),
            EvictionPolicy::LFU => self.evict_lfu(required_size),
            EvictionPolicy::TTL(ttl) => self.evict_expired(ttl),
            EvictionPolicy::SizeBased => self.evict_largest(),
            EvictionPolicy::Adaptive => self.evict_adaptive(required_size),
        }
    }
    
    /// Evict least recently used entries
    fn evict_lru(&self, required_size: usize) -> Result<()> {
        let mut evicted_size = 0;
        let mut keys_to_evict = Vec::new();
        
        {
            let lru = self.lru_order.read().unwrap();
            let data = self.data.read().unwrap();
            
            for (_, key) in lru.iter() {
                if let Some(result) = data.get(key) {
                    keys_to_evict.push(key.clone());
                    evicted_size += result.size_bytes;
                    
                    if evicted_size >= required_size {
                        break;
                    }
                }
            }
        }
        
        // Evict the selected keys
        self.remove_keys(&keys_to_evict);
        
        self.logger.info(&format!("Evicted {} LRU entries ({} bytes)", keys_to_evict.len(), evicted_size));
        Ok(())
    }
    
    /// Evict least frequently used entries
    fn evict_lfu(&self, required_size: usize) -> Result<()> {
        let mut candidates: Vec<_> = {
            let data = self.data.read().unwrap();
            data.iter()
                .map(|(key, result)| (key.clone(), result.access_count, result.size_bytes))
                .collect()
        };
        
        // Sort by access count (ascending)
        candidates.sort_by_key(|(_, access_count, _)| *access_count);
        
        let mut evicted_size = 0;
        let mut keys_to_evict = Vec::new();
        
        for (key, _, size) in candidates {
            keys_to_evict.push(key);
            evicted_size += size;
            
            if evicted_size >= required_size {
                break;
            }
        }
        
        self.remove_keys(&keys_to_evict);
        
        self.logger.info(&format!("Evicted {} LFU entries ({} bytes)", keys_to_evict.len(), evicted_size));
        Ok(())
    }
    
    /// Evict expired entries
    fn evict_expired(&self, ttl: Duration) -> Result<()> {
        let now = SystemTime::now();
        let mut keys_to_evict = Vec::new();
        
        {
            let data = self.data.read().unwrap();
            for (key, result) in data.iter() {
                if now.duration_since(result.created_at).unwrap_or_default() > ttl {
                    keys_to_evict.push(key.clone());
                }
            }
        }
        
        self.remove_keys(&keys_to_evict);
        
        self.logger.info(&format!("Evicted {} expired entries", keys_to_evict.len()));
        Ok(())
    }
    
    /// Evict largest entries first
    fn evict_largest(&self) -> Result<()> {
        let mut candidates: Vec<_> = {
            let data = self.data.read().unwrap();
            data.iter()
                .map(|(key, result)| (key.clone(), result.size_bytes))
                .collect()
        };
        
        // Sort by size (descending)
        candidates.sort_by_key(|(_, size)| std::cmp::Reverse(*size));
        
        // Evict largest 10% of entries
        let evict_count = (candidates.len() / 10).max(1);
        let keys_to_evict: Vec<_> = candidates.into_iter()
            .take(evict_count)
            .map(|(key, _)| key)
            .collect();
        
        self.remove_keys(&keys_to_evict);
        
        self.logger.info(&format!("Evicted {} largest entries", keys_to_evict.len()));
        Ok(())
    }
    
    /// Adaptive eviction based on access patterns
    fn evict_adaptive(&self, required_size: usize) -> Result<()> {
        let now = SystemTime::now();
        let mut candidates: Vec<_> = {
            let data = self.data.read().unwrap();
            data.iter()
                .map(|(key, result)| {
                    let age_hours = now.duration_since(result.created_at)
                        .unwrap_or_default()
                        .as_secs_f64() / 3600.0;
                    let time_since_access_hours = now.duration_since(result.last_accessed)
                        .unwrap_or_default()
                        .as_secs_f64() / 3600.0;
                    
                    // Calculate eviction score (higher = more likely to evict)
                    let score = age_hours + time_since_access_hours * 2.0 - result.access_count as f64 * 0.1;
                    
                    (key.clone(), result.size_bytes, score)
                })
                .collect()
        };
        
        // Sort by eviction score (descending)
        candidates.sort_by(|(_, _, score_a), (_, _, score_b)| score_b.partial_cmp(score_a).unwrap_or(std::cmp::Ordering::Equal));
        
        let mut evicted_size = 0;
        let mut keys_to_evict = Vec::new();
        
        for (key, size, _) in candidates {
            keys_to_evict.push(key);
            evicted_size += size;
            
            if evicted_size >= required_size {
                break;
            }
        }
        
        self.remove_keys(&keys_to_evict);
        
        self.logger.info(&format!("Evicted {} entries using adaptive policy ({} bytes)", keys_to_evict.len(), evicted_size));
        Ok(())
    }
    
    /// Remove specified keys from cache
    fn remove_keys(&self, keys: &[CacheKey]) {
        let mut data = self.data.write().unwrap();
        let mut lru = self.lru_order.write().unwrap();
        let mut size = self.current_size.write().unwrap();
        let mut stats = self.stats.write().unwrap();
        
        for key in keys {
            if let Some(result) = data.remove(key) {
                *size -= result.size_bytes;
                stats.evictions += 1;
            }
            
            // Remove from LRU tracking
            lru.retain(|_, k| k != key);
        }
        
        stats.entry_count = data.len();
        stats.total_size_bytes = *size;
    }
    
    /// Estimate size of value in bytes
    fn estimate_size(&self, _value: &T) -> usize {
        // Simplified size estimation
        // In practice, this would use serialization size or introspection
        std::mem::size_of::<T>()
    }
    
    /// Clear entire cache
    pub fn clear(&self) {
        self.data.write().unwrap().clear();
        self.lru_order.write().unwrap().clear();
        *self.current_size.write().unwrap() = 0;
        
        {
            let mut stats = self.stats.write().unwrap();
            *stats = CacheStats::default();
        }
        
        self.logger.info("Cache cleared");
    }
    
    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        let stats = self.stats.read().unwrap().clone();
        let mut result = stats;
        result.hit_rate = result.hit_rate();
        result
    }
    
    /// Get current cache size
    pub fn size(&self) -> usize {
        *self.current_size.read().unwrap()
    }
    
    /// Get current entry count
    pub fn len(&self) -> usize {
        self.data.read().unwrap().len()
    }
    
    /// Check if cache is empty
    pub fn is_empty(&self) -> bool {
        self.data.read().unwrap().is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_cache_key_creation() {
        let params = vec![1.0, 2.0, 3.0];
        let input_data = b"test_data";
        let key = CacheKey::new("test_op", &params, input_data, 32);
        
        assert_eq!(key.operation, "test_op");
        assert_eq!(key.precision, 32);
        assert_ne!(key.parameters_hash, 0);
        assert_ne!(key.input_hash, 0);
    }
    
    #[test]
    fn test_cached_result() {
        let mut result = CachedResult::new("test_data".to_string(), 100.0, 64);
        
        assert_eq!(result.access_count, 0);
        let data = result.access();
        assert_eq!(data, "test_data");
        assert_eq!(result.access_count, 1);
    }
    
    #[test]
    fn test_cache_basic_operations() {
        let config = CacheConfig::default();
        let cache = PhotonicCache::<String>::new(config);
        
        let key = CacheKey::new("test", &[1.0], b"input", 32);
        
        // Test miss
        assert!(cache.get(&key).is_none());
        
        // Test put and hit
        cache.put(key.clone(), "result".to_string(), 50.0).unwrap();
        assert_eq!(cache.get(&key).unwrap(), "result");
        
        // Check statistics
        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.entry_count, 1);
    }
    
    #[test]
    fn test_cache_eviction() {
        let config = CacheConfig {
            max_entries: 2,
            eviction_policy: EvictionPolicy::LRU,
            ..Default::default()
        };
        let cache = PhotonicCache::<String>::new(config);
        
        let key1 = CacheKey::new("test1", &[1.0], b"input1", 32);
        let key2 = CacheKey::new("test2", &[2.0], b"input2", 32);
        let key3 = CacheKey::new("test3", &[3.0], b"input3", 32);
        
        // Fill cache to capacity
        cache.put(key1.clone(), "result1".to_string(), 10.0).unwrap();
        cache.put(key2.clone(), "result2".to_string(), 10.0).unwrap();
        
        // Access key1 to make it more recent
        cache.get(&key1);
        
        // Add third item, should evict key2 (LRU)
        cache.put(key3.clone(), "result3".to_string(), 10.0).unwrap();
        
        assert!(cache.get(&key1).is_some());
        assert!(cache.get(&key2).is_none()); // Should be evicted
        assert!(cache.get(&key3).is_some());
    }
    
    #[test]
    fn test_ttl_eviction() {
        let config = CacheConfig {
            eviction_policy: EvictionPolicy::TTL(Duration::from_millis(10)),
            ..Default::default()
        };
        let cache = PhotonicCache::<String>::new(config);
        
        let key = CacheKey::new("test", &[1.0], b"input", 32);
        cache.put(key.clone(), "result".to_string(), 10.0).unwrap();
        
        // Should be accessible immediately
        assert!(cache.get(&key).is_some());
        
        // Wait for expiration
        std::thread::sleep(Duration::from_millis(15));
        
        // Trigger eviction by adding new entry
        let key2 = CacheKey::new("test2", &[2.0], b"input2", 32);
        cache.put(key2, "result2".to_string(), 10.0).unwrap();
        
        // Original should be gone due to TTL
        let stats = cache.stats();
        assert!(stats.evictions > 0);
    }
    
    #[test]
    fn test_cache_stats() {
        let config = CacheConfig::default();
        let cache = PhotonicCache::<String>::new(config);
        
        let key1 = CacheKey::new("test1", &[1.0], b"input1", 32);
        let key2 = CacheKey::new("test2", &[2.0], b"input2", 32);
        
        // Generate some cache activity
        cache.get(&key1); // miss
        cache.put(key1.clone(), "result1".to_string(), 100.0).unwrap();
        cache.get(&key1); // hit
        cache.get(&key2); // miss
        
        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 2);
        assert_eq!(stats.entry_count, 1);
        assert_eq!(stats.total_computation_time_saved_ms, 100.0);
        assert!(stats.hit_rate() > 0.0);
        
        let summary = stats.summary();
        assert!(summary.contains("hit rate"));
        assert!(summary.contains("saved"));
    }
}