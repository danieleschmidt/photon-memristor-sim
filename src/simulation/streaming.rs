//! Streaming data processing for large-scale photonic simulations

use crate::core::{Result, PhotonicError, OpticalField};
use crate::simulation::cache::{CacheKey, SimulationCacheManager};
use nalgebra::{DVector, DMatrix};
use std::sync::{Arc, RwLock, mpsc};
use std::thread;
use std::time::{Duration, Instant};
use serde::{Serialize, Deserialize};

/// Streaming data chunk for processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataChunk<T> {
    /// Chunk identifier
    pub id: usize,
    /// Sequence number within stream
    pub sequence: u64,
    /// Actual data payload
    pub data: T,
    /// Timestamp when chunk was created
    pub timestamp: Instant,
    /// Processing metadata
    pub metadata: ChunkMetadata,
}

/// Metadata for data chunks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkMetadata {
    /// Source identifier
    pub source: String,
    /// Chunk size in bytes
    pub size_bytes: usize,
    /// Compression ratio (if compressed)
    pub compression_ratio: Option<f32>,
    /// Checksum for integrity
    pub checksum: u64,
}

/// Stream processing statistics
#[derive(Debug, Clone)]
pub struct StreamStats {
    /// Total chunks processed
    pub chunks_processed: u64,
    /// Total bytes processed
    pub bytes_processed: u64,
    /// Average processing time per chunk
    pub avg_processing_time: Duration,
    /// Current throughput (chunks/sec)
    pub throughput: f64,
    /// Error count
    pub error_count: u64,
    /// Buffer utilization
    pub buffer_utilization: f32,
}

/// Configuration for stream processor
#[derive(Debug, Clone)]
pub struct StreamConfig {
    /// Maximum buffer size for incoming data
    pub max_buffer_size: usize,
    /// Number of worker threads
    pub num_workers: usize,
    /// Batch size for processing
    pub batch_size: usize,
    /// Timeout for chunk processing
    pub chunk_timeout: Duration,
    /// Enable compression
    pub enable_compression: bool,
    /// Enable caching
    pub enable_caching: bool,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            max_buffer_size: 1000,
            num_workers: 4,
            batch_size: 10,
            chunk_timeout: Duration::from_secs(30),
            enable_compression: true,
            enable_caching: true,
        }
    }
}

/// Stream processor for handling large-scale photonic simulation data
pub struct StreamProcessor<T> {
    /// Configuration
    config: StreamConfig,
    /// Input channel for receiving data chunks
    input_rx: Arc<RwLock<Option<mpsc::Receiver<DataChunk<T>>>>>,
    /// Output channel for sending processed results
    output_tx: mpsc::Sender<ProcessedChunk<T>>,
    /// Worker thread handles
    workers: Vec<thread::JoinHandle<()>>,
    /// Processing statistics
    stats: Arc<RwLock<StreamStats>>,
    /// Cache manager for processed results
    cache_manager: Option<Arc<SimulationCacheManager>>,
    /// Processing function
    processor_fn: Arc<dyn Fn(&T) -> Result<T> + Send + Sync>,
}

/// Processed chunk result
#[derive(Debug, Clone)]
pub struct ProcessedChunk<T> {
    /// Original chunk info
    pub original_id: usize,
    /// Processed data
    pub result: Result<T>,
    /// Processing time
    pub processing_time: Duration,
    /// Whether result was cached
    pub cached: bool,
}

impl<T> StreamProcessor<T>
where
    T: Clone + Send + Sync + 'static + Serialize + for<'de> Deserialize<'de>,
{
    /// Create new stream processor
    pub fn new<F>(
        config: StreamConfig,
        processor_fn: F,
        cache_manager: Option<Arc<SimulationCacheManager>>,
    ) -> Result<(Self, mpsc::Sender<DataChunk<T>>, mpsc::Receiver<ProcessedChunk<T>>)>
    where
        F: Fn(&T) -> Result<T> + Send + Sync + 'static,
    {
        let (input_tx, input_rx) = mpsc::channel();
        let (output_tx, output_rx) = mpsc::channel();
        
        let input_rx = Arc::new(RwLock::new(Some(input_rx)));
        let stats = Arc::new(RwLock::new(StreamStats {
            chunks_processed: 0,
            bytes_processed: 0,
            avg_processing_time: Duration::from_millis(0),
            throughput: 0.0,
            error_count: 0,
            buffer_utilization: 0.0,
        }));
        
        let processor = StreamProcessor {
            config: config.clone(),
            input_rx: input_rx.clone(),
            output_tx: output_tx.clone(),
            workers: Vec::new(),
            stats: stats.clone(),
            cache_manager: cache_manager.clone(),
            processor_fn: Arc::new(processor_fn),
        };
        
        Ok((processor, input_tx, output_rx))
    }
    
    /// Start processing with worker threads
    pub fn start(&mut self) -> Result<()> {
        let input_rx = self.input_rx.write().unwrap().take()
            .ok_or_else(|| PhotonicError::simulation_error("Stream processor already started"))?;
        
        // Create worker threads
        for worker_id in 0..self.config.num_workers {
            let input_rx = Arc::new(RwLock::new(input_rx));
            let output_tx = self.output_tx.clone();
            let stats = self.stats.clone();
            let cache_manager = self.cache_manager.clone();
            let processor_fn = self.processor_fn.clone();
            let config = self.config.clone();
            
            let worker_handle = thread::spawn(move || {
                Self::worker_loop(
                    worker_id,
                    input_rx,
                    output_tx,
                    stats,
                    cache_manager,
                    processor_fn,
                    config,
                );
            });
            
            self.workers.push(worker_handle);
        }
        
        Ok(())
    }
    
    /// Worker thread main loop
    fn worker_loop(
        worker_id: usize,
        input_rx: Arc<RwLock<mpsc::Receiver<DataChunk<T>>>>,
        output_tx: mpsc::Sender<ProcessedChunk<T>>,
        stats: Arc<RwLock<StreamStats>>,
        cache_manager: Option<Arc<SimulationCacheManager>>,
        processor_fn: Arc<dyn Fn(&T) -> Result<T> + Send + Sync>,
        config: StreamConfig,
    ) {
        println!("Worker {} started", worker_id);
        
        loop {
            // Receive chunk from input channel
            let chunk = {
                let rx = input_rx.read().unwrap();
                match rx.recv_timeout(Duration::from_millis(100)) {
                    Ok(chunk) => chunk,
                    Err(mpsc::RecvTimeoutError::Timeout) => continue,
                    Err(mpsc::RecvTimeoutError::Disconnected) => {
                        println!("Worker {} shutting down - input disconnected", worker_id);
                        break;
                    }
                }
            };
            
            let start_time = Instant::now();
            
            // Check cache if enabled
            let cached_result = if let Some(ref cache) = cache_manager {
                if config.enable_caching {
                    Self::check_cache(&chunk, cache)
                } else {
                    None
                }
            } else {
                None
            };
            
            let (result, cached) = if let Some(cached_result) = cached_result {
                (Ok(cached_result), true)
            } else {
                // Process the chunk
                let result = processor_fn(&chunk.data);
                
                // Cache the result if successful and caching is enabled
                if let (Ok(ref processed_data), Some(ref cache)) = (&result, &cache_manager) {
                    if config.enable_caching {
                        Self::cache_result(&chunk, processed_data, cache);
                    }
                }
                
                (result, false)
            };
            
            let processing_time = start_time.elapsed();
            
            // Send processed result
            let processed_chunk = ProcessedChunk {
                original_id: chunk.id,
                result,
                processing_time,
                cached,
            };
            
            if output_tx.send(processed_chunk).is_err() {
                println!("Worker {} shutting down - output disconnected", worker_id);
                break;
            }
            
            // Update statistics
            Self::update_stats(&stats, &chunk, processing_time, cached);
        }
    }
    
    /// Check cache for existing result
    fn check_cache(chunk: &DataChunk<T>, cache_manager: &SimulationCacheManager) -> Option<T> {
        // Generate cache key based on chunk data
        let cache_key = Self::generate_cache_key(chunk);
        
        // This is a simplified cache check - in practice you'd need to implement
        // specific cache retrieval based on the data type T
        // For now, we'll return None to indicate no cache hit
        None
    }
    
    /// Cache processing result
    fn cache_result(chunk: &DataChunk<T>, result: &T, cache_manager: &SimulationCacheManager) {
        let cache_key = Self::generate_cache_key(chunk);
        
        // This is a simplified cache store - in practice you'd need to implement
        // specific cache storage based on the data type T
        // For now, we'll just generate the key
    }
    
    /// Generate cache key for chunk
    fn generate_cache_key(chunk: &DataChunk<T>) -> CacheKey {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        chunk.id.hash(&mut hasher);
        chunk.sequence.hash(&mut hasher);
        chunk.metadata.checksum.hash(&mut hasher);
        
        CacheKey {
            param_hash: hasher.finish(),
            sim_type: "streaming".to_string(),
            version: 1,
        }
    }
    
    /// Update processing statistics
    fn update_stats(
        stats: &Arc<RwLock<StreamStats>>,
        chunk: &DataChunk<T>,
        processing_time: Duration,
        cached: bool,
    ) {
        let mut stats = stats.write().unwrap();
        
        stats.chunks_processed += 1;
        stats.bytes_processed += chunk.metadata.size_bytes as u64;
        
        // Update average processing time
        let total_time = stats.avg_processing_time * (stats.chunks_processed - 1) as u32 + processing_time;
        stats.avg_processing_time = total_time / stats.chunks_processed as u32;
        
        // Calculate throughput (simple moving average)
        let current_time = Instant::now();
        let time_since_start = current_time.duration_since(chunk.timestamp);
        if time_since_start > Duration::from_millis(1) {
            stats.throughput = stats.chunks_processed as f64 / time_since_start.as_secs_f64();
        }
        
        if !cached {
            // Only count as processed if not from cache
            stats.chunks_processed += 0; // Already incremented above
        }
    }
    
    /// Get current processing statistics
    pub fn stats(&self) -> StreamStats {
        self.stats.read().unwrap().clone()
    }
    
    /// Stop processing and wait for workers to finish
    pub fn stop(self) -> Result<()> {
        // Workers will stop when input channel is closed (dropped)
        for worker in self.workers {
            worker.join().map_err(|_| {
                PhotonicError::simulation_error("Failed to join worker thread")
            })?;
        }
        
        Ok(())
    }
}

/// Batch processor for processing multiple chunks together
pub struct BatchProcessor<T> {
    /// Batch size
    batch_size: usize,
    /// Current batch
    current_batch: Vec<DataChunk<T>>,
    /// Batch processing function
    processor_fn: Arc<dyn Fn(&[DataChunk<T>]) -> Result<Vec<T>> + Send + Sync>,
    /// Statistics
    stats: Arc<RwLock<StreamStats>>,
}

impl<T> BatchProcessor<T>
where
    T: Clone + Send + Sync + 'static,
{
    /// Create new batch processor
    pub fn new<F>(batch_size: usize, processor_fn: F) -> Self
    where
        F: Fn(&[DataChunk<T>]) -> Result<Vec<T>> + Send + Sync + 'static,
    {
        Self {
            batch_size,
            current_batch: Vec::with_capacity(batch_size),
            processor_fn: Arc::new(processor_fn),
            stats: Arc::new(RwLock::new(StreamStats {
                chunks_processed: 0,
                bytes_processed: 0,
                avg_processing_time: Duration::from_millis(0),
                throughput: 0.0,
                error_count: 0,
                buffer_utilization: 0.0,
            })),
        }
    }
    
    /// Add chunk to current batch
    pub fn add_chunk(&mut self, chunk: DataChunk<T>) -> Option<Vec<T>> {
        self.current_batch.push(chunk);
        
        if self.current_batch.len() >= self.batch_size {
            self.process_current_batch()
        } else {
            None
        }
    }
    
    /// Process current batch if not empty
    pub fn flush(&mut self) -> Option<Vec<T>> {
        if !self.current_batch.is_empty() {
            self.process_current_batch()
        } else {
            None
        }
    }
    
    /// Process current batch and return results
    fn process_current_batch(&mut self) -> Option<Vec<T>> {
        if self.current_batch.is_empty() {
            return None;
        }
        
        let start_time = Instant::now();
        let batch = std::mem::take(&mut self.current_batch);
        
        match (self.processor_fn)(&batch) {
            Ok(results) => {
                let processing_time = start_time.elapsed();
                self.update_batch_stats(&batch, processing_time, false);
                Some(results)
            }
            Err(_) => {
                let processing_time = start_time.elapsed();
                self.update_batch_stats(&batch, processing_time, true);
                None
            }
        }
    }
    
    /// Update batch processing statistics
    fn update_batch_stats(&mut self, batch: &[DataChunk<T>], processing_time: Duration, error: bool) {
        let mut stats = self.stats.write().unwrap();
        
        stats.chunks_processed += batch.len() as u64;
        stats.bytes_processed += batch.iter().map(|c| c.metadata.size_bytes as u64).sum::<u64>();
        
        if error {
            stats.error_count += 1;
        }
        
        // Update average processing time
        let batch_avg_time = processing_time / batch.len() as u32;
        let total_time = stats.avg_processing_time * (stats.chunks_processed - batch.len() as u64) as u32 + 
                        batch_avg_time * batch.len() as u32;
        stats.avg_processing_time = total_time / stats.chunks_processed as u32;
    }
    
    /// Get current statistics
    pub fn stats(&self) -> StreamStats {
        self.stats.read().unwrap().clone()
    }
}

/// Utility functions for creating data chunks
pub mod chunk_utils {
    use super::*;
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    
    /// Create data chunk from raw data
    pub fn create_chunk<T>(
        id: usize,
        sequence: u64,
        data: T,
        source: String,
    ) -> DataChunk<T>
    where
        T: Serialize,
    {
        let serialized = bincode::serialize(&data).unwrap_or_default();
        let size_bytes = serialized.len();
        
        let mut hasher = DefaultHasher::new();
        serialized.hash(&mut hasher);
        let checksum = hasher.finish();
        
        DataChunk {
            id,
            sequence,
            data,
            timestamp: Instant::now(),
            metadata: ChunkMetadata {
                source,
                size_bytes,
                compression_ratio: None,
                checksum,
            },
        }
    }
    
    /// Split large data into chunks
    pub fn split_into_chunks<T>(
        data: Vec<T>,
        chunk_size: usize,
        source: String,
    ) -> Vec<DataChunk<Vec<T>>>
    where
        T: Clone + Serialize,
    {
        data.chunks(chunk_size)
            .enumerate()
            .map(|(i, chunk_data)| {
                create_chunk(i, i as u64, chunk_data.to_vec(), source.clone())
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;
    use std::time::Duration;
    
    #[test]
    fn test_stream_processor_creation() {
        let config = StreamConfig::default();
        let processor_fn = |data: &String| Ok(data.to_uppercase());
        
        let result = StreamProcessor::new(config, processor_fn, None);
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_batch_processor() {
        let mut processor = BatchProcessor::new(3, |batch: &[DataChunk<String>]| {
            Ok(batch.iter().map(|chunk| chunk.data.to_uppercase()).collect())
        });
        
        // Add chunks to batch
        let chunk1 = chunk_utils::create_chunk(1, 1, "hello".to_string(), "test".to_string());
        let chunk2 = chunk_utils::create_chunk(2, 2, "world".to_string(), "test".to_string());
        
        assert!(processor.add_chunk(chunk1).is_none()); // Batch not full
        assert!(processor.add_chunk(chunk2).is_none()); // Still not full
        
        let chunk3 = chunk_utils::create_chunk(3, 3, "test".to_string(), "test".to_string());
        let results = processor.add_chunk(chunk3); // Now full, should process
        
        assert!(results.is_some());
        let results = results.unwrap();
        assert_eq!(results.len(), 3);
        assert_eq!(results[0], "HELLO");
        assert_eq!(results[1], "WORLD");
        assert_eq!(results[2], "TEST");
    }
    
    #[test]
    fn test_chunk_utils() {
        let data = vec!["a", "b", "c", "d", "e"];
        let chunks = chunk_utils::split_into_chunks(data, 2, "test".to_string());
        
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].data, vec!["a", "b"]);
        assert_eq!(chunks[1].data, vec!["c", "d"]);
        assert_eq!(chunks[2].data, vec!["e"]);
    }
}