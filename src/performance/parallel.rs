//! Parallel processing and concurrency for photonic simulations

use crate::core::{Result, PhotonicError, OpticalField, Logger, Monitor};
use rayon::prelude::*;
use std::sync::{Arc, Mutex, RwLock};
use std::collections::HashMap;
use std::time::Instant;

/// Parallel computation configuration
#[derive(Debug, Clone)]
pub struct ParallelConfig {
    /// Number of worker threads
    pub num_threads: Option<usize>,
    /// Chunk size for data parallelism
    pub chunk_size: usize,
    /// Enable work stealing
    pub work_stealing: bool,
    /// Load balancing strategy
    pub load_balancing: LoadBalancingStrategy,
    /// Memory pool size per thread (bytes)
    pub memory_pool_size: usize,
}

/// Load balancing strategies
#[derive(Debug, Clone, PartialEq)]
pub enum LoadBalancingStrategy {
    /// Static load balancing - equal chunks
    Static,
    /// Dynamic load balancing - work stealing
    Dynamic,
    /// Adaptive load balancing based on performance
    Adaptive,
    /// Round-robin assignment to workers
    RoundRobin,
    /// Assign to least loaded worker
    LeastLoaded,
    /// Random assignment to workers
    Random,
}

impl LoadBalancingStrategy {
    /// Assign a worker based on the strategy
    pub fn assign_worker(&self, loads: &[f64], current_task: usize) -> Result<usize> {
        let num_workers = loads.len();
        if num_workers == 0 {
            return Err(crate::core::PhotonicError::ValidationError("No workers available".to_string()));
        }
        
        let worker_id = match self {
            LoadBalancingStrategy::Static => current_task % num_workers,
            LoadBalancingStrategy::Dynamic | LoadBalancingStrategy::Adaptive => {
                // Find worker with least load
                loads.iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i)
                    .unwrap_or(0)
            },
            LoadBalancingStrategy::RoundRobin => current_task % num_workers,
            LoadBalancingStrategy::LeastLoaded => {
                loads.iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i)
                    .unwrap_or(0)
            },
            LoadBalancingStrategy::Random => {
                use rand::Rng;
                rand::thread_rng().gen_range(0..num_workers)
            },
        };
        
        Ok(worker_id)
    }
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self {
            num_threads: None, // Use system default
            chunk_size: 1000,
            work_stealing: true,
            load_balancing: LoadBalancingStrategy::Adaptive,
            memory_pool_size: 64 * 1024 * 1024, // 64MB per thread
        }
    }
}

/// Parallel executor for photonic simulations
pub struct ParallelExecutor {
    config: ParallelConfig,
    thread_pool: Arc<rayon::ThreadPool>,
    logger: Arc<Logger>,
    monitor: Option<Arc<Monitor>>,
    work_metrics: Arc<RwLock<WorkMetrics>>,
}

/// Work distribution metrics
#[derive(Debug, Default, Clone)]
struct WorkMetrics {
    total_tasks: usize,
    completed_tasks: usize,
    failed_tasks: usize,
    average_task_time: f64,
    thread_utilization: HashMap<usize, f64>,
}

impl ParallelExecutor {
    /// Create new parallel executor
    pub fn new(config: ParallelConfig) -> Result<Self> {
        let thread_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(config.num_threads.unwrap_or_else(|| get_num_cpus()))
            .thread_name(|index| format!("photonic-worker-{}", index))
            .build()
            .map_err(|e| PhotonicError::simulation(format!("Failed to create thread pool: {}", e)))?;
        
        Ok(Self {
            config,
            thread_pool: Arc::new(thread_pool),
            logger: Arc::new(Logger::new("parallel_executor")),
            monitor: None,
            work_metrics: Arc::new(RwLock::new(WorkMetrics::default())),
        })
    }
    
    /// Create executor with monitoring
    pub fn with_monitor(config: ParallelConfig, monitor: Arc<Monitor>) -> Result<Self> {
        let mut executor = Self::new(config)?;
        executor.monitor = Some(monitor);
        Ok(executor)
    }
    
    /// Execute operations in parallel on a collection
    pub fn parallel_map<T, U, F>(&self, items: Vec<T>, operation: F) -> Result<Vec<U>>
    where
        T: Send + Sync,
        U: Send,
        F: Fn(T) -> Result<U> + Send + Sync,
    {
        let start_time = Instant::now();
        let operation = Arc::new(operation);
        
        self.logger.info(&format!("Starting parallel execution of {} items", items.len()));
        
        let results: Result<Vec<U>> = items
            .into_par_iter()
            .chunks(self.config.chunk_size)
            .map(|chunk| {
                chunk.into_iter()
                    .map(|item| {
                        let task_start = Instant::now();
                        let result = operation(item);
                        let task_duration = task_start.elapsed();
                        
                        // Update metrics
                        {
                            let mut metrics = self.work_metrics.write().unwrap();
                            metrics.total_tasks += 1;
                            if result.is_ok() {
                                metrics.completed_tasks += 1;
                            } else {
                                metrics.failed_tasks += 1;
                            }
                            
                            // Update average task time
                            let prev_avg = metrics.average_task_time;
                            let count = metrics.total_tasks as f64;
                            metrics.average_task_time = (prev_avg * (count - 1.0) + task_duration.as_secs_f64()) / count;
                        }
                        
                        result
                    })
                    .collect::<Result<Vec<U>>>()
            })
            .collect::<Result<Vec<Vec<U>>>>()
            .map(|chunks| chunks.into_iter().flatten().collect());
        
        let total_duration = start_time.elapsed();
        
        match &results {
            Ok(output) => {
                self.logger.info(&format!(
                    "Parallel execution completed: {} items in {:.2}ms",
                    output.len(),
                    total_duration.as_millis()
                ));
                
                if let Some(monitor) = &self.monitor {
                    monitor.record_metric("parallel_execution_duration_ms", total_duration.as_millis() as f64, HashMap::new());
                    monitor.record_metric("parallel_throughput_items_per_sec", output.len() as f64 / total_duration.as_secs_f64(), HashMap::new());
                }
            }
            Err(e) => {
                self.logger.error(&format!("Parallel execution failed: {}", e));
            }
        }
        
        results
    }
    
    /// Execute reduction operation in parallel
    pub fn parallel_reduce<T, F, G, R>(&self, items: Vec<T>, identity: R, reduce_op: F, combine_op: G) -> Result<R>
    where
        T: Send + Sync,
        R: Send + Sync + Clone,
        F: Fn(R, T) -> R + Send + Sync,
        G: Fn(R, R) -> R + Send + Sync,
    {
        let start_time = Instant::now();
        
        let result = items
            .into_par_iter()
            .fold(|| identity.clone(), |acc, item| reduce_op(acc, item))
            .reduce(|| identity.clone(), |a, b| combine_op(a, b));
        
        let duration = start_time.elapsed();
        self.logger.info(&format!("Parallel reduction completed in {:.2}ms", duration.as_millis()));
        
        if let Some(monitor) = &self.monitor {
            monitor.record_metric("parallel_reduce_duration_ms", duration.as_millis() as f64, HashMap::new());
        }
        
        Ok(result)
    }
    
    /// Execute work with adaptive load balancing
    pub fn adaptive_parallel_execute<T, U, F>(&self, mut items: Vec<T>, operation: F) -> Result<Vec<U>>
    where
        T: Send + Sync,
        U: Send,
        F: Fn(T) -> Result<U> + Send + Sync,
    {
        match self.config.load_balancing {
            LoadBalancingStrategy::Static => self.parallel_map(items, operation),
            LoadBalancingStrategy::Dynamic => {
                // Use work stealing with smaller chunks
                let small_chunk_size = (self.config.chunk_size / 4).max(1);
                let mut config = self.config.clone();
                config.chunk_size = small_chunk_size;
                
                let temp_executor = Self::new(config)?;
                temp_executor.parallel_map(items, operation)
            }
            LoadBalancingStrategy::RoundRobin => self.parallel_map(items, operation),
            LoadBalancingStrategy::LeastLoaded => self.parallel_map(items, operation),
            LoadBalancingStrategy::Random => self.parallel_map(items, operation),
            LoadBalancingStrategy::Adaptive => {
                // Start with small chunks and adapt based on performance
                if items.len() < 100 {
                    // Small dataset - use sequential or small parallel
                    self.parallel_map(items, operation)
                } else {
                    // Large dataset - use adaptive chunking
                    let sample_size = items.len().min(100);
                    let sample: Vec<_> = items.drain(0..sample_size).collect();
                    
                    let start_time = Instant::now();
                    let sample_results = self.parallel_map(sample, &operation)?;
                    let sample_duration = start_time.elapsed();
                    
                    // Estimate optimal chunk size based on sample performance
                    let avg_time_per_item = sample_duration.as_secs_f64() / sample_size as f64;
                    let optimal_chunk_size = if avg_time_per_item > 0.001 {
                        // Slow operations - larger chunks to reduce overhead
                        (self.config.chunk_size * 2).min(10000)
                    } else {
                        // Fast operations - smaller chunks for better load balancing
                        (self.config.chunk_size / 2).max(10)
                    };
                    
                    let mut config = self.config.clone();
                    config.chunk_size = optimal_chunk_size;
                    let adaptive_executor = Self::new(config)?;
                    
                    let mut remaining_results = adaptive_executor.parallel_map(items, operation)?;
                    
                    // Combine sample and remaining results
                    let mut all_results = sample_results;
                    all_results.extend(remaining_results);
                    
                    Ok(all_results)
                }
            }
        }
    }
    
    /// Get current performance metrics
    pub fn get_metrics(&self) -> WorkMetrics {
        let guard = self.work_metrics.read().unwrap();
        guard.clone()
    }
    
    /// Reset performance metrics
    pub fn reset_metrics(&self) {
        let mut metrics = self.work_metrics.write().unwrap();
        *metrics = WorkMetrics::default();
    }
    
    /// Get number of worker threads
    pub fn worker_count(&self) -> usize {
        self.thread_pool.current_num_threads()
    }
    
    /// Check if executor is healthy
    pub fn is_healthy(&self) -> bool {
        // Check if thread pool is still running and responsive
        self.thread_pool.current_num_threads() > 0
    }
}

/// Memory pool for efficient allocation
pub struct MemoryPool {
    pools: Vec<Arc<Mutex<Vec<Vec<u8>>>>>,
    chunk_sizes: Vec<usize>,
    logger: Arc<Logger>,
}

/// Memory pool statistics
#[derive(Debug, Clone)]
pub struct MemoryPoolStats {
    pub allocated_blocks: usize,
    pub total_capacity: usize,
    pub available_blocks: usize,
}

impl MemoryPool {
    /// Create new memory pool
    pub fn new() -> Self {
        let chunk_sizes = vec![1024, 4096, 16384, 65536, 262144, 1048576]; // Powers of 2 from 1KB to 1MB
        let pools = chunk_sizes.iter()
            .map(|_| Arc::new(Mutex::new(Vec::new())))
            .collect();
        
        Self {
            pools,
            chunk_sizes,
            logger: Arc::new(Logger::new("memory_pool")),
        }
    }
    
    /// Create new memory pool with specified block size and count
    pub fn with_config(block_size: usize, block_count: usize) -> Result<Self> {
        let chunk_sizes = vec![block_size];
        let pools = vec![Arc::new(Mutex::new(Vec::with_capacity(block_count)))];
        
        Ok(Self {
            pools,
            chunk_sizes,
            logger: Arc::new(Logger::new("memory_pool")),
        })
    }
    
    /// Allocate buffer from pool
    pub fn allocate(&self, size: usize) -> Vec<u8> {
        // Find appropriate pool
        for (i, &chunk_size) in self.chunk_sizes.iter().enumerate() {
            if size <= chunk_size {
                let mut pool = self.pools[i].lock().unwrap();
                if let Some(mut buffer) = pool.pop() {
                    buffer.clear();
                    buffer.resize(size, 0);
                    return buffer;
                }
                break;
            }
        }
        
        // Allocate new buffer if pool is empty or size is too large
        vec![0; size]
    }
    
    /// Return buffer to pool
    pub fn deallocate(&self, mut buffer: Vec<u8>) {
        let capacity = buffer.capacity();
        
        for (i, &chunk_size) in self.chunk_sizes.iter().enumerate() {
            if capacity <= chunk_size {
                let mut pool = self.pools[i].lock().unwrap();
                if pool.len() < 100 { // Limit pool size
                    buffer.clear();
                    pool.push(buffer);
                }
                return;
            }
        }
        
        // Buffer too large for pool - let it drop
    }
    
    /// Get pool statistics
    pub fn get_stats(&self) -> Vec<(usize, usize)> {
        self.chunk_sizes.iter()
            .zip(self.pools.iter())
            .map(|(&size, pool)| (size, pool.lock().unwrap().len()))
            .collect()
    }
    
    /// Get detailed statistics
    pub fn statistics(&self) -> MemoryPoolStats {
        let stats = self.get_stats();
        let available_blocks = stats.iter().map(|(_, count)| count).sum();
        let total_capacity = stats.iter().map(|(size, count)| size * count).sum();
        
        MemoryPoolStats {
            allocated_blocks: 0, // We don't track allocated blocks, so assume 0 for now
            total_capacity,
            available_blocks,
        }
    }
}

/// SIMD-accelerated operations for photonic simulations
#[cfg(target_arch = "x86_64")]
pub mod simd {
    use std::arch::x86_64::*;
    
    /// SIMD-accelerated complex number multiplication
    #[target_feature(enable = "sse2")]
    pub unsafe fn complex_mul_simd(a_real: &[f64], a_imag: &[f64], 
                                  b_real: &[f64], b_imag: &[f64],
                                  result_real: &mut [f64], result_imag: &mut [f64]) {
        assert_eq!(a_real.len(), a_imag.len());
        assert_eq!(a_real.len(), b_real.len());
        assert_eq!(a_real.len(), b_imag.len());
        assert_eq!(a_real.len(), result_real.len());
        assert_eq!(a_real.len(), result_imag.len());
        
        let len = a_real.len();
        let chunks = len / 2; // Process 2 f64s at a time with SSE2
        
        for i in 0..chunks {
            let idx = i * 2;
            
            let a_r = _mm_loadu_pd(&a_real[idx]);
            let a_i = _mm_loadu_pd(&a_imag[idx]);
            let b_r = _mm_loadu_pd(&b_real[idx]);
            let b_i = _mm_loadu_pd(&b_imag[idx]);
            
            // Complex multiplication: (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
            let ac = _mm_mul_pd(a_r, b_r);
            let bd = _mm_mul_pd(a_i, b_i);
            let ad = _mm_mul_pd(a_r, b_i);
            let bc = _mm_mul_pd(a_i, b_r);
            
            let result_r = _mm_sub_pd(ac, bd);
            let result_i = _mm_add_pd(ad, bc);
            
            _mm_storeu_pd(&mut result_real[idx], result_r);
            _mm_storeu_pd(&mut result_imag[idx], result_i);
        }
        
        // Handle remaining elements
        for i in (chunks * 2)..len {
            let a_r = a_real[i];
            let a_i = a_imag[i];
            let b_r = b_real[i];
            let b_i = b_imag[i];
            
            result_real[i] = a_r * b_r - a_i * b_i;
            result_imag[i] = a_r * b_i + a_i * b_r;
        }
    }
    
    /// SIMD-accelerated vector addition
    #[target_feature(enable = "sse2")]
    pub unsafe fn add_f64_simd(a: &[f64], b: &[f64], result: &mut [f64]) {
        assert_eq!(a.len(), b.len());
        assert_eq!(a.len(), result.len());
        
        let len = a.len();
        let chunks = len / 2;
        
        for i in 0..chunks {
            let idx = i * 2;
            let va = _mm_loadu_pd(&a[idx]);
            let vb = _mm_loadu_pd(&b[idx]);
            let vresult = _mm_add_pd(va, vb);
            _mm_storeu_pd(&mut result[idx], vresult);
        }
        
        // Handle remaining elements
        for i in (chunks * 2)..len {
            result[i] = a[i] + b[i];
        }
    }
}

/// Utility function to get number of CPU cores
fn get_num_cpus() -> usize {
    // Fallback implementation
    std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(4)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_parallel_executor_creation() {
        let config = ParallelConfig::default();
        let executor = ParallelExecutor::new(config).unwrap();
        assert!(executor.thread_pool.current_num_threads() > 0);
    }
    
    #[test]
    fn test_parallel_map() {
        let config = ParallelConfig::default();
        let executor = ParallelExecutor::new(config).unwrap();
        
        let items: Vec<i32> = (0..1000).collect();
        let results = executor.parallel_map(items, |x| Ok(x * 2)).unwrap();
        
        assert_eq!(results.len(), 1000);
        assert_eq!(results[0], 0);
        assert_eq!(results[999], 1998);
    }
    
    #[test]
    fn test_parallel_reduce() {
        let config = ParallelConfig::default();
        let executor = ParallelExecutor::new(config).unwrap();
        
        let items: Vec<i32> = (1..=100).collect();
        let sum = executor.parallel_reduce(items, 0, |acc, x| acc + x).unwrap();
        
        assert_eq!(sum, 5050); // Sum of 1 to 100
    }
    
    #[test]
    fn test_memory_pool() {
        let pool = MemoryPool::new();
        
        // Allocate and deallocate buffers
        let buffer1 = pool.allocate(1024);
        let buffer2 = pool.allocate(4096);
        
        assert_eq!(buffer1.len(), 1024);
        assert_eq!(buffer2.len(), 4096);
        
        pool.deallocate(buffer1);
        pool.deallocate(buffer2);
        
        let stats = pool.get_stats();
        assert!(stats.len() > 0);
    }
    
    #[test]
    fn test_load_balancing_strategies() {
        let config = ParallelConfig {
            load_balancing: LoadBalancingStrategy::Static,
            ..Default::default()
        };
        let executor = ParallelExecutor::new(config).unwrap();
        
        let items: Vec<i32> = (0..100).collect();
        let results = executor.adaptive_parallel_execute(items, |x| Ok(x * 2)).unwrap();
        
        assert_eq!(results.len(), 100);
    }
    
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_simd_operations() {
        let a_real = vec![1.0, 2.0, 3.0, 4.0];
        let a_imag = vec![1.0, 1.0, 1.0, 1.0];
        let b_real = vec![2.0, 2.0, 2.0, 2.0];
        let b_imag = vec![1.0, 1.0, 1.0, 1.0];
        let mut result_real = vec![0.0; 4];
        let mut result_imag = vec![0.0; 4];
        
        unsafe {
            simd::complex_mul_simd(&a_real, &a_imag, &b_real, &b_imag, &mut result_real, &mut result_imag);
        }
        
        // Verify complex multiplication: (1+1i) * (2+1i) = 1+3i
        assert!((result_real[0] - 1.0).abs() < 1e-10);
        assert!((result_imag[0] - 3.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_adaptive_execution() {
        let config = ParallelConfig {
            load_balancing: LoadBalancingStrategy::Adaptive,
            chunk_size: 10,
            ..Default::default()
        };
        let executor = ParallelExecutor::new(config).unwrap();
        
        // Test with small dataset
        let small_items: Vec<i32> = (0..50).collect();
        let small_results = executor.adaptive_parallel_execute(small_items, |x| Ok(x + 1)).unwrap();
        assert_eq!(small_results.len(), 50);
        
        // Test with large dataset
        let large_items: Vec<i32> = (0..500).collect();
        let large_results = executor.adaptive_parallel_execute(large_items, |x| {
            // Simulate some work
            std::thread::sleep(std::time::Duration::from_micros(10));
            Ok(x + 1)
        }).unwrap();
        assert_eq!(large_results.len(), 500);
    }
    
    #[test]
    fn test_metrics_tracking() {
        let config = ParallelConfig::default();
        let executor = ParallelExecutor::new(config).unwrap();
        
        let items: Vec<i32> = (0..100).collect();
        let _results = executor.parallel_map(items, |x| Ok(x * 2)).unwrap();
        
        let metrics = executor.get_metrics();
        assert!(metrics.total_tasks > 0);
        assert_eq!(metrics.completed_tasks, metrics.total_tasks);
        assert_eq!(metrics.failed_tasks, 0);
        assert!(metrics.average_task_time >= 0.0);
    }
}