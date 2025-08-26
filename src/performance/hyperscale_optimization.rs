//! Hyperscale Optimization for Photonic Memristor Systems
//! Generation 3: MAKE IT SCALE - High-performance computing with auto-scaling capabilities

use std::sync::{Arc, Mutex, RwLock};
use std::collections::{HashMap, VecDeque};
use std::thread;
use std::time::{Duration, Instant};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Performance metrics for scaling decisions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub throughput: f64,           // Operations per second
    pub latency_p50: f64,          // 50th percentile latency (ms)
    pub latency_p95: f64,          // 95th percentile latency (ms)
    pub cpu_utilization: f64,      // CPU usage percentage
    pub memory_utilization: f64,   // Memory usage percentage
    pub error_rate: f64,          // Error rate percentage
    pub queue_depth: usize,        // Current queue size
    pub active_workers: usize,     // Number of active workers
    pub timestamp: u64,           // Unix timestamp
}

/// Auto-scaling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoScalingConfig {
    pub min_workers: usize,
    pub max_workers: usize,
    pub target_cpu_utilization: f64,
    pub target_latency_ms: f64,
    pub scale_up_threshold: f64,
    pub scale_down_threshold: f64,
    pub scale_up_cooldown: Duration,
    pub scale_down_cooldown: Duration,
    pub metrics_window_size: usize,
}

impl Default for AutoScalingConfig {
    fn default() -> Self {
        Self {
            min_workers: 2,
            max_workers: 32,
            target_cpu_utilization: 70.0,
            target_latency_ms: 100.0,
            scale_up_threshold: 80.0,
            scale_down_threshold: 40.0,
            scale_up_cooldown: Duration::from_secs(30),
            scale_down_cooldown: Duration::from_secs(60),
            metrics_window_size: 60,
        }
    }
}

/// Intelligent cache for optimized data access
#[derive(Debug)]
pub struct IntelligentCache<K: Clone + std::hash::Hash + Eq, V: Clone> {
    cache: Arc<RwLock<HashMap<K, CacheEntry<V>>>>,
    max_size: usize,
    access_tracker: Arc<Mutex<VecDeque<(K, Instant)>>>,
    hit_count: Arc<Mutex<u64>>,
    miss_count: Arc<Mutex<u64>>,
}

#[derive(Debug, Clone)]
struct CacheEntry<V> {
    value: V,
    access_count: u64,
    last_access: Instant,
    creation_time: Instant,
    ttl: Option<Duration>,
}

impl<K: Clone + std::hash::Hash + Eq, V: Clone> IntelligentCache<K, V> {
    pub fn new(max_size: usize) -> Self {
        Self {
            cache: Arc::new(RwLock::new(HashMap::new())),
            max_size,
            access_tracker: Arc::new(Mutex::new(VecDeque::new())),
            hit_count: Arc::new(Mutex::new(0)),
            miss_count: Arc::new(Mutex::new(0)),
        }
    }

    pub fn get(&self, key: &K) -> Option<V> {
        let mut cache = self.cache.write().unwrap();
        
        if let Some(entry) = cache.get_mut(key) {
            // Update access statistics
            entry.access_count += 1;
            entry.last_access = Instant::now();
            
            // Check TTL
            if let Some(ttl) = entry.ttl {
                if entry.creation_time.elapsed() > ttl {
                    cache.remove(key);
                    *self.miss_count.lock().unwrap() += 1;
                    return None;
                }
            }
            
            *self.hit_count.lock().unwrap() += 1;
            
            // Track access for LRU
            let mut tracker = self.access_tracker.lock().unwrap();
            tracker.push_back((key.clone(), Instant::now()));
            
            Some(entry.value.clone())
        } else {
            *self.miss_count.lock().unwrap() += 1;
            None
        }
    }

    pub fn put(&self, key: K, value: V, ttl: Option<Duration>) {
        let mut cache = self.cache.write().unwrap();
        
        // Evict if necessary
        if cache.len() >= self.max_size {
            self.evict_lru(&mut cache);
        }
        
        let entry = CacheEntry {
            value,
            access_count: 1,
            last_access: Instant::now(),
            creation_time: Instant::now(),
            ttl,
        };
        
        cache.insert(key.clone(), entry);
        
        // Track access
        let mut tracker = self.access_tracker.lock().unwrap();
        tracker.push_back((key, Instant::now()));
    }

    fn evict_lru(&self, cache: &mut HashMap<K, CacheEntry<V>>) {
        let mut tracker = self.access_tracker.lock().unwrap();
        
        // Find least recently used item
        if let Some((lru_key, _)) = tracker.front() {
            cache.remove(lru_key);
            tracker.pop_front();
        }
    }

    pub fn get_stats(&self) -> CacheStats {
        let hits = *self.hit_count.lock().unwrap();
        let misses = *self.miss_count.lock().unwrap();
        let total_requests = hits + misses;
        
        CacheStats {
            hit_rate: if total_requests > 0 { hits as f64 / total_requests as f64 } else { 0.0 },
            total_requests,
            cache_size: self.cache.read().unwrap().len(),
            max_size: self.max_size,
        }
    }

    pub fn clear(&self) {
        self.cache.write().unwrap().clear();
        self.access_tracker.lock().unwrap().clear();
        *self.hit_count.lock().unwrap() = 0;
        *self.miss_count.lock().unwrap() = 0;
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStats {
    pub hit_rate: f64,
    pub total_requests: u64,
    pub cache_size: usize,
    pub max_size: usize,
}

/// Resource pool for optimized resource management
pub struct ResourcePool<T> {
    resources: Arc<Mutex<VecDeque<T>>>,
    factory: Arc<dyn Fn() -> T + Send + Sync>,
    max_size: usize,
    current_size: Arc<Mutex<usize>>,
    checkout_count: Arc<Mutex<u64>>,
    checkin_count: Arc<Mutex<u64>>,
}

impl<T> ResourcePool<T> {
    pub fn new<F>(factory: F, max_size: usize) -> Self 
    where 
        F: Fn() -> T + Send + Sync + 'static,
    {
        Self {
            resources: Arc::new(Mutex::new(VecDeque::new())),
            factory: Arc::new(factory),
            max_size,
            current_size: Arc::new(Mutex::new(0)),
            checkout_count: Arc::new(Mutex::new(0)),
            checkin_count: Arc::new(Mutex::new(0)),
        }
    }

    pub fn checkout(&self) -> Option<T> {
        let mut resources = self.resources.lock().unwrap();
        
        if let Some(resource) = resources.pop_front() {
            *self.checkout_count.lock().unwrap() += 1;
            Some(resource)
        } else if *self.current_size.lock().unwrap() < self.max_size {
            // Create new resource
            let resource = (self.factory)();
            *self.current_size.lock().unwrap() += 1;
            *self.checkout_count.lock().unwrap() += 1;
            Some(resource)
        } else {
            None // Pool exhausted
        }
    }

    pub fn checkin(&self, resource: T) {
        let mut resources = self.resources.lock().unwrap();
        resources.push_back(resource);
        *self.checkin_count.lock().unwrap() += 1;
    }

    pub fn get_stats(&self) -> PoolStats {
        let resources = self.resources.lock().unwrap();
        let checkouts = *self.checkout_count.lock().unwrap();
        let checkins = *self.checkin_count.lock().unwrap();
        
        PoolStats {
            available_resources: resources.len(),
            total_size: *self.current_size.lock().unwrap(),
            max_size: self.max_size,
            utilization: 1.0 - (resources.len() as f64 / self.max_size as f64),
            checkout_count: checkouts,
            checkin_count: checkins,
            outstanding_resources: checkouts - checkins,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoolStats {
    pub available_resources: usize,
    pub total_size: usize,
    pub max_size: usize,
    pub utilization: f64,
    pub checkout_count: u64,
    pub checkin_count: u64,
    pub outstanding_resources: u64,
}

/// High-performance batch processor
pub struct BatchProcessor<T, R> {
    batch_size: usize,
    timeout: Duration,
    processor: Arc<dyn Fn(Vec<T>) -> Vec<R> + Send + Sync>,
    pending_batch: Arc<Mutex<Vec<T>>>,
    last_flush: Arc<Mutex<Instant>>,
    processed_count: Arc<Mutex<u64>>,
    batch_count: Arc<Mutex<u64>>,
}

impl<T, R> BatchProcessor<T, R> 
where 
    T: Send + 'static,
    R: Send + 'static,
{
    pub fn new<F>(batch_size: usize, timeout: Duration, processor: F) -> Self
    where
        F: Fn(Vec<T>) -> Vec<R> + Send + Sync + 'static,
    {
        Self {
            batch_size,
            timeout,
            processor: Arc::new(processor),
            pending_batch: Arc::new(Mutex::new(Vec::new())),
            last_flush: Arc::new(Mutex::new(Instant::now())),
            processed_count: Arc::new(Mutex::new(0)),
            batch_count: Arc::new(Mutex::new(0)),
        }
    }

    pub fn add(&self, item: T) -> Option<Vec<R>> {
        let mut batch = self.pending_batch.lock().unwrap();
        batch.push(item);

        // Check if we should flush
        if batch.len() >= self.batch_size || self.should_flush_by_timeout() {
            let items = std::mem::take(&mut *batch);
            drop(batch); // Release lock early
            
            *self.last_flush.lock().unwrap() = Instant::now();
            Some(self.process_batch(items))
        } else {
            None
        }
    }

    pub fn flush(&self) -> Vec<R> {
        let mut batch = self.pending_batch.lock().unwrap();
        let items = std::mem::take(&mut *batch);
        drop(batch);
        
        *self.last_flush.lock().unwrap() = Instant::now();
        self.process_batch(items)
    }

    fn should_flush_by_timeout(&self) -> bool {
        let last_flush = self.last_flush.lock().unwrap();
        last_flush.elapsed() > self.timeout
    }

    fn process_batch(&self, items: Vec<T>) -> Vec<R> {
        if items.is_empty() {
            return Vec::new();
        }

        let result = (self.processor)(items);
        
        *self.processed_count.lock().unwrap() += result.len() as u64;
        *self.batch_count.lock().unwrap() += 1;
        
        result
    }

    pub fn get_stats(&self) -> BatchStats {
        let processed = *self.processed_count.lock().unwrap();
        let batches = *self.batch_count.lock().unwrap();
        let pending = self.pending_batch.lock().unwrap().len();
        
        BatchStats {
            processed_items: processed,
            batch_count: batches,
            pending_items: pending,
            avg_batch_size: if batches > 0 { processed as f64 / batches as f64 } else { 0.0 },
            batch_size_limit: self.batch_size,
            timeout_ms: self.timeout.as_millis() as u64,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchStats {
    pub processed_items: u64,
    pub batch_count: u64,
    pub pending_items: usize,
    pub avg_batch_size: f64,
    pub batch_size_limit: usize,
    pub timeout_ms: u64,
}

/// Auto-scaling orchestrator
#[derive(Debug)]
pub struct AutoScaler {
    config: AutoScalingConfig,
    metrics_history: Arc<Mutex<VecDeque<PerformanceMetrics>>>,
    last_scale_up: Arc<Mutex<Instant>>,
    last_scale_down: Arc<Mutex<Instant>>,
    current_workers: Arc<Mutex<usize>>,
    scaling_decisions: Arc<Mutex<Vec<ScalingDecision>>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingDecision {
    pub timestamp: u64,
    pub action: ScalingAction,
    pub reason: String,
    pub previous_workers: usize,
    pub new_workers: usize,
    pub metrics: PerformanceMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScalingAction {
    ScaleUp,
    ScaleDown,
    NoAction,
}

impl AutoScaler {
    pub fn new(config: AutoScalingConfig) -> Self {
        let initial_workers = config.min_workers;
        
        Self {
            config,
            metrics_history: Arc::new(Mutex::new(VecDeque::new())),
            last_scale_up: Arc::new(Mutex::new(Instant::now())),
            last_scale_down: Arc::new(Mutex::new(Instant::now())),
            current_workers: Arc::new(Mutex::new(initial_workers)),
            scaling_decisions: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub fn record_metrics(&self, metrics: PerformanceMetrics) {
        let mut history = self.metrics_history.lock().unwrap();
        history.push_back(metrics.clone());

        // Maintain window size
        while history.len() > self.config.metrics_window_size {
            history.pop_front();
        }

        // Make scaling decision
        let decision = self.make_scaling_decision(&metrics);
        
        if let ScalingAction::ScaleUp | ScalingAction::ScaleDown = decision.action {
            let mut decisions = self.scaling_decisions.lock().unwrap();
            decisions.push(decision);
        }
    }

    fn make_scaling_decision(&self, current_metrics: &PerformanceMetrics) -> ScalingDecision {
        let current_workers = *self.current_workers.lock().unwrap();
        let now = Instant::now();
        
        // Check cooldown periods
        let scale_up_ready = now.duration_since(*self.last_scale_up.lock().unwrap()) > self.config.scale_up_cooldown;
        let scale_down_ready = now.duration_since(*self.last_scale_down.lock().unwrap()) > self.config.scale_down_cooldown;

        // Calculate average metrics over window
        let history = self.metrics_history.lock().unwrap();
        let avg_cpu = if !history.is_empty() {
            history.iter().map(|m| m.cpu_utilization).sum::<f64>() / history.len() as f64
        } else {
            current_metrics.cpu_utilization
        };

        let avg_latency = if !history.is_empty() {
            history.iter().map(|m| m.latency_p95).sum::<f64>() / history.len() as f64
        } else {
            current_metrics.latency_p95
        };

        // Scaling decision logic
        let (action, reason, new_workers) = if scale_up_ready && 
            current_workers < self.config.max_workers &&
            (avg_cpu > self.config.scale_up_threshold || avg_latency > self.config.target_latency_ms * 1.5) {
            
            let new_count = std::cmp::min(current_workers * 2, self.config.max_workers);
            *self.current_workers.lock().unwrap() = new_count;
            *self.last_scale_up.lock().unwrap() = now;
            
            (ScalingAction::ScaleUp, 
             format!("High load detected: CPU={:.1}%, Latency={:.1}ms", avg_cpu, avg_latency),
             new_count)
        } else if scale_down_ready && 
            current_workers > self.config.min_workers &&
            avg_cpu < self.config.scale_down_threshold && avg_latency < self.config.target_latency_ms * 0.5 {
            
            let new_count = std::cmp::max(current_workers / 2, self.config.min_workers);
            *self.current_workers.lock().unwrap() = new_count;
            *self.last_scale_down.lock().unwrap() = now;
            
            (ScalingAction::ScaleDown,
             format!("Low load detected: CPU={:.1}%, Latency={:.1}ms", avg_cpu, avg_latency),
             new_count)
        } else {
            (ScalingAction::NoAction, "No scaling action needed".to_string(), current_workers)
        };

        ScalingDecision {
            timestamp: std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)
                .unwrap().as_secs(),
            action,
            reason,
            previous_workers: current_workers,
            new_workers,
            metrics: current_metrics.clone(),
        }
    }

    pub fn get_current_workers(&self) -> usize {
        *self.current_workers.lock().unwrap()
    }

    pub fn get_scaling_history(&self) -> Vec<ScalingDecision> {
        self.scaling_decisions.lock().unwrap().clone()
    }

    pub fn get_metrics_summary(&self) -> Option<MetricsSummary> {
        let history = self.metrics_history.lock().unwrap();
        if history.is_empty() {
            return None;
        }

        let count = history.len() as f64;
        let avg_throughput = history.iter().map(|m| m.throughput).sum::<f64>() / count;
        let avg_cpu = history.iter().map(|m| m.cpu_utilization).sum::<f64>() / count;
        let avg_memory = history.iter().map(|m| m.memory_utilization).sum::<f64>() / count;
        let avg_latency_p95 = history.iter().map(|m| m.latency_p95).sum::<f64>() / count;
        let avg_error_rate = history.iter().map(|m| m.error_rate).sum::<f64>() / count;

        Some(MetricsSummary {
            window_size: history.len(),
            avg_throughput,
            avg_cpu_utilization: avg_cpu,
            avg_memory_utilization: avg_memory,
            avg_latency_p95,
            avg_error_rate,
            current_workers: self.get_current_workers(),
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsSummary {
    pub window_size: usize,
    pub avg_throughput: f64,
    pub avg_cpu_utilization: f64,
    pub avg_memory_utilization: f64,
    pub avg_latency_p95: f64,
    pub avg_error_rate: f64,
    pub current_workers: usize,
}

/// Hyperscale photonic simulation engine
pub struct HyperscaleEngine {
    cache: IntelligentCache<String, Vec<f64>>,
    resource_pool: ResourcePool<Vec<f64>>,
    batch_processor: BatchProcessor<SimulationTask, SimulationResult>,
    auto_scaler: AutoScaler,
    performance_monitor: Arc<Mutex<PerformanceMonitor>>,
}

#[derive(Debug, Clone)]
pub struct SimulationTask {
    pub id: String,
    pub params: HashMap<String, f64>,
    pub priority: u8,
}

#[derive(Debug, Clone)]
pub struct SimulationResult {
    pub task_id: String,
    pub results: HashMap<String, f64>,
    pub execution_time_ms: u64,
    pub cache_hits: u32,
}

#[derive(Debug)]
struct PerformanceMonitor {
    start_time: Instant,
    total_tasks: u64,
    completed_tasks: u64,
    total_execution_time: Duration,
    error_count: u64,
}

impl HyperscaleEngine {
    pub fn new(config: AutoScalingConfig) -> Self {
        let cache = IntelligentCache::new(10000);
        let resource_pool = ResourcePool::new(|| vec![0.0; 1000], 100);
        
        let batch_processor = BatchProcessor::new(
            32,  // batch size
            Duration::from_millis(50),  // timeout
            |tasks: Vec<SimulationTask>| -> Vec<SimulationResult> {
                // Parallel processing using rayon
                tasks.into_par_iter().map(|task| {
                    let start = Instant::now();
                    
                    // Simulate computation
                    let mut results = HashMap::new();
                    results.insert("conductance".to_string(), 
                                 task.params.get("voltage").unwrap_or(&0.0) * 1e-6);
                    results.insert("temperature".to_string(), 
                                 300.0 + task.params.get("voltage").unwrap_or(&0.0) * 50.0);
                    
                    SimulationResult {
                        task_id: task.id,
                        results,
                        execution_time_ms: start.elapsed().as_millis() as u64,
                        cache_hits: 0,
                    }
                }).collect()
            }
        );
        
        let auto_scaler = AutoScaler::new(config);
        
        let performance_monitor = Arc::new(Mutex::new(PerformanceMonitor {
            start_time: Instant::now(),
            total_tasks: 0,
            completed_tasks: 0,
            total_execution_time: Duration::new(0, 0),
            error_count: 0,
        }));

        Self {
            cache,
            resource_pool,
            batch_processor,
            auto_scaler,
            performance_monitor,
        }
    }

    pub fn submit_task(&self, task: SimulationTask) -> Option<Vec<SimulationResult>> {
        // Update performance metrics
        {
            let mut monitor = self.performance_monitor.lock().unwrap();
            monitor.total_tasks += 1;
        }

        // Process task through batch processor
        let results = self.batch_processor.add(task);

        // Update completion metrics
        if let Some(ref results) = results {
            let mut monitor = self.performance_monitor.lock().unwrap();
            monitor.completed_tasks += results.len() as u64;
            
            for result in results {
                monitor.total_execution_time += Duration::from_millis(result.execution_time_ms);
            }
        }

        // Record metrics for auto-scaling
        self.record_current_metrics();

        results
    }

    pub fn flush_pending(&self) -> Vec<SimulationResult> {
        let results = self.batch_processor.flush();
        
        // Update completion metrics
        let mut monitor = self.performance_monitor.lock().unwrap();
        monitor.completed_tasks += results.len() as u64;
        
        for result in &results {
            monitor.total_execution_time += Duration::from_millis(result.execution_time_ms);
        }
        
        results
    }

    fn record_current_metrics(&self) {
        let monitor = self.performance_monitor.lock().unwrap();
        
        let throughput = if monitor.start_time.elapsed().as_secs() > 0 {
            monitor.completed_tasks as f64 / monitor.start_time.elapsed().as_secs() as f64
        } else {
            0.0
        };

        let avg_latency = if monitor.completed_tasks > 0 {
            monitor.total_execution_time.as_millis() as f64 / monitor.completed_tasks as f64
        } else {
            0.0
        };

        let error_rate = if monitor.total_tasks > 0 {
            monitor.error_count as f64 / monitor.total_tasks as f64 * 100.0
        } else {
            0.0
        };

        let metrics = PerformanceMetrics {
            throughput,
            latency_p50: avg_latency * 0.8,  // Approximation
            latency_p95: avg_latency * 1.2,  // Approximation
            cpu_utilization: 50.0 + (throughput / 100.0) * 30.0,  // Simulated
            memory_utilization: 30.0 + (monitor.completed_tasks as f64 / 1000.0) * 20.0,  // Simulated
            error_rate,
            queue_depth: 0,  // Batch processor handles this
            active_workers: self.auto_scaler.get_current_workers(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        };

        drop(monitor);
        self.auto_scaler.record_metrics(metrics);
    }

    pub fn get_performance_stats(&self) -> HyperscaleStats {
        let cache_stats = self.cache.get_stats();
        let pool_stats = self.resource_pool.get_stats();
        let batch_stats = self.batch_processor.get_stats();
        let metrics_summary = self.auto_scaler.get_metrics_summary();
        let scaling_history = self.auto_scaler.get_scaling_history();

        let monitor = self.performance_monitor.lock().unwrap();
        
        HyperscaleStats {
            cache_stats,
            pool_stats,
            batch_stats,
            metrics_summary,
            scaling_decisions: scaling_history.len(),
            total_tasks: monitor.total_tasks,
            completed_tasks: monitor.completed_tasks,
            uptime_seconds: monitor.start_time.elapsed().as_secs(),
            current_workers: self.auto_scaler.get_current_workers(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HyperscaleStats {
    pub cache_stats: CacheStats,
    pub pool_stats: PoolStats,
    pub batch_stats: BatchStats,
    pub metrics_summary: Option<MetricsSummary>,
    pub scaling_decisions: usize,
    pub total_tasks: u64,
    pub completed_tasks: u64,
    pub uptime_seconds: u64,
    pub current_workers: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_intelligent_cache() {
        let cache = IntelligentCache::new(3);
        
        // Test basic operations
        cache.put("key1".to_string(), "value1".to_string(), None);
        cache.put("key2".to_string(), "value2".to_string(), None);
        
        assert_eq!(cache.get(&"key1".to_string()), Some("value1".to_string()));
        assert_eq!(cache.get(&"nonexistent".to_string()), None);
        
        let stats = cache.get_stats();
        assert!(stats.hit_rate > 0.0);
    }

    #[test]
    fn test_resource_pool() {
        let pool = ResourcePool::new(|| vec![1, 2, 3], 2);
        
        let resource1 = pool.checkout().unwrap();
        let resource2 = pool.checkout().unwrap();
        let resource3 = pool.checkout(); // Should be None (pool exhausted)
        
        assert_eq!(resource1, vec![1, 2, 3]);
        assert_eq!(resource2, vec![1, 2, 3]);
        assert!(resource3.is_none());
        
        pool.checkin(resource1);
        let resource4 = pool.checkout().unwrap(); // Should succeed now
        assert_eq!(resource4, vec![1, 2, 3]);
    }

    #[test]
    fn test_auto_scaler() {
        let config = AutoScalingConfig {
            min_workers: 1,
            max_workers: 4,
            scale_up_threshold: 80.0,
            scale_down_threshold: 20.0,
            ..Default::default()
        };
        
        let scaler = AutoScaler::new(config);
        assert_eq!(scaler.get_current_workers(), 1);
        
        // Simulate high load
        let high_load_metrics = PerformanceMetrics {
            throughput: 100.0,
            latency_p50: 50.0,
            latency_p95: 120.0,
            cpu_utilization: 90.0,
            memory_utilization: 60.0,
            error_rate: 1.0,
            queue_depth: 50,
            active_workers: 1,
            timestamp: 0,
        };
        
        scaler.record_metrics(high_load_metrics);
        
        // Should scale up (after cooldown in real scenario)
        let decisions = scaler.get_scaling_history();
        assert!(!decisions.is_empty());
    }
}