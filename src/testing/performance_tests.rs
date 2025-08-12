//! Performance benchmarks and testing for photonic simulation components

use super::*;
use crate::core::*;
use crate::optimization::quantum_inspired::*;
use crate::performance::{parallel::*, cache::*};
use std::time::{Duration, Instant};
use std::collections::HashMap;

/// Performance benchmark runner
pub struct BenchmarkRunner {
    warmup_iterations: usize,
    measurement_iterations: usize,
    target_duration: Duration,
    statistical_significance: f64,
}

impl Default for BenchmarkRunner {
    fn default() -> Self {
        Self {
            warmup_iterations: 5,
            measurement_iterations: 50,
            target_duration: Duration::from_millis(200),
            statistical_significance: 0.95,
        }
    }
}

impl BenchmarkRunner {
    pub fn new() -> Self {
        Self::default()
    }
    
    pub fn with_iterations(mut self, warmup: usize, measurement: usize) -> Self {
        self.warmup_iterations = warmup;
        self.measurement_iterations = measurement;
        self
    }
    
    pub fn with_target_duration(mut self, duration: Duration) -> Self {
        self.target_duration = duration;
        self
    }
    
    pub fn benchmark<F, T>(&self, name: &str, mut operation: F) -> BenchmarkResult
    where
        F: FnMut() -> T,
    {
        println!("📊 Benchmarking: {}", name);
        
        // Warmup phase
        for _ in 0..self.warmup_iterations {
            let _ = operation();
        }
        
        // Measurement phase
        let mut measurements = Vec::with_capacity(self.measurement_iterations);
        let start_time = Instant::now();
        
        for _ in 0..self.measurement_iterations {
            let iter_start = Instant::now();
            let _ = operation();
            let iter_duration = iter_start.elapsed();
            measurements.push(iter_duration);
        }
        
        let total_time = start_time.elapsed();
        
        // Calculate statistics
        let execution_time = self.calculate_median(&measurements);
        let throughput = 1.0 / execution_time.as_secs_f64();
        let memory_usage = self.estimate_memory_usage();
        
        let result = BenchmarkResult {
            name: name.to_string(),
            execution_time,
            throughput,
            memory_usage,
            iterations: self.measurement_iterations,
        };
        
        let status = if execution_time <= self.target_duration {
            "✅ PASS"
        } else {
            "⚠️ SLOW"
        };
        
        println!("   {} {:.2}ms (target: {:.0}ms), {:.0} ops/sec",
                status,
                execution_time.as_millis(),
                self.target_duration.as_millis(),
                throughput);
        
        result
    }
    
    fn calculate_median(&self, measurements: &[Duration]) -> Duration {
        let mut sorted = measurements.to_vec();
        sorted.sort();
        
        let mid = sorted.len() / 2;
        if sorted.len() % 2 == 0 {
            Duration::from_nanos(
                ((sorted[mid - 1].as_nanos() + sorted[mid].as_nanos()) / 2) as u64
            )
        } else {
            sorted[mid]
        }
    }
    
    fn estimate_memory_usage(&self) -> f64 {
        // Simplified memory estimation - in production, use proper memory profiling
        use std::process::Command;
        
        if let Ok(output) = Command::new("ps")
            .args(&["-o", "rss", "-p"])
            .arg(std::process::id().to_string())
            .output()
        {
            if let Ok(output_str) = String::from_utf8(output.stdout) {
                if let Some(rss_line) = output_str.lines().nth(1) {
                    if let Ok(rss_kb) = rss_line.trim().parse::<f64>() {
                        return rss_kb / 1024.0; // Convert to MB
                    }
                }
            }
        }
        
        0.0 // Fallback if memory measurement fails
    }
}

/// Quantum optimization benchmarks
pub struct QuantumOptimizationBenchmarks {
    runner: BenchmarkRunner,
}

impl QuantumOptimizationBenchmarks {
    pub fn new() -> Self {
        Self {
            runner: BenchmarkRunner::default(),
        }
    }
    
    pub fn run_all(&self) -> Vec<BenchmarkResult> {
        let mut results = Vec::new();
        
        results.push(self.benchmark_superposition_creation());
        results.push(self.benchmark_quantum_evolution());
        results.push(self.benchmark_interference_application());
        results.push(self.benchmark_measurement_collapse());
        results.push(self.benchmark_annealing_optimization());
        results.push(self.benchmark_classical_comparison());
        
        results
    }
    
    fn benchmark_superposition_creation(&self) -> BenchmarkResult {
        self.runner.benchmark("quantum_superposition_creation", || {
            QuantumSuperposition::new(16)
        })
    }
    
    fn benchmark_quantum_evolution(&self) -> BenchmarkResult {
        let mut planner = QuantumTaskPlanner::new(8).unwrap();
        
        self.runner.benchmark("quantum_evolution", || {
            planner.evolve(0.01);
        })
    }
    
    fn benchmark_interference_application(&self) -> BenchmarkResult {
        let mut planner = QuantumTaskPlanner::new(8).unwrap();
        let target = TaskAssignment {
            task_id: 0,
            resources: vec![0.8, 0.6, 0.4, 0.2, 0.1, 0.3, 0.5, 0.7],
            priority: 0.9,
            execution_time: 1.5,
            dependencies: vec![],
        };
        
        self.runner.benchmark("quantum_interference", || {
            planner.apply_interference(&target);
        })
    }
    
    fn benchmark_measurement_collapse(&self) -> BenchmarkResult {
        let mut planner = QuantumTaskPlanner::new(8).unwrap();
        
        self.runner.benchmark("quantum_measurement", || {
            planner.measure()
        })
    }
    
    fn benchmark_annealing_optimization(&self) -> BenchmarkResult {
        let mut planner = QuantumTaskPlanner::new(8).unwrap();
        
        self.runner.benchmark("quantum_annealing", || {
            planner.quantum_anneal(20, 1.0).unwrap()
        })
    }
    
    fn benchmark_classical_comparison(&self) -> BenchmarkResult {
        self.runner.benchmark("classical_optimization", || {
            // Simulate classical greedy optimization
            let resources = vec![0.8, 0.6, 0.4, 0.2, 0.1, 0.3, 0.5, 0.7];
            let mut best_score = 0.0;
            let mut best_allocation = resources.clone();
            
            for _ in 0..100 {
                let mut allocation = resources.clone();
                for i in 0..allocation.len() {
                    allocation[i] += (rand::random::<f64>() - 0.5) * 0.1;
                    allocation[i] = allocation[i].max(0.0).min(1.0);
                }
                
                let score: f64 = allocation.iter().sum();
                if score > best_score {
                    best_score = score;
                    best_allocation = allocation;
                }
            }
            
            TaskAssignment {
                task_id: 0,
                resources: best_allocation,
                priority: 0.8,
                execution_time: 2.0,
                dependencies: vec![],
            }
        })
    }
}

/// Parallel processing benchmarks
pub struct ParallelProcessingBenchmarks {
    runner: BenchmarkRunner,
}

impl ParallelProcessingBenchmarks {
    pub fn new() -> Self {
        Self {
            runner: BenchmarkRunner::default(),
        }
    }
    
    pub fn run_all(&self) -> Vec<BenchmarkResult> {
        let mut results = Vec::new();
        
        results.push(self.benchmark_thread_pool_creation());
        results.push(self.benchmark_parallel_map_operation());
        results.push(self.benchmark_load_balancing());
        results.push(self.benchmark_memory_pool_allocation());
        results.push(self.benchmark_simd_operations());
        
        results
    }
    
    fn benchmark_thread_pool_creation(&self) -> BenchmarkResult {
        self.runner.benchmark("thread_pool_creation", || {
            let config = ParallelConfig::default();
            ParallelExecutor::new(config).unwrap()
        })
    }
    
    fn benchmark_parallel_map_operation(&self) -> BenchmarkResult {
        let config = ParallelConfig::default();
        let executor = ParallelExecutor::new(config).unwrap();
        let data: Vec<f64> = (0..1000).map(|i| i as f64).collect();
        
        self.runner.benchmark("parallel_map", || {
            executor.parallel_map(data.clone(), |x| Ok(x * x + 1.0)).unwrap()
        })
    }
    
    fn benchmark_load_balancing(&self) -> BenchmarkResult {
        let loads = vec![0.2, 0.8, 0.5, 0.1, 0.9, 0.3, 0.7, 0.4];
        let strategies = vec![
            LoadBalancingStrategy::RoundRobin,
            LoadBalancingStrategy::LeastLoaded,
            LoadBalancingStrategy::Random,
        ];
        
        self.runner.benchmark("load_balancing", || {
            for strategy in &strategies {
                for i in 0..100 {
                    strategy.assign_worker(&loads, i).unwrap();
                }
            }
        })
    }
    
    fn benchmark_memory_pool_allocation(&self) -> BenchmarkResult {
        let pool = MemoryPool::with_config(1024 * 1024, 8).unwrap(); // 1MB blocks
        
        self.runner.benchmark("memory_pool_allocation", || {
            let mut blocks = Vec::new();
            
            // Allocate blocks
            for _ in 0..10 {
                let block = pool.allocate(64 * 1024); // 64KB
                blocks.push(block);
            }
            
            // Deallocate blocks
            for block in blocks {
                let _ = pool.deallocate(block);
            }
        })
    }
    
    fn benchmark_simd_operations(&self) -> BenchmarkResult {
        let data_a: Vec<f32> = (0..1024).map(|i| i as f32).collect();
        let data_b: Vec<f32> = (0..1024).map(|i| (i * 2) as f32).collect();
        
        self.runner.benchmark("simd_vectorized_add", || {
            #[cfg(target_arch = "x86_64")]
            {
                use std::arch::x86_64::*;
                
                let mut result = vec![0.0f32; 1024];
                
                unsafe {
                    for i in (0..1024).step_by(8) {
                        let a = _mm256_loadu_ps(data_a.as_ptr().add(i));
                        let b = _mm256_loadu_ps(data_b.as_ptr().add(i));
                        let sum = _mm256_add_ps(a, b);
                        _mm256_storeu_ps(result.as_mut_ptr().add(i), sum);
                    }
                }
                
                result
            }
            
            #[cfg(not(target_arch = "x86_64"))]
            {
                data_a.iter().zip(data_b.iter()).map(|(a, b)| a + b).collect::<Vec<f32>>()
            }
        })
    }
}

/// Caching system benchmarks
pub struct CachingBenchmarks {
    runner: BenchmarkRunner,
}

impl CachingBenchmarks {
    pub fn new() -> Self {
        Self {
            runner: BenchmarkRunner::default(),
        }
    }
    
    pub fn run_all(&self) -> Vec<BenchmarkResult> {
        let mut results = Vec::new();
        
        results.push(self.benchmark_cache_creation());
        results.push(self.benchmark_cache_put_operation());
        results.push(self.benchmark_cache_get_operation());
        results.push(self.benchmark_cache_eviction());
        results.push(self.benchmark_cache_hit_rate());
        
        results
    }
    
    fn benchmark_cache_creation(&self) -> BenchmarkResult {
        self.runner.benchmark("cache_creation", || {
            let config = CacheConfig::default();
            PhotonicCache::<Vec<f64>>::new(config).unwrap()
        })
    }
    
    fn benchmark_cache_put_operation(&self) -> BenchmarkResult {
        let mut cache = PhotonicCache::new(CacheConfig::default()).unwrap();
        let keys: Vec<CacheKey> = (0..100)
            .map(|i| CacheKey::from_params(&format!("key_{}", i), vec![i as f64]))
            .collect();
        let values: Vec<CachedResult<Vec<f64>>> = (0..100)
            .map(|i| CachedResult::with_estimated_size(vec![i as f64], 0.9))
            .collect();
        
        self.runner.benchmark("cache_put", || {
            for (key, value) in keys.iter().zip(values.iter()) {
                cache.put_cached(key.clone(), value.clone()).unwrap();
            }
        })
    }
    
    fn benchmark_cache_get_operation(&self) -> BenchmarkResult {
        let mut cache = PhotonicCache::new(CacheConfig::default()).unwrap();
        let keys: Vec<CacheKey> = (0..100)
            .map(|i| CacheKey::from_params(&format!("key_{}", i), vec![i as f64]))
            .collect();
        
        // Populate cache
        for (i, key) in keys.iter().enumerate() {
            let value = CachedResult::with_estimated_size(vec![i as f64], 0.9);
            cache.put_cached(key.clone(), value).unwrap();
        }
        
        self.runner.benchmark("cache_get", || {
            for key in &keys {
                let _ = cache.get(key);
            }
        })
    }
    
    fn benchmark_cache_eviction(&self) -> BenchmarkResult {
        let config = CacheConfig {
            max_entries: 50,
            eviction_policy: EvictionPolicy::LRU,
            ..Default::default()
        };
        let mut cache = PhotonicCache::new(config).unwrap();
        
        self.runner.benchmark("cache_eviction", || {
            // Add more entries than the cache can hold
            for i in 0..100 {
                let key = CacheKey::from_params(&format!("key_{}", i), vec![i as f64]);
                let value = CachedResult::with_estimated_size(vec![i as f64], 0.9);
                cache.put_cached(key, value).unwrap();
            }
        })
    }
    
    fn benchmark_cache_hit_rate(&self) -> BenchmarkResult {
        let mut cache = PhotonicCache::new(CacheConfig::default()).unwrap();
        let keys: Vec<CacheKey> = (0..100)
            .map(|i| CacheKey::from_params(&format!("key_{}", i), vec![i as f64]))
            .collect();
        
        // Populate cache with some keys
        for (i, key) in keys.iter().take(50).enumerate() {
            let value = CachedResult::with_estimated_size(vec![i as f64], 0.9);
            cache.put_cached(key.clone(), value).unwrap();
        }
        
        self.runner.benchmark("cache_hit_rate_mixed", || {
            for key in &keys {
                let _ = cache.get(key); // Mix of hits and misses
            }
        })
    }
}

/// Comprehensive performance test suite
pub async fn run_performance_tests() -> Result<BenchmarkResults> {
    println!("🚀 Running comprehensive performance benchmarks...");
    println!("Target: <200ms per operation for optimal performance");
    println!("{}", "=".repeat(70));
    
    let mut results = BenchmarkResults::new();
    let overall_start = Instant::now();
    
    // Quantum optimization benchmarks
    println!("\n🌌 Quantum Optimization Benchmarks");
    println!("{}", "-".repeat(40));
    let quantum_benchmarks = QuantumOptimizationBenchmarks::new();
    let quantum_results = quantum_benchmarks.run_all();
    for result in quantum_results {
        results.add_result(result);
    }
    
    // Parallel processing benchmarks
    println!("\n⚡ Parallel Processing Benchmarks");
    println!("{}", "-".repeat(40));
    let parallel_benchmarks = ParallelProcessingBenchmarks::new();
    let parallel_results = parallel_benchmarks.run_all();
    for result in parallel_results {
        results.add_result(result);
    }
    
    // Caching system benchmarks
    println!("\n💾 Caching System Benchmarks");
    println!("{}", "-".repeat(40));
    let caching_benchmarks = CachingBenchmarks::new();
    let caching_results = caching_benchmarks.run_all();
    for result in caching_results {
        results.add_result(result);
    }
    
    // Additional system-level benchmarks
    println!("\n🔬 System-Level Benchmarks");
    println!("{}", "-".repeat(40));
    let system_results = run_system_benchmarks().await?;
    for result in system_results {
        results.add_result(result);
    }
    
    results.total_time = overall_start.elapsed();
    
    // Performance summary
    println!("\n📊 Performance Summary");
    println!("{}", "=".repeat(50));
    println!("{}", results.summary());
    
    let fast_benchmarks = results.benchmarks.iter()
        .filter(|b| b.execution_time <= Duration::from_millis(200))
        .count();
    
    let success_rate = (fast_benchmarks as f64 / results.benchmarks.len() as f64) * 100.0;
    
    println!("\n🎯 Performance Targets:");
    println!("   Fast operations (<200ms): {}/{} ({:.1}%)",
            fast_benchmarks, results.benchmarks.len(), success_rate);
    
    if success_rate >= 80.0 {
        println!("✅ Performance targets met!");
    } else {
        println!("⚠️ Performance needs optimization");
    }
    
    // Identify bottlenecks
    let mut slow_operations: Vec<_> = results.benchmarks.iter()
        .filter(|b| b.execution_time > Duration::from_millis(200))
        .collect();
    slow_operations.sort_by_key(|b| std::cmp::Reverse(b.execution_time));
    
    if !slow_operations.is_empty() {
        println!("\n🐌 Optimization Opportunities:");
        for (i, benchmark) in slow_operations.iter().take(5).enumerate() {
            println!("   {}. {} - {:.1}ms",
                    i + 1, benchmark.name, benchmark.execution_time.as_millis());
        }
    }
    
    Ok(results)
}

/// System-level performance benchmarks
async fn run_system_benchmarks() -> Result<Vec<BenchmarkResult>> {
    let runner = BenchmarkRunner::default();
    let mut results = Vec::new();
    
    // Memory allocation benchmark
    results.push(runner.benchmark("memory_allocation", || {
        let mut vectors = Vec::new();
        for _ in 0..100 {
            vectors.push(vec![0.0f64; 1000]);
        }
        vectors
    }));
    
    // File I/O benchmark
    results.push(runner.benchmark("file_io", || {
        use std::fs;
        let data = "test data ".repeat(1000);
        fs::write("/tmp/benchmark_test.txt", &data).unwrap();
        let read_data = fs::read_to_string("/tmp/benchmark_test.txt").unwrap();
        let _ = fs::remove_file("/tmp/benchmark_test.txt");
        read_data
    }));
    
    // CPU intensive computation
    results.push(runner.benchmark("cpu_intensive", || {
        let mut sum = 0.0;
        for i in 0..10000 {
            sum += (i as f64).sin().cos().tan();
        }
        sum
    }));
    
    Ok(results)
}

/// Performance regression detector
pub struct RegressionDetector {
    baseline_results: HashMap<String, BenchmarkResult>,
    tolerance_percentage: f64,
}

impl RegressionDetector {
    pub fn new(tolerance_percentage: f64) -> Self {
        Self {
            baseline_results: HashMap::new(),
            tolerance_percentage,
        }
    }
    
    pub fn set_baseline(&mut self, results: &BenchmarkResults) {
        self.baseline_results.clear();
        for result in &results.benchmarks {
            self.baseline_results.insert(result.name.clone(), result.clone());
        }
    }
    
    pub fn detect_regressions(&self, current_results: &BenchmarkResults) -> Vec<RegressionReport> {
        let mut reports = Vec::new();
        
        for current in &current_results.benchmarks {
            if let Some(baseline) = self.baseline_results.get(&current.name) {
                let performance_change = (current.execution_time.as_nanos() as f64 
                                        - baseline.execution_time.as_nanos() as f64) 
                                       / baseline.execution_time.as_nanos() as f64 * 100.0;
                
                if performance_change > self.tolerance_percentage {
                    reports.push(RegressionReport {
                        benchmark_name: current.name.clone(),
                        baseline_time: baseline.execution_time,
                        current_time: current.execution_time,
                        performance_change,
                        severity: if performance_change > 50.0 { 
                            RegressionSeverity::Critical 
                        } else if performance_change > 25.0 {
                            RegressionSeverity::Major
                        } else {
                            RegressionSeverity::Minor
                        },
                    });
                }
            }
        }
        
        reports
    }
}

/// Performance regression report
#[derive(Debug, Clone)]
pub struct RegressionReport {
    pub benchmark_name: String,
    pub baseline_time: Duration,
    pub current_time: Duration,
    pub performance_change: f64,
    pub severity: RegressionSeverity,
}

#[derive(Debug, Clone)]
pub enum RegressionSeverity {
    Minor,
    Major,
    Critical,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_performance_benchmarks() -> Result<()> {
        let results = run_performance_tests().await?;
        
        assert!(results.benchmarks.len() > 0, "Should have benchmark results");
        assert!(results.total_time > Duration::ZERO, "Should have total execution time");
        
        // Ensure no benchmark takes excessively long (>5 seconds)
        for benchmark in &results.benchmarks {
            assert!(benchmark.execution_time < Duration::from_secs(5),
                   "Benchmark '{}' took too long: {:.1}s", 
                   benchmark.name, benchmark.execution_time.as_secs_f64());
        }
        
        Ok(())
    }
    
    #[test]
    fn test_regression_detection() {
        let mut detector = RegressionDetector::new(20.0); // 20% tolerance
        
        let baseline = BenchmarkResults {
            benchmarks: vec![
                BenchmarkResult {
                    name: "test_operation".to_string(),
                    execution_time: Duration::from_millis(100),
                    throughput: 10.0,
                    memory_usage: 50.0,
                    iterations: 50,
                }
            ],
            total_time: Duration::from_millis(100),
        };
        
        detector.set_baseline(&baseline);
        
        // Test with regression
        let current = BenchmarkResults {
            benchmarks: vec![
                BenchmarkResult {
                    name: "test_operation".to_string(),
                    execution_time: Duration::from_millis(150), // 50% slower
                    throughput: 6.67,
                    memory_usage: 60.0,
                    iterations: 50,
                }
            ],
            total_time: Duration::from_millis(150),
        };
        
        let regressions = detector.detect_regressions(&current);
        
        assert_eq!(regressions.len(), 1);
        assert_eq!(regressions[0].benchmark_name, "test_operation");
        assert!(regressions[0].performance_change > 20.0);
    }
}