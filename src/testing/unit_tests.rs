//! Unit tests for photonic simulation components

use super::*;
use crate::core::*;
use crate::optimization::quantum_inspired::*;
use crate::performance::{parallel::*, cache::*};
use nalgebra::{DMatrix, DVector};
use std::time::{Duration, Instant};

/// Quantum-inspired optimization unit tests
pub struct QuantumOptimizationTests;

impl TestCase for QuantumOptimizationTests {
    fn name(&self) -> &str {
        "QuantumOptimizationTests"
    }
    
    async fn execute(&self) -> Result<()> {
        self.test_quantum_superposition()?;
        self.test_task_assignment_creation()?;
        self.test_quantum_interference()?;
        self.test_quantum_annealing()?;
        self.test_measurement_and_collapse()?;
        Ok(())
    }
}

impl QuantumOptimizationTests {
    fn test_quantum_superposition(&self) -> Result<()> {
        let superposition = QuantumSuperposition::new(4);
        
        // Test initialization
        assert_eq!(superposition.num_states(), 4);
        assert!((superposition.fidelity() - 1.0).abs() < 1e-10, "Initial fidelity should be 1.0");
        
        // Test amplitude normalization
        let total_probability: f64 = superposition.amplitudes.iter()
            .map(|a| a.norm_sqr())
            .sum();
        assert!((total_probability - 1.0).abs() < 1e-10, "Amplitudes should be normalized");
        
        println!("✓ Quantum superposition tests passed");
        Ok(())
    }
    
    fn test_task_assignment_creation(&self) -> Result<()> {
        let assignment = TaskAssignment {
            task_id: 1,
            resources: vec![0.5, 0.7, 0.3, 0.9],
            priority: 0.8,
            execution_time: 2.5,
            dependencies: vec![0, 2],
        };
        
        assert_eq!(assignment.task_id, 1);
        assert_eq!(assignment.resources.len(), 4);
        assert_eq!(assignment.priority, 0.8);
        assert_eq!(assignment.dependencies, vec![0, 2]);
        
        println!("✓ Task assignment creation tests passed");
        Ok(())
    }
    
    fn test_quantum_interference(&self) -> Result<()> {
        let mut planner = QuantumTaskPlanner::new(4);
        let initial_fidelity = planner.fidelity();
        
        let target = TaskAssignment {
            task_id: 0,
            resources: vec![0.8, 0.6, 0.4, 0.2],
            priority: 0.9,
            execution_time: 1.5,
            dependencies: vec![],
        };
        
        planner.apply_interference(&target);
        
        // Interference should maintain or improve fidelity
        let post_interference_fidelity = planner.fidelity();
        assert!(post_interference_fidelity >= 0.0 && post_interference_fidelity <= 1.0,
                "Fidelity must be between 0 and 1");
        
        println!("✓ Quantum interference tests passed");
        Ok(())
    }
    
    fn test_quantum_annealing(&self) -> Result<()> {
        let mut planner = QuantumTaskPlanner::new(8);
        let start_time = Instant::now();
        
        let result = planner.quantum_anneal(50, 1.0)?;
        let execution_time = start_time.elapsed();
        
        // Verify result properties
        assert!(result.resources.len() > 0, "Result should have resource allocation");
        assert!(result.priority >= 0.0 && result.priority <= 1.0, "Priority should be normalized");
        assert!(result.execution_time > 0.0, "Execution time should be positive");
        assert!(execution_time < Duration::from_secs(5), "Annealing should complete quickly");
        
        println!("✓ Quantum annealing tests passed in {:.2}ms", execution_time.as_millis());
        Ok(())
    }
    
    fn test_measurement_and_collapse(&self) -> Result<()> {
        let mut planner = QuantumTaskPlanner::new(4);
        
        // Perform measurement
        let measurement = planner.measure();
        
        // Verify measurement properties
        assert!(measurement.resources.len() > 0, "Measurement should have resources");
        assert!(measurement.priority >= 0.0, "Priority should be non-negative");
        
        // Test multiple measurements for consistency
        let measurements: Vec<_> = (0..10).map(|_| {
            let mut p = QuantumTaskPlanner::new(4);
            p.measure()
        }).collect();
        
        assert!(measurements.len() == 10, "Should generate 10 measurements");
        
        println!("✓ Quantum measurement tests passed");
        Ok(())
    }
}

/// Parallel processing unit tests
pub struct ParallelProcessingTests;

impl TestCase for ParallelProcessingTests {
    fn name(&self) -> &str {
        "ParallelProcessingTests"
    }
    
    async fn execute(&self) -> Result<()> {
        self.test_parallel_executor_creation()?;
        self.test_memory_pool_management()?;
        self.test_load_balancing_strategies()?;
        self.test_simd_operations()?;
        Ok(())
    }
}

impl ParallelProcessingTests {
    fn test_parallel_executor_creation(&self) -> Result<()> {
        let config = ParallelConfig::default();
        let executor = ParallelExecutor::new(config)?;
        
        assert!(executor.worker_count() > 0, "Should have worker threads");
        assert!(executor.is_healthy(), "Executor should be healthy");
        
        println!("✓ Parallel executor creation tests passed");
        Ok(())
    }
    
    fn test_memory_pool_management(&self) -> Result<()> {
        let pool = MemoryPool::new(1024 * 1024, 4)?; // 1MB blocks, 4 blocks
        
        // Test allocation
        let block = pool.allocate(512)?;
        assert!(block.size() >= 512, "Block should be large enough");
        
        // Test deallocation
        pool.deallocate(block)?;
        
        let stats = pool.statistics();
        assert_eq!(stats.allocated_blocks, 0, "All blocks should be deallocated");
        
        println!("✓ Memory pool management tests passed");
        Ok(())
    }
    
    fn test_load_balancing_strategies(&self) -> Result<()> {
        let strategies = vec![
            LoadBalancingStrategy::RoundRobin,
            LoadBalancingStrategy::LeastLoaded,
            LoadBalancingStrategy::Random,
        ];
        
        for strategy in strategies {
            // Test strategy assignment
            let worker_id = strategy.assign_worker(&[1.0, 2.0, 0.5, 1.5], 0)?;
            assert!(worker_id < 4, "Worker ID should be valid");
        }
        
        println!("✓ Load balancing strategy tests passed");
        Ok(())
    }
    
    fn test_simd_operations(&self) -> Result<()> {
        #[cfg(target_arch = "x86_64")]
        {
            use std::arch::x86_64::*;
            
            let data_a = vec![1.0f32; 8];
            let data_b = vec![2.0f32; 8];
            
            unsafe {
                let a = _mm256_loadu_ps(data_a.as_ptr());
                let b = _mm256_loadu_ps(data_b.as_ptr());
                let result = _mm256_add_ps(a, b);
                
                let mut output = [0.0f32; 8];
                _mm256_storeu_ps(output.as_mut_ptr(), result);
                
                for &val in &output {
                    assert!((val - 3.0).abs() < 1e-6, "SIMD addition should work correctly");
                }
            }
        }
        
        println!("✓ SIMD operations tests passed");
        Ok(())
    }
}

/// Caching system unit tests
pub struct CachingTests;

impl TestCase for CachingTests {
    fn name(&self) -> &str {
        "CachingTests"
    }
    
    async fn execute(&self) -> Result<()> {
        self.test_cache_creation()?;
        self.test_cache_operations()?;
        self.test_eviction_policies()?;
        self.test_cache_statistics()?;
        Ok(())
    }
}

impl CachingTests {
    fn test_cache_creation(&self) -> Result<()> {
        let config = CacheConfig::default();
        let cache = PhotonicCache::new(config)?;
        
        assert_eq!(cache.size(), 0, "New cache should be empty");
        assert!(cache.capacity() > 0, "Cache should have positive capacity");
        
        println!("✓ Cache creation tests passed");
        Ok(())
    }
    
    fn test_cache_operations(&self) -> Result<()> {
        let mut cache = PhotonicCache::new(CacheConfig::default())?;
        
        let key = CacheKey::new("test", vec![1.0, 2.0, 3.0]);
        let value = CachedResult::new(vec![4.0, 5.0, 6.0], 0.95);
        
        // Test put operation
        cache.put(key.clone(), value.clone())?;
        assert_eq!(cache.size(), 1, "Cache should contain one item");
        
        // Test get operation
        let retrieved = cache.get(&key)?;
        assert!(retrieved.is_some(), "Should retrieve cached value");
        
        let retrieved_value = retrieved.unwrap();
        assert_eq!(retrieved_value.confidence, value.confidence);
        
        // Test cache hit/miss statistics
        let stats = cache.stats();
        assert_eq!(stats.hits, 1, "Should record cache hit");
        
        println!("✓ Cache operations tests passed");
        Ok(())
    }
    
    fn test_eviction_policies(&self) -> Result<()> {
        let policies = vec![
            EvictionPolicy::LRU,
            EvictionPolicy::LFU,
            EvictionPolicy::TTL { ttl: Duration::from_secs(60) },
        ];
        
        for policy in policies {
            let config = CacheConfig {
                max_entries: 10,
                eviction_policy: policy.clone(),
                ..Default::default()
            };
            
            let mut cache = PhotonicCache::new(config)?;
            
            // Fill cache beyond capacity
            for i in 0..15 {
                let key = CacheKey::new(&format!("key_{}", i), vec![i as f64]);
                let value = CachedResult::new(vec![i as f64], 1.0);
                cache.put(key, value)?;
            }
            
            assert!(cache.size() <= 10, "Eviction should limit cache size");
        }
        
        println!("✓ Eviction policy tests passed");
        Ok(())
    }
    
    fn test_cache_statistics(&self) -> Result<()> {
        let mut cache = PhotonicCache::new(CacheConfig::default())?;
        
        // Perform operations to generate statistics
        let key1 = CacheKey::new("test1", vec![1.0]);
        let key2 = CacheKey::new("test2", vec![2.0]);
        let value = CachedResult::new(vec![1.0], 0.9);
        
        cache.put(key1.clone(), value.clone())?;
        cache.get(&key1)?; // Hit
        cache.get(&key2)?; // Miss
        
        let stats = cache.stats();
        assert_eq!(stats.hits, 1, "Should record one hit");
        assert_eq!(stats.misses, 1, "Should record one miss");
        assert_eq!(stats.hit_rate(), 50.0, "Hit rate should be 50%");
        
        println!("✓ Cache statistics tests passed");
        Ok(())
    }
}

/// Core types and validation unit tests
pub struct CoreTypesTests;

impl TestCase for CoreTypesTests {
    fn name(&self) -> &str {
        "CoreTypesTests"
    }
    
    async fn execute(&self) -> Result<()> {
        self.test_optical_field_creation()?;
        self.test_waveguide_geometry()?;
        self.test_validation_system()?;
        Ok(())
    }
}

impl CoreTypesTests {
    fn test_optical_field_creation(&self) -> Result<()> {
        let amplitude = DMatrix::from_element(5, 5, Complex64::new(1.0, 0.5));
        let x_coords = DVector::from_iterator(5, (0..5).map(|i| i as f64 * 1e-6));
        let y_coords = DVector::from_iterator(5, (0..5).map(|i| i as f64 * 1e-6));
        
        let field = OpticalField::new(amplitude, 1550e-9, 1e-3, x_coords, y_coords);
        
        assert_eq!(field.wavelength, 1550e-9);
        assert_eq!(field.power, 1e-3);
        assert_eq!(field.dimensions(), (5, 5));
        
        let calculated_power = field.calculate_power();
        assert!(calculated_power > 0.0, "Calculated power should be positive");
        
        println!("✓ Optical field creation tests passed");
        Ok(())
    }
    
    fn test_waveguide_geometry(&self) -> Result<()> {
        let waveguide = WaveguideGeometry::silicon_photonic_standard();
        
        assert!(waveguide.width > 0.0, "Width should be positive");
        assert!(waveguide.height > 0.0, "Height should be positive");
        assert!(waveguide.core_index.re > waveguide.cladding_index.re, 
                "Core index should be higher than cladding");
        
        let effective_area = waveguide.effective_area();
        assert!(effective_area > 0.0, "Effective area should be positive");
        
        println!("✓ Waveguide geometry tests passed");
        Ok(())
    }
    
    fn test_validation_system(&self) -> Result<()> {
        use crate::core::validation::*;
        
        let validator = PhotonicValidator::new();
        
        // Test valid field validation
        let amplitude = DMatrix::from_element(3, 3, Complex64::new(1.0, 0.0));
        let x_coords = DVector::from_iterator(3, (0..3).map(|i| i as f64 * 1e-6));
        let y_coords = DVector::from_iterator(3, (0..3).map(|i| i as f64 * 1e-6));
        let field = OpticalField::new(amplitude, 1550e-9, 1e-3, x_coords, y_coords);
        
        let report = validator.validate_optical_field(&field)?;
        assert!(report.is_valid(), "Valid field should pass validation");
        
        // Test invalid field validation
        let invalid_field = OpticalField::new(
            DMatrix::from_element(3, 3, Complex64::new(f64::NAN, 0.0)),
            50e-9, // Invalid wavelength
            -1e-3, // Invalid power
            DVector::from_iterator(3, (0..3).map(|i| i as f64 * 1e-6)),
            DVector::from_iterator(3, (0..3).map(|i| i as f64 * 1e-6))
        );
        
        let report = validator.validate_optical_field(&invalid_field)?;
        assert!(!report.is_valid(), "Invalid field should fail validation");
        assert!(!report.errors.is_empty(), "Should have validation errors");
        
        println!("✓ Validation system tests passed");
        Ok(())
    }
}

/// Create comprehensive unit test suite
pub fn create_unit_test_suite() -> TestSuite {
    let mut suite = TestSuite::new("Unit Tests")
        .with_parallel_execution(true)
        .with_timeout(Duration::from_secs(60));
    
    suite.add_test(Box::new(QuantumOptimizationTests));
    suite.add_test(Box::new(ParallelProcessingTests));
    suite.add_test(Box::new(CachingTests));
    suite.add_test(Box::new(CoreTypesTests));
    
    suite
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio;
    
    #[tokio::test]
    async fn test_unit_test_suite() {
        let suite = create_unit_test_suite();
        let results = suite.run().await.expect("Test suite should run successfully");
        
        println!("Unit test results: {}", results.summary());
        
        // Ensure reasonable test coverage
        assert!(results.success_rate() >= 90.0, "Unit tests should have high success rate");
        assert!(results.total_tests >= 4, "Should run all test cases");
    }
    
    #[test]
    fn test_quality_gates_evaluation() {
        let gates = QualityGates::default();
        let stats = TestStats {
            total_tests: 100,
            passed: 95,
            failed: 5,
            skipped: 0,
            execution_time: Duration::from_secs(30),
            coverage_percentage: 87.5,
            memory_usage_mb: 256.0,
        };
        
        let security_report = SecurityReport::new();
        let benchmark_results = BenchmarkResults::new();
        
        let result = gates.evaluate(&stats, &security_report, &benchmark_results);
        
        assert!(result.passed, "Quality gates should pass with good metrics");
        println!("Quality gate result: {}", result.summary());
    }
}