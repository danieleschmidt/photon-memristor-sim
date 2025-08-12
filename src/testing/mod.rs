//! Comprehensive testing framework for photonic simulation components

pub mod unit_tests;
pub mod integration_tests;
pub mod performance_tests;
pub mod security_tests;
pub mod coverage_analysis;

use crate::core::Result;
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Test execution statistics
#[derive(Debug, Clone)]
pub struct TestStats {
    pub total_tests: usize,
    pub passed: usize,
    pub failed: usize,
    pub skipped: usize,
    pub execution_time: Duration,
    pub coverage_percentage: f64,
    pub memory_usage_mb: f64,
}

impl TestStats {
    pub fn new() -> Self {
        Self {
            total_tests: 0,
            passed: 0,
            failed: 0,
            skipped: 0,
            execution_time: Duration::ZERO,
            coverage_percentage: 0.0,
            memory_usage_mb: 0.0,
        }
    }
    
    pub fn success_rate(&self) -> f64 {
        if self.total_tests == 0 {
            0.0
        } else {
            self.passed as f64 / self.total_tests as f64 * 100.0
        }
    }
    
    pub fn summary(&self) -> String {
        format!(
            "Test Results: {}/{} passed ({:.1}%), Coverage: {:.1}%, Time: {:.2}s, Memory: {:.1}MB",
            self.passed,
            self.total_tests,
            self.success_rate(),
            self.coverage_percentage,
            self.execution_time.as_secs_f64(),
            self.memory_usage_mb
        )
    }
}

/// Test failure information
#[derive(Debug, Clone)]
pub struct TestFailure {
    pub test_name: String,
    pub error_message: String,
    pub stack_trace: String,
    pub execution_time: Duration,
}

/// Comprehensive test suite runner
pub struct TestSuite {
    pub name: String,
    pub tests: Vec<Box<dyn TestCase>>,
    pub setup_hooks: Vec<Box<dyn Fn() -> Result<()>>>,
    pub teardown_hooks: Vec<Box<dyn Fn() -> Result<()>>>,
    pub parallel_execution: bool,
    pub timeout: Duration,
}

impl TestSuite {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            tests: Vec::new(),
            setup_hooks: Vec::new(),
            teardown_hooks: Vec::new(),
            parallel_execution: false,
            timeout: Duration::from_secs(30),
        }
    }
    
    pub fn with_parallel_execution(mut self, enabled: bool) -> Self {
        self.parallel_execution = enabled;
        self
    }
    
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }
    
    pub fn add_test(&mut self, test: Box<dyn TestCase>) {
        self.tests.push(test);
    }
    
    pub async fn run(&self) -> Result<TestStats> {
        println!("🧪 Running test suite: {}", self.name);
        println!("📊 {} tests to execute", self.tests.len());
        
        let start_time = Instant::now();
        let mut stats = TestStats::new();
        stats.total_tests = self.tests.len();
        
        // Run setup hooks
        for setup in &self.setup_hooks {
            if let Err(e) = setup() {
                eprintln!("❌ Setup failed: {}", e);
                return Ok(stats);
            }
        }
        
        // Execute tests
        if self.parallel_execution {
            stats = self.run_parallel_tests().await?;
        } else {
            stats = self.run_sequential_tests().await?;
        }
        
        // Run teardown hooks
        for teardown in &self.teardown_hooks {
            if let Err(e) = teardown() {
                eprintln!("⚠️ Teardown warning: {}", e);
            }
        }
        
        stats.execution_time = start_time.elapsed();
        stats.memory_usage_mb = self.get_memory_usage();
        
        println!("✅ {}", stats.summary());
        
        Ok(stats)
    }
    
    async fn run_sequential_tests(&self) -> Result<TestStats> {
        let mut stats = TestStats::new();
        stats.total_tests = self.tests.len();
        
        for test in &self.tests {
            match test.execute().await {
                Ok(_) => {
                    stats.passed += 1;
                    print!(".");
                }
                Err(e) => {
                    stats.failed += 1;
                    print!("F");
                    eprintln!("❌ Test '{}' failed: {}", test.name(), e);
                }
            }
        }
        println!();
        
        Ok(stats)
    }
    
    async fn run_parallel_tests(&self) -> Result<TestStats> {
        // For trait objects, we'll run tests sequentially but asynchronously
        // This avoids the lifetime issues with spawning tasks that reference self
        let mut stats = TestStats::new();
        stats.total_tests = self.tests.len();
        
        for test in &self.tests {
            let test_name = test.name().to_string();
            match test.execute().await {
                Ok(_) => {
                    stats.passed += 1;
                    print!(".");
                },
                Err(e) => {
                    stats.failed += 1;
                    print!("F");
                    eprintln!("❌ Test '{}' failed: {}", test_name, e);
                }
            }
        }
        println!();
        
        Ok(stats)
    }
    
    fn get_memory_usage(&self) -> f64 {
        // Simplified memory usage estimation
        use std::mem;
        let size_estimate = mem::size_of_val(&*self) + 
                          self.tests.len() * 1024; // Rough estimate per test
        size_estimate as f64 / 1024.0 / 1024.0 // Convert to MB
    }
}

/// Test case trait for individual tests  
pub trait TestCase: Send + Sync {
    fn name(&self) -> &str;
    fn execute_sync(&self) -> Result<()>; // Sync method for dyn compatibility
    
    // Default async implementation that calls execute_sync
    fn execute(&self) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<()>> + Send + '_>> {
        let result = self.execute_sync();
        Box::pin(async move { result })
    }
    
    fn timeout(&self) -> Duration {
        Duration::from_secs(10)
    }
    fn should_skip(&self) -> bool {
        false
    }
}

/// Quality gate enforcement
pub struct QualityGates {
    pub min_test_coverage: f64,
    pub min_success_rate: f64,
    pub max_execution_time: Duration,
    pub max_memory_usage: f64,
    pub security_scan_required: bool,
    pub performance_benchmarks_required: bool,
}

impl Default for QualityGates {
    fn default() -> Self {
        Self {
            min_test_coverage: 85.0,
            min_success_rate: 95.0,
            max_execution_time: Duration::from_secs(300),
            max_memory_usage: 512.0, // MB
            security_scan_required: true,
            performance_benchmarks_required: true,
        }
    }
}

impl QualityGates {
    pub fn evaluate(&self, stats: &TestStats, security_report: &SecurityReport, 
                   benchmark_results: &BenchmarkResults) -> QualityGateResult {
        let mut result = QualityGateResult::new();
        
        // Test coverage check
        if stats.coverage_percentage < self.min_test_coverage {
            result.add_failure(format!(
                "Test coverage {:.1}% below minimum {:.1}%",
                stats.coverage_percentage, self.min_test_coverage
            ));
        }
        
        // Success rate check
        if stats.success_rate() < self.min_success_rate {
            result.add_failure(format!(
                "Test success rate {:.1}% below minimum {:.1}%",
                stats.success_rate(), self.min_success_rate
            ));
        }
        
        // Execution time check
        if stats.execution_time > self.max_execution_time {
            result.add_warning(format!(
                "Test execution time {:.1}s exceeds recommended {:.1}s",
                stats.execution_time.as_secs_f64(),
                self.max_execution_time.as_secs_f64()
            ));
        }
        
        // Memory usage check
        if stats.memory_usage_mb > self.max_memory_usage {
            result.add_warning(format!(
                "Memory usage {:.1}MB exceeds recommended {:.1}MB",
                stats.memory_usage_mb, self.max_memory_usage
            ));
        }
        
        // Security scan check
        if self.security_scan_required && !security_report.vulnerabilities.is_empty() {
            let high_severity = security_report.vulnerabilities.iter()
                .filter(|v| v.severity == Severity::High || v.severity == Severity::Critical)
                .count();
            
            if high_severity > 0 {
                result.add_failure(format!(
                    "Security scan found {} high/critical vulnerabilities",
                    high_severity
                ));
            }
        }
        
        // Performance benchmark check
        if self.performance_benchmarks_required {
            for benchmark in &benchmark_results.benchmarks {
                if benchmark.execution_time > Duration::from_millis(200) {
                    result.add_warning(format!(
                        "Benchmark '{}' exceeds 200ms target: {:.1}ms",
                        benchmark.name,
                        benchmark.execution_time.as_millis()
                    ));
                }
            }
        }
        
        result
    }
}

/// Quality gate evaluation result
#[derive(Debug, Clone)]
pub struct QualityGateResult {
    pub passed: bool,
    pub failures: Vec<String>,
    pub warnings: Vec<String>,
}

impl QualityGateResult {
    pub fn new() -> Self {
        Self {
            passed: true,
            failures: Vec::new(),
            warnings: Vec::new(),
        }
    }
    
    pub fn add_failure(&mut self, message: String) {
        self.failures.push(message);
        self.passed = false;
    }
    
    pub fn add_warning(&mut self, message: String) {
        self.warnings.push(message);
    }
    
    pub fn summary(&self) -> String {
        let mut summary = if self.passed {
            "✅ QUALITY GATES PASSED".to_string()
        } else {
            "❌ QUALITY GATES FAILED".to_string()
        };
        
        if !self.failures.is_empty() {
            summary.push_str(&format!("\n\n🚨 Failures ({}):", self.failures.len()));
            for (i, failure) in self.failures.iter().enumerate() {
                summary.push_str(&format!("\n  {}. {}", i + 1, failure));
            }
        }
        
        if !self.warnings.is_empty() {
            summary.push_str(&format!("\n\n⚠️ Warnings ({}):", self.warnings.len()));
            for (i, warning) in self.warnings.iter().enumerate() {
                summary.push_str(&format!("\n  {}. {}", i + 1, warning));
            }
        }
        
        summary
    }
}

/// Security vulnerability severity levels
#[derive(Debug, Clone, PartialEq)]
pub enum Severity {
    Low,
    Medium,
    High,
    Critical,
}

/// Security vulnerability information
#[derive(Debug, Clone)]
pub struct Vulnerability {
    pub id: String,
    pub description: String,
    pub severity: Severity,
    pub file_path: Option<String>,
    pub line_number: Option<usize>,
    pub recommendation: String,
}

/// Security scan report
#[derive(Debug, Clone)]
pub struct SecurityReport {
    pub scan_time: Duration,
    pub vulnerabilities: Vec<Vulnerability>,
    pub files_scanned: usize,
    pub lines_scanned: usize,
}

impl SecurityReport {
    pub fn new() -> Self {
        Self {
            scan_time: Duration::ZERO,
            vulnerabilities: Vec::new(),
            files_scanned: 0,
            lines_scanned: 0,
        }
    }
    
    pub fn summary(&self) -> String {
        let critical = self.vulnerabilities.iter().filter(|v| v.severity == Severity::Critical).count();
        let high = self.vulnerabilities.iter().filter(|v| v.severity == Severity::High).count();
        let medium = self.vulnerabilities.iter().filter(|v| v.severity == Severity::Medium).count();
        let low = self.vulnerabilities.iter().filter(|v| v.severity == Severity::Low).count();
        
        format!(
            "Security Scan: {} files, {} vulnerabilities (Critical: {}, High: {}, Medium: {}, Low: {})",
            self.files_scanned, self.vulnerabilities.len(), critical, high, medium, low
        )
    }
}

/// Performance benchmark result
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub name: String,
    pub execution_time: Duration,
    pub throughput: f64,
    pub memory_usage: f64,
    pub iterations: usize,
}

/// Collection of benchmark results
#[derive(Debug, Clone)]
pub struct BenchmarkResults {
    pub benchmarks: Vec<BenchmarkResult>,
    pub total_time: Duration,
}

impl BenchmarkResults {
    pub fn new() -> Self {
        Self {
            benchmarks: Vec::new(),
            total_time: Duration::ZERO,
        }
    }
    
    pub fn add_result(&mut self, result: BenchmarkResult) {
        self.total_time += result.execution_time;
        self.benchmarks.push(result);
    }
    
    pub fn summary(&self) -> String {
        let avg_time = if !self.benchmarks.is_empty() {
            self.total_time.as_millis() as f64 / self.benchmarks.len() as f64
        } else {
            0.0
        };
        
        format!(
            "Performance Benchmarks: {} tests, avg {:.1}ms, total {:.1}s",
            self.benchmarks.len(),
            avg_time,
            self.total_time.as_secs_f64()
        )
    }
}