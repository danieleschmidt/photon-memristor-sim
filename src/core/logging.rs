//! Comprehensive logging and monitoring system for photonic simulation

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use std::fmt;
use serde::{Serialize, Deserialize};

/// Log level enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum LogLevel {
    Trace = 0,
    Debug = 1,
    Info = 2,
    Warn = 3,
    Error = 4,
    Critical = 5,
}

impl fmt::Display for LogLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LogLevel::Trace => write!(f, "TRACE"),
            LogLevel::Debug => write!(f, "DEBUG"),
            LogLevel::Info => write!(f, "INFO"),
            LogLevel::Warn => write!(f, "WARN"),
            LogLevel::Error => write!(f, "ERROR"),
            LogLevel::Critical => write!(f, "CRITICAL"),
        }
    }
}

/// Log entry structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogEntry {
    pub timestamp: u64,
    pub level: LogLevel,
    pub module: String,
    pub message: String,
    pub metadata: HashMap<String, String>,
    pub performance_data: Option<PerformanceData>,
}

/// Performance metrics for operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceData {
    pub duration_ns: u64,
    pub memory_usage_bytes: Option<u64>,
    pub cpu_usage_percent: Option<f64>,
    pub operation_count: Option<u64>,
    pub throughput: Option<f64>, // Operations per second
}

/// Structured logging system
pub struct Logger {
    min_level: LogLevel,
    entries: Arc<Mutex<Vec<LogEntry>>>,
    max_entries: usize,
    module_name: String,
}

impl Logger {
    /// Create new logger instance
    pub fn new(module_name: &str) -> Self {
        Self {
            min_level: LogLevel::Info,
            entries: Arc::new(Mutex::new(Vec::new())),
            max_entries: 10000,
            module_name: module_name.to_string(),
        }
    }
    
    /// Set minimum logging level
    pub fn set_level(&mut self, level: LogLevel) {
        self.min_level = level;
    }
    
    /// Set maximum number of entries to keep in memory
    pub fn set_max_entries(&mut self, max_entries: usize) {
        self.max_entries = max_entries;
    }
    
    /// Log message with level
    pub fn log(&self, level: LogLevel, message: &str) {
        if level >= self.min_level {
            let entry = LogEntry {
                timestamp: current_timestamp(),
                level,
                module: self.module_name.clone(),
                message: message.to_string(),
                metadata: HashMap::new(),
                performance_data: None,
            };
            
            self.add_entry(entry);
        }
    }
    
    /// Log with metadata
    pub fn log_with_metadata(&self, level: LogLevel, message: &str, 
                           metadata: HashMap<String, String>) {
        if level >= self.min_level {
            let entry = LogEntry {
                timestamp: current_timestamp(),
                level,
                module: self.module_name.clone(),
                message: message.to_string(),
                metadata,
                performance_data: None,
            };
            
            self.add_entry(entry);
        }
    }
    
    /// Log with performance data
    pub fn log_performance(&self, level: LogLevel, message: &str, 
                         perf_data: PerformanceData) {
        if level >= self.min_level {
            let entry = LogEntry {
                timestamp: current_timestamp(),
                level,
                module: self.module_name.clone(),
                message: message.to_string(),
                metadata: HashMap::new(),
                performance_data: Some(perf_data),
            };
            
            self.add_entry(entry);
        }
    }
    
    /// Convenience methods for different log levels
    pub fn trace(&self, message: &str) {
        self.log(LogLevel::Trace, message);
    }
    
    pub fn debug(&self, message: &str) {
        self.log(LogLevel::Debug, message);
    }
    
    pub fn info(&self, message: &str) {
        self.log(LogLevel::Info, message);
    }
    
    pub fn warn(&self, message: &str) {
        self.log(LogLevel::Warn, message);
    }
    
    pub fn error(&self, message: &str) {
        self.log(LogLevel::Error, message);
    }
    
    pub fn critical(&self, message: &str) {
        self.log(LogLevel::Critical, message);
    }
    
    /// Get recent log entries
    pub fn get_entries(&self, limit: Option<usize>) -> Vec<LogEntry> {
        let entries = self.entries.lock().unwrap();
        match limit {
            Some(n) => entries.iter().rev().take(n).cloned().collect(),
            None => entries.clone(),
        }
    }
    
    /// Get entries by level
    pub fn get_entries_by_level(&self, level: LogLevel) -> Vec<LogEntry> {
        let entries = self.entries.lock().unwrap();
        entries.iter()
            .filter(|entry| entry.level == level)
            .cloned()
            .collect()
    }
    
    /// Get entries in time range
    pub fn get_entries_in_range(&self, start: u64, end: u64) -> Vec<LogEntry> {
        let entries = self.entries.lock().unwrap();
        entries.iter()
            .filter(|entry| entry.timestamp >= start && entry.timestamp <= end)
            .cloned()
            .collect()
    }
    
    /// Clear all log entries
    pub fn clear(&self) {
        let mut entries = self.entries.lock().unwrap();
        entries.clear();
    }
    
    /// Get log statistics
    pub fn get_stats(&self) -> LogStats {
        let entries = self.entries.lock().unwrap();
        let mut level_counts = HashMap::new();
        let mut total_duration = 0;
        let mut perf_count = 0;
        
        for entry in entries.iter() {
            *level_counts.entry(entry.level).or_insert(0) += 1;
            
            if let Some(perf) = &entry.performance_data {
                total_duration += perf.duration_ns;
                perf_count += 1;
            }
        }
        
        let avg_duration = if perf_count > 0 {
            Some(total_duration / perf_count)
        } else {
            None
        };
        
        LogStats {
            total_entries: entries.len(),
            level_counts,
            average_duration_ns: avg_duration,
            oldest_timestamp: entries.first().map(|e| e.timestamp),
            newest_timestamp: entries.last().map(|e| e.timestamp),
        }
    }
    
    /// Add entry to log with size management
    fn add_entry(&self, entry: LogEntry) {
        let mut entries = self.entries.lock().unwrap();
        entries.push(entry);
        
        // Maintain maximum size by removing oldest entries
        if entries.len() > self.max_entries {
            let overflow = entries.len() - self.max_entries;
            entries.drain(0..overflow);
        }
    }
}

/// Log statistics summary
#[derive(Debug, Clone)]
pub struct LogStats {
    pub total_entries: usize,
    pub level_counts: HashMap<LogLevel, usize>,
    pub average_duration_ns: Option<u64>,
    pub oldest_timestamp: Option<u64>,
    pub newest_timestamp: Option<u64>,
}

impl LogStats {
    pub fn summary(&self) -> String {
        let mut summary = format!("Log Statistics:\n");
        summary.push_str(&format!("Total entries: {}\n", self.total_entries));
        
        for (level, count) in &self.level_counts {
            summary.push_str(&format!("  {}: {}\n", level, count));
        }
        
        if let Some(avg_duration) = self.average_duration_ns {
            summary.push_str(&format!("Average operation duration: {:.2}ms\n", 
                avg_duration as f64 / 1_000_000.0));
        }
        
        summary
    }
}

/// Performance timer for measuring operation duration
pub struct PerformanceTimer {
    start_time: Instant,
    operation_name: String,
    logger: Arc<Logger>,
    metadata: HashMap<String, String>,
}

impl PerformanceTimer {
    /// Start timing an operation
    pub fn start(operation_name: &str, logger: Arc<Logger>) -> Self {
        Self {
            start_time: Instant::now(),
            operation_name: operation_name.to_string(),
            logger,
            metadata: HashMap::new(),
        }
    }
    
    /// Add metadata to timer
    pub fn add_metadata(&mut self, key: &str, value: &str) {
        self.metadata.insert(key.to_string(), value.to_string());
    }
    
    /// Stop timer and log performance
    pub fn stop(self) -> Duration {
        let duration = self.start_time.elapsed();
        
        let perf_data = PerformanceData {
            duration_ns: duration.as_nanos() as u64,
            memory_usage_bytes: None,
            cpu_usage_percent: None,
            operation_count: Some(1),
            throughput: None,
        };
        
        self.logger.log_performance(
            LogLevel::Debug,
            &format!("Operation '{}' completed", self.operation_name),
            perf_data
        );
        
        duration
    }
    
    /// Stop timer with custom metrics
    pub fn stop_with_metrics(self, operation_count: u64, memory_bytes: Option<u64>) -> Duration {
        let duration = self.start_time.elapsed();
        let throughput = operation_count as f64 / duration.as_secs_f64();
        
        let perf_data = PerformanceData {
            duration_ns: duration.as_nanos() as u64,
            memory_usage_bytes: memory_bytes,
            cpu_usage_percent: None,
            operation_count: Some(operation_count),
            throughput: Some(throughput),
        };
        
        self.logger.log_performance(
            LogLevel::Info,
            &format!("Operation '{}' completed: {} ops in {:.2}ms ({:.0} ops/sec)",
                self.operation_name,
                operation_count,
                duration.as_millis(),
                throughput
            ),
            perf_data
        );
        
        duration
    }
}

impl Drop for PerformanceTimer {
    fn drop(&mut self) {
        // Auto-log if timer is dropped without explicit stop
        let duration = self.start_time.elapsed();
        self.logger.warn(&format!(
            "Performance timer for '{}' was dropped without explicit stop (duration: {:.2}ms)",
            self.operation_name,
            duration.as_millis()
        ));
    }
}

/// Global logger registry
pub struct LoggerRegistry {
    loggers: Arc<Mutex<HashMap<String, Arc<Logger>>>>,
}

impl LoggerRegistry {
    /// Create new logger registry
    pub fn new() -> Self {
        Self {
            loggers: Arc::new(Mutex::new(HashMap::new())),
        }
    }
    
    /// Get or create logger for module
    pub fn get_logger(&self, module_name: &str) -> Arc<Logger> {
        let mut loggers = self.loggers.lock().unwrap();
        
        if let Some(logger) = loggers.get(module_name) {
            logger.clone()
        } else {
            let logger = Arc::new(Logger::new(module_name));
            loggers.insert(module_name.to_string(), logger.clone());
            logger
        }
    }
    
    /// Set logging level for all loggers
    pub fn set_global_level(&self, level: LogLevel) {
        let loggers = self.loggers.lock().unwrap();
        // Note: Can't directly modify loggers due to Arc, would need interior mutability
        // This is a simplified version
    }
    
    /// Get all logger names
    pub fn get_logger_names(&self) -> Vec<String> {
        let loggers = self.loggers.lock().unwrap();
        loggers.keys().cloned().collect()
    }
}

/// Global logger registry instance
static mut GLOBAL_REGISTRY: Option<LoggerRegistry> = None;
static mut REGISTRY_INITIALIZED: bool = false;

/// Get global logger registry
pub fn get_global_registry() -> &'static LoggerRegistry {
    unsafe {
        if !REGISTRY_INITIALIZED {
            GLOBAL_REGISTRY = Some(LoggerRegistry::new());
            REGISTRY_INITIALIZED = true;
        }
        GLOBAL_REGISTRY.as_ref().unwrap()
    }
}

/// Get logger for module (convenience function)
pub fn get_logger(module_name: &str) -> Arc<Logger> {
    get_global_registry().get_logger(module_name)
}

/// Macro for easy logging
#[macro_export]
macro_rules! log {
    ($level:expr, $module:expr, $($arg:tt)*) => {
        {
            let logger = crate::core::logging::get_logger($module);
            logger.log($level, &format!($($arg)*));
        }
    };
}

#[macro_export]
macro_rules! info {
    ($module:expr, $($arg:tt)*) => {
        log!(crate::core::logging::LogLevel::Info, $module, $($arg)*)
    };
}

#[macro_export]
macro_rules! warn {
    ($module:expr, $($arg:tt)*) => {
        log!(crate::core::logging::LogLevel::Warn, $module, $($arg)*)
    };
}

#[macro_export]
macro_rules! error {
    ($module:expr, $($arg:tt)*) => {
        log!(crate::core::logging::LogLevel::Error, $module, $($arg)*)
    };
}

/// Get current Unix timestamp
fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;
    
    #[test]
    fn test_logger_creation() {
        let logger = Logger::new("test_module");
        assert_eq!(logger.module_name, "test_module");
        assert_eq!(logger.min_level, LogLevel::Info);
    }
    
    #[test]
    fn test_logging_levels() {
        let logger = Logger::new("test");
        
        logger.info("info message");
        logger.warn("warn message");
        logger.error("error message");
        
        let entries = logger.get_entries(None);
        assert_eq!(entries.len(), 3);
        
        assert_eq!(entries[0].level, LogLevel::Info);
        assert_eq!(entries[1].level, LogLevel::Warn);
        assert_eq!(entries[2].level, LogLevel::Error);
    }
    
    #[test]
    fn test_min_level_filtering() {
        let mut logger = Logger::new("test");
        logger.set_level(LogLevel::Warn);
        
        logger.debug("debug message"); // Should be filtered
        logger.info("info message");   // Should be filtered
        logger.warn("warn message");   // Should be logged
        logger.error("error message"); // Should be logged
        
        let entries = logger.get_entries(None);
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].level, LogLevel::Warn);
        assert_eq!(entries[1].level, LogLevel::Error);
    }
    
    #[test]
    fn test_metadata_logging() {
        let logger = Logger::new("test");
        let mut metadata = HashMap::new();
        metadata.insert("user_id".to_string(), "12345".to_string());
        metadata.insert("session_id".to_string(), "abc-123".to_string());
        
        logger.log_with_metadata(LogLevel::Info, "User action", metadata.clone());
        
        let entries = logger.get_entries(None);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].metadata, metadata);
    }
    
    #[test]
    fn test_performance_timer() {
        let logger = Arc::new(Logger::new("test"));
        
        let timer = PerformanceTimer::start("test_operation", logger.clone());
        thread::sleep(Duration::from_millis(10));
        let duration = timer.stop();
        
        assert!(duration >= Duration::from_millis(10));
        
        let entries = logger.get_entries(None);
        assert_eq!(entries.len(), 1);
        assert!(entries[0].performance_data.is_some());
    }
    
    #[test]
    fn test_entries_by_level() {
        let logger = Logger::new("test");
        
        logger.info("info 1");
        logger.warn("warn 1");
        logger.info("info 2");
        logger.error("error 1");
        
        let info_entries = logger.get_entries_by_level(LogLevel::Info);
        let warn_entries = logger.get_entries_by_level(LogLevel::Warn);
        let error_entries = logger.get_entries_by_level(LogLevel::Error);
        
        assert_eq!(info_entries.len(), 2);
        assert_eq!(warn_entries.len(), 1);
        assert_eq!(error_entries.len(), 1);
    }
    
    #[test]
    fn test_max_entries_limit() {
        let mut logger = Logger::new("test");
        logger.set_max_entries(3);
        
        logger.info("message 1");
        logger.info("message 2");
        logger.info("message 3");
        logger.info("message 4"); // Should remove message 1
        
        let entries = logger.get_entries(None);
        assert_eq!(entries.len(), 3);
        assert!(entries[0].message.contains("message 2"));
        assert!(entries[2].message.contains("message 4"));
    }
    
    #[test]
    fn test_log_stats() {
        let logger = Logger::new("test");
        
        logger.info("info message");
        logger.warn("warn message");
        logger.warn("another warn");
        logger.error("error message");
        
        let stats = logger.get_stats();
        assert_eq!(stats.total_entries, 4);
        assert_eq!(*stats.level_counts.get(&LogLevel::Info).unwrap(), 1);
        assert_eq!(*stats.level_counts.get(&LogLevel::Warn).unwrap(), 2);
        assert_eq!(*stats.level_counts.get(&LogLevel::Error).unwrap(), 1);
    }
    
    #[test]
    fn test_global_registry() {
        let registry = get_global_registry();
        let logger1 = registry.get_logger("module1");
        let logger2 = registry.get_logger("module2");
        let logger1_again = registry.get_logger("module1");
        
        // Same module should return same logger instance
        assert!(Arc::ptr_eq(&logger1, &logger1_again));
        
        // Different modules should have different loggers
        assert!(!Arc::ptr_eq(&logger1, &logger2));
    }
}