//! Enhanced error handling with monitoring integration

use crate::core::{PhotonicError, Result, Logger, Monitor, LogLevel};
use crate::core::validation::ValidationReport;
use std::sync::Arc;
use std::collections::HashMap;
// use std::backtrace::Backtrace; // Requires nightly Rust
use std::fmt;

/// Enhanced error with context and monitoring
#[derive(Debug)]
pub struct EnhancedError {
    pub inner: PhotonicError,
    pub context: Vec<String>,
    pub metadata: HashMap<String, String>,
    pub backtrace: Option<String>,
    pub validation_report: Option<ValidationReport>,
}

impl EnhancedError {
    /// Create new enhanced error
    pub fn new(error: PhotonicError) -> Self {
        Self {
            inner: error,
            context: Vec::new(),
            metadata: HashMap::new(),
            backtrace: Some("Backtrace disabled in stable Rust".to_string()),
            validation_report: None,
        }
    }
    
    /// Add context to error
    pub fn with_context(mut self, context: &str) -> Self {
        self.context.push(context.to_string());
        self
    }
    
    /// Add metadata to error
    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }
    
    /// Add validation report
    pub fn with_validation_report(mut self, report: ValidationReport) -> Self {
        self.validation_report = Some(report);
        self
    }
    
    /// Get formatted error message
    pub fn formatted_message(&self) -> String {
        let mut message = format!("Error: {}\n", self.inner);
        
        if !self.context.is_empty() {
            message.push_str("Context:\n");
            for (i, ctx) in self.context.iter().enumerate() {
                message.push_str(&format!("  {}. {}\n", i + 1, ctx));
            }
        }
        
        if !self.metadata.is_empty() {
            message.push_str("Metadata:\n");
            for (key, value) in &self.metadata {
                message.push_str(&format!("  {}: {}\n", key, value));
            }
        }
        
        if let Some(report) = &self.validation_report {
            message.push_str(&format!("Validation: {}\n", report.summary()));
        }
        
        if let Some(backtrace) = &self.backtrace {
            message.push_str(&format!("Backtrace: {}\n", backtrace));
        }
        
        message
    }
}

impl fmt::Display for EnhancedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.formatted_message())
    }
}

impl std::error::Error for EnhancedError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.inner)
    }
}

impl From<PhotonicError> for EnhancedError {
    fn from(error: PhotonicError) -> Self {
        Self::new(error)
    }
}

/// Error handler with monitoring integration
pub struct ErrorHandler {
    logger: Arc<Logger>,
    monitor: Option<Arc<Monitor>>,
    error_count: std::sync::atomic::AtomicU64,
}

impl ErrorHandler {
    /// Create new error handler
    pub fn new(logger: Arc<Logger>) -> Self {
        Self {
            logger,
            monitor: None,
            error_count: std::sync::atomic::AtomicU64::new(0),
        }
    }
    
    /// Create error handler with monitoring
    pub fn with_monitor(logger: Arc<Logger>, monitor: Arc<Monitor>) -> Self {
        Self {
            logger,
            monitor: Some(monitor),
            error_count: std::sync::atomic::AtomicU64::new(0),
        }
    }
    
    /// Handle error with logging and monitoring
    pub fn handle_error(&self, error: &EnhancedError) {
        // Increment error count
        let count = self.error_count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        
        // Log error
        self.logger.log_with_metadata(
            LogLevel::Error,
            &error.formatted_message(),
            error.metadata.clone(),
        );
        
        // Record metrics if monitoring is available
        if let Some(monitor) = &self.monitor {
            let mut tags = HashMap::new();
            tags.insert("error_type".to_string(), format!("{:?}", error.inner));
            
            if !error.context.is_empty() {
                tags.insert("context_count".to_string(), error.context.len().to_string());
            }
            
            monitor.record_metric("error_count", count as f64, tags.clone());
            monitor.record_metric("error_occurred", 1.0, tags);
        }
    }
    
    /// Handle recoverable error (warning level)
    pub fn handle_recoverable(&self, error: &EnhancedError, recovery_action: &str) {
        self.logger.log_with_metadata(
            LogLevel::Warn,
            &format!("Recoverable error handled: {}\nRecovery: {}", 
                error.inner, recovery_action),
            error.metadata.clone(),
        );
        
        if let Some(monitor) = &self.monitor {
            let mut tags = HashMap::new();
            tags.insert("recovery_action".to_string(), recovery_action.to_string());
            monitor.record_metric("recoverable_error", 1.0, tags);
        }
    }
    
    /// Get error statistics
    pub fn get_error_count(&self) -> u64 {
        self.error_count.load(std::sync::atomic::Ordering::SeqCst)
    }
}

/// Error recovery strategies
pub enum RecoveryStrategy {
    Retry { max_attempts: u32, delay_ms: u64 },
    FallbackValue { value: f64 },
    SkipOperation,
    ResetToDefaults,
    Custom { action: Box<dyn Fn() -> Result<()> + Send + Sync> },
}

/// Resilient operation executor
pub struct ResilientExecutor {
    error_handler: Arc<ErrorHandler>,
    default_strategy: RecoveryStrategy,
}

impl ResilientExecutor {
    /// Create new resilient executor
    pub fn new(error_handler: Arc<ErrorHandler>) -> Self {
        Self {
            error_handler,
            default_strategy: RecoveryStrategy::Retry { 
                max_attempts: 3, 
                delay_ms: 100 
            },
        }
    }
    
    /// Execute operation with error recovery
    pub fn execute<T, F>(&self, operation: F, strategy: Option<RecoveryStrategy>) -> Result<T>
    where
        F: Fn() -> Result<T>,
        T: Clone + Default,
    {
        let strategy = strategy.unwrap_or_else(|| {
            match &self.default_strategy {
                RecoveryStrategy::Retry { max_attempts, delay_ms } => {
                    RecoveryStrategy::Retry { 
                        max_attempts: *max_attempts, 
                        delay_ms: *delay_ms 
                    }
                }
                _ => RecoveryStrategy::SkipOperation,
            }
        });
        
        match strategy {
            RecoveryStrategy::Retry { max_attempts, delay_ms } => {
                for attempt in 1..=max_attempts {
                    match operation() {
                        Ok(result) => return Ok(result),
                        Err(error) => {
                            let enhanced = EnhancedError::from(error)
                                .with_context(&format!("Attempt {}/{}", attempt, max_attempts));
                            
                            if attempt < max_attempts {
                                self.error_handler.handle_recoverable(
                                    &enhanced,
                                    &format!("Retrying after {}ms delay", delay_ms)
                                );
                                std::thread::sleep(std::time::Duration::from_millis(delay_ms));
                            } else {
                                self.error_handler.handle_error(&enhanced);
                                return Err(enhanced.inner);
                            }
                        }
                    }
                }
                unreachable!()
            }
            RecoveryStrategy::FallbackValue { value } => {
                match operation() {
                    Ok(result) => Ok(result),
                    Err(error) => {
                        let enhanced = EnhancedError::from(error)
                            .with_context("Using fallback value");
                        
                        self.error_handler.handle_recoverable(
                            &enhanced,
                            &format!("Returning fallback value: {}", value)
                        );
                        
                        // This is a simplified approach - in practice, you'd need
                        // more sophisticated type handling for fallback values
                        Ok(T::default())
                    }
                }
            }
            RecoveryStrategy::SkipOperation => {
                match operation() {
                    Ok(result) => Ok(result),
                    Err(error) => {
                        let enhanced = EnhancedError::from(error)
                            .with_context("Skipping operation");
                        
                        self.error_handler.handle_recoverable(
                            &enhanced,
                            "Operation skipped, returning default value"
                        );
                        
                        Ok(T::default())
                    }
                }
            }
            RecoveryStrategy::ResetToDefaults => {
                match operation() {
                    Ok(result) => Ok(result),
                    Err(error) => {
                        let enhanced = EnhancedError::from(error)
                            .with_context("Resetting to defaults");
                        
                        self.error_handler.handle_recoverable(
                            &enhanced,
                            "System reset to default state"
                        );
                        
                        Ok(T::default())
                    }
                }
            }
            RecoveryStrategy::Custom { action } => {
                match operation() {
                    Ok(result) => Ok(result),
                    Err(error) => {
                        let enhanced = EnhancedError::from(error)
                            .with_context("Executing custom recovery");
                        
                        self.error_handler.handle_recoverable(
                            &enhanced,
                            "Custom recovery action executed"
                        );
                        
                        action()?;
                        Ok(T::default())
                    }
                }
            }
        }
    }
}

/// Circuit breaker for preventing cascading failures
pub struct CircuitBreaker {
    failure_count: std::sync::atomic::AtomicU32,
    success_count: std::sync::atomic::AtomicU32,
    state: std::sync::RwLock<CircuitBreakerState>,
    failure_threshold: u32,
    success_threshold: u32,
    timeout: std::time::Duration,
    last_failure_time: std::sync::RwLock<Option<std::time::Instant>>,
    error_handler: Arc<ErrorHandler>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CircuitBreakerState {
    Closed,    // Normal operation
    Open,      // Preventing calls due to failures
    HalfOpen,  // Testing if service has recovered
}

impl CircuitBreaker {
    /// Create new circuit breaker
    pub fn new(
        failure_threshold: u32,
        success_threshold: u32,
        timeout: std::time::Duration,
        error_handler: Arc<ErrorHandler>,
    ) -> Self {
        Self {
            failure_count: std::sync::atomic::AtomicU32::new(0),
            success_count: std::sync::atomic::AtomicU32::new(0),
            state: std::sync::RwLock::new(CircuitBreakerState::Closed),
            failure_threshold,
            success_threshold,
            timeout,
            last_failure_time: std::sync::RwLock::new(None),
            error_handler,
        }
    }
    
    /// Execute operation through circuit breaker
    pub fn call<T, F>(&self, operation: F) -> Result<T>
    where
        F: Fn() -> Result<T>,
    {
        // Check if circuit breaker allows the call
        if !self.can_execute() {
            let error = PhotonicError::simulation("Circuit breaker is open".to_string());
            let enhanced = EnhancedError::from(error)
                .with_context("Circuit breaker prevented operation")
                .with_metadata("state", &format!("{:?}", self.get_state()))
                .with_metadata("failure_count", &self.get_failure_count().to_string());
            
            self.error_handler.handle_error(&enhanced);
            return Err(enhanced.inner);
        }
        
        // Execute operation
        match operation() {
            Ok(result) => {
                self.on_success();
                Ok(result)
            }
            Err(error) => {
                self.on_failure();
                let enhanced = EnhancedError::from(error)
                    .with_context("Operation failed through circuit breaker");
                self.error_handler.handle_error(&enhanced);
                Err(enhanced.inner)
            }
        }
    }
    
    /// Check if circuit breaker allows execution
    fn can_execute(&self) -> bool {
        match self.get_state() {
            CircuitBreakerState::Closed => true,
            CircuitBreakerState::Open => {
                // Check if timeout has passed
                if let Some(last_failure) = *self.last_failure_time.read().unwrap() {
                    if last_failure.elapsed() >= self.timeout {
                        self.transition_to_half_open();
                        true
                    } else {
                        false
                    }
                } else {
                    true
                }
            }
            CircuitBreakerState::HalfOpen => true,
        }
    }
    
    /// Handle successful operation
    fn on_success(&self) {
        let success_count = self.success_count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        
        match self.get_state() {
            CircuitBreakerState::HalfOpen => {
                if success_count >= self.success_threshold {
                    self.transition_to_closed();
                }
            }
            _ => {}
        }
    }
    
    /// Handle failed operation
    fn on_failure(&self) {
        let failure_count = self.failure_count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        *self.last_failure_time.write().unwrap() = Some(std::time::Instant::now());
        
        if failure_count >= self.failure_threshold {
            self.transition_to_open();
        }
    }
    
    /// Transition to closed state
    fn transition_to_closed(&self) {
        *self.state.write().unwrap() = CircuitBreakerState::Closed;
        self.failure_count.store(0, std::sync::atomic::Ordering::SeqCst);
        self.success_count.store(0, std::sync::atomic::Ordering::SeqCst);
    }
    
    /// Transition to open state
    fn transition_to_open(&self) {
        *self.state.write().unwrap() = CircuitBreakerState::Open;
    }
    
    /// Transition to half-open state
    fn transition_to_half_open(&self) {
        *self.state.write().unwrap() = CircuitBreakerState::HalfOpen;
        self.success_count.store(0, std::sync::atomic::Ordering::SeqCst);
    }
    
    /// Get current state
    pub fn get_state(&self) -> CircuitBreakerState {
        *self.state.read().unwrap()
    }
    
    /// Get failure count
    pub fn get_failure_count(&self) -> u32 {
        self.failure_count.load(std::sync::atomic::Ordering::SeqCst)
    }
    
    /// Get success count
    pub fn get_success_count(&self) -> u32 {
        self.success_count.load(std::sync::atomic::Ordering::SeqCst)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Logger;
    
    #[test]
    fn test_enhanced_error_creation() {
        let base_error = PhotonicError::invalid_parameter("test", 1.0, "valid range");
        let enhanced = EnhancedError::new(base_error)
            .with_context("Test context")
            .with_metadata("test_key", "test_value");
        
        assert_eq!(enhanced.context.len(), 1);
        assert_eq!(enhanced.metadata.len(), 1);
        assert!(enhanced.backtrace.is_some());
    }
    
    #[test]
    fn test_error_handler() {
        let logger = Arc::new(Logger::new("test"));
        let error_handler = ErrorHandler::new(logger);
        
        let base_error = PhotonicError::simulation("Test error".to_string());
        let enhanced = EnhancedError::from(base_error);
        
        error_handler.handle_error(&enhanced);
        assert_eq!(error_handler.get_error_count(), 1);
    }
    
    #[test]
    fn test_resilient_executor() {
        let logger = Arc::new(Logger::new("test"));
        let error_handler = Arc::new(ErrorHandler::new(logger));
        let executor = ResilientExecutor::new(error_handler);
        
        // Test successful operation
        let result = executor.execute(|| Ok(42), None);
        assert_eq!(result.unwrap(), 42);
        
        // Test operation with fallback
        let result = executor.execute(
            || Err(PhotonicError::simulation("Test error".to_string())),
            Some(RecoveryStrategy::FallbackValue { value: 0.0 })
        );
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_circuit_breaker() {
        let logger = Arc::new(Logger::new("test"));
        let error_handler = Arc::new(ErrorHandler::new(logger));
        let circuit_breaker = CircuitBreaker::new(
            3, // failure threshold
            2, // success threshold
            std::time::Duration::from_millis(100), // timeout
            error_handler,
        );
        
        assert_eq!(circuit_breaker.get_state(), CircuitBreakerState::Closed);
        
        // Simulate failures to open circuit breaker
        for _ in 0..3 {
            let _ = circuit_breaker.call(|| {
                Err(PhotonicError::simulation("Test failure".to_string()))
            });
        }
        
        assert_eq!(circuit_breaker.get_state(), CircuitBreakerState::Open);
    }
}