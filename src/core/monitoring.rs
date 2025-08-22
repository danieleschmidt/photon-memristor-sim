//! Real-time monitoring and metrics collection for photonic simulation

use crate::core::logging::Logger;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant, SystemTime};
use std::thread;
use std::sync::mpsc;
use serde::{Serialize, Deserialize};

/// System health status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HealthStatus {
    Healthy,
    Warning,
    Critical,
    Unknown,
}

/// Metric data point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricPoint {
    pub timestamp: u64,
    pub value: f64,
    pub tags: HashMap<String, String>,
}

/// Time-series metric storage
#[derive(Debug, Clone)]
pub struct TimeSeries {
    pub name: String,
    pub points: VecDeque<MetricPoint>,
    pub max_points: usize,
    pub retention_duration: Duration,
}

impl TimeSeries {
    /// Create new time series
    pub fn new(name: &str, max_points: usize, retention_duration: Duration) -> Self {
        Self {
            name: name.to_string(),
            points: VecDeque::new(),
            max_points,
            retention_duration,
        }
    }
    
    /// Add data point
    pub fn add_point(&mut self, value: f64, tags: HashMap<String, String>) {
        let point = MetricPoint {
            timestamp: current_timestamp_ns(),
            value,
            tags,
        };
        
        self.points.push_back(point);
        
        // Maintain size limit
        if self.points.len() > self.max_points {
            self.points.pop_front();
        }
        
        // Remove expired points
        let cutoff = current_timestamp_ns() - self.retention_duration.as_nanos() as u64;
        while let Some(front) = self.points.front() {
            if front.timestamp < cutoff {
                self.points.pop_front();
            } else {
                break;
            }
        }
    }
    
    /// Get recent points
    pub fn get_recent(&self, limit: usize) -> Vec<&MetricPoint> {
        self.points.iter().rev().take(limit).collect()
    }
    
    /// Get points in time range
    pub fn get_range(&self, start: u64, end: u64) -> Vec<&MetricPoint> {
        self.points.iter()
            .filter(|p| p.timestamp >= start && p.timestamp <= end)
            .collect()
    }
    
    /// Calculate statistics
    pub fn stats(&self) -> MetricStats {
        if self.points.is_empty() {
            return MetricStats::default();
        }
        
        let values: Vec<f64> = self.points.iter().map(|p| p.value).collect();
        let sum: f64 = values.iter().sum();
        let count = values.len() as f64;
        let mean = sum / count;
        
        let variance: f64 = values.iter()
            .map(|v| (v - mean).powi(2))
            .sum::<f64>() / count;
        
        let std_dev = variance.sqrt();
        
        let mut sorted_values = values.clone();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        MetricStats {
            count: count as u64,
            sum,
            mean,
            std_dev,
            min: sorted_values[0],
            max: sorted_values[sorted_values.len() - 1],
            median: sorted_values[sorted_values.len() / 2],
            p95: sorted_values[(sorted_values.len() as f64 * 0.95) as usize],
            p99: sorted_values[(sorted_values.len() as f64 * 0.99) as usize],
        }
    }
}

/// Metric statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricStats {
    pub count: u64,
    pub sum: f64,
    pub mean: f64,
    pub std_dev: f64,
    pub min: f64,
    pub max: f64,
    pub median: f64,
    pub p95: f64,
    pub p99: f64,
}

impl Default for MetricStats {
    fn default() -> Self {
        Self {
            count: 0,
            sum: 0.0,
            mean: 0.0,
            std_dev: 0.0,
            min: 0.0,
            max: 0.0,
            median: 0.0,
            p95: 0.0,
            p99: 0.0,
        }
    }
}

/// Alert condition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alert {
    pub id: String,
    pub metric_name: String,
    pub condition: AlertCondition,
    pub threshold: f64,
    pub duration: Duration,
    pub severity: AlertSeverity,
    pub message: String,
    pub is_active: bool,
    pub triggered_at: Option<u64>,
    pub resolved_at: Option<u64>,
}

/// Alert condition types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertCondition {
    GreaterThan,
    LessThan,
    Equal,
    NotEqual,
    PercentileAbove(f64),
    PercentileBelow(f64),
    StandardDeviations(f64),
}

/// Alert severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
}

/// Monitoring system
pub struct Monitor {
    metrics: Arc<RwLock<HashMap<String, TimeSeries>>>,
    alerts: Arc<Mutex<HashMap<String, Alert>>>,
    logger: Arc<Logger>,
    _worker_handle: Option<thread::JoinHandle<()>>,
    worker_tx: mpsc::Sender<MonitorCommand>,
    health_status: Arc<RwLock<HealthStatus>>,
}

/// Commands for monitor worker thread
enum MonitorCommand {
    Stop,
    CheckAlerts,
    RecordMetric { name: String, value: f64, tags: HashMap<String, String> },
}

impl Monitor {
    /// Create new monitoring system
    pub fn new() -> Self {
        let logger = Arc::new(Logger::new("monitor"));
        let (tx, rx) = mpsc::channel();
        
        let metrics = Arc::new(RwLock::new(HashMap::new()));
        let alerts = Arc::new(Mutex::new(HashMap::new()));
        let health_status = Arc::new(RwLock::new(HealthStatus::Healthy));
        
        // Spawn worker thread for background monitoring
        let worker_metrics = metrics.clone();
        let worker_alerts = alerts.clone();
        let worker_logger = logger.clone();
        let worker_health = health_status.clone();
        
        let worker_handle = thread::spawn(move || {
            Self::worker_loop(rx, worker_metrics, worker_alerts, worker_logger, worker_health);
        });
        
        Self {
            metrics,
            alerts,
            logger,
            _worker_handle: Some(worker_handle),
            worker_tx: tx,
            health_status,
        }
    }
    
    /// Record metric value
    pub fn record_metric(&self, name: &str, value: f64, tags: HashMap<String, String>) {
        let _ = self.worker_tx.send(MonitorCommand::RecordMetric {
            name: name.to_string(),
            value,
            tags,
        });
    }
    
    /// Record metric without tags
    pub fn record(&self, name: &str, value: f64) {
        self.record_metric(name, value, HashMap::new());
    }
    
    /// Get metric time series
    pub fn get_metric(&self, name: &str) -> Option<TimeSeries> {
        let metrics = self.metrics.read().unwrap();
        metrics.get(name).cloned()
    }
    
    /// Get all metric names
    pub fn get_metric_names(&self) -> Vec<String> {
        let metrics = self.metrics.read().unwrap();
        metrics.keys().cloned().collect()
    }
    
    /// Add alert
    pub fn add_alert(&self, alert: Alert) {
        let mut alerts = self.alerts.lock().unwrap();
        alerts.insert(alert.id.clone(), alert);
    }
    
    /// Remove alert
    pub fn remove_alert(&self, alert_id: &str) -> bool {
        let mut alerts = self.alerts.lock().unwrap();
        alerts.remove(alert_id).is_some()
    }
    
    /// Get active alerts
    pub fn get_active_alerts(&self) -> Vec<Alert> {
        let alerts = self.alerts.lock().unwrap();
        alerts.values()
            .filter(|alert| alert.is_active)
            .cloned()
            .collect()
    }
    
    /// Get all alerts
    pub fn get_all_alerts(&self) -> Vec<Alert> {
        let alerts = self.alerts.lock().unwrap();
        alerts.values().cloned().collect()
    }
    
    /// Get current health status
    pub fn get_health_status(&self) -> HealthStatus {
        *self.health_status.read().unwrap()
    }
    
    /// Force alert check
    pub fn check_alerts(&self) {
        let _ = self.worker_tx.send(MonitorCommand::CheckAlerts);
    }
    
    /// Get monitoring dashboard data
    pub fn get_dashboard(&self) -> MonitoringDashboard {
        let metrics = self.metrics.read().unwrap();
        let alerts = self.alerts.lock().unwrap();
        
        let metric_summaries: HashMap<String, MetricStats> = metrics.iter()
            .map(|(name, series)| (name.clone(), series.stats()))
            .collect();
        
        let active_alert_count = alerts.values().filter(|a| a.is_active).count();
        let health_status = self.get_health_status();
        
        MonitoringDashboard {
            health_status,
            active_alerts: active_alert_count,
            total_metrics: metrics.len(),
            metric_summaries,
            uptime_seconds: 0, // Would need startup time tracking
        }
    }
    
    /// Worker thread main loop
    fn worker_loop(
        rx: mpsc::Receiver<MonitorCommand>,
        metrics: Arc<RwLock<HashMap<String, TimeSeries>>>,
        alerts: Arc<Mutex<HashMap<String, Alert>>>,
        logger: Arc<Logger>,
        health_status: Arc<RwLock<HealthStatus>>,
    ) {
        let mut last_alert_check = Instant::now();
        let alert_check_interval = Duration::from_secs(30);
        
        loop {
            // Check for commands
            match rx.recv_timeout(Duration::from_millis(100)) {
                Ok(MonitorCommand::Stop) => break,
                Ok(MonitorCommand::CheckAlerts) => {
                    Self::check_alerts_impl(&metrics, &alerts, &logger);
                    last_alert_check = Instant::now();
                }
                Ok(MonitorCommand::RecordMetric { name, value, tags }) => {
                    Self::record_metric_impl(&metrics, &name, value, tags);
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    // Periodic tasks
                    if last_alert_check.elapsed() >= alert_check_interval {
                        Self::check_alerts_impl(&metrics, &alerts, &logger);
                        last_alert_check = Instant::now();
                    }
                }
                Err(mpsc::RecvTimeoutError::Disconnected) => break,
            }
            
            // Update health status
            Self::update_health_status(&metrics, &alerts, &health_status);
        }
        
        logger.info("Monitor worker thread stopped");
    }
    
    /// Record metric implementation
    fn record_metric_impl(
        metrics: &Arc<RwLock<HashMap<String, TimeSeries>>>,
        name: &str,
        value: f64,
        tags: HashMap<String, String>,
    ) {
        let mut metrics_map = metrics.write().unwrap();
        
        let series = metrics_map.entry(name.to_string()).or_insert_with(|| {
            TimeSeries::new(name, 10000, Duration::from_secs(24 * 3600))
        });
        
        series.add_point(value, tags);
    }
    
    /// Check alerts implementation
    fn check_alerts_impl(
        metrics: &Arc<RwLock<HashMap<String, TimeSeries>>>,
        alerts: &Arc<Mutex<HashMap<String, Alert>>>,
        logger: &Arc<Logger>,
    ) {
        let metrics_map = metrics.read().unwrap();
        let mut alerts_map = alerts.lock().unwrap();
        
        for alert in alerts_map.values_mut() {
            if let Some(series) = metrics_map.get(&alert.metric_name) {
                let should_trigger = Self::evaluate_alert_condition(alert, series);
                
                if should_trigger && !alert.is_active {
                    alert.is_active = true;
                    alert.triggered_at = Some(current_timestamp_ns());
                    alert.resolved_at = None;
                    
                    logger.warn(&format!("Alert triggered: {} - {}", alert.id, alert.message));
                } else if !should_trigger && alert.is_active {
                    alert.is_active = false;
                    alert.resolved_at = Some(current_timestamp_ns());
                    
                    logger.info(&format!("Alert resolved: {}", alert.id));
                }
            }
        }
    }
    
    /// Evaluate alert condition
    fn evaluate_alert_condition(alert: &Alert, series: &TimeSeries) -> bool {
        if series.points.is_empty() {
            return false;
        }
        
        match alert.condition {
            AlertCondition::GreaterThan => {
                series.points.back().unwrap().value > alert.threshold
            }
            AlertCondition::LessThan => {
                series.points.back().unwrap().value < alert.threshold
            }
            AlertCondition::Equal => {
                (series.points.back().unwrap().value - alert.threshold).abs() < 1e-10
            }
            AlertCondition::NotEqual => {
                (series.points.back().unwrap().value - alert.threshold).abs() >= 1e-10
            }
            AlertCondition::PercentileAbove(p) => {
                let stats = series.stats();
                match p {
                    95.0 => stats.p95 > alert.threshold,
                    99.0 => stats.p99 > alert.threshold,
                    _ => false, // Simplified
                }
            }
            AlertCondition::PercentileBelow(p) => {
                let stats = series.stats();
                match p {
                    95.0 => stats.p95 < alert.threshold,
                    99.0 => stats.p99 < alert.threshold,
                    _ => false,
                }
            }
            AlertCondition::StandardDeviations(n) => {
                let stats = series.stats();
                let latest = series.points.back().unwrap().value;
                (latest - stats.mean).abs() > n * stats.std_dev
            }
        }
    }
    
    /// Update overall health status
    fn update_health_status(
        _metrics: &Arc<RwLock<HashMap<String, TimeSeries>>>,
        alerts: &Arc<Mutex<HashMap<String, Alert>>>,
        health_status: &Arc<RwLock<HealthStatus>>,
    ) {
        let alerts_map = alerts.lock().unwrap();
        let active_alerts: Vec<&Alert> = alerts_map.values().filter(|a| a.is_active).collect();
        
        let new_status = if active_alerts.is_empty() {
            HealthStatus::Healthy
        } else {
            let has_critical = active_alerts.iter().any(|a| matches!(a.severity, AlertSeverity::Critical));
            if has_critical {
                HealthStatus::Critical
            } else {
                HealthStatus::Warning
            }
        };
        
        let mut status = health_status.write().unwrap();
        *status = new_status;
    }
}

impl Drop for Monitor {
    fn drop(&mut self) {
        let _ = self.worker_tx.send(MonitorCommand::Stop);
    }
}

/// Monitoring dashboard data
#[derive(Debug, Serialize, Deserialize)]
pub struct MonitoringDashboard {
    pub health_status: HealthStatus,
    pub active_alerts: usize,
    pub total_metrics: usize,
    pub metric_summaries: HashMap<String, MetricStats>,
    pub uptime_seconds: u64,
}

/// Performance monitoring for operations
pub struct PerformanceMonitor {
    monitor: Arc<Monitor>,
    operation_name: String,
    start_time: Instant,
    tags: HashMap<String, String>,
}

impl PerformanceMonitor {
    /// Start monitoring operation performance
    pub fn start(monitor: Arc<Monitor>, operation_name: &str) -> Self {
        Self {
            monitor,
            operation_name: operation_name.to_string(),
            start_time: Instant::now(),
            tags: HashMap::new(),
        }
    }
    
    /// Add tag to performance monitoring
    pub fn tag(&mut self, key: &str, value: &str) {
        self.tags.insert(key.to_string(), value.to_string());
    }
    
    /// Finish monitoring and record metrics
    pub fn finish(self) -> Duration {
        let duration = self.start_time.elapsed();
        
        // Record duration metric
        self.monitor.record_metric(
            &format!("{}_duration_ms", self.operation_name),
            duration.as_millis() as f64,
            self.tags.clone(),
        );
        
        // Record throughput if operation count is in tags
        if let Some(count_str) = self.tags.get("operation_count") {
            if let Ok(count) = count_str.parse::<f64>() {
                let throughput = count / duration.as_secs_f64();
                self.monitor.record_metric(
                    &format!("{}_throughput_ops_per_sec", self.operation_name),
                    throughput,
                    self.tags,
                );
            }
        }
        
        duration
    }
}

/// Get current timestamp in nanoseconds
fn current_timestamp_ns() -> u64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64
}

/// Create standard alerts for photonic simulation
pub fn create_standard_alerts() -> Vec<Alert> {
    vec![
        Alert {
            id: "high_optical_power".to_string(),
            metric_name: "optical_power_watts".to_string(),
            condition: AlertCondition::GreaterThan,
            threshold: 1.0, // 1W
            duration: Duration::from_secs(30),
            severity: AlertSeverity::Warning,
            message: "Optical power exceeds safe operating limit".to_string(),
            is_active: false,
            triggered_at: None,
            resolved_at: None,
        },
        Alert {
            id: "low_simulation_accuracy".to_string(),
            metric_name: "simulation_accuracy".to_string(),
            condition: AlertCondition::LessThan,
            threshold: 0.95, // 95%
            duration: Duration::from_secs(60),
            severity: AlertSeverity::Critical,
            message: "Simulation accuracy has dropped below acceptable threshold".to_string(),
            is_active: false,
            triggered_at: None,
            resolved_at: None,
        },
        Alert {
            id: "high_device_temperature".to_string(),
            metric_name: "device_temperature_kelvin".to_string(),
            condition: AlertCondition::GreaterThan,
            threshold: 400.0, // 400K (127°C)
            duration: Duration::from_secs(10),
            severity: AlertSeverity::Critical,
            message: "Device temperature exceeds maximum operating limit".to_string(),
            is_active: false,
            triggered_at: None,
            resolved_at: None,
        },
        Alert {
            id: "slow_convergence".to_string(),
            metric_name: "convergence_time_ms".to_string(),
            condition: AlertCondition::PercentileAbove(95.0),
            threshold: 10000.0, // 10 seconds
            duration: Duration::from_secs(5 * 60),
            severity: AlertSeverity::Warning,
            message: "Simulation convergence is consistently slow".to_string(),
            is_active: false,
            triggered_at: None,
            resolved_at: None,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;
    
    #[test]
    fn test_time_series_creation() {
        let mut series = TimeSeries::new("test_metric", 100, Duration::from_hours(1));
        assert_eq!(series.name, "test_metric");
        assert_eq!(series.points.len(), 0);
    }
    
    #[test]
    fn test_time_series_add_points() {
        let mut series = TimeSeries::new("test_metric", 100, Duration::from_hours(1));
        
        series.add_point(1.0, HashMap::new());
        series.add_point(2.0, HashMap::new());
        series.add_point(3.0, HashMap::new());
        
        assert_eq!(series.points.len(), 3);
        assert_eq!(series.points[0].value, 1.0);
        assert_eq!(series.points[2].value, 3.0);
    }
    
    #[test]
    fn test_time_series_stats() {
        let mut series = TimeSeries::new("test_metric", 100, Duration::from_hours(1));
        
        for i in 1..=10 {
            series.add_point(i as f64, HashMap::new());
        }
        
        let stats = series.stats();
        assert_eq!(stats.count, 10);
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 10.0);
        assert_eq!(stats.mean, 5.5);
    }
    
    #[test]
    fn test_monitor_creation() {
        let monitor = Monitor::new();
        assert_eq!(monitor.get_health_status(), HealthStatus::Healthy);
        assert_eq!(monitor.get_metric_names().len(), 0);
    }
    
    #[test]
    fn test_metric_recording() {
        let monitor = Monitor::new();
        
        monitor.record("test_metric", 42.0);
        
        // Give worker thread time to process
        thread::sleep(Duration::from_millis(50));
        
        let metric = monitor.get_metric("test_metric");
        assert!(metric.is_some());
        
        let series = metric.unwrap();
        assert_eq!(series.points.len(), 1);
        assert_eq!(series.points[0].value, 42.0);
    }
    
    #[test]
    fn test_alert_creation() {
        let monitor = Monitor::new();
        
        let alert = Alert {
            id: "test_alert".to_string(),
            metric_name: "test_metric".to_string(),
            condition: AlertCondition::GreaterThan,
            threshold: 10.0,
            duration: Duration::from_secs(30),
            severity: AlertSeverity::Warning,
            message: "Test alert".to_string(),
            is_active: false,
            triggered_at: None,
            resolved_at: None,
        };
        
        monitor.add_alert(alert);
        
        let alerts = monitor.get_all_alerts();
        assert_eq!(alerts.len(), 1);
        assert_eq!(alerts[0].id, "test_alert");
    }
    
    #[test]
    fn test_performance_monitor() {
        let monitor = Arc::new(Monitor::new());
        
        let mut perf_mon = PerformanceMonitor::start(monitor.clone(), "test_operation");
        perf_mon.tag("operation_count", "100");
        
        thread::sleep(Duration::from_millis(10));
        let duration = perf_mon.finish();
        
        assert!(duration >= Duration::from_millis(10));
        
        // Give worker thread time to process
        thread::sleep(Duration::from_millis(50));
        
        let duration_metric = monitor.get_metric("test_operation_duration_ms");
        assert!(duration_metric.is_some());
        
        let throughput_metric = monitor.get_metric("test_operation_throughput_ops_per_sec");
        assert!(throughput_metric.is_some());
    }
    
    #[test]
    fn test_standard_alerts() {
        let alerts = create_standard_alerts();
        assert!(!alerts.is_empty());
        
        let power_alert = alerts.iter().find(|a| a.id == "high_optical_power");
        assert!(power_alert.is_some());
        assert_eq!(power_alert.unwrap().threshold, 1.0);
    }
    
    #[test]
    fn test_dashboard_data() {
        let monitor = Monitor::new();
        monitor.record("test_metric", 42.0);
        
        thread::sleep(Duration::from_millis(50));
        
        let dashboard = monitor.get_dashboard();
        assert_eq!(dashboard.health_status, HealthStatus::Healthy);
        assert_eq!(dashboard.active_alerts, 0);
        assert_eq!(dashboard.total_metrics, 1);
    }
}