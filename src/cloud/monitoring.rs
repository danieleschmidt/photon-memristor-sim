//! Cloud-scale monitoring and observability for photonic neural networks

use crate::core::{Result, PhotonicError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, SystemTime};

/// Cloud monitoring system
#[derive(Debug, Clone)]
pub struct CloudMonitor {
    /// Metrics collection interval
    pub metrics_interval: Duration,
    /// Active deployments being monitored
    pub deployments: HashMap<String, DeploymentMonitor>,
}

/// Deployment-specific monitor
#[derive(Debug, Clone)]
pub struct DeploymentMonitor {
    /// Deployment identifier
    pub deployment_id: String,
    /// Region monitors
    pub region_monitors: HashMap<String, RegionMonitor>,
    /// Global metrics aggregation
    pub global_metrics: GlobalMetrics,
}

/// Region-specific monitor
#[derive(Debug, Clone)]
pub struct RegionMonitor {
    /// Region identifier
    pub region_id: String,
    /// Instance monitors
    pub instances: HashMap<String, InstanceMonitor>,
    /// Regional aggregate metrics
    pub regional_metrics: RegionalMetrics,
}

/// Instance-specific monitor
#[derive(Debug, Clone)]
pub struct InstanceMonitor {
    /// Instance identifier
    pub instance_id: String,
    /// Current instance metrics
    pub metrics: InstanceMetrics,
    /// Last health check timestamp
    pub last_health_check: SystemTime,
}

/// Global metrics across all regions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalMetrics {
    /// Total requests per second across all regions
    pub total_rps: f64,
    /// Average response time across all regions
    pub avg_response_time_ms: f64,
    /// Global error rate percentage
    pub global_error_rate: f64,
    /// Total optical power efficiency
    pub total_optical_efficiency: f64,
    /// Total photonic throughput TOPS
    pub total_photonic_tops: f64,
    /// Total operational cost per hour
    pub total_cost_per_hour: f64,
}

/// Regional aggregate metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegionalMetrics {
    /// Regional requests per second
    pub regional_rps: f64,
    /// Regional average response time
    pub avg_response_time_ms: f64,
    /// Regional error rate
    pub error_rate: f64,
    /// Regional optical efficiency
    pub optical_efficiency: f64,
    /// Regional photonic throughput
    pub photonic_tops: f64,
    /// Regional cost per hour
    pub cost_per_hour: f64,
    /// Network latency to other regions
    pub inter_region_latency_ms: HashMap<String, f64>,
}

/// Instance-level metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstanceMetrics {
    /// CPU utilization percentage
    pub cpu_utilization: f64,
    /// Memory utilization percentage
    pub memory_utilization: f64,
    /// Network I/O bytes per second
    pub network_io_bps: f64,
    /// Disk I/O operations per second
    pub disk_iops: f64,
    /// Photonic-specific metrics
    pub photonic_metrics: PhotonicInstanceMetrics,
    /// Health status
    pub health_status: HealthStatus,
}

/// Photonic-specific instance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhotonicInstanceMetrics {
    /// Optical power consumption (watts)
    pub optical_power_watts: f64,
    /// Wavelength channel utilization
    pub wavelength_utilization: f64,
    /// Matrix multiplication throughput (TOPS)
    pub matrix_throughput_tops: f64,
    /// Thermal drift (degrees Celsius)
    pub thermal_drift_celsius: f64,
    /// Optical loss (dB)
    pub optical_loss_db: f64,
}

/// Health status enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HealthStatus {
    Healthy,
    Warning { message: String },
    Critical { message: String },
    Unknown,
}

/// Alert configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertConfig {
    /// Alert name
    pub name: String,
    /// Metric to monitor
    pub metric: String,
    /// Threshold value
    pub threshold: f64,
    /// Comparison operator
    pub operator: String, // "greater_than", "less_than", "equals"
    /// Severity level
    pub severity: String, // "info", "warning", "critical"
}

impl CloudMonitor {
    /// Create new cloud monitor
    pub fn new(metrics_interval: Duration) -> Self {
        Self {
            metrics_interval,
            deployments: HashMap::new(),
        }
    }

    /// Add deployment to monitoring
    pub fn add_deployment(&mut self, deployment_id: String) -> Result<()> {
        let deployment_monitor = DeploymentMonitor {
            deployment_id: deployment_id.clone(),
            region_monitors: HashMap::new(),
            global_metrics: GlobalMetrics {
                total_rps: 0.0,
                avg_response_time_ms: 0.0,
                global_error_rate: 0.0,
                total_optical_efficiency: 0.0,
                total_photonic_tops: 0.0,
                total_cost_per_hour: 0.0,
            },
        };

        self.deployments.insert(deployment_id, deployment_monitor);
        Ok(())
    }

    /// Add region to deployment monitoring
    pub fn add_region(&mut self, deployment_id: &str, region_id: String) -> Result<()> {
        let deployment = self.deployments.get_mut(deployment_id)
            .ok_or_else(|| PhotonicError::InvalidParameter {
                param: "deployment_id".to_string(),
                value: deployment_id.to_string(),
                constraint: "must exist".to_string(),
            })?;

        let region_monitor = RegionMonitor {
            region_id: region_id.clone(),
            instances: HashMap::new(),
            regional_metrics: RegionalMetrics {
                regional_rps: 0.0,
                avg_response_time_ms: 0.0,
                error_rate: 0.0,
                optical_efficiency: 0.0,
                photonic_tops: 0.0,
                cost_per_hour: 0.0,
                inter_region_latency_ms: HashMap::new(),
            },
        };

        deployment.region_monitors.insert(region_id, region_monitor);
        Ok(())
    }

    /// Collect metrics from all monitored deployments
    pub fn collect_metrics(&mut self) -> Result<HashMap<String, GlobalMetrics>> {
        let mut all_metrics = HashMap::new();

        for (deployment_id, deployment) in &mut self.deployments {
            // Aggregate regional metrics to global
            let mut global_metrics = GlobalMetrics {
                total_rps: 0.0,
                avg_response_time_ms: 0.0,
                global_error_rate: 0.0,
                total_optical_efficiency: 0.0,
                total_photonic_tops: 0.0,
                total_cost_per_hour: 0.0,
            };

            let mut region_count = 0;
            for region_monitor in deployment.region_monitors.values() {
                global_metrics.total_rps += region_monitor.regional_metrics.regional_rps;
                global_metrics.avg_response_time_ms += region_monitor.regional_metrics.avg_response_time_ms;
                global_metrics.global_error_rate += region_monitor.regional_metrics.error_rate;
                global_metrics.total_optical_efficiency += region_monitor.regional_metrics.optical_efficiency;
                global_metrics.total_photonic_tops += region_monitor.regional_metrics.photonic_tops;
                global_metrics.total_cost_per_hour += region_monitor.regional_metrics.cost_per_hour;
                region_count += 1;
            }

            // Average metrics where appropriate
            if region_count > 0 {
                global_metrics.avg_response_time_ms /= region_count as f64;
                global_metrics.global_error_rate /= region_count as f64;
            }

            deployment.global_metrics = global_metrics.clone();
            all_metrics.insert(deployment_id.clone(), global_metrics);
        }

        Ok(all_metrics)
    }

    /// Check alerts for all deployments
    pub fn check_alerts(&self, alert_configs: &[AlertConfig]) -> Result<Vec<Alert>> {
        let mut alerts = Vec::new();

        for (deployment_id, deployment) in &self.deployments {
            for alert_config in alert_configs {
                if let Some(alert) = self.evaluate_alert(deployment_id, deployment, alert_config)? {
                    alerts.push(alert);
                }
            }
        }

        Ok(alerts)
    }

    /// Evaluate single alert condition
    fn evaluate_alert(&self, deployment_id: &str, deployment: &DeploymentMonitor, config: &AlertConfig) -> Result<Option<Alert>> {
        let current_value = match config.metric.as_str() {
            "total_rps" => deployment.global_metrics.total_rps,
            "avg_response_time_ms" => deployment.global_metrics.avg_response_time_ms,
            "global_error_rate" => deployment.global_metrics.global_error_rate,
            "total_optical_efficiency" => deployment.global_metrics.total_optical_efficiency,
            "total_photonic_tops" => deployment.global_metrics.total_photonic_tops,
            "total_cost_per_hour" => deployment.global_metrics.total_cost_per_hour,
            _ => return Ok(None),
        };

        let threshold_exceeded = match config.operator.as_str() {
            "greater_than" => current_value > config.threshold,
            "less_than" => current_value < config.threshold,
            "equals" => (current_value - config.threshold).abs() < f64::EPSILON,
            _ => false,
        };

        if threshold_exceeded {
            Ok(Some(Alert {
                name: config.name.clone(),
                deployment_id: deployment_id.to_string(),
                metric: config.metric.clone(),
                current_value,
                threshold: config.threshold,
                severity: config.severity.clone(),
                timestamp: SystemTime::now(),
                message: format!("Alert {} triggered: {} {} {} (current: {})", 
                               config.name, config.metric, config.operator, config.threshold, current_value),
            }))
        } else {
            Ok(None)
        }
    }

    /// Get deployment metrics
    pub fn get_deployment_metrics(&self, deployment_id: &str) -> Option<&GlobalMetrics> {
        self.deployments.get(deployment_id).map(|d| &d.global_metrics)
    }
}

/// Alert notification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alert {
    /// Alert name
    pub name: String,
    /// Deployment that triggered the alert
    pub deployment_id: String,
    /// Metric that triggered the alert
    pub metric: String,
    /// Current metric value
    pub current_value: f64,
    /// Alert threshold
    pub threshold: f64,
    /// Severity level
    pub severity: String,
    /// Alert timestamp
    pub timestamp: SystemTime,
    /// Alert message
    pub message: String,
}

impl Default for CloudMonitor {
    fn default() -> Self {
        Self::new(Duration::from_secs(10))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cloud_monitor_creation() {
        let monitor = CloudMonitor::default();
        assert_eq!(monitor.metrics_interval, Duration::from_secs(10));
        assert_eq!(monitor.deployments.len(), 0);
    }

    #[test]
    fn test_add_deployment() {
        let mut monitor = CloudMonitor::default();
        let result = monitor.add_deployment("test-deployment".to_string());
        assert!(result.is_ok());
        assert_eq!(monitor.deployments.len(), 1);
    }

    #[test]
    fn test_add_region() {
        let mut monitor = CloudMonitor::default();
        monitor.add_deployment("test-deployment".to_string()).unwrap();
        let result = monitor.add_region("test-deployment", "us-west-2".to_string());
        assert!(result.is_ok());

        let deployment = monitor.deployments.get("test-deployment").unwrap();
        assert_eq!(deployment.region_monitors.len(), 1);
    }

    #[test]
    fn test_metrics_collection() {
        let mut monitor = CloudMonitor::default();
        monitor.add_deployment("test-deployment".to_string()).unwrap();
        monitor.add_region("test-deployment", "us-west-2".to_string()).unwrap();

        let result = monitor.collect_metrics();
        assert!(result.is_ok());
        
        let metrics = result.unwrap();
        assert_eq!(metrics.len(), 1);
        assert!(metrics.contains_key("test-deployment"));
    }
}