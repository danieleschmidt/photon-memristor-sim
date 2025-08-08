//! Cloud-scale deployment and distributed computing for photonic arrays
//!
//! This module provides infrastructure for deploying photonic neural networks
//! across distributed cloud environments with automatic scaling, load balancing,
//! and global optimization coordination.

pub mod deployment;
pub mod scaling;
pub mod distributed_optimizer;
pub mod monitoring;

use crate::core::{Result, PhotonicError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Cloud deployment configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloudConfig {
    /// Target cloud regions
    pub regions: Vec<CloudRegion>,
    /// Auto-scaling configuration
    pub scaling: AutoScalingConfig,
    /// Global optimization settings
    pub optimization: GlobalOptimizationConfig,
    /// Monitoring and telemetry
    pub monitoring: MonitoringConfig,
    /// Security and compliance
    pub security: SecurityConfig,
}

/// Cloud region specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloudRegion {
    /// Region identifier (e.g., "us-west-2", "eu-central-1")
    pub region_id: String,
    /// Available compute resources
    pub compute_resources: ComputeResources,
    /// Network latency characteristics
    pub network_latency_ms: f64,
    /// Data sovereignty requirements
    pub data_sovereignty: DataSovereignty,
    /// Cost per compute unit
    pub cost_per_hour: f64,
}

/// Compute resources available in a region
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputeResources {
    /// CPU cores available
    pub cpu_cores: usize,
    /// GPU devices for acceleration
    pub gpu_devices: Vec<GPUDevice>,
    /// Memory in GB
    pub memory_gb: f64,
    /// Network bandwidth in Gbps
    pub network_bandwidth_gbps: f64,
    /// Specialized photonic simulators
    pub photonic_accelerators: Vec<PhotonicAccelerator>,
}

/// GPU device specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GPUDevice {
    pub model: String,
    pub memory_gb: f64,
    pub compute_capability: f64,
    pub tensor_cores: bool,
}

/// Photonic accelerator (future hardware)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhotonicAccelerator {
    pub model: String,
    pub wavelength_channels: usize,
    pub matrix_dimensions: (usize, usize),
    pub precision_bits: usize,
    pub throughput_tops: f64,
}

/// Data sovereignty and compliance requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataSovereignty {
    /// Data must remain in region
    pub data_residency_required: bool,
    /// Compliance frameworks (GDPR, HIPAA, etc.)
    pub compliance_frameworks: Vec<String>,
    /// Encryption requirements
    pub encryption_requirements: EncryptionRequirements,
}

/// Encryption requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptionRequirements {
    pub data_at_rest: bool,
    pub data_in_transit: bool,
    pub key_management: KeyManagementConfig,
}

/// Key management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyManagementConfig {
    pub provider: String, // "aws-kms", "azure-keyvault", "gcp-kms", "self-managed"
    pub key_rotation_days: u32,
    pub multi_region_keys: bool,
}

/// Auto-scaling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoScalingConfig {
    /// Minimum number of instances
    pub min_instances: usize,
    /// Maximum number of instances
    pub max_instances: usize,
    /// CPU utilization threshold for scale-up
    pub cpu_scale_up_threshold: f64,
    /// CPU utilization threshold for scale-down
    pub cpu_scale_down_threshold: f64,
    /// Memory utilization threshold
    pub memory_threshold: f64,
    /// Custom metrics for scaling
    pub custom_metrics: Vec<CustomScalingMetric>,
    /// Cool-down periods
    pub scale_up_cooldown_seconds: u64,
    pub scale_down_cooldown_seconds: u64,
}

/// Custom scaling metric
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomScalingMetric {
    pub name: String,
    pub threshold: f64,
    pub comparison: String, // "greater_than", "less_than"
    pub action: String, // "scale_up", "scale_down"
}

/// Global optimization across regions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalOptimizationConfig {
    /// Coordination algorithm
    pub algorithm: String, // "federated_averaging", "quantum_consensus", "hierarchical"
    /// Synchronization frequency
    pub sync_frequency_seconds: u64,
    /// Parameter compression for network efficiency
    pub parameter_compression: CompressionConfig,
    /// Differential privacy settings
    pub privacy: PrivacyConfig,
}

/// Parameter compression configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionConfig {
    pub algorithm: String, // "gradient_quantization", "sparse_updates", "low_rank"
    pub compression_ratio: f64,
    pub error_compensation: bool,
}

/// Privacy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivacyConfig {
    pub differential_privacy: bool,
    pub epsilon: f64, // Privacy budget
    pub delta: f64,
    pub secure_aggregation: bool,
}

/// Monitoring and observability configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    /// Metrics collection interval
    pub metrics_interval_seconds: u64,
    /// Log aggregation settings
    pub logging: LoggingConfig,
    /// Distributed tracing
    pub tracing: TracingConfig,
    /// Alerting rules
    pub alerts: Vec<AlertRule>,
}

/// Logging configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    pub level: String, // "debug", "info", "warn", "error"
    pub structured_logging: bool,
    pub retention_days: u32,
    pub centralized_logging: bool,
}

/// Distributed tracing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TracingConfig {
    pub enabled: bool,
    pub sampling_rate: f64,
    pub jaeger_endpoint: Option<String>,
    pub service_name: String,
}

/// Alert rule definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertRule {
    pub name: String,
    pub condition: String,
    pub threshold: f64,
    pub severity: String, // "critical", "warning", "info"
    pub notification_channels: Vec<String>,
}

/// Security configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    /// Network security
    pub network_security: NetworkSecurity,
    /// Access control
    pub access_control: AccessControl,
    /// Vulnerability scanning
    pub vulnerability_scanning: VulnerabilityScanning,
}

/// Network security settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkSecurity {
    pub vpc_isolation: bool,
    pub firewall_rules: Vec<FirewallRule>,
    pub ddos_protection: bool,
    pub waf_enabled: bool,
}

/// Firewall rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FirewallRule {
    pub name: String,
    pub direction: String, // "inbound", "outbound"
    pub protocol: String, // "tcp", "udp", "icmp"
    pub port_range: Option<(u16, u16)>,
    pub source_cidrs: Vec<String>,
    pub action: String, // "allow", "deny"
}

/// Access control configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessControl {
    pub authentication_method: String, // "oauth2", "saml", "certificate"
    pub authorization_model: String, // "rbac", "abac"
    pub multi_factor_auth: bool,
    pub session_timeout_minutes: u32,
}

/// Vulnerability scanning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VulnerabilityScanning {
    pub enabled: bool,
    pub scan_frequency_hours: u32,
    pub severity_threshold: String, // "low", "medium", "high", "critical"
    pub auto_remediation: bool,
}

/// Default cloud configuration for photonic neural networks
impl Default for CloudConfig {
    fn default() -> Self {
        Self {
            regions: vec![
                CloudRegion {
                    region_id: "us-west-2".to_string(),
                    compute_resources: ComputeResources {
                        cpu_cores: 64,
                        gpu_devices: vec![GPUDevice {
                            model: "A100".to_string(),
                            memory_gb: 80.0,
                            compute_capability: 8.0,
                            tensor_cores: true,
                        }],
                        memory_gb: 256.0,
                        network_bandwidth_gbps: 100.0,
                        photonic_accelerators: vec![PhotonicAccelerator {
                            model: "PhotonicML-1".to_string(),
                            wavelength_channels: 1024,
                            matrix_dimensions: (2048, 2048),
                            precision_bits: 16,
                            throughput_tops: 1000.0,
                        }],
                    },
                    network_latency_ms: 50.0,
                    data_sovereignty: DataSovereignty {
                        data_residency_required: false,
                        compliance_frameworks: vec!["SOC2".to_string()],
                        encryption_requirements: EncryptionRequirements {
                            data_at_rest: true,
                            data_in_transit: true,
                            key_management: KeyManagementConfig {
                                provider: "aws-kms".to_string(),
                                key_rotation_days: 90,
                                multi_region_keys: true,
                            },
                        },
                    },
                    cost_per_hour: 5.0,
                },
                CloudRegion {
                    region_id: "eu-central-1".to_string(),
                    compute_resources: ComputeResources {
                        cpu_cores: 32,
                        gpu_devices: vec![GPUDevice {
                            model: "V100".to_string(),
                            memory_gb: 32.0,
                            compute_capability: 7.0,
                            tensor_cores: true,
                        }],
                        memory_gb: 128.0,
                        network_bandwidth_gbps: 50.0,
                        photonic_accelerators: vec![],
                    },
                    network_latency_ms: 75.0,
                    data_sovereignty: DataSovereignty {
                        data_residency_required: true,
                        compliance_frameworks: vec!["GDPR".to_string(), "ISO27001".to_string()],
                        encryption_requirements: EncryptionRequirements {
                            data_at_rest: true,
                            data_in_transit: true,
                            key_management: KeyManagementConfig {
                                provider: "azure-keyvault".to_string(),
                                key_rotation_days: 30,
                                multi_region_keys: false,
                            },
                        },
                    },
                    cost_per_hour: 4.5,
                },
            ],
            scaling: AutoScalingConfig {
                min_instances: 2,
                max_instances: 100,
                cpu_scale_up_threshold: 70.0,
                cpu_scale_down_threshold: 30.0,
                memory_threshold: 80.0,
                custom_metrics: vec![
                    CustomScalingMetric {
                        name: "optical_power_efficiency".to_string(),
                        threshold: 0.8,
                        comparison: "less_than".to_string(),
                        action: "scale_up".to_string(),
                    },
                    CustomScalingMetric {
                        name: "photonic_throughput_ops_per_sec".to_string(),
                        threshold: 1000000.0,
                        comparison: "greater_than".to_string(),
                        action: "scale_down".to_string(),
                    },
                ],
                scale_up_cooldown_seconds: 300,
                scale_down_cooldown_seconds: 600,
            },
            optimization: GlobalOptimizationConfig {
                algorithm: "quantum_consensus".to_string(),
                sync_frequency_seconds: 60,
                parameter_compression: CompressionConfig {
                    algorithm: "gradient_quantization".to_string(),
                    compression_ratio: 0.1,
                    error_compensation: true,
                },
                privacy: PrivacyConfig {
                    differential_privacy: true,
                    epsilon: 1.0,
                    delta: 1e-5,
                    secure_aggregation: true,
                },
            },
            monitoring: MonitoringConfig {
                metrics_interval_seconds: 10,
                logging: LoggingConfig {
                    level: "info".to_string(),
                    structured_logging: true,
                    retention_days: 30,
                    centralized_logging: true,
                },
                tracing: TracingConfig {
                    enabled: true,
                    sampling_rate: 0.1,
                    jaeger_endpoint: Some("http://jaeger:14268/api/traces".to_string()),
                    service_name: "photonic-neural-network".to_string(),
                },
                alerts: vec![
                    AlertRule {
                        name: "high_optical_loss".to_string(),
                        condition: "optical_loss_db > threshold".to_string(),
                        threshold: 10.0,
                        severity: "critical".to_string(),
                        notification_channels: vec!["slack".to_string(), "email".to_string()],
                    },
                    AlertRule {
                        name: "thermal_drift".to_string(),
                        condition: "temperature_drift_c > threshold".to_string(),
                        threshold: 5.0,
                        severity: "warning".to_string(),
                        notification_channels: vec!["email".to_string()],
                    },
                ],
            },
            security: SecurityConfig {
                network_security: NetworkSecurity {
                    vpc_isolation: true,
                    firewall_rules: vec![
                        FirewallRule {
                            name: "allow_https".to_string(),
                            direction: "inbound".to_string(),
                            protocol: "tcp".to_string(),
                            port_range: Some((443, 443)),
                            source_cidrs: vec!["0.0.0.0/0".to_string()],
                            action: "allow".to_string(),
                        },
                        FirewallRule {
                            name: "deny_all_default".to_string(),
                            direction: "inbound".to_string(),
                            protocol: "tcp".to_string(),
                            port_range: None,
                            source_cidrs: vec!["0.0.0.0/0".to_string()],
                            action: "deny".to_string(),
                        },
                    ],
                    ddos_protection: true,
                    waf_enabled: true,
                },
                access_control: AccessControl {
                    authentication_method: "oauth2".to_string(),
                    authorization_model: "rbac".to_string(),
                    multi_factor_auth: true,
                    session_timeout_minutes: 60,
                },
                vulnerability_scanning: VulnerabilityScanning {
                    enabled: true,
                    scan_frequency_hours: 24,
                    severity_threshold: "medium".to_string(),
                    auto_remediation: false,
                },
            },
        }
    }
}

/// Deployment status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeploymentStatus {
    pub deployment_id: String,
    pub status: String, // "pending", "deploying", "running", "scaling", "error"
    pub regions: HashMap<String, RegionStatus>,
    pub total_instances: usize,
    pub health_score: f64,
    pub last_updated: chrono::DateTime<chrono::Utc>,
}

/// Region-specific deployment status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegionStatus {
    pub instances: usize,
    pub health_score: f64,
    pub current_load: f64,
    pub network_latency_ms: f64,
    pub last_scaling_event: Option<chrono::DateTime<chrono::Utc>>,
}

/// Deployment metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeploymentMetrics {
    pub total_requests_per_second: f64,
    pub average_response_time_ms: f64,
    pub error_rate_percent: f64,
    pub optical_power_efficiency: f64,
    pub photonic_throughput_tops: f64,
    pub cost_per_hour: f64,
    pub carbon_emissions_kg_co2_per_hour: f64,
}

/// Global deployment manager
pub struct GlobalDeploymentManager {
    config: CloudConfig,
    deployments: HashMap<String, DeploymentStatus>,
}

impl GlobalDeploymentManager {
    /// Create new global deployment manager
    pub fn new(config: CloudConfig) -> Self {
        Self {
            config,
            deployments: HashMap::new(),
        }
    }

    /// Deploy photonic neural network globally
    pub fn deploy_global(&mut self, deployment_id: String) -> Result<()> {
        // Validate configuration
        self.validate_config()?;
        
        // Create deployment plan
        let deployment_plan = self.create_deployment_plan(&deployment_id)?;
        
        // Initialize deployment status
        let mut region_statuses = HashMap::new();
        for region in &self.config.regions {
            region_statuses.insert(
                region.region_id.clone(),
                RegionStatus {
                    instances: 0,
                    health_score: 0.0,
                    current_load: 0.0,
                    network_latency_ms: region.network_latency_ms,
                    last_scaling_event: None,
                },
            );
        }

        let deployment_status = DeploymentStatus {
            deployment_id: deployment_id.clone(),
            status: "pending".to_string(),
            regions: region_statuses,
            total_instances: 0,
            health_score: 0.0,
            last_updated: chrono::Utc::now(),
        };

        self.deployments.insert(deployment_id.clone(), deployment_status);
        
        // Execute deployment
        self.execute_deployment(&deployment_id, deployment_plan)?;
        
        Ok(())
    }

    /// Validate cloud configuration
    fn validate_config(&self) -> Result<()> {
        if self.config.regions.is_empty() {
            return Err(PhotonicError::InvalidParameter {
                param: "regions".to_string(),
                value: "empty".to_string(),
                constraint: "at least one region required".to_string(),
            });
        }

        for region in &self.config.regions {
            if region.compute_resources.cpu_cores == 0 {
                return Err(PhotonicError::InvalidParameter {
                    param: "cpu_cores".to_string(),
                    value: "0".to_string(),
                    constraint: format!("Region {} must have CPU cores", region.region_id),
                });
            }
        }

        // Validate scaling configuration
        if self.config.scaling.min_instances >= self.config.scaling.max_instances {
            return Err(PhotonicError::InvalidParameter {
                param: "scaling".to_string(),
                value: format!("min={}, max={}", self.config.scaling.min_instances, self.config.scaling.max_instances),
                constraint: "min_instances < max_instances".to_string(),
            });
        }

        Ok(())
    }

    /// Create deployment plan
    fn create_deployment_plan(&self, deployment_id: &str) -> Result<DeploymentPlan> {
        // Cost-performance optimization
        let mut region_allocations = Vec::new();
        
        for region in &self.config.regions {
            // Calculate initial allocation based on resources and cost
            let performance_score = region.compute_resources.cpu_cores as f64 * 
                                   region.compute_resources.gpu_devices.len() as f64;
            let cost_efficiency = performance_score / region.cost_per_hour;
            
            let initial_instances = ((cost_efficiency / 10.0).ceil() as usize)
                .max(1)
                .min(self.config.scaling.max_instances / self.config.regions.len());
            
            region_allocations.push(RegionAllocation {
                region_id: region.region_id.clone(),
                initial_instances,
                max_instances: self.config.scaling.max_instances / self.config.regions.len(),
            });
        }

        Ok(DeploymentPlan {
            deployment_id: deployment_id.to_string(),
            region_allocations,
            global_optimization_enabled: self.config.optimization.algorithm != "none",
            monitoring_enabled: true,
        })
    }

    /// Execute deployment plan
    fn execute_deployment(&mut self, deployment_id: &str, plan: DeploymentPlan) -> Result<()> {
        // Update status to deploying
        if let Some(status) = self.deployments.get_mut(deployment_id) {
            status.status = "deploying".to_string();
            status.last_updated = chrono::Utc::now();
        }

        // Deploy to each region
        for allocation in &plan.region_allocations {
            self.deploy_to_region(deployment_id, allocation)?;
        }

        // Update status to running
        if let Some(status) = self.deployments.get_mut(deployment_id) {
            status.status = "running".to_string();
            status.total_instances = plan.region_allocations
                .iter()
                .map(|a| a.initial_instances)
                .sum();
            status.health_score = 1.0; // Initial perfect health
            status.last_updated = chrono::Utc::now();
        }

        Ok(())
    }

    /// Deploy to specific region
    fn deploy_to_region(&mut self, deployment_id: &str, allocation: &RegionAllocation) -> Result<()> {
        println!("Deploying {} instances to region {}", 
                allocation.initial_instances, 
                allocation.region_id);

        // Update region status
        if let Some(deployment) = self.deployments.get_mut(deployment_id) {
            if let Some(region_status) = deployment.regions.get_mut(&allocation.region_id) {
                region_status.instances = allocation.initial_instances;
                region_status.health_score = 1.0;
            }
        }

        Ok(())
    }

    /// Get deployment status
    pub fn get_deployment_status(&self, deployment_id: &str) -> Option<&DeploymentStatus> {
        self.deployments.get(deployment_id)
    }

    /// Scale deployment based on metrics
    pub fn auto_scale(&mut self, deployment_id: &str, metrics: &DeploymentMetrics) -> Result<()> {
        let deployment = self.deployments.get_mut(deployment_id)
            .ok_or_else(|| PhotonicError::InvalidParameter {
                param: "deployment_id".to_string(),
                value: deployment_id.to_string(),
                constraint: "must exist".to_string(),
            })?;

        let should_scale_up = metrics.average_response_time_ms > 200.0 || 
                             metrics.optical_power_efficiency < 0.8;
        let should_scale_down = metrics.average_response_time_ms < 50.0 && 
                               metrics.optical_power_efficiency > 0.95 &&
                               deployment.total_instances > self.config.scaling.min_instances;

        if should_scale_up && deployment.total_instances < self.config.scaling.max_instances {
            deployment.total_instances += 1;
            deployment.status = "scaling".to_string();
            println!("Scaling up deployment {} to {} instances", 
                    deployment_id, deployment.total_instances);
        } else if should_scale_down {
            deployment.total_instances -= 1;
            deployment.status = "scaling".to_string();
            println!("Scaling down deployment {} to {} instances", 
                    deployment_id, deployment.total_instances);
        }

        deployment.last_updated = chrono::Utc::now();
        Ok(())
    }

    /// Update health scores based on monitoring data
    pub fn update_health_scores(&mut self, deployment_id: &str, region_metrics: HashMap<String, RegionMetrics>) -> Result<()> {
        let deployment = self.deployments.get_mut(deployment_id)
            .ok_or_else(|| PhotonicError::InvalidParameter {
                param: "deployment_id".to_string(),
                value: deployment_id.to_string(),
                constraint: "must exist".to_string(),
            })?;

        let mut total_health = 0.0;
        let mut region_count = 0;

        for (region_id, metrics) in region_metrics {
            if let Some(region_status) = deployment.regions.get_mut(&region_id) {
                // Calculate health score based on multiple factors
                let latency_score = (1.0 - (metrics.network_latency_ms / 1000.0).min(1.0)).max(0.0);
                let error_score = (1.0 - (metrics.error_rate_percent / 100.0).min(1.0)).max(0.0);
                let efficiency_score = metrics.optical_power_efficiency;
                
                region_status.health_score = (latency_score + error_score + efficiency_score) / 3.0;
                region_status.current_load = metrics.cpu_utilization_percent / 100.0;
                
                total_health += region_status.health_score;
                region_count += 1;
            }
        }

        deployment.health_score = if region_count > 0 { total_health / region_count as f64 } else { 0.0 };
        deployment.last_updated = chrono::Utc::now();

        Ok(())
    }
}

/// Deployment plan
#[derive(Debug, Clone)]
pub struct DeploymentPlan {
    pub deployment_id: String,
    pub region_allocations: Vec<RegionAllocation>,
    pub global_optimization_enabled: bool,
    pub monitoring_enabled: bool,
}

/// Region allocation in deployment plan
#[derive(Debug, Clone)]
pub struct RegionAllocation {
    pub region_id: String,
    pub initial_instances: usize,
    pub max_instances: usize,
}

/// Region-specific metrics
#[derive(Debug, Clone)]
pub struct RegionMetrics {
    pub cpu_utilization_percent: f64,
    pub memory_utilization_percent: f64,
    pub network_latency_ms: f64,
    pub error_rate_percent: f64,
    pub optical_power_efficiency: f64,
    pub photonic_throughput_tops: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cloud_config_creation() {
        let config = CloudConfig::default();
        assert_eq!(config.regions.len(), 2);
        assert!(config.scaling.min_instances < config.scaling.max_instances);
    }

    #[test]
    fn test_deployment_manager_creation() {
        let config = CloudConfig::default();
        let manager = GlobalDeploymentManager::new(config);
        assert_eq!(manager.deployments.len(), 0);
    }

    #[test]
    fn test_config_validation() {
        let config = CloudConfig::default();
        let manager = GlobalDeploymentManager::new(config);
        assert!(manager.validate_config().is_ok());
        
        // Test invalid config
        let mut invalid_config = CloudConfig::default();
        invalid_config.regions.clear();
        let invalid_manager = GlobalDeploymentManager::new(invalid_config);
        assert!(invalid_manager.validate_config().is_err());
    }

    #[test]
    fn test_deployment_plan_creation() {
        let config = CloudConfig::default();
        let manager = GlobalDeploymentManager::new(config);
        let plan = manager.create_deployment_plan("test-deployment").unwrap();
        
        assert_eq!(plan.deployment_id, "test-deployment");
        assert_eq!(plan.region_allocations.len(), 2);
        assert!(plan.monitoring_enabled);
    }

    #[test]
    fn test_auto_scaling_decision() {
        let config = CloudConfig::default();
        let mut manager = GlobalDeploymentManager::new(config);
        
        // Create a test deployment
        manager.deploy_global("test-deployment".to_string()).unwrap();
        
        // Test scale-up scenario
        let high_load_metrics = DeploymentMetrics {
            total_requests_per_second: 1000.0,
            average_response_time_ms: 300.0, // High latency
            error_rate_percent: 1.0,
            optical_power_efficiency: 0.7, // Low efficiency
            photonic_throughput_tops: 100.0,
            cost_per_hour: 10.0,
            carbon_emissions_kg_co2_per_hour: 5.0,
        };
        
        let initial_instances = manager.get_deployment_status("test-deployment")
            .unwrap().total_instances;
        
        manager.auto_scale("test-deployment", &high_load_metrics).unwrap();
        
        let final_instances = manager.get_deployment_status("test-deployment")
            .unwrap().total_instances;
        
        assert!(final_instances > initial_instances);
    }

    #[test]
    fn test_health_score_calculation() {
        let config = CloudConfig::default();
        let mut manager = GlobalDeploymentManager::new(config);
        
        manager.deploy_global("test-deployment".to_string()).unwrap();
        
        let mut region_metrics = HashMap::new();
        region_metrics.insert("us-west-2".to_string(), RegionMetrics {
            cpu_utilization_percent: 50.0,
            memory_utilization_percent: 60.0,
            network_latency_ms: 100.0,
            error_rate_percent: 0.1,
            optical_power_efficiency: 0.95,
            photonic_throughput_tops: 500.0,
        });
        
        manager.update_health_scores("test-deployment", region_metrics).unwrap();
        
        let deployment = manager.get_deployment_status("test-deployment").unwrap();
        assert!(deployment.health_score > 0.8); // Should be healthy
    }
}