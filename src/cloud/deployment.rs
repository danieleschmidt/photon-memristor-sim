//! Cloud deployment infrastructure and orchestration

use crate::cloud::{CloudConfig, DeploymentStatus};
use crate::core::{Result, PhotonicError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tokio::time::{sleep, Duration};

/// Container orchestration platform
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OrchestrationPlatform {
    Kubernetes,
    DockerSwarm,
    AWSFargate,
    AzureContainerInstances,
    GoogleCloudRun,
}

/// Deployment orchestrator
pub struct DeploymentOrchestrator {
    config: CloudConfig,
    platform: OrchestrationPlatform,
    active_deployments: Arc<Mutex<HashMap<String, ActiveDeployment>>>,
}

/// Active deployment tracking
#[derive(Debug, Clone)]
pub struct ActiveDeployment {
    pub deployment_id: String,
    pub status: DeploymentStatus,
    pub containers: HashMap<String, Vec<ContainerInstance>>,
    pub load_balancers: HashMap<String, LoadBalancer>,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub last_health_check: chrono::DateTime<chrono::Utc>,
}

/// Container instance information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContainerInstance {
    pub instance_id: String,
    pub region: String,
    pub status: ContainerStatus,
    pub resource_allocation: ResourceAllocation,
    pub health_status: HealthStatus,
    pub metrics: ContainerMetrics,
    pub network_endpoint: NetworkEndpoint,
}

/// Container status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ContainerStatus {
    Pending,
    Starting,
    Running,
    Stopping,
    Stopped,
    Failed,
    Unknown,
}

/// Resource allocation for container
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceAllocation {
    pub cpu_cores: f64,
    pub memory_gb: f64,
    pub gpu_devices: Vec<String>,
    pub storage_gb: f64,
    pub network_bandwidth_mbps: f64,
}

/// Health status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthStatus {
    pub overall_health: f64, // 0.0 to 1.0
    pub last_health_check: chrono::DateTime<chrono::Utc>,
    pub health_checks: Vec<HealthCheck>,
}

/// Individual health check
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheck {
    pub name: String,
    pub status: bool,
    pub response_time_ms: f64,
    pub error_message: Option<String>,
}

/// Container metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContainerMetrics {
    pub cpu_utilization_percent: f64,
    pub memory_utilization_percent: f64,
    pub gpu_utilization_percent: Vec<f64>,
    pub network_rx_mbps: f64,
    pub network_tx_mbps: f64,
    pub disk_io_ops_per_sec: f64,
    pub photonic_operations_per_sec: f64,
    pub optical_power_watts: f64,
}

/// Network endpoint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkEndpoint {
    pub public_ip: Option<String>,
    pub private_ip: String,
    pub port: u16,
    pub protocol: String, // "http", "https", "grpc"
    pub domain_name: Option<String>,
}

/// Load balancer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadBalancer {
    pub lb_id: String,
    pub region: String,
    pub algorithm: LoadBalancingAlgorithm,
    pub health_check: LoadBalancerHealthCheck,
    pub backend_instances: Vec<String>,
    pub ssl_termination: bool,
    pub metrics: LoadBalancerMetrics,
}

/// Load balancing algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LoadBalancingAlgorithm {
    RoundRobin,
    WeightedRoundRobin,
    LeastConnections,
    WeightedLeastConnections,
    IpHash,
    GeographicProximity,
    OpticalLatencyOptimized, // Custom for photonic networks
}

/// Load balancer health check
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadBalancerHealthCheck {
    pub path: String,
    pub port: u16,
    pub protocol: String,
    pub interval_seconds: u32,
    pub timeout_seconds: u32,
    pub healthy_threshold: u32,
    pub unhealthy_threshold: u32,
}

/// Load balancer metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadBalancerMetrics {
    pub requests_per_second: f64,
    pub average_response_time_ms: f64,
    pub error_rate_percent: f64,
    pub active_connections: u64,
    pub bytes_processed_per_second: u64,
}

impl DeploymentOrchestrator {
    /// Create new deployment orchestrator
    pub fn new(config: CloudConfig, platform: OrchestrationPlatform) -> Self {
        Self {
            config,
            platform,
            active_deployments: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Deploy photonic neural network to cloud
    pub async fn deploy(&self, deployment_spec: &DeploymentSpec) -> Result<String> {
        let deployment_id = format!("pnn-{}", uuid::Uuid::new_v4());
        
        // Validate deployment specification
        self.validate_deployment_spec(deployment_spec)?;
        
        // Create deployment plan
        let deployment_plan = self.create_deployment_plan(&deployment_id, deployment_spec).await?;
        
        // Execute deployment across regions
        let active_deployment = self.execute_deployment_plan(deployment_plan).await?;
        
        // Register active deployment
        {
            let mut deployments = self.active_deployments.lock().unwrap();
            deployments.insert(deployment_id.clone(), active_deployment);
        }
        
        // Start health monitoring
        self.start_health_monitoring(&deployment_id).await?;
        
        Ok(deployment_id)
    }

    /// Validate deployment specification
    fn validate_deployment_spec(&self, spec: &DeploymentSpec) -> Result<()> {
        if spec.model_config.layers.is_empty() {
            return Err(PhotonicError::InvalidParameter {
                param: "layers".to_string(),
                value: "empty".to_string(),
                constraint: "at least one layer required".to_string(),
            });
        }

        if spec.resource_requirements.min_memory_gb <= 0.0 {
            return Err(PhotonicError::InvalidParameter {
                param: "min_memory_gb".to_string(),
                value: spec.resource_requirements.min_memory_gb.to_string(),
                constraint: "positive value".to_string(),
            });
        }

        if spec.target_regions.is_empty() {
            return Err(PhotonicError::InvalidParameter {
                param: "target_regions".to_string(),
                value: "empty".to_string(),
                constraint: "at least one region".to_string(),
            });
        }

        // Validate regions exist in config
        for region in &spec.target_regions {
            if !self.config.regions.iter().any(|r| r.region_id == *region) {
                return Err(PhotonicError::InvalidParameter {
                    param: "region".to_string(),
                    value: region.clone(),
                    constraint: "must exist in cloud configuration".to_string(),
                });
            }
        }

        Ok(())
    }

    /// Create deployment plan
    async fn create_deployment_plan(
        &self,
        deployment_id: &str,
        spec: &DeploymentSpec,
    ) -> Result<DeploymentPlan> {
        let mut region_plans = Vec::new();
        
        for region_id in &spec.target_regions {
            let region_config = self.config.regions.iter()
                .find(|r| r.region_id == *region_id)
                .unwrap();
            
            // Calculate optimal instance count based on resources
            let instances_needed = self.calculate_instances_needed(spec, region_config).await?;
            
            // Create container specifications
            let container_specs = self.create_container_specs(spec, region_config)?;
            
            region_plans.push(RegionDeploymentPlan {
                region_id: region_id.clone(),
                instances: instances_needed,
                container_specs,
                load_balancer_config: self.create_load_balancer_config(region_id)?,
            });
        }

        Ok(DeploymentPlan {
            deployment_id: deployment_id.to_string(),
            region_plans,
            global_config: spec.clone(),
        })
    }

    /// Calculate required instances for region
    async fn calculate_instances_needed(
        &self,
        spec: &DeploymentSpec,
        region_config: &crate::cloud::CloudRegion,
    ) -> Result<usize> {
        // Consider model complexity, expected load, and available resources
        let model_complexity = spec.model_config.layers.len() as f64;
        let expected_qps = spec.performance_requirements.target_qps;
        let memory_per_instance = spec.resource_requirements.min_memory_gb;
        
        // Calculate instances based on memory constraints
        let instances_by_memory = (spec.resource_requirements.total_memory_gb / memory_per_instance).ceil() as usize;
        
        // Calculate instances based on compute requirements
        let compute_intensity = model_complexity * expected_qps / 1000.0; // Normalize
        let instances_by_compute = (compute_intensity / region_config.compute_resources.cpu_cores as f64).ceil() as usize;
        
        // Take the maximum to ensure both constraints are met
        let required_instances = instances_by_memory.max(instances_by_compute).max(1);
        
        // Cap at region limits
        Ok(required_instances.min(self.config.scaling.max_instances))
    }

    /// Create container specifications
    fn create_container_specs(
        &self,
        spec: &DeploymentSpec,
        region_config: &crate::cloud::CloudRegion,
    ) -> Result<Vec<ContainerSpec>> {
        let mut container_specs = Vec::new();
        
        // Main inference container
        container_specs.push(ContainerSpec {
            name: "photonic-inference".to_string(),
            image: spec.container_image.clone(),
            resource_allocation: ResourceAllocation {
                cpu_cores: spec.resource_requirements.min_cpu_cores,
                memory_gb: spec.resource_requirements.min_memory_gb,
                gpu_devices: if !region_config.compute_resources.gpu_devices.is_empty() {
                    vec![region_config.compute_resources.gpu_devices[0].model.clone()]
                } else {
                    vec![]
                },
                storage_gb: 50.0, // Default storage
                network_bandwidth_mbps: 1000.0, // 1 Gbps
            },
            environment_variables: spec.environment_variables.clone(),
            health_check: HealthCheckConfig {
                path: "/health".to_string(),
                port: 8080,
                interval_seconds: 30,
                timeout_seconds: 5,
                retries: 3,
            },
            ports: vec![8080, 8081], // Main + metrics port
        });
        
        // Monitoring sidecar
        container_specs.push(ContainerSpec {
            name: "monitoring-agent".to_string(),
            image: "photonic-monitoring:latest".to_string(),
            resource_allocation: ResourceAllocation {
                cpu_cores: 0.5,
                memory_gb: 1.0,
                gpu_devices: vec![],
                storage_gb: 10.0,
                network_bandwidth_mbps: 100.0,
            },
            environment_variables: HashMap::new(),
            health_check: HealthCheckConfig {
                path: "/metrics".to_string(),
                port: 9090,
                interval_seconds: 60,
                timeout_seconds: 10,
                retries: 2,
            },
            ports: vec![9090],
        });

        Ok(container_specs)
    }

    /// Create load balancer configuration
    fn create_load_balancer_config(&self, region_id: &str) -> Result<LoadBalancerConfig> {
        Ok(LoadBalancerConfig {
            name: format!("pnn-lb-{}", region_id),
            algorithm: LoadBalancingAlgorithm::OpticalLatencyOptimized,
            health_check: LoadBalancerHealthCheck {
                path: "/health".to_string(),
                port: 8080,
                protocol: "http".to_string(),
                interval_seconds: 30,
                timeout_seconds: 5,
                healthy_threshold: 2,
                unhealthy_threshold: 3,
            },
            ssl_termination: true,
            sticky_sessions: false,
        })
    }

    /// Execute deployment plan
    async fn execute_deployment_plan(&self, plan: DeploymentPlan) -> Result<ActiveDeployment> {
        let mut containers = HashMap::new();
        let mut load_balancers = HashMap::new();
        
        for region_plan in &plan.region_plans {
            // Deploy containers to region
            let region_containers = self.deploy_containers_to_region(region_plan).await?;
            containers.insert(region_plan.region_id.clone(), region_containers);
            
            // Set up load balancer
            let load_balancer = self.setup_load_balancer(region_plan).await?;
            load_balancers.insert(region_plan.region_id.clone(), load_balancer);
        }

        let deployment_status = DeploymentStatus {
            deployment_id: plan.deployment_id.clone(),
            status: "running".to_string(),
            regions: HashMap::new(), // Will be populated by health checks
            total_instances: plan.region_plans.iter().map(|rp| rp.instances).sum(),
            health_score: 1.0,
            last_updated: chrono::Utc::now(),
        };

        Ok(ActiveDeployment {
            deployment_id: plan.deployment_id,
            status: deployment_status,
            containers,
            load_balancers,
            created_at: chrono::Utc::now(),
            last_health_check: chrono::Utc::now(),
        })
    }

    /// Deploy containers to specific region
    async fn deploy_containers_to_region(&self, region_plan: &RegionDeploymentPlan) -> Result<Vec<ContainerInstance>> {
        let mut instances = Vec::new();
        
        for i in 0..region_plan.instances {
            for spec in &region_plan.container_specs {
                let instance_id = format!("{}-{}-{}", region_plan.region_id, spec.name, i);
                
                // Create container instance
                let instance = self.create_container_instance(&instance_id, spec, &region_plan.region_id).await?;
                instances.push(instance);
                
                // Wait for startup
                sleep(Duration::from_millis(1000)).await;
            }
        }

        Ok(instances)
    }

    /// Create single container instance
    async fn create_container_instance(
        &self,
        instance_id: &str,
        spec: &ContainerSpec,
        region: &str,
    ) -> Result<ContainerInstance> {
        println!("Creating container instance: {} in region {}", instance_id, region);
        
        // Simulate container creation based on platform
        let network_endpoint = self.allocate_network_endpoint(region).await?;
        
        Ok(ContainerInstance {
            instance_id: instance_id.to_string(),
            region: region.to_string(),
            status: ContainerStatus::Running,
            resource_allocation: spec.resource_allocation.clone(),
            health_status: HealthStatus {
                overall_health: 1.0,
                last_health_check: chrono::Utc::now(),
                health_checks: vec![
                    HealthCheck {
                        name: "http_endpoint".to_string(),
                        status: true,
                        response_time_ms: 50.0,
                        error_message: None,
                    },
                    HealthCheck {
                        name: "photonic_system".to_string(),
                        status: true,
                        response_time_ms: 10.0,
                        error_message: None,
                    },
                ],
            },
            metrics: ContainerMetrics {
                cpu_utilization_percent: 25.0,
                memory_utilization_percent: 40.0,
                gpu_utilization_percent: vec![60.0],
                network_rx_mbps: 100.0,
                network_tx_mbps: 80.0,
                disk_io_ops_per_sec: 500.0,
                photonic_operations_per_sec: 1000000.0,
                optical_power_watts: 5.0,
            },
            network_endpoint,
        })
    }

    /// Allocate network endpoint
    async fn allocate_network_endpoint(&self, region: &str) -> Result<NetworkEndpoint> {
        // Simulate IP allocation
        let private_ip = format!("10.0.{}.{}", 
            region.chars().map(|c| c as u8).sum::<u8>() % 255,
            rand::random::<u8>());
        
        Ok(NetworkEndpoint {
            public_ip: Some(format!("54.{}.{}.{}", 
                rand::random::<u8>() % 255,
                rand::random::<u8>() % 255, 
                rand::random::<u8>() % 255)),
            private_ip,
            port: 8080,
            protocol: "https".to_string(),
            domain_name: Some(format!("pnn-{}.photonic.ai", region)),
        })
    }

    /// Set up load balancer for region
    async fn setup_load_balancer(&self, region_plan: &RegionDeploymentPlan) -> Result<LoadBalancer> {
        println!("Setting up load balancer for region: {}", region_plan.region_id);
        
        Ok(LoadBalancer {
            lb_id: format!("lb-{}", region_plan.region_id),
            region: region_plan.region_id.clone(),
            algorithm: region_plan.load_balancer_config.algorithm.clone(),
            health_check: region_plan.load_balancer_config.health_check.clone(),
            backend_instances: Vec::new(), // Will be populated after container deployment
            ssl_termination: region_plan.load_balancer_config.ssl_termination,
            metrics: LoadBalancerMetrics {
                requests_per_second: 0.0,
                average_response_time_ms: 0.0,
                error_rate_percent: 0.0,
                active_connections: 0,
                bytes_processed_per_second: 0,
            },
        })
    }

    /// Start health monitoring for deployment
    async fn start_health_monitoring(&self, deployment_id: &str) -> Result<()> {
        let deployment_id = deployment_id.to_string();
        let deployments = Arc::clone(&self.active_deployments);
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(30));
            
            loop {
                interval.tick().await;
                
                // Perform health checks
                if let Some(mut deployment) = {
                    let mut deployments_guard = deployments.lock().unwrap();
                    deployments_guard.get_mut(&deployment_id).cloned()
                } {
                    // Update health status
                    deployment.last_health_check = chrono::Utc::now();
                    
                    // Update metrics
                    for (_region_id, containers) in &mut deployment.containers {
                        for container in containers {
                            // Simulate metric updates
                            container.metrics.cpu_utilization_percent = 20.0 + rand::random::<f64>() * 60.0;
                            container.metrics.photonic_operations_per_sec = 800000.0 + rand::random::<f64>() * 400000.0;
                            
                            // Update health checks
                            for health_check in &mut container.health_status.health_checks {
                                health_check.status = rand::random::<f64>() > 0.05; // 95% success rate
                                health_check.response_time_ms = 10.0 + rand::random::<f64>() * 90.0;
                            }
                            
                            // Calculate overall health
                            let healthy_checks = container.health_status.health_checks.iter()
                                .filter(|hc| hc.status)
                                .count();
                            container.health_status.overall_health = healthy_checks as f64 / 
                                container.health_status.health_checks.len() as f64;
                        }
                    }
                    
                    // Update deployment in collection
                    let mut deployments_guard = deployments.lock().unwrap();
                    deployments_guard.insert(deployment_id.clone(), deployment);
                }
            }
        });
        
        Ok(())
    }

    /// Get deployment status
    pub fn get_deployment(&self, deployment_id: &str) -> Option<ActiveDeployment> {
        let deployments = self.active_deployments.lock().unwrap();
        deployments.get(deployment_id).cloned()
    }

    /// Scale deployment
    pub async fn scale_deployment(&self, deployment_id: &str, target_instances: usize) -> Result<()> {
        println!("Scaling deployment {} to {} instances", deployment_id, target_instances);
        
        let mut deployments = self.active_deployments.lock().unwrap();
        if let Some(deployment) = deployments.get_mut(deployment_id) {
            deployment.status.total_instances = target_instances;
            deployment.status.status = "scaling".to_string();
            deployment.status.last_updated = chrono::Utc::now();
        }
        
        Ok(())
    }

    /// Terminate deployment
    pub async fn terminate_deployment(&self, deployment_id: &str) -> Result<()> {
        println!("Terminating deployment: {}", deployment_id);
        
        let mut deployments = self.active_deployments.lock().unwrap();
        if let Some(deployment) = deployments.get_mut(deployment_id) {
            deployment.status.status = "terminating".to_string();
            
            // Terminate all containers
            for (region_id, containers) in &mut deployment.containers {
                for container in containers {
                    container.status = ContainerStatus::Stopping;
                    println!("Stopping container {} in region {}", container.instance_id, region_id);
                }
            }
        }
        
        // Remove from active deployments after cleanup
        tokio::time::sleep(Duration::from_secs(30)).await;
        deployments.remove(deployment_id);
        
        Ok(())
    }
}

/// Deployment specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeploymentSpec {
    pub model_config: ModelConfig,
    pub resource_requirements: ResourceRequirements,
    pub performance_requirements: PerformanceRequirements,
    pub target_regions: Vec<String>,
    pub container_image: String,
    pub environment_variables: HashMap<String, String>,
}

/// Neural network model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub model_name: String,
    pub layers: Vec<LayerConfig>,
    pub optimization_config: OptimizationConfig,
}

/// Layer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerConfig {
    pub layer_type: String,
    pub input_dim: usize,
    pub output_dim: usize,
    pub photonic_config: PhotonicLayerConfig,
}

/// Photonic layer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhotonicLayerConfig {
    pub wavelength_nm: f64,
    pub optical_power_mw: f64,
    pub device_type: String,
}

/// Optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationConfig {
    pub algorithm: String,
    pub learning_rate: f64,
    pub batch_size: usize,
}

/// Resource requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceRequirements {
    pub min_cpu_cores: f64,
    pub min_memory_gb: f64,
    pub total_memory_gb: f64,
    pub gpu_required: bool,
    pub photonic_accelerator_required: bool,
}

/// Performance requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceRequirements {
    pub target_qps: f64,
    pub max_latency_ms: f64,
    pub target_accuracy: f64,
}

/// Container specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContainerSpec {
    pub name: String,
    pub image: String,
    pub resource_allocation: ResourceAllocation,
    pub environment_variables: HashMap<String, String>,
    pub health_check: HealthCheckConfig,
    pub ports: Vec<u16>,
}

/// Health check configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheckConfig {
    pub path: String,
    pub port: u16,
    pub interval_seconds: u32,
    pub timeout_seconds: u32,
    pub retries: u32,
}

/// Load balancer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadBalancerConfig {
    pub name: String,
    pub algorithm: LoadBalancingAlgorithm,
    pub health_check: LoadBalancerHealthCheck,
    pub ssl_termination: bool,
    pub sticky_sessions: bool,
}

/// Deployment plan
#[derive(Debug, Clone)]
pub struct DeploymentPlan {
    pub deployment_id: String,
    pub region_plans: Vec<RegionDeploymentPlan>,
    pub global_config: DeploymentSpec,
}

/// Region deployment plan
#[derive(Debug, Clone)]
pub struct RegionDeploymentPlan {
    pub region_id: String,
    pub instances: usize,
    pub container_specs: Vec<ContainerSpec>,
    pub load_balancer_config: LoadBalancerConfig,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cloud::CloudConfig;

    #[tokio::test]
    async fn test_deployment_orchestrator_creation() {
        let config = CloudConfig::default();
        let orchestrator = DeploymentOrchestrator::new(config, OrchestrationPlatform::Kubernetes);
        
        assert_eq!(orchestrator.active_deployments.lock().unwrap().len(), 0);
    }

    #[tokio::test]
    async fn test_deployment_spec_validation() {
        let config = CloudConfig::default();
        let orchestrator = DeploymentOrchestrator::new(config, OrchestrationPlatform::Kubernetes);
        
        let valid_spec = DeploymentSpec {
            model_config: ModelConfig {
                model_name: "test-model".to_string(),
                layers: vec![LayerConfig {
                    layer_type: "photonic_linear".to_string(),
                    input_dim: 784,
                    output_dim: 256,
                    photonic_config: PhotonicLayerConfig {
                        wavelength_nm: 1550.0,
                        optical_power_mw: 10.0,
                        device_type: "ring_resonator".to_string(),
                    },
                }],
                optimization_config: OptimizationConfig {
                    algorithm: "adam".to_string(),
                    learning_rate: 0.001,
                    batch_size: 32,
                },
            },
            resource_requirements: ResourceRequirements {
                min_cpu_cores: 4.0,
                min_memory_gb: 8.0,
                total_memory_gb: 16.0,
                gpu_required: true,
                photonic_accelerator_required: false,
            },
            performance_requirements: PerformanceRequirements {
                target_qps: 1000.0,
                max_latency_ms: 100.0,
                target_accuracy: 0.95,
            },
            target_regions: vec!["us-west-2".to_string()],
            container_image: "photonic-nn:latest".to_string(),
            environment_variables: HashMap::new(),
        };
        
        assert!(orchestrator.validate_deployment_spec(&valid_spec).is_ok());
    }

    #[tokio::test]
    async fn test_instance_calculation() {
        let config = CloudConfig::default();
        let orchestrator = DeploymentOrchestrator::new(config.clone(), OrchestrationPlatform::Kubernetes);
        
        let spec = DeploymentSpec {
            model_config: ModelConfig {
                model_name: "test-model".to_string(),
                layers: vec![LayerConfig {
                    layer_type: "photonic_linear".to_string(),
                    input_dim: 784,
                    output_dim: 256,
                    photonic_config: PhotonicLayerConfig {
                        wavelength_nm: 1550.0,
                        optical_power_mw: 10.0,
                        device_type: "ring_resonator".to_string(),
                    },
                }],
                optimization_config: OptimizationConfig {
                    algorithm: "adam".to_string(),
                    learning_rate: 0.001,
                    batch_size: 32,
                },
            },
            resource_requirements: ResourceRequirements {
                min_cpu_cores: 4.0,
                min_memory_gb: 8.0,
                total_memory_gb: 16.0,
                gpu_required: true,
                photonic_accelerator_required: false,
            },
            performance_requirements: PerformanceRequirements {
                target_qps: 1000.0,
                max_latency_ms: 100.0,
                target_accuracy: 0.95,
            },
            target_regions: vec!["us-west-2".to_string()],
            container_image: "photonic-nn:latest".to_string(),
            environment_variables: HashMap::new(),
        };
        
        let region_config = &config.regions[0];
        let instances = orchestrator.calculate_instances_needed(&spec, region_config).await.unwrap();
        
        assert!(instances >= 1);
        assert!(instances <= config.scaling.max_instances);
    }

    #[test]
    fn test_container_spec_creation() {
        let config = CloudConfig::default();
        let orchestrator = DeploymentOrchestrator::new(config.clone(), OrchestrationPlatform::Kubernetes);
        
        let spec = DeploymentSpec {
            model_config: ModelConfig {
                model_name: "test-model".to_string(),
                layers: vec![],
                optimization_config: OptimizationConfig {
                    algorithm: "adam".to_string(),
                    learning_rate: 0.001,
                    batch_size: 32,
                },
            },
            resource_requirements: ResourceRequirements {
                min_cpu_cores: 2.0,
                min_memory_gb: 4.0,
                total_memory_gb: 8.0,
                gpu_required: false,
                photonic_accelerator_required: false,
            },
            performance_requirements: PerformanceRequirements {
                target_qps: 100.0,
                max_latency_ms: 200.0,
                target_accuracy: 0.9,
            },
            target_regions: vec!["us-west-2".to_string()],
            container_image: "photonic-nn:test".to_string(),
            environment_variables: HashMap::new(),
        };
        
        let region_config = &config.regions[0];
        let container_specs = orchestrator.create_container_specs(&spec, region_config).unwrap();
        
        assert_eq!(container_specs.len(), 2); // Main + monitoring containers
        assert_eq!(container_specs[0].name, "photonic-inference");
        assert_eq!(container_specs[1].name, "monitoring-agent");
    }
}