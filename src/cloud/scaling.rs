//! Auto-scaling module for dynamically adapting cloud deployments

use crate::cloud::{AutoScalingConfig, DeploymentMetrics, CustomScalingMetric};
use crate::core::{Result, PhotonicError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tokio::time::{Duration, Instant};

/// Auto-scaling engine
pub struct AutoScaler {
    config: AutoScalingConfig,
    metrics_history: Arc<Mutex<HashMap<String, Vec<MetricDataPoint>>>>,
    scaling_decisions: Arc<Mutex<Vec<ScalingDecision>>>,
    last_scale_events: Arc<Mutex<HashMap<String, Instant>>>,
}

/// Metric data point with timestamp
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricDataPoint {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub value: f64,
    pub metadata: Option<HashMap<String, String>>,
}

/// Scaling decision record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingDecision {
    pub deployment_id: String,
    pub region: String,
    pub decision_type: ScalingDecisionType,
    pub trigger_metric: String,
    pub metric_value: f64,
    pub threshold: f64,
    pub current_instances: usize,
    pub target_instances: usize,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub executed: bool,
    pub cooldown_until: chrono::DateTime<chrono::Utc>,
}

/// Type of scaling decision
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ScalingDecisionType {
    ScaleUp,
    ScaleDown,
    NoChange,
    Cooldown,
}

/// Scaling action
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingAction {
    pub deployment_id: String,
    pub region: String,
    pub action_type: ScalingActionType,
    pub from_instances: usize,
    pub to_instances: usize,
    pub reason: String,
}

/// Types of scaling actions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScalingActionType {
    Increment,
    Decrement,
    SetTarget,
    EmergencyScale,
}

/// Photonic-specific scaling metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhotonicScalingMetrics {
    pub optical_power_efficiency: f64,
    pub thermal_drift_celsius: f64,
    pub wavelength_stability_ppm: f64,
    pub crosstalk_db: f64,
    pub insertion_loss_db: f64,
    pub switching_speed_ns: f64,
    pub quantum_coherence: f64,
}

impl AutoScaler {
    /// Create new auto-scaler
    pub fn new(config: AutoScalingConfig) -> Self {
        Self {
            config,
            metrics_history: Arc::new(Mutex::new(HashMap::new())),
            scaling_decisions: Arc::new(Mutex::new(Vec::new())),
            last_scale_events: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Evaluate scaling decision for deployment
    pub async fn evaluate_scaling(
        &self,
        deployment_id: &str,
        region: &str,
        current_instances: usize,
        metrics: &DeploymentMetrics,
        photonic_metrics: &PhotonicScalingMetrics,
    ) -> Result<Option<ScalingAction>> {
        // Record metrics
        self.record_metrics(deployment_id, region, metrics, photonic_metrics).await;

        // Check cooldown periods
        if self.is_in_cooldown(deployment_id, region).await {
            return Ok(None);
        }

        // Evaluate standard metrics
        let standard_decision = self.evaluate_standard_metrics(
            deployment_id,
            region, 
            current_instances,
            metrics,
        ).await?;

        // Evaluate photonic-specific metrics
        let photonic_decision = self.evaluate_photonic_metrics(
            deployment_id,
            region,
            current_instances,
            photonic_metrics,
        ).await?;

        // Combine decisions with priority to photonic metrics
        let final_decision = self.combine_scaling_decisions(standard_decision, photonic_decision)?;

        // Record scaling decision
        self.record_scaling_decision(deployment_id, region, &final_decision).await;

        // Convert decision to action
        match final_decision.decision_type {
            ScalingDecisionType::ScaleUp => {
                let target_instances = (current_instances + 1).min(self.config.max_instances);
                Ok(Some(ScalingAction {
                    deployment_id: deployment_id.to_string(),
                    region: region.to_string(),
                    action_type: ScalingActionType::Increment,
                    from_instances: current_instances,
                    to_instances: target_instances,
                    reason: format!("Scale up triggered by {}: {} > {}", 
                                  final_decision.trigger_metric, 
                                  final_decision.metric_value,
                                  final_decision.threshold),
                }))
            },
            ScalingDecisionType::ScaleDown => {
                let target_instances = (current_instances.saturating_sub(1)).max(self.config.min_instances);
                Ok(Some(ScalingAction {
                    deployment_id: deployment_id.to_string(),
                    region: region.to_string(),
                    action_type: ScalingActionType::Decrement,
                    from_instances: current_instances,
                    to_instances: target_instances,
                    reason: format!("Scale down triggered by {}: {} < {}", 
                                  final_decision.trigger_metric,
                                  final_decision.metric_value,
                                  final_decision.threshold),
                }))
            },
            _ => Ok(None),
        }
    }

    /// Record metrics for historical analysis
    async fn record_metrics(
        &self,
        deployment_id: &str,
        region: &str,
        metrics: &DeploymentMetrics,
        photonic_metrics: &PhotonicScalingMetrics,
    ) {
        let mut history = self.metrics_history.lock().unwrap();
        let timestamp = chrono::Utc::now();

        // Record standard metrics
        let standard_metrics = [
            ("cpu_utilization", metrics.average_response_time_ms / 10.0), // Approximation
            ("response_time_ms", metrics.average_response_time_ms),
            ("error_rate_percent", metrics.error_rate_percent),
            ("requests_per_second", metrics.total_requests_per_second),
        ];

        for (metric_name, value) in standard_metrics {
            let key = format!("{}:{}:{}", deployment_id, region, metric_name);
            let data_point = MetricDataPoint {
                timestamp,
                value,
                metadata: None,
            };
            
            history.entry(key).or_insert_with(Vec::new).push(data_point);
        }

        // Record photonic metrics
        let photonic_metric_values = [
            ("optical_power_efficiency", photonic_metrics.optical_power_efficiency),
            ("thermal_drift_celsius", photonic_metrics.thermal_drift_celsius),
            ("wavelength_stability_ppm", photonic_metrics.wavelength_stability_ppm),
            ("crosstalk_db", photonic_metrics.crosstalk_db),
            ("insertion_loss_db", photonic_metrics.insertion_loss_db),
            ("switching_speed_ns", photonic_metrics.switching_speed_ns),
            ("quantum_coherence", photonic_metrics.quantum_coherence),
        ];

        for (metric_name, value) in photonic_metric_values {
            let key = format!("{}:{}:{}", deployment_id, region, metric_name);
            let data_point = MetricDataPoint {
                timestamp,
                value,
                metadata: Some([("type".to_string(), "photonic".to_string())].into()),
            };
            
            history.entry(key).or_insert_with(Vec::new).push(data_point);
        }

        // Cleanup old metrics (keep last 1000 data points per metric)
        for metric_history in history.values_mut() {
            if metric_history.len() > 1000 {
                metric_history.drain(0..metric_history.len() - 1000);
            }
        }
    }

    /// Check if deployment is in cooldown period
    async fn is_in_cooldown(&self, deployment_id: &str, region: &str) -> bool {
        let last_events = self.last_scale_events.lock().unwrap();
        let key = format!("{}:{}", deployment_id, region);
        
        if let Some(last_event_time) = last_events.get(&key) {
            let cooldown_duration = Duration::from_secs(self.config.scale_up_cooldown_seconds);
            return last_event_time.elapsed() < cooldown_duration;
        }
        
        false
    }

    /// Evaluate standard scaling metrics
    async fn evaluate_standard_metrics(
        &self,
        deployment_id: &str,
        region: &str,
        current_instances: usize,
        metrics: &DeploymentMetrics,
    ) -> Result<ScalingDecision> {
        // Estimate CPU utilization from response time (heuristic)
        let estimated_cpu = (metrics.average_response_time_ms / 2.0).min(100.0);
        
        // Check scale-up conditions
        if estimated_cpu > self.config.cpu_scale_up_threshold {
            return Ok(ScalingDecision {
                deployment_id: deployment_id.to_string(),
                region: region.to_string(),
                decision_type: ScalingDecisionType::ScaleUp,
                trigger_metric: "estimated_cpu_utilization".to_string(),
                metric_value: estimated_cpu,
                threshold: self.config.cpu_scale_up_threshold,
                current_instances,
                target_instances: (current_instances + 1).min(self.config.max_instances),
                timestamp: chrono::Utc::now(),
                executed: false,
                cooldown_until: chrono::Utc::now() + chrono::Duration::seconds(self.config.scale_up_cooldown_seconds as i64),
            });
        }

        // Check scale-down conditions
        if estimated_cpu < self.config.cpu_scale_down_threshold && current_instances > self.config.min_instances {
            return Ok(ScalingDecision {
                deployment_id: deployment_id.to_string(),
                region: region.to_string(),
                decision_type: ScalingDecisionType::ScaleDown,
                trigger_metric: "estimated_cpu_utilization".to_string(),
                metric_value: estimated_cpu,
                threshold: self.config.cpu_scale_down_threshold,
                current_instances,
                target_instances: current_instances.saturating_sub(1).max(self.config.min_instances),
                timestamp: chrono::Utc::now(),
                executed: false,
                cooldown_until: chrono::Utc::now() + chrono::Duration::seconds(self.config.scale_down_cooldown_seconds as i64),
            });
        }

        // Check custom metrics
        for custom_metric in &self.config.custom_metrics {
            let metric_value = match custom_metric.name.as_str() {
                "error_rate_percent" => metrics.error_rate_percent,
                "requests_per_second" => metrics.total_requests_per_second,
                "photonic_throughput_tops" => metrics.photonic_throughput_tops,
                "optical_power_efficiency" => metrics.optical_power_efficiency,
                _ => continue,
            };

            let should_trigger = match custom_metric.comparison.as_str() {
                "greater_than" => metric_value > custom_metric.threshold,
                "less_than" => metric_value < custom_metric.threshold,
                _ => false,
            };

            if should_trigger {
                let decision_type = match custom_metric.action.as_str() {
                    "scale_up" => ScalingDecisionType::ScaleUp,
                    "scale_down" => ScalingDecisionType::ScaleDown,
                    _ => ScalingDecisionType::NoChange,
                };

                if decision_type != ScalingDecisionType::NoChange {
                    let target_instances = match decision_type {
                        ScalingDecisionType::ScaleUp => (current_instances + 1).min(self.config.max_instances),
                        ScalingDecisionType::ScaleDown => current_instances.saturating_sub(1).max(self.config.min_instances),
                        _ => current_instances,
                    };

                    return Ok(ScalingDecision {
                        deployment_id: deployment_id.to_string(),
                        region: region.to_string(),
                        decision_type,
                        trigger_metric: custom_metric.name.clone(),
                        metric_value,
                        threshold: custom_metric.threshold,
                        current_instances,
                        target_instances,
                        timestamp: chrono::Utc::now(),
                        executed: false,
                        cooldown_until: chrono::Utc::now() + chrono::Duration::seconds(self.config.scale_up_cooldown_seconds as i64),
                    });
                }
            }
        }

        // No scaling needed
        Ok(ScalingDecision {
            deployment_id: deployment_id.to_string(),
            region: region.to_string(),
            decision_type: ScalingDecisionType::NoChange,
            trigger_metric: "none".to_string(),
            metric_value: 0.0,
            threshold: 0.0,
            current_instances,
            target_instances: current_instances,
            timestamp: chrono::Utc::now(),
            executed: false,
            cooldown_until: chrono::Utc::now(),
        })
    }

    /// Evaluate photonic-specific scaling metrics
    async fn evaluate_photonic_metrics(
        &self,
        deployment_id: &str,
        region: &str,
        current_instances: usize,
        photonic_metrics: &PhotonicScalingMetrics,
    ) -> Result<ScalingDecision> {
        // Critical photonic conditions that require immediate scaling
        
        // Optical power efficiency too low - scale up for more redundancy
        if photonic_metrics.optical_power_efficiency < 0.7 {
            return Ok(ScalingDecision {
                deployment_id: deployment_id.to_string(),
                region: region.to_string(),
                decision_type: ScalingDecisionType::ScaleUp,
                trigger_metric: "optical_power_efficiency".to_string(),
                metric_value: photonic_metrics.optical_power_efficiency,
                threshold: 0.7,
                current_instances,
                target_instances: (current_instances + 2).min(self.config.max_instances), // Scale up by 2 for critical issues
                timestamp: chrono::Utc::now(),
                executed: false,
                cooldown_until: chrono::Utc::now() + chrono::Duration::seconds(60), // Shorter cooldown for critical issues
            });
        }

        // Thermal drift too high - scale up to distribute load
        if photonic_metrics.thermal_drift_celsius > 5.0 {
            return Ok(ScalingDecision {
                deployment_id: deployment_id.to_string(),
                region: region.to_string(),
                decision_type: ScalingDecisionType::ScaleUp,
                trigger_metric: "thermal_drift_celsius".to_string(),
                metric_value: photonic_metrics.thermal_drift_celsius,
                threshold: 5.0,
                current_instances,
                target_instances: (current_instances + 1).min(self.config.max_instances),
                timestamp: chrono::Utc::now(),
                executed: false,
                cooldown_until: chrono::Utc::now() + chrono::Duration::seconds(120),
            });
        }

        // Crosstalk too high - scale up for signal isolation
        if photonic_metrics.crosstalk_db > -20.0 { // crosstalk should be negative (better isolation)
            return Ok(ScalingDecision {
                deployment_id: deployment_id.to_string(),
                region: region.to_string(),
                decision_type: ScalingDecisionType::ScaleUp,
                trigger_metric: "crosstalk_db".to_string(),
                metric_value: photonic_metrics.crosstalk_db,
                threshold: -20.0,
                current_instances,
                target_instances: (current_instances + 1).min(self.config.max_instances),
                timestamp: chrono::Utc::now(),
                executed: false,
                cooldown_until: chrono::Utc::now() + chrono::Duration::seconds(180),
            });
        }

        // Quantum coherence loss - scale up for fault tolerance
        if photonic_metrics.quantum_coherence < 0.8 {
            return Ok(ScalingDecision {
                deployment_id: deployment_id.to_string(),
                region: region.to_string(),
                decision_type: ScalingDecisionType::ScaleUp,
                trigger_metric: "quantum_coherence".to_string(),
                metric_value: photonic_metrics.quantum_coherence,
                threshold: 0.8,
                current_instances,
                target_instances: (current_instances + 1).min(self.config.max_instances),
                timestamp: chrono::Utc::now(),
                executed: false,
                cooldown_until: chrono::Utc::now() + chrono::Duration::seconds(90),
            });
        }

        // Positive scaling conditions (scale down when conditions are excellent)
        let all_metrics_excellent = 
            photonic_metrics.optical_power_efficiency > 0.95 &&
            photonic_metrics.thermal_drift_celsius < 1.0 &&
            photonic_metrics.crosstalk_db < -30.0 &&
            photonic_metrics.quantum_coherence > 0.98 &&
            current_instances > self.config.min_instances;

        if all_metrics_excellent {
            return Ok(ScalingDecision {
                deployment_id: deployment_id.to_string(),
                region: region.to_string(),
                decision_type: ScalingDecisionType::ScaleDown,
                trigger_metric: "all_photonic_metrics_excellent".to_string(),
                metric_value: 1.0,
                threshold: 1.0,
                current_instances,
                target_instances: current_instances.saturating_sub(1).max(self.config.min_instances),
                timestamp: chrono::Utc::now(),
                executed: false,
                cooldown_until: chrono::Utc::now() + chrono::Duration::seconds(self.config.scale_down_cooldown_seconds as i64),
            });
        }

        // No photonic-specific scaling needed
        Ok(ScalingDecision {
            deployment_id: deployment_id.to_string(),
            region: region.to_string(),
            decision_type: ScalingDecisionType::NoChange,
            trigger_metric: "photonic_stable".to_string(),
            metric_value: 0.0,
            threshold: 0.0,
            current_instances,
            target_instances: current_instances,
            timestamp: chrono::Utc::now(),
            executed: false,
            cooldown_until: chrono::Utc::now(),
        })
    }

    /// Combine multiple scaling decisions with priority logic
    fn combine_scaling_decisions(
        &self,
        standard_decision: ScalingDecision,
        photonic_decision: ScalingDecision,
    ) -> Result<ScalingDecision> {
        // Priority order:
        // 1. Photonic scale-up (critical for system stability)
        // 2. Standard scale-up 
        // 3. Standard scale-down
        // 4. Photonic scale-down
        // 5. No change

        match (&photonic_decision.decision_type, &standard_decision.decision_type) {
            // Photonic scale-up takes highest priority
            (ScalingDecisionType::ScaleUp, _) => Ok(photonic_decision),
            
            // Standard scale-up if no photonic scaling needed
            (ScalingDecisionType::NoChange, ScalingDecisionType::ScaleUp) => Ok(standard_decision),
            
            // Standard scale-down if no other scaling needed
            (ScalingDecisionType::NoChange, ScalingDecisionType::ScaleDown) => Ok(standard_decision),
            
            // Photonic scale-down only if no standard scaling needed
            (ScalingDecisionType::ScaleDown, ScalingDecisionType::NoChange) => Ok(photonic_decision),
            
            // Conflicting scale decisions - prefer scale-up for safety
            (ScalingDecisionType::ScaleDown, ScalingDecisionType::ScaleUp) => Ok(standard_decision),
            (ScalingDecisionType::ScaleUp, ScalingDecisionType::ScaleDown) => Ok(photonic_decision),
            
            // Both scale in same direction - choose the more aggressive one
            (ScalingDecisionType::ScaleUp, ScalingDecisionType::ScaleUp) => {
                if photonic_decision.target_instances > standard_decision.target_instances {
                    Ok(photonic_decision)
                } else {
                    Ok(standard_decision)
                }
            },
            (ScalingDecisionType::ScaleDown, ScalingDecisionType::ScaleDown) => {
                if photonic_decision.target_instances < standard_decision.target_instances {
                    Ok(photonic_decision)
                } else {
                    Ok(standard_decision)
                }
            },
            
            // No scaling needed
            _ => Ok(standard_decision),
        }
    }

    /// Record scaling decision for audit and analysis
    async fn record_scaling_decision(&self, deployment_id: &str, region: &str, decision: &ScalingDecision) {
        let mut decisions = self.scaling_decisions.lock().unwrap();
        decisions.push(decision.clone());
        
        // Keep only last 1000 decisions
        let len = decisions.len();
        if len > 1000 {
            decisions.drain(0..len - 1000);
        }

        // Update last scale event time if action will be taken
        if matches!(decision.decision_type, ScalingDecisionType::ScaleUp | ScalingDecisionType::ScaleDown) {
            let mut last_events = self.last_scale_events.lock().unwrap();
            let key = format!("{}:{}", deployment_id, region);
            last_events.insert(key, Instant::now());
        }
    }

    /// Get scaling history for a deployment
    pub async fn get_scaling_history(&self, deployment_id: &str, region: Option<&str>) -> Vec<ScalingDecision> {
        let decisions = self.scaling_decisions.lock().unwrap();
        
        decisions.iter()
            .filter(|d| {
                d.deployment_id == deployment_id && 
                region.map_or(true, |r| d.region == r)
            })
            .cloned()
            .collect()
    }

    /// Get metrics history
    pub async fn get_metrics_history(&self, deployment_id: &str, region: &str, metric_name: &str) -> Option<Vec<MetricDataPoint>> {
        let history = self.metrics_history.lock().unwrap();
        let key = format!("{}:{}:{}", deployment_id, region, metric_name);
        history.get(&key).cloned()
    }

    /// Analyze scaling patterns and provide recommendations
    pub async fn analyze_scaling_patterns(&self, deployment_id: &str) -> ScalingAnalysis {
        let history = self.get_scaling_history(deployment_id, None).await;
        
        let total_scaling_events = history.len();
        let scale_up_events = history.iter().filter(|d| matches!(d.decision_type, ScalingDecisionType::ScaleUp)).count();
        let scale_down_events = history.iter().filter(|d| matches!(d.decision_type, ScalingDecisionType::ScaleDown)).count();
        
        // Calculate average time between scaling events
        let mut time_between_events = Vec::new();
        for i in 1..history.len() {
            let duration = history[i].timestamp.signed_duration_since(history[i-1].timestamp);
            time_between_events.push(duration.num_seconds() as f64);
        }
        let avg_time_between_events = if time_between_events.is_empty() { 
            0.0 
        } else { 
            time_between_events.iter().sum::<f64>() / time_between_events.len() as f64 
        };

        // Identify most common scaling triggers
        let mut trigger_counts: HashMap<String, usize> = HashMap::new();
        for decision in &history {
            *trigger_counts.entry(decision.trigger_metric.clone()).or_insert(0) += 1;
        }
        let most_common_trigger = trigger_counts.into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(trigger, _)| trigger)
            .unwrap_or_else(|| "none".to_string());

        // Generate recommendations
        let mut recommendations = Vec::new();
        
        if scale_up_events as f64 / total_scaling_events as f64 > 0.7 {
            recommendations.push("Consider increasing baseline instance count - frequent scale-ups detected".to_string());
        }
        
        if avg_time_between_events < 300.0 { // Less than 5 minutes
            recommendations.push("Scaling events are too frequent - consider adjusting thresholds or cooldown periods".to_string());
        }
        
        if most_common_trigger.contains("photonic") {
            recommendations.push(format!("Photonic metric '{}' is frequently triggering scaling - investigate hardware stability", most_common_trigger));
        }

        ScalingAnalysis {
            deployment_id: deployment_id.to_string(),
            total_scaling_events,
            scale_up_events,
            scale_down_events,
            avg_time_between_events_seconds: avg_time_between_events,
            most_common_trigger,
            recommendations,
            analysis_timestamp: chrono::Utc::now(),
        }
    }

    /// Predict future scaling needs based on historical patterns
    pub async fn predict_scaling_needs(&self, deployment_id: &str, region: &str, hours_ahead: u32) -> Result<ScalingPrediction> {
        // Get historical metrics
        let cpu_history = self.get_metrics_history(deployment_id, region, "cpu_utilization").await
            .unwrap_or_default();
        let photonic_efficiency_history = self.get_metrics_history(deployment_id, region, "optical_power_efficiency").await
            .unwrap_or_default();
        
        if cpu_history.len() < 10 || photonic_efficiency_history.len() < 10 {
            return Ok(ScalingPrediction {
                deployment_id: deployment_id.to_string(),
                region: region.to_string(),
                hours_ahead,
                predicted_instances_needed: None,
                confidence: 0.0,
                key_factors: vec!["Insufficient historical data".to_string()],
                recommended_action: "Monitor for more data".to_string(),
            });
        }

        // Simple trend analysis (in a production system, you'd use more sophisticated ML models)
        let recent_cpu_values: Vec<f64> = cpu_history.iter()
            .rev()
            .take(20)
            .map(|dp| dp.value)
            .collect();
        
        let cpu_trend = if recent_cpu_values.len() > 1 {
            let first_half_avg = recent_cpu_values[recent_cpu_values.len()/2..].iter().sum::<f64>() / (recent_cpu_values.len()/2) as f64;
            let second_half_avg = recent_cpu_values[..recent_cpu_values.len()/2].iter().sum::<f64>() / (recent_cpu_values.len()/2) as f64;
            second_half_avg - first_half_avg
        } else {
            0.0
        };

        let recent_efficiency_values: Vec<f64> = photonic_efficiency_history.iter()
            .rev()
            .take(20)
            .map(|dp| dp.value)
            .collect();
            
        let efficiency_trend = if recent_efficiency_values.len() > 1 {
            let first_half_avg = recent_efficiency_values[recent_efficiency_values.len()/2..].iter().sum::<f64>() / (recent_efficiency_values.len()/2) as f64;
            let second_half_avg = recent_efficiency_values[..recent_efficiency_values.len()/2].iter().sum::<f64>() / (recent_efficiency_values.len()/2) as f64;
            second_half_avg - first_half_avg
        } else {
            0.0
        };

        // Predict based on trends
        let current_avg_cpu = recent_cpu_values.iter().sum::<f64>() / recent_cpu_values.len() as f64;
        let current_avg_efficiency = recent_efficiency_values.iter().sum::<f64>() / recent_efficiency_values.len() as f64;
        
        let projected_cpu = current_avg_cpu + (cpu_trend * hours_ahead as f64 / 24.0);
        let projected_efficiency = current_avg_efficiency + (efficiency_trend * hours_ahead as f64 / 24.0);

        // Determine scaling needs
        let mut predicted_instances = 1;
        let mut key_factors = Vec::new();
        let mut confidence = 0.5_f64;

        if projected_cpu > self.config.cpu_scale_up_threshold {
            predicted_instances += ((projected_cpu - self.config.cpu_scale_up_threshold) / 20.0).ceil() as usize;
            key_factors.push(format!("Projected CPU utilization: {:.1}%", projected_cpu));
            confidence += 0.2;
        }

        if projected_efficiency < 0.8 {
            predicted_instances += 1;
            key_factors.push(format!("Projected optical efficiency: {:.2}", projected_efficiency));
            confidence += 0.3;
        }

        let recommended_action = if predicted_instances > 1 {
            "Consider pre-scaling to handle predicted load"
        } else {
            "Current capacity appears sufficient"
        };

        Ok(ScalingPrediction {
            deployment_id: deployment_id.to_string(),
            region: region.to_string(),
            hours_ahead,
            predicted_instances_needed: Some(predicted_instances),
            confidence: confidence.min(1.0_f64),
            key_factors,
            recommended_action: recommended_action.to_string(),
        })
    }
}

/// Scaling analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingAnalysis {
    pub deployment_id: String,
    pub total_scaling_events: usize,
    pub scale_up_events: usize,
    pub scale_down_events: usize,
    pub avg_time_between_events_seconds: f64,
    pub most_common_trigger: String,
    pub recommendations: Vec<String>,
    pub analysis_timestamp: chrono::DateTime<chrono::Utc>,
}

/// Scaling prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingPrediction {
    pub deployment_id: String,
    pub region: String,
    pub hours_ahead: u32,
    pub predicted_instances_needed: Option<usize>,
    pub confidence: f64,
    pub key_factors: Vec<String>,
    pub recommended_action: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cloud::CustomScalingMetric;

    fn create_test_config() -> AutoScalingConfig {
        AutoScalingConfig {
            min_instances: 2,
            max_instances: 10,
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
            ],
            scale_up_cooldown_seconds: 300,
            scale_down_cooldown_seconds: 600,
        }
    }

    fn create_test_metrics() -> (DeploymentMetrics, PhotonicScalingMetrics) {
        let deployment_metrics = DeploymentMetrics {
            total_requests_per_second: 1000.0,
            average_response_time_ms: 150.0, // High response time
            error_rate_percent: 0.5,
            optical_power_efficiency: 0.85,
            photonic_throughput_tops: 500.0,
            cost_per_hour: 25.0,
            carbon_emissions_kg_co2_per_hour: 12.0,
        };

        let photonic_metrics = PhotonicScalingMetrics {
            optical_power_efficiency: 0.65, // Low efficiency - should trigger scale-up
            thermal_drift_celsius: 3.0,
            wavelength_stability_ppm: 10.0,
            crosstalk_db: -25.0,
            insertion_loss_db: 5.0,
            switching_speed_ns: 2.0,
            quantum_coherence: 0.95,
        };

        (deployment_metrics, photonic_metrics)
    }

    #[tokio::test]
    async fn test_auto_scaler_creation() {
        let config = create_test_config();
        let scaler = AutoScaler::new(config);
        
        assert_eq!(scaler.config.min_instances, 2);
        assert_eq!(scaler.config.max_instances, 10);
    }

    #[tokio::test]
    async fn test_photonic_scaling_decision() {
        let config = create_test_config();
        let scaler = AutoScaler::new(config);
        let (deployment_metrics, photonic_metrics) = create_test_metrics();

        let action = scaler.evaluate_scaling(
            "test-deployment",
            "us-west-2", 
            3, // current instances
            &deployment_metrics,
            &photonic_metrics,
        ).await.unwrap();

        assert!(action.is_some());
        let scaling_action = action.unwrap();
        assert_eq!(scaling_action.action_type, ScalingActionType::Increment);
        assert!(scaling_action.to_instances > scaling_action.from_instances);
        assert!(scaling_action.reason.contains("optical_power_efficiency"));
    }

    #[tokio::test] 
    async fn test_standard_scaling_decision() {
        let config = create_test_config();
        let scaler = AutoScaler::new(config);
        
        // Create metrics that should trigger standard scaling
        let high_load_metrics = DeploymentMetrics {
            total_requests_per_second: 2000.0,
            average_response_time_ms: 300.0, // Very high response time
            error_rate_percent: 0.1,
            optical_power_efficiency: 0.9, // Good optical efficiency
            photonic_throughput_tops: 800.0,
            cost_per_hour: 40.0,
            carbon_emissions_kg_co2_per_hour: 20.0,
        };

        let good_photonic_metrics = PhotonicScalingMetrics {
            optical_power_efficiency: 0.92, // Good efficiency
            thermal_drift_celsius: 1.5,
            wavelength_stability_ppm: 5.0,
            crosstalk_db: -28.0,
            insertion_loss_db: 3.0,
            switching_speed_ns: 1.5,
            quantum_coherence: 0.98,
        };

        let action = scaler.evaluate_scaling(
            "test-deployment",
            "us-west-2",
            2,
            &high_load_metrics,
            &good_photonic_metrics,
        ).await.unwrap();

        assert!(action.is_some());
        let scaling_action = action.unwrap();
        assert_eq!(scaling_action.action_type, ScalingActionType::Increment);
        assert!(scaling_action.reason.contains("estimated_cpu_utilization"));
    }

    #[tokio::test]
    async fn test_metrics_recording() {
        let config = create_test_config();
        let scaler = AutoScaler::new(config);
        let (deployment_metrics, photonic_metrics) = create_test_metrics();

        // Record metrics
        scaler.record_metrics("test-deployment", "us-west-2", &deployment_metrics, &photonic_metrics).await;

        // Check that metrics were recorded
        let cpu_history = scaler.get_metrics_history("test-deployment", "us-west-2", "response_time_ms").await;
        assert!(cpu_history.is_some());
        assert_eq!(cpu_history.unwrap().len(), 1);

        let photonic_history = scaler.get_metrics_history("test-deployment", "us-west-2", "optical_power_efficiency").await;
        assert!(photonic_history.is_some());
        assert_eq!(photonic_history.unwrap().len(), 1);
    }

    #[tokio::test]
    async fn test_scaling_decision_combination() {
        let config = create_test_config();
        let scaler = AutoScaler::new(config);

        let standard_scale_up = ScalingDecision {
            deployment_id: "test".to_string(),
            region: "us-west-2".to_string(),
            decision_type: ScalingDecisionType::ScaleUp,
            trigger_metric: "cpu".to_string(),
            metric_value: 75.0,
            threshold: 70.0,
            current_instances: 3,
            target_instances: 4,
            timestamp: chrono::Utc::now(),
            executed: false,
            cooldown_until: chrono::Utc::now(),
        };

        let photonic_scale_up = ScalingDecision {
            deployment_id: "test".to_string(),
            region: "us-west-2".to_string(),
            decision_type: ScalingDecisionType::ScaleUp,
            trigger_metric: "optical_power_efficiency".to_string(),
            metric_value: 0.6,
            threshold: 0.7,
            current_instances: 3,
            target_instances: 5, // More aggressive scaling
            timestamp: chrono::Utc::now(),
            executed: false,
            cooldown_until: chrono::Utc::now(),
        };

        let combined = scaler.combine_scaling_decisions(standard_scale_up, photonic_scale_up).unwrap();
        
        // Should choose the more aggressive photonic scaling
        assert_eq!(combined.target_instances, 5);
        assert_eq!(combined.trigger_metric, "optical_power_efficiency");
    }

    #[tokio::test]
    async fn test_scaling_analysis() {
        let config = create_test_config();
        let scaler = AutoScaler::new(config);

        // Record some scaling decisions
        let decision1 = ScalingDecision {
            deployment_id: "test-deployment".to_string(),
            region: "us-west-2".to_string(),
            decision_type: ScalingDecisionType::ScaleUp,
            trigger_metric: "optical_power_efficiency".to_string(),
            metric_value: 0.6,
            threshold: 0.7,
            current_instances: 2,
            target_instances: 3,
            timestamp: chrono::Utc::now(),
            executed: false,
            cooldown_until: chrono::Utc::now(),
        };

        {
            let mut decisions = scaler.scaling_decisions.lock().unwrap();
            decisions.push(decision1);
        }

        let analysis = scaler.analyze_scaling_patterns("test-deployment").await;
        
        assert_eq!(analysis.deployment_id, "test-deployment");
        assert_eq!(analysis.total_scaling_events, 1);
        assert_eq!(analysis.scale_up_events, 1);
        assert_eq!(analysis.most_common_trigger, "optical_power_efficiency");
        assert!(!analysis.recommendations.is_empty());
    }
}