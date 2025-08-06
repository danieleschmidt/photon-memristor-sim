//! Core physics and mathematical foundations for photonic simulation

pub mod error;
pub mod types;
pub mod waveguide;
pub mod propagation;
pub mod coupling;
pub mod validation;
pub mod logging;
pub mod monitoring;
pub mod error_handling;

pub use error::{PhotonicError, Result};
pub use types::{OpticalField, WaveguideGeometry, DeviceGeometry, Complex64};
pub use waveguide::{WaveguideMode, EffectiveIndexCalculator};
pub use propagation::{PropagationSolver, BeamPropagationMethod};
pub use coupling::{CouplingCalculator, OverlapIntegral};
pub use validation::{PhotonicValidator, ValidationReport, Validatable};
pub use logging::{Logger, LogLevel, PerformanceTimer, get_logger};
pub use monitoring::{Monitor, HealthStatus, TimeSeries, Alert, MonitoringDashboard};
pub use error_handling::{EnhancedError, ErrorHandler, ResilientExecutor, CircuitBreaker};