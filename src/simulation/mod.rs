//! Simulation engine and photonic array implementations

pub mod array;
pub mod result;
pub mod solver;

pub use array::{PhotonicArray, ArrayTopology};
pub use result::{SimulationResult, SimulationMetrics};
pub use solver::{SimulationSolver, SolverConfig};