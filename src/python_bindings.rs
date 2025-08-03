//! Python bindings using PyO3

use crate::core::{OpticalField, WaveguideGeometry, Complex64};
use crate::simulation::{PhotonicArray, ArrayTopology};
use nalgebra::{DMatrix, DVector};
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Python wrapper for OpticalField
#[pyclass]
#[derive(Clone)]
pub struct PyOpticalField {
    inner: OpticalField,
}

#[pymethods]
impl PyOpticalField {
    #[new]
    pub fn new(
        amplitude_real: Vec<Vec<f64>>,
        amplitude_imag: Vec<Vec<f64>>,
        wavelength: f64,
        power: f64,
    ) -> PyResult<Self> {
        if amplitude_real.is_empty() || amplitude_imag.is_empty() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Amplitude arrays cannot be empty"
            ));
        }
        
        let rows = amplitude_real.len();
        let cols = amplitude_real[0].len();
        
        if amplitude_imag.len() != rows || amplitude_imag[0].len() != cols {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Real and imaginary parts must have same shape"
            ));
        }
        
        let mut amplitude = DMatrix::zeros(rows, cols);
        
        for i in 0..rows {
            for j in 0..cols {
                amplitude[(i, j)] = Complex64::new(amplitude_real[i][j], amplitude_imag[i][j]);
            }
        }
        
        let x_coords = DVector::from_fn(cols, |i, _| i as f64 * 1e-6);
        let y_coords = DVector::from_fn(rows, |i, _| i as f64 * 1e-6);
        
        let inner = OpticalField::new(amplitude, wavelength, power, x_coords, y_coords);
        
        Ok(PyOpticalField { inner })
    }
    
    /// Get wavelength
    #[getter]
    pub fn wavelength(&self) -> f64 {
        self.inner.wavelength
    }
    
    /// Get power
    #[getter]
    pub fn power(&self) -> f64 {
        self.inner.power
    }
    
    /// Calculate total power
    pub fn calculate_power(&self) -> f64 {
        self.inner.calculate_power()
    }
    
    /// Normalize to specified power
    pub fn normalize_to_power(&mut self, target_power: f64) {
        self.inner.normalize_to_power(target_power);
    }
    
    /// Get field dimensions
    pub fn dimensions(&self) -> (usize, usize) {
        self.inner.dimensions()
    }
}

/// Python wrapper for PhotonicArray
#[pyclass]
pub struct PyPhotonicArray {
    inner: PhotonicArray,
}

#[pymethods]
impl PyPhotonicArray {
    #[new]
    #[pyo3(signature = (topology_type, rows, cols))]
    pub fn new(topology_type: &str, rows: usize, cols: usize) -> PyResult<Self> {
        let topology = match topology_type {
            "crossbar" => ArrayTopology::Crossbar { rows, cols },
            "broadcast_weight" => ArrayTopology::BroadcastWeight { 
                inputs: rows, 
                weights: cols 
            },
            "mesh" => ArrayTopology::Mesh { size: rows.max(cols) },
            _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Invalid topology type. Use 'crossbar', 'broadcast_weight', or 'mesh'"
            )),
        };
        
        let inner = PhotonicArray::new(topology);
        Ok(PyPhotonicArray { inner })
    }
    
    /// Get array dimensions
    pub fn dimensions(&self) -> (usize, usize) {
        self.inner.dimensions()
    }
    
    /// Forward propagation through array
    pub fn forward(&mut self, inputs: Vec<PyOpticalField>) -> PyResult<Vec<PyOpticalField>> {
        let rust_inputs: Vec<OpticalField> = inputs.into_iter()
            .map(|py_field| py_field.inner)
            .collect();
        
        let outputs = self.inner.forward(&rust_inputs)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;
        
        let py_outputs = outputs.into_iter()
            .map(|field| PyOpticalField { inner: field })
            .collect();
        
        Ok(py_outputs)
    }
    
    /// Get total power consumption
    pub fn total_power(&self) -> f64 {
        self.inner.total_power()
    }
    
    /// Get array metrics
    pub fn metrics(&self, py: Python) -> PyResult<Py<PyDict>> {
        let metrics = self.inner.metrics();
        let dict = PyDict::new(py);
        
        dict.set_item("total_devices", metrics.total_devices)?;
        dict.set_item("active_devices", metrics.active_devices)?;
        dict.set_item("total_power", metrics.total_power)?;
        dict.set_item("average_temperature", metrics.average_temperature)?;
        dict.set_item("memory_usage", metrics.memory_usage)?;
        
        Ok(dict.into())
    }
}

/// Simple matrix multiplication for JAX interface
#[pyfunction]
pub fn jax_photonic_matmul(
    inputs: Vec<f64>,
    weights: Vec<Vec<f64>>,
    _wavelength: f64,
) -> PyResult<Vec<f64>> {
    if weights.is_empty() || weights[0].len() != inputs.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Weight matrix dimensions don't match input vector"
        ));
    }
    
    let output_size = weights.len();
    let mut outputs = vec![0.0; output_size];
    
    for i in 0..output_size {
        for j in 0..inputs.len() {
            outputs[i] += inputs[j] * weights[i][j];
        }
    }
    
    Ok(outputs)
}

/// JAX gradient computation
#[pyfunction]
pub fn jax_photonic_matmul_vjp(
    inputs: Vec<f64>,
    weights: Vec<Vec<f64>>,
    grad_outputs: Vec<f64>,
    _wavelength: f64,
) -> PyResult<(Vec<f64>, Vec<Vec<f64>>)> {
    if weights.is_empty() || grad_outputs.len() != weights.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Gradient output size doesn't match weight matrix rows"
        ));
    }
    
    // Compute gradients
    let mut grad_inputs = vec![0.0; inputs.len()];
    let mut grad_weights = vec![vec![0.0; inputs.len()]; weights.len()];
    
    for i in 0..weights.len() {
        for j in 0..inputs.len() {
            grad_inputs[j] += grad_outputs[i] * weights[i][j];
            grad_weights[i][j] = grad_outputs[i] * inputs[j];
        }
    }
    
    Ok((grad_inputs, grad_weights))
}

/// Waveguide mode calculation
#[pyfunction]
pub fn calculate_waveguide_mode(
    width: f64,
    height: f64,
    core_index: f64,
    cladding_index: f64,
    wavelength: f64,
) -> PyResult<(f64, Vec<Vec<f64>>)> {
    use crate::core::waveguide::EffectiveIndexCalculator;
    
    let geometry = WaveguideGeometry {
        width,
        height,
        core_index: Complex64::new(core_index, 0.0),
        cladding_index: Complex64::new(cladding_index, 0.0),
        substrate_index: Complex64::new(cladding_index, 0.0),
        sidewall_angle: 80.0_f64.to_radians(),
    };
    
    let calculator = EffectiveIndexCalculator::new(geometry, wavelength);
    let n_eff = calculator.calculate_effective_index(0)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;
    
    let profile = calculator.generate_mode_profile(n_eff, 32, 32)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;
    
    // Convert to intensity profile
    let mut intensity = vec![vec![0.0; 32]; 32];
    for i in 0..32 {
        for j in 0..32 {
            intensity[i][j] = profile[(i, j)].norm_sqr();
        }
    }
    
    Ok((n_eff.re, intensity))
}

/// Create device simulator
#[pyfunction]
pub fn create_device_simulator(device_type: &str) -> PyResult<String> {
    match device_type {
        "pcm" => Ok("PCM device simulator created".to_string()),
        "oxide" => Ok("Oxide memristor simulator created".to_string()),
        "ring" => Ok("Ring resonator simulator created".to_string()),
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Unknown device type"
        )),
    }
}

/// Python module definition
#[pymodule]
fn _core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyOpticalField>()?;
    m.add_class::<PyPhotonicArray>()?;
    m.add_function(wrap_pyfunction!(jax_photonic_matmul, m)?)?;
    m.add_function(wrap_pyfunction!(jax_photonic_matmul_vjp, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_waveguide_mode, m)?)?;
    m.add_function(wrap_pyfunction!(create_device_simulator, m)?)?;
    
    // Add constants
    m.add("VERSION", env!("CARGO_PKG_VERSION"))?;
    m.add("SPEED_OF_LIGHT", 299792458.0)?;
    
    Ok(())
}