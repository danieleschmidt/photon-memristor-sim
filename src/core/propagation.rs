//! Light propagation methods and solvers

use crate::core::{PhotonicError, Result, Complex64, OpticalField};
use nalgebra::{DMatrix, DVector};
use std::f64::consts::PI;

/// Trait for optical propagation solvers
pub trait PropagationSolver {
    /// Propagate optical field through a distance
    fn propagate(&self, field: &OpticalField, distance: f64) -> Result<OpticalField>;
    
    /// Set refractive index profile
    fn set_index_profile(&mut self, index_profile: DMatrix<Complex64>);
}

/// Beam Propagation Method solver
pub struct BeamPropagationMethod {
    /// Refractive index profile
    index_profile: DMatrix<Complex64>,
    /// Reference refractive index
    reference_index: f64,
    /// Step size in propagation direction
    step_size: f64,
    /// Grid spacing (dx, dy)
    grid_spacing: (f64, f64),
}

impl BeamPropagationMethod {
    /// Create new BPM solver
    pub fn new(
        index_profile: DMatrix<Complex64>,
        reference_index: f64,
        step_size: f64,
        grid_spacing: (f64, f64),
    ) -> Self {
        Self {
            index_profile,
            reference_index,
            step_size,
            grid_spacing,
        }
    }
    
    /// Calculate differential phase from index profile
    fn calculate_phase_operator(&self, wavelength: f64) -> DMatrix<Complex64> {
        let k0 = 2.0 * PI / wavelength;
        let k_ref = k0 * self.reference_index;
        
        self.index_profile.map(|n| {
            let delta_k = k0 * n - Complex64::new(k_ref, 0.0);
            Complex64::new(0.0, -delta_k.re * self.step_size)
        })
    }
    
    /// Apply Fresnel diffraction operator using FFT
    fn apply_diffraction(&self, field: &DMatrix<Complex64>, wavelength: f64) -> DMatrix<Complex64> {
        let k0 = 2.0 * PI / wavelength;
        let k_ref = k0 * self.reference_index;
        
        let (ny, nx) = field.shape();
        let mut result = field.clone();
        
        // Simple finite difference approximation for Laplacian
        for i in 1..ny-1 {
            for j in 1..nx-1 {
                let laplacian = (field[(i-1, j)] + field[(i+1, j)] - 2.0 * field[(i, j)]) / (self.grid_spacing.1 * self.grid_spacing.1) +
                              (field[(i, j-1)] + field[(i, j+1)] - 2.0 * field[(i, j)]) / (self.grid_spacing.0 * self.grid_spacing.0);
                
                let diffraction_term = Complex64::new(0.0, 1.0) * laplacian * self.step_size / (2.0 * k_ref);
                result[(i, j)] = field[(i, j)] + field[(i, j)] * diffraction_term;
            }
        }
        
        result
    }
    
    /// Single propagation step
    fn propagation_step(&self, field: &DMatrix<Complex64>, wavelength: f64) -> DMatrix<Complex64> {
        // Split-step method: diffraction -> phase -> diffraction
        
        // Half diffraction step
        let mut field_temp = self.apply_diffraction(field, wavelength);
        
        // Full phase step
        let phase_op = self.calculate_phase_operator(wavelength);
        for i in 0..field_temp.nrows() {
            for j in 0..field_temp.ncols() {
                field_temp[(i, j)] *= phase_op[(i, j)].exp();
            }
        }
        
        // Half diffraction step
        self.apply_diffraction(&field_temp, wavelength)
    }
}

impl PropagationSolver for BeamPropagationMethod {
    fn propagate(&self, field: &OpticalField, distance: f64) -> Result<OpticalField> {
        if distance < 0.0 {
            return Err(PhotonicError::invalid_parameter(
                "distance",
                distance,
                ">= 0"
            ));
        }
        
        let num_steps = (distance / self.step_size).ceil() as usize;
        let actual_step = distance / num_steps as f64;
        
        let mut current_field = field.amplitude.clone();
        
        // Propagate step by step
        for _ in 0..num_steps {
            current_field = self.propagation_step(&current_field, field.wavelength);
        }
        
        // Create output field
        let mut output_field = field.clone();
        output_field.amplitude = current_field;
        
        // Update power (accounting for losses)
        output_field.power = output_field.calculate_power();
        
        Ok(output_field)
    }
    
    fn set_index_profile(&mut self, index_profile: DMatrix<Complex64>) {
        self.index_profile = index_profile;
    }
}

/// Transfer matrix method for linear devices
pub struct TransferMatrixMethod {
    /// Device length
    length: f64,
    /// Transfer matrix
    transfer_matrix: DMatrix<Complex64>,
}

impl TransferMatrixMethod {
    /// Create new transfer matrix solver
    pub fn new(length: f64) -> Self {
        // Initialize with identity matrix
        let transfer_matrix = DMatrix::identity(2, 2);
        Self {
            length,
            transfer_matrix,
        }
    }
    
    /// Set transfer matrix for the device
    pub fn set_transfer_matrix(&mut self, matrix: DMatrix<Complex64>) -> Result<()> {
        if matrix.shape() != (2, 2) {
            return Err(PhotonicError::invalid_parameter(
                "matrix_size",
                format!("{}x{}", matrix.nrows(), matrix.ncols()),
                "2x2"
            ));
        }
        self.transfer_matrix = matrix;
        Ok(())
    }
    
    /// Create transfer matrix for uniform waveguide section
    pub fn uniform_waveguide(length: f64, effective_index: Complex64, wavelength: f64) -> Self {
        let k0 = 2.0 * PI / wavelength;
        let beta = k0 * effective_index;
        let phase = Complex64::new(0.0, -beta.re * length) * (-beta.im * length).exp();
        
        let mut matrix = DMatrix::zeros(2, 2);
        matrix[(0, 0)] = phase;
        matrix[(1, 1)] = phase;
        
        Self {
            length,
            transfer_matrix: matrix,
        }
    }
    
    /// Apply transfer matrix to field amplitudes
    pub fn apply_to_amplitudes(&self, input_amplitudes: &DVector<Complex64>) -> Result<DVector<Complex64>> {
        if input_amplitudes.len() != 2 {
            return Err(PhotonicError::invalid_parameter(
                "input_size",
                input_amplitudes.len(),
                "2 (forward and backward amplitudes)"
            ));
        }
        
        Ok(&self.transfer_matrix * input_amplitudes)
    }
}

/// Free-space propagation using Fresnel diffraction
pub struct FresnelPropagator {
    /// Grid spacing
    grid_spacing: (f64, f64),
}

impl FresnelPropagator {
    /// Create new Fresnel propagator
    pub fn new(grid_spacing: (f64, f64)) -> Self {
        Self { grid_spacing }
    }
    
    /// Fresnel number calculation
    pub fn fresnel_number(&self, aperture_size: f64, distance: f64, wavelength: f64) -> f64 {
        aperture_size * aperture_size / (wavelength * distance)
    }
    
    /// Apply Fresnel propagation kernel
    fn fresnel_kernel(&self, distance: f64, wavelength: f64, nx: usize, ny: usize) -> DMatrix<Complex64> {
        let mut kernel = DMatrix::zeros(ny, nx);
        let k = 2.0 * PI / wavelength;
        
        for i in 0..ny {
            for j in 0..nx {
                let x = ((j as f64) - (nx as f64 / 2.0)) * self.grid_spacing.0;
                let y = ((i as f64) - (ny as f64 / 2.0)) * self.grid_spacing.1;
                
                let r_squared = x * x + y * y;
                let phase = k * r_squared / (2.0 * distance);
                
                kernel[(i, j)] = Complex64::new(0.0, phase).exp() / Complex64::new(distance, 0.0);
            }
        }
        
        kernel
    }
}

impl PropagationSolver for FresnelPropagator {
    fn propagate(&self, field: &OpticalField, distance: f64) -> Result<OpticalField> {
        if distance <= 0.0 {
            return Err(PhotonicError::invalid_parameter(
                "distance",
                distance,
                "> 0"
            ));
        }
        
        let (ny, nx) = field.amplitude.shape();
        let kernel = self.fresnel_kernel(distance, field.wavelength, nx, ny);
        
        // Apply convolution (simplified as element-wise multiplication)
        let propagated_amplitude = field.amplitude.component_mul(&kernel);
        
        let mut output_field = field.clone();
        output_field.amplitude = propagated_amplitude;
        output_field.power = output_field.calculate_power();
        
        Ok(output_field)
    }
    
    fn set_index_profile(&mut self, _index_profile: DMatrix<Complex64>) {
        // Free space propagation doesn't use index profile
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DVector;
    
    #[test]
    fn test_bpm_creation() {
        let index_profile = DMatrix::from_element(10, 10, Complex64::new(1.5, 0.0));
        let bpm = BeamPropagationMethod::new(index_profile, 1.5, 1e-6, (100e-9, 100e-9));
        
        assert_eq!(bpm.reference_index, 1.5);
        assert_eq!(bpm.step_size, 1e-6);
    }
    
    #[test]
    fn test_transfer_matrix_uniform_waveguide() {
        let length = 100e-6;
        let n_eff = Complex64::new(2.4, 0.0);
        let wavelength = 1550e-9;
        
        let tm = TransferMatrixMethod::uniform_waveguide(length, n_eff, wavelength);
        
        // Test that matrix is 2x2
        assert_eq!(tm.transfer_matrix.shape(), (2, 2));
        
        // Test propagation through waveguide
        let input = DVector::from_vec(vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)]);
        let output = tm.apply_to_amplitudes(&input).unwrap();
        
        // Forward amplitude should have acquired phase
        assert!((output[0].norm() - 1.0).abs() < 1e-10);
        assert!(output[0].arg().abs() > 0.0);
    }
    
    #[test]
    fn test_fresnel_propagator() {
        let propagator = FresnelPropagator::new((100e-9, 100e-9));
        
        // Create simple Gaussian field
        let amplitude = DMatrix::from_fn(10, 10, |i, j| {
            let x = (j as f64 - 5.0) * 100e-9;
            let y = (i as f64 - 5.0) * 100e-9;
            let r_sq = x * x + y * y;
            Complex64::new((-r_sq / (100e-9 * 100e-9)).exp(), 0.0)
        });
        
        let x_coords = DVector::from_fn(10, |i, _| i as f64 * 100e-9);
        let y_coords = DVector::from_fn(10, |i, _| i as f64 * 100e-9);
        
        let field = OpticalField::new(amplitude, 1550e-9, 1e-3, x_coords, y_coords);
        
        let propagated = propagator.propagate(&field, 1e-3).unwrap();
        
        // Check that field was modified
        assert_ne!(field.amplitude, propagated.amplitude);
    }
}