//! Optical coupling calculations and mode overlap integrals

use crate::core::{PhotonicError, Result, Complex64, WaveguideMode};
use nalgebra::{DMatrix, DVector};
use std::f64::consts::PI;

/// Overlap integral calculator for mode coupling
pub struct OverlapIntegral;

impl OverlapIntegral {
    /// Calculate overlap integral between two modes
    pub fn calculate(
        mode1: &WaveguideMode,
        mode2: &WaveguideMode,
        dx: f64,
        dy: f64,
    ) -> Result<Complex64> {
        if mode1.field_profile.shape() != mode2.field_profile.shape() {
            return Err(PhotonicError::invalid_parameter(
                "mode_profiles",
                format!("{}x{} vs {}x{}", 
                    mode1.field_profile.nrows(), mode1.field_profile.ncols(),
                    mode2.field_profile.nrows(), mode2.field_profile.ncols()),
                "same dimensions"
            ));
        }
        
        let mut overlap = Complex64::new(0.0, 0.0);
        
        for i in 0..mode1.field_profile.nrows() {
            for j in 0..mode1.field_profile.ncols() {
                let e1 = mode1.field_profile[(i, j)];
                let e2_conj = mode2.field_profile[(i, j)].conj();
                overlap += e1 * e2_conj * dx * dy;
            }
        }
        
        Ok(overlap)
    }
    
    /// Calculate power coupling coefficient between modes
    pub fn power_coupling_coefficient(
        mode1: &WaveguideMode,
        mode2: &WaveguideMode,
        dx: f64,
        dy: f64,
    ) -> Result<f64> {
        let overlap = Self::calculate(mode1, mode2, dx, dy)?;
        Ok(overlap.norm_sqr())
    }
    
    /// Calculate mode orthogonality
    pub fn orthogonality_check(
        modes: &[WaveguideMode],
        dx: f64,
        dy: f64,
        tolerance: f64,
    ) -> Result<bool> {
        for (i, mode1) in modes.iter().enumerate() {
            for (j, mode2) in modes.iter().enumerate() {
                if i != j {
                    let overlap = Self::calculate(mode1, mode2, dx, dy)?;
                    if overlap.norm() > tolerance {
                        return Ok(false);
                    }
                }
            }
        }
        Ok(true)
    }
}

/// Evanescent coupling calculator for directional couplers
pub struct CouplingCalculator {
    /// Gap between waveguides
    gap: f64,
    /// Wavelength
    wavelength: f64,
    /// Coupling length
    length: f64,
}

impl CouplingCalculator {
    /// Create new coupling calculator
    pub fn new(gap: f64, wavelength: f64, length: f64) -> Self {
        Self {
            gap,
            wavelength,
            length,
        }
    }
    
    /// Calculate coupling coefficient using exponential decay model
    pub fn coupling_coefficient(
        &self,
        mode1: &WaveguideMode,
        mode2: &WaveguideMode,
    ) -> Result<Complex64> {
        // Simplified exponential decay model
        let k0 = 2.0 * PI / self.wavelength;
        let n_eff = (mode1.effective_index + mode2.effective_index) / 2.0;
        
        // Estimate decay constant (simplified)
        let gamma = k0 * (n_eff.norm_sqr() - 1.0).sqrt(); // Assuming cladding index = 1
        
        // Coupling strength decreases exponentially with gap
        let coupling_strength = 0.1 * (-gamma * self.gap).exp(); // Empirical factor
        
        Ok(Complex64::new(coupling_strength, 0.0))
    }
    
    /// Calculate coupling matrix for N-waveguide system
    pub fn coupling_matrix(&self, modes: &[WaveguideMode]) -> Result<DMatrix<Complex64>> {
        let n = modes.len();
        let mut matrix = DMatrix::zeros(n, n);
        
        // Diagonal elements (self-coupling)
        for i in 0..n {
            matrix[(i, i)] = modes[i].beta;
        }
        
        // Off-diagonal elements (mutual coupling)
        for i in 0..n {
            for j in i+1..n {
                let coupling = self.coupling_coefficient(&modes[i], &modes[j])?;
                matrix[(i, j)] = coupling;
                matrix[(j, i)] = coupling.conj(); // Hermitian coupling matrix
            }
        }
        
        Ok(matrix)
    }
    
    /// Calculate power transfer for directional coupler
    pub fn power_transfer_ratio(&self, delta_beta: f64) -> f64 {
        let kappa = 0.1; // Simplified coupling coefficient
        let length = self.length;
        
        let gamma = (kappa * kappa + (delta_beta / 2.0) * (delta_beta / 2.0)).sqrt();
        let coupling_strength = (kappa / gamma).powi(2);
        
        coupling_strength * (gamma * length).sin().powi(2)
    }
}

/// Ring resonator coupling calculator
pub struct RingCouplingCalculator {
    /// Ring radius
    radius: f64,
    /// Coupling gap
    gap: f64,
    /// Coupling length
    coupling_length: f64,
}

impl RingCouplingCalculator {
    /// Create new ring coupling calculator
    pub fn new(radius: f64, gap: f64, coupling_length: f64) -> Self {
        Self {
            radius,
            gap,
            coupling_length,
        }
    }
    
    /// Calculate field coupling coefficient
    pub fn field_coupling_coefficient(
        &self,
        _bus_mode: &WaveguideMode,
        _ring_mode: &WaveguideMode,
    ) -> Result<Complex64> {
        // Simplified coupling calculation for ring resonator
        let overlap_area = self.coupling_length * (-self.gap / 100e-9).exp(); // Exponential decay
        
        // Coupling strength based on mode overlap and geometry
        let coupling_strength = 0.1 * overlap_area / (self.radius * PI); // Normalized by ring circumference
        
        Ok(Complex64::new(coupling_strength, 0.0))
    }
    
    /// Calculate transmission and drop port responses
    pub fn transmission_response(
        &self,
        wavelengths: &DVector<f64>,
        bus_mode: &WaveguideMode,
        ring_mode: &WaveguideMode,
        ring_loss: f64,
    ) -> Result<(DVector<Complex64>, DVector<Complex64>)> {
        let t = self.field_coupling_coefficient(bus_mode, ring_mode)?;
        let kappa = t.norm();
        let transmission_coeff = (1.0 - kappa * kappa).sqrt();
        
        let mut transmission = DVector::zeros(wavelengths.len());
        let mut drop = DVector::zeros(wavelengths.len());
        
        for (i, &wavelength) in wavelengths.iter().enumerate() {
            // Ring round-trip phase
            let k0 = 2.0 * PI / wavelength;
            let beta = k0 * ring_mode.effective_index.re;
            let round_trip_phase = beta * 2.0 * PI * self.radius;
            
            // Loss factor
            let loss_factor = (-ring_loss * PI * self.radius).exp();
            
            // Ring reflection coefficient
            let r = Complex64::new(loss_factor, 0.0) * 
                   Complex64::new(0.0, round_trip_phase).exp();
            
            // Transmission and drop calculations
            let denominator = Complex64::new(1.0, 0.0) - 
                            Complex64::new(transmission_coeff * r.re, transmission_coeff * r.im);
            
            let t_through = (Complex64::new(transmission_coeff, 0.0) - r) / denominator;
            let t_drop = Complex64::new(kappa, 0.0) * 
                        (Complex64::new(1.0, 0.0) - r) / denominator;
            
            transmission[i] = t_through;
            drop[i] = t_drop;
        }
        
        Ok((transmission, drop))
    }
    
    /// Calculate quality factor
    pub fn quality_factor(
        &self,
        resonance_wavelength: f64,
        ring_mode: &WaveguideMode,
        total_loss: f64,
    ) -> f64 {
        let _k0 = 2.0 * PI / resonance_wavelength;
        let n_g = ring_mode.effective_index.re; // Simplified group index
        let circumference = 2.0 * PI * self.radius;
        
        // Free spectral range
        let fsr = resonance_wavelength * resonance_wavelength / (n_g * circumference);
        
        // Quality factor from losses
        let finesse = PI / (2.0 * total_loss);
        fsr * finesse / resonance_wavelength
    }
}

/// Grating coupler for fiber-chip coupling
pub struct GratingCoupler {
    /// Grating period
    period: f64,
    /// Duty cycle
    duty_cycle: f64,
    /// Number of periods
    num_periods: usize,
    /// Etch depth
    etch_depth: f64,
}

impl GratingCoupler {
    /// Create new grating coupler
    pub fn new(period: f64, duty_cycle: f64, num_periods: usize, etch_depth: f64) -> Self {
        Self {
            period,
            duty_cycle,
            num_periods,
            etch_depth,
        }
    }
    
    /// Calculate coupling efficiency to fiber mode
    pub fn coupling_efficiency(
        &self,
        wavelength: f64,
        waveguide_mode: &WaveguideMode,
        _fiber_mode_field_diameter: f64,
    ) -> Result<f64> {
        // Simplified Bragg condition check
        let _k0 = 2.0 * PI / wavelength;
        let n_eff = waveguide_mode.effective_index.re;
        let fiber_angle = 8.0_f64.to_radians(); // Typical fiber angle
        
        // Phase matching condition
        let bragg_wavelength = self.period * (n_eff - fiber_angle.sin());
        let wavelength_detuning = (wavelength - bragg_wavelength) / bragg_wavelength;
        
        // Coupling strength based on overlap and grating parameters
        let grating_strength = self.etch_depth / wavelength; // Normalized etch depth
        let mode_overlap = 0.8; // Simplified overlap factor
        
        // Gaussian envelope from finite grating length
        let length = self.num_periods as f64 * self.period;
        let effective_length = length * 0.7; // Effective interaction length
        
        let envelope = (-PI * wavelength_detuning.powi(2) * effective_length / wavelength).exp();
        
        let efficiency = mode_overlap * grating_strength * envelope;
        Ok(efficiency.min(1.0)) // Cap at 100%
    }
    
    /// Calculate bandwidth
    pub fn bandwidth(&self, wavelength: f64) -> f64 {
        let length = self.num_periods as f64 * self.period;
        // Bandwidth inversely proportional to grating length
        wavelength / (2.0 * length / wavelength)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DMatrix;
    
    fn create_test_mode(effective_index: f64) -> WaveguideMode {
        let field_profile = DMatrix::from_fn(10, 10, |i, j| {
            let x = (j as f64 - 5.0) * 0.1;
            let y = (i as f64 - 5.0) * 0.1;
            let r_sq = x * x + y * y;
            Complex64::new((-r_sq).exp(), 0.0)
        });
        
        WaveguideMode::new(
            Complex64::new(effective_index, 0.0),
            field_profile,
            0,
            1550e-9,
        )
    }
    
    #[test]
    fn test_overlap_integral() {
        let mode1 = create_test_mode(2.4);
        let mode2 = create_test_mode(2.4);
        
        let overlap = OverlapIntegral::calculate(&mode1, &mode2, 0.1, 0.1).unwrap();
        
        // Self-overlap should be close to 1 for normalized modes
        assert!(overlap.norm() > 0.8);
    }
    
    #[test]
    fn test_coupling_calculator() {
        let calculator = CouplingCalculator::new(200e-9, 1550e-9, 100e-6);
        
        let mode1 = create_test_mode(2.4);
        let mode2 = create_test_mode(2.4);
        
        let coupling = calculator.coupling_coefficient(&mode1, &mode2).unwrap();
        
        // Coupling should be positive and less than propagation constant
        assert!(coupling.norm() > 0.0);
        assert!(coupling.norm() < mode1.beta.norm());
    }
    
    #[test]
    fn test_ring_coupling() {
        let ring_calc = RingCouplingCalculator::new(10e-6, 200e-9, 2e-6);
        
        let bus_mode = create_test_mode(2.4);
        let ring_mode = create_test_mode(2.4);
        
        let coupling = ring_calc.field_coupling_coefficient(&bus_mode, &ring_mode).unwrap();
        
        assert!(coupling.norm() > 0.0);
        assert!(coupling.norm() < 1.0);
    }
    
    #[test]
    fn test_grating_coupler() {
        let grating = GratingCoupler::new(630e-9, 0.5, 20, 70e-9);
        
        let mode = create_test_mode(2.4);
        let efficiency = grating.coupling_efficiency(1550e-9, &mode, 10e-6).unwrap();
        
        assert!(efficiency >= 0.0);
        assert!(efficiency <= 1.0);
    }
}