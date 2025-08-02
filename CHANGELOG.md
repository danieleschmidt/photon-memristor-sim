# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial project foundation and documentation
- Project charter defining scope, success criteria, and stakeholder alignment
- Comprehensive architecture documentation with system design and data flow
- Development roadmap with versioned milestones through v1.0+
- Architecture Decision Records (ADR) structure with initial templates
- Community files: Contributing guidelines, Security policy, Code of Conduct

### Changed
- Enhanced README.md with comprehensive project overview and examples

### Deprecated
- Nothing deprecated yet

### Removed
- Nothing removed yet

### Fixed
- Nothing fixed yet

### Security
- Initial security policy and vulnerability reporting procedures established

## [0.1.0] - 2025-01-XX

### Added
- Initial Rust core simulation engine architecture
- Python bindings with maturin integration
- Basic waveguide propagation models
- Initial PCM (Phase Change Material) device model for Ge2Sb2Te5
- JAX integration for automatic differentiation
- Transfer matrix method for linear device simulation
- Core data structures and Python API interfaces
- Basic examples and documentation
- Project structure with Cargo.toml configuration

### Technical Details
- Rust-based core with Python bindings via PyO3
- WASM support for browser-based simulation
- JAX custom_vjp integration for gradient computation
- Modular architecture supporting device plugins
- Multi-threading support with Rayon
- Memory-efficient data structures with nalgebra

---

## Versioning Strategy

We use semantic versioning (MAJOR.MINOR.PATCH) where:
- **MAJOR**: Incompatible API changes or architectural overhauls
- **MINOR**: New functionality added in a backwards compatible manner
- **PATCH**: Backwards compatible bug fixes and performance improvements

## Release Schedule

- **Major releases**: Annually with significant capability expansion
- **Minor releases**: Quarterly with new features and device models  
- **Patch releases**: Monthly with bug fixes and optimizations
- **Hotfix releases**: As needed for critical security or stability issues

## Change Categories

- **Added**: New features, device models, simulation methods
- **Changed**: Changes in existing functionality or APIs
- **Deprecated**: Soon-to-be removed features (marked for future removal)
- **Removed**: Features removed in this version
- **Fixed**: Bug fixes and corrections
- **Security**: Vulnerability fixes and security improvements

## Contributing to Changelog

When contributing, please:
1. Add entries under `[Unreleased]` section
2. Use appropriate category (Added, Changed, etc.)
3. Write clear, concise descriptions
4. Include breaking changes with migration notes
5. Reference issue/PR numbers where applicable

Example entry format:
```
### Added
- New PCM device model for GSST materials (#123)
- FDTD solver with PML boundaries (#456)
  - Breaking: Old BPM solver API changed, see migration guide
```