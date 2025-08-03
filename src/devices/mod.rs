//! Device models for photonic components

pub mod pcm;
pub mod oxide;
pub mod ring_resonator;
pub mod mzi;
pub mod traits;

pub use pcm::{PCMDevice, PCMMaterial, CrystallizationModel};
pub use oxide::{OxideMemristor, OxideType, FilamentaryModel};
pub use ring_resonator::{MicroringResonator, RingGeometry};
pub use mzi::{MachZehnderInterferometer, MZIConfiguration};
pub use traits::{PhotonicDevice, MemristiveDevice, TunableDevice};