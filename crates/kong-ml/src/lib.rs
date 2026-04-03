pub mod tokenizer;
pub mod config;
pub mod model;
pub mod detector;
pub mod registry;

use candle_core::Device;

pub fn preferred_device() -> Device {
    let device = try_metal_device().unwrap_or(Device::Cpu);

    if std::env::var_os("KONG_ML_LOG_DEVICE").is_some() {
        let backend = match device.location() {
            candle_core::DeviceLocation::Metal { gpu_id } => {
                format!("Metal (gpu_id={gpu_id})")
            }
            other => format!("{other:?}")
        };
        eprintln!("kong-ml: using {backend} device");
    }

    device
}

#[cfg(all(target_os = "macos", feature = "metal"))]
fn try_metal_device() -> Option<Device> {
    Device::new_metal(0).ok()
}

#[cfg(not(all(target_os = "macos", feature = "metal")))]
fn try_metal_device() -> Option<Device> {
    None
}