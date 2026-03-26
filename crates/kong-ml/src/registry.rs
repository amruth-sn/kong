use std::path::{Path, PathBuf};

use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use kong_types::binary::Arch;

use crate::config::XdaConfig;
use crate::detector::XdaDetector;
use crate::model::XdaModel;

pub struct ModelRegistry {
    cache_dir: PathBuf,
    confidence_threshold: f32,
}

impl ModelRegistry {
    pub fn new(confidence_threshold: f32) -> Self {
        let cache_dir = dirs::data_local_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("kong")
            .join("models");

        Self {
            cache_dir,
            confidence_threshold,
        }
    }

    pub fn with_cache_dir(cache_dir: PathBuf, confidence_threshold: f32) -> Self {
        Self {
            cache_dir,
            confidence_threshold,
        }
    }

    pub fn has_model(&self, arch: Arch) -> bool {
        self.weights_path(arch).exists() && self.config_path(arch).exists()
    }

    pub fn available_architectures(&self) -> Vec<Arch> {
        let all = [
            Arch::X86_64,
            Arch::Aarch64,
            Arch::Arm,
            Arch::Riscv64,
        ];
        all.into_iter().filter(|a| self.has_model(*a)).collect()
    }

    pub fn load_detector(&self, arch: Arch, device: &Device) -> Result<Option<XdaDetector>, Box<dyn std::error::Error>> {
        if !self.has_model(arch) {
            return Ok(None);
        }

        let config = XdaConfig::from_json(&self.config_path(arch))?;
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[self.weights_path(arch)],
                DType::F32,
                device,
            )?
        };
        let model = XdaModel::load(&config, vb)?;
        let detector = XdaDetector::new(model, device.clone(), self.confidence_threshold);

        Ok(Some(detector))
    }

    pub fn install(&self, arch: Arch, weights_src: &Path, config_src: &Path) -> Result<(), Box<dyn std::error::Error>> {
        std::fs::create_dir_all(&self.cache_dir)?;
        std::fs::copy(weights_src, self.weights_path(arch))?;
        std::fs::copy(config_src, self.config_path(arch))?;
        Ok(())
    }

    fn weights_path(&self, arch: Arch) -> PathBuf {
        self.cache_dir.join(format!("xda_{}.safetensors", arch.model_name()))
    }

    fn config_path(&self, arch: Arch) -> PathBuf {
        self.cache_dir.join(format!("xda_{}_config.json", arch.model_name()))
    }
}