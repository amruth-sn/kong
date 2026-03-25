use serde::{Deserialize, Serialize};
use serde_json::to_string;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Arch {
    X86_64,
    X86,
    Aarch64,
    Arm,
    Mips,
    Mips64,
    Riscv64,
    PowerPc,
}

impl Arch {
    pub fn model_name(&self) -> String {
        to_string(self).unwrap()
    }
}

impl std::fmt::Display for Arch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.model_name())
    }
}