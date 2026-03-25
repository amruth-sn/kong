use serde::{Deserialize, Serialize};
use strum_macros::AsRefStr;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, AsRefStr)]
#[serde(rename_all = "snake_case")]
#[strum(serialize_all = "snake_case")]
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

impl std::fmt::Display for Arch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_ref())
    }
}