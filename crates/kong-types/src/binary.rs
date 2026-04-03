use serde::{Deserialize, Serialize};
use strum_macros::IntoStaticStr;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, IntoStaticStr)]
#[serde(rename_all = "snake_case")]
#[strum(serialize_all = "snake_case")]
pub enum Arch {
    X86_64,
    Aarch64,
    Arm,
    Riscv64,
}

impl Arch {
    // we use strum as_ref() to get the string representation of the enum, NOT a match statement
    pub fn model_name(&self) -> &'static str {
        self.into()
    }
}

impl std::fmt::Display for Arch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.model_name())
    }
}