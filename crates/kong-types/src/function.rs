use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionBoundary {
    pub start: u64,
    pub end: u64,
    pub confidence: f32,
    pub source: DetectionSource,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DetectionSource {
    Symbol,         // from symbol table
    Heuristic,      // from prologue/call-target analysis
    Ml,             // from XDA model
    ExceptionInfo,  // from .eh_frame / .pdata
}

#[derive(Debug, Clone)]
pub struct AmbiguousRegion {
    pub start: u64,
    pub bytes: Vec<u8>,
    pub reason: AmbiguityReason,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AmbiguityReason {
    NoPrologue,       // reachable but no recognizable prologue
    IndirectTarget,   // suspected indirect call/jump target
    GapRegion,        // unreached .text gap between known functions
    OptimizedCode,    // code region with non-standard patterns
}