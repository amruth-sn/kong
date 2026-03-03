"""Data types for Ghidra objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class FunctionClassification(Enum):
    IMPORTED = "imported"
    THUNK = "thunk"
    TRIVIAL = "trivial"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


@dataclass
class FunctionInfo:
    address: int
    name: str
    size: int
    is_thunk: bool = False
    params: list[ParameterInfo] = field(default_factory=list)
    return_type: str = "undefined"
    local_vars: list[VariableInfo] = field(default_factory=list)
    calling_convention: str = "unknown"
    classification: FunctionClassification | None = None

    @property
    def address_hex(self) -> str:
        return f"0x{self.address:08x}"


@dataclass
class ParameterInfo:
    name: str
    data_type: str
    ordinal: int
    size: int = 0


@dataclass
class VariableInfo:
    name: str
    data_type: str
    size: int = 0
    stack_offset: int | None = None


@dataclass
class XRef:
    from_addr: int
    to_addr: int
    ref_type: str = "unknown"

    @property
    def from_hex(self) -> str:
        return f"0x{self.from_addr:08x}"

    @property
    def to_hex(self) -> str:
        return f"0x{self.to_addr:08x}"


@dataclass
class StringEntry:
    address: int
    value: str
    length: int = 0
    xref_addrs: list[int] = field(default_factory=list)

    @property
    def address_hex(self) -> str:
        return f"0x{self.address:08x}"


@dataclass
class BinaryInfo:
    arch: str
    format: str
    endianness: str
    word_size: int
    compiler: str = "unknown"
    name: str = ""
    path: str = ""
    min_address: int = 0
    max_address: int = 0


@dataclass
class StructField:
    name: str
    data_type: str
    offset: int
    size: int

    @property
    def end_offset(self) -> int:
        return self.offset + self.size


@dataclass
class StructDefinition:
    name: str
    size: int
    fields: list[StructField] = field(default_factory=list)

    @property
    def field_count(self) -> int:
        return len(self.fields)

    def field_at_offset(self, offset: int) -> StructField | None:
        for f in self.fields:
            if f.offset == offset:
                return f
        return None


@dataclass
class PcodeOp:
    """A single pcode operation within a basic block."""
    mnemonic: str
    address: int
    inputs: list[str] = field(default_factory=list)
    output: str = ""


@dataclass
class BasicBlock:
    """A basic block within a function's control flow graph."""
    start_addr: int
    end_addr: int
    instructions: list[str] = field(default_factory=list)
    pcode_ops: list[PcodeOp] = field(default_factory=list)

    @property
    def start_hex(self) -> str:
        return f"0x{self.start_addr:08x}"

    @property
    def end_hex(self) -> str:
        return f"0x{self.end_addr:08x}"


class BlockEdgeType(Enum):
    FALL_THROUGH = "fall_through"
    BRANCH = "branch"
    CALL = "call"
    COMPUTED = "computed"
    UNKNOWN = "unknown"


@dataclass
class BlockEdge:
    """A directed edge in the control flow graph."""
    from_addr: int
    to_addr: int
    edge_type: BlockEdgeType = BlockEdgeType.UNKNOWN


@dataclass
class ControlFlowGraph:
    """Control flow graph for a single function."""
    function_addr: int
    blocks: list[BasicBlock] = field(default_factory=list)
    edges: list[BlockEdge] = field(default_factory=list)

    @property
    def block_count(self) -> int:
        return len(self.blocks)

    def block_at(self, addr: int) -> BasicBlock | None:
        for b in self.blocks:
            if b.start_addr == addr:
                return b
        return None

    def successors(self, addr: int) -> list[int]:
        return [e.to_addr for e in self.edges if e.from_addr == addr]

    def predecessors(self, addr: int) -> list[int]:
        return [e.from_addr for e in self.edges if e.to_addr == addr]
