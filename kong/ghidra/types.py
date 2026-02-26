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
