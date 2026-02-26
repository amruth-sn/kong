"""Configuration management for Kong."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from kong.ghidra.environment import find_ghidra_install


@dataclass
class GhidraConfig:
    install_dir: str | None = None
    project_dir: str = "/tmp/kong_ghidra"
    project_name: str = "kong_project"

    def __post_init__(self) -> None:
        if self.install_dir is None:
            self.install_dir = find_ghidra_install()


@dataclass
class OutputConfig:
    directory: Path = field(default_factory=lambda: Path("./kong_output"))
    formats: list[str] = field(default_factory=lambda: ["source", "json", "ghidra"])


@dataclass
class KongConfig:
    ghidra: GhidraConfig = field(default_factory=GhidraConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    headless: bool = False
    verbose: bool = False
