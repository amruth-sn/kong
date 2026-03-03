"""Signature db for matching known library functions.

Loads JSON signature files and matches function names/aliases against them.
This matches Ghidra's auto-detected import names and symbol names against known signatures.
Constants-based and pattern-based matching is handled by the LLM during analysis.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kong.ghidra.types import FunctionInfo

logger = logging.getLogger(__name__)

SIGNATURES_DIR = Path(__file__).parent.parent / "signatures"


@dataclass
class SignatureEntry:
    name: str
    aliases: list[str] = field(default_factory=list)
    description: str = ""
    signature: str = ""
    category: str = ""  # e.g., "stdlib", "crypto"
    indicators: dict[str, str | list[str]] = field(default_factory=dict)


@dataclass
class SignatureMatch:
    """A match between a function and a known signature."""
    function_address: int
    function_name: str
    matched_name: str  # canonical name from DB
    signature: str
    category: str
    description: str


class SignatureDB:
    """In-memory signature database loaded from JSON files.

    Provides fast name-based lookup via a normalized name index.
    """

    def __init__(self) -> None:
        self._entries: list[SignatureEntry] = []
        self._name_index: dict[str, SignatureEntry] = {}  # normalized name -> entry

    @staticmethod
    def _normalize(name: str) -> str:
        """Normalize a function name for matching.

        Strips leading underscores and lowercases, so that
        __memcpy, _memcpy, MEMCPY all match memcpy.
        """
        return name.lstrip("_").lower()

    def load_file(self, path: Path, category: str = "") -> int:
        """Load signatures from a JSON file. Returns count loaded."""
        if not path.exists():
            logger.warning("Signature file not found: %s", path)
            return 0

        with open(path) as f:
            data = json.load(f)

        count = 0
        for item in data:
            entry = SignatureEntry(
                name=item["name"],
                aliases=item.get("aliases", []),
                description=item.get("description", ""),
                signature=item.get("signature", ""),
                category=category or path.stem,
                indicators=item.get("indicators", {}),
            )
            self._entries.append(entry)

            # Index by normalized canonical name and all aliases
            for name in [entry.name] + entry.aliases:
                self._name_index[self._normalize(name)] = entry
            count += 1

        return count

    def load_directory(self, directory: Path | None = None) -> int:
        """Load all .json files from a directory. Returns total count."""
        if directory is None:
            directory = SIGNATURES_DIR

        if not directory.is_dir():
            logger.warning("Signatures directory not found: %s", directory)
            return 0

        total = 0
        for path in sorted(directory.glob("*.json")):
            loaded = self.load_file(path, category=path.stem)
            logger.info("Loaded %d signatures from %s", loaded, path.name)
            total += loaded

        return total

    def lookup(self, name: str) -> SignatureEntry | None:
        """Look up a function name in the database."""
        return self._name_index.get(self._normalize(name))

    def match_functions(
        self,
        functions: list[FunctionInfo],
    ) -> list[SignatureMatch]:
        """Match a list of functions against the signature database.

        Returns matches for functions whose names (as detected by Ghidra)
        match known signatures. This catches imported functions that Ghidra
        already identified by name.
        """
        matches = []
        for func in functions:
            entry = self.lookup(func.name)
            if entry is not None:
                matches.append(SignatureMatch(
                    function_address=func.address,
                    function_name=func.name,
                    matched_name=entry.name,
                    signature=entry.signature,
                    category=entry.category,
                    description=entry.description,
                ))
        return matches

    @property
    def size(self) -> int:
        return len(self._entries)

    @property
    def entries(self) -> list[SignatureEntry]:
        return list(self._entries)
