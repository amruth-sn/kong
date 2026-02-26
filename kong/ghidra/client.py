"""Ghidra client — in-process via PyGhidra/JPype.

Opens a binary directly in the current process using PyGhidra, then calls
Ghidra Java APIs through JPype.  No subprocess, no RPC, no port management.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pyghidra

from kong.ghidra.types import (
    BinaryInfo,
    FunctionClassification,
    FunctionInfo,
    ParameterInfo,
    StringEntry,
    VariableInfo,
    XRef,
)

logger = logging.getLogger(__name__)


class GhidraClientError(Exception):
    """Raised when a Ghidra operation fails."""


def _classify(size: int, is_thunk: bool) -> FunctionClassification:
    if is_thunk:
        return FunctionClassification.THUNK
    if size <= 16:
        return FunctionClassification.TRIVIAL
    if size <= 64:
        return FunctionClassification.SMALL
    if size <= 256:
        return FunctionClassification.MEDIUM
    return FunctionClassification.LARGE


class GhidraClient:
    """In-process Ghidra client via PyGhidra.

    Usage::

        with GhidraClient("/path/to/binary") as client:
            funcs = client.list_functions()
            decomp = client.get_decompilation(funcs[0].address)
    """

    def __init__(self, binary_path: str, install_dir: str | None = None) -> None:
        self.binary_path = str(Path(binary_path).resolve())
        if install_dir is None:
            from kong.ghidra.environment import find_ghidra_install
            install_dir = find_ghidra_install()
        self.install_dir = install_dir
        self._program: Any = None
        self._flat_api: Any = None
        self._ctx: Any = None  # context manager from open_program

    @property
    def program(self) -> Any:
        if self._program is None:
            raise GhidraClientError("Not open. Call open() first.")
        return self._program

    @property
    def flat_api(self) -> Any:
        if self._flat_api is None:
            raise GhidraClientError("Not open. Call open() first.")
        return self._flat_api

    def open(self) -> GhidraClient:
        """Start PyGhidra JVM and open the binary for analysis."""
        if not Path(self.binary_path).exists():
            raise GhidraClientError(f"Binary not found: {self.binary_path}")

        logger.info("Starting PyGhidra JVM ...")
        pyghidra.start(install_dir=self.install_dir)

        logger.info("Opening program: %s", self.binary_path)
        self._ctx = pyghidra.open_program(self.binary_path)
        self._flat_api = self._ctx.__enter__()
        self._program = self._flat_api.getCurrentProgram()
        logger.info("Program loaded: %s", self._program.getName())
        return self

    def close(self) -> None:
        """Close the program and release resources."""
        if self._ctx is not None:
            try:
                self._ctx.__exit__(None, None, None)
            except Exception:
                logger.debug("Error closing program context", exc_info=True)
            self._ctx = None
        self._program = None
        self._flat_api = None

    def __enter__(self) -> GhidraClient:
        return self.open()

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Query methods
    # ------------------------------------------------------------------

    def get_binary_info(self) -> BinaryInfo:
        """Get metadata about the loaded binary."""
        prog = self.program
        lang = prog.getLanguage()
        return BinaryInfo(
            name=str(prog.getName()),
            path=str(prog.getExecutablePath()),
            arch=str(lang.getProcessor().toString()),
            endianness="big" if lang.isBigEndian() else "little",
            word_size=int(prog.getDefaultPointerSize()),
            format=str(prog.getExecutableFormat()),
            compiler=str(prog.getCompilerSpec().getCompilerSpecID()),
            min_address=int(prog.getMinAddress().getOffset()),
            max_address=int(prog.getMaxAddress().getOffset()),
        )

    def list_functions(self) -> list[FunctionInfo]:
        """List all functions in the binary."""
        fm = self.program.getFunctionManager()
        functions: list[FunctionInfo] = []
        for func in fm.getFunctions(True):
            size = int(func.getBody().getNumAddresses())
            is_thunk = bool(func.isThunk())
            functions.append(
                FunctionInfo(
                    address=int(func.getEntryPoint().getOffset()),
                    name=str(func.getName()),
                    size=size,
                    is_thunk=is_thunk,
                    classification=_classify(size, is_thunk),
                )
            )
        return functions

    def get_function_info(self, addr: int) -> FunctionInfo:
        """Get detailed information about a specific function."""
        func = self._get_function_at(addr)
        size = int(func.getBody().getNumAddresses())
        is_thunk = bool(func.isThunk())

        params = [
            ParameterInfo(
                name=str(p.getName()),
                data_type=str(p.getDataType().getDisplayName()),
                ordinal=int(p.getOrdinal()),
                size=int(p.getLength()),
            )
            for p in func.getParameters()
        ]
        local_vars = [
            VariableInfo(
                name=str(v.getName()),
                data_type=str(v.getDataType().getDisplayName()),
                size=int(v.getLength()),
                stack_offset=int(v.getStackOffset()) if v.isStackVariable() else None,
            )
            for v in func.getLocalVariables()
        ]

        return FunctionInfo(
            address=int(func.getEntryPoint().getOffset()),
            name=str(func.getName()),
            size=size,
            is_thunk=is_thunk,
            params=params,
            return_type=str(func.getReturnType().getDisplayName()),
            local_vars=local_vars,
            calling_convention=str(func.getCallingConventionName()),
            classification=_classify(size, is_thunk),
        )

    def get_decompilation(self, addr: int) -> str:
        """Get the decompiled C source for a function."""
        from ghidra.app.decompiler import DecompInterface
        from ghidra.util.task import ConsoleTaskMonitor

        func = self._get_function_at(addr)
        di = DecompInterface()
        try:
            di.openProgram(self.program)
            result = di.decompileFunction(func, 30, ConsoleTaskMonitor())
            if not result.decompileCompleted():
                raise GhidraClientError(f"Decompilation failed for function at 0x{addr:08x}")
            decomp_func = result.getDecompiledFunction()
            if decomp_func is None:
                raise GhidraClientError(f"Decompilation failed for function at 0x{addr:08x}")
            return str(decomp_func.getC())
        finally:
            di.dispose()

    def get_xrefs_to(self, addr: int) -> list[XRef]:
        """Get all cross-references TO a given address."""
        target = self._to_addr(addr)
        refs = self.flat_api.getReferencesTo(target)
        return [
            XRef(
                from_addr=int(ref.getFromAddress().getOffset()),
                to_addr=int(ref.getToAddress().getOffset()),
                ref_type=str(ref.getReferenceType().getName()),
            )
            for ref in refs
        ]

    def get_xrefs_from(self, addr: int) -> list[XRef]:
        """Get all cross-references FROM an address."""
        source = self._to_addr(addr)
        refs = self.program.getReferenceManager().getReferencesFrom(source)
        return [
            XRef(
                from_addr=int(ref.getFromAddress().getOffset()),
                to_addr=int(ref.getToAddress().getOffset()),
                ref_type=str(ref.getReferenceType().getName()),
            )
            for ref in refs
        ]

    def get_callers(self, addr: int) -> list[int]:
        """Get addresses of functions that call the function at addr."""
        target = self._to_addr(addr)
        refs = self.flat_api.getReferencesTo(target)
        return list({
            int(ref.getFromAddress().getOffset())
            for ref in refs
            if ref.getReferenceType().isCall()
        })

    def get_callees(self, addr: int) -> list[int]:
        """Get addresses of functions called by the function at addr."""
        source = self._to_addr(addr)
        refs = self.program.getReferenceManager().getReferencesFrom(source)
        return list({
            int(ref.getToAddress().getOffset())
            for ref in refs
            if ref.getReferenceType().isCall()
        })

    def get_strings(self) -> list[StringEntry]:
        """Get all defined strings in the binary."""
        prog = self.program
        listing = prog.getListing()
        entries: list[StringEntry] = []
        for data in listing.getDefinedData(True):
            dt_name = str(data.getDataType().getName()).lower()
            if "string" not in dt_name:
                continue
            addr = data.getAddress()
            addr_offset = int(addr.getOffset())
            value = data.getValue()
            xrefs = [
                int(ref.getFromAddress().getOffset())
                for ref in self.flat_api.getReferencesTo(addr)
            ]
            entries.append(
                StringEntry(
                    address=addr_offset,
                    value=str(value) if value is not None else "",
                    length=int(data.getLength()),
                    xref_addrs=xrefs,
                )
            )
        return entries

    # ------------------------------------------------------------------
    # Mutation methods — wrapped in transactions
    # ------------------------------------------------------------------

    def rename_function(self, addr: int, new_name: str) -> None:
        """Rename a function at the given address."""
        func = self._get_function_at(addr)
        from ghidra.program.model.symbol import SourceType
        tx = self.program.startTransaction("rename_function")
        try:
            func.setName(new_name, SourceType.USER_DEFINED)
        finally:
            self.program.endTransaction(tx, True)
        logger.info("Renamed function at 0x%08x to '%s'", addr, new_name)

    def set_function_signature(self, addr: int, signature_str: str) -> None:
        """Set a function's full signature from a C-style string."""
        from ghidra.app.cmd.function import ApplyFunctionSignatureCmd
        from ghidra.app.util.parser import FunctionSignatureParser
        from ghidra.program.model.symbol import SourceType

        dtm = self.program.getDataTypeManager()
        parser = FunctionSignatureParser(dtm, None)
        func_def = parser.parse(None, signature_str)
        cmd = ApplyFunctionSignatureCmd(
            self._to_addr(addr),
            func_def,
            SourceType.USER_DEFINED,
        )
        tx = self.program.startTransaction("set_function_signature")
        try:
            cmd.applyTo(self.program)
        finally:
            self.program.endTransaction(tx, True)
        logger.info("Set signature at 0x%08x to '%s'", addr, signature_str)

    def add_comment(
        self,
        addr: int,
        comment: str,
        comment_type: str = "plate",
    ) -> None:
        """Add a comment to an address.

        Args:
            addr: Address to comment.
            comment: Comment text.
            comment_type: One of "plate", "pre", "post", "eol", "repeatable".
        """
        from ghidra.program.model.listing import CodeUnit

        type_map = {
            "plate": CodeUnit.PLATE_COMMENT,
            "pre": CodeUnit.PRE_COMMENT,
            "post": CodeUnit.POST_COMMENT,
            "eol": CodeUnit.EOL_COMMENT,
            "repeatable": CodeUnit.REPEATABLE_COMMENT,
        }
        if comment_type not in type_map:
            raise ValueError(
                f"Invalid comment_type: {comment_type}. "
                f"Use one of {list(type_map.keys())}"
            )

        code_unit = self.program.getListing().getCodeUnitAt(self._to_addr(addr))
        tx = self.program.startTransaction("add_comment")
        try:
            code_unit.setComment(type_map[comment_type], comment)
        finally:
            self.program.endTransaction(tx, True)
        logger.info("Added %s comment at 0x%08x", comment_type, addr)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _to_addr(self, addr: int) -> Any:
        """Convert an integer offset to a Ghidra Address."""
        return self.program.getAddressFactory().getDefaultAddressSpace().getAddress(addr)

    def _get_function_at(self, addr: int) -> Any:
        """Get Ghidra Function object at an address."""
        func = self.program.getFunctionManager().getFunctionAt(self._to_addr(addr))
        if func is None:
            raise GhidraClientError(f"No function found at 0x{addr:08x}")
        return func
