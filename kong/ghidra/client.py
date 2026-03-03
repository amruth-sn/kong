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
    BasicBlock,
    BinaryInfo,
    BlockEdge,
    BlockEdgeType,
    ControlFlowGraph,
    FunctionClassification,
    FunctionInfo,
    ParameterInfo,
    PcodeOp,
    StringEntry,
    StructDefinition,
    StructField,
    VariableInfo,
    XRef,
)
from kong.ghidra.environment import find_ghidra_install


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
    # Type management methods
    # ------------------------------------------------------------------

    def create_struct(self, definition: StructDefinition) -> None:
        """Create a struct data type in Ghidra's DataTypeManager.

        Fields are placed at explicit offsets within the struct. Gaps between
        fields are left as undefined bytes (Ghidra fills them automatically).
        """
        from ghidra.program.model.data import (
            CategoryPath,
            StructureDataType,
        )

        dtm = self.program.getDataTypeManager()
        category = CategoryPath("/kong")

        struct_dt = StructureDataType(category, definition.name, definition.size)
        for fld in definition.fields:
            ghidra_type = self._resolve_data_type(fld.data_type, fld.size)
            struct_dt.replaceAtOffset(fld.offset, ghidra_type, fld.size, fld.name, None)

        tx = self.program.startTransaction("create_struct")
        try:
            dtm.addDataType(struct_dt, None)
        finally:
            self.program.endTransaction(tx, True)
        logger.info("Created struct '%s' (%d bytes, %d fields)", definition.name, definition.size, definition.field_count)

    def apply_type_to_param(
        self,
        func_addr: int,
        param_ordinal: int,
        type_name: str,
        as_pointer: bool = True,
    ) -> None:
        """Apply a data type (optionally as a pointer) to a function parameter."""
        from ghidra.program.model.data import PointerDataType
        from ghidra.program.model.symbol import SourceType

        func = self._get_function_at(func_addr)
        params = func.getParameters()
        if param_ordinal >= len(params):
            raise GhidraClientError(
                f"Parameter ordinal {param_ordinal} out of range for "
                f"function at 0x{func_addr:08x} ({len(params)} params)"
            )

        resolved = self._lookup_type(type_name)
        if resolved is None:
            raise GhidraClientError(f"Type '{type_name}' not found in DataTypeManager")

        target_type = PointerDataType(resolved) if as_pointer else resolved

        tx = self.program.startTransaction("apply_type_to_param")
        try:
            param = params[param_ordinal]
            param.setDataType(target_type, SourceType.USER_DEFINED)
        finally:
            self.program.endTransaction(tx, True)
        logger.info(
            "Applied type '%s%s' to param %d of function at 0x%08x",
            type_name, " *" if as_pointer else "", param_ordinal, func_addr,
        )

    def list_custom_types(self) -> list[StructDefinition]:
        """List all struct types in the /kong category."""
        from ghidra.program.model.data import CategoryPath, Structure

        dtm = self.program.getDataTypeManager()
        category = dtm.getCategory(CategoryPath("/kong"))
        if category is None:
            return []

        results: list[StructDefinition] = []
        for dt in category.getDataTypes():
            if not isinstance(dt, Structure):
                continue
            fields = []
            for i in range(dt.getNumDefinedComponents()):
                comp = dt.getComponent(i)
                fields.append(StructField(
                    name=str(comp.getFieldName() or f"field_{i}"),
                    data_type=str(comp.getDataType().getDisplayName()),
                    offset=int(comp.getOffset()),
                    size=int(comp.getLength()),
                ))
            results.append(StructDefinition(
                name=str(dt.getName()),
                size=int(dt.getLength()),
                fields=fields,
            ))
        return results

    def get_type(self, name: str) -> StructDefinition | None:
        """Look up a struct type by name from the /kong category."""
        from ghidra.program.model.data import Structure

        resolved = self._lookup_type(name)
        if resolved is None or not isinstance(resolved, Structure):
            return None

        fields = []
        for i in range(resolved.getNumDefinedComponents()):
            comp = resolved.getComponent(i)
            fields.append(StructField(
                name=str(comp.getFieldName() or f"field_{i}"),
                data_type=str(comp.getDataType().getDisplayName()),
                offset=int(comp.getOffset()),
                size=int(comp.getLength()),
            ))
        return StructDefinition(
            name=str(resolved.getName()),
            size=int(resolved.getLength()),
            fields=fields,
        )

    def _lookup_type(self, name: str) -> Any:
        """Search for a data type by name, checking /kong category first."""
        from ghidra.program.model.data import CategoryPath
        from java.util import ArrayList

        dtm = self.program.getDataTypeManager()
        category = dtm.getCategory(CategoryPath("/kong"))
        if category is not None:
            for dt in category.getDataTypes():
                if str(dt.getName()) == name:
                    return dt

        results = ArrayList()
        dtm.findDataTypes(name, results)
        return results.get(0) if results.size() > 0 else None

    def _resolve_data_type(self, type_name: str, size: int) -> Any:
        """Resolve a C type name to a Ghidra DataType, falling back to sized defaults."""
        from ghidra.program.model.data import (
            ByteDataType,
            CharDataType,
            IntegerDataType,
            LongDataType,
            LongLongDataType,
            PointerDataType,
            ShortDataType,
            UnsignedIntegerDataType,
            UnsignedLongDataType,
            UnsignedLongLongDataType,
            UnsignedShortDataType,
            Undefined1DataType,
            Undefined2DataType,
            Undefined4DataType,
            Undefined8DataType,
        )

        _BUILTIN: dict[str, Any] = {
            "byte": ByteDataType.dataType,
            "char": CharDataType.dataType,
            "short": ShortDataType.dataType,
            "int": IntegerDataType.dataType,
            "long": LongDataType.dataType,
            "long long": LongLongDataType.dataType,
            "uint8_t": ByteDataType.dataType,
            "uint16_t": UnsignedShortDataType.dataType,
            "uint32_t": UnsignedIntegerDataType.dataType,
            "uint64_t": UnsignedLongLongDataType.dataType,
            "int8_t": CharDataType.dataType,
            "int16_t": ShortDataType.dataType,
            "int32_t": IntegerDataType.dataType,
            "int64_t": LongLongDataType.dataType,
            "unsigned int": UnsignedIntegerDataType.dataType,
            "unsigned long": UnsignedLongDataType.dataType,
            "unsigned short": UnsignedShortDataType.dataType,
        }

        if type_name.endswith("*"):
            base_name = type_name.rstrip("* ").strip()
            base_type = self._resolve_data_type(base_name, self.program.getDefaultPointerSize())
            return PointerDataType(base_type)

        low = type_name.lower().strip()
        if low in _BUILTIN:
            return _BUILTIN[low]

        found = self._lookup_type(type_name)
        if found is not None:
            return found

        _SIZED_DEFAULTS: dict[int, Any] = {
            1: Undefined1DataType.dataType,
            2: Undefined2DataType.dataType,
            4: Undefined4DataType.dataType,
            8: Undefined8DataType.dataType,
        }
        return _SIZED_DEFAULTS.get(size, Undefined4DataType.dataType)

    # ------------------------------------------------------------------
    # Control flow graph methods
    # ------------------------------------------------------------------

    def get_basic_blocks(self, addr: int) -> list[BasicBlock]:
        """Get basic blocks for the function at addr."""
        from ghidra.program.model.block import SimpleBlockModel
        from ghidra.util.task import ConsoleTaskMonitor

        func = self._get_function_at(addr)
        body = func.getBody()
        model = SimpleBlockModel(self.program)
        monitor = ConsoleTaskMonitor()

        blocks: list[BasicBlock] = []
        block_iter = model.getCodeBlocksContaining(body, monitor)
        while block_iter.hasNext():
            block = block_iter.next()
            start = int(block.getFirstStartAddress().getOffset())
            end = int(block.getMaxAddress().getOffset())
            instructions = self._disassemble_range(start, end)
            blocks.append(BasicBlock(
                start_addr=start,
                end_addr=end,
                instructions=instructions,
            ))
        return blocks

    def get_control_flow_graph(self, addr: int) -> ControlFlowGraph:
        """Build a complete control flow graph for the function at addr."""
        from ghidra.program.model.block import SimpleBlockModel
        from ghidra.util.task import ConsoleTaskMonitor

        func = self._get_function_at(addr)
        body = func.getBody()
        model = SimpleBlockModel(self.program)
        monitor = ConsoleTaskMonitor()

        blocks: list[BasicBlock] = []
        edges: list[BlockEdge] = []
        block_iter = model.getCodeBlocksContaining(body, monitor)
        while block_iter.hasNext():
            block = block_iter.next()
            start = int(block.getFirstStartAddress().getOffset())
            end = int(block.getMaxAddress().getOffset())
            instructions = self._disassemble_range(start, end)
            blocks.append(BasicBlock(
                start_addr=start,
                end_addr=end,
                instructions=instructions,
            ))

            dest_iter = block.getDestinations(monitor)
            while dest_iter.hasNext():
                dest_ref = dest_iter.next()
                dest_block = dest_ref.getDestinationBlock()
                if dest_block is None:
                    continue
                dest_addr = int(dest_block.getFirstStartAddress().getOffset())
                if not body.contains(dest_block.getFirstStartAddress()):
                    continue
                flow_type = dest_ref.getFlowType()
                edge_type = self._classify_flow(flow_type)
                edges.append(BlockEdge(
                    from_addr=start,
                    to_addr=dest_addr,
                    edge_type=edge_type,
                ))

        return ControlFlowGraph(
            function_addr=int(func.getEntryPoint().getOffset()),
            blocks=blocks,
            edges=edges,
        )

    def get_pcode_ops(self, addr: int) -> list[PcodeOp]:
        """Get high-level pcode operations for the function at addr."""
        from ghidra.app.decompiler import DecompInterface
        from ghidra.util.task import ConsoleTaskMonitor

        func = self._get_function_at(addr)
        di = DecompInterface()
        try:
            di.openProgram(self.program)
            result = di.decompileFunction(func, 30, ConsoleTaskMonitor())
            if not result.decompileCompleted():
                raise GhidraClientError(
                    f"Decompilation failed for pcode at 0x{addr:08x}"
                )
            high_func = result.getHighFunction()
            if high_func is None:
                return []

            ops: list[PcodeOp] = []
            op_iter = high_func.getPcodeOps()
            while op_iter.hasNext():
                op = op_iter.next()
                mnemonic = str(op.getMnemonic())
                op_addr = int(op.getSeqnum().getTarget().getOffset())
                inputs = [
                    str(op.getInput(i)) for i in range(op.getNumInputs())
                ]
                output = str(op.getOutput()) if op.getOutput() is not None else ""
                ops.append(PcodeOp(
                    mnemonic=mnemonic,
                    address=op_addr,
                    inputs=inputs,
                    output=output,
                ))
            return ops
        finally:
            di.dispose()

    def _disassemble_range(self, start: int, end: int) -> list[str]:
        """Get disassembly text for instructions in an address range."""
        listing = self.program.getListing()
        start_addr = self._to_addr(start)
        end_addr = self._to_addr(end)
        instructions: list[str] = []
        instr = listing.getInstructionAt(start_addr)
        while instr is not None and instr.getAddress().compareTo(end_addr) <= 0:
            instructions.append(str(instr))
            instr = instr.getNext()
        return instructions

    @staticmethod
    def _classify_flow(flow_type: object) -> BlockEdgeType:
        """Map a Ghidra FlowType to our BlockEdgeType enum."""
        name = str(flow_type)
        if "FALL_THROUGH" in name:
            return BlockEdgeType.FALL_THROUGH
        if "UNCONDITIONAL_JUMP" in name or "CONDITIONAL_JUMP" in name:
            return BlockEdgeType.BRANCH
        if "CALL" in name:
            return BlockEdgeType.CALL
        if "COMPUTED" in name or "INDIRECT" in name:
            return BlockEdgeType.COMPUTED
        return BlockEdgeType.UNKNOWN

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
