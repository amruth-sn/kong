"""Tests for the PyGhidra in-process client."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from kong.ghidra.client import GhidraClient, GhidraClientError, _classify
from kong.ghidra.types import FunctionClassification


# ---------------------------------------------------------------------------
# Helpers — build mock Java objects
# ---------------------------------------------------------------------------

def _mock_program():
    """Create a mock Ghidra Program with enough methods for client tests."""
    prog = MagicMock()
    prog.getName.return_value = "test_binary"
    prog.getExecutablePath.return_value = "/tmp/test_binary"
    prog.getExecutableFormat.return_value = "Executable and Linking Format (ELF)"
    prog.getDefaultPointerSize.return_value = 8
    prog.getMinAddress.return_value.getOffset.return_value = 0x400000
    prog.getMaxAddress.return_value.getOffset.return_value = 0x4FFFFF

    lang = MagicMock()
    lang.getProcessor.return_value.toString.return_value = "x86"
    lang.isBigEndian.return_value = False
    prog.getLanguage.return_value = lang

    prog.getCompilerSpec.return_value.getCompilerSpecID.return_value = "gcc"

    # Address factory
    addr_space = MagicMock()
    addr_space.getAddress.side_effect = lambda offset: _mock_address(offset)
    prog.getAddressFactory.return_value.getDefaultAddressSpace.return_value = addr_space

    return prog


def _mock_address(offset):
    addr = MagicMock()
    addr.getOffset.return_value = offset
    return addr


def _mock_function(addr, name, size, is_thunk=False):
    func = MagicMock()
    func.getEntryPoint.return_value = _mock_address(addr)
    func.getName.return_value = name
    func.getBody.return_value.getNumAddresses.return_value = size
    func.isThunk.return_value = is_thunk
    func.getParameters.return_value = []
    func.getLocalVariables.return_value = []
    func.getReturnType.return_value.getDisplayName.return_value = "undefined"
    func.getCallingConventionName.return_value = "unknown"
    return func


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_pyghidra():
    """Patch pyghidra.start and pyghidra.open_program."""
    with patch("kong.ghidra.client.pyghidra") as mock_mod:
        mock_mod.start.return_value = MagicMock()
        yield mock_mod


@pytest.fixture
def client(mock_pyghidra, tmp_path):
    """Create an opened GhidraClient with mocked pyghidra."""
    binary = tmp_path / "test_binary"
    binary.write_bytes(b"\x00" * 16)

    prog = _mock_program()
    flat_api = MagicMock()
    flat_api.getCurrentProgram.return_value = prog

    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=flat_api)
    ctx.__exit__ = MagicMock(return_value=False)
    mock_pyghidra.open_program.return_value = ctx

    c = GhidraClient(str(binary))
    c.open()
    return c


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestConnection:
    def test_open_sets_program(self, client):
        assert client._program is not None
        assert client._flat_api is not None

    def test_close_clears_state(self, client):
        client.close()
        assert client._program is None
        assert client._flat_api is None

    def test_operations_before_open_raise(self, tmp_path):
        binary = tmp_path / "test_binary"
        binary.write_bytes(b"\x00" * 16)
        c = GhidraClient(str(binary))
        with pytest.raises(GhidraClientError, match="Not open"):
            c.program

    def test_open_nonexistent_binary_raises(self, mock_pyghidra):
        c = GhidraClient("/nonexistent/binary")
        with pytest.raises(GhidraClientError, match="Binary not found"):
            c.open()

    def test_context_manager(self, mock_pyghidra, tmp_path):
        binary = tmp_path / "test_binary"
        binary.write_bytes(b"\x00" * 16)

        prog = _mock_program()
        flat_api = MagicMock()
        flat_api.getCurrentProgram.return_value = prog

        ctx = MagicMock()
        ctx.__enter__ = MagicMock(return_value=flat_api)
        ctx.__exit__ = MagicMock(return_value=False)
        mock_pyghidra.open_program.return_value = ctx

        with GhidraClient(str(binary)) as c:
            assert c._program is not None
        # After exit, state is cleared
        assert c._program is None


class TestBinaryInfo:
    def test_get_binary_info(self, client):
        info = client.get_binary_info()
        assert info.name == "test_binary"
        assert info.arch == "x86"
        assert info.endianness == "little"
        assert info.word_size == 8
        assert info.format == "Executable and Linking Format (ELF)"
        assert info.compiler == "gcc"


class TestListFunctions:
    def test_list_functions(self, client):
        funcs_iter = [
            _mock_function(0x401000, "main", 120),
            _mock_function(0x401100, "FUN_00401100", 10),
            _mock_function(0x401200, "puts", 4, is_thunk=True),
        ]
        client.program.getFunctionManager.return_value.getFunctions.return_value = funcs_iter

        funcs = client.list_functions()
        assert len(funcs) == 3
        assert funcs[0].name == "main"
        assert funcs[0].address == 0x401000
        assert funcs[0].classification == FunctionClassification.MEDIUM
        assert funcs[1].classification == FunctionClassification.TRIVIAL
        assert funcs[2].classification == FunctionClassification.THUNK
        assert funcs[2].is_thunk is True

    def test_function_address_hex(self, client):
        client.program.getFunctionManager.return_value.getFunctions.return_value = [
            _mock_function(0x401000, "main", 100),
        ]
        funcs = client.list_functions()
        assert funcs[0].address_hex == "0x00401000"

    def test_classification_boundaries(self):
        assert _classify(16, False) == FunctionClassification.TRIVIAL
        assert _classify(17, False) == FunctionClassification.SMALL
        assert _classify(64, False) == FunctionClassification.SMALL
        assert _classify(65, False) == FunctionClassification.MEDIUM
        assert _classify(256, False) == FunctionClassification.MEDIUM
        assert _classify(257, False) == FunctionClassification.LARGE
        assert _classify(100, True) == FunctionClassification.THUNK


class TestDecompilation:
    def test_get_decompilation_success(self, client):
        func = _mock_function(0x401000, "main", 120)
        client.program.getFunctionManager.return_value.getFunctionAt.return_value = func

        mock_result = MagicMock()
        mock_result.decompileCompleted.return_value = True
        mock_result.getDecompiledFunction.return_value.getC.return_value = (
            "void main(void) {\n  puts(\"hello\");\n}\n"
        )

        MockDI = MagicMock()
        di_instance = MagicMock()
        di_instance.decompileFunction.return_value = mock_result
        MockDI.return_value = di_instance

        MockMonitor = MagicMock()

        ghidra_decompiler = MagicMock()
        ghidra_decompiler.DecompInterface = MockDI
        ghidra_task = MagicMock()
        ghidra_task.ConsoleTaskMonitor = MockMonitor

        with patch.dict("sys.modules", {
            "ghidra": MagicMock(),
            "ghidra.app": MagicMock(),
            "ghidra.app.decompiler": ghidra_decompiler,
            "ghidra.util": MagicMock(),
            "ghidra.util.task": ghidra_task,
        }):
            result = client.get_decompilation(0x401000)
            assert "void main" in result
            di_instance.dispose.assert_called_once()

    def test_get_decompilation_failure(self, client):
        func = _mock_function(0x401000, "main", 120)
        client.program.getFunctionManager.return_value.getFunctionAt.return_value = func

        mock_result = MagicMock()
        mock_result.decompileCompleted.return_value = False

        MockDI = MagicMock()
        di_instance = MagicMock()
        di_instance.decompileFunction.return_value = mock_result
        MockDI.return_value = di_instance

        ghidra_decompiler = MagicMock()
        ghidra_decompiler.DecompInterface = MockDI
        ghidra_task = MagicMock()
        ghidra_task.ConsoleTaskMonitor = MagicMock()

        with patch.dict("sys.modules", {
            "ghidra": MagicMock(),
            "ghidra.app": MagicMock(),
            "ghidra.app.decompiler": ghidra_decompiler,
            "ghidra.util": MagicMock(),
            "ghidra.util.task": ghidra_task,
        }):
            with pytest.raises(GhidraClientError, match="Decompilation failed"):
                client.get_decompilation(0x401000)


class TestXRefs:
    def test_get_xrefs_to(self, client):
        ref = MagicMock()
        ref.getFromAddress.return_value = _mock_address(0x401050)
        ref.getToAddress.return_value = _mock_address(0x401000)
        ref.getReferenceType.return_value.getName.return_value = "UNCONDITIONAL_CALL"
        client.flat_api.getReferencesTo.return_value = [ref]

        xrefs = client.get_xrefs_to(0x401000)
        assert len(xrefs) == 1
        assert xrefs[0].from_addr == 0x401050
        assert xrefs[0].ref_type == "UNCONDITIONAL_CALL"

    def test_get_callers(self, client):
        refs = []
        for from_addr in [0x401050, 0x401100]:
            ref = MagicMock()
            ref.getFromAddress.return_value = _mock_address(from_addr)
            ref.getReferenceType.return_value.isCall.return_value = True
            refs.append(ref)
        client.flat_api.getReferencesTo.return_value = refs

        callers = client.get_callers(0x401000)
        assert 0x401050 in callers
        assert 0x401100 in callers


class TestMutations:
    def _ghidra_modules(self):
        """Return a dict of mock ghidra modules for patch.dict(sys.modules)."""
        mock_symbol = MagicMock()
        mock_symbol.SourceType = MagicMock()
        mock_listing = MagicMock()
        mock_listing.CodeUnit = MagicMock()
        mock_listing.CodeUnit.PLATE_COMMENT = 0
        mock_listing.CodeUnit.PRE_COMMENT = 1
        mock_listing.CodeUnit.POST_COMMENT = 2
        mock_listing.CodeUnit.EOL_COMMENT = 3
        mock_listing.CodeUnit.REPEATABLE_COMMENT = 4
        return {
            "ghidra": MagicMock(),
            "ghidra.program": MagicMock(),
            "ghidra.program.model": MagicMock(),
            "ghidra.program.model.symbol": mock_symbol,
            "ghidra.program.model.listing": mock_listing,
        }

    def test_rename_function(self, client):
        func = _mock_function(0x401000, "FUN_00401000", 100)
        client.program.getFunctionManager.return_value.getFunctionAt.return_value = func
        client.program.startTransaction.return_value = 1

        with patch.dict("sys.modules", self._ghidra_modules()):
            client.rename_function(0x401000, "my_function")
            func.setName.assert_called_once()
            client.program.startTransaction.assert_called_once()
            client.program.endTransaction.assert_called_once_with(1, True)

    def test_add_comment(self, client):
        code_unit = MagicMock()
        client.program.getListing.return_value.getCodeUnitAt.return_value = code_unit
        client.program.startTransaction.return_value = 1

        with patch.dict("sys.modules", self._ghidra_modules()):
            client.add_comment(0x401000, "This is RC4 init", "plate")
            code_unit.setComment.assert_called_once()
            client.program.endTransaction.assert_called_once_with(1, True)

    def test_add_comment_invalid_type(self, client):
        with patch.dict("sys.modules", self._ghidra_modules()):
            with pytest.raises(ValueError, match="Invalid comment_type"):
                client.add_comment(0x401000, "test", "invalid_type")

    def test_no_function_at_address(self, client):
        client.program.getFunctionManager.return_value.getFunctionAt.return_value = None
        with pytest.raises(GhidraClientError, match="No function found"):
            client.rename_function(0xDEAD, "nope")


class TestFunctionInfo:
    def test_get_function_info(self, client):
        param0 = MagicMock()
        param0.getName.return_value = "argc"
        param0.getDataType.return_value.getDisplayName.return_value = "int"
        param0.getOrdinal.return_value = 0
        param0.getLength.return_value = 4

        param1 = MagicMock()
        param1.getName.return_value = "argv"
        param1.getDataType.return_value.getDisplayName.return_value = "char * *"
        param1.getOrdinal.return_value = 1
        param1.getLength.return_value = 8

        local0 = MagicMock()
        local0.getName.return_value = "buf"
        local0.getDataType.return_value.getDisplayName.return_value = "char[256]"
        local0.getLength.return_value = 256
        local0.isStackVariable.return_value = True
        local0.getStackOffset.return_value = -0x100

        func = _mock_function(0x401000, "main", 200)
        func.getParameters.return_value = [param0, param1]
        func.getLocalVariables.return_value = [local0]
        func.getReturnType.return_value.getDisplayName.return_value = "int"
        func.getCallingConventionName.return_value = "__cdecl"

        client.program.getFunctionManager.return_value.getFunctionAt.return_value = func

        info = client.get_function_info(0x401000)
        assert info.name == "main"
        assert info.return_type == "int"
        assert len(info.params) == 2
        assert info.params[0].name == "argc"
        assert info.params[1].data_type == "char * *"
        assert len(info.local_vars) == 1
        assert info.local_vars[0].stack_offset == -0x100
        assert info.classification == FunctionClassification.MEDIUM
