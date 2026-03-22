from __future__ import annotations

import functools
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from src.extract_labels import extract_function_boundaries, generate_byte_labels


def _probe_compile(cmd_prefix: list[str]) -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp)
        src = p / "probe.c"
        out = p / "probe.bin"
        src.write_text("int main(void) { return 0; }\n")
        try:
            subprocess.run(
                [*cmd_prefix, "-g", "-O0", "-o", str(out), str(src)],
                check=True,
                capture_output=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            return False
        return out.read_bytes()[:4] == b"\x7fELF"


@functools.lru_cache(maxsize=1)
def _elf_compile_prefix() -> list[str]:
    if sys.platform == "linux":
        candidates: tuple[list[str], ...] = (["gcc"], ["clang"])
    else:
        candidates = (
            ["x86_64-linux-gnu-gcc"],
            ["aarch64-linux-gnu-gcc"],
            ["x86_64-elf-gcc"],
            ["zig", "cc", "-target", "x86_64-linux-gnu"],
            ["zig", "cc", "-target", "aarch64-linux-gnu"],
            ["gcc"],
            ["clang"],
        )
    for prefix in candidates:
        if shutil.which(prefix[0]) and _probe_compile(prefix):
            return prefix
    pytest.skip(
        "No C compiler found that emits ELF (native macOS clang produces Mach-O; "
        "install Zig, brew install x86_64-elf-gcc, or a Linux cross-compiler)."
    )


def _compile_test_binary(src: str, out: Path) -> Path:
    """Compile a C string to an ELF binary with debug info."""
    src_file = out / "test.c"
    bin_file = out / "test.elf"
    src_file.write_text(src)

    prefix = _elf_compile_prefix()
    subprocess.run(
        [*prefix, "-g", "-O0", "-o", str(bin_file), str(src_file)],
        check=True,
    )
    return bin_file


def test_extract_boundaries_from_simple_binary():
    src = """
    int add(int a, int b) { return a + b; }
    int mul(int a, int b) { return a * b; }
    int main() { return add(1, 2) + mul(3, 4); }
    """
    with tempfile.TemporaryDirectory() as tmp:
        binary = _compile_test_binary(src, Path(tmp))
        boundaries = extract_function_boundaries(str(binary))

        names = {b["name"] for b in boundaries}
        assert "add" in names
        assert "mul" in names
        assert "main" in names

        for b in boundaries:
            assert b["start"] < b["end"]
            assert b["end"] - b["start"] > 0


def test_generate_byte_labels():
    src = "int foo() { return 42; } int main() { return foo(); }"
    with tempfile.TemporaryDirectory() as tmp:
        binary = _compile_test_binary(src, Path(tmp))
        boundaries = extract_function_boundaries(str(binary))
        text_bytes, labels = generate_byte_labels(str(binary), boundaries)

        assert len(text_bytes) == len(labels)
        assert len(text_bytes) > 0

        # At least some bytes should be labeled as function_start (1)
        assert 1 in labels
        # At least some bytes should be labeled as function_body (2)
        assert 2 in labels
        # At least some bytes should be non-function (0) unless .text is fully covered