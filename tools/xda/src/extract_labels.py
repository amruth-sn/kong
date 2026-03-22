from elftools.elf.elffile import ELFFile

def extract_function_boundaries(elf_path: str) -> list[dict]:
    """Extract function start/end addresses from DWARF debug info.

    Returns list of {"name": str | None, "start": int, "end": int}.
    """
    functions: list[dict] = []
    with open(elf_path, "rb") as f:
        elf = ELFFile(f)
        if not elf.has_dwarf_info():
            return []

        dwarf = elf.get_dwarf_info()
        for cu in dwarf.iter_CUs():
            for die in cu.iter_DIEs():
                if die.tag != "DW_TAG_subprogram":
                    continue
                if "DW_AT_low_pc" not in die.attributes:
                    continue

                low_pc = die.attributes["DW_AT_low_pc"].value
                high_pc_attr = die.attributes.get("DW_AT_high_pc")
                if high_pc_attr is None:
                    continue

                if high_pc_attr.form in ("DW_FORM_addr",):
                    high_pc = high_pc_attr.value
                else:
                    high_pc = low_pc + high_pc_attr.value

                if high_pc <= low_pc:
                    continue

                name_attr = die.attributes.get("DW_AT_name")
                name = name_attr.value.decode() if name_attr else None
                functions.append({"name": name, "start": low_pc, "end": high_pc})

    return functions

def generate_byte_labels(
    elf_path: str,
    boundaries: list[dict],
) -> tuple[bytes, list[int]]:
    """Generate per-byte labels for the .text section.

    Labels: 0 = non-function, 1 = function_start, 2 = function_body.
    Returns (text_section_bytes, labels).
    """
    with open(elf_path, "rb") as f:
        elf = ELFFile(f)
        text = elf.get_section_by_name(".text")
        if text is None:
            return b"", []

        text_start = text["sh_addr"]
        text_bytes = text.data()

    labels = [0] * len(text_bytes)

    for func in boundaries:
        start_off = func["start"] - text_start
        end_off = func["end"] - text_start

        if start_off < 0 or start_off >= len(text_bytes):
            continue
        end_off = min(end_off, len(text_bytes))

        labels[start_off] = 1  # function_start
        for i in range(start_off + 1, end_off):
            labels[i] = 2  # function_body

    return bytes(text_bytes), labels