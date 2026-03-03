# VM-Based Protection (VMProtect / Themida / Custom VM)

## What It Looks Like

The original native code is compiled into a custom bytecode and interpreted at runtime by an embedded virtual machine. The function body is replaced with a call into the VM dispatcher, which fetches, decodes, and executes bytecodes from an embedded program. The original logic is not present as native instructions — it exists only as data consumed by the interpreter.

## Identification

- Large `switch` statement or function pointer table in a loop, dispatching on a byte/word read from a data pointer
- A "virtual instruction pointer" (VIP) incremented after each dispatch
- A "virtual stack pointer" (VSP) or register file used for operands
- Common dispatcher structure:
  ```
  while (1) {
      opcode = *vip++;
      switch (opcode) {
          case 0x01: /* push */ ...
          case 0x02: /* pop */ ...
          case 0x03: /* add */ ...
          case 0x10: /* jmp */ ...
          case 0xFF: return;
      }
  }
  ```
- The bytecode data is typically stored in a separate section or embedded inline
- Handler cases are small (1-5 operations each) and uniform in structure
- Many handlers manipulate the same set of variables (the virtual registers / stack)

## Distinguishing VM from CFF

Both use `while(1)/switch` patterns, but:
- **CFF**: state variable is reassigned to large hex constants, cases contain real program logic
- **VM**: dispatch variable comes from a data pointer (`*vip++`), cases are generic operations (push, pop, add, xor), and the real logic is encoded in the bytecode data, not the handler code

## What Kong Can Do (MVP)

Full VM deobfuscation (lifting bytecodes back to native semantics) is out of scope for MVP. Kong can:

1. **Detect** the VM dispatcher pattern and flag the function as VM-protected
2. **Identify** the handler table and enumerate virtual opcodes
3. **Classify** the function as `vm_protected` with low confidence
4. **Report** the number of handlers and estimated bytecode size

## Detection Heuristics

- Function has a large switch (20+ cases) inside a loop
- The switch variable is loaded from a pointer that is incremented (`*ptr++` or `ptr[offset]` with incrementing offset)
- Most cases are small (under 10 lines) and structurally similar
- Cases manipulate a common set of array/pointer variables (the virtual registers or stack)

## Recovery Strategy (Future)

1. Identify the VM dispatcher and handler table
2. Map each handler to a virtual opcode semantics (push, pop, add, sub, jmp, call, etc.)
3. Extract the bytecode stream from the embedded data
4. Disassemble the bytecodes using the handler semantics
5. Lift the virtual instruction trace back to C-like pseudocode
6. Name and type the recovered function

This requires dedicated VM-lifting infrastructure and is planned for a future release.
