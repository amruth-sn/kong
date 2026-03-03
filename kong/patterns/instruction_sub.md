# Instruction Substitution

## What It Looks Like

Simple arithmetic and logical operations are replaced with equivalent but more complex expressions using mathematical identities. The function's logic is preserved, but each operation becomes harder to understand at a glance.

## Identification

- Unnecessarily complex arithmetic where a simpler equivalent exists
- Multiple operations used where one would suffice
- Common substitution patterns:
  - `a + b` → `(a ^ b) + 2 * (a & b)`
  - `a - b` → `a + (~b + 1)`
  - `a ^ b` → `(a | b) & ~(a & b)` or `(~a & b) | (a & ~b)`
  - `a | b` → `(a & ~b) | b`
  - `a & b` → `(~(~a | ~b))`
  - `-a` → `~a + 1`
  - Boolean NOT: `!a` → `a ^ 1` or `(a + 1) & 1`
- These patterns nest: a substituted expression may itself contain substituted sub-expressions, creating deeply nested arithmetic

## Example 1: XOR via AND/OR

### Before (obfuscated)

```c
int FUN_00403000(int a, int b) {
    return (~a & b) | (a & ~b);
}
```

### After (recovered)

```c
int xor_values(int a, int b) {
    return a ^ b;
}
```

## Example 2: Addition via XOR/AND

### Before (obfuscated)

```c
int FUN_00403100(int x, int y) {
    int t1 = x ^ y;
    int t2 = x & y;
    int t3 = t2 << 1;
    return (t1 ^ t3) + 2 * (t1 & t3);
}
```

### After (recovered)

```c
int add(int x, int y) {
    return x + y;
}
```

## Example 3: Negation Chain

### Before (obfuscated)

```c
int FUN_00403200(int val) {
    return (~val + 1) + (~val + 1);
}
```

### After (recovered)

```c
int negate_double(int val) {
    return -2 * val;
}
```

## Recovery Strategy

1. **Extract all arithmetic expressions** from the decompiled function
2. **Use tool: `simplify_expression`** on each expression — z3 will reduce equivalent forms to canonical representations
3. **Replace complex expressions** with their simplified equivalents in the decompilation
4. **Look for patterns** — if a function is just `return simplified_expr`, the function's purpose becomes immediately clear
5. **Name and type** based on the simplified operations
