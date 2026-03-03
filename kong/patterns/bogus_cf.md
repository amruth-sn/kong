# Bogus Control Flow (BCF)

## What It Looks Like

Fake branches are injected into the function using conditions that always evaluate to the same value (opaque predicates). The "dead" branch typically contains junk code or cloned/modified copies of real code. The function's actual logic is unchanged, but the control flow graph becomes much larger and harder to follow.

## Identification

- Conditions that are always true or always false for any input — opaque predicates
- Common opaque predicate patterns:
  - `(x * (x + 1)) % 2 == 0` — always true (product of consecutive integers is even)
  - `(x * x) % 2 == x % 2` — always true
  - `(x ^ x) != 0` — always false
  - `(x & ~x) != 0` — always false
  - `(x | ~x) == -1` — always true
  - Conditions involving global variables that are never written to
- Dead branches often contain:
  - Unreachable code that still references valid addresses
  - Copies of nearby code with slight modifications
  - Random-looking arithmetic that never produces visible side effects

## Example 1: Simple Opaque Predicate

### Before (obfuscated)

```c
int FUN_00401000(int a, int b) {
    int result;
    if ((a * (a + 1)) % 2 == 0) {
        result = a + b;
    } else {
        result = a ^ b ^ 0xDEAD;  // unreachable
    }
    if ((b & ~b) != 0) {
        result ^= 0xBEEF;  // unreachable
    }
    return result;
}
```

### After (recovered)

```c
int add_values(int a, int b) {
    return a + b;
}
```

## Example 2: Multiple Opaque Predicates

### Before (obfuscated)

```c
void FUN_00402000(char *buf, int len) {
    int i = 0;
    while (i < len) {
        if ((i * (i - 1)) % 2 != 0) {
            buf[i] = buf[i] ^ buf[i-1] ^ 0xFF;  // unreachable
        }
        buf[i] = buf[i] ^ 0x42;
        if ((i | ~i) == 0) {
            break;  // unreachable
        }
        i++;
    }
}
```

### After (recovered)

```c
void xor_buf(char *buf, int len) {
    for (int i = 0; i < len; i++) {
        buf[i] ^= 0x42;
    }
}
```

## Recovery Strategy

1. **Scan all branch conditions** in the function
2. **Use tool: `simplify_expression`** on each condition to check for opaque predicates
3. **For each resolved predicate:**
   - If always true → keep the "then" branch, discard the "else" branch
   - If always false → keep the "else" branch (if any), discard the "then" branch
4. **Use tool: `eliminate_dead_code`** to remove the unreachable branches from the decompilation
5. **Re-analyze the simplified code** — with dead branches removed, the function's real logic is exposed
6. **Name and type** based on the cleaned control flow
