# String Encryption

## What It Looks Like

String literals are encrypted at compile time and decrypted at runtime just before use. Instead of seeing `"Hello, World!"` in the decompilation, you see a function that XORs a byte array with a key, or runs RC4 over an encrypted blob, then passes the result to the original call site.

## Identification

- Byte arrays initialized with non-ASCII / non-printable data used as function arguments where strings are expected
- XOR loops that iterate over a local/global byte array before passing it to `printf`, `strcmp`, `puts`, etc.
- Calls to small helper functions that take encrypted data + key and return `char *`
- Common patterns:
  - Single-byte XOR: `for (i = 0; i < len; i++) buf[i] ^= key;`
  - Multi-byte XOR: `for (i = 0; i < len; i++) buf[i] ^= key[i % key_len];`
  - RC4 decryption: KSA + PRGA applied to a byte array
  - Stack strings: characters pushed one-by-one onto the stack (e.g., `buf[0]='H'; buf[1]='e'; ...`)
  - Base64 decode followed by XOR

## Example 1: Single-Byte XOR

### Before (obfuscated)

```c
void FUN_00404000(void) {
    char buf[14] = { 0x2a, 0x27, 0x2e, 0x2e, 0x29, 0x4e, 0x42, 0x3f, 0x29, 0x36, 0x2e, 0x26, 0x43, 0x42 };
    for (int i = 0; i < 14; i++) {
        buf[i] ^= 0x42;
    }
    puts(buf);
}
```

### After (recovered)

```c
void print_greeting(void) {
    // Decrypted string: "Hello, World!\0"
    puts("Hello, World!");
}
```

## Example 2: Decryption Helper

### Before (obfuscated)

```c
char *FUN_00404100(unsigned char *data, int len, unsigned char key) {
    char *result = malloc(len + 1);
    for (int i = 0; i < len; i++) {
        result[i] = data[i] ^ key;
    }
    result[len] = 0;
    return result;
}
```

### After (recovered)

```c
char *xor_decrypt_string(unsigned char *data, int len, unsigned char key) {
    char *result = malloc(len + 1);
    for (int i = 0; i < len; i++) {
        result[i] = data[i] ^ key;
    }
    result[len] = 0;
    return result;
}
```

## Recovery Strategy

1. **Identify encrypted byte arrays** — look for local/global arrays with non-ASCII data passed to string-consuming functions
2. **Identify the decryption routine** — typically a small loop (XOR, add/sub, or RC4) immediately before the string is used
3. **Use tool: `simplify_expression`** on the decryption arithmetic to confirm it's a reversible operation
4. **If the key is a constant**, attempt to decrypt the string statically to annotate the function
5. **Name the decryption helper** if it's a standalone function (e.g., `xor_decrypt_string`)
6. **Annotate call sites** with the decrypted string values as comments
