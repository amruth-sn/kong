# Control Flow Flattening (CFF)

## What It Looks Like

A function's natural control flow is replaced with a single `while(1)` / `switch(state)` loop. Every basic block becomes a case in the switch, and an integer "state variable" determines which block executes next. The state variable is reassigned at the end of every case to route to the next block.

## Identification

- Single `while(1)`, `while(true)`, or `for(;;)` wrapping the entire function body
- Large `switch` statement inside the loop (often 5+ cases)
- Integer local variable reassigned in every case — this is the state variable
- No natural `if`/`else` or loop structures outside the switch
- State values are typically large hex constants (e.g., `0x3a2b`, `0x9e22`)
- The function entry assigns the initial state before the loop

## Example 1: XOR Decrypt

### Before (obfuscated)

```c
void FUN_004015a0(int *data, int len) {
    int state = 0x3a2b;
    while (1) {
        switch (state) {
            case 0x3a2b:
                if (len <= 0) { state = 0x9e22; }
                else { state = 0x7f01; }
                break;
            case 0x7f01:
                data[0] ^= 0x55;
                state = 0x1c44;
                break;
            case 0x1c44:
                data++;
                len--;
                state = 0x3a2b;
                break;
            case 0x9e22:
                return;
        }
    }
}
```

### After (recovered)

```c
void xor_decrypt(int *data, int len) {
    for (int i = 0; i < len; i++) {
        data[i] ^= 0x55;
    }
}
```

## Example 2: Linked List Search

### Before (obfuscated)

```c
void *FUN_00401200(void *list, int key) {
    int state = 0xa100;
    void *node = list;
    while (1) {
        switch (state) {
            case 0xa100:
                if (node == NULL) { state = 0xd300; }
                else { state = 0xb200; }
                break;
            case 0xb200:
                if (*(int *)node == key) { state = 0xe400; }
                else { state = 0xc300; }
                break;
            case 0xc300:
                node = *(void **)((char *)node + 8);
                state = 0xa100;
                break;
            case 0xd300:
                return NULL;
            case 0xe400:
                return node;
        }
    }
}
```

### After (recovered)

```c
void *linked_list_find(void *list, int key) {
    for (void *node = list; node != NULL; node = *(void **)((char *)node + 8)) {
        if (*(int *)node == key) {
            return node;
        }
    }
    return NULL;
}
```

## Recovery Strategy

1. **Identify the state variable** — the local integer reassigned in every switch case
2. **Use tool: `trace_state_machine`** to extract the state transition graph automatically
3. **Trace state transitions** — for each case, record `(current_state → next_state)` and any associated condition
4. **Build a state transition graph** from the extracted data
5. **Collapse linear chains** — consecutive states with unconditional transitions become sequential statements
6. **Identify conditional branches** — cases where the next state depends on a condition become `if`/`else`
7. **Identify loops** — cycles in the state graph become `while` or `for` loops
8. **Reconstruct structured code** — map the graph back to `if`/`else`, `while`, `for` constructs
9. **Name and type** the function based on the recovered logic
