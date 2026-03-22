#!/usr/bin/env bash
# tools/xda/scripts/build_corpus.sh
# Cross-compiles C projects at multiple optimization levels for a target architecture.
# Usage: ./build_corpus.sh <arch> <output_dir>
# Example: ./build_corpus.sh aarch64 ./corpus/aarch64

set -euo pipefail

ARCH="${1:?Usage: build_corpus.sh <arch> <output_dir>}"
OUT="${2:?Usage: build_corpus.sh <arch> <output_dir>}"

ZIG_TARGET=""
case "$ARCH" in
    x86_64)  CC="x86_64-linux-gnu-gcc"; STRIP="x86_64-linux-gnu-strip"; ZIG_TARGET="x86_64-linux-gnu" ;;
    aarch64) CC="aarch64-linux-gnu-gcc"; STRIP="aarch64-linux-gnu-strip"; ZIG_TARGET="aarch64-linux-gnu" ;;
    arm)     CC="arm-linux-gnueabihf-gcc"; STRIP="arm-linux-gnueabihf-strip"; ZIG_TARGET="arm-linux-gnueabihf" ;;
    mips)    CC="mips-linux-gnu-gcc"; STRIP="mips-linux-gnu-strip"; ZIG_TARGET="mips-linux-gnu" ;;
    riscv64) CC="riscv64-linux-gnu-gcc"; STRIP="riscv64-linux-gnu-strip"; ZIG_TARGET="riscv64-linux-gnu" ;;
    *)       echo "Unknown arch: $ARCH"; exit 1 ;;
esac

USE_ZIG=0
if command -v "$CC" >/dev/null 2>&1; then
    : # native cross-compiler found
elif command -v zig >/dev/null 2>&1 && [ -n "$ZIG_TARGET" ]; then
    USE_ZIG=1
    echo "Using zig cc -target $ZIG_TARGET (native $CC not found)"
else
    echo "$CC not found and zig not available. Install a cross-compiler or zig."; exit 1
fi

mkdir -p "$OUT"

# Source files to compile — create a set of diverse test programs
SOURCES_DIR=$(mktemp -d)
trap 'rm -rf "$SOURCES_DIR"' EXIT

# Generate diverse source files
cat > "$SOURCES_DIR/algorithms.c" << 'CSRC'
#include <stdlib.h>
#include <string.h>

void bubble_sort(int *arr, int n) {
    for (int i = 0; i < n - 1; i++)
        for (int j = 0; j < n - i - 1; j++)
            if (arr[j] > arr[j + 1]) { int t = arr[j]; arr[j] = arr[j + 1]; arr[j + 1] = t; }
}
int binary_search(int *arr, int n, int target) {
    int lo = 0, hi = n - 1;
    while (lo <= hi) { int mid = (lo + hi) / 2; if (arr[mid] == target) return mid; if (arr[mid] < target) lo = mid + 1; else hi = mid - 1; }
    return -1;
}
void *mempool_alloc(size_t size) { static char pool[4096]; static size_t offset = 0; if (offset + size > 4096) return NULL; void *p = &pool[offset]; offset += size; return p; }
struct node { int val; struct node *next; };
struct node *list_insert(struct node *head, int val) { struct node *n = malloc(sizeof(*n)); n->val = val; n->next = head; return n; }
int list_sum(struct node *head) { int s = 0; while (head) { s += head->val; head = head->next; } return s; }
unsigned crc32_byte(unsigned crc, unsigned char b) { crc ^= b; for (int i = 0; i < 8; i++) crc = (crc >> 1) ^ (0xEDB88320 & -(crc & 1)); return crc; }
int main() { int arr[] = {5,3,1,4,2}; bubble_sort(arr, 5); return binary_search(arr, 5, 3); }
CSRC

cat > "$SOURCES_DIR/string_ops.c" << 'CSRC'
#include <string.h>
#include <ctype.h>
int str_count_char(const char *s, char c) { int n = 0; while (*s) { if (*s == c) n++; s++; } return n; }
void str_reverse(char *s) { int len = strlen(s); for (int i = 0; i < len / 2; i++) { char t = s[i]; s[i] = s[len - 1 - i]; s[len - 1 - i] = t; } }
void str_to_upper(char *s) { while (*s) { *s = toupper(*s); s++; } }
int str_is_palindrome(const char *s) { int l = 0, r = strlen(s) - 1; while (l < r) { if (s[l++] != s[r--]) return 0; } return 1; }
char *str_find_substr(const char *haystack, const char *needle) { int nlen = strlen(needle); if (!nlen) return (char*)haystack; for (; *haystack; haystack++) { if (!strncmp(haystack, needle, nlen)) return (char*)haystack; } return 0; }
int main() { char buf[] = "hello"; str_reverse(buf); return str_is_palindrome(buf); }
CSRC

cat > "$SOURCES_DIR/state_machine.c" << 'CSRC'
enum state { IDLE, RUNNING, PAUSED, ERROR, DONE };
enum event { START, PAUSE, RESUME, FAIL, FINISH };
enum state transition(enum state s, enum event e) {
    switch (s) {
        case IDLE: return e == START ? RUNNING : s;
        case RUNNING: switch (e) { case PAUSE: return PAUSED; case FAIL: return ERROR; case FINISH: return DONE; default: return s; }
        case PAUSED: return e == RESUME ? RUNNING : (e == FAIL ? ERROR : s);
        case ERROR: return s;
        case DONE: return s;
    }
    return s;
}
int run_machine(enum event *events, int n) { enum state s = IDLE; for (int i = 0; i < n; i++) s = transition(s, events[i]); return s; }
int main() { enum event seq[] = {START, PAUSE, RESUME, FINISH}; return run_machine(seq, 4); }
CSRC

# Compile each source at each optimization level
for src in "$SOURCES_DIR"/*.c; do
    name=$(basename "$src" .c)
    for opt in O0 O1 O2 O3 Os; do
        out_debug="${OUT}/${name}_${opt}_debug"
        out_stripped="${OUT}/${name}_${opt}_stripped"

        if [ "$USE_ZIG" -eq 1 ]; then
            zig cc -target "$ZIG_TARGET" -g "-${opt}" -o "$out_debug" "$src" 2>/dev/null || continue
            cp "$out_debug" "$out_stripped"
            zig cc -target "$ZIG_TARGET" -s -o "$out_stripped" "$src" 2>/dev/null || continue
        else
            "$CC" -g "-${opt}" -o "$out_debug" "$src" 2>/dev/null || continue
            cp "$out_debug" "$out_stripped"
            "$STRIP" "$out_stripped"
        fi

        echo "Built: ${name}_${opt}"
    done
done

echo "Corpus built in $OUT"