# Benchmarks

## Case Study: [XZ Backdoor](https://en.wikipedia.org/wiki/XZ_Utils_backdoor) (CVE-2024-3094)

In March 2024, a Microsoft engineer noticed SSH logins taking 500ms longer than usual. That observation unraveled one of the most sophisticated supply chain attacks in open-source history: a malicious maintainer had spent two years embedding a backdoor into XZ Utils, a compression library linked by virtually every Linux distribution's SSH daemon. The backdoored `liblzma.so.5.4.1` hijacked OpenSSH's RSA signature verification to execute arbitrary commands on targeted systems.

The malicious functions were hand-written to blend in with liblzma's legitimate compression code — no symbols, no obvious strings, no exports. We chose this binary because it represents the hardest class of RE problem: a real-world implant buried inside a large, legitimate codebase, where the malicious code is a small fraction of the total and deliberately avoids standing out.

Kong analyzed the stripped binary (558 functions enumerated, 396 queued) in **15 minutes** for **$6.63**.

| Metric | Value |
|--------|-------|
| Functions analyzed | 355 / 396 |
| High confidence (>=80%) | 308 (87%) |
| Medium confidence (60-79%) | 33 (9%) |
| Low confidence (<60%) | 14 (4%) |

### Backdoor Kill Chain — Fully Reconstructed

Kong independently identified all five core backdoor functions and reconstructed the attack chain with no prior knowledge of CVE-2024-3094:

| Function | Confidence | Role |
|----------|------------|------|
| `init_rsa_public_decrypt` | 95% | Parses ELF dynamic symbols at load time to locate `RSA_public_decrypt` |
| `function_hook_replace` | 90% | Overwrites the GOT entry, which changes memory protection, swaps pointer, and restores permissions |
| `rsa_public_decrypt_wrapper` | 95% | The hook: intercepts RSA verification, checks for root + magic value, decrypts payload |
| `initialize_cipher_context` | 92% | Sets up ChaCha20 state with 256-bit key, 96-bit nonce |
| `chacha20_encrypt` | 95% | Decrypts shellcode embedded in RSA signature data |

Kong's analysis of `rsa_public_decrypt_wrapper`:

> *"XZ backdoor: intercepts RSA_public_decrypt. When running as root and magic matches, decrypts and executes shellcode via ChaCha20."*

It also identified the supporting infrastructure: ELF dynamic section parsing, `/proc/self/maps` memory permission reads, and `dladdr1`-based symbol resolution. This meant that Kong correctly classified this infra as part of the implant's runtime hooking mechanism.

### Legitimate Code

Beyond the backdoor, Kong correctly recovered the full breadth of liblzma's real functionality — LZMA/LZMA2 encoders and decoders, match finders, range coders, streaming state machines, CRC32/CRC64 (generic and CLMUL-accelerated), SHA-256, XZ container format handling, and branch-call-jump filters for x86, ARM64, and RISC-V.

Five functions were flagged for potential control-flow flattening. All were correctly identified as false positives in the reasoning (legitimate 7-23 state resumption machines inherent to liblzma's streaming API).

### Why This Matters

The XZ backdoor was discovered by a human noticing a timing anomaly. Finding the implant through static analysis of the stripped binary without symbols, without source, without knowing what to look for, is the kind of task that traditionally takes an experienced reverse engineer days of manual work.

Kong reconstructed the full kill chain autonomously in 15 minutes. This suggests a path toward automated triage of suspected supply chain compromises: point Kong at a suspicious binary and get a structured assessment of what it does — including code that shouldn't be there.
