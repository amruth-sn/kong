/* ============================================================
 * Binary:   hexdump
 * Arch:     x86
 * Format:   Mac OS X Mach-O
 * Compiler: gcc
 *
 * Functions: 34 total, 33 analyzed, 1 skipped, 0 errors
 * Renamed:   33 | Confirmed: 0
 * LLM calls: 33
 * Duration:  4m 29.2s
 * Cost:      $0.6914
 * ============================================================ */

/* ==================== I/O ==================== */

/**
 * @name  format_character_output
 * @brief Formats and outputs character data with special handling for control characters, multibyte sequences, and printable characters. Uses different output formats based on character properties.
 * @confidence 85%
 * @classification io
 * @address 0x1000005f8
 */

/* Formats and outputs character data with special handling for control characters, multibyte
   sequences, and printable characters. Uses different output formats based on character properties.
    */

void format_character_output(long param_1,byte *param_2,size_t param_3)

{
  byte bVar1;
  uint uVar2;
  int iVar3;
  size_t sVar4;
  byte *pbVar5;
  char *pcVar6;
  int iVar7;
  size_t sVar8;
  byte local_5e [6];
  size_t local_58;
  long local_50;
  wchar_t local_48;
  char local_42 [10];
  long local_38;
  
  local_38 = *(long *)PTR____stack_chk_guard_100004040;
  if (*(int *)(param_1 + 0x28) < 1) {
    bVar1 = *param_2;
    if ((bVar1 < 0xe) && ((0x3f81U >> (bVar1 & 0x1f) & 1) != 0)) {
      pcVar6 = &DAT_100003144 + *(int *)(&DAT_100003144 + (ulong)(uint)bVar1 * 4);
      goto LAB_10000062d;
    }
    if ((canonical_display_mode != 0) && (1 < *(int *)PTR____mb_cur_max_100004028)) {
      local_58 = 0;
      pbVar5 = (byte *)0x0;
      local_50 = param_1;
      while( true ) {
        sVar8 = param_3;
        sVar4 = _mbrtowc(&local_48,(char *)param_2,sVar8,(mbstate_t *)(param_1 + 0x30));
        if (sVar4 == 0xffffffffffffffff) break;
        if (sVar4 == 0) {
          sVar4 = 1;
LAB_10000076c:
          iVar7 = (int)sVar4 + (int)local_58 - 1;
          param_1 = local_50;
          goto LAB_100000785;
        }
        if ((param_2 == local_5e) && (sVar4 == 0xfffffffffffffffe)) break;
        if (sVar4 != 0xfffffffffffffffe) goto LAB_10000076c;
        param_3 = read_with_putback(local_5e,(long)*(int *)PTR____mb_cur_max_100004028);
        pbVar5 = param_2;
        param_2 = local_5e;
        local_58 = sVar8;
      }
      *(undefined8 *)(param_1 + 0xa0) = 0;
      *(undefined8 *)(param_1 + 0xa8) = 0;
      *(undefined8 *)(param_1 + 0x90) = 0;
      *(undefined8 *)(param_1 + 0x98) = 0;
      *(undefined8 *)(param_1 + 0x80) = 0;
      *(undefined8 *)(param_1 + 0x88) = 0;
      *(undefined8 *)(param_1 + 0x70) = 0;
      *(undefined8 *)(param_1 + 0x78) = 0;
      *(undefined8 *)(param_1 + 0x60) = 0;
      *(undefined8 *)(param_1 + 0x68) = 0;
      *(undefined8 *)(param_1 + 0x50) = 0;
      *(undefined8 *)(param_1 + 0x58) = 0;
      *(undefined8 *)(param_1 + 0x40) = 0;
      *(undefined8 *)(param_1 + 0x48) = 0;
      ((mbstate_t *)(param_1 + 0x30))->_mbstateL = 0;
      *(undefined8 *)(param_1 + 0x38) = 0;
      if (param_2 == local_5e) {
        param_2 = pbVar5;
      }
      local_48 = (uint)*param_2;
      param_1 = local_50;
LAB_100000809:
      pcVar6 = local_42;
      ___sprintf_chk(pcVar6,0,10,"%03o",*param_2);
      goto LAB_10000062d;
    }
    local_48 = (uint)bVar1;
    iVar7 = 0;
LAB_100000785:
    if ((uint)local_48 < 0x80) {
      uVar2 = *(uint *)(PTR___DefaultRuneLocale_100004000 + (ulong)(uint)local_48 * 4 + 0x3c) &
              0x40000;
    }
    else {
      uVar2 = ___maskrune(local_48,0x40000);
    }
    if (uVar2 == 0) goto LAB_100000809;
    if (canonical_display_mode != 0) {
      **(undefined1 **)(param_1 + 0x10) = 0x43;
      iVar3 = _strcmp(*(char **)(param_1 + 0x18),"%3C");
      if (iVar3 == 0) {
        iVar3 = _wcwidth(local_48);
        if (-1 < iVar3) {
          uVar2 = 3U - iVar3;
          if ((int)(3U - iVar3) < 1) {
            uVar2 = 0;
          }
          _printf("%*s%C",(ulong)uVar2,"",(ulong)(uint)local_48);
          *(int *)(param_1 + 0x28) = iVar7;
          goto LAB_100000642;
        }
      }
      else {
        assert_conv_c_format_check();
      }
      assert_width_nonnegative();
      goto LAB_10000084e;
    }
    **(undefined1 **)(param_1 + 0x10) = 99;
    _printf(*(char **)(param_1 + 0x18),(ulong)(uint)local_48);
  }
  else {
    *(int *)(param_1 + 0x28) = *(int *)(param_1 + 0x28) - 1;
    pcVar6 = "**";
LAB_10000062d:
    **(undefined1 **)(param_1 + 0x10) = 0x73;
    _printf(*(char **)(param_1 + 0x18),pcVar6);
  }
LAB_100000642:
  if (*(long *)PTR____stack_chk_guard_100004040 == local_38) {
    return;
  }
LAB_10000084e:
                    /* WARNING: Subroutine does not return */
  ___stack_chk_fail();
}



/**
 * @name  read_and_deduplicate_buffer
 * @brief Reads data from stdin into buffers, detects duplicate consecutive chunks, and suppresses duplicates by printing '*' marker
 * @confidence 75%
 * @classification io
 * @address 0x100000d94
 */

/* Reads data from stdin into buffers, detects duplicate consecutive chunks, and suppresses
   duplicates by printing '*' marker */

void * read_and_deduplicate_buffer(void)

{
  undefined *puVar1;
  void *pvVar2;
  int iVar3;
  uint uVar4;
  size_t sVar5;
  uint uVar6;
  int iVar7;
  uint uVar8;
  ulong uVar9;
  
  pvVar2 = DAT_100005030;
  if (DAT_100005030 == (void *)0x0) {
    DAT_100005030 = (void *)_malloc_type_calloc(1,(long)(int)max_record_size,0x1347a052);
    if (DAT_100005030 == (void *)0x0) goto LAB_10000103e;
    DAT_100005038 = (void *)_malloc_type_calloc(1,(long)(int)max_record_size,0xf8b7ace6);
    if (DAT_100005038 == (void *)0x0) {
                    /* WARNING: Subroutine does not return */
      FUN_100002e23();
    }
    uVar9 = (ulong)max_record_size;
  }
  else {
    DAT_100005030 = DAT_100005038;
    DAT_100005038 = pvVar2;
    uVar9 = (ulong)(int)max_record_size;
    current_file_offset = current_file_offset + uVar9;
  }
  puVar1 = PTR____stdinp_100004050;
  uVar8 = (uint)uVar9;
  iVar7 = 0;
  uVar6 = max_bytes_to_read;
  while (uVar6 != 0) {
    uVar8 = (uint)uVar9;
    while( true ) {
      if (((DAT_100005028 & 1) == 0) &&
         (iVar3 = process_input_files(0), uVar6 = max_bytes_to_read, iVar3 == 0)) goto LAB_100000f65;
      uVar4 = uVar8;
      if ((int)uVar6 < (int)uVar8) {
        uVar4 = uVar6;
      }
      if (uVar6 == 0xffffffff) {
        uVar4 = uVar8;
      }
      sVar5 = _fread((void *)((long)DAT_100005030 + (long)iVar7),1,(long)(int)uVar4,*(FILE **)puVar1
                    );
      iVar3 = (int)sVar5;
      if (iVar3 != 0) break;
      iVar3 = _ferror(*(FILE **)puVar1);
      if (iVar3 != 0) {
        warn_format_string();
      }
      DAT_100005028 = 0;
      uVar6 = max_bytes_to_read;
      if (max_bytes_to_read == 0) goto LAB_100000f65;
    }
    DAT_100005028 = 1;
    if (max_bytes_to_read == 0xffffffff) {
      uVar6 = 0xffffffff;
    }
    else {
      max_bytes_to_read = max_bytes_to_read - iVar3;
      uVar6 = max_bytes_to_read;
    }
    uVar9 = (ulong)(uVar8 - iVar3);
    if (uVar8 - iVar3 == 0) {
      if (((pvVar2 == (void *)0x0) || ((duplicate_suppress_mode & 0xfffffffd) == 0)) ||
         (iVar7 = _bcmp(DAT_100005030,DAT_100005038,(long)(int)max_record_size), iVar7 != 0)) {
        if (1 < duplicate_suppress_mode - 1) {
          return DAT_100005030;
        }
        duplicate_suppress_mode = 3;
        return DAT_100005030;
      }
      if (duplicate_suppress_mode == 3) {
        _puts("*");
      }
      uVar9 = (ulong)(int)max_record_size;
      current_file_offset = current_file_offset + uVar9;
      duplicate_suppress_mode = 1;
      iVar7 = 0;
      uVar6 = max_bytes_to_read;
    }
    else {
      iVar7 = iVar7 + iVar3;
    }
    uVar8 = (uint)uVar9;
  }
LAB_100000f65:
  if ((canonical_display_mode == 0) || (bytes_to_skip <= current_file_offset)) {
    if (uVar8 != max_record_size) {
      if (((pvVar2 == (void *)0x0) || (uVar8 != 0)) ||
         ((duplicate_suppress_mode == 0 ||
          (iVar3 = _bcmp(DAT_100005030,DAT_100005038,(long)iVar7), iVar3 != 0)))) {
        pvVar2 = DAT_100005030;
        ___bzero((long)DAT_100005030 + (long)iVar7,(long)(int)uVar8);
        end_of_data_offset = iVar7 + current_file_offset;
        return pvVar2;
      }
      if (duplicate_suppress_mode != 1) {
        _puts("*");
      }
    }
    return (void *)0x0;
  }
  skip_past_end_of_input_error();
LAB_10000103e:
                    /* WARNING: Subroutine does not return */
  FUN_100002e23();
}



/**
 * @name  process_input_files
 * @brief Processes command line arguments as input files, opening each file or defaulting to stdin if no files specified
 * @confidence 85%
 * @classification io
 * @address 0x1000010c6
 */

/* Processes command line arguments as input files, opening each file or defaulting to stdin if no
   files specified */

undefined8 process_input_files(long param_1)

{
  undefined *puVar1;
  FILE *pFVar2;
  long *plVar3;
  undefined8 uVar4;
  char *pcVar5;
  
  puVar1 = PTR____stdinp_100004050;
  plVar3 = current_input_files;
  if (param_1 != 0) {
    current_input_files = (long *)param_1;
    return 1;
  }
LAB_100001110:
  do {
    if ((char *)*plVar3 == (char *)0x0) {
      if (DAT_100005048 != 0) {
        DAT_100005048 = DAT_100005048 + 1;
        return 0;
      }
      uVar4 = 0;
      pcVar5 = "stdin";
      DAT_100005048 = 1;
      if (bytes_to_skip != 0) {
LAB_100001168:
        skip_input_bytes(pcVar5,uVar4);
        plVar3 = current_input_files;
      }
    }
    else {
      DAT_100005048 = 1;
      pFVar2 = _freopen((char *)*plVar3,"r",*(FILE **)puVar1);
      if (pFVar2 == (FILE *)0x0) {
        _warn("%s",*current_input_files);
        exit_status = 1;
        current_input_files = current_input_files + 1;
        plVar3 = current_input_files;
        goto LAB_100001110;
      }
      plVar3 = current_input_files;
      if (bytes_to_skip != 0) {
        pcVar5 = (char *)*current_input_files;
        uVar4 = 1;
        goto LAB_100001168;
      }
    }
    if (*plVar3 != 0) {
      plVar3 = plVar3 + 1;
      current_input_files = plVar3;
    }
    if (bytes_to_skip == 0) {
      return 1;
    }
  } while( true );
}



/**
 * @name  read_with_putback
 * @brief Reads up to max_bytes characters from stdin into buffer, then puts all read characters back using ungetc
 * @confidence 85%
 * @classification io
 * @address 0x1000011db
 */

/* Reads up to max_bytes characters from stdin into buffer, then puts all read characters back using
   ungetc */

ulong read_with_putback(long param_1,ulong param_2)

{
  undefined *puVar1;
  int iVar2;
  ulong uVar3;
  ulong uVar4;
  long lVar5;
  
  uVar4 = param_2;
  if (max_bytes_to_read < param_2) {
    uVar4 = (long)(int)max_bytes_to_read;
  }
  if ((long)(int)max_bytes_to_read == 0xffffffffffffffff) {
    uVar4 = param_2;
  }
  if (uVar4 == 0) {
LAB_10000125b:
    uVar4 = 0;
  }
  else {
    uVar3 = 0;
    do {
      iVar2 = _getchar();
      if (iVar2 == -1) {
        uVar4 = uVar3;
        if (uVar3 == 0) goto LAB_10000125b;
        break;
      }
      *(char *)(param_1 + uVar3) = (char)iVar2;
      uVar3 = uVar3 + 1;
    } while (uVar4 != uVar3);
    puVar1 = PTR____stdinp_100004050;
    lVar5 = 0;
    do {
      _ungetc((uint)*(byte *)(param_1 + uVar3 - 1 + lVar5),*(FILE **)puVar1);
      lVar5 = lVar5 - 1;
    } while (uVar4 + lVar5 != 0);
  }
  return uVar4;
}



/**
 * @name  skip_input_bytes
 * @brief Skips bytes from stdin based on file type and terminal characteristics, updating global position counters
 * @confidence 75%
 * @classification io
 * @address 0x10000126a
 */

/* Skips bytes from stdin based on file type and terminal characteristics, updating global position
   counters */

void skip_input_bytes(undefined8 param_1,int param_2)

{
  undefined *puVar1;
  int iVar2;
  undefined1 local_b0 [4];
  ushort local_ac;
  long local_50;
  byte local_1c [4];
  
  puVar1 = PTR____stdinp_100004050;
  if (param_2 == 0) goto LAB_100001339;
  iVar2 = _fileno(*(FILE **)PTR____stdinp_100004050);
  iVar2 = _fstat_INODE64(iVar2,local_b0);
  if (iVar2 != 0) goto LAB_10000135d;
  local_ac = local_ac & 0xf000;
  if (local_ac < 0x6000) {
    if (local_ac == 0x1000) goto LAB_100001339;
    if (local_ac == 0x2000) goto LAB_1000012fd;
  }
  else if (local_ac == 0x6000) {
LAB_1000012fd:
    iVar2 = _fileno(*(FILE **)puVar1);
    iVar2 = _ioctl(iVar2,0x4004667a,local_1c);
    if (iVar2 != 0) {
LAB_10000135d:
                    /* WARNING: Subroutine does not return */
      _err(1,"%s",param_1);
    }
    if ((local_1c[0] & 1) != 0) goto LAB_100001339;
  }
  else {
    if (local_ac == 0xc000) goto LAB_100001339;
    if ((local_ac == 0x8000) && (bytes_to_skip - local_50 != 0 && local_50 <= bytes_to_skip)) {
      current_file_offset = current_file_offset + local_50;
      bytes_to_skip = bytes_to_skip - local_50;
      return;
    }
  }
  iVar2 = _fseeko(*(FILE **)puVar1,bytes_to_skip,0);
  if (iVar2 == 0) {
    current_file_offset = current_file_offset + bytes_to_skip;
    bytes_to_skip = 0;
    return;
  }
LAB_100001339:
  consume_buffered_input();
  return;
}



/**
 * @name  consume_buffered_input
 * @brief Reads characters from stdin up to a limit, updating global counters for bytes consumed and remaining buffer space
 * @confidence 65%
 * @classification io
 * @address 0x100001373
 */

/* Reads characters from stdin up to a limit, updating global counters for bytes consumed and
   remaining buffer space */

void consume_buffered_input(void)

{
  int iVar1;
  long lVar2;
  
  lVar2 = 0;
  if (0 < bytes_to_skip) {
    do {
      iVar1 = _getchar();
      if (iVar1 == -1) break;
      lVar2 = lVar2 + 1;
    } while (lVar2 < bytes_to_skip);
  }
  current_file_offset = current_file_offset + lVar2;
  bytes_to_skip = bytes_to_skip - lVar2;
  return;
}



/**
 * @name  print_with_padding_once
 * @brief Prints a message with special handling for the first call - adds padding spaces before the first message, then prints the message and sets a flag to skip padding on subsequent calls
 * @confidence 85%
 * @classification io
 * @address 0x100002174
 */

/* Prints a message with special handling for the first call - adds padding spaces before the first
   message, then prints the message and sets a flag to skip padding on subsequent calls */

void print_with_padding_once(undefined8 param_1)

{
  if (DAT_10000504c == '\x01') {
    parse_format_specification("\"         \"");
  }
  parse_format_specification(param_1);
  DAT_10000504c = 1;
  return;
}



/**
 * @name  discard_long_line
 * @brief Discards the rest of a line from input after detecting it exceeds maximum length. Warns the user and reads characters until newline or EOF.
 * @confidence 75%
 * @classification io
 * @address 0x100002ec3
 */

/* Discards the rest of a line from input after detecting it exceeds maximum length. Warns the user
   and reads characters until newline or EOF. */

void discard_long_line(void)

{
  int iVar1;
  
  _warnx("line too long");
  do {
    iVar1 = _getchar();
    if (iVar1 == 10) {
      return;
    }
  } while (iVar1 != -1);
  return;
}



/* ==================== String Operations ==================== */

/**
 * @name  process_escape_sequences
 * @brief Processes a null-terminated string in-place, converting C-style escape sequences (\n, \t, \r, etc.) to their actual byte values
 * @confidence 95%
 * @classification string
 * @address 0x100002590
 */

/* Processes a null-terminated string in-place, converting C-style escape sequences (\n, \t, \r,
   etc.) to their actual byte values */

void process_escape_sequences(byte *param_1)

{
  byte bVar1;
  byte *pbVar2;
  
  pbVar2 = param_1;
  do {
    if (*pbVar2 == 0x5c) {
      bVar1 = pbVar2[1];
      if (bVar1 < 0x66) {
        if (bVar1 == 0x61) {
          bVar1 = 7;
        }
        else if (bVar1 == 0x62) {
          bVar1 = 8;
        }
        else if (bVar1 == 0) {
          param_1[0] = 0x5c;
          param_1[1] = 0;
          return;
        }
      }
      else {
        switch(bVar1) {
        case 0x6e:
          bVar1 = 10;
          break;
        case 0x6f:
        case 0x70:
        case 0x71:
        case 0x73:
        case 0x75:
          break;
        case 0x72:
          bVar1 = 0xd;
          break;
        case 0x74:
          bVar1 = 9;
          break;
        case 0x76:
          bVar1 = 0xb;
          break;
        default:
          if (bVar1 == 0x66) {
            bVar1 = 0xc;
          }
        }
      }
      pbVar2 = pbVar2 + 1;
      *param_1 = bVar1;
    }
    else {
      *param_1 = *pbVar2;
      if (*pbVar2 == 0) {
        return;
      }
    }
    pbVar2 = pbVar2 + 1;
    param_1 = param_1 + 1;
  } while( true );
}



/* ==================== Initialization ==================== */

/**
 * @name  main_hexdump
 * @brief Program entry point that sets locale, parses command-line arguments to determine execution mode, processes data structures, and handles program exit with error checking
 * @confidence 95%
 * @classification init
 * @address 0x1000013b3
 */

/* Program entry point that sets locale, parses command-line arguments to determine execution mode,
   processes data structures, and handles program exit with error checking */

void main_hexdump(undefined4 param_1,undefined8 *param_2)

{
  FILE *pFVar1;
  int iVar2;
  char *pcVar3;
  long *plVar4;
  long *plVar5;
  undefined8 *local_20;
  
  local_20 = param_2;
  _setlocale(0,"");
  pcVar3 = _strrchr((char *)*param_2,0x6f);
  if ((((pcVar3 == (char *)0x0) || (*pcVar3 != 'o')) || (pcVar3[1] != 'd')) || (pcVar3[2] != '\0'))
  {
    parse_hexdump_command_line(param_1,&local_20);
  }
  else {
    parse_od_command_line(param_1,&local_20);
  }
  max_record_size = 0;
  plVar4 = format_list_head;
  if (format_list_head != (long *)0x0) {
    do {
      iVar2 = validate_format_strings(plVar4);
      *(int *)(plVar4 + 2) = iVar2;
      if (max_record_size < iVar2) {
        max_record_size = iVar2;
      }
      plVar4 = (long *)*plVar4;
      plVar5 = format_list_head;
    } while (plVar4 != (long *)0x0);
    for (; plVar5 != (long *)0x0; plVar5 = (long *)*plVar5) {
      compile_format_descriptors(plVar5);
    }
  }
  process_input_files(local_20);
  process_formatted_output();
  pFVar1 = *(FILE **)PTR____stdoutp_100004058;
  iVar2 = _ferror(pFVar1);
  if ((iVar2 == 0) && (iVar2 = _fflush(pFVar1), iVar2 == 0)) {
                    /* WARNING: Subroutine does not return */
    _exit(exit_status);
  }
                    /* WARNING: Subroutine does not return */
  _err(1,"stdout");
}



/* ==================== Handlers ==================== */

/**
 * @name  skip_past_end_of_input_error
 * @brief Prints an error message about skipping past end of input and terminates the program via _errx()
 * @confidence 75%
 * @classification handler
 * @address 0x100002e51
 */

/* Prints an error message about skipping past end of input and terminates the program via _errx()
    */

void skip_past_end_of_input_error(void)

{
                    /* WARNING: Subroutine does not return */
  _errx(1,"cannot skip past end of input");
}



/**
 * @name  error_bad_skip_value
 * @brief Prints an error message for invalid skip value and terminates the program
 * @confidence 72%
 * @classification handler
 * @address 0x100002e66
 */

/* Prints an error message for invalid skip value and terminates the program */

void error_bad_skip_value(void)

{
                    /* WARNING: Subroutine does not return */
  _errx(1,"%s: bad skip value",*(undefined8 *)PTR__optarg_100004130);
}



/**
 * @name  handle_invalid_skip_amount
 * @brief Prints an error message about an invalid skip amount and terminates the program via _errx()
 * @confidence 72%
 * @classification handler
 * @address 0x100002e88
 */

/* Prints an error message about an invalid skip amount and terminates the program via _errx() */

void handle_invalid_skip_amount(void)

{
                    /* WARNING: Subroutine does not return */
  _errx(1,"%s: invalid skip amount",*(undefined8 *)PTR__optarg_100004130);
}



/**
 * @name  report_unrecognized_format_char
 * @brief Calls errx() with error status 1 to report an unrecognized format character. Does not return.
 * @confidence 85%
 * @classification handler
 * @address 0x100002ea7
 */

/* Calls errx() with error status 1 to report an unrecognized format character. Does not return. */

void report_unrecognized_format_char(char param_1)

{
                    /* WARNING: Subroutine does not return */
  _errx(1,"%c: unrecognised format character",(int)param_1);
}



/**
 * @name  error_missing_conversion_character
 * @brief Calls _errx with exit code 1 and an error message about a missing conversion character, indicating a format string parsing error
 * @confidence 75%
 * @classification handler
 * @address 0x100002ee6
 */

/* Calls _errx with exit code 1 and an error message about a missing conversion character,
   indicating a format string parsing error */

void error_missing_conversion_character(void)

{
                    /* WARNING: Subroutine does not return */
  _errx(1,"missing conversion character");
}



/**
 * @name  error_missing_conversion_character
 * @brief Calls _errx() to report a missing conversion character error and exit the program with status code 1
 * @confidence 75%
 * @classification handler
 * @address 0x100002efb
 */

/* Calls _errx() to report a missing conversion character error and exit the program with status
   code 1 */

void error_missing_conversion_character(void)

{
                    /* WARNING: Subroutine does not return */
  _errx(1,"missing conversion character");
}



/**
 * @name  handle_bad_conversion_character
 * @brief Writes a null byte at offset +3 of a parameter, then calls _errx to report a bad conversion character error and exit with status 1
 * @confidence 55%
 * @classification handler
 * @address 0x100002f10
 */

/* Writes a null byte at offset +3 of a parameter, then calls _errx to report a bad conversion
   character error and exit with status 1 */

void handle_bad_conversion_character(long param_1)

{
  *(undefined1 *)(param_1 + 3) = 0;
                    /* WARNING: Subroutine does not return */
  _errx(1,"%%%s: bad conversion character",param_1);
}



/**
 * @name  exit_with_precision_error
 * @brief Prints an error message and exits the program. The error message indicates a missing precision or byte count specification in a format string context.
 * @confidence 60%
 * @classification handler
 * @address 0x100002f2c
 */

/* Prints an error message and exits the program. The error message indicates a missing precision or
   byte count specification in a format string context. */

void exit_with_precision_error(void)

{
                    /* WARNING: Subroutine does not return */
  _errx(1,"%%s: requires a precision or a byte count");
}



/**
 * @name  error_multiple_conversion_characters
 * @brief Prints an error message and terminates the program due to multiple conversion characters in byte count specification
 * @confidence 75%
 * @classification handler
 * @address 0x100002f41
 */

/* Prints an error message and terminates the program due to multiple conversion characters in byte
   count specification */

void error_multiple_conversion_characters(void)

{
                    /* WARNING: Subroutine does not return */
  _errx(1,"byte count with multiple conversion characters");
}



/**
 * @name  error_missing_conversion_character
 * @brief Calls _errx() with error code 1 and message about missing conversion character, indicating an error handler for format string parsing failures
 * @confidence 75%
 * @classification handler
 * @address 0x100002f56
 */

/* Calls _errx() with error code 1 and message about missing conversion character, indicating an
   error handler for format string parsing failures */

void error_missing_conversion_character(void)

{
                    /* WARNING: Subroutine does not return */
  _errx(1,"missing conversion character");
}



/* ==================== Parsers ==================== */

/**
 * @name  process_formatted_output
 * @brief Processes formatted output by traversing linked data structures and applying type-specific formatting rules
 * @confidence 75%
 * @classification parser
 * @address 0x100000938
 */

/* Processes formatted output by traversing linked data structures and applying type-specific
   formatting rules */

void process_formatted_output(void)

{
  byte bVar1;
  int iVar2;
  undefined1 *puVar3;
  undefined8 *puVar4;
  uint uVar5;
  double *pdVar6;
  long lVar7;
  undefined8 *puVar8;
  ulong uVar9;
  double *pdVar10;
  char *pcVar11;
  double *pdVar12;
  undefined8 *puVar13;
  int iVar14;
  undefined1 uVar15;
  double dVar16;
  double *local_40;
  
  local_40 = (double *)read_and_deduplicate_buffer();
  if (local_40 != (double *)0x0) {
    uVar15 = 0;
    puVar4 = format_list_head;
    do {
      for (; pdVar10 = current_file_offset, puVar4 != (undefined8 *)0x0; puVar4 = (undefined8 *)*puVar4) {
        pdVar6 = current_file_offset;
        pdVar12 = local_40;
        for (puVar8 = (undefined8 *)puVar4[1];
            (puVar8 != (undefined8 *)0x0 && ((*(byte *)(puVar8 + 2) & 1) == 0));
            puVar8 = (undefined8 *)*puVar8) {
          for (iVar14 = *(int *)((long)puVar8 + 0x14); iVar14 != 0; iVar14 = iVar14 - 1) {
            for (puVar13 = (undefined8 *)puVar8[1]; puVar13 != (undefined8 *)0x0;
                puVar13 = (undefined8 *)*puVar13) {
              if (((end_of_data_offset != (double *)0x0) && ((long)end_of_data_offset <= (long)pdVar6)) &&
                 ((*(ushort *)(puVar13 + 1) & 0x402) == 0)) {
                parse_printf_format_specifier(puVar13);
              }
              if ((iVar14 == 1) && (puVar3 = (undefined1 *)puVar13[4], puVar3 != (undefined1 *)0x0))
              {
                uVar15 = *puVar3;
                *puVar3 = 0;
              }
              iVar2 = *(int *)(puVar13 + 1);
              if (0x3f < iVar2) {
                if (iVar2 < 0x100) {
                  if (iVar2 == 0x40) {
                    pcVar11 = (char *)puVar13[3];
                    bVar1 = *(byte *)pdVar12;
                    uVar5 = (uint)bVar1;
                    if ((long)(char)bVar1 < 0) {
                      ___maskrune((uint)bVar1,0x40000);
                      uVar5 = 0x2e;
                    }
                    else if ((PTR___DefaultRuneLocale_100004000[(long)(char)bVar1 * 4 + 0x3e] & 4)
                             == 0) {
                      uVar5 = 0x2e;
                    }
                    _printf(pcVar11,(ulong)uVar5);
                  }
                  else if (iVar2 == 0x80) {
                    pcVar11 = (char *)puVar13[3];
                    pdVar6 = pdVar12;
                    goto LAB_100000c2b;
                  }
                }
                else if (iVar2 == 0x100) {
                  format_character_for_display(puVar13,pdVar12);
                }
                else {
                  if (iVar2 == 0x200) {
                    switch(*(undefined4 *)((long)puVar13 + 0xc)) {
                    case 1:
                      pcVar11 = (char *)puVar13[3];
                      pdVar6 = (double *)(ulong)*(byte *)pdVar12;
                      goto LAB_100000c2b;
                    case 2:
                      pdVar6 = (double *)(ulong)*(ushort *)pdVar12;
                      break;
                    default:
                      goto switchD_100000a15_caseD_3;
                    case 4:
                      pdVar6 = (double *)(ulong)(uint)*(float *)pdVar12;
                      break;
                    case 8:
                      pdVar6 = (double *)*pdVar12;
                    }
                    goto LAB_100000c27;
                  }
                  if (iVar2 == 0x400) {
                    pcVar11 = "%s";
                    pdVar6 = (double *)puVar13[3];
                    goto LAB_100000c2b;
                  }
                }
                goto switchD_100000a15_caseD_3;
              }
              switch(iVar2) {
              case 1:
                pcVar11 = (char *)puVar13[3];
                pdVar6 = current_file_offset;
                goto LAB_100000c2b;
              case 2:
                pcVar11 = (char *)puVar13[3];
                pdVar6 = (double *)"";
LAB_100000c2b:
                _printf(pcVar11,pdVar6);
                break;
              case 3:
              case 5:
              case 6:
              case 7:
                break;
              case 4:
                if (end_of_data_offset == (double *)0x0) {
                  lVar7 = (long)max_record_size - (long)current_file_offset % (long)max_record_size;
                }
                else {
                  lVar7 = (long)end_of_data_offset - (long)current_file_offset;
                }
                format_character_output(puVar13,pdVar12,lVar7);
                break;
              case 8:
                pcVar11 = (char *)puVar13[3];
                uVar9 = (ulong)*(byte *)pdVar12;
LAB_100000ac3:
                _printf(pcVar11,uVar9);
                break;
              default:
                if (iVar2 == 0x10) {
                  iVar2 = *(int *)((long)puVar13 + 0xc);
                  if (iVar2 == 0x10) {
                    _printf((char *)puVar13[3]);
                  }
                  else {
                    if (iVar2 == 8) {
                      dVar16 = *pdVar12;
                      pcVar11 = (char *)puVar13[3];
                    }
                    else {
                      if (iVar2 != 4) break;
                      pcVar11 = (char *)puVar13[3];
                      dVar16 = (double)*(float *)pdVar12;
                    }
                    _printf(pcVar11,dVar16);
                  }
                }
                else if (iVar2 == 0x20) {
                  switch(*(undefined4 *)((long)puVar13 + 0xc)) {
                  case 1:
                    pcVar11 = (char *)puVar13[3];
                    pdVar6 = (double *)(long)*(char *)pdVar12;
                    goto LAB_100000c2b;
                  case 2:
                    pdVar6 = (double *)(long)*(short *)pdVar12;
                    break;
                  default:
                    goto switchD_100000a15_caseD_3;
                  case 4:
                    pdVar6 = (double *)(long)(int)*(float *)pdVar12;
                    break;
                  case 8:
                    uVar9 = (ulong)(uint)(int)*(char *)pdVar12;
                    pcVar11 = (char *)puVar13[3];
                    goto LAB_100000ac3;
                  }
LAB_100000c27:
                  pcVar11 = (char *)puVar13[3];
                  goto LAB_100000c2b;
                }
              }
switchD_100000a15_caseD_3:
              if ((iVar14 == 1) && ((undefined1 *)puVar13[4] != (undefined1 *)0x0)) {
                *(undefined1 *)puVar13[4] = uVar15;
              }
              pdVar6 = (double *)((long)current_file_offset + (long)*(int *)((long)puVar13 + 0xc));
              pdVar12 = (double *)((long)pdVar12 + (long)*(int *)((long)puVar13 + 0xc));
              current_file_offset = pdVar6;
            }
          }
        }
        current_file_offset = pdVar10;
      }
      local_40 = (double *)read_and_deduplicate_buffer();
      puVar4 = format_list_head;
    } while (local_40 != (double *)0x0);
  }
  if (trailing_format_element != 0) {
    if (end_of_data_offset == (double *)0x0) {
      if (current_file_offset == (double *)0x0) {
        return;
      }
      end_of_data_offset = current_file_offset;
    }
    for (puVar4 = *(undefined8 **)(trailing_format_element + 8); puVar4 != (undefined8 *)0x0;
        puVar4 = (undefined8 *)*puVar4) {
      if (*(int *)(puVar4 + 1) == 0x400) {
        pdVar10 = (double *)puVar4[3];
        pcVar11 = "%s";
LAB_100000d15:
        _printf(pcVar11,pdVar10);
      }
      else if (*(int *)(puVar4 + 1) == 1) {
        pcVar11 = (char *)puVar4[3];
        pdVar10 = end_of_data_offset;
        goto LAB_100000d15;
      }
    }
  }
  return;
}



/**
 * @name  parse_printf_format_specifier
 * @brief Parses a printf-style format string, skips format flags, and modifies the format specifier
 * @confidence 85%
 * @classification parser
 * @address 0x100001048
 */

/* Parses a printf-style format string, skips format flags, and modifies the format specifier */

void parse_printf_format_specifier(long param_1)

{
  char cVar1;
  void *pvVar2;
  long lVar3;
  char *pcVar4;
  char *pcVar5;
  
  *(undefined4 *)(param_1 + 8) = 2;
  **(undefined1 **)(param_1 + 0x10) = 0x73;
  *(undefined1 *)(*(long *)(param_1 + 0x10) + 1) = 0;
  pcVar4 = *(char **)(param_1 + 0x18);
  do {
    cVar1 = *pcVar4;
    pcVar4 = pcVar4 + 1;
  } while (cVar1 != '%');
  cVar1 = *pcVar4;
  pcVar5 = pcVar4;
  while ((cVar1 != '\0' && (pvVar2 = _memchr(" -0+#",(int)cVar1,6), pvVar2 != (void *)0x0))) {
    cVar1 = pcVar5[1];
    pcVar5 = pcVar5 + 1;
  }
  lVar3 = 0;
  do {
    cVar1 = pcVar5[lVar3];
    pcVar4[lVar3] = cVar1;
    lVar3 = lVar3 + 1;
  } while (cVar1 != '\0');
  return;
}



/**
 * @name  parse_hexdump_command_line
 * @brief Parses command-line options for hexdump/hd utility, setting output formats and processing parameters
 * @confidence 85%
 * @classification parser
 * @address 0x1000014b0
 */

/* Parses command-line options for hexdump/hd utility, setting output formats and processing
   parameters */

void parse_hexdump_command_line(int param_1,long *param_2)

{
  byte bVar1;
  char *pcVar2;
  char *pcVar3;
  int iVar4;
  byte *local_38;
  
  pcVar2 = (char *)*param_2;
  local_38 = (byte *)_strrchr(*(char **)pcVar2,0x68);
  if ((((local_38 != (byte *)0x0) && (*local_38 == 0x68)) && (local_38[1] == 100)) &&
     (local_38[2] == 0)) {
    parse_format_specification("\"%08.8_Ax\n\"");
    parse_format_specification("\"%08.8_ax  \" 8/1 \"%02x \" \"  \" 8/1 \"%02x \" ");
    parse_format_specification("\"  |\" 16/1 \"%_p\" \"|\\n\"");
  }
  do {
    while( true ) {
      while (iVar4 = _getopt(param_1,pcVar2,"bcCde:f:n:os:vx"), 0x61 < iVar4) {
        switch(iVar4) {
        case 0x62:
          parse_format_specification("\"%07.7_Ax\n\"");
          parse_format_specification("\"%07.7_ax \" 16/1 \"%03o \" \"\\n\"");
          break;
        case 99:
          parse_format_specification("\"%07.7_Ax\n\"");
          parse_format_specification("\"%07.7_ax \" 16/1 \"%3_c \" \"\\n\"");
          break;
        case 100:
          parse_format_specification("\"%07.7_Ax\n\"");
          parse_format_specification("\"%07.7_ax \" 8/2 \"  %05u \" \"\\n\"");
          break;
        case 0x65:
          parse_format_specification(*(undefined8 *)PTR__optarg_100004130);
          break;
        case 0x66:
          read_format_file(*(undefined8 *)PTR__optarg_100004130);
          break;
        case 0x6e:
          pcVar3 = *(char **)PTR__optarg_100004130;
          max_bytes_to_read = _atoi(pcVar3);
          if (max_bytes_to_read < 0) {
                    /* WARNING: Subroutine does not return */
            _errx(1,"%s: bad length value",pcVar3);
          }
          break;
        case 0x6f:
          parse_format_specification("\"%07.7_Ax\n\"");
          parse_format_specification("\"%07.7_ax \" 8/2 \" %06o \" \"\\n\"");
          break;
        case 0x73:
          bytes_to_skip = _strtoll(*(char **)PTR__optarg_100004130,(char **)&local_38,0);
          if (bytes_to_skip < 0) {
            error_bad_skip_value();
            goto LAB_100001770;
          }
          bVar1 = *local_38;
          if (bVar1 < 0x6b) {
            if (bVar1 == 0x62) {
              bytes_to_skip = bytes_to_skip << 9;
            }
            else if (bVar1 == 0x67) {
              bytes_to_skip = bytes_to_skip << 0x1e;
            }
          }
          else if (bVar1 == 0x6b) {
            bytes_to_skip = bytes_to_skip << 10;
          }
          else if (bVar1 == 0x6d) {
            bytes_to_skip = bytes_to_skip << 0x14;
          }
          break;
        case 0x76:
          duplicate_suppress_mode = 0;
          break;
        case 0x78:
          parse_format_specification("\"%07.7_Ax\n\"");
          parse_format_specification("\"%07.7_ax \" 8/2 \"   %04x \" \"\\n\"");
        }
      }
      if (iVar4 != 0x43) break;
      parse_format_specification("\"%08.8_Ax\n\"");
      parse_format_specification("\"%08.8_ax  \" 8/1 \"%02x \" \"  \" 8/1 \"%02x \" ");
      parse_format_specification("\"  |\" 16/1 \"%_p\" \"|\\n\"");
    }
    if (iVar4 == -1) {
      if (format_list_head == 0) {
        parse_format_specification("\"%07.7_Ax\n\"");
        parse_format_specification("\"%07.7_ax \" 8/2 \"%04x \" \"\\n\"");
      }
      *param_2 = *param_2 + (long)*(int *)PTR__optind_100004138 * 8;
      return;
    }
  } while (iVar4 != 0x3f);
LAB_100001770:
  _fprintf(*(FILE **)PTR____stderrp_100004048,"%s\n%s\n%s\n%s\n",
           "usage: hexdump [-bcCdovx] [-e fmt] [-f fmt_file] [-n length]",
           "               [-s skip] [file ...]",
           "       hd      [-bcdovx]  [-e fmt] [-f fmt_file] [-n length]",
           "               [-s skip] [file ...]");
                    /* WARNING: Subroutine does not return */
  _exit(1);
}



/**
 * @name  parse_od_command_line
 * @brief Parses command line arguments for the 'od' (octal dump) utility, handling format options, address base, skip offset, and length parameters
 * @confidence 85%
 * @classification parser
 * @address 0x100001808
 */

/* Parses command line arguments for the 'od' (octal dump) utility, handling format options, address
   base, skip offset, and length parameters */

void parse_od_command_line(int param_1,long *param_2)

{
  byte bVar1;
  undefined *puVar2;
  long *plVar3;
  byte bVar4;
  undefined1 uVar5;
  int iVar6;
  int *piVar7;
  size_t sVar8;
  undefined8 *puVar9;
  sbyte sVar10;
  long lVar11;
  char *pcVar12;
  byte *pbVar13;
  byte *pbVar14;
  int iVar15;
  byte *local_48;
  byte *local_40;
  long *local_38;
  
  parse_format_specification("\"%07.7_Ao\n\"");
  parse_format_specification("\"%07.7_ao  \"");
  canonical_display_mode = 1;
  pcVar12 = (char *)*param_2;
  local_38 = param_2;
  do {
    iVar6 = _getopt(param_1,pcVar12,"A:aBbcDdeFfHhIij:LlN:Oost:vXx");
    plVar3 = format_list_head;
    switch(iVar6) {
    case 0x41:
      pbVar14 = *(byte **)PTR__optarg_100004130;
      bVar4 = *pbVar14;
      if (bVar4 < 0x6f) {
        if (bVar4 == 100) {
LAB_100001a6c:
          *(byte *)(*(long *)(format_list_head[1] + 0x20) + 7) = bVar4;
          *(undefined1 *)(*(long *)(*(long *)(*format_list_head + 8) + 0x20) + 7) =
               **(undefined1 **)PTR__optarg_100004130;
          break;
        }
        if (bVar4 == 0x6e) {
          *(undefined **)(format_list_head[1] + 0x20) = &DAT_100005008;
          *(char **)(*(long *)(*plVar3 + 8) + 0x20) = s__10000500c;
          break;
        }
      }
      else if ((bVar4 == 0x78) || (bVar4 == 0x6f)) goto LAB_100001a6c;
      pcVar12 = "%s: invalid address base";
LAB_100001d28:
                    /* WARNING: Subroutine does not return */
      _errx(1,pcVar12,pbVar14);
    case 0x42:
    case 0x6f:
      generate_default_formats("o2");
      break;
    case 0x43:
    case 0x45:
    case 0x47:
    case 0x4a:
    case 0x4b:
    case 0x4d:
    case 0x50:
    case 0x51:
    case 0x52:
    case 0x53:
    case 0x54:
    case 0x55:
    case 0x56:
    case 0x57:
    case 0x59:
    case 0x5a:
    case 0x5b:
    case 0x5c:
    case 0x5d:
    case 0x5e:
    case 0x5f:
    case 0x60:
    case 0x67:
    case 0x6b:
    case 0x6d:
    case 0x6e:
    case 0x70:
    case 0x71:
    case 0x72:
    case 0x75:
    case 0x77:
      goto switchD_100001877_caseD_43;
    case 0x44:
      generate_default_formats("u4");
      break;
    case 0x46:
    case 0x65:
      generate_default_formats("fD");
      break;
    case 0x48:
    case 0x58:
      generate_default_formats("x4");
      break;
    case 0x49:
    case 0x4c:
    case 0x6c:
      generate_default_formats("dL");
      break;
    case 0x4e:
      pbVar14 = *(byte **)PTR__optarg_100004130;
      max_bytes_to_read = _atoi((char *)pbVar14);
      if (max_bytes_to_read < 1) {
        pcVar12 = "%s: invalid length";
        goto LAB_100001d28;
      }
      break;
    case 0x4f:
      generate_default_formats("o4");
      break;
    case 0x61:
      generate_default_formats("a");
      break;
    case 0x62:
      generate_default_formats("o1");
      break;
    case 99:
      generate_default_formats("c");
      break;
    case 100:
      generate_default_formats("u2");
      break;
    case 0x66:
      generate_default_formats("fF");
      break;
    case 0x68:
    case 0x78:
      generate_default_formats("x2");
      break;
    case 0x69:
      generate_default_formats("dI");
      break;
    case 0x6a:
      piVar7 = ___error();
      *piVar7 = 0;
      bytes_to_skip = _strtoll(*(char **)PTR__optarg_100004130,(char **)&local_40,0);
      pbVar14 = local_40;
      bVar4 = *local_40;
      if (bVar4 < 0x6b) {
        if (bVar4 == 0x62) {
          sVar10 = 9;
          goto LAB_100001aaf;
        }
        if (bVar4 == 0x67) {
          sVar10 = 0x1e;
          goto LAB_100001aaf;
        }
      }
      else {
        if (bVar4 == 0x6d) {
          sVar10 = 0x14;
        }
        else {
          if (bVar4 != 0x6b) goto LAB_100001ab5;
          sVar10 = 10;
        }
LAB_100001aaf:
        bytes_to_skip = bytes_to_skip << sVar10;
      }
LAB_100001ab5:
      piVar7 = ___error();
      if (((*piVar7 != 0) || (bytes_to_skip < 0)) || (sVar8 = _strlen((char *)pbVar14), 1 < sVar8))
      goto LAB_100001ade;
      break;
    case 0x73:
      generate_default_formats("d2");
      break;
    case 0x74:
      generate_default_formats(*(undefined8 *)PTR__optarg_100004130);
      break;
    case 0x76:
      duplicate_suppress_mode = 0;
      break;
    default:
      goto switchD_100001877_default;
    }
  } while( true );
LAB_100001ade:
  iVar6 = handle_invalid_skip_amount();
switchD_100001877_default:
  if (iVar6 != -1) {
switchD_100001877_caseD_43:
    puVar2 = PTR____stderrp_100004048;
    _fwrite("usage: od [-aBbcDdeFfHhIiLlOosvXx] [-A base] [-j skip] [-N length] [-t type]\n",0x4d,1,
            *(FILE **)PTR____stderrp_100004048);
    _fwrite("          [[+]offset[.][Bb]] [file ...]\n",0x28,1,*(FILE **)puVar2);
                    /* WARNING: Subroutine does not return */
    _exit(1);
  }
  if (*(long *)*format_list_head == 0) {
    generate_default_formats("oS");
  }
  plVar3 = local_38;
  iVar6 = *(int *)PTR__optind_100004138;
  puVar9 = (undefined8 *)((long)iVar6 * 8 + *local_38);
  *local_38 = (long)puVar9;
  if (param_1 == iVar6) {
    return;
  }
  if (param_1 - iVar6 == 1) {
    pbVar14 = (byte *)*puVar9;
    lVar11 = 1;
    if (*pbVar14 != 0x2b) {
      return;
    }
  }
  else {
    pbVar14 = (byte *)puVar9[1];
    bVar4 = *pbVar14;
    lVar11 = 1;
    if (bVar4 != 0x2b) {
      if (param_1 - iVar6 < 2) {
        return;
      }
      if (9 < bVar4 - 0x30) {
        if (bVar4 != 0x78) {
          return;
        }
        if ((PTR___DefaultRuneLocale_100004000[(ulong)pbVar14[1] * 4 + 0x3e] & 1) == 0) {
          return;
        }
      }
      lVar11 = 0;
    }
  }
  pbVar13 = pbVar14 + lVar11;
  if (pbVar14[lVar11] == 0x30) {
    if (pbVar13[1] == 0x78) {
      pbVar13 = pbVar13 + 2;
      goto LAB_100001ba9;
    }
LAB_100001bcc:
    pbVar14 = pbVar14 + lVar11 - 1;
    do {
      bVar4 = pbVar14[1];
      pbVar14 = pbVar14 + 1;
    } while (bVar4 - 0x30 < 10);
    iVar6 = 0;
    bVar1 = 1;
  }
  else {
    if ((pbVar14[lVar11] != 0x78) ||
       ((PTR___DefaultRuneLocale_100004000[(ulong)pbVar13[1] * 4 + 0x3e] & 1) == 0))
    goto LAB_100001bcc;
    pbVar13 = pbVar13 + 1;
LAB_100001ba9:
    pbVar14 = pbVar13 - 1;
    do {
      bVar4 = pbVar14[1];
      pbVar14 = pbVar14 + 1;
    } while ((PTR___DefaultRuneLocale_100004000[(ulong)bVar4 * 4 + 0x3e] & 1) != 0);
    bVar1 = 0;
    iVar6 = 0x10;
  }
  if (pbVar13 == pbVar14) {
    return;
  }
  iVar15 = 10;
  if (bVar4 != 0x2e) {
    iVar15 = iVar6;
  }
  if (!(bool)(bVar4 != 0x2e | bVar1)) {
    return;
  }
  iVar6 = 8;
  if (iVar15 != 0) {
    iVar6 = iVar15;
  }
  bytes_to_skip = _strtoll((char *)pbVar13,(char **)&local_48,iVar6);
  if (local_48 != pbVar14) {
    bytes_to_skip = 0;
    return;
  }
  bVar4 = *pbVar14;
  if (bVar4 == 0x2e) {
LAB_100001c79:
    bVar4 = pbVar14[1];
  }
  else {
    if (bVar4 == 0x42) {
      sVar10 = 10;
LAB_100001c73:
      bytes_to_skip = bytes_to_skip << sVar10;
      goto LAB_100001c79;
    }
    if (bVar4 == 0x62) {
      sVar10 = 9;
      goto LAB_100001c73;
    }
  }
  if (bVar4 != 0) {
    bytes_to_skip = 0;
    return;
  }
  if (iVar15 == 0x10) {
    uVar5 = 0x78;
  }
  else {
    if (iVar15 != 10) goto LAB_100001cc5;
    uVar5 = 100;
  }
  *(undefined1 *)(*(long *)(format_list_head[1] + 0x20) + 7) = uVar5;
  *(undefined1 *)(*(long *)(*(long *)(*format_list_head + 8) + 0x20) + 7) = uVar5;
LAB_100001cc5:
  *(undefined8 *)(*plVar3 + 8) = 0;
  return;
}



/**
 * @name  generate_default_formats
 * @brief Parses format specifiers in a string and generates corresponding hexdump-style format strings
 * @confidence 85%
 * @classification parser
 * @address 0x100001e20
 */

/* Parses format specifiers in a string and generates corresponding hexdump-style format strings */

void generate_default_formats(byte *param_1)

{
  byte bVar1;
  byte bVar2;
  undefined1 auVar3 [16];
  uint uVar4;
  int *piVar5;
  ulong uVar6;
  int iVar7;
  byte *pbVar8;
  char *pcVar9;
  byte *pbVar10;
  undefined8 uVar11;
  bool bVar12;
  byte *local_40;
  char *local_38;
  
  bVar1 = *param_1;
  do {
    if (bVar1 == 0) {
      return;
    }
    pbVar8 = param_1 + 1;
    switch(bVar1) {
    case 0x61:
      pcVar9 = "16/1 \"%3_u \" \"\\n\"";
      break;
    case 99:
      pcVar9 = "16/1 \"%3_c \" \"\\n\"";
      break;
    case 100:
switchD_100001e62_caseD_64:
      bVar2 = *pbVar8;
      uVar4 = (uint)bVar2;
      if (bVar2 < 0x4c) {
        if (uVar4 == 0x43) {
          pbVar10 = (byte *)0x1;
          goto LAB_100001f8d;
        }
        if (uVar4 == 0x49) {
          pbVar10 = (byte *)0x4;
          goto LAB_100001f8d;
        }
LAB_100001f14:
        pbVar10 = (byte *)0x4;
        if (uVar4 - 0x30 < 10) {
          piVar5 = ___error();
          *piVar5 = 0;
          pbVar10 = (byte *)_strtoul((char *)pbVar8,(char **)&local_40,10);
          piVar5 = ___error();
          if ((*piVar5 != 0) || (pbVar10 == (byte *)0x0)) goto LAB_10000212c;
          if (((byte *)0x8 < pbVar10) ||
             (pbVar8 = local_40, (0x116UL >> ((ulong)pbVar10 & 0x3f) & 1) == 0)) {
            pcVar9 = "unsupported int size %lu";
            goto LAB_100002155;
          }
        }
      }
      else {
        if (bVar2 == 0x4c) {
          pbVar10 = (byte *)0x8;
        }
        else {
          if (uVar4 != 0x53) goto LAB_100001f14;
          pbVar10 = (byte *)0x2;
        }
LAB_100001f8d:
        pbVar8 = param_1 + 2;
      }
      uVar6 = ~(-1L << ((char)pbVar10 * '\b' & 0x3fU));
      iVar7 = 0;
      do {
        iVar7 = iVar7 + 1;
        uVar6 = uVar6 >> (bVar1 == 0x78) + 3;
      } while (uVar6 != 0);
      pcVar9 = "0";
      if (bVar1 == 0x75) {
        pcVar9 = "";
      }
      bVar12 = bVar1 == 100;
      if (bVar12) {
        pcVar9 = "";
      }
      auVar3._8_8_ = 0;
      auVar3._0_8_ = pbVar10;
      _asprintf(&local_38,"%lu/%lu \"%*s%%%s%d%c\" \"\\n\"",
                SUB168((ZEXT816(0) << 0x40 | ZEXT816(0x10)) / auVar3,0),pbVar10,
                (ulong)(((int)pbVar10 * 4 - (uint)bVar12) - iVar7),"",pcVar9,
                (ulong)((uint)bVar12 + iVar7),(ulong)(uint)(int)(char)bVar1);
      if (local_38 == (char *)0x0) {
                    /* WARNING: Subroutine does not return */
        FUN_100002e23();
      }
      goto LAB_1000020e4;
    case 0x66:
      bVar1 = *pbVar8;
      if (bVar1 == 0x44) {
        pbVar8 = param_1 + 2;
        goto LAB_1000020ab;
      }
      if (bVar1 != 0x4c) {
        if (bVar1 == 0x46) {
          iVar7 = 6;
          uVar11 = 4;
          pbVar8 = param_1 + 2;
          goto LAB_1000020b7;
        }
        if (9 < bVar1 - 0x30) {
LAB_1000020ab:
          iVar7 = 0xf;
          uVar11 = 8;
          goto LAB_1000020b7;
        }
        piVar5 = ___error();
        *piVar5 = 0;
        pbVar10 = (byte *)_strtoul((char *)pbVar8,(char **)&local_40,10);
        piVar5 = ___error();
        if ((*piVar5 == 0) && (pbVar10 != (byte *)0x0)) {
          pbVar8 = local_40;
          if (pbVar10 == (byte *)0x4) {
            iVar7 = 6;
            uVar11 = 4;
            goto LAB_1000020b7;
          }
          if (pbVar10 == (byte *)0x10) goto LAB_10000203c;
          if (pbVar10 != (byte *)0x8) {
            pcVar9 = "unsupported floating point size %lu";
            goto LAB_100002155;
          }
          goto LAB_1000020ab;
        }
LAB_10000212c:
        pcVar9 = "%s: invalid size";
        pbVar10 = pbVar8;
LAB_100002155:
                    /* WARNING: Subroutine does not return */
        _errx(1,pcVar9,pbVar10);
      }
      pbVar8 = param_1 + 2;
LAB_10000203c:
      iVar7 = 0x12;
      uVar11 = 0x10;
LAB_1000020b7:
      _asprintf(&local_38,"%lu/%lu \" %%%d.%de \" \"\\n\"",(ulong)(byte)(0x10 / (byte)uVar11),uVar11
                ,(ulong)(iVar7 + 8));
      if (local_38 == (char *)0x0) goto LAB_100002127;
LAB_1000020e4:
      print_with_padding_once();
      _free(local_38);
      goto LAB_1000020f2;
    default:
      if ((bVar1 - 0x6f < 10) && ((0x241U >> (bVar1 - 0x6f & 0x1f) & 1) != 0))
      goto switchD_100001e62_caseD_64;
    case 0x62:
    case 0x65:
      report_unrecognized_format_char(bVar1);
LAB_100002127:
                    /* WARNING: Subroutine does not return */
      FUN_100002e23();
    }
    print_with_padding_once(pcVar9);
LAB_1000020f2:
    bVar1 = *pbVar8;
    param_1 = pbVar8;
  } while( true );
}



/**
 * @name  read_format_file
 * @brief Reads a configuration file line by line, skips whitespace and comments (lines starting with #), and processes non-comment lines
 * @confidence 85%
 * @classification parser
 * @address 0x1000021a8
 */

/* Reads a configuration file line by line, skips whitespace and comments (lines starting with #),
   and processes non-comment lines */

void read_format_file(char *param_1)

{
  byte *pbVar1;
  byte bVar2;
  undefined *puVar3;
  uint uVar4;
  FILE *pFVar5;
  char *pcVar6;
  byte *pbVar7;
  byte local_848 [2064];
  long local_38;
  
  local_38 = *(long *)PTR____stack_chk_guard_100004040;
  pFVar5 = _fopen(param_1,"r");
  if (pFVar5 == (FILE *)0x0) {
                    /* WARNING: Subroutine does not return */
    _err(1,"%s",param_1);
  }
  pcVar6 = _fgets((char *)local_848,0x801,pFVar5);
  puVar3 = PTR___DefaultRuneLocale_100004000;
  do {
    if (pcVar6 == (char *)0x0) {
      _fclose(pFVar5);
      if (*(long *)PTR____stack_chk_guard_100004040 == local_38) {
        return;
      }
                    /* WARNING: Subroutine does not return */
      ___stack_chk_fail();
    }
    pcVar6 = _strchr((char *)local_848,10);
    if (pcVar6 == (char *)0x0) {
      discard_long_line();
    }
    else {
      *pcVar6 = '\0';
      pbVar7 = local_848;
      bVar2 = local_848[0];
      while (bVar2 != 0) {
        if ((char)bVar2 < '\0') {
          uVar4 = ___maskrune((uint)bVar2,0x4000);
        }
        else {
          uVar4 = *(uint *)(puVar3 + (ulong)bVar2 * 4 + 0x3c) & 0x4000;
        }
        if (uVar4 == 0) {
          if ((*pbVar7 != 0) && (*pbVar7 != 0x23)) {
            parse_format_specification(pbVar7);
          }
          break;
        }
        pbVar1 = pbVar7 + 1;
        pbVar7 = pbVar7 + 1;
        bVar2 = *pbVar1;
      }
    }
    pcVar6 = _fgets((char *)local_848,0x801,pFVar5);
  } while( true );
}



/**
 * @name  parse_format_specification
 * @brief Parses a structured text format containing numeric values, optional quoted strings, and builds a linked list of parsed elements
 * @confidence 75%
 * @classification parser
 * @address 0x1000022d6
 */

/* Parses a structured text format containing numeric values, optional quoted strings, and builds a
   linked list of parsed elements */

void parse_format_specification(byte *param_1)

{
  byte *pbVar1;
  undefined *puVar2;
  uint uVar3;
  int iVar4;
  long lVar5;
  long *plVar6;
  long *plVar7;
  char *pcVar8;
  byte bVar9;
  size_t sVar10;
  byte *pbVar11;
  byte *pbVar12;
  
  lVar5 = _malloc_type_calloc(1,0x18,0x1020040edceb4c7);
  if (lVar5 == 0) {
                    /* WARNING: Subroutine does not return */
    FUN_100002e23();
  }
  plVar6 = &format_list_head;
  if (format_list_head != 0) {
    plVar6 = DAT_100005050;
  }
  *plVar6 = lVar5;
  pbVar12 = param_1;
  plVar6 = (long *)(lVar5 + 8);
  DAT_100005050 = (long *)lVar5;
  while( true ) {
    puVar2 = PTR___DefaultRuneLocale_100004000;
    pbVar11 = pbVar12 - 1;
    do {
      bVar9 = pbVar11[1];
      if ((long)(char)bVar9 < 0) {
        uVar3 = ___maskrune((uint)bVar9,0x4000);
      }
      else {
        uVar3 = *(uint *)(puVar2 + (long)(char)bVar9 * 4 + 0x3c) & 0x4000;
      }
      pbVar11 = pbVar11 + 1;
    } while (uVar3 != 0);
    if (bVar9 == 0) {
      return;
    }
    plVar7 = (long *)_malloc_type_calloc(1,0x28,0x1030040c0889a8e);
    if (plVar7 == (long *)0x0) {
                    /* WARNING: Subroutine does not return */
      FUN_100002e23();
    }
    *plVar6 = (long)plVar7;
    *(undefined4 *)((long)plVar7 + 0x14) = 1;
    bVar9 = *pbVar11;
    uVar3 = bVar9 - 0x30;
    pbVar12 = pbVar11;
    if (uVar3 < 10) {
      while (uVar3 < 10) {
        bVar9 = pbVar12[1];
        pbVar12 = pbVar12 + 1;
        uVar3 = bVar9 - 0x30;
      }
      if ((char)bVar9 < '\0') {
        uVar3 = ___maskrune((uint)bVar9,0x4000);
      }
      else {
        uVar3 = *(uint *)(puVar2 + (ulong)bVar9 * 4 + 0x3c) & 0x4000;
      }
      if ((bVar9 != 0x2f) && (uVar3 == 0)) break;
      iVar4 = _atoi((char *)pbVar11);
      *(int *)((long)plVar7 + 0x14) = iVar4;
      *(undefined4 *)(plVar7 + 2) = 2;
      do {
        bVar9 = pbVar12[1];
        if ((long)(char)bVar9 < 0) {
          uVar3 = ___maskrune((uint)bVar9,0x4000);
        }
        else {
          uVar3 = *(uint *)(puVar2 + (long)(char)bVar9 * 4 + 0x3c) & 0x4000;
        }
        pbVar12 = pbVar12 + 1;
      } while (uVar3 != 0);
      bVar9 = *pbVar12;
      pbVar11 = pbVar12;
    }
    if (bVar9 == 0x2f) {
      do {
        bVar9 = pbVar11[1];
        if ((long)(char)bVar9 < 0) {
          uVar3 = ___maskrune((uint)bVar9,0x4000);
        }
        else {
          uVar3 = *(uint *)(puVar2 + (long)(char)bVar9 * 4 + 0x3c) & 0x4000;
        }
        pbVar11 = pbVar11 + 1;
      } while (uVar3 != 0);
      bVar9 = *pbVar11;
    }
    if (bVar9 - 0x30 < 10) {
      pbVar12 = pbVar11 - 1;
      do {
        bVar9 = pbVar12[1];
        pbVar12 = pbVar12 + 1;
      } while (bVar9 - 0x30 < 10);
      if ((char)bVar9 < '\0') {
        uVar3 = ___maskrune((uint)bVar9,0x4000);
      }
      else {
        uVar3 = *(uint *)(puVar2 + (ulong)(uint)bVar9 * 4 + 0x3c) & 0x4000;
      }
      if (uVar3 == 0) break;
      iVar4 = _atoi((char *)pbVar11);
      *(int *)(plVar7 + 3) = iVar4;
      do {
        bVar9 = pbVar12[1];
        if ((long)(char)bVar9 < 0) {
          uVar3 = ___maskrune((uint)bVar9,0x4000);
        }
        else {
          uVar3 = *(uint *)(puVar2 + (long)(char)bVar9 * 4 + 0x3c) & 0x4000;
        }
        pbVar12 = pbVar12 + 1;
      } while (uVar3 != 0);
      bVar9 = *pbVar12;
      pbVar11 = pbVar12;
    }
    if (bVar9 != 0x22) break;
    pbVar12 = pbVar11 + 2;
    sVar10 = 1;
    while (pbVar1 = pbVar12 - 1, *pbVar1 != 0x22) {
      pbVar12 = pbVar12 + 1;
      sVar10 = sVar10 + 1;
      if (*pbVar1 == 0) goto LAB_100002568;
    }
    pcVar8 = (char *)_malloc_type_malloc(sVar10,0x1acbde61);
    plVar7[4] = (long)pcVar8;
    if (pcVar8 == (char *)0x0) {
                    /* WARNING: Subroutine does not return */
      FUN_100002e23();
    }
    _strlcpy(pcVar8,(char *)(pbVar11 + 1),sVar10);
    process_escape_sequences(plVar7[4]);
    plVar6 = plVar7;
  }
LAB_100002568:
                    /* WARNING: Subroutine does not return */
  _errx(1,"\"%s\": bad format",param_1);
}



/**
 * @name  validate_format_strings
 * @brief Validates printf format specifiers in a linked list of format strings, checking for proper conversion characters, flags, and precision specifiers
 * @confidence 85%
 * @classification parser
 * @address 0x100002634
 */

/* Validates printf format specifiers in a linked list of format strings, checking for proper
   conversion characters, flags, and precision specifiers */

void validate_format_strings(long param_1)

{
  byte bVar1;
  undefined8 *puVar2;
  void *pvVar3;
  uint uVar4;
  byte *pbVar5;
  byte *pbVar6;
  
  puVar2 = *(undefined8 **)(param_1 + 8);
  while( true ) {
    if (puVar2 == (undefined8 *)0x0) {
      return;
    }
    if (*(int *)(puVar2 + 3) == 0) break;
LAB_100002752:
    puVar2 = (undefined8 *)*puVar2;
  }
  pbVar6 = (byte *)puVar2[4];
  do {
    if (*pbVar6 == 0x25) {
      pbVar5 = pbVar6 + 1;
      do {
        pbVar6 = pbVar5;
        bVar1 = *pbVar6;
        uVar4 = (uint)bVar1;
        if (bVar1 == 0) {
                    /* WARNING: Subroutine does not return */
          error_missing_conversion_character();
        }
        pvVar3 = _memchr("#-+ 0123456789",uVar4,0xf);
        pbVar5 = pbVar6 + 1;
      } while (pvVar3 != (void *)0x0);
      if ((bVar1 == 0x2e) && (uVar4 = (uint)*pbVar5, pbVar6 = pbVar5, uVar4 - 0x30 < 10)) {
        _atoi((char *)pbVar5);
        do {
          uVar4 = (uint)pbVar5[1];
          pbVar5 = pbVar5 + 1;
          pbVar6 = pbVar5;
        } while (uVar4 - 0x30 < 10);
      }
      if (uVar4 < 0x5f) {
        if ((uVar4 == 0x45) || (uVar4 == 0x47)) goto switchD_100002709_caseD_65;
        if (uVar4 == 0x58) goto switchD_100002709_caseD_64;
      }
      else {
        switch(uVar4) {
        case 0x5f:
          pbVar5 = pbVar6 + 1;
          pbVar6 = pbVar6 + 1;
          if ((*pbVar5 - 99 < 0x13) && ((0x42001U >> (*pbVar5 - 99 & 0x1f) & 1) != 0))
          goto switchD_100002709_caseD_63;
          break;
        case 99:
switchD_100002709_caseD_63:
          break;
        case 100:
        case 0x69:
        case 0x6f:
        case 0x75:
        case 0x78:
switchD_100002709_caseD_64:
          break;
        case 0x65:
        case 0x66:
        case 0x67:
switchD_100002709_caseD_65:
          break;
        case 0x73:
        }
      }
    }
    else if (*pbVar6 == 0) goto LAB_100002752;
    pbVar6 = pbVar6 + 1;
  } while( true );
}



/**
 * @name  compile_format_descriptors
 * @brief Parses printf-style format strings and builds linked list of format descriptors with width calculations and whitespace trimming for hexdump display formatting
 * @confidence 85%
 * @classification parser
 * @address 0x1000027e7
 */

/* Parses printf-style format strings and builds linked list of format descriptors with width
   calculations and whitespace trimming for hexdump display formatting */

void compile_format_descriptors(long param_1)

{
  bool bVar1;
  undefined *puVar2;
  bool bVar3;
  undefined4 uVar4;
  uint uVar5;
  long *plVar6;
  void *pvVar7;
  long *plVar8;
  byte bVar9;
  int iVar10;
  long *plVar11;
  char *pcVar12;
  byte *pbVar13;
  byte *pbVar14;
  uint uVar15;
  int local_4c;
  
  plVar11 = *(long **)(param_1 + 8);
  if (plVar11 != (long *)0x0) {
    local_4c = 0;
    do {
      if (*(byte *)plVar11[4] != 0) {
        bVar1 = false;
        plVar8 = plVar11 + 1;
        pcVar12 = (char *)plVar11[4];
        do {
          plVar6 = (long *)_malloc_type_calloc(1,0xb0,0x10300405d1070d2);
          if (plVar6 == (long *)0x0) {
LAB_100002ce7:
                    /* WARNING: Subroutine does not return */
            FUN_100002e23();
          }
          *plVar8 = (long)plVar6;
          pbVar14 = (byte *)pcVar12;
          pbVar13 = (byte *)pcVar12;
          while( true ) {
            pbVar13 = pbVar13 + 1;
            if (*pbVar14 == 0) {
              plVar6[3] = (long)pcVar12;
              *(undefined4 *)(plVar6 + 1) = 0x400;
              goto LAB_100002bf7;
            }
            if (*pbVar14 == 0x25) break;
            pbVar14 = pbVar14 + 1;
          }
          uVar5 = *(uint *)(plVar11 + 3);
          if (uVar5 == 0) {
            do {
              pbVar14 = pbVar13;
              bVar9 = *pbVar14;
              uVar15 = (uint)bVar9;
              if (bVar9 == 0) goto LAB_100002ce2;
              pvVar7 = _memchr("#-+ 0123456789",uVar15,0xf);
              pbVar13 = pbVar14 + 1;
            } while (pvVar7 != (void *)0x0);
            iVar10 = 0;
            if ((bVar9 == 0x2e) && (uVar15 = (uint)*pbVar13, pbVar14 = pbVar13, uVar15 - 0x30 < 10))
            {
              local_4c = _atoi((char *)pbVar13);
              do {
                uVar15 = (uint)pbVar13[1];
                pbVar13 = pbVar13 + 1;
              } while (uVar15 - 0x30 < 10);
              iVar10 = 2;
              pbVar14 = pbVar13;
            }
          }
          else {
            do {
              uVar15 = (uint)pbVar14[1];
              if (pbVar14[1] == 0) {
                error_missing_conversion_character();
LAB_100002ce2:
                error_missing_conversion_character();
                goto LAB_100002ce7;
              }
              pbVar14 = pbVar14 + 1;
              pvVar7 = _memchr(".#-+ 0123456789",uVar15,0x10);
            } while (pvVar7 != (void *)0x0);
            iVar10 = 1;
          }
          pbVar13 = pbVar14 + 1;
          if ((char)uVar15 == '\0') {
            pbVar13 = pbVar14;
          }
          if (uVar15 < 0x5f) {
            if ((uVar15 == 0x45) || (uVar15 == 0x47)) goto switchD_100002963_caseD_65;
            if (uVar15 == 0x58) goto switchD_100002963_caseD_6f;
switchD_100002963_caseD_60:
            pbVar14[1] = 0;
LAB_100002d09:
            pcVar12 = "%%%s: bad conversion character";
            goto LAB_100002d10;
          }
          switch(uVar15) {
          case 0x5f:
            pbVar13 = pbVar14 + 2;
            bVar9 = pbVar14[1];
            if (0x6d < bVar9) {
              if (bVar9 != 0x6e) {
                if (bVar9 == 0x70) {
                  *(undefined4 *)(plVar6 + 1) = 0x40;
                }
                else {
                  if (bVar9 != 0x75) goto LAB_100002d2b;
                  *(undefined4 *)(plVar6 + 1) = 0x100;
                }
LAB_100002ad3:
                if (1 < uVar5) goto LAB_100002d24;
                goto switchD_1000029b2_caseD_1;
              }
              trailing_format_element = plVar11;
              *(undefined4 *)(plVar11 + 2) = 1;
              *(undefined4 *)(plVar6 + 1) = 0x400;
              pcVar12 = "\n";
              goto LAB_100002ae5;
            }
            if (bVar9 == 0x41) {
              trailing_format_element = plVar11;
              *(byte *)(plVar11 + 2) = *(byte *)(plVar11 + 2) | 1;
LAB_100002b7e:
              *(undefined4 *)(plVar6 + 1) = 1;
              if ((0x14 < *pbVar13 - 100) || ((0x100801U >> (*pbVar13 - 100 & 0x1f) & 1) == 0)) {
                    /* WARNING: Subroutine does not return */
                handle_bad_conversion_character(pbVar14);
              }
              pbVar13 = pbVar14 + 3;
              goto LAB_100002ae5;
            }
            if (bVar9 == 0x61) goto LAB_100002b7e;
            if (bVar9 == 99) {
              *(undefined4 *)(plVar6 + 1) = 4;
              goto LAB_100002ad3;
            }
LAB_100002d2b:
            pbVar14[2] = 0;
            goto LAB_100002d09;
          default:
            goto switchD_100002963_caseD_60;
          case 99:
            *(undefined4 *)(plVar6 + 1) = 8;
            if (uVar5 < 2) goto switchD_1000029b2_caseD_1;
            goto switchD_1000029b2_caseD_3;
          case 100:
          case 0x69:
            uVar4 = 0x20;
            break;
          case 0x65:
          case 0x66:
          case 0x67:
switchD_100002963_caseD_65:
            *(undefined4 *)(plVar6 + 1) = 0x10;
            switch(uVar5 << 0x1e | uVar5 >> 2) {
            case 0:
            case 2:
              goto switchD_1000029b2_caseD_8;
            case 1:
              goto switchD_1000029b2_caseD_0;
            default:
              goto switchD_1000029b2_caseD_3;
            case 4:
              *(undefined4 *)((long)plVar6 + 0xc) = 0x10;
              goto LAB_100002ae5;
            }
          case 0x6f:
          case 0x75:
          case 0x78:
switchD_100002963_caseD_6f:
            uVar4 = 0x200;
            break;
          case 0x73:
            *(undefined4 *)(plVar6 + 1) = 0x80;
            if (iVar10 == 1) {
              *(uint *)((long)plVar6 + 0xc) = uVar5;
              goto LAB_100002ae5;
            }
            if (iVar10 == 2) {
              *(int *)((long)plVar6 + 0xc) = local_4c;
              goto LAB_100002ae5;
            }
            exit_with_precision_error();
LAB_100002d24:
            pbVar14[2] = 0;
            goto LAB_100002cfb;
          }
          *(undefined4 *)(plVar6 + 1) = uVar4;
          switch(uVar5) {
          case 0:
          case 4:
switchD_1000029b2_caseD_0:
            *(undefined4 *)((long)plVar6 + 0xc) = 4;
            break;
          case 1:
switchD_1000029b2_caseD_1:
            *(undefined4 *)((long)plVar6 + 0xc) = 1;
            break;
          case 2:
            *(undefined4 *)((long)plVar6 + 0xc) = 2;
            break;
          default:
            goto switchD_1000029b2_caseD_3;
          case 8:
switchD_1000029b2_caseD_8:
            *(undefined4 *)((long)plVar6 + 0xc) = 8;
          }
LAB_100002ae5:
          bVar9 = *pbVar13;
          *pbVar14 = 0;
          iVar10 = _asprintf((char **)(plVar6 + 3),"%s%s",pcVar12);
          if (iVar10 == -1) {
                    /* WARNING: Subroutine does not return */
            FUN_100002e23();
          }
          *pbVar13 = bVar9;
          pbVar14 = pbVar14 + (plVar6[3] - (long)pcVar12);
          plVar6[2] = (long)pbVar14;
          bVar3 = bVar1;
          if ((((*(byte *)(plVar6 + 1) & 1) == 0) && ((int)plVar11[3] != 0)) &&
             (bVar3 = true, bVar1)) {
            error_multiple_conversion_characters();
switchD_1000029b2_caseD_3:
            pbVar14[1] = 0;
LAB_100002cfb:
            pcVar12 = "%s: bad byte count";
LAB_100002d10:
                    /* WARNING: Subroutine does not return */
            _errx(1,pcVar12,pbVar14);
          }
          bVar1 = bVar3;
          plVar8 = plVar6;
          pcVar12 = (char *)pbVar13;
        } while (*pbVar13 != 0);
      }
LAB_100002bf7:
      if (((int)plVar11[3] == 0) && (plVar8 = (long *)plVar11[1], plVar8 != (long *)0x0)) {
        iVar10 = 0;
        do {
          iVar10 = iVar10 + *(int *)((long)plVar8 + 0xc);
          plVar8 = (long *)*plVar8;
        } while (plVar8 != (long *)0x0);
        *(int *)(plVar11 + 3) = iVar10;
      }
      puVar2 = PTR___DefaultRuneLocale_100004000;
      plVar11 = (long *)*plVar11;
    } while (plVar11 != (long *)0x0);
    for (plVar11 = *(long **)(param_1 + 8); plVar11 != (long *)0x0; plVar11 = (long *)*plVar11) {
      if (((*plVar11 == 0) &&
          (iVar10 = max_record_size - *(int *)(param_1 + 0x10),
          iVar10 != 0 && *(int *)(param_1 + 0x10) <= max_record_size)) &&
         (((*(byte *)(plVar11 + 2) & 2) == 0 && ((int)plVar11[3] != 0)))) {
        *(int *)((long)plVar11 + 0x14) = *(int *)((long)plVar11 + 0x14) + iVar10 / (int)plVar11[3];
      }
      if (1 < *(int *)((long)plVar11 + 0x14)) {
        plVar8 = (long *)plVar11[1];
        do {
          plVar6 = plVar8;
          plVar8 = (long *)*plVar6;
        } while ((long *)*plVar6 != (long *)0x0);
        bVar9 = *(byte *)plVar6[3];
        if (bVar9 != 0) {
          pbVar13 = (byte *)plVar6[3] - 1;
          do {
            if ((char)bVar9 < '\0') {
              uVar5 = ___maskrune((uint)bVar9,0x4000);
            }
            else {
              uVar5 = *(uint *)(puVar2 + (ulong)bVar9 * 4 + 0x3c) & 0x4000;
            }
            bVar9 = pbVar13[2];
            pbVar13 = pbVar13 + 1;
          } while (bVar9 != 0);
          if (uVar5 != 0) {
            plVar6[4] = (long)pbVar13;
          }
        }
      }
    }
  }
  return;
}



/* ==================== Utilities ==================== */

/**
 * @name  format_character_for_display
 * @brief Formats a single character for display, handling control characters and special cases with human-readable representations
 * @confidence 85%
 * @classification utility
 * @address 0x100000853
 */

/* Formats a single character for display, handling control characters and special cases with
   human-readable representations */

void format_character_for_display(long param_1,byte *param_2)

{
  byte bVar1;
  uint uVar2;
  char *pcVar3;
  char *pcVar4;
  
  bVar1 = *param_2;
  if ((ulong)bVar1 < 0x20) {
    **(undefined1 **)(param_1 + 0x10) = 0x73;
    pcVar4 = *(char **)(param_1 + 0x18);
    if (*param_2 == 10 && canonical_display_mode != 0) {
      pcVar3 = "nl";
    }
    else {
      pcVar3 = &DAT_1000030c4 + *(int *)(&DAT_1000030c4 + (ulong)*param_2 * 4);
    }
  }
  else if (bVar1 == 0x7f) {
    **(undefined1 **)(param_1 + 0x10) = 0x73;
    pcVar4 = *(char **)(param_1 + 0x18);
    pcVar3 = "del";
  }
  else {
    if ((bVar1 != 0x20) || (canonical_display_mode == 0)) {
      if ((char)bVar1 < '\0') {
        uVar2 = ___maskrune((uint)bVar1,0x40000);
      }
      else {
        uVar2 = *(uint *)(PTR___DefaultRuneLocale_100004000 + (ulong)bVar1 * 4 + 0x3c) & 0x40000;
      }
      if (uVar2 == 0) {
        **(undefined1 **)(param_1 + 0x10) = 0x78;
      }
      else {
        **(undefined1 **)(param_1 + 0x10) = 99;
      }
      _printf(*(char **)(param_1 + 0x18),(ulong)*param_2);
      return;
    }
    **(undefined1 **)(param_1 + 0x10) = 0x73;
    pcVar4 = *(char **)(param_1 + 0x18);
    pcVar3 = " sp";
  }
  _printf(pcVar4,pcVar3);
  return;
}



/**
 * @name  assert_conv_c_format_check
 * @brief Assertion function that checks if a format string equals "%3C" in the conv_c function at line 153 of conv.c
 * @confidence 95%
 * @classification utility
 * @address 0x100002ddc
 */

/* Assertion function that checks if a format string equals "%3C" in the conv_c function at line 153
   of conv.c */

void assert_conv_c_format_check(void)

{
                    /* WARNING: Subroutine does not return */
  ___assert_rtn("conv_c","conv.c",0x99,"strcmp(pr->fmt, \"%3C\") == 0");
}



/**
 * @name  assert_width_nonnegative
 * @brief Assertion failure handler that reports a failed assertion about width being non-negative
 * @confidence 100%
 * @classification utility
 * @address 0x100002dff
 */

/* Assertion failure handler that reports a failed assertion about width being non-negative */

void assert_width_nonnegative(void)

{
                    /* WARNING: Subroutine does not return */
  ___assert_rtn("conv_c","conv.c",0x9b,"width >= 0");
}



/**
 * @name  warn_format_string
 * @brief Outputs a warning message using a format string retrieved from a global data pointer
 * @confidence 85%
 * @classification utility
 * @address 0x100002e33
 */

/* Outputs a warning message using a format string retrieved from a global data pointer */

void warn_format_string(void)

{
  _warn("%s",*(undefined8 *)(current_input_files - 8));
  return;
}


