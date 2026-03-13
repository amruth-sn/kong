/* ============================================================
 * Binary:   minitest_stripped
 * Arch:     AARCH64
 * Format:   Mac OS X Mach-O
 * Compiler: default
 *
 * Functions: 14 total, 14 analyzed, 0 skipped, 0 errors
 * Renamed:   0 | Confirmed: 14
 * LLM calls: 14
 * Duration:  0m 26.1s
 * Cost:      $0.0527
 * ============================================================ */

/* ==================== Crypto ==================== */

/**
 * @name  xor_buffer
 * @brief Performs in-place XOR encryption/decryption on a memory buffer using a single-byte key. Iterates through each byte and XORs it with the provided key.
 * @confidence 95%
 * @classification crypto
 * @address 0x100000b38
 */

/* WARNING: Unknown calling convention -- yet parameter storage is locked */
/* Performs in-place XOR encryption/decryption on a memory buffer using a single-byte key. Iterates
   through each byte in the buffer and XORs it with the provided key byte. */

void xor_buffer(long buffer,int length,byte xor_key)

{
  int local_14;
  
  for (local_14 = 0; local_14 < length; local_14 = local_14 + 1) {
    *(byte *)(buffer + local_14) = *(byte *)(buffer + local_14) ^ xor_key;
  }
  return;
}



/**
 * @name  rot13_encode
 * @brief Applies ROT13 cipher encoding to a null-terminated string in-place. Rotates lowercase letters by 13 positions (a-z wrapping), rotates uppercase letters by 13 positions (A-Z wrapping), and leaves non-alphabetic characters unchanged.
 * @confidence 92%
 * @classification crypto
 * @address 0x100000b9c
 */

/* WARNING: Unknown calling convention -- yet parameter storage is locked */
/* Applies ROT13 cipher encoding to a null-terminated string in-place, rotating alphabetic
   characters by 13 positions while preserving case and leaving non-alphabetic characters unchanged
    */

void rot13_encode(char *str)

{
  char *local_8;
  
  for (local_8 = str; *local_8 != '\0'; local_8 = local_8 + 1) {
    if ((*local_8 < 'a') || ('z' < *local_8)) {
      if (('@' < *local_8) && (*local_8 < '[')) {
        *local_8 = (char)((*local_8 - 0x34) % 0x1a) + 'A';
      }
    }
    else {
      *local_8 = (char)((*local_8 - 0x54) % 0x1a) + 'a';
    }
  }
  return;
}



/* ==================== I/O ==================== */

/**
 * @name  print_linked_list_items
 * @brief Iterates through a linked list starting from a global variable and calls printf to print each node, but has a format string bug - printf expects 2 arguments but none are provided.
 * @confidence 75%
 * @classification io
 * @address 0x100000a18
 */

/* WARNING: Unknown calling convention -- yet parameter storage is locked */
/* Iterates through a linked list starting from a global variable and calls printf to print each
   node, but has a format string bug - printf expects 2 arguments (%d and %s) but none are provided
    */

uint print_linked_list_items(uint param_1)

{
  long local_18;
  
  for (local_18 = linked_list_head; local_18 != 0; local_18 = *(long *)(local_18 + 0x10)) {
    param_1 = _printf("  [%d] = \"%s\"\n");
  }
  return param_1;
}



/* ==================== Memory Management ==================== */

/**
 * @name  remove_entry_from_list
 * @brief Searches a linked list for an entry with matching entry_id, unlinks it from the list, and frees both the entry node and its associated data. Returns 1 on success, 0 if entry not found.
 * @confidence 85%
 * @classification memory
 * @address 0x100000dc8
 */

/* WARNING: Unknown calling convention -- yet parameter storage is locked */
/* Searches a linked list for an entry with matching entry_id, unlinks it from the list, and frees
   both the entry node and its associated data. Returns 1 on success, 0 if entry not found. */

int remove_entry_from_list(int entry_id)

{
  void *pvVar1;
  long *local_20;
  
  local_20 = &linked_list_head;
  while( true ) {
    if (*local_20 == 0) {
      return 0;
    }
    if (*(int *)*local_20 == entry_id) break;
    local_20 = (long *)(*local_20 + 0x10);
  }
  pvVar1 = (void *)*local_20;
  *local_20 = *(long *)((long)pvVar1 + 0x10);
  _free(*(void **)((long)pvVar1 + 8));
  _free(pvVar1);
  return 1;
}



/**
 * @name  create_node_or_entry
 * @brief Allocates a 24-byte structure, initializes it with an enum/int value at offset 0, a duplicated string at offset 8, and sets a null pointer at offset 16. Returns the allocated structure or NULL on malloc failure.
 * @confidence 88%
 * @classification memory
 * @address 0x100000ee4
 */

/* WARNING: Unknown calling convention -- yet parameter storage is locked */
/* Allocates a 24-byte structure, initializes it with an enum/int value, a duplicated string, and a
   null pointer. Returns the allocated structure or NULL on malloc failure. */

undefined4 * create_node_or_entry(undefined4 param_1,char *param_2)

{
  char *pcVar1;
  undefined4 *local_18;
  
  local_18 = _malloc(0x18);
  if (local_18 == (undefined4 *)0x0) {  }
  else {
    *local_18 = param_1;
    pcVar1 = _strdup(param_2);
    *(char **)(local_18 + 2) = pcVar1;
    *(undefined8 *)(local_18 + 4) = 0;
  }
  return local_18;
}



/* ==================== String Operations ==================== */

/**
 * @name  is_palindrome
 * @brief Checks if a string is a palindrome by comparing characters from both ends moving inward toward the middle. Returns 1 if palindrome, 0 otherwise.
 * @confidence 95%
 * @classification string
 * @address 0x100000a8c
 */

/* WARNING: Unknown calling convention -- yet parameter storage is locked */
/* Checks if a string is a palindrome by comparing characters from both ends moving inward toward
   the middle. Returns 1 if palindrome, 0 otherwise. */

int is_palindrome(char *str)

{
  size_t sVar1;
  int local_28;
  
  sVar1 = _strlen(str);
  local_28 = 0;
  while( true ) {
    if ((int)sVar1 / 2 <= local_28) {
      return 1;
    }
    if (str[local_28] != str[((int)sVar1 - 1) - local_28]) break;
    local_28 = local_28 + 1;
  }
  return 0;
}



/**
 * @name  string_reverse
 * @brief Creates a new dynamically allocated string containing the reverse of the input string. Allocates memory, reverses the characters in a loop, null-terminates the result, and returns the new string.
 * @confidence 95%
 * @classification string
 * @address 0x100000c88
 */

/* WARNING: Unknown calling convention -- yet parameter storage is locked */
/* Creates a new dynamically allocated string containing the reverse of the input string */

char * string_reverse(char *str)

{
  size_t sVar1;
  int iVar2;
  int local_34;
  char *local_18;
  
  sVar1 = _strlen(str);
  iVar2 = (int)sVar1;
  local_18 = _malloc((long)(iVar2 + 1));
  if (local_18 == (char *)0x0) {  }
  else {
    for (local_34 = 0; local_34 < iVar2; local_34 = local_34 + 1) {
      local_18[local_34] = str[(iVar2 - 1) - local_34];
    }
    local_18[iVar2] = '\0';
  }
  return local_18;
}



/* ==================== Initialization ==================== */

/**
 * @name  main
 * @brief Main entry point that demonstrates hash tables with linked list collision handling, string operations (palindrome checking, ROT13, XOR encoding), and various utility functions
 * @confidence 95%
 * @classification init
 * @address 0x100000548
 */

/* WARNING: Unknown calling convention -- yet parameter storage is locked */
/* Main entry point that demonstrates various data structures and algorithms including hash tables,
   linked lists, string manipulation, and encoding/decoding operations */

int main(void)

{
  uint uVar1;
  int iVar2;
  size_t sVar3;
  char *pcVar4;
  int *piVar5;
  int local_cc;
  int local_c8;
  int local_c0;
  char local_a8 [8];
  char local_a0 [8];
  char acStack_98 [64];
  undefined8 auStack_58 [6];
  long local_28;
  
  local_28 = *(long *)PTR____stack_chk_guard_100004000;
  _memcpy(auStack_58,&PTR_s_hello_100004048,0x30);
  for (local_c0 = 0; local_c0 < 6; local_c0 = local_c0 + 1) {
    uVar1 = hash_string((byte *)auStack_58[local_c0]);
    linked_list_insert_head(uVar1 % 1000,(char *)auStack_58[local_c0]);
  }
  count_linked_list_nodes();
  sum_linked_list();
  uVar1 = _printf("List (%d nodes, sum=%d):\n");
  print_linked_list_items(uVar1);
  for (local_c8 = 0; local_c8 < 6; local_c8 = local_c8 + 1) {
    iVar2 = is_palindrome((char *)auStack_58[local_c8]);
    if (iVar2 != 0) {
      _printf("  palindrome: %s\n");
    }
  }
  ___strncpy_chk(local_c8 - 6,acStack_98,"secret message",0x40);
  sVar3 = _strlen(acStack_98);
  xor_buffer((long)acStack_98,(int)sVar3,0x42);
  _printf("Encoded: ");
  for (local_cc = 0; local_cc < 0xe; local_cc = local_cc + 1) {
    _printf("%02x ");
  }
  _printf("\n",(ulong)(local_cc - 0xe));
  xor_buffer((long)acStack_98,0xe,0x42);
  _printf("Decoded: %s\n");
  local_a8[0] = s_Hello_World_100001036[0];
  local_a8[1] = s_Hello_World_100001036[1];
  local_a8[2] = s_Hello_World_100001036[2];
  local_a8[3] = s_Hello_World_100001036[3];
  local_a8[4] = s_Hello_World_100001036[4];
  local_a8[5] = s_Hello_World_100001036[5];
  local_a8[6] = s_Hello_World_100001036[6];
  local_a8[7] = s_Hello_World_100001036[7];
  local_a0[0] = s_Hello_World_100001036[8];
  local_a0[1] = s_Hello_World_100001036[9];
  local_a0[2] = s_Hello_World_100001036[10];
  local_a0[3] = s_Hello_World_100001036[0xb];
  rot13_encode(local_a8);
  _printf("ROT13: %s\n");
  pcVar4 = string_reverse("kong");
  if (pcVar4 != (char *)0x0) {
    _printf("Reversed: %s\n");
    _free(pcVar4);
  }
  uVar1 = hash_string((byte *)"world");
  piVar5 = linked_list_search(uVar1 % 1000);
  if (piVar5 != (int *)0x0) {
    _printf("Found: [%d] = %s\n");
  }
  uVar1 = hash_string((byte *)"hello");
  remove_entry_from_list(uVar1 % 1000);
  count_linked_list_nodes();
  uVar1 = _printf("After remove (%d nodes):\n");
  print_linked_list_items(uVar1);
  free_linked_list();
  if (*(long *)PTR____stack_chk_guard_100004000 - local_28 != 0) {
                    /* WARNING: Subroutine does not return */
    ___stack_chk_fail(*(long *)PTR____stack_chk_guard_100004000 - local_28);
  }
  return 0;
}



/* ==================== Cleanup ==================== */

/**
 * @name  free_linked_list
 * @brief Iterates through a linked list structure, freeing the data pointer (at offset 8) and node structure (at offset 0) at each step, then resets the global head pointer to NULL.
 * @confidence 95%
 * @classification cleanup
 * @address 0x100000e7c
 */

/* WARNING: Unknown calling convention -- yet parameter storage is locked */
/* Iterates through a linked list structure, freeing the data pointer and node structure at each
   step, then resets the global head pointer to NULL */

void free_linked_list(void)

{
  void *pvVar1;
  long local_18;
  
  local_18 = linked_list_head;
  while (local_18 != (void *)0x0) {
    pvVar1 = *(void **)((long)local_18 + 0x10);
    _free(*(void **)((long)local_18 + 8));
    _free(local_18);
    local_18 = pvVar1;
  }
  linked_list_head = (void *)0x0;
  return;
}



/* ==================== Utilities ==================== */

/**
 * @name  hash_string
 * @brief Computes a hash of a null-terminated string using a linear congruential-style algorithm with XOR mixing. Initializes with 0x1505, multiplies by 0x21, and XORs with each character.
 * @confidence 90%
 * @classification utility
 * @address 0x1000008bc
 */

/* WARNING: Unknown calling convention -- yet parameter storage is locked */
/* Computes a hash of a null-terminated string using a linear congruential-style algorithm with XOR
   mixing. Initializes accumulator to 0x1505, then iterates through each character, multiplying by
   0x21 and XORing with the character value. */

uint hash_string(byte *str)

{
  uint local_c;
  byte *local_8;
  
  local_c = 0x1505;
  for (local_8 = str; *local_8 != 0; local_8 = local_8 + 1) {
    local_c = local_c * 0x21 ^ (uint)*local_8;
  }
  return local_c;
}



/**
 * @name  linked_list_insert_head
 * @brief Creates a new node and inserts it at the head of a singly-linked list. Updates the new node's next pointer to point to the current head, then updates the global head pointer.
 * @confidence 92%
 * @classification utility
 * @address 0x100000914
 */

/* WARNING: Unknown calling convention -- yet parameter storage is locked */
/* Creates a new node and inserts it at the head of a singly-linked list. Updates the new node's
   next pointer to point to the current head, then updates the global head pointer to the new node.
    */

void linked_list_insert_head(undefined4 param_1,char *param_2)

{
  undefined4 *puVar1;
  
  puVar1 = create_node_or_entry(param_1,param_2);
  if (puVar1 != (undefined4 *)0x0) {
    *(undefined4 **)(puVar1 + 4) = linked_list_head;
    linked_list_head = puVar1;
  }
  return;
}



/**
 * @name  count_linked_list_nodes
 * @brief Counts the total number of nodes in a globally-referenced singly linked list by following next pointers at offset 0x10 until NULL is reached.
 * @confidence 95%
 * @classification utility
 * @address 0x100000970
 */

/* WARNING: Unknown calling convention -- yet parameter storage is locked */
/* Traverses a linked list starting from a global data pointer and counts the total number of nodes
   by following next pointers at offset 0x10 until a null pointer is encountered. */

int count_linked_list_nodes(void)

{
  long local_10;
  int local_4;
  
  local_4 = 0;
  for (local_10 = linked_list_head; local_10 != 0; local_10 = *(long *)(local_10 + 0x10)) {
    local_4 = local_4 + 1;
  }
  return local_4;
}



/**
 * @name  sum_linked_list
 * @brief Traverses a globally-referenced singly linked list and returns the sum of integer values stored at the beginning of each node. The linked list is terminated by NULL at offset 4.
 * @confidence 95%
 * @classification utility
 * @address 0x1000009c0
 */

/* WARNING: Unknown calling convention -- yet parameter storage is locked */
/* Traverses a globally-referenced singly linked list and returns the sum of integer values stored
   at the beginning of each node. The linked list is terminated by a NULL pointer at offset 4 of the
   last node. */

int sum_linked_list(void)

{
  int *local_10;
  int local_4;
  
  local_4 = 0;
  for (local_10 = linked_list_head; local_10 != (int *)0x0; local_10 = *(int **)(local_10 + 4)) {
    local_4 = local_4 + *local_10;
  }
  return local_4;
}



/**
 * @name  linked_list_search
 * @brief Searches a linked list for a node containing a specific integer value by traversing from a global head pointer and following next pointers until a match is found or the list ends.
 * @confidence 92%
 * @classification utility
 * @address 0x100000d5c
 */

/* WARNING: Unknown calling convention -- yet parameter storage is locked */
/* Searches a linked list for a node containing a specific integer value by traversing from a global
   head pointer and following next pointers until a match is found or the list ends. */

int * linked_list_search(int search_value)

{
  int *local_18;
  
  local_18 = linked_list_head;
  while( true ) {
    if (local_18 == (int *)0x0) {
      return (int *)0x0;
    }
    if (*local_18 == search_value) break;
    local_18 = *(int **)(local_18 + 4);
  }
  return local_18;
}


