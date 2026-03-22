from dataclasses import dataclass

@dataclass
class ByteTokenizer:
    """Byte-level tokenizer for XDA. Maps raw bytes 0-255 to token IDs 0-255.
    Special tokens occupy IDs 256-260: [PAD], [CLS], [SEP], [MASK], [UNK]."""
    pad_id: int = 256
    cls_id: int = 257
    sep_id: int = 258
    mask_id: int = 259
    unk_id: int = 260
    vocab_size: int = 261

    def encode(self, bytes: bytes, max_length: int | None = None) -> list[int]:
        """Encode a sequence of bytes into a list of token IDs.
        Args:
            bytes: The sequence of bytes to encode.
            max_length: The maximum length of the tokenized sequence. If None, no padding is performed.
        Returns:
            A list of token IDs.
        """
        ids = [self.cls_id] + list(bytes) + [self.sep_id]
        if max_length is not None:
            ids = ids[:max_length]
            ids += [self.pad_id] * (max_length - len(ids))
        return ids

    def attention_mask(self, token_ids: list[int]) -> list[int]:
        return [0 if t == self.pad_id else 1 for t in token_ids]