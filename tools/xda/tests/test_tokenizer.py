from src.tokenizer import ByteTokenizer

def test_special_token_ids():
    tok = ByteTokenizer()
    assert tok.pad_id == 256
    assert tok.cls_id == 257
    assert tok.sep_id == 258
    assert tok.mask_id == 259
    assert tok.unk_id == 260
    assert tok.vocab_size == 261

def test_encode_bytes():
    tok = ByteTokenizer()
    raw = bytes([0x55, 0x48, 0x89, 0xe5])
    ids = tok.encode(raw)
    # [CLS] + byte IDs + [SEP]
    assert ids == [257, 0x55, 0x48, 0x89, 0xe5, 258]

def test_encode_with_padding():
    tok = ByteTokenizer()
    raw = bytes([0x55, 0x48])
    ids = tok.encode(raw, max_length=8)
    assert ids == [257, 0x55, 0x48, 258, 256, 256, 256, 256]
    assert len(ids) == 8

def test_attention_mask():
    tok = ByteTokenizer()
    raw = bytes([0x55, 0x48])
    ids = tok.encode(raw, max_length=8)
    mask = tok.attention_mask(ids)
    assert mask == [1, 1, 1, 1, 0, 0, 0, 0]