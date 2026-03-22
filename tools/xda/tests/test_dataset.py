import torch
from src.dataset import ByteDataset, chunk_labeled_bytes
from src.tokenizer import ByteTokenizer


def test_chunk_labeled_bytes():
    text_bytes = bytes(range(256)) * 4  # 1024 bytes
    labels = [0] * 1024
    labels[0] = 1    # function start at byte 0
    labels[512] = 1  # function start at byte 512

    chunks = chunk_labeled_bytes(text_bytes, labels, window_size=512, stride=256)

    assert len(chunks) > 0
    for chunk in chunks:
        assert len(chunk["bytes"]) == 512
        assert len(chunk["labels"]) == 512


def test_byte_dataset_shapes():
    text_bytes = bytes(range(256)) * 4
    labels = [0] * 1024
    labels[0] = 1

    chunks = chunk_labeled_bytes(text_bytes, labels, window_size=512, stride=256)
    ds = ByteDataset(chunks, ByteTokenizer())

    item = ds[0]
    assert item["input_ids"].shape == (514,)    # 512 + [CLS] + [SEP]
    assert item["attention_mask"].shape == (514,)
    assert item["labels"].shape == (514,)
    assert item["input_ids"][0].item() == 257    # [CLS]
    assert item["input_ids"][-1].item() == 258   # [SEP]
    # Labels for [CLS] and [SEP] should be -100 (ignored in loss)
    assert item["labels"][0].item() == -100
    assert item["labels"][-1].item() == -100


def test_byte_dataset_label_values():
    text_bytes = bytes([0x55, 0x48, 0x89, 0xe5, 0xc3, 0x00, 0x00, 0x00])
    labels = [1, 2, 2, 2, 2, 0, 0, 0]

    chunks = chunk_labeled_bytes(text_bytes, labels, window_size=8, stride=8)
    ds = ByteDataset(chunks, ByteTokenizer())

    item = ds[0]
    # Labels offset by 1 for [CLS] prefix
    assert item["labels"][1].item() == 1   # function_start
    assert item["labels"][2].item() == 2   # function_body
    assert item["labels"][6].item() == 0   # non-function
