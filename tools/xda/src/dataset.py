import torch
from torch.utils.data import Dataset

from .tokenizer import ByteTokenizer


def chunk_labeled_bytes(
    text_bytes: bytes,
    labels: list[int],
    window_size: int = 512,
    stride: int = 256,
) -> list[dict]:
    """Chunk labeled bytes into fixed-size windows with overlap."""
    chunks = []
    for i in range(0, max(1, len(text_bytes) - window_size + 1), stride):
        end = i + window_size
        if end > len(text_bytes):
            # Pad the last chunk
            b = text_bytes[i:] + bytes(end - len(text_bytes))
            l = labels[i:] + [0] * (end - len(text_bytes))
        else:
            b = text_bytes[i:end]
            l = labels[i:end]
        chunks.append({"bytes": b, "labels": l})
    return chunks


class ByteDataset(Dataset):
    """PyTorch dataset for XDA training. Each item is a fixed-size byte window."""

    def __init__(self, chunks: list[dict], tokenizer: ByteTokenizer) -> None:
        self.chunks = chunks
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        chunk = self.chunks[idx]
        raw_bytes = chunk["bytes"]
        raw_labels = chunk["labels"]

        # Tokenize: [CLS] + bytes + [SEP]
        input_ids = self.tokenizer.encode(raw_bytes)
        attention_mask = self.tokenizer.attention_mask(input_ids)

        # Labels: -100 for special tokens (ignored by CrossEntropyLoss)
        labels = [-100] + list(raw_labels) + [-100]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }