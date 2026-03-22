from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .model import XdaConfig, XdaModel
from .tokenizer import ByteTokenizer


class MaskedByteDataset(Dataset):

    def __init__(
        self,
        chunks: list[dict],
        tokenizer: ByteTokenizer,
        mask_prob: float = 0.2,
        mask_token_ratio: float = 0.5,
    ) -> None:
        self.chunks = chunks
        self.tokenizer = tokenizer
        self.mask_prob = mask_prob
        self.mask_token_ratio = mask_token_ratio

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        raw_bytes = bytes(self.chunks[idx]["bytes"])
        input_ids = self.tokenizer.encode(raw_bytes)
        labels = list(input_ids)

        for i in range(1, len(input_ids) - 1):
            if random.random() < self.mask_prob:
                if random.random() < self.mask_token_ratio:
                    input_ids[i] = self.tokenizer.mask_id
                else:
                    input_ids[i] = random.randint(0, 255)
            else:
                labels[i] = -100

        labels[0] = -100
        labels[-1] = -100

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def pretrain(config_path: str, data_path: str) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    train_chunks = json.loads(Path(data_path, "train.json").read_text())
    print(f"Loaded {len(train_chunks)} training chunks")

    tokenizer = ByteTokenizer()
    dataset = MaskedByteDataset(
        train_chunks,
        tokenizer,
        mask_prob=cfg["training"]["mask_probability"],
        mask_token_ratio=cfg["training"]["mask_token_ratio"],
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    model_cfg = XdaConfig(
        hidden_size=cfg["model"]["hidden_size"],
        num_hidden_layers=cfg["model"]["num_hidden_layers"],
        num_attention_heads=cfg["model"]["num_attention_heads"],
        intermediate_size=cfg["model"]["intermediate_size"],
        num_labels=tokenizer.vocab_size,
    )
    model = XdaModel(model_cfg).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    total_steps = len(loader) * cfg["training"]["epochs"]
    warmup_steps = int(total_steps * cfg["training"]["warmup_ratio"])
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps,
    )

    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    scaler = torch.amp.GradScaler(enabled=cfg["training"]["fp16"] and device.type == "cuda")

    checkpoint_dir = Path(cfg["output"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(cfg["training"]["epochs"]):
        model.train()
        total_loss = 0.0
        steps = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{cfg['training']['epochs']}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast(device.type, enabled=cfg["training"]["fp16"]):
                logits = model(input_ids, attention_mask)
                loss = loss_fn(logits.view(-1, model_cfg.num_labels), labels.view(-1))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

            total_loss += loss.item()
            steps += 1
            pbar.set_postfix(loss=f"{total_loss / steps:.4f}")

        avg_loss = total_loss / steps
        print(f"Epoch {epoch + 1}: avg_loss={avg_loss:.4f}")

        if (epoch + 1) % cfg["output"]["save_every_n_epochs"] == 0:
            torch.save(model.state_dict(), checkpoint_dir / f"pretrain_epoch{epoch + 1}.pt")

    torch.save(model.state_dict(), checkpoint_dir / "pretrain_final.pt")
    print(f"Pre-training complete. Final checkpoint: {checkpoint_dir / 'pretrain_final.pt'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    args = parser.parse_args()
    pretrain(args.config, args.data)
