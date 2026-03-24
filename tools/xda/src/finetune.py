from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .dataset import ByteDataset
from .model import XdaConfig, XdaModel
from .tokenizer import ByteTokenizer


def evaluate(model: XdaModel, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    tp = [0, 0, 0]
    fp = [0, 0, 0]
    fn = [0, 0, 0]

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, attention_mask)
            preds = logits.argmax(dim=-1)

            mask = labels != -100
            preds = preds[mask]
            labels = labels[mask]

            for cls in range(3):
                tp[cls] += ((preds == cls) & (labels == cls)).sum().item()
                fp[cls] += ((preds == cls) & (labels != cls)).sum().item()
                fn[cls] += ((preds != cls) & (labels == cls)).sum().item()

    results = {}
    for cls, name in enumerate(["non_function", "function_start", "function_body"]):
        precision = tp[cls] / (tp[cls] + fp[cls]) if tp[cls] + fp[cls] > 0 else 0
        recall = tp[cls] / (tp[cls] + fn[cls]) if tp[cls] + fn[cls] > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
        results[name] = {"precision": precision, "recall": recall, "f1": f1}

    return results


def finetune(config_path: str, data_path: str, arch: str = "unknown", resume_from: str | None = None) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    use_cuda = device.type == "cuda"
    print(f"Using device: {device}")

    train_chunks = json.loads(Path(data_path, "train.json").read_text())
    val_chunks = json.loads(Path(data_path, "val.json").read_text())
    print(f"Train: {len(train_chunks)} chunks, Val: {len(val_chunks)} chunks")

    label_counts = [0, 0, 0]
    for chunk in train_chunks:
        for label in chunk["labels"]:
            if 0 <= label <= 2:
                label_counts[label] += 1
    total = sum(label_counts)
    raw_weights = [total / (3 * c) if c > 0 else 1.0 for c in label_counts]
    class_weights = torch.tensor(
        [w ** 0.5 for w in raw_weights],
        dtype=torch.float32,
    ).to(device)
    print(f"Label counts: non_func={label_counts[0]}, func_start={label_counts[1]}, func_body={label_counts[2]}")
    print(f"Class weights (sqrt-dampened): {class_weights.tolist()}")

    tokenizer = ByteTokenizer()
    train_ds = ByteDataset(train_chunks, tokenizer)
    val_ds = ByteDataset(val_chunks, tokenizer)

    train_loader = DataLoader(
        train_ds, batch_size=cfg["training"]["batch_size"],
        shuffle=True, num_workers=4 if use_cuda else 0,
        pin_memory=use_cuda,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["training"]["batch_size"],
        num_workers=4 if use_cuda else 0,
    )

    model_cfg = XdaConfig(
        hidden_size=cfg["model"]["hidden_size"],
        num_hidden_layers=cfg["model"]["num_hidden_layers"],
        num_attention_heads=cfg["model"]["num_attention_heads"],
        intermediate_size=cfg["model"]["intermediate_size"],
        num_labels=3,
    )
    model = XdaModel(model_cfg)

    if resume_from and Path(resume_from).exists():
        print(f"Resuming from checkpoint: {resume_from}")
        state = torch.load(resume_from, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        print(f"Loaded full model state ({len(state)} tensors)")
    else:
        pretrained_path = cfg.get("pretrained_checkpoint")
        if pretrained_path and Path(pretrained_path).exists():
            print(f"Loading pre-trained weights from {pretrained_path}")
            state = torch.load(pretrained_path, map_location="cpu", weights_only=True)
            encoder_state = {k: v for k, v in state.items() if k.startswith("encoder.")}
            model.load_state_dict(encoder_state, strict=False)
            print(f"Loaded {len(encoder_state)} encoder weight tensors")

    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    total_steps = len(train_loader) * cfg["training"]["epochs"]
    warmup_steps = int(total_steps * cfg["training"]["warmup_ratio"])
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps,
    )

    loss_fn = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)
    scaler = torch.amp.GradScaler(enabled=cfg["training"]["fp16"] and use_cuda)

    checkpoint_dir = Path(cfg["output"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    log_dir = Path(cfg["output"].get("log_dir", "runs")) / arch
    writer = SummaryWriter(log_dir=str(log_dir))
    print(f"Tensorboard logs: {log_dir}")

    best_f1 = 0.0
    global_step = 0

    for epoch in range(cfg["training"]["epochs"]):
        model.train()
        total_loss = 0.0
        steps = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg['training']['epochs']}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast(device.type, enabled=cfg["training"]["fp16"]):
                logits = model(input_ids, attention_mask)
                loss = loss_fn(logits.view(-1, 3), labels.view(-1))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

            total_loss += loss.item()
            steps += 1
            global_step += 1
            pbar.set_postfix(loss=f"{total_loss / steps:.4f}")

            writer.add_scalar("train/loss", loss.item(), global_step)
            writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)

        avg_loss = total_loss / steps
        results = evaluate(model, val_loader, device)
        fs_f1 = results["function_start"]["f1"]

        writer.add_scalar("train/epoch_loss", avg_loss, epoch + 1)
        for name, metrics in results.items():
            writer.add_scalar(f"val/{name}_f1", metrics["f1"], epoch + 1)
            writer.add_scalar(f"val/{name}_precision", metrics["precision"], epoch + 1)
            writer.add_scalar(f"val/{name}_recall", metrics["recall"], epoch + 1)

        print(f"Epoch {epoch + 1}: loss={avg_loss:.4f}")
        for name, metrics in results.items():
            print(f"  {name}: P={metrics['precision']:.4f} R={metrics['recall']:.4f} F1={metrics['f1']:.4f}")

        if cfg["output"]["save_best"] and fs_f1 > best_f1:
            best_f1 = fs_f1
            torch.save(model.state_dict(), checkpoint_dir / "best.pt")
            print(f"  New best function_start F1: {fs_f1:.4f}")

        prev = checkpoint_dir / f"epoch{epoch}.pt"
        if prev.exists():
            prev.unlink()
        torch.save(model.state_dict(), checkpoint_dir / f"epoch{epoch + 1}.pt")

    config_out = {
        "vocab_size": model_cfg.vocab_size,
        "hidden_size": model_cfg.hidden_size,
        "num_hidden_layers": model_cfg.num_hidden_layers,
        "num_attention_heads": model_cfg.num_attention_heads,
        "intermediate_size": model_cfg.intermediate_size,
        "max_position_embeddings": model_cfg.max_position_embeddings,
        "num_labels": model_cfg.num_labels,
        "classifier_hidden": model_cfg.classifier_hidden,
    }
    (checkpoint_dir / "config.json").write_text(json.dumps(config_out, indent=2))

    writer.close()
    print(f"Fine-tuning complete. Best F1: {best_f1:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--arch", default="unknown")
    parser.add_argument("--resume", default=None, help="Resume from a full finetune checkpoint")
    args = parser.parse_args()
    finetune(args.config, args.data, args.arch, args.resume)
