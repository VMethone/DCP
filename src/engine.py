"""Shared train/eval step logic used by both train.py and eval.py."""
import torch
import torch.nn as nn

from .utils.metrics import f1_macro_multilabel, accuracy_single_label


def build_scheduler(optimizer, total_steps: int, warmup_ratio: float):
    warmup_steps = max(1, int(total_steps * warmup_ratio))

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 1.0 - progress)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def move_batch(batch, device):
    return {
        "pixel_values": batch["pixel_values"].to(device),
        "input_ids": batch["input_ids"].to(device),
        "attention_mask": batch["attention_mask"].to(device),
        "missing_type_ids": batch["missing_type_ids"].to(device),
    }


def train_one_epoch(model, loader, optimizer, scheduler, device, multi_label=True, grad_clip=1.0, log_every=50, log_fn=print):
    model.train()
    criterion = nn.BCEWithLogitsLoss() if multi_label else nn.CrossEntropyLoss()
    running_loss = 0.0

    for step, batch in enumerate(loader):
        inputs = move_batch(batch, device)
        labels = batch["labels"].to(device)

        logits = model(**inputs)
        if multi_label:
            loss = criterion(logits, labels)
        else:
            loss = criterion(logits, labels.argmax(dim=-1))

        optimizer.zero_grad()
        loss.backward()
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.trainable_parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()
        if log_every and (step + 1) % log_every == 0:
            log_fn(f"  step {step + 1}/{len(loader)}  loss={running_loss / (step + 1):.4f}  lr={scheduler.get_last_lr()[0]:.2e}")

    return running_loss / max(1, len(loader))


@torch.no_grad()
def evaluate(model, loader, device, multi_label=True):
    model.eval()
    all_logits, all_labels = [], []
    for batch in loader:
        inputs = move_batch(batch, device)
        logits = model(**inputs)
        all_logits.append(logits.cpu())
        all_labels.append(batch["labels"])

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    if multi_label:
        return {"f1_macro": f1_macro_multilabel(logits, labels)}
    return {"accuracy": accuracy_single_label(logits, labels)}
