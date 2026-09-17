import torch
from sklearn.metrics import f1_score


def f1_macro_multilabel(logits: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5) -> float:
    preds = (torch.sigmoid(logits) > threshold).float()
    return f1_score(labels.numpy(), preds.numpy(), average="macro", zero_division=0)


def accuracy_single_label(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    target = labels.argmax(dim=-1) if labels.dim() > 1 else labels
    return (preds == target).float().mean().item()
