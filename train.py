"""Train DCP-CLIP on MM-IMDb under a missing-modality scenario.

Usage:
    python train.py --config configs/mmimdb.yaml
    python train.py --config configs/mmimdb.yaml data.missing_rate=0.5 train.epochs=10
"""
import argparse
import os
import random
import sys

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from transformers import CLIPProcessor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.datasets.mmimdb import MMIMDbDataset, make_collate_fn
from src.models import DCPCLIP
from src.engine import build_scheduler, train_one_epoch, evaluate


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path, overrides):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for kv in overrides:
        key, value = kv.split("=", 1)
        section, field = key.split(".")
        try:
            value = yaml.safe_load(value)
        except Exception:
            pass
        cfg[section][field] = value
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("overrides", nargs="*", help="dot.key=value overrides, e.g. data.missing_rate=0.5")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    dcfg, mcfg, tcfg = cfg["data"], cfg["model"], cfg["train"]

    set_seed(tcfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(tcfg["output_dir"], exist_ok=True)

    processor = CLIPProcessor.from_pretrained(mcfg["clip_name"])
    collate_fn = make_collate_fn(processor, max_text_len=dcfg["max_text_len"])

    train_set = MMIMDbDataset(
        dcfg["root"], "train", dcfg["scenario"], dcfg["missing_rate"], dcfg["seed"], dcfg["image_size"]
    )
    dev_set = MMIMDbDataset(
        dcfg["root"], "dev", dcfg["scenario"], dcfg["missing_rate"], dcfg["seed"], dcfg["image_size"]
    )
    test_set = MMIMDbDataset(
        dcfg["root"], "test", dcfg["scenario"], dcfg["missing_rate"], dcfg["seed"], dcfg["image_size"]
    )

    train_loader = DataLoader(
        train_set, batch_size=tcfg["batch_size"], shuffle=True, num_workers=tcfg["num_workers"],
        collate_fn=collate_fn, drop_last=True,
    )
    dev_loader = DataLoader(
        dev_set, batch_size=tcfg["batch_size"], shuffle=False, num_workers=tcfg["num_workers"], collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_set, batch_size=tcfg["batch_size"], shuffle=False, num_workers=tcfg["num_workers"], collate_fn=collate_fn
    )

    model = DCPCLIP(
        clip_name=mcfg["clip_name"],
        num_classes=mcfg["num_classes"],
        multi_label=mcfg["multi_label"],
        prompt_len_correlated=mcfg["prompt_len_correlated"],
        prompt_len_dynamic=mcfg["prompt_len_dynamic"],
        prompt_len_common=mcfg["prompt_len_common"],
        common_dim=mcfg["common_dim"],
        depth_J=mcfg["depth_J"],
        reduction=mcfg["reduction"],
    ).to(device)

    n_trainable = model.num_trainable_parameters()
    n_total = model.num_total_parameters()
    print(f"Trainable params: {n_trainable:,} ({100 * n_trainable / n_total:.2f}% of {n_total:,})")

    optimizer = torch.optim.Adam(model.trainable_parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    total_steps = tcfg["epochs"] * len(train_loader)
    scheduler = build_scheduler(optimizer, total_steps, tcfg["warmup_ratio"])

    best_metric = -1.0
    best_path = os.path.join(tcfg["output_dir"], "best.pt")

    for epoch in range(1, tcfg["epochs"] + 1):
        print(f"Epoch {epoch}/{tcfg['epochs']}")
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device,
            multi_label=mcfg["multi_label"], grad_clip=tcfg["grad_clip"], log_every=tcfg["log_every"],
        )
        print(f"  train_loss={train_loss:.4f}")

        if epoch % tcfg["eval_every_epoch"] == 0:
            dev_metrics = evaluate(model, dev_loader, device, multi_label=mcfg["multi_label"])
            metric_name, metric_val = next(iter(dev_metrics.items()))
            print(f"  dev {metric_name}={metric_val:.4f}")

            if metric_val > best_metric:
                best_metric = metric_val
                trainable_state = {k: v for k, v in model.state_dict().items() if "clip." not in k}
                torch.save({"model": trainable_state, "config": cfg, "epoch": epoch, "metric": best_metric}, best_path)
                print(f"  saved new best checkpoint ({metric_name}={best_metric:.4f}) -> {best_path}")

    print("Training done. Evaluating best checkpoint on test set...")
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"], strict=False)
    test_metrics = evaluate(model, test_loader, device, multi_label=mcfg["multi_label"])
    print(f"Test metrics: {test_metrics}")


if __name__ == "__main__":
    main()
