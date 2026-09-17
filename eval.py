"""Standalone evaluation of a trained DCP-CLIP checkpoint on MM-IMDb.

Usage:
    python eval.py --config configs/mmimdb.yaml --checkpoint outputs/mmimdb/best.pt --split test
"""
import argparse
import os
import sys

import torch
import yaml
from torch.utils.data import DataLoader
from transformers import CLIPProcessor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.datasets.mmimdb import MMIMDbDataset, make_collate_fn
from src.models import DCPCLIP
from src.engine import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "dev", "test"])
    parser.add_argument("--missing_rate", type=float, default=None, help="override the missing rate for evaluation")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    dcfg, mcfg = cfg["data"], cfg["model"]
    if args.missing_rate is not None:
        dcfg["missing_rate"] = args.missing_rate

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = CLIPProcessor.from_pretrained(mcfg["clip_name"])
    collate_fn = make_collate_fn(processor, max_text_len=dcfg["max_text_len"])

    dataset = MMIMDbDataset(
        dcfg["root"], args.split, dcfg["scenario"], dcfg["missing_rate"], dcfg["seed"], dcfg["image_size"]
    )
    loader = DataLoader(dataset, batch_size=cfg["train"]["batch_size"], shuffle=False, collate_fn=collate_fn)

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

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"], strict=False)

    metrics = evaluate(model, loader, device, multi_label=mcfg["multi_label"])
    print(f"[{args.split}] scenario={dcfg['scenario']} missing_rate={dcfg['missing_rate']} -> {metrics}")


if __name__ == "__main__":
    main()
