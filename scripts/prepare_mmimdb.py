"""Sanity-checks a downloaded MM-IMDb dataset directory before training.

This script does NOT download the dataset (it is a large, manually-licensed
release). Get it from http://lisi1.unal.edu.co/mmimdb/ (or an equivalent
mirror) and extract it so that:

    <root>/dataset/<id>.json
    <root>/dataset/<id>.jpeg
    <root>/split.json   -> {"train": [...], "dev": [...], "test": [...]}

If your copy only ships a flat list of ids with genre labels (no
split.json), see `build_split_json()` below for a quick way to create one
from an 80/10/10 random split.

Usage:
    python scripts/prepare_mmimdb.py --root data/mmimdb
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.datasets.mmimdb import GENRES


def build_split_json(root, train_ratio=0.8, dev_ratio=0.1, seed=0):
    dataset_dir = os.path.join(root, "dataset")
    ids = sorted(f[:-5] for f in os.listdir(dataset_dir) if f.endswith(".json"))
    rng = random.Random(seed)
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(n * train_ratio)
    n_dev = int(n * dev_ratio)
    splits = {
        "train": ids[:n_train],
        "dev": ids[n_train : n_train + n_dev],
        "test": ids[n_train + n_dev :],
    }
    with open(os.path.join(root, "split.json"), "w", encoding="utf-8") as f:
        json.dump(splits, f)
    print(f"Wrote split.json with {len(splits['train'])}/{len(splits['dev'])}/{len(splits['test'])} train/dev/test ids")
    return splits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--build_split", action="store_true", help="create split.json from an 80/10/10 random split")
    args = parser.parse_args()

    dataset_dir = os.path.join(args.root, "dataset")
    if not os.path.isdir(dataset_dir):
        raise SystemExit(f"Expected a 'dataset' subfolder at {dataset_dir}. See this script's docstring.")

    split_path = os.path.join(args.root, "split.json")
    if not os.path.exists(split_path):
        if args.build_split:
            splits = build_split_json(args.root)
        else:
            raise SystemExit(f"No split.json found at {split_path}. Re-run with --build_split to create one.")
    else:
        with open(split_path, "r", encoding="utf-8") as f:
            splits = json.load(f)

    genre_set = set(GENRES)
    missing_files, unknown_genres, genre_counts = 0, 0, {g: 0 for g in GENRES}
    total = 0
    for split_name, ids in splits.items():
        for movie_id in ids:
            total += 1
            json_path = os.path.join(dataset_dir, f"{movie_id}.json")
            jpeg_path = os.path.join(dataset_dir, f"{movie_id}.jpeg")
            if not (os.path.exists(json_path) and os.path.exists(jpeg_path)):
                missing_files += 1
                continue
            with open(json_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            for g in meta.get("genres", []):
                if g in genre_set:
                    genre_counts[g] += 1
                else:
                    unknown_genres += 1

    print(f"Splits: " + ", ".join(f"{k}={len(v)}" for k, v in splits.items()))
    print(f"Total samples checked: {total}, missing json/jpeg pairs: {missing_files}")
    print(f"Genre labels outside the standard 23-class vocabulary: {unknown_genres}")
    print("Genre frequency:")
    for g, c in sorted(genre_counts.items(), key=lambda x: -x[1]):
        print(f"  {g:12s} {c}")


if __name__ == "__main__":
    main()
