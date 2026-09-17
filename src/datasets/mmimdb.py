"""MM-IMDb dataset (Arevalo et al., 2017) with missing-modality simulation.

Expects the raw MM-IMDb release layout:
    <root>/dataset/<id>.json   (has "plot": list[str] and "genres": list[str])
    <root>/dataset/<id>.jpeg
    <root>/split.json          ({"train": [...ids], "dev": [...], "test": [...]})
as distributed at http://lisi1.unal.edu.co/mmimdb/. If your downloaded copy
uses a different layout (e.g. the pyarrow-table format used by some
ViLT/MMP codebases), adapt ``_load_json``/``__getitem__`` accordingly -- the
rest of the pipeline (missing-modality assignment, collate_fn, model) is
independent of the storage format.
"""
import json
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from ..utils.missing import assign_missing_types, MISSING_TEXT, MISSING_IMAGE

# Standard 23-genre vocabulary used across MM-IMDb missing-modality papers
# (ViLT/MMP/DCP all report F1-Macro over these 23 classes).
GENRES = [
    "Drama", "Comedy", "Romance", "Thriller", "Crime", "Action", "Adventure",
    "Horror", "Documentary", "Mystery", "Sci-Fi", "Fantasy", "Family",
    "Biography", "War", "History", "Music", "Animation", "Musical",
    "Western", "Sport", "Short", "Film-Noir",
]


class MMIMDbDataset(Dataset):
    GENRES = GENRES

    def __init__(
        self,
        root: str,
        split: str,
        scenario: str = "missing_both",
        missing_rate: float = 0.7,
        seed: int = 0,
        image_size: int = 224,
    ):
        assert split in ("train", "dev", "test"), split
        self.root = root
        self.split = split
        self.image_size = image_size

        with open(os.path.join(root, "split.json"), "r", encoding="utf-8") as f:
            splits = json.load(f)
        self.ids = list(splits[split])

        self.genre_to_idx = {g: i for i, g in enumerate(self.GENRES)}
        # Fixed missing-pattern per split (paper evaluates on a fixed missing
        # assignment, not one re-sampled every epoch). Use a different seed
        # per split so train/dev/test don't share the exact same pattern.
        split_seed = seed + {"train": 0, "dev": 1, "test": 2}[split]
        rng = np.random.RandomState(split_seed)
        self.missing_ids = assign_missing_types(len(self.ids), scenario, missing_rate, rng)

    def __len__(self):
        return len(self.ids)

    def _load_meta(self, movie_id):
        with open(os.path.join(self.root, "dataset", f"{movie_id}.json"), "r", encoding="utf-8") as f:
            return json.load(f)

    def __getitem__(self, idx):
        movie_id = self.ids[idx]
        meta = self._load_meta(movie_id)
        missing_type = int(self.missing_ids[idx])

        if missing_type == MISSING_TEXT:
            text = " "  # placeholder; masked out downstream by the model
        else:
            plot = meta.get("plot", [""])
            text = (plot[0] if plot else "") or " "

        if missing_type == MISSING_IMAGE:
            image = Image.new("RGB", (self.image_size, self.image_size), color=(0, 0, 0))
        else:
            img_path = os.path.join(self.root, "dataset", f"{movie_id}.jpeg")
            image = Image.open(img_path).convert("RGB")

        label = torch.zeros(len(self.GENRES), dtype=torch.float32)
        for g in meta.get("genres", []):
            if g in self.genre_to_idx:
                label[self.genre_to_idx[g]] = 1.0

        return {"image": image, "text": text, "missing_type": missing_type, "label": label}


def make_collate_fn(processor, max_text_len: int = 77):
    """``processor`` is a HuggingFace ``CLIPProcessor`` (image processor + tokenizer)."""

    def collate(batch):
        images = [b["image"] for b in batch]
        texts = [b["text"] for b in batch]
        missing_types = torch.tensor([b["missing_type"] for b in batch], dtype=torch.long)
        labels = torch.stack([b["label"] for b in batch])

        img_inputs = processor.image_processor(images=images, return_tensors="pt")
        text_inputs = processor.tokenizer(
            texts, padding="max_length", truncation=True, max_length=max_text_len, return_tensors="pt"
        )

        return {
            "pixel_values": img_inputs["pixel_values"],
            "input_ids": text_inputs["input_ids"],
            "attention_mask": text_inputs["attention_mask"],
            "missing_type_ids": missing_types,
            "labels": labels,
        }

    return collate
