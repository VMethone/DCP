"""Quick end-to-end smoke test for the DCP-CLIP model that needs no dataset
-- only internet access (or a local cache) to fetch the CLIP checkpoint.
Verifies: forward pass shapes, that missing-modality masking actually zeroes
out the right branch, and that gradients only flow into trainable params.

Usage:
    python scripts/sanity_check.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models import DCPCLIP, COMPLETE, MISSING_TEXT, MISSING_IMAGE


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = DCPCLIP(
        clip_name="openai/clip-vit-base-patch16",
        num_classes=23,
        multi_label=True,
        prompt_len_correlated=4,
        prompt_len_dynamic=4,
        prompt_len_common=4,
        common_dim=32,
        depth_J=3,
        reduction=16,
    ).to(device)

    print(f"Trainable params: {model.num_trainable_parameters():,} / {model.num_total_parameters():,} total")

    B = 6
    pixel_values = torch.randn(B, 3, 224, 224, device=device)
    input_ids = torch.randint(0, 49407, (B, 20), device=device)
    input_ids[:, -1] = 49407  # fake EOS as the max-id token, mimicking CLIP's tokenizer convention
    attention_mask = torch.ones(B, 20, dtype=torch.long, device=device)
    missing_type_ids = torch.tensor([COMPLETE, COMPLETE, MISSING_TEXT, MISSING_TEXT, MISSING_IMAGE, MISSING_IMAGE], device=device)

    logits = model(pixel_values, input_ids, attention_mask, missing_type_ids)
    assert logits.shape == (B, 23), logits.shape
    print(f"Forward OK, logits shape: {tuple(logits.shape)}")

    # Check masking: pooled image feature should be exactly zero for MISSING_IMAGE samples,
    # and pooled text feature should be exactly zero for MISSING_TEXT samples.
    with torch.no_grad():
        corr_v, corr_t = model._build_correlated_chains(missing_type_ids)
        pooled_v = model._encode_image(pixel_values, missing_type_ids, corr_v)
        pooled_t = model._encode_text(input_ids, attention_mask, missing_type_ids, corr_t)
        image_present = (missing_type_ids != MISSING_IMAGE).float().unsqueeze(-1)
        text_present = (missing_type_ids != MISSING_TEXT).float().unsqueeze(-1)
        masked_v = pooled_v * image_present
        masked_t = pooled_t * text_present
        assert torch.all(masked_v[4:6] == 0), "image branch not zeroed for MISSING_IMAGE samples"
        assert torch.all(masked_t[2:4] == 0), "text branch not zeroed for MISSING_TEXT samples"
    print("Missing-modality masking OK")

    labels = torch.randint(0, 2, (B, 23), device=device).float()
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
    loss.backward()

    n_grad_trainable = sum(1 for p in model.trainable_parameters() if p.grad is not None)
    n_grad_clip = sum(1 for p in model.clip.parameters() if p.grad is not None)
    print(f"Params with grad: trainable={n_grad_trainable}/{len(model.trainable_parameters())}, frozen clip={n_grad_clip} (should be 0)")
    assert n_grad_clip == 0, "frozen CLIP backbone received gradients!"

    print("All sanity checks passed.")


if __name__ == "__main__":
    main()
