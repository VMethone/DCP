# Deep Correlated Prompting (DCP) — Reproduction

An implementation of **"Deep Correlated Prompting for Visual Recognition with
Missing Modalities"** (Hu, Shi, Feng, Shang, Wan — NeurIPS 2024,
[arXiv:2410.06558](https://arxiv.org/abs/2410.06558)), built on a frozen
HuggingFace CLIP backbone.

This is an independent, from-the-paper reimplementation (not the authors'
code at https://github.com/hulianyuyy/Deep_Correlated_Prompting) — see
**Design decisions** below for every place the paper under-specifies an
implementation detail and what choice was made here.

## What's implemented

- `src/models/prompt_modules.py` — the three prompt-generation primitives:
  `CorrelatedPromptGenerator` (Eq. 3–5), `DynamicPromptGenerator` (Eq. 6–7),
  `ModalCommonProjector` (Eq. 8).
- `src/models/dcp_clip.py` — `DCPCLIP`, the full model: drives a HuggingFace
  `CLIPModel` layer-by-layer, splicing in correlated/dynamic/modal-common
  prompts per Sec. 3.3, with per-sample missing-case selection.
- `src/models/clip_utils.py` — low-level helpers to run individual
  `CLIPEncoderLayer`s and build the causal+padding attention mask for the
  text tower once prompts are prepended.
- `src/utils/missing.py` — the missing-rate sampling protocol of Sec. 4.1
  (`missing_both` / `missing_text` / `missing_image` scenarios).
- `src/datasets/mmimdb.py` — MM-IMDb dataset + collate function with
  missing-modality simulation.
- `train.py` / `eval.py` — training and evaluation entry points.
- `scripts/prepare_mmimdb.py` — sanity-checks a downloaded MM-IMDb directory.
- `scripts/sanity_check.py` — end-to-end smoke test of the model (forward
  pass, masking, gradient isolation) — needs the CLIP checkpoint but no
  dataset.

Only MM-IMDb has a ready-made `Dataset` class. UPMC Food-101 and Hateful
Memes use the exact same model/training code — see **Extending to other
datasets** below.

## Setup

```bash
pip install -r requirements.txt
```

Verify the model architecture end-to-end (downloads
`openai/clip-vit-base-patch16`, ~600MB, on first run):

```bash
python scripts/sanity_check.py
```

## Getting the data

You mentioned the dataset download itself isn't urgent since training will
happen on a remote (SSH) machine — that's fine, nothing else here depends on
it being present locally.

MM-IMDb (Arevalo et al., 2017) is distributed at
http://lisi1.unal.edu.co/mmimdb/. Extract it so you have:

```
data/mmimdb/dataset/<id>.json   # {"plot": [...], "genres": [...], ...}
data/mmimdb/dataset/<id>.jpeg
data/mmimdb/split.json          # {"train": [...], "dev": [...], "test": [...]}
```

If your copy doesn't ship a `split.json`, generate an 80/10/10 split:

```bash
python scripts/prepare_mmimdb.py --root data/mmimdb --build_split
```

Otherwise just validate the structure:

```bash
python scripts/prepare_mmimdb.py --root data/mmimdb
```

If the copy you find uses a different on-disk layout (e.g. some
ViLT/MMP-derived repos repackage it as a pyarrow table), only
`src/datasets/mmimdb.py`'s `__getitem__`/`_load_meta` need to change — the
missing-modality sampling, collate function, and model are independent of
storage format.

## Training

```bash
python train.py --config configs/mmimdb.yaml
```

Override any config value from the CLI with `section.key=value`:

```bash
python train.py --config configs/mmimdb.yaml data.missing_rate=0.5 data.scenario=missing_text train.epochs=10
```

`configs/mmimdb.yaml` matches the paper's reported hyperparameters: prompt
length 36 (split 12/12/12 across correlated/dynamic/modal-common), prompt
depth J=6, Adam lr=1e-2, weight decay 2e-2, 10% linear warmup then linear
decay to zero, batch size 4, missing rate η=0.7.

Checkpoints (only the trainable prompt/classifier weights, not the frozen
CLIP backbone) are saved to `train.output_dir/best.pt`.

## Evaluation

```bash
python eval.py --config configs/mmimdb.yaml --checkpoint outputs/mmimdb/best.pt --split test
# sweep a different missing rate than the one used for training:
python eval.py --config configs/mmimdb.yaml --checkpoint outputs/mmimdb/best.pt --split test --missing_rate 0.3
```

## Design decisions (where the paper is ambiguous)

The paper's Sec. 3.3 gives equations for the prompt mechanisms but leaves a
handful of implementation choices unspecified. Each is documented in code
(mainly the `DCPCLIP` docstring in `src/models/dcp_clip.py`) and summarized
here:

1. **Missing-modality masking.** The paper says: *"we stop feeding the
   inputs into the corresponding encoder and use a zero-filled tensor as the
   output instead."* To support batches that mix different missing cases per
   sample (required by the paper's random missing-rate protocol) while
   keeping everything vectorized, this implementation always runs both
   encoders over the whole batch (missing samples get a placeholder
   black image / whitespace text) and then **zeroes each sample's pooled
   feature** according to its own presence mask before concatenation. This
   is functionally equivalent for the downstream classifier — the masked
   branch contributes nothing — but is much simpler to batch.

2. **Correlated-prompt cross-modal fusion (Eq. 5)** is a pure parameter
   chain: `P_{m,i}` depends only on `P_{m,i-1}` of *both* modalities, never
   on real hidden states (only Eq. 6, the dynamic prompt, reads real input
   tokens). So it's always computable regardless of whether a sample's other
   modality is actually present, letting the present modality's correlated
   prompts still benefit from cross-modal information even for
   missing-modality samples.

3. **Dynamic / modal-common prompt depth.** Both are inserted once at the
   input level and simply ride along through all later layers (`Eq. 2`'s
   "retained" behavior) — this matches the paper's own default ablation
   configuration (prompt depth = 1 for both, Tables 2 and 3), which is also
   what the headline results use. The paper's ablation over deeper variants
   (depth > 1 re-injection) is not implemented, since the mechanism isn't
   fully specified and the default (depth=1) is what's used for the main
   numbers.

4. **36-token prompt budget split.** The paper sets total prompt length
   `Lp=36` but doesn't specify how it's divided across the three prompt
   types; this implementation defaults to an even 12/12/12 split
   (`prompt_len_correlated` / `prompt_len_dynamic` / `prompt_len_common` in
   the config), each independently tunable.

5. **Dynamic prompt generator (Eq. 7)** is described as "a self-attention
   layer" that must "deal with the varying length of tokens for different
   inputs" while producing a fixed-length prompt. Implemented as
   cross-attention with a fixed bank of learnable query tokens attending to
   the (variable-length) input sequence as key/value — a standard
   variable-to-fixed-length pooling mechanism consistent with the stated
   purpose.

6. **Text attention mask with prepended prompts.** Prompts are placed before
   the BOS token and the whole prompt+text sequence uses one standard causal
   mask (each position attends to itself and everything before it). This
   means text tokens can attend to all prompts (as intended) and prompts
   attend causally among themselves — a common simplification in
   prompt-tuning implementations rather than giving prompts full bidirectional
   visibility among themselves.

7. **Final task representation.** Per Fig. 1, pooled image/text tokens are
   concatenated directly (not projected into CLIP's shared embedding space
   via `visual_projection`/`text_projection`) before the final FC classifier
   — matching "concatenate the task-related token of both encoders... pass
   it through a fully-connected layer."

None of these are exotic choices — they're the standard moves in the
prompt-tuning literature the paper builds on (VPT, CoOp, MaPLe) — but since
the official code isn't what this was built from, exact numbers may differ
from Table 4 of the paper even when the qualitative trends (DCP > MMP >
CoOp > baseline, robustness across missing rates) should hold.

## Extending to other datasets

Nothing in `DCPCLIP` or `src/engine.py` is MM-IMDb-specific. To add UPMC
Food-101 or Hateful Memes:

1. Write a `Dataset` analogous to `src/datasets/mmimdb.py` returning
   `{"image": PIL.Image, "text": str, "missing_type": int, "label": tensor}`,
   reusing `assign_missing_types` from `src/utils/missing.py`.
2. Reuse `make_collate_fn` as-is.
3. Food-101 is single-label (101-way) — set `multi_label: false` and
   `num_classes: 101` in a new config; `train.py`/`eval.py` already branch
   on `multi_label` for the loss (`CrossEntropyLoss` vs `BCEWithLogitsLoss`)
   and metric (accuracy vs F1-macro).
4. Hateful Memes is binary classification scored by AUROC — set
   `multi_label: false`, `num_classes: 2`, and add an AUROC metric next to
   `f1_macro_multilabel`/`accuracy_single_label` in `src/utils/metrics.py`
   (e.g. `sklearn.metrics.roc_auc_score` on the softmax'd positive-class
   probability).

## Known local-environment note

Running `scripts/sanity_check.py` here failed to download the CLIP
checkpoint because `C:` has essentially no free space left on this machine
(`df` showed 8MB free). This doesn't affect the code — the architecture was
instead validated with a tiny randomly-initialized CLIP config (no download)
confirming the forward pass, missing-modality masking, and frozen-backbone
gradient isolation all work correctly. On your SSH machine, make sure
`HF_HOME`/the default `~/.cache/huggingface` has a few GB free before running
`train.py`.
