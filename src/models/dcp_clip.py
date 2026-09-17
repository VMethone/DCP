"""Deep Correlated Prompting (DCP) built on top of a frozen HuggingFace CLIP
backbone. Implements the method of:

  Hu, Shi, Feng, Shang, Wan. "Deep Correlated Prompting for Visual
  Recognition with Missing Modalities." NeurIPS 2024. arXiv:2410.06558

Design notes / places where the paper under-specifies exact details and we
made an explicit, documented choice:

1. Missing-modality masking (Sec. 4.1, "For the missing modality, we stop
   feeding the inputs into the corresponding encoder and use a zero-filled
   tensor as the output instead."). To support batches with a mix of
   missing-modality cases per sample (as required by the paper's random
   missing-rate protocol) while keeping the encoders' forward pass fully
   batched/vectorized, we always run both encoders on the whole batch (the
   dataset substitutes a placeholder image/text for genuinely absent
   modalities) and then zero out each sample's pooled feature according to
   its own presence mask. This is functionally equivalent to "not feeding
   the input / using a zero output" for the downstream classifier, since the
   masked-out branch never contributes to the concatenated representation.

2. Correlated-prompt cross-modal fusion (Eq. 5) is computed as a pure
   parameter chain (P_{m,i} depends only on P_{m,i-1} of both modalities,
   never on real hidden states -- see Eq. 3), so it is always available for
   fusion regardless of whether a given sample's modality is actually
   present. This matches the paper's equations exactly and lets the
   "present" modality's correlated prompts still benefit from cross-modal
   information even when the other modality is missing for that sample.

3. Dynamic and modal-common prompts are inserted once at the input level and
   simply ride along (retained, cf. Eq. 2) through all subsequent layers --
   this matches the paper's stated default configuration (prompt depth = 1
   for both, Tables 2-3), which is also what the main results use.

4. The default 36-token prompt budget (Sec. 4.1) is split evenly across the
   three prompt types (12/12/12) since the paper does not specify the exact
   split; this is configurable via ``prompt_len_*``.
"""
import torch
import torch.nn as nn
from transformers import CLIPModel

from .prompt_modules import CorrelatedPromptGenerator, DynamicPromptGenerator, ModalCommonProjector
from .clip_utils import (
    get_vision_submodules,
    get_text_submodules,
    run_vision_layer,
    run_text_layer,
    build_text_attention_mask,
)

# Missing-case ids, following the paper's m in {c, m1, m2} notation.
COMPLETE = 0        # both modalities present
MISSING_TEXT = 1    # image present, text missing
MISSING_IMAGE = 2   # text present, image missing
MISSING_TYPE_IDS = {"complete": COMPLETE, "missing_text": MISSING_TEXT, "missing_image": MISSING_IMAGE}
N_MISSING_TYPES = 3


class DCPCLIP(nn.Module):
    COMPLETE, MISSING_TEXT, MISSING_IMAGE = COMPLETE, MISSING_TEXT, MISSING_IMAGE

    def __init__(
        self,
        clip_name: str = "openai/clip-vit-base-patch16",
        num_classes: int = 23,
        multi_label: bool = True,
        prompt_len_correlated: int = 12,
        prompt_len_dynamic: int = 12,
        prompt_len_common: int = 12,
        common_dim: int = 128,
        depth_J: int = 6,
        reduction: int = 16,
    ):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(clip_name)
        for p in self.clip.parameters():
            p.requires_grad_(False)

        vcfg = self.clip.config.vision_config
        tcfg = self.clip.config.text_config
        self.v_dim = vcfg.hidden_size
        self.t_dim = tcfg.hidden_size
        self.J = max(1, min(depth_J, vcfg.num_hidden_layers, tcfg.num_hidden_layers))
        self.multi_label = multi_label

        Lr, Ld, Lc = prompt_len_correlated, prompt_len_dynamic, prompt_len_common
        self.Lr, self.Ld, self.Lc = Lr, Ld, Lc

        # --- correlated prompts: Eq. 3-5 ---
        self.corr_init_v = nn.Parameter(torch.randn(N_MISSING_TYPES, Lr, self.v_dim) * 0.02)
        self.corr_init_t = nn.Parameter(torch.randn(N_MISSING_TYPES, Lr, self.t_dim) * 0.02)
        self.corr_gen_v = nn.ModuleList(
            [CorrelatedPromptGenerator(self.v_dim, other_dim=self.t_dim, reduction=reduction) for _ in range(self.J - 1)]
        )
        self.corr_gen_t = nn.ModuleList(
            [CorrelatedPromptGenerator(self.t_dim, other_dim=self.v_dim, reduction=reduction) for _ in range(self.J - 1)]
        )

        # --- dynamic prompts: Eq. 6-7 (input level only) ---
        self.dyn_gen_v = DynamicPromptGenerator(self.v_dim, Ld) if Ld > 0 else None
        self.dyn_gen_t = DynamicPromptGenerator(self.t_dim, Ld) if Ld > 0 else None

        # --- modal-common prompts: Eq. 8 (input level only) ---
        if Lc > 0:
            self.common_latent = nn.Parameter(torch.randn(N_MISSING_TYPES, Lc, common_dim) * 0.02)
            self.proj_common_v = ModalCommonProjector(common_dim, self.v_dim, reduction=reduction)
            self.proj_common_t = ModalCommonProjector(common_dim, self.t_dim, reduction=reduction)
        else:
            self.common_latent = None
            self.proj_common_v = None
            self.proj_common_t = None

        self.classifier = nn.Linear(self.v_dim + self.t_dim, num_classes)

    def train(self, mode: bool = True):
        super().train(mode)
        self.clip.eval()  # backbone always frozen/eval, only prompts + classifier train
        return self

    # ------------------------------------------------------------------
    def _build_correlated_chains(self, missing_ids: torch.Tensor):
        cv = [self.corr_init_v[missing_ids]]  # (B, Lr, v_dim)
        ct = [self.corr_init_t[missing_ids]]  # (B, Lr, t_dim)
        for i in range(self.J - 1):
            nv = self.corr_gen_v[i](cv[-1], ct[-1])
            nt = self.corr_gen_t[i](ct[-1], cv[-1])
            cv.append(nv)
            ct.append(nt)
        return cv, ct

    def _encode_image(self, pixel_values, missing_ids, corr_chain_v):
        embeddings, pre_ln, layers, post_ln = get_vision_submodules(self.clip)
        x = embeddings(pixel_values)
        if pre_ln is not None:
            x = pre_ln(x)
        B = x.size(0)

        cls_tok, patches = x[:, :1], x[:, 1:]
        dyn = self.dyn_gen_v(x) if self.dyn_gen_v is not None else x.new_zeros(B, 0, self.v_dim)
        common = (
            self.proj_common_v(self.common_latent[missing_ids]) if self.proj_common_v is not None else x.new_zeros(B, 0, self.v_dim)
        )
        corr = corr_chain_v[0]
        Ld, Lc, Lr = dyn.size(1), common.size(1), corr.size(1)

        seq = torch.cat([cls_tok, dyn, common, corr, patches], dim=1)
        for i, layer in enumerate(layers):
            seq = run_vision_layer(layer, seq)
            if i < self.J - 1:
                cls_tok = seq[:, :1]
                dyn_part = seq[:, 1 : 1 + Ld]
                common_part = seq[:, 1 + Ld : 1 + Ld + Lc]
                rest = seq[:, 1 + Ld + Lc + Lr :]
                seq = torch.cat([cls_tok, dyn_part, common_part, corr_chain_v[i + 1], rest], dim=1)

        pooled = post_ln(seq[:, 0])
        return pooled

    def _encode_text(self, input_ids, attention_mask, missing_ids, corr_chain_t):
        embeddings, layers, final_ln = get_text_submodules(self.clip)
        x = embeddings(input_ids=input_ids)
        B = x.size(0)

        dyn = (
            self.dyn_gen_t(x, key_padding_mask=(attention_mask == 0))
            if self.dyn_gen_t is not None
            else x.new_zeros(B, 0, self.t_dim)
        )
        common = (
            self.proj_common_t(self.common_latent[missing_ids]) if self.proj_common_t is not None else x.new_zeros(B, 0, self.t_dim)
        )
        corr = corr_chain_t[0]
        Ld, Lc, Lr = dyn.size(1), common.size(1), corr.size(1)
        prompt_len = Ld + Lc + Lr

        seq = torch.cat([dyn, common, corr, x], dim=1)
        combined_mask = build_text_attention_mask(attention_mask, prompt_len, seq.dtype)

        for i, layer in enumerate(layers):
            seq = run_text_layer(layer, seq, combined_mask)
            if i < self.J - 1:
                dyn_part = seq[:, :Ld]
                common_part = seq[:, Ld : Ld + Lc]
                rest = seq[:, Ld + Lc + Lr :]
                seq = torch.cat([dyn_part, common_part, corr_chain_t[i + 1], rest], dim=1)

        seq = final_ln(seq)
        eos_pos = input_ids.argmax(dim=-1) + prompt_len
        pooled = seq[torch.arange(B, device=seq.device), eos_pos]
        return pooled

    # ------------------------------------------------------------------
    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        missing_type_ids: torch.Tensor,
        image_present: torch.Tensor = None,
        text_present: torch.Tensor = None,
    ) -> torch.Tensor:
        """All tensors are batched; ``missing_type_ids`` (LongTensor[B]) and the
        optional per-sample presence masks let a single batch mix different
        missing-modality cases (see class docstring, point 1)."""
        if image_present is None:
            image_present = (missing_type_ids != MISSING_IMAGE).float()
        if text_present is None:
            text_present = (missing_type_ids != MISSING_TEXT).float()

        corr_chain_v, corr_chain_t = self._build_correlated_chains(missing_type_ids)
        pooled_v = self._encode_image(pixel_values, missing_type_ids, corr_chain_v)
        pooled_t = self._encode_text(input_ids, attention_mask, missing_type_ids, corr_chain_t)

        pooled_v = pooled_v * image_present.to(pooled_v.dtype).unsqueeze(-1)
        pooled_t = pooled_t * text_present.to(pooled_t.dtype).unsqueeze(-1)

        feat = torch.cat([pooled_v, pooled_t], dim=-1)
        return self.classifier(feat)

    # ------------------------------------------------------------------
    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())

    def num_total_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
