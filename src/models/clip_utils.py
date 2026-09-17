"""Low-level helpers for driving a HuggingFace CLIPModel layer-by-layer so
that prompts can be inserted/replaced at arbitrary depths.

We deliberately avoid calling ``CLIPVisionTransformer.forward`` /
``CLIPTextTransformer.forward`` directly, since those methods run all
encoder layers in one shot. Deep Correlated Prompting needs to intercept the
hidden states between layers (to discard/regenerate the correlated-prompt
slice), so we call the embeddings module and each ``CLIPEncoderLayer``
individually instead.
"""
import torch


def get_vision_submodules(clip_model):
    vm = clip_model.vision_model
    pre_ln = getattr(vm, "pre_layrnorm", None) or getattr(vm, "pre_layernorm", None)
    return vm.embeddings, pre_ln, vm.encoder.layers, vm.post_layernorm


def get_text_submodules(clip_model):
    tm = clip_model.text_model
    return tm.embeddings, tm.encoder.layers, tm.final_layer_norm


def run_vision_layer(layer, hidden_states):
    """CLIPEncoderLayer forward for the (bidirectional) vision transformer."""
    out = layer(hidden_states, attention_mask=None, causal_attention_mask=None)
    return out[0] if isinstance(out, tuple) else out.hidden_states if hasattr(out, "hidden_states") else out


def run_text_layer(layer, hidden_states, combined_mask):
    """CLIPEncoderLayer forward for the causal text transformer."""
    out = layer(hidden_states, attention_mask=combined_mask, causal_attention_mask=None)
    return out[0] if isinstance(out, tuple) else out.hidden_states if hasattr(out, "hidden_states") else out


def build_text_attention_mask(padding_mask: torch.Tensor, prompt_len: int, dtype: torch.dtype) -> torch.Tensor:
    """Builds an additive (bsz, 1, L', L') mask combining:
      - a standard causal mask over the *whole* prompt+text sequence (prompts
        are placed first, so every text position can attend to all prompts
        and to earlier text positions, matching CLIP's normal causal
        attention pattern),
      - the tokenizer's padding mask, extended with `prompt_len` always-valid
        positions for the prepended prompt tokens.

    Returns a float tensor with 0 at allowed positions and a large negative
    value at disallowed ones, as expected by CLIPEncoderLayer's
    ``attention_mask`` argument.
    """
    B, L = padding_mask.shape
    device = padding_mask.device
    total_len = L + prompt_len
    neg_inf = torch.finfo(dtype).min

    causal = torch.full((total_len, total_len), neg_inf, device=device, dtype=dtype)
    causal = torch.triu(causal, diagonal=1)
    causal = causal.unsqueeze(0).unsqueeze(0).expand(B, 1, total_len, total_len)

    prompt_pad = torch.ones(B, prompt_len, dtype=padding_mask.dtype, device=device)
    full_padding = torch.cat([prompt_pad, padding_mask], dim=1)  # (B, total_len)
    pad_additive = (1.0 - full_padding[:, None, None, :].to(dtype)) * neg_inf

    combined = causal + pad_additive
    return combined
