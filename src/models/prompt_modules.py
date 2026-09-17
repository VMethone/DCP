"""Building blocks for Deep Correlated Prompting (Hu et al., NeurIPS 2024).

Each class below implements one equation from Section 3.3 of the paper:
  - CorrelatedPromptGenerator: Eq. 3 / Eq. 4 / Eq. 5 (bottleneck MLP F, bi-modal fusion)
  - DynamicPromptGenerator:    Eq. 6 / Eq. 7 (input-conditioned prompt generator D)
  - ModalCommonProjector:      Eq. 8 (projection G from a shared modal-common latent)
"""
import torch
import torch.nn as nn


class CorrelatedPromptGenerator(nn.Module):
    """F(.) = LN(Fc(GELU(Fc(.)))) -- Eq. 4.

    Generates the correlated prompt of layer i from the correlated prompt of
    layer i-1 (Eq. 3). When ``other_dim`` is given, the prompt of the *other*
    modality's encoder at layer i-1 is concatenated in before the bottleneck,
    implementing the bi-modal fusion of Eq. 5.
    """

    def __init__(self, dim: int, other_dim: int = None, reduction: int = 16):
        super().__init__()
        self.bimodal = other_dim is not None
        in_dim = dim + other_dim if self.bimodal else dim
        hidden = max(1, in_dim // reduction)
        self.fc1 = nn.Linear(in_dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.ln = nn.LayerNorm(dim)

    def forward(self, prompt_self: torch.Tensor, prompt_other: torch.Tensor = None) -> torch.Tensor:
        if self.bimodal:
            assert prompt_other is not None, "bimodal generator requires the other modality's prompt"
            x = torch.cat([prompt_self, prompt_other], dim=-1)
        else:
            x = prompt_self
        return self.ln(self.fc2(self.act(self.fc1(x))))


class DynamicPromptGenerator(nn.Module):
    """D(.) = LN(MLP(LN(MHA(.)))) -- Eq. 7.

    The paper generates a dynamic prompt from the (variable-length) input
    token sequence via a self-attention layer. To produce a *fixed* number of
    prompt tokens regardless of input length (needed since prompts of every
    missing case must have the same length to be concatenated into one
    sequence), we implement the attention with a fixed bank of learnable
    query tokens attending to the input sequence as key/value -- a standard
    variable-to-fixed-length attention pooling (e.g. as in Perceiver /
    BLIP-2's Q-Former), consistent with the paper's stated purpose ("to deal
    with the varying length of tokens for different inputs").
    """

    def __init__(self, dim: int, num_prompts: int, num_heads: int = 1):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, num_prompts, dim) * 0.02)
        self.mha = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        bsz = x.size(0)
        q = self.query.expand(bsz, -1, -1)
        attn_out, _ = self.mha(q, x, x, key_padding_mask=key_padding_mask)
        h = self.ln1(attn_out)
        h = self.ln2(h + self.mlp(h))
        return h


class ModalCommonProjector(nn.Module):
    """G(.) projects the shared modal-common latent into a modality-specific
    space -- Eq. 8. Instantiated as an MLP with reduction factor r=16 as in
    the paper.
    """

    def __init__(self, common_dim: int, out_dim: int, reduction: int = 16):
        super().__init__()
        hidden = max(1, common_dim // reduction)
        self.fc1 = nn.Linear(common_dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))
