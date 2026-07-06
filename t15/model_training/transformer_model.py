"""BiT_Phoneme transformer phoneme decoder for T15.

Self-contained port of `src/neural_decoder/bit.py` from the
`transformers_with_dietcorp_cp` repo. Implementation follows
TRANSFORMER_IMPLEMENTATION.md §4 exactly:

  - Causal (lower-triangular) temporal mask, per source code.
  - T5-style relative position bias (single-scalar per relative offset).
  - Patch embedding via einops Rearrange (1 channel, patch_size x neural_dim).
  - GaussianSmoothing inlined as depthwise Conv1d along time.
  - Forward signature: ``model(neural_input, X_len=None, day_idx=None)`` —
    ``day_idx`` accepted for trainer compatibility but ignored.
  - n_classes is the *number of phonemes* (40); output linear is dim -> 41
    (40 + 1 blank). The trainer is responsible for subtracting 1 from the
    YAML's ``dataset.n_classes`` when constructing the model.

Deps: torch, einops only.
"""

import math
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as Fnn
from torch.utils.checkpoint import checkpoint

from einops import rearrange
from einops.layers.torch import Rearrange


# --------------------------------------------------------------------- helpers
def pad_to_multiple(x: torch.Tensor, multiple: int) -> torch.Tensor:
    """Right-pad along time (dim=1) so x.size(1) is a multiple of ``multiple``."""
    t = x.size(1)
    rem = t % multiple
    if rem == 0:
        return x
    return Fnn.pad(x, (0, 0, 0, multiple - rem))


def get_sinusoidal_pos_emb(seq_len: int, dim: int, device=None) -> torch.Tensor:
    position = torch.arange(seq_len, device=device).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, device=device) * -(math.log(10000.0) / dim)
    )
    pe = torch.zeros(seq_len, dim, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


def create_temporal_mask(seq_len: int, device=None) -> torch.Tensor:
    """Boolean lower-triangular mask of shape ``[1, 1, seq_len, seq_len]``.

    Each timestep ``t`` may attend to positions ``<= t``.
    """
    i = torch.arange(seq_len, device=device).unsqueeze(1)
    j = torch.arange(seq_len, device=device).unsqueeze(0)
    return (j <= i).unsqueeze(0).unsqueeze(0)


def _gaussian_kernel_1d(kernel_size: int, sigma: float) -> torch.Tensor:
    half = (kernel_size - 1) / 2.0
    x = torch.arange(kernel_size, dtype=torch.float32) - half
    k = torch.exp(-(x ** 2) / (2.0 * sigma ** 2))
    return k / k.sum()


class GaussianSmoothing(nn.Module):
    """Depthwise temporal Gaussian smoothing.

    Equivalent to bit.py's GaussianSmoothing(channels, kernel_size, sigma, dim=1).
    """

    def __init__(self, channels: int, kernel_size: int, sigma: float, dim: int = 1):
        super().__init__()
        del dim  # only dim=1 (time) is supported
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.channels = channels
        self.padding = kernel_size // 2
        kernel = _gaussian_kernel_1d(kernel_size, sigma)
        weight = kernel.view(1, 1, -1).repeat(channels, 1, 1)
        self.register_buffer("weight", weight, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, channels, T)
        return Fnn.conv1d(x, self.weight, padding=self.padding, groups=self.channels)


class _UnfoldPatcher(nn.Module):
    """Window the time axis with kernel=patch_height and stride=patch_stride.

    Input:  (B, T_pad, F) — F = neural_dim
    Output: (B, P, patch_height * F) where
            P = (T_pad - patch_height) // patch_stride + 1.
    With patch_stride < patch_height the patches overlap.
    """

    def __init__(self, patch_height: int, patch_stride: int):
        super().__init__()
        self.patch_height = patch_height
        self.patch_stride = patch_stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F) — unfold along time → (B, P, F, patch_height)
        patches = x.unfold(1, self.patch_height, self.patch_stride)
        # Permute to (B, P, patch_height, F) so the flat ordering matches the
        # original einops `(p1 p2 c)` layout: time-within-patch slow, channel fast.
        patches = patches.permute(0, 1, 3, 2).contiguous()
        b, p, ph, f = patches.shape
        return patches.reshape(b, p, ph * f)


# ------------------------------------------------------------------ submodules
class FFN(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Attention(nn.Module):
    """Multi-head attention with T5-style relative position bias.

    Source ``bit.py`` shares one ``dropout`` between attn-probs and output proj.
    Table 10 distinguishes the two, so we expose ``attn_dropout`` separately.
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        dropout: float = 0.2,
        attn_dropout: float = 0.4,
        max_rel_dist: int = 200,
        use_relative_bias: bool = True,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

        self.max_rel_dist = max_rel_dist
        self.use_relative_bias = use_relative_bias
        if use_relative_bias:
            self.rel_pos_bias = nn.Embedding(2 * max_rel_dist - 1, 1)

    def forward(self, x: torch.Tensor, temporal_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (rearrange(t, "b n (h d) -> b h n d", h=self.heads) for t in qkv)

        # Build additive attention bias (relative position + causal mask).
        # Shape (n, n) — SDPA broadcasts over batch and heads.
        attn_bias = None
        if self.use_relative_bias or temporal_mask is not None:
            seq_len = x.size(1)
            attn_bias = x.new_zeros(seq_len, seq_len)
            if self.use_relative_bias:
                i = torch.arange(seq_len, device=x.device).unsqueeze(1)
                j = torch.arange(seq_len, device=x.device).unsqueeze(0)
                rel = (i - j).clamp(-self.max_rel_dist + 1, self.max_rel_dist - 1) + self.max_rel_dist - 1
                attn_bias = attn_bias + self.rel_pos_bias(rel).squeeze(-1)
            if temporal_mask is not None:
                attn_bias = attn_bias.masked_fill(temporal_mask == 0, float("-inf"))

        # Memory-efficient SDPA — never materialises the full (b, h, n, n) matrix.
        dropout_p = self.attn_dropout.p if self.training else 0.0
        out = Fnn.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias, dropout_p=dropout_p, scale=self.scale)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim_ratio: int,
        dropout: float = 0.2,
        attn_dropout: float = 0.4,
        max_rel_dist: int = 200,
        use_relative_bias: bool = True,
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.layers = nn.ModuleList()
        mlp_dim = mlp_dim_ratio * dim
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(
                            dim=dim,
                            heads=heads,
                            dim_head=dim_head,
                            dropout=dropout,
                            attn_dropout=attn_dropout,
                            max_rel_dist=max_rel_dist,
                            use_relative_bias=use_relative_bias,
                        ),
                        FFN(dim=dim, hidden_dim=mlp_dim, dropout=dropout),
                    ]
                )
            )

    @staticmethod
    def _run_layer(attn, ffn, x, mask):
        return ffn(attn(x, temporal_mask=mask) + x) + x

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_rep: bool = False,
        rep_layer_idx: Optional[int] = None,
    ):
        rep = None
        use_ckpt = self.use_gradient_checkpointing and self.training
        for i, (attn, ffn) in enumerate(self.layers):
            if use_ckpt:
                x = checkpoint(self._run_layer, attn, ffn, x, mask, use_reentrant=False)
            else:
                x = attn(x, temporal_mask=mask) + x
                x = ffn(x) + x
            if return_rep and rep_layer_idx is not None and i == rep_layer_idx:
                rep = x  # capture before final LayerNorm
        x = self.norm(x)
        if return_rep:
            if rep_layer_idx is None:
                rep = x  # use normed output when no specific layer requested
            return x, rep
        return x


# ------------------------------------------------------------------- BiT model
class BiT_Phoneme(nn.Module):
    """BiT phoneme decoder.

    Output dim is ``n_classes + 1`` (+1 for the CTC blank). The trainer should
    pass ``n_classes = dataset.n_classes - 1`` so a YAML ``n_classes: 41`` (which
    includes blank) yields the correct 41-way output linear.
    """

    def __init__(
        self,
        *,
        patch_size: int,
        neural_dim: int,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim_ratio: int,
        dropout: float,
        attn_dropout: float,
        input_dropout: float,
        gaussian_smooth_width: float,
        gaussian_smooth_size: int,
        n_classes: int,
        T5_style_pos: bool,
        max_mask_pct: float,
        num_masks: int,
        mask_token_zeros: bool,
        max_rel_dist: int = 200,
        use_gradient_checkpointing: bool = False,
        patch_stride: Optional[int] = None,
    ):
        super().__init__()

        # Patcher windows the time axis with kernel = patch_size and
        # stride = patch_stride (defaults to patch_size for the original
        # non-overlapping behaviour). When patch_stride < patch_size the
        # patches overlap, matching the RNN baseline regime and producing
        # more CTC output frames per second.
        patch_height = patch_size
        patch_width = neural_dim
        if patch_stride is None:
            patch_stride = patch_size
        if patch_stride <= 0 or patch_stride > patch_size:
            raise ValueError(
                f"patch_stride must satisfy 0 < patch_stride <= patch_size; "
                f"got patch_stride={patch_stride}, patch_size={patch_size}."
            )
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.patch_stride = patch_stride
        self.dim = dim
        self.n_classes = n_classes
        self.gaussian_smooth_width = gaussian_smooth_width
        self.T5_style_pos = T5_style_pos
        self.max_mask_pct = max_mask_pct
        self.num_masks = num_masks
        self.patch_dim = patch_height * patch_width

        # Patcher: unfold(time, kernel=patch_height, stride=patch_stride) →
        # (B, P, patch_height * neural_dim) where
        # P = floor((T_pad - patch_height) / patch_stride) + 1.
        self.to_patch = _UnfoldPatcher(patch_height=patch_height, patch_stride=patch_stride)
        self.patch_to_emb = nn.Sequential(
            nn.LayerNorm(self.patch_dim),
            nn.Linear(self.patch_dim, dim),
            nn.LayerNorm(dim),
        )
        self.to_patch_embedding = nn.Sequential(self.to_patch, *self.patch_to_emb)

        # Depthwise temporal Gaussian smoothing (channels = neural_dim).
        self.gaussianSmoother = GaussianSmoothing(
            channels=patch_width,
            kernel_size=gaussian_smooth_size,
            sigma=gaussian_smooth_width,
        )

        if mask_token_zeros:
            self.mask_token = nn.Parameter(torch.zeros(self.patch_dim), requires_grad=False)
        else:
            self.mask_token = nn.Parameter(torch.randn(self.patch_dim))

        self.dropout = nn.Dropout(input_dropout)

        self.transformer = Transformer(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim_ratio=mlp_dim_ratio,
            dropout=dropout,
            attn_dropout=attn_dropout,
            max_rel_dist=max_rel_dist,
            use_relative_bias=T5_style_pos,
            use_gradient_checkpointing=use_gradient_checkpointing,
        )

        # Logits: n_classes phonemes + 1 blank.
        self.projection = nn.Linear(dim, n_classes + 1)

    # --------------------------------------------------------------- helpers
    def compute_length(self, X_len: torch.Tensor) -> torch.Tensor:
        """Number of valid output patches for each unpadded length.

        With kernel=patch_height and stride=patch_stride (and inputs padded
        so that P_full = (T_pad - patch_height) // patch_stride + 1 patches
        are always produced), the count of patches whose receptive field
        ends within the unpadded region is:
            ceil((X_len - patch_height + 1) / patch_stride),  clamped to >= 1.
        For the non-overlapping case (patch_stride == patch_height) this
        reduces to ceil(X_len / patch_height), matching prior behaviour.
        """
        ph = self.patch_height
        ps = self.patch_stride
        x = X_len.to(torch.float32)
        if ps == ph:
            out = torch.ceil(x / ph)
        else:
            out = torch.ceil((x - ph + 1).clamp(min=1.0) / ps)
        return out.clamp(min=1.0).to(torch.int32)

    # --------------------------------------------------------------- forward
    def forward(
        self,
        neural_input: torch.Tensor,
        X_len: Optional[torch.Tensor] = None,
        day_idx: Optional[torch.Tensor] = None,
        return_rep: bool = False,
        rep_layer_idx: Optional[int] = None,
    ):
        """Args:
            neural_input:  (B, T, neural_dim)
            X_len:         (B,) pre-pad lengths, used by SpecAugment when active
            day_idx:       accepted for trainer-API compat; ignored
            return_rep:    if True, also return intermediate patch representation
            rep_layer_idx: transformer block index (0-based) to extract rep from;
                           None = normed output of the final block
        Returns:
            logits (B, P, n_classes+1) — or (logits, rep) when return_rep=True,
            where rep is (B, P, dim).
        """
        del day_idx  # ignored — single-subject model

        # 1. Pad time so the unfolding is well-defined and trailing positions
        #    aren't lost. We need T_pad >= patch_height and (T_pad - patch_height)
        #    divisible by patch_stride.
        T = neural_input.size(1)
        ph, ps = self.patch_height, self.patch_stride
        if T < ph:
            pad = ph - T
        else:
            rem = (T - ph) % ps
            pad = 0 if rem == 0 else (ps - rem)
        if pad > 0:
            neural_input = torch.nn.functional.pad(neural_input, (0, 0, 0, pad))
        x = neural_input

        # 2. Per-channel temporal Gaussian smoothing.
        x = x.permute(0, 2, 1)  # (B, neural_dim, T_pad)
        x = self.gaussianSmoother(x)
        x = x.permute(0, 2, 1)  # (B, T_pad, neural_dim)

        # 3. Patch + (optional) SpecAugment. Patcher consumes (B, T_pad, F).
        if self.training and self.max_mask_pct > 0 and self.num_masks > 0:
            patches = self.to_patch(x)  # (B, P, patch_dim)
            if X_len is None:
                X_len = torch.full(
                    (patches.size(0),),
                    fill_value=neural_input.size(1),
                    device=neural_input.device,
                    dtype=torch.long,
                )
            patches, _ = self.apply_time_mask(patches, X_len)
            x = self.patch_to_emb(patches)
        else:
            x = self.to_patch_embedding(x)

        # 5. Input dropout.
        x = self.dropout(x)

        # 6. Optional sinusoidal pos emb when T5-style relative bias is off.
        b, seq_len, _ = x.shape
        if not self.T5_style_pos:
            pos_emb = get_sinusoidal_pos_emb(seq_len, self.dim, device=x.device)
            x = x + pos_emb.unsqueeze(0)

        # 7. Causal temporal mask + transformer (per source bit.py).
        temporal_mask = create_temporal_mask(seq_len, device=x.device)
        if return_rep:
            x, rep = self.transformer(x, mask=temporal_mask, return_rep=True, rep_layer_idx=rep_layer_idx)
            return self.projection(x), rep

        x = self.transformer(x, mask=temporal_mask)

        # 8. Output projection.
        return self.projection(x)

    # --------------------------------------------------------------- masking
    def apply_time_mask(
        self, X: torch.Tensor, X_len: torch.Tensor
    ):
        """Vectorised SpecAugment time masking on patched tokens.

        Args:
            X:     (B, P, D) patched tokens (pre-projection: D = patch_dim)
            X_len: (B,) pre-pad timestep lengths

        Returns:
            X_masked: (B, P, D) with masked patches replaced by ``self.mask_token``
            mask:     (B, P) boolean — True at masked positions
        """
        B, P, D = X.shape
        device = X.device

        valid_lens = self.compute_length(X_len).to(device).long()
        max_mask_lens = (self.max_mask_pct * valid_lens).long()

        B_rep = B * self.num_masks
        valid_lens_rep = valid_lens.repeat_interleave(self.num_masks)
        max_mask_lens_rep = max_mask_lens.repeat_interleave(self.num_masks)

        t = (torch.rand(B_rep, device=device) * (max_mask_lens_rep + 1).float()).floor().long()
        max_start = (valid_lens_rep - t + 1).clamp(min=1)
        t0 = (torch.rand(B_rep, device=device) * max_start.float()).floor().long()

        arange = torch.arange(P, device=device).unsqueeze(0)
        t0_exp = t0.unsqueeze(1)
        t1_exp = (t0 + t).unsqueeze(1)
        chunks = (arange >= t0_exp) & (arange < t1_exp)  # (B_rep, P)

        batch_idx = torch.arange(B, device=device).repeat_interleave(self.num_masks)
        patch_idx = chunks.nonzero(as_tuple=False)
        b_indices = batch_idx[patch_idx[:, 0]]
        p_indices = patch_idx[:, 1]

        mask = torch.zeros(B, P, dtype=torch.bool, device=device)
        mask[b_indices, p_indices] = True

        X_masked = X.clone()
        X_masked[mask] = self.mask_token.to(X.dtype)
        return X_masked, mask
