# hi_longformer.py
#
# Hierarchical Local-Global Longformer for neural→phoneme decoding.
# Architecture is copied from LocalGlobalViT_Phoneme (no inheritance),
# and extended to always output:
#   logits_phone: [B, T_patches, nPhones+1]
#   logits_broad: [B, T_patches, nBroad+1]
#
# Also provides:
#   compute_hierarchical_ctc_loss(...) for joint phoneme + broad-class CTC.

from typing import Optional, Tuple, Dict, List, Tuple as TypingTuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

# ---------------------------------------------------------
# Phone / broad-class definitions
# ---------------------------------------------------------

PHONE_DEF = [
    'AA', 'AE', 'AH', 'AO', 'AW',
    'AY', 'B',  'CH', 'D', 'DH',
    'EH', 'ER', 'EY', 'F', 'G',
    'HH', 'IH', 'IY', 'JH', 'K',
    'L', 'M', 'N', 'NG', 'OW',
    'OY', 'P', 'R', 'S', 'SH',
    'T', 'TH', 'UH', 'UW', 'V',
    'W', 'Y', 'Z', 'ZH'
]

PHONE_DEF_SIL = PHONE_DEF + ['SIL']  # 40 phones including silence

BROAD_CLASS_DEF = [
    'VOWEL',     # 0
    'STOP',      # 1
    'NASAL',     # 2
    'FRICATIVE', # 3
    'AFFRICATE', # 4
    'LIQUID',    # 5
    'GLIDE',     # 6
    'SIL',       # 7
]

PHONE_TO_BROAD: Dict[str, str] = {
    # Vowels (monophthongs & diphthongs)
    'AA': 'VOWEL', 'AE': 'VOWEL', 'AH': 'VOWEL', 'AO': 'VOWEL',
    'AW': 'VOWEL', 'AY': 'VOWEL', 'EH': 'VOWEL', 'ER': 'VOWEL',
    'EY': 'VOWEL', 'IH': 'VOWEL', 'IY': 'VOWEL', 'OW': 'VOWEL',
    'OY': 'VOWEL', 'UH': 'VOWEL', 'UW': 'VOWEL',

    # Stops
    'B': 'STOP', 'D': 'STOP', 'G': 'STOP',
    'K': 'STOP', 'P': 'STOP', 'T': 'STOP',

    # Nasals
    'M': 'NASAL', 'N': 'NASAL', 'NG': 'NASAL',

    # Fricatives (incl. aspirate-ish HH)
    'F': 'FRICATIVE', 'V': 'FRICATIVE', 'TH': 'FRICATIVE', 'DH': 'FRICATIVE',
    'S': 'FRICATIVE', 'Z': 'FRICATIVE', 'SH': 'FRICATIVE', 'ZH': 'FRICATIVE',
    'HH': 'FRICATIVE',

    # Affricates
    'CH': 'AFFRICATE', 'JH': 'AFFRICATE',

    # Liquids
    'L': 'LIQUID', 'R': 'LIQUID',

    # Glides / semivowels
    'W': 'GLIDE', 'Y': 'GLIDE',

    # Silence
    'SIL': 'SIL',
}

PHONE_TO_INDEX = {ph: i for i, ph in enumerate(PHONE_DEF_SIL)}
INDEX_TO_PHONE = {i: ph for ph, i in PHONE_TO_INDEX.items()}

BROAD_TO_INDEX = {bc: i for i, bc in enumerate(BROAD_CLASS_DEF)}
INDEX_TO_BROAD = {i: bc for bc, i in BROAD_TO_INDEX.items()}


def build_phone_to_broad_lut(device: torch.device, one_indexed: bool = True) -> torch.Tensor:
    """
    LUT from phone index to broad-class index.
    
    Args:
        device: torch device
        one_indexed: If True, LUT is 1-indexed (size 41: index 0 unused/blank, indices 1-40 map to phones)
                    If False, LUT is 0-indexed (size 40: indices 0-39 map to phones)
    """
    nPhones = len(PHONE_DEF_SIL)
    lut_size = nPhones + 1 if one_indexed else nPhones
    lut = torch.zeros(lut_size, dtype=torch.long, device=device)
    
    # Map blank/padding (index 0) to SIL broad class
    blank_bc_idx = BROAD_TO_INDEX['SIL']
    lut[0] = blank_bc_idx
    
    if one_indexed:
        # 1-indexed: lut[1] through lut[40] map to phones
        for ph_idx, ph in enumerate(PHONE_DEF_SIL):
            bc_name = PHONE_TO_BROAD[ph]
            bc_idx = BROAD_TO_INDEX[bc_name]
            lut[ph_idx + 1] = bc_idx  # ph_idx is 0-39, store at lut[1-40]
    else:
        # 0-indexed: lut[0] through lut[39] map to phones
        for ph_idx, ph in enumerate(PHONE_DEF_SIL):
            bc_name = PHONE_TO_BROAD[ph]
            bc_idx = BROAD_TO_INDEX[bc_name]
            lut[ph_idx] = bc_idx
    
    return lut


def phones_to_broad_indices(
    phone_idx_seq: torch.Tensor,
    lut: Optional[torch.Tensor] = None,
    one_indexed: bool = True,
) -> torch.Tensor:
    """
    Map a 1D tensor of phone indices to broad-class indices.
    
    Args:
        phone_idx_seq: LongTensor [T_ph] - phone indices (1-indexed: 0=blank, 1-40=phones for CTC)
        lut: optional LUT (will be built if None)
        one_indexed: If True, assumes phone_idx_seq is 1-indexed (matches CTC target format)
    Returns:
        broad_idx_seq: LongTensor [T_ph] with same shape as phone_idx_seq
    """
    if lut is None:
        lut = build_phone_to_broad_lut(phone_idx_seq.device, one_indexed=one_indexed)
    
    nPhones = len(PHONE_DEF_SIL)
    
    if one_indexed:
        # 1-indexed targets: valid range is [0, nPhones] where 0=blank, 1-40=phones
        max_valid = nPhones  # 40
        phone_idx_seq_clamped = phone_idx_seq.clamp(min=0, max=max_valid)
    else:
        # 0-indexed targets: valid range is [0, nPhones-1]
        max_valid = nPhones - 1  # 39
        phone_idx_seq_clamped = phone_idx_seq.clamp(min=0, max=max_valid)
    
    # Warn if there were out-of-bounds indices
    if torch.any(phone_idx_seq != phone_idx_seq_clamped):
        out_of_bounds_mask = (phone_idx_seq < 0) | (phone_idx_seq > max_valid if one_indexed else phone_idx_seq >= max_valid + 1)
        n_oob = out_of_bounds_mask.sum().item()
        max_idx = phone_idx_seq.max().item()
        min_idx = phone_idx_seq.min().item()
        expected_max = max_valid
        expected_range = f"[0, {expected_max}]" if one_indexed else f"[0, {expected_max}]"
        print(f"⚠️ WARNING: Found {n_oob} out-of-bounds phone indices. "
              f"Range was [{min_idx}, {max_idx}], expected {expected_range}. "
              f"Clamping to valid range.")
    
    return lut[phone_idx_seq_clamped]


# ---------------------------------------------------------
# Utilities (parity with BiT / your original longformer)
# ---------------------------------------------------------

def get_sinusoidal_pos_emb(seq_len, dim, device=None):
    position = torch.arange(seq_len, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, device=device) * -(math.log(10000.0) / dim))
    pe = torch.zeros(seq_len, dim, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class PatchEmbedLN(nn.Module):
    def __init__(self, in_dim: int, embed_dim: int, eps: float = 1e-5):
        super().__init__()
        self.ln1 = nn.LayerNorm(in_dim, eps=eps)
        self.proj = nn.Linear(in_dim, embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, L, in_dim]
        return self.ln2(self.proj(self.ln1(x)))


class RelativePositionBias1D(nn.Module):
    def __init__(self, num_heads: int, max_distance: int = 4096):
        super().__init__()
        self.max_distance = max_distance
        self.bias = nn.Parameter(torch.zeros(2 * max_distance + 1, num_heads))
        nn.init.trunc_normal_(self.bias, std=0.02)

    def forward(self, Lq: int, Lk: int) -> torch.Tensor:
        pos_q = torch.arange(Lq)
        pos_k = torch.arange(Lk)
        rel = pos_q[:, None] - pos_k[None, :]
        rel = rel.clamp(-self.max_distance, self.max_distance) + self.max_distance
        b = self.bias[rel.long()]  # [Lq,Lk,H]
        return b.permute(2, 0, 1).contiguous()  # [H,Lq,Lk]


# ---------------------------------------------------------
# Local-Global (Longformer-style) causal attention
# ---------------------------------------------------------

class LocalGlobalCausalAttention(nn.Module):
    """Sliding-window causal attention + G global tokens."""
    def __init__(self, dim: int, heads: int, dim_head: int, window: int, num_global: int,
                 dropout: float = 0.1, use_relative_bias: bool = True):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.inner = heads * dim_head
        self.window = window
        self.num_global = num_global
        self.scale = dim_head ** -0.5
        self.use_relative_bias = use_relative_bias

        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * self.inner, bias=False)
        self.out = nn.Sequential(nn.Linear(self.inner, dim), nn.Dropout(dropout))
        self.dropout = nn.Dropout(dropout)
        self.rel = RelativePositionBias1D(heads) if use_relative_bias else None

    def _build_mask(self, L: int, device: torch.device):
        G = self.num_global
        N = G + L
        mask = torch.full((N, N), float('-inf'), device=device)

        # Global queries: causal over all tokens
        idx = torch.arange(N, device=device)
        causal = (idx[None, :] <= idx[:, None])  # [N,N] lower-triangular
        mask[:G, :] = torch.where(causal[:G, :], 0.0, float('-inf'))

        # Time-token queries: local causal window + all globals
        for t in range(L):
            q = G + t
            mask[q, :G] = 0.0
            start = max(0, t - (self.window - 1))
            end = t
            if start <= end:
                mask[q, G + start:G + end + 1] = 0.0
        return mask  # [N,N]

    def forward(self, x: torch.Tensor, global_tokens: torch.Tensor) -> torch.Tensor:
        # x: [B,L,D], global_tokens: [B,G,D]
        B, L, D = x.shape
        G = self.num_global
        assert global_tokens.size(1) == G
        H, Dh = self.heads, self.dim_head

        z = torch.cat([global_tokens, self.norm(x)], dim=1)  # [B, G+L, D]
        qkv = self.qkv(z).reshape(B, G + L, 3, H, Dh)
        q, k, v = qkv.unbind(dim=2)  # [B,N,H,Dh]
        q, k, v = (t.permute(0, 2, 1, 3) for t in (q, k, v))  # [B,H,N,Dh]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B,H,N,N]
        if self.use_relative_bias:
            bias = self.rel(G + L, G + L)[None]  # [1,H,N,N]
            attn = attn + bias

        m = self._build_mask(L, x.device)[None, None, :, :]  # [1,1,N,N]
        attn = attn + m
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        y = attn @ v  # [B,H,N,Dh]
        y = y.permute(0, 2, 1, 3).reshape(B, G + L, H * Dh)
        y = self.out(y)  # [B,G+L,D]
        return y[:, G:, :]  # drop globals


# ---------------------------------------------------------
# FFN / Transformer block
# ---------------------------------------------------------

class FFN(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class LGViTBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_ratio, window, num_global, dropout=0.1, use_relative_bias=True):
        super().__init__()
        self.attn = LocalGlobalCausalAttention(dim, heads, dim_head, window, num_global, dropout, use_relative_bias)
        self.ffn = FFN(dim, int(dim * mlp_ratio), dropout)

    def forward(self, x, global_tokens):
        x = x + self.attn(x, global_tokens)
        x = x + self.ffn(x)
        return x


# ---------------------------------------------------------
# Simple causal Gaussian smoother
# ---------------------------------------------------------

class CausalGaussianSmoother(nn.Module):
    def __init__(self, width: float, feat_dim: int):
        super().__init__()
        self.enabled = width is not None and width > 0
        if not self.enabled:
            self.register_buffer("kernel", torch.ones(1))
        else:
            sigma = float(width)
            k = max(3, int(6 * sigma) | 1)
            rad = k // 2
            x = torch.arange(-rad, rad + 1, dtype=torch.float32)
            g = torch.exp(-0.5 * (x / sigma) ** 2)
            g[x > 0] = 0
            g = g / g.sum().clamp(min=1e-8)
            self.register_buffer("kernel", g[None, None, :])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return x
        B, F, T = x.shape
        kc = self.kernel.repeat(F, 1, 1)
        y = torch.nn.functional.conv1d(x, kc, padding=self.kernel.shape[-1]-1, groups=F)
        return y[:, :, :T]


# ---------------------------------------------------------
# Hierarchical Local-Global ViT (no inheritance)
# ---------------------------------------------------------

class HiLocalGlobalViT_Phoneme(nn.Module):
    """
    BIT-compatible Local-Global (Longformer-style) ViT with two CTC heads:

    - Main head: phoneme logits over PHONE_DEF_SIL (nPhones = 40) + blank
    - Aux head:  broad-class logits over BROAD_CLASS_DEF (nBroad = 8) + blank
    """

    def __init__(
        self,
        *,
        patch_size: Tuple[int, int] = (5, 256),
        dim: int = 384,
        depth: int = 6,
        heads: int = 6,
        mlp_dim_ratio: float = 4.0,
        dim_head: int = 64,
        dropout: float = 0.1,
        input_dropout: float = 0.2,
        gaussianSmoothWidth: float = 2.0,
        T5_style_pos: bool = True,
        max_mask_pct: float = 0.0,
        num_masks: int = 0,
        mask_token_zeros: bool = False,
        num_masks_channels: int = 0,
        max_mask_channels: int = 0,
        dist_dict_path: str = "",
        consistency: bool = False,
        window: int = 5,
        num_global_tokens: int = 2,
        eps: float = 1e-5,
    ):
        super().__init__()

        nPhones = len(PHONE_DEF_SIL)
        nBroad = len(BROAD_CLASS_DEF)

        patch_height, patch_width = patch_size
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.patch_dim = patch_height * patch_width
        self.dim = dim
        self.nClasses_phone = nPhones
        self.nClasses_broad = nBroad
        self.gaussianSmoothWidth = gaussianSmoothWidth
        self.T5_style_pos = T5_style_pos
        self.max_mask_pct = max_mask_pct
        self.num_masks = num_masks
        self.num_masks_channels = num_masks_channels
        self.max_channels_to_mask = max_mask_channels
        self.dist_dict_path = dist_dict_path
        self.consistency = consistency

        self.dropout_in = nn.Dropout(input_dropout)
        self.gaussianSmoother = CausalGaussianSmoother(gaussianSmoothWidth, feat_dim=patch_width)

        # mask token
        if mask_token_zeros:
            self.mask_token = nn.Parameter(torch.zeros(self.patch_dim), requires_grad=False)
        else:
            self.mask_token = nn.Parameter(torch.randn(self.patch_dim))

        # embedding
        self.to_patch = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width)
        self.patch_to_emb = nn.Sequential(
            nn.LayerNorm(self.patch_dim, eps=eps),
            nn.Linear(self.patch_dim, dim),
            nn.LayerNorm(dim, eps=eps),
        )
        self.to_patch_embedding = nn.Sequential(self.to_patch, *self.patch_to_emb)

        # learned global tokens (shared across layers)
        self.num_global_tokens = num_global_tokens
        self.global_tokens = nn.Parameter(torch.randn(1, num_global_tokens, dim) * 0.02)

        # blocks
        self.blocks = nn.ModuleList([
            LGViTBlock(dim, heads, dim_head, mlp_dim_ratio, window, num_global_tokens,
                       dropout, use_relative_bias=T5_style_pos)
            for _ in range(depth)
        ])

        self.norm_out = nn.LayerNorm(dim, eps=eps)
        # +1 for CTC blank
        self.projection_phone = nn.Linear(dim, nPhones + 1)
        self.projection_broad = nn.Linear(dim, nBroad + 1)

        if self.T5_style_pos is False:
            self.register_buffer('pos_embedding', None, persistent=False)

    # ---- helpers (copied from original) ----

    def compute_length(self, X_len: torch.Tensor) -> torch.Tensor:
        return torch.ceil(X_len / self.patch_height).to(dtype=torch.int32)

    def apply_original_augs(self, neuralInput: torch.Tensor, n_masks_nptl_augs: int, aug_values):
        device = neuralInput.device
        neuralInput = neuralInput.repeat_interleave(n_masks_nptl_augs, dim=0)
        neuralInput += torch.randn_like(neuralInput) * aug_values[0]
        neuralInput += (
            torch.randn([neuralInput.shape[0], 1, neuralInput.shape[2]], device=device)
            * aug_values[1]
        )
        return neuralInput

    def apply_time_mask(self, X: torch.Tensor, X_len: torch.Tensor,
                        constant_mask: bool = False, mask_range: List[int] = []):
        B, P, D = X.shape
        device = X.device
        if self.num_masks <= 0 or self.max_mask_pct <= 0:
            return X, torch.zeros(B, P, dtype=torch.bool, device=device)

        if constant_mask:
            valid_lens = torch.min((X_len // self.patch_height).to(device)).repeat(B)
        else:
            valid_lens = (X_len // self.patch_height).to(device)

        max_mask_lens = (self.max_mask_pct * valid_lens).long()
        B_rep = B * self.num_masks
        valid_lens_rep = valid_lens.repeat_interleave(self.num_masks)
        max_mask_lens_rep = max_mask_lens.repeat_interleave(self.num_masks)

        if constant_mask:
            t = (torch.rand(self.num_masks, device=device).repeat(B)
                 * (max_mask_lens_rep + 1).float()).floor().long().clamp(min=1)
        else:
            t = (torch.rand(B_rep, device=device)
                 * (max_mask_lens_rep + 1).float()).floor().long()

        max_start = (valid_lens_rep - t + 1).clamp(min=1)
        if constant_mask:
            t0 = (torch.rand(self.num_masks, device=device).repeat(B)
                  * max_start.float()).floor().long()
        else:
            t0 = (torch.rand(B_rep, device=device)
                  * max_start.float()).floor().long()

        arange = torch.arange(P, device=device).unsqueeze(0)
        mask_chunks = (arange >= t0.unsqueeze(1)) & (arange < (t0 + t).unsqueeze(1))
        batch_idx = torch.arange(B, device=device).repeat_interleave(self.num_masks)
        patch_idx = mask_chunks.nonzero(as_tuple=False)
        b_indices = batch_idx[patch_idx[:, 0]]
        p_indices = patch_idx[:, 1]

        mask = torch.zeros(B, P, dtype=torch.bool, device=device)
        mask[b_indices, p_indices] = True

        X_masked = X.clone()
        X_masked[mask] = self.mask_token
        return X_masked, mask

    # ---- forward ----

    def forward(self, neuralInput: torch.Tensor, X_len: torch.Tensor, day_idx=None):
        """
        Args:
            neuralInput: [B, T, C]
            X_len: [B] original time lengths
        Returns:
            logits_phone: [B, P, nPhones+1]
            logits_broad: [B, P, nBroad+1]
        """
        # pad to multiple of patch height
        T = neuralInput.shape[1]
        rem = T % self.patch_height
        if rem != 0:
            pad = self.patch_height - rem
            pad_tensor = torch.zeros(
                neuralInput.shape[0], pad, neuralInput.shape[2],
                device=neuralInput.device, dtype=neuralInput.dtype
            )
            neuralInput = torch.cat([neuralInput, pad_tensor], dim=1)

        # gaussian smoother over features
        x = torch.permute(neuralInput, (0, 2, 1))  # [B,F,T]
        x = self.gaussianSmoother(x)
        x = torch.permute(x, (0, 2, 1))            # [B,T,F]

        x = x.unsqueeze(1)  # [B,1,T,F]

        if self.training and self.max_mask_pct > 0:
            tokens = self.to_patch(x)
            if self.consistency:
                x1, _ = self.apply_time_mask(tokens, X_len)
                x2, _ = self.apply_time_mask(tokens, X_len)
                tokens = torch.cat([x1, x2], dim=0)
            else:
                tokens, _ = self.apply_time_mask(tokens, X_len)
            tokens = self.patch_to_emb(tokens)
        else:
            tokens = self.to_patch_embedding(x)  # [B,P,D]

        tokens = self.dropout_in(tokens)

        B, P, D = tokens.shape

        if self.T5_style_pos is False:
            pos = get_sinusoidal_pos_emb(P, self.dim, device=tokens.device)
            tokens = tokens + pos.unsqueeze(0)

        # repeat global tokens per batch
        g = self.global_tokens.repeat(B, 1, 1)

        # pass through blocks
        for blk in self.blocks:
            tokens = blk(tokens, g)

        h = self.norm_out(tokens)
        logits_phone = self.projection_phone(h)
        logits_broad = self.projection_broad(h)
        return logits_phone, logits_broad


# ---------------------------------------------------------
# Hierarchical CTC loss helper
# ---------------------------------------------------------

def compute_hierarchical_ctc_loss(
    logits_phone: torch.Tensor,          # [B, T, nPhones+1]
    logits_broad: torch.Tensor,          # [B, T, nBroad+1]
    input_lengths: torch.Tensor,         # [B] in patch steps
    phone_targets: torch.Tensor,         # [sum_T_ph]
    phone_target_lengths: torch.Tensor,  # [B]
    blank_index: int = 0,
    hier_aux_weight: float = 0.3,
) -> TypingTuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Combined CTC loss:
        L = L_phone + hier_aux_weight * L_broad
    """

    assert logits_phone.shape[:2] == logits_broad.shape[:2], \
        "Phone and broad logits must have same [B, T] shape"

    device = logits_phone.device
    input_lengths = input_lengths.to(device=device, dtype=torch.long)
    phone_target_lengths = phone_target_lengths.to(device=device, dtype=torch.long)
    phone_targets = phone_targets.to(device=device, dtype=torch.long)

    # [T,B,C]
    logp_phone = logits_phone.log_softmax(dim=-1).transpose(0, 1)
    logp_broad = logits_broad.log_softmax(dim=-1).transpose(0, 1)

    ctc_phone = nn.CTCLoss(blank=blank_index, zero_infinity=True)
    ctc_broad = nn.CTCLoss(blank=blank_index, zero_infinity=True)

    loss_phone = ctc_phone(
        logp_phone,
        phone_targets,
        input_lengths,
        phone_target_lengths,
    )

    lut = build_phone_to_broad_lut(device, one_indexed=True)
    broad_targets = phones_to_broad_indices(phone_targets, lut=lut, one_indexed=True)
    broad_target_lengths = phone_target_lengths  # same lengths

    loss_broad = ctc_broad(
        logp_broad,
        broad_targets,
        input_lengths,
        broad_target_lengths,
    )

    loss_total = loss_phone + hier_aux_weight * loss_broad
    return loss_total, loss_phone.detach(), loss_broad.detach()
