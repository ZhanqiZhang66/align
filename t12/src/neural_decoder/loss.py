import torch
import torch.nn as nn
from typing import Dict, Iterable, List, Optional, TextIO, Tuple, Union
from torch import Tensor
from typing import Optional
import torch.nn.functional as F


# code from Icefall package
def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    """
    Args:
      lengths:
        A 1-D tensor containing sentence lengths.
      max_len:
        The length of masks.
    Returns:
      Return a 2-D bool tensor, where masked positions
      are filled with `True` and non-masked positions are
      filled with `False`.

    >>> lengths = torch.tensor([1, 3, 2, 5])
    >>> make_pad_mask(lengths)
    tensor([[False,  True,  True,  True,  True],
            [False, False, False,  True,  True],
            [False, False,  True,  True,  True],
            [False, False, False, False, False]])
    """
    assert lengths.ndim == 1, lengths.ndim
    max_len = max(max_len, lengths.max())
    n = lengths.size(0)
    seq_range = torch.arange(0, max_len, device=lengths.device)
    expaned_lengths = seq_range.unsqueeze(0).expand(n, max_len)

    return expaned_lengths >= lengths.unsqueeze(-1)
    
def kl(p, q):
    # p: log-probs with no grad, q: log-probs with grad
    return F.kl_div(q, p, log_target=True, reduction="none")

def kl_phone_prior_loss(
    encoder_out: torch.Tensor,      # (N,T,C)
    p_prior: torch.Tensor,     # (C-1,) excluding blank
    blank_id: int = 0,
    eps: float = 1e-8
) -> torch.Tensor:
    probs = encoder_out.softmax(-1)                       # (N,T,C)

    p_blank = probs[..., blank_id]                   # (N,T)
    w = (1.0 - p_blank).detach()                     # (N,T) don't let it game the weights
    w = torch.clamp(w, min=0.01)                     # avoid zero weight
    w = w / (w.sum() + eps)                          # normalize over all frames

    q = (probs[..., 1:] * w[..., None]).sum(dim=(0,1))   # (C-1,)
    q = q / (q.sum() + eps)

    kl = (p_prior * (p_prior.log() - (q + eps).log())).sum()
    return kl

def forward_ctc(
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Compute CTC loss.
        Args:
          encoder_out:
            Encoder output, of shape (N, T, C).
          encoder_out_lens:
            Encoder output lengths, of shape (N,).
          targets:
            Target Tensor of shape (sum(target_lengths)). The targets are assumed
            to be un-padded and concatenated within 1 dimension.
        """
        # Compute CTC log-prob
        ctc_output = encoder_out.log_softmax(2) # (N, T, C)

        ctc_loss = torch.nn.functional.ctc_loss(
            log_probs=ctc_output.permute(1, 0, 2).cpu(),  # (T, N, C)
            targets=targets.cpu(),
            input_lengths=encoder_out_lens.cpu(),
            target_lengths=target_lengths.cpu(),
            reduction="mean",
        )
        return ctc_loss
    
def entropy_min_loss(
    encoder_out: torch.Tensor,          # [N, T, C]
    encoder_out_lens: torch.Tensor,     # [N]
    *,
    blank_id: int = 0,
    exclude_blank: bool = True,
    exclude_sil: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Minimize entropy of model posteriors to encourage confident (spiky) logits.
    CTC-aware: entropy is on conditional non-blank distribution.
    """

    N, T, C = encoder_out.shape
    probs = torch.softmax(encoder_out, dim=-1)  # [N,T,C]

    # -------- nonblank mass (important for weighting) --------
    p_blank = probs[..., blank_id]              # [N,T]
    nonblank_mass = (1.0 - p_blank).detach()   # <- critical trick

    # -------- remove blank correctly --------
    if exclude_blank:
        if blank_id == 0:
            probs_nb = probs[..., 1:]
        elif blank_id == C - 1:
            probs_nb = probs[..., :-1]
        else:
            probs_nb = torch.cat(
                [probs[..., :blank_id], probs[..., blank_id+1:]],
                dim=-1
            )

        # condition on non-blank
        probs_nb = probs_nb / (probs_nb.sum(-1, keepdim=True) + eps)
        p = probs_nb
    else:
        p = probs

    # -------- remove SIL safely (assumes SIL is last class in your mapping) --------
    if exclude_sil:
        p = p[..., :-1]
        p = p / (p.sum(-1, keepdim=True) + eps)

    # -------- entropy per frame --------
    ent = -(p * (p + eps).log()).sum(dim=-1)  # [N,T]

    # -------- mask to valid frames --------
    t = torch.arange(T, device=encoder_out.device)[None, :].expand(N, T)
    valid = (t < encoder_out_lens[:, None]).float()

    # -------- weight by nonblank mass --------
    valid = valid * nonblank_mass

    return (ent * valid).sum() / (valid.sum() + eps)


def weighted_forward_ctc(
    encoder_out: torch.Tensor,        # (N, T, C)
    encoder_out_lens: torch.Tensor,  # (N,)
    targets: torch.Tensor,           # (sum target lengths)
    target_lengths: torch.Tensor,    # (N,)
    class_weights: torch.Tensor,     # (C,)
    blank: int = 0,
) -> torch.Tensor:

    N, T, C = encoder_out.shape

    # log probs
    log_probs = encoder_out.log_softmax(2)  # (N, T, C)

    # 🔥 Apply class weights in log-space
    # broadcast: (N, T, C) * (C,) -> (N, T, C)
    class_weights = class_weights.to(device=log_probs.device, dtype=log_probs.dtype)
    weighted_log_probs = log_probs * class_weights.view(1, 1, C)

    # CTC expects (T, N, C)
    weighted_log_probs = weighted_log_probs.permute(1, 0, 2)

    loss = F.ctc_loss(
        log_probs=weighted_log_probs.cpu(),
        targets=targets.cpu(),
        input_lengths=encoder_out_lens.cpu(),
        target_lengths=target_lengths.cpu(),
        blank=blank,
        reduction="mean",
        zero_infinity=True,
    )

    return loss



def forward_cr_ctc(
    encoder_out: torch.Tensor,      # (2N, T, C)
    encoder_out_lens: torch.Tensor, # (2N,)
    targets: torch.Tensor,
    target_lengths: torch.Tensor,
):
    """
    Compute CTC + Consistency Regularization loss (CR-CTC).
    """

    # ---------- CTC LOSS ----------
    ctc_output = encoder_out.log_softmax(dim=2)  # (2N, T, C)

    ctc_loss = F.ctc_loss(
        log_probs=ctc_output.permute(1, 0, 2),  # (T,2N,C)
        targets=targets.cpu(),
        input_lengths=encoder_out_lens.cpu(),
        target_lengths=target_lengths.cpu(),
        reduction="sum",
    )
    # caller will multiply by 0.5 outside (since batch doubling)

    # ---------- CR LOSS (bidirectional KL) ----------
    B, T, C = ctc_output.shape
    assert B % 2 == 0
    N = B // 2

    z1 = ctc_output[:N]     # (N,T,C)
    z2 = ctc_output[N:]     # (N,T,C)

    # stop-gradient
    z1_det = z1.detach()
    z2_det = z2.detach()

    # KL(target || input) — but using log probabilities
    def kl(log_p_target, log_q_input):
        return F.kl_div(
            log_q_input,        # input (log q, receives gradient)
            log_p_target,       # target (log p, no gradient)
            reduction="none",
            log_target=True,
        )

    # bidirectional KL
    kl1 = kl(z2_det, z1)  # KL(sg(z2) || z1)  # (N,T,C)
    kl2 = kl(z1_det, z2)  # KL(sg(z1) || z2)  # (N,T,C)

    # ---------- MASKING ----------
    lengths1 = encoder_out_lens[:N]
    lengths2 = encoder_out_lens[N:]

    mask1 = make_pad_mask(lengths1, max_len=T).unsqueeze(-1)  # (N,T,1)
    mask2 = make_pad_mask(lengths2, max_len=T).unsqueeze(-1)  # (N,T,1)

    # Mask kl1 with mask1 (for X1 lengths) and kl2 with mask2 (for X2 lengths)
    kl1 = kl1.clone()
    kl2 = kl2.clone()
    kl1.masked_fill_(mask1, 0.0)
    kl2.masked_fill_(mask2, 0.0)

    # Sum the masked KL divergences
    cr_loss = (kl1 + kl2).sum()

    return ctc_loss, cr_loss



def memo_loss_from_logits(
    logits_aug: Tensor,
    adjusted_len: int,
    blank_id: Optional[int] = 0,
    T: float = 1
) -> Tensor:
  
    """
    Computes negative entropy loss from augmented logits.

    Parameters
    ----------
    logits_aug : Tensor
        Logits from multiple augmentations. Shape: [n_aug, T, D]
    adjusted_len : int
        If provided, truncate to this length along time dimension.
    blank_id : Optional[int]
        If not None, filter out time steps where the most likely
        token is the blank_id.
    T : float
        Temperature for softmax. 
        
    Returns
    -------
    loss : Tensor
        Scalar loss tensor (requires grad).
    """
    
    logits_aug /= T # temperature scaling
    probs_aug = torch.nn.functional.softmax(logits_aug, dim=-1)   # [n_aug, T, D]
    marginal_probs = probs_aug.mean(dim=0)                        # [T, D]
    marginal_probs = marginal_probs[:adjusted_len] # [T', D], where T' <= T

    if blank_id is not None:
        max_indices = marginal_probs.argmax(dim=1)
        marginal_probs = marginal_probs[max_indices != blank_id]

    loss = - (marginal_probs * marginal_probs.log()).sum(dim=-1).mean()
    return loss



def future_prediction_loss(reps, lengths, predictor, steps=(1, 2), loss_type="cosine"):
    """
    reps:      (B, T, D) tensor of representations (here: time x logits)
    lengths:   (B,) effective sequence length for each item (after model.compute_length)
    predictor: nn.Module mapping D -> D
    steps:     iterable of positive integers (how many patches ahead to predict)
    loss_type: "cosine" or "mse"
    """
    B, T, D = reps.shape
    device = reps.device

    if isinstance(steps, int):
        steps = [steps]

    losses = []
    for k in steps:
        if k <= 0 or k >= T:
            continue

        # reps[:, :-k] predicts reps[:, k:]
        src = reps[:, :-k, :]   # (B, T-k, D)
        tgt = reps[:, k:,  :]   # (B, T-k, D)

        # Build mask so we only use valid time steps for each sequence
        max_valid = lengths - k                # (B,)
        t_idx = torch.arange(T - k, device=device)[None, :]  # (1, T-k)
        mask = t_idx < max_valid[:, None]      # (B, T-k)

        if not mask.any():
            continue

        src = src[mask]        # (N_valid, D)
        tgt = tgt[mask]        # (N_valid, D)

        pred = predictor(src)  # (N_valid, D)

        if loss_type == "cosine":
            pred_norm = F.normalize(pred, dim=-1)
            tgt_norm = F.normalize(tgt, dim=-1)
            loss_k = 1.0 - (pred_norm * tgt_norm).sum(dim=-1).mean()
        else:  # mse
            loss_k = F.mse_loss(pred, tgt)

        losses.append(loss_k)

    if len(losses) == 0:
        return reps.new_tensor(0.0)

    return sum(losses) / len(losses)

import torch
import torch.nn.functional as F


def phone_contrastive_loss(
    reps,          # [B, T, D] encoder representations
    phone_ids,     # [B, T]    int64, -1 where invalid
    valid_mask,    # [B, T]    bool, True for real frames
    temperature=0.1,
    max_frames=1024,
):
    """
    Hard InfoNCE with token-level alignment from runs.

    Positives = frames with the *same phone_id* (>=0).
    Only uses frames where valid_mask & phone_ids>=0.
    """
    B, T, D = reps.shape
    device = reps.device

    valid = valid_mask & (phone_ids >= 0)
    if not valid.any():
        return reps.new_tensor(0.0)

    # flatten
    reps_flat  = reps[valid]           # [N, D]
    labels_flat = phone_ids[valid]     # [N]

    N = reps_flat.size(0)
    if N == 0:
        return reps.new_tensor(0.0)

    # optional subsampling for efficiency
    if N > max_frames:
        idx = torch.randperm(N, device=device)[:max_frames]
        reps_flat   = reps_flat[idx]
        labels_flat = labels_flat[idx]
        N = max_frames

    # normalize embeddings
    reps_flat = F.normalize(reps_flat, dim=-1)

    # similarity matrix
    sim = reps_flat @ reps_flat.t() / temperature   # [N, N]

    # mask self-similarity
    mask = torch.ones_like(sim, dtype=torch.bool)
    mask.fill_(True)
    mask.fill_diagonal_(False)

    # positives: same label
    labels_equal = labels_flat.unsqueeze(0) == labels_flat.unsqueeze(1)  # [N, N]
    pos_mask = labels_equal & mask

    # for each anchor i, positives j where pos_mask[i,j]
    # InfoNCE: log (sum_j exp(sim_ij) over positives / sum_j exp(sim_ij) over all j≠i)

    # logsumexp over all non-self j
    sim_masked = sim.masked_fill(~mask, float('-inf'))
    log_denom = torch.logsumexp(sim_masked, dim=-1)  # [N]

    # logsumexp over positives (may be -inf if no positives)
    sim_pos = sim.masked_fill(~pos_mask, float('-inf'))
    log_num = torch.logsumexp(sim_pos, dim=-1)       # [N]

    # anchors that actually have at least one positive
    valid_anchor = torch.isfinite(log_num)
    if valid_anchor.sum() == 0:
        return reps.new_tensor(0.0)

    loss_per_i = -(log_num[valid_anchor] - log_denom[valid_anchor])
    return loss_per_i.mean()

def cross_trial_phone_contrastive_loss(
    reps,          # [B, T, D] encoder representations
    phone_ids,     # [B, T]    int64, -1 where invalid
    valid_mask,    # [B, T]    bool, True for real frames
    temperature: float = 0.1,
    max_frames: int = 1024,
):
    """
    InfoNCE over frames, with *cross-trial* positives:
      - Positives: same phone_id, but from a *different* sequence in the batch.
      - Negatives: all other frames (different phone OR same phone but same sequence).
      - Ignores padding/blank frames via valid_mask & phone_ids >= 0.

    Args:
        reps: [B, T, D] encoder representations.
        phone_ids: [B, T] int64, -1 wherever we don't trust the label.
        valid_mask: [B, T] bool, True for non-padded frames.
        temperature: softmax temperature.
        max_frames: optional cap on total frames used for efficiency.

    Returns:
        Scalar contrastive loss (tensor).
    """
    B, T, D = reps.shape
    device = reps.device

    # Only keep real, labeled frames
    valid = valid_mask & (phone_ids >= 0)
    if not valid.any():
        return reps.new_tensor(0.0)

    # Flatten
    reps_flat   = reps[valid]          # [N, D]
    labels_flat = phone_ids[valid]     # [N]

    # Sequence id for each frame: 0..B-1
    batch_ids = torch.arange(B, device=device).unsqueeze(1).expand(B, T)
    batch_ids_flat = batch_ids[valid]  # [N]

    N = reps_flat.size(0)
    if N == 0:
        return reps.new_tensor(0.0)

    # Optional subsampling for efficiency
    if N > max_frames:
        idx = torch.randperm(N, device=device)[:max_frames]
        reps_flat   = reps_flat[idx]
        labels_flat = labels_flat[idx]
        batch_ids_flat = batch_ids_flat[idx]
        N = max_frames

    # Normalize embeddings
    reps_flat = F.normalize(reps_flat, dim=-1)  # [N, D]

    # Similarity matrix
    sim = reps_flat @ reps_flat.t() / temperature  # [N, N]

    # Mask out self-similarity
    mask = torch.ones_like(sim, dtype=torch.bool, device=device)
    mask.fill_diagonal_(False)  # j != i

    # Positives: same label, different sequence, not self
    same_label = labels_flat.unsqueeze(0) == labels_flat.unsqueeze(1)   # [N, N]
    same_seq   = batch_ids_flat.unsqueeze(0) == batch_ids_flat.unsqueeze(1)  # [N, N]
    pos_mask   = same_label & (~same_seq) & mask                        # [N, N]

    # Denominator: all non-self frames
    sim_all = sim.masked_fill(~mask, float('-inf'))
    log_denom = torch.logsumexp(sim_all, dim=-1)  # [N]

    # Numerator: only positives
    sim_pos = sim.masked_fill(~pos_mask, float('-inf'))
    log_num = torch.logsumexp(sim_pos, dim=-1)    # [N]

    # Only keep anchors that have at least one positive
    has_pos = torch.isfinite(log_num)
    if has_pos.sum() == 0:
        # No cross-trial positives in this batch
        return reps.new_tensor(0.0)

    loss_per_anchor = -(log_num[has_pos] - log_denom[has_pos])
    return loss_per_anchor.mean()

import torch
from edit_distance import SequenceMatcher


def ctc_run_alignment_phone_ids(
    logits,          # [B, T, C] raw logits
    targets,         # [B, U_max]
    input_lengths,   # [B]
    target_lengths,  # [B]
    blank=0,
):
    """
    For each example:
      1) greedy path = argmax_t logits[t]
      2) collapse runs in time
      3) align run-sequence to ground-truth target sequence with edit distance
      4) assign each *run* to a target phone index where aligned ('equal')
         and mark all its frames with that phone id.

    Returns:
      phone_ids:  [B, T] int64, -1 where we don't trust the assignment
      valid_mask: [B, T] bool, True where t < input_length
    """
    device = logits.device
    B, T, C = logits.shape

    phone_ids = torch.full(
        (B, T), fill_value=-1, dtype=torch.long, device=device
    )
    valid_mask = torch.zeros((B, T), dtype=torch.bool, device=device)

    with torch.no_grad():
        # greedy path over time
        path = torch.argmax(logits, dim=-1)  # [B, T]

        for b in range(B):
            T_b = int(input_lengths[b].item())
            U_b = int(target_lengths[b].item())
            if T_b == 0 or U_b == 0:
                continue

            valid_mask[b, :T_b] = True

            # --- 1) time-limited path and gt sequence ---
            path_b = path[b, :T_b].cpu().tolist()         # length T_b
            gt_b   = targets[b, :U_b].cpu().tolist()      # length U_b

            # --- 2) build runs over time ---
            run_symbols = []   # symbol per run (includes blanks)
            run_starts  = []   # inclusive
            run_ends    = []   # exclusive

            prev_sym = None
            for t in range(T_b):
                sym = path_b[t]
                if t == 0 or sym != prev_sym:
                    # start new run
                    run_symbols.append(sym)
                    run_starts.append(t)
                    run_ends.append(t + 1)
                else:
                    # extend current run
                    run_ends[-1] = t + 1
                prev_sym = sym

            num_runs = len(run_symbols)
            if num_runs == 0:
                continue

            # --- 3) build sequence of non-blank runs for alignment ---
            alignable_run_indices = [
                r for r, s in enumerate(run_symbols) if s != blank
            ]
            if len(alignable_run_indices) == 0:
                # everything is blank, nothing to align
                continue

            run_seq = [run_symbols[r] for r in alignable_run_indices]

            # --- 4) align run_seq to ground truth via edit distance ---
            # a = ground truth, b = run_seq
            sm = SequenceMatcher(a=gt_b, b=run_seq)
            opcodes = sm.get_opcodes()

            # map from *original* run index -> gt index, or None
            run_to_gt = {r: None for r in range(num_runs)}

            for tag, i1, i2, j1, j2 in opcodes:
                # a[i1:i2] ↔ b[j1:j2]
                if tag == "equal":
                    # same symbols – trust these
                    length = min(i2 - i1, j2 - j1)
                    for k in range(length):
                        gt_idx = i1 + k          # in gt_b
                        run_align_idx = j1 + k   # index in run_seq
                        orig_run_idx = alignable_run_indices[run_align_idx]
                        run_to_gt[orig_run_idx] = gt_idx
                # for 'replace', 'insert', 'delete' we leave mapping as None

            # --- 5) assign per-frame phone ids from run mapping ---
            for r in range(num_runs):
                gt_idx = run_to_gt[r]
                if gt_idx is None:
                    continue  # we don't trust this run
                phone_label = gt_b[gt_idx]
                start_t = run_starts[r]
                end_t   = run_ends[r]
                phone_ids[b, start_t:end_t] = phone_label

    return phone_ids, valid_mask

def forward_ctc_ntp(
    encoder_out: torch.Tensor,
    encoder_out_lens: torch.Tensor,
    targets: torch.Tensor,
    target_lengths: torch.Tensor,
    ntp_logits: Optional[torch.Tensor] = None,
    ntp_targets: Optional[torch.Tensor] = None,
    lambda_ntp: float = 0.1,
    ntp_ignore_index: int = -100,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute combined CTC + NTP loss.

    Args:
      encoder_out:
        Encoder output, of shape (N, T, C).
      encoder_out_lens:
        Encoder output lengths, of shape (N,).
      targets:
        1D target tensor of shape (sum(target_lengths,)), unpadded and concatenated,
        as expected by torch.nn.functional.ctc_loss.
      target_lengths:
        Target lengths, shape (N,).
      ntp_logits:
        Optional NTP logits of shape (N, T_ntp, C).
      ntp_targets:
        Optional NTP targets of shape (N, T_ntp), using ntp_ignore_index for padding.
        You are responsible for aligning these to whatever time grid you want.
      lambda_ntp:
        Weight on the NTP loss term.
      ntp_ignore_index:
        Ignore index for NTP cross-entropy (e.g., -100).

    Returns:
      total_loss: scalar tensor
      ctc_loss:   scalar tensor (detached from graph is your choice; here we keep it)
      ntp_loss:   scalar tensor
    """
    # ---- CTC loss (same as forward_ctc) ----
    ctc_output = encoder_out.log_softmax(2)  # (N, T, C)

    ctc_loss = torch.nn.functional.ctc_loss(
        log_probs=ctc_output.permute(1, 0, 2).cpu(),  # (T, N, C)
        targets=targets.cpu(),
        input_lengths=encoder_out_lens.cpu(),
        target_lengths=target_lengths.cpu(),
        reduction="mean",
    ).to(encoder_out.device)

    # ---- NTP loss ----
    ntp_loss = torch.tensor(0.0, device=encoder_out.device)

    if ntp_logits is not None and ntp_targets is not None:
        # ntp_logits: [N, T_ntp, C], ntp_targets: [N, T_ntp]
        N, T_ntp, C = ntp_logits.shape
        assert ntp_targets.shape == (N, T_ntp), \
            f"ntp_targets shape {ntp_targets.shape} must match (N, T_ntp)=({N}, {T_ntp})"

        ntp_loss = F.cross_entropy(
            ntp_logits.reshape(-1, C),
            ntp_targets.reshape(-1),
            ignore_index=ntp_ignore_index,
        )

    total_loss = ctc_loss + lambda_ntp * ntp_loss

    return total_loss, ctc_loss, ntp_loss

import torch
import torch.nn.functional as F

# ----------------------------
# DPO utilities
# ----------------------------

def _sequence_logprobs_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    pad_token_id: int,
) -> torch.Tensor:
    """
    logits: [B, T, V]
    labels: [B, T]  (token ids, padded with pad_token_id)
    Returns:
        log_probs: [B]  (sum of log p(y|x) over non-pad tokens)
    """
    # [B, T, V] -> [B, T, V]
    log_probs = F.log_softmax(logits, dim=-1)

    # mask out padding positions
    mask = (labels != pad_token_id)  # [B, T]

    # gather token log-probs
    token_logp = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)  # [B, T]

    # zero out pads then sum over time
    token_logp = token_logp * mask
    seq_logp = token_logp.sum(dim=-1)  # [B]

    return seq_logp


def dpo_loss(
    policy_chosen_logits: torch.Tensor,   # [B, T, V]
    policy_rejected_logits: torch.Tensor, # [B, T, V]
    ref_chosen_logits: torch.Tensor,      # [B, T, V]
    ref_rejected_logits: torch.Tensor,    # [B, T, V]
    chosen_labels: torch.Tensor,          # [B, T]
    rejected_labels: torch.Tensor,        # [B, T]
    pad_token_id: int,
    beta: float = 0.1,
    reduction: str = "mean",
):
    """
    Direct Preference Optimization loss.

    For each preference pair (x, y+, y-):

        L = - log σ( β * [ (log πθ(y+|x) - log π_ref(y+|x))
                          - (log πθ(y-|x) - log π_ref(y-|x)) ] )

    Args:
        policy_chosen_logits:   logits from the *policy* model on chosen responses
        policy_rejected_logits: logits from the *policy* model on rejected responses
        ref_chosen_logits:      logits from the *reference* model on chosen responses
        ref_rejected_logits:    logits from the *reference* model on rejected responses
        chosen_labels:          token ids for chosen sequences (padded)
        rejected_labels:        token ids for rejected sequences (padded)
        pad_token_id:           id used for padding
        beta:                   DPO inverse temperature (paper uses ~0.1–0.3 typically)
        reduction:              "mean", "sum", or "none"

    Returns:
        loss: scalar if reduction != "none", else [B]
        extra: dict with diagnostics (policy_advantage, ref_advantage, dpo_logits, accuracy)
    """
    # sequence log-probs under policy
    logp_policy_chosen = _sequence_logprobs_from_logits(
        policy_chosen_logits, chosen_labels, pad_token_id
    )  # [B]
    logp_policy_rejected = _sequence_logprobs_from_logits(
        policy_rejected_logits, rejected_labels, pad_token_id
    )  # [B]

    # sequence log-probs under reference
    logp_ref_chosen = _sequence_logprobs_from_logits(
        ref_chosen_logits, chosen_labels, pad_token_id
    )  # [B]
    logp_ref_rejected = _sequence_logprobs_from_logits(
        ref_rejected_logits, rejected_labels, pad_token_id
    )  # [B]

    # advantages (log π(y+|x) - log π(y-|x)) for policy and reference
    policy_advantage = logp_policy_chosen - logp_policy_rejected  # [B]
    ref_advantage = logp_ref_chosen - logp_ref_rejected          # [B]

    # DPO logit
    dpo_logits = beta * (policy_advantage - ref_advantage)       # [B]

    # DPO loss: - log σ(dpo_logits)
    per_example_loss = -F.logsigmoid(dpo_logits)                 # [B]

    # "accuracy" in terms of preference:
    # higher dpo_logits means policy favors chosen over rejected
    accuracy = (dpo_logits > 0).float()

    if reduction == "mean":
        loss = per_example_loss.mean()
    elif reduction == "sum":
        loss = per_example_loss.sum()
    elif reduction == "none":
        loss = per_example_loss
    else:
        raise ValueError(f"Unknown reduction: {reduction}")

    extra = {
        "policy_advantage": policy_advantage.detach(),
        "ref_advantage": ref_advantage.detach(),
        "dpo_logits": dpo_logits.detach(),
        "accuracy": accuracy.mean().detach(),
    }
    return loss, extra

import torch
import torch.nn.functional as F

def _duplicate_important_targets(
    targets: torch.Tensor,
    target_lengths: torch.Tensor,
    importance_mask: torch.Tensor,
    dup_factor: int = 2,
):
    """
    Expand targets by duplicating "important" labels.

    Args
    ----
    targets:        1D tensor of concatenated targets (sum_L,)
    target_lengths: 1D tensor of per-utterance lengths (N,)
    importance_mask:1D bool or 0/1 tensor, same shape as `targets`.
                    True/1 = important label to upweight.
    dup_factor:     duplicate each important label this many times.

    Returns
    -------
    new_targets:        1D tensor of concatenated expanded targets
    new_target_lengths: 1D tensor of new lengths per utterance
    """
    assert targets.dim() == 1
    assert importance_mask.shape == targets.shape
    assert target_lengths.dim() == 1

    new_targets_list = []
    new_lengths = []

    offset = 0
    for L in target_lengths:
        L = int(L.item())
        seq = targets[offset:offset + L]
        imp = importance_mask[offset:offset + L]

        expanded = []
        for p, is_imp in zip(seq, imp):
            if bool(is_imp):
                # duplicate important label
                expanded.extend([p] * dup_factor)
            else:
                expanded.append(p)

        new_targets_list.append(torch.stack(expanded))
        new_lengths.append(len(expanded))
        offset += L

    new_targets = torch.cat(new_targets_list, dim=0)
    new_target_lengths = torch.tensor(new_lengths, dtype=target_lengths.dtype)

    return new_targets, new_target_lengths


def forward_ctc_weighted(
    encoder_out: torch.Tensor,
    encoder_out_lens: torch.Tensor,
    targets: torch.Tensor,
    target_lengths: torch.Tensor,
    importance_mask: torch.Tensor | None = None,
    dup_factor: int = 2,
) -> torch.Tensor:
    """
    CTC loss with optional up-weighting of certain target labels by duplication.

    Args:
      encoder_out: (N, T, C)
      encoder_out_lens: (N,)
      targets: 1D concatenated targets, length sum(target_lengths)
      target_lengths: (N,)
      importance_mask: 1D tensor same shape as `targets`.
                       If None, behaves like standard CTC.
      dup_factor: how many times to duplicate each important label.

    Returns:
      Scalar CTC loss.
    """
    # Optionally expand targets & lengths
    if importance_mask is not None:
        targets, target_lengths = _duplicate_important_targets(
            targets=targets,
            target_lengths=target_lengths,
            importance_mask=importance_mask,
            dup_factor=dup_factor,
        )

    # Standard CTC from here on
    ctc_output = encoder_out.log_softmax(2)  # (N, T, C)

    ctc_loss = F.ctc_loss(
        log_probs=ctc_output.permute(1, 0, 2).cpu(),  # (T, N, C)
        targets=targets.cpu(),
        input_lengths=encoder_out_lens.cpu(),
        target_lengths=target_lengths.cpu(),
        reduction="mean",
    )
    return ctc_loss
