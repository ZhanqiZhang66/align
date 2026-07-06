# neural_decoder/dann.py
import torch
import torch.nn as nn

class GradientReversalFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha: float):
        ctx.alpha = float(alpha)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


def grad_reverse(x, alpha: float):
    return GradientReversalFn.apply(x, alpha)


class DomainDiscriminator(nn.Module):
    """
    Domain discriminator that can be either linear or MLP.
    
    Modes:
      - Standard: one head with n_domains logits (DANN k+1 style when n_domains>1)
      - MDAN: truely_mdan=True builds K binary heads (source-i vs target), stacked to [B, K]
    """
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        n_domains: int = 2,
        dropout: float = 0.1,
        linear_discriminator: bool = True,
        truely_mdan: bool = False,
    ):
        super().__init__()
        self.linear_discriminator = linear_discriminator
        self.truely_mdan = bool(truely_mdan)
        self.n_domains = n_domains

        if not self.truely_mdan:
            # Single head: logits over n_domains
            if linear_discriminator:
                self.net = nn.Linear(in_dim, n_domains)
            else:
                self.net = nn.Sequential(
                    nn.Linear(in_dim, hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, n_domains),
                )
        else:
            # MDAN: K binary heads (each outputs 1 logit)
            if n_domains < 1:
                raise ValueError("n_domains must be >=1 for truely_mdan")
            self.heads = nn.ModuleList()
            for _ in range(n_domains):
                if linear_discriminator:
                    self.heads.append(nn.Linear(in_dim, 1))
                else:
                    self.heads.append(
                        nn.Sequential(
                            nn.Linear(in_dim, hidden_dim),
                            nn.ReLU(inplace=True),
                            nn.Dropout(dropout),
                            nn.Linear(hidden_dim, hidden_dim),
                            nn.ReLU(inplace=True),
                            nn.Dropout(dropout),
                            nn.Linear(hidden_dim, 1),
                        )
                    )
    
    def forward(self, x):
        """
        x: [B, D] or [B, T, D] (or [B, ..., D])
        """
        if x.dim() < 2:
            raise ValueError(f"Expected x with dim>=2, got shape {tuple(x.shape)}")

        # Flatten all leading dims except the last feature dim
        orig_shape = x.shape[:-1]   # e.g., (B,D) or (B,T,D)
        d = x.shape[-1]
        x_flat = x.reshape(-1, d)   # [prod(orig_shape), D]

        if not self.truely_mdan:          
            out = self.net(x_flat)  # [B_flat, n_domains]
            return out

        # MDAN: stack binary heads -> [B_flat, K]
        logits = [head(x_flat) for head in self.heads]  # each [B_flat,1]
        return torch.cat(logits, dim=1)


def randomly_mask_channelsteps(rep: torch.Tensor, lengths: torch.Tensor, mask_prob: float) -> torch.Tensor:
    """
    Randomly mask out channels (feature dimensions D) in the representation before pooling.
    Only masks valid timesteps (within sequence length).
    
    Args:
        rep: [B, T, D] representation tensor
        lengths: [B] valid sequence lengths
        mask_prob: Probability of masking each channel (dimension D) for each valid timestep (0.0 to 1.0)
    
    Returns:
        rep: [B, T, D] representation with randomly masked channels set to zero
    """
    if mask_prob <= 0.0:
        return rep

    B, T, D = rep.shape
    device = rep.device

    # Create mask for valid timesteps
    t = torch.arange(T, device=device)[None, :].expand(B, T)
    valid_mask = (t < lengths[:, None]).float()  # [B, T]

    # Randomly mask channels (independently for each D at each timestep)
    random_mask = torch.rand(B, T, D, device=device)  # [B, T, D]
    # mask_tensor: 1 to keep, 0 to mask, using mask_prob
    channel_mask = (random_mask >= mask_prob).float()  # [B, T, D]

    # Broadcast valid_mask over D
    valid_mask_expanded = valid_mask[:, :, None]  # [B, T, 1]
    mask = channel_mask * valid_mask_expanded + (1.0 - valid_mask_expanded)
    # For valid timesteps: mask channels using channel_mask; for invalid timesteps: keep as 0 (already outside sequence length)

    # Apply mask: set masked channels to zero
    masked_rep = rep * mask  # [B, T, D]

    return masked_rep


def masked_mean_pool(rep: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """
    rep: [B, T, D]
    lengths: [B] int
    returns: [B, D]
    """
    B, T, D = rep.shape
    device = rep.device
    t = torch.arange(T, device=device)[None, :].expand(B, T)
    mask = (t < lengths[:, None]).float()  # [B, T]
    denom = mask.sum(dim=1).clamp_min(1.0)  # [B]
    pooled = (rep * mask[:, :, None]).sum(dim=1) / denom[:, None]
    return pooled


def unpack_batch(batch):
    # batch is (X, y, X_len, y_len, dayIdx, ...)
    X, y, X_len, y_len, day = batch[:5]
    return X, X_len, y, y_len, day