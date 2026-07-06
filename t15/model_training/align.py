# DANN utilities for domain adaptation (GRL, domain discriminator, pooling).
# Mirrors repos/transformers_with_dietcorp_cp/src/neural_decoder/dann.py
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
        orig_shape = x.shape[:-1]
        d = x.shape[-1]
        x_flat = x.reshape(-1, d)

        if not self.truely_mdan:
            out = self.net(x_flat)
            return out
        logits = [head(x_flat) for head in self.heads]
        return torch.cat(logits, dim=1)


def randomly_mask_channelsteps(rep: torch.Tensor, lengths: torch.Tensor, mask_prob: float) -> torch.Tensor:
    """Randomly mask channels in representation before pooling. Only valid timesteps are masked."""
    if mask_prob <= 0.0:
        return rep
    B, T, D = rep.shape
    device = rep.device
    t = torch.arange(T, device=device)[None, :].expand(B, T)
    valid_mask = (t < lengths[:, None]).float()
    random_mask = (torch.rand(B, T, D, device=device) >= mask_prob).float()
    mask = random_mask * valid_mask[:, :, None] + (1.0 - valid_mask[:, :, None])
    return rep * mask


def masked_mean_pool(rep: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """
    rep: [B, T, D]
    lengths: [B] int
    returns: [B, D]
    """
    B, T, D = rep.shape
    device = rep.device
    t = torch.arange(T, device=device)[None, :].expand(B, T)
    mask = (t < lengths[:, None]).float()
    denom = mask.sum(dim=1).clamp_min(1.0)
    pooled = (rep * mask[:, :, None]).sum(dim=1) / denom[:, None]
    return pooled
