import math
import numbers
import torch
from torch import nn
from torch.nn import functional as F


import numpy as np

def mask_electrodes(X, max_mask_size):
    
    X = X.clone()
    
    batch_size, _, _  = X.shape
    
    area_6v_superior = np.array([
    [62,  51,  43,  35,  94,  87,  79,  78],
    [60,  53,  41,  33,  95,  86,  77,  76],
    [63,  54,  47,  44,  93,  84,  75,  74],
    [58,  55,  48,  40,  92,  85,  73,  72],
    [59,  45,  46,  38,  91,  82,  71,  70],
    [61,  49,  42,  36,  90,  83,  69,  68],
    [56,  52,  39,  34,  89,  81,  67,  66],
    [57,  50,  37,  32,  88,  80,  65,  64]
    ])

    area_6v_inferior = np.array([
        [125, 126, 112, 103,  31,  28,  11,  8],
        [123, 124, 110, 102,  29,  26,   9,  5],
        [121, 122, 109, 101,  27,  19,  18,  4],
        [119, 120, 108, 100,  25,  15,  12,  6],
        [117, 118, 107,  99,  23,  13,  10,  3],
        [115, 116, 106,  97,  21,  20,   7,  2],
        [113, 114, 105,  98,  17,  24,  14,  0],
        [127, 111, 104,  96,  30,  22,  16,  1]
    ])
        
    for b in range(batch_size):
        
        M = np.random.randint(0, max_mask_size+1)
        
        if M > 0:
            
            masked_indices = return_mask_electrodes_optimized(M)
            rows, cols = np.array(masked_indices).T  # Shape (2, M)
            superior_masked_indices = area_6v_superior[rows, cols]
            inferior_masked_indices = area_6v_inferior[rows, cols]
            masked_channels = np.concatenate((superior_masked_indices, inferior_masked_indices))
            masked_channels_all = np.concatenate((masked_channels, masked_channels+128))
            X[b, :, masked_channels_all] = 0
            
    return X

def return_mask_electrodes_optimized(M, grid_size=8):
    """
    Optimized electrode masking with vectorized operations.
    
    Args:
        M (int): Number of electrodes to mask
        grid_size (int): Size of square grid (default 8x8)
        
    Returns:
        ndarray: Masked electrode indices sorted by distance
    """
    # Precompute grid coordinates using broadcasting
    rows, cols = np.divmod(np.arange(grid_size**2), grid_size)
    
    # Random center selection
    center_idx = np.random.randint(grid_size**2)
    
    # Vectorized distance calculation
    distances = np.hypot(rows - rows[center_idx], 
                        cols - cols[center_idx])
    
    # Create mask excluding center and sort
    mask = np.ones(grid_size**2, bool)
    valid_indices = np.where(mask)[0]
    
    # Sort with tie-breaking using 64-bit precision
    sorted_indices = valid_indices[
        np.lexsort((np.random.random(len(valid_indices)),  # Tiebreaker
                   distances[valid_indices]))
    ]
    
    return [(idx // grid_size, idx % grid_size) for idx in sorted_indices[:M]]

class WhiteNoise(nn.Module):
    def __init__(self, std=0.1):
        super().__init__()
        self.std = std

    def forward(self, x):
        noise = torch.randn_like(x) * self.std
        return x + noise

class MeanDriftNoise(nn.Module):
    def __init__(self, std=0.1):
        super().__init__()
        self.std = std

    def forward(self, x):
        _, C = x.shape
        noise = torch.randn(1, C) * self.std
        return x + noise

class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """

    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [torch.arange(size, dtype=torch.float32) for size in kernel_size]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= (
                1
                / (std * math.sqrt(2 * math.pi))
                * torch.exp(-(((mgrid - mean) / std) ** 2) / 2)
            )

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer("weight", kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                "Only 1, 2 and 3 dimensions are supported. Received {}.".format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups, padding="same")


import random
from typing import Tuple

def spec_time_warp(
    X: torch.Tensor,
    X_len: torch.Tensor,
    max_warp: int = 5,
    time_dim: int = 1,   # time is dim=1 for [B, T, C]
) -> torch.Tensor:
    """
    Simple time-warp: roll each sequence along time dimension
    by a random shift in [-max_warp, max_warp], bounded by length.
    X: [B, T, C]
    """
    if max_warp <= 0:
        return X

    B = X.size(0)
    T = X.size(time_dim)

    if T <= 1:
        return X

    X_warped = X.clone()

    for b in range(B):
        valid_T = int(X_len[b].item())
        if valid_T <= 1:
            continue

        local_max_warp = min(max_warp, max(1, valid_T // 4))
        if local_max_warp <= 0:
            continue

        shift = random.randint(-local_max_warp, local_max_warp)
        if shift == 0:
            continue

        X_warped[b] = torch.roll(X_warped[b], shifts=shift, dims=time_dim)

    return X_warped


def spec_time_mask(
    X: torch.Tensor,
    X_len: torch.Tensor,
    num_masks: int = 2,
    max_width: int = 40,
    time_dim: int = 1,   # time is dim=1
) -> torch.Tensor:
    """
    Time masking along time_dim (T), respecting per-sample lengths.
    X: [B, T, C]
    """
    if num_masks <= 0 or max_width <= 0:
        return X

    B = X.size(0)
    X_aug = X

    for b in range(B):
        valid_T = int(X_len[b].item())
        if valid_T <= 0:
            continue

        for _ in range(num_masks):
            width = random.randint(0, min(max_width, valid_T))
            if width == 0:
                continue
            t0 = random.randint(0, max(0, valid_T - width))
            t1 = t0 + width

            # Mask time slice [t0:t1] across channels
            X_aug[b, t0:t1, :] = 0.0

    return X_aug


def apply_specaugment_two_views(
    X: torch.Tensor,
    X_len: torch.Tensor,
    args,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    1) Time warping ONCE (shared base)
    2) Independent electrode (channel) masking + time masking per view

    X: [B, T, C], X_len: [B]
    """
    # Hyperparameters from args or defaults
    max_time_warp = args.get("spec_time_warp_W", 5)
    num_time_masks = args.get("spec_num_time_masks", 2)
    time_mask_width = args.get("spec_time_mask_param", 40)
    max_mask_electrodes = args.get("spec_max_mask_electrodes", 8)

    # 1) Shared time warp (pre-branch)
    X_base = spec_time_warp(X, X_len, max_warp=max_time_warp, time_dim=1)

    # 2) Two independent masked copies
    X1 = X_base.clone()
    X2 = X_base.clone()

    # Electrode (channel) masking per view
    X1 = mask_electrodes(X1, max_mask_electrodes)
    X2 = mask_electrodes(X2, max_mask_electrodes)

    # Time masking per view
    X1 = spec_time_mask(X1, X_len, num_masks=num_time_masks, max_width=time_mask_width, time_dim=1)
    X2 = spec_time_mask(X2, X_len, num_masks=num_time_masks, max_width=time_mask_width, time_dim=1)

    # # Optional: re-apply your old noise/offset per view
    # if args.get("whiteNoiseSD", 0) > 0:
    #     X1 = X1 + torch.randn_like(X1) * args["whiteNoiseSD"]
    #     X2 = X2 + torch.randn_like(X2) * args["whiteNoiseSD"]

    # if args.get("constantOffsetSD", 0) > 0:
    #     offset1 = (
    #         torch.randn([X1.shape[0], 1, X1.shape[2]], device=X1.device)
    #         * args["constantOffsetSD"]
    #     )
    #     offset2 = (
    #         torch.randn([X2.shape[0], 1, X2.shape[2]], device=X2.device)
    #         * args["constantOffsetSD"]
    #     )
    #     X1 = X1 + offset1
    #     X2 = X2 + offset2

    return X1, X2
