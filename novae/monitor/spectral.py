import numpy as np
import torch

from .._constants import Nums


@torch.no_grad()
def spectral_stats(z: torch.Tensor) -> dict[str, float | np.ndarray]:
    z = z - z.mean(0, keepdim=True)
    cov = z.T @ z / (z.shape[0] - 1)

    eigvals: torch.Tensor = torch.linalg.eigvalsh(cov)
    eigvals = eigvals.clamp(min=0)

    participation_ratio = eigvals.sum() ** 2 / eigvals.square().sum()

    p = eigvals / eigvals.sum()
    erank = torch.exp(-(p * (p + Nums.EPS).log()).sum())

    return {
        "participation_ratio": participation_ratio.item(),
        "effective_rank": erank.item(),
        "eigvals": eigvals.numpy(force=True),
    }
