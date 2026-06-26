import numpy as np
import torch

from .._constants import Nums


@torch.no_grad()
def spectral_stats(z: np.ndarray) -> tuple[float, float, np.ndarray]:
    """Compute spectral statistics on a batch of embeddings.

    Args:
        z: A batch of embeddings of size `(B, O)`.

    Returns:
        A tuple containing:
        - participation_ratio: The participation ratio of the eigenvalues.
        - erank: The effective rank of the covariance matrix.
        - eigvals: The eigenvalues of the covariance matrix.
    """
    z = z / np.linalg.norm(z, axis=1, keepdims=True)
    cov = np.cov(z.T)

    eigvals = np.linalg.eigvalsh(cov)
    eigvals: np.ndarray = eigvals.clip(min=0)

    participation_ratio = eigvals.sum() ** 2 / (eigvals**2).sum()

    p: np.ndarray = eigvals / eigvals.sum()
    erank = np.exp(-(p * np.log(p + Nums.EPS)).sum())

    return participation_ratio, erank, eigvals[::-1]
