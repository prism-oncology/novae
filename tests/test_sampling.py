import numpy as np
import pandas as pd
from anndata import AnnData

from novae import utils
from novae._constants import Keys


def _fake_input(p: float) -> AnnData:
    obs = pd.DataFrame({Keys.IS_VALID_OBS: np.random.rand(1000) <= p})
    obs.index = obs.index.astype(str)
    obsm = {Keys.REPR: np.random.rand(1000, 10)}
    return AnnData(obs=obs, obsm=obsm)


def test_sample_latent() -> None:
    adatas = [_fake_input(0.5) for _ in range(3)]
    n_valid_cells = sum([np.sum(adata.obs[Keys.IS_VALID_OBS]) for adata in adatas])

    latent = utils.sample_latent(adatas, sampling_size=1_000_000)
    assert latent.shape[0] == n_valid_cells

    latent = utils.sample_latent(adatas, sampling_size=n_valid_cells)
    assert latent.shape[0] == n_valid_cells

    latent = utils.sample_latent(adatas, sampling_size=n_valid_cells - 1)
    assert latent.shape[0] == n_valid_cells - 3

    latent = utils.sample_latent(adatas, sampling_size=100)
    assert latent.shape[0] <= 100
