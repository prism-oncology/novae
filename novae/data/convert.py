import numpy as np
import torch
from anndata import AnnData
from fast_array_utils import stats
from fast_array_utils.conv import to_dense
from torch import Tensor

from .._constants import Keys, Nums
from ..module import CellEmbedder


class ZScoreTorch:
    tensors: list[Tensor] | None
    genes_indices_list: list[Tensor]

    def __init__(self, adatas: list[AnnData], cell_embedder: CellEmbedder):
        """Converting AnnData objects to PyTorch tensors with z-score of log1p expression data.

        Args:
            adatas: A list of `AnnData` objects.
            cell_embedder: A [novae.module.CellEmbedder][] object.
        """
        super().__init__()
        self.adatas = [adata[:, adata.var[Keys.USE_GENE]] for adata in adatas]
        self.cell_embedder = cell_embedder

        self.genes_indices_list = [self._adata_to_genes_indices(adata) for adata in self.adatas]
        self.tensors = None

        self.means_dict, self.stds_dict = self._compute_means_stds()

        if not _use_lazy_loading(self.adatas):  # load tensors in memory for low numbers of elements (n_obs * n_vars)
            self.tensors = [self.to_tensor(adata, enforce_single_slide=False) for adata in self.adatas]

    def _adata_to_genes_indices(self, adata: AnnData) -> Tensor:
        return self.cell_embedder.genes_to_indices(adata.var_names)[None, :]  # ty: ignore[invalid-return-type]

    def _compute_means_stds(self) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        means, stds = {}, {}

        for adata in self.adatas:
            slide_ids = adata.obs[Keys.SLIDE_ID]
            for slide_id in slide_ids.cat.categories:
                adata_slide: AnnData = adata[adata.obs[Keys.SLIDE_ID] == slide_id]

                mean, var = stats.mean_var(adata_slide.X, axis=0)
                mean, var = to_dense(mean, to_cpu_memory=True), to_dense(var, to_cpu_memory=True)

                means[slide_id] = mean.astype(np.float32)
                stds[slide_id] = np.sqrt(var).astype(np.float32)

        return {slide_id: torch.tensor(_means) for slide_id, _means in means.items()}, {
            slide_id: torch.tensor(_std) for slide_id, _std in stds.items()
        }

    def to_tensor(self, adata: AnnData, enforce_single_slide: bool = True) -> Tensor:
        """Get the normalized gene expressions of the cells in the dataset.
        Only the genes of interest are kept (known genes and highly variable).

        Args:
            adata: An `AnnData` object.
            enforce_single_slide: Whether we know for sure that the AnnData object corresponds to only one slide.

        Returns:
            A `Tensor` containing the normalized gene expresions.
        """
        if enforce_single_slide or len(adata.obs[Keys.SLIDE_ID].unique()) == 1:
            slide_id = adata.obs[Keys.SLIDE_ID].iloc[0]
            mean, std = self.means_dict[slide_id], self.stds_dict[slide_id]
        else:
            mean = torch.stack([self.means_dict[slide_id] for slide_id in adata.obs[Keys.SLIDE_ID].values])
            std = torch.stack([self.stds_dict[slide_id] for slide_id in adata.obs[Keys.SLIDE_ID].values])

        X = torch.tensor(to_dense(adata.X, to_cpu_memory=True), dtype=torch.float32)
        X = (X - mean) / (std + Nums.EPS)

        return X

    def __getitem__(self, item: tuple[int, slice]) -> tuple[Tensor, Tensor]:
        """Get the expression values for a subset of cells (corresponding to a subgraph).

        Args:
            item: A `tuple` containing the index of the `AnnData` object and the indices of the cells in the neighborhoods.

        Returns:
            A `Tensor` of normalized gene expressions and a `Tensor` of gene indices.
        """
        adata_index, obs_indices = item

        if self.tensors is not None:
            return self.tensors[adata_index][obs_indices], self.genes_indices_list[adata_index]

        adata = self.adatas[adata_index]
        adata_view = adata[obs_indices]

        return self.to_tensor(adata_view), self.genes_indices_list[adata_index]


class EmbeddingTorch:
    def __init__(self, adatas: list[AnnData], embedding_name: str):
        """Converting pre-computed single-cell embeddings to PyTorch tensors.

        Args:
            adatas: A list of `AnnData` objects.
            embedding_name: The name of the embedding in `adata.obsm`.
        """
        super().__init__()
        self.adatas = adatas
        self.embedding_name = embedding_name

        # Dummy variable since gene indices are not used when using pre-computed embeddings
        self.dummy_genes_indices = torch.zeros((1, 1), dtype=torch.long)

    def __getitem__(self, item: tuple[int, slice]) -> tuple[Tensor, Tensor]:
        """Get the single-cell embedding values for a subset of cells (corresponding to a subgraph).

        Args:
            item: A `tuple` containing the index of the `AnnData` object and the indices of the cells in the neighborhoods.

        Returns:
            A `Tensor` of single-cell embedding values and a dummy `Tensor` of gene indices to match the converter signature.
        """
        adata_index, obs_indices = item

        adata = self.adatas[adata_index]
        embeddings = adata[obs_indices].obsm[self.embedding_name]
        embeddings = torch.tensor(to_dense(embeddings, to_cpu_memory=True), dtype=torch.float32)

        return embeddings, self.dummy_genes_indices  # keep the same return type as ZScoreTorch


def get_torch_converter(adatas: list[AnnData], cell_embedder: CellEmbedder | str) -> ZScoreTorch | EmbeddingTorch:
    assert cell_embedder is not None, "Received None as a `cell_embedder`."

    return (
        EmbeddingTorch(adatas, cell_embedder) if isinstance(cell_embedder, str) else ZScoreTorch(adatas, cell_embedder)
    )


def _use_lazy_loading(adatas: list[AnnData]) -> bool:
    return int(sum(np.prod(adata.shape) for adata in adatas)) >= Nums.LAZY_LOADING_SIZE_THRESHOLD
