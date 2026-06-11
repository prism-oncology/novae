import logging
from typing import Callable, cast

import pandas as pd
import scanpy as sc
from anndata import AnnData

from ..._constants import Keys

log = logging.getLogger(__name__)


def _read_anndata_from_hub(
    name: str, row: pd.Series, annotations: bool = False, embeddings: str | None = None
) -> AnnData:
    adata = _read_h5ad_from_hub(f"{row['species']}/{row['tissue']}/{name}.h5ad")

    if "slide_id" in adata.obs:  # old datasets used "slide_id" instead of "novae_sid"
        adata.obs.rename(columns={"slide_id": Keys.SLIDE_ID}, inplace=True)
    elif Keys.SLIDE_ID not in adata.obs:
        adata.obs[Keys.SLIDE_ID] = pd.Series(name, index=adata.obs_names, dtype="category")

    if annotations:
        try:
            df_annot = pd.read_parquet(f"hf://datasets/prism-oncology/novae/annotations/{name}.parquet")
            adata.obs[df_annot.columns] = df_annot
        except FileNotFoundError:
            log.warning(f"Annotations unavailable for {name}. They will not be added to the adata.obs.")
        except Exception as e:
            log.warning(f"Failed to read annotations for {name}: {e}.")

    if embeddings is not None:
        try:
            adata_embeddings = _read_h5ad_from_hub(f"embeddings/{embeddings}/{name}.h5ad")
            obsm_key = adata_embeddings.uns[Keys.OBSM_KEY]
            adata.obsm[obsm_key] = adata_embeddings.obsm[obsm_key]
        except FileNotFoundError:
            log.warning(f"Embeddings '{embeddings}' unavailable for {name}. They will not be added to the adata.obsm.")
        except Exception as e:
            log.warning(f"Failed to read embeddings '{embeddings}' for {name}: {e}.")

    return adata


def _read_h5ad_from_hub(path: str) -> AnnData:
    from huggingface_hub import hf_hub_download

    local_file = hf_hub_download(repo_id="prism-oncology/novae", filename=path, repo_type="dataset")

    return sc.read_h5ad(local_file)


def load_dataset(
    pattern: str | None = None,
    tissue: list[str] | str | None = None,
    species: list[str] | str | None = None,
    technology: list[str] | str | None = None,
    custom_filter: Callable[[pd.DataFrame], pd.Series] | None = None,
    top_k: int | None = None,
    annotations: bool = False,
    embeddings: str | None = None,
    dry_run: bool = False,
) -> list[AnnData] | pd.DataFrame:
    """Automatically load slides from the Novae dataset repository.

    !!! info "Selecting slides"
        The function arguments allow to filter the slides based on the tissue, species, and name pattern.
        Internally, the function reads [this dataset metadata file](https://huggingface.co/datasets/prism-oncology/novae/blob/main/metadata.csv) to select the slides that match the provided filters.

    Args:
        pattern: Optional pattern to match the slides names, or directly a slide name.
        tissue: Optional tissue (or tissue list) to filter the slides. E.g., `"brain"` or `"colon"`.
        species: Optional species (or species list) to filter the slides. E.g., `"human"` or `"mouse"`.
        technology: Optional technology (or technology list) to filter the slides. E.g., `"xenium"` or `"visium_hd"`.
        custom_filter: Custom filter function that takes the metadata DataFrame (see above link) and returns a boolean Series to decide which rows should be kept.
        top_k: Optional number of slides to keep. If `None`, keeps all slides.
        annotations: If `True`, this will add cell-type annotations and/or pre-computed novae spatial domains in `adata.obs`. Not all slides have these annotations available.
        embeddings: Optional embedding type to load, e.g., `"corpus360M[multi-species]-model170M"` for scConcept embeddings.
        dry_run: If `True`, the function will only return the metadata of slides that match the filters.

    Returns:
        A list of `AnnData` objects, each object corresponds to one slide, or the metadata DataFrame if `dry_run=True`.
    """
    metadata = pd.read_csv("hf://datasets/prism-oncology/novae/metadata.csv", index_col=0)

    FILTER_COLUMN = [("species", species), ("tissue", tissue), ("technology", technology)]
    VALID_VALUES = {column: metadata[column].unique() for column, _ in FILTER_COLUMN}

    for column, value in FILTER_COLUMN:
        if value is not None:
            values = [value] if isinstance(value, str) else value
            valid_values = VALID_VALUES[column]

            assert all(value in valid_values for value in values), (
                f"Found invalid {column} value in {values}. Valid values are {valid_values}."
            )

            metadata = metadata[metadata[column].isin(values)]

    if custom_filter is not None:
        metadata = metadata[custom_filter(metadata)]

    assert not metadata.empty, "No dataset found for the provided filters."

    if pattern is not None:
        where = metadata.index.str.match(pattern)
        assert len(where), f"No dataset found for the provided pattern ({', '.join(list(metadata.index))})."
        metadata = metadata[where]

    assert not metadata.empty, "No dataset found for the provided filters."

    if top_k is not None:
        metadata = metadata.head(top_k)

    if dry_run:
        return cast(pd.DataFrame, metadata)

    log.info(f"Found {len(metadata)} h5ad file(s) matching the filters.")
    return [
        _read_anndata_from_hub(name, row, annotations=annotations, embeddings=embeddings)
        for name, row in metadata.iterrows()
    ]
