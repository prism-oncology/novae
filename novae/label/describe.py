import pandas as pd
import scanpy as sc
from anndata import AnnData

from .. import plot


def domains_description(
    adata: AnnData,
    obs_key: str,
    domain_ids: list[str],
    cell_type_key: str | None,
    pathways: list[str] | None,
    n_genes: int,
) -> str:
    sections: list[str] = [
        _deg_description(adata, obs_key, domain_ids, n_genes),
        _domain_size_description(adata, obs_key, domain_ids),
    ]

    if cell_type_key is not None:
        sections.append(_cell_type_description(adata, obs_key, domain_ids, cell_type_key))

    if pathways is not None:
        sections.append(_pathway_description(adata, obs_key, domain_ids, pathways))

    return "\n\n".join(sections)


def _deg_description(adata: AnnData, obs_key: str, domain_ids: list[str], n_genes: int) -> str:
    if "rank_genes_groups" not in adata.uns or adata.uns["rank_genes_groups"]["params"]["groupby"] != obs_key:
        sc.tl.rank_genes_groups(adata, groupby=obs_key)

    return "Gene markers (DEGs):\n" + "\n".join(
        f"Domain {domain_id}: {_top_degs(adata, domain_id, n_genes)}" for domain_id in domain_ids
    )


def _top_degs(adata: AnnData, domain_id: str, n_genes: int) -> str:
    return ", ".join(sc.get.rank_genes_groups_df(adata, group=domain_id)["names"][:n_genes])


def _domain_size_description(adata: AnnData, obs_key: str, domain_ids: list[int] | list[str]) -> str:
    perc = adata.obs[obs_key].value_counts(normalize=True)

    return "Domain sizes:\n" + "\n".join(
        f"Domain {domain_id}: {perc.get(domain_id, 0):.2%}" for domain_id in domain_ids
    )


def _cell_type_description(
    adata: AnnData, obs_key: str, domain_ids: list[str], cell_type_key: str, top_k: int = 3
) -> str:
    if cell_type_key not in adata.obs:
        raise KeyError(f"`cell_type_key` key '{cell_type_key}' was not found in `adata.obs`.")

    cell_type_percentages = pd.crosstab(adata.obs[obs_key], adata.obs[cell_type_key], normalize="index")

    lines = []
    for domain_id in domain_ids:
        if domain_id not in cell_type_percentages.index:
            continue

        row = cell_type_percentages.loc[domain_id].sort_values(ascending=False)
        row = row[row > 0].head(top_k)

        if row.empty:
            continue

        values = ", ".join(f"{cell_type}={pct:.2%}" for cell_type, pct in row.items())
        lines.append(f"Domain {domain_id}: {values}")

    return "Cell-type composition per domain:\n" + "\n".join(lines)


def _pathway_description(adata: AnnData, obs_key: str, domain_ids: list[str], pathways: list[str]) -> str:
    pathway_scores = plot.pathway_scores(adata, obs_key=obs_key, pathways=pathways, show=False, return_df=True)

    lines = []
    for domain_id in domain_ids:
        values = ", ".join(f"{name}={value:.4f}" for name, value in pathway_scores.loc[domain_id].items())
        lines.append(f"Domain {domain_id}: {values}")

    return "Pathway scores:\n" + "\n".join(lines)
