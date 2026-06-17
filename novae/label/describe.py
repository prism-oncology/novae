import logging

import pandas as pd
import scanpy as sc
from anndata import AnnData

from .. import plot
from .._constants import Keys

log = logging.getLogger(__name__)


def domains_description(
    adata: AnnData,
    obs_key: str,
    domain_ids: list[str],
    cell_type_key: str | None,
    pathways: dict[str, list[str]] | str | None,
    n_genes: int,
) -> str:
    sections: list[str] = [
        _deg_description(adata, obs_key, domain_ids, n_genes),
        _domain_size_description(adata, obs_key, domain_ids),
        _interfaces_description(adata, obs_key),
    ]

    if cell_type_key is not None:
        sections.append(_cell_type_description(adata, obs_key, domain_ids, cell_type_key))

    if pathways is not None:
        sections.append(_pathway_description(adata, obs_key, domain_ids, pathways))

    return "\n\n".join(sections)


def _deg_description(adata: AnnData, obs_key: str, domain_ids: list[str], n_genes: int) -> str:
    if (
        "rank_genes_groups" not in adata.uns
        or adata.uns["rank_genes_groups"]["params"]["groupby"] != obs_key
        or adata.uns["rank_genes_groups"]["params"]["reference"] != "rest"
    ):
        sc.tl.rank_genes_groups(adata, groupby=obs_key)

    top_degs = {
        domain_id: sc.get.rank_genes_groups_df(adata, group=domain_id)["names"][:n_genes] for domain_id in domain_ids
    }

    extra_degs = []
    for domain1, domain2 in zip(domain_ids, domain_ids[1:]):
        if len(set(top_degs[domain1]) & set(top_degs[domain2])) >= n_genes / 4:  # similar domains, add extra info
            for group1, group2 in [(domain1, domain2), (domain2, domain1)]:
                sc.tl.rank_genes_groups(adata, groupby=obs_key, groups=[group1], reference=group2)
                extra_degs.append(
                    f"Domain {group1} vs Domain {group2}: "
                    + ", ".join(sc.get.rank_genes_groups_df(adata, group=group1)["names"][:n_genes])
                )

    return (
        "Gene markers (DEGs), ordered from higher to lower significance:\n"
        + "\n".join(f"Domain {domain_id}: {', '.join(degs)}" for domain_id, degs in top_degs.items())
        + ("\n\n" + "\n".join(extra_degs) if extra_degs else "")
    )


def _interfaces_description(adata: AnnData, obs_key: str, top_k: int = 4) -> str:
    if Keys.ADJ not in adata.obsp:
        log.warning("Adjacency matrix not found in `adata.obsp`. Skipping interface description.")
        return ""

    connectivities = adata.obsp[Keys.ADJ]

    i, j = connectivities.nonzero()
    di = adata.obs[obs_key].iloc[i].values
    dj = adata.obs[obs_key].iloc[j].values

    where = di != dj
    interfaces = pd.DataFrame({"di": di[where], "dj": dj[where]}).value_counts().unstack(fill_value=0)
    interfaces = interfaces / interfaces.sum()

    return "Cell-cell neighborhood connectivites:\n" + "\n".join(
        f"{domain} top connectivities: "
        + ", ".join(
            f"{i} {v}"
            for i, v in interfaces[domain]
            .sort_values(ascending=False)
            .iloc[:top_k]
            .apply(lambda x: f"({x:.2%})")
            .items()
        )
        for domain in interfaces
    )


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


def _pathway_description(
    adata: AnnData, obs_key: str, domain_ids: list[str], pathways: dict[str, list[str]] | str
) -> str:
    pathway_scores: pd.DataFrame = plot.pathway_scores(
        adata, obs_key=obs_key, pathways=pathways, show=False, return_df=True
    )

    lines = []
    for domain_id in domain_ids:
        values = ", ".join(f"{name}={value:.4f}" for name, value in pathway_scores.loc[domain_id].items())
        lines.append(f"Domain {domain_id}: {values}")

    return "Pathway scores:\n" + "\n".join(lines)
