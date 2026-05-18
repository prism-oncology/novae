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
    gene_marker_dict = _markers_as_dict(adata, obs_key, domain_ids, n_genes)

    input_markers = "Gene markers:\n" + "\n".join(
        f"Domain {domain_id}: {', '.join(gene_marker_dict[domain_id])}" for domain_id in domain_ids
    )
    input_percentages = _format_domain_cell_percentages(adata, obs_key, domain_ids)

    prompt_sections = [input_markers, input_percentages]
    if cell_type_key is not None:
        cell_type_pct = _get_cell_type_pct(adata, obs_key=obs_key, cell_type_key=cell_type_key)
        input_cell_types = _format_domain_cell_type_percentages(cell_type_pct, domain_ids)
        prompt_sections.append(input_cell_types)
    if pathways is not None:
        pathway_scores = plot.pathway_scores(adata, obs_key=obs_key, pathways=pathways, show=False, return_df=True)
        input_pathway = _format_pathway_scores(pathway_scores, domain_ids)
        prompt_sections.append(input_pathway)

    return "\n\n".join(prompt_sections)


def _format_pathway_scores(
    pathway_scores: pd.DataFrame | None,
    domain_ids: list[int] | list[str],
) -> str:
    if pathway_scores is None:
        return ""

    lines = []
    for domain_id in domain_ids:
        values = ", ".join(f"{name}={value:.4f}" for name, value in pathway_scores.loc[domain_id].items())
        lines.append(f"Domain {domain_id}: {values}")

    return "Pathway scores:\n" + "\n".join(lines)


def _format_domain_cell_percentages(adata: AnnData, obs_key: str, domain_ids: list[int] | list[str]) -> str:
    freq = adata.obs[obs_key].value_counts(normalize=True)
    lines = [f"Domain {domain_id}: {freq.get(domain_id, 0):.2%}" for domain_id in domain_ids]
    return "Cell percentages by domain:\n" + "\n".join(lines)


def _format_domain_cell_type_percentages(
    cell_type_pct: pd.DataFrame | None,
    domain_ids: list[int] | list[str],
    top_k: int = 3,
) -> str:
    if cell_type_pct is None:
        return ""

    lines = []
    for domain_id in domain_ids:
        if domain_id not in cell_type_pct.index:
            continue
        row = cell_type_pct.loc[domain_id].sort_values(ascending=False)
        row = row[row > 0].head(top_k)
        if row.empty:
            continue
        values = ", ".join(f"{cell_type}={pct:.2%}" for cell_type, pct in row.items())
        lines.append(f"Domain {domain_id}: {values}")

    return "" if not lines else "Cell-type composition by domain:\n" + "\n".join(lines)


def _markers_as_dict(adata: AnnData, obs_key: str, domain_ids: list[str], n_genes: int = 15):
    rank_genes_groups = adata.uns.get("rank_genes_groups")
    groupby = None if rank_genes_groups is None else rank_genes_groups.get("params", {}).get("groupby")
    if rank_genes_groups is None or groupby != obs_key:
        sc.tl.rank_genes_groups(adata, groupby=obs_key)

    names = adata.uns["rank_genes_groups"]["names"][:n_genes]
    return {domain: list(names[str(domain)]) for domain in domain_ids}


def _get_cell_type_pct(adata: AnnData, obs_key: str, cell_type_key: str) -> pd.DataFrame:
    if cell_type_key not in adata.obs:
        raise KeyError(f"`cell_type_key` key '{cell_type_key}' was not found in `adata.obs`.")

    df = adata.obs[[obs_key, cell_type_key]].copy()

    pct = pd.crosstab(df[obs_key], df[cell_type_key], normalize="index")
    return pct
