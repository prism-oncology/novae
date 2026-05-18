import pandas as pd
import scanpy as sc
from anndata import AnnData


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


def _markers_as_dict(adata: AnnData, obs_key: str, domain_ids: list, n_genes: int = 15):
    rank_genes_groups = adata.uns.get("rank_genes_groups")
    groupby = None if rank_genes_groups is None else rank_genes_groups.get("params", {}).get("groupby")
    if rank_genes_groups is None or groupby != obs_key:
        sc.tl.rank_genes_groups(adata, groupby=obs_key)

    names = adata.uns["rank_genes_groups"]["names"][:n_genes]
    return {domain: list(names[str(domain)]) for domain in domain_ids}
