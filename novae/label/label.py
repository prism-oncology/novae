import logging

import pandas as pd
from anndata import AnnData

from .. import plot, utils
from .._constants import Keys
from .clients import _get_api_request_func, _validate_api_key
from .describe import (
    _format_domain_cell_percentages,
    _format_domain_cell_type_percentages,
    _format_pathway_scores,
    _markers_as_dict,
)

log = logging.getLogger(__name__)


def _create_prompt(tissue: str = "unknown", species: str | None = None, spatial_context: str | None = None) -> str:
    """
    Prompt for domain labeling.
    """

    species_text = species if species else ""
    spatial_context_text = spatial_context if spatial_context else ""

    return (
        f"You are an expert in spatial transcriptomics analysis specializing in {species_text} tissue domain labeling. "
        f"Identify the most likely spatial domain name or tissue region (niche) for each domain of a {tissue} tissue based on marker genes and potentially enriched pathway scores. "
        "Consider spatial context, functional zones, and tissue organization when assigning domain names. "
        f"{spatial_context_text} "
        "Be concise but specific. Some domain may represent mixed or transitional regions. "
        "CRITICAL OUTPUT RULES: "
        "- The 'domain_name' must contain ONLY a short domain label. "
        "- Do NOT label domain using cell-type names, but using niche names. For instance, do not label a 'B and T cells' domain, but use 'Tertiary Lymphoid Structure' instead. "
        "- Use 2-5 words per label. "
        "- Prefer established histological/spatial terms. "
        "- Do NOT include explanations, examples, or additional details. "
        "- Do NOT use phrases like 'including', 'such as', or 'with'. "
        "- Do NOT skip any domain. "
        "- Do NOT add explanations. "
        "Return only valid JSON matching the provided schema."
    )


def _output_schema(
    domain_ids: list,
    domain_key: str,
    label_key: str,
    confidence_score_key: str,
    additionalProperties: bool = False,
) -> dict:
    schema = {
        "type": "object",
        "properties": {
            label_key: {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        domain_key: {"type": "string", "enum": domain_ids},
                        label_key: {
                            "type": "string",
                            "description": "Most likely domain name. May be a mixed label if needed.",
                        },
                        confidence_score_key: {
                            "type": "number",
                            "description": "A confidence score between 0 and 1 for the labeling.",
                        },
                    },
                    "required": [domain_key, label_key, confidence_score_key],
                    "additionalProperties": additionalProperties,
                },
            }
        },
        "required": [label_key],
        "additionalProperties": additionalProperties,
    }
    return schema


def _get_cell_type_pct(adata: AnnData, obs_key: str, cell_type_key: str) -> pd.DataFrame:
    if cell_type_key not in adata.obs:
        raise KeyError(f"`cell_type_key` key '{cell_type_key}' was not found in `adata.obs`.")

    df = adata.obs[[obs_key, cell_type_key]].copy()

    pct = pd.crosstab(df[obs_key], df[cell_type_key], normalize="index")
    return pct


def label_domains(
    adata: AnnData | None = None,
    pathways: dict[str, list[str]] | str | None = None,
    obs_key: str | None = None,
    cell_type_key: str | None = None,
    provider: str = "openai",
    model: str = "gpt-4.1",
    api_key: str | None = None,
    tissue: str = "unknown",
    species: str | None = None,
    n_genes: int = 15,
    spatial_context: str | None = None,
    return_prompt: bool = False,
    max_tokens: int = 1024,
    seed: int | None = None,
) -> pd.DataFrame | dict[str, object]:
    """Label spatial domains with an LLM using domain marker genes.

    Args:
        adata: An `AnnData` object, or a list of `AnnData` objects. Optional if the model was initialized with `adata`.
        pathways: Either a dictionary of pathways (keys are pathway names, values are lists of gene names), or a path to a [GSEA](https://www.gsea-msigdb.org/gsea/msigdb/index.jsp) JSON file.
        obs_key: Key in `adata.obs` containing domain IDs to label. By default, it uses the last available Novae domain key.
        cell_type_key: Optional key in `adata.obs` containing cell-type annotation labels. When provided, cell-type composition per domain is added to the LLM input.
        provider: LLM provider to use. Supported providers: 'openai', 'anthropic'.
        model: OpenAI model name used for labeling.
        api_key: OpenAI API key. If `None`, uses `OPENAI_API_KEY` from the environment.
        tissue: Tissue name (for example, `"liver"`).
        species: Species name (for example, `"human"` or `"mouse"`).
        n_genes: Number of marker genes per domain passed to the LLM prompt.
        spatial_context: Optional biological/spatial context to include in the prompt.
        return_prompt: If `True`, returns only the generated request payload (`messages` and `output_schema`) so you can copy/paste it into an LLM manually; no LLM request is made.
        seed: Optional random seed passed to the labeling utility.
        max_tokens: Maximum number of tokens the model is allowed to generate for the labeling response.

    Returns:
        A DataFrame with domain labels. If `return_prompt=True`, returns a dictionary containing `messages` and `output_schema`.
    """

    obs_key = utils.check_available_domains_key([adata], obs_key)

    domain_ids = pd.Index(pd.unique(adata.obs[obs_key].dropna()))

    gene_marker_dict = _markers_as_dict(adata, obs_key, domain_ids, n_genes)

    domain_ids = list(gene_marker_dict.keys())

    input_markers = "Gene markers:\n" + "\n".join(
        f"Domain {domain_id}: {', '.join(gene_marker_dict[domain_id])}" for domain_id in domain_ids
    )
    input_percentages = _format_domain_cell_percentages(adata, obs_key, domain_ids)

    pathway_scores = (
        None
        if pathways is None
        else plot.pathway_scores(adata, obs_key=obs_key, pathways=pathways, show=False, return_df=True)
    )

    cell_type_pct = (
        None if cell_type_key is None else _get_cell_type_pct(adata, obs_key=obs_key, cell_type_key=cell_type_key)
    )
    input_cell_types = _format_domain_cell_type_percentages(cell_type_pct, domain_ids)

    input_pathway = _format_pathway_scores(pathway_scores, domain_ids)

    prompt_sections = [input_markers, input_percentages]
    if input_cell_types:
        prompt_sections.append(input_cell_types)
    if input_pathway:
        prompt_sections.append(input_pathway)

    prompt = _create_prompt(species=species, tissue=tissue, spatial_context=spatial_context)

    messages = [
        {
            "role": "developer",
            "content": prompt,
        },
        {
            "role": "user",
            "content": "Label the following domains.\n\n" + "\n\n".join(prompt_sections),
        },
    ]

    output_schema = _output_schema(
        domain_ids=domain_ids,
        domain_key=obs_key,
        label_key=Keys.LABEL_SUFFIX,
        confidence_score_key=Keys.CONFIDENCE_SCORE,
    )

    if return_prompt:
        return {"messages": messages, "output_schema": output_schema}

    is_openai = provider.lower().startswith("openai")

    api_key = _validate_api_key(
        api_key,
        env_var=Keys.OPENAI_API_KEY if is_openai else Keys.ANTHROPIC_API_KEY,
        provider=provider,
    )

    api_request_func = _get_api_request_func(provider=provider, model=model)

    result = api_request_func(
        model=model,
        api_key=api_key,
        messages=messages,
        max_tokens=max_tokens,
        output_schema=output_schema,
        seed=seed,
    )

    return pd.DataFrame(result[Keys.LABEL_SUFFIX])
