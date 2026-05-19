import logging
from typing import Any

import pandas as pd
from anndata import AnnData

from .. import utils
from .._constants import Keys
from .clients import api_request
from .describe import domains_description

log = logging.getLogger(__name__)


def label_domains(
    adata: AnnData,
    obs_key: str | None = None,
    tissue: str = "unknown",
    species: str | None = None,
    n_genes: int = 15,
    cell_type_key: str | None = None,
    pathways: dict[str, list[str]] | str | None = None,
    spatial_context: str | None = None,
    provider: str = "openai",
    model: str = "gpt-4.1",
    api_key: str | None = None,
    max_tokens: int = 1024,
    seed: int | None = None,
    return_prompt: bool = False,
) -> pd.DataFrame | dict[str, dict[str, Any]]:
    """Label the Novae spatial domains with an LLM based on descriptive information: DEGs, domain sizes, pathway expressions, and cell-type proportions.

    Args:
        adata: An `AnnData` object containing the spatial domains assigned by Novae.
        obs_key: Key in `adata.obs` containing domain IDs to label. By default, it uses the last available Novae domain key.
        tissue: Tissue name (for example, `"liver"`).
        species: Species name (for example, `"human"` or `"mouse"`).
        n_genes: Number of marker genes per domain passed to the LLM prompt.
        cell_type_key: Optional key in `adata.obs` containing cell-type annotations. When provided, cell-type composition per domain is added to the LLM input.
        pathways: Either a dictionary of pathways (keys are pathway names, values are lists of gene names), or a path to a [GSEA](https://www.gsea-msigdb.org/gsea/msigdb/index.jsp) JSON file. When provided, pathway enrichment scores per domain are added to the LLM input.
        spatial_context: Optional extra biological or spatial context to include in the prompt.
        provider: LLM provider to use. Either `'openai'` or `'anthropic'`.
        model: OpenAI model name used for labeling.
        api_key: OpenAI API key. If `None`, uses `OPENAI_API_KEY` from the environment.
        max_tokens: Maximum number of tokens the model is allowed to generate for the labeling response (only for anthropic).
        seed: Optional random seed passed to the labeling utility.
        return_prompt: If `True`, returns only the generated request payload (`messages` and `output_schema`) so you can copy/paste it into an LLM manually. No LLM request is made, and no API key is required.

    Returns:
        A `DataFrame` with domain labels. If `return_prompt=True`, returns a dictionary containing `messages` and `output_schema`.
    """

    obs_key = utils.check_available_domains_key([adata], obs_key)
    domain_ids = list(pd.unique(adata.obs[obs_key].dropna()))

    description = domains_description(
        adata=adata,
        obs_key=obs_key,
        domain_ids=domain_ids,
        cell_type_key=cell_type_key,
        pathways=pathways,
        n_genes=n_genes,
    )

    messages = [
        {
            "role": "developer",
            "content": _get_system_prompt(tissue=tissue, species=species, spatial_context=spatial_context),
        },
        {
            "role": "user",
            "content": f"Label the following domains.\n\n{description}",
        },
    ]

    output_schema = _get_output_schema(domain_ids=domain_ids, domain_key=obs_key)

    if return_prompt:
        return {"messages": messages, "output_schema": output_schema}

    result = api_request(
        api_key=api_key,
        provider=provider,
        model=model,
        messages=messages,
        output_schema=output_schema,
        max_tokens=max_tokens,
        seed=seed,
    )

    return pd.DataFrame(result[Keys.LABEL_SUFFIX])


def _get_system_prompt(tissue: str = "unknown", species: str | None = None, spatial_context: str | None = None) -> str:
    return (
        f"You are an expert in spatial transcriptomics analysis specializing in {species or ''} tissue domain labeling. "
        f"Identify the most likely spatial domain name or tissue region (niche) for each domain of a {tissue} tissue based on marker genes and potentially enriched pathway scores. "
        "Consider spatial context, functional zones, and tissue organization when assigning domain names. "
        f"{spatial_context or ''} "
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


def _get_output_schema(domain_ids: list, domain_key: str) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            Keys.LABEL_SUFFIX: {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        domain_key: {"type": "string", "enum": domain_ids},
                        Keys.LABEL_SUFFIX: {
                            "type": "string",
                            "description": "Most likely domain label. May be a mixed label if needed.",
                        },
                        Keys.CONFIDENCE_SCORE: {
                            "type": "number",
                            "description": "A confidence score between 0 and 1 for the labeling.",
                        },
                    },
                    "required": [domain_key, Keys.LABEL_SUFFIX, Keys.CONFIDENCE_SCORE],
                    "additionalProperties": False,
                },
            }
        },
        "required": [Keys.LABEL_SUFFIX],
        "additionalProperties": False,
    }
