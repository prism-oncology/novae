# Contributing to *novae*

Contributions are welcome as we aim to continue improving `novae`. For instance, you can contribute by:

- Opening an issue
- Discussing the current state of the code
- Making a Pull Request (PR)

If you want to open a PR, follow the following instructions.

## Making a Pull Request (PR)

To add some new code to **novae**, you should:

1. Fork the repository
2. Install `novae` in editable mode with the `dev` dependencies (see next section)
3. Create your personal branch from `main`
4. Implement your changes according to the 'Coding guidelines' below
5. Create a pull request on the `main` branch of the original repository. Add explanations about your developed features, and wait for discussion and validation of your pull request

### Installing `novae` in editable mode

When contributing, installing `novae` in editable mode is recommended. We also recommend installing the `dev` dependencies.

For that, you can use [`uv`](https://docs.astral.sh/uv/getting-started/installation/) as below:

```sh
git clone https://github.com/prism-oncology/novae.git
cd novae

uv sync --all-extras --dev   # all extras and dev dependencies
```

### Coding guidelines

#### Styling and formatting

We use [`pre-commit`](https://pre-commit.com/) to run code quality controls before the commits. This will run `ruff` and others minor checks.


You can set it up at the root of the repository like this:
```sh
uv run pre-commit install
```

Then, it will run the pre-commit automatically before each commit.

You can also run the pre-commit manually:
```sh
uv run poe pre-commit
```

Apart from this, we recommend to follow the standard styling conventions:
- Follow the [PEP8](https://peps.python.org/pep-0008/) style guide.
- Provide meaningful names to all your variables and functions.
- Provide type hints to your function inputs/outputs.
- Add docstrings in the Google style.
- Try as much as possible to follow the same coding style as the rest of the repository.

#### Testing

When create a pull request, tests are run automatically. But you can also run the tests yourself before making the PR. For that, run `pytest` at the root of the repository. You can also add new tests in the `./tests` directory.

To check the coverage of the tests:

```sh
uv run poe test

open htmlcov/index.html
```

#### Documentation

You can update the documentation in the `./docs` directory. Refer to the [mkdocs-material documentation](https://squidfunk.github.io/mkdocs-material/) for more help.

To serve the documentation locally:

```sh
uv run poe docs
```

## Adding data to the Hugging Face Hub

If you have rights to add data to the [Hugging Face dataset](https://huggingface.co/datasets/prism-oncology/novae), you can upload a new file via the `hf` CLI:

```sh
hf upload prism-oncology/novae --repo-type dataset <local_file> <upstream_file>
```

### Adding a new slide

To upload a new slide, add it to the right sub-directory, e.g.:

```sh
hf upload prism-oncology/novae --repo-type dataset \
    Xenium_Prime_Breast_Cancer_FFPE_outs.h5ad \
    human/breast/Xenium_Prime_Breast_Cancer_FFPE_outs.h5ad
```

Also make sure to update the [`metadata.csv` file](https://huggingface.co/datasets/prism-oncology/novae/blob/main/metadata.csv) to list your new added data file.

### Adding an embedding

If you add an embedding, create a new `h5ad` file with only the embeddings and an `"obsm_key"` in `.uns` to store the embedding name:

```python
adata_small = AnnData(obsm={"X_scConcept": adata.obsm["X_scConcept"]})
adata_small.uns["obsm_key"] = "X_scConcept"
```

This can be added under the embeddings directory of the dataset, for instance:

```sh
hf upload prism-oncology/novae --repo-type dataset \
    adata_small.h5ad \
    'embeddings/corpus360M[multi-species]-model170M/Xenium_Prime_Breast_Cancer_FFPE_outs.h5ad'
```

### Adding annotations

For the annotations, simply add a subset of the `obs` and save it as a parquet file.

```python
adata.obs[["cell_type", "novae_domains"]].to_parquet("local_annotations.parquet")
```

And add it to the `annotations` sub-directory.
