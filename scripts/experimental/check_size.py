from pathlib import Path

import scanpy as sc


def main(directory: Path) -> None:
    paths = list(directory.glob("*.h5ad"))

    for path in paths:
        name = path.stem
        adata = sc.read_h5ad(path)

        print(name, adata.obsm["X_scConcept"].shape)


if __name__ == "__main__":
    for directory in Path("/gpfs/workdir/blampeyq/res_novae").glob("X_scConcept*"):
        print(directory)
        main(directory)
