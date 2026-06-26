# isort: skip_file
# else segmentation fault during imports
import pyarrow  # noqa: F401

from pathlib import Path

import numpy as np
import scanpy as sc

other_dir_names = [f"X_scConcept{2 + i}" for i in range(3)]


def _print_stats(adata: sc.AnnData, name: str) -> None:
    X = adata.obsm["X_scConcept"]
    mean_l2_norm = np.linalg.norm(X, axis=1).mean()
    print(
        f"{name}\nshape{X.shape}, mean:{X.mean(0).mean()}, std:{X.std(0).mean()} max:{X.max()}, min:{X.min()}, mean_l2_norm:{mean_l2_norm}\n\n"
    )


def main(directory: Path) -> None:
    paths = list(directory.glob("*.h5ad"))

    for path in paths:
        adata = sc.read_h5ad(path)

        _print_stats(adata, "default")

        for other_dir_name in other_dir_names:
            other_path = directory.parent / other_dir_name / path.name
            if not other_path.exists():
                print(f"Not existing for {other_dir_name}.")
                continue

            _print_stats(sc.read_h5ad(other_path), other_dir_name)


if __name__ == "__main__":
    main(Path("/gpfs/workdir/blampeyq/res_novae/X_scConcept"))
