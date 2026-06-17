import novae

from ._utils import adata_small as adata


def test_init_prototypes():
    model = novae.Novae(adata, num_prototypes=20)

    prototypes = model.swav_head._prototypes.data.clone()
    model.init_prototypes(adata)
    assert (model.swav_head._prototypes.data != prototypes).all()


def test_preserve_clustering():
    model = novae.Novae(adata)
    model.mode.trained = True

    clusters_levels = model.swav_head.clusters_levels
    leiden_codes = model.swav_head.leiden_clustering()

    model.save_pretrained("tests/test_proto_clustering")

    model2 = novae.Novae.from_pretrained("tests/test_proto_clustering")

    assert (model2.swav_head.clusters_levels == clusters_levels).all()
    assert (model2.swav_head.leiden_clustering() == leiden_codes).all()


def test_preserve_zero_shot_prototypes():
    model = novae.Novae(adata, num_prototypes=50)
    model.mode.trained = True

    prototypes_before = model.swav_head._prototypes.data.clone()

    model.compute_representations(adata, zero_shot=True)
    prototypes = model.swav_head._prototypes.data.clone()
    leiden_codes = model.swav_head.leiden_clustering(resolution=2)

    assert (prototypes != prototypes_before).all(), "Prototypes should be updated after computing representations."

    model.save_pretrained("tests/test_proto_zero_shot")
    model2 = novae.Novae.from_pretrained("tests/test_proto_zero_shot")

    prototypes_after = model2.swav_head._prototypes.data.clone()
    assert (prototypes_after == prototypes).all(), "Prototypes should be preserved after re-loading a zero-shot model."

    leiden_codes_after = model2.swav_head.leiden_clustering(resolution=2)
    assert (leiden_codes_after == leiden_codes).all(), "Leiden should be preserved after re-loading a zero-shot model."
