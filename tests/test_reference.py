import novae
from novae._constants import Keys
from novae.utils import get_reference


def test_get_reference():
    adatas = novae.toy_dataset()

    novae.spatial_neighbors(adatas)

    assert get_reference(adatas, "largest").n_obs == 1958

    assert len(get_reference(adatas, "all")) == 3

    assert get_reference(adatas, 1)[0].n_obs == 1957

    assert [adata.n_obs for adata in get_reference(adatas, [1, 2])] == [1957, 1956]

    sid1 = adatas[1].obs[Keys.SLIDE_ID].iloc[0]
    assert get_reference(adatas, sid1)[0].obs[Keys.SLIDE_ID].iloc[0] == sid1


def test_zero_shot_with_different_references():
    adatas = novae.toy_dataset(compute_spatial_neighbors=True)

    model = novae.Novae(adatas, num_prototypes=20)
    model.mode.trained = True  # trick to avoid assert error in compute_representations

    model.compute_representations(adatas, zero_shot=True, reference="all")
    key_added = model.assign_domains(adatas[0], resolution=1)
    domains_1 = adatas[0].obs[key_added].copy()

    model.compute_representations(adatas, zero_shot=True, reference="all")
    key_added = model.assign_domains(adatas[0], resolution=1)
    domains_1_repro = adatas[0].obs[key_added].copy()

    assert domains_1.equals(domains_1_repro), "Domains should be the same when using the same reference."

    model.compute_representations(adatas, zero_shot=True, reference="largest")
    key_added = model.assign_domains(adatas[0], resolution=1)
    domains_2 = adatas[0].obs[key_added].copy()

    model.compute_representations(adatas, zero_shot=True, reference=1)
    key_added = model.assign_domains(adatas[0], resolution=1)
    domains_3 = adatas[0].obs[key_added].copy()

    assert not domains_1.equals(domains_2) and not domains_2.equals(domains_3) and not domains_1.equals(domains_3), (
        "Domains should be different when using different references."
    )
