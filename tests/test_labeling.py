import novae


def test_label_domains():
    adata = novae.toy_dataset(n_panels=1, compute_spatial_neighbors=True)[0]
    prompt = novae.label_domains(adata, "domain", return_prompt=True)

    assert "messages" in prompt
    assert "output_schema" in prompt
    assert len(prompt["messages"]) == 2
