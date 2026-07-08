from ._utils import (
    fill_invalid_indices,
    get_reference,
    iter_slides,
    lower_var_names,
    parse_device_args,
    pretty_model_repr,
    pretty_num_parameters,
    tqdm,
    unique_leaves_indices,
    unique_obs,
    valid_indices,
    store_inference_mode,
)
from ._train import configure_optimizers, train
from ._validate import (
    check_available_domains_key,
    check_has_spatial_adjancency,
    check_slide_name_key,
    prepare_adatas,
    check_model_name,
)
from .build import spatial_neighbors
from .correct import batch_effect_correction
from .mode import Mode
