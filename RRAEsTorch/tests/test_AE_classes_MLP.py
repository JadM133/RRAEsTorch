import RRAEsTorch.config
import pytest
from RRAEsTorch.AE_classes import (
    RRAE_MLP,
    Vanilla_AE_MLP,
    IRMAE_MLP,
    LoRAE_MLP,
)
import numpy.random as random
import numpy as np
import torch

@pytest.mark.parametrize("dim_D", (10, 15, 50))
@pytest.mark.parametrize("latent", (200, 400, 800))
@pytest.mark.parametrize("num_modes", (1, 2, 6))
class Test_AEs_shapes:
    def test_RRAE_MLP(self, latent, num_modes, dim_D):
        x = random.normal(size=(dim_D, 500))
        x = torch.tensor(x, dtype=torch.float32)
        model = RRAE_MLP(x.shape[1], latent, num_modes)
        y = model.encode(x)
        assert y.shape == (dim_D, latent)
        y = model.perform_in_latent(y, k_max=num_modes)
        _, sing_vals, _ = torch.linalg.svd(y, full_matrices=False)
        assert sing_vals[num_modes + 1] < 1e-4
        assert y.shape == (dim_D, latent)
        assert model.decode(y).shape == (dim_D, 500)

    def test_Vanilla_MLP(self, latent, num_modes, dim_D):
        x = random.normal(size=(dim_D, 500))
        x = torch.tensor(x, dtype=torch.float32)
        model = Vanilla_AE_MLP(x.shape[1], latent)
        y = model.encode(x)
        assert y.shape == (dim_D, latent)
        x = model.decode(y)
        assert x.shape == (dim_D, 500)

    def test_IRMAE_MLP(self, latent, num_modes, dim_D):
        x = random.normal(size=(dim_D, 500))
        x = torch.tensor(x, dtype=torch.float32)
        model = IRMAE_MLP(x.shape[1], latent, linear_l=2)
        y = model.encode(x)
        assert y.shape == (dim_D, latent)
        assert len(model._encode.layers_l) == 2
        x = model.decode(y)
        assert x.shape == (dim_D, 500)

    def test_LoRAE_MLP(self, latent, num_modes, dim_D):
        x = random.normal(size=(dim_D, 500))
        x = torch.tensor(x, dtype=torch.float32)
        model = LoRAE_MLP(x.shape[1], latent)
        y = model.encode(x)
        assert y.shape == (dim_D, latent)
        assert len(model._encode.layers_l) == 1
        x = model.decode(y)
        assert x.shape == (dim_D, 500)

def test_getting_SVD_coeffs():
    data = random.uniform(size=(15, 500))
    data = torch.tensor(data, dtype=torch.float32)
    model_s = RRAE_MLP(data.shape[1], 200, 3)
    basis, coeffs = model_s.latent(data, k_max=3, get_basis_coeffs=True)
    assert basis.shape == (200, 3)
    assert coeffs.shape == (3, 15)

