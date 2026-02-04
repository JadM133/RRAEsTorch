import RRAEsTorch.config
import pytest
from RRAEsTorch.AE_classes import (
    RRAE_CNN,
    Vanilla_AE_CNN,
    IRMAE_CNN,
    LoRAE_CNN,
)
from RRAEsTorch.wrappers import vmap_wrap
import numpy.random as random
import numpy as np
import torch

methods = ["encode", "decode"]

v_RRAE_CNN = vmap_wrap(RRAE_CNN, -1, 1, methods)
v_Vanilla_AE_CNN = vmap_wrap(Vanilla_AE_CNN, -1, 1, methods)
v_IRMAE_CNN = vmap_wrap(IRMAE_CNN, -1, 1, methods)
v_LoRAE_CNN = vmap_wrap(LoRAE_CNN, -1, 1, methods)

@pytest.mark.parametrize("width", (10, 17, 149))
@pytest.mark.parametrize("height", (20,))
@pytest.mark.parametrize("latent", (200,))
@pytest.mark.parametrize("num_modes", (1,))
@pytest.mark.parametrize("channels", (1, 3, 5))
@pytest.mark.parametrize("num_samples", (10, 100))
class Test_AEs_shapes:
    def test_RRAE_CNN(self, latent, num_modes, width, height, channels, num_samples):
        x = random.normal(size=(channels, width, height, num_samples))
        x = torch.tensor(x, dtype=torch.float32)
        kwargs = {"kwargs_dec": {"stride": 2}}
        model = v_RRAE_CNN(
            x.shape[0], x.shape[1], x.shape[2], latent, num_modes, **kwargs
        )
        y = model.encode(x)
        assert y.shape == (latent, num_samples)
        y = model.latent(x, k_max=num_modes)
        _, sing_vals, _ = torch.linalg.svd(y, full_matrices=False)
        assert sing_vals[num_modes + 1] < 1e-5
        assert y.shape == (latent, num_samples)
        assert model.decode(y).shape == (channels, width, height, num_samples)

    def test_Vanilla_CNN(self, latent, num_modes, width, height, channels, num_samples):
        x = random.normal(size=(channels, width, height, num_samples))
        x = torch.tensor(x, dtype=torch.float32)
        kwargs = {"kwargs_dec": {"stride": 2}}
        model = v_Vanilla_AE_CNN(
            x.shape[0], x.shape[1], x.shape[2], latent, **kwargs
        )
        y = model.encode(x)
        assert y.shape == (latent, num_samples)
        x = model.decode(y)
        assert x.shape == (channels, width, height, num_samples)


    def test_IRMAE_CNN(self, latent, num_modes, width, height, channels, num_samples):
        x = random.normal(size=(channels, width, height, num_samples))
        x = torch.tensor(x, dtype=torch.float32)
        model = v_IRMAE_CNN(
            x.shape[0], x.shape[1], x.shape[2], latent, linear_l=2
        )
        y = model.encode(x)
        assert y.shape == (latent, num_samples)
        assert len(model._encode.layers[-1].layers_l) == 2
        x = model.decode(y)
        assert x.shape == (channels, width, height, num_samples)

    def test_LoRAE_CNN(self, latent, num_modes, width, height, channels, num_samples):
        x = random.normal(size=(channels, width, height, num_samples))
        x = torch.tensor(x, dtype=torch.float32)
        model = v_LoRAE_CNN(x.shape[0], x.shape[1], x.shape[2], latent)
        y = model.encode(x)
        assert y.shape == (latent, num_samples)
        assert len(model._encode.layers[-1].layers_l) == 1
        x = model.decode(y)
        assert x.shape == (channels, width, height, num_samples)
