import RRAEsTorch.config
import pytest
from RRAEsTorch.AE_classes import (
    RRAE_CNN,
    Vanilla_AE_CNN,
    IRMAE_CNN,
    LoRAE_CNN,
)
import numpy.random as random
import numpy as np
import torch

methods = ["encode", "decode"]

@pytest.mark.parametrize("width", (10, 17, 149))
@pytest.mark.parametrize("height", (20,))
@pytest.mark.parametrize("latent", (200,))
@pytest.mark.parametrize("num_modes", (1,))
@pytest.mark.parametrize("channels", (1, 3, 5))
@pytest.mark.parametrize("num_samples", (10, 100))
class Test_AEs_shapes:
    def test_RRAE_CNN(self, latent, num_modes, width, height, channels, num_samples):
        x = random.normal(size=(num_samples, channels, width, height))
        x = torch.tensor(x, dtype=torch.float32)
        kwargs = {"kwargs_dec": {"stride": 2}}
        model = RRAE_CNN(
            x.shape[1], x.shape[2], x.shape[3], latent, num_modes, **kwargs
        )
        y = model.encode(x)
        assert y.shape == (num_samples, latent)
        y = model.latent(x, k_max=num_modes)
        _, sing_vals, _ = torch.linalg.svd(y, full_matrices=False)
        assert sing_vals[num_modes + 1] < 1e-5
        assert y.shape == (num_samples, latent)
        assert model.decode(y).shape == (num_samples, channels, width, height)

    def test_Vanilla_CNN(self, latent, num_modes, width, height, channels, num_samples):
        x = random.normal(size=(num_samples, channels, width, height))
        x = torch.tensor(x, dtype=torch.float32)
        kwargs = {"kwargs_dec": {"stride": 2}}
        model = Vanilla_AE_CNN(
            x.shape[1], x.shape[2], x.shape[3], latent, **kwargs
        )
        y = model.encode(x)
        assert y.shape == (num_samples, latent)
        x = model.decode(y)
        assert x.shape == (num_samples, channels, width, height)


    def test_IRMAE_CNN(self, latent, num_modes, width, height, channels, num_samples):
        x = random.normal(size=(num_samples, channels, width, height))
        x = torch.tensor(x, dtype=torch.float32)
        model = IRMAE_CNN(
            x.shape[1], x.shape[2], x.shape[3], latent, linear_l=2
        )
        y = model.encode(x)
        assert y.shape == (num_samples, latent)
        assert len(model._encode.layers[-1].layers_l) == 2
        x = model.decode(y)
        assert x.shape == (num_samples, channels, width, height)

    def test_LoRAE_CNN(self, latent, num_modes, width, height, channels, num_samples):
        x = random.normal(size=(num_samples, channels, width, height))
        x = torch.tensor(x, dtype=torch.float32)
        model = LoRAE_CNN(x.shape[1], x.shape[2], x.shape[3], latent)
        y = model.encode(x)
        assert y.shape == (num_samples, latent)
        assert len(model._encode.layers[-1].layers_l) == 1
        x = model.decode(y)
        assert x.shape == (num_samples, channels, width, height)
