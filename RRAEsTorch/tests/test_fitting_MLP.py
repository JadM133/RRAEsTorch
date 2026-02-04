import RRAEsTorch.config
import numpy.random as random
import pytest
from RRAEsTorch.AE_classes import (
    RRAE_MLP,
    Vanilla_AE_MLP,
    IRMAE_MLP,
    LoRAE_MLP,
)
from torchvision.ops import MLP
from RRAEsTorch.training_classes import RRAE_Trainor_class, Trainor_class, AE_Trainor_class
import torch

@pytest.mark.parametrize(
    "model_cls, sh, lf",
    [
        (Vanilla_AE_MLP, (500, 10), "default"),
        (LoRAE_MLP, (500, 10), "nuc"),
    ],
)
def test_fitting(model_cls, sh, lf):
    x = random.normal(size=sh)
    x = torch.tensor(x, dtype=torch.float32)
    trainor = AE_Trainor_class(
        x,
        model_cls,
        in_size=x.shape[0],
        data_size=x.shape[-1],
        samples=x.shape[-1],  # Only for weak
        norm_in="meanstd",
        norm_out="minmax",
        out_train=x,
        latent_size=2000,
        k_max=2,
    )
    kwargs = {
        "step_st": [2],
        "loss_kwargs": {"lambda_nuc": 0.001, "find_layer": lambda model: model._encode.layers_l[0].weight},
        "loss_type": lf
    }
    try:
        trainor.fit(
            x,
            x,
            verbose=False,
            training_kwargs=kwargs,
        )
    except Exception as e:
        assert False, f"Fitting failed with the following exception {repr(e)}"



def test_RRAE_fitting():
    sh = (500, 10)
    model_cls = RRAE_MLP
    x = random.normal(size=sh)
    x = torch.tensor(x, dtype=torch.float32)
    trainor = RRAE_Trainor_class(
        x,
        model_cls,
        in_size=x.shape[0],
        latent_size=2000,
        k_max=2,
    )
    training_kwargs = {
        "step_st": [2],
        "loss_type":"RRAE"
    }
    ft_kwargs = {
        "step_st": [2],
    }
    try:
        trainor.fit(
            x,
            x,
            verbose=False,
            training_kwargs=training_kwargs,
            ft_kwargs=ft_kwargs,    
        )
    except Exception as e:
        assert False, f"Fitting failed with the following exception {repr(e)}"

def test_IRMAE_fitting():
    model_cls = IRMAE_MLP
    lf = "default"
    sh = (500, 10)
    x = random.normal(size=sh)
    x = torch.tensor(x, dtype=torch.float32)
    trainor = AE_Trainor_class(
        x,
        model_cls,
        in_size=x.shape[0],
        data_size=x.shape[-1],
        latent_size=2000,
        k_max=2,
        linear_l=4,
    )
    kwargs = {"step_st": [2], "loss_type":lf}
    try:
        trainor.fit(
            x,
            x,
            verbose=False,
            training_kwargs=kwargs,
        )
    except Exception as e:
        assert False, f"Fitting failed with the following exception {repr(e)}"

def test_fitting():
    sh = (50, 100)
    model_cls = MLP
    x = random.normal(size=sh)
    x = torch.tensor(x, dtype=torch.float32)
    trainor = Trainor_class(
        x,
        model_cls,
        in_channels=x.shape[0],
        hidden_channels=[100, x.shape[0]]
    )
    training_kwargs = {
        "step_st": [2],
        "loss_type": "default"
    }

    try:
        trainor.fit(
            x,
            x,
            verbose=False,
            **training_kwargs,
        )
    except Exception as e:
        assert False, f"Fitting failed with the following exception {repr(e)}"