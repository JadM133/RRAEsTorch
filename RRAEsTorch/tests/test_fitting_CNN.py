import RRAEsTorch.config
import pytest
from RRAEsTorch.AE_classes import (
    RRAE_CNN,
    Vanilla_AE_CNN,
    IRMAE_CNN,
    LoRAE_CNN,
)
from RRAEsTorch.training_classes import RRAE_Trainor_class, Trainor_class, AE_Trainor_class
import numpy.random as random
import torch

@pytest.mark.parametrize(
    "model_cls, sh, lf",
    [
        (Vanilla_AE_CNN, (10, 1, 2, 2), "default"),
        (LoRAE_CNN, (10, 6, 16, 16), "nuc"),
    ],
)
def test_AE_fitting(model_cls, sh, lf):
    x = random.normal(size=sh)
    x = torch.tensor(x, dtype=torch.float32)
    trainor = AE_Trainor_class(
        x,
        model_cls,
        latent_size=100,
        channels=x.shape[1],
        width=x.shape[2],
        height=x.shape[3],
        samples=x.shape[0],  # Only for weak
        k_max=2,
    )
    kwargs = {
        "step_st": [2],
        "loss_kwargs": {
            "lambda_nuc": 0.001,
            "find_layer": lambda model: model.encode.layers[-2].layers[-1].weight,
            "loss_type": lf
        },
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


def test_IRMAE_fitting():
    model_cls = IRMAE_CNN
    lf = "default"
    sh = (10, 3, 12, 12)
    x = random.normal(size=sh)
    x = torch.tensor(x, dtype=torch.float32)
    trainor = AE_Trainor_class(
        x,
        model_cls,
        latent_size=100,
        channels=x.shape[1],
        width=x.shape[2],
        height=x.shape[3],
        k_max=2,
        linear_l=4,
    )
    kwargs = {"step_st": [2], "loss_type": lf}
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
    sh = (10, 1, 20, 20)
    model_cls = RRAE_CNN
    x = random.normal(size=sh)
    x = torch.tensor(x, dtype=torch.float32)
    trainor = RRAE_Trainor_class(
        x,
        model_cls,
        latent_size=100,
        channels=x.shape[1],
        width=x.shape[2],
        height=x.shape[3],
        k_max=2,
    )
    training_kwargs = {
        "step_st": [2],
        "loss_type": "default"
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