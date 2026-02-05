import RRAEsTorch.config
from RRAEsTorch.AE_classes import RRAE_CNN
from RRAEsTorch.training_classes import RRAE_Trainor_class
import numpy.random as random
import torch

def test_save():  # Only to test if saving/loading is causing a problem
    data = random.normal(size=(1, 1, 28, 28))
    data = torch.tensor(data, dtype=torch.float32)
    model_cls = RRAE_CNN

    trainor = RRAE_Trainor_class(
        data,
        model_cls,
        latent_size=100,
        channels=data.shape[1],
        width=data.shape[2],
        height=data.shape[3],
        pre_func_inp=lambda x: x * 2 / 17,
        pre_func_out=lambda x: x / 2,
        k_max=2,
    )

    trainor.save_model("test_")
    new_trainor = RRAE_Trainor_class()
    new_trainor.load_model("test_", erase=True)
    try:
        pr = trainor.model(data[0:1], k_max=2)
    except Exception as e:
        raise ValueError(f"Original trainor failed with following exception {e}")
    try:
        pr = new_trainor.model(data[0:1], k_max=2)
    except Exception as e:
        raise ValueError(f"Failed with following exception {e}")


def test_save_with_final_act():
    data = random.normal(size=(1, 1, 28, 28))
    data = torch.tensor(data, dtype=torch.float32)
    
    model_cls = RRAE_CNN

    trainor = RRAE_Trainor_class(
        data,
        model_cls,
        latent_size=100,
        channels=data.shape[1],
        width=data.shape[2],
        height=data.shape[3],
        kwargs_dec={"final_activation": torch.sigmoid},
        k_max=2,
    )

    trainor.save_model("test_")
    new_trainor = RRAE_Trainor_class()
    new_trainor.load_model("test_", erase=True)
    try:
        pr = new_trainor.model(data[0:1], k_max=2)
        assert torch.max(pr) <= 1.0, "Final activation not working"
        assert torch.min(pr) >= 0.0, "Final activation not working"
    except Exception as e:
        raise ValueError(f"Failed with following exception {e}")