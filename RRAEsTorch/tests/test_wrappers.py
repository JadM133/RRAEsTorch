from RRAEsTorch.wrappers import vmap_wrap, norm_wrap
from torchvision.ops import MLP
import pytest
import numpy as np
import math
import numpy.random as random
import torch

def test_vmap_wrapper():
    # Usually MLP only accepts a vector, here we give
    # a tensor and vectorize over the last axis twice
    data = random.normal(size=(50, 60, 600))
    data = torch.tensor(data, dtype=torch.float32)
    
    model_cls = vmap_wrap(MLP, -1, 2)
    model = model_cls(50, [64, 100])
    try:
        model(data)
    except ValueError:
        pytest.fail("Vmap wrapper is not working properly.")

def test_norm_wrapper():
    # Testing the keep_normalized kwarg
    data = random.normal(size=(50,))
    data = torch.tensor(data, dtype=torch.float32)
    model_cls = norm_wrap(MLP, data, "minmax", None, data, "minmax", None)
    model = model_cls(50, [64, 100])
    try:
        assert not torch.allclose(model(data), model(data, keep_normalized=True))
    except AssertionError:
        pytest.fail("The keep_normalized kwarg for norm wrapper is not behaving as expected.")

    # Testing minmax with knwon mins and maxs
    data = np.linspace(-1, 1, 100)
    data = torch.tensor(data, dtype=torch.float32)
    model_cls = norm_wrap(MLP, data, "minmax", None, data, "minmax", None)
    model = model_cls(50, [64, 100])
    try:
        assert 0.55 == model.norm_in.default(None, 0.1)
        assert -0.8 == model.inv_norm_out.default(None, 0.1)
    except AssertionError:
        pytest.fail("Something wrong with minmax wrapper.")

    # Testing meanstd with knwon mean and std
    data = random.normal(size=(50,))
    data = (data-np.mean(data))/np.std(data)
    data = data*2.0 + 1.0  # mean of 1 and std of 2
    data = torch.tensor(data, dtype=torch.float32)

    model_cls = norm_wrap(MLP, data, "meanstd", None, data, "meanstd", None)
    model = model_cls(50, [64, 100])
    try:
        assert math.isclose(2, model.norm_in.default(None, 5), rel_tol=1e-1, abs_tol=1e-1)
        assert math.isclose(7, model.inv_norm_out.default(None, 3), rel_tol=1e-1, abs_tol=1e-1)
    except AssertionError:
        pytest.fail("Something wrong with norm wrapper.")