from RRAEsTorch.utilities import stable_SVD
from torch.linalg import svd as normal_svd
import numpy.random as random
import pytest
import numpy as np
import torch

def stable_SVD_to_scalar(A):
    U, s, Vt = stable_SVD(A)
    return torch.linalg.norm((U * s) @ Vt)  # Any scalar depending on U, s, and Vt.

def normal_svd_to_scalar(A):
    U, s, Vt = normal_svd(A, full_matrices=False)
    return torch.linalg.norm((U * s) @ Vt)  # Any scalar depending on U, s, and Vt.

@pytest.mark.parametrize(
    "length, width",
    [(10, 10), (100, 10), (10, 100), (50000, 100), (1000, 1000), (100, 50000)],
)
def test_random_normal(length, width):
    A = random.uniform(low=0.0, high=1.0, size=(length, width))
    A = torch.tensor(A, dtype=torch.float32)

    A = A.clone().detach().requires_grad_(True)

    stable_value = stable_SVD_to_scalar(A)
    stable_value.backward()
    stable_grad = A.grad

    A = A.clone().detach().requires_grad_(True)

    normal_value = normal_svd_to_scalar(A)
    normal_value.backward()
    normal_grad = A.grad
    
    assert torch.allclose(stable_value, normal_value, atol=1e-5, rtol=1e-5)
    assert torch.allclose(stable_grad, normal_grad, atol=1e-5, rtol=1e-5)