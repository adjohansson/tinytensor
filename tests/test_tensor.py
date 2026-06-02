import numpy as np

from tinytensor.core import Parameter
from tinytensor.layers import Linear, Sigmoid, ReLU, BCE, CrossEntropyLoss

from tinytensor.tensor import LinearCombination


def test_combination():
    x1_np = np.array([2.0, 3.0])
    x2_np = np.array([0.0, 1.0])

    x1 = Parameter("x1", x1_np, requires_grad=True)
    x2 = Parameter("x2", x2_np, requires_grad=True)

    combination = LinearCombination([x1, x2])

    out_np = combination(x1.with_value(x1_np), x2.with_value(x2_np)).value.sum(0)

    combination.backward_pass()

    import torch
    import torch.nn.functional as F

    x1_torch = torch.from_numpy(x1_np).float()
    x1_torch.requires_grad = True
    x2_torch = torch.from_numpy(x2_np).float()
    x2_torch.requires_grad = True

    coefficients_torch = torch.tensor([float(c.value) for c in combination.coefficients], requires_grad=True)

    combination_torch = coefficients_torch[0]*x1_torch + coefficients_torch[1]*x2_torch
    out_torch = combination_torch.sum(0)
    out_torch.backward()

    assert np.allclose(out_np, out_torch.detach().numpy())
    assert np.allclose(x1.grad, x1_torch.grad.numpy())  # pyright: ignore
    assert np.allclose(x2.grad, x2_torch.grad.numpy())  # pyright: ignore
    assert np.allclose(combination.coefficients[0].grad, coefficients_torch.grad[0].numpy())  # pyright: ignore
    assert np.allclose(combination.coefficients[1].grad, coefficients_torch.grad[1].numpy())  # pyright: ignore

