import numpy as np

from tiny_neural_engine.layers import Linear, Sigmoid, ReLU, BCE, Parameter, CrossEntropyLoss


def test_linear():
    x_np = np.random.randn(5, 10)
    x = Parameter("input", x_np, requires_grad=True)

    linear = Linear(x, 1)
    w_np = linear.weight.value
    b_np = linear.bias.value

    out_np = linear([x.with_value(x.value)]).value.sum(0)

    linear.backward_pass()

    import torch
    import torch.nn.functional as F

    x_torch = torch.from_numpy(x_np).float()
    x_torch.requires_grad = True
    w_torch = torch.from_numpy(w_np).float()
    w_torch.requires_grad = True
    b_torch = torch.from_numpy(b_np).float()
    b_torch.requires_grad = True

    out_torch_full = F.linear(x_torch, w_torch.T, b_torch)
    out_torch = out_torch_full.sum(0)
    out_torch.backward()

    assert np.allclose(out_np, out_torch.detach().numpy())
    assert np.allclose(x.grad, x_torch.grad.numpy())  # pyright: ignore
    assert np.allclose(linear.weight.grad, w_torch.grad.numpy())  # pyright: ignore
    assert np.allclose(linear.bias.grad, b_torch.grad.numpy())  # pyright: ignore


def test_sigmoid():
    x_np = np.random.randn(5, 10)
    x = Parameter("input", x_np, requires_grad=True)

    linear = Linear(x, 1)
    sigmoid = Sigmoid(linear)

    w_np = linear.weight.value
    b_np = linear.bias.value

    logits = linear([x.with_value(x.value)])
    out_np = sigmoid([logits]).value.sum(0)

    sigmoid.backward_pass()

    import torch
    import torch.nn.functional as F

    x_torch = torch.from_numpy(x_np).float()
    x_torch.requires_grad = True
    w_torch = torch.from_numpy(w_np).float()
    w_torch.requires_grad = True
    b_torch = torch.from_numpy(b_np).float()
    b_torch.requires_grad = True

    logits_torch = F.linear(x_torch, w_torch.T, b_torch)
    out_full_torch = F.sigmoid(logits_torch)
    out_torch = out_full_torch.sum(0)
    out_torch.backward()

    assert np.allclose(out_np, out_torch.detach().numpy())
    assert np.allclose(x.grad, x_torch.grad.numpy())  # pyright: ignore
    assert np.allclose(linear.weight.grad, w_torch.grad.numpy())  # pyright: ignore
    assert np.allclose(linear.bias.grad, b_torch.grad.numpy())  # pyright: ignore


def test_relu():
    x_np = np.random.randn(5, 10)
    x = Parameter("input", x_np, requires_grad=True)

    linear = Linear(x, 1)
    relu = ReLU(linear)

    w_np = linear.weight.value
    b_np = linear.bias.value

    logits = linear([x.with_value(x.value)])
    out_np = relu([logits]).value.sum(0)

    relu.backward_pass()

    import torch
    import torch.nn.functional as F

    x_torch = torch.from_numpy(x_np).float()
    x_torch.requires_grad = True
    w_torch = torch.from_numpy(w_np).float()
    w_torch.requires_grad = True
    b_torch = torch.from_numpy(b_np).float()
    b_torch.requires_grad = True

    logits_torch = F.linear(x_torch, w_torch.T, b_torch)
    out_full_torch = F.relu(logits_torch)
    out_torch = out_full_torch.sum(0)
    out_torch.backward()

    #print("x.grad", x.grad, x_torch.grad.numpy())
    #print("linear.weight.grad", linear.weight.grad)

    assert np.allclose(out_np, out_torch.detach().numpy())
    assert np.allclose(x.grad, x_torch.grad.numpy())  # pyright: ignore
    assert np.allclose(linear.weight.grad, w_torch.grad.numpy())  # pyright: ignore
    assert np.allclose(linear.bias.grad, b_torch.grad.numpy())  # pyright: ignore


def test_bce():
    l_np = np.random.randn(5)
    t_np = np.array([1, 1, 0, 0, 1])

    logits = Parameter("logits", l_np, requires_grad=True)
    target = Parameter("targets", t_np, requires_grad=False)

    bce = BCE(logits, target)

    out_np = bce([logits.with_value(logits.value), target.with_value(target.value)]).value

    bce.backward_pass()

    import torch
    import torch.nn.functional as F

    l_torch = torch.from_numpy(l_np).float()
    l_torch.requires_grad = True
    t_torch = torch.from_numpy(t_np).float()

    bce_torch = F.binary_cross_entropy_with_logits(l_torch, t_torch)

    bce_torch.backward()
    assert np.allclose(out_np, bce_torch.detach().numpy())
    assert np.allclose(logits.grad, l_torch.grad.numpy())  # pyright: ignore


def test_ce():
    l_np = np.random.randn(5, 4)
    t_np_int = np.array([2, 3, 0, 0, 1])
    t_np = np.zeros((5, 4))
    t_np[np.arange(5), t_np_int] = 1.0

    logits = Parameter("logits", l_np, requires_grad=True)
    target = Parameter("targets", t_np, requires_grad=False)

    loss = CrossEntropyLoss(logits, target)

    out_np = loss([logits.with_value(logits.value), target.with_value(target.value)]).value

    loss.backward_pass()

    import torch
    import torch.nn.functional as F

    l_torch = torch.from_numpy(l_np).float()
    l_torch.requires_grad = True
    t_torch = torch.from_numpy(t_np_int).long()

    loss_torch = F.cross_entropy(l_torch, t_torch)

    loss_torch.backward()
    assert np.allclose(out_np, loss_torch.detach().numpy())
    assert np.allclose(logits.grad, l_torch.grad.numpy())  # pyright: ignore


if __name__ == '__main__':
    test_linear()
    test_sigmoid()
    test_relu()
    test_bce()
    test_ce()
    print("Successful!")

