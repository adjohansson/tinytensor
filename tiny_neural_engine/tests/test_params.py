from tiny_neural_engine.layers import Linear, Parameter
import numpy as np


def test_cutoff():
    """
    Test that only parameters with requires_grad have gradients computed in backward pass
    """

    x = Parameter("input", np.random.randn(10, 3), requires_grad=False)
    l1 = Linear(x, 3)
    l2 = Linear(x, 3)

    l1.weight.requires_grad = False
    l1.bias.requires_grad = False

    _output = l2(x.with_value(x.value)).value

    assert not l1._requires_grad
    assert l2._requires_grad

    l2.backward_pass()
    assert l1.weight.grad is None
    assert l1.bias.grad is None
    assert l2.weight.grad is not None
    assert l2.bias.grad is not None


if __name__ == '__main__':
    test_cutoff()
    print("Tests passed!")

