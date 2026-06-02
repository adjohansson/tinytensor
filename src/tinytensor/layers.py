import numpy as np

from typing import Optional

from tinytensor.core import Node, NodeValue, Operation, Parameter


class Linear(Operation):
    def __init__(self, input_node: Node, out_features: int):
        fan_in = input_node.shape[1]
        assert isinstance(fan_in, int)
        self.weight = Parameter("weight", np.random.randn(fan_in, out_features) / np.sqrt(fan_in), requires_grad=True)
        self.bias = Parameter("bias", np.random.randn(out_features)*0.01, requires_grad=True)
        self.in_node = input_node

        self._in_value: Optional[np.ndarray] = None
        super().__init__(in_nodes=[input_node], parameters=[self.weight, self.bias], shape=(None, out_features))

    def _forward(self, node_values: dict[Node, NodeValue]) -> NodeValue:
        self._in_value = node_values[self.in_node].value

        out_value = self._in_value @ self.weight.value + self.bias.value
        return self.with_value(out_value)

    def _backward(self, grad: NodeValue) -> list[NodeValue]:
        assert self._in_value is not None

        grads = [
            self.weight.with_value(self._in_value.T @ grad.value),  # (batch, in_features).T @ (batch, out_features) -> (in_features, out_features)
            self.bias.with_value(grad.value.sum(axis=0)),  # (batch, out_features,) -> (out_features,)
        ]
        if isinstance(self.in_node, Operation) or isinstance(self.in_node, Parameter) and self.in_node.requires_grad:
            grads.append(
                self.in_node.with_value(grad.value @ self.weight.value.T)  # (batch, out) @ (in, out).T -> (batch, in)
            )

        return grads


class Sigmoid(Operation):
    def __init__(self, input_node: Node):
        super().__init__(in_nodes=[input_node], parameters=[], shape=input_node.shape)
        self.in_node = input_node
        self._out_value: Optional[np.ndarray] = None

    def _forward(self, node_values: dict[Node, NodeValue]) -> NodeValue:
        in_value = node_values[self.in_node]

        self._out_value = 1.0 / (1.0 + np.exp(-in_value.value))
        return self.with_value(self._out_value)

    def _backward(self, grad: NodeValue) -> list[NodeValue]:
        assert self._out_value is not None

        function_grad = self._out_value * (1.0 - self._out_value)  # ds(x)/dx
        grads = [
            self.in_node.with_value(function_grad * grad.value)
        ]
        return grads


class ReLU(Operation):
    def __init__(self, input_node: Node):
        super().__init__(in_nodes=[input_node], parameters=[], shape=input_node.shape)
        self.in_node = input_node
        self._in_value: Optional[np.ndarray] = None

    def _forward(self, node_values: dict[Node, NodeValue]) -> NodeValue:
        self._in_value = node_values[self.in_node].value

        out_value = np.clip(self._in_value, a_min=0.0, a_max=None)
        return self.with_value(out_value)

    def _backward(self, grad: NodeValue) -> list[NodeValue]:
        assert self._in_value is not None

        in_grad = np.where(self._in_value >= 0.0, grad.value, 0.0)
        return [self.in_node.with_value(in_grad)]


def _log_sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0,
                    -np.log(1.0 + np.exp(-x)),
                    x - np.log(np.exp(x) + 1.0))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0, 
                    1.0 / (1.0 + np.exp(-x)),
                    np.exp(x) / (np.exp(x) + 1.0))


def softmax(x: np.ndarray, axis: Optional[int]) -> np.ndarray:
    baseline = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x-baseline)
    return e / e.sum(axis=axis, keepdims=True)


def log_softmax(x: np.ndarray, axis: Optional[int]) -> np.ndarray:
    baseline = np.max(x, axis=axis, keepdims=True)
    d = x - baseline
    return d - np.log(np.sum(np.exp(d), axis=axis, keepdims=True))


class BCE(Operation):
    def __init__(self, input_node: Node, target_node: Node, reduction="mean"):
        assert input_node.shape == target_node.shape

        super().__init__(in_nodes=[input_node, target_node], parameters=[], shape=())
        self.in_node = input_node
        self.target_node = target_node
        self.reduction = reduction

        self._in_value: Optional[np.ndarray] = None
        self._target_value: Optional[np.ndarray] = None

    def _forward(self, node_values: dict[Node, NodeValue]) -> NodeValue:
        self._in_value = node_values[self.in_node].value
        self._target_value = node_values[self.target_node].value

        out_value = -self._target_value * _log_sigmoid(self._in_value) - (1.0-self._target_value)*_log_sigmoid(-self._in_value)
        if self.reduction == "mean":
            out_value = out_value.mean()
        elif self.reduction == "sum":
            out_value = out_value.sum()
        else:
            raise ValueError()

        return self.with_value(out_value)

    def _backward(self, grad: NodeValue) -> list[NodeValue]:
        x = self._in_value
        target = self._target_value
        assert x is not None
        assert target is not None

        function_grad = _sigmoid(x) - target
        in_grad = function_grad * grad.value  # Shape (batch, 1)*(1,) = (batch,)
        if self.reduction == "mean":
            in_grad /= in_grad.size

        assert in_grad.shape == x.shape

        return [self.in_node.with_value(in_grad)]


class CrossEntropyLoss(Operation):
    def __init__(self, input_node: Node, target_node: Node, reduction="mean"):
        assert input_node.shape == target_node.shape

        super().__init__(in_nodes=[input_node, target_node], parameters=[], shape=())
        self.in_node = input_node
        self.target_node = target_node
        self.reduction = reduction

        self._in_value: Optional[np.ndarray] = None
        self._target_value: Optional[np.ndarray] = None

    def _forward(self, node_values: dict[Node, NodeValue]) -> NodeValue:
        self._in_value = node_values[self.in_node].value
        self._target_value = node_values[self.target_node].value

        out_value = -(self._target_value * log_softmax(self._in_value, axis=-1)).sum(axis=-1)
        if self.reduction == "mean":
            out_value = out_value.mean()
        elif self.reduction == "sum":
            out_value = out_value.sum()
        else:
            raise ValueError()

        return self.with_value(out_value)

    def _backward(self, grad: NodeValue) -> list[NodeValue]:
        x = self._in_value
        target = self._target_value
        assert x is not None
        assert target is not None
        target_sum = np.sum(target, axis=-1, keepdims=True)

        function_grad = target_sum * softmax(x, axis=-1) - target  # Shape (batch, categories)
        in_grad = function_grad * grad.value  # Shape (batch, categories)*() = (batch, categories)
        if self.reduction == "mean":
            in_grad *= in_grad.shape[-1]
            in_grad /= in_grad.size

        assert in_grad.shape == x.shape

        return [self.in_node.with_value(in_grad)]

