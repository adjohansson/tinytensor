import numpy as np

from abc import ABC, abstractmethod
from dataclasses import dataclass

import itertools

from typing import Tuple, List, Dict, Optional, Any


def _shapes_agree(shape1: Tuple[Optional[int], ...], shape2: Tuple[Optional[int]]) -> bool:
    if shape1 == shape2:
        return True

    for s, v in zip(shape1, shape2):
        if s is None or v is None:
            continue
        if s != v:
            return False
    return True
    


class Node:
    """
    Represents a point in the computation graph of a given module.

    Each step in a computation graph is assigned a Node with a unique id_ and a static shape.
    As results are computed for the node, these are returned as 'NodeValue' instances.
    """
    _N_NODES: int = 0  # Number of created nodes in total

    def __init__(self, shape: Tuple[Optional[int], ...]):
        self._id = Node._N_NODES
        Node._N_NODES += 1
        self._shape = shape

    @property
    def id_(self) -> int:
        return self._id

    def with_value(self, value: np.ndarray) -> 'NodeValue':
        """
        State that the value of this node is 'value'. Does modify the node internally.
        """
        assert _shapes_agree(self._shape, value.shape)
        return NodeValue(self, value)

    @property
    def shape(self) -> Tuple[Optional[int], ...]:
        return self._shape

    def __hash__(self) -> int:
        return hash(self._id)


class Parameter(Node):
    """
    A node which holds independently controllable parameters.
    """
    def __init__(self, name: str, value: np.ndarray, requires_grad: bool):
        super().__init__(shape=value.shape)
        self.name = name
        self.value = value
        self.grad: Optional[np.ndarray] = None
        self.requires_grad = requires_grad

    def set(self, value: np.ndarray):
        assert _shapes_agree(self.shape, value.shape)
        self.value = value

    def add_grad(self, grad: np.ndarray):
        assert self.requires_grad

        if self.grad is None:
            self.grad = np.zeros(self.shape)  # pyright: ignore
        self.grad += grad

    def __repr__(self):
        return f"Parameter({self.name}, shape={self.shape})"


class DataNode(Node):
    """
    A node which holds fixed non-trainable data.
    """
    def __init__(self, name: str, shape: Tuple[Optional[int], ...]):
        super().__init__(shape=shape)
        self.name = name
        self._data: Optional[np.ndarray] = None
        self._filled = False
        
    def fill(self, data: np.ndarray):
        assert _shapes_agree(self.shape, data.shape)
        self._data = data
        self._filled = True
    
    def collect(self) -> 'NodeValue':
        if not self._filled:
            raise ValueError(f"Cannot collect value from unfilled DataNode: {self}")
        assert self._data is not None
        return self.with_value(self._data)

    def __repr__(self):
        return f"DataNode({self.name}, shape={self.shape})"


@dataclass
class NodeValue:
    """
    Represents a node in the computation graph after it has been assigned a concrete value.
    """
    node: Node
    value: np.ndarray


class Operation(Node):
    def __init__(self, in_nodes: List[Node], parameters: List[Parameter], shape: Tuple[Optional[int], ...]):
        super().__init__(shape=shape)
        self._in_nodes = {node.id_: node for node in in_nodes}
        self._parameters = {node.id_: node for node in parameters}
        self._in_node_values: Dict[int, Optional[NodeValue]] = {node.id_: None for node in in_nodes}
        self._out_value: Optional[NodeValue] = None

    @abstractmethod
    def forward(self, node_values: List[NodeValue], *args: Any, **kwargs: Any) -> NodeValue:
        """Recieves a specification of the values at all input nodes. Should compute the value of the operations's output node."""

    @abstractmethod
    def backward(self, grad: NodeValue) -> List[NodeValue]:
        """
        Recieves a specification of the loss gradient at the output node (or it is assumed to be scalar 1.0 if grad=None).
        The function should then compute the corresponding loss gradient at the input nodes.
        """

    def __call__(self, node_values: List[NodeValue], *args: Any, **kwargs: Any) -> NodeValue:
        assert set(node_value.node.id_ for node_value in node_values)
        self._out_value = self.forward(node_values, *args, **kwargs)
        for node_value in node_values:
            self._in_node_values[node_value.node.id_] = node_value
        return self._out_value

    def forward_pass(self) -> NodeValue:
        """
        Recursively retrieves the output values from previous nodes in the graph and returns the output value
        """
        node_values = []
        for node in self._in_nodes.values():
            
            if isinstance(node, Parameter):
                value = node.value
                node_with_value = node.with_value(value)
            elif isinstance(node, DataNode):
                node_with_value = node.collect()
            elif isinstance(node, Operation):
                node_with_value = node.forward_pass()
            else:
                raise ValueError()
            node_values.append(node_with_value)
        return self(node_values=node_values)

    def backward_pass(self, grad: Optional[NodeValue] = None):
        assert self._out_value is not None
        if grad is None:
            grad = self.with_value(np.ones(self._out_value.value.shape))
        input_grads = self.backward(grad)
        for grad_value in input_grads:
            input_node = grad_value.node
            if isinstance(input_node, Operation):
                if not any(param.requires_grad for param in input_node.parameters()):
                    continue
                input_node.backward_pass(grad_value)
            elif isinstance(input_node, Parameter) and input_node.requires_grad:
                input_node.add_grad(grad=grad_value.value)

    def parameters(self) -> List[Parameter]:
        child_parameters = list(itertools.chain.from_iterable(
            in_node.parameters() if isinstance(in_node, Operation) else [in_node] if isinstance(in_node, Parameter) else [] 
            for in_node in self._in_nodes.values()
        ))
        return child_parameters + list(self._parameters.values())

    def zero_grad(self):
        for parameter in self.parameters():
            parameter.grad = None


class Linear(Operation):
    def __init__(self, input_node: Node, out_features: int):
        fan_in = input_node.shape[1]
        assert isinstance(fan_in, int)
        self.weight = Parameter("weight", np.random.randn(fan_in, out_features) / np.sqrt(fan_in), requires_grad=True)
        self.bias = Parameter("bias", np.random.randn(out_features)*0.01, requires_grad=True)
        self.in_node = input_node
        super().__init__(in_nodes=[input_node], parameters=[self.weight, self.bias], shape=(None, out_features))

    def forward(self, node_values: List[NodeValue]) -> NodeValue:
        in_value = node_values[0]

        out_value = in_value.value @ self.weight.value + self.bias.value
        return self.with_value(out_value)

    def backward(self, grad: NodeValue) -> List[NodeValue]:
        in_node_value = self._in_node_values[self.in_node.id_]
        assert in_node_value is not None
        in_value = in_node_value.value  # Shape (batch, in_features)

        grads = [
            self.weight.with_value(in_value.T @ grad.value),  # (batch, in_features).T @ (batch, out_features) -> (in_features, out_features)
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

    def forward(self, node_values: List[NodeValue]) -> NodeValue:
        in_value = node_values[0]

        out_value = 1.0 / (1.0 + np.exp(-in_value.value))
        return self.with_value(out_value)

    def backward(self, grad: NodeValue) -> List[NodeValue]:
        assert self._out_value is not None
        out_value = self._out_value.value

        function_grad = out_value * (1.0 - out_value)  # ds(x)/dx
        grads = [
            self.in_node.with_value(function_grad * grad.value)
        ]
        return grads


class ReLU(Operation):
    def __init__(self, input_node: Node):
        super().__init__(in_nodes=[input_node], parameters=[], shape=input_node.shape)
        self.in_node = input_node

    def forward(self, node_values: List[NodeValue]) -> NodeValue:
        in_value = node_values[0]

        out_value = np.clip(in_value.value, a_min=0.0, a_max=None)
        return self.with_value(out_value)

    def backward(self, grad: NodeValue) -> List[NodeValue]:
        in_node_value = self._in_node_values[self.in_node.id_]
        assert in_node_value is not None
        in_value = in_node_value.value

        in_grad = np.where(in_value >= 0.0, grad.value, 0.0)
        return [self.in_node.with_value(in_grad)]


def _log_sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0,
                    -np.log(1.0 + np.exp(-x)),
                    x - np.log(np.exp(x) + 1.0))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0, 
                    1.0 / (1.0 + np.exp(-x)),
                    np.exp(x) / (np.exp(x) + 1.0))


def _softmax(x: np.ndarray, axis: Optional[int]) -> np.ndarray:
    baseline = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x-baseline)
    return e / e.sum(axis=axis, keepdims=True)


def _log_softmax(x: np.ndarray, axis: Optional[int]) -> np.ndarray:
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

    def forward(self, node_values: List[NodeValue]) -> NodeValue:
        in_value = None
        target_value = None
        for node_value in node_values:
            if node_value.node.id_ == self.in_node.id_:
                in_value = node_value.value
            elif node_value.node.id_ == self.target_node.id_:
                target_value = node_value.value
        assert in_value is not None
        assert target_value is not None

        out_value = -target_value * _log_sigmoid(in_value) - (1.0-target_value)*_log_sigmoid(-in_value)
        if self.reduction == "mean":
            out_value = out_value.mean()
        elif self.reduction == "sum":
            out_value = out_value.sum()
        else:
            raise ValueError()

        return self.with_value(out_value)

    def backward(self, grad: NodeValue) -> List[NodeValue]:
        x = self._in_node_values[self.in_node.id_].value  # pyright: ignore
        target = self._in_node_values[self.target_node.id_].value  # pyright: ignore

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

    def forward(self, node_values: List[NodeValue]) -> NodeValue:
        in_value = None
        target_value = None
        for node_value in node_values:
            if node_value.node.id_ == self.in_node.id_:
                in_value = node_value.value
            elif node_value.node.id_ == self.target_node.id_:
                target_value = node_value.value
        assert in_value is not None
        assert target_value is not None

        out_value = -(target_value * _log_softmax(in_value, axis=-1)).sum(axis=-1)
        if self.reduction == "mean":
            out_value = out_value.mean()
        elif self.reduction == "sum":
            out_value = out_value.sum()
        else:
            raise ValueError()

        return self.with_value(out_value)

    def backward(self, grad: NodeValue) -> List[NodeValue]:
        x = self._in_node_values[self.in_node.id_].value  # pyright: ignore
        target = self._in_node_values[self.target_node.id_].value  # pyright: ignore
        target_sum = np.sum(target, axis=-1, keepdims=True)

        function_grad = target_sum * _softmax(x, axis=-1) - target  # Shape (batch, categories)
        in_grad = function_grad * grad.value  # Shape (batch, categories)*() = (batch, categories)
        if self.reduction == "mean":
            in_grad *= in_grad.shape[-1]
            in_grad /= in_grad.size

        assert in_grad.shape == x.shape

        return [self.in_node.with_value(in_grad)]

