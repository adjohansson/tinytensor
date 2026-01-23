import numpy as np

from abc import abstractmethod
from dataclasses import dataclass

from typing import Set, Tuple, List, Dict, Optional, Any


def _shapes_agree(shape1: Tuple[Optional[int], ...], shape2: Tuple[Optional[int]]) -> bool:
    if shape1 == shape2:
        return True

    for s, v in zip(shape1, shape2):
        if s is None or v is None:
            continue
        if s != v:
            return False
    return True


class EmptyDataError(ValueError):
    pass


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

    def with_value(self, value: np.ndarray) -> 'NodeValue':
        """
        State that the value of this node is 'value'. Does modify the node internally.
        """
        if not _shapes_agree(self._shape, value.shape):
            raise ValueError(f"Cannot assign value of shape {value.shape} to node of shape {self._shape}.")
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
        if not _shapes_agree(self.shape, value.shape):
            raise ValueError(f"Cannot assign value of shape {value.shape} to parameter of shape {self._shape}.")
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
        self._in_nodes = in_nodes
        self._parameters = parameters
        self._out_shape: Optional[Tuple[int, ...]] = None

        self._all_parameters = self._seek_parameters()
        self._requires_grad: Optional[bool] = None

    @abstractmethod
    def _forward(self, node_values: Dict[Node, NodeValue], *args: Any, **kwargs: Any) -> NodeValue:
        """Recieves a specification of the values at all input nodes. Should compute the value of the operations's output node."""

    @abstractmethod
    def _backward(self, grad: NodeValue) -> List[NodeValue]:
        """
        Recieves a specification of the loss gradient at the output node (or it is assumed to be scalar 1.0 if grad=None).
        The function should then compute the corresponding loss gradient at the input nodes.
        """

    def __call__(self, *node_values: NodeValue) -> NodeValue:
        return self._forward_pass({node_value.node: node_value for node_value in node_values})

    def _forward_pass(self, values: Optional[Dict[Node, NodeValue]] = None) -> NodeValue:
        """
        Recursively retrieves the output values from previous nodes in the graph and returns the output value
        """
        self._requires_grad = any(param.requires_grad for param in self._parameters)
        if values is None:
            values = {}

        in_node_values = {}
        for node in self._in_nodes:

            # If this value is cached, we return it
            if node in values:
                value = values[node]
            
            # If it is a DataNode, it needs to be cached already, 
            # otherwise there is no source of truth
            elif isinstance(node, DataNode):
                raise EmptyDataError(f"DataNode {node} was not filled during forward pass.")

            # If it is an operation, we recursively compute its value from the parents in the graph
            elif isinstance(node, Operation):
                value = node._forward_pass(values)
                self._requires_grad = self._requires_grad or node._requires_grad
            else:
                raise ValueError()

            # Register the value at the node
            in_node_values[node] = value

        # Use the input notes' values to compute the output of this node
        out_value = self._forward(in_node_values)
        self._out_shape = out_value.value.shape
        return out_value

    def backward_pass(self, grad: Optional[NodeValue] = None):
        assert self._out_shape is not None
        if grad is None:
            grad = self.with_value(np.ones(self._out_shape))
        input_grads = self._backward(grad)
        for grad_value in input_grads:
            input_node = grad_value.node
            if isinstance(input_node, Operation):
                assert input_node._requires_grad is not None
                if input_node._requires_grad:
                    input_node.backward_pass(grad_value)
            elif isinstance(input_node, Parameter) and input_node.requires_grad:
                input_node.add_grad(grad=grad_value.value)

    def _seek_parameters(self) -> Set[Parameter]:
        parameters = set(self._parameters)
        for in_node in self._in_nodes:
            if isinstance(in_node, Parameter):
                parameters |= {in_node}
            elif isinstance(in_node, Operation):
                parameters |= in_node._all_parameters

        return parameters

    def parameters(self) -> List[Parameter]:
        return list(self._all_parameters)

    def zero_grad(self):
        for parameter in self._all_parameters:
            parameter.grad = None


class Linear(Operation):
    def __init__(self, input_node: Node, out_features: int):
        fan_in = input_node.shape[1]
        assert isinstance(fan_in, int)
        self.weight = Parameter("weight", np.random.randn(fan_in, out_features) / np.sqrt(fan_in), requires_grad=True)
        self.bias = Parameter("bias", np.random.randn(out_features)*0.01, requires_grad=True)
        self.in_node = input_node

        self._in_value: Optional[np.ndarray] = None
        super().__init__(in_nodes=[input_node], parameters=[self.weight, self.bias], shape=(None, out_features))

    def _forward(self, node_values: Dict[Node, NodeValue]) -> NodeValue:
        self._in_value = node_values[self.in_node].value

        out_value = self._in_value @ self.weight.value + self.bias.value
        return self.with_value(out_value)

    def _backward(self, grad: NodeValue) -> List[NodeValue]:
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

    def _forward(self, node_values: Dict[Node, NodeValue]) -> NodeValue:
        in_value = node_values[self.in_node]

        self._out_value = 1.0 / (1.0 + np.exp(-in_value.value))
        return self.with_value(self._out_value)

    def _backward(self, grad: NodeValue) -> List[NodeValue]:
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

    def _forward(self, node_values: Dict[Node, NodeValue]) -> NodeValue:
        self._in_value = node_values[self.in_node].value

        out_value = np.clip(self._in_value, a_min=0.0, a_max=None)
        return self.with_value(out_value)

    def _backward(self, grad: NodeValue) -> List[NodeValue]:
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

    def _forward(self, node_values: Dict[Node, NodeValue]) -> NodeValue:
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

    def _backward(self, grad: NodeValue) -> List[NodeValue]:
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

    def _forward(self, node_values: Dict[Node, NodeValue]) -> NodeValue:
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

    def _backward(self, grad: NodeValue) -> List[NodeValue]:
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

