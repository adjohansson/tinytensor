"""
This module defines the core graph computation objects.
"""

import numpy as np

from abc import abstractmethod
from dataclasses import dataclass

from typing import Optional, Any


def _shapes_agree(shape1: tuple[Optional[int], ...], shape2: tuple[Optional[int]]) -> bool:
    if shape1 == shape2:
        return True
    if len(shape1) != len(shape2):
        return False

    for s, v in zip(shape1, shape2):
        if s is None or v is None:
            continue
        if s != v:
            return False
    return True


class EmptyDataError(ValueError):
    pass


class ShapeError(ValueError):
    pass


class Node:
    """
    Represents a point in the computation graph of a given module.

    Each step in a computation graph is assigned a Node with a unique id_ and a static shape.
    As results are computed for the node, these are returned as 'NodeValue' instances.
    """
    _N_NODES: int = 0  # Number of created nodes in total

    def __init__(self, shape: tuple[Optional[int], ...]):
        self._id = Node._N_NODES
        Node._N_NODES += 1
        self._shape = shape

    def with_value(self, value: np.ndarray) -> 'NodeValue':
        """
        Create object representing a value 'value' stored at this node. This operation does not mutate the underlying node.
        """
        if not _shapes_agree(self._shape, value.shape):
            raise ShapeError(f"Cannot assign value of shape {value.shape} to node of shape {self._shape}.")
        return NodeValue(self, value)

    @property
    def shape(self) -> tuple[Optional[int], ...]:
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
        """
        Set the value contained in the parameter.
        """
        if not _shapes_agree(self.shape, value.shape):
            raise ShapeError(f"Cannot assign value of shape {value.shape} to parameter of shape {self._shape}.")
        self.value = value

    def _add_grad(self, grad: np.ndarray):
        """
        Aggregate gradient 'grad' onto the Parameter.
        """
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

    def __init__(self, name: str, shape: tuple[Optional[int], ...]):
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
    """
    A node which is a differentiable function of the values at a number of input nodes.
    Can hold trainable parameters.
    """

    def __init__(self, in_nodes: list[Node], parameters: list[Parameter], shape: tuple[Optional[int], ...]):
        super().__init__(shape=shape)
        self._in_nodes = in_nodes
        self._parameters = parameters
        self._out_shape: Optional[tuple[int, ...]] = None

        self._all_parameters = self._seek_parameters()
        self._requires_grad: Optional[bool] = None

    @abstractmethod
    def _forward(self, node_values: dict[Node, NodeValue], *args: Any, **kwargs: Any) -> NodeValue:
        """Recieves a specification of the values at all input nodes. Should compute the value of the operations's output node."""

    @abstractmethod
    def _backward(self, grad: NodeValue) -> list[NodeValue]:
        """
        Recieves a specification of the loss gradient at the output node (or it is assumed to be scalar 1.0 if grad=None).
        The function should then compute the corresponding loss gradient at the input nodes.
        """

    def __call__(self, *node_values: NodeValue) -> NodeValue:
        return self._forward_pass({node_value.node: node_value for node_value in node_values})

    def _forward_pass(self, values: Optional[dict[Node, NodeValue]] = None) -> NodeValue:
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
                raise RuntimeError(f"Internal logic error: unrecognized node type {type(node)}")

            # Register the value at the node
            in_node_values[node] = value

        # Use the input notes' values to compute the output of this node
        out_value = self._forward(in_node_values)
        self._out_shape = out_value.value.shape
        return out_value

    def backward_pass(self, grad: Optional[NodeValue] = None):
        """
        Run a backpropagation backward pass with this node as the final node.
        If the parameter 'grad' is passed, it is used as the output gradient, otherwise an output gradient of 1.0 is assumed.
        """
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
                input_node._add_grad(grad=grad_value.value)

    def _seek_parameters(self) -> set[Parameter]:
        parameters = set(self._parameters)
        for in_node in self._in_nodes:
            if isinstance(in_node, Parameter):
                parameters |= {in_node}
            elif isinstance(in_node, Operation):
                parameters |= in_node._all_parameters

        return parameters

    def parameters(self) -> list[Parameter]:
        """
        Enumerate all parameters upstream of this Operation.
        """
        return list(self._all_parameters)

    def zero_grad(self):
        """
        Zero the gradients of all parameters upstream of this Operation.
        """
        for parameter in self._all_parameters:
            parameter.grad = None

