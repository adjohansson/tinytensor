import numpy as np

from tinytensor.core import Operation, Node, NodeValue, Parameter

from typing import Optional


class LinearCombination(Operation):
    """
    An operation that recieves multiple nodes of the same shape and
    linearly combines them with trainable coefficients.
    """

    def __init__(self, input_nodes: list[Node], coefficients: Optional[list[float]] = None, requires_grad: bool = True):
        fan_in = len(input_nodes)
        self.len = len(input_nodes)

        (fan_in, int)

        if coefficients is not None:
            if not len(coefficients) == len(input_nodes):
                raise ValueError(f"Number of coefficients did not match number of input nodes: coefficients={coefficients}, input_nodes={input_nodes}")
            
            self.coefficients = [
                Parameter(f"coefficient_{i}", np.array(c), requires_grad=requires_grad)
                for i, c in enumerate(coefficients)
            ]
        else:
            self.coefficients = [
                Parameter(f"coefficient_{i}", np.random.randn() / np.sqrt(fan_in), requires_grad=requires_grad)
                for i in range(self.len)
            ]

        shape = input_nodes[0].shape
        for node in input_nodes:
            assert node.shape == shape

        self.in_nodes = input_nodes

        self._in_values: Optional[list[np.ndarray]] = None
        super().__init__(in_nodes=self.in_nodes, parameters=self.coefficients, shape=shape)

    def _forward(self, node_values: dict[Node, NodeValue]) -> NodeValue:
        self._in_values = [node_values[node].value for node in self.in_nodes]

        out_value = sum(
            (tensor*coefficient.value for tensor, coefficient in zip(self._in_values, self.coefficients)),
            start=0.0
        )
        return self.with_value(out_value)

    def _backward(self, grad: NodeValue) -> list[NodeValue]:
        assert self._in_values is not None

        grads = []

        for coefficient, node, in_value in zip(self.coefficients, self.in_nodes, self._in_values):
            grads.append(
                coefficient.with_value(np.sum(in_value * grad.value))
            )
            if isinstance(node, Operation) and node._requires_grad or isinstance(node, Parameter) and node.requires_grad:
                grads.append(
                    node.with_value(coefficient.value*grad.value)
                )

        return grads

