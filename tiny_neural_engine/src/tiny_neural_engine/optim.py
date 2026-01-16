from abc import ABC, abstractmethod

from .layers import Parameter

from typing import List


class Optimizer(ABC):
    def __init__(self, parameters: List[Parameter]):
        self.parameters = parameters

    def zero_grad(self):
        for param in self.parameters:
            param.grad = None

    @abstractmethod
    def step(self):
        pass
                

class SGD(Optimizer):
    def __init__(self, parameters: List[Parameter], lr: float):
        super().__init__(parameters)
        self.lr = lr

    def step(self):
        for param in self.parameters:
            if param.grad is not None:
                param.set(
                    param.value - self.lr*param.grad
                )


