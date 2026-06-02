from tinytensor.core import DataNode
from tinytensor.layers import Linear, ReLU, CrossEntropyLoss
import numpy as np


def main():
    # Construct the graph

    # Input data source
    input_ = DataNode(name="input", shape=(None, 28*28))

    # Network graph
    linear1 = Linear(input_, out_features=32)
    relu1 = ReLU(linear1)
    linear2 = Linear(relu1, out_features=32)
    relu2 = ReLU(linear2)
    logits = Linear(relu2, out_features=10)

    # Target data source
    target = DataNode(name="target", shape=(None, 10))

    # Loss node
    loss = CrossEntropyLoss(input_node=logits, target_node=target)

    # Supply values
    x = np.random.randn(16, 28 * 28)
    y = np.random.random(size=(16, 10))

    # Compute logits
    logits_value = logits(input_.with_value(x))
    print("logits:", logits_value.value)

    # Compute loss using the previously computed logits
    loss_value = loss(logits_value, target.with_value(y))
    print("loss:", loss_value.value)

    # Do the backward pass
    loss.backward_pass()

    for param in loss.parameters():
        print(param, param.grad)


if __name__ == '__main__':
    main()

