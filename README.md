# Tinytensor — a minimal autograd library with an explicit computation graph

Tinytensor is a small experimental automatic differentiation library written in Python.  
It is conceptually similar to PyTorch, but uses a **predefined computation graph** rather than dynamic graph construction.

This project was built as a learning exercise and a hobby project, with the goals of:
- gaining a deeper understanding of how autograd engines work internally,
- experimenting with alternative API designs,
- practicing clean Python packaging and unit testing.

Note that he code is not optimized for performance and is **not intended for production use**.  
Correctness is validated via unit tests, but I make no guarantees beyond that.

## Basic design
- The full computation graph is defined up-front using static `Node` objects.
- The inputs to the computation, and results of computations are represented as `NodeValue` objects.
- Forward passes are very flexible; the user can request the value at any given node, and to carry out the computation, the user can provide any set of upstream `NodeValues` that are enough to uniquely determine the result.
- Gradients are computed by initiating a backward pass at the terminal loss node.

## Example usage
### Defining the computation graph

The user first defines the computation graph by wiring together layers and data nodes:
```python
from tinytensor.layers import Linear, ReLU, CrossEntropyLoss, DataNode

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
```
At this stage, no computation is performed. This step only defines the structure of the computation graph.

### Forward passes with concrete values

To evaluate the graph, concrete values must be provided for all required upstream `DataNode`s.
This is done using `node.with_value(...)`, which returns a `NodeValue` instance.
```python
import numpy as np

x = np.random.randn(16, 28 * 28)
y = np.random.random(size=(16, 10))

# Compute logits
logits_value = logits(input_.with_value(x))
print("logits:", logits_value.value)

# Compute loss using the previously computed logits
loss_value = loss(logits_value, target.with_value(y))
print("loss:", loss_value.value)
```
A key design choice is that `NodeValue`s can be reused:
- Intermediate results (such as `logits_value`) do not need to be recomputed when used in downstream nodes.
- Fixed data (e.g. `targets`) can be mixed with previously computed node values.

### Backward pass and gradients
Once a loss has been evaluated, gradients can be computed via a backward pass:
```python
loss.backward_pass()

for param in loss.parameters():
    print(param, param.grad)
```

