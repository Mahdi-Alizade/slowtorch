# D:\Mahdi Alizade\Projects\slowtorch\tests\test_nn.py

import random
from slowtorch.tensor import Tensor
from slowtorch.nn import Module, Linear


def test_linear_layer_forward_shapes_and_parameters():
    random.seed(42)
    # Batch size: 3, Input features: 4, Output features: 2
    linear = Linear(in_features=4, out_features=2, bias=True)

    params = linear.parameters()
    assert len(params) == 2
    assert linear.weight.shape == (4, 2)
    assert linear.bias.shape == (2,)

    # Input: 3x4
    batch_input = [
        [1.0, 2.0, 3.0, 4.0],
        [0.5, 1.5, 2.5, 3.5],
        [0.0, 1.0, 0.0, 1.0]
    ]
    x = Tensor(batch_input)
    y = linear(x)

    assert y.shape == (3, 2)


def test_linear_layer_backward_and_zero_grad():
    # Deterministic weights and bias for precise mathematical assertion
    linear = Linear(in_features=2, out_features=2, bias=True)
    linear.weight.data = [
        [1.0, 2.0],
        [3.0, 4.0]
    ]
    linear.bias.data = [0.1, 0.2]

    # Input batch size: 2
    x = Tensor([[1.0, 1.0], [2.0, 2.0]], requires_grad=False)

    out = linear(x)
    # out = x @ W + b
    # row0: [1*1 + 1*3 + 0.1 = 4.1, 1*2 + 1*4 + 0.2 = 6.2]
    # row1: [2*1 + 2*3 + 0.1 = 8.1, 2*2 + 2*4 + 0.2 = 12.2]
    assert abs(out.data[0][0] - 4.1) < 1e-6
    assert abs(out.data[0][1] - 6.2) < 1e-6
    assert abs(out.data[1][0] - 8.1) < 1e-6
    assert abs(out.data[1][1] - 12.2) < 1e-6

    loss = out.sum()
    loss.backward()

    # dloss/dout is 1 everywhere (2x2)
    # dloss/db = sum along batch rows = [1+1, 1+1] = [2.0, 2.0]
    assert linear.bias.grad == [2.0, 2.0]

    # dloss/dW = x.T @ dloss/dout
    # x.T is [[1, 2], [1, 2]]
    # [[1*1 + 2*1, 1*1 + 2*1], [1*1 + 2*1, 1*1 + 2*1]] = [[3.0, 3.0], [3.0, 3.0]]
    assert linear.weight.grad == [[3.0, 3.0], [3.0, 3.0]]

    # Test zero_grad functionality
    linear.zero_grad()
    assert linear.weight.grad == [[0.0, 0.0], [0.0, 0.0]]
    assert linear.bias.grad == [0.0, 0.0]


def test_custom_module_hierarchy():
    class SimpleMLP(Module):
        def __init__(self):
            super().__init__()
            self.fc1 = Linear(2, 3)
            self.fc2 = Linear(3, 1)

        def forward(self, x):
            h = self.fc1(x)
            out = self.fc2(h)
            return out

    model = SimpleMLP()
    # 2 parameters from fc1 (weight, bias) and 2 from fc2
    assert len(model.parameters()) == 4

    x = Tensor([[1.0, 2.0]])
    res = model(x)
    assert res.shape == (1, 1)