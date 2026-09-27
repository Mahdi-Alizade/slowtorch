import random
from slowtorch.tensor import Tensor
from slowtorch.nn import Module, Linear, ReLU, Sigmoid


def test_linear_layer_forward_shapes_and_parameters():
    random.seed(42)
    linear = Linear(in_features=4, out_features=2, bias=True)

    params = linear.parameters()
    assert len(params) == 2
    assert linear.weight.shape == (4, 2)
    assert linear.bias.shape == (2,)

    batch_input = [
        [1.0, 2.0, 3.0, 4.0],
        [0.5, 1.5, 2.5, 3.5],
        [0.0, 1.0, 0.0, 1.0]
    ]
    x = Tensor(batch_input)
    y = linear(x)

    assert y.shape == (3, 2)


def test_linear_layer_backward_and_zero_grad():
    linear = Linear(in_features=2, out_features=2, bias=True)
    linear.weight.data = [
        [1.0, 2.0],
        [3.0, 4.0]
    ]
    linear.bias.data = [0.1, 0.2]

    x = Tensor([[1.0, 1.0], [2.0, 2.0]], requires_grad=False)
    out = linear(x)

    assert abs(out.data[0][0] - 4.1) < 1e-6
    assert abs(out.data[0][1] - 6.2) < 1e-6
    assert abs(out.data[1][0] - 8.1) < 1e-6
    assert abs(out.data[1][1] - 12.2) < 1e-6

    loss = out.sum()
    loss.backward()

    assert linear.bias.grad == [2.0, 2.0]
    assert linear.weight.grad == [[3.0, 3.0], [3.0, 3.0]]

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
    assert len(model.parameters()) == 4

    x = Tensor([[1.0, 2.0]])
    res = model(x)
    assert res.shape == (1, 1)


def test_mlp_with_activations_forward_and_backward():
    class NonLinearMLP(Module):
        def __init__(self):
            super().__init__()
            self.fc1 = Linear(2, 2)
            self.act1 = ReLU()
            self.fc2 = Linear(2, 1)
            self.act2 = Sigmoid()

        def forward(self, x):
            z1 = self.fc1(x)
            a1 = self.act1(z1)
            z2 = self.fc2(a1)
            a2 = self.act2(z2)
            return a2

    model = NonLinearMLP()
    x = Tensor([[1.0, -1.0]])
    pred = model(x)

    # Sigmoid output must always be strictly between 0.0 and 1.0
    assert 0.0 < pred.data[0][0] < 1.0

    # Test backpropagation flow across all layers
    loss = pred.sum()
    loss.backward()

    # Gradients should have reached fc1 weight and bias
    assert model.fc1.weight.grad is not None
    assert model.fc1.bias.grad is not None