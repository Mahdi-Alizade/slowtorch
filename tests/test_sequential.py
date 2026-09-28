from slowtorch.tensor import Tensor
from slowtorch.nn import Sequential, Linear, ReLU, Sigmoid


def test_sequential_forward_and_len():
    model = Sequential(
        Linear(2, 4),
        ReLU(),
        Linear(4, 1),
        Sigmoid()
    )

    assert len(model) == 4
    assert isinstance(model[0], Linear)
    assert isinstance(model[1], ReLU)

    x = Tensor([[1.0, 2.0], [-1.0, 0.5]])
    out = model(x)

    assert out.shape == (2, 1)
    # Output of Sigmoid is strictly within (0, 1)
    for val in out.data:
        assert 0.0 < val[0] < 1.0


def test_sequential_parameters_and_state_dict():
    model = Sequential(
        Linear(3, 2, bias=True),
        Linear(2, 1, bias=False)
    )

    # 2 params from first layer (weight, bias), 1 from second layer (weight)
    assert len(model.parameters()) == 3

    state = model.state_dict()
    assert "0.weight" in state
    assert "0.bias" in state
    assert "1.weight" in state


def test_sequential_backward():
    model = Sequential(
        Linear(2, 2, bias=True),
        ReLU()
    )

    x = Tensor([[1.0, 1.0]])
    out = model(x)
    loss = out.sum()
    loss.backward()

    # Gradients should reach layer 0 weights and bias
    assert model[0].weight.grad is not None
    assert model[0].bias.grad is not None