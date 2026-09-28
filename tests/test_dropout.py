import random
from slowtorch.tensor import Tensor
from slowtorch.nn import Dropout, Sequential, Linear


def test_dropout_eval_mode():
    drop = Dropout(p=0.5)
    drop.eval()

    x = Tensor([[1.0, 2.0], [3.0, 4.0]])
    out = drop(x)

    # In eval mode, output must be identical to input
    assert out.data == x.data


def test_dropout_train_mode_scaling_and_masking():
    random.seed(42)
    p = 0.5
    drop = Dropout(p=p)
    drop.train()

    # Large vector to statistically verify zeroing and scaling
    data = [[10.0] * 100]
    x = Tensor(data, requires_grad=True)
    out = drop(x)

    # Inverted dropout scales remaining elements by 1 / (1 - p) = 2.0
    # So surviving values must be 20.0 and dropped ones must be 0.0
    has_zeros = False
    has_scaled = False

    for val in out.data[0]:
        if val == 0.0:
            has_zeros = True
        elif abs(val - 20.0) < 1e-5:
            has_scaled = True

    assert has_zeros
    assert has_scaled

    loss = out.sum()
    loss.backward()

    # Gradients must match the mask: 2.0 where active, 0.0 where dropped
    for i in range(100):
        if out.data[0][i] == 0.0:
            assert x.grad[0][i] == 0.0
        else:
            assert abs(x.grad[0][i] - 2.0) < 1e-5


def test_module_recursive_train_eval_toggle():
    model = Sequential(
        Linear(2, 4),
        Dropout(0.3),
        Linear(4, 1)
    )

    assert model.training is True
    assert model[1].training is True

    model.eval()
    assert model.training is False
    assert model[0].training is False
    assert model[1].training is False

    model.train()
    assert model.training is True
    assert model[1].training is True