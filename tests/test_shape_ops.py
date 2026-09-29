import pytest
from slowtorch.tensor import Tensor


def test_reshape_forward_and_backward():
    # 2x3 matrix reshaped to 3x2, and then 1D of length 6
    x = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)

    y = x.reshape(3, 2)
    assert y.shape == (3, 2)
    assert y.data == [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]

    # Reshape with -1 inference
    z = y.reshape(-1)
    assert z.shape == (6,)
    assert z.data == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    loss = z.sum()
    loss.backward()

    # Gradients should restore back into x's shape (2, 3) as ones
    assert x.grad == [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]


def test_reshape_invalid_dimensions():
    x = Tensor([[1.0, 2.0], [3.0, 4.0]])

    with pytest.raises(ValueError):
        _ = x.reshape(5)

    with pytest.raises(ValueError):
        _ = x.reshape(-1, -1)


def test_transpose_and_property_T_backward():
    # A is 2x3
    a = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    at = a.T

    assert at.shape == (3, 2)
    assert at.data == [
        [1.0, 4.0],
        [2.0, 5.0],
        [3.0, 6.0]
    ]

    # Weighted sum
    multiplier = Tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    loss = (at * multiplier).sum()
    loss.backward()

    # dloss/dat = multiplier
    # dloss/da = multiplier.T = [[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]]
    expected_grad = [[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]]
    assert a.grad == expected_grad