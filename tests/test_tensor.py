import pytest
import math
from slowtorch.tensor import Tensor


def test_scalar_addition_and_multiplication_backward():
    a = Tensor(2.0, requires_grad=True)
    b = Tensor(3.0, requires_grad=True)
    c = Tensor(4.0, requires_grad=True)

    prod = a * b
    y = prod + c

    y.backward()

    assert y.data == 10.0
    assert a.grad == 3.0
    assert b.grad == 2.0
    assert c.grad == 1.0


def test_vector_addition_and_multiplication_backward():
    v1 = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    v2 = Tensor([4.0, 5.0, 6.0], requires_grad=True)

    out = v1 * v2
    out.backward()

    assert out.data == [4.0, 10.0, 18.0]
    assert v1.grad == [4.0, 5.0, 6.0]
    assert v2.grad == [1.0, 2.0, 3.0]


def test_matrix_multiplication_forward_and_backward():
    mat_a = [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0]
    ]
    mat_b = [
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0]
    ]

    a = Tensor(mat_a, requires_grad=True)
    b = Tensor(mat_b, requires_grad=True)

    c = a @ b

    assert c.shape == (2, 2)
    assert c.data == [[22.0, 28.0], [49.0, 64.0]]

    loss = c.sum()
    loss.backward()

    assert a.grad == [[3.0, 7.0, 11.0], [3.0, 7.0, 11.0]]
    assert b.grad == [[5.0, 5.0], [7.0, 7.0], [9.0, 9.0]]


def test_matrix_multiplication_dimension_mismatch():
    a = Tensor([[1.0, 2.0], [3.0, 4.0]])
    b = Tensor([[1.0, 2.0, 3.0]])

    with pytest.raises(ValueError):
        _ = a @ b


def test_relu_forward_and_backward():
    # 2D matrix with both positive and negative values
    x = Tensor([[-2.0, 3.0], [0.0, -1.0]], requires_grad=True)
    y = x.relu()

    assert y.data == [[0.0, 3.0], [0.0, 0.0]]

    loss = y.sum()
    loss.backward()

    # Gradient should be 1.0 for x > 0 and 0.0 for x <= 0
    assert x.grad == [[0.0, 1.0], [0.0, 0.0]]


def test_sigmoid_forward_and_backward():
    # Test at 0.0 where sigmoid(0) = 0.5 and derivative is 0.5 * (1 - 0.5) = 0.25
    x = Tensor(0.0, requires_grad=True)
    y = x.sigmoid()

    assert abs(y.data - 0.5) < 1e-6

    y.backward()
    assert abs(x.grad - 0.25) < 1e-6