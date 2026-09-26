# D:\Mahdi Alizade\Projects\slowtorch\tests\test_tensor.py

import pytest
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
    # A is 2x3, B is 3x2
    # C = A @ B (2x2)
    # loss = sum(C)
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
    # row0: [1*1 + 2*3 + 3*5 = 22, 1*2 + 2*4 + 3*6 = 28]
    # row1: [4*1 + 5*3 + 6*5 = 49, 4*2 + 5*4 + 6*6 = 64]
    assert c.data == [[22.0, 28.0], [49.0, 64.0]]

    loss = c.sum()
    loss.backward()

    # dloss/dc is all 1.0
    # dloss/da = dloss/dc @ b.T
    # b.T is:
    # [1.0, 3.0, 5.0]
    # [2.0, 4.0, 6.0]
    # dloss/dc is [[1, 1], [1, 1]]
    # row0 of grad_a = [1*1 + 1*2, 1*3 + 1*4, 1*5 + 1*6] = [3.0, 7.0, 11.0]
    # row1 of grad_a = [3.0, 7.0, 11.0]
    assert a.grad == [[3.0, 7.0, 11.0], [3.0, 7.0, 11.0]]

    # dloss/db = a.T @ dloss/dc
    # a.T is:
    # [1.0, 4.0]
    # [2.0, 5.0]
    # [3.0, 6.0]
    # row0 of grad_b = [1*1 + 4*1, 1*1 + 4*1] = [5.0, 5.0]
    # row1 of grad_b = [2*1 + 5*1, 2*1 + 5*1] = [7.0, 7.0]
    # row2 of grad_b = [3*1 + 6*1, 3*1 + 6*1] = [9.0, 9.0]
    assert b.grad == [[5.0, 5.0], [7.0, 7.0], [9.0, 9.0]]


def test_matrix_multiplication_dimension_mismatch():
    a = Tensor([[1.0, 2.0], [3.0, 4.0]])
    b = Tensor([[1.0, 2.0, 3.0]])

    with pytest.raises(ValueError):
        _ = a @ b