# D:\Mahdi Alizade\Projects\slowtorch\tests\test_tensor.py

from slowtorch.tensor import Tensor


def test_scalar_addition_and_multiplication_backward():
    # Equation: y = a * b + c
    # dy/da = b, dy/db = a, dy/dc = 1.0
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