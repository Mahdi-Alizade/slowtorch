from slowtorch.tensor import Tensor
from slowtorch.nn import MaxPool2d


def test_maxpool2d_forward():
    # 1 batch, 1 channel, 4x4 input
    # [[ 1,  2,  5,  6],
    #  [ 3,  4,  7,  8],
    #  [ 9, 10, 13, 14],
    #  [11, 12, 15, 16]]
    grid = [
        [1.0, 2.0, 5.0, 6.0],
        [3.0, 4.0, 7.0, 8.0],
        [9.0, 10.0, 13.0, 14.0],
        [11.0, 12.0, 15.0, 16.0]
    ]
    x = Tensor([[grid]], requires_grad=True)

    pool = MaxPool2d(kernel_size=2, stride=2)
    out = pool(x)

    assert out.shape == (1, 1, 2, 2)
    expected = [[[[4.0, 8.0], [12.0, 16.0]]]]
    assert out.data == expected


def test_maxpool2d_backward_gradient_routing():
    # Only max items receive gradient; others get 0
    grid = [
        [10.0, 2.0],
        [3.0, 4.0]
    ]
    x = Tensor([[grid]], requires_grad=True)

    pool = MaxPool2d(kernel_size=2)
    out = pool(x)

    loss = out.sum()
    loss.backward()

    # (0, 0) was the maximum (10.0), so it gets 1.0, all other cells get 0.0
    expected_grad = [[
        [1.0, 0.0],
        [0.0, 0.0]
    ]]
    assert x.grad[0] == expected_grad