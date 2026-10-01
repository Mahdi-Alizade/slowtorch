from slowtorch.tensor import Tensor
from slowtorch.nn import Conv2d


def test_conv2d_forward_shapes_and_values():
    # Batch=1, Channel=1, 3x3 input
    # [[1, 2, 3],
    #  [4, 5, 6],
    #  [7, 8, 9]]
    x_data = [[[[float(r * 3 + c + 1) for c in range(3)] for r in range(3)]]]
    x = Tensor(x_data, requires_grad=True)

    # 1 input channel, 1 output channel, 2x2 kernel, stride=1, padding=0, bias=False
    conv = Conv2d(in_channels=1, out_channels=1, kernel_size=2, stride=1, padding=0, bias=False)
    # Set weights to ones [[1, 1], [1, 1]]
    conv.weight.data = [[[[1.0, 1.0], [1.0, 1.0]]]]

    out = conv(x)

    # Out shape should be (1, 1, 2, 2)
    assert out.shape == (1, 1, 2, 2)

    # Window (0,0): 1 + 2 + 4 + 5 = 12
    # Window (0,1): 2 + 3 + 5 + 6 = 16
    # Window (1,0): 4 + 5 + 7 + 8 = 24
    # Window (1,1): 5 + 6 + 8 + 9 = 28
    expected = [[[[12.0, 16.0], [24.0, 28.0]]]]
    assert out.data == expected


def test_conv2d_backward_gradients():
    x_data = [[[[1.0, 2.0], [3.0, 4.0]]]]
    x = Tensor(x_data, requires_grad=True)

    conv = Conv2d(in_channels=1, out_channels=1, kernel_size=2, stride=1, padding=0, bias=True)
    conv.weight.data = [[[[0.5, 0.5], [0.5, 0.5]]]]
    conv.bias.data = [0.1]

    out = conv(x)
    # out = 0.5*(1+2+3+4) + 0.1 = 5.1
    assert abs(out.data[0][0][0][0] - 5.1) < 1e-5

    loss = out.sum()
    loss.backward()

    # dloss/dbias = 1.0
    assert abs(conv.bias.grad[0] - 1.0) < 1e-5

    # dloss/dweight = x
    assert conv.weight.grad[0][0] == [[1.0, 2.0], [3.0, 4.0]]

    # dloss/dx = weight
    assert x.grad[0][0] == [[0.5, 0.5], [0.5, 0.5]]


def test_conv2d_padding_and_stride():
    # Input 4x4, kernel 3x3, padding 1, stride 2 -> Output 2x2
    x_data = [[[[1.0 for _ in range(4)] for _ in range(4)]]]
    x = Tensor(x_data)

    conv = Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=2, padding=1)
    out = conv(x)

    assert out.shape == (1, 2, 2, 2)