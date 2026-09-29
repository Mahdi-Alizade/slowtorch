from slowtorch.tensor import Tensor
from slowtorch.nn import LayerNorm


def test_layernorm_forward_zero_mean_unit_variance():
    # Batch size 2, Features 4
    x_data = [
        [1.0, 2.0, 3.0, 4.0],
        [10.0, 20.0, 30.0, 40.0]
    ]
    x = Tensor(x_data, requires_grad=True)

    ln = LayerNorm(4, eps=1e-5)
    out = ln(x)

    assert out.shape == (2, 4)

    # For each row, mean must be ~0 and variance ~1
    for r in range(2):
        row = out.data[r]
        mean_val = sum(row) / 4.0
        var_val = sum((val - mean_val) ** 2 for val in row) / 4.0

        assert abs(mean_val) < 1e-4
        assert abs(var_val - 1.0) < 1e-3


def test_layernorm_affine_transform():
    x = Tensor([[1.0, 2.0, 3.0]], requires_grad=False)
    ln = LayerNorm(3, eps=1e-5)

    # Set custom gamma (2.0) and beta (0.5)
    ln.weight.data = [2.0, 2.0, 2.0]
    ln.bias.data = [0.5, 0.5, 0.5]

    out = ln(x)
    # Normalized x for [1, 2, 3] is [-1.2247, 0.0, 1.2247]
    # Scaled by 2 and shifted by 0.5 -> [-1.949, 0.5, 2.949]
    assert abs(out.data[0][1] - 0.5) < 1e-4


def test_layernorm_backward_gradients():
    x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    ln = LayerNorm(2, eps=1e-5)

    out = ln(x)
    loss = out.sum()
    loss.backward()

    # Weight and bias gradients must be populated
    assert ln.weight.grad is not None
    assert ln.bias.grad is not None
    # For a sum loss on normalized rows, sum of dL/dx for each row is analytically 0
    row0_grad_sum = sum(x.grad[0])
    row1_grad_sum = sum(x.grad[1])
    assert abs(row0_grad_sum) < 1e-5
    assert abs(row1_grad_sum) < 1e-5