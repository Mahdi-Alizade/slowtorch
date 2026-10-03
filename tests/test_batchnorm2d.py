from slowtorch.tensor import Tensor
from slowtorch.nn import BatchNorm2d


def test_batchnorm2d_train_zero_mean_unit_variance():
    # Batch size 2, Channels 1, Spatial 2x2 -> M = 8 elements
    x_data = [
        [[[1.0, 2.0], [3.0, 4.0]]],
        [[[5.0, 6.0], [7.0, 8.0]]]
    ]
    x = Tensor(x_data, requires_grad=True)

    bn = BatchNorm2d(num_features=1, eps=1e-5)
    bn.train()

    out = bn(x)
    assert out.shape == (2, 1, 2, 2)

    # Calculate overall mean and variance across (N, H, W) for Channel 0
    all_vals = []
    for n in range(2):
        for h in range(2):
            for w in range(2):
                all_vals.append(out.data[n][0][h][w])

    mean_c = sum(all_vals) / len(all_vals)
    var_c = sum((v - mean_c) ** 2 for v in all_vals) / len(all_vals)

    assert abs(mean_c) < 1e-4
    assert abs(var_c - 1.0) < 1e-3


def test_batchnorm2d_running_stats_update():
    bn = BatchNorm2d(num_features=1, momentum=0.1)
    bn.train()

    # Initial running stats
    assert bn.running_mean == [0.0]
    assert bn.running_var == [1.0]

    # Input with mean 10.0
    x_data = [[[[10.0, 10.0], [10.0, 10.0]]]]
    x = Tensor(x_data)

    _ = bn(x)

    # New running mean = (1 - 0.1)*0.0 + 0.1*10.0 = 1.0
    assert abs(bn.running_mean[0] - 1.0) < 1e-5


def test_batchnorm2d_eval_mode():
    bn = BatchNorm2d(num_features=1)
    bn.running_mean = [5.0]
    bn.running_var = [4.0]  # std = 2.0
    bn.eval()

    # Input 9.0 -> (9.0 - 5.0) / sqrt(4.0) = 2.0
    x = Tensor([[[[9.0]]]], requires_grad=True)
    out = bn(x)

    assert abs(out.data[0][0][0][0] - 2.0) < 1e-4

    loss = out.sum()
    loss.backward()

    # dL/dx = 1.0 * gamma * inv_std = 1.0 * 1.0 * 0.5 = 0.5
    assert abs(x.grad[0][0][0][0] - 0.5) < 1e-4


def test_batchnorm2d_backward_gradients_sum_to_zero():
    # Analytically, for any batchnorm forward under uniform gradient, sum of dx over batch is 0
    x_data = [
        [[[1.0, 2.0], [3.0, 4.0]]],
        [[[5.0, 6.0], [7.0, 8.0]]]
    ]
    x = Tensor(x_data, requires_grad=True)

    bn = BatchNorm2d(num_features=1)
    bn.train()

    out = bn(x)
    loss = out.sum()
    loss.backward()

    total_dx = 0.0
    for n in range(2):
        for h in range(2):
            for w in range(2):
                total_dx = total_dx + x.grad[n][0][h][w]

    assert abs(total_dx) < 1e-5
    assert bn.weight.grad is not None
    assert bn.bias.grad is not None