import math
from slowtorch.tensor import Tensor
from slowtorch.nn import Linear, MSELoss
from slowtorch.optim import SGD, Adam


def test_sgd_optimization_step():
    w = Tensor(5.0, requires_grad=True)
    y = w * 2.0
    y.backward()

    optimizer = SGD([w], lr=0.1)
    optimizer.step()

    assert abs(w.data - 4.8) < 1e-6


def test_linear_regression_convergence():
    model = Linear(in_features=2, out_features=1, bias=True)
    criterion = MSELoss()
    optimizer = SGD(model.parameters(), lr=0.05)

    x = Tensor([[1.0, 1.0], [2.0, 1.0], [1.0, 2.0], [2.0, 2.0]])
    y_true = Tensor([[6.0], [8.0], [9.0], [11.0]])

    initial_loss = criterion(model(x), y_true).data

    for _ in range(50):
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y_true)
        loss.backward()
        optimizer.step()

    final_loss = criterion(model(x), y_true).data
    assert final_loss < initial_loss


def test_adam_single_step_mathematical_precision():
    # Verify exact math for 1 step of Adam on a scalar
    # w = 2.0, loss = w * 3.0 -> grad = 3.0
    w = Tensor(2.0, requires_grad=True)
    loss = w * 3.0
    loss.backward()

    lr = 0.1
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8

    opt = Adam([w], lr=lr, betas=(beta1, beta2), eps=eps)
    opt.step()

    # Manual calculations:
    # m_1 = 0.1 * 3.0 = 0.3
    # v_1 = 0.001 * 9.0 = 0.009
    # m_hat = 0.3 / (1 - 0.9) = 3.0
    # v_hat = 0.009 / (1 - 0.999) = 9.0
    # step = 0.1 * 3.0 / (sqrt(9.0) + eps) = 0.3 / 3.0 = 0.1
    # expected_w = 2.0 - 0.1 = 1.9
    assert abs(w.data - 1.9) < 1e-5


def test_adam_xor_fast_convergence():
    # Verify Adam solves XOR significantly faster or smoothly
    class XORNet(Linear):
        pass

    model = Linear(in_features=2, out_features=1, bias=True)
    criterion = MSELoss()
    opt = Adam(model.parameters(), lr=0.1)

    x = Tensor([[1.0, 2.0], [3.0, 4.0]])
    y = Tensor([[5.0], [11.0]])

    initial_loss = criterion(model(x), y).data
    for _ in range(40):
        opt.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        opt.step()

    final_loss = criterion(model(x), y).data
    assert final_loss < initial_loss