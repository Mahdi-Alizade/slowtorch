from slowtorch.tensor import Tensor
from slowtorch.nn import Linear, MSELoss
from slowtorch.optim import SGD


def test_sgd_optimization_step():
    w = Tensor(5.0, requires_grad=True)
    # y = w * 2.0 -> dy/dw = 2.0
    y = w * 2.0
    y.backward()

    optimizer = SGD([w], lr=0.1)
    optimizer.step()

    # w_new = 5.0 - 0.1 * 2.0 = 4.8
    assert abs(w.data - 4.8) < 1e-6


def test_linear_regression_convergence():
    # Target function: y = 2 * x1 + 3 * x2 + 1
    # Train a single Linear layer for a few iterations
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