from slowtorch.tensor import Tensor
from slowtorch.optim import SGD, StepLR


def test_steplr_schedule_decay():
    w = Tensor(1.0, requires_grad=True)
    optimizer = SGD([w], lr=0.1)

    # Decay by 0.5 every 2 epochs
    scheduler = StepLR(optimizer, step_size=2, gamma=0.5)

    assert abs(optimizer.lr - 0.1) < 1e-6

    # Epoch 1
    scheduler.step()
    assert abs(optimizer.lr - 0.1) < 1e-6

    # Epoch 2 -> decay applies (0.1 * 0.5 = 0.05)
    scheduler.step()
    assert abs(optimizer.lr - 0.05) < 1e-6

    # Epoch 3
    scheduler.step()
    assert abs(optimizer.lr - 0.05) < 1e-6

    # Epoch 4 -> decay applies again (0.05 * 0.5 = 0.025)
    scheduler.step()
    assert abs(optimizer.lr - 0.025) < 1e-6


def test_steplr_optimizer_step_integration():
    w = Tensor(10.0, requires_grad=True)
    w.grad = 1.0

    optimizer = SGD([w], lr=1.0)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.1)

    # Initial lr = 1.0 -> w = 10 - 1.0 * 1 = 9.0
    optimizer.step()
    assert abs(w.data - 9.0) < 1e-6

    # Next epoch -> lr becomes 0.1
    scheduler.step()
    optimizer.step()
    # w = 9.0 - 0.1 * 1 = 8.9
    assert abs(w.data - 8.9) < 1e-6