import math
from slowtorch.tensor import Tensor
from slowtorch.nn import Softmax, CrossEntropyLoss


def test_softmax_probabilities_sum_to_one():
    # Large numbers to test numerical stability
    logits = Tensor([[1000.0, 1001.0, 1002.0], [-500.0, -500.0, -500.0]], requires_grad=True)
    sm = Softmax()
    probs = sm(logits)

    assert probs.shape == (2, 3)

    # Each row must sum to 1.0
    for r in range(2):
        row_sum = sum(probs.data[r])
        assert abs(row_sum - 1.0) < 1e-5


def test_cross_entropy_loss_calculation():
    # Batch size 2, Classes 3
    # Perfect scenario where prediction strongly aligns with ground truth
    logits = Tensor([
        [10.0, 0.0, 0.0],
        [0.0, 10.0, 0.0]
    ], requires_grad=True)
    targets = [0, 1]

    criterion = CrossEntropyLoss()
    loss = criterion(logits, targets)

    # Loss should be extremely close to 0.0
    assert loss.data < 0.01

    loss.backward()

    # Gradients for correct classes should be negative (close to 0)
    assert logits.grad[0][0] < 0.0
    assert logits.grad[1][1] < 0.0


def test_cross_entropy_gradient_correctness():
    # Logits: [0.0, 0.0], target: 0
    # Softmax output is [0.5, 0.5]
    # dLoss/dz0 = (0.5 - 1) / 1 = -0.5
    # dLoss/dz1 = (0.5 - 0) / 1 = 0.5
    logits = Tensor([[0.0, 0.0]], requires_grad=True)
    targets = [0]

    criterion = CrossEntropyLoss()
    loss = criterion(logits, targets)
    loss.backward()

    expected_loss = -math.log(0.5)
    assert abs(loss.data - expected_loss) < 1e-5
    assert abs(logits.grad[0][0] - (-0.5)) < 1e-5
    assert abs(logits.grad[0][1] - 0.5) < 1e-5