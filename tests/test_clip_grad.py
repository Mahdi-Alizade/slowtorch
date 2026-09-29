import math
from slowtorch.tensor import Tensor
from slowtorch.nn import Linear
from slowtorch.nn.utils import clip_grad_norm_


def test_clip_grad_norm_rescaling():
    # p1: scalar with grad = 3.0
    # p2: vector with grad = [4.0, 0.0]
    # Total norm = sqrt(3^2 + 4^2 + 0^2) = sqrt(9 + 16) = 5.0
    p1 = Tensor(1.0, requires_grad=True)
    p1.grad = 3.0

    p2 = Tensor([2.0, 3.0], requires_grad=True)
    p2.grad = [4.0, 0.0]

    max_norm = 2.5
    returned_norm = clip_grad_norm_([p1, p2], max_norm=max_norm)

    assert abs(returned_norm - 5.0) < 1e-5

    # Target scaling ratio: 2.5 / 5.0 = 0.5
    # p1.grad should be 3.0 * 0.5 = 1.5
    # p2.grad should be [4.0 * 0.5, 0.0] = [2.0, 0.0]
    assert abs(p1.grad - 1.5) < 1e-4
    assert abs(p2.grad[0] - 2.0) < 1e-4
    assert abs(p2.grad[1] - 0.0) < 1e-4

    # Recalculate new total norm: sqrt(1.5^2 + 2.0^2) = 2.5
    new_norm = clip_grad_norm_([p1, p2], max_norm=max_norm)
    assert abs(new_norm - 2.5) < 1e-4


def test_clip_grad_norm_no_clipping_when_below_threshold():
    p = Tensor([[1.0, 2.0]], requires_grad=True)
    p.grad = [[0.3, 0.4]]  # norm = sqrt(0.09 + 0.16) = 0.5

    orig_grad = [[0.3, 0.4]]
    returned_norm = clip_grad_norm_([p], max_norm=1.0)

    assert abs(returned_norm - 0.5) < 1e-5
    # Gradients should remain unchanged
    assert abs(p.grad[0][0] - orig_grad[0][0]) < 1e-6
    assert abs(p.grad[0][1] - orig_grad[0][1]) < 1e-6