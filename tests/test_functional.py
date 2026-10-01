import math
from slowtorch.tensor import Tensor
import slowtorch.nn.functional as F


def test_cosine_similarity_properties():
    # 1. Collinear vectors -> similarity = 1.0
    u1 = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    v1 = Tensor([2.0, 4.0, 6.0], requires_grad=True)
    sim1 = F.cosine_similarity(u1, v1)
    assert abs(sim1.data - 1.0) < 1e-5

    # 2. Orthogonal vectors -> similarity = 0.0
    u2 = Tensor([1.0, 0.0], requires_grad=True)
    v2 = Tensor([0.0, 1.0], requires_grad=True)
    sim2 = F.cosine_similarity(u2, v2)
    assert abs(sim2.data - 0.0) < 1e-5

    # 3. Opposite vectors -> similarity = -1.0
    u3 = Tensor([2.0, -1.0])
    v3 = Tensor([-2.0, 1.0])
    sim3 = F.cosine_similarity(u3, v3)
    assert abs(sim3.data - (-1.0)) < 1e-5


def test_cosine_similarity_backward():
    u = Tensor([[1.0, 0.0]], requires_grad=True)
    v = Tensor([[1.0, 1.0]], requires_grad=True)

    sim = F.cosine_similarity(u, v)
    # Cosine( [1, 0], [1, 1] ) = 1 / sqrt(2) = 0.7071
    assert abs(sim.data[0] - (1.0 / math.sqrt(2.0))) < 1e-4

    loss = sim.sum()
    loss.backward()

    assert u.grad is not None
    assert v.grad is not None


def test_pairwise_distance_and_backward():
    # 3-4-5 right triangle: Distance between (0, 0) and (3, 4) is 5.0
    p1 = Tensor([[0.0, 0.0]], requires_grad=True)
    p2 = Tensor([[3.0, 4.0]], requires_grad=True)

    dist = F.pairwise_distance(p1, p2)
    assert abs(dist.data[0] - 5.0) < 1e-5

    loss = dist.sum()
    loss.backward()

    # d(dist)/dp1 = (p1 - p2) / dist = [-3/5, -4/5] = [-0.6, -0.8]
    assert abs(p1.grad[0][0] - (-0.6)) < 1e-4
    assert abs(p1.grad[0][1] - (-0.8)) < 1e-4

    # d(dist)/dp2 = -d(dist)/dp1 = [0.6, 0.8]
    assert abs(p2.grad[0][0] - 0.6) < 1e-4
    assert abs(p2.grad[0][1] - 0.8) < 1e-4