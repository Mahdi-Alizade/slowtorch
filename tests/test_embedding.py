import pytest
from slowtorch.tensor import Tensor
from slowtorch.nn import Embedding


def test_embedding_forward_1d_and_2d():
    # Vocab size 5, embedding dimension 3
    emb = Embedding(num_embeddings=5, embedding_dim=3)

    # 1D lookup: (3,) -> output shape (3, 3)
    indices_1d = Tensor([0, 2, 4])
    out_1d = emb(indices_1d)
    assert out_1d.shape == (3, 3)
    assert out_1d.data[0] == emb.weight.data[0]
    assert out_1d.data[1] == emb.weight.data[2]
    assert out_1d.data[2] == emb.weight.data[4]

    # 2D batch lookup: (2, 2) -> output shape (2, 2, 3)
    indices_2d = Tensor([[1, 3], [0, 2]])
    out_2d = emb(indices_2d)
    assert out_2d.shape == (2, 2, 3)
    assert out_2d.data[0][0] == emb.weight.data[1]
    assert out_2d.data[1][1] == emb.weight.data[2]


def test_embedding_repeated_indices_gradient_accumulation():
    # Vocab size 4, embedding dimension 2
    emb = Embedding(num_embeddings=4, embedding_dim=2)

    # Index 1 appears THREE times across batch
    indices = Tensor([[1, 2], [1, 1]])
    out = emb(indices)

    loss = out.sum()
    loss.backward()

    # Index 1 was hit 3 times, each contributing 1.0 -> weight.grad[1] must be [3.0, 3.0]
    assert abs(emb.weight.grad[1][0] - 3.0) < 1e-5
    assert abs(emb.weight.grad[1][1] - 3.0) < 1e-5

    # Index 2 was hit 1 time -> weight.grad[2] must be [1.0, 1.0]
    assert abs(emb.weight.grad[2][0] - 1.0) < 1e-5
    assert abs(emb.weight.grad[2][1] - 1.0) < 1e-5

    # Index 0 and 3 were never selected -> weight.grad must remain 0.0
    assert emb.weight.grad[0] == [0.0, 0.0]
    assert emb.weight.grad[3] == [0.0, 0.0]


def test_embedding_padding_idx_zero_grad():
    # padding_idx=0 should stay 0 and receive 0 grad
    emb = Embedding(num_embeddings=3, embedding_dim=2, padding_idx=0)
    assert emb.weight.data[0] == [0.0, 0.0]

    indices = Tensor([[0, 1]])
    out = emb(indices)

    loss = out.sum()
    loss.backward()

    # padding_idx must not accumulate gradients
    assert emb.weight.grad[0] == [0.0, 0.0]
    assert emb.weight.grad[1] == [1.0, 1.0]


def test_embedding_out_of_bounds_index():
    emb = Embedding(num_embeddings=3, embedding_dim=2)
    with pytest.raises(IndexError):
        _ = emb(Tensor([5]))