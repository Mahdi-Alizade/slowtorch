
import pytest
from slowtorch.tensor import Tensor
from slowtorch.nn import MultiheadAttention


def test_multihead_attention_forward_shapes():
    # Batch size 2, Sequence length 4, Embedding dim 8, 2 Heads (head_dim=4)
    mha = MultiheadAttention(embed_dim=8, num_heads=2)

    x_data = [[[0.5 for _ in range(8)] for _ in range(4)] for _ in range(2)]
    x = Tensor(x_data, requires_grad=True)

    out = mha(x, x, x)

    # Output shape must match input: (2, 4, 8)
    assert out.shape == (2, 4, 8)


def test_multihead_attention_with_causal_mask():
    # Sequence length 3, embed_dim 4, 1 head
    mha = MultiheadAttention(embed_dim=4, num_heads=1)

    x_data = [[[1.0 for _ in range(4)] for _ in range(3)]]
    x = Tensor(x_data, requires_grad=True)

    # Upper triangular mask filled with -1e9 to prevent looking forward
    causal_mask = [
        [0.0, -1e9, -1e9],
        [0.0, 0.0, -1e9],
        [0.0, 0.0, 0.0]
    ]

    out = mha(x, x, x, attn_mask=causal_mask)
    assert out.shape == (1, 3, 4)

    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert mha.q_proj.weight.grad is not None
    assert mha.k_proj.weight.grad is not None
    assert mha.v_proj.weight.grad is not None
    assert mha.out_proj.weight.grad is not None


def test_multihead_attention_invalid_dimensions():
    with pytest.raises(ValueError):
        # 10 is not divisible by 3
        _ = MultiheadAttention(embed_dim=10, num_heads=3)