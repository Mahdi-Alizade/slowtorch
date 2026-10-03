import math
import pytest
from slowtorch.tensor import Tensor
from slowtorch.nn import PositionalEncoding


def test_positional_encoding_math_values():
    # d_model=4, max_len=10, dropout=0.0
    pe_module = PositionalEncoding(d_model=4, max_len=10, dropout=0.0)

    # Position 0:
    # 2i=0 -> sin(0) = 0.0
    # 2i+1=1 -> cos(0) = 1.0
    # 2i=2 -> sin(0) = 0.0
    # 2i+1=3 -> cos(0) = 1.0
    assert abs(pe_module.pe.data[0][0][0] - 0.0) < 1e-5
    assert abs(pe_module.pe.data[0][0][1] - 1.0) < 1e-5
    assert abs(pe_module.pe.data[0][0][2] - 0.0) < 1e-5
    assert abs(pe_module.pe.data[0][0][3] - 1.0) < 1e-5

    # Position 1, dimension 0: sin(1 / 10000^0) = sin(1.0) = 0.84147
    expected_pos1_d0 = math.sin(1.0)
    assert abs(pe_module.pe.data[0][1][0] - expected_pos1_d0) < 1e-4


def test_positional_encoding_forward_and_backward():
    # Batch size 2, Sequence length 5, d_model 6
    pe_module = PositionalEncoding(d_model=6, max_len=20, dropout=0.0)

    x_zeros = [[[0.0 for _ in range(6)] for _ in range(5)] for _ in range(2)]
    x = Tensor(x_zeros, requires_grad=True)

    out = pe_module(x)
    assert out.shape == (2, 5, 6)

    # Since x was all zeros and dropout=0, out must equal the PE slice across batch items
    for b in range(2):
        for s in range(5):
            assert out.data[b][s] == pe_module.pe.data[0][s]

    loss = out.sum()
    loss.backward()

    # Gradients should pass through cleanly to x
    for b in range(2):
        for s in range(5):
            for d in range(6):
                assert x.grad[b][s][d] == 1.0


def test_positional_encoding_exceeds_max_len():
    pe_module = PositionalEncoding(d_model=4, max_len=5)
    # Sequence length 6 exceeds max_len 5
    x = Tensor([[[0.0 for _ in range(4)] for _ in range(6)]])
    with pytest.raises(ValueError):
        _ = pe_module(x)