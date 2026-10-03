from slowtorch.tensor import Tensor
from slowtorch.nn import TransformerEncoderLayer, TransformerEncoder, LayerNorm


def test_transformer_encoder_stack_shapes_and_gradients():
    # d_model=8, nhead=2, dim_feedforward=16
    base_layer = TransformerEncoderLayer(d_model=8, nhead=2, dim_feedforward=16, dropout=0.0)
    final_norm = LayerNorm(8)

    # 2 stacked encoder layers
    encoder = TransformerEncoder(encoder_layer=base_layer, num_layers=2, norm=final_norm)

    assert len(encoder) == 2

    # Input: batch 2, seq_len 4, d_model 8
    x_data = [[[0.1 for _ in range(8)] for _ in range(4)] for _ in range(2)]
    x = Tensor(x_data, requires_grad=True)

    out = encoder(x)
    assert out.shape == (2, 4, 8)

    loss = out.sum()
    loss.backward()

    # Gradients must flow back to x and all stacked layers
    assert x.grad is not None
    assert encoder[0].self_attn.q_proj.weight.grad is not None
    assert encoder[1].self_attn.q_proj.weight.grad is not None
    assert final_norm.weight.grad is not None


def test_transformer_encoder_without_final_norm():
    base_layer = TransformerEncoderLayer(d_model=4, nhead=1, dim_feedforward=8, dropout=0.0)
    encoder = TransformerEncoder(encoder_layer=base_layer, num_layers=3, norm=None)

    x_data = [[[0.5 for _ in range(4)] for _ in range(2)]]
    x = Tensor(x_data)

    out = encoder(x)
    assert out.shape == (1, 2, 4)