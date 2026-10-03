from slowtorch.tensor import Tensor
from slowtorch.nn import TransformerEncoderLayer


def test_transformer_encoder_layer_forward_and_backward():
    # Batch size 2, Sequence length 3, d_model 8, 2 heads, dim_feedforward 16
    layer = TransformerEncoderLayer(d_model=8, nhead=2, dim_feedforward=16, dropout=0.0)

    x_data = [[[0.2 for _ in range(8)] for _ in range(3)] for _ in range(2)]
    x = Tensor(x_data, requires_grad=True)

    out = layer(x)

    # Output dimensions must preserve input shape exactly: (2, 3, 8)
    assert out.shape == (2, 3, 8)

    loss = out.sum()
    loss.backward()

    # Gradients must reach input tensor
    assert x.grad is not None

    # Gradients must propagate to attention projection weights
    assert layer.self_attn.q_proj.weight.grad is not None
    assert layer.self_attn.out_proj.weight.grad is not None

    # Gradients must propagate to FFN linear layers
    assert layer.linear1.weight.grad is not None
    assert layer.linear2.weight.grad is not None

    # Gradients must reach LayerNorm weights
    assert layer.norm1.weight.grad is not None
    assert layer.norm2.weight.grad is not None


def test_transformer_encoder_layer_eval_mode():
    layer = TransformerEncoderLayer(d_model=4, nhead=1, dim_feedforward=8, dropout=0.5)
    layer.eval()

    x_data = [[[1.0 for _ in range(4)] for _ in range(2)]]
    x = Tensor(x_data)

    out = layer(x)
    assert out.shape == (1, 2, 4)