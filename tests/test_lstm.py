from slowtorch.tensor import Tensor
from slowtorch.nn import LSTMCell, LSTM


def test_lstm_cell_forward_and_backward():
    # Batch 1, Input size 2, Hidden size 3
    cell = LSTMCell(input_size=2, hidden_size=3, bias=True)
    x = Tensor([[1.0, 2.0]], requires_grad=True)

    h_next, c_next = cell(x)

    assert h_next.shape == (1, 3)
    assert c_next.shape == (1, 3)

    # Output bounded by tanh: within (-1, 1)
    for val in h_next.data[0]:
        assert -1.0 <= val <= 1.0

    loss = h_next.sum() + c_next.sum()
    loss.backward()

    # Gradients must reach input, weights, and biases
    assert x.grad is not None
    assert cell.weight_ih.grad is not None
    assert cell.weight_hh.grad is not None
    assert cell.bias_ih.grad is not None
    assert cell.bias_hh.grad is not None


def test_lstm_sequence_shapes_batch_first():
    # batch_size=2, seq_len=4, input_size=3, hidden_size=5
    lstm = LSTM(input_size=3, hidden_size=5, batch_first=True)

    seq_data = [[[0.1 for _ in range(3)] for _ in range(4)] for _ in range(2)]
    x = Tensor(seq_data, requires_grad=True)

    out, (h_n, c_n) = lstm(x)

    # Output shape: (batch_size, seq_len, hidden_size) = (2, 4, 5)
    assert out.shape == (2, 4, 5)
    # State shapes: (batch_size, hidden_size) = (2, 5)
    assert h_n.shape == (2, 5)
    assert c_n.shape == (2, 5)

    loss = out.sum()
    loss.backward()

    assert x.grad is not None


def test_lstm_gradient_flow_to_initial_step():
    lstm = LSTM(input_size=2, hidden_size=2, batch_first=True)

    # 3-step sequence
    x_data = [[[1.0, 0.5], [0.5, -0.5], [-1.0, 1.0]]]
    x = Tensor(x_data, requires_grad=True)

    out, (h_n, c_n) = lstm(x)

    # Loss depends only on the final state
    loss = h_n.sum()
    loss.backward()

    # BPTT must propagate gradients to step 0
    step0_grads = x.grad[0][0]
    assert any(abs(g) > 1e-7 for g in step0_grads)