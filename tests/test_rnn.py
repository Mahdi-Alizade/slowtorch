from slowtorch.tensor import Tensor
from slowtorch.nn import RNNCell, RNN


def test_rnn_cell_forward_and_backward():
    # Input size 2, hidden size 3, batch size 1
    cell = RNNCell(input_size=2, hidden_size=3, bias=True)
    x = Tensor([[1.0, 2.0]], requires_grad=True)

    h_next = cell(x)
    assert h_next.shape == (1, 3)

    loss = h_next.sum()
    loss.backward()

    # Gradients must reach input, weights, and biases
    assert x.grad is not None
    assert cell.weight_ih.grad is not None
    assert cell.weight_hh.grad is not None
    assert cell.bias_ih.grad is not None
    assert cell.bias_hh.grad is not None


def test_rnn_unroll_seq_len_shapes():
    # seq_len=4, batch_size=2, input_size=3, hidden_size=5
    rnn = RNN(input_size=3, hidden_size=5, batch_first=False)

    seq_data = [[[0.5 for _ in range(3)] for _ in range(2)] for _ in range(4)]
    x = Tensor(seq_data, requires_grad=True)

    output, h_n = rnn(x)

    # Output should be (seq_len, batch_size, hidden_size) = (4, 2, 5)
    assert output.shape == (4, 2, 5)
    # Final hidden state should be (batch_size, hidden_size) = (2, 5)
    assert h_n.shape == (2, 5)


def test_rnn_bptt_gradient_propagation():
    # Verify BPTT: Loss on the final time step propagates gradients to the first time step input
    seq_len = 3
    batch_size = 1
    in_size = 2
    hidden_size = 2

    rnn = RNN(input_size=in_size, hidden_size=hidden_size, batch_first=True)

    # (batch_size=1, seq_len=3, in_size=2)
    x_data = [[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]]
    x = Tensor(x_data, requires_grad=True)

    output, h_n = rnn(x)

    # Compute loss ONLY on the final step hidden state
    loss = h_n.sum()
    loss.backward()

    # If BPTT works, gradients must flow all the way back to step 0 of x
    step0_grad = x.grad[0][0]
    step1_grad = x.grad[0][1]
    step2_grad = x.grad[0][2]

    assert any(abs(g) > 1e-7 for g in step0_grad)
    assert any(abs(g) > 1e-7 for g in step1_grad)
    assert any(abs(g) > 1e-7 for g in step2_grad)