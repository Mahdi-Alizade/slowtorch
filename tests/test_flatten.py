from slowtorch.tensor import Tensor
from slowtorch.nn import Flatten, Linear, Sequential


def test_flatten_4d_to_2d_and_backward():
    # Shape: (2, 2, 2, 2) -> 2 batches, each having 8 features
    sample0 = [
        [[1.0, 2.0], [3.0, 4.0]],
        [[5.0, 6.0], [7.0, 8.0]]
    ]
    sample1 = [
        [[9.0, 10.0], [11.0, 12.0]],
        [[13.0, 14.0], [15.0, 16.0]]
    ]
    x = Tensor([sample0, sample1], requires_grad=True)

    flatten = Flatten()
    out = flatten(x)

    assert out.shape == (2, 8)
    assert out.data[0] == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    assert out.data[1] == [9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]

    loss = out.sum()
    loss.backward()

    # Gradients should restore back into (2, 2, 2, 2) filled with 1.0s
    for n in range(2):
        for c in range(2):
            for h in range(2):
                for w in range(2):
                    assert x.grad[n][c][h][w] == 1.0


def test_flatten_pipeline_with_linear():
    # End-to-end pipeline: 4D Conv-like output -> Flatten -> Linear -> Output
    model = Sequential(
        Flatten(),
        Linear(in_features=4, out_features=1, bias=False)
    )
    model[1].weight.data = [[1.0], [2.0], [3.0], [4.0]]

    # 1 batch, 1 channel, 2x2 image = 4 features
    x = Tensor([[[[1.0, 1.0], [1.0, 1.0]]]], requires_grad=True)
    out = model(x)

    # 1*1 + 1*2 + 1*3 + 1*4 = 10.0
    assert abs(out.data[0][0] - 10.0) < 1e-5

    loss = out.sum()
    loss.backward()

    # Gradients flow back through linear and flatten to the 4D input
    expected_grad = [[[[1.0, 2.0], [3.0, 4.0]]]]
    assert x.grad == expected_grad