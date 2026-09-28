import os
import pytest
from slowtorch.tensor import Tensor
from slowtorch.nn import Module, Linear, ReLU


class SampleNet(Module):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(2, 3, bias=True)
        self.relu = ReLU()
        self.fc2 = Linear(3, 1, bias=True)

    def forward(self, x):
        h = self.relu(self.fc1(x))
        return self.fc2(h)


def test_state_dict_and_load_state_dict():
    net1 = SampleNet()
    net2 = SampleNet()

    # Give net1 custom distinct values
    net1.fc1.weight.data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    net1.fc1.bias.data = [0.1, 0.2, 0.3]
    net1.fc2.weight.data = [[0.5], [0.6], [0.7]]
    net1.fc2.bias.data = [0.05]

    state = net1.state_dict()
    assert "fc1.weight" in state
    assert "fc1.bias" in state
    assert "fc2.weight" in state
    assert "fc2.bias" in state

    # Load into net2
    net2.load_state_dict(state)

    x = Tensor([[1.0, -1.0]])
    out1 = net1(x)
    out2 = net2(x)

    assert abs(out1.data[0][0] - out2.data[0][0]) < 1e-6


def test_load_state_dict_strict_error():
    net = SampleNet()
    bad_state = {"fc1.weight": [[1.0, 2.0, 3.0]]}

    with pytest.raises(KeyError):
        net.load_state_dict(bad_state, strict=True)


def test_file_save_and_load(tmp_path):
    model_path = str(tmp_path / "model.json")

    m1 = SampleNet()
    m2 = SampleNet()

    m1.fc1.weight.data = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    m1.save(model_path)

    assert os.path.exists(model_path)

    m2.load(model_path)
    assert m2.fc1.weight.data == m1.fc1.weight.data