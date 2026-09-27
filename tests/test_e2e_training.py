import random
from slowtorch.tensor import Tensor
from slowtorch.nn import Module, Linear, Sigmoid, MSELoss
from slowtorch.optim import SGD


class XORModel(Module):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(2, 4, bias=True)
        self.act1 = Sigmoid()
        self.fc2 = Linear(4, 1, bias=True)
        self.act2 = Sigmoid()

    def forward(self, x):
        h = self.fc1(x)
        h = self.act1(h)
        out = self.fc2(h)
        out = self.act2(out)
        return out


def test_xor_convergence_and_accuracy():
    random.seed(42)

    inputs = [
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0]
    ]
    targets = [
        [0.0],
        [1.0],
        [1.0],
        [0.0]
    ]

    x = Tensor(inputs)
    y = Tensor(targets)

    model = XORModel()
    criterion = MSELoss()
    optimizer = SGD(model.parameters(), lr=1.5)

    initial_loss = criterion(model(x), y).data

    for _ in range(1500):
        optimizer.zero_grad()
        predictions = model(x)
        loss = criterion(predictions, y)
        loss.backward()
        optimizer.step()

    final_loss = criterion(model(x), y).data

    # Loss must decrease drastically
    assert final_loss < 0.05
    assert final_loss < initial_loss

    # Check discrete classification labels
    preds = model(x)
    for i in range(len(inputs)):
        predicted_val = preds.data[i][0]
        expected_val = targets[i][0]
        predicted_label = 1.0 if predicted_val >= 0.5 else 0.0
        assert predicted_label == expected_val