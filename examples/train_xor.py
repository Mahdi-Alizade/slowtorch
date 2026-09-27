import sys
import os
import random

# Add root project folder to sys.path so slowtorch is found
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from slowtorch.tensor import Tensor
from slowtorch.nn import Module, Linear, Sigmoid, MSELoss
from slowtorch.optim import SGD


class XORClassifier(Module):
    def __init__(self):
        super().__init__()
        # Input layer: 2 inputs -> 4 hidden units
        self.fc1 = Linear(in_features=2, out_features=4, bias=True)
        self.act1 = Sigmoid()
        # Output layer: 4 hidden units -> 1 output probability
        self.fc2 = Linear(in_features=4, out_features=1, bias=True)
        self.act2 = Sigmoid()

    def forward(self, x):
        h = self.fc1(x)
        h = self.act1(h)
        out = self.fc2(h)
        out = self.act2(out)
        return out


def main():
    # Set seed for reproducible results
    random.seed(42)

    # XOR Truth Table:
    # [0, 0] -> [0]
    # [0, 1] -> [1]
    # [1, 0] -> [1]
    # [1, 1] -> [0]
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

    model = XORClassifier()
    criterion = MSELoss()
    optimizer = SGD(model.parameters(), lr=1.5)

    epochs = 2000
    print("Starting XOR training loop...")
    print("---------------------------------------------")

    for epoch in range(1, epochs + 1):
        # 1. Zero out previous gradients
        optimizer.zero_grad()

        # 2. Forward pass
        predictions = model(x)

        # 3. Compute loss
        loss = criterion(predictions, y)

        # 4. Backward pass (compute gradients)
        loss.backward()

        # 5. Update weights and biases
        optimizer.step()

        # Print progress every 200 epochs
        if epoch % 200 == 0 or epoch == 1:
            loss_val = round(loss.data, 5)
            print("Epoch: " + str(epoch) + " | Loss: " + str(loss_val))

    print("---------------------------------------------")
    print("Training finished! Final predictions:")

    # Evaluate final output
    final_preds = model(x)
    for i in range(len(inputs)):
        inp = inputs[i]
        expected = targets[i][0]
        actual = final_preds.data[i][0]
        pred_label = 1 if actual >= 0.5 else 0

        print("Input: " + str(inp) + " | Expected: " + str(expected) + " | Predicted Raw: " + str(round(actual, 4)) + " | Label: " + str(pred_label))


if __name__ == "__main__":
    main()