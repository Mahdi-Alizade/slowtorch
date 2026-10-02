import sys
import os
import random

# Add project root directory to sys.path for direct script execution
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from slowtorch.tensor import Tensor
from slowtorch.nn import Sequential, Conv2d, ReLU, MaxPool2d, Flatten, Linear, CrossEntropyLoss, Softmax
from slowtorch.optim import Adam
from slowtorch.data import TensorDataset, DataLoader


def generate_synthetic_image_dataset(num_samples_per_class=30):
    images = []
    labels = []

    # Image size: 6x6 with 1 channel
    for _ in range(num_samples_per_class):
        # Class 0: Horizontal bar pattern in middle rows (rows 2 and 3)
        h_img = []
        for r in range(6):
            row = []
            for c in range(6):
                base_val = 1.0 if (r == 2 or r == 3) else 0.0
                noise = random.gauss(0.0, 0.08)
                val = base_val + noise
                val = max(0.0, min(1.0, val))
                row.append(val)
            h_img.append(row)
        images.append([h_img])
        labels.append(0)

        # Class 1: Vertical bar pattern in middle columns (cols 2 and 3)
        v_img = []
        for r in range(6):
            row = []
            for c in range(6):
                base_val = 1.0 if (c == 2 or c == 3) else 0.0
                noise = random.gauss(0.0, 0.08)
                val = base_val + noise
                val = max(0.0, min(1.0, val))
                row.append(val)
            v_img.append(row)
        images.append([v_img])
        labels.append(1)

    return images, labels


def main():
    random.seed(42)

    raw_images, raw_labels = generate_synthetic_image_dataset(num_samples_per_class=30)
    total_samples = len(raw_images)

    x_tensor = Tensor(raw_images, requires_grad=False)
    y_tensor = Tensor([[float(label)] for label in raw_labels], requires_grad=False)

    dataset = TensorDataset(x_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=10, shuffle=True)

    # Build CNN Architecture
    # Input: (N, 1, 6, 6)
    # Conv2d -> (N, 2, 6, 6)
    # MaxPool2d -> (N, 2, 3, 3)
    # Flatten -> (N, 18)
    # Linear -> (N, 2)
    model = Sequential(
        Conv2d(in_channels=1, out_channels=2, kernel_size=3, padding=1, bias=True),
        ReLU(),
        MaxPool2d(kernel_size=2, stride=2),
        Flatten(),
        Linear(in_features=18, out_features=2, bias=True)
    )

    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.04)
    softmax = Softmax()

    epochs = 40
    print("Starting CNN Training (Pattern Recognition: Horizontal vs Vertical)...")
    print("Dataset samples: " + str(total_samples) + " images of shape (1, 6, 6).")
    print("Architecture: Conv2d(1, 2, k=3, p=1) -> ReLU -> MaxPool2d(2) -> Flatten -> Linear(18, 2)")
    print("-----------------------------------------------------------------------------------")

    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        correct_predictions = 0

        for batch_x, batch_y in loader:
            optimizer.zero_grad()

            logits = model(batch_x)
            targets = [int(item[0]) for item in batch_y.data]

            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            batch_size = len(targets)
            running_loss = running_loss + loss.data * batch_size

            for i in range(batch_size):
                pred_c = 0 if logits.data[i][0] > logits.data[i][1] else 1
                if pred_c == targets[i]:
                    correct_predictions = correct_predictions + 1

        epoch_loss = running_loss / float(total_samples)
        accuracy = (correct_predictions / float(total_samples)) * 100.0

        if epoch % 5 == 0 or epoch == 1:
            print("Epoch: " + str(epoch).rjust(2) + " | Loss: " + str(round(epoch_loss, 4)).ljust(7) + " | Accuracy: " + str(round(accuracy, 2)) + "%")

    print("-----------------------------------------------------------------------------------")
    print("Training finished! Testing on 2 synthetic test patterns:")

    # Clean Horizontal Test Image
    test_h = [[
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ]]

    # Clean Vertical Test Image
    test_v = [[
        [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 1.0, 0.0, 0.0]
    ]]

    test_tensor = Tensor([test_h, test_v])
    test_logits = model(test_tensor)
    test_probs = softmax(test_logits)

    names = ["Horizontal Pattern (Class 0)", "Vertical Pattern (Class 1)"]
    for i in range(2):
        p0 = round(test_probs.data[i][0], 3)
        p1 = round(test_probs.data[i][1], 3)
        pred_label = 0 if p0 > p1 else 1
        print("Input: " + names[i] + " -> Predicted Class " + str(pred_label) + " (Probabilities: [P(H)=" + str(p0) + ", P(V)=" + str(p1) + "])")


if __name__ == "__main__":
    main()