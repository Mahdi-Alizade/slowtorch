import sys
import os
import random
import math

# Add root folder to sys.path for direct script execution
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from slowtorch.tensor import Tensor
from slowtorch.nn import Module, Linear, ReLU, CrossEntropyLoss, Softmax
from slowtorch.optim import Adam


class MultiClassClassifier(Module):
    def __init__(self, in_features=2, hidden_dim=8, num_classes=3):
        super().__init__()
        self.fc1 = Linear(in_features, hidden_dim, bias=True)
        self.relu = ReLU()
        self.fc2 = Linear(hidden_dim, num_classes, bias=True)

    def forward(self, x):
        h = self.fc1(x)
        h = self.relu(h)
        logits = self.fc2(h)
        return logits


def generate_synthetic_clusters(points_per_cluster=20):
    # 3 distinct 2D clusters
    # Cluster 0 center: (-2.0, -2.0)
    # Cluster 1 center: (2.0, 2.0)
    # Cluster 2 center: (-2.0, 2.0)
    centers = [(-2.0, -2.0), (2.0, 2.0), (-2.0, 2.0)]
    data = []
    labels = []

    for class_id in range(len(centers)):
        cx, cy = centers[class_id]
        for _ in range(points_per_cluster):
            # Add small random noise around cluster center
            x = cx + random.gauss(0.0, 0.45)
            y = cy + random.gauss(0.0, 0.45)
            data.append([x, y])
            labels.append(class_id)

    return data, labels


def main():
    random.seed(1337)

    raw_x, raw_y = generate_synthetic_clusters(points_per_cluster=25)
    total_samples = len(raw_x)

    x = Tensor(raw_x)
    targets = raw_y

    model = MultiClassClassifier(in_features=2, hidden_dim=8, num_classes=3)
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.03)

    softmax = Softmax()
    epochs = 150

    print("Starting Multi-Class Classification Training...")
    print("Dataset samples: " + str(total_samples) + " across 3 classes.")
    print("---------------------------------------------------------")

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()

        # Forward pass: raw logits returned
        logits = model(x)

        # Loss computation using numerically stable CrossEntropy
        loss = criterion(logits, targets)

        # Backward propagation
        loss.backward()

        # Weight updates via Adam
        optimizer.step()

        if epoch % 25 == 0 or epoch == 1:
            # Calculate classification accuracy
            correct_count = 0
            for i in range(total_samples):
                row = logits.data[i]
                predicted_class = 0
                max_logit = row[0]
                for c in range(1, len(row)):
                    if row[c] > max_logit:
                        max_logit = row[c]
                        predicted_class = c
                if predicted_class == targets[i]:
                    correct_count = correct_count + 1

            acc = (correct_count / float(total_samples)) * 100.0
            print("Epoch: " + str(epoch).rjust(3) + " | Loss: " + str(round(loss.data, 4)).ljust(7) + " | Accuracy: " + str(round(acc, 2)) + "%")

    print("---------------------------------------------------------")
    print("Training Complete! Testing 3 custom coordinates:")

    test_points = [
        [-2.0, -2.0],  # Should be Class 0
        [2.0, 2.0],    # Should be Class 1
        [-2.0, 2.0]    # Should be Class 2
    ]

    test_tensor = Tensor(test_points)
    out_logits = model(test_tensor)
    out_probs = softmax(out_logits)

    for idx in range(len(test_points)):
        probs = [round(p, 3) for p in out_probs.data[idx]]
        pred_c = 0
        max_p = probs[0]
        for c in range(1, len(probs)):
            if probs[c] > max_p:
                max_p = probs[c]
                pred_c = c

        print("Point: " + str(test_points[idx]) + " -> Class " + str(pred_c) + " (Probabilities: " + str(probs) + ")")


if __name__ == "__main__":
    main()